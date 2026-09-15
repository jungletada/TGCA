"""Wait until a scheduled time, then poll COCO and launch one diagnostic queue.

Stdlib-only monitor: no CUDA context, no model load, no edits to watched runs.
All monitor artifacts live in a separate results directory. No experiment is
stopped or retried, and only successful full COCO completion releases the queue.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))
PYTHON = '/home/peng/anaconda3/envs/tgca-repro/bin/python'


def result_path(path):
    path = Path(path).resolve()
    root = (REPO / 'results').resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Monitor and experiment paths must be inside results/')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def audit_coco(root):
    """No inference. A final checkpoint alone is NOT completion."""
    if (root / 'QUEUE_FAILED').exists():
        return 'failed', (root / 'QUEUE_FAILED').read_text().strip()
    if not (root / 'QUEUE_COMPLETE').exists():
        return 'waiting', 'COCO full queue has not completed (training/CAM/classification may still be active)'
    required = ['manifest.json', 'comparison.csv', 'C2P_COCO_REPORT.md',
                'COCO/RUN_COMPLETE', 'COCO/TRAIN_COMPLETE', 'COCO/CAM_COMPLETE',
                'COCO/raw_cam/EVAL_COMPLETE', 'COCO/raw_cam/metrics.json',
                'COCO/mctformerplus_final.pth', 'COCO/checkpoint_sha256.txt',
                'COCO/classification/classification_metrics.json',
                'baseline_classification/classification_metrics.json']
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return 'failed', 'COCO completion marker conflicts with missing files: ' + ', '.join(missing)
    try:
        manifest = read_json(root / 'manifest.json')
        before, after = manifest['source_sha256_before'], manifest['source_sha256_after']
        if not manifest.get('source_integrity_unchanged') or not before or before != after:
            raise ValueError('COCO source integrity audit did not pass')
        for path, expected in after.items():
            if sha256(path) != expected:
                raise ValueError('Changed COCO source: ' + path)
        cam = read_json(root / 'COCO/raw_cam/metrics.json')
        if cam['num_images'] != 82783 or cam['num_classes_including_bg'] != 81:
            raise ValueError('Incomplete COCO CAM coverage')
        checkpoint = root / 'COCO/mctformerplus_final.pth'
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != (root / 'COCO/checkpoint_sha256.txt').read_text().split()[0]:
            raise ValueError('COCO final checkpoint hash mismatch')
        for branch in ('COCO/classification', 'baseline_classification'):
            metrics = read_json(root / branch / 'classification_metrics.json')
            if metrics['num_images'] != 40504 or not metrics.get('finite'):
                raise ValueError('Incomplete/non-finite COCO classification: ' + branch)
            if branch == 'COCO/classification' and metrics['checkpoint']['sha256'] != checkpoint_hash:
                raise ValueError('COCO classification/checkpoint mismatch')
        with (root / 'comparison.csv').open() as stream:
            methods = [row['pooling'] for row in csv.DictReader(stream)]
        if sorted(methods) != ['c2p-all-product', 'gwrp']:
            raise ValueError('Incomplete COCO comparison table')
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return 'failed', f'COCO completion audit failed: {exc}'
    return 'complete', 'Full COCO queue, image coverage, report and source/checkpoint hashes verified'


def code_hashes():
    # Allow later docs-only commits, but never silently launch altered training
    # code. Record tracked Python/shell dependencies without touching sources.
    paths = subprocess.check_output(['git', 'ls-files', '-z', '*.py', '*.sh'], cwd=REPO).decode().split('\0')
    return {path: sha256(REPO / path) for path in paths if path}


def has_tmux(name):
    return subprocess.run(['tmux', 'has-session', '-t', '=' + name], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def readiness(next_output, session, expected_code):
    if next_output.exists() or has_tmux(session):
        return 'already_present', 'Diagnostic output/session already exists; refusing duplicate launch'
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() != 'main':
        return 'blocked', 'Repository is no longer on main'
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO).strip():
        return 'waiting', 'Tracked worktree is dirty; wait for a clean committed checkout'
    if code_hashes() != expected_code:
        return 'blocked', 'Training/analysis code changed since scheduling; review before launching'
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if any(line.strip().isdigit() for line in processes.splitlines()):
            return 'waiting', 'GPU compute process still active; no preemption'
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        if not memory.strip() or int(memory.splitlines()[0]) < 30000:
            return 'waiting', 'GPU0 has less than 30000 MiB free'
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        return 'waiting', f'GPU status unavailable: {exc}'
    if shutil.disk_usage(REPO / 'results').free < 10 * 1024**3:
        return 'waiting', 'Less than 10 GiB disk free for the diagnostic queue'
    return 'ready', 'Clean main, unchanged code, idle GPU and sufficient disk space'


def write_event(output, state, message, **extra):
    payload = dict(time=datetime.now(JST).isoformat(), state=state, message=message, **extra)
    line = json.dumps(payload, ensure_ascii=False)
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(line + '\n')
    (output / 'status.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(line, flush=True)


def launch_once(output, next_output, session):
    if next_output.exists() or has_tmux(session):
        raise FileExistsError('Diagnostic target exists; no duplicate launch')
    command = [PYTHON, '-u', '-m', 'experiments.run_diag_trajectory', '--dataset', 'VOC12',
               '--output', str(next_output), '--execute']
    log = output / 'diagnostic_queue.log'
    with (output / 'LAUNCH_CLAIM.json').open('x') as stream:
        json.dump({'command': command, 'session': session, 'output': str(next_output),
                   'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}, stream, indent=2)
    # The target directory is created exclusively by the existing runner.
    # noclobber protects the separate queue log in case of an unexpected retry.
    shell_command = 'set -C\nexec ' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(REPO), shell_command], check=True)
    with (output / 'commands.sh').open('a') as stream:
        stream.write(shlex.join(command) + '\n')
    (output / 'LAUNCHED').write_text(datetime.now(JST).isoformat() + '\n')


def parse_start(value):
    start = datetime.fromisoformat(value)
    if start.tzinfo is None:
        raise ValueError('--start-at must include timezone, e.g. 2026-09-16T00:00:00+09:00')
    return start.astimezone(JST)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--watch-run', required=True)
    parser.add_argument('--start-at', required=True)
    parser.add_argument('--interval-minutes', type=float, default=30)
    parser.add_argument('--monitor-output', required=True)
    parser.add_argument('--next-output', required=True)
    parser.add_argument('--next-session', required=True)
    args = parser.parse_args()
    root, output, next_output = map(result_path, (args.watch_run, args.monitor_output, args.next_output))
    start = parse_start(args.start_at)
    if args.interval_minutes <= 0 or not root.is_dir():
        raise ValueError('Positive interval and an existing watched run required')
    if output == next_output or output.is_relative_to(root) or next_output.is_relative_to(root):
        raise ValueError('Monitor/next run must not be inside the immutable watched run')
    locks = REPO / 'results/monitor_locks'
    locks.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.lock'
    with (locks / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        expected_code = code_hashes()
        (output / 'manifest.json').write_text(json.dumps({
            'config': vars(args), 'start_jst': start.isoformat(), 'code_sha256': expected_code,
            'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'next_matrix': 'VOC GWRP/C2P-all-product, seeds0/1/2, 45 epochs each; no C/D/E',
        }, indent=2) + '\n')
        (output / 'commands.sh').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
        write_event(output, 'scheduled', 'Waiting until the first scheduled check', next_check_at=start.isoformat())
        while datetime.now(JST) < start:
            time.sleep(min(60, max(0, (start - datetime.now(JST)).total_seconds())))
        while True:
            state, message = audit_coco(root)
            write_event(output, 'coco_' + state, message)
            if state == 'failed':
                (output / 'STOPPED').write_text(message + '\n')
                return
            if state == 'complete':
                ready, reason = readiness(next_output, args.next_session, expected_code)
                write_event(output, ready, reason)
                if ready in {'already_present', 'blocked'}:
                    (output / 'STOPPED').write_text(reason + '\n')
                    return
                if ready == 'ready':
                    try:
                        launch_once(output, next_output, args.next_session)
                    except Exception as exc:
                        write_event(output, 'launch_failed', str(exc))
                        (output / 'STOPPED').write_text(str(exc) + '\n')
                        raise
                    write_event(output, 'launched', 'Diagnostic VOC queue handed off to tmux',
                                session=args.next_session, result_root=str(next_output),
                                log=str(output / 'diagnostic_queue.log'))
                    return
            next_check = datetime.now(JST) + timedelta(minutes=args.interval_minutes)
            write_event(output, 'waiting', 'Next check scheduled', next_check_at=next_check.isoformat())
            while datetime.now(JST) < next_check:
                time.sleep(min(60, max(0, (next_check - datetime.now(JST)).total_seconds())))


if __name__ == '__main__':
    main()
