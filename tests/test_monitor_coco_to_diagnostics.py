from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from experiments import monitor_coco_to_diagnostics as monitor


def completed_run(tmp_path):
    root = tmp_path / 'coco'
    root.mkdir()
    for name in ['QUEUE_COMPLETE', 'C2P_COCO_REPORT.md', 'COCO/RUN_COMPLETE',
                 'COCO/TRAIN_COMPLETE', 'COCO/CAM_COMPLETE', 'COCO/raw_cam/EVAL_COMPLETE']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('complete\n')
    checkpoint = root / 'COCO/mctformerplus_final.pth'
    checkpoint.write_bytes(b'test checkpoint fixture, not a model')
    digest = monitor.sha256(checkpoint)
    (root / 'COCO/checkpoint_sha256.txt').write_text(digest + '  ' + str(checkpoint))
    source = tmp_path / 'immutable_source'; source.write_bytes(b'source')
    hashes = {str(source): monitor.sha256(source)}
    payloads = {
        'manifest.json': {'source_integrity_unchanged': True, 'source_sha256_before': hashes, 'source_sha256_after': hashes},
        'COCO/raw_cam/metrics.json': {'num_images': 82783, 'num_classes_including_bg': 81},
        'COCO/classification/classification_metrics.json': {'num_images': 40504, 'finite': True, 'checkpoint': {'sha256': digest}},
        'baseline_classification/classification_metrics.json': {'num_images': 40504, 'finite': True},
    }
    for name, payload in payloads.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    (root / 'comparison.csv').write_text('pooling\ngwrp\nc2p-all-product\n')
    return root


def test_incomplete_and_failure_never_release(tmp_path):
    root = tmp_path
    assert monitor.audit_coco(root)[0] == 'waiting'
    (root / 'mctformerplus_final.pth').write_bytes(b'final does not imply evaluation completed')
    assert monitor.audit_coco(root)[0] == 'waiting'
    (root / 'QUEUE_FAILED').write_text('training failed')
    assert monitor.audit_coco(root) == ('failed', 'training failed')


def test_complete_full_audit_read_only(tmp_path):
    root = completed_run(tmp_path)
    before = {str(p): monitor.sha256(p) for p in root.rglob('*') if p.is_file()}
    assert monitor.audit_coco(root)[0] == 'complete'
    after = {str(p): monitor.sha256(p) for p in root.rglob('*') if p.is_file()}
    assert before == after


@pytest.mark.parametrize('problem', ['source', 'checkpoint', 'coverage', 'missing', 'failure_wins'])
def test_corrupt_completion_cannot_launch(tmp_path, problem):
    root = completed_run(tmp_path)
    if problem == 'source':
        (tmp_path / 'immutable_source').write_bytes(b'changed')
    elif problem == 'checkpoint':
        (root / 'COCO/mctformerplus_final.pth').write_bytes(b'changed')
    elif problem == 'coverage':
        (root / 'COCO/raw_cam/metrics.json').write_text(json.dumps({'num_images': 1, 'num_classes_including_bg': 81}))
    elif problem == 'missing':
        (root / 'C2P_COCO_REPORT.md').unlink()  # disposable fixture only
    else:
        (root / 'QUEUE_FAILED').write_text('failed')
    assert monitor.audit_coco(root)[0] == 'failed'


def test_timezone_and_path_validation():
    assert monitor.parse_start('2026-09-15T15:00:00+00:00').isoformat() == '2026-09-16T00:00:00+09:00'
    with pytest.raises(ValueError, match='timezone'):
        monitor.parse_start('2026-09-16T00:00:00')
    with pytest.raises(ValueError, match='inside results'):
        monitor.result_path('/tmp/not-an-experiment')


def test_launch_is_one_shot_and_command_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'has_tmux', lambda name: False)
    monkeypatch.setattr(monitor.subprocess, 'check_output', lambda *a, **k: 'commit\n')
    calls = []
    monkeypatch.setattr(monitor.subprocess, 'run', lambda command, **kw: calls.append(command))
    monitor.launch_once(tmp_path, tmp_path / 'next', 'diagnostic-test')
    assert len(calls) == 1 and calls[0][:3] == ['tmux', 'new-session', '-d']
    claim = json.loads((tmp_path / 'LAUNCH_CLAIM.json').read_text())
    assert claim['command'][-1] == '--execute'
    assert claim['command'][claim['command'].index('--dataset') + 1] == 'VOC12'
    assert not (tmp_path / 'next').exists()
    with pytest.raises(FileExistsError):
        monitor.launch_once(tmp_path, tmp_path / 'next', 'diagnostic-test')
    assert len(calls) == 1


def test_readiness_waits_for_gpu_and_dirty_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'has_tmux', lambda name: False)
    monkeypatch.setattr(monitor, 'code_hashes', lambda: {'code': 'same'})
    monkeypatch.setattr(monitor.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20 * 1024**3))
    states = {'dirty': b'', 'pids': '1234\n'}
    def output(command, **kwargs):
        if command[:2] == ['git', 'branch']:
            return 'main\n'
        if command[:2] == ['git', 'status']:
            return states['dirty']
        if '--query-compute-apps=pid' in command:
            return states['pids']
        return '48000\n'
    monkeypatch.setattr(monitor.subprocess, 'check_output', output)
    assert monitor.readiness(tmp_path / 'new', 'session', {'code': 'same'})[0] == 'waiting'
    states['pids'] = ''
    assert monitor.readiness(tmp_path / 'new', 'session', {'code': 'same'})[0] == 'ready'
    states['dirty'] = b' M train_model_v2.py'
    assert monitor.readiness(tmp_path / 'new', 'session', {'code': 'same'})[0] == 'waiting'
    states['dirty'] = b''
    assert monitor.readiness(tmp_path / 'new', 'session', {'code': 'changed'})[0] == 'blocked'
    (tmp_path / 'new').mkdir()
    assert monitor.readiness(tmp_path / 'new', 'session', {})[0] == 'already_present'


def test_midnight_schedule_and_thirty_minute_poll(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    watched = repo / 'results/coco'; watched.mkdir(parents=True)
    output, next_output = repo / 'results/monitor', repo / 'results/diagnostic'
    clock = [datetime.fromisoformat('2026-09-15T23:59:00+09:00')]
    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]
    monkeypatch.setattr(monitor, 'datetime', FakeDatetime)
    monkeypatch.setattr(monitor.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)))
    monkeypatch.setattr(monitor, 'REPO', repo)
    monkeypatch.setattr(monitor, 'code_hashes', lambda: {'code': 'same'})
    monkeypatch.setattr(monitor.subprocess, 'check_output', lambda *a, **kw: 'sha\n')
    checks, launched = [], []
    def audit(root):
        checks.append(clock[0].isoformat())
        return ('waiting', 'active') if len(checks) == 1 else ('complete', 'done')
    monkeypatch.setattr(monitor, 'audit_coco', audit)
    monkeypatch.setattr(monitor, 'readiness', lambda *a: ('ready', 'idle'))
    monkeypatch.setattr(monitor, 'launch_once', lambda *a: launched.append(clock[0].isoformat()))
    monkeypatch.setattr(monitor.sys, 'argv', ['monitor', '--watch-run', str(watched),
                        '--monitor-output', str(output), '--next-output', str(next_output),
                        '--next-session', 'test', '--start-at', '2026-09-16T00:00:00+09:00'])
    monitor.main()
    assert checks == ['2026-09-16T00:00:00+09:00', '2026-09-16T00:30:00+09:00']
    assert launched == ['2026-09-16T00:30:00+09:00']
    assert json.loads((output / 'status.json').read_text())['state'] == 'launched'
