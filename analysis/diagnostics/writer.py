"""Exclusive run creation and append-only epoch diagnostics under results/."""

import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def result_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to((REPO / 'results').resolve()) or path == (REPO / 'results').resolve():
        raise ValueError('Diagnostic outputs must be strictly inside repository results/')
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f'{name}:{t.dtype}:{tuple(t.shape)}'.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    with result_path(path).open('x') as stream:
        json.dump(clean_json(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def append_csv(path, rows, fields):
    path = result_path(path)
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(clean_json(rows))


class ProbeWriter:
    def __init__(self, output, manifest):
        self.output = result_path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(manifest, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip()),
                        command=shlex.join([sys.executable, *sys.argv]), torch_version=torch.__version__)
        write_json(self.output / 'manifest.json', manifest)
        (self.output / 'commands.sh').write_text(manifest['command'] + '\n')
        self.epochs = set()

    def begin(self, epoch):
        if epoch in self.epochs:
            raise FileExistsError(f'Probe epoch {epoch} already attempted')
        (self.output / f'epoch_{epoch:03d}.STARTED').touch(exist_ok=False)
        self.epochs.add(epoch)

    def finish(self, epoch, rows, wide, metadata, spectra, autocorrelation, centers, counts):
        fields = ['epoch', 'global_step', 'metric', 'layer', 'class_id', 'mean', 'std', 'n_valid', 'n_total']
        append_csv(self.output / 'metrics.csv', rows, fields)
        append_csv(self.output / 'metrics_wide.csv', [wide], list(wide))
        np.save(self.output / f'spectrum_{epoch:03d}.npy', spectra)
        np.savez_compressed(self.output / f'spectral_support_{epoch:03d}.npz', autocorrelation=autocorrelation, frequency_centers=centers, bin_counts=counts)
        write_json(self.output / f'epoch_{epoch:03d}.json', metadata)
        (self.output / f'epoch_{epoch:03d}.COMPLETE').touch(exist_ok=False)
