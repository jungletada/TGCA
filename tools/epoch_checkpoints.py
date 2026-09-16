"""Opt-in immutable epoch snapshots; serialization must not consume training RNG."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class EpochCheckpointWriter:
    """One end-of-epoch file plus hash index. Never replace an existing snapshot.

    State is captured after scheduler.step(epoch) and validation, before the next
    iterator is created. Worker RNG state is not captured: no bitwise resume claim.
    The legacy trainer's --resume behavior is deliberately not changed here.
    """
    def __init__(self, work_space, model_name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', model_name):
            raise ValueError('Invalid model filename')
        self.directory = Path(work_space) / 'epoch_checkpoints'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model_name = model_name
        self.provenance = dict(
            git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            command=[sys.executable, *sys.argv],
            snapshot_timing='after training, scheduler step, validation and best update',
            exact_resume_supported=False,
            resume_note='State saved for analysis/recovery; legacy --resume not extended; worker RNG not captured.')

    def save(self, payload, optimizer, scheduler, scaler, metrics, args, max_accuracy):
        epoch = int(payload['epoch'])
        if epoch < 0:
            raise ValueError('Negative epoch')
        path = self.directory / f'{self.model_name}_epoch_{epoch + 1:03d}.pth'
        if path.exists():
            raise FileExistsError(path)
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
        snapshot = dict(payload, optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), rng_state=rng, args=dict(vars(args)),
                        metrics=metrics, max_accuracy=float(max_accuracy), epoch_one_based=epoch+1,
                        snapshot_provenance=self.provenance)
        # Hard-link publication is atomic AND refuses overwrites. Temporary file
        # is created in the destination filesystem; no destructive rename.
        fd, temporary = tempfile.mkstemp(prefix='.epoch-', suffix='.tmp', dir=self.directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(snapshot, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        row = dict(epoch=epoch, epoch_one_based=epoch+1, filename=path.name,
                   sha256=sha256(path), bytes=path.stat().st_size)
        with (self.directory / 'index.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return path
