"""Summarize the existing native-last3 readout from frozen extracted maps.

This is the actual pooling readout, not an additional pooling variant.
No inference or source-file edits are performed.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.c2p_pooling_attention import METRICS, map_metrics, paired_bootstrap, sha256


def run(root):
    root = Path(root)
    output = root / 'native_last3_summary.csv'
    if output.exists():
        raise FileExistsError(output)
    arrays, hashes = [], {}
    reference_ids = None
    for name in ['gwrp', 'c2p']:
        source = root / f'{name}_positive_attention.npz'
        hashes[str(source)] = sha256(source)
        with np.load(source) as archive:
            offsets, ids = archive['offsets'], archive['image_ids']
            if reference_ids is None:
                reference_ids = ids
                reference_offsets = offsets
            else:
                np.testing.assert_array_equal(ids, reference_ids)
                np.testing.assert_array_equal(offsets, reference_offsets)
            # Average RAW attention across L10-L12 first, THEN conditionalize.
            # This is not the equally weighted average of conditional layer maps.
            attention = archive['attention'][-3:].mean(0)
            arrays.append(np.array([map_metrics(attention[a:b])
                                    for a, b in zip(offsets[:-1], offsets[1:]) if b-a >= 2]))
        assert sha256(source) == hashes[str(source)]
    means, ci, counts = paired_bootstrap(np.array(arrays))
    rows = []
    for i, metric in enumerate(METRICS):
        row = dict(stratum='multi', readout='native-last3', metric=metric, n_images=int(counts[i]))
        for m, name in enumerate(['gwrp', 'c2p', 'delta']):
            row.update({name: means[m, i], name+'_lo': ci[0, m, i], name+'_hi': ci[1, m, i]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)
    (root / 'native_last3_manifest.json').write_text(json.dumps({
        'command': shlex.join([sys.executable, '-m', 'analysis.c2p_pooling_attention_readout'] + sys.argv[1:]),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_hashes_unchanged': hashes, 'bootstrap_replicates': 5000,
        'bootstrap_seed': 20260914, 'bootstrap_unit': 'paired image',
    }, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
