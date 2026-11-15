"""Offline bootstrap CIs for the S37 aligned-instrument medians (R04 optional).

Reads a run_align verdict.json that saved `gains` (12 paired gains) and
`bases`. The 12 finals derive from 6 independent starts (CE i and GR i share
start i), so the primary interval is a CLUSTER bootstrap over the 6 start
clusters; a naive record-level bootstrap over the 12 gains is reported for
comparison. Fixed seed; percentile method; 10k resamples.
"""
import argparse
import json

import numpy as np

KEYS = ('fresh_raw', 'ret_raw', 'fresh_z', 'ret_z')
ARCHIVED = 'research_outputs/uw/align-20261112T185915/verdict.json'


def med(v):
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('verdict')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=13000)
    ap.add_argument('--archived', default=ARCHIVED)
    cli = ap.parse_args()

    v = json.load(open(cli.verdict))
    g = {k: np.array([r[k] for r in v['gains']]) for k in KEYS}
    n = len(v['gains'])
    assert n == 12, n
    # cluster k = {gain of CE row k, gain of GR row k}, k = i % 6
    cl = {k: np.stack([g[k][:6], g[k][6:]]) for k in KEYS}

    rng = np.random.RandomState(cli.seed)
    boot_c = {k: np.empty(cli.B) for k in KEYS}
    boot_r = {k: np.empty(cli.B) for k in KEYS}
    for b in range(cli.B):
        cs = rng.randint(0, 6, 6)
        rs = rng.randint(0, n, n)
        for k in KEYS:
            boot_c[k][b] = med(cl[k][:, cs].ravel())
            boot_r[k][b] = med(g[k][rs])

    out = {'verdict_path': cli.verdict,
           'archived_medians': None,
           'medians': {k: med(g[k]) for k in KEYS},
           'cluster_ci95': {k: [float(np.percentile(boot_c[k], 2.5)),
                                float(np.percentile(boot_c[k], 97.5))]
                            for k in KEYS},
           'record_ci95': {k: [float(np.percentile(boot_r[k], 2.5)),
                               float(np.percentile(boot_r[k], 97.5))]
                           for k in KEYS},
           'cluster_ci_excludes_zero': {
               k: bool(np.percentile(boot_c[k], 2.5) > 0
                       or np.percentile(boot_c[k], 97.5) < 0)
               for k in KEYS},
           'B': cli.B, 'seed': cli.seed}
    try:
        a = json.load(open(cli.archived))
        out['archived_medians'] = a['median_gains']
        out['median_abs_dev_vs_archived'] = {
            k: abs(med(g[k]) - a['median_gains'][k]) for k in KEYS}
    except (OSError, KeyError):
        pass
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
