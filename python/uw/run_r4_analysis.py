"""R4-verdict analysis (2026-09-29 third-round review): recompute the
symmetric-band cap vs the honest-luck line, per unit, from unmodified
artifacts (CPU only).

The review's counterexample: with c = median honest fm, b+ = max-c,
b- = c-min, b = max(b+, b-), E = fm0 - c, the luck line is L = b+/E
but the symmetric band |fm-c| <= b caps elevation at gamma = b/E.
Slack beyond luck = (b- - b+)_+/E. Recompute, per unit:
  - b+, b-, gamma(band cap), luck, slack
  - washes: elevation, exceeds-luck?, exceeds-band? (fm > c + b)
  - cheap-manipulation feasible points (P2 rows): same two tests.
No training.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = sorted(glob.glob(os.path.join(
    ROOT, 'research_outputs/uw/r2-2*/r2_analysis.json')))[-1]
E3 = 'research_outputs/uw/ew3-20260925T113900'
RUN3 = os.path.join(ROOT, E3)
pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k']['30']
FM0 = pt['m6_fm']


def band_stats(fms, med, E):
    fms = list(fms)
    c = float(med)
    bp = max(fms) - c
    bm = c - min(fms)
    b = max(bp, bm)
    return dict(c=c, b_plus=bp, b_minus=bm, b=b, E=E,
                gamma=b / E, luck=bp / E, slack=max(bm - bp, 0.0) / E,
                hi=c + b, fmax=max(fms))


def main():
    r2 = json.load(open(R2))
    out = {'units': {}, 'cheap': {}}
    for key, u in r2['units'].items():
        fms = list(u['fm_seeds']) if isinstance(u['fm_seeds'], list) \
            else list(u['fm_seeds'].values())
        st = band_stats(fms, u['med'], u['E'])   # per-unit med/E stored
        # sanity: r2's luck_max must equal b+/E
        assert abs(st['luck'] - u['luck_max']) < 1e-6, key
        washes = [{'fm': w['fm'],
                   'elev': (w['fm'] - st['c']) / st['E'],
                   'exceeds_luck': w['fm'] > st['c'] + st['b_plus'],
                   'exceeds_band': w['fm'] > st['hi']}
                  for w in u['wash']]
        out['units'][key] = dict(st, wash=washes,
                                 n_exceed_luck=sum(1 for w in washes
                                                   if w['exceeds_luck']),
                                 n_exceed_band=sum(1 for w in washes
                                                   if w['exceeds_band']))
    # P2 cheap points against the k30-A provider band threshold
    p2 = json.load(open(sorted(glob.glob(os.path.join(
        ROOT, 'research_outputs/uw/r3p2-2*/p2_results.json')))[-1]))
    prov = out['units']['t44500_k30']
    feas = [r for r in p2['rows'] if r['feasible']]
    out['cheap'] = {
        'n_feasible': len(feas),
        'n_exceed_luck': sum(1 for r in feas
                             if r['fm'] > prov['c'] + prov['b_plus']),
        'n_exceed_band': sum(1 for r in feas
                             if r['fm'] > prov['hi']),
        'best_elev_feasible': max(
            (r['fm'] - prov['c']) / prov['E'] for r in feas)}
    print('%-14s %6s %6s %6s %6s  luckX bandX' % ('unit', 'gamma', 'luck',
                                                  'slack', 'washElev'))
    for k, u in out['units'].items():
        best = max(w['elev'] for w in u['wash'])
        print('%-14s %.3f  %.3f  %.3f  %.3f   %d/6  %d/6'
              % (k, u['gamma'], u['luck'], u['slack'], best,
                 u['n_exceed_luck'], u['n_exceed_band']))
    print('cheap: feasible %d, exceed-luck %d, exceed-BAND %d'
          % (out['cheap']['n_feasible'], out['cheap']['n_exceed_luck'],
             out['cheap']['n_exceed_band']))
    outdir = os.path.join(ROOT, 'research_outputs', 'uw',
                          'r4-' + __import__('time').strftime(
                              '%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, 'r4_analysis.json'), 'w'),
              indent=1)
    print('OUT', outdir)


if __name__ == '__main__':
    main()
