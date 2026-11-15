"""UW E-W3a: audit-band calibration curve from archived E-W3 evals (zero
new compute; decision-support for the 2026-09-23 stress-test plan (a)).

Claim under test (E-W3 audit-validity finding): acceptance bands calibrated
on k=0 (full-data retrain) stochasticity under-cover the honest retrain
ensemble dispersion at deletion scale -- dispersion must be measured per k.
Inputs: research_outputs/uw/ew3-20260925T113900/ (noise_s*.json for k=0,
rt_k*_s*.json for k in {30,300,1000}, pt.json for theta0 context).
Contrast: S0-K archived test-side dispersion (flat with k, 10 seeds x 4 k,
research_outputs/s0k/varcurve-*/results.json) -- quoted for context.

Output: calibration_curve.json next to the inputs + printed table.
Honest limits: 6 seeds at k=0, 3 seeds per deletion k (small-sample
dispersion estimates); F-side metrics evaluated on the E-W3 frozen F_k /
subsample; test-side m1/m2 from the same evals.
"""
import glob
import json
import os

import numpy as np

RUN = 'research_outputs/uw/ew3-20260925T113900'
S0K = sorted(glob.glob('research_outputs/s0k/varcurve-*/results.json'))
METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')


def load_per_k(k):
    if k == 0:
        files = sorted(glob.glob(os.path.join(RUN, 'noise_s*.json')))
    else:
        files = sorted(glob.glob(os.path.join(RUN,
                                              'rt_k{}_s*.json'.format(k))))
    return [json.load(open(f))['per_k'][str(k if k else 30)]
            for f in files]          # k=0 jsons keyed by first k grid entry
    # NOTE: noise jsons store per_k under every k key with IDENTICAL
    # F-side inputs (the same F_k from the k grid) -- for k=0 models the
    # F-side stats measure "full-data model fit on F_k", which is exactly
    # the band source.  We take the k=30 entry for the k=0 dispersion.


def main():
    ks = [0, 30, 300, 1000]
    bands = json.load(open(os.path.join(RUN, 'bands.json')))
    table = {}
    for k in ks:
        entries = load_per_k(k)
        disp = {}
        for m in METRICS:
            v = np.array([e[m] for e in entries], dtype=float)
            disp[m] = {'max_dev': float(np.max(np.abs(v - np.median(v)))),
                       'std': float(v.std()),
                       'range': float(v.max() - v.min()),
                       'n_seeds': len(v)}
        table[str(k)] = disp
    pt = json.load(open(os.path.join(RUN, 'pt.json')))['per_k']
    # ratio per deletion k = honest theta_R dispersion on F_k  /  k=0 noise
    # band on the SAME F_k (bands.json) -- the exact quantity whose
    # under-coverage produced the E-W3 honest-rejection FPRs
    ratio = {}
    for k in (30, 300, 1000):
        ratio[str(k)] = {
            m: round(table[str(k)][m]['max_dev']
                     / max(bands[str(k)][m], 1e-12), 3) for m in METRICS}
    curve = {'run': RUN, 'metrics': METRICS,
             'dispersion_by_k': table,
             'ratio_thetaR_disp_over_k0_band': ratio,
             'theta0_dev_vs_k0_band': {
                 str(k): {m: round(abs(pt[str(k)][m]
                                        - float(np.median(
                                            [e[m] for e in load_per_k(k)]))),
                                  4) for m in METRICS} for k in (30, 300,
                                                                 1000)},
             'note': ('dispersion = max|dev from median| over available '
                      'seeds (band analog); k=0 has 6 seeds, deletion ks '
                      'have 3; denominators = bands.json (k=0 noise '
                      'dispersion on the SAME F_k)')}
    if S0K:
        acc = {}
        for f in S0K:
            for arm in json.load(open(f))['arms'].values():
                if arm.get('status') == 'done':
                    acc.setdefault(arm['k'], []).append(arm['acc'])
        curve['s0k_test_side_contrast'] = {
            'source': [os.path.relpath(f) for f in S0K],
            'acc_std_by_k': {str(k): round(float(np.std(v)), 5)
                             for k, v in sorted(acc.items())},
            'note': ('S0-K archived test-side accuracy std (10 seeds per '
                     'k, k in {0,500,2500,5000}) is FLAT in k -- the '
                     'contrast half of the calibration claim')}
    with open(os.path.join(RUN, 'calibration_curve.json'), 'w') as fh:
        json.dump(curve, fh, indent=1)
    print('k | ' + ' | '.join(METRICS) + '   (thetaR disp / k0 band)')
    for k in (30, 300, 1000):
        r = curve['ratio_thetaR_disp_over_k0_band'][str(k)]
        print('{} | '.format(k) + ' | '.join(
            '{:.2f}'.format(r[m]) for m in METRICS))
    if 's0k_test_side_contrast' in curve:
        print('S0-K test-side acc std by k:',
              curve['s0k_test_side_contrast']['acc_std_by_k'])
    print('saved', os.path.join(RUN, 'calibration_curve.json'))


if __name__ == '__main__':
    main()
