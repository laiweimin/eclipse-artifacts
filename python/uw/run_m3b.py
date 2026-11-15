"""R5-verdict M3 closure (2026-09-29 reviewer round 5): (a) correct
Clopper-Pearson exact binomial intervals (the previous 'exact' label
was Wald clipped), (b) a clean, theorem-backed split-conformal
experiment on the 30 archived ft-family honest models (fixed disjoint
fit/cal/test assignment + random-reshuffle robustness), (c) per-request
breakdown of the Phase-2 cross-draw results. Pure CPU, reads unmodified
m3/m3p2 metrics caches; no new training, no GPU.
"""
import glob
import json
import math
import os
import time

import numpy as np

METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
ALPHAS = (0.20, 0.10)
FIXED_SPLIT = {'fit': ['ew9f'], 'cal': ['p1aud', 'e2'],
               'test': ['e3', 'e4']}
N_RESHUFFLE = 200


def binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_cdf(k, n, p):
    return sum(binom_pmf(i, n, p) for i in range(k + 1))


def cp_interval(k, n, level=0.95):
    """Clopper-Pearson exact interval. CDF(k,p)=P(X<=k) is DECREASING
    in p. Lower solves CDF(k-1, p) = 1 - a/2; upper solves CDF(k, p) =
    a/2."""
    a = (1 - level) / 2
    if k == 0:
        lo = 0.0
    else:
        l, h = 0.0, 1.0
        for _ in range(200):
            m = (l + h) / 2
            if binom_cdf(k - 1, n, m) > 1 - a:
                l = m
            else:
                h = m
        lo = (l + h) / 2
    if k == n:
        hi = 1.0
    else:
        l, h = 0.0, 1.0
        for _ in range(200):
            m = (l + h) / 2
            if binom_cdf(k, n, m) > a:
                l = m
            else:
                h = m
        hi = (l + h) / 2
    return lo, hi


def conformal_q(scores, alpha):
    s = sorted(scores)
    idx = math.ceil((len(s) + 1) * (1.0 - alpha))
    return s[idx - 1] if idx <= len(s) else float('inf')


def loo_median(v):
    v = np.asarray(v, float)
    return np.array([np.median(np.delete(v, i)) for i in range(len(v))])


def main():
    rng = np.random.RandomState(47301)
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    tags = sorted(ens_mm)

    outdir = os.path.join('research_outputs', 'uw',
                          'm3b-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)

    def split_scores(fit_tags, cal_tags, test_tags):
        fit = [e for t in fit_tags for e in ens_mm[t]]
        c = {m: float(np.median([e[m] for e in fit])) for m in METRICS}
        sig = {m: float(np.median(np.abs(
            np.array([e[m] for e in fit]) - c[m]))) for m in METRICS}

        def sc(e):
            return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                       for m in METRICS)
        cal_s = [sc(e) for e in (x for t in cal_tags for x in ens_mm[t])]
        out = {}
        for alpha in ALPHAS:
            q = conformal_q(cal_s, alpha)
            n_cal = len(cal_s)
            out[alpha] = {
                'q': q, 'n_cal': n_cal,
                'test': [{'tag': t, 's': sc(e),
                          'reject': sc(e) > q}
                         for t in test_tags for e in ens_mm[t]]}
        # resource-matched baseline: max-dev bands from the SAME fit set
        bands = {m: float(max(abs(e[m] - c[m]) for e in fit))
                 for m in METRICS}
        base_rej = []
        for t in test_tags:
            for e in ens_mm[t]:
                base_rej.append(any(abs(e[m] - c[m]) > bands[m]
                                    for m in METRICS))
        return out, base_rej

    # ---- fixed split ----------------------------------------------------
    res, base = split_scores(FIXED_SPLIT['fit'], FIXED_SPLIT['cal'],
                             FIXED_SPLIT['test'])
    fixed = {}
    for alpha in ALPHAS:
        r = res[alpha]
        k = sum(1 for t in r['test'] if t['reject'])
        n = len(r['test'])
        lo, hi = cp_interval(k, n)
        fixed['alpha_{:.2f}'.format(alpha)] = {
            'k': k, 'n': n, 'fpr': k / n, 'cp95': [lo, hi],
            'q': r['q'], 'n_cal': r['n_cal'],
            'per_test': r['test']}
    fixed['baseline_fitbudget_fpr'] = float(np.mean(base))
    fixed['baseline_k_n'] = [int(sum(base)), len(base)]
    print('FIXED SPLIT: a0.20 {}/{} CP{:.3f}-{:.3f} | a0.10 {}/{} '
          'CP{:.3f}-{:.3f} | baseline {}/{}'.format(
              fixed['alpha_0.20']['k'], fixed['alpha_0.20']['n'],
              *fixed['alpha_0.20']['cp95'],
              fixed['alpha_0.10']['k'], fixed['alpha_0.10']['n'],
              *fixed['alpha_0.10']['cp95'],
              fixed['baseline_k_n'][0], fixed['baseline_k_n'][1]),
          flush=True)

    # ---- reshuffle robustness (descriptive; each split valid alone) -----
    allmods = [(t, e) for t in tags for e in ens_mm[t]]
    resha = {('{:.2f}'.format(a)): [] for a in ALPHAS}
    base_rs = []
    for it in range(N_RESHUFFLE):
        perm = rng.permutation(len(allmods))
        fit = [allmods[i] for i in perm[:6]]
        cal = [allmods[i] for i in perm[6:18]]
        test = [allmods[i] for i in perm[18:]]
        c = {m: float(np.median([e[m] for _, e in fit]))
             for m in METRICS}
        sig = {m: float(np.median(np.abs(
            np.array([e[m] for _, e in fit]) - c[m])))
               for m in METRICS}

        def sc(e):
            return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                       for m in METRICS)
        cal_s = [sc(e) for _, e in cal]
        bands = {m: float(max(abs(e[m] - c[m]) for _, e in fit))
                 for m in METRICS}
        base_rs.append(np.mean([
            any(abs(e[m] - c[m]) > bands[m] for m in METRICS)
            for _, e in test]))
        for alpha in ALPHAS:
            q = conformal_q(cal_s, alpha)
            resha['{:.2f}'.format(alpha)].append(np.mean(
                [sc(e) > q for _, e in test]))
    rob = {'n_splits': N_RESHUFFLE,
           'baseline_fitbudget_mean': float(np.mean(base_rs))}
    for a, v in resha.items():
        v = np.array(v)
        rob['alpha_{}'.format(a)] = {
            'mean': float(v.mean()),
            'q05': float(np.quantile(v, 0.05)),
            'q95': float(np.quantile(v, 0.95)),
            'frac_at_or_below_alpha': float(np.mean(v <= float(a)))}
        print('RESHUFFLE alpha', a, 'mean', round(float(v.mean()), 3),
              '5-95%', round(float(np.quantile(v, 0.05)), 3),
              round(float(np.quantile(v, 0.95)), 3), flush=True)

    # ---- phase 2 per-request breakdown ----------------------------------
    p2 = sorted(glob.glob(
        'research_outputs/uw/m3p2-2*/verdict.json'))[-1]
    p2d = json.load(open(p2))
    per_draw = {}
    for alpha in ALPHAS:
        for hold in ('F1', 'F2', 'F3', 'F4'):
            rows = p2d['honest_loeo']['{:.2f}|{}'.format(alpha, hold)]
            k = sum(1 for r in rows if r['reject'])
            per_draw['{:.2f}|{}'.format(alpha, hold)] = {
                'k': k, 'n': len(rows)}
    eq = {a: float(np.mean([per_draw['{:.2f}|{}'.format(a, h)]['k']
                            / per_draw['{:.2f}|{}'.format(a, h)]['n']
                            for h in ('F1', 'F2', 'F3', 'F4')]))
          for a in ALPHAS}
    print('PHASE2 per-draw a0.20:',
          {h: per_draw['0.20|{}'.format(h)] for h in ('F1', 'F2', 'F3',
                                                      'F4')},
          'request-equal-weight', round(eq[0.20], 3), flush=True)

    out = {'fixed_split': fixed, 'robustness': rob,
           'phase2_per_draw': per_draw,
           'phase2_request_equal_weight': eq,
           'cp_reference': 'Clopper-Pearson 95% (previous labels were '
                           'Wald clipped; superseded)'}
    json.dump(out, open(os.path.join(outdir, 'm3b.json'), 'w'),
              indent=1, default=float)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
