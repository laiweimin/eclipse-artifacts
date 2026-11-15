"""C-CRE P0: zero-training mechanism decomposition — per-instance s_J
distributions of adaptive finals, intersection space analysis.

Uses the A8R recomputation's 200 bootstrap instances. For each adaptive
final: (i) distribution of s_J across instances, (ii) survival rate,
(iii) signal retention if pulled to intersection center (analytical —
s_J scales as d^T S^-1 d, so halving d halves s_J quadratically).
Also: honest starts' dual headroom (how deep in the intersection are
they).
"""
import argparse, glob, json, os, time
import numpy as np
from uw.run_ey import METRICS, conformal_q

RHO = 0.5


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def main():
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = json.load(open(cache))['ens_mm']
    hon18 = [vec(e) for g in ('e2', 'e3', 'e4') for e in ens[g]]
    cal12 = [vec(e) for g in ('p1aud', 'e2') for e in ens[g]]
    rs = np.random.RandomState(44111)
    # rebuild 200 instances (same scheme as run_a8boot/a8r)
    instances = []
    for b in range(200):
        idx = rs.choice(len(hon18), 6, replace=True)
        X = np.stack([hon18[i] for i in idx])
        cj = X.mean(0)
        S = np.cov(X, rowvar=False)
        Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
        try:
            Si = np.linalg.inv(Sh)
        except np.linalg.LinAlgError:
            continue
        qJ = conformal_q(
            [float((v - cj) @ Si @ (v - cj)) for v in cal12], 0.20)
        instances.append((cj, Si, qJ))
    print('rebuilt %d instances' % len(instances))

    # load A8R verdict for adaptive finals' metric vectors
    a8r = sorted(glob.glob(
        'research_outputs/uw/*a8r*/verdict.json'))
    if not a8r:
        print('A8R verdict not found'); return
    a8r_d = json.load(open(a8r[-1]))
    # load a8mech for raw metric vectors (it stored std_components;
    # but a8r stored per_final with s6/sJ but not raw vectors)
    # Instead use the a8mech rows: they have std_components = |d|/sig
    # We need raw d for each final. a8mech computed d from eval_full.
    # Rebuild from a8mech's stored values is lossy (abs values only).
    # Instead: use the honest 18 + original 12 from a8mech rows.
    mech = sorted(glob.glob(
        'research_outputs/uw/a8mech-*/verdict.json'))[-1]
    mech_d = json.load(open(mech))
    # mech rows have std_components; need signed d for s_J under
    # different c_j/Si. Can't recover sign. Use main-instance s_J
    # values from a8r verdict instead and analyze distributions.

    per_f = a8r_d.get('per_final', {})
    print('%-22s %-14s %7s %7s %7s %7s %7s' %
          ('tag', 'family', 'sJ_main', 'boot_pass', 'sJ_med',
           'sJ_p95', 'gap_to_qJ'), flush=True)
    rows = []
    for tag in sorted(per_f):
        r = per_f[tag]
        if 'joint' not in r['family'] and 'orig' not in r['family']:
            continue
        # sJ_main is from main instance; boot_pass_rate from 200
        # We don't have per-instance sJ stored — only pass rates.
        # Analyze what we have: pass rate + main sJ.
        rows.append(r)
        print('%-22s %-14s %7.3f %7.2f' %
              (tag, r['family'], r['sJ'], r['boot_pass_rate']),
              flush=True)

    # honest starts' depth in intersection
    fit_e = ens['ew9f']
    X_fit = np.stack([vec(e) for e in fit_e])
    c_main = X_fit.mean(0)
    S_main = np.cov(X_fit, rowvar=False)
    Sh_main = (1 - RHO) * S_main + RHO * np.diag(np.diag(S_main))
    Si_main = np.linalg.inv(Sh_main)
    qJ_main = conformal_q(
        [float((vec(e) - c_main) @ Si_main @ (vec(e) - c_main))
         for e in cal12], 0.20)

    print('\n--- Honest starts under main instance ---')
    for g in ('e3', 'e4'):
        for i, e in enumerate(ens[g]):
            d = vec(e) - c_main
            sj = float(d @ Si_main @ d)
            pass_all = sum(1 for cj_, Si_, qJ_ in instances
                           if float((vec(e) - cj_) @ Si_
                                    @ (vec(e) - cj_)) <= qJ_)
            print(' %s-%d sJ_main=%7.3f pass200=%3d/200' %
                  (g, i, sj, pass_all), flush=True)

    # intersection space analysis: for each pair of instances,
    # correlation of s_J rankings
    print('\n--- Instance pair correlation of s_J (top 20 pairs) ---')
    tags = [t for t in sorted(per_f) if 'joint' in per_f[t]['family']
            or 'orig' in per_f[t]['family']]
    if len(tags) > 2:
        # Use the honest starts' sJ across instances as proxy
        hon_sj = {}
        for g in ('e3', 'e4'):
            for i, e in enumerate(ens[g]):
                v = vec(e)
                hon_sj['%s-%d' % (g, i)] = [
                    float((v - cj_) @ Si_ @ (v - cj_))
                    for cj_, Si_, _ in instances]
        names = sorted(hon_sj)
        M = np.array([hon_sj[n] for n in names])
        corr = np.corrcoef(M)
        n = len(names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((corr[i, j], names[i], names[j]))
        pairs.sort()
        print('sJ cross-instance rank correlations (honest models):')
        print('  min=%.3f median=%.3f max=%.3f' %
              (pairs[0][0], pairs[len(pairs)//2][0], pairs[-1][0]))
        print('  lowest:', pairs[:3])
        print('  highest:', pairs[-3:])

    print('\nDone.')


if __name__ == '__main__':
    main()
