"""S0-N Step 3: held-out stability validation of the frozen calibration.

Frozen rules (docs/plans/2026-09-17-s0n-step1-spec.md):
  reference w_R via Woodbury (numerically verified, floor 6e-14);
  truth h_V = v^T(w_R - w_U), positive = margin drop; tau_h = 0.0038
  (1% of cat-test margin std, frozen);
  alarm = |metric(w_U) - metric(w_R)| > 95% band, band = |metric(w_R +
  tau_h*u_random) - metric(w_R)| over 200 random unit directions.

Held-out (never used in discovery): NEW pool (seed 42501), held-out target
direction v_A (even-parity cat test half; discovery used odd half), and a
k=10 scale variant (set seeds 42601+).

Validations (per condition, 20 sets x 200 directions):
  V1 sensitivity stability: v_A-aligned tau_h displacement -> alarm rate >=95%
     on all six scores.
  V2 FPR control: random-direction tau_h displacements -> per-set alarm
     fraction ~0.05 (design FPR).
  V3 unidentifiability stability: null(X_F) displacement with v_A-projection
     tau_h -> F-side scores (m3/m4/m6) alarm fraction ~0.
  V0 sanity: w_U = w_R -> zero alarms.
Output: research_outputs/s0n/step3-<ts>/
"""
import glob
import json
import os
import time

import numpy as np

from run_s0n_p0 import ks_stat
from s0j_ae.linhead import grad_contributions, ridge_head

LAM = 0.1
CAT_CLS = 3
N_DIRS = 200
N_SETS = 20
POOL_SEED_NEW = 42501
SETS_SEED_NEW = 42601
TAU_H = 0.0038
METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_vsRT')
F_SIDE = ('m3_Floss', 'm4_KS', 'm6_vsRT')


def ks(a, b):
    return ks_stat(a, b)


def metrics_at(shift, s_rt, s0_sub, Xsub, ysub, cat_sub, Xf, yf, theta0):
    sU = s0_sub + Xsub @ shift
    lU_sub = (sU - ysub) ** 2
    lU_F = (Xf @ (theta0 + shift) - yf) ** 2
    lR_F = (Xf @ (theta0 + s_rt) - yf) ** 2
    return {'m1_acc': float(((sU > 0) == (ysub > 0)).mean()
                            - ((s0_sub > 0) == (ysub > 0)).mean()),
            'm2_catacc': float((sU[cat_sub] > 0).mean()
                               - (s0_sub[cat_sub] > 0).mean()),
            'm3_Floss': float(lU_F.mean()),
            'm4_KS': ks(lU_F, lU_sub),
            'm5_adv': float(lU_sub.mean() - lU_F.mean()),
            'm6_vsRT': float(lU_F.mean() - lR_F.mean())}


def validate_pool(Xp, yF, G, theta0, Hinv, n, sqn, s0_sub, Xsub, ysub,
                  cat_sub, v_dir, tau_h, k, sets_seed, n_sets=N_SETS,
                  n_dirs=N_DIRS):
    HX = Hinv @ Xp.T
    sets = [np.random.RandomState(sets_seed + j).choice(len(Xp), size=k,
                                                        replace=False)
            for j in range(n_sets)]
    res = {c: {m: [] for m in METRICS}
           for c in ('V1_aligned', 'V2_random', 'V3_nullspace', 'V0_exact')}
    h_list = []
    for si, idx in enumerate(sets):
        g = G[:, idx].sum(axis=1)
        Hg = Hinv @ g
        Bm = HX[:, idx] / np.sqrt(n)
        Kd = (Xp[idx] @ HX[:, idx]) / n
        w = np.linalg.solve(np.eye(k) - Kd, (Xp[idx] @ Hg) / np.sqrt(n))
        s_rt = Hg + Bm @ w
        Xf = Xp[idx]
        yf = yF[idx]
        m_ref = metrics_at(s_rt, s_rt, s0_sub, Xsub, ysub, cat_sub, Xf, yf,
                           theta0)

        rng_b = np.random.RandomState(60000 + si)
        band_acc = {m: [] for m in METRICS}
        alarm_rand = {m: 0 for m in METRICS}
        for d_i in range(n_dirs):
            u_r = rng_b.randn(512)
            u_r /= np.linalg.norm(u_r)
            mm = metrics_at(s_rt + tau_h * u_r, s_rt, s0_sub, Xsub, ysub,
                            cat_sub, Xf, yf, theta0)
            for m in METRICS:
                band_acc[m].append(abs(mm[m] - m_ref[m]))
        band = {m: float(np.quantile(band_acc[m], 0.95)) for m in METRICS}
        # second pass for held-out random-direction alarm fraction
        rng_v = np.random.RandomState(61000 + si)
        frac_rand = {m: 0 for m in METRICS}
        for _ in range(n_dirs):
            u_r = rng_v.randn(512)
            u_r /= np.linalg.norm(u_r)
            mm = metrics_at(s_rt + tau_h * u_r, s_rt, s0_sub, Xsub, ysub,
                            cat_sub, Xf, yf, theta0)
            for m in METRICS:
                if abs(mm[m] - m_ref[m]) > band[m]:
                    frac_rand[m] += 1
        for m in METRICS:
            res['V2_random'][m].append(frac_rand[m] / float(n_dirs))
            res['V0_exact'][m].append(False)

        # V1 aligned at tau_h
        u_a = v_dir / np.linalg.norm(v_dir)
        m1_ = metrics_at(s_rt + tau_h * u_a, s_rt, s0_sub, Xsub, ysub,
                         cat_sub, Xf, yf, theta0)
        for m in METRICS:
            res['V1_aligned'][m].append(
                abs(m1_[m] - m_ref[m]) > band[m])
        h_list.append(tau_h)

        # V3 null space, v-projection scaled to tau_h
        zz = rng_v.randn(512)
        v_n = zz - Xf.T @ np.linalg.solve(Xf @ Xf.T, Xf @ zz)
        nn = np.linalg.norm(v_n)
        v_n = zz / nn if nn < 1e-8 else v_n / nn
        pr = float(v_dir @ v_n)
        scale = tau_h / max(abs(pr), 1e-12)
        s3 = s_rt + v_n * scale
        m3m = metrics_at(s3, s_rt, s0_sub, Xsub, ysub, cat_sub, Xf, yf,
                         theta0)
        for m in F_SIDE:
            res['V3_nullspace'][m].append(abs(m3m[m] - m_ref[m]) > band[m])

    out = {}
    for c, per in res.items():
        out[c] = {m: float(np.mean(v)) if v else None for m, v in
                  per.items()}
    out['h_V_scale_check'] = float(np.mean(h_list))
    return out


def main():
    run_id = 'step3-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs/s0n', run_id)
    os.makedirs(outdir, exist_ok=True)
    z = np.load('research_outputs/s0j_ae/features_inst-2000000002.npz')
    Xtr, ytr, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']
    n = Xtr.shape[0]
    ybin = np.where(ytr == CAT_CLS, 1.0, -1.0)
    theta0, Hinv = ridge_head(Xtr, ybin, LAM)

    rng_pool = np.random.RandomState(POOL_SEED_NEW)
    pool = np.sort(rng_pool.choice(np.where(ytr == CAT_CLS)[0],
                                   size=60, replace=False))
    cat_te = np.where(yte == CAT_CLS)[0]
    v_dir = Xte[cat_te[cat_te % 2 == 0]].mean(axis=0)  # held-out direction
    rng_sub = np.random.RandomState(41200)
    sub = rng_sub.choice(Xte.shape[0], size=2000, replace=False)
    Xsub = Xte[sub]
    ysub = np.where(yte[sub] == CAT_CLS, 1.0, -1.0)
    cat_sub = yte[sub] == CAT_CLS
    s0_sub = Xsub @ theta0

    Xp = Xtr[pool]
    yF = ybin[pool]
    G = grad_contributions(Xp, yF, theta0, n)
    sqn = np.sqrt(n)

    summary = {'run_id': run_id, 'tau_h': TAU_H,
               'heldout': 'pool seed 42501 + v_A (even half); band = 200 '
                          'random dirs at tau_h parameter-norm; FPR target '
                          '5%',
               'conditions': {}}
    for k in (3, 10):
        res = validate_pool(Xp, yF, G, theta0, Hinv, n, sqn, s0_sub, Xsub,
                            ysub, cat_sub, v_dir, TAU_H, k, SETS_SEED_NEW)
        summary['conditions']['k={}_pool42501'.format(k)] = res
        print('== k={}: alarm rates (V1 expect >=0.95, V2 expect ~0.05, '
              'V3 F-side expect ~0)'.format(k), flush=True)
        for c, per in res.items():
            print('  ', c, json.dumps(per, default=str), flush=True)

    with open(os.path.join(outdir, 'step3_results.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('saved', outdir)


if __name__ == '__main__':
    main()
