"""S0-N P0: metric faithfulness audit on the existing k=3 enumeration.

For every deletion set F (34,220, pool of 60 cat records) we have the exact
retrain shift (Woodbury) and the approximate unlearned models (guo one-step
P=H_D^{-1}; fisher damped diagonal). P0 computes the standard unlearning
certification metrics on each approximate model and asks the meta-evaluation
question: when the unlearning actually damages the target direction (true
margin shift vs exact retrain), do the metrics notice?

Metrics (adapted to the ridge cat-vs-rest head; declared in the report):
  m1  overall-acc delta on a fixed 2000-sample test subsample
  m2  cat-acc delta on the same subsample
  m3  mean squared loss on F under theta_U           (forgetting amount)
  m4  KS statistic: F losses vs subsample losses under theta_U
      (TOFU-style forget quality, adapted; closed-form KS, no scipy)
  m5  LOSS-attack advantage: mean subsample loss - mean F loss under theta_U
  m6  mean F loss under theta_U - under theta_RT     (vs-retrain completeness)

Truth: d_true = v_b . (theta_U - theta_RT)  (cat-margin drop vs exact retrain,
positive = damage). Detection = AUC(|metric_U - metric_RT| -> d_true > q75).
Faithfulness = calibration of |metric error| against |d_true|.

Output: research_outputs/s0n/p0-<ts>/ (non-overwrite).
"""
import argparse
import itertools
import json
import os
import subprocess
import time

import numpy as np

from s0j_ae.linhead import grad_contributions, ridge_head

OUT_ROOT = 'research_outputs/s0n'
ENUM_DIR_GLOB = 'research_outputs/s0j_ae/s0j-enum-*'
LAM = 0.1
CAT_CLS = 3
SUB_N = 2000
SUB_SEED = 41200


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (closed form, no scipy)."""
    a = np.sort(a)
    b = np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / len(a)
    cdf_b = np.searchsorted(b, allv, side='right') / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def auc(scores, labels):
    """AUC via rank statistic (labels in {0,1})."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    summed = scores[order]
    i = 0
    while i < len(summed):
        j = i
        while j + 1 < len(summed) and summed[j + 1] == summed[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def sqloss(X, y, theta):
    r = X @ theta - y
    return r * r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default=0, type=int)
    cli = p.parse_args()
    run_id = 'p0-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(outdir, exist_ok=True)
    try:
        git_rev = subprocess.check_output(['git', 'rev-parse', '--short',
                                           'HEAD'], text=True).strip()
    except Exception:
        git_rev = 'unknown'

    device = torch_device(cli.gpu)
    z = np.load('research_outputs/s0j_ae/features_inst-2000000002.npz')
    Xtr, ytr, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']
    n = Xtr.shape[0]
    d = Xtr.shape[1]
    ybin = np.where(ytr == CAT_CLS, 1.0, -1.0)
    theta0, Hinv = ridge_head(Xtr, ybin, LAM)

    enum_dirs = sorted(glob_enum())
    ed = enum_dirs[-1]
    z = np.load(os.path.join(ed, 'enum_scatter.npz'))
    sets = z['sets']
    eB_guo_saved = z['e_guo_B']
    N = sets.shape[0]

    rng_pool = np.random.RandomState(41002)
    pool = np.sort(rng_pool.choice(np.where(ytr == CAT_CLS)[0],
                                   size=60, replace=False))
    Xp = Xtr[pool]
    G = grad_contributions(Xp, ybin[pool], theta0, n)
    HG = Hinv @ G
    HX = Hinv @ Xp.T
    sqn = np.sqrt(n)
    eye3 = np.eye(3)

    cat_te = np.where(yte == CAT_CLS)[0]
    v_b = Xte[cat_te[cat_te % 2 == 1]].mean(axis=0)

    rng_sub = np.random.RandomState(SUB_SEED)
    sub = rng_sub.choice(Xte.shape[0], size=SUB_N, replace=False)
    Xsub = Xte[sub]
    ysub = np.where(yte[sub] == CAT_CLS, 1.0, -1.0)
    cat_sub = yte[sub] == CAT_CLS

    # precompute theta0 scores
    s0_sub = Xsub @ theta0
    l0_sub = (s0_sub - ysub) ** 2
    acc0 = float(((s0_sub > 0) == (ysub > 0)).mean())
    cat0 = float(((s0_sub[cat_sub] > 0)).mean())
    b_l0 = float(l0_sub.mean())

    keys = ('d_true', 'm1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv',
            'm6_vsRT')
    out = {op: {k: np.zeros(N) for k in keys} for op in ('guo', 'fisher')}

    t0 = time.time()
    print('P0 over {} sets...'.format(N), flush=True)
    for s_i in range(N):
        idx = sets[s_i]
        g = G[:, idx].sum(axis=1)
        Hg = HG[:, idx].sum(axis=1)
        Bm = HX[:, idx] / sqn
        Kd = (Xp[idx] @ HX[:, idx]) / n
        w = np.linalg.solve(eye3 - Kd, (Xp[idx] @ Hg) / sqn)
        s_rt = Hg + Bm @ w
        s_guo = Hg
        df_diag = (Xp[idx] ** 2).sum(axis=0) / float(n)
        s_fish = g / (df_diag + LAM)
        Xf = Xp[idx]
        yf = ybin[pool][idx]

        for op, shift in (('guo', s_guo), ('fisher', s_fish)):
            sU_sub = s0_sub + Xsub @ shift
            sR_sub = s0_sub + Xsub @ s_rt
            lU_sub = (sU_sub - ysub) ** 2
            lR_sub = (sR_sub - ysub) ** 2
            accU = float(((sU_sub > 0) == (ysub > 0)).mean())
            accR = float(((sR_sub > 0) == (ysub > 0)).mean())
            catU = float((sU_sub[cat_sub] > 0).mean())
            catR = float((sR_sub[cat_sub] > 0).mean())
            lU_F = sqloss(Xf, yf, theta0 + shift)
            lR_F = sqloss(Xf, yf, theta0 + s_rt)
            ksU = ks_stat(lU_F, lU_sub)
            ksR = ks_stat(lR_F, lR_sub)
            o = out[op]
            o['d_true'][s_i] = -(v_b @ (shift - s_rt))  # positive = margin drop = damage
            o['m1_acc'][s_i] = (accU - accR)
            o['m2_catacc'][s_i] = (catU - catR)
            o['m3_Floss'][s_i] = float(lU_F.mean())
            o['m4_KS'][s_i] = ksU - ksR
            o['m5_adv'][s_i] = float(lU_sub.mean() - lU_F.mean())
            o['m6_vsRT'][s_i] = float(lU_F.mean() - lR_F.mean())
        if s_i % 5000 == 0:
            print('  {}/{} ({:.0f}s)'.format(s_i, N, time.time() - t0),
                  flush=True)

    # ---------- meta-evaluation ----------
    q75 = float(np.quantile(out['guo']['d_true'], 0.75))
    metrics = {'m1_acc': 'abs', 'm2_catacc': 'abs-signed', 'm3_Floss': 'raw',
               'm4_KS': 'abs-signed', 'm5_adv': 'signed', 'm6_vsRT': 'raw'}
    summary = {'run_id': run_id, 'git_rev': git_rev, 'enum_dir': ed,
               'N': int(N), 'tau_sub': {'acc0': acc0, 'cat0': cat0,
                                        'mean_loss0': b_l0},
               'truth_q75': q75, 'operators': {}}
    for op in ('guo', 'fisher'):
        dt = out[op]['d_true']
        q75_op = float(np.quantile(dt, 0.75))
        damage = dt > q75_op
        per_metric = {}
        for m, mode in metrics.items():
            if mode in ('abs', 'abs-signed'):
                dev = np.abs(out[op][m])
            else:
                dev = out[op][m]
            # deviation from the retrain value: for signed metrics the
            # retrain-referenced forms (m2, m4, m6) are already differences;
            # center by median to remove constant offsets
            dev = dev - np.median(dev)
            per_metric[m] = {
                'auc_detect_damage': auc(np.abs(dev), damage),
                'corr_with_abs_truth': float(np.corrcoef(
                    np.abs(dev), np.abs(dt))[0, 1]),
                'std': float(np.std(dev)),
            }
        # agreement/disagreement: pairwise correlation of |dev| profiles
        devs = {m: (np.abs(out[op][m]) - np.median(np.abs(out[op][m])))
                for m in metrics}
        disagree = {}
        ml = list(metrics)
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                c = float(np.corrcoef(np.abs(devs[ml[i]]),
                                      np.abs(devs[ml[j]]))[0, 1])
                disagree['{}|{}'.format(ml[i], ml[j])] = round(c, 3)
        # blind spot: among top-damage sets (truth), how visible is each
        # metric relative to its own noise? visibility = |dev| z-score
        vis = {}
        for m in metrics:
            dev = np.abs(out[op][m]) - np.median(np.abs(out[op][m]))
            zd = dev / max(float(np.std(dev)), 1e-12)
            vis[m] = float(zd[damage].mean())
        summary['operators'][op] = {'truth_q75': q75_op,
                                    'damage_fraction': float(damage.mean()),
                                    'metrics': per_metric,
                                    'pairwise_corr': disagree,
                                    'visibility_zscore_top_damage': vis}
    with open(os.path.join(outdir, 'p0_results.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    np.savez_compressed(os.path.join(outdir, 'p0_scatter.npz'), sets=sets,
                        **{op + '_' + k: v for op in out
                           for k, v in out[op].items()})
    print(json.dumps(summary, indent=1))


def glob_enum():
    import glob
    return glob.glob(ENUM_DIR_GLOB)


def torch_device(i):
    import torch
    return torch.device('cuda:{}'.format(i))


if __name__ == '__main__':
    main()
