"""E-SP: pool-geometry discriminator for the CIFAR-100 boundary
(prereg: docs/plans/2026-10-26-emob-erl-prereg.md, E-SP section).

CIFAR-10 cat cell with build_pool cut to 500 records (seed 54501);
F = f_deep(pool500, 30). Honest family: theta0 ft'd 1 epoch on
D minus F, 18 models (seeds 47501..47518). Audit A6-SP: fit = first
6 (median/MAD), cal = mid 6 (conformal alpha=0.20), starts = passing
of last 6. Sets (train-side same-class, matching E-C100): cons 200 /
fresh 100 / aux 100 / ret 30, seeds 52801/52501/53304/52701.
Encoding = frozen ce_gen. Readout = AUC(ng) resampling CI + z-TPR@5%.
Gates as E-C100: G1 >=4/6 CI>0; G2 >=4/6 z-TPR inc AND max >= 0.20.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3 import build_pool as real_build_pool
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import (BETAS, ce_gen, rec, score_sets,
                         working_point, zcal)

THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt')
FT_SEED_BASE = 47501
POOL_SEED = 54501


def small_pool(cls_idx):
    full = real_build_pool(cls_idx)
    rs = np.random.RandomState(POOL_SEED)
    return np.sort(rs.choice(full, 500, replace=False))


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = []
    for _ in range(n):
        idx = rs.choice(len(fm), len(fm), replace=True)
        diffs.append(rank_auc(fm[idx], fe) - rank_auc(bm[idx], be))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ez.build_pool = small_pool
    ctx = ez.setup_c10(device)
    ez.build_pool = real_build_pool
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    pool = ctx['pool']
    assert len(pool) == 500, len(pool)
    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(ctx['n']), fset)
    pool_rest = np.setdiff1d(pool, fset)
    cons = np.sort(np.random.RandomState(52801).choice(pool_rest, 200,
                                                       replace=False))
    rest = np.setdiff1d(pool_rest, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 100,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    ret_pool = np.setdiff1d(rest, np.concatenate([fresh, aux]))
    ret = np.sort(np.random.RandomState(52701).choice(
        ret_pool, min(30, len(ret_pool)), replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(trset, fresh, device)
    ax_ = rec(trset, aux, device)
    rx = rec(trset, ret, device)
    cx = rec(trset, cons, device)
    sets = {'mem': mem, 'aux': ax_, 'eval': fx, 'ret': rx}
    print('pool=%d F=%d keep=%d cons=%d fresh=%d aux=%d ret=%d'
          % (len(pool), len(fset), len(keep), len(cons), len(fresh),
             len(aux), len(ret)), flush=True)

    # honest family (18 x 1-epoch ft from theta0 on keep)
    sd0 = torch.load(THETA0, map_location=device)
    fam = []
    n_fam = 6 if cli.smoke else 18
    for i in range(n_fam):
        sd = ctx['fit_full'](keep, FT_SEED_BASE + i, 1, init=sd0)
        e = ctx['eval_full'](sd)
        fam.append((sd, {m: float(e[m]) for m in METRICS}))
        if i % 6 == 5:
            print(' fam %d done' % (i + 1), flush=True)
    if cli.smoke:
        fit_m, cal_m, starts_m = fam, fam, fam[:1]
    else:
        fit_m, cal_m, starts_m = fam[:6], fam[6:12], fam[12:]
    c2 = {m: float(np.median([e[m] for _, e in fit_m]))
          for m in METRICS}
    sig2 = {m: float(np.median(np.abs(
        np.array([e[m] for _, e in fit_m]) - c2[m])))
        for m in METRICS}

    def sc2(e):
        return max(abs(e[m] - c2[m]) / max(sig2[m], 1e-12)
                   for m in METRICS)

    q2 = conformal_q([sc2(e) for _, e in cal_m], 0.20)
    passes = [sc2(e) <= q2 for _, e in starts_m]
    print('A6-SP q20=%.4f | starts pass %d/%d'
          % (q2, sum(passes), len(starts_m)), flush=True)

    def score_sd2(sd):
        return sc2(ctx['eval_full'](sd))

    fit_scores = [score_sets(model, sd, sets) for sd, _ in fit_m]

    rows = []
    for si, (sd, _) in enumerate(starts_m):
        if not cli.smoke and sc2(dict(fam[12 + si][1])) > q2:
            continue
        tag = 'sp-s%d' % si
        ce_fin = ce_gen(model, mem, cx, score_sd2, q2,
                        {'sd': sd, 'n': 4 if cli.smoke else 60})
        torch.save(ce_fin,
                   os.path.join(cli.outdir, 'ce_%s.pt' % tag))
        b_sc = score_sets(model, sd, sets)
        f_sc = score_sets(model, ce_fin, sets)
        row = {'start': tag,
               's6_base': float(score_sd2(sd)),
               's6_fin': float(score_sd2(ce_fin)),
               'pass_fin': bool(score_sd2(ce_fin) <= q2), 'wp': {}}
        for ch in ('negloss', 'neggrad'):
            mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
            eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
            base, fin, lo, hi = rank_ci(mf, mb, ef, eb)
            row['wp'][ch] = {'auc_base': base, 'auc_fin': fin,
                             'ci': [lo, hi]}
            zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
            zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
            zab = zcal(b_sc['aux'][ch],
                       [fs['aux'][ch] for fs in fit_scores])
            zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
            zdec = {'base': {}, 'fin': {}}
            for beta in BETAS:
                zdec['base'][str(beta)] = working_point(
                    zmb, zab, zeb, beta)
                zdec['fin'][str(beta)] = working_point(
                    zmf, zab, zeb, beta)
            row['wp']['z_' + ch] = zdec
        rows.append(row)
        print(' %s pass=%s auc(ng) %.3f->%.3f CI[%.3f,%.3f] '
              'zTPR5 %.2f->%.2f'
              % (tag, row['pass_fin'],
                 row['wp']['neggrad']['auc_base'],
                 row['wp']['neggrad']['auc_fin'],
                 row['wp']['neggrad']['ci'][0],
                 row['wp']['neggrad']['ci'][1],
                 row['wp']['z_neggrad']['base']['0.05']['tpr'],
                 row['wp']['z_neggrad']['fin']['0.05']['tpr']),
              flush=True)

    g1 = sum(1 for r in rows if r['wp']['neggrad']['ci'][0] > 0)
    passing = [r for r in rows if r['pass_fin']]
    inc = sum(1 for r in passing
              if r['wp']['z_neggrad']['fin']['0.05']['tpr']
              > r['wp']['z_neggrad']['base']['0.05']['tpr'])
    mx = max([r['wp']['z_neggrad']['fin']['0.05']['tpr']
              for r in passing], default=0.0)
    need = 1 if cli.smoke else 4
    verdict = ('GEOMETRY-EXPLAINS' if g1 < need and inc < need
               else 'DATASET-REAL' if g1 >= need and inc >= need
               and mx >= 0.20 else 'MIXED/WEAK')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': float(q2),
           'pool_seed': POOL_SEED, 'n_fam': n_fam,
           'gates': {'G1_auc_positive': g1,
                     'G2_zTPR_positive_passing': inc,
                     'G2_max': float(mx)},
           'verdict': verdict, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-esp' if cli.smoke else 'esp',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
