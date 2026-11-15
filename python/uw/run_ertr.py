"""E1 (prereg docs/plans/2026-11-03-ertr-exact-retrain-encoding-prereg.md):
exact-retrain-referenced encoding. Replaces the A6 reference family
(one-epoch fine-tunes) with TRUE from-scratch retrains on D minus F
(Table IV recipe verbatim), everything else identical to the primary
protocol (F, set seeds, ce_gen, scoring). 15 fresh retrains
(6 fit / 6 cal / 3 starts) + the 3 existing ew3 retrains as
secondary starts. Gates frozen: EXACT-RT-SURVIVES (>=2/6
accepted-and-CI-positive) / PARTIAL (1) / NEGATIVE (0).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_lfpr import ce_gen, channels
from uw.run_gm import auc as rank_auc

NEW_SEEDS = list(range(47601, 47616))       # 15 fresh retrains
FIT = NEW_SEEDS[:6]
CAL = NEW_SEEDS[6:12]
STARTS_NEW = NEW_SEEDS[12:15]
STARTS_OLD = [44101, 44102, 44103]          # ew3 existing retrains
OLD_GLOB = ('research_outputs/uw/ew3-20260925T113900/ckpt/'
            'rt_k30_s*.pt')


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    keep = np.setdiff1d(np.arange(ctx['n']), fset).astype(int)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    from uw.run_mb import rec
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    def retrain_path(s):
        return os.path.join(ck, 'rt_s%d.pt' % s)

    def train_or_load(s):
        p = retrain_path(s)
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = ctx['fit_full'](keep, s, 40)          # from-scratch recipe
        torch.save(sd, p)
        print(' retrain seed %d done' % s, flush=True)
        return sd

    def mvec(sd):
        return {m: ctx['eval_full'](sd)[m] for m in METRICS}

    # ---- reference family: 15 fresh retrains ----------------------
    fit_sds = [train_or_load(s) for s in FIT]
    cal_sds = [train_or_load(s) for s in CAL]
    start_sds = [(s, train_or_load(s)) for s in STARTS_NEW]
    old_paths = sorted(glob.glob(OLD_GLOB))
    assert len(old_paths) == 3, old_paths
    for s, p in zip(STARTS_OLD, old_paths):
        start_sds.append((s, torch.load(p, map_location=device)))

    fit_mm = [mvec(sd) for sd in fit_sds]
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(sd):
        e = mvec(sd)
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_scores = [sc(sd) for sd in cal_sds]
    q_rt = conformal_q(cal_scores, 0.20)
    print('A6-RT: q-hat = %.3f (n_cal=6, max-order; cal scores %s)'
          % (q_rt, [round(x, 2) for x in cal_scores]), flush=True)

    honest_all = [(s, sd) for s, sd in
                  [(FIT[i], fit_sds[i]) for i in range(6)] +
                  [(CAL[i], cal_sds[i]) for i in range(6)] +
                  start_sds]
    hon_pass = [(s, float(sc(sd)) <= q_rt) for s, sd in honest_all]
    print('honest pass rate: %d/%d' % (sum(p for _, p in hon_pass),
                                       len(hon_pass)), flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q_rt': q_rt, 'cal_scores': cal_scores,
           'honest_pass': {str(s): bool(p) for s, p in hon_pass},
           'rows': []}
    for s, sd0 in start_sds:
        s0 = float(sc(sd0))
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = ce_gen(model, mem, cx, sc, q_rt,
                     {'sd': sd0, 'n': 60})
        sf = float(sc(fin))
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        torch.save(fin, os.path.join(ck, 'enc_s%d.pt' % s))
        row = {'start_seed': s, 'start_src': ('new' if s >= 47600
                                              else 'ew3-existing'),
               's6_start': s0, 's6_fin': sf,
               'pass_start': bool(s0 <= q_rt), 'pass_fin': bool(sf <= q_rt),
               'auc_base': base_auc, 'auc_fin': fin_auc,
               'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
        out['rows'].append(row)
        print('start %d (%s): s6 %.3f->%.3f (q %.3f) pass=%s auc '
              '%.3f->%.3f CI[%.3f,%.3f]'
              % (s, row['start_src'], s0, sf, q_rt, row['pass_fin'],
                 base_auc, fin_auc, lo, hi), flush=True)

    surv = sum(1 for r in out['rows']
               if r['pass_fin'] and r['ci_positive'])
    out['n_survive'] = surv
    out['verdict'] = ('EXACT-RT-SURVIVES' if surv >= 2
                      else 'EXACT-RT-PARTIAL' if surv == 1
                      else 'EXACT-RT-NEGATIVE')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/ertr-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
