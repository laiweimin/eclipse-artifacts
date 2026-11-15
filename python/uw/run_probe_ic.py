"""Probe IND-CAL (review ruling 15b, prereg
docs/plans/2026-11-03-review-probe-ab-prereg.md): attack feasibility
under independent (provider-estimated) calibration. The provider
trains its OWN 12 honest ft models (fresh seeds 45501-45512, recipe
verbatim from run_a9z3.ft_arm), estimates centers/scales from its
first 6 and a split-conformal threshold from its last 6 (same rule
structure as the audit), encodes under the ESTIMATED audit, and the
finals are submitted to the TRUE audit (ew9F centers, q-hat 5.245).
Starts: e3-s1 / e3-s3 / e4-s0. Gates frozen in the prereg card:
IND-CAL-SURVIVES / REQUIRES-KNOWN-CAL / MIXED.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import ce_gen, channels
from uw.run_r1 import load_audit
from uw.run_gm import auc as rank_auc
from uw.run_ey import METRICS

_ICB = int(os.environ.get('IC_BASE', '45501'))
PROV_SEEDS = list(range(_ICB, _ICB + 12))
STARTS = ('e3-s1', 'e3-s3', 'e4-s0')


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def ft_arm(device, sd0, trset, yfull, keep, s):
    """Full-epoch honest ft, verbatim from run_a9z3.ft_arm_cifar."""
    import torch.nn.functional as F
    from tools.models import resnet18
    torch.manual_seed(s)
    m = resnet18(num_classes=10).to(device)
    m.load_state_dict(sd0)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    rs = np.random.RandomState(s + 7)
    order = rs.permutation(len(keep))
    done = 0
    m.train()
    while done < len(keep):
        b = keep[order[done % len(keep):done % len(keep) + 100]]
        x = torch.stack([trset[int(j)][1] for j in b]).to(device)
        y = torch.tensor(yfull[b]).long().to(device)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        done += 100
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
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

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    # ---- provider ensemble (12 fresh honest ft) ------------------
    theta0 = torch.load(
        'research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt',
        map_location=device)
    prov_m = []
    for s in PROV_SEEDS:
        prov_m.append(ft_arm(device, theta0, trset, yfull, keep, s))
        print(' provider ft seed %d done' % s, flush=True)

    def mvec(sd):
        return np.array([ctx['eval_full'](sd)[m] for m in METRICS],
                        dtype=float)

    prov_vecs = [mvec(sd) for sd in prov_m]
    # mirror the audit's estimator exactly: median centers, MAD scales
    c_est = np.median(np.stack(prov_vecs[:6]), axis=0)
    sig_est = np.median(np.abs(np.stack(prov_vecs[:6]) - c_est),
                        axis=0)

    def s_est(sd):
        v = mvec(sd)
        return float(np.max(np.abs((v - c_est)
                                   / np.maximum(sig_est, 1e-12))))

    # split-conformal alpha=0.20 on the provider's own cal six
    # (conformal_q semantics mirrored from run_ey.conformal_q)
    import math
    cal_scores = [s_est(sd) for sd in prov_m[6:]]
    n_cal = len(cal_scores)
    idx = math.ceil((n_cal + 1) * 0.8)
    q_est = float(sorted(cal_scores)[idx - 1]
                  if idx <= n_cal else float('inf'))

    true_scores_prov = [float(sc(ctx['eval_full'](sd)))
                        for sd in prov_m]
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q_true': q20, 'q_est': q_est,
           'prov_true_scores': true_scores_prov,
           'rows': []}
    print('q_est=%.3f vs q_true=%.3f | provider true-audit scores '
          '[%.2f, %.2f]' % (q_est, q20, min(true_scores_prov),
                            max(true_scores_prov)), flush=True)

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    start_paths = {'e3-s1': e3[1], 'e3-s3': e3[3], 'e4-s0': e4[0]}

    for tag in STARTS:
        sd0 = torch.load(start_paths[tag], map_location=device)
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = ce_gen(model, mem, cx, s_est, q_est,
                     {'sd': sd0, 'n': 60})
        s6_true = float(sc(ctx['eval_full'](fin)))
        se_fin = s_est(fin)
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        row = {'start': tag, 's_est_base': s_est(sd0),
               's_est_fin': se_fin, 'pass_est': bool(se_fin <= q_est),
               's6_true_fin': s6_true,
               'pass_true': bool(s6_true <= q20),
               'auc_base': base_auc, 'auc_fin': fin_auc,
               'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
        out['rows'].append(row)
        print('%-6s s_est %.3f->%.3f (q_est %.3f) | true s6 %.3f '
              '(q %.3f) pass=%s | auc %.3f->%.3f CI[%.3f,%.3f]'
              % (tag, row['s_est_base'], se_fin, q_est, s6_true,
                 q20, row['pass_true'], base_auc, fin_auc, lo, hi),
              flush=True)

    surv = sum(1 for r in out['rows'] if r['pass_true']
               and r['ci_positive'])
    out['n_survive'] = surv
    out['verdict'] = ('IND-CAL-SURVIVES' if surv >= 1
                      else 'REQUIRES-KNOWN-CAL'
                      if surv == 0 and not any(r['pass_true']
                                               for r in out['rows'])
                      else 'MIXED')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/probe-ic-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
