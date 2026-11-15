"""C9: audit-strength vs attack-capability tradeoff curve.

Progressively weaken the audit (drop metrics) and measure how much
stronger the encoding attack becomes. This characterizes WHICH
metrics actually constrain the attack.

Audit levels (frozen):
  A6-full: m1+m2+m3+m4+m5+m6 (existing, q20=5.245)
  A4-noMIA: m1+m2+m3+m4+m6 (drop m5 MIA AUC)
  A4-noKS: m1+m2+m3+m5+m6 (drop m4 KS distance)
  A2-basic: m1+m2 only (deployable minimum)
  A1-acc: m1 only (bare utility check)

For each level: recalibrate threshold from honest cal ensemble,
run CE encoding from 2 starts, measure z-TPR@5% (neggrad).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_lfpr import channels, rec, zcal

ITERS = 60
AUDIT_LEVELS = {
    'A6full': ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS',
               'm5_adv', 'm6_fm'),
    'A4noMIA': ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS',
                'm6_fm'),
    'A4noKS': ('m1_acc', 'm2_catacc', 'm3_Floss', 'm5_adv',
               'm6_fm'),
    'A2basic': ('m1_acc', 'm2_catacc'),
    'A1acc': ('m1_acc',),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c9audit-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    cx = rec(tset, cons, device)

    # ---- load full audit cache and build per-level audits --------
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = json.load(open(cache))['ens_mm']
    fit = ens['ew9f']
    cal_tags = ('p1aud', 'e2')

    def make_audit(metric_subset):
        c = {m: float(np.median(
            [e[m] for e in fit if m in e]))
            for m in metric_subset}
        sig = {m: float(np.median(np.abs(
            np.array([e[m] for e in fit if m in e]) - c[m])))
            for m in metric_subset}

        def score(e):
            return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                       for m in metric_subset
                       if m in e)

        cal_s = [score(e) for t in cal_tags
                 for e in ens[t] if all(m in e
                                        for m in metric_subset)]
        q20 = conformal_q(cal_s, 0.20)
        return score, q20

    audits = {}
    for name, metrics in AUDIT_LEVELS.items():
        score_fn, q20 = make_audit(metrics)
        audits[name] = (score_fn, q20)
        # count cal entries for logging
        n_cal = sum(1 for t in cal_tags
                    for e in ens[t]
                    if all(m in e for m in metrics))
        print(' %s: q20=%.3f (cal n=%d)'
              % (name, q20, n_cal), flush=True)

    def score_with(name, sd):
        fn, _ = audits[name]
        e = ctx['eval_full'](sd)
        return fn(e)

    # ---- z-decoder machinery ---------------------------------------
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in fit_sds]

    def zread(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in ev_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][chn],
                      [fs['mem'][chn] for fs in fit_scores])
            za = zcal(s['aux'][chn],
                      [fs['aux'][chn] for fs in fit_scores])
            ze = zcal(s['eval'][chn],
                      [fs['eval'][chn] for fs in fit_scores])
            tau = float(np.quantile(za, 0.95))
            out['z_' + chn] = {
                'tpr5': float(np.mean(zm >= tau)),
                'fpr5': float(np.mean(ze >= tau)),
                'auc': float(rank_auc(zm, ze))}
        return out

    def encode(sd_start, audit_name):
        _, q20 = audits[audit_name]

        def obj(sd, nx, ny):
            model.load_state_dict(sd)
            model.eval()
            rm = -F.cross_entropy(model(mem[0]), mem[1])
            contrast = F.cross_entropy(model(nx), ny)
            return rm.mean() - 0.5 * contrast

        cur = sd_start
        lr = LR0
        rs = np.random.RandomState(53600)
        stall = 0
        for _it in range(ITERS):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone()
                 for nm, p in model.named_parameters()
                 if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_with(audit_name, cand) <= q20:
                    v = float(obj(cand, nx, ny))
                    v0 = float(obj(cur, nx, ny))
                    if v < v0 - 1e-9:
                        cur, acc = cand, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > 12:
                break
        return cur

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts = [('s1', e3[1]), ('s2', e3[3])]

    rows = []
    for aname in AUDIT_LEVELS:
        for sname, sp in starts:
            sd0 = torch.load(sp, map_location=device)
            sdf = encode(sd0, aname)
            torch.save(sdf, os.path.join(
                ck, '%s_%s.pt' % (aname, sname)))
            zr = zread(sdf)
            _, q20 = audits[aname]
            s6 = score_with(aname, sdf)
            m1 = float(ctx['eval_full'](sdf)['m1_acc'])
            rows.append({'audit': aname, 'start': sname,
                         's6': float(s6), 'q20': float(q20),
                         'pass': bool(s6 <= q20),
                         'm1': m1, 'zr': zr})
            print(' %s %s pass=%s m1 %.3f tpr5(gr) %.2f '
                  'auc %.3f fpr %.3f'
                  % (aname, sname, rows[-1]['pass'], m1,
                     zr['z_neggrad']['tpr5'],
                     zr['z_neggrad']['auc'],
                     zr['z_neggrad']['fpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # summary: per audit level
    print('\\n=== AUDIT STRENGTH vs ATTACK CAPABILITY ===',
          flush=True)
    for aname in AUDIT_LEVELS:
        lv = [r for r in rows if r['audit'] == aname]
        if not lv:
            continue
        tpr = [r['zr']['z_neggrad']['tpr5'] for r in lv]
        auc_ = [r['zr']['z_neggrad']['auc'] for r in lv]
        print(' %-9s: tpr5(gr) %s | auc %s | pass %s | m1 %s'
              % (aname,
                 ['%.2f' % t for t in tpr],
                 ['%.3f' % a for a in auc_],
                 ['%s' % r['pass'] for r in lv],
                 ['%.3f' % r['m1'] for r in lv]), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
