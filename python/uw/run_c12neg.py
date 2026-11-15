"""C12: encoding attack under WEAKER unlearning operator (NegGrad).

Hypothesis: the 1-ep ft operator creates a TIGHT honest family
(small dispersion → narrow audit bands → limited attack space).
NegGrad creates a WIDER family (larger dispersion → wider bands →
more room for encoding). Test: run CE encoding against a NegGrad-
referenced audit and measure z-TPR@5%.

Setup: build audit from NegGrad honest ensemble (ew6/ew10 family),
encode against it, evaluate with same z-decoder protocol.
"""
import argparse
import glob
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c12neg-' + time.strftime('%Y%m%dT%H%M%S')
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

    # ---- find NegGrad honest ensemble from existing metrics ------
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = json.load(open(cache))['ens_mm']
    print('available families:', list(ens.keys()), flush=True)

    # use e2 family (contains neggrad+ft models from ew2 run)
    # or ew9f (ft-only). We need the WIDEST family.
    # From paper: NegGrad γ=0.86 means very wide dispersion
    # Let's use e2 family which has mixed operators
    for fam in ens:
        if len(ens[fam]) >= 6:
            accs = [e['m1_acc'] for e in ens[fam]]
            fms = [e['m6_fm'] for e in ens[fam]]
            print(' %s: n=%d m1 range [%.3f,%.3f] fm range '
                  '[%.2f,%.2f]'
                  % (fam, len(ens[fam]), min(accs), max(accs),
                     min(fms), max(fms)), flush=True)

    # Use the family with WIDEST m3_Floss dispersion as "weaker
    # operator" reference
    best_fam, best_spread = None, 0
    for fam in ens:
        if len(ens[fam]) >= 6:
            m3s = [e['m3_Floss'] for e in ens[fam]]
            spread = max(m3s) - min(m3s)
            if spread > best_spread:
                best_spread = spread
                best_fam = fam
    print('widest m3 family: %s (spread %.3f)'
          % (best_fam, best_spread), flush=True)

    # build audit from this family
    fit_ens = ens[best_fam][:6]  # use first 6 as fit
    cal_ens = ens[best_fam][6:]  # rest as cal
    if len(cal_ens) < 4:
        cal_ens = ens.get('p1aud', ens.get('e2', []))

    c = {m: float(np.median(
        [e[m] for e in fit_ens])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_ens]) - c[m])))
        for m in METRICS}

    def score(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_s = [score(e) for e in cal_ens if len(e) >= len(METRICS)]
    q20 = conformal_q(cal_s, 0.20)
    print('NegGrad audit: q20=%.3f (cal n=%d)'
          % (q20, len(cal_s)), flush=True)

    def score_sd(sd):
        return score(ctx['eval_full'](sd))

    # ---- z-decoder using same fit family's checkpoints ----------
    # find checkpoints for this family
    # For now: use ew9F (ft family) as z reference (same as before)
    # This tests: encoding vs NegGrad audit, readout with ft-family
    # z-decoder (the audit and decoder can use different references)
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

    def encode(sd_start):
        """CE encoding against the NegGrad-referenced audit."""

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
                if score_sd(cand) <= q20:
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
    starts = [('s1', e3[1]), ('s2', e3[3]), ('s3', e3[4])]

    rows = []
    # baseline: existing CE finals (ft-referenced audit)
    for pth in sorted(glob.glob(
            'research_outputs/uw/lfpr-*/ckpt/ce_e3-s*.pt'))[:3]:
        sd = torch.load(pth, map_location=device)
        zr = zread(sd)
        s6_ng = score_sd(sd)  # score under NegGrad audit
        rows.append({'arm': 'CE_ft_audit', 'start':
                     os.path.basename(pth),
                     'pass_ng': bool(s6_ng <= q20),
                     's6_ng': float(s6_ng), 'zr': zr})
        print(' CE(ft-audit) %s tpr5(gr) %.2f auc %.3f'
              % (os.path.basename(pth),
                 zr['z_neggrad']['tpr5'],
                 zr['z_neggrad']['auc']), flush=True)

    # new: encode against NegGrad audit
    for sname, sp in starts:
        sd0 = torch.load(sp, map_location=device)
        sdf = encode(sd0)
        torch.save(sdf, os.path.join(
            ck, 'negaud_%s.pt' % sname))
        zr = zread(sdf)
        s6 = score_sd(sdf)
        m1 = float(ctx['eval_full'](sdf)['m1_acc'])
        rows.append({'arm': 'CE_ng_audit', 'start': sname,
                     'pass': bool(s6 <= q20), 's6': float(s6),
                     'm1': m1, 'zr': zr})
        print(' CE(ng-audit) %s pass=%s m1 %.3f tpr5(gr) %.2f '
              'auc %.3f fpr %.3f'
              % (sname, rows[-1]['pass'], m1,
                 zr['z_neggrad']['tpr5'],
                 zr['z_neggrad']['auc'],
                 zr['z_neggrad']['fpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'audit_family': best_fam,
                   'q20': float(q20),
                   'rows': rows}, fh, indent=1)

    ft = [r for r in rows if r['arm'] == 'CE_ft_audit']
    ng = [r for r in rows if r['arm'] == 'CE_ng_audit']
    ft_med = float(np.median(
        [r['zr']['z_neggrad']['tpr5'] for r in ft]))
    ng_med = float(np.median(
        [r['zr']['z_neggrad']['tpr5'] for r in ng]))
    ng_pass = all(r.get('pass', False) for r in ng)
    print('FT-audit tpr med %.2f | NG-audit tpr med %.2f | '
          'NG pass %s'
          % (ft_med, ng_med, ng_pass), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
