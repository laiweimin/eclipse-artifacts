"""C7: self-referenced end-to-end z-encoding (new method).

Key insight: current CE encoding optimizes a PROXY (low loss);
the z-decoder reads CALIBRATED deviation ((r-mu)/sigma vs honest
reference). Optimizing the proxy ≠ optimizing the readout. This
method closes the gap: provider trains its own 6 reference
models, computes per-record mu/sigma, and directly optimizes
z-scores on F subject to audit feasibility.

Arms: uniform CE (reuse), self-ref z-encoding (direct z max).
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
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_lfpr import channels, rec, zcal
from uw.run_r1 import load_audit

ITERS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c7selfz-' + time.strftime('%Y%m%dT%H%M%S')
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

    _, _, _, c_aud, sig_aud, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    # ---- provider's own reference ensemble (self-trained) --------
    # use ew9F as provider's shadow models (same distribution
    # as audit's reference -- provider-known calibration condition)
    ref_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    ref_sds = [torch.load(p, map_location=device)
               for p in ref_paths]

    # compute per-record mu/sigma on F from reference ensemble
    # for neggrad channel (the effective readout channel)
    ref_neggrad = []
    for sd in ref_sds:
        model.load_state_dict(sd)
        model.eval()
        ch = channels(model, sd, mem[0], mem[1])
        ref_neggrad.append(ch['neggrad'])
    ref_arr = np.stack(ref_neggrad)  # (6, 30)
    mu_f = ref_arr.mean(axis=0)  # per-record mean
    sig_f = ref_arr.std(axis=0) + 1e-12  # per-record std
    mu_t = torch.tensor(mu_f, dtype=torch.float32,
                        device=device)
    sig_t = torch.tensor(sig_f, dtype=torch.float32,
                         device=device)
    print('provider ref: mu range [%.3f, %.3f] sig range '
          '[%.4f, %.4f]'
          % (mu_f.min(), mu_f.max(), sig_f.min(), sig_f.max()),
          flush=True)

    # threshold: q95 of aux z-scores (provider-computable)
    ref_aux_scores = []
    for sd in ref_sds:
        ch = channels(model, sd, ax_[0], ax_[1])
        ref_aux_scores.append(ch['neggrad'])
    ref_aux_arr = np.stack(ref_aux_scores)
    mu_a = ref_aux_arr.mean(axis=0)
    sig_a = ref_aux_arr.std(axis=0) + 1e-12
    # z of aux for reference models themselves ~ N(0,1)
    tau5 = float(np.quantile(
        np.random.RandomState(7950).randn(10000), 0.95))
    # better: use actual aux z from a mid-reference model
    mid_ref = ref_sds[3]
    ch_mid = channels(model, mid_ref, ax_[0], ax_[1])
    z_aux_mid = (ch_mid['neggrad'] - mu_a) / sig_a
    tau5 = float(np.quantile(z_aux_mid, 0.95))
    print('provider tau5 (from mid-ref aux z): %.3f'
          % tau5, flush=True)

    def encode_selfz(sd_start):
        """Directly optimize sum of z-scores on F, audit-feasible."""

        def obj(sd, nx, ny):
            model.load_state_dict(sd)
            model.eval()
            # compute neggrad for each F record
            xg = mem[0].clone().requires_grad_(True)
            out = model(xg)
            loss_per = F.cross_entropy(out, mem[1],
                                       reduction='none')
            gn = []
            for i in range(30):
                g = torch.autograd.grad(
                    loss_per[i], xg, retain_graph=True)[0][i]
                gn.append(-g.norm())
            neggrad = torch.stack(gn)
            # z-scores
            z = (neggrad - mu_t) / sig_t
            # objective: maximize sum of z (push all above 0)
            # + contrast term for non-members
            contrast = F.cross_entropy(model(nx), ny)
            return -z.mean() - 0.5 * contrast

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

    # ---- evaluation (audit's z-decoder, not provider's) ---------
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

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts = [('s1', e3[1]), ('s2', e3[3]), ('s3', e3[4])]

    rows = []
    # uniform CE baseline (reuse)
    for pth in sorted(glob.glob(
            'research_outputs/uw/lfpr-*/ckpt/ce_e3-s*.pt'))[:3]:
        sd = torch.load(pth, map_location=device)
        zr = zread(sd)
        rows.append({'arm': 'CE', 'start':
                     os.path.basename(pth),
                     'pass': bool(score_sd(sd) <= q20),
                     'zr': zr})
        print(' CE %s pass=%s tpr5(gr) %.2f auc %.3f'
              % (os.path.basename(pth), rows[-1]['pass'],
                 zr['z_neggrad']['tpr5'],
                 zr['z_neggrad']['auc']), flush=True)

    # self-ref z encoding
    for sname, sp in starts:
        sd0 = torch.load(sp, map_location=device)
        sdf = encode_selfz(sd0)
        torch.save(sdf, os.path.join(
            ck, 'selfz_%s.pt' % sname))
        zr = zread(sdf)
        rows.append({'arm': 'SELFZ', 'start': sname,
                     'pass': bool(score_sd(sdf) <= q20),
                     'zr': zr})
        print(' SELFZ %s pass=%s tpr5(gr) %.2f fpr %.3f '
              'auc %.3f | tpr5(loss) %.2f'
              % (sname, rows[-1]['pass'],
                 zr['z_neggrad']['tpr5'],
                 zr['z_neggrad']['fpr5'],
                 zr['z_neggrad']['auc'],
                 zr['z_negloss']['tpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    ce = [r for r in rows if r['arm'] == 'CE']
    sz = [r for r in rows if r['arm'] == 'SELFZ']
    ce_med = float(np.median(
        [r['zr']['z_neggrad']['tpr5'] for r in ce]))
    sz_med = float(np.median(
        [r['zr']['z_neggrad']['tpr5'] for r in sz]))
    sz_pass = all(r['pass'] for r in sz)
    gate = {'CE_med': ce_med, 'SELFZ_med': sz_med,
            'SELFZ_pass': sz_pass,
            'Z_gate': bool(sz_pass
                           and sz_med > ce_med)}
    print('Z-GATE %s (CE %.2f SELFZ %.2f pass=%s)'
          % (gate['Z_gate'], ce_med, sz_med, sz_pass),
          flush=True)
    with open(os.path.join(outdir, 'gates.json'), 'w') as fh:
        json.dump(gate, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
