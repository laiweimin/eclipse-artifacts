"""UW-Gm new-F preregistered validation (frozen
docs/plans/2026-10-07-uwgm-newf-preregistration.md).

New deletion request F' (seed 53301, 30 cat records); fresh honest
ft family (30 models); audit fit z1 / cal z2+z3 / pools z4,z5;
gradient-joint vs contrastive-CE arms under the new audit; full
label-blind evaluation (paired bootstrap CI, retained, TPR,
four decode channels).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ey import K, METRICS, conformal_q, record_subsets
from uw.run_ez import (apply_delta, cached_train, make_context, obj_val,
                       setup_c10)
from uw.run_gm import auc as rank_auc
from uw.run_gm2 import trace_loss
from uw.run_ew3b import LR0, lr_update
from uw.run_ew3 import f_deep

RUN3 = 'research_outputs/uw/ew3-20260925T113900'


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run_newf(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    n_ft = 2 if cli.smoke else 6
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    # ---- frozen draws ------------------------------------------------
    fset = np.sort(np.random.RandomState(53301).choice(
        cat_tr, 30, replace=False))
    cons = np.sort(np.random.RandomState(53302).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(53303).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    ret = np.sort(np.random.RandomState(53305).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    keep = np.setdiff1d(np.arange(ctx['n']), fset)
    fx = rec(tset, fresh, device)
    rx = rec(trset, ret, device)
    # ---- honest family ------------------------------------------------
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    ens_mm, ens_paths, ens_sds = {}, {}, {}
    for e in range(5):
        tag = 'z{}'.format(e + 1)
        ens_mm[tag], ens_paths[tag], ens_sds[tag] = [], [], []
        for j in range(n_ft):
            seed = 49501 + e * 10 + j
            name = '{}_ft_s{}'.format(tag, seed)
            sd, mm = cached_train(ctx, ck, name, keep, seed, 1,
                                  init=sd0, cosine=False,
                                  eval_fn=ctx['eval_full'],
                                  outdir=cli.outdir)
            ens_sds[tag].append(sd)
            ens_mm[tag].append(mm)
            ens_paths[tag].append(os.path.join(ck, name + '.pt'))
        print(' newf ens', tag, flush=True)
    c = {m: float(np.median([e[m] for e in ens_mm['z1']]))
         for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in ens_mm['z1']]) - c[m])))
        for m in METRICS}

    def sc(mm):
        return max(abs(mm[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)
    cal = [m for t in ('z2', 'z3') for m in ens_mm[t]]
    q20 = conformal_q([sc(e) for e in cal], 0.20)
    honest = [m for t in ('z2', 'z3', 'z4', 'z5')
              for m in ens_mm.get(t, [])]
    print(' newf q20 %.3f honest %.3f'
          % (q20, float(np.mean([sc(e) <= q20 for e in honest]))),
          flush=True)

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    # per-record tensors for F'
    fx_set = rec(trset, fset, device)
    mem_x, mem_y = fx_set
    cx = rec(tset, cons, device)

    def gen(sd, obj_kind):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53400)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model = ctx['model']
            model.load_state_dict(cur)
            model.zero_grad()
            if obj_kind == 'joint':
                tl = trace_loss(ctx, cur, mem_x, mem_y, nx, ny)
            else:
                model.eval()
                tl = (torch.nn.functional.cross_entropy(
                    model(mem_x), mem_y)
                    - 0.5 * torch.nn.functional.cross_entropy(
                        model(nx), ny))
            tl.backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_sd(cand) <= q20:
                    model.load_state_dict(cand)
                    model.eval()
                    v = float(trace_loss(
                        ctx, cand, mem_x, mem_y, nx, ny)
                        if obj_kind == 'joint'
                        else torch.nn.functional.cross_entropy(
                            model(mem_x), mem_y)
                        - 0.5 * torch.nn.functional.cross_entropy(
                            model(nx), ny))
                    model.load_state_dict(cur)
                    model.eval()
                    v0 = float(trace_loss(
                        ctx, cur, mem_x, mem_y, nx, ny)
                        if obj_kind == 'joint'
                        else torch.nn.functional.cross_entropy(
                            model(mem_x), mem_y)
                        - 0.5 * torch.nn.functional.cross_entropy(
                            model(nx), ny))
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

    # ---- evaluation ----------------------------------------------------
    def gnorms(sd, x, y):
        model = ctx['model']
        model.load_state_dict(sd)
        model.eval()
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        loss = torch.nn.functional.cross_entropy(out, y,
                                                 reduction='none')
        return -np.array([
            float(torch.autograd.grad(loss[i], xg,
                                      retain_graph=True)[0][i]
                  .norm()) for i in range(len(y))])

    starts = [p for p in ens_paths['z4']
              if score_sd(torch.load(p, map_location=device))
              <= q20][:3]
    if len(starts) < 3:
        starts += [p for p in ens_paths['z5']
                   if score_sd(torch.load(p, map_location=device))
                   <= q20][:3 - len(starts)]
    rows = []
    rsm = np.random.RandomState(53500)
    for sp in starts:
        sd = torch.load(sp, map_location=device)
        r_ = {'start': os.path.basename(sp)}
        for kind in ('joint', 'ce'):
            fin = gen(sd, kind)
            bm = gnorms(sd, mem_x, mem_y)
            bn = gnorms(sd, fx[0], fx[1])
            fm = gnorms(fin, mem_x, mem_y)
            fn = gnorms(fin, fx[0], fx[1])
            base = rank_auc(bm, bn)
            fina = rank_auc(fm, fn)
            diffs = []
            for _b in range(2000):
                idx = rsm.choice(30, 30, replace=True)
                diffs.append(rank_auc(fm[idx], fn)
                             - rank_auc(bm[idx], bn))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rbr = rank_auc(gnorms(sd, mem_x, mem_y),
                           gnorms(sd, rx[0], rx[1]))
            rfr = rank_auc(gnorms(fin, mem_x, mem_y),
                           gnorms(fin, rx[0], rx[1]))
            thr5 = np.quantile(fn, 0.95)
            r_[kind] = {'base': base, 'final': fina,
                        'ci': [float(lo), float(hi)],
                        'ret_base': float(rbr),
                        'ret_final': float(rfr),
                        'tpr5': float(np.mean(fm >= thr5)),
                        'pass': bool(score_sd(fin) <= q20)}
            print(' newf %s %s %.3f->%.3f CI[%.3f,%.3f] ret %.2f->'
                  '%.2f tpr5 %.2f' % (r_['start'], kind, base, fina,
                                       lo, hi, rbr, rfr,
                                       r_[kind]['tpr5']), flush=True)
        rows.append(r_)
    joint_ok = sum(1 for r_ in rows
                   if r_['joint']['ci'][0] > 0
                   and r_['joint']['pass'])
    jg = [r_['joint']['final'] - r_['joint']['base'] for r_ in rows]
    cg = [r_['ce']['final'] - r_['ce']['base'] for r_ in rows]
    med_ok = float(np.median(jg)) >= float(np.median(cg))
    v = ('NEWF_REPLICATED' if joint_ok >= 2 and med_ok
         else ('NEWF_PARTIAL' if joint_ok >= 1 else 'NEWF_NULL'))
    out = {'q20': float(q20), 'rows': rows, 'joint_gains': jg,
           'ce_gains': cg, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'newf-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('NewF ->', cli.outdir, flush=True)
    run_newf(cli)


if __name__ == '__main__':
    main()
