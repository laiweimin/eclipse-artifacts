"""UW-Gm task-matrix completion: contrastive-CE arms on the
original e3/e4 pools under the original frozen audit (mirrors the
G1b gradient-joint protocol exactly), with full label-blind
evaluation (fresh-200 AUC, paired bootstrap CI, retained
before/after, TPR@5/10, four decode channels).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import K
from uw.run_ew3 import f_deep


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run_ce(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    ret = np.sort(np.random.RandomState(52701).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    rx = rec(trset, ret, device)
    cx = rec(tset, cons, device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    model = ctx['model']

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx),
                                                          ny))

    def gen(sd):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53600)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            ce_obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_sd(cand) <= q20:
                    v = float(ce_obj(cand, nx, ny))
                    v0 = float(ce_obj(cur, nx, ny))
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

    def gnorms(sd, x, y):
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

    def channels(sd, x, y):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            out = model(x)
            logp = -torch.nn.functional.cross_entropy(
                out, y, reduction='none')
            p = torch.softmax(out, 1)
            conf = p.gather(1, y[:, None]).squeeze(1)
            zy = out.gather(1, y[:, None]).squeeze(1)
            zc = out.clone()
            zc.scatter_(1, y[:, None], -1e18)
            margin = zy - zc.max(1).values
        return {'negloss': logp.cpu().numpy(),
                'margin': margin.cpu().numpy(),
                'conf': conf.cpu().numpy()}
    starts = [('e3-s1', sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[1]),
              ('e3-s3', sorted(glob.glob(
                  'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[3]),
              ('e3-s4', sorted(glob.glob(
                  'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[4]),
              ('e4-s0', sorted(glob.glob(
                  'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[0]),
              ('e4-s1', sorted(glob.glob(
                  'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[1]),
              ('e4-s2', sorted(glob.glob(
                  'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[2])]
    rsm = np.random.RandomState(53700)
    rows = []
    for tag, sp in (starts[:2] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        fin = gen(sd)
        bm, bn = gnorms(sd, *mem), None
        bn = gnorms(sd, fx[0], fx[1])
        fm = gnorms(fin, mem[0], mem[1])
        fn = gnorms(fin, fx[0], fx[1])
        base = rank_auc(bm, bn)
        fina = rank_auc(fm, fn)
        diffs = []
        for _b in range(2000):
            idx = rsm.choice(30, 30, replace=True)
            diffs.append(rank_auc(fm[idx], fn)
                         - rank_auc(bm[idx], bn))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rbr = rank_auc(bm, gnorms(sd, rx[0], rx[1]))
        rfr = rank_auc(fm, gnorms(fin, rx[0], rx[1]))
        thr5 = np.quantile(fn, 0.95)
        thr10 = np.quantile(fn, 0.90)
        dec = {}
        for ch in ('negloss', 'margin', 'conf'):
            m_ = channels(fin, *mem)[ch]
            f_ = channels(fin, *fx)[ch]
            dec[ch] = rank_auc(m_, f_)
        row = {'start': tag, 'base': base, 'final': fina,
               'ci': [float(lo), float(hi)],
               'ret_base': float(rbr), 'ret_final': float(rfr),
               'tpr5': float(np.mean(fm >= thr5)),
               'tpr10': float(np.mean(fm >= thr10)),
               'decode': dec,
               'pass': bool(score_sd(fin) <= q20)}
        rows.append(row)
        print(' ce %s %.3f->%.3f CI[%.3f,%.3f] ret %.2f->%.2f tpr5 '
              '%.2f tpr10 %.2f' % (tag, base, fina, lo, hi, rbr,
                                    rfr, row['tpr5'], row['tpr10']),
              flush=True)
    ok = sum(1 for r_ in rows if r_['ci'][0] > 0 and r_['pass'])
    v = ('CE_REPLICATED_4OF6' if ok >= 4 else
         'CE_PARTIAL' if ok >= 2 else 'CE_WEAK')
    out = {'rows': rows, 'n_ci_positive': ok, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'ce-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_ce(cli)


if __name__ == '__main__':
    main()
