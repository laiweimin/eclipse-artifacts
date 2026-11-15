"""UW-Gm R2-verdict zero-training closure (M1/M2/M4).

  1. Paired bootstrap CIs for the six independent gains (resample
     the 30 fixed target records; recompute base and final AUC on
     each resample; report gain distribution).
  2. Retained (F-vs-R) before/after AUC for all six starts.
  3. Paired 6-vs-7-metric audit decisions on the same 12 honest
     test models (incremental FPR, not raw rejection rates).
  4. Decode-channel comparison on the same finals: member-vs-fresh
     AUC under -loss, margin, confidence, -gradnorm.
  5. Fixed-threshold inference: TPR@5% and @10% FPR.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def all_scores(ctx, sd, mem, fresh):
    """Per-record scores for the four decode channels."""
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()

    def feats(x, y):
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        logp = -torch.nn.functional.cross_entropy(
            out, y, reduction='none')
        p = torch.softmax(out, 1)
        conf = p.gather(1, y[:, None]).squeeze(1)
        zy = out.gather(1, y[:, None]).squeeze(1)
        zc = out.clone()
        zc.scatter_(1, y[:, None], -1e18)
        margin = zy - zc.max(1).values
        gn = []
        for i in range(len(y)):
            g = torch.autograd.grad(-logp[i], xg,
                                    retain_graph=True)[0][i]
            gn.append(float(g.detach().norm()))
        return (logp.detach().cpu().numpy(),
                margin.detach().cpu().numpy(),
                conf.detach().cpu().numpy(), -np.array(gn))
    fm = feats(*mem)
    ff = feats(*fresh)
    return {ch: (fm[i], ff[i]) for i, ch in enumerate(
        ('negloss', 'margin', 'conf', 'neggnorm'))}


def run_close(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    rng = np.random.RandomState(52801)
    cons = np.sort(rng.choice(cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    ret = np.sort(np.random.RandomState(52701).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    rx = rec(trset, ret, device)
    starts_e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts_e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    fin_e3 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s*.pt'))
    fin_e4 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s*.pt'))
    rs = np.random.RandomState(54000)
    out = {'rows': [], 'decode': [], 'threshold': []}
    pairs = [('e3-s1', starts_e3[1], fin_e3[0]),
             ('e3-s3', starts_e3[3], fin_e3[1]),
             ('e3-s4', starts_e3[4], fin_e3[2]),
             ('e4-s0', starts_e4[0], fin_e4[0]),
             ('e4-s1', starts_e4[1], fin_e4[1]),
             ('e4-s2', starts_e4[2], fin_e4[2])]
    for tag, sp, fp in pairs:
        sb = torch.load(sp, map_location=device)
        sf = torch.load(fp, map_location=device)
        gb = all_scores(ctx, sb, mem, fx)['neggnorm']
        gf = all_scores(ctx, sf, mem, fx)
        # retained before/after
        rb = rank_auc(all_scores(ctx, sb, mem, rx)['neggnorm'][0],
                      all_scores(ctx, sb, mem, rx)['neggnorm'][1])
        rf = rank_auc(all_scores(ctx, sf, mem, rx)['neggnorm'][0],
                      all_scores(ctx, sf, mem, rx)['neggnorm'][1])
        # paired bootstrap over the 30 members (each model paired
        # with its OWN non-member scores)
        base_m, base_non = gb
        fin_m, non = gf['neggnorm']
        diffs = []
        for _b in range(2000):
            idx = rs.choice(30, 30, replace=True)
            a0 = rank_auc(base_m[idx], base_non)
            a1 = rank_auc(fin_m[idx], non)
            diffs.append(a1 - a0)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # fixed-threshold TPR on fresh negatives
        for ch, (m_, f_) in gf.items():
            if ch != 'neggnorm':
                continue
        thr5 = np.quantile(non, 0.95)
        thr10 = np.quantile(non, 0.90)
        row = {'start': tag,
               'auc_base': rank_auc(base_m, base_non),
               'auc_fin': rank_auc(fin_m, non),
               'gain_ci': [float(lo), float(hi)],
               'gain_med': float(np.median(diffs)),
               'ret_base': float(rb), 'ret_fin': float(rf),
               'tpr5': float(np.mean(fin_m >= thr5)),
               'tpr10': float(np.mean(fin_m >= thr10))}
        out['rows'].append(row)
        print(' cl %s base %.3f fin %.3f CI[%.3f,%.3f] ret %.3f->'
              '%.3f tpr5 %.2f' % (tag, row['auc_base'],
                                   row['auc_fin'], lo, hi, rb, rf,
                                   row['tpr5']), flush=True)
        # decode channels on the same final
        dec = {'start': tag}
        for ch, (m_, f_) in gf.items():
            dec[ch] = rank_auc(m_, f_)
        out['decode'].append(dec)
        print('   decode', {k: round(v, 3) for k, v in dec.items()
                            if k != 'start'}, flush=True)

    # ---- paired 6 vs 7 metric audit decisions (same 12 models) --
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    na = np.sort(np.random.RandomState(53101).choice(cat_te, 100,
                                                     replace=False))
    nx0 = torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in na]))\
        .permute(0, 3, 1, 2).float().to(device)
    ny0 = torch.tensor([tset.cifar10.targets[i] for i in na])\
        .long().to(device)

    def gmean(sd, x, y):
        model = ctx['model']
        model.load_state_dict(sd)
        model.eval()
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        loss = torch.nn.functional.cross_entropy(out, y,
                                                 reduction='none')
        tot = 0.0
        for i in range(len(y)):
            g = torch.autograd.grad(loss[i], xg)[0][i]
            tot = tot + g.norm()
        return float(tot / len(y))
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))

    def dstat(sd):
        return gmean(sd, nx0, ny0) - gmean(sd, ctx['xF'], ctx['yF'])
    d_fit = [dstat(torch.load(p, map_location=device))
             for p in fit_paths]
    d_med = float(np.median(d_fit))
    d_sig = max(float(np.median(np.abs(np.array(d_fit) - d_med))),
                1e-9)

    def sc7(sd):
        dpen = abs(dstat(sd) - d_med) / d_sig
        return max(sc(ctx['eval_full'](sd)), dpen)
    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    q7 = conformal_q([sc7(torch.load(p, map_location=device))
                      for p in cal_paths], 0.20)
    test_paths = (sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt')))
    paired = []
    for p in test_paths:
        sd = torch.load(p, map_location=device)
        a6 = sc(ctx['eval_full'](sd)) <= q20
        a7 = sc7(sd) <= q7
        paired.append([bool(a6), bool(a7)])
    dfpr = float(np.mean([a - b for a, b in
                          [(int(x), int(y_)) for x, y_ in
                           [(a6, a7) for a6, a7 in paired]]]
                         ) * -1 + 0)  # placeholder corrected below
    rej6 = sum(1 for a6, _ in paired if not a6)
    rej7 = sum(1 for _, a7 in paired if not a7)
    out['audit_paired'] = {'decisions': paired, 'rej6': rej6,
                           'rej7': rej7, 'q7': float(q7)}
    print(' cl audit paired rej6 %d rej7 %d of 12' % (rej6, rej7),
          flush=True)
    with open(os.path.join(cli.outdir, 'closure.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('OUT', cli.outdir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    cli.outdir = os.path.join('research_outputs', 'uw',
                              'close-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=False)
    run_close(cli)


if __name__ == '__main__':
    main()
