"""Stronger-m5 audit experiment (prereg docs/plans/2026-11-09-sm5-icdraws-prereg.md).

m5s = MLP probe (5-16-1) on per-sample output features [ce, margin,
entropy, true_logit, top_other] (all black-box+losses accessible),
in-sample AUC, same protocol shape as the frozen m5 spec. Strong audit
= m1..m4, m5s, m6, conformal q at alpha=0.20 on the primary
fit6/cal12/test12 honest models; this instance's own frozen probe
subsample (retained 3000, seed 62000).
Tier A: rescore the 12 primary finals (GR gate-g1b 6 + CE lfpr 6).
Tier B: encode 3 e4 starts under the strong audit (run_ce hyperparams).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from uw.run_ez import apply_delta, setup_c10
from uw.run_ew3 import metrics_from_logits, _ce_rows, _margin_rows
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc


def rows(z, y):
    ce = _ce_rows(z, y)
    mg = _margin_rows(z, y)
    zy = z[np.arange(len(y)), y]
    zc = z.copy()
    zc[np.arange(len(y)), y] = -1e18
    ent = zy - (zc.max(axis=1) + np.log(
        np.exp(zc - zc.max(axis=1, keepdims=True)).sum(axis=1) + 1e-30))
    feat = np.column_stack([ce, mg, ent, zy, zc.max(axis=1)])
    return feat.astype(np.float32)


def mlp_auc(fp, fn):
    X = np.concatenate([fp, fn])
    yv = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    tX = torch.tensor(Xn)
    ty = torch.tensor(yv, dtype=torch.float32).unsqueeze(1)
    net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(),
                        nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        lossf(net(tX), ty).backward()
        opt.step()
    with torch.no_grad():
        s = net(tX).squeeze(1).numpy()
    from uw.run_ew3 import auc_mw
    return float(auc_mw(s[:len(fp)], s[len(fp):]))


def run_sm5(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    te = np.arange(len(tset.cifar10.targets))
    cat_te = te[np.asarray(tset.cifar10.targets) == 3]
    cat_tr = ctx['pool']
    yfull = ctx['yfull']
    # frozen probe subsample: 3000 RETAINED TRAIN records, seed 62000
    poolset = set(cat_tr.tolist())
    fset_set = set(ctx['fset'].tolist())
    ret_train = np.array([i for i in ctx['keep']
                          if i not in poolset and i not in fset_set])
    sub = np.sort(np.random.RandomState(62000).choice(
        ret_train, 3000, replace=False))
    fidx = ctx['fset']
    xF, yF = ctx['xF'], ctx['yF']

    def rec_train(idx):
        x = torch.tensor(np.stack([trset.cifar10.data[i] / 255.0
                                   for i in idx]))\
            .permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor(np.asarray(trset.cifar10.targets)[idx])\
            .long().to(device)
        return x, y

    def rec(idx):
        x = torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                   for i in idx]))\
            .permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([tset.cifar10.targets[i] for i in idx])\
            .long().to(device)
        return x, y

    xS, yS = rec_train(sub)
    xT = torch.cat([rec(b)[0] for b in np.array_split(te, 20)])
    yT = torch.tensor(np.asarray(tset.cifar10.targets)[te])\
        .long().to(device)
    sub_np = np.asarray(trset.cifar10.targets)[sub]
    model = ctx['model']

    def logits_all(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zF = model(xF).cpu().numpy()
            zS = model(xS).cpu().numpy()
            zT = []
            for b in torch.split(xT, 1000):
                zT.append(model(b).cpu().numpy())
        return zF, zS, np.concatenate(zT)

    def strong_metrics(sd):
        zF, zS, zT = logits_all(sd)
        mm = metrics_from_logits(zF, yF.cpu().numpy(), zS, sub_np,
                                 zT, yT.cpu().numpy())
        m5s = mlp_auc(rows(zF, yF.cpu().numpy()),
                      rows(zS, sub_np))
        return {'m1': mm['m1_acc'], 'm2': mm['m2_catacc'],
                'm3': mm['m3_Floss'], 'm4': mm['m4_KS'],
                'm5s': m5s, 'm6': mm['m6_fm']}

    KEYS = ('m1', 'm2', 'm3', 'm4', 'm5s', 'm6')
    fit = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    cal = sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt')) + \
        sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt'))
    test = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt')) + \
        sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    test = [p for p in test][:12]
    mm_fit = [strong_metrics(torch.load(p, map_location=device))
              for p in fit]
    c_ = {k: float(np.median([m[k] for m in mm_fit]))
          for k in KEYS}
    sg = {k: max(float(np.median(np.abs(
        np.array([m[k] for m in mm_fit]) - c_[k]))), 1e-9)
        for k in KEYS}

    def sc_strong(sd):
        mm = strong_metrics(sd)
        return max(abs(mm[k] - c_[k]) / sg[k] for k in KEYS)

    q_s = np.quantile([sc_strong(torch.load(p, map_location=device))
                       for p in cal] + [np.inf], 0.80,
                      interpolation='higher')
    q_s = float(np.sort([sc_strong(torch.load(p, map_location=device))
                         for p in cal])[int(np.ceil(0.8 * len(cal))) - 1])
    print(' sm5 q %.3f' % q_s, flush=True)
    hon_test = sum(sc_strong(torch.load(p, map_location=device))
                   <= q_s for p in test)
    print(' sm5 honest test pass %d/%d' % (hon_test, len(test)),
          flush=True)

    # Tier A: rescore primary finals
    tierA = []
    for tag, gl in (('GR', 'research_outputs/uw/gate-g1b-'
                     '20261007T102014/ckpt/fb_s*.pt'),
                    ('CE', 'research_outputs/uw/lfpr-2026*/ckpt/*.pt')):
        for p in sorted(glob.glob(gl))[:6]:
            sd = torch.load(p, map_location=device)
            s_ = sc_strong(sd)
            tierA.append({'tag': tag, 'ckpt': os.path.basename(p),
                          'score': float(s_),
                          'pass': bool(s_ <= q_s)})
            print(' sm5 A %s %s %.3f %s' % (tag,
                                            os.path.basename(p), s_,
                                            s_ <= q_s), flush=True)
    nA = sum(r['pass'] for r in tierA)

    # Tier B: encode 3 e4 starts under strong audit
    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset
                  and i not in sub.tolist()]), 300, replace=False))
    cx = rec(cons)
    mem = (xF, yF)

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(
                    model(nx), ny))

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
                if sc_strong(cand) <= q_s:
                    model.load_state_dict(cur)
                    model.zero_grad()
                    if float(ce_obj(cand, nx, ny)) < \
                            float(ce_obj(cur, nx, ny)) - 1e-9:
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

    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(np.array([i for i in cat_te
                               if i not in poolset]), sub)[:2000]
        if False else np.setdiff1d(
            np.array([i for i in cat_te if i not in poolset]),
            np.concatenate([sub, cons])), 200, replace=False))
    fx = rec(fresh)

    def auc_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1],
                reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1],
                reduction='none').cpu().numpy()
        return rank_auc(zm, zf)

    def ci_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1],
                reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1],
                reduction='none').cpu().numpy()
        rs = np.random.RandomState(54100)
        bs = [rank_auc(zm[rs.choice(len(zm), len(zm), True)], zf)
              for _ in range(2000 if not cli.smoke else 50)]
        return (float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)))

    tierB = []
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[:3]
    for p in e4:
        sd = torch.load(p, map_location=device)
        base = auc_nl(sd)
        fB = gen(sd)
        aB = auc_nl(fB)
        lo, hi = ci_nl(fB)
        tierB.append({'start': os.path.basename(p), 'base': base,
                      'gain': aB - base, 'ci_lo': lo - base,
                      'ci_hi': hi - base,
                      'pass': bool(sc_strong(fB) <= q_s)})
        print(' sm5 B %s base %.3f g %.3f ci[%.3f,%.3f] pass %s'
              % (os.path.basename(p), base, aB - base, lo - base,
                 hi - base, sc_strong(fB) <= q_s), flush=True)
    vA = ('STRONG-M5-PASSES' if nA == len(tierA)
          else 'STRONG-M5-DETECTS')
    ci_pos = sum(1 for r in tierB if r['ci_lo'] > 0)
    vB = ('STRONG-ENCODING-WORKS' if ci_pos >= 2 else
          ('COMPRESSED' if ci_pos == 0 else 'PARTIAL'))
    out = {'q_strong': q_s, 'honest_test_pass': [hon_test, len(test)],
           'tierA': tierA, 'tierB': tierB,
           'verdictA': vA, 'verdictB': vB}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT A:', vA, '| B:', vB, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'sm5-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_sm5(cli)


if __name__ == '__main__':
    main()
