"""C13: CE encoding at k=300 deletion scale.

Hypothesis: larger F (300 records) changes the bandwidth-utility
constraint balance. Need: (1) create k=300 honest ft starts,
(2) build k=300 reference ensemble, (3) CE encode, (4) z-decode.

Existing k=300 assets: ew3 run has bands/centers for k=300 but
NO ft model checkpoints (only k=1000 ft saved). Must create.
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

K300 = 300
N_REF = 6
N_START = 3
ITERS = 60
FT_SEEDS_REF = tuple(range(46001, 46019))
FT_SEEDS_START = (46101, 46102, 46103)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'k300cal-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    trset = CIFAR10(train=True,
                    transform=_tf.Compose([
                        _tf.RandomHorizontalFlip(),
                        _tf.RandomCrop(32, 4), _tf.ToTensor()]))
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
    n = len(yfull)
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K300)  # 300 records
    keep = np.setdiff1d(np.arange(n), fset)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))

    # F records for k=300
    mem_x = torch.tensor(np.stack(
        [trset.cifar10.data[i] / 255.0
         for i in fset])).permute(0, 3, 1, 2).float().to(device)
    mem_y = torch.tensor(
        [yfull[i] for i in fset]).long().to(device)
    mem = (mem_x, mem_y)
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    cx = rec(tset, cons, device)

    from uw.run_ew3 import metrics_from_logits
    sub3 = np.sort(np.random.RandomState(62000).choice(
        keep, 3000, replace=False))
    xS = torch.tensor(np.stack(
        [trset.cifar10.data[i] / 255.0 for i in sub3])
        ).permute(0, 3, 1, 2).float().to(device)
    yS = torch.tensor([yfull[i] for i in sub3]).long().to(device)
    te_all = np.arange(len(tset.cifar10.targets))
    yT_np = np.asarray(tset.cifar10.targets)
    KEYS6 = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS',
             'm5_adv', 'm6_fm')

    def k300_metrics(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zF = model(mem[0]).cpu().numpy()
            zS_ = model(xS).cpu().numpy()
            zT = []
            for b in np.array_split(te_all, 20):
                xb = torch.tensor(np.stack(
                    [tset.cifar10.data[i] / 255.0 for i in b])
                    ).permute(0, 3, 1, 2).float().to(device)
                zT.append(model(xb).cpu().numpy())
        return metrics_from_logits(zF, mem[1].cpu().numpy(),
                                   zS_, yS.cpu().numpy(),
                                   np.concatenate(zT), yT_np)

    audit_state = {}

    def score_sd(sd):
        mm = k300_metrics(sd)
        return max(abs(mm[k2] - audit_state['c'][k2])
                   / audit_state['sg'][k2] for k2 in KEYS6)

    from tools.models import resnet18

    def ft_unlearn(sd0, s):
        torch.manual_seed(s)
        m = resnet18(num_classes=10).to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        rs = np.random.RandomState(s + 7)
        order = rs.permutation(len(keep))
        m.train()
        for i in range(0, len(keep), 100):
            b = keep[order[i:i + 100]].astype(int)
            x = torch.stack([trset[int(j)][1]
                             for j in b]).to(device)
            y = torch.tensor(yfull[b]).long().to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        m.eval()
        return {k: v.detach().clone()
                for k, v in m.state_dict().items()}

    # ---- step 1: create reference ensemble (6 k=300 ft models) --
    theta0 = torch.load(
        'research_outputs/uw/ew3-20260925T113900/ckpt/'
        'theta0.pt', map_location=device)
    ref_sds = []
    for i, s in enumerate(FT_SEEDS_REF):
        pth = os.path.join(ck, 'ref_ft_s%d.pt' % s)
        if os.path.exists(pth):
            ref_sds.append(torch.load(pth, map_location=device))
        else:
            sd = ft_unlearn(theta0, s)
            torch.save(sd, pth)
            ref_sds.append(sd)
        print(' ref %d done (seed %d)'
              % (i, s), flush=True)

    # ---- recalibrated k=300 audit (prereg #4) ----
    fit_ms = [k300_metrics(s) for s in ref_sds[:6]]
    audit_state['c'] = {k2: float(np.median(
        [m[k2] for m in fit_ms])) for k2 in KEYS6}
    audit_state['sg'] = {k2: max(float(np.median(np.abs(
        np.array([m[k2] for m in fit_ms])
        - audit_state['c'][k2]))), 1e-9) for k2 in KEYS6}
    cal_sc = sorted(score_sd(s) for s in ref_sds[6:])
    q20 = float(cal_sc[int(np.ceil(0.8 * len(cal_sc))) - 1])
    print(' k300 recal q %.3f (from %d refs)' % (q20, len(ref_sds)),
          flush=True)

    # ---- step 2: z-decoder using k=300 reference -----------------
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    ref_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in ref_sds]

    def zread(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in ev_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][chn],
                      [fs['mem'][chn] for fs in ref_scores])
            za = zcal(s['aux'][chn],
                      [fs['aux'][chn] for fs in ref_scores])
            ze = zcal(s['eval'][chn],
                      [fs['eval'][chn] for fs in ref_scores])
            tau = float(np.quantile(za, 0.95))
            out['z_' + chn] = {
                'tpr5': float(np.mean(zm >= tau)),
                'fpr5': float(np.mean(ze >= tau)),
                'auc': float(rank_auc(zm, ze))}
        return out

    # honest baseline z-TPR (reference models themselves)
    hon_tpr = []
    for sd in ref_sds:
        zr = zread(sd)
        hon_tpr.append(zr['z_neggrad']['tpr5'])
    print(' honest k=300 z-TPR@5%%: %s med=%.2f'
          % (['%.2f' % t for t in hon_tpr],
             float(np.median(hon_tpr))), flush=True)

    # ---- step 3: CE encoding ---------------------------------------
    def encode(sd_start):
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

    rows = []
    for i, s in enumerate(FT_SEEDS_START):
        pth = os.path.join(ck, 'start_ft_s%d.pt' % s)
        if os.path.exists(pth):
            sd0 = torch.load(pth, map_location=device)
        else:
            sd0 = ft_unlearn(theta0, s)
            torch.save(sd0, pth)
        sdf = encode(sd0)
        torch.save(sdf, os.path.join(
            ck, 'enc_s%d.pt' % s))
        zr = zread(sdf)
        s6 = score_sd(sdf)
        m1 = float(ctx['eval_full'](sdf)['m1_acc'])
        rows.append({'start': 'k300_s%d' % s,
                     'pass': bool(s6 <= q20), 's6': float(s6),
                     'm1': m1, 'zr': zr})
        print(' k300 enc s%d pass=%s m1 %.3f tpr5(gr) %.2f '
              'auc %.3f fpr %.3f'
              % (s, rows[-1]['pass'], m1,
                 zr['z_neggrad']['tpr5'],
                 zr['z_neggrad']['auc'],
                 zr['z_neggrad']['fpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'q20': float(q20),
                   'honest_tpr': [float(t) for t in hon_tpr],
                   'rows': rows}, fh, indent=1)

    enc_tpr = [r['zr']['z_neggrad']['tpr5'] for r in rows]
    hon_med = float(np.median(hon_tpr))
    enc_med = float(np.median(enc_tpr))
    all_pass = all(r['pass'] for r in rows)
    print('K300-GATE: honest med %.2f encoded med %.2f '
          'diff %+.2f pass=%s'
          % (hon_med, enc_med, enc_med - hon_med, all_pass),
          flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
