"""C14: multi-class deletion encoding attack.

Audit design gap: m2 monitors ONE class (cat). If F spans
multiple classes (10 cat + 10 dog + 10 frog), the encoding on
non-cat records faces NO class-specific audit constraint.

Test: create multi-class F, build multi-class reference, encode,
evaluate z-TPR separately for monitored vs non-monitored classes.
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

PER_CLASS = 10  # 10 per class × 3 classes = 30 total
CLASSES = [3, 5, 6]  # cat, dog, frog
ITERS = 60
N_REF = 6
REF_SEEDS = (47001, 47002, 47003, 47004, 47005, 47006)
START_SEEDS = (47101, 47102, 47103)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c14multi-' + time.strftime('%Y%m%dT%H%M%S')
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

    # ---- multi-class F: 10 per class × 3 classes ------------------
    cat_pool = ctx['pool']  # cat training pool
    fset_multi = []
    for ci, cls in enumerate(CLASSES):
        if cls == 3:
            pool_c = cat_pool
        else:
            pool_c = np.where(yfull == cls)[0]
        sel = np.sort(np.random.RandomState(
            48000 + cls).choice(pool_c, PER_CLASS, replace=False))
        fset_multi.extend(sel.tolist())
    fset_multi = np.sort(np.array(fset_multi))
    keep = np.setdiff1d(np.arange(n), fset_multi)

    # fresh non-members for each class
    targets_te = np.asarray(tset.cifar10.targets)
    fresh_by_cls = {}
    aux_by_cls = {}
    for cls in CLASSES:
        cls_te = np.where(targets_te == cls)[0]
        # exclude any test records that might overlap
        sel_f = np.sort(np.random.RandomState(
            49000 + cls).choice(cls_te, 100, replace=False))
        sel_a = np.sort(np.random.RandomState(
            49100 + cls).choice(
                np.setdiff1d(cls_te, sel_f), 50, replace=False))
        fresh_by_cls[cls] = sel_f
        aux_by_cls[cls] = sel_a

    # F records
    mem_x = torch.tensor(np.stack(
        [trset.cifar10.data[i] / 255.0
         for i in fset_multi])).permute(0, 3, 1, 2).float().to(device)
    mem_y = torch.tensor(
        [yfull[i] for i in fset_multi]).long().to(device)
    mem = (mem_x, mem_y)
    # class labels for F records (for per-class evaluation)
    f_classes = np.array([yfull[i] for i in fset_multi])

    # evaluation sets per class
    ev_sets = {'mem': mem}
    for cls in CLASSES:
        cname = {3: 'cat', 5: 'dog', 6: 'frog'}[cls]
        ev_sets['fresh_' + cname] = rec(
            tset, fresh_by_cls[cls][:50], device)
        ev_sets['aux_' + cname] = rec(
            tset, aux_by_cls[cls], device)
    cons = np.sort(np.random.RandomState(52801).choice(
        np.where(targets_te == 3)[0], 300, replace=False))
    cx = rec(tset, cons, device)

    _, _, _, c_aud, sig_aud, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

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

    # ---- reference ensemble (multi-class ft models) ---------------
    theta0 = torch.load(
        'research_outputs/uw/ew3-20260925T113900/ckpt/'
        'theta0.pt', map_location=device)
    ref_sds = []
    for i, s in enumerate(REF_SEEDS):
        pth = os.path.join(ck, 'ref_s%d.pt' % s)
        if os.path.exists(pth):
            ref_sds.append(torch.load(pth, map_location=device))
        else:
            sd = ft_unlearn(theta0, s)
            torch.save(sd, pth)
            ref_sds.append(sd)
        print(' ref %d done' % i, flush=True)

    ref_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in ref_sds]

    def zread_per_class(sd):
        """z-TPR@5% separately for each class's F records."""
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in ev_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm_all = zcal(s['mem'][chn],
                          [fs['mem'][chn] for fs in ref_scores])
            for cls in CLASSES:
                cname = {3: 'cat', 5: 'dog',
                         6: 'frog'}[cls]
                zf = ev_sets['fresh_' + cname]
                s_fresh = channels(model, sd, zf[0], zf[1])
                za = zcal(s_fresh[chn],
                          [fs['fresh_' + cname][chn]
                           for fs in ref_scores])
                ze = zcal(s['aux_' + cname][chn],
                          [fs['aux_' + cname][chn]
                           for fs in ref_scores])
                tau = float(np.quantile(za, 0.95))
                mask = f_classes == cls
                zm_cls = zm_all[mask]
                out['%s_%s' % (cname, chn)] = {
                    'tpr5': float(np.mean(zm_cls >= tau)),
                    'fpr5': float(np.mean(ze >= tau)),
                    'auc': float(rank_auc(zm_cls, ze)),
                    'n': int(mask.sum())}
            # overall
            za_all = zcal(
                s['aux_cat'][chn],
                [fs['aux_cat'][chn] for fs in ref_scores])
            tau_all = float(np.quantile(za_all, 0.95))
            out['all_%s' % chn] = {
                'tpr5': float(np.mean(zm_all >= tau_all))}
        return out

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
    # honest baseline
    for i, sd in enumerate(ref_sds[:3]):
        zr = zread_per_class(sd)
        rows.append({'arm': 'HON', 'start': 'ref%d' % i, 'zr': zr})
        print(' HON %d cat_tpr %.2f dog_tpr %.2f frog_tpr %.2f'
              % (i,
                 zr.get('cat_neggrad', {}).get('tpr5', 0),
                 zr.get('dog_neggrad', {}).get('tpr5', 0),
                 zr.get('frog_neggrad', {}).get('tpr5', 0)),
              flush=True)

    # encode
    for sname, s in zip(['s1', 's2', 's3'], START_SEEDS):
        pth = os.path.join(ck, 'start_%s.pt' % sname)
        if os.path.exists(pth):
            sd0 = torch.load(pth, map_location=device)
        else:
            sd0 = ft_unlearn(theta0, s)
            torch.save(sd0, pth)
        sdf = encode(sd0)
        torch.save(sdf, os.path.join(
            ck, 'enc_%s.pt' % sname))
        zr = zread_per_class(sdf)
        s6 = score_sd(sdf)
        m1 = float(ctx['eval_full'](sdf)['m1_acc'])
        rows.append({'arm': 'ENC', 'start': sname,
                     'pass': bool(s6 <= q20),
                     's6': float(s6), 'm1': m1, 'zr': zr})
        print(' ENC %s pass=%s m1 %.3f | cat_tpr %.2f auc %.3f | '
              'dog_tpr %.2f auc %.3f | frog_tpr %.2f auc %.3f'
              % (sname, rows[-1]['pass'], m1,
                 zr.get('cat_neggrad', {}).get('tpr5', 0),
                 zr.get('cat_neggrad', {}).get('auc', 0),
                 zr.get('dog_neggrad', {}).get('tpr5', 0),
                 zr.get('dog_neggrad', {}).get('auc', 0),
                 zr.get('frog_neggrad', {}).get('tpr5', 0),
                 zr.get('frog_neggrad', {}).get('auc', 0)),
              flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'q20': float(q20), 'rows': rows}, fh, indent=1)

    print('\\n=== MULTI-CLASS RESULTS (neggrad) ===', flush=True)
    for cls_name in ('cat', 'dog', 'frog'):
        hon = [r['zr'].get(cls_name + '_neggrad',
                           {}).get('tpr5', 0)
               for r in rows if r['arm'] == 'HON']
        enc = [r['zr'].get(cls_name + '_neggrad',
                           {}).get('tpr5', 0)
               for r in rows if r['arm'] == 'ENC']
        hon_auc = [r['zr'].get(cls_name + '_neggrad',
                               {}).get('auc', 0)
                   for r in rows if r['arm'] == 'HON']
        enc_auc = [r['zr'].get(cls_name + '_neggrad',
                               {}).get('auc', 0)
                   for r in rows if r['arm'] == 'ENC']
        print(' %s: HON tpr %.2f auc %.3f | ENC tpr %.2f auc '
              '%.3f | d_tpr %+.2f d_auc %+.3f'
              % (cls_name,
                 float(np.median(hon)),
                 float(np.median(hon_auc)),
                 float(np.median(enc)),
                 float(np.median(enc_auc)),
                 float(np.median(enc))
                 - float(np.median(hon)),
                 float(np.median(enc_auc))
                 - float(np.median(hon_auc))), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
