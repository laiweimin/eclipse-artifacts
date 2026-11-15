"""C6: concentrated membership encoding attack (new method).

Key insight: audit tolerance is AGGREGATE (mean forget loss, KS
distance over all F). By encoding only |F_target|=5 records with
full budget and leaving 25 as "cover" (zero gradient), the
aggregate stats stay in-band while target records get ~6x signal
density per record.

Arms: uniform CE (= existing, reuse), concentrated-5, conc-10.
Readout: z-TPR@5% on TARGET subset vs NON-TARGET subset
separately (the key differential), plus audit pass + aggregate
z on all F.
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

TARGET_SIZES = (5, 10)
ITERS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c6conc-' + time.strftime('%Y%m%dT%H%M%S')
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

    # z-decoder reference
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in fit_sds]

    def zread_detailed(sd, target_mask):
        """z-TPR@5% separately for target and cover records."""
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
            zt = zm[target_mask]
            zc = zm[~target_mask]
            out['z_' + chn] = {
                'all_tpr5': float(np.mean(zm >= tau)),
                'target_tpr5': float(np.mean(zt >= tau)),
                'cover_tpr5': float(np.mean(zc >= tau)),
                'target_auc': float(rank_auc(
                    zt, ze)) if target_mask.sum() >= 3 else None,
                'fpr5': float(np.mean(ze >= tau))}
        return out

    def encode(sd_start, target_indices):
        """CE encoding on target subset only."""
        n_t = len(target_indices)
        tx = mem[0][target_indices]
        ty = mem[1][target_indices]

        def obj(sd, nx, ny):
            model.load_state_dict(sd)
            model.eval()
            rm = -F.cross_entropy(model(tx), ty)
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

    # starts
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts = [('s1', e3[1]), ('s2', e3[3]), ('s3', e3[4])]

    # target selection: top-|F_target| by base model's F margin
    # (same record difficulty ordering, frozen across arms)
    sd_base = torch.load(e3[1], map_location=device)
    model.load_state_dict(sd_base)
    model.eval()
    with torch.no_grad():
        out = model(mem[0])
        zy = out.gather(1, mem[1][:, None]).squeeze(1)
        zc = out.clone()
        zc.scatter_(1, mem[1][:, None], -1e18)
        margins = (zy - zc.max(1).values).cpu().numpy()
    order = np.argsort(margins)  # LOWEST margin first (hardest, widest ref sigma)

    rows = []
    # uniform baseline: reuse existing CE finals
    for pth in sorted(glob.glob(
            'research_outputs/uw/lfpr-*/ckpt/ce_e3-s*.pt'))[:3]:
        sd = torch.load(pth, map_location=device)
        all_mask = np.ones(30, dtype=bool)
        zr = zread_detailed(sd, all_mask)
        rows.append({'arm': 'uniform', 'start':
                     os.path.basename(pth),
                     'pass': bool(score_sd(sd) <= q20),
                     'zr': zr})
        print(' UNI %s pass=%s all_tpr %.2f'
              % (os.path.basename(pth), rows[-1]['pass'],
                 zr['z_neggrad']['all_tpr5']), flush=True)

    for nt in TARGET_SIZES:
        arm = 'conc%d' % nt
        t_idx = np.sort(order[:nt])
        t_mask = np.zeros(30, dtype=bool)
        t_mask[t_idx] = True
        for sname, sp in starts:
            sd0 = torch.load(sp, map_location=device)
            sdf = encode(sd0, t_idx.tolist())
            torch.save(sdf, os.path.join(
                ck, '%s_%s.pt' % (arm, sname)))
            zr = zread_detailed(sdf, t_mask)
            rows.append({'arm': arm, 'start': sname,
                         'n_target': nt,
                         'target_idx': t_idx.tolist(),
                         'pass': bool(score_sd(sdf) <= q20),
                         'zr': zr})
            print(' %s %s pass=%s | target_tpr %.2f cover_tpr '
                  '%.2f all_tpr %.2f | target_auc %s'
                  % (arm, sname, rows[-1]['pass'],
                     zr['z_neggrad']['target_tpr5'],
                     zr['z_neggrad']['cover_tpr5'],
                     zr['z_neggrad']['all_tpr5'],
                     '%.3f' % zr['z_neggrad']['target_auc']
                     if zr['z_neggrad']['target_auc'] else 'n/a'),
                  flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # gates
    uni = [r for r in rows if r['arm'] == 'uniform']
    c5r = [r for r in rows if r['arm'] == 'conc5']
    c10r = [r for r in rows if r['arm'] == 'conc10']
    gate = {}
    for nm, arm_rows in (('conc5', c5r), ('conc10', c10r)):
        if not arm_rows:
            continue
        all_pass = all(r['pass'] for r in arm_rows)
        t_tpr = [r['zr']['z_neggrad']['target_tpr5']
                 for r in arm_rows]
        c_tpr = [r['zr']['z_neggrad']['cover_tpr5']
                 for r in arm_rows]
        u_tpr = [r['zr']['z_neggrad']['all_tpr5']
                 for r in uni]
        u_med = float(np.median(u_tpr))
        t_med = float(np.median(t_tpr))
        # C-gate: target TPR >= uniform TPR + 0.15 AND all pass
        gate[nm] = {
            'all_pass': all_pass,
            'target_tpr_med': t_med,
            'cover_tpr_med': float(np.median(c_tpr)),
            'uniform_tpr_med': u_med,
            'differential': t_med - u_med,
            'C_gate': bool(all_pass and t_med >= u_med + 0.15)}
        print(' %s C-gate %s (pass=%s target %.2f uniform '
              '%.2f diff %+.2f cover %.2f)'
              % (nm, gate[nm]['C_gate'], all_pass,
                 t_med, u_med, t_med - u_med,
                 gate[nm]['cover_tpr_med']), flush=True)
    with open(os.path.join(outdir, 'gates.json'), 'w') as fh:
        json.dump(gate, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
