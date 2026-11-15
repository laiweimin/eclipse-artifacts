"""C8: repeated-request compositional leakage attack.

New principle: provider serves the SAME deletion request k times
(different encoding random seeds). Each return model passes the
audit independently. Adversary averages z-scores across k models
-> noise drops by sqrt(k), signal adds linearly.

Test: encode same start with 5 different seeds, evaluate (i)
individual TPR@5%, (ii) averaged-z TPR@5%. Compare against
single-model baseline.
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

N_ENCODE = 5
ITERS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c8rep-' + time.strftime('%Y%m%dT%H%M%S')
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

    def encode(sd_start, enc_seed):
        """CE encoding with specific random seed for optimizer."""

        def obj(sd, nx, ny):
            model.load_state_dict(sd)
            model.eval()
            rm = -F.cross_entropy(model(mem[0]), mem[1])
            contrast = F.cross_entropy(model(nx), ny)
            return rm.mean() - 0.5 * contrast

        cur = sd_start
        lr = LR0
        rs = np.random.RandomState(enc_seed)
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

    # z-decoder machinery
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in fit_sds]

    def zscores(sd):
        """Return raw z-vectors (not just summary) for averaging."""
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
            out[chn] = {'mem': zm, 'aux': za, 'eval': ze}
        return out

    def wp_from_z(zm, za, ze):
        tau = float(np.quantile(za, 0.95))
        return {'tpr5': float(np.mean(zm >= tau)),
                'fpr5': float(np.mean(ze >= tau)),
                'auc': float(rank_auc(zm, ze))}

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts = [('s1', e3[1]), ('s2', e3[3])]

    rows = []
    for sname, sp in starts:
        sd0 = torch.load(sp, map_location=device)
        all_z = {chn: {'mem': [], 'aux': [], 'eval': []}
                 for chn in ('negloss', 'neggrad')}
        for k in range(N_ENCODE):
            enc_seed = 54000 + k * 111
            sdf = encode(sd0, enc_seed)
            torch.save(sdf, os.path.join(
                ck, '%s_enc%d.pt' % (sname, k)))
            zs = zscores(sdf)
            for chn in ('negloss', 'neggrad'):
                all_z[chn]['mem'].append(zs[chn]['mem'])
                all_z[chn]['aux'].append(zs[chn]['aux'])
                all_z[chn]['eval'].append(zs[chn]['eval'])
            # individual model working point
            wp = {chn: wp_from_z(zs[chn]['mem'],
                                 zs[chn]['aux'],
                                 zs[chn]['eval'])
                  for chn in ('negloss', 'neggrad')}
            rows.append({'start': sname, 'k': k,
                         'pass': bool(score_sd(sdf) <= q20),
                         'wp': wp})
            print(' %s enc%d pass=%s tpr5(gr) %.2f'
                  % (sname, k, rows[-1]['pass'],
                     wp['neggrad']['tpr5']), flush=True)

        # compositional: average z across k models
        for chn in ('negloss', 'neggrad'):
            for nk in range(2, N_ENCODE + 1):
                zm_avg = np.mean(
                    all_z[chn]['mem'][:nk], axis=0)
                za_avg = np.mean(
                    all_z[chn]['aux'][:nk], axis=0)
                ze_avg = np.mean(
                    all_z[chn]['eval'][:nk], axis=0)
                wp_avg = wp_from_z(zm_avg, za_avg, ze_avg)
                rows.append({
                    'start': sname, 'k': 'avg%d' % nk,
                    'wp': {chn: wp_avg}})
                print(' %s avg%d(%s) tpr5 %.2f fpr %.3f '
                      'auc %.3f'
                      % (sname, nk, chn,
                         wp_avg['tpr5'], wp_avg['fpr5'],
                         wp_avg['auc']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # summary: individual medians vs averaged
    ind = [r for r in rows if isinstance(r['k'], int)]
    for chn in ('neggrad',):
        ind_tpr = [r['wp'][chn]['tpr5'] for r in ind
                   if chn in r.get('wp', {})]
        print('INDIVIDUAL tpr5(%s): %s med=%.2f'
              % (chn,
                 ['%.2f' % t for t in ind_tpr],
                 float(np.median(ind_tpr))), flush=True)
        for nk in (3, 5):
            avgs = [r['wp'][chn]['tpr5']
                    for r in rows
                    if r['k'] == 'avg%d' % nk
                    and chn in r.get('wp', {})]
            if avgs:
                print('AVG%d tpr5(%s): %s med=%.2f'
                      % (nk, chn,
                         ['%.2f' % t for t in avgs],
                         float(np.median(avgs))), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
