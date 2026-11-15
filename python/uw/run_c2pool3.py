"""C2-S3: robustness of the top-k member-blind pool verifier across
pool constructions (prereg card sec.7, frozen before running).

Cells: pool size 50/100(baseline)/200 (|F|=30, same-class decoys);
member count 10/20 (pool=100, F-subsampled, k-mismatch grid
k=5/15/30/50); mixed-class decoys (pool=100, |F|=30). Models = 12
honest (H0) + 12 encoded finals (H1), same as S2. Statistic family
top-k mean; per-cell threshold = q95 of honest control pools;
primary = member increment (H1 treat - H1 decoy-control detection);
leakage vs honest empirical null.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal

N_RES = 50
KS = (5, 15, 30, 50)
CELLS = [('pool50', 30, 50, 'same'),
         ('base', 30, 100, 'same'),
         ('pool200', 30, 200, 'same'),
         ('f10', 10, 100, 'same'),
         ('f20', 20, 100, 'same'),
         ('mixed', 30, 100, 'mixed')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c2pool3-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    all_te = np.arange(len(tset.cifar10.targets))
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    db1 = np.setdiff1d(rest, fresh)
    same_bank = np.sort(np.random.RandomState(7901).choice(
        db1, 200, replace=False))
    mixed_bank = np.sort(np.random.RandomState(7902).choice(
        np.setdiff1d(all_te, cat_te), 400, replace=False))
    mem = (ctx['xF'], ctx['yF'])

    def rec_sub(ids, ref):
        x = torch.tensor(np.stack(
            [tset.cifar10.data[i] / 255.0
             for i in ids])).permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([tset.cifar10.targets[i]
                          for i in ids]).long().to(device)
        return x, y

    # F members + both decoy banks scored per model
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    banks = {'mem': mem,
             'same': rec_sub(same_bank, tset),
             'mixed': rec_sub(mixed_bank, tset)}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in banks.items()}
                  for sd in fit_sds]

    def zvecs(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in banks.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            out[chn] = {nm: zcal(s[nm][chn],
                                 [fs[nm][chn]
                                  for fs in fit_scores])
                        for nm in banks}
        return out

    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6] \
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/'
            'e4_ft_s*.pt'))[:6]
    ce = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:]
    g1 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/'
        'ckpt/fb_s*.pt')) + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T104042/'
            'ckpt/fb_s*.pt'))
    h1_paths = ce + g1
    h1_names = (['CE:%d' % i for i in range(len(ce))]
                + ['GR:%d' % i for i in range(len(g1))])

    def arm_rows(paths, names, arm):
        rows = []
        for pth, nm in zip(paths, names):
            z = zvecs(torch.load(pth, map_location=device))
            for chn in ('neggrad',):  # S2 的有效通道
                zmem = z[chn]['mem']
                zsame = z[chn]['same']
                zmix = z[chn]['mixed']
                rs = np.random.RandomState(8100)
                cells = {}
                for cname, nf, psz, dsrc in CELLS:
                    zp = zsame if dsrc == 'same' else zmix
                    tt, cc, amf_t, amf_c = [], [], [], []
                    for r in range(N_RES):
                        fpos = rs.choice(len(zmem), nf,
                                         replace=False)
                        n_dec = psz - nf
                        dec = rs.choice(len(zp), n_dec,
                                        replace=False)
                        treat = np.concatenate(
                            [zmem[fpos], zp[dec]])
                        ctrl = zp[rs.choice(
                            len(zp), psz, replace=False)]
                        def topk(v, k):
                            s = np.sort(v)[::-1]
                            return float(np.mean(s[:k]))
                        tt.append({('top%d' % k):
                                   topk(treat, k)
                                   for k in KS})
                        cc.append({('top%d' % k):
                                   topk(ctrl, k)
                                   for k in KS})
                        amf_t.append(1.0 if np.argmax(
                            treat) < nf else 0.0)
                        amf_c.append(1.0 if np.argmax(
                            ctrl) < nf else 0.0)
                    cells[cname] = {'t': tt, 'c': cc,
                                    'amf_t': amf_t,
                                    'amf_c': amf_c}
                rows.append({'arm': arm, 'model': nm,
                             'ch': chn, 'cells': cells})
            print(' %s %s scored' % (arm, nm), flush=True)
        return rows

    hon = arm_rows(honest_paths,
                   ['HON:%d' % i
                    for i in range(len(honest_paths))], 'H0')
    h1 = arm_rows(h1_paths, h1_names, 'H1')

    results = {}
    for cname, nf, psz, dsrc in CELLS:
        for k in KS:
            sk = 'top%d' % k
            hr = [r for r in hon]
            h1r = [r for r in h1]
            null = [c[sk] for r in hr
                    for c in r['cells'][cname]['c']]
            q95 = float(np.quantile(null, 0.95))
            def det(rows, side):
                vals = []
                for r in rows:
                    cell = r['cells'][cname]
                    pool = (cell['c'] if side == 'c'
                            else cell['t'])
                    vals += [1 if p[sk] > q95 else 0
                             for p in pool]
                return np.array(vals)
            h0c, h0t = det(hr, 'c'), det(hr, 't')
            h1c, h1t = det(h1r, 'c'), det(h1r, 't')
            lk0 = float(np.mean([
                a for r in hr
                for a in r['cells'][cname]['amf_t']]))
            lk1 = float(np.mean([
                a for r in h1r
                for a in r['cells'][cname]['amf_t']]))
            results['%s|%s' % (cname, sk)] = {
                'q95': q95, 'H0c': float(h0c.mean()),
                'H1c': float(h1c.mean()),
                'H1t': float(h1t.mean()),
                'incr': float(h1t.mean() - h1c.mean()),
                'lk0': lk0, 'lk1': lk1}
            print(' %s|top%d H0c %.3f H1c %.3f H1t %.3f '
                  'incr %+.3f lk %.2f->%.2f'
                  % (cname, k, h0c.mean(), h1c.mean(),
                     h1t.mean(),
                     results['%s|%s' % (cname, sk)]['incr'],
                     lk0, lk1), flush=True)

    # S3 gate: baseline cell (base|top15) must replicate; >=3 of
    # 4 perturbation cells hold at their best-k and at k=15
    def hold(key):
        r = results.get(key)
        return (r is not None and r['incr'] >= 0.15
                and r['H1t'] >= 0.5 and r['H0c'] <= 0.10)

    base_ok = hold('base|top15')
    pert_cells = ['pool50', 'pool200', 'f10', 'mixed']
    best_k_hold = {}
    for c in pert_cells:
        best_k_hold[c] = any(
            hold('%s|top%d' % (c, k)) for k in KS)
    k15_hold = {c: hold('%s|top15' % c) for c in pert_cells}
    n_hold = sum(best_k_hold.values())
    gate = {'base_replicates': base_ok,
            'bestk_holds': best_k_hold,
            'k15_holds': k15_hold,
            'n_pert_hold': n_hold,
            'S3_pass': bool(base_ok and n_hold >= 3)}
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'results': results, 'gate': gate}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('S3 gate', json.dumps(gate), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
