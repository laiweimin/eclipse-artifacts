"""C2 gap-1: H1-family diversification control arms (design card
docs/plans/2026-10-16-c2-gap1-diversification-design.md, frozen;
run under the 24h goal's default experiment authorization).

Arms (from theta0 trained WITH F, ew3 run's theta0.pt):
  D1 undertrained: 20-step ft on D\\F
  D2 mis-lr: lr=0.1, 1-epoch ft on D\\F
  D3 partial: half-epoch (~195/390 iters) ft on D\\F
  D4 no-deletion: theta0 returned as-is (positive control)
Protocol = S5 verbatim: 400-record same-class decoy bank
(fresh UNION same_bank), base cell (30 members / pool 100 /
same-class), k in {5,15,30,50}, 10 resample seeds, honest-control
calibration. Gate (frozen): D4 det >= 0.8 AND >=2 of D1-D3
increments >= +10pp -> GO; all D1-D3 < +10pp while H1 (encoded,
from S5 rows) >= +17pp -> STOP ("self-encoder detector").
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal

N_RES = 50
KS = (5, 15, 30, 50)
THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/'
          'theta0.pt')
FT_SEEDS = (44101, 44102)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c2gap1-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(outdir, exist_ok=False)
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
    fset = f_deep(cat_tr, K)
    keep = np.setdiff1d(np.arange(n), fset)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    same_bank = np.setdiff1d(rest, fresh)
    same_bank = np.sort(np.random.RandomState(7901).choice(
        same_bank, 200, replace=False))
    bank400 = np.concatenate([fresh, same_bank])
    mem = (ctx['xF'], ctx['yF'])

    def rec_ids(ids):
        x = torch.tensor(np.stack(
            [tset.cifar10.data[i] / 255.0
             for i in ids])).permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([tset.cifar10.targets[i]
                          for i in ids]).long().to(device)
        return x, y

    from tools.models import resnet18

    def fresh_m():
        return resnet18(num_classes=10).to(device)

    def ft_arm(sd0, s, steps=None, lr=0.01):
        torch.manual_seed(s)
        m = fresh_m()
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        rs = np.random.RandomState(s + 7)
        order = rs.permutation(len(keep))
        total = len(keep) if steps is None else min(
            steps * 100, len(keep))
        done = 0
        m.train()
        while done < total:
            b = keep[order[done % len(keep):
                         done % len(keep) + 100]].astype(int)
            x = torch.stack([trset[int(j)][1]
                             for j in b]).to(device)
            y = torch.tensor(yfull[b]).long().to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
            done += 100
        m.eval()
        return {k: v.detach().clone()
                for k, v in m.state_dict().items()}

    # ---- z machinery (S5 protocol) ----------------------------------
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    banks = {'mem': mem, 'same': rec_ids(bank400)}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in banks.items()}
                  for sd in fit_sds]

    def zvec(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in banks.items()}
        return {'neggrad': {
            'mem': zcal(s['mem']['neggrad'],
                        [fs['mem']['neggrad']
                         for fs in fit_scores]),
            'pool': zcal(s['same']['neggrad'],
                         [fs['same']['neggrad']
                          for fs in fit_scores])}}

    def topk(v, k):
        srt = np.sort(v)[::-1]
        return float(np.mean(srt[:k]))

    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6] \
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/'
            'e4_ft_s*.pt'))[:6]

    def model_pools(sd):
        z = zvec(sd)
        zm, zp = z['neggrad']['mem'], z['neggrad']['pool']
        rows = []
        for rs_i in range(10):
            rs = np.random.RandomState(8100 + 100 * rs_i)
            for r in range(N_RES):
                dec = rs.choice(len(zp), 70, replace=False)
                treat = np.concatenate([zm, zp[dec]])
                ctrl = zp[rs.choice(
                    len(zp), 100, replace=False)]
                rows.append((
                    {('top%d' % k): topk(treat, k)
                     for k in KS},
                    {('top%d' % k): topk(ctrl, k)
                     for k in KS}))
        return rows

    def score_arm(name, sds):
        pools = [model_pools(sd) for sd in sds]
        print(' %s: %d models scored' % (name, len(sds)),
              flush=True)
        return {'name': name, 'pools': pools}

    # honest reference for threshold
    hon_pools = []
    for p in honest_paths:
        hon_pools.append(model_pools(
            torch.load(p, map_location=device)))
    print(' honest: %d models scored' % len(hon_pools),
          flush=True)

    sd0 = torch.load(THETA0, map_location=device)
    arms = {}
    arms['D4'] = [sd0]  # no-deletion positive control
    arms['D1'] = [ft_arm(sd0, s, steps=20) for s in FT_SEEDS]
    arms['D2'] = [ft_arm(sd0, s, lr=0.1) for s in FT_SEEDS]
    arms['D3'] = [ft_arm(sd0, s, steps=195) for s in FT_SEEDS]

    results = {}
    for sk in ['top%d' % k for k in KS]:
        null = [c[sk] for mp in hon_pools
                for t, c in mp]
        q95 = float(np.quantile(null, 0.95))
        det = lambda pools: float(np.mean([
            1 if t[sk] > q95 else 0
            for mp in pools for t, c in mp]))
        h0c = det(hon_pools)
        res = {'q95': q95, 'H0c': h0c}
        # D4 detection (treatment pools vs honest threshold)
        for name in ('D4', 'D1', 'D2', 'D3'):
            mp = [model_pools(sd) for sd in arms[name]]
            # decoy-control detection for member increment
            res[name] = {
                'treatDet': float(np.mean([
                    1 if t[sk] > q95 else 0
                    for m in mp for t, c in m])),
                'ctrlDet': float(np.mean([
                    1 if c[sk] > q95 else 0
                    for m in mp for t, c in m]))}
            res[name]['incr'] = (res[name]['treatDet']
                                 - res[name]['ctrlDet'])
        results[sk] = res
        print(' %s H0c %.3f | D4 %.3f | D1 %.3f(+%.3f) '
              'D2 %.3f(+%.3f) D3 %.3f(+%.3f)'
              % (sk, h0c, res['D4']['treatDet'],
                 res['D1']['treatDet'], res['D1']['incr'],
                 res['D2']['treatDet'], res['D2']['incr'],
                 res['D3']['treatDet'], res['D3']['incr']),
              flush=True)

    # gate: use best-k across D1-D3 (design says arm-level)
    gate = {'D4_sanity': False, 'arms_over_10pp': 0,
            'verdict': None}
    for sk in ['top%d' % k for k in KS]:
        if results[sk]['D4']['treatDet'] >= 0.8:
            gate['D4_sanity'] = True
    best = {a: max(results[sk][a]['incr']
                   for sk in ['top%d' % k for k in KS])
            for a in ('D1', 'D2', 'D3')}
    gate['best_increments'] = best
    gate['arms_over_10pp'] = sum(
        1 for v in best.values() if v >= 0.10)
    if gate['D4_sanity'] and gate['arms_over_10pp'] >= 2:
        gate['verdict'] = 'GO'
    elif (gate['arms_over_10pp'] == 0):
        gate['verdict'] = 'STOP_candidate'
    else:
        gate['verdict'] = 'PARTIAL'
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'results': results, 'gate': gate}, fh,
                  indent=1)
    print('GATE', json.dumps(gate), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
