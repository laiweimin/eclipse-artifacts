"""UW E-W12: wash-path Pareto + binding evolution (P/Q, one GPU run
with step logging) + cross-probe robustness (R, existing wash ckpts).
Frozen prereg 2026-09-27-uw-ew12-*."""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, loo_bands, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = sorted(__import__('glob').glob(
    'research_outputs/uw/ew6-2*'))[-1]
RUN10L = sorted(__import__('glob').glob(
    'research_outputs/uw/ew10L-2*'))[-1]
S2_SEED = 46400
MAX_ITERS = 60
STALL_LIMIT = 15


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf
    device = torch.device('cuda:0')
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    outdir = os.path.join('research_outputs', 'uw',
                          'ew12-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)

    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    evalset = CIFAR10(train=True,
                      transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    fset = f_deep(pool, 30)
    pos_f = np.searchsorted(pool, fset)
    sub1 = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    model = get_model('ResNet18').to(device)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    pt3 = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k']

    seeds6 = [44101, 44102, 44103, 45201, 45202, 45203]
    sd6 = {}
    for s in seeds6:
        p = (os.path.join(RUN3, 'ckpt',
                          'rt_k30_s{}.pt'.format(s))
             if s < 45200 else os.path.join(
                 RUN6, 'ckpt',
                 'u0_t44500_k30_s{}.pt'.format(s)))
        sd6[s] = torch.load(p, map_location=device)
    sdw = {s: torch.load(os.path.join(
        RUN10L, 'ckpt', 'wash_s{}.pt'.format(s)),
        map_location=device) for s in seeds6}

    def batch_F():
        lx, ly = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(evalset, fset),
                batch_size=500, shuffle=False):
            lx.append(x)
            ly.append(y)
        return torch.cat(lx).to(device), torch.cat(ly).to(
            device).long()

    xF, yF = batch_F()

    def eval_sub(sd, sub):
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(evalset,
                                    np.concatenate([pool, sub])),
            batch_size=500, shuffle=False)
        model.load_state_dict(sd)
        model.eval()
        zs, ys = [], []
        with torch.no_grad():
            for _i, x, y in testloader:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        z_te = torch.cat(zs).numpy()
        y_te = torch.cat(ys).numpy()
        zs = []
        with torch.no_grad():
            for _g, x, _y in loader:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return metrics_from_logits(
            z_ps[:len(pool)][pos_f], yfull[fset],
            z_ps[len(pool):], yfull[sub], z_te, y_te)

    # ---- reference ensembles under S1 and S2
    sub2 = np.sort(np.random.RandomState(S2_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    mm1 = {s: eval_sub(sd6[s], sub1) for s in seeds6}
    mm2 = {s: eval_sub(sd6[s], sub2) for s in seeds6}
    cen1 = {m: float(np.median([mm1[s][m] for s in seeds6]))
            for m in TIER_A}
    bnd1 = {m: max(abs(mm1[s][m] - cen1[m]) for s in seeds6)
            for m in TIER_A}
    cen2 = {m: float(np.median([mm2[s][m] for s in seeds6]))
            for m in TIER_A}
    bnd2 = {m: max(abs(mm2[s][m] - cen2[m]) for s in seeds6)
            for m in TIER_A}
    fm0 = pt3['30']['m6_fm']

    # ---- Arm P+Q: re-run wash on two leaking seeds with full logging
    pq_out = {}
    for s in (44101, 44102):
        cur = sd6[s]
        lr = LR0
        best = eval_sub(cur, sub1)['m6_fm']
        log = [{'iter': 0, 'fm': best,
                'metrics': eval_sub(cur, sub1)}]
        stall = 0
        for it in range(1, MAX_ITERS + 1):
            model.load_state_dict(cur)
            model.eval()
            z = model(xF)
            zy = z.gather(1, yF[:, None]).squeeze(1)
            zc = z.clone()
            zc.scatter_(1, yF[:, None], -1e18)
            margin = (zy - zc.max(1).values).mean()
            model.zero_grad()
            margin.backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(12):
                cand = {k_: (v + lr * g[k_] if k_ in g else v)
                        for k_, v in cur.items()}
                mm = eval_sub(cand, sub1)
                ok, _v = accept(mm, cen1,
                                {m: max(bnd1[m], 1e-12)
                                 for m in TIER_A}, TIER_A)
                if ok and mm['m6_fm'] > best + 1e-7:
                    cur, best, acc = cand, mm['m6_fm'], True
                    log.append({'iter': it, 'fm': best,
                                'metrics': mm})
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        pq_out[str(s)] = log
        m1s = [e['metrics']['m1_acc'] for e in log]
        print('P/Q seed {}: {} steps, m1 range {:.2f}pp'.format(
            s % 100, len(log) - 1, (max(m1s) - min(m1s)) * 100))
        # binding evolution: which metric saturates first
        sat_order = []
        for e in log[1:]:
            for m in TIER_A:
                ratio = abs(e['metrics'][m] - cen1[m]) / max(
                    bnd1[m], 1e-12)
                if ratio >= 0.8 and m not in sat_order:
                    sat_order.append(m)
        print('  first saturated: {}'.format(sat_order[:3]))
        pq_out[str(s) + '_sat'] = sat_order[:3]

    # ---- Arm R: cross-probe test on existing washes
    r_out = {}
    n_pass2 = 0
    for s in seeds6:
        wm2 = eval_sub(sdw[s], sub2)
        ok2, viol2 = accept(wm2, cen2,
                            {m: max(bnd2[m], 1e-12) for m in TIER_A},
                            TIER_A)
        r_out[str(s)] = {'pass_S2': ok2,
                         'viol_S2': viol2 if not ok2 else []}
        if ok2:
            n_pass2 += 1
        print('R seed {}: S2 pass={} viol={}'.format(
            s % 100, ok2, viol2 if not ok2 else ''))
    out = {'pq': pq_out, 'r_per_seed': r_out,
           'r_n_pass_S2': n_pass2,
           'r_verdict': ('GENERALIZES' if n_pass2 >= 5 else
                         'PROBE_SPECIFIC')}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT(R):', out['r_verdict'],
          '({}/{} pass S2)'.format(n_pass2, len(seeds6)))
    print('saved', outdir)


if __name__ == '__main__':
    main()
