"""UW R3-P2: cheap-manipulation baseline (frozen prereg
docs/plans/2026-09-23-uw-r3-minimal-loop.md, author-approved).

Question (reviewer R2 Q-c1): is the wash's forget-margin elevation just
confidence inflation reachable by zero-retrain manipulations?

Families, applied to each of the 6 provider retrains of the k=30
theta0-A unit: logit temperature T (exact linear-head identity
z' = T*z), class-bias delta on the deleted class (z'[:,cat] += delta),
and the joint grid. Manipulated logits are EXACTLY what a provider
returning the physically modified model (fc.weight *= T; fc.bias
adjust) would produce, so audit metrics computed on transformed cached
logits equal a full re-evaluation.

Verdict (frozen): best feasible cheap elevation >= 0.9x luck ceiling
-> CHEAP_REACHES_LUCK (attack narrative further deflated); < 0.5x luck
-> CHEAP_LOCKED (optimization is not confidence manipulation);
else IN_BETWEEN_REPORTED.
"""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import loo_bands
from uw.run_r3_p1 import PROV_SEEDS, mk_bands_centers, RUN3, RUN6

K = 30
# T-only 1.00..2.00 step .05; delta-only 0..2.0 step .1; joint coarse
CHEAP_GRIDS = sorted(
    {(round(1.0 + 0.05 * i, 2), 0.0) for i in range(21)}
    | {(1.0, round(0.1 * j, 1)) for j in range(21)}
    | {(round(t, 1), round(d, 1))
       for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
       for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)})


def tx_logits(z, T, delta, cat=CAT_CLS):
    """Exact linear-head manipulation: fc' = T*fc (+delta on cat logit)."""
    out = z * T
    if delta:
        out = out.copy()
        out[:, cat] += delta
    return out


def verdict_cheap(best_elev, luck):
    if best_elev >= 0.9 * luck:
        return 'CHEAP_REACHES_LUCK', (
            'cheap manipulation reaches {:.2f} vs luck {:.3f}; zero-'
            'retrain manipulation suffices'.format(best_elev, luck))
    if best_elev < 0.5 * luck:
        return 'CHEAP_LOCKED', (
            'best cheap elevation {:.3f} << luck {:.3f}; bands lock '
            'confidence inflation'.format(best_elev, luck))
    return 'IN_BETWEEN_REPORTED', (
        'best cheap elevation {:.3f} vs luck {:.3f}'.format(best_elev,
                                                            luck))


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'r3p2-smoke-' if cli.smoke else 'r3p2-'
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)

    device = torch.device('cuda:{}'.format(cli.gpu))
    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    trainset_eval = CIFAR10(train=True,
                            transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]
    fset = f_deep(pool, K)
    pos = np.searchsorted(pool, fset)
    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']

    model = get_model('ResNet18').to(device)

    def ckpt_of(s):
        if s in (44101, 44102, 44103):
            return os.path.join(RUN3, 'ckpt',
                                'rt_k{}_s{}.pt'.format(K, s))
        return os.path.join(RUN6, 'ckpt',
                            'u0_t44500_k30_s{}.pt'.format(s))

    def base_logits(sd):
        model.load_state_dict(sd)
        model.eval()
        zs, ys = [], []
        with torch.no_grad():
            for _id, x, y in testloader:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        z_te = torch.cat(zs).numpy()
        y_te = torch.cat(ys).numpy()
        zs = []
        with torch.no_grad():
            for _g, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return (z_ps[:len(pool)][pos], yfull[fset], z_ps[len(pool):],
                y_sub, z_te, y_te)

    prov_mm = {}
    for s in PROV_SEEDS:
        if s in (44101, 44102, 44103):
            prov_mm[s] = json.load(open(os.path.join(
                RUN3, 'rt_k{}_s{}.json'.format(K, s))))['per_k'][str(K)]
        else:
            prov_mm[s] = json.load(open(os.path.join(
                RUN6, 'u0_t44500_k30_s{}.json'.format(s))))
    centers, bands = mk_bands_centers(prov_mm)
    band = bands[next(iter(bands))]
    fm_med = float(np.median([prov_mm[s]['m6_fm'] for s in PROV_SEEDS]))
    luck = (max(prov_mm[s]['m6_fm'] for s in PROV_SEEDS) - fm_med) \
        / (fm0 - fm_med)

    grid = [(1.0, 0.0)] if cli.smoke else CHEAP_GRIDS
    rows = []
    for s in PROV_SEEDS[:(1 if cli.smoke else 6)]:
        packs = base_logits(torch.load(ckpt_of(s), map_location=device))
        z_F, y_F, z_sub, y_sub, z_te, y_te = packs
        mm_id = metrics_from_logits(*packs)
        assert abs(mm_id['m1_acc'] - prov_mm[s]['m1_acc']) < 1e-6
        best = None
        viol_counts = {m: 0 for m in TIER_A}
        for T, d in grid:
            args2 = (tx_logits(z_F, T, d), y_F, tx_logits(z_sub, T, d),
                     y_sub, tx_logits(z_te, T, d), y_te)
            mm = metrics_from_logits(*args2)
            ok, viol = accept(mm, centers, band, TIER_A)
            for m in viol:
                viol_counts[m] += 1
            elev = (mm['m6_fm'] - fm_med) / (fm0 - fm_med)
            rec = {'seed': s, 'T': T, 'delta': d, 'feasible': ok,
                   'viol': viol, 'fm': mm['m6_fm'], 'elev': elev,
                   'm1': mm['m1_acc']}
            rows.append(rec)
            if ok and (best is None or elev > best['elev']):
                best = rec
        print('seed', s, 'best cheap:',
              None if best is None else
              (round(best['elev'], 3), 'T={}'.format(best['T']),
               'd={}'.format(best['delta'])),
              'viol:', viol_counts, flush=True)

    feas = [r for r in rows if r['feasible']]
    best_elev = max((r['elev'] for r in feas), default=-1.0)
    v, why = verdict_cheap(best_elev, luck)
    out = {'config': {'grid_n': len(grid), 'seeds': list(PROV_SEEDS),
                      'smoke': bool(cli.smoke)},
           'luck_ceiling': luck, 'best_cheap_elevation': best_elev,
           'verdict': v, 'verdict_why': why,
           'n_feasible': len(feas), 'rows': rows}
    with open(os.path.join(outdir, 'p2_results.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT:', v, '|', why, flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
