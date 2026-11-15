"""UW E-X2: stronger search variants vs region capacity (frozen prereg
docs/plans/2026-09-28-uw-ex-multi-config-preregistration.md).

V1 basin-hopping: from the standard wash endpoint, 3 perturbation rounds
(sigma = 1e-4, 1e-3, 1e-3 isotropic on all parameters) each followed by
re-ascent (same accept rule); keep a round only if it ends feasible with
higher fm.  V2 slack-first prepass: 15 accepted steps whose acceptance
maximizes the minimum band slack (along the fm gradient), then standard
fm ascent.  Units: k30-A retrain family (endpoints = r3p3 unaware-wash
checkpoints) and the lowest-gamma E-X1 arm (--ex1).

Estimand: gain = best elevation over the unit's family median minus the
family luck ceiling (card section 1). ALL seeds reported.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, loo_bands, lr_update
from uw.run_ex1 import verdict_family
from uw.run_r3_p1 import PROV_SEEDS, RUN6, RUN3

K = 30
MATERIAL = 0.10
PERT_SIGMAS = (1e-4, 1e-3, 1e-3)
SLACK_STEPS = 15
MAX_ITERS = 60
RE_ASCENT_ITERS = 30
STALL_LIMIT = 15


def min_slack(mm, centers, band):
    return min(band[m] - abs(mm[m] - centers[m]) for m in TIER_A)


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('v1', 'v2'), required=True)
    ap.add_argument('--retrain', action='store_true',
                    help='k30-A retrain family (endpoints from r3p3)')
    ap.add_argument('--ex1', default=None,
                    help='E-X1 output dir for the second unit')
    ap.add_argument('--arm', choices=('a', 'b', 'c'), default=None)
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    iters = 6 if cli.smoke else MAX_ITERS
    slack_steps = 3 if cli.smoke else SLACK_STEPS
    sigmas = PERT_SIGMAS[:(1 if cli.smoke else 3)]

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ex2-smoke-' if cli.smoke else 'ex2-'
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
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']

    # ---- unit selection -------------------------------------------------
    if cli.retrain:
        fam_mm = {}
        for s in PROV_SEEDS:
            fam_mm[s] = json.load(open(os.path.join(
                RUN3, 'rt_k{}_s{}.json'.format(K, s))))['per_k'][str(K)] \
                if s in (44101, 44102, 44103) else json.load(open(
                    os.path.join(RUN6,
                                 'u0_t44500_k30_s{}.json'.format(s))))
        r3p3 = sorted(glob.glob(
            'research_outputs/uw/r3p3-2*'))[-1]
        start_ck = {s: os.path.join(r3p3, 'ckpt_da_s{}.pt'.format(s))
                    for s in PROV_SEEDS}
        base_ck = {s: (os.path.join(RUN3, 'ckpt',
                                    'rt_k{}_s{}.pt'.format(K, s))
                       if s in (44101, 44102, 44103)
                       else os.path.join(RUN6, 'ckpt',
                                         'u0_t44500_k30_s{}.pt'.format(s)))
                   for s in PROV_SEEDS}
        unit = 'retrain'
        seeds = list(PROV_SEEDS)
    else:
        if cli.ex1:
            ex1 = cli.ex1
        else:
            cands = []
            for d in sorted(glob.glob('research_outputs/uw/ex1-2*')):
                vj0 = json.load(open(os.path.join(d, 'verdict.json')))
                if vj0['arm'] == cli.arm:
                    cands.append(d)
            ex1 = cands[-1]
        vj = json.load(open(os.path.join(ex1, 'verdict.json')))
        assert vj['arm'] == cli.arm
        fam_mm = {int(k): v for k, v in vj['honest_mm'].items()}
        seeds = list(fam_mm)
        start_ck = {s: os.path.join(
            ex1, 'ckpt', 'arm{}_wash_s{}.pt'.format(cli.arm, s))
            for s in seeds}
        base_ck = {s: os.path.join(
            ex1, 'ckpt', 'arm{}_s{}.pt'.format(cli.arm, s))
            for s in seeds}
        unit = 'ex1_' + cli.arm
        r3p3 = None

    model = get_model('ResNet18').to(device)
    med = {m: float(np.median([e[m] for e in fam_mm.values()]))
           for m in TIER_A}
    bands = loo_bands(fam_mm, med)
    band = bands[next(iter(bands))]
    fm_med = float(np.median([e['m6_fm'] for e in fam_mm.values()]))
    E = fm0 - fm_med
    luck = (max(e['m6_fm'] for e in fam_mm.values()) - fm_med) / E

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)

    def eval_model(sd):
        model.load_state_dict(sd)
        zs, ys = [], []
        model.eval()
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
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset), batch_size=500,
            shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()

    def grad_delta(sd):
        model.load_state_dict(sd)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        margin = (zy - zc.max(1).values).mean()
        model.zero_grad()
        margin.backward()
        return {nm: p.grad.detach().clone() for nm, p in
                model.named_parameters() if p.grad is not None}

    def apply_delta(base_sd, delta, lr):
        return {k_: (v + lr * delta[k_] if k_ in delta else v)
                for k_, v in base_sd.items()}

    def perturb(sd, sigma, gen):
        out = {}
        for k_, v in sd.items():
            if v.dtype.is_floating_point:
                noise = torch.randn(v.shape, generator=gen,
                                    device=v.device)
                out[k_] = v + sigma * noise
            else:
                out[k_] = v
        return out

    def ascend(start, start_fm, max_iters, slack_mode=False):
        cur, lr, best_fm = start, LR0, start_fm
        stall = 0
        cur_slack = min_slack(eval_model(cur), med, band)
        for _it in range(1, max_iters + 1):
            g = grad_delta(cur)
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                mm = eval_model(cand)
                ok, _v = accept(mm, med, band, TIER_A)
                if slack_mode:
                    take = ok and min_slack(mm, med, band) > cur_slack \
                        + 1e-9
                else:
                    take = ok and mm['m6_fm'] > best_fm + 1e-7
                if take:
                    cur = cand
                    acc = True
                    cur_slack = min_slack(mm, med, band)
                    best_fm = max(best_fm, mm['m6_fm'])
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return cur, best_fm

    rows = []
    n_seeds = 2 if cli.smoke else len(seeds)
    for s in seeds[:n_seeds]:
        fnj = 'v_{}_{}_s{}.json'.format(cli.variant, unit, s)
        if os.path.exists(os.path.join(outdir, fnj)):
            rec = json.load(open(os.path.join(outdir, fnj)))
        else:
            gen = torch.Generator(device=device).manual_seed(50000 + s)
            if cli.variant == 'v1':
                cur = torch.load(start_ck[s], map_location=device)
                base_fm = eval_model(cur)['m6_fm']
                best_fm = base_fm
                rounds_kept = 0
                for sig in sigmas:
                    cand = perturb(cur, sig, gen)
                    cand2, fm2 = ascend(cand, -1e9, RE_ASCENT_ITERS)
                    mm2 = eval_model(cand2)
                    if accept(mm2, med, band, TIER_A)[0] \
                            and fm2 > best_fm + 1e-7:
                        cur, best_fm, rounds_kept = cand2, fm2, \
                            rounds_kept + 1
                final = eval_model(cur)
                rec = {'fm': final['m6_fm'],
                       'base_fm': base_fm, 'rounds_kept': rounds_kept,
                       'final_mm': final,
                       'I1': i1(final['m6_fm'], fam_mm[s]['m6_fm'], fm0)}
            else:
                cur = torch.load(base_ck[s], map_location=device)
                cur, _ = ascend(cur, eval_model(cur)['m6_fm'],
                                slack_steps, slack_mode=True)
                cur, best_fm = ascend(cur, eval_model(cur)['m6_fm'],
                                      iters)
                final = eval_model(cur)
                rec = {'fm': final['m6_fm'], 'final_mm': final,
                       'I1': i1(final['m6_fm'], fam_mm[s]['m6_fm'], fm0)}
            with open(os.path.join(outdir, fnj), 'w') as fh:
                json.dump(rec, fh, indent=1)
        rows.append({'seed': s, 'fm': rec['fm'],
                     'elev': (rec['fm'] - fm_med) / E,
                     'I1': rec['I1']})
        print(unit, cli.variant, s, 'I1', round(rec['I1'], 4), 'elev',
              round((rec['fm'] - fm_med) / E, 3), flush=True)

    best_elev = max(r['elev'] for r in rows)
    gain = best_elev - luck
    v, why = verdict_family(gain, best_elev)
    out = {'unit': unit, 'variant': cli.variant, 'luck': luck,
           'best_elev': best_elev, 'gain': gain, 'verdict': v,
           'verdict_why': why, 'rows': rows}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('EX2', unit, cli.variant, ':', v, '|', why, flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
