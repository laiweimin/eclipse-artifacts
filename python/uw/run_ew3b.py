"""UW E-W3b: optimized-wash loophole check (frozen mini-prereg
docs/plans/2026-09-23-uw-ew3b-wash-loophole.md).

Question: does the E-W3 acceptance region (Tier-A under LOO honest-ensemble
bands -- the minimal band any deployable audit must grant) contain a model
with material F influence, reachable by (1) constrained gradient ascent on
F margin from theta_R or (2) greedy per-layer-group rollback of the honest
update?  Positive control (unconstrained ascent) guards optimizer weakness.
All inputs reused from ew3-20260925T113900; no new training.
"""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)

RUN = 'research_outputs/uw/ew3-20260925T113900'
K = 30
RT_SEEDS = (44101, 44102, 44103)
LR0 = 1e-3          # erratum: 1e-4 gave first-step effects below the
#                   acceptance threshold (delta_fm ~ lr*||g||^2)
LR_MAX = 3e-3
MAX_ITERS = 150
STALL_LIMIT = 20
CONTROL_STEPS = 50
CONTROL_GAIN = 0.5
I1_PASS = 0.20


def loo_bands(entries, centers):
    """Self-inclusive honest-ensemble band (erratum 2026-09-23): per
    metric, max |dev from center| over ALL seeds -- the argmax seed's dev
    equals the band so it passes with equality (c=0 feasible for every
    seed).  Slightly wider than the LOO variant (attacker-favorable, so
    AUDIT-CLOSED conclusions are the conservative direction)."""
    out = {}
    for s in entries:
        out[s] = {m: max(max(abs(entries[t][m] - centers[m])
                             for t in entries), 1e-12) for m in TIER_A}
    return out


def pick_start_t(ts, feas_flags):
    """Smallest t whose blend is feasible; None if none."""
    for t, f in zip(ts, feas_flags):
        if f:
            return float(t)
    return None


def lr_update(lr, accepted):
    return min(lr * 1.1, LR_MAX) if accepted else lr * 0.5


def group_of(name):
    """Frozen layer-group assignment for the greedy rollback."""
    if 'running_mean' in name or 'running_var' in name:
        return 'bn_stats'
    if name.startswith('bn') or '.bn' in name:
        return 'bn_params'
    if name == 'fc.weight':
        return 'fc_w'
    if name == 'fc.bias':
        return 'fc_b'
    for lay in ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'):
        if name.startswith(lay):
            return lay
    return '_none'


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    iters = 8 if a.smoke else MAX_ITERS
    ctl_steps = 3 if a.smoke else CONTROL_STEPS
    greedy_seed = 44101

    tag = 'ew3b-smoke-' if a.smoke else 'ew3b-'
    outdir = os.path.join('research_outputs', 'uw',
                          tag + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:{}'.format(a.gpu))

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
    fset = f_deep(pool, K)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    pos_f = np.searchsorted(pool, fset)
    y_F = yfull[fset]
    y_sub = yfull[sub]

    model = get_model('ResNet18').to(device)
    sd0 = torch.load(os.path.join(RUN, 'ckpt', 'theta0.pt'),
                     map_location=device)
    rt_sd = {s: torch.load(os.path.join(RUN, 'ckpt',
                                        'rt_k30_s{}.pt'.format(s)),
                           map_location=device) for s in RT_SEEDS}
    centers_full = json.load(open(os.path.join(RUN, 'centers.json')))
    centers = centers_full[str(K)]
    rt_mm = {s: json.load(open(os.path.join(
        RUN, 'rt_k30_s{}.json'.format(s))))['per_k'][str(K)]
        for s in RT_SEEDS}
    bands = loo_bands(rt_mm, centers)
    fm0 = json.load(open(os.path.join(RUN, 'pt.json')))['per_k'][str(K)]['m6_fm']

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
        zs, ids = [], []
        with torch.no_grad():
            for gid, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
                ids.append(gid)
        z_ps = torch.cat(zs).numpy()
        ids_ps = torch.cat(ids).numpy()
        row = {int(g): i for i, g in enumerate(ids_ps)}
        z_pool = z_ps[[row[int(g)] for g in pool]]
        z_sub = z_ps[[row[int(g)] for g in sub]]
        return metrics_from_logits(z_pool[pos_f], y_F, z_sub, y_sub,
                                   z_te, y_te)

    def feasible(sd, s):
        mm = eval_model(sd)
        ok, viol = accept(mm, centers, bands[s], TIER_A)
        return ok, viol, mm

    # F batch for gradients (canonical images)
    xf, yfl = [], []
    loader_f = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, fset), batch_size=500,
        shuffle=False)
    for _gid, x, y in loader_f:
        xf.append(x)
        yfl.append(y)
    xF = torch.cat(xf).to(device)
    yF = torch.cat(yfl).to(device).long()

    def grad_delta(sd):
        """ASCENT gradient of mean F margin wrt each parameter
        (eval-mode BN).  erratum: the first version backpropagated
        -margin and stepped +lr*grad -- a descent step (caught by the
        positive control: fixed-lr smoke diverged to -300, adaptive
        smoke accepted nothing); backprop +margin so +lr*grad ascends."""
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
        out = {}
        for k, v in base_sd.items():
            if k in delta:
                out[k] = v + lr * delta[k]
            else:
                out[k] = v
        return out

    results = {'run_id': os.path.basename(outdir), 'smoke': a.smoke,
               'k': K, 'bands_loo': {str(s): bands[s] for s in RT_SEEDS},
               'centers': centers, 'fm0': fm0, 'per_seed': {}}

    def fm_only(sd):
        """Light F-margin read (canonical F images, no test pass)."""
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    # ---- positive control: UNCONSTRAINED adaptive ascent from each
    # theta_R (erratum: fixed-lr diverged; same backtracking as the main
    # loop minus feasibility)
    ctrl = {}
    for s in RT_SEEDS:
        cur = rt_sd[s]
        lr = LR0
        best = fm_only(cur)
        fm_start = best
        for _ in range(ctl_steps):
            g = grad_delta(cur)
            accepted = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                fmc = fm_only(cand)
                if fmc > best + 1e-7:
                    cur, best, accepted = cand, fmc, True
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, accepted)
        ctrl[s] = {'fm_start': fm_start, 'fm_end': best,
                   'gain': best - fm_start}
        print('control', s, 'gain', round(best - fm_start, 4), flush=True)
    ctrl_ok = all(c['gain'] >= CONTROL_GAIN for c in ctrl.values())
    with open(os.path.join(outdir, 'control.json'), 'w') as fh:
        json.dump({'control': ctrl, 'ctrl_ok': ctrl_ok,
                   'threshold': CONTROL_GAIN}, fh, indent=1)

    # ---- constrained optimization per seed (self-inclusive bands make
    # every seed feasible at c=0; the infeasible-start branch is a guard)
    opt_summary = {}
    for s in RT_SEEDS:
        ok0, _v, mm0 = feasible(rt_sd[s], s)
        t0 = 0.0
        sd_start = rt_sd[s]
        if not ok0:
            opt_summary[s] = {'status': 'no_feasible_start',
                              'I1': 0.0}
            print('opt', s, 'no feasible start (guard hit)', flush=True)
            continue
        fm_R = rt_mm[s]['m6_fm']
        cur = sd_start
        lr = LR0
        stall = 0
        best_fm = eval_model(cur)['m6_fm']
        log = [{'iter': 0, 't0': t0, 'fm': best_fm, 'feas': True}]
        for it in range(1, iters + 1):
            g = grad_delta(cur)
            accepted = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                ok, _v, mm = feasible(cand, s)
                if ok and mm['m6_fm'] > best_fm + 1e-7:
                    cur, best_fm, accepted = cand, mm['m6_fm'], True
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, accepted)
            if not accepted:
                stall += 1
            else:
                stall = 0
            log.append({'iter': it, 'lr': lr, 'fm': best_fm,
                        'accepted': accepted})
            if stall > (2 if a.smoke else STALL_LIMIT):
                break
        _, _, mm_best = feasible(cur, s)
        rec = {'status': 'ok', 't0': t0, 'iters_run': len(log) - 1,
               'fm_start': fm_R, 'fm_best': best_fm,
               'I1': i1(best_fm, fm_R, fm0),
               'm1_best': mm_best['m1_acc'], 'log': log}
        opt_summary[s] = rec
        with open(os.path.join(outdir, 'opt_s{}.json'.format(s)), 'w') as fh:
            json.dump(rec, fh, indent=1)
        print('opt', s, 'I1', round(rec['I1'], 4), flush=True)

    # ---- greedy per-layer rollback for the c=0-feasible seed
    greedy = None
    s = greedy_seed
    ok0, _v, _m = feasible(rt_sd[s], s)
    if ok0:
        groups = {}
        for k_ in rt_sd[s]:
            groups.setdefault(group_of(k_), []).append(k_)
        sd_cur = rt_sd[s]
        c_cur = {g: 0.0 for g in groups}
        cs = [round(0.05 * i, 2) for i in range(21)]
        for g in sorted(groups):
            if g == '_none':
                continue
            best_c = 0.0
            for c in cs:
                trial = dict(sd_cur)
                for key in groups[g]:
                    trial[key] = sd_cur[key] + c * (sd0[key]
                                                    - sd_cur[key])
                ok, _vv, _mm = feasible(trial, s)
                if ok:
                    best_c = c
                else:
                    break
            sd_cur = {k_: (sd_cur[k_] + best_c * (sd0[k_] - sd_cur[k_]))
                      if k_ in groups[g] else sd_cur[k_]
                      for k_ in sd_cur}
            c_cur[g] = best_c
        mm_g = eval_model(sd_cur)
        num = sum(float((sd_cur[k_] - sd0[k_]).float().norm() ** 2)
                  for k_ in sd_cur)
        den = sum(float((rt_sd[s][k_] - sd0[k_]).float().norm() ** 2)
                  for k_ in sd_cur)
        rho = 1.0 - (num ** 0.5) / (den ** 0.5)
        greedy = {'seed': s, 'c_per_group': c_cur, 'm6_fm': mm_g['m6_fm'],
                  'I1': i1(mm_g['m6_fm'], rt_mm[s]['m6_fm'], fm0),
                  'rho_param': rho, 'm1': mm_g['m1_acc']}
        with open(os.path.join(outdir, 'greedy_s{}.json'.format(s)),
                  'w') as fh:
            json.dump(greedy, fh, indent=1)
        print('greedy I1', round(greedy['I1'], 4), 'rho',
              round(rho, 4), flush=True)

    # ---- verdict (frozen §2)
    best_i1 = 0.0
    for s, rec in opt_summary.items():
        if rec['status'] == 'ok':
            best_i1 = max(best_i1, rec['I1'])
    if greedy:
        best_i1 = max(best_i1, greedy['I1'])
    if not ctrl_ok:
        status, reason = 'UNDECIDED', (
            'positive control failed: optimizer too weak to claim closure')
    elif best_i1 >= I1_PASS:
        status, reason = 'LOOPHOLE', (
            'feasible model with I1 >= 0.20 found -- UW partially reopens')
    else:
        status, reason = 'AUDIT-CLOSED', (
            'control passed; constrained optimization and greedy rollback '
            'both stall below I1 = 0.20 -- E-W3 closure extends from '
            'interpolation to locally-optimized and non-uniform washes')
    verdict = {'status': status, 'reason': reason, 'best_I1': best_i1,
               'control_ok': ctrl_ok, 'opt': {str(s): {
                   'status': r.get('status'),
                   'I1': r.get('I1')} for s, r in opt_summary.items()},
               'greedy_I1': greedy['I1'] if greedy else None}
    results['verdict'] = verdict
    with open(os.path.join(outdir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(verdict, fh, indent=1)
    print('VERDICT:', status, '--', reason, flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
