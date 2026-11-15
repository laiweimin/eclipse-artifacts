"""UW E-W4: optimized-wash multi-k curve + multi-start + motive readouts
(frozen prereg docs/plans/2026-09-23-uw-ew4-preregistration.md).

Reuses ew3-20260925T113900 checkpoints/centers (no new training) and the
E-W3b-final optimizer (adaptive backtracking, +margin ascent, feasibility
= Tier-A under self-inclusive honest-ensemble bands per k).  New in this
round: k in {30,300,1000}; 4 starts per (k,seed) (theta_R, two 0.3/0.15
blends toward the other honest seeds, one random direction with a scale
ladder); accepted steps log FULL metrics (E-W3b mechanism-logging gap);
qret/cat-margin recovery readouts; per-(k,seed) 5-step unconstrained
controls.
"""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        interpolate_sd, metrics_from_logits)
from uw.run_ew3b import LR0, LR_MAX, loo_bands, lr_update

RUN = 'research_outputs/uw/ew3-20260925T113900'
K_GRID = (30, 300, 1000)
RT_SEEDS = {30: (44101, 44102, 44103), 300: (44111, 44112, 44113),
            1000: (44121, 44122, 44123)}
MAX_ITERS = 60
STALL_LIMIT = 15
CONTROL_STEPS = 5
CONTROL_GAIN = 0.5
BLEND_TS = (0.3, 0.15)
RAND_BASE = 45500
RAND_SCALES = (0.1, 0.05, 0.02, 0.01)
U_DEN_MIN = 1e-3
PASS_I1_LARGE = 0.10
PASS_I1_SMALL = 0.20


def qret(u0, u_R, u_W, min_den=U_DEN_MIN):
    """Utility recovery fraction; flag when the deletion moves utility by
    less than min_den (vacuous denominator)."""
    den = u0 - u_R
    if abs(den) < min_den:
        return None, True
    return float(1.0 - (u0 - u_W) / den), False


def verdict_from(i1_star, ctrl_fail_ks, skip_frac):
    """Frozen §4: PASS / KILL / UNDECIDED from per-k best I1."""
    if ctrl_fail_ks or skip_frac > 0.5:
        return 'UNDECIDED', ('control failure at {} / skip_frac {:.2f}'
                             .format(sorted(ctrl_fail_ks), skip_frac))
    a = i1_star.get(30, 0.0)
    b = i1_star.get(300, 0.0)
    c = i1_star.get(1000, 0.0)
    if c >= PASS_I1_LARGE or (a >= PASS_I1_SMALL and b >= PASS_I1_SMALL):
        return 'PASS', ('I1*(1000)={:.3f} / I1*(30)={:.3f}, '
                        'I1*(300)={:.3f}'.format(c, a, b))
    if b < PASS_I1_LARGE and c < PASS_I1_LARGE:
        return 'KILL', ('I1*(300)={:.3f} and I1*(1000)={:.3f} both < 0.10'
                        '-- small-k-only loophole'.format(b, c))
    return 'UNDECIDED', ('intermediate shape: I1*(30)={:.3f}, 300={:.3f}, '
                         '1000={:.3f}'.format(a, b, c))


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
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    iters = 8 if cli.smoke else MAX_ITERS
    k_grid = (30,) if cli.smoke else K_GRID
    seeds_of = {k: RT_SEEDS[k][:(1 if cli.smoke else 3)] for k in k_grid}
    starts_of = ('S0', 'S3') if cli.smoke else ('S0', 'S1', 'S2', 'S3')

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ew4-smoke-' if cli.smoke else 'ew4-'
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

    model = get_model('ResNet18').to(device)
    sd0 = torch.load(os.path.join(RUN, 'ckpt', 'theta0.pt'),
                     map_location=device)
    pt = json.load(open(os.path.join(RUN, 'pt.json')))['per_k']
    # bands/centers ALWAYS from the full 3-seed ensemble (a single-seed
    # ensemble collapses the band to the 1e-12 guard -- smoke artifact
    # caught in the first E-W4 smoke); seeds_of only limits which seeds
    # the optimizer runs on.
    rt_sd = {}
    for k in k_grid:
        for s in seeds_of[k]:
            rt_sd[s] = torch.load(
                os.path.join(RUN, 'ckpt', 'rt_k{}_s{}.pt'.format(k, s)),
                map_location=device)
    rt_mm = {k: {s: json.load(open(os.path.join(
        RUN, 'rt_k{}_s{}.json'.format(k, s))))['per_k'][str(k)]
        for s in RT_SEEDS[k]} for k in k_grid}

    fsets = {k: f_deep(pool, k) for k in k_grid}
    pos_of = {k: np.searchsorted(pool, fsets[k]) for k in k_grid}
    centers, bands = {}, {}
    for k in k_grid:
        centers[k] = {m: float(np.median(
            [rt_mm[k][s][m] for s in RT_SEEDS[k]])) for m in TIER_A}
        bands[k] = loo_bands(rt_mm[k], centers[k])

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)

    def eval_model(sd, k):
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
        return metrics_from_logits(z_pool[pos_of[k]],
                                   yfull[fsets[k]], z_sub, y_sub,
                                   z_te, y_te)

    xF, yF = {}, {}
    for k in k_grid:
        xf, yfl = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(trainset_eval, fsets[k]),
                batch_size=500, shuffle=False):
            xf.append(x)
            yfl.append(y)
        xF[k] = torch.cat(xf).to(device)
        yF[k] = torch.cat(yfl).to(device).long()

    def fm_only(sd, k):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF[k])
        zy = z.gather(1, yF[k][:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[k][:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    def grad_delta(sd, k):
        model.load_state_dict(sd)
        model.eval()
        z = model(xF[k])
        zy = z.gather(1, yF[k][:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[k][:, None], -1e18)
        margin = (zy - zc.max(1).values).mean()
        model.zero_grad()
        margin.backward()
        return {nm: p.grad.detach().clone() for nm, p in
                model.named_parameters() if p.grad is not None}

    def apply_delta(base_sd, delta, lr):
        return {k_: (v + lr * delta[k_] if k_ in delta else v)
                for k_, v in base_sd.items()}

    def feasible(sd, k, s):
        mm = eval_model(sd, k)
        ok, viol = accept(mm, centers[k], bands[k][s], TIER_A)
        return ok, viol, mm

    def rand_dir_sd(sd, rng, scale):
        """sd + (scale * ||sd - sd0||) * unit random direction in param
        space (drawn on the named-parameter set)."""
        disp = (sum(float((sd[nm] - sd0[nm]).float().norm() ** 2)
                    for nm, _ in model.named_parameters()) ** 0.5)
        delta = {nm: torch.as_tensor(
            rng.standard_normal(tuple(p.shape)).astype('float32')
        ).to(p.device) for nm, p in model.named_parameters()}
        dn = sum(float(v.norm() ** 2) for v in delta.values()) ** 0.5
        if dn < 1e-12:
            return None
        g = scale * disp / dn
        return {k_: (sd[k_] + g * delta[k_] if k_ in delta else sd[k_])
                for k_ in sd}

    def run_opt(k, s, start_name, sd_start, iters_):
        ok0, _v, mm0 = feasible(sd_start, k, s)
        if not ok0:
            return {'status': 'infeasible_start'}
        fm_R = rt_mm[k][s]['m6_fm']
        fm_0 = pt[str(k)]['m6_fm']
        cur = sd_start
        lr = LR0
        best_fm = mm0['m6_fm']
        accepted_log = [{'iter': 0, 'fm': best_fm, 'metrics': mm0}]
        stall = 0
        for it in range(1, iters_ + 1):
            g = grad_delta(cur, k)
            accepted = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                ok, _vv, mm = feasible(cand, k, s)
                if ok and mm['m6_fm'] > best_fm + 1e-7:
                    cur, best_fm, accepted = cand, mm['m6_fm'], True
                    accepted_log.append({'iter': it, 'fm': best_fm,
                                         'metrics': mm})
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, accepted)
            stall = 0 if accepted else stall + 1
            if stall > STALL_LIMIT:
                break
        mm_best = accepted_log[-1]['metrics']
        return {'status': 'ok', 'start': start_name,
                'iters_run': it, 'n_accepted': len(accepted_log) - 1,
                'fm_start': fm_R, 'fm_best': best_fm,
                'I1': i1(best_fm, fm_R, fm_0),
                'm1_best': mm_best['m1_acc'],
                'util_margin_best': mm_best['util_margin'],
                'cat_margin_best': mm_best['cat_margin'],
                'accepted_log': accepted_log}

    all_runs, ctrl_fail = [], {k: [] for k in k_grid}
    n_skip = n_total = 0
    for k in k_grid:
        for si, s in enumerate(seeds_of[k]):
            others = [t for t in seeds_of[k] if t != s]
            # ---- control
            cfn = 'ctrl_k{}_s{}.json'.format(k, s)
            if not os.path.exists(os.path.join(outdir, cfn)):
                cur = rt_sd[s]
                lr = LR0
                best = fm_only(cur, k)
                f0 = best
                for _ in range(3 if cli.smoke else CONTROL_STEPS):
                    g = grad_delta(cur, k)
                    acc = False
                    for _bt in range(12):
                        cand = apply_delta(cur, g, lr)
                        fmc = fm_only(cand, k)
                        if fmc > best + 1e-7:
                            cur, best, acc = cand, fmc, True
                            break
                        lr *= 0.5
                        if lr < 1e-9:
                            break
                    lr = lr_update(lr, acc)
                ctrl = {'k': k, 'seed': s, 'fm_start': f0,
                        'gain': best - f0}
                with open(os.path.join(outdir, cfn), 'w') as fh:
                    json.dump(ctrl, fh, indent=1)
            else:
                ctrl = json.load(open(os.path.join(outdir, cfn)))
            if ctrl['gain'] < CONTROL_GAIN:
                ctrl_fail[k].append(s)
            print('ctrl', k, s, round(ctrl['gain'], 3), flush=True)
            # ---- starts: S0 base, S1/S2 blends, S3 random ladder
            specs = [('S0', ('base',))]
            for oi, t_s in enumerate(others):
                specs.append(('S{}'.format(oi + 1), ('blend', t_s)))
            specs.append(('S3', ('rand',)))
            for name, spec in specs:
                if name not in starts_of:
                    continue
                fn = 'opt_k{}_s{}_{}.json'.format(k, s, name)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    sd_start = None
                    if spec[0] == 'base':
                        sd_start = rt_sd[s]
                    elif spec[0] == 'blend':
                        for t in BLEND_TS:
                            cand_sd = interpolate_sd(
                                rt_sd[s], rt_sd[spec[1]], t)
                            ok, _vv, _m = feasible(cand_sd, k, s)
                            if ok:
                                sd_start = cand_sd
                                break
                    else:                     # rand ladder
                        rng = np.random.RandomState(
                            RAND_BASE + 100 * k_grid.index(k)
                            + 10 * si + 3)
                        for sc in RAND_SCALES:
                            cand_sd = rand_dir_sd(rt_sd[s], rng, sc)
                            if cand_sd is None:
                                continue
                            ok, _vv, _m = feasible(cand_sd, k, s)
                            if ok:
                                sd_start = cand_sd
                                break
                    if sd_start is None:
                        rec = {'status': 'no_feasible_start'}
                    else:
                        rec = run_opt(k, s, name, sd_start, iters)
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1)
                n_total += 1
                if rec.get('status') != 'ok':
                    n_skip += 1
                all_runs.append({'k': k, 'seed': s, 'start': name,
                                 'rec': rec})
                print('opt', k, s, name, rec.get('status'),
                      round(rec.get('I1', 0.0), 4), flush=True)

    i1_star, best_rec = {}, {}
    for k in k_grid:
        ok_i1 = [r['rec']['I1'] for r in all_runs
                 if r['k'] == k and r['rec'].get('status') == 'ok'
                 and r['seed'] not in ctrl_fail[k]]
        i1_star[k] = float(max(ok_i1)) if ok_i1 else 0.0
        best_rec[k] = next((r for r in all_runs
                            if r['k'] == k
                            and r['rec'].get('status') == 'ok'
                            and abs(r['rec']['I1'] - i1_star[k]) < 1e-12),
                           None)
    motive = {}
    for k in k_grid:
        if best_rec[k] is None:
            continue
        s = best_rec[k]['seed']
        q_u, vac = qret(pt[str(k)]['util_margin'],
                        rt_mm[k][s]['util_margin'],
                        best_rec[k]['rec']['util_margin_best'])
        q_c, vac_c = qret(pt[str(k)]['cat_margin'],
                          rt_mm[k][s]['cat_margin'],
                          best_rec[k]['rec']['cat_margin_best'])
        motive[k] = {'qret_util': q_u, 'qret_cat': q_c,
                     'vacuous_den_util': vac, 'vacuous_den_cat': vac_c,
                     'seed': s, 'start': best_rec[k]['start']}
    cf = [k for k in k_grid if ctrl_fail[k]]
    status, reason = verdict_from(
        {k: i1_star[k] for k in k_grid}, cf, n_skip / max(n_total, 1))
    verdict = {'status': status, 'reason': reason,
               'i1_star': {str(k): i1_star[k] for k in k_grid},
               'motive': {str(k): v for k, v in motive.items()},
               'control_fail': {str(k): ctrl_fail[k] for k in k_grid},
               'skip_frac': n_skip / max(n_total, 1)}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(verdict, fh, indent=1)
    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write('# UW E-W4 (frozen prereg 2026-09-23)\n\nverdict: {}'
                 ' -- {}\n'.format(status, reason))
    print('VERDICT:', status, '--', reason, flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
