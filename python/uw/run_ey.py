"""UW E-Y: adaptive attack on the conformal-calibrated audit (frozen
prereg docs/plans/2026-09-30-uw-ey-adaptive-conformal-attack.md).

Oracle-knowledge attacker ascends one of three harm objectives
(mean / bottom-10 / top-10 per-record margin on F, subsets frozen by
theta0's per-record margins) from the e3 ensemble's six honest models,
under the constraint s(theta) <= q_hat@0.20 of the FROZEN fixed-split
conformal audit (fit=ew9f 6, cal=p1aud+e2 12; centers/scales from fit
only). Selection baseline = best honest candidate ON THE SAME OBJECTIVE
from the same pool. Zero new training.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        build_pool, f_deep, metrics_from_logits)
from uw.run_ew3b import LR0, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
K = 30
METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
FIT_TAGS = ('ew9f', )
CAL_TAGS = ('p1aud', 'e2')
START_TAGS = ('e3', 'e4')
GATE = 0.10
MAX_ITERS = 60
STALL_LIMIT = 15


def record_subsets(theta0_margins, k=10):
    """Frozen bottom-k / top-k record indices by theta0 margin."""
    order = np.argsort(theta0_margins, kind='stable')
    return order[:k], order[-k:]


def gain_of(atk, sel, med, theta0):
    e = theta0 - med
    return (atk - med) / e - (sel - med) / e


def verdict_ey(gains, replicated):
    mat = [k for k, g in gains.items() if g >= GATE]
    if mat and replicated:
        return 'MATERIAL', ('objectives over gate and replicated: {}'
                            .format(mat))
    if mat:
        return 'CANDIDATE_UNREPLICATED', ('over gate, unreplicated: {}'
                                          .format(mat))
    return 'NULL', ('no objective reaches {} ({})'.format(
        GATE, {k: round(v, 3) for k, v in gains.items()}))


def conformal_q(scores, alpha):
    s = sorted(scores)
    idx = math.ceil((len(s) + 1) * (1.0 - alpha))
    return s[idx - 1] if idx <= len(s) else float('inf')


def per_record_margin(z, y):
    zy = z[np.arange(len(y)), y]
    zc = z.copy()
    zc[np.arange(len(y)), y] = -1e18
    return zy - zc.max(axis=1)


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
    iters = 6 if cli.smoke else MAX_ITERS
    n_starts = 2 if cli.smoke else 6

    tag = 'ey-smoke-' if cli.smoke else 'ey-'
    outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw', tag + time.strftime('%Y%m%dT%H%M%S'))
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
    model = get_model('ResNet18').to(device)

    # ---- frozen audit (fit-only centers/scales, cal threshold) --------
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    fit = [e for t in FIT_TAGS for e in ens_mm[t]]
    c = {m: float(np.median([e[m] for e in fit])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit]) - c[m]))) for m in METRICS}
    cal_s = [max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                 for m in METRICS)
             for t in CAL_TAGS for e in ens_mm[t]]
    q = conformal_q(cal_s, 0.20)

    def score_of(mm):
        return max(abs(mm[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    # ---- per-record margins of theta0 and honest pools ----------------
    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset),
            batch_size=500, shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()

    def margins_of(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF).cpu().numpy()
        return per_record_margin(z, yF.cpu().numpy())

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_margins = margins_of(sd0)
    bot, top = record_subsets(th_margins, k=10)
    print('theta0 margins: min %.2f med %.2f max %.2f | bottom10 '
          'boundary %.2f, top10 boundary %.2f'
          % (th_margins.min(), np.median(th_margins), th_margins.max(),
             th_margins[bot].max(), th_margins[top].min()), flush=True)

    def obj_val(margins, obj):
        if obj == 'O1':
            return float(margins.mean())
        if obj == 'O2':
            return float(margins[bot].mean())
        return float(margins[top].mean())

    bot_t = torch.as_tensor(bot, device=device)
    top_t = torch.as_tensor(top, device=device)

    def obj_torch(z, obj):
        """Differentiable objective on logits z (batch = F)."""
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        m = zy - zc.max(1).values
        if obj == 'O1':
            return m.mean()
        if obj == 'O2':
            return m[bot_t].mean()
        return m[top_t].mean()

    # honest pools per start-tag: margins + objectives
    pools = {}
    for st in START_TAGS:
        paths = sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/{}_ft_s*.pt'.format(st)))
        mgs = [margins_of(torch.load(p, map_location=device))
               for p in paths]
        pools[st] = {'paths': paths, 'margins': mgs,
                     'objs': {o: [obj_val(mg, o) for mg in mgs]
                              for o in ('O1', 'O2', 'O3')}}
        print('pool', st, 'O1 range %.2f-%.2f'
              % (min(pools[st]['objs']['O1']),
                 max(pools[st]['objs']['O1'])), flush=True)

    def eval_full(sd):
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

    def ascend(start_sd, obj):
        """Feasible-direction ascent on obj under s <= q (oracle)."""
        cur = start_sd
        lr = LR0
        mm0 = eval_full(cur)
        best = obj_val(per_record_margin_from_state(cur), obj)
        stall = 0
        for it in range(1, iters + 1):
            model.load_state_dict(cur)
            model.eval()
            loss = obj_torch(model(xF), obj)
            model.zero_grad()
            loss.backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(12):
                cand = {k2: (v + lr * g[k2] if k2 in g else v)
                        for k2, v in cur.items()}
                mm = eval_full(cand)
                if score_of(mm) <= q:
                    mg = margins_of(cand)
                    v = obj_val(mg, obj)
                    if v > best + 1e-7:
                        cur, best, acc = cand, v, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return cur, best

    def per_record_margin_from_state(sd):
        return margins_of(sd)

    out = {'frozen': {'q20': float(q), 'fit': list(FIT_TAGS),
                      'cal': list(CAL_TAGS)},
           'theta0_objs': {}, 'pools': {}, 'runs': {}}
    for o in ('O1', 'O2', 'O3'):
        out['theta0_objs'][o] = obj_val(th_margins, o)
    for st in START_TAGS:
        out['pools'][st] = {o: {'min': float(min(v)), 'max':
                                float(max(v)),
                                'median': float(np.median(v))}
                            for o, v in pools[st]['objs'].items()}

    results = {}
    for st in START_TAGS[:1 if cli.smoke else 2]:
        for o in ('O1', 'O2', 'O3'):
            gains = []
            for j, p in enumerate(pools[st]['paths'][:n_starts]):
                fn = '{}_{}_s{}.json'.format(st, o, j)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    sd = torch.load(p, map_location=device)
                    fin, best = ascend(sd, o)
                    mm = eval_full(fin)
                    rec = {'best': best, 'final_s': score_of(mm),
                           'passes': score_of(mm) <= q,
                           'final_mm': mm}
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1, default=float)
                sel = max(pools[st]['objs'][o])
                med = float(np.median(pools[st]['objs'][o]))
                th0 = out['theta0_objs'][o]
                g = gain_of(rec['best'], sel, med, th0)
                gains.append(g)
                print(st, o, 'start', j, 'best %.3f sel %.3f gain %.3f '
                      'pass %s' % (rec['best'], sel, g, rec['passes']),
                      flush=True)
            results['{}_{}'.format(st, o)] = {
                'gains': gains, 'max_gain': max(gains)}
    # verdict on the primary pool (e3); replication = e4 same sign+gate
    prim = {o: results['e3_{}'.format(o)]['max_gain']
            for o in ('O1', 'O2', 'O3') if 'e3_{}'.format(o) in results}
    if not cli.smoke and 'e4_O1' in results:
        repl = all(results['e4_{}'.format(o)]['max_gain'] >= GATE
                   for o in prim if results['e3_{}'.format(o)]['max_gain']
                   >= GATE) if prim else False
        v, why = verdict_ey(prim, replicated=bool(prim) and repl)
    else:
        v, why = verdict_ey(prim, replicated=False)
    out.update({'results': results, 'verdict': v, 'verdict_why': why})
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('E-Y VERDICT:', v, '|', why, flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
