"""R6-verdict closure (2026-09-29, reviewer round 6): zero-training
supplements computed under the FROZEN fixed-split audit of m3b.

(a) Matched-total-reference baseline: max-deviation bands from
    fit+cal (18 models, the same total references the conformal
    branch uses) evaluated on the same 12 test models.
(b) Attacks under the frozen fixed-split audit: the 18 archived
    ex3 washes and 3 r3p1 adaptive ckpts (all on F1) scored with
    the m3b fixed-split centers/scales (fit: ew9f) and thresholds
    (cal: p1aud+e2); rejections at alpha=0.20/0.10.
(c) New-draw equal-weight (2/9) -- pure arithmetic from m3p2.
"""
import glob
import json
import os
import time

import numpy as np

METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
ALPHAS = (0.20, 0.10)
FIXED = {'fit': ['ew9f'], 'cal': ['p1aud', 'e2'], 'test': ['e3', 'e4']}


def conformal_q(scores, alpha):
    import math
    s = sorted(scores)
    idx = math.ceil((len(s) + 1) * (1.0 - alpha))
    return s[idx - 1] if idx <= len(s) else float('inf')


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED,
                            TIER_A, build_pool, f_deep,
                            metrics_from_logits)
    import torchvision.transforms as _tf

    outdir = os.path.join('research_outputs', 'uw',
                          'm3c-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:0')

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
    fset = f_deep(pool, 30)
    pos = np.searchsorted(pool, fset)
    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)
    model = get_model('ResNet18').to(device)

    def eval_f(sd):
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

    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    fit = [e for t in FIXED['fit'] for e in ens_mm[t]]
    c = {m: float(np.median([e[m] for e in fit])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit]) - c[m]))) for m in METRICS}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_s = [sc(e) for t in FIXED['cal'] for e in ens_mm[t]]
    q = {a: conformal_q(cal_s, a) for a in ALPHAS}

    # (a) matched-total-reference baseline: max-dev from fit+cal (18)
    fc = fit + [e for t in FIXED['cal'] for e in ens_mm[t]]
    c18 = {m: float(np.median([e[m] for e in fc])) for m in METRICS}
    b18 = {m: float(max(abs(e[m] - c18[m]) for e in fc)) for m in METRICS}
    test = [e for t in FIXED['test'] for e in ens_mm[t]]
    rej18 = [any(abs(e[m] - c18[m]) > b18[m] for m in METRICS)
             for e in test]
    print('matched-18 max-dev baseline: %d/12' % sum(rej18), flush=True)

    # (b) attacks under the frozen fixed-split audit
    atk_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/*_wash_s*.pt'))
    atk_paths += sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/adapt_s*.pt'))
    rows = []
    for pth in atk_paths:
        mm = eval_f(torch.load(pth, map_location=device))
        s = sc(mm)
        rows.append({'ckpt': os.path.basename(pth), 's': float(s),
                     'q02': float(q[0.20]), 'q01': float(q[0.10]),
                     'reject02': bool(s > q[0.20]),
                     'reject01': bool(s > q[0.10])})
        print(os.path.basename(pth), round(s, 3),
              'rej@.20', rows[-1]['reject02'], flush=True)

    # (c) new-draw equal weight
    p2 = json.load(open(sorted(glob.glob(
        'research_outputs/uw/m3p2-2*/verdict.json'))[-1]))
    per = p2['honest_loeo']
    new_draws = []
    for h in ('F2', 'F3', 'F4'):
        r = per['0.20|{}'.format(h)]
        new_draws.append(sum(1 for x in r if x['reject']) / len(r))
    eq_new = float(np.mean(new_draws))

    out = {'matched18_baseline_k': int(sum(rej18)),
           'matched18_baseline_n': 12,
           'attacks_fixed_audit': rows,
           'attack_reject02': sum(1 for r in rows if r['reject02']),
           'attack_reject01': sum(1 for r in rows if r['reject01']),
           'n_attacks': len(rows),
           'new_draw_equal_weight_020': eq_new,
           'frozen_audit': {'q02': float(q[0.20]),
                            'q01': float(q[0.10]),
                            'fit_tags': FIXED['fit'],
                            'cal_tags': FIXED['cal'],
                            'test_tags': FIXED['test']}}
    json.dump(out, open(os.path.join(outdir, 'm3c.json'), 'w'),
              indent=1, default=float)
    print('new-draw equal weight @0.20:', round(eq_new, 3), flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
