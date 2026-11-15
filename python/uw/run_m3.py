"""UW M3: coverage-targeted calibration of the banded audit (frozen
prereg docs/plans/2026-09-29-uw-m3-coverage-preregistration.md,
reviewer-M3 spec). Standard split conformal on the max normalized
deviation score -- no new algorithm claimed.

Phase 1 (no new training): ft-family cross-ensemble coverage.
30 honest ft models across 5 independent ensembles (ckpts archived);
leave-one-ensemble-out conformal at alpha=0.20 (primary) / 0.10;
baseline = current single-ensemble n=6 max-deviation audit evaluated
on all 20 ordered ensemble pairs; attack side = ex3 wash checkpoints
(18) scored under the same LOEO rule plus analytic logit-scaled
points. All decisions reported.

Phase 2 (approval-gated, --phase2): retrain-family cross-request
coverage on 3 fresh F-draws x 3 retrains.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        build_pool, f_deep,
                        metrics_from_logits)

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
K = 30
METRICS = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
ALPHAS = (0.20, 0.10)
FT_ENS = {
    'ew9f': ('research_outputs/uw/ew9F-20260927T001555/ckpt',
             'ft_k30_s4590{}.pt', range(1, 7)),
    'p1aud': (None, None, None),      # resolved at runtime via glob
    'e2': (None, None, None),
    'e3': (None, None, None),
    'e4': (None, None, None),
}
# Phase 2 frozen seeds
F_SEEDS = (47101, 47201, 47301)
RT_SEEDS = ((48101, 48102, 48103), (48201, 48202, 48203),
            (48301, 48302, 48303))


def conformal_q(scores, alpha):
    """Split-conformal threshold: ceil((n+1)(1-alpha))-th smallest;
    infinite when the index exceeds n (no finite-sample guarantee)."""
    s = sorted(scores)
    idx = math.ceil((len(s) + 1) * (1.0 - alpha))
    return s[idx - 1] if idx <= len(s) else float('inf')


def loo_median(v):
    v = np.asarray(v, float)
    return np.array([np.median(np.delete(v, i)) for i in range(len(v))])


def pooled_sigma(ens_metrics, metric):
    """Per-ensemble MAD of |x - ensemble median|, pooled as the median
    across ensembles (frozen in the card)."""
    mads = []
    for e in ens_metrics.values():
        x = np.asarray([m[metric] for m in e], float)
        mads.append(np.median(np.abs(x - np.median(x))))
    return float(np.median(mads))


def maxdev_reject(mm, med, bands, metrics=TIER_A):
    return any(abs(mm[m] - med[m]) > bands[m] for m in metrics)


def verdict_m3(fpr, baseline, alpha):
    if fpr <= alpha and baseline - fpr >= 0.30:
        return 'COVERAGE_DELIVERED', (
            'honest FPR {:.2f} <= alpha {:.2f}; baseline {:.2f} -> '
            'improvement {:.2f}'.format(fpr, alpha, baseline,
                                        baseline - fpr))
    if baseline - fpr >= 0.30:
        return 'IMPROVED_BUT_IMPRECISE', (
            'improvement {:.2f} but FPR {:.2f} above alpha {}'
            .format(baseline - fpr, fpr, alpha))
    return 'NOT_DELIVERED', ('baseline {:.2f} vs FPR {:.2f}'
                             .format(baseline, fpr))




def _jdefault(o):
    import numpy as _np
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    raise TypeError(repr(o))

def binom_ci(k, n, z=1.96):
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def f_draw(pool, seed, k=K):
    """Card-frozen fresh deletion draw: first k of RandomState(seed)
    permutation of the pool (f_deep convention)."""
    return np.sort(np.random.RandomState(seed).permutation(pool)[:k])


def resolve_ft_ensembles():
    out = {}
    for p in sorted(glob.glob(
            'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt')):
        out.setdefault('ew9f', []).append(p)
    for p in sorted(glob.glob(
            'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt')):
        out.setdefault('p1aud', []).append(p)
    for tag in ('e2', 'e3', 'e4'):
        for p in sorted(glob.glob(
                'research_outputs/uw/ex3-2*/ckpt/{}_ft_s*.pt'.format(
                    tag))):
            out.setdefault(tag, []).append(p)
    return out


def run_phase2(cli):
    """Cross-deletion-draw coverage (card section 3, author-approved).

    Draws: F1 = f_deep(pool,30) with 12 archived honest retrains;
    F2/F3/F4 = f_draw(pool, 47101/47201/47301) with 3 fresh 40-epoch
    retrains each (seeds 481xx/482xx/483xx). LOEO across draws with
    the same frozen score/alphas/verdict as Phase 1; baseline =
    single-draw max-deviation cross-draw rejection; attacks = the 15
    archived wash/adaptive checkpoints built on F1, scored against F1
    centers with sigma/threshold from the other draws.
    """
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw', 'm3p2-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
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
    y_t = torch.as_tensor(yfull, dtype=torch.long)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    epochs = 2 if cli.smoke else 40

    def fit(keep, seed):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        for _ep in range(epochs):
            m.train()
            for _ids, x, _yy in loader:
                opt.zero_grad()
                lf(m(x.to(device)), y_t[_ids.tolist()].to(
                    device)).backward()
                opt.step()
            sched.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)
    zs0, ys0 = [], []
    for _id, x, y in testloader:
        zs0.append(x)
        ys0.append(y)
    x_te = torch.cat(zs0)
    y_te = torch.cat(ys0).numpy()

    def eval_f(sd, fset):
        pos = np.searchsorted(pool, fset)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z_te = model(x_te.to(device)).cpu().numpy()
        zs = []
        with torch.no_grad():
            for _g, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    F1 = f_deep(pool, K)
    draws = {
        'F1': dict(fset=F1, ckpts=(
            [os.path.join(RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s))
             for s in (44101, 44102, 44103)]
            + ['research_outputs/uw/ew6-20260925T213308/ckpt/'
               'u0_t44500_k30_s{}.pt'.format(s)
               for s in (45201, 45202, 45203)]
            + sorted(glob.glob(
                'research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt')))),
    }
    for name, fseed, rseeds in (('F2', 47101, (48101, 48102, 48103)),
                                ('F3', 47201, (48201, 48202, 48203)),
                                ('F4', 47301, (48301, 48302, 48303))):
        fs = f_draw(pool, fseed)
        keep = np.setdiff1d(np.arange(n), fs)
        cks = []
        for s in (rseeds[:1] if cli.smoke else rseeds):
            pth = os.path.join(ck, '{}_s{}.pt'.format(name, s))
            if not os.path.exists(pth):
                torch.save(fit(keep, s), pth)
                print('fit', name, s, flush=True)
            cks.append(pth)
        draws[name] = dict(fset=fs, ckpts=cks)

    # metrics per draw (cached)
    cache = os.path.join(outdir, 'metrics_cache.json')
    if os.path.exists(cache):
        dm = json.load(open(cache))['draws']
    else:
        dm = {}
        for name, d in draws.items():
            dm[name] = [eval_f(torch.load(p, map_location=device),
                               d['fset']) for p in d['ckpts']]
            print('evaled', name, len(dm[name]), flush=True)
        json.dump({'draws': dm}, open(cache, 'w'), indent=1,
                  default=_jdefault)

    tags = sorted(dm)

    def score(mm, med, sig):
        return max(abs(mm[m] - med[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    results = {'honest_loeo': {}, 'baseline_maxdev': {}, 'attacks': {}}
    for alpha in ALPHAS:
        rejects = []
        for hold in tags:
            cal = [t for t in tags if t != hold]
            sig = {m: pooled_sigma({t: dm[t] for t in cal}, m)
                   for m in METRICS}
            cs = []
            for t in cal:
                loo = {m: loo_median([e[m] for e in dm[t]])
                       for m in METRICS}
                for j, e in enumerate(dm[t]):
                    cs.append(score(e, {m: loo[m][j] for m in METRICS},
                                    sig))
            q = conformal_q(cs, alpha)
            hloo = {m: loo_median([e[m] for e in dm[hold]])
                    for m in METRICS}
            for j, e in enumerate(dm[hold]):
                s = score(e, {m: hloo[m][j] for m in METRICS}, sig)
                results['honest_loeo'].setdefault(
                    '{:.2f}|{}'.format(alpha, hold), []).append(
                        {'s': float(s), 'q': float(q),
                         'reject': bool(s > q)})
                rejects.append(s > q)
        k, n_ = int(sum(rejects)), len(rejects)
        lo, hi = binom_ci(k, n_)
        results['honest_loeo']['summary_{:.2f}'.format(alpha)] = {
            'fpr': k / n_, 'ci': [lo, hi], 'k': k, 'n': n_}
        print('alpha', alpha, 'cross-draw FPR', round(k / n_, 3),
              'CI', (round(lo, 3), round(hi, 3)), flush=True)

    rejs = []
    for cald in tags:
        med = {m: float(np.median([e[m] for e in dm[cald]]))
               for m in METRICS}
        bands = {m: float(max(abs(e[m] - med[m]) for e in dm[cald]))
                 for m in METRICS}
        for testd in tags:
            if testd == cald:
                continue
            rs = [maxdev_reject(e, med, bands, METRICS)
                  for e in dm[testd]]
            results['baseline_maxdev']['{}->{}'.format(
                cald, testd)] = [bool(r) for r in rs]
            rejs.extend(rs)
    results['baseline_maxdev']['summary'] = {
        'fpr': float(np.mean(rejs)), 'n': len(rejs)}
    print('baseline cross-draw max-dev FPR',
          round(float(np.mean(rejs)), 3), flush=True)

    # attacks on F1: r3p3 defense-aware washes + r3p1 adaptive ckpts
    atk_paths = sorted(glob.glob(
        'research_outputs/uw/r3p3-2*/ckpt_da_s*.pt'))
    atk_paths += sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/adapt_s*.pt'))
    atk = []
    for p in atk_paths:
        mm = eval_f(torch.load(p, map_location=device), F1)
        cal = [t for t in tags if t != 'F1']
        sig = {m: pooled_sigma({t: dm[t] for t in cal}, m)
               for m in METRICS}
        med = {m: float(np.median([e[m] for e in dm['F1']]))
               for m in METRICS}
        row = {'ckpt': os.path.basename(p)}
        for alpha in ALPHAS:
            cs = []
            for t in cal:
                loo = {m: loo_median([e[m] for e in dm[t]])
                       for m in METRICS}
                for j, e in enumerate(dm[t]):
                    cs.append(score(e, {m: loo[m][j] for m in METRICS},
                                    sig))
            q = conformal_q(cs, alpha)
            s = score(mm, med, sig)
            row['alpha_{:.2f}'.format(alpha)] = {
                's': float(s), 'q': float(q), 'reject': bool(s > q)}
        atk.append(row)
    results['attacks'] = atk
    n_rej = sum(1 for r in atk if r['alpha_0.20']['reject'])
    results['attack_reject_rate_0.20'] = n_rej / max(len(atk), 1)

    s_main = results['honest_loeo']['summary_0.20']
    v, why = verdict_m3(s_main['fpr'],
                        results['baseline_maxdev']['summary']['fpr'],
                        0.20)
    results['verdict'] = v
    results['verdict_why'] = why
    print('M3 PHASE2 VERDICT:', v, '|', why,
          '| attack reject', results['attack_reject_rate_0.20'],
          flush=True)
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(results, fh, indent=1, default=_jdefault)
    print('OUT', outdir, flush=True)


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--phase2', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()

    if cli.phase2:
        run_phase2(cli)
        return

    tag = 'm3-smoke-' if cli.smoke else 'm3-'
    outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw',
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

    model = get_model('ResNet18').to(device)

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

    def eval_ckpt(path):
        return eval_model(torch.load(path, map_location=device))

    # ---------------- phase 1: ft family ---------------------------------
    ens_paths = resolve_ft_ensembles()
    ens_mm = {}
    cache = os.path.join(outdir, 'metrics_cache.json')
    if os.path.exists(cache):
        ens_mm = json.load(open(cache))['ens_mm']
    else:
        for tag_, paths in ens_paths.items():
            ens_mm[tag_] = [eval_ckpt(p) for p in paths]
            print('evaled', tag_, len(ens_mm[tag_]), flush=True)
        json.dump({'ens_mm': ens_mm}, open(cache, 'w'), indent=1)

    tags = sorted(ens_mm)
    if cli.smoke:
        tags = tags[:2]

    def conformal_score(mm, med, sig):
        return max(abs(mm[m] - med[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    results = {'honest_loeo': {}, 'baseline_maxdev': {},
               'attacks': {}, 'alphas': ALPHAS}
    for alpha in ALPHAS:
        rejects = []
        for hold in tags:
            cal_tags = [t for t in tags if t != hold]
            cal_ens = {t: ens_mm[t] for t in cal_tags}
            sig = {m: pooled_sigma(cal_ens, m) for m in METRICS}
            cal_scores = []
            for t in cal_tags:
                loo = {m: loo_median([e[m] for e in ens_mm[t]])
                       for m in METRICS}
                for j, e in enumerate(ens_mm[t]):
                    med = {m: loo[m][j] for m in METRICS}
                    cal_scores.append(conformal_score(e, med, sig))
            q = conformal_q(cal_scores, alpha)
            hold_loo = {m: loo_median([e[m] for e in ens_mm[hold]])
                        for m in METRICS}
            for j, e in enumerate(ens_mm[hold]):
                med = {m: hold_loo[m][j] for m in METRICS}
                s = conformal_score(e, med, sig)
                results['honest_loeo'].setdefault(
                    '{:.2f}|{}'.format(alpha, hold), []).append(
                        {'s': float(s), 'q': float(q),
                         'reject': bool(s > q)})
                rejects.append(s > q)
        k, n_ = sum(rejects), len(rejects)
        lo, hi = binom_ci(k, n_)
        results['honest_loeo']['summary_{:.2f}'.format(alpha)] = {
            'fpr': k / n_, 'ci': [lo, hi], 'k': k, 'n': n_}
        print('alpha', alpha, 'honest FPR', round(k / n_, 3),
              'CI', (round(lo, 3), round(hi, 3)), flush=True)

    # baseline: single-ensemble max-dev bands tested on other ensembles
    rejs = []
    for cal in tags:
        med = {m: float(np.median([mm[m] for mm in ens_mm[cal]]))
               for m in METRICS}
        bands = {m: float(max(abs(mm[m] - med[m])
                              for mm in ens_mm[cal])) for m in METRICS}
        for test in tags:
            if test == cal:
                continue
            for mm in ens_mm[test]:
                r = maxdev_reject(mm, med, bands, METRICS)
                results['baseline_maxdev'].setdefault(
                    '{}->{}'.format(cal, test), []).append(r)
                rejs.append(r)
    results['baseline_maxdev']['summary'] = {
        'fpr': float(np.mean(rejs)), 'n': len(rejs)}
    print('baseline max-dev FPR', round(float(np.mean(rejs)), 3),
          flush=True)

    # attack side: ex3 wash ckpts under the same LOEO rule
    atk = []
    for p in sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/*_wash_s*.pt')):
        tag_ = os.path.basename(p).split('_')[0]     # e2/e3/e4
        if tag_ not in tags:
            continue
        mm = eval_ckpt(p)
        cal_tags = [t for t in tags if t != tag_]
        cal_ens = {t: ens_mm[t] for t in cal_tags}
        sig = {m: pooled_sigma(cal_ens, m) for m in METRICS}
        med = {m: float(np.median([q[m] for q in ens_mm[tag_]]))
               for m in METRICS}
        row = {'ckpt': os.path.basename(p)}
        for alpha in ALPHAS:
            cal_scores = []
            for t in cal_tags:
                loo = {m: loo_median([e[m] for e in ens_mm[t]])
                       for m in METRICS}
                for j, e in enumerate(ens_mm[t]):
                    cal_scores.append(conformal_score(
                        e, {m: loo[m][j] for m in METRICS}, sig))
            q = conformal_q(cal_scores, alpha)
            s = conformal_score(mm, med, sig)
            row['alpha_{:.2f}'.format(alpha)] = {'s': s, 'q': q,
                                                 'reject': s > q}
        atk.append(row)
        print('attack', os.path.basename(p),
              {k: v['reject'] for k, v in row.items()
               if k.startswith('alpha')}, flush=True)
    results['attacks'] = atk

    s_main = results['honest_loeo']['summary_0.20']
    v, why = verdict_m3(s_main['fpr'],
                        results['baseline_maxdev']['summary']['fpr'],
                        0.20)
    results['verdict'] = v
    results['verdict_why'] = why
    n_rej_atk = sum(1 for r in atk if r['alpha_0.20']['reject'])
    results['attack_reject_rate_0.20'] = n_rej_atk / max(len(atk), 1)
    print('M3 PHASE1 VERDICT:', v, '|', why,
          '| attack reject rate', results['attack_reject_rate_0.20'],
          flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(results, fh, indent=1, default=_jdefault)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
