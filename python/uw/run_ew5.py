"""UW E-W5: worst-case F families + multi-theta0 replication + paper
skeleton inputs (frozen prereg docs/plans/2026-09-23-uw-ew5-preregistration.md).

Arm A: two NEW theta_0 (seeds 44600/44700, same recipe), k in {30,300},
3 theta_R each, S0-only optimized wash under self-inclusive ensemble
bands; replication judged vs the 44500 baseline (E-W4: 0.248/0.214).
Arm B: theta_0 = 44500 (E-W3 ckpt), k = 30, three constructed F families
(topF/botF/lossF by theta_0 margin/loss on the pool), 3 theta_R each;
family sensitivity vs the uniform baseline 0.248.
Optimizer = E-W4-final semantics (adaptive backtracking, +margin ascent,
60 iters, stall 15).  Resume via per-unit jsons (--outdir).
"""
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
NEW_THETA0 = (44600, 44700)
BASE_THETA0 = 44500
K_A = (30, 300)
KB = 30
FAMS = ('topF', 'botF', 'lossF')
RT_A = lambda tidx, ki, s: 44900 + 100 * tidx + 10 * ki + s
RT_B = lambda fidx, s: 44800 + 10 * fidx + s
MAX_ITERS = 60
STALL_LIMIT = 15
CONTROL_STEPS = 5
CONTROL_GAIN = 0.5
REPL_TOL = 0.10
REPL_FAIL = 0.10
FAM_2X = 0.50
FAM_DEAD = 0.10


def fam_select(order_key, fam, k):
    """Frozen F-family construction from a per-pool-record score."""
    if fam == 'topF':
        idx = np.argsort(-order_key)
    elif fam == 'botF':
        idx = np.argsort(order_key)
    elif fam == 'lossF':
        idx = np.argsort(-order_key)      # caller passes losses
    else:
        raise ValueError(fam)
    return np.sort(idx[:k])


def armA_verdict(i1_base, i1_new, tol=REPL_TOL, fail=REPL_FAIL):
    """replicated / failed / partial per frozen §1."""
    if i1_new is None:
        return 'failed', 'no valid runs'
    if abs(i1_new - i1_base) <= tol:
        return 'replicated', ('|{:.3f}-({:.3f})| <= 0.10'.format(
            i1_new, i1_base))
    if i1_new < fail:
        return 'failed', ('I1*={:.3f} < 0.10'.format(i1_new))
    return 'partial', ('I1*={:.3f} deviates >0.10 from {:.3f} but '
                       '>= 0.10'.format(i1_new, i1_base))


def armB_verdict(i1s, uniform, two_x=FAM_2X, dead=FAM_DEAD):
    mx = max(i1s.values()) if i1s else 0.0
    if not i1s:
        return 'undecided', 'no valid runs'
    if all(v < dead for v in i1s.values()):
        return 'uniform_only', 'all families < 0.10'
    if mx >= two_x:
        fam = max(i1s, key=lambda f: i1s[f])
        return 'family_material', ('{} I1*={:.3f} >= 2x uniform '
                                   '({:.3f})'.format(fam, mx, uniform))
    return 'bounded', ('max family I1*={:.3f} < 2x uniform'.format(mx))


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
    epochs = 2 if cli.smoke else 40
    iters = 8 if cli.smoke else MAX_ITERS
    new_theta0 = NEW_THETA0[:1] if cli.smoke else NEW_THETA0
    k_a = (30,) if cli.smoke else K_A
    fams = FAMS[:1] if cli.smoke else FAMS
    n_rt = 1 if cli.smoke else 3

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ew5-smoke-' if cli.smoke else 'ew5-'
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
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

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def fit(keep, seed, epochs_):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        for _ep in range(epochs_):
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

    def get_theta0(seed):
        if seed == BASE_THETA0:
            return torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                              map_location=device)
        p = os.path.join(ck, 'theta0_s{}.pt'.format(seed))
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = fit(np.arange(n), seed, epochs)
        torch.save(sd, p)
        print('theta0', seed, 'trained', flush=True)
        return sd

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)

    def eval_model(sd, fset):
        pos = np.searchsorted(pool, fset)
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
        z_pool = z_ps[:len(pool)]
        z_sub = z_ps[len(pool):]
        return metrics_from_logits(z_pool[pos], yfull[fset], z_sub,
                                   y_sub, z_te, y_te)

    def f_batch(fset):
        xf, yfl = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(trainset_eval, fset),
                batch_size=500, shuffle=False):
            xf.append(x)
            yfl.append(y)
        return torch.cat(xf).to(device), torch.cat(yfl).to(device).long()

    def fm_only(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    def grad_delta(sd, xF, yF):
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

    def unit(sd0, fset, rt_seeds, tagname):
        """One audit unit: retrains, ensemble bands, S0 wash per seed."""
        keep = np.setdiff1d(np.arange(n), fset)
        rt_sd, rt_mm = {}, {}
        for s in rt_seeds:
            fn = '{}_s{}.json'.format(tagname, s)
            pth = os.path.join(ck, fn.replace('.json', '.pt'))
            if os.path.exists(os.path.join(outdir, fn)):
                rt_mm[s] = json.load(open(os.path.join(outdir, fn)))['mm']
                continue
            if os.path.exists(pth):
                sd = torch.load(pth, map_location=device)
            else:
                sd = fit(keep, s, epochs)
                torch.save(sd, pth)
            mm = eval_model(sd, fset)
            rt_mm[s] = mm
            with open(os.path.join(outdir, fn), 'w') as fh:
                json.dump({'mm': mm}, fh, indent=1)
        centers = {m: float(np.median([rt_mm[s][m] for s in rt_seeds]))
                   for m in TIER_A}
        bands = loo_bands(rt_mm, centers)
        pt_fn = '{}_pt.json'.format(tagname)
        if os.path.exists(os.path.join(outdir, pt_fn)):
            pt = json.load(open(os.path.join(outdir, pt_fn)))
        else:
            pt = eval_model(sd0, fset)
            with open(os.path.join(outdir, pt_fn), 'w') as fh:
                json.dump(pt, fh, indent=1)
        xF, yF = f_batch(fset)
        out = []
        for s in rt_seeds:
            fn = '{}_opt_s{}.json'.format(tagname, s)
            if os.path.exists(os.path.join(outdir, fn)):
                out.append(json.load(open(os.path.join(outdir, fn))))
                continue
            sd = torch.load(os.path.join(
                ck, '{}_s{}.pt'.format(tagname, s)), map_location=device)
            # control
            cur, lr, best, f0 = sd, LR0, None, None
            best = fm_only(sd, xF, yF)
            f0 = best
            for _ in range(3 if cli.smoke else CONTROL_STEPS):
                g = grad_delta(cur, xF, yF)
                acc = False
                for _bt in range(12):
                    cand = apply_delta(cur, g, lr)
                    fmc = fm_only(cand, xF, yF)
                    if fmc > best + 1e-7:
                        cur, best, acc = cand, fmc, True
                        break
                    lr *= 0.5
                    if lr < 1e-9:
                        break
                lr = lr_update(lr, acc)
            ctrl_gain = best - f0
            # constrained S0 wash
            cur = sd
            lr = LR0
            best_fm = eval_model(cur, fset)['m6_fm']
            log = [{'iter': 0, 'fm': best_fm}]
            stall = 0
            for it in range(1, iters + 1):
                g = grad_delta(cur, xF, yF)
                acc = False
                for _bt in range(12):
                    cand = apply_delta(cur, g, lr)
                    mm = eval_model(cand, fset)
                    ok, _v = accept(mm, centers, bands[s], TIER_A)
                    if ok and mm['m6_fm'] > best_fm + 1e-7:
                        cur, best_fm, acc = cand, mm['m6_fm'], True
                        log.append({'iter': it, 'fm': best_fm})
                        break
                    lr *= 0.5
                    if lr < 1e-9:
                        break
                lr = lr_update(lr, acc)
                stall = 0 if acc else stall + 1
                if stall > STALL_LIMIT:
                    break
            mm_final = eval_model(cur, fset)
            rec = {'seed': s, 'ctrl_gain': ctrl_gain,
                   'fm_start': rt_mm[s]['m6_fm'], 'fm_best': best_fm,
                   'I1': i1(best_fm, rt_mm[s]['m6_fm'], pt['m6_fm']),
                   'n_accepted': len(log) - 1, 'fm_0': pt['m6_fm'],
                   'util_margin_0': pt['util_margin'],
                   'util_margin_R': rt_mm[s]['util_margin'],
                   'util_margin_best': mm_final['util_margin'],
                   'cat_margin_best': mm_final['cat_margin'],
                   'cat_margin_0': pt['cat_margin'],
                   'cat_margin_R': rt_mm[s]['cat_margin'],
                   'log': log}
            with open(os.path.join(outdir, fn), 'w') as fh:
                json.dump(rec, fh, indent=1)
            out.append(rec)
            print(tagname, s, 'I1', round(rec['I1'], 4), 'ctrl',
                  round(ctrl_gain, 1), flush=True)
        return out

    results = {'armA': {}, 'armB': {}}
    # ---- Arm A
    for tidx, ts in enumerate(new_theta0, start=1):
        sd0 = get_theta0(ts)
        for ki, k in enumerate(k_a):
            fset = f_deep(pool, k)
            seeds = [RT_A(tidx, ki, s) for s in range(1, n_rt + 1)]
            runs = unit(sd0, fset, seeds,
                        'armA_t{}_k{}'.format(ts, k))
            valid = [r['I1'] for r in runs if r['ctrl_gain']
                     >= CONTROL_GAIN]
            results['armA']['{}_k{}'.format(ts, k)] = {
                'i1_star': max(valid) if valid else None,
                'per_seed': [{'seed': r['seed'], 'I1': r['I1'],
                              'ctrl_gain': r['ctrl_gain']}
                             for r in runs]}
    # ---- Arm B
    sd0b = get_theta0(BASE_THETA0)
    # per-record margins/losses on the pool under theta0 (canonical form)
    model.load_state_dict(sd0b)
    model.eval()
    zs = []
    with torch.no_grad():
        for _g, x, _y in loader_ps:
            zs.append(model(x.to(device)).cpu())
    z_pool = torch.cat(zs).numpy()[:len(pool)]
    y_pool_t = yfull[pool]
    mg = _margin_rows_np(z_pool, y_pool_t)
    ls = _ce_rows_np(z_pool, y_pool_t)
    for fidx, fam in enumerate(fams):
        key = ls if fam == 'lossF' else mg
        fset = pool[fam_select(key, fam, KB)]
        seeds = [RT_B(fidx, s) for s in range(1, n_rt + 1)]
        runs = unit(sd0b, fset, seeds, 'armB_{}'.format(fam))
        valid = [r['I1'] for r in runs if r['ctrl_gain'] >= CONTROL_GAIN]
        results['armB'][fam] = {
            'i1_star': max(valid) if valid else None,
            'per_seed': [{'seed': r['seed'], 'I1': r['I1'],
                          'ctrl_gain': r['ctrl_gain']} for r in runs]}

    # ---- verdict (frozen §1/§2)
    base = {30: 0.2477, 300: 0.2139}     # E-W4 baseline (44500)
    verdictA = {}
    for key, r in results['armA'].items():
        k = int(key.split('_k')[1])
        st, why = armA_verdict(base.get(k, 0.0), r['i1_star'])
        verdictA[key] = {'status': st, 'reason': why}
    i1s = {f: r['i1_star'] for f, r in results['armB'].items()
           if r['i1_star'] is not None}
    stB, whyB = armB_verdict(i1s, base[30])
    verdict = {'armA': verdictA,
               'armB': {'status': stB, 'reason': whyB,
                        'i1_by_family': results['armB']},
               'baseline_44500': base}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(verdict, fh, indent=1)
    with open(os.path.join(outdir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    print('VERDICT A:', json.dumps(verdictA), flush=True)
    print('VERDICT B:', stB, '--', whyB, flush=True)
    print('saved', outdir, flush=True)


def _margin_rows_np(z, y):
    zy = z[np.arange(len(y)), y]
    zc = z.copy()
    zc[np.arange(len(y)), y] = -1e18
    return zy - zc.max(axis=1)


def _ce_rows_np(z, y):
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return lse - z[np.arange(len(y)), y]


if __name__ == '__main__':
    main()
