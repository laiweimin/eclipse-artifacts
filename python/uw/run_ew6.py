"""UW E-W6: retrial-and-select attack + dispersion-gating law (frozen
prereg docs/plans/2026-09-23-uw-ew6-preregistration.md).

Four units (theta_0, k): (44500,30), (44600,30), (44700,30), (44700,300).
Each: 3 archived theta_R seeds + 3 NEW (seed_all(45200+ui*10+s)); ensemble
bands/centers recomputed over all 6; S0 optimized wash per seed under the
6-seed bands; per-unit success(M) curves (exact over random M-subsets);
dispersion statistic b_fm = max|fm - median| / E_fm = |fm_0 - median fm|;
law test = Spearman rank corr of I1* vs b/E over all measured cells
(E-W4/E-W5 archived + this round).  All accepted steps log full metrics.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, loo_bands, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN5 = 'research_outputs/uw/ew5-20260925T160840'
UNITS = (('44500', 30), ('44600', 30), ('44700', 30), ('44700', 300))
MAX_ITERS = 60
STALL_LIMIT = 15
CONTROL_STEPS = 5
CONTROL_GAIN = 0.5
LAW_RHO = 0.7
TAUS = (0.10, 0.20)


def spearman(x, y):
    """Rank correlation without scipy (average ranks for ties)."""
    def ranks(v):
        order = np.argsort(v, kind='mergesort')
        sv = np.asarray(v)[order]
        r = np.empty(len(v))
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def success_curve(i1s, tau, ms=(1, 2, 3, 4, 5, 6)):
    """P(at least one of a uniformly random M-subset has I1 >= tau),
    exact enumeration over C(6, M)."""
    n = len(i1s)
    out = {}
    for m in ms:
        if m > n:
            continue
        combos = list(itertools.combinations(range(n), m))
        hits = sum(1 for c in combos if any(i1s[i] >= tau for i in c))
        out[m] = hits / len(combos)
    return out


def verdict_from(units_out, rho, n_units=4):
    """Frozen §2: attack claim established iff >=3/4 units have a seed
    with I1 >= 0.20 at M=6 AND (law holds OR success >= 5/6 cells)."""
    n_strong = sum(1 for u in units_out.values()
                   if max(u['i1_per_seed']) >= 0.20)
    law_ok = rho >= LAW_RHO
    if n_strong >= 3 and law_ok:
        return 'ATTACK_ESTABLISHED', ('{}/4 units leak at 0.20; law '
                                      'rho={:.2f}'.format(n_strong, rho))
    if n_strong >= 3:
        return 'ATTACK_ESTABLISHED', ('{}/4 units leak at 0.20; law '
                                      'rho={:.2f} < 0.7 (carried by unit '
                                      'success alone)'.format(n_strong,
                                                              rho))
    dry = [k for k, u in units_out.items()
           if max(u['i1_per_seed']) < 0.10]
    return 'BACK_TO_AUTHOR', ('only {}/4 units leak at 0.20; dry cells '
                              '{}; law rho={:.2f}'.format(n_strong, dry,
                                                          rho))


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
    n_new = 1 if cli.smoke else 3
    units = UNITS[-1:] if cli.smoke else UNITS

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ew6-smoke-' if cli.smoke else 'ew6-'
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
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
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

    def theta0_of(ts):
        if ts == '44500':
            return torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                              map_location=device)
        return torch.load(os.path.join(RUN5, 'ckpt',
                                       'theta0_s{}.pt'.format(ts)),
                          map_location=device)

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
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    def f_batch(fset):
        xf, yfl = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(trainset_eval, fset),
                batch_size=500, shuffle=False):
            xf.append(x)
            yfl.append(y)
        return torch.cat(xf).to(device), torch.cat(yfl).to(device).long()

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

    def fm_only(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    units_out = {}
    for ui, (ts, k) in enumerate(units):
        fset = f_deep(pool, k)
        keep = np.setdiff1d(np.arange(n), fset)
        tag = 'u{}_t{}_k{}'.format(ui, ts, k)
        # archived seeds
        if ts == '44500':
            old = [44101, 44102, 44103][:1 if cli.smoke else 3]
            old_mm = {s: json.load(open(os.path.join(
                RUN3, 'rt_k{}_s{}.json'.format(k, s))))['per_k'][str(k)]
                for s in old}
            old_ck = {s: os.path.join(
                RUN3, 'ckpt', 'rt_k{}_s{}.pt'.format(k, s)) for s in old}
            pt = json.load(open(os.path.join(
                RUN3, 'pt.json')))['per_k'][str(k)]
        else:
            t5 = 'armA_t{}_k{}'.format(ts, k)
            import glob as _g
            fns = sorted(_g.glob(os.path.join(
                RUN5, '{}_s*.json'.format(t5))))
            fns = [f for f in fns if 'opt' not in os.path.basename(f)]
            old = [int(os.path.basename(f).rsplit('_s', 1)[1][:-5])
                   for f in fns][:1 if cli.smoke else 3]
            old_mm = {s: json.load(open(os.path.join(
                RUN5, '{}_s{}.json'.format(t5, s))))['mm'] for s in old}
            old_ck = {s: os.path.join(
                RUN5, 'ckpt', '{}_s{}.pt'.format(t5, s)) for s in old}
            pt = json.load(open(os.path.join(RUN5, t5 + '_pt.json')))
        seeds = list(old) + [45200 + ui * 10 + s
                             for s in range(1, n_new + 1)]
        mm_all = dict(old_mm)
        for s in seeds[len(old):]:
            fn = '{}_s{}.json'.format(tag, s)
            pth = os.path.join(ck, fn.replace('.json', '.pt'))
            if not os.path.exists(os.path.join(outdir, fn)):
                if os.path.exists(pth):
                    sd = torch.load(pth, map_location=device)
                else:
                    sd = fit(keep, s, epochs)
                    torch.save(sd, pth)
                mm_all[s] = eval_model(sd, fset)
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(mm_all[s], fh, indent=1)
            else:
                mm_all[s] = json.load(open(os.path.join(outdir, fn)))
        centers = {m: float(np.median([mm_all[s][m] for s in seeds]))
                   for m in TIER_A}
        bands = loo_bands(mm_all, centers)
        fm_med = float(np.median([mm_all[s]['m6_fm'] for s in seeds]))
        b_fm = float(max(abs(mm_all[s]['m6_fm'] - fm_med)
                         for s in seeds))
        xF, yF = f_batch(fset)
        i1s, ctrl_ok = [], True
        for s in seeds:
            fn = '{}_opt_s{}.json'.format(tag, s)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                sd = (torch.load(old_ck[s], map_location=device)
                      if s in old else torch.load(
                          os.path.join(ck, '{}_s{}.pt'.format(tag, s)),
                          map_location=device))
                ctrl_best = fm_only(sd, xF, yF)
                cur, lr, best = sd, LR0, ctrl_best
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
                if best - ctrl_best < CONTROL_GAIN:
                    ctrl_ok = False
                cur = sd
                lr = LR0
                best_fm = eval_model(cur, fset)['m6_fm']
                log = [{'iter': 0, 'fm': best_fm,
                        'metrics': eval_model(cur, fset)}]
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
                            log.append({'iter': it, 'fm': best_fm,
                                        'metrics': mm})
                            break
                        lr *= 0.5
                        if lr < 1e-9:
                            break
                    lr = lr_update(lr, acc)
                    stall = 0 if acc else stall + 1
                    if stall > STALL_LIMIT:
                        break
                mm_final = log[-1]['metrics']
                rec = {'seed': s, 'I1': i1(best_fm, mm_all[s]['m6_fm'],
                                           pt['m6_fm']),
                       'n_accepted': len(log) - 1,
                       'm1_best': mm_final['m1_acc'],
                       'util_margin_best': mm_final['util_margin'],
                       'cat_margin_best': mm_final['cat_margin'],
                       'ctrl_ok': best - ctrl_best >= CONTROL_GAIN,
                       'log': log}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1)
            i1s.append(rec['I1'] if rec.get('ctrl_ok', True) else 0.0)
            print(tag, s, 'I1', round(rec['I1'], 4), flush=True)
        units_out['t{}_k{}'.format(ts, k)] = {
            'i1_per_seed': i1s, 'seeds': seeds, 'ctrl_ok': ctrl_ok,
            'b_fm': b_fm, 'E_fm': abs(pt['m6_fm'] - fm_med),
            'b_over_E': b_fm / max(abs(pt['m6_fm'] - fm_med), 1e-12),
            'success': {str(tau): {str(m): v for m, v in
                                   success_curve(i1s, tau).items()}
                        for tau in TAUS},
            'i1_star': float(max(i1s))}

    # ---- law test over all measured cells (archived + this round)
    law_cells = []
    for k in (30, 300, 1000):                  # E-W4 44500 cells
        seeds_k = {30: (44101, 44102, 44103), 300: (44111, 44112, 44113),
                   1000: (44121, 44122, 44123)}[k]
        rts = [json.load(open(os.path.join(
            RUN3, 'rt_k{}_s{}.json'.format(k, s))))['per_k'][str(k)]
            for s in seeds_k]
        pt3 = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k']
        fms = [e['m6_fm'] for e in rts]
        med = float(np.median(fms))
        b = max(abs(v - med) for v in fms)
        e = abs(pt3[str(k)]['m6_fm'] - med)
        law_cells.append(('ew4_44500_k{}'.format(k),
                          b / max(e, 1e-12), None))
    import glob as _g
    for t5 in ('armA_t44600_k30', 'armA_t44600_k300', 'armA_t44700_k30',
               'armA_t44700_k300', 'armB_topF', 'armB_botF',
               'armB_lossF'):
        fns = [f for f in sorted(_g.glob(os.path.join(
            RUN5, t5 + '_s*.json'))) if 'opt' not in os.path.basename(f)]
        rts = [json.load(open(f))['mm'] for f in fns]
        pt5 = json.load(open(os.path.join(RUN5, t5 + '_pt.json')))
        fms = [e['m6_fm'] for e in rts]
        med = float(np.median(fms))
        b = max(abs(v - med) for v in fms)
        e = abs(pt5['m6_fm'] - med)
        law_cells.append((t5, b / max(e, 1e-12), None))
    for key, u in units_out.items():           # this round (6 seeds)
        law_cells.append(('ew6_' + key, u['b_over_E'], u['i1_star']))
    # attach I1* for archived cells (E-W4/E-W5 recorded values)
    archived_i1 = {'ew4_44500_k30': 0.2477, 'ew4_44500_k300': 0.2139,
                   'ew4_44500_k1000': 0.1066,
                   'armA_t44600_k30': 0.1199, 'armA_t44600_k300': 0.1702,
                   'armA_t44700_k30': 0.3302, 'armA_t44700_k300': 0.0037,
                   'armB_topF': 0.1893, 'armB_botF': 0.1296,
                   'armB_lossF': 0.2053}
    cells = []
    for name, be, i1v in law_cells:
        v = archived_i1.get(name, i1v)
        cells.append({'cell': name, 'b_over_E': be, 'i1_star': v})
    xs = [c['b_over_E'] for c in cells]
    ys = [c['i1_star'] for c in cells]
    rho = spearman(xs, ys)
    status, reason = verdict_from(units_out, rho)
    verdict = {'status': status, 'reason': reason, 'spearman': rho,
               'law_cells': cells,
               'units': {k: {m: v for m, v in u.items() if m != 'seeds'}
                         for k, u in units_out.items()}}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(verdict, fh, indent=1)
    print('LAW rho =', round(rho, 3), 'over', len(cells), 'cells',
          flush=True)
    print('VERDICT:', status, '--', reason, flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
