"""UW E-W8: second dataset (CIFAR-100) + stronger audit families
(frozen prereg docs/plans/2026-09-26-uw-ew8-preregistration.md).

Arm D: CIFAR-100, ResNet-18(num_classes=100), pool = class 0 (500
records), k in {30,100,250}, theta0 seed 45400, rt seeds 45600+10ki+s,
sub RandomState(45700); same Tier-A audit + ensemble bands + S0 wash.
Arm E: CIFAR-10 ew6 unit t44500_k30 reuse; stronger audits E1 (RBF-MMD
on [CE,margin]), E2 (MLP-MIA AUC on 5 features), E3 (per-record extreme
test); frontiers under Tier-A+{E1,E2,E3,all}.
"""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import TIER_A, accept, i1
from uw.run_ew3b import LR0, loo_bands, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = sorted(__import__('glob').glob(
    'research_outputs/uw/ew6-2*'))[-1]
K_D = (30, 100, 250)
POOL_CLS = 0
THETA0_D = 45400
RT_D = lambda ki, s: 45600 + 10 * ki + s
SUB_D = 45700
SUB_N = 3000
MAX_ITERS = 60
STALL_LIMIT = 15


# ------------- pure-numpy stronger-audit primitives (tested) ----------

def mmd_rbf(x, y, sub=None):
    """Unbiased RBF-MMD^2 on standardized features, median-heuristic
    bandwidth over the pooled sample."""
    z = np.vstack([x, y]).astype(float)
    mu, sd = z.mean(0), z.std(0) + 1e-12
    x = (x - mu) / sd
    y = (y - mu) / sd
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    k = np.exp(-d2 / med)
    n, m = len(x), len(y)
    kxx = k[:n, :n]
    kyy = k[n:, n:]
    kxy = k[:n, n:]
    mxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    myy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(mxx + myy - 2 * kxy.mean())


def mlp_auc(feats_pos, feats_neg, seed, hidden=32, steps=200, lr=0.05):
    """Deterministic 2-layer tanh MLP probe (numpy), returns AUC."""
    from uw.run_ew3 import auc_mw
    X = np.vstack([feats_pos, feats_neg]).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    X = (X - mu) / sd
    y = np.concatenate([np.ones(len(feats_pos)),
                        np.zeros(len(feats_neg))])
    rng = np.random.RandomState(seed)
    W1 = rng.randn(X.shape[1], hidden) * 0.1
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden) * 0.1
    b2 = 0.0
    for _ in range(steps):
        h = np.tanh(X @ W1 + b1)
        p = 1 / (1 + np.exp(-(h @ W2 + b2)))
        g2 = ((p - y) * p * (1 - p))[:, None] * W2[None, :] / len(y)
        gw2 = h.T @ ((p - y)) / len(y)
        gb2 = (p - y).mean()
        gh = (1 - h ** 2) * g2
        gw1 = X.T @ gh / len(y)
        gb1 = gh.mean(0)
        W2 -= lr * gw2
        b2 -= lr * gb2
        W1 -= lr * gw1
        b1 -= lr * gb1
    h = np.tanh(X @ W1 + b1)
    s = h @ W2 + b2
    return auc_mw(s[:len(feats_pos)], s[len(feats_pos):])


def per_record_ok(margins, med, ens_worst):
    """E3 acceptance: wash's worst per-record deviation <= the honest
    ensemble's worst (self-inclusive)."""
    dev = np.abs(margins - med).max()
    return bool(dev <= ens_worst + 1e-12)


class Indexed:
    """(index, x, y) wrapper over a torchvision dataset."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return i, x, y

    def __getattr__(self, name):
        return getattr(self.ds, name)


def main():
    import torch
    import torchvision.transforms as T
    from tools.models import resnet18

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--arm', default='D', choices=['D', 'E'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    epochs = 2 if cli.smoke else 40
    iters = 8 if cli.smoke else MAX_ITERS
    device = torch.device('cuda:{}'.format(cli.gpu))

    tag = 'ew8{}-smoke-'.format(cli.arm) if cli.smoke \
        else 'ew8{}-'.format(cli.arm)
    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    def batch_x(dataset, idx):
        ls = []
        for _g, x, _y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(Indexed(dataset), idx),
                batch_size=500, shuffle=False):
            ls.append(x)
        return torch.cat(ls).to(device)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def make_model(num_classes=10):
        m = resnet18(num_classes=num_classes).to(device)
        return m

    def fit(trainset, yfull, keep, seed, num_classes):
        seed_all(seed)
        m = make_model(num_classes)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(Indexed(trainset), keep),
            batch_size=100, shuffle=True)
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

    if cli.arm == 'D':
        from torchvision.datasets import CIFAR100
        tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, 4),
                        T.ToTensor()])
        ev = T.Compose([T.ToTensor()])
        trainset = CIFAR100('data', train=True, transform=tr)
        evalset = CIFAR100('data', train=True, transform=ev)
        testset = CIFAR100('data', train=False, transform=ev)
        yfull = np.asarray(trainset.targets)
        n = len(trainset)
        pool = np.where(yfull == POOL_CLS)[0]           # all 500
        sub = np.sort(np.random.RandomState(SUB_D).choice(
            np.setdiff1d(np.arange(n), pool), size=SUB_N,
            replace=False))
        loader_ps = torch.utils.data.DataLoader(
            torch.utils.data.Subset(Indexed(evalset),
                                    np.concatenate([pool, sub])),
            batch_size=500, shuffle=False)
        loader_te = torch.utils.data.DataLoader(
            Indexed(testset), batch_size=500, shuffle=False)
        model = make_model(100)

        def eval_logits(sd):
            model.load_state_dict(sd)
            model.eval()
            zs, ys = [], []
            with torch.no_grad():
                for _i, x, y in loader_te:
                    zs.append(model(x.to(device)).cpu())
                    ys.append(y)
            z_te = torch.cat(zs).numpy()
            y_te = torch.cat(ys).numpy()
            zs = []
            with torch.no_grad():
                for _g, x, _y in loader_ps:
                    zs.append(model(x.to(device)).cpu())
            z_ps = torch.cat(zs).numpy()
            return z_te, y_te, z_ps[:len(pool)], z_ps[len(pool):]

        def metrics_D(z_te, y_te, z_pool_F, z_sub, y_F):
            from uw.run_ew3 import (_ce_rows, _margin_rows, ks_stat,
                                    logistic_auc)
            pred_te = z_te.argmax(1)
            m1 = float((pred_te == y_te).mean())
            cls_te = y_te == POOL_CLS
            m2 = float((pred_te[cls_te] == y_te[cls_te]).mean())
            y_S = yfull[sub]
            ce_F = _ce_rows(z_pool_F, y_F)
            ce_S = _ce_rows(z_sub, y_S)
            mg_F = _margin_rows(z_pool_F, y_F)
            m3 = float(ce_F.mean())
            m4 = ks_stat(ce_F, ce_S)
            m5 = logistic_auc(np.column_stack([ce_F, mg_F]),
                              np.column_stack([ce_S, _margin_rows(
                                  z_sub, y_S)]))
            return {'m1_acc': m1, 'm2_catacc': m2, 'm3_Floss': m3,
                    'm4_KS': m4, 'm5_adv': m5,
                    'm6_fm': float(mg_F.mean()),
                    'util_margin': float(_margin_rows(
                        z_te, y_te).mean())}

        theta0_p = os.path.join(ck, 'theta0.pt')
        if os.path.exists(theta0_p):
            sd0 = torch.load(theta0_p, map_location=device)
        else:
            sd0 = fit(trainset, yfull, np.arange(n), THETA0_D, 100)
            torch.save(sd0, theta0_p)
            print('theta0 trained', flush=True)
        z_te, y_te, z_pool0, z_sub0 = eval_logits(sd0)

        results = {'units': {}}
        for ki, k in enumerate(K_D[: (1 if cli.smoke else 3)]):
            fset = np.sort(np.random.RandomState(
                45500 + k).permutation(pool)[:k])
            pos = np.searchsorted(pool, fset)
            keep = np.setdiff1d(np.arange(n), fset)
            mm_seeds, sd_seeds = {}, {}
            for s in range(1, (2 if cli.smoke else 3) + 1):
                seed = RT_D(ki, s)
                p = os.path.join(ck, 'rt_k{}_s{}.pt'.format(k, seed))
                fn = 'rt_k{}_s{}.json'.format(k, seed)
                if os.path.exists(os.path.join(outdir, fn)):
                    mm_seeds[seed] = json.load(open(
                        os.path.join(outdir, fn)))
                    sd_seeds[seed] = torch.load(p, map_location=device)
                    continue
                sd = fit(trainset, yfull, keep, seed, 100)
                torch.save(sd, p)
                sd_seeds[seed] = sd
                z_te2, y_te2, zp, zs2 = eval_logits(sd)
                mm = metrics_D(z_te2, y_te2, zp[pos], zs2,
                               yfull[fset])
                mm_seeds[seed] = mm
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(mm, fh, indent=1)
            centers = {m: float(np.median(
                [mm_seeds[x][m] for x in mm_seeds]))
                for m in TIER_A}
            bands = loo_bands(mm_seeds, centers)
            fm0 = metrics_D(z_te, y_te, z_pool0[pos], z_sub0,
                            yfull[fset])['m6_fm']
            fm_med = float(np.median(
                [mm_seeds[x]['m6_fm'] for x in mm_seeds]))
            b_fm = float(max(abs(mm_seeds[x]['m6_fm'] - fm_med)
                             for x in mm_seeds))
            xF = batch_x(evalset, fset)
            yF = torch.as_tensor(yfull[fset],
                                 dtype=torch.long).to(device)
            i1s = []
            for x in mm_seeds:
                fn = 'opt_k{}_s{}.json'.format(k, x)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    cur = sd_seeds[x]
                    lr = LR0
                    best = None

                    def fm_of(sd_):
                        model.load_state_dict(sd_)
                        model.eval()
                        with torch.no_grad():
                            z = model(xF)
                        zy = z.gather(1, yF[:, None]).squeeze(1)
                        zc = z.clone()
                        zc.scatter_(1, yF[:, None], -1e18)
                        return float((zy - zc.max(1).values).mean())

                    best = fm_of(cur)
                    log = [{'iter': 0, 'fm': best}]
                    stall = 0
                    for it in range(1, iters + 1):
                        model.load_state_dict(cur)
                        model.eval()
                        z = model(xF)
                        zy = z.gather(1, yF[:, None]).squeeze(1)
                        zc = z.clone()
                        zc.scatter_(1, yF[:, None], -1e18)
                        margin = (zy - zc.max(1).values).mean()
                        model.zero_grad()
                        margin.backward()
                        g = {nm: p_.grad.detach().clone()
                             for nm, p_ in model.named_parameters()
                             if p_.grad is not None}
                        acc = False
                        for _bt in range(12):
                            cand = {k_: (v + lr * g[k_] if k_ in g
                                         else v)
                                    for k_, v in cur.items()}
                            model.load_state_dict(cand)
                            model.eval()
                            zs, ys = [], []
                            with torch.no_grad():
                                for _i2, xx, yy in loader_te:
                                    zs.append(model(
                                        xx.to(device)).cpu())
                                    ys.append(yy)
                            z_t2 = torch.cat(zs).numpy()
                            y_t2 = torch.cat(ys).numpy()
                            zs = []
                            with torch.no_grad():
                                for _g, xx, _yy in loader_ps:
                                    zs.append(model(
                                        xx.to(device)).cpu())
                            z_p2 = torch.cat(zs).numpy()
                            mm2 = metrics_D(z_t2, y_t2,
                                            z_p2[:len(pool)][pos],
                                            z_p2[len(pool):],
                                            yfull[fset])
                            ok, _v = accept(mm2, centers, bands[x],
                                            TIER_A)
                            fmc = mm2['m6_fm']
                            if ok and fmc > best + 1e-7:
                                cur, best, acc = cand, fmc, True
                                log.append({'iter': it, 'fm': best})
                                break
                            lr *= 0.5
                            if lr < 1e-9:
                                break
                        lr = lr_update(lr, acc)
                        stall = 0 if acc else stall + 1
                        if stall > STALL_LIMIT:
                            break
                    rec = {'I1': i1(best, mm_seeds[x]['m6_fm'], fm0),
                           'log': log}
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1)
                i1s.append(rec['I1'])
                print('D', k, x, 'I1', round(rec['I1'], 4),
                      flush=True)
            i1s_all = i1s
            results['units'][str(k)] = {
                'i1_star': float(max(i1s_all)), 'i1_per_seed': i1s_all,
                'b_over_E': b_fm / max(abs(fm0 - fm_med), 1e-12),
                'leak_010': sum(1 for v in i1s_all if v >= 0.10),
                'leak_020': sum(1 for v in i1s_all if v >= 0.20)}
        rep = any(u['i1_star'] >= 0.10
                  for u in results['units'].values())
        results['verdict'] = ('REPLICATED' if rep else
                              'DATASET_BOUNDARY')
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(results, fh, indent=1)
        print('VERDICT(D):', results['verdict'], flush=True)
        print('saved', outdir, flush=True)
        return

    # ---------------- Arm E ----------------
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from uw.run_ew3 import CAT_CLS, INST, POOL_SEED, SUB_SEED, \
        build_pool, f_deep, metrics_from_logits, _ce_rows, \
        _margin_rows

    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset = ing.trainset
    testloader = ing.testloader
    evalset = CIFAR10('data', train=True,
                      transform=T.Compose([T.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    fset = f_deep(pool, 30)
    pos = np.searchsorted(pool, fset)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=3000, replace=False))
    model = make_model(10)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    seeds6 = [44101, 44102, 44103, 45201, 45202, 45203]
    sd6 = {}
    for s in seeds6:
        if s < 45200:      # archived E-W3 seeds
            p = os.path.join(RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s))
        else:              # E-W6 new seeds
            p = os.path.join(RUN6, 'ckpt',
                             'u0_t44500_k30_s{}.pt'.format(s))
        sd6[s] = torch.load(p, map_location=device)
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(Indexed(evalset),
                                np.concatenate([pool, sub])),
        batch_size=500, shuffle=False)

    def logits_ps(sd):
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
            for _g, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return z_te, y_te, z_ps[:len(pool)], z_ps[len(pool):]

    def rich(zF, zS, z_te, y_te):
        yF = yfull[fset]
        yS = yfull[sub]
        ce_F, ce_S = _ce_rows(zF, yF), _ce_rows(zS, yS)
        mg_F, mg_S = _margin_rows(zF, yF), _margin_rows(zS, yS)

        def ent(z):
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            return -(p * np.log(p + 1e-12)).sum(1)

        def conf(z):
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            return p.max(1)

        def corr(z, y):
            return (z.argmax(1) == y).astype(float)

        fF = np.column_stack([ce_F, mg_F, ent(zF), conf(zF),
                              corr(zF, yF)])
        fS = np.column_stack([ce_S, mg_S, ent(zS), conf(zS),
                              corr(zS, yS)])
        return ce_F, ce_S, mg_F, mg_S, fF, fS

    cache = {}
    for name, sd in [('t0', sd0)] + list(sd6.items()):
        z_te, y_te, z_pool, z_sub = logits_ps(sd)
        ce_F, ce_S, mg_F, mg_S, fF, fS = rich(z_pool[pos], z_sub,
                                              z_te, y_te)
        mm = metrics_from_logits(z_pool[pos], yfull[fset], z_sub,
                                 yfull[sub], z_te, y_te)
        cache[name] = {'mm': mm, 'ce_F': ce_F, 'ce_S': ce_S,
                       'mg_F': mg_F, 'mg_S': mg_S, 'fF': fF, 'fS': fS}
    # ensemble stats for E1/E2/E3
    mmd6 = [mmd_rbf(cache[s]['fF'][:, :2], cache[s]['fS'][:, :2])
            for s in seeds6]
    mmd0 = mmd_rbf(cache['t0']['fF'][:, :2], cache['t0']['fS'][:, :2])
    mmd_med = float(np.median(mmd6))
    band_mmd = max(abs(v - mmd_med) for v in mmd6)
    auc6 = [mlp_auc(cache[s]['fF'], cache[s]['fS'], 45800)
            for s in seeds6]
    auc0 = mlp_auc(cache['t0']['fF'], cache['t0']['fS'], 45800)
    auc_med = float(np.median(auc6))
    band_auc = max(abs(v - auc_med) for v in auc6)
    mg6 = np.stack([cache[s]['mg_F'] for s in seeds6])
    mg_med = np.median(mg6, axis=0)
    ens_worst = np.abs(mg6 - mg_med).max()
    centersE = {m: float(np.median([cache[s]['mm'][m]
                                    for s in seeds6]))
                for m in TIER_A}
    bandsE = loo_bands({s: cache[s]['mm'] for s in seeds6},
                       centersE)
    fm0 = cache['t0']['mm']['m6_fm']

    def feas_E(sd, variant, seed):
        z_te, y_te, z_pool, z_sub = logits_ps(sd)
        ce_F, ce_S, mg_F, mg_S, fF, fS = rich(z_pool[pos], z_sub,
                                              z_te, y_te)
        from uw.run_ew3 import metrics_from_logits as mfl
        mm = mfl(z_pool[pos], yfull[fset], z_sub, yfull[sub],
                 z_te, y_te)
        ok, viol = accept(mm, centersE, bandsE[seed], TIER_A)
        if not ok:
            return False, mm
        if 'E1' in variant:
            v = mmd_rbf(fF[:, :2], fS[:, :2])
            if abs(v - mmd_med) > band_mmd:
                return False, mm
        if 'E2' in variant:
            v = mlp_auc(fF, fS, 45800)
            if abs(v - auc_med) > band_auc:
                return False, mm
        if 'E3' in variant:
            if not per_record_ok(mg_F, mg_med, ens_worst):
                return False, mm
        return True, mm

    variants = ['A', 'AE1', 'AE2', 'AE3', 'AE123']
    out = {'baseline_I1_star': 0.248, 'variants': {}}
    for variant in variants[: (2 if cli.smoke else 5)]:
        i1s = []
        for s in seeds6[:(1 if cli.smoke else 6)]:
            fn = 'E_{}_s{}.json'.format(variant, s)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                cur = sd6[s]
                lr = LR0
                xF = batch_x(evalset, fset)
                yF = torch.as_tensor(yfull[fset],
                                     dtype=torch.long).to(device)

                def fm_of(sd_):
                    model.load_state_dict(sd_)
                    model.eval()
                    with torch.no_grad():
                        z = model(xF)
                    zy = z.gather(1, yF[:, None]).squeeze(1)
                    zc = z.clone()
                    zc.scatter_(1, yF[:, None], -1e18)
                    return float((zy - zc.max(1).values).mean())

                best = fm_of(cur)
                log = [{'iter': 0, 'fm': best}]
                stall = 0
                for it in range(1, iters + 1):
                    model.load_state_dict(cur)
                    model.eval()
                    z = model(xF)
                    zy = z.gather(1, yF[:, None]).squeeze(1)
                    zc = z.clone()
                    zc.scatter_(1, yF[:, None], -1e18)
                    margin = (zy - zc.max(1).values).mean()
                    model.zero_grad()
                    margin.backward()
                    g = {nm: p_.grad.detach().clone()
                         for nm, p_ in model.named_parameters()
                         if p_.grad is not None}
                    acc = False
                    for _bt in range(12):
                        cand = {k_: (v + lr * g[k_] if k_ in g else v)
                                for k_, v in cur.items()}
                        ok, mm2 = feas_E(cand, variant, s)
                        if ok and mm2['m6_fm'] > best + 1e-7:
                            cur, best, acc = cand, mm2['m6_fm'], True
                            log.append({'iter': it, 'fm': best})
                            break
                        lr *= 0.5
                        if lr < 1e-9:
                            break
                    lr = lr_update(lr, acc)
                    stall = 0 if acc else stall + 1
                    if stall > STALL_LIMIT:
                        break
                rec = {'I1': i1(best, cache[s]['mm']['m6_fm'], fm0),
                       'log': log}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1)
            i1s.append(rec['I1'])
            print('E', variant, s, 'I1', round(rec['I1'], 4),
                  flush=True)
        out['variants'][variant] = {
            'i1_star': float(max(i1s)), 'i1_per_seed': i1s}
    full = out['variants'].get('AE123', {}).get('i1_star')
    out['verdict'] = {
        'full_stack_closes': bool(full is not None and full < 0.10),
        'note': 'report-only per prereg'}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT(E):', json.dumps(out['verdict']), flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
