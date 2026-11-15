"""UW E-W10: scale endpoint k=2000 (I), NegGrad operator reference
(J), wash footprint (L).  Frozen prereg 2026-09-27-uw-ew10-*.md."""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, group_of, loo_bands, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = sorted(__import__('glob').glob(
    'research_outputs/uw/ew6-2*'))[-1]
K_I = 2000
RT_I = lambda s: 46100 + s
NG_SEED = lambda s: 46200 + s
NG_STEPS = 300
MAX_ITERS = 60
STALL_LIMIT = 15


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--arm', default='I', choices=['I', 'J', 'L'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    epochs = 2 if cli.smoke else 40
    iters = 8 if cli.smoke else MAX_ITERS
    device = torch.device('cuda:{}'.format(cli.gpu))
    tag = 'ew10{}-smoke-'.format(cli.arm) if cli.smoke \
        else 'ew10{}-'.format(cli.arm)
    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    evalset = CIFAR10(train=True,
                      transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]
    model = get_model('ResNet18').to(device)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    pt3 = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k']

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(evalset, idx_ps), batch_size=500,
        shuffle=False)

    def eval_model(sd, k):
        fset = f_deep(pool, k)
        pos = np.searchsorted(pool, fset)
        model.load_state_dict(sd)
        zs, ys = [], []
        model.eval()
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
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    def batch_F(k):
        fset = f_deep(pool, k)
        lx, ly = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(evalset, fset), batch_size=500,
                shuffle=False):
            lx.append(x)
            ly.append(y)
        return torch.cat(lx).to(device), torch.cat(ly).to(
            device).long(), fset

    def grad_delta(sd, xF, yF, sign=1.0):
        model.load_state_dict(sd)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        margin = (zy - zc.max(1).values).mean()
        model.zero_grad()
        (sign * margin).backward()
        return {nm: p.grad.detach().clone() for nm, p in
                model.named_parameters() if p.grad is not None}

    def apply_delta(base, delta, lr):
        return {k_: (v + lr * delta[k_] if k_ in delta else v)
                for k_, v in base.items()}

    def retrain(k, seed):
        fset = f_deep(pool, k)
        keep = np.setdiff1d(np.arange(n), fset)
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        for _ep in range(epochs):
            m.train()
            for _ids, x, _yy in loader:
                opt.zero_grad()
                lf(m(x.to(device)),
                   y_t[_ids.tolist()].to(device)).backward()
                opt.step()
            sched.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    def neggrad(k, seed, steps):
        xF, yF, fset = batch_F(k)
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        lf = torch.nn.CrossEntropyLoss()
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        m.train()
        for _ in range(steps):
            idx = torch.randint(0, len(yF), (100,), generator=gen)
            opt.zero_grad()
            lf(m(xF[idx.to(device)]), yF[idx.to(device)]).backward()
            for p in m.parameters():       # gradient ascent on F loss
                if p.grad is not None:
                    p.grad = -p.grad
            opt.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    def optimize(sd_start, k, centers, bands, seed, xF, yF,
                 save_path=None):
        cur = sd_start
        lr = LR0
        best = eval_model(cur, k)['m6_fm']
        log = [{'iter': 0, 'fm': best}]
        stall = 0
        for it in range(1, iters + 1):
            g = grad_delta(cur, xF, yF)
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                mm = eval_model(cand, k)
                ok, _v = accept(mm, centers, bands[seed], TIER_A)
                if ok and mm['m6_fm'] > best + 1e-7:
                    cur, best, acc = cand, mm['m6_fm'], True
                    log.append({'iter': it, 'fm': best})
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        if save_path is not None:
            torch.save(cur, save_path)
        return best, len(log) - 1

    if cli.arm == 'I':
        xF, yF, fset = batch_F(K_I)
        seeds = [RT_I(s) for s in range(1, (2 if cli.smoke else 6) + 1)]
        sd_seeds, mm_seeds = {}, {}
        for s in seeds:
            fn = 'rt_k2000_s{}.json'.format(s)
            pth = os.path.join(ck, fn.replace('.json', '.pt'))
            if os.path.exists(os.path.join(outdir, fn)):
                mm_seeds[s] = json.load(open(os.path.join(outdir, fn)))
                sd_seeds[s] = torch.load(pth, map_location=device)
                continue
            sd = retrain(K_I, s)
            torch.save(sd, pth)
            sd_seeds[s] = sd
            mm_seeds[s] = eval_model(sd, K_I)
            with open(os.path.join(outdir, fn), 'w') as fh:
                json.dump(mm_seeds[s], fh, indent=1)
            print('rt2000', s, 'm1', round(mm_seeds[s]['m1_acc'], 4),
                  flush=True)
        centers = {m: float(np.median([mm_seeds[x][m] for x in seeds]))
                   for m in TIER_A}
        bands = loo_bands(mm_seeds, centers)
        fm_med = float(np.median([mm_seeds[x]['m6_fm'] for x in seeds]))
        b_fm = float(max(abs(mm_seeds[x]['m6_fm'] - fm_med)
                         for x in seeds))
        fm0 = pt3['30']['m6_fm']
        fm0k = eval_model(sd0, K_I)['m6_fm']
        i1s = []
        for s in seeds:
            fn = 'opt_k2000_s{}.json'.format(s)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, nacc = optimize(sd_seeds[s], K_I, centers,
                                      bands, s, xF, yF)
                rec = {'I1': i1(best, mm_seeds[s]['m6_fm'], fm0k),
                       'n_accepted': nacc}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1)
            i1s.append(rec['I1'])
            print('I', s, 'I1', round(rec['I1'], 4), flush=True)
        out = {'i1_star': float(max(i1s)), 'i1_per_seed': i1s,
               'gamma': b_fm / max(abs(fm0k - fm_med), 1e-12),
               'leak_010': sum(1 for v in i1s if v >= 0.10),
               'leak_020': sum(1 for v in i1s if v >= 0.20)}
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('VERDICT(I): I1*', round(out['i1_star'], 4),
              'gamma', round(out['gamma'], 4), flush=True)
        print('saved', outdir, flush=True)
        return

    if cli.arm == 'J':
        k = 30
        xF, yF, fset = batch_F(k)
        seeds = [NG_SEED(s) for s in
                 range(1, (2 if cli.smoke else 6) + 1)]
        sd_seeds, mm_seeds = {}, {}
        for s in seeds:
            fn = 'ng_k30_s{}.json'.format(s)
            pth = os.path.join(ck, fn.replace('.json', '.pt'))
            if os.path.exists(os.path.join(outdir, fn)):
                mm_seeds[s] = json.load(open(os.path.join(outdir, fn)))
                sd_seeds[s] = torch.load(pth, map_location=device)
                continue
            sd = neggrad(k, s, 6 if cli.smoke else NG_STEPS)
            torch.save(sd, pth)
            sd_seeds[s] = sd
            mm_seeds[s] = eval_model(sd, k)
            with open(os.path.join(outdir, fn), 'w') as fh:
                json.dump(mm_seeds[s], fh, indent=1)
            print('ng', s, 'm1', round(mm_seeds[s]['m1_acc'], 4),
                  flush=True)
        centers = {m: float(np.median([mm_seeds[x][m] for x in seeds]))
                   for m in TIER_A}
        bands = loo_bands(mm_seeds, centers)
        fm_med = float(np.median([mm_seeds[x]['m6_fm'] for x in seeds]))
        b_fm = float(max(abs(mm_seeds[x]['m6_fm'] - fm_med)
                         for x in seeds))
        fm0 = pt3['30']['m6_fm']
        i1s = []
        for s in seeds:
            fn = 'opt_k30_s{}.json'.format(s)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, nacc = optimize(sd_seeds[s], k, centers, bands,
                                      s, xF, yF)
                rec = {'I1': i1(best, mm_seeds[s]['m6_fm'], fm0),
                       'n_accepted': nacc}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1)
            i1s.append(rec['I1'])
            print('J', s, 'I1', round(rec['I1'], 4), flush=True)
        out = {'i1_star': float(max(i1s)), 'i1_per_seed': i1s,
               'gamma_ng': b_fm / max(abs(fm0 - fm_med), 1e-12),
               'leak_010': sum(1 for v in i1s if v >= 0.10),
               'leak_020': sum(1 for v in i1s if v >= 0.20)}
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('VERDICT(J): I1*', round(out['i1_star'], 4),
              'gamma', round(out['gamma_ng'], 4), flush=True)
        print('saved', outdir, flush=True)
        return

    # ---- Arm L: footprint on the six-seed k=30 unit
    k = 30
    xF, yF, fset = batch_F(k)
    seeds6 = [44101, 44102, 44103, 45201, 45202, 45203]
    sd6 = {}
    for s in seeds6:
        p = (os.path.join(RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s))
             if s < 45200 else os.path.join(
                 RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)))
        sd6[s] = torch.load(p, map_location=device)
    mm6 = {s: eval_model(sd6[s], k) for s in seeds6}
    centers = {m: float(np.median([mm6[s][m] for s in seeds6]))
               for m in TIER_A}
    bands = loo_bands(mm6, centers)
    fm0 = pt3['30']['m6_fm']
    rows = []
    for s in seeds6[:(1 if cli.smoke else 6)]:
        fn = 'foot_s{}.json'.format(s)
        if os.path.exists(os.path.join(outdir, fn)):
            rec = json.load(open(os.path.join(outdir, fn)))
        else:
            savep = os.path.join(ck, 'wash_s{}.pt'.format(s))
            best, nacc = optimize(sd6[s], k, centers, bands, s, xF,
                                  yF, save_path=savep)
            sd_w = torch.load(savep, map_location=device)
            num = sum(float((sd_w[kk] - sd0[kk]).float().norm() ** 2)
                      for kk in sd_w)
            den = sum(float((sd6[s][kk] - sd0[kk]).float().norm() ** 2)
                      for kk in sd_w)
            ratio = (num ** 0.5) / (den ** 0.5)
            groups = {}
            gden = {}
            for kk in sd_w:
                g = group_of(kk)
                d2 = float((sd_w[kk] - sd6[s][kk]).float().norm() ** 2)
                groups[g] = groups.get(g, 0.0) + d2
                t2 = float((sd6[s][kk] - sd0[kk]).float().norm() ** 2)
                gden[g] = gden.get(g, 0.0) + t2
            rec = {'I1': i1(best, mm6[s]['m6_fm'], fm0),
                   'disp_ratio': ratio,
                   'group_share': {g: round(v / max(num, 1e-12), 3)
                                   for g, v in groups.items()}}
            with open(os.path.join(outdir, fn), 'w') as fh:
                json.dump(rec, fh, indent=1)
        rows.append(rec)
        print('L', s, 'I1', round(rec['I1'], 4), 'disp',
              round(rec['disp_ratio'], 4), flush=True)
    ratios = [r['disp_ratio'] for r in rows]
    out = {'median_disp_ratio': float(np.median(ratios)),
           'min': float(min(ratios)), 'max': float(max(ratios)),
           'rows': rows}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT(L): median disp ratio',
          round(out['median_disp_ratio'], 4), flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
