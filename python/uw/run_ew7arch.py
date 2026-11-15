"""UW E-W7-arch: one MobileNetV2 cell (attack-claim architecture row,
optional follow-up approved with the skeleton path on 2026-09-26).

Cell: MobileNetV2, k=30, theta_0 seed 45300, 3 theta_R seeds 45301-03,
uniform F = f_deep(pool, 30); self-inclusive ensemble bands (3 seeds);
S0 optimized wash per seed (E-W4/E-W6 semantics); readouts = per-seed I1,
unit I1*, leak counts at 0.10/0.20, b/E.  Same recipe/data conventions as
run_ew6 (40ep cosine from scratch).
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

THETA0_SEED = 45300
RT_SEEDS = (45301, 45302, 45303)
K = 30
MAX_ITERS = 60
STALL_LIMIT = 15


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from torchvision.models import mobilenet_v2
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    epochs = 2 if cli.smoke else 40
    iters = 8 if cli.smoke else MAX_ITERS
    rt_seeds = RT_SEEDS[:1] if cli.smoke else RT_SEEDS

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ew7arch-smoke-' if cli.smoke else 'ew7arch-'
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
    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(n), fset)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]
    model = mobilenet_v2(num_classes=10).to(device)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def fit(train_idx, seed):
        seed_all(seed)
        m = mobilenet_v2(num_classes=10).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, train_idx), batch_size=100,
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

    def get_sd(name, train_idx, seed):
        p = os.path.join(ck, name + '.pt')
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = fit(train_idx, seed)
        torch.save(sd, p)
        print(name, 'trained', flush=True)
        return sd

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)
    pos_f = np.searchsorted(pool, fset)

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
        return metrics_from_logits(z_ps[:len(pool)][pos_f], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    xf, yfl = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset), batch_size=500,
            shuffle=False):
        xf.append(x)
        yfl.append(y)
    xF = torch.cat(xf).to(device)
    yF = torch.cat(yfl).to(device).long()

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

    sd0 = get_sd('theta0', np.arange(n), THETA0_SEED)
    pt = eval_model(sd0)
    mm_all, sd_all = {}, {}
    for s in rt_seeds:
        sd_all[s] = get_sd('rt_s{}'.format(s), keep, s)
        mm_all[s] = eval_model(sd_all[s])
    centers = {m: float(np.median([mm_all[s][m] for s in rt_seeds]))
               for m in TIER_A}
    bands = loo_bands(mm_all, centers)
    fm_med = float(np.median([mm_all[s]['m6_fm'] for s in rt_seeds]))
    b_fm = float(max(abs(mm_all[s]['m6_fm'] - fm_med) for s in rt_seeds))
    i1s = []
    for s in rt_seeds:
        cur = sd_all[s]
        lr = LR0
        best_fm = eval_model(cur)['m6_fm']
        log = [{'iter': 0, 'fm': best_fm, 'metrics': eval_model(cur)}]
        stall = 0
        for it in range(1, iters + 1):
            g = grad_delta(cur)
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                mm = eval_model(cand)
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
        rec_i1 = i1(best_fm, mm_all[s]['m6_fm'], pt['m6_fm'])
        i1s.append(rec_i1)
        with open(os.path.join(
                outdir, 'opt_s{}.json'.format(s)), 'w') as fh:
            json.dump({'seed': s, 'I1': rec_i1,
                       'n_accepted': len(log) - 1, 'log': log}, fh,
                      indent=1)
        print('opt', s, 'I1', round(rec_i1, 4), flush=True)
    E = abs(pt['m6_fm'] - fm_med)
    out = {'arch': 'MobileNetV2', 'k': K, 'i1_per_seed': i1s,
           'i1_star': float(max(i1s)),
           'leak_010': sum(1 for v in i1s if v >= 0.10),
           'leak_020': sum(1 for v in i1s if v >= 0.20),
           'b_over_E': b_fm / max(E, 1e-12),
           'b_fm': b_fm, 'E_fm': E,
           'pt': pt, 'rt': {str(s): mm_all[s] for s in rt_seeds}}
    with open(os.path.join(outdir, 'cell.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('CELL I1* =', out['i1_star'], 'b/E =', round(out['b_over_E'], 3),
          flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
