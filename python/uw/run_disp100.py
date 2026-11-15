"""A2: k=100 dispersion ratio (prereg 2026-11-12 path A).

Replicates the ew3 pipeline exactly (same theta0, pool/f_deep, sub seed
52000, metrics_from_logits) for k=100; noise side = the six archived
full-data retrains re-evaluated on F(100); honest side = 3 fresh
from-scratch retrains on D minus F(100).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms as _tf

import uw.run_ew3 as ew3
from uw.run_ew3 import (CAT_CLS, EPOCHS, POOL_N, POOL_SEED, SUB_N,
                        SUB_SEED, THETA0_SEED, build_pool, f_deep,
                        metrics_from_logits, seed_all)
from tools.cifar_10 import CIFAR10

EW3_DIR = 'research_outputs/uw/ew3-20260925T113900'
K100 = 100
RT_SEEDS_100 = (44141, 44142, 44143)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.models import get_model

    args = build_args(seed=int(ew3.INST), dataset='CIFAR10', epochs=EPOCHS,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    tset_eval = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(tset_eval.targets)
    n = len(tset_eval)

    cat_idx = np.where(yfull == CAT_CLS)[0]
    pool = build_pool(cat_idx)
    assert len(pool) == POOL_N, len(pool)
    f = f_deep(pool, K100)
    keep = np.setdiff1d(np.arange(n), f)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    pos_of = np.searchsorted(pool, f)
    y_pool = yfull[pool]
    y_sub = yfull[sub]

    model = get_model('ResNet18').to(device)

    def collect_test():
        zs, ys = [], []
        with torch.no_grad():
            for batch in testloader:
                x, y = batch[-2], batch[-1]
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        return torch.cat(zs).numpy(), torch.cat(ys).numpy()

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tset_eval, idx_ps), batch_size=500,
        shuffle=False)

    def eval_one(sd):
        model.load_state_dict(sd)
        model.eval()
        z_te, y_te = collect_test()
        zs = []
        with torch.no_grad():
            for _b in loader_ps:
                zs.append(model(_b[-2].to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        z_pool, z_sub = z_ps[:len(pool)], z_ps[len(pool):]
        return metrics_from_logits(z_pool[pos_of], y_pool[pos_of],
                                   z_sub, y_sub, z_te, y_te)

    # ---- noise side: archived full-data retrains ----
    noise_ms = []
    for s in [44000 + i for i in range(1, 7)]:
        p = os.path.join(EW3_DIR, 'ckpt', 'noise_s{}.pt'.format(s))
        noise_ms.append(eval_one(torch.load(p, map_location=device)))
        print(' noise', s, 'm3', round(noise_ms[-1]['m3_Floss'], 4),
              flush=True)

    # ---- honest side: fresh retrains on D minus F(100) ----
    def fit(keep_idx, seed):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                           T_max=EPOCHS)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep_idx), batch_size=100,
            shuffle=True)
        for _ep in range(EPOCHS):
            m.train()
            for _ids, x, _yy in loader:
                opt.zero_grad()
                lf(m(x.to(device)),
                   torch.as_tensor(yfull[_ids.tolist()]).to(device)
                   ).backward()
                opt.step()
            sched.step()
        m.eval()
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    rt_ms = []
    for s in RT_SEEDS_100:
        ck = os.path.join(cli.outdir, 'ckpt', 'rt100_s{}.pt'.format(s))
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        if os.path.exists(ck):
            sd = torch.load(ck, map_location=device)
        else:
            sd = fit(keep, s)
            torch.save(sd, ck)
        rt_ms.append(eval_one(sd))
        print(' rt', s, 'm3', round(rt_ms[-1]['m3_Floss'], 4), flush=True)

    keys = ('m3_Floss', 'm4_KS', 'm5_adv', 'm6_fm', 'm1_acc')
    ratios = {}
    for k in keys:
        hv = np.array([m[k] for m in rt_ms])
        nv = np.array([m[k] for m in noise_ms])
        hd = float(np.max(np.abs(hv - np.median(hv))))
        nd = float(np.max(np.abs(nv - np.median(nv))))
        ratios[k] = hd / max(nd, 1e-12)
    out = {'k': K100, 'ratios': ratios,
           'honest_m3': [round(float(m['m3_Floss']), 5) for m in rt_ms],
           'noise_m3': [round(float(m['m3_Floss']), 5) for m in noise_ms],
           'rt_seeds': list(RT_SEEDS_100)}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('RATIOS k=100:', {k: round(v, 2) for k, v in ratios.items()},
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'disp100-' + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()
