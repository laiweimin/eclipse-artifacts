"""UW E-W11: wash-footprint re-analysis (M), determinism floor (N),
per-class side effects (O).  Frozen prereg 2026-09-27-uw-ew11-*."""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        build_pool, f_deep, metrics_from_logits)
from uw.run_ew3b import group_of

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = sorted(__import__('glob').glob(
    'research_outputs/uw/ew6-2*'))[-1]
RUN10L = sorted(__import__('glob').glob(
    'research_outputs/uw/ew10L-2*'))[-1]
DET_SEED = 46300
K = 30


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--arm', default='M', choices=['M', 'N', 'O'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    epochs = 2 if cli.smoke else 40
    device = torch.device('cuda:{}'.format(cli.gpu))
    tag = 'ew11{}-smoke-'.format(cli.arm) if cli.smoke \
        else 'ew11{}-'.format(cli.arm)
    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)

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
    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(n), fset)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
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

    def retrain(seed):
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

    seeds6 = [44101, 44102, 44103, 45201, 45202, 45203]
    sd6 = {}
    for s in seeds6:
        p = (os.path.join(RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s))
             if s < 45200 else os.path.join(
                 RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)))
        sd6[s] = torch.load(p, map_location=device)
    sdw = {s: torch.load(os.path.join(
        RUN10L, 'ckpt', 'wash_s{}.pt'.format(s)),
        map_location=device) for s in seeds6}

    if cli.arm == 'M':
        rows = []
        for s in seeds6:
            num = sum(float((sdw[s][k_] - sd6[s][k_]).float().norm() ** 2)
                      for k_ in sdw[s]) ** 0.5
            den = sum(float((sd6[s][k_] - sd0[k_]).float().norm() ** 2)
                      for k_ in sdw[s]) ** 0.5
            groups = {}
            gden = {}
            for k_ in sdw[s]:
                g = group_of(k_)
                groups[g] = groups.get(g, 0.0) + float(
                    (sdw[s][k_] - sd6[s][k_]).float().norm() ** 2)
                gden[g] = gden.get(g, 0.0) + float(
                    (sd6[s][k_] - sd0[k_]).float().norm() ** 2)
            rows.append({'seed': s,
                         'delta_over_disp': num / max(den, 1e-12),
                         'group_share_delta': {
                             g: round(v / max(num ** 2, 1e-12), 3)
                             for g, v in groups.items()},
                         'group_share_disp': {
                             g: round(v / max(den ** 2, 1e-12), 3)
                             for g, v in gden.items()}})
            print('M', s, 'd/D', round(num / max(den, 1e-12), 5),
                  flush=True)
        ratios = [r['delta_over_disp'] for r in rows]
        out = {'median_delta_over_disp': float(np.median(ratios)),
               'min': float(min(ratios)), 'max': float(max(ratios)),
               'rows': rows}
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('VERDICT(M): median d/D',
              round(out['median_delta_over_disp'], 5), flush=True)
        print('saved', outdir, flush=True)
        return

    if cli.arm == 'N':
        runs = []
        for r in range(1, (2 if cli.smoke else 3) + 1):
            fn = 'det_run{}.json'.format(r)
            if os.path.exists(os.path.join(outdir, fn)):
                mm = json.load(open(os.path.join(outdir, fn)))
            else:
                sd = retrain(DET_SEED)
                model.load_state_dict(sd)
                model.eval()
                zs, ys = [], []
                with torch.no_grad():
                    for _i, x, y in testloader:
                        zs.append(model(x.to(device)).cpu())
                        ys.append(y)
                z_te = torch.cat(zs).numpy()
                y_te = torch.cat(ys).numpy()
                pos_f = np.searchsorted(pool, fset)
                lx, ly = [], []
                for _g, x, y in torch.utils.data.DataLoader(
                        torch.utils.data.Subset(
                            evalset, np.concatenate([pool, sub])),
                        batch_size=500, shuffle=False):
                    lx.append(x)
                    ly.append(y)
                with torch.no_grad():
                    model.load_state_dict(sd)
                    zp = []
                    for xb in lx:
                        zp.append(model(xb.to(device)).cpu())
                z_ps = torch.cat(zp).numpy()
                mm = metrics_from_logits(
                    z_ps[:len(pool)][pos_f], yfull[fset],
                    z_ps[len(pool):], yfull[sub], z_te, y_te)
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(mm, fh, indent=1)
            runs.append(mm)
            print('N run', r, 'm1', round(mm['m1_acc'], 5), flush=True)
        disp = {m: float(max(abs(e[m] - np.median([x[m] for x in runs]))
                             for e in runs)) for m in TIER_A}
        b_fm = float(max(abs(e['m6_fm'] - np.median(
            [x['m6_fm'] for x in runs])) for e in runs))
        fm_med = float(np.median([x['m6_fm'] for x in runs]))
        E = abs(pt3['30']['m6_fm'] - fm_med)
        out = {'per_metric_max_dev': disp, 'b_fm': b_fm,
               'gamma_floor': b_fm / max(E, 1e-12),
               'same_seed_equal': bool(len({
                   json.dumps({k: round(v, 8) for k, v in r.items()})
                   for r in runs}) == 1)}
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('VERDICT(N): gamma_floor',
              round(out['gamma_floor'], 4), 'equal',
              out['same_seed_equal'], flush=True)
        print('saved', outdir, flush=True)
        return

    # ---- Arm O: per-class side effects
    loader_te = testloader

    def per_class(sd):
        model.load_state_dict(sd)
        model.eval()
        counts = np.zeros(10)
        correct = np.zeros(10)
        with torch.no_grad():
            for _i, x, y in loader_te:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                yy = y.numpy()
                for c in range(10):
                    m = yy == c
                    counts[c] += m.sum()
                    correct[c] += (pred[m] == c).sum()
        return correct / np.maximum(counts, 1)

    pc0 = per_class(sd0)
    pc_r = {s: per_class(sd6[s]) for s in seeds6}
    pc_med = np.median(np.stack([pc_r[s] for s in seeds6]), axis=0)
    rows = []
    for s in seeds6:
        pcw = per_class(sdw[s])
        dev = np.abs(pcw - pc_med)
        noncat = np.delete(dev, CAT_CLS)
        rows.append({'seed': s,
                     'max_noncat_dev_pp': float(noncat.max() * 100),
                     'cat_dev_pp': float(dev[CAT_CLS] * 100)})
        print('O', s, 'max noncat dev pp',
              round(float(noncat.max() * 100), 3), flush=True)
    out = {'rows': rows,
           'median_max_noncat_dev_pp': float(np.median(
               [r['max_noncat_dev_pp'] for r in rows])),
           'ref_seed_noncat_dev_pp': float(np.delete(
               np.abs(pc_r[44102] - pc_med), CAT_CLS).max() * 100)}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT(O): median max noncat dev pp',
          round(out['median_max_noncat_dev_pp'], 3),
          '| honest-reference spread pp',
          round(out['ref_seed_noncat_dev_pp'], 3), flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
