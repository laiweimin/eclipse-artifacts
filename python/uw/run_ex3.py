"""UW E-X3: ft-ensemble redraws deciding whether the provider-known ft
gain (+0.086/+0.158, ew9F) is real-but-unstable or a sampling artifact
(frozen prereg docs/plans/2026-09-28-uw-ex-multi-config-preregistration.md).

Three fresh 1-epoch ft honest ensembles (6 seeds each) on k=30 theta0-A,
each with its own centers/bands and 6 standard washes; per-ensemble gain
= best wash elevation over that ensemble's median (deletion-delta units)
minus its own honest luck ceiling. Combined with the two existing draws
(ew9F provider ensemble: gain +0.086 from R2 analysis; P1 auditor
ensemble: gain -0.066) = 5 independent draws, full distribution reported.

Verdict (frozen): >=2/5 draws with gain >= 0.10 -> UNSTABLE_BUT_REAL;
<=1/5 -> SAMPLING_ARTIFACT. All seeds reported.
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
K = 30
ENSEMBLES = {'e2': (46801, 46802, 46803, 46804, 46805, 46806),
             'e3': (46811, 46812, 46813, 46814, 46815, 46816),
             'e4': (46821, 46822, 46823, 46824, 46825, 46826)}
EXISTING_GAINS = {'ew9F_e1': 0.086, 'p1_auditor': -0.066}   # R2/P1 records
MATERIAL = 0.10
MAX_ITERS = 60
STALL_LIMIT = 15


def verdict_ex3(gains):
    n_mat = sum(1 for g in gains if g >= MATERIAL)
    if n_mat >= 2:
        return 'UNSTABLE_BUT_REAL', (
            '{}/{} draws >= 0.10 ({}); report as real but ensemble-'
            'dependent'.format(n_mat, len(gains),
                               [round(g, 3) for g in gains]))
    return 'SAMPLING_ARTIFACT', (
        '{}/{} draws >= 0.10 ({}); the provider-known ft gain does not '
        'replicate across ensembles'.format(n_mat, len(gains),
                                            [round(g, 3) for g in gains]))


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
    iters = 8 if cli.smoke else MAX_ITERS
    ensembles = {'e2': ENSEMBLES['e2'][:1]} if cli.smoke else ENSEMBLES

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ex3-smoke-' if cli.smoke else 'ex3-'
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
    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(n), fset)
    pos = np.searchsorted(pool, fset)
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)

    model = get_model('ResNet18').to(device)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    y_t = torch.as_tensor(yfull, dtype=torch.long)

    def ft_run(seed):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        m.train()
        for _ids, x, _yy in loader:
            opt.zero_grad()
            lf(m(x.to(device)), y_t[_ids.tolist()].to(device)).backward()
            opt.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)

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

    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset), batch_size=500,
            shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()

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

    out = {'ensembles': {}}
    for tag_, seeds in ensembles.items():
        ft_mm, ft_ck = {}, {}
        for s in seeds:
            pth = os.path.join(ck, '{}_ft_s{}.pt'.format(tag_, s))
            fnj = '{}_ft_s{}.json'.format(tag_, s)
            if os.path.exists(os.path.join(outdir, fnj)):
                ft_mm[s] = json.load(open(os.path.join(outdir, fnj)))
                ft_ck[s] = torch.load(pth, map_location=device)
                continue
            sd = ft_run(s)
            torch.save(sd, pth)
            ft_ck[s] = sd
            ft_mm[s] = eval_model(sd)
            with open(os.path.join(outdir, fnj), 'w') as fh:
                json.dump(ft_mm[s], fh, indent=1)
            print(tag_, 'ft', s, 'fm', round(ft_mm[s]['m6_fm'], 3),
                  flush=True)
        med = {m: float(np.median([e[m] for e in ft_mm.values()]))
               for m in TIER_A}
        bands = loo_bands(ft_mm, med)
        band = bands[next(iter(bands))]
        fm_med = float(np.median([e['m6_fm'] for e in ft_mm.values()]))
        E = fm0 - fm_med
        b_fm = float(max(abs(e['m6_fm'] - fm_med)
                         for e in ft_mm.values()))
        luck = (max(e['m6_fm'] for e in ft_mm.values()) - fm_med) / E
        rows = []
        for s in seeds:
            fnj = '{}_wash_s{}.json'.format(tag_, s)
            if os.path.exists(os.path.join(outdir, fnj)):
                rec = json.load(open(os.path.join(outdir, fnj)))
            else:
                cur, lr, best_fm = ft_ck[s], LR0, ft_mm[s]['m6_fm']
                stall = 0
                for it in range(1, iters + 1):
                    g = grad_delta(cur)
                    acc = False
                    for _bt in range(12):
                        cand = apply_delta(cur, g, lr)
                        mm = eval_model(cand)
                        ok, _v = accept(mm, med, band, TIER_A)
                        if ok and mm['m6_fm'] > best_fm + 1e-7:
                            cur, best_fm, acc = cand, mm['m6_fm'], True
                            break
                        lr *= 0.5
                        if lr < 1e-9:
                            break
                    lr = lr_update(lr, acc)
                    stall = 0 if acc else stall + 1
                    if stall > STALL_LIMIT:
                        break
                final = eval_model(cur)
                torch.save(cur, os.path.join(
                    ck, '{}_wash_s{}.pt'.format(tag_, s)))
                rec = {'I1': i1(final['m6_fm'], ft_mm[s]['m6_fm'], fm0),
                       'fm': final['m6_fm'], 'final_mm': final}
                with open(os.path.join(outdir, fnj), 'w') as fh:
                    json.dump(rec, fh, indent=1)
            rows.append({'seed': s, 'I1': rec['I1'],
                         'elev': (rec['fm'] - fm_med) / E})
            print(tag_, 'wash', s, 'I1', round(rec['I1'], 4), flush=True)
        best_elev = max(r['elev'] for r in rows)
        out['ensembles'][tag_] = {
            'gamma_ft': b_fm / max(abs(E), 1e-12), 'luck': luck,
            'best_elev': best_elev, 'gain': best_elev - luck,
            'rows': rows,
            'honest_mm': {str(s): ft_mm[s] for s in seeds}}
        print(tag_, 'gamma', round(out['ensembles'][tag_]['gamma_ft'], 3),
              'luck', round(luck, 3), 'gain',
              round(out['ensembles'][tag_]['gain'], 3), flush=True)

    gains = [v['gain'] for v in out['ensembles'].values()] \
        + list(EXISTING_GAINS.values())
    v, why = verdict_ex3(gains)
    out.update({'existing_gains': EXISTING_GAINS, 'all_gains':
                {k: round(g, 4) for k, g in zip(
                    list(out['ensembles']) + list(EXISTING_GAINS), gains)},
                'verdict': v, 'verdict_why': why})
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('EX3 VERDICT:', v, '|', why, flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
