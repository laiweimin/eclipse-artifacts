"""UW E-W9: approximate-operator reference (F), metric ablation (G),
heterogeneity (H).  Frozen prereg docs/plans/2026-09-27-uw-ew9-*.md."""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, loo_bands, lr_update
from uw.run_ew6 import spearman

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN5 = 'research_outputs/uw/ew5-20260925T160840'
RUN6 = sorted(__import__('glob').glob(
    'research_outputs/uw/ew6-2*'))[-1]
MAX_ITERS = 60
STALL_LIMIT = 15
CONTROL_STEPS = 5
CONTROL_GAIN = 0.5
FT_SEED = lambda ki, s: 45900 + 10 * ki + s


def rank_r2(x, y):
    return spearman(x, y) ** 2


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--arm', default='F', choices=['F', 'G', 'H'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    iters = 8 if cli.smoke else MAX_ITERS
    device = torch.device('cuda:{}'.format(cli.gpu))
    tag = 'ew9{}-smoke-'.format(cli.arm) if cli.smoke \
        else 'ew9{}-'.format(cli.arm)
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
        torch.utils.data.Subset(evalset, idx_ps),
        batch_size=500, shuffle=False)

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
        ls_x, ls_y = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(evalset, fset), batch_size=500,
                shuffle=False):
            ls_x.append(x)
            ls_y.append(y)
        return torch.cat(ls_x).to(device), torch.cat(ls_y).to(
            device).long(), fset

    def ft_run(k, seed):
        """1-epoch finetune unlearning operator from theta0."""
        fset = f_deep(pool, k)
        keep = np.setdiff1d(np.arange(n), fset)
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
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

    def apply_delta(base, delta, lr):
        return {k_: (v + lr * delta[k_] if k_ in delta else v)
                for k_, v in base.items()}

    def fm_only(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    def per_record_margins(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return (zy - zc.max(1).values).cpu().numpy()

    def optimize(sd_start, k, centers, bands, seed, xF, yF,
                 tier, save_path=None):
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
                ok, _v = accept(mm, centers, bands, tier)
                if ok and mm['m6_fm'] > best + 1e-7:
                    cur, best, acc = cand, mm['m6_fm'], True
                    log.append({'iter': it, 'fm': best,
                                'metrics': mm})
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

    if cli.arm == 'F':
        out = {'units': {}}
        for ki, k in enumerate((30, 1000)[:(1 if cli.smoke else 2)]):
            xF, yF, fset = batch_F(k)
            seeds = [FT_SEED(ki, s) for s in
                     range(1, (2 if cli.smoke else 6) + 1)]
            sd_seeds, mm_seeds = {}, {}
            for s in seeds:
                fn = 'ft_k{}_s{}.json'.format(k, s)
                pth = os.path.join(ck, fn.replace('.json', '.pt'))
                if os.path.exists(os.path.join(outdir, fn)):
                    mm_seeds[s] = json.load(open(
                        os.path.join(outdir, fn)))
                    sd_seeds[s] = torch.load(pth, map_location=device)
                    continue
                sd = ft_run(k, s)
                torch.save(sd, pth)
                sd_seeds[s] = sd
                mm_seeds[s] = eval_model(sd, k)
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(mm_seeds[s], fh, indent=1)
            centers = {m: float(np.median(
                [mm_seeds[x][m] for x in seeds])) for m in TIER_A}
            bands = loo_bands(mm_seeds, centers)
            fm_med = float(np.median(
                [mm_seeds[x]['m6_fm'] for x in seeds]))
            b_fm = float(max(abs(mm_seeds[x]['m6_fm'] - fm_med)
                             for x in seeds))
            fm0 = pt3[str(k)]['m6_fm']
            i1s = []
            for s in seeds:
                fn = 'opt_k{}_s{}.json'.format(k, s)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    # control
                    cb = fm_only(sd_seeds[s], xF, yF)
                    curc, lrc, bestc = sd_seeds[s], LR0, cb
                    for _ in range(3 if cli.smoke else CONTROL_STEPS):
                        g = grad_delta(curc, xF, yF)
                        acc = False
                        for _bt in range(12):
                            cand = apply_delta(curc, g, lrc)
                            fmc = fm_only(cand, xF, yF)
                            if fmc > bestc + 1e-7:
                                curc, bestc, acc = cand, fmc, True
                                break
                            lrc *= 0.5
                            if lrc < 1e-9:
                                break
                        lrc = lr_update(lrc, acc)
                    ctrl_ok = bestc - cb >= CONTROL_GAIN
                    best, nacc = optimize(
                        sd_seeds[s], k, centers, bands[s], s, xF, yF,
                        TIER_A)
                    rec = {'I1': i1(best, mm_seeds[s]['m6_fm'], fm0)
                           if ctrl_ok else 0.0,
                           'n_accepted': nacc, 'ctrl_ok': ctrl_ok}
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1)
                i1s.append(rec['I1'])
                print('F', k, s, 'I1', round(rec['I1'], 4),
                      flush=True)
            out['units'][str(k)] = {
                'i1_star': float(max(i1s)), 'i1_per_seed': i1s,
                'gamma_ft': b_fm / max(abs(fm0 - fm_med), 1e-12),
                'leak_010': sum(1 for v in i1s if v >= 0.10),
                'leak_020': sum(1 for v in i1s if v >= 0.20)}
        passed = any(u['i1_star'] >= 0.10
                     for u in out['units'].values())
        out['verdict'] = ('APPROX_AUDIT_WASHES' if passed else
                          'APPROX_AUDIT_CLOSES')
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('VERDICT(F):', out['verdict'], flush=True)
        print('saved', outdir, flush=True)
        return

    # six-seed unit ckpts for G / H
    seeds6 = [44101, 44102, 44103, 45201, 45202, 45203]
    sd6 = {}
    for s in seeds6:
        p = (os.path.join(RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s))
             if s < 45200 else os.path.join(
                 RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)))
        sd6[s] = torch.load(p, map_location=device)
    mm6 = {s: eval_model(sd6[s], 30) for s in seeds6}
    centers = {m: float(np.median([mm6[s][m] for s in seeds6]))
               for m in TIER_A}
    bands = loo_bands(mm6, centers)
    fm0 = pt3['30']['m6_fm']
    xF, yF, fset = batch_F(30)

    if cli.arm == 'G':
        out = {'variants': {}}
        variants = ['drop_m1', 'drop_m2', 'drop_m3', 'drop_m4',
                    'drop_m5'][:(2 if cli.smoke else 5)]
        for v in variants:
            tier = tuple(m for m in TIER_A
                         if m != v.replace('drop_', ''))
            i1s = []
            for s in seeds6[:(1 if cli.smoke else 6)]:
                fn = '{}_s{}.json'.format(v, s)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    best, nacc = optimize(sd6[s], 30, centers,
                                          bands[s], s, xF, yF, tier)
                    rec = {'I1': i1(best, mm6[s]['m6_fm'], fm0),
                           'n_accepted': nacc}
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1)
                i1s.append(rec['I1'])
                print('G', v, s, 'I1', round(rec['I1'], 4),
                      flush=True)
            out['variants'][v] = {'i1_star': float(max(i1s)),
                                  'i1_per_seed': i1s}
        with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        print('saved', outdir, flush=True)
        return

    # ---- Arm H
    import glob as _g
    ew4 = sorted(_g.glob('research_outputs/uw/ew4-2*'))[-1]
    ew8d = sorted(_g.glob('research_outputs/uw/ew8D-2*'))[-1]
    v6 = json.load(open(os.path.join(RUN6, 'verdict.json')))
    cells = [(c['b_over_E'], c['i1_star']) for c in v6['law_cells']]
    e8d = json.load(open(os.path.join(ew8d, 'verdict.json')))
    for k, u in e8d['units'].items():
        cells.append((u['b_over_E'], u['i1_star']))
    rho = spearman([c[0] for c in cells], [c[1] for c in cells])
    r2 = rank_r2([c[0] for c in cells], [c[1] for c in cells])
    # residuals in rank space
    order = np.argsort([c[0] for c in cells])
    h1 = {'n_cells': len(cells), 'spearman': round(rho, 3),
          'rank_r2': round(r2, 3)}
    # H2: 3 vs 6 seed band-source ablation (four units)
    r5 = json.load(open(os.path.join(RUN5, 'results.json')))
    h2 = {}
    pairs = {'t44600_k30': '44600_k30', 't44700_k30': '44700_k30',
             't44700_k300': '44700_k300'}
    lc = {c['cell']: c for c in v6['law_cells']}
    for u6, u5 in pairs.items():
        h2[u6] = {
            'i1_star_3seed': max(x['I1'] for x in
                                 r5['armA'][u5]['per_seed']),
            'gamma_3seed': lc['armA_t' + u5]['b_over_E'],
            'i1_star_6seed': v6['units'][u6]['i1_star'],
            'gamma_6seed': v6['units'][u6]['b_over_E']}
    # 44500_k30: 3-seed baseline from E-W4 (law cell), 6-seed from ew6
    h2['t44500_k30'] = {
        'i1_star_3seed': lc['ew4_44500_k30']['i1_star'],
        'gamma_3seed': lc['ew4_44500_k30']['b_over_E'],
        'i1_star_6seed': v6['units']['t44500_k30']['i1_star'],
        'gamma_6seed': v6['units']['t44500_k30']['b_over_E']}
    # H3: leak-record identity on 44102 with checkpointed wash
    p_wash = os.path.join(ck, 'wash_44102.pt')
    best, nacc = optimize(sd6[44102], 30, centers, bands[44102],
                          44102, xF, yF, TIER_A,
                          save_path=p_wash)
    sd_w = torch.load(p_wash, map_location=device)
    mg0 = per_record_margins(sd0, xF, yF)
    mgR = per_record_margins(sd6[44102], xF, yF)
    mgW = per_record_margins(sd_w, xF, yF)
    rec_recovery = (mgW - mgR) / np.maximum(mg0 - mgR, 1e-9)
    rank0 = np.argsort(np.argsort(mg0))
    from uw.run_ew3 import ks_stat  # noqa: F401
    corr = spearman(rank0.tolist(), rec_recovery.tolist())
    h3 = {'spearman_recovery_vs_theta0_margin_rank': round(corr, 3),
          'mean_recovery_high_half': float(np.mean(
              rec_recovery[rank0 >= len(rank0) // 2])),
          'mean_recovery_low_half': float(np.mean(
              rec_recovery[rank0 < len(rank0) // 2])),
          'I1': round(i1(best, mm6[44102]['m6_fm'], fm0), 4)}
    out = {'H1_variance_explained': h1, 'H2_band_source': h2,
           'H3_leak_identity': h3}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('H1:', h1, flush=True)
    print('H3:', h3, flush=True)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
