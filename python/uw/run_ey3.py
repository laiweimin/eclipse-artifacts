"""UW E-Y3: coverage-security tradeoff curve + multi-unit replication.

Arm A (tradeoff): ft family, frozen fixed split (fit=ew9f, cal=p1aud+e2).
Sweep alpha in {0.05,0.10,0.20,0.30,0.40}: threshold q(alpha) from the
same cal scores; honest pass rate over the 24 non-fit models; O2 attack
(3 starts, e3 pool) under each alpha; report gain + pass. The curve
quantifies: looser threshold -> better honest coverage, more attack room.

Arm B (multi-unit, retrain family, k=30 same F): units t44600 and
t44700. Six honest retrains each (ew5 3 + ew6 3); audit = fit all 6,
q at alpha=0.20 is the 6/6 order (max, as E-Y2's retrain arm); O2
ascent from 3 starts; selection from the 6. theta0 subsets per unit.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED,
                        build_pool, f_deep, metrics_from_logits)
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import (K, METRICS, MAX_ITERS, STALL_LIMIT, conformal_q,
                       gain_of, per_record_margin, record_subsets)
from uw.run_ey2 import mk_audit

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN5 = 'research_outputs/uw/ew5-20260925T160840'
RUN6 = 'research_outputs/uw/ew6-20260925T213308'
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40)


def obj_val(margins, obj, bot, top):
    if obj == 'O1':
        return float(margins.mean())
    if obj == 'O2':
        return float(margins[bot].mean())
    return float(margins[top].mean())


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
    iters = 4 if cli.smoke else MAX_ITERS
    n_starts = 2 if cli.smoke else 3

    tag = 'ey3-smoke-' if cli.smoke else 'ey3-'
    outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw', tag + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
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

    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset),
            batch_size=500, shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()
    yF_np = yF.cpu().numpy()

    def eval_full(sd):
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

    def margins_of(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF).cpu().numpy()
        return per_record_margin(z, yF_np)

    def ascend(start_sd, obj, bot_t, top_t, score_fn, q_lim):
        cur = start_sd
        lr = LR0
        best = obj_val(margins_of(cur), obj,
                       bot_t.cpu().numpy(), top_t.cpu().numpy())
        stall = 0
        for it in range(1, iters + 1):
            model.load_state_dict(cur)
            model.eval()
            z = model(xF)
            zy = z.gather(1, yF[:, None]).squeeze(1)
            zc = z.clone()
            zc.scatter_(1, yF[:, None], -1e18)
            m = zy - zc.max(1).values
            loss = (m.mean() if obj == 'O1'
                    else m[bot_t].mean() if obj == 'O2'
                    else m[top_t].mean())
            model.zero_grad()
            loss.backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(12):
                cand = {k2: (v + lr * g[k2] if k2 in g else v)
                        for k2, v in cur.items()}
                mm = eval_full(cand)
                if score_fn(mm) <= q_lim:
                    v = obj_val(margins_of(cand), obj,
                                bot_t.cpu().numpy(), top_t.cpu().numpy())
                    if v > best + 1e-7:
                        cur, best, acc = cand, v, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return best, eval_full(cur)

    # ================= Arm A: tradeoff curve (ft family) ================
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    fit_mm = ens_mm['ew9f']
    cal_mm = [e for t in ('p1aud', 'e2') for e in ens_mm[t]]
    c, sig, _, sc = mk_audit(fit_mm, cal_mm)
    cal_s = [sc(e) for e in cal_mm]

    honest_all = [e for t in ('p1aud', 'e2', 'e3', 'e4')
                  for e in ens_mm[t]]            # 24 non-fit models
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = margins_of(sd0)
    bot, top = record_subsets(th_m, k=10)
    bot_t = torch.as_tensor(bot, device=device)
    top_t = torch.as_tensor(top, device=device)
    th_O2 = obj_val(th_m, 'O2', bot, top)

    e3_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e3_mg = [margins_of(torch.load(p, map_location=device))
             for p in e3_paths]
    sel_O2 = max(obj_val(mg, 'O2', bot, top) for mg in e3_mg)
    med_O2 = float(np.median([obj_val(mg, 'O2', bot, top)
                              for mg in e3_mg]))

    curve = []
    for alpha in ALPHAS[:2 if cli.smoke else 5]:
        q = conformal_q(cal_s, alpha)
        hon_pass = float(np.mean([sc(e) <= q for e in honest_all]))
        gains, passes = [], []
        for j, p in enumerate(e3_paths[:n_starts]):
            fn = 'curve_a{}_s{}.json'.format(alpha, j)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, mm = ascend(torch.load(p, map_location=device),
                                  'O2', bot_t, top_t, sc, q)
                rec = {'best': best, 'pass': sc(mm) <= q}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1, default=float)
            gains.append(gain_of(rec['best'], sel_O2, med_O2, th_O2))
            passes.append(rec['pass'])
        curve.append({'alpha': alpha, 'q': float(q),
                      'honest_pass': hon_pass,
                      'max_gain': max(gains),
                      'pass_frac': float(np.mean(passes))})
        print('curve a=%.2f q=%.2f honPass %.2f gain %.3f pass %.2f'
              % (alpha, q, hon_pass, max(gains), np.mean(passes)),
              flush=True)

    # ================= Arm B: multi-unit (retrain, k=30) ================
    units = {
        't44600': dict(paths=[os.path.join(
            RUN5, 'ckpt', 'armA_t44600_k30_s{}.pt'.format(s))
            for s in (45001, 45002, 45003)]
            + [os.path.join(RUN6, 'ckpt',
                            'u1_t44600_k30_s{}.pt'.format(s))
               for s in (45211, 45212, 45213)],
            th0=os.path.join(RUN5, 'ckpt', 'theta0_s44600.pt')),
        't44700': dict(paths=[os.path.join(
            RUN5, 'ckpt', 'armA_t44700_k30_s{}.pt'.format(s))
            for s in (45101, 45102, 45103)]
            + [os.path.join(RUN6, 'ckpt',
                            'u2_t44700_k30_s{}.pt'.format(s))
               for s in (45221, 45222, 45223)],
            th0=os.path.join(RUN5, 'ckpt', 'theta0_s44700.pt')),
    }
    multi = {}
    for uname, u in units.items():
        mms = [eval_full(torch.load(p, map_location=device))
               for p in u['paths']]
        c_u, sig_u, q_u, sc_u = mk_audit(mms, mms)     # 6/6 order
        mgs = [margins_of(torch.load(p, map_location=device))
               for p in u['paths']]
        th_u = margins_of(torch.load(u['th0'], map_location=device))
        b_u, t_u = record_subsets(th_u, k=10)
        bt = torch.as_tensor(b_u, device=device)
        tt = torch.as_tensor(t_u, device=device)
        th_O2u = obj_val(th_u, 'O2', b_u, t_u)
        sel = max(obj_val(mg, 'O2', b_u, t_u) for mg in mgs)
        med = float(np.median([obj_val(mg, 'O2', b_u, t_u)
                               for mg in mgs]))
        gains, passes = [], []
        for j, p in enumerate(u['paths'][:n_starts]):
            fn = 'multi_{}_s{}.json'.format(uname, j)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, mm = ascend(torch.load(p, map_location=device),
                                  'O2', bt, tt, sc_u, q_u)
                rec = {'best': best, 'pass': sc_u(mm) <= q_u}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1, default=float)
            gains.append(gain_of(rec['best'], sel, med, th_O2u))
            passes.append(rec['pass'])
        multi[uname] = {'max_gain': max(gains), 'gains': gains,
                        'pass_frac': float(np.mean(passes))}
        print('multi', uname, 'O2 max_gain %.3f pass %.2f'
              % (max(gains), np.mean(passes)), flush=True)

    out = {'curve': curve, 'multi': multi,
           'sel_O2': sel_O2, 'med_O2': med_O2, 'th_O2': th_O2}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
