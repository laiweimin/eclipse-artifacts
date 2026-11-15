"""UW E-Y2: non-oracle arm, per-record flip verification, retrain-family
replication (continuation of docs/plans/2026-09-30-uw-ey-adaptive-
conformal-attack.md, frozen extensions).

Arms:
  oracle-resave : deterministic rerun of the E-Y oracle attack with
                  final-state saving + per-record flip verification
                  (ft family, pools e3/e4, objectives O1/O2/O3).
  nonoracle     : attacker estimates centers/scales/threshold from its
                  OWN pool (NO1: 6 models of its ensemble; NO2: 12
                  models e3+e4 split 6/6), ascends under the estimated
                  constraint, and the result is judged by the TRUE
                  frozen audit. Gain vs own-pool objective-matched
                  selection, plus true-audit pass rate.
  retrain       : retrain family on F1 (fit = provider 6, cal =
                  auditor 6, q at alpha=0.20 = 6th of 6), attack from
                  auditor starts, selection from auditor pool.

Verdicts per arm, frozen: MATERIAL (any objective gain >= 0.10 on the
primary pool with true-audit pass), TRANSFER-FAIL (gain but does not
pass true audit), NULL. All results reported.
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
from uw.run_ey import (GATE, K, METRICS, MAX_ITERS, STALL_LIMIT,
                       conformal_q, gain_of, per_record_margin,
                       record_subsets)


def obj_val(margins, obj, bot=None, top=None):
    if obj == 'O1':
        return float(margins.mean())
    if obj == 'O2':
        return float(margins[bot].mean())
    return float(margins[top].mean())

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN5 = 'research_outputs/uw/ew5-20260925T160840'
RUN6 = 'research_outputs/uw/ew6-20260925T213308'
PROV = ('44101', '44102', '44103', '45201', '45202', '45203')


def mk_audit(entries_fit, entries_cal, alpha=0.20):
    import numpy as _np
    c = {m: float(_np.median([e[m] for e in entries_fit])) for m in METRICS}
    sig = {m: float(_np.median(np.abs(
        _np.array([e[m] for e in entries_fit]) - c[m]))) for m in METRICS}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in METRICS)
    q = conformal_q([sc(e) for e in entries_cal], alpha)
    return c, sig, q, sc


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
    n_starts = 2 if cli.smoke else 6

    tag = 'ey2-smoke-' if cli.smoke else 'ey2-'
    outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw', tag + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=True)   # resumable
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

    bot_t = None
    top_t = None

    def ascend(start_sd, obj, score_fn, q_lim):
        cur = start_sd
        lr = LR0
        best = obj_val(margins_of(cur), obj, bot, top)
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
                    else m[bot_t].mean() if obj == 'O2' else m[top_t].mean())
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
                    v = obj_val(margins_of(cand), obj, bot, top)
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
        return cur, best

    # ---- shared: theta0 subsets -----------------------------------------
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = margins_of(sd0)
    bot, top = record_subsets(th_m, k=10)
    bot_t = torch.as_tensor(bot, device=device)
    top_t = torch.as_tensor(top, device=device)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in ('O1', 'O2', 'O3')}

    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']

    def pool_paths(tag):
        return sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/{}_ft_s*.pt'.format(tag)))

    out = {'theta0_objs': th_objs, 'arms': {}}

    def run_pool_arm(arm, start_paths, pool_margins, audit, true_audit,
                     prefix):
        """Ascent under audit; judged by true_audit when different."""
        c_a, sig_a, q_a, sc_a = audit
        _, _, q_true, sc_true = true_audit
        res = {}
        for o in ('O1', 'O2', 'O3'):
            gains, passes, flips = [], [], []
            for j, p in enumerate(start_paths[:n_starts]):
                fn = '{}_{}_s{}.json'.format(prefix, o, j)
                if os.path.exists(os.path.join(outdir, fn)):
                    rec = json.load(open(os.path.join(outdir, fn)))
                else:
                    sd = torch.load(p, map_location=device)
                    fin, best = ascend(sd, o, sc_a, q_a)
                    mm = eval_full(fin)
                    mgs = margins_of(fin)
                    torch.save(fin, os.path.join(
                        ck, '{}_{}_s{}.pt'.format(prefix, o, j)))
                    # per-record flip accounting vs pool median
                    med_marg = np.median(
                        np.array([pm for pm in pool_margins]), axis=0)
                    atk_corr = mgs > 0
                    hon_med_corr = med_marg > 0
                    rec = {'best': best,
                           'true_s': sc_true(mm),
                           'true_pass': sc_true(mm) <= q_true,
                           'est_s': sc_a(mm),
                           'est_pass': sc_a(mm) <= q_a,
                           'm1': mm['m1_acc'], 'm2': mm['m2_catacc'],
                           'F_acc_atk': float(atk_corr.mean()),
                           'F_acc_honmed': float(hon_med_corr.mean()),
                           'F_acc_theta0': float((th_m > 0).mean()),
                           'flips_to_mem': int((atk_corr &
                                               ~hon_med_corr).sum()),
                           'flips_to_loss': int((hon_med_corr &
                                                ~atk_corr).sum())}
                    with open(os.path.join(outdir, fn), 'w') as fh:
                        json.dump(rec, fh, indent=1, default=float)
                sel = max(obj_val(pm, o, bot, top) for pm in pool_margins)
                med = float(np.median([obj_val(pm, o, bot, top)
                                       for pm in pool_margins]))
                g = gain_of(rec['best'], sel, med, th_objs[o])
                gains.append(g)
                passes.append(rec['true_pass'])
                flips.append(rec['flips_to_mem'])
                print(arm, o, j, 'best %.2f gain %.3f truepass %s '
                      'flipstomem %d' % (rec['best'], g,
                                         rec['true_pass'],
                                         rec['flips_to_mem']), flush=True)
            res[o] = {'gains': gains, 'max_gain': max(gains),
                      'true_pass_frac': float(np.mean(passes)),
                      'mean_flips_to_mem': float(np.mean(flips))}
        return res

    # ---- true frozen audit (ft family) ----------------------------------
    fit_mm = ens_mm['ew9f']
    cal_mm = [e for t in ('p1aud', 'e2') for e in ens_mm[t]]
    true_audit = mk_audit(fit_mm, cal_mm)

    e3_paths, e4_paths = pool_paths('e3'), pool_paths('e4')
    e3_mg = [margins_of(torch.load(p, map_location=device))
             for p in e3_paths]
    e4_mg = [margins_of(torch.load(p, map_location=device))
             for p in e4_paths]

    # ---- arm 1: oracle-resave (deterministic rerun + flips) ------------
    out['arms']['oracle'] = {
        'e3': run_pool_arm('oracle', e3_paths, e3_mg, true_audit,
                           true_audit, 'orc_e3'),
        'e4': run_pool_arm('oracle', e4_paths, e4_mg, true_audit,
                           true_audit, 'orc_e4')}

    # ---- arm 2: non-oracle ----------------------------------------------
    e3_mm6 = [eval_full(torch.load(p, map_location=device))
              for p in e3_paths]
    e4_mm6 = [eval_full(torch.load(p, map_location=device))
              for p in e4_paths]
    # NO1: attacker audit from own 6 (e3 as fit AND cal -> q = max)
    no1 = mk_audit(e3_mm6, e3_mm6)
    out['arms']['nonoracle_NO1'] = run_pool_arm(
        'NO1', e3_paths, e3_mg, no1, true_audit, 'no1_e3')
    # NO2: attacker audit from own 12 (fit e3, cal e4)
    no2 = mk_audit(e3_mm6, e4_mm6)
    out['arms']['nonoracle_NO2'] = run_pool_arm(
        'NO2', e3_paths, e3_mg, no2, true_audit, 'no2_e3')

    # ---- arm 3: retrain family ------------------------------------------
    prov_ck = [torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s)), map_location=device)
        for s in (44101, 44102, 44103)]
    prov_ck += [torch.load(os.path.join(
        RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
        map_location=device) for s in (45201, 45202, 45203)]
    aud_ck = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))]
    prov_mm = [eval_full(sd) for sd in prov_ck]
    aud_mm = [eval_full(sd) for sd in aud_ck]
    rt_audit = mk_audit(prov_mm, aud_mm)
    aud_mg = [margins_of(sd) for sd in aud_ck]
    out['arms']['retrain'] = run_pool_arm(
        'retrain', [sorted(glob.glob(
            'research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))[i]
            for i in range(len(aud_ck))], aud_mg, rt_audit, rt_audit,
        'rt_aud')

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
