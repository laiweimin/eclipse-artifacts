"""UW R3-P3: defense-aware optimization + multi-subsample defense FPR
(frozen prereg docs/plans/2026-09-23-uw-r3-minimal-loop.md,
author-approved).

Part 1 (defense cost): 12 honest models (provider 6 + P1 auditor 6)
evaluated on three probe subsamples (S1 calibration 52000, S2 ew12's
46400, S3 fresh 53001). Per-ensemble-per-subsample bands (max |dev|
from median over that ensemble on that subsample); rejection of honest
models under single-subsample and any-of-subsample rules = the
multi-subsample defense's false-positive cost.

Part 2 (defense-aware attacker): washes optimized from the 6 provider
retrains requiring feasibility on m1/m2/m3 bands plus m4/m5 bands for
EACH of S1/S2/S3 simultaneously. Held-out evaluation on S4 (53002);
compared with the S1-only (unaware) washes regenerated under provider
bands.

Verdict (frozen): S4 pass rate of defense-aware washes >= unaware
S4 pass rate + 2/6 -> DEFENSE_CIRCUMVENTED; <= unaware -> DEFENSE_ROBUST
(at these settings); else PARTIAL.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, lr_update
from uw.run_r3_p1 import (PROV_SEEDS, mk_bands_centers, RUN3, RUN6)

K = 30
SUB_SEEDS = {'s1': SUB_SEED, 's2': 46400, 's3': 53001, 's4': 53002}
TRAIN_SUBS = ('s1', 's2', 's3')
HOLD_SUB = 's4'
MAX_ITERS = 60
STALL_LIMIT = 15


def accept_multi(mm, centers, bands):
    """m1/m2/m3 once + per-sub (m4, m5) against the same centers/bands
    (per-subsample centers/bands are folded in by the caller)."""
    viol = [m for m in ('m1_acc', 'm2_catacc', 'm3_Floss')
            if abs(mm[m] - centers[m]) > bands[m]]
    for tag in TRAIN_SUBS:
        m4, m5 = mm['per_sub'][tag]
        if abs(m4 - centers['m4_KS']) > bands['m4_KS']:
            viol.append('{}:m4_KS'.format(tag))
        if abs(m5 - centers['m5_adv']) > bands['m5_adv']:
            viol.append('{}:m5_adv'.format(tag))
    return (not viol), viol


def joint_reject(per_sub_flags):
    return any(per_sub_flags.values())


def verdict_p3(da_pass, un_pass):
    if da_pass >= un_pass + 2:
        return 'DEFENSE_CIRCUMVENTED', (
            'defense-aware washes pass held-out {}/6 vs unaware {}/6'
            .format(da_pass, un_pass))
    if da_pass <= un_pass:
        return 'DEFENSE_ROBUST', (
            'defense-aware washes pass held-out {}/6 <= unaware {}/6'
            .format(da_pass, un_pass))
    return 'PARTIAL', ('{} vs {} of 6 on held-out'.format(da_pass,
                                                          un_pass))


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
    ap.add_argument('--r3p1', default=None,
                    help='r3p1 output dir for auditor checkpoints')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    iters = 6 if cli.smoke else MAX_ITERS
    n_da_seeds = 2 if cli.smoke else 6

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'r3p3-smoke-' if cli.smoke else 'r3p3-'
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)
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
    pos = np.searchsorted(pool, fset)
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']

    subs = {}
    for tag, sd in SUB_SEEDS.items():
        subs[tag] = np.sort(np.random.RandomState(sd).choice(
            np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    model = get_model('ResNet18').to(device)

    y_te = None

    def _test_y():
        nonlocal y_te
        if y_te is None:
            ys = []
            for _id, x, y in testloader:
                ys.append(y)
            y_te = torch.cat(ys).numpy()
        return y_te

    def loader_of(idx):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, idx), batch_size=500,
            shuffle=False)

    pool_loader = loader_of(pool)
    sub_loaders = {t: loader_of(subs[t]) for t in SUB_SEEDS}
    xF, yF = [], []
    for _g, x, y in loader_of(fset):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()

    def _forward(loader):
        zs = []
        model.eval()
        with torch.no_grad():
            for _g, x, _y in loader:
                zs.append(model(x.to(device)).cpu())
        return torch.cat(zs).numpy()

    def eval_full(sd):
        """All-aux metrics: m1/m2/m3/m6 on test+F plus per-sub m4/m5."""
        model.load_state_dict(sd)
        z_te = _forward(testloader)
        y_te = _test_y()
        z_pool = _forward(pool_loader)
        z_F = z_pool[pos]
        per_sub = {}
        for t in SUB_SEEDS:
            z_sub = _forward(sub_loaders[t])
            mm = metrics_from_logits(z_F, yfull[fset], z_sub,
                                     yfull[subs[t]], z_te, y_te)
            per_sub[t] = (mm['m4_KS'], mm['m5_adv'])
        head = metrics_from_logits(z_F, yfull[fset],
                                   _forward(sub_loaders['s1']),
                                   yfull[subs['s1']], z_te, y_te)
        return {'m1_acc': head['m1_acc'], 'm2_catacc': head['m2_catacc'],
                'm3_Floss': head['m3_Floss'], 'm6_fm': head['m6_fm'],
                'util_margin': head['util_margin'],
                'cat_margin': head['cat_margin'],
                'per_sub': per_sub}

    # ---- ensembles -----------------------------------------------------
    def ckpt_of(s):
        if s in (44101, 44102, 44103):
            return os.path.join(RUN3, 'ckpt',
                                'rt_k{}_s{}.pt'.format(K, s))
        return os.path.join(RUN6, 'ckpt',
                            'u0_t44500_k30_s{}.pt'.format(s))

    prov_ck = {s: torch.load(ckpt_of(s), map_location=device)
               for s in PROV_SEEDS}
    prov_full = {s: eval_full(sd) for s, sd in prov_ck.items()}
    print('provider full evals done', flush=True)

    r3p1 = cli.r3p1 or sorted(glob.glob(
        'research_outputs/uw/r3p1-2*'))[-1]
    aud_ck = {}
    for p in sorted(glob.glob(os.path.join(r3p1, 'ckpt',
                                           'aud_s*.pt'))):
        s = int(os.path.basename(p)[6:-3])
        aud_ck[s] = torch.load(p, map_location=device)
    if cli.smoke and not aud_ck:
        aud_ck = dict(list(prov_ck.items())[:2])   # plumbing only
    aud_full = {s: eval_full(sd) for s, sd in aud_ck.items()}
    print('auditor full evals done ({} models)'.format(len(aud_ck)),
          flush=True)

    def flat(entries, key, sub=None):
        if sub is None:
            return float(np.median([e[key] for e in entries.values()])), \
                float(max(abs(e[key] -
                              np.median([q[key] for q in
                                         entries.values()]))
                          for e in entries.values()))
        vals = [e['per_sub'][sub] for e in entries.values()]
        idx = 0 if key == 'm4_KS' else 1
        v = [x[idx] for x in vals]
        return float(np.median(v)), float(max(abs(x - np.median(v))
                                              for x in v))

    out = {'r3p1_dir': r3p1, 'sub_seeds': SUB_SEEDS}

    # ---- part 1: multi-subsample defense FPR ---------------------------
    part1 = {}
    for ens_name, ens in (('provider', prov_full), ('auditor',
                                                    aud_full)):
        rows = {}
        for t in ('s1', 's2', 's3'):
            c4, b4 = flat(ens, 'm4_KS', t)
            c5, b5 = flat(ens, 'm5_adv', t)
            rows[t] = (c4, b4, c5, b5)
        part1[ens_name] = {'bands': rows}
        # FPR of the OTHER ensemble on this ensemble's bands (and self)
        for eval_name, ev in (('provider', prov_full),
                              ('auditor', aud_full)):
            rej_single = {t: [] for t in ('s1', 's2', 's3')}
            for s, e in ev.items():
                for t in ('s1', 's2', 's3'):
                    c4, b4, c5, b5 = rows[t]
                    m4, m5 = e['per_sub'][t]
                    rej_single[t].append(
                        abs(m4 - c4) > b4 or abs(m5 - c5) > b5)
            joint = [any(rej_single[t][i] for t in ('s1', 's2', 's3'))
                     for i in range(len(ev))]
            part1[ens_name]['fpr_' + eval_name] = {
                t: float(np.mean(rej_single[t]))
                for t in ('s1', 's2', 's3')}
            part1[ens_name]['fpr_' + eval_name]['any3'] = \
                float(np.mean(joint))
    out['part1'] = part1
    print('part1 FPR:', json.dumps({k: v.get('fpr_auditor',
                                              v.get('fpr_provider'))
                                    for k, v in part1.items()})[:300],
          flush=True)

    # ---- part 2: defense-aware washes ----------------------------------
    p_mm = {s: json.load(open(os.path.join(
        RUN3, 'rt_k{}_s{}.json'.format(K, s))))['per_k'][str(K)]
        if s in (44101, 44102, 44103) else json.load(open(os.path.join(
            RUN6, 'u0_t44500_k30_s{}.json'.format(s))))
        for s in PROV_SEEDS}
    p_centers, _ = mk_bands_centers(p_mm)
    p_band123 = {m: max(abs(q[m] - p_centers[m]) for q in p_mm.values())
                 for m in ('m1_acc', 'm2_catacc', 'm3_Floss')}
    # per-sub bands from provider ensemble on each TRAIN subsample
    psb = {}
    for t in TRAIN_SUBS:
        c4, b4 = flat(prov_full, 'm4_KS', t)
        c5, b5 = flat(prov_full, 'm5_adv', t)
        psb[t] = (c4, b4, c5, b5)
    hold = {}
    c4, b4 = flat(prov_full, 'm4_KS', HOLD_SUB)
    c5, b5 = flat(prov_full, 'm5_adv', HOLD_SUB)
    hold = (c4, b4, c5, b5)

    def da_feasible(mm):
        viol = [m for m in ('m1_acc', 'm2_catacc', 'm3_Floss')
                if abs(mm[m] - p_centers[m]) > p_band123[m]]
        for t in TRAIN_SUBS:
            m4, m5 = mm['per_sub'][t]
            _, b4, _, b5 = psb[t]
            if abs(m4 - psb[t][0]) > b4:
                viol.append('{}:m4'.format(t))
            if abs(m5 - psb[t][2]) > b5:
                viol.append('{}:m5'.format(t))
        return (not viol), viol

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

    def fm_only(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    def run_wash(s, feasible):
        """Deterministic wash from provider retrain s under the given
        feasibility predicate; returns (final_sd, final_mm, n_acc)."""
        cur = prov_ck[s]
        lr = LR0
        best_fm = prov_full[s]['m6_fm']
        n_acc = 0
        stall = 0
        for it in range(1, iters + 1):
            g = grad_delta(cur)
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                mm = eval_full(cand)
                ok, _v = feasible(mm)
                if ok and mm['m6_fm'] > best_fm + 1e-7:
                    cur, best_fm, acc = cand, mm['m6_fm'], True
                    n_acc += 1
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return cur, eval_full(cur), n_acc

    da = {}
    for s in PROV_SEEDS[:n_da_seeds]:
        fn = 'da_s{}.json'.format(s)
        p_j = os.path.join(outdir, fn)
        if os.path.exists(p_j):
            da[s] = json.load(open(p_j))
            continue
        cur, final, n_acc = run_wash(s, da_feasible)
        torch.save(cur, os.path.join(outdir, 'ckpt_da_s{}.pt'.format(s)))
        h4, h5 = final['per_sub'][HOLD_SUB]
        hold_ok = (abs(h4 - hold[0]) <= hold[1]
                   and abs(h5 - hold[2]) <= hold[3])
        da[s] = {'I1': i1(final['m6_fm'], prov_full[s]['m6_fm'], fm0),
                 'fm': final['m6_fm'], 'hold_ok': hold_ok,
                 'final': {k: v for k, v in final.items()
                           if k != 'per_sub'},
                 'n_acc': n_acc}
        with open(p_j, 'w') as fh:
            json.dump(da[s], fh, indent=1)
        print('DA seed', s, 'I1', round(da[s]['I1'], 4), 'hold_ok',
              hold_ok, flush=True)

    # unaware comparison: S1-only washes (the paper's original attack)
    # regenerated here (deterministic) and evaluated on the S4 hold.
    def s1_feasible(mm):
        viol = [m for m in ('m1_acc', 'm2_catacc', 'm3_Floss')
                if abs(mm[m] - p_centers[m]) > p_band123[m]]
        m4, m5 = mm['per_sub']['s1']
        _, b4, _, b5 = psb['s1']
        if abs(m4 - psb['s1'][0]) > b4:
            viol.append('s1:m4')
        if abs(m5 - psb['s1'][2]) > b5:
            viol.append('s1:m5')
        return (not viol), viol

    un = {}
    for s in PROV_SEEDS[:n_da_seeds]:
        fn = 'unaware_s{}.json'.format(s)
        p_j = os.path.join(outdir, fn)
        if os.path.exists(p_j):
            un[s] = json.load(open(p_j))
            continue
        cur, final, n_acc = run_wash(s, s1_feasible)
        h4, h5 = final['per_sub'][HOLD_SUB]
        hold_ok = (abs(h4 - hold[0]) <= hold[1]
                   and abs(h5 - hold[2]) <= hold[3])
        un[s] = {'I1': i1(final['m6_fm'], prov_full[s]['m6_fm'], fm0),
                 'hold_ok': hold_ok}
        with open(p_j, 'w') as fh:
            json.dump(un[s], fh, indent=1)
        print('unaware seed', s, 'I1', round(un[s]['I1'], 4),
              'hold_ok', hold_ok, flush=True)

    da_pass = sum(1 for v in da.values() if v['hold_ok'])
    un_pass = sum(1 for v in un.values() if v['hold_ok'])
    v, why = verdict_p3(da_pass, un_pass)
    out.update({'part2': {'da': {str(k): v for k, v in da.items()},
                          'unaware': {str(k): v for k, v in un.items()},
                          'da_pass_hold': da_pass,
                          'unaware_pass_hold': un_pass,
                          'verdict': v, 'verdict_why': why}})
    print('P3 VERDICT:', v, '|', why, flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
