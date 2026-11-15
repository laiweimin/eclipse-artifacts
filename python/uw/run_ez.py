"""UW E-Z: supplementary arms for the adaptive-attack paper (frozen
prereg: docs/plans/2026-10-02-uw-ez-attack-supplement-preregistration.md).

Arms:
  mob         MobileNetV2 / CIFAR-10 replication cell: trains theta0 +
              30 honest 1-ep ft models (5 ensembles x 6 seeds), builds
              the split-conformal audit (fit z1, cal z2+z3, pools
              z4/z5), oracle O1/O2/O3 + NO1/NO2 + alpha sweep + flip
              verification.  Same F1 deletion request as E-Y.
  c100        ResNet-18 / CIFAR-100 cell, identical structure
              (pool = full class 0, m2 = class-0 accuracy).
  probe       multi-subsample audit defense vs per-record O2 on cached
              CIFAR-10 checkpoints: unaware (S1-constrained, judged on
              max over S1/S2/S3) and aware attackers; honest single-
              vs multi-subsample pass rates.
  poolsize    non-oracle estimation-pool size ablation (12 / 6 / 3),
              starts and selection fixed on e3.
  retrainfix  retrain-family attack from independent starts: trains 6
              new retrains (provider's own pool), audit frozen as
              E-Y2's retrain arm (fit=PROV 6, cal=auditor 6).

All recipes inherit E-Y/E-Y2/E-Y3 unchanged (SGD lr0.01 mom0.9 wd5e-4
cosine bs100; ft = 1 ep from theta0 on keep; eval = ToTensor-only).
Smoke mode shrinks iters/starts/epochs; full runs use fresh timestamped
output roots (explicit --outdir resumes a crashed run).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

import uw.run_ew3 as ew3mod
from uw.run_ew3 import (POOL_SEED, SUB_N, SUB_SEED, build_pool, f_deep,
                        _ce_rows, _margin_rows, ks_stat, logistic_auc,
                        metrics_from_logits)
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import (GATE, K, MAX_ITERS, METRICS, STALL_LIMIT,
                       conformal_q, gain_of, per_record_margin,
                       record_subsets)
from uw.run_ey2 import mk_audit

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = 'research_outputs/uw/ew6-20260925T213308'
OBJ = ('O1', 'O2', 'O3')
ALPHAS = (0.10, 0.20, 0.30, 0.40)
N_ENS, N_FT = 5, 6
RT_SEEDS = (47301, 47302, 47303, 47304, 47305, 47306)
CELLS = {
    'mob': dict(dataset='CIFAR10', arch='mobilenet_v2', nc=10, cls=3,
                pool_scheme='build_pool', t0=47101, ens0=47111),
    'c100': dict(dataset='CIFAR100', arch='ResNet18', nc=100, cls=0,
                 pool_scheme='full_class', t0=47201, ens0=47211),
    'mobrt': dict(dataset='CIFAR10', arch='mobilenet_v2', nc=10, cls=3,
                  pool_scheme='build_pool', t0=47101, ens0=47311,
                  family='rt', n_ens=4, pool2=False, objs=('O2',),
                  t0_src='research_outputs/uw/ez1-mob-2*/ckpt/'
                         'theta0.pt'),
    'c100b': dict(dataset='CIFAR100', arch='ResNet18', nc=100, cls=0,
                  pool_scheme='full_class', t0=47401, ens0=47411),
    'k100': dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=3,
                 pool_scheme='build_pool', k=100, t0=0, ens0=47511,
                 t0_src='research_outputs/uw/ew3-20260925T113900/'
                        'ckpt/theta0.pt'),
    'calib2': dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=3,
                   pool_scheme='build_pool', t0=0, ens0=47611,
                   t0_src='research_outputs/uw/ew3-20260925T113900/'
                          'ckpt/theta0.pt'),
}
ARM_PREFIX = {'mob': 'ez1', 'c100': 'ez2', 'probe': 'ez3',
              'poolsize': 'ez4', 'retrainfix': 'ez5', 'mobrt': 'ez6',
              'c100b': 'ez7', 'k100': 'ez8', 'capc100': 'ez9',
              'calib2': 'r4a'}


def seed_all(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obj_val(margins, obj, bot, top):
    if obj == 'O1':
        return float(margins.mean())
    if obj == 'O2':
        return float(margins[bot].mean())
    return float(margins[top].mean())


def apply_delta(base_sd, delta, lr):
    return {k: (v + lr * delta[k] if k in delta else v)
            for k, v in base_sd.items()}


def ascend(start_sd, obj, score_fn, q_lim, eval_obj, grad_step, iters,
           stall_limit):
    """Feasible-direction ascent under score_fn(sd) <= q_lim."""
    cur = start_sd
    lr = LR0
    best = eval_obj(start_sd)
    stall = 0
    for _it in range(1, iters + 1):
        g = grad_step(cur, obj)
        acc = False
        for _bt in range(12):
            cand = apply_delta(cur, g, lr)
            if score_fn(cand) <= q_lim:
                v = eval_obj(cand)
                if v > best + 1e-7:
                    cur, best, acc = cand, v, True
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > stall_limit:
            break
    return cur, best


def best_gain_pass(gains, passes):
    return max((g if p else -1e9) for g, p in zip(gains, passes))


def run_attack(model, xF, yF, bot, top, margins_of, score_fn, q_lim,
               outdir, ck, prefix, start_paths, pool_margins, th_objs,
               objs=('O1', 'O2', 'O3'), n_starts=3, iters=MAX_ITERS,
               judged=None):
    """Ascent from start_paths under (score_fn, q_lim); judged(sd) ->
    (s, ok) when different from the ascent audit (non-oracle /
    probe-unaware).  score_fn/judged take state dicts.  Per-start
    records are resumable on disk."""
    import torch
    device = xF.device
    bot_t = torch.as_tensor(bot, device=device)
    top_t = torch.as_tensor(top, device=device)

    def grad_step(sd, o):
        model.load_state_dict(sd)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        m = zy - zc.max(1).values
        loss = (m.mean() if o == 'O1' else
                m[bot_t].mean() if o == 'O2' else m[top_t].mean())
        model.zero_grad()
        loss.backward()
        return {nm: p.grad.detach().clone() for nm, p in
                model.named_parameters() if p.grad is not None}

    def eval_obj_for(o):
        return lambda sd: obj_val(margins_of(sd), o, bot, top)

    med_marg = np.median(np.array(pool_margins), axis=0)
    med_corr = med_marg > 0
    res = {}
    for o in objs:
        gains, passes, est_passes, flips = [], [], [], []
        for j, p in enumerate(start_paths[:n_starts]):
            fn = '{}_{}_s{}.json'.format(prefix, o, j)
            fp = os.path.join(outdir, fn)
            if os.path.exists(fp):
                rec = json.load(open(fp))
            else:
                sd = torch.load(p, map_location=device)
                fin, best = ascend(sd, o, score_fn, q_lim,
                                   eval_obj_for(o), grad_step, iters,
                                   STALL_LIMIT)
                mgs = margins_of(fin)
                atk_corr = mgs > 0
                rec = {'best': best,
                       'est_pass': bool(score_fn(fin) <= q_lim)}
                if judged is not None:
                    s_t, ok_t = judged(fin)
                    rec['true_s'], rec['true_pass'] = float(s_t), \
                        bool(ok_t)
                rec.update({
                    'F_acc_atk': float(atk_corr.mean()),
                    'flips_to_mem': int((atk_corr & ~med_corr).sum()),
                    'flips_to_loss': int((med_corr & ~atk_corr).sum())})
                torch.save(fin, os.path.join(
                    ck, '{}_{}_s{}.pt'.format(prefix, o, j)))
                with open(fp, 'w') as fh:
                    json.dump(rec, fh, indent=1, default=float)
            sel = max(obj_val(pm, o, bot, top) for pm in pool_margins)
            med = float(np.median([obj_val(pm, o, bot, top)
                                   for pm in pool_margins]))
            g = gain_of(rec['best'], sel, med, th_objs[o])
            gains.append(g)
            passes.append(rec.get('true_pass', rec['est_pass']))
            est_passes.append(rec['est_pass'])
            flips.append(rec.get('flips_to_mem', -1))
            print(' ', prefix, o, j, 'best %.3f gain %.3f pass %s'
                  % (rec['best'], g, passes[-1]), flush=True)
        res[o] = {'gains': gains, 'max_gain': max(gains),
                  'best_gain_pass': best_gain_pass(gains, passes),
                  'pass_frac': float(np.mean(passes)),
                  'est_pass_frac': float(np.mean(est_passes)),
                  'mean_flips_to_mem': float(np.mean(flips))}
    return res


def make_context(spec, device):
    """Dataset/model/eval scaffolding shared by all arms."""
    import torch
    import torchvision.transforms as T

    tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, 4),
                    T.ToTensor()])
    ev = T.Compose([T.ToTensor()])
    if spec['dataset'] == 'CIFAR10':
        from torchvision.datasets import CIFAR10 as DS
    else:
        from torchvision.datasets import CIFAR100 as DS
    trainset = DS('data', train=True, download=False, transform=tr)
    evalset = DS('data', train=True, download=False, transform=ev)
    testset = DS('data', train=False, download=False, transform=ev)
    yfull = np.asarray(trainset.targets)
    n = len(trainset)

    ew3mod.CAT_CLS = spec['cls']
    cls_idx = np.where(yfull == spec['cls'])[0]
    if spec['pool_scheme'] == 'multi':
        tr_idx = np.where(yfull == 9)[0]
        p1 = np.sort(np.random.RandomState(43801).choice(
            cls_idx, 250, replace=False))
        p2 = np.sort(np.random.RandomState(43802).choice(
            tr_idx, 250, replace=False))
        pool = np.sort(np.concatenate([p1, p2]))
        fset = np.sort(np.concatenate([
            np.random.RandomState(44815).permutation(p1)[:15],
            np.random.RandomState(44915).permutation(p2)[:15]]))
    else:
        pool = (build_pool(cls_idx)
                if spec['pool_scheme'] == 'build_pool'
                else np.sort(cls_idx))
        fset = f_deep(pool, spec.get('k', K))
    keep = np.setdiff1d(np.arange(n), fset)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]
    pos = np.searchsorted(pool, fset)

    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(evalset, np.concatenate([pool, sub])),
        batch_size=500, shuffle=False)
    loader_te = torch.utils.data.DataLoader(testset, batch_size=500,
                                            shuffle=False)
    if spec['arch'] == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(num_classes=spec['nc']).to(device)

        def make_model():
            return mobilenet_v2(num_classes=spec['nc']).to(device)
    else:
        from tools.models import resnet18
        model = resnet18(num_classes=spec['nc']).to(device)

        def make_model():
            return resnet18(num_classes=spec['nc']).to(device)

    xF, yF = [], []
    for x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(evalset, fset), batch_size=500,
            shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()
    yF_np = yF.cpu().numpy()

    def eval_full(sd):
        model.load_state_dict(sd)
        model.eval()
        zs, ys = [], []
        with torch.no_grad():
            for x, y in loader_te:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        z_te, y_te = torch.cat(zs).numpy(), torch.cat(ys).numpy()
        zs = []
        with torch.no_grad():
            for x, _y in loader_ps:
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

    def fit_full(train_idx, seed, epochs, init=None, cosine=True,
                 lr=0.01):
        seed_all(seed)
        m = make_model()
        if init is not None:
            m.load_state_dict(init)
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs) if cosine else None)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, train_idx), batch_size=100,
            shuffle=True)
        m.train()
        for _ep in range(epochs):
            for x, yb in loader:
                opt.zero_grad()
                lf(m(x.to(device)), yb.to(device)).backward()
                opt.step()
            if sched is not None:
                sched.step()
        m.eval()
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    return dict(yfull=yfull, n=n, pool=pool, fset=fset, keep=keep,
                pos=pos, sub=sub, model=model, make_model=make_model,
                xF=xF, yF=yF, eval_full=eval_full, margins_of=margins_of,
                fit_full=fit_full, loader_te=loader_te)


def cached_train(ctx, ck, name, train_idx, seed, epochs, init=None,
                 cosine=True, eval_fn=None, outdir=None):
    """Train-or-load a checkpoint; optionally cache its metrics json."""
    import torch
    p = os.path.join(ck, name + '.pt')
    if os.path.exists(p):
        sd = torch.load(p, map_location='cpu')
    else:
        sd = ctx['fit_full'](train_idx, seed, epochs, init=init,
                             cosine=cosine)
        torch.save(sd, p)
        print(' trained', name, flush=True)
    mm = None
    if eval_fn is not None:
        fj = os.path.join(outdir, name + '_mm.json')
        if os.path.exists(fj):
            mm = json.load(open(fj))
        else:
            mm = eval_fn(sd)
            with open(fj, 'w') as fh:
                json.dump(mm, fh, indent=1, default=float)
    return sd, mm


def run_cell(cli, spec):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else MAX_ITERS
    n_starts = 1 if cli.smoke else 3
    t0_ep = 1 if cli.smoke else 40
    n_ft = 2 if cli.smoke else N_FT
    objs = ('O2',) if cli.smoke else spec.get('objs', OBJ)
    alphas = (0.20, 0.30) if cli.smoke else ALPHAS

    ctx = make_context(spec, device)
    keep = ctx['keep']
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    if 't0_src' in spec:
        import shutil
        src = sorted(glob.glob(spec['t0_src']))[-1]
        dst = os.path.join(ck, 'theta0.pt')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(' theta0 copied from', src, flush=True)

    sd0, _ = cached_train(ctx, ck, 'theta0', np.arange(ctx['n']),
                          spec['t0'], t0_ep)
    family = spec.get('family', 'ft')
    if family == 'rt':
        fam, ens_ep, ens_init, ens_cos = 'rt', (1 if cli.smoke else 40), \
            None, True
    else:
        fam, ens_ep, ens_init, ens_cos = 'ft', 1, sd0, False
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in OBJ}

    n_ens_t = spec.get('n_ens', N_ENS)
    ens_sds, ens_mm, ens_paths = {}, {}, {}
    for e in range(n_ens_t):
        tag = 'z{}'.format(e + 1)
        ens_sds[tag], ens_mm[tag], ens_paths[tag] = [], [], []
        for j in range(n_ft):
            seed = spec['ens0'] + e * 10 + j
            name = '{}_{}_s{}'.format(tag, fam, seed)
            sd, mm = cached_train(
                ctx, ck, name, keep, seed, ens_ep, init=ens_init,
                cosine=ens_cos,
                eval_fn=ctx['eval_full'], outdir=cli.outdir)
            ens_sds[tag].append(sd)
            ens_mm[tag].append(mm)
            ens_paths[tag].append(os.path.join(ck, name + '.pt'))
        print(' ensemble', tag, 'fm med %.2f'
              % float(np.median([m['m6_fm'] for m in ens_mm[tag]])),
              flush=True)

    fit_mm = ens_mm['z1']
    cal_mm = [m for t in ('z2', 'z3') for m in ens_mm[t]]
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(mm):
        return max(abs(mm[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_s = [sc(e) for e in cal_mm]
    q = {a: conformal_q(cal_s, a) for a in alphas}

    honest_all = [m for t in ens_mm if t != 'z1' for m in ens_mm[t]]
    honest_pass = {a: float(np.mean([sc(e) <= q[a]
                                     for e in honest_all]))
                   for a in alphas}
    print('audit q:', {a: round(v, 3) for a, v in q.items()},
          '| honest pass:', {a: round(v, 3)
                             for a, v in honest_pass.items()},
          flush=True)

    if spec.get('pool2', True):
        t0, t1 = ('z4', 'z3') if cli.smoke else ('z4', 'z5')
    else:
        t0, t1 = 'z4', None
    pool_mgs = {t: [ctx['margins_of'](sd) for sd in ens_sds[t]]
                for t in ((t0,) if t1 is None else (t0, t1))}

    guard = {}
    for o in OBJ:
        vals = [obj_val(mg, o, bot, top) for mg in pool_mgs[t0]]
        e_o = th_objs[o] - float(np.median(vals))
        std_o = float(np.std(vals))
        guard[o] = {'E': e_o, 'pool_std': std_o,
                    'measurable': bool(e_o > 0 and std_o <= e_o)}
    print('E-guard:', {o: (round(g['E'], 3), round(g['pool_std'], 3),
                            g['measurable'])
                       for o, g in guard.items()}, flush=True)

    out = {'spec': dict(spec, t0_epochs=t0_ep, n_ft=n_ft, iters=iters,
                        n_starts=n_starts), 'theta0_objs': th_objs,
           'q': q, 'honest_pass': honest_pass, 'E_guard': guard,
           'arms': {}}

    def arm(tag, metric_sc, q_lim, objs_, prefix, true_ms=None):
        def score_sd(sd):
            return metric_sc(ctx['eval_full'](sd))

        def judged(sd):
            s = true_ms(ctx['eval_full'](sd))
            return s, s <= q[0.20]
        return run_attack(
            ctx['model'], ctx['xF'], ctx['yF'], bot, top,
            ctx['margins_of'], score_sd, q_lim, cli.outdir, ck, prefix,
            ens_paths[tag], pool_mgs[tag], th_objs, objs=objs_,
            n_starts=n_starts, iters=iters,
            judged=(judged if true_ms is not None else None))

    out['arms']['oracle_primary'] = arm(t0, sc, q[0.20], objs, 'orc')
    if t1 is not None:
        out['arms']['oracle_repl_O2'] = arm(t1, sc, q[0.20], ('O2',),
                                            'orcR')
    if guard['O2']['measurable']:
        no1 = mk_audit(ens_mm[t0], ens_mm[t0])
        out['arms']['NO1_O2'] = arm(t0, no1[3], no1[2], ('O2',), 'no1',
                                    true_ms=sc)
        if t1 is not None:
            no2 = mk_audit(ens_mm[t0], ens_mm[t1])
            out['arms']['NO2_O2'] = arm(t0, no2[3], no2[2], ('O2',),
                                        'no2', true_ms=sc)
        else:
            out['arms']['NO2_O2'] = None
    else:
        out['arms']['NO1_O2'] = None
        if t1 is not None:
            out['arms']['NO2_O2'] = None
    out['arms']['sweep'] = {}
    for a in alphas:
        if a == 0.20:
            continue
        out['arms']['sweep'][str(a)] = arm(
            t0, sc, q[a], ('O2',), 'sw' + str(a).replace('.', ''))

    o2_gp = out['arms']['oracle_primary']['O2']['best_gain_pass']
    if t1 is not None:
        o2_repl = out['arms']['oracle_repl_O2']['O2']['best_gain_pass']
    else:
        o2_repl = o2_gp   # single-pool cell: replication waived (card)
    any_gp = max(out['arms']['oracle_primary'][o]['best_gain_pass']
                 for o in objs)
    if not cli.smoke and not guard['O2']['measurable']:
        v, why = 'NOT_MEASURABLE', 'O2 E-guard failed'
    elif o2_gp >= GATE and o2_repl >= GATE:
        v = 'MATERIAL_SINGLE_POOL' if t1 is None else 'MATERIAL'
        why = ('O2 over gate on the single pool (replication waived)'
               if t1 is None else
               'O2 over gate on primary and replication pool')
    elif any_gp >= GATE:
        v = 'CANDIDATE_UNREPLICATED'
        why = 'over gate on primary, not replicated (O2)'
    else:
        v = 'NULL'
        why = 'no objective reaches {} with pass'.format(GATE)
    out.update({'verdict': v, 'verdict_why': why})
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', why, flush=True)


def setup_c10(device):
    return make_context(
        dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=3,
             pool_scheme='build_pool'), device)


def run_probe(cli):
    import torch
    import torchvision.transforms as _tf
    from torchvision.datasets import CIFAR10
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else MAX_ITERS
    n_starts = 1 if cli.smoke else 3
    ctx = setup_c10(device)
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    pool, pos, yfull = ctx['pool'], ctx['pos'], ctx['yfull']

    evalset = CIFAR10('data', train=True, download=False,
                      transform=_tf.Compose([_tf.ToTensor()]))
    subs = [np.sort(np.random.RandomState(SUB_SEED + 100 * j).choice(
        np.setdiff1d(np.arange(ctx['n']), pool), size=SUB_N,
        replace=False)) for j in range(3)]
    loaders_ps = [torch.utils.data.DataLoader(
        torch.utils.data.Subset(evalset, np.concatenate([pool, s])),
        batch_size=500, shuffle=False) for s in subs]
    loader_te = torch.utils.data.DataLoader(
        CIFAR10('data', train=False, download=False,
                transform=_tf.Compose([_tf.ToTensor()])), batch_size=500,
        shuffle=False)
    yF_np = yfull[ctx['fset']]

    def eval_multi(sd):
        model.load_state_dict(sd)
        model.eval()
        zs, ys = [], []
        with torch.no_grad():
            for x, y in loader_te:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        z_te, y_te = torch.cat(zs).numpy(), torch.cat(ys).numpy()
        base = {'m1_acc': float((z_te.argmax(1) == y_te).mean()),
                'm2_catacc': float((z_te.argmax(1)[y_te == 3]
                                    == y_te[y_te == 3]).mean())}
        m4s, m5s = [], []
        for j, ld in enumerate(loaders_ps):
            zs = []
            with torch.no_grad():
                for x, _y in ld:
                    zs.append(model(x.to(device)).cpu())
            z_ps = torch.cat(zs).numpy()
            z_F, z_S = z_ps[:len(pool)][pos], z_ps[len(pool):]
            y_S = yfull[subs[j]]
            ce_F, ce_S = _ce_rows(z_F, yF_np), _ce_rows(z_S, y_S)
            mg_F, mg_S = _margin_rows(z_F, yF_np), _margin_rows(z_S, y_S)
            m4s.append(float(ks_stat(ce_F, ce_S)))
            m5s.append(float(logistic_auc(
                np.column_stack([ce_F, mg_F]),
                np.column_stack([ce_S, mg_S]))))
            base['m3_Floss'] = float(ce_F.mean())
            base['m6_fm'] = float(mg_F.mean())
        return {'base': base, 'm4': m4s, 'm5': m5s}

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    aud_paths = sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
    e2_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt'))
    e3_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    assert len(fit_paths) == 6 and len(aud_paths) == 6, \
        (len(fit_paths), len(aud_paths))

    def evals(paths, tag):
        out = []
        for p in paths:
            fj = os.path.join(
                cli.outdir, '{}_{}.json'.format(
                    tag, os.path.basename(p)[:-3]))
            if os.path.exists(fj):
                out.append(json.load(open(fj)))
            else:
                r = eval_multi(torch.load(p, map_location=device))
                with open(fj, 'w') as fh:
                    json.dump(r, fh, indent=1, default=float)
                print(' eval', tag, os.path.basename(p), flush=True)
                out.append(r)
        return out

    fit_e = evals(fit_paths, 'fit')
    cal_e = evals(aud_paths, 'cal') + evals(e2_paths, 'cal')

    def mk_multiaudit(fit_entries, cal_entries, alpha=0.20):
        coords = (['m1_acc', 'm2_catacc', 'm3_Floss', 'm6_fm']
                  + [('m4', j) for j in range(3)]
                  + [('m5', j) for j in range(3)])

        def getv(e, co):
            return (e['base'][co] if isinstance(co, str)
                    else e[co[0]][co[1]])

        c = {co: float(np.median([getv(e, co) for e in fit_entries]))
             for co in coords}
        sig = {co: float(np.median(np.abs(
            np.array([getv(e, co) for e in fit_entries]) - c[co])))
            for co in coords}

        def sc(e):
            return max(abs(getv(e, co) - c[co]) / max(sig[co], 1e-12)
                       for co in coords)
        q = conformal_q([sc(e) for e in cal_entries], alpha)
        return c, sig, q, sc

    def single_from_multi(e, j=0):
        return {**e['base'], 'm4_KS': e['m4'][j], 'm5_adv': e['m5'][j]}

    aud_single = mk_audit([single_from_multi(e) for e in fit_e],
                          [single_from_multi(e) for e in cal_e])
    aud_multi = mk_multiaudit(fit_e, cal_e)

    honest = cal_e + evals(e3_paths, 'hon') + evals(e4_paths, 'hon')
    out = {'sub_seeds': [SUB_SEED, SUB_SEED + 100, SUB_SEED + 200],
           'q_single': aud_single[2], 'q_multi': aud_multi[2],
           'honest_pass_single': float(np.mean(
               [aud_single[3](single_from_multi(e)) <= aud_single[2]
                for e in honest])),
           'honest_pass_multi': float(np.mean(
               [aud_multi[3](e) <= aud_multi[2] for e in honest])),
           'arms': {}}
    print('q_single %.3f (frozen ref 5.245) | q_multi %.3f | honest '
          'single %.3f multi %.3f'
          % (out['q_single'], out['q_multi'],
             out['honest_pass_single'], out['honest_pass_multi']),
          flush=True)

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in OBJ}
    e3_mg = [ctx['margins_of'](torch.load(p, map_location=device))
             for p in e3_paths]

    def sc_single(sd):
        return aud_single[3](single_from_multi(eval_multi(sd)))

    def sc_multi(sd):
        return aud_multi[3](eval_multi(sd))

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    for name, scf, qlim, judged in (
            ('unaware', sc_single, aud_single[2],
             lambda sd: (sc_multi(sd), sc_multi(sd) <= aud_multi[2])),
            ('aware', sc_multi, aud_multi[2], None)):
        out['arms'][name] = run_attack(
            model, xF, yF, bot, top, ctx['margins_of'], scf, qlim,
            cli.outdir, ck, name, e3_paths, e3_mg, th_objs,
            objs=('O2',), n_starts=n_starts, iters=iters, judged=judged)

    aw = out['arms']['aware']['O2']
    un = out['arms']['unaware']['O2']
    g_drop = (1.0 - aw['best_gain_pass']
              / max(un['best_gain_pass'], 1e-9))
    if aw['best_gain_pass'] < GATE or aw['pass_frac'] == 0.0:
        v, why = 'DEFENSE_CLOSED', 'aware attacker below gate or 0 pass'
    elif g_drop > 0.5:
        v = 'DEFENSE_PARTIAL'
        why = 'aware gain drops {:.0%} vs single-sub'.format(g_drop)
    else:
        v = 'INEFFECTIVE'
        why = 'aware gain within 50% of single-sub'
    out.update({'verdict': v, 'verdict_why': why,
                'gain_drop_frac': g_drop})
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', why, flush=True)


def run_poolsize(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else MAX_ITERS
    n_starts = 1 if cli.smoke else 3
    ctx = setup_c10(device)

    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    true_audit = mk_audit(ens_mm['ew9f'],
                          [e for t in ('p1aud', 'e2')
                           for e in ens_mm[t]])
    e3_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e3_mm, e4_mm = ens_mm['e3'], ens_mm['e4']
    assert len(e3_mm) == 6 and len(e4_mm) == 6, \
        (len(e3_mm), len(e4_mm))

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in OBJ}
    e3_mg = [ctx['margins_of'](torch.load(p, map_location=device))
             for p in e3_paths]

    def true_judged(sd):
        s = true_audit[3](ctx['eval_full'](sd))
        return s, s <= true_audit[2]

    out = {'true_q': true_audit[2], 'sizes': {}}
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    for label, nf, nc_ in (('12', 6, 6), ('10', 5, 5), ('8', 4, 4),
                           ('6', 3, 3)):
        est = mk_audit(e3_mm[:nf], e4_mm[:nc_])

        def est_sd(sd, _sc=est[3]):
            return _sc(ctx['eval_full'](sd))
        res = run_attack(
            ctx['model'], ctx['xF'], ctx['yF'], bot, top,
            ctx['margins_of'], est_sd, est[2], cli.outdir, ck,
            'no2s' + label, e3_paths, e3_mg, th_objs, objs=('O2',),
            n_starts=n_starts, iters=iters, judged=true_judged)
        out['sizes'][label] = {'est_q': est[2], 'res': res}
        print(' size', label, 'est_q %.3f true_q %.3f best_gp %.3f'
              % (est[2], true_audit[2],
                 res['O2']['best_gain_pass']), flush=True)

    g12 = out['sizes']['12']['res']['O2']['best_gain_pass']
    g6 = out['sizes']['6']['res']['O2']['best_gain_pass']
    if abs(g6 - g12) <= 0.05 and out['sizes']['6']['res']['O2'][
            'pass_frac'] > 0:
        v, why = 'ESTIMATION_ROBUST', 'size-6 within 0.05 of size-12'
    else:
        v, why = 'DEGRADED', 'size-6 departs from size-12 (report curve)'
    out.update({'verdict': v, 'verdict_why': why})
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', why, flush=True)


def run_retrainfix(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else MAX_ITERS
    n_starts = 1 if cli.smoke else 3
    rt_ep = 1 if cli.smoke else 40
    rt_seeds = RT_SEEDS[:2] if cli.smoke else RT_SEEDS
    objs = ('O2',) if cli.smoke else OBJ
    ctx = setup_c10(device)

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in OBJ}

    prov_ck = [torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s)), map_location=device)
        for s in (44101, 44102, 44103)]
    prov_ck += [torch.load(os.path.join(
        RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
        map_location=device) for s in (45201, 45202, 45203)]
    aud_ck = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))]
    prov_mm = [ctx['eval_full'](sd) for sd in prov_ck]
    aud_mm = [ctx['eval_full'](sd) for sd in aud_ck]
    audit = mk_audit(prov_mm, aud_mm)
    print('retrain audit q@0.20 = %.3f' % audit[2], flush=True)

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    new_sds, new_mm = [], []
    for s in rt_seeds:
        name = 'rtnew_s{}'.format(s)
        sd, mm = cached_train(
            ctx, ck, name, ctx['keep'], s, rt_ep,
            eval_fn=ctx['eval_full'], outdir=cli.outdir)
        new_sds.append(sd)
        new_mm.append(mm)
    new_paths = [os.path.join(ck, 'rtnew_s{}.pt'.format(s))
                 for s in rt_seeds]
    new_mg = [ctx['margins_of'](sd) for sd in new_sds]
    honest_new = float(np.mean(
        [audit[3](m) <= audit[2] for m in new_mm]))

    def audit_sd(sd):
        return audit[3](ctx['eval_full'](sd))

    res = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top,
        ctx['margins_of'], audit_sd, audit[2], cli.outdir, ck, 'rtfix',
        new_paths, new_mg, th_objs, objs=objs, n_starts=n_starts,
        iters=iters)
    best = max(res[o]['best_gain_pass'] for o in objs)
    out = {'q': audit[2], 'honest_pass_new': honest_new,
           'arms': {'independent_start': res}}
    if best >= GATE:
        v, why = 'MATERIAL', 'independent-start gain over gate with pass'
    else:
        v = 'VIOLATION_DEPENDENT'
        why = ('E-Y2 retrain reading depended on auditor-cal starts; '
               'independent starts do not reach gate')
    out.update({'verdict': v, 'verdict_why': why})
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', why, flush=True)


def run_capcheck(cli):
    """E-Z9: one-sided fm cap + conformal vs the ez7 (c100b) attack."""
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else MAX_ITERS
    n_starts = 1 if cli.smoke else 3
    ctx = make_context(CELLS['c100'], device)
    ez7 = sorted(glob.glob('research_outputs/uw/ez7-c100b-2*'))[-1]
    ck7 = os.path.join(ez7, 'ckpt')

    def ens_mm(tag):
        return [json.load(open(p)) for p in sorted(glob.glob(
            os.path.join(ez7, '{}_ft_s*_mm.json'.format(tag))))]

    fit_mm, cal_mm = ens_mm('z1'), ens_mm('z2') + ens_mm('z3')
    assert len(fit_mm) == 6 and len(cal_mm) == 12
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(mm):
        return max(abs(mm[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    q = conformal_q([sc(e) for e in cal_mm], 0.20)
    fm_max = float(max(e['m6_fm'] for e in fit_mm))
    print('ez7 audit q %.3f (ez7 ref 4.077) | fm_max %.3f'
          % (q, fm_max), flush=True)

    def score_sd(sd):
        mm = ctx['eval_full'](sd)
        return (0.0 if (sc(mm) <= q and mm['m6_fm'] <= fm_max)
                else 1.0)

    sd0 = torch.load(os.path.join(ck7, 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in OBJ}
    z4_paths = sorted(glob.glob(os.path.join(ck7, 'z4_ft_s*.pt')))
    z4_mg = [ctx['margins_of'](torch.load(p, map_location=device))
             for p in z4_paths]

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    res = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top,
        ctx['margins_of'], score_sd, 0.5, cli.outdir, ck, 'capO2',
        z4_paths, z4_mg, th_objs, objs=('O2',),
        n_starts=n_starts, iters=iters)
    best = res['O2']['best_gain_pass']
    v = 'CAP_LEAKS' if best >= GATE else 'CAP_CLOSES'
    out = {'ez7_dir': ez7, 'q': float(q), 'fm_max': fm_max,
           'uncapped_ref_gains': [0.131, 0.390, 0.868],
           'arms': res, 'verdict': v,
           'verdict_why': ('one-sided cap admits a passing O2 gain '
                           'over gate on CIFAR-100' if v == 'CAP_LEAKS'
                           else 'one-sided cap closes O2 on '
                                'CIFAR-100 too')}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', out['verdict_why'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('mob', 'c100', 'probe', 'poolsize',
                             'retrainfix', 'mobrt', 'c100b', 'k100',
                             'capc100', 'calib2'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + ARM_PREFIX[cli.arm] + \
        '-' + cli.arm + '-' + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('E-Z arm', cli.arm, '->', cli.outdir, flush=True)
    if cli.arm in CELLS:
        run_cell(cli, CELLS[cli.arm])
    elif cli.arm == 'probe':
        run_probe(cli)
    elif cli.arm == 'poolsize':
        run_poolsize(cli)
    elif cli.arm == 'capc100':
        run_capcheck(cli)
    else:
        run_retrainfix(cli)


if __name__ == '__main__':
    main()
