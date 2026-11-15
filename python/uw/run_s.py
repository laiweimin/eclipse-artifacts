"""UW E-S: capability-restoration arms (frozen prereg
docs/plans/2026-10-03-uw-s-capability-restoration-preregistration.md).

  s1 : pooled-estimator submission -- attacker c/sigma pooled over its
       own 12 (e3+e4, both ensembles), in-sample conformal q at 0.20;
       O2 ascent under the conservative estimate, judged by the true
       audit; commit rules (best-O2, random); bootstrap-draw
       robustness (3 resampled estimators x 3 starts).
  s2 : last-layer ascent + F-ft on the two cells where class bias
       already fails (CIFAR-100 ez7 unit; retrain family ez5), vs
       the full-model attack readings (0.87 / 0.74).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import GATE, conformal_q, gain_of, record_subsets
from uw.run_ez import (CELLS, apply_delta, make_context, obj_val,
                       setup_c10)
from uw.run_ey2 import mk_audit
from uw.run_r1 import CACHE, RUN3, load_audit
from uw.run_r2 import theta_ctx
from uw.run_ew3b import LR0, lr_update

RUN6 = 'research_outputs/uw/ew6-20260925T213308'
M = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')


def pooled_est(entries, alpha=0.20):
    c = {m: float(np.median([e[m] for e in entries])) for m in M}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in entries]) - c[m]))) for m in M}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in M)
    q = conformal_q([sc(e) for e in entries], alpha)
    return sc, float(q)


def ascend_under(ctx, start_sd, score_sd, q_lim, bot, top, iters):
    """Local feasible-direction ascent (full model) under score_sd."""
    import torch
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    device = xF.device
    bot_t = torch.as_tensor(bot, device=device)
    cur, lr = start_sd, LR0
    best = obj_val(ctx['margins_of'](cur), 'O2', bot, top)
    stall = 0
    for _it in range(iters):
        model.load_state_dict(cur)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        loss = (zy - zc.max(1).values)[bot_t].mean()
        model.zero_grad()
        loss.backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(12):
            cand = apply_delta(cur, g, lr)
            if score_sd(cand) <= q_lim:
                v = obj_val(ctx['margins_of'](cand), 'O2', bot, top)
                if v > best + 1e-7:
                    cur, best, acc = cand, v, True
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 15:
            break
    return cur, best


def ll_ascend_under(ctx, start_sd, score_sd, q_lim, bot, top, iters):
    """Last-layer-only variant of ascend_under."""
    sd = {k: v.clone() for k, v in start_sd.items()}
    out, best = ascend_under_ll(ctx, sd, score_sd, q_lim, bot, top,
                                iters)
    return out, best


def ascend_under_ll(ctx, sd, score_sd, q_lim, bot, top, iters):
    import torch
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    device = xF.device
    bot_t = torch.as_tensor(bot, device=device)
    cur, lr = sd, LR0
    best = obj_val(ctx['margins_of'](cur), 'O2', bot, top)
    stall = 0
    for _it in range(iters):
        model.load_state_dict(cur)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        loss = (zy - zc.max(1).values)[bot_t].mean()
        model.zero_grad()
        loss.backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters()
             if p.grad is not None and nm.startswith('fc.')}
        acc = False
        for _bt in range(12):
            cand = apply_delta(cur, g, lr)
            if score_sd(cand) <= q_lim:
                v = obj_val(ctx['margins_of'](cand), 'O2', bot, top)
                if v > best + 1e-7:
                    cur, best, acc = cand, v, True
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 15:
            break
    return cur, best


def run_s1(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc_true, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    ens = json.load(open(CACHE))['ens_mm']
    own = ens['e3'] + ens['e4']
    sc_est, q_est = pooled_est(own)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs]
    med, sel = float(np.median(o2s)), max(o2s)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    out = {'q_est': q_est, 'q_true': float(q20)}
    n_st = 2 if cli.smoke else 6

    def run_set(est_sc, est_q, paths, tag):
        rows = []

        def sfun(sd):
            return est_sc(ctx['eval_full'](sd))
        for p in paths[:n_st]:
            sd = torch.load(p, map_location=device)
            fin, best = ascend_under(ctx, sd, sfun, est_q, bot, top,
                                     iters)
            ts = sc_true(ctx['eval_full'](fin))
            rows.append({'best': best, 'true_pass': bool(ts <= q20),
                         'true_s': float(ts),
                         'gain': float(gain_of(best, sel, med, th_o2)),
                         'w2c': int(((ctx['margins_of'](fin) > 0)
                                     & ~(ctx['margins_of'](sd) > 0)
                                     ).sum())})
            print(' s1', tag, os.path.basename(p),
                  'gain %.3f pass %s w2c %d'
                  % (rows[-1]['gain'], rows[-1]['true_pass'],
                     rows[-1]['w2c']), flush=True)
        best_row = max(rows, key=lambda r: r['best'])
        joint = [r for r in rows if r['true_pass'] and r['w2c'] >= 1]
        return {'per_start': rows,
                'commit_bestO2': {'pass': best_row['true_pass'],
                                  'gain': best_row['gain']},
                'random_commit_joint': len(joint) / len(rows)}

    out['main'] = run_set(sc_est, q_est, starts, 'main')
    rng = np.random.RandomState(47800)
    boot = []
    for b in range(1 if cli.smoke else 3):
        idx = rng.choice(len(own), len(own), replace=True)
        sc_b, q_b = pooled_est([own[i] for i in idx])
        boot.append({'q': q_b,
                     'res': run_set(sc_b, q_b, starts[:3],
                                    'boot{}'.format(b))})
    out['bootstrap'] = boot
    ok_main = out['main']['commit_bestO2']['pass'] and \
        out['main']['commit_bestO2']['gain'] >= GATE
    frac = (sum(1 for b in boot
                if b['res']['commit_bestO2']['pass']
                and b['res']['commit_bestO2']['gain'] >= GATE)
            / len(boot)) if boot else 0.0
    out['verdict'] = ('SUBMISSION_ESTABLISHED'
                      if ok_main and frac >= 2 / 3
                      else 'S1_FAIL')
    out['boot_success_frac'] = frac
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], '| boot frac %.2f' % frac,
          flush=True)


def audit_of_c100():
    ez7 = sorted(glob.glob('research_outputs/uw/ez7-c100b-2*'))[-1]

    def mm_of(tag):
        return [json.load(open(p)) for p in sorted(glob.glob(
            os.path.join(ez7, '{}_ft_s*_mm.json'.format(tag))))]
    fit_mm = mm_of('z1')
    cal_mm = mm_of('z2') + mm_of('z3')
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in M}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in M}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in M)
    return ez7, sc, float(conformal_q([sc(e) for e in cal_mm], 0.20))


def run_s2(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    out = {}

    # ---- cell 1: CIFAR-100 (ez7) --------------------------------
    ctx = make_context(CELLS['c100'], device)
    ez7, sc, qa = audit_of_c100()
    sd0c = torch.load(os.path.join(ez7, 'ckpt', 'theta0.pt'),
                      map_location=device)
    th_mc = ctx['margins_of'](sd0c)
    bot, top = record_subsets(th_mc, k=10)
    th_o2 = obj_val(th_mc, 'O2', bot, top)
    z4 = sorted(glob.glob(os.path.join(ez7, 'ckpt', 'z4_ft_s*.pt')))
    z4mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in z4]
    o2s = [obj_val(m, 'O2', bot, top) for m in z4mg]
    med, sel = float(np.median(o2s)), max(o2s)
    passing = [p for p in z4
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= qa][:3 if not cli.smoke else 1]
    ll, ff = [], []
    for p in passing:
        sd = torch.load(p, map_location=device)
        def sfun(sd):
            return sc(ctx['eval_full'](sd))
        _, best = ascend_under_ll(ctx, sd, sfun, qa, bot, top,
                                  iters)
        mm = ctx['eval_full'] if False else None
        ll.append({'gain': float(gain_of(best, sel, med, th_o2))})
        for lr_, st_ in ((0.01, 1),) if cli.smoke else (
                (0.003, 1), (0.003, 2), (0.01, 1), (0.01, 2)):
            sd2 = ctx['fit_full'](ctx['fset'], 47900 + st_, st_,
                                  init=sd, cosine=False, lr=lr_)
            ok = sc(ctx['eval_full'](sd2)) <= qa
            v = obj_val(ctx['margins_of'](sd2), 'O2', bot, top)
            ff.append({'lr': lr_, 'steps': st_, 'pass': bool(ok),
                       'gain': float(gain_of(v, sel, med, th_o2))})
        print(' s2 c100', os.path.basename(p), 'll',
              round(ll[-1]['gain'], 3),
              [(r['pass'], round(r['gain'], 2)) for r in ff[-4:]],
              flush=True)
    out['c100'] = {'ll': ll, 'ff': ff}

    # ---- cell 2: retrain family (ez5) ---------------------------
    ctx2 = setup_c10(device)
    prov = [ctx2['eval_full'](torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s)), map_location=device))
        for s in (44101, 44102, 44103)]
    prov += [ctx2['eval_full'](torch.load(os.path.join(
        RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
        map_location=device)) for s in (45201, 45202, 45203)]
    aud = [ctx2['eval_full'](torch.load(p, map_location=device))
           for p in sorted(glob.glob(
               'research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))]
    audit = mk_audit(prov, aud)
    _, th_m2, bot, top = theta_ctx(ctx2)
    th_o2 = obj_val(th_m2, 'O2', bot, top)
    new_p = [os.path.join(
        sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[-1],
        'ckpt', 'rtnew_s{}.pt'.format(s))
        for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    new_mg = [ctx2['margins_of'](torch.load(p, map_location=device))
              for p in new_p]
    o2n = [obj_val(m, 'O2', bot, top) for m in new_mg]
    medn, seln = float(np.median(o2n)), max(o2n)
    passing2 = [p for p, mm2 in zip(new_p, [
        ctx2['eval_full'](torch.load(x, map_location=device))
        for x in new_p]) if audit[3](mm2) <= audit[2]][:3
                                                      if not cli.smoke
                                                      else 1]
    ll2, ff2 = [], []
    for p in passing2:
        sd = torch.load(p, map_location=device)
        def sfun2(sd):
            return audit[3](ctx2['eval_full'](sd))
        _, best = ascend_under_ll(ctx2, sd, sfun2, audit[2], bot,
                                  top, iters)
        ll2.append({'gain': float(gain_of(best, seln, medn, th_o2))})
        for lr_, st_ in ((0.01, 1),) if cli.smoke else (
                (0.003, 1), (0.003, 2), (0.01, 1), (0.01, 2)):
            sd2 = ctx2['fit_full'](ctx2['fset'], 47950 + st_, st_,
                                   init=sd, cosine=False, lr=lr_)
            ok = audit[3](ctx2['eval_full'](sd2)) <= audit[2]
            v = obj_val(ctx2['margins_of'](sd2), 'O2', bot, top)
            ff2.append({'lr': lr_, 'steps': st_, 'pass': bool(ok),
                        'gain': float(gain_of(v, seln, medn, th_o2))})
        print(' s2 rt', os.path.basename(p), 'll',
              round(ll2[-1]['gain'], 3),
              [(r['pass'], round(r['gain'], 2)) for r in ff2[-4:]],
              flush=True)
    out['retrain'] = {'ll': ll2, 'ff': ff2}

    def best_passing(rows, extra):
        vals = [r['gain'] for r in rows]
        vals += [r['gain'] for r in extra if r['pass']]
        return max(vals) if vals else -1e9
    c100_best = max([max(best_passing(out['c100']['ll'],
                                     out['c100']['ff']), -1e9)])
    rt_best = max([max(best_passing(out['retrain']['ll'],
                                    out['retrain']['ff']), -1e9)])
    out['best_ll_passing_c100'] = c100_best
    out['best_ll_passing_rt'] = rt_best
    out['verdict'] = ('DIFFERENTIAL_ESTABLISHED'
                      if c100_best < GATE and rt_best < GATE
                      else 'DIFFERENTIAL_CLOSED')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], '| c100 best %.3f rt best %.3f'
          % (c100_best, rt_best), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('s1', 's2'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 's-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('E-S arm', cli.arm, '->', cli.outdir, flush=True)
    dict(s1=run_s1, s2=run_s2)[cli.arm](cli)


if __name__ == '__main__':
    main()
