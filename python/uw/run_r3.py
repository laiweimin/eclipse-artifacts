"""UW R3: continued solidification (frozen prereg
docs/plans/2026-10-03-uw-r3-solidify-preregistration.md).

Arms:
  percls     : per-class test-accuracy differential of the six
               passing O2 finals vs their own starts + absolute
               delta-O2 margins (forward only).
  trep       : combined per-record score max(s,T) replicated on
               (i) the CIFAR-10 e4 pool and (ii) the retrain
               family (ez5 frozen audit + the six new retrains).
  subsetsens : O2' = bottom-10 by honest-pool per-record MEDIAN
               margins (instead of theta0's); attack under the
               true audit from passing starts.
  subext     : (i) F-ft lr grid on passing starts; (ii) class
               bias on the retrain family under the retrain audit.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import (GATE, METRICS, conformal_q, gain_of,
                       per_record_margin, record_subsets)
from uw.run_ez import (apply_delta, obj_val, run_attack, setup_c10)
from uw.run_ey2 import mk_audit
from uw.run_r1 import CACHE, RUN3, load_audit
from uw.run_r2 import ey2_dir, theta_ctx

RUN6 = 'research_outputs/uw/ew6-20260925T213308'
EZ5 = None


def ez5_dir():
    return sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[-1]


def perclass_diff(ctx, sd_a, sd_b):
    """Per-class test accuracy of sd_a minus sd_b."""
    import torch
    ctx['model'].load_state_dict(sd_a)
    ctx['model'].eval()
    za, ya = [], []
    with torch.no_grad():
        for x, y in ctx['loader_te']:
            za.append(ctx['model'](x.to(ctx['xF'].device)).cpu())
            ya.append(y)
    za, ya = torch.cat(za).numpy(), torch.cat(ya).numpy()
    ctx['model'].load_state_dict(sd_b)
    zb = []
    with torch.no_grad():
        for x, _y in ctx['loader_te']:
            zb.append(ctx['model'](x.to(ctx['xF'].device)).cpu())
    zb = torch.cat(zb).numpy()
    pa, pb = za.argmax(1), zb.argmax(1)
    return [float((pa[ya == k] == k).mean()
                  - (pb[ya == k] == k).mean()) for k in range(10)]


def run_percls(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    rows = []
    for pth in sorted(glob.glob(os.path.join(
            ey2_dir(), 'ckpt', 'orc_e3_O2_s*.pt'))):
        j0 = int(os.path.basename(pth)[:-3].rsplit('s', 1)[1])
        fin = torch.load(pth, map_location=device)
        st = torch.load(starts[j0], map_location=device)
        mg_f = ctx['margins_of'](fin)
        mg_s = ctx['margins_of'](st)
        o2s = [obj_val(ctx['margins_of'](torch.load(p,
                                                    map_location=device)),
                       'O2', bot, top) for p in starts]
        rows.append({
            'final': os.path.basename(pth),
            'abs_dO2': float(mg_f[bot].mean() - mg_s[bot].mean()),
            'start_O2': float(mg_s[bot].mean()),
            'final_O2': float(mg_f[bot].mean()),
            'pool_median_O2': float(np.median(o2s)),
            'perclass_diff': perclass_diff(ctx, fin, st)})
        print(' percls', rows[-1]['final'],
              'abs_dO2 %.3f' % rows[-1]['abs_dO2'],
              'max|dclass| %.3f' % max(abs(x) for x in
                                       rows[-1]['perclass_diff']),
              flush=True)
    with open(os.path.join(cli.outdir, 'percls.json'), 'w') as fh:
        json.dump(rows, fh, indent=1, default=float)


def tfit(ctx, fit_paths):
    import torch
    device = ctx['xF'].device
    R = np.stack([ctx['margins_of'](torch.load(p, map_location=device))
                  for p in fit_paths])
    c_j = np.median(R, axis=0)
    s_j = np.maximum(np.median(np.abs(R - c_j), axis=0), 1e-9)
    return c_j, s_j


def run_trep(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    ctx = setup_c10(device)
    out = {}

    # ---- (i) e4 pool under the frozen ft audit ------------------
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    th_objs = {'O2': obj_val(th_m, 'O2', bot, top)}
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    e4mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in e4]
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e3mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in e3]
    passing = [p for p in e4
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20][:3]
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    c_j, s_j = tfit(ctx, fit_paths)

    def T_sd(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    cal_T = [T_sd(torch.load(p, map_location=device))
             for p in cal_paths]
    cal_s = [sc(e) for e in cal]
    qC = conformal_q([max(a, b) for a, b in zip(cal_s, cal_T)], 0.20)
    ho = float(np.mean([
        max(sc(ctx['eval_full'](torch.load(p, map_location=device))),
            T_sd(torch.load(p, map_location=device))) <= qC
        for p in (e3 + e4)]))
    out['e4'] = {'qC': float(qC), 'honest24': ho}

    def sc_C(sd):
        return max(sc(ctx['eval_full'](sd)), T_sd(sd))

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    out['e4']['attack'] = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top,
        ctx['margins_of'], sc_C, qC, cli.outdir, ck, 'e4comb',
        passing, e4mg[:len(passing)], th_objs, objs=('O2',),
        n_starts=(1 if cli.smoke else 3), iters=iters)
    print(' e4 qC %.3f honest %.3f best_gp %.3f'
          % (qC, ho, out['e4']['attack']['O2']['best_gain_pass']),
          flush=True)

    # ---- (ii) retrain family (ez5 audit + new retrains) ---------
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
    new_p = [os.path.join(ez5_dir(), 'ckpt', 'rtnew_s{}.pt'.format(s))
             for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    new_mg = [ctx['margins_of'](torch.load(p, map_location=device))
              for p in new_p]
    new_mm = [ctx['eval_full'](torch.load(p, map_location=device))
              for p in new_p]
    c2, s2 = tfit(ctx, [os.path.join(RUN3, 'ckpt',
                                     'rt_k30_s{}.pt'.format(s))
                        for s in (44101, 44102, 44103)]
                  + [os.path.join(
                      RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s))
                     for s in (45201, 45202, 45203)])

    def T2_sd(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c2, 0.0) / s2))

    cal_T2 = [T2_sd(sd) for sd in aud_ck]
    qT2 = conformal_q(cal_T2, 0.20)
    qC2 = conformal_q([max(audit[3](m), t)
                       for m, t in zip(aud_mm, cal_T2)], 0.20)
    ho2 = float(np.mean([
        max(audit[3](m), T2_sd(torch.load(p, map_location=device)))
        <= qC2 for m, p in zip(new_mm, new_p)]))
    passing2 = [p for p, m in zip(new_p, new_mm)
                if audit[3](m) <= audit[2]][:3]
    out['retrain'] = {'q': float(audit[2]), 'qC': float(qC2),
                      'honest6': ho2}

    def sc_C2(sd):
        return max(audit[3](ctx['eval_full'](sd)), T2_sd(sd))

    out['retrain']['attack'] = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top,
        ctx['margins_of'], sc_C2, qC2, cli.outdir, ck, 'rtcomb',
        passing2, new_mg[:len(passing2)], th_objs, objs=('O2',),
        n_starts=(1 if cli.smoke else 3), iters=iters)
    print(' rt qC %.3f honest %.3f best_gp %.3f'
          % (qC2, ho2, out['retrain']['attack']['O2'][
              'best_gain_pass']), flush=True)

    b1 = out['e4']['attack']['O2']['best_gain_pass']
    b2 = out['retrain']['attack']['O2']['best_gain_pass']
    out['verdict'] = ('TREP_CLOSES' if max(b1, b2) < GATE
                      else 'TREP_LEAKS')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def run_subsetsens(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    med_mg = np.median(np.stack(mgs), axis=0)
    bot2 = np.argsort(med_mg, kind='stable')[:10]   # pool-median O2'
    th_o = float(np.median(th_m))
    o2p_vals = [float(m[bot2].mean()) for m in mgs]
    med, sel = float(np.median(o2p_vals)), max(o2p_vals)
    th_o2p = float(th_m[bot2].mean())
    passing = [p for p in starts
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20][:3]
    out = {'subset': 'pool-median bottom-10', 'E': th_o2p - med,
           'sel': sel}

    def o2p(sd):
        return float(ctx['margins_of'](sd)[bot2].mean())

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    res = {'O2p': None}
    # lightweight ascent: reuse run_attack with an obj shim is not
    # possible (bot is baked in); run a local loop instead.
    bot_t = torch.as_tensor(bot2, device=device)
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    from uw.run_ew3b import LR0, lr_update
    gains, passes = [], []
    for p in (passing if not cli.smoke else passing[:1]):
        cur = torch.load(p, map_location=device)
        lr, best = LR0, o2p(cur)
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
            g = {nm: pp.grad.detach().clone() for nm, pp in
                 model.named_parameters() if pp.grad is not None}
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                if sc(ctx['eval_full'](cand)) <= q20:
                    v = o2p(cand)
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
        mm = ctx['eval_full'](cur)
        g_ = gain_of(best, sel, med, th_o2p)
        gains.append(g_)
        passes.append(bool(sc(mm) <= q20))
        print(' o2p', os.path.basename(p), 'gain %.3f pass %s'
              % (g_, passes[-1]), flush=True)
    out['gains'] = gains
    out['passes'] = passes
    out['best_gp'] = max((g if p else -1e9)
                         for g, p in zip(gains, passes))
    out['verdict'] = ('SENS_ROBUST' if out['best_gp'] >= GATE
                      else 'SENS_FRAGILE')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def run_subext(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    o2s = [obj_val(ctx['margins_of'](torch.load(p, map_location=device)),
                   'O2', bot, top) for p in starts]
    med, sel = float(np.median(o2s)), max(o2s)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    passing = [p for p in starts
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20][:3 if not cli.smoke else 1]
    out = {}

    # (i) F-ft lr grid
    fft = []
    for p in passing:
        for lr_ in ((0.003,) if cli.smoke else
                    (0.001, 0.003, 0.01, 0.03)):
            for st_ in ((1,) if cli.smoke else (1, 2, 5)):
                sd = ctx['fit_full'](ctx['fset'], 48000 + st_, st_,
                                      init=torch.load(p,
                                                      map_location=device),
                                      cosine=False, lr=lr_)
                mm = ctx['eval_full'](sd)
                v = obj_val(ctx['margins_of'](sd), 'O2', bot, top)
                fft.append({'start': os.path.basename(p), 'lr': lr_,
                            'steps': st_, 'pass': bool(sc(mm) <= q20),
                            'gain': float(gain_of(v, sel, med, th_o2))})
        print(' fftgrid', os.path.basename(p),
              [(r['lr'], r['steps'], round(r['gain'], 2), r['pass'])
               for r in fft if r['start'] == os.path.basename(p)][-6:],
              flush=True)
    out['fft_grid'] = fft

    # (ii) class bias on the retrain family
    prov_ck = [torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s)), map_location=device)
        for s in (44101, 44102, 44103)]
    prov_ck += [torch.load(os.path.join(
        RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
        map_location=device) for s in (45201, 45202, 45203)]
    aud_ck = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))]
    audit = mk_audit([ctx['eval_full'](sd) for sd in prov_ck],
                     [ctx['eval_full'](sd) for sd in aud_ck])
    new_p = [os.path.join(ez5_dir(), 'ckpt', 'rtnew_s{}.pt'.format(s))
             for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    new_mg = [ctx['margins_of'](torch.load(p, map_location=device))
              for p in new_p]
    o2n = [obj_val(m, 'O2', bot, top) for m in new_mg]
    medn, seln = float(np.median(o2n)), max(o2n)
    thn = obj_val(th_m, 'O2', bot, top)
    passing2 = [p for p, sd in zip(new_p, [
            torch.load(x, map_location=device) for x in new_p])
        if audit[3](ctx['eval_full'](sd)) <= audit[2]][:3]
    bias = []
    for p in passing2:
        bb, bv = None, None
        for b in ((1.0, 4.0) if cli.smoke else (0.5, 1, 2, 4, 8)):
            sd2 = {k: v.clone() for k, v in torch.load(
                p, map_location=device).items()}
            sd2['fc.bias'][3] += b
            mm = ctx['eval_full'](sd2)
            if audit[3](mm) <= audit[2]:
                bb, bv = b, obj_val(ctx['margins_of'](sd2), 'O2',
                                    bot, top)
        bias.append({'start': os.path.basename(p), 'b_max': bb,
                     'gain': (float(gain_of(bv, seln, medn, thn))
                              if bv is not None else None)})
        print(' rt bias', os.path.basename(p), bb, bias[-1]['gain'],
              flush=True)
    out['retrain_bias'] = bias
    with open(os.path.join(cli.outdir, 'subext.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('percls', 'trep', 'subsetsens', 'subext'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'r3-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('R3 arm', cli.arm, '->', cli.outdir, flush=True)
    dict(percls=run_percls, trep=run_trep, subsetsens=run_subsetsens,
         subext=run_subext)[cli.arm](cli)


if __name__ == '__main__':
    main()
