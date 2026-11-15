"""UW E-U1: interactive path-bisection threshold extraction (frozen
prereg docs/plans/2026-10-04-uw-u-interactive-extraction-*.md).

Per start (no audit knowledge): ascend O2 unconstrained, recording
the path; submit exponentially spaced path points to the audit
(binary feedback) until first rejection; bisect between last-pass
and first-fail; final submission = a safety point on the passing
side.  Baselines under the same feedback: class-bias bisection and
last-layer path bisection.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from uw.run_ey import GATE, gain_of, record_subsets
from uw.run_ez import apply_delta, obj_val, setup_c10
from uw.run_r1 import load_audit
from uw.run_r2 import theta_ctx
from uw.run_ew3b import LR0, lr_update


def ascend_path(ctx, start_sd, bot, top, iters, record_every=1):
    """Unconstrained O2 ascent returning the state path."""
    import torch
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    device = xF.device
    bot_t = torch.as_tensor(bot, device=device)
    cur, lr = start_sd, LR0
    best = obj_val(ctx['margins_of'](cur), 'O2', bot, top)
    path = [cur]
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
            v = obj_val(ctx['margins_of'](cand), 'O2', bot, top)
            if v > best + 1e-7:
                cur, best, acc = cand, v, True
                break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        path.append(cur)
        stall = 0 if acc else stall + 1
        if stall > 15:
            break
    return path


def ll_ascend_path(ctx, start_sd, bot, top, iters):
    import torch
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    device = xF.device
    bot_t = torch.as_tensor(bot, device=device)
    cur, lr = start_sd, LR0
    best = obj_val(ctx['margins_of'](cur), 'O2', bot, top)
    path = [cur]
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
            v = obj_val(ctx['margins_of'](cand), 'O2', bot, top)
            if v > best + 1e-7:
                cur, best, acc = cand, v, True
                break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        path.append(cur)
        stall = 0 if acc else stall + 1
        if stall > 15:
            break
    return path


def bisect_submit(ctx, path, sc, q, max_queries=12):
    """Submit path points (binary feedback); return (final, queries)."""
    n = len(path)
    probes = sorted(set(min(n - 1, max(0, int(n * f)))
                        for f in (0.25, 0.5, 0.75, 1.0)))
    queries = 0
    lo = 0                      # index 0 assumed passing (the start)
    hi = None
    for p in probes:
        queries += 1
        if sc(ctx['eval_full'](path[p])) <= q:
            lo = p
        else:
            hi = p
            break
    if hi is None:
        return path[lo], queries, True          # no rejection seen
    while hi - lo > 1 and queries < max_queries:
        mid = (lo + hi) // 2
        queries += 1
        if sc(ctx['eval_full'](path[mid])) <= q:
            lo = mid
        else:
            hi = mid
    return path[lo], queries, False


def bias_bisect(ctx, start_sd, sc, q, bot, top):
    """Baseline: class-bias scan + bisection under the same feedback."""
    import torch
    grid = [0.5, 1, 2, 4, 8, 16]
    queries, lo_b, hi_b = 0, 0.0, None
    for b in grid:
        queries += 1
        sd2 = {k: v.clone() for k, v in start_sd.items()}
        sd2['fc.bias'][3] += b
        if sc(ctx['eval_full'](sd2)) <= q:
            lo_b = b
        else:
            hi_b = b
            break
    if hi_b is not None:
        while hi_b - lo_b > 0.25 and queries < 12:
            mid = (lo_b + hi_b) / 2
            queries += 1
            sd2 = {k: v.clone() for k, v in start_sd.items()}
            sd2['fc.bias'][3] += mid
            if sc(ctx['eval_full'](sd2)) <= q:
                lo_b = mid
            else:
                hi_b = mid
    sd2 = {k: v.clone() for k, v in start_sd.items()}
    sd2['fc.bias'][3] += lo_b
    return sd2, queries


def run_u1(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 12 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs]
    med, sel = float(np.median(o2s)), max(o2s)
    n_st = 2 if cli.smoke else 6
    out = {'full': [], 'lastlayer': [], 'bias': []}
    for j, p in enumerate(starts[:n_st]):
        sd = torch.load(p, map_location=device)
        path = ascend_path(ctx, sd, bot, top, iters)
        fin, nq, no_rej = bisect_submit(ctx, path, sc, q20)
        ts = sc(ctx['eval_full'](fin))
        o2v = obj_val(ctx['margins_of'](fin), 'O2', bot, top)
        out['full'].append({
            'j': j, 'queries': nq, 'no_rejection': no_rej,
            'true_pass': bool(ts <= q20), 'gain': float(
                gain_of(o2v, sel, med, th_o2))})
        print(' u1 full', j, 'q', nq, 'no_rej', no_rej,
              'pass', out['full'][-1]['true_pass'],
              'gain %.3f' % out['full'][-1]['gain'], flush=True)
    for j, p in enumerate(starts[:n_st]):
        sd = torch.load(p, map_location=device)
        path = ll_ascend_path(ctx, sd, bot, top, iters)
        fin, nq, no_rej = bisect_submit(ctx, path, sc, q20)
        o2v = obj_val(ctx['margins_of'](fin), 'O2', bot, top)
        ts = sc(ctx['eval_full'](fin))
        out['lastlayer'].append({
            'j': j, 'queries': nq,
            'true_pass': bool(ts <= q20),
            'gain': float(gain_of(o2v, sel, med, th_o2))})
        print(' u1 ll', j, 'q', nq, 'pass',
              out['lastlayer'][-1]['true_pass'],
              'gain %.3f' % out['lastlayer'][-1]['gain'], flush=True)
    passing0 = [j for j, p in enumerate(starts)
                if sc(ctx['eval_full'](torch.load(p,
                                                  map_location=device)))
                <= q20][:n_st]
    for j in passing0:
        sd = torch.load(starts[j], map_location=device)
        fin, nq = bias_bisect(ctx, sd, sc, q20, bot, top)
        o2v = obj_val(ctx['margins_of'](fin), 'O2', bot, top)
        out['bias'].append({
            'j': j, 'queries': nq,
            'true_pass': bool(sc(ctx['eval_full'](fin)) <= q20),
            'gain': float(gain_of(o2v, sel, med, th_o2))})
        print(' u1 bias', j, 'q', nq, 'pass',
              out['bias'][-1]['true_pass'],
              'gain %.3f' % out['bias'][-1]['gain'], flush=True)
    ok = sum(1 for r in out['full']
             if r['true_pass'] and r['gain'] >= GATE
             and r['queries'] <= 12)
    out['verdict'] = ('FEEDBACK_SUBMISSION_ESTABLISHED'
                      if ok >= (n_st - 1) else 'U1_PATH_FAIL')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='u1', choices=('u1',))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'u-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('E-U arm', cli.arm, '->', cli.outdir, flush=True)
    run_u1(cli)


if __name__ == '__main__':
    main()
