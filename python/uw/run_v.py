"""UW E-V: P2 probe -- transformation-robust re-memorization.

For one audit-passing start, build three passing finals under the
TRUE audit (full-model ascent, last-layer ascent, class bias), then
measure wrong-to-correct flip rates on the original F records AND
on deterministic transformed views (horizontal flip, four 2-px
shifts).  Frozen gate: full-model flips transformed views at >= 2x
the rate of BOTH cheap channels on >= 2 views.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ey import record_subsets
from uw.run_ez import apply_delta, obj_val, setup_c10
from uw.run_r1 import load_audit
from uw.run_r2 import theta_ctx
from uw.run_u import bias_bisect
from uw.run_s import ascend_under, ascend_under_ll


def views_of(x):
    """Original + hflip + four 2px shifts (pad+crop), all (N,3,32,32)."""
    out = [('orig', x)]
    out.append(('hflip', torch.flip(x, dims=[3])))
    for name, shift in (('l2', (-0, -2)), ('r2', (-0, 2)),
                        ('u2', (-2, -0)), ('d2', (2, -0))):
        pad = F.pad(x, (2, 2, 2, 2))
        out.append((name, pad[:, :,
                              2 + shift[0]:34 + shift[0],
                              2 + shift[1]:34 + shift[1]]))
    return out


def flip_rate(ctx, sd_fin, sd_start, xviews, yv):
    mg_f = per_margin(ctx, sd_fin, xviews, yv)
    mg_s = per_margin(ctx, sd_start, xviews, yv)
    return int(((mg_f > 0) & ~(mg_s > 0)).sum()), \
        int(((mg_s > 0) & ~(mg_f > 0)).sum())


def per_margin(ctx, sd, x, y):
    ctx['model'].load_state_dict(sd)
    ctx['model'].eval()
    with torch.no_grad():
        z = ctx['model'](x)
    zy = z.gather(1, y[:, None]).squeeze(1)
    zc = z.clone()
    zc.scatter_(1, y[:, None], -1e18)
    return (zy - zc.max(1).values).cpu().numpy()


def run_v(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 8 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    passing = [p for p in starts
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20][:3]
    xF, yF = ctx['xF'], ctx['yF']
    xviews = views_of(xF)
    out = {'rows': []}
    for p in passing:
        sd = torch.load(p, map_location=device)

        def sfun(sdx):
            return sc(ctx['eval_full'](sdx))
        full_fin, _ = ascend_under(ctx, sd, sfun, q20, bot, top, iters)
        ll_fin, _ = ascend_under_ll(ctx, sd, sfun, q20, bot, top,
                                    iters)
        bias_fin, _ = bias_bisect(ctx, sd, sc, q20, bot, top)
        row = {'start': os.path.basename(p), 'views': {}}
        for name, xv in xviews:
            f_w2c, f_c2w = flip_rate(ctx, full_fin, sd, xv, yY)
            l_w2c, l_c2w = flip_rate(ctx, ll_fin, sd, xv, yY)
            b_w2c, b_c2w = flip_rate(ctx, bias_fin, sd, xv, yY)
            row['views'][name] = {'full': [int(f_w2c), int(f_c2w)],
                                  'll': [int(l_w2c), int(l_c2w)],
                                  'bias': [int(b_w2c), int(b_c2w)]}
        out['rows'].append(row)
        print(' v', os.path.basename(p),
              {k: (v['full'][0], v['ll'][0], v['bias'][0])
               for k, v in row['views'].items()}, flush=True)
    wins = 0
    for row in out['rows']:
        for name, v in row['views'].items():
            if name == 'orig':
                continue
            fv, lv, bv = v['full'][0], v['ll'][0], v['bias'][0]
            if fv >= 2 * max(lv, bv) and fv >= 2:
                wins += 1
    out['transform_wins'] = wins
    out['verdict'] = ('STRUCTURAL_DIFFERENTIAL'
                      if wins >= 2 * len(out['rows'])
                      else 'P2_PROBE_FAIL')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], '| wins', wins, flush=True)


def pick_passing(ctx, path, sc, q):
    best, idx = None, 0
    for i, sd in enumerate(path):
        if sc(ctx['eval_full'](sd)) <= q:
            v = float('nan')
            best, idx = sd, i
    return (best if best is not None else path[0]), idx, len(path)


yY = None


def main():
    global yY
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'v-probe-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    ctx0 = setup_c10(torch.device('cuda:{}'.format(cli.gpu)))
    yY = ctx0['yF']
    print('E-V probe ->', cli.outdir, flush=True)
    run_v(cli)


if __name__ == '__main__':
    main()
