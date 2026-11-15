"""A9-z2: confirmatory one-sided top-k z detector on MobileNetV2
(prereg amendment in docs/plans/2026-10-26-a9z-detector-prereg.md).
z-cal = ez1-mob z1 (6 refs); honest T = z2..z5 (24 models);
threshold q95; test = 6 E-MOB passing encoded finals. Gate >=4/6."""
import argparse, glob, json, os, time
import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K
from uw.run_ew3 import f_deep as real_f_deep
from uw.run_lfpr import channels

MOB = 'research_outputs/uw/ez1-mob-20261002T080210'
EMOB = sorted(glob.glob(
    'research_outputs/uw/emob-20261026T141216/ckpt/ce_*.pt'))


def T1(model, sd, mem, mu, sg, k=15):
    z = (channels(model, sd, *mem)['neggrad'] - mu) / (sg + 1e-12)
    return float(np.mean(np.sort(z)[::-1][:k]))


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = ez.make_context(ez.CELLS['mob'], device)
    model = ctx['model']
    fit = [torch.load(p, map_location=device) for p in sorted(
        glob.glob(MOB + '/ckpt/z1_ft_s*.pt'))]
    assert len(fit) == 6
    base = np.stack([channels(model, sd, ctx['xF'], ctx['yF'])
                     ['neggrad'] for sd in fit])
    mu, sg = base.mean(0), base.std(0)
    hon = [torch.load(p, map_location=device) for p in sorted(
        glob.glob(MOB + '/ckpt/z[2345]_ft_s*.pt'))]
    assert len(hon) == 24
    honT = [T1(model, sd, (ctx['xF'], ctx['yF']), mu, sg)
            for sd in hon]
    q95 = float(np.quantile(honT, 0.95))
    fins = [torch.load(p, map_location=device) for p in EMOB]
    assert len(fins) == 6
    finT = [T1(model, sd, (ctx['xF'], ctx['yF']), mu, sg)
            for sd in fins]
    det = int(sum(t > q95 for t in finT))
    hon_det = int(sum(t > q95 for t in honT))
    print('q95=%.3f hon_det=%d/24 finals_det=%d/6 T=%s'
          % (q95, hon_det, det, [round(t, 2) for t in finT]),
          flush=True)
    verdict = 'CONFIRMED' if det >= 4 else 'NOT-CONFIRMED'
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q95': q95, 'honest_det': hon_det,
           'finals_T': finT, 'finals_det': det, 'verdict': verdict}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/a9z2-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
