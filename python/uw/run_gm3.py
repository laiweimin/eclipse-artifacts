"""UW E-Gm3: headline cells for the gradient-trace attack.

  arm1 (retrain): honest retrains carry no trace (flipped ~0.51);
       directed trace ascent under the RETRAIN-family audit
       restores it -- behavioral-indistinguishable, white-box-
       distinguishable returns.  Includes the ll ablation.
  arm2 (fftc): 1-step naive F-ft finals that PASS the ft audit --
       their flipped AUC (does naive re-memorization already
       restore the trace while passing?).
  arm3 (c100): directed ascent under the ez7 CIFAR-100 audit from
       z4 starts; members = c100 F, non-members = class-0 test.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_ez import setup_c10
from uw.run_gm import grad_norms, auc
from uw.run_gm2 import ascend_trace, flipped_auc
from uw.run_ey2 import mk_audit

RUN6 = 'research_outputs/uw/ew6-20260925T213308'
RUN3 = 'research_outputs/uw/ew3-20260925T113900'


def nonmembers(device, cls=3, seed=52001):
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    idx_all = np.where(np.asarray(tset.cifar10.targets) == cls)[0]
    rng = np.random.RandomState(seed)
    nm = np.sort(rng.choice(idx_all, 30, replace=False))
    x = torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                               for i in nm]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([tset.cifar10.targets[i] for i in nm])\
        .long().to(device)
    return x, y


def run_arm1(cli, device, out, iters):
    ctx = setup_c10(device)
    prov = [ctx['eval_full'](torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k30_s{}.pt'.format(s)), map_location=device))
        for s in (44101, 44102, 44103)]
    prov += [ctx['eval_full'](torch.load(os.path.join(
        RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
        map_location=device)) for s in (45201, 45202, 45203)]
    aud = [ctx['eval_full'](torch.load(p, map_location=device))
           for p in sorted(glob.glob(
               'research_outputs/uw/r3p1-2*/ckpt/aud_s*.pt'))]
    audit = mk_audit(prov, aud)

    def score_sd(sd):
        return audit[3](ctx['eval_full'](sd))
    mem_x, mem_y = ctx['xF'], ctx['yF']
    non_x, non_y = nonmembers(device)
    new_p = [os.path.join(
        sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[-1],
        'ckpt', 'rtnew_s{}.pt'.format(s))
        for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    passing = [torch.load(p, map_location=device) for p in new_p
               if score_sd(torch.load(p, map_location=device))
               <= audit[2]][:3]
    out['retrain'] = {'full': [], 'll': []}
    for chan in ('full', 'll'):
        for k, sd in enumerate(passing):
            base = flipped_auc(ctx, sd, mem_x, mem_y, non_x, non_y)
            fin, tl = ascend_trace(
                ctx, sd, score_sd, audit[2], mem_x, mem_y,
                non_x, non_y, iters, mask=None if chan == 'full'
                else 'fc.')
            fa = flipped_auc(ctx, fin, mem_x, mem_y, non_x, non_y)
            ok = bool(score_sd(fin) <= audit[2])
            out['retrain'][chan].append(
                {'k': k, 'base': base, 'final': fa, 'pass': ok})
            print(' gm3 rt', chan, k, '%.3f -> %.3f pass %s'
                  % (base, fa, ok), flush=True)
    fb = max([r['final'] for r in out['retrain']['full']
              if r['pass']] or [0.0])
    out['retrain']['verdict'] = ('RESTORED' if fb >= 0.65
                                 else 'NOT_RESTORED')
    print(' arm1', out['retrain']['verdict'], 'full best %.3f' % fb,
          flush=True)


def run_arm2(cli, device, out):
    from uw.run_r1 import load_audit
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    mem_x, mem_y = ctx['xF'], ctx['yF']
    non_x, non_y = nonmembers(device)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    rows = []
    for j in (1, 3, 4):
        sd = torch.load(starts[j], map_location=device)
        sd2 = ctx['fit_full'](ctx['fset'], 49300 + j, 1, init=sd,
                              cosine=False)
        ok = bool(sc(ctx['eval_full'](sd2)) <= q20)
        fa = flipped_auc(ctx, sd2, mem_x, mem_y, non_x, non_y)
        rows.append({'j': j, 'pass': ok, 'flipped': fa})
        print(' gm3 fft', j, 'pass %s flipped %.3f' % (ok, fa),
              flush=True)
    out['fft_1step'] = rows


def run_arm3(cli, device, out, iters):
    from uw.run_ez import CELLS, make_context
    from uw.run_s import audit_of_c100
    ctx = make_context(CELLS['c100'], device)
    ez7, sc, qa = audit_of_c100()

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    mem_x, mem_y = ctx['xF'], ctx['yF']
    non_x, non_y = nonmembers(device, cls=0)
    z4 = sorted(glob.glob(os.path.join(ez7, 'ckpt',
                                       'z4_ft_s*.pt')))
    passing = [torch.load(p, map_location=device) for p in z4
               if score_sd(torch.load(p, map_location=device))
               <= qa][:3]
    out['c100'] = {'full': []}
    for k, sd in enumerate(passing):
        base = flipped_auc(ctx, sd, mem_x, mem_y, non_x, non_y)
        fin, tl = ascend_trace(ctx, sd, score_sd, qa, mem_x, mem_y,
                               non_x, non_y, iters)
        fa = flipped_auc(ctx, fin, mem_x, mem_y, non_x, non_y)
        ok = bool(score_sd(fin) <= qa)
        out['c100']['full'].append({'k': k, 'base': base,
                                    'final': fa, 'pass': ok})
        print(' gm3 c100', k, '%.3f -> %.3f pass %s'
              % (base, fa, ok), flush=True)
    fb = max([r['final'] for r in out['c100']['full']
              if r['pass']] or [0.0])
    out['c100']['verdict'] = ('REPLICATED' if fb >= 0.65
                              else 'NOT_REPLICATED')
    print(' arm3', out['c100']['verdict'], 'best %.3f' % fb,
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'gm3-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    print('E-Gm3 ->', cli.outdir, flush=True)
    out = {}
    run_arm1(cli, device, out, iters)
    run_arm2(cli, device, out)
    run_arm3(cli, device, out, iters)
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('OUT', cli.outdir, flush=True)


if __name__ == '__main__':
    main()
