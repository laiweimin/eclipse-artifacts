"""UW E-Gm2: directed gradient-trace restoration under the audit
(frozen prereg docs/plans/2026-10-05-uw-gm-gradient-trace-*.md).

Trace objective (differentiable, double-backward):
  minimize mean(gnorm_members) - 0.5 * mean(gnorm_nonmembers)
(θ0 reference: member median 0.31 vs non-member 3.51).  Both
channels -- full-model and last-layer -- optimize the SAME
objective under the SAME true-audit constraint; readout is the
flipped AUC (1 - AUC of member-vs-non gradient norms), compared
against the frozen gates (DIFFERENTIAL: full >= 0.65 passing and
ll < 0.58).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ey import conformal_q, gain_of
from uw.run_ez import obj_val, setup_c10
from uw.run_gm import grad_norms, auc
from uw.run_r1 import load_audit
from uw.run_r2 import theta_ctx
from uw.run_ew3b import LR0, lr_update
from uw.run_ez import apply_delta


def trace_loss(ctx, sd, mem_x, mem_y, non_x, non_y):
    """Differentiable trace objective via double backward."""
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()

    def gnorm_mean(x, y):
        x = x.clone().requires_grad_(True)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y,
                                                 reduction='none')
        total = 0.0
        for i in range(len(y)):
            g = torch.autograd.grad(loss[i], x, create_graph=True)[0][i]
            total = total + g.norm()
        return total / len(y)
    return gnorm_mean(mem_x, mem_y) - 0.5 * gnorm_mean(non_x, non_y)


def ascend_trace(ctx, sd, score_sd, q_lim, mem_x, mem_y, non_x,
                 non_y, iters, mask=None):
    model = ctx['model']
    cur, lr = sd, LR0
    best = float(trace_loss(ctx, cur, mem_x, mem_y, non_x, non_y))
    stall = 0
    for _it in range(iters):
        model.load_state_dict(cur)
        model.zero_grad()
        tl = trace_loss(ctx, cur, mem_x, mem_y, non_x, non_y)
        tl.backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters()
             if p.grad is not None
             and (mask is None or nm.startswith(mask))}
        acc = False
        for _bt in range(8):
            cand = apply_delta(cur, g, -lr)      # descend trace loss
            if score_sd(cand) <= q_lim:
                v = float(trace_loss(ctx, cand, mem_x, mem_y,
                                     non_x, non_y))
                if v < best - 1e-9:
                    cur, best, acc = cand, v, True
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            break
    return cur, best


def flipped_auc(ctx, sd, mem_x, mem_y, non_x, non_y):
    m = grad_norms(ctx, sd, mem_x, mem_y)
    n = grad_norms(ctx, sd, non_x, non_y)
    return 1.0 - auc(m, n)


def run_gm2(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_test = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    rng = np.random.RandomState(52001)
    nm_idx = np.sort(rng.choice(cat_test, 30, replace=False))
    non_x = torch.tensor(np.stack(
        [tset.cifar10.data[i] / 255.0 for i in nm_idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    non_y = torch.tensor([tset.cifar10.targets[i]
                          for i in nm_idx]).long().to(device)
    mem_x, mem_y = ctx['xF'], ctx['yF']
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    passing = [j for j in (1, 3, 4)
               if score_sd(torch.load(starts[j],
                                      map_location=device)) <= q20]
    rows = {'full': [], 'll': []}
    for chan in ('full', 'll'):
        for j in (passing if not cli.smoke else passing[:1]):
            sd = torch.load(starts[j], map_location=device)
            base = flipped_auc(ctx, sd, mem_x, mem_y, non_x, non_y)
            fin, tl = ascend_trace(
                ctx, sd, score_sd, q20, mem_x, mem_y, non_x, non_y,
                iters, mask=None if chan == 'full' else 'fc.')
            fa = flipped_auc(ctx, fin, mem_x, mem_y, non_x, non_y)
            ok = bool(score_sd(fin) <= q20)
            rows[chan].append({'j': j, 'base_flipped': base,
                               'final_flipped': fa, 'pass': ok,
                               'trace_loss': tl})
            print(' gm2', chan, j, 'base %.3f -> final %.3f pass %s'
                  % (base, fa, ok), flush=True)
    full_best = max([r['final_flipped'] for r in rows['full']
                     if r['pass']] or [0.0])
    ll_best = max([r['final_flipped'] for r in rows['ll']
                   if r['pass']] or [0.0])
    v = ('DIFFERENTIAL' if full_best >= 0.65 and ll_best < 0.58
         else 'NULL')
    out = {'rows': rows, 'full_best': full_best, 'll_best': ll_best,
           'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, '| full %.3f ll %.3f' % (full_best, ll_best),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'gm2-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('E-Gm2 ->', cli.outdir, flush=True)
    run_gm2(cli)


if __name__ == '__main__':
    main()
