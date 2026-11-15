"""UW E-Gm4: reinforcement batch for the gradient-trace attack.

  A: retrain-cell re-attack with a pairwise ranking surrogate
     (member gradient norms pushed below non-member norms,
     pairwise softplus margin) under the retrain audit.
  B: record-specificity -- directed final vs TWO non-member sets
     (cat-class and non-cat class-0 test records).
  C: audit-slack comparison -- naive 1-step ft finals vs directed
     finals: normalized audit score position (fragility check).
  D: gradient-DIRECTION feature -- cosine similarity of each
     record's input-gradient to theta0's gradient on the same
     record; AUC of that alignment feature across model classes.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import grad_norms, auc
from uw.run_gm2 import ascend_trace, flipped_auc
from uw.run_ey2 import mk_audit
from uw.run_ew3b import LR0, lr_update

RUN6 = 'research_outputs/uw/ew6-20260925T213308'
RUN3 = 'research_outputs/uw/ew3-20260925T113900'


def nonmem(device, cls, seed=52001):
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    t = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    idx = np.where(np.asarray(t.cifar10.targets) == cls)[0]
    rng = np.random.RandomState(seed)
    nm = np.sort(rng.choice(idx, 30, replace=False))
    x = torch.tensor(np.stack([t.cifar10.data[i] / 255.0
                               for i in nm]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([t.cifar10.targets[i] for i in nm]).long().to(device)
    return x, y


def pairwise_loss(ctx, sd, mem_x, mem_y, non_x, non_y, margin=1.0):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()

    def norms(x, y):
        x = x.clone().requires_grad_(True)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y,
                                                 reduction='none')
        ns = []
        for i in range(len(y)):
            g = torch.autograd.grad(loss[i], x, create_graph=True)[0][i]
            ns.append(g.norm())
        return torch.stack(ns)
    gm, gn = norms(mem_x, mem_y), norms(non_x, non_y)
    diff = gm[:, None] - gn[None, :] + margin
    return torch.nn.functional.softplus(diff).mean()


def ascend_pairwise(ctx, sd, score_sd, q_lim, mem_x, mem_y, non_x,
                    non_y, iters):
    model = ctx['model']
    cur, lr = sd, LR0
    best = float(pairwise_loss(ctx, cur, mem_x, mem_y, non_x, non_y))
    stall = 0
    for _it in range(iters):
        model.load_state_dict(cur)
        model.zero_grad()
        pairwise_loss(ctx, cur, mem_x, mem_y, non_x, non_y).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = apply_delta(cur, g, -lr)
            if score_sd(cand) <= q_lim:
                v = float(pairwise_loss(ctx, cand, mem_x, mem_y,
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


def grad_vecs(ctx, sd, x, y):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    vs = []
    for i in range(len(y)):
        g = torch.autograd.grad(loss[i], x, retain_graph=True)[0][i]\
            .flatten()
        vs.append(g.detach().cpu().numpy())
    return np.stack(vs)


def cos_align_auc(ctx, sd, ref_vecs, mem_x, mem_y, non_x, non_y):
    def align(x, y, ref):
        v = grad_vecs(ctx, sd, x, y)
        cs = [float(np.dot(v[i], ref[i])
                    / (np.linalg.norm(v[i]) * np.linalg.norm(ref[i])
                       + 1e-12)) for i in range(len(y))]
        return np.array(cs)
    a = align(mem_x, mem_y, ref_vecs[:len(mem_y)])
    b = align(non_x, non_y, ref_vecs[len(mem_y):])
    return auc(a, b)


def run_gm4(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    ctx = setup_c10(device)
    from uw.run_r1 import load_audit
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    mem_x, mem_y = ctx['xF'], ctx['yF']
    noncat_x, noncat_y = nonmem(device, 3)
    cls0_x, cls0_y = nonmem(device, 0, seed=52101)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    out = {}

    # ---- A: retrain cell, pairwise objective --------------------
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

    def rt_score(sd):
        return audit[3](ctx['eval_full'](sd))
    new_p = [os.path.join(
        sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[-1],
        'ckpt', 'rtnew_s{}.pt'.format(s))
        for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    passing = [torch.load(p, map_location=device) for p in new_p
               if rt_score(torch.load(p, map_location=device))
               <= audit[2]][:3]
    out['A_pairwise_retrain'] = []
    for k, sd in enumerate(passing):
        fin, pl = ascend_pairwise(ctx, sd, rt_score, audit[2],
                                  mem_x, mem_y, noncat_x, noncat_y,
                                  iters)
        fa = flipped_auc(ctx, fin, mem_x, mem_y, noncat_x, noncat_y)
        ok = bool(rt_score(fin) <= audit[2])
        out['A_pairwise_retrain'].append(
            {'k': k, 'base': flipped_auc(ctx, sd, mem_x, mem_y,
                                         noncat_x, noncat_y),
             'final': fa, 'pass': ok})
        print(' gm4 A', k, '-> %.3f pass %s' % (fa, ok), flush=True)

    # ---- B/C: directed finals on the ft cell + specificity + slack
    j0 = 1
    sd = torch.load(starts[j0], map_location=device)

    def ft_score(sdx):
        return sc(ctx['eval_full'](sdx))
    dfin, _ = ascend_trace(ctx, sd, ft_score, q20, mem_x, mem_y,
                           noncat_x, noncat_y, iters)
    nfin = ctx['fit_full'](ctx['fset'], 49400 + j0, 1, init=sd,
                           cosine=False)
    out['B_specificity'] = {
        'base_cat': flipped_auc(ctx, sd, mem_x, mem_y, noncat_x,
                                noncat_y),
        'base_cls0': flipped_auc(ctx, sd, mem_x, mem_y, cls0_x,
                                 cls0_y),
        'directed_cat': flipped_auc(ctx, dfin, mem_x, mem_y,
                                    noncat_x, noncat_y),
        'directed_cls0': flipped_auc(ctx, dfin, mem_x, mem_y,
                                     cls0_x, cls0_y),
        'naive_cat': flipped_auc(ctx, nfin, mem_x, mem_y, noncat_x,
                                 noncat_y)}
    out['C_slack'] = {'directed_s_over_q': float(ft_score(dfin) / q20),
                      'naive_s_over_q': float(ft_score(nfin) / q20),
                      'naive_pass': bool(ft_score(nfin) <= q20)}
    print(' gm4 B', out['B_specificity'], flush=True)
    print(' gm4 C', out['C_slack'], flush=True)

    # ---- D: direction-alignment feature --------------------------
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    ref = np.concatenate([
        grad_vecs(ctx, sd0, mem_x, mem_y),
        grad_vecs(ctx, sd0, noncat_x, noncat_y)])
    out['D_direction'] = {}
    for name, sdx in (('theta0', sd0), ('directed', dfin),
                      ('naive', nfin)):
        out['D_direction'][name] = cos_align_auc(
            ctx, sdx, ref, mem_x, mem_y, noncat_x, noncat_y)
        print(' gm4 D', name, out['D_direction'][name], flush=True)
    fa_best = max([r['final'] for r in out['A_pairwise_retrain']
                   if r['pass']] or [0.0])
    out['verdict_A'] = ('RESTORED_V2' if fa_best >= 0.65
                        else 'STILL_NOT_RESTORED')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT A', out['verdict_A'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'gm4-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=True)
    print('E-Gm4 ->', cli.outdir, flush=True)
    run_gm4(cli)


if __name__ == '__main__':
    main()
