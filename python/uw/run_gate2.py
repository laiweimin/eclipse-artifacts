"""UW-Gm gates G2/G3 (reviewer's remaining loops; run after G1c
PASS so there is a live attack to test).

  g2 : same-start same-budget same-audit paired controls, all
       evaluated with the G1b label-blind protocol (200 fresh
       non-members): (i) plain F-CE first-order descent;
       (ii) contrastive CE (F down, N_c up) in loss space;
       (iii) F-only gradient-norm objective; (iv) N_c-only;
       (v) joint (= the attack, reference).  Class bias grid from
       prior runs.  Gate: any first-order variant matching the
       joint gradient objective's independent gains?
  g3 : seventh statistic d(theta) = E_Na||grad_x l|| - E_F||...||
       frozen on the fit 6, added to the joint audit score,
       recalibrated on cal 12; honest pass on test 12; adaptive
       attack (G1b protocol) against the new audit.
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
from uw.run_gm2 import trace_loss
from uw.run_r1 import CACHE, RUN3, load_audit
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep


def sets_(device):
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    ctx = setup_c10(device)
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)

    def rec(src, idx):
        x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                                   for i in idx]))\
            .permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([src.cifar10.targets[i] for i in idx])\
            .long().to(device)
        return x, y
    out = {'ctx': ctx}
    rng = np.random.RandomState(52801)
    cons = np.sort(rng.choice(cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    out['cons'] = rec(tset, cons)
    out['fresh'] = rec(tset, fresh)
    out['mem'] = (ctx['xF'], ctx['yF'])
    r3 = np.random.RandomState(52701)
    ret = np.sort(r3.choice(np.setdiff1d(cat_tr, fset), 30,
                            replace=False))
    out['ret'] = rec(trset, ret)
    return out


def gen_ascent(ctx, sd, score_sd, q_lim, obj_fn, iters, mem, cons,
               sign=-1, mask=None):
    """Generic audit-feasible descent on obj_fn (sign=-1 minimizes)."""
    model = ctx['model']
    cur, lr = sd, LR0
    rs = np.random.RandomState(52811)
    cons_x, cons_y = cons
    stall = 0
    for _it in range(iters):
        idx = rs.choice(len(cons_y), 60, replace=False)
        nx, ny = cons_x[idx], cons_y[idx]
        model.load_state_dict(cur)
        model.zero_grad()
        obj_fn(ctx, cur, mem, (nx, ny)).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters()
             if p.grad is not None and (mask is None
                                        or nm.startswith(mask))}
        acc = False
        for _bt in range(8):
            cand = apply_delta(cur, g, sign * lr)
            if score_sd(cand) <= q_lim:
                if obj_fn(ctx, cand, mem, (nx, ny)) \
                        < obj_fn(ctx, cur, mem, (nx, ny)) - 1e-9:
                    cur, acc = cand, True
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            break
    return cur


def ce_f(ctx, sd, mem, cons):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    mx, my = mem
    return torch.nn.functional.cross_entropy(model(mx), my)


def ce_contrast(ctx, sd, mem, cons):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    mx, my = mem
    nx, ny = cons
    return (torch.nn.functional.cross_entropy(model(mx), my)
            - 0.5 * torch.nn.functional.cross_entropy(model(nx), ny))


def trace_f_only(ctx, sd, mem, cons):
    return trace_loss(ctx, sd, mem[0], mem[1], mem[0][:1],
                      mem[1][:1])  # degenerate; replaced below


def gnorm_mean(ctx, sd, x, y):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    xg = x.clone().requires_grad_(True)
    out = model(xg)
    loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    tot = 0.0
    for i in range(len(y)):
        g = torch.autograd.grad(loss[i], xg, create_graph=True)[0][i]
        tot = tot + g.norm()
    return tot / len(y)


def labelblind(ctx, sd, S):
    def nf(x, y):
        return -grad_norms(ctx, sd, x, y)
    fm = nf(*S['mem'])
    ff = nf(*S['fresh'])
    return auc(fm, ff)


def run_g2(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    S = sets_(device)
    ctx = S['ctx']
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    starts = [p for p in starts
              if score_sd(torch.load(p, map_location=device))
              <= q20][:3]

    def t_joint(csd, sd_, mem, cons):
        return trace_loss(ctx, sd_, mem[0], mem[1], cons[0], cons[1])

    def t_fonly(csd, sd_, mem, cons):
        return gnorm_mean(ctx, sd_, mem[0], mem[1])

    def t_nonly(csd, sd_, mem, cons):
        return -gnorm_mean(ctx, sd_, cons[0], cons[1])
    arms = {'ce_f': ce_f, 'ce_contrast': ce_contrast,
            'gF_only': t_fonly, 'gN_only': t_nonly,
            'joint': t_joint}
    rows = []
    for p in starts:
        sd = torch.load(p, map_location=device)
        base = labelblind(ctx, sd, S)
        r_ = {'start': os.path.basename(p), 'base': base}
        for name, fn in arms.items():
            fin = gen_ascent(ctx, sd, score_sd, q20, fn, iters,
                             S['mem'], S['cons'])
            r_[name] = labelblind(ctx, fin, S)
            print(' g2', os.path.basename(p), name,
                  '%.3f -> %.3f' % (base, r_[name]), flush=True)
        rows.append(r_)
    # bias grid (cheap, same protocol)
    for p in starts:
        sd = torch.load(p, map_location=device)
        best = -1e9
        for b in (0.25, 0.5, 1, 2, 4):
            sd2 = {k: v.clone() for k, v in sd.items()}
            sd2['fc.bias'][3] += b
            if score_sd(sd2) <= q20:
                best = max(best, labelblind(ctx, sd2, S))
        for r_ in rows:
            if r_['start'] == os.path.basename(p):
                r_['bias'] = best
    joint_g = [r_['joint'] - r_['base'] for r_ in rows]
    rival = max(max(r_['ce_f'], r_['ce_contrast'], r_['gF_only'],
                    r_['gN_only'], r_.get('bias', -1e9)) - r_['base']
                for r_ in rows)
    v = ('G2_FIRST_ORDER_PARITY' if rival >= max(joint_g) - 0.03
         else 'G2_GRADIENT_OBJECTIVE_ADVANTAGE')
    out = {'rows': rows, 'joint_gains': joint_g,
           'best_rival_gain': rival, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, '| joint', [round(x, 3) for x in joint_g],
          '| rival %.3f' % rival, flush=True)


def run_g3(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    S = sets_(device)
    ctx = S['ctx']
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    # seventh statistic: d(theta) = E_Na gnorm - E_F gnorm, frozen
    # sign/normalization from the fit 6.
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    na = np.sort(np.random.RandomState(53101).choice(cat_te, 100,
                                                     replace=False))
    nx0 = torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in na]))\
        .permute(0, 3, 1, 2).float().to(device)
    ny0 = torch.tensor([tset.cifar10.targets[i] for i in na])\
        .long().to(device)

    def d_stat(sd):
        return float(gnorm_mean(ctx, sd, nx0, ny0)
                     - gnorm_mean(ctx, sd, ctx['xF'], ctx['yF']))
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    d_fit = [d_stat(torch.load(p, map_location=device))
             for p in fit_paths]
    d_med = float(np.median(d_fit))
    d_mad = float(np.median(np.abs(np.array(d_fit) - d_med)))
    d_sig = max(d_mad, 1e-9)

    def sc7(sd):
        mm = ctx['eval_full'](sd)
        dpen = abs(d_stat(sd) - d_med) / d_sig
        return max(sc(mm), dpen)
    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    cal7 = [sc7(torch.load(p, map_location=device))
            for p in cal_paths]
    q7 = conformal_q(cal7, 0.20)
    test_paths = (sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt')))
    honest = float(np.mean([
        sc7(torch.load(p, map_location=device)) <= q7
        for p in test_paths]))
    print(' g3 q7 %.3f honest12 %.3f' % (q7, honest), flush=True)

    def score7(sd):
        return sc7(sd)
    starts = [p for p in sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
        if score7(torch.load(p, map_location=device)) <= q7][:3]
    rows = []
    for p in starts:
        sd = torch.load(p, map_location=device)
        base = labelblind(ctx, sd, S)

        def tj(csd, sd_, mem, cons):
            return trace_loss(ctx, sd_, mem[0], mem[1],
                              cons[0], cons[1])
        fin = gen_ascent(ctx, sd, score7, q7, tj, iters, S['mem'],
                         S['cons'])
        fin_auc = labelblind(ctx, fin, S)
        ok = bool(score7(fin) <= q7)
        rows.append({'start': os.path.basename(p), 'base': base,
                     'final': fin_auc, 'pass7': ok})
        print(' g3 atk', os.path.basename(p),
              '%.3f -> %.3f pass %s' % (base, fin_auc, ok),
              flush=True)
    gains = [r_['final'] - r_['base'] for r_ in rows if r_['pass7']]
    v = ('G3_GRADIENT_AUDIT_LEAKS'
         if gains and max(gains) >= 0.05
         else 'G3_GRADIENT_AUDIT_BLOCKS')
    out = {'q7': float(q7), 'honest12': honest, 'rows': rows,
           'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('g2', 'g3'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'gate-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('Gate', cli.arm, '->', cli.outdir, flush=True)
    dict(g2=run_g2, g3=run_g3)[cli.arm](cli)


if __name__ == '__main__':
    main()
