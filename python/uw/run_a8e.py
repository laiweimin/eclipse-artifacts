"""C-A8E P1: joint-constrained adaptive encoding (prereg:
docs/decisions/2026-11-01-method-improvement-ledger.md section 5).

ce_gen frozen protocol with the acceptance check replaced by
s_J <= q_J (rho = 0.5 shrunk covariance of the ew9F fit metric
vectors, mean-centered; q_J = conformal alpha=0.20 of the 12 cal
models — identical construction to run_a8joint). Paired starts =
the six lfpr honest starts; readouts evaluated with the same code
path as the existing marginal CE finals (loaded as the paired
reference group, no regeneration). Counter arms: 'fonly' and
'grad' objectives under the same s_J constraint.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_lfpr import (BETAS, channels, rec, score_sets,
                         working_point, zcal)

RHO = 0.5
E3 = 'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'
E4 = 'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'
CE_REF = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_obj(model, mem, cx, score_ok, q, it, obj):
    """ce_gen loop verbatim; objective in {ce, fonly, grad}."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        if obj == 'grad':
            xg = mem[0].clone().requires_grad_(True)
            xg2 = nx.clone().requires_grad_(True)
            lf = torch.nn.functional.cross_entropy(
                model(xg), mem[1], reduction='none')
            ln = torch.nn.functional.cross_entropy(
                model(xg2), ny, reduction='none')
            gf = torch.autograd.grad(lf.sum(), xg, create_graph=True
                                     )[0].flatten(1).norm(1)
            gn = torch.autograd.grad(ln.sum(), xg2, create_graph=True
                                     )[0].flatten(1).norm(1)
            return (gf - 0.5 * gn).sum()
        lf = torch.nn.functional.cross_entropy(
            model(mem[0]), mem[1])
        if obj == 'fonly':
            return lf
        ln = torch.nn.functional.cross_entropy(model(nx), ny)
        return lf - 0.5 * ln

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    traj, reason = [], 'max-iter'
    probe = None
    for itn in range(it['n']):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(loss(cur, nx, ny))
        loss(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() - lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            sj_c = score_ok(cand)
            if probe is None:
                probe = {'dJ': float(loss(cand, nx, ny)) - v0,
                         'dsJ': sj_c - score_ok(cur),
                         'lr0': lr}
            if sj_c <= q:
                v = float(loss(cand, nx, ny))
                if v < v0 - 1e-9:
                    cur, acc = cand, True
                    nacc += 1
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        traj.append({'it': itn, 'acc': acc, 'lr': lr})
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            reason = 'stall'
            break
        if nacc == 0 and itn >= 15:
            reason = 'no-improve'
            break
    return cur, {'reason': reason, 'n_accepted': nacc,
                 'probe': probe, 'traj_tail': traj[-5:]}


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
             for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    ret = np.sort(np.random.RandomState(52701).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    rx = rec(trset, ret, device)
    cx = rec(tset, cons, device)
    sets = {'mem': mem, 'aux': ax_, 'eval': fx, 'ret': rx}

    fit, cal, test, c, sig, sc, q = load_audit()
    X = np.stack([vec(e) for e in fit])
    c_j = X.mean(0)
    Sh = ((1 - RHO) * np.cov(X, rowvar=False)
          + RHO * np.diag(np.diag(np.cov(X, rowvar=False))))
    Sinv = np.linalg.inv(Sh)
    qJ = conformal_q([float((vec(e) - c_j) @ Sinv @ (vec(e) - c_j))
                      for e in cal], 0.20)
    q20 = q[0.20]
    print('qJ=%.4f q6=%.4f cond=%.0f' % (qJ, q20,
                                          np.linalg.cond(Sh)), flush=True)

    def sJ_of(sd):
        d = vec(ctx['eval_full'](sd)) - c_j
        return float(d @ Sinv @ d)

    def score_ok(sd):
        return sJ_of(sd)

    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    e3 = sorted(glob.glob(E3.replace('*', '*')))
    e4 = sorted(glob.glob(E4))
    starts = [('e3-s1', e3[1]), ('e3-s3', e3[3]), ('e3-s4', e3[4]),
              ('e4-s0', e4[0]), ('e4-s1', e4[1]), ('e4-s2', e4[2])]
    if cli.phase == 'p2a':
        starts = [('f-e3-0', e3[0]), ('f-e3-2', e3[2]),
                  ('f-e3-5', e3[5]), ('f-e4-3', e4[3]),
                  ('f-e4-4', e4[4]), ('f-e4-5', e4[5])]

    def evaluate(tag, group, sd, fin):
        b_sc = score_sets(model, sd, sets)
        f_sc = score_sets(model, fin, sets)
        row = {'tag': tag, 'group': group,
               's6_fin': float(sc(ctx['eval_full'](fin))),
               'sJ_fin': sJ_of(fin),
               'pass6': bool(sc(ctx['eval_full'](fin)) <= q20),
               'passJ': bool(sJ_of(fin) <= qJ), 'wp': {}}
        for ch in ('neggrad',):
            mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
            eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
            base, fin_auc, lo, hi = rank_ci(mf, mb, ef, eb)
            row['wp'][ch] = {'auc_base': base, 'auc_fin': fin_auc,
                             'ci': [lo, hi]}
            zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
            zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
            zab = zcal(b_sc['aux'][ch],
                       [fs['aux'][ch] for fs in fit_scores])
            zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
            zaf = zcal(f_sc['aux'][ch],
                       [fs['aux'][ch] for fs in fit_scores])
            zef = zcal(ef, [fs['eval'][ch] for fs in fit_scores])
            zdec = {'base': working_point(zmb, zab, zeb, 0.05),
                    'fin': working_point(zmf, zaf, zef, 0.05)}
            row['wp']['z_' + ch] = zdec
        rows.append(row)
        z = row['wp']['z_neggrad']
        print(' %-10s %-8s pass6=%s passJ=%s auc %.3f->%.3f '
              'CI[%.3f,%.3f] zTPR %.2f->%.2f'
              % (tag, group, row['pass6'], row['passJ'],
                 row['wp']['neggrad']['auc_base'],
                 row['wp']['neggrad']['auc_fin'],
                 row['wp']['neggrad']['ci'][0],
                 row['wp']['neggrad']['ci'][1],
                 z['base']['tpr'], z['fin']['tpr']), flush=True)

    rows = []
    metas = {}
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'qJ': float(qJ), 'q20': float(q20), 'rows': []}

    # paired reference: existing marginal CE finals (evaluated, not
    # regenerated)
    if cli.phase == 'p1':
        for (tag, sp) in (starts[:1] if cli.smoke else starts):
            fin = torch.load(sorted(glob.glob(CE_REF))[
                [t for t, _ in starts].index(tag)],
                map_location=device)
            evaluate(tag, 'ce-marginal-ref', torch.load(
                sp, map_location=device), fin)

    # main arm: joint-constrained adaptive CE encoding
    for (tag, sp) in (starts[:1] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        fin, meta = gen_obj(model, mem, cx, score_ok, qJ,
                             {'sd': sd,
                              'n': 4 if cli.smoke else 60}, 'ce')
        torch.save(fin, os.path.join(cli.outdir,
                                     'ceJ_%s.pt' % tag))
        metas[tag] = meta
        evaluate(tag, 'ce-joint', sd, fin)

    # counter arms
    if not cli.smoke and cli.phase == 'p1':
        for (tag, sp) in starts[:2]:
            sd = torch.load(sp, map_location=device)
            fin, meta = gen_obj(model, mem, cx, score_ok, qJ,
                                {'sd': sd, 'n': 60}, 'fonly')
            metas[tag + ':fonly'] = meta
            evaluate(tag, 'fonly-joint', sd, fin)
        for (tag, sp) in starts[:3]:
            sd = torch.load(sp, map_location=device)
            fin, meta = gen_obj(model, mem, cx, score_ok, qJ,
                                {'sd': sd, 'n': 60}, 'grad')
            metas[tag + ':grad'] = meta
            evaluate(tag, 'grad-joint', sd, fin)
        # 120-iteration sensitivity arm (one start)
        sd = torch.load(starts[0][1], map_location=device)
        fin, meta = gen_obj(model, mem, cx, score_ok, qJ,
                            {'sd': sd, 'n': 120}, 'ce')
        metas['e3-s1:ce120'] = meta
        evaluate('e3-s1', 'ce-joint-120', sd, fin)
    if cli.phase == 'p2a':
        for (tag, sp) in starts:
            sd = torch.load(sp, map_location=device)
            fin, meta = gen_obj(model, mem, cx, score_ok, qJ,
                                {'sd': sd, 'n': 60}, 'grad')
            metas[tag + ':grad'] = meta
            evaluate(tag, 'grad-joint', sd, fin)

    # gates (main arm vs paired marginal reference)
    ref = {r['tag']: r for r in rows if r['group'] == 'ce-marginal-ref'}
    main = [r for r in rows if r['group'] == 'ce-joint']
    g1 = sum(1 for r in main if r['passJ']
             and r['wp']['neggrad']['ci'][0] > 0)
    g2 = sum(1 for r in main if r['tag'] in ref
             and r['wp']['neggrad']['auc_fin']
             >= ref[r['tag']]['wp']['neggrad']['auc_fin'] - 0.03)
    g3 = sum(1 for r in main
             if r['wp']['z_neggrad']['fin']['tpr']
             - r['wp']['z_neggrad']['base']['tpr'] > 0.02)
    need = 1 if cli.smoke else 4
    # reviewer (3): paired gain-difference 90% CI lower bound
    diffs = [r['wp']['neggrad']['auc_fin']
             - ref[r['tag']]['wp']['neggrad']['auc_fin']
             for r in main if r['tag'] in ref]
    ci_lb = float(np.percentile(diffs, 5)) if diffs else None
    g2_ci = ci_lb is not None and ci_lb > -0.03
    # exact binomial p per gate (>= observed successes under .5)
    from math import comb
    n6 = len(main) if not cli.smoke else 1
    def pbinom(k, n):
        return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    dual = sum(1 for r in main if r['pass6'] and r['passJ'])
    surv = g1 >= need and g2_ci and g3 >= need
    verdict = ('SURVIVES-A8' if surv else
               'BLOCKED-A8' if g1 < need else 'PARTIAL')
    out['gates'] = {'G1_feasible': g1, 'G1_p': pbinom(g1, n6),
                    'G2_paired_diffs': diffs,
                    'G2_ci_lb_90': ci_lb, 'G2_noninferior': g2_ci,
                    'G3_zTPR_gt_0.02': g3, 'G3_p': pbinom(g3, n6),
                    'dual_pass_count': dual,
                    'n_main': n6,
                    'note': 'mechanism-level evidence; no '
                            'significance claim from 4/6 gates'}
    out['gen_metas'] = metas
    out['verdict'] = verdict
    out['rows'] = rows
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--phase', default='p1', choices=['p1', 'p2a', 'p2b'])
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-a8e' if cli.smoke else 'a8e',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
