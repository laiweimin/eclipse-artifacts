"""E-MOB: replicate the encoding attack + reference-calibrated readout
on MobileNetV2 / CIFAR-10 (prereg: docs/plans/2026-10-26-emob-erl-prereg.md).

Zero new honest training: reuses the 30 honest ft models of
ez1-mob-20261002T080210 (z1..z5 x 6 seeds, six-metric mm jsons on disk).
Fresh A6-mob audit instance: fit = z1, cal = z2+z3, starts = passing
models of z4/z5. Encoding = ce_gen frozen protocol verbatim; set
construction seeds identical to run_lfpr (cons 52801 / fresh 52501 /
aux 53304 / ret 52701). Gates G0/G1/G2 preregistered.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import CELLS, make_context
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import (BETAS, ce_gen, channels, rec, score_sets,
                         working_point, zcal)

MOB_RUN = 'research_outputs/uw/ez1-mob-20261002T080210'


def mob_audit():
    """A6-mob from archived per-model metrics (median/MAD on z1;
    conformal q on z2+z3)."""
    def load(prefix):
        out = []
        for p in sorted(glob.glob(
                os.path.join(MOB_RUN, '%s_ft_s*_mm.json' % prefix))):
            with open(p) as fh:
                out.append(json.load(fh))
        return out

    fit = load('z1')
    assert len(fit) == 6, 'fit ensemble must have 6 models'
    cal = load('z2') + load('z3')
    assert len(cal) == 12
    c = {m: float(np.median([e[m] for e in fit])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit]) - c[m]))) for m in METRICS}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    q = {a: conformal_q([sc(e) for e in cal], a)
         for a in (0.05, 0.10, 0.20, 0.30)}
    return fit, cal, c, sig, sc, q


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    """Paired AUC (member vs fresh-eval) + bootstrap CI of the gain
    over member indices (same indices for base and final)."""
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = []
    for _ in range(n):
        idx = rs.choice(len(fm), len(fm), replace=True)
        diffs.append(rank_auc(fm[idx], fe) - rank_auc(bm[idx], be))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = make_context(CELLS['mob'], device)
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

    fit, cal, c, sig, sc, q = mob_audit()
    q20 = q[0.20]
    print('A6-mob q20=%.4f (ez1 self-q20=5.4816, different split)'
          % q20, flush=True)

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    # G0: held-out honest acceptance (z5) + start selection from z4/z5
    def mm_paths(prefix):
        return sorted(glob.glob(
            os.path.join(MOB_RUN, 'ckpt', '%s_ft_s*.pt' % prefix)))

    z5_mm = [json.load(open(p)) for p in sorted(glob.glob(
        os.path.join(MOB_RUN, 'z5_ft_s*_mm.json')))]
    z5_pass = [sc(e) <= q20 for e in z5_mm]
    g0 = {'z5_pass': [bool(x) for x in z5_pass],
          'n_pass': int(sum(z5_pass))}
    print('G0 z5 honest pass %d/6' % g0['n_pass'], flush=True)

    cands = []
    for p in mm_paths('z4') + mm_paths('z5'):
        mm_p = os.path.join(
            MOB_RUN,
            os.path.basename(p)[:-3] + '_mm.json')
        with open(mm_p) as fh:
            e = json.load(fh)
        if sc(e) <= q20:
            cands.append(p)
    starts = cands[:1 if cli.smoke else 6]
    print('starts (%d): %s' % (len(starts),
          [os.path.basename(s) for s in starts]), flush=True)

    # fit-ensemble scores for z calibration + identity sanity
    fit_paths = mm_paths('z1')
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    m1_arch = sorted(e['m1_acc'] for e in fit)
    m1_got = sorted(float(ctx['eval_full'](sd)['m1_acc'])
                    for sd in fit_sds)
    m1_match = max(abs(a - b) for a, b in zip(m1_arch, m1_got)) < 2e-3
    print('fit identity match:', m1_match, flush=True)
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    rows = []
    for si, sp in enumerate(starts):
        sd = torch.load(sp, map_location=device)
        tag = 'm-s%d' % si
        ce_fin = ce_gen(model, mem, cx, score_sd, q20,
                        {'sd': sd, 'n': 4 if cli.smoke else 60})
        torch.save(ce_fin, os.path.join(cli.ckpt, 'ce_%s.pt' % tag))
        for fam, fin in (('ce', ce_fin),):
            b_sc = score_sets(model, sd, sets)
            f_sc = score_sets(model, fin, sets)
            s6_b, s6_f = float(score_sd(sd)), float(score_sd(fin))
            row = {'start_path': sp, 'start': tag, 'family': fam,
                   's6_base': s6_b, 's6_fin': s6_f,
                   'pass_fin': bool(s6_f <= q20),
                   'headroom': float(q20 - s6_f), 'wp': {}}
            row['m1_base'] = float(ctx['eval_full'](sd)['m1_acc'])
            row['m1_fin'] = float(ctx['eval_full'](fin)['m1_acc'])
            for ch in ('negloss', 'neggrad'):
                mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
                ab, af = b_sc['aux'][ch], f_sc['aux'][ch]
                eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
                base, fin_auc, lo, hi = rank_ci(mf, mb, ef, eb)
                row['wp'][ch] = {'auc_base': base, 'auc_fin': fin_auc,
                                 'ci': [lo, hi]}
                zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
                zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
                zab = zcal(ab, [fs['aux'][ch] for fs in fit_scores])
                zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
                zdec = {'base': {}, 'fin': {}}
                for beta in BETAS:
                    zdec['base'][str(beta)] = working_point(
                        zmb, zab, zeb, beta)
                    zdec['fin'][str(beta)] = working_point(
                        zmf, zab, zeb, beta)
                row['wp']['z_' + ch] = zdec
            rows.append(row)
            zt = row['wp']['z_neggrad']
            print(' %s pass=%s | auc(ng) %.3f->%.3f CI[%.3f,%.3f] | '
                  'zTPR5 %.2f->%.2f (fpr %.3f)'
                  % (tag, row['pass_fin'],
                     row['wp']['neggrad']['auc_base'],
                     row['wp']['neggrad']['auc_fin'],
                     row['wp']['neggrad']['ci'][0],
                     row['wp']['neggrad']['ci'][1],
                     zt['base']['0.05']['tpr'],
                     zt['fin']['0.05']['tpr'],
                     zt['fin']['0.05']['fpr']), flush=True)

    # preregistered gates
    g1 = sum(1 for r in rows if r['wp']['neggrad']['ci'][0] > 0)
    passing = [r for r in rows if r['pass_fin']]
    g2_pos = sum(1 for r in passing
                 if r['wp']['z_neggrad']['fin']['0.05']['tpr']
                 > r['wp']['z_neggrad']['base']['0.05']['tpr'])
    g2_max = max([r['wp']['z_neggrad']['fin']['0.05']['tpr']
                  for r in passing], default=0.0)
    verdict = ('REPLICATED' if (g1 >= (4 if not cli.smoke else 1)
                                and g2_pos >= (4 if not cli.smoke else 1)
                                and g2_max >= 0.20)
               else 'NOT-REPLICATED')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': float(q20), 'fit_m1_match': bool(m1_match),
           'gates': {'G0_honest_z5': g0,
                     'G1_auc_positive': g1,
                     'G2_zTPR_positive_passing': g2_pos,
                     'G2_max_passing_zTPR': float(g2_max)},
           'verdict': verdict, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-emob' if cli.smoke else 'emob',
            time.strftime('%Y%m%dT%H%M%S'))
    cli.ckpt = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(cli.ckpt, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
