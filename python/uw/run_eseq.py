"""E-SEQ: consecutive deletion requests on the same model
(prereg: docs/plans/2026-10-26-emob-erl-prereg.md, E-SEQ section).

G1 stale detection: request-1 encoded finals (and their honest base,
the null) scored under a fresh request-2 audit -- all should be
rejected (the model still contains F2).
G2 survival: honest request-2 processing (1ep ft on D minus F1\F2)
of the finals + base; F1 z-readout retention (paired, E-RL metric).
G3 compounding: second encoding (F2 target) from the honest-processed
final; audit pass + F2 readability + F1 residual.

Request-2 honest family: 18 F1-family honest models ft'd on keep12
(1 epoch, seeds 4740x). Audit-2: fit=ew9F', cal=e3', q-hat conformal
alpha=0.20; starts check = e4'.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep as real_f_deep
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_lfpr import (BETAS, ce_gen, rec, score_sets,
                         working_point, zcal)


def probe(model, ctx, sd, sets, fit_scores, score_sd, mem_key='mem'):
    sc_ = score_sets(model, sd, sets)
    out = {'s6': float(score_sd(sd)),
           'm1': float(ctx['eval_full'](sd)['m1_acc']), 'wp': {}}
    for ch in ('neggrad',):
        mf, ab, ef = sc_[mem_key][ch], sc_['aux'][ch], sc_['eval'][ch]
        zmf = zcal(mf, [fs[mem_key][ch] for fs in fit_scores])
        zab = zcal(ab, [fs['aux'][ch] for fs in fit_scores])
        zef = zcal(ef, [fs['eval'][ch] for fs in fit_scores])
        out['wp'][ch] = {str(b): working_point(zmf, zab, zef, b)
                         for b in BETAS}
    return out


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx1 = ez.setup_c10(device)
    model = ctx1['model']

    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx1['pool']
    f1 = real_f_deep(cat_tr, K)
    # F2: disjoint 30 from the class pool
    f2 = np.sort(np.random.RandomState(54301).choice(
        np.setdiff1d(cat_tr, f1), 30, replace=False))
    keep12 = np.setdiff1d(np.arange(ctx1['n']),
                          np.union1d(f1, f2))

    # request-2 context (module-level f_deep swap)
    ez.f_deep = lambda pool, k=None: f2
    ctx2 = ez.setup_c10(device)
    ez.f_deep = real_f_deep
    assert np.array_equal(ctx2['xF'].cpu().numpy().shape,
                          ctx1['xF'].cpu().numpy().shape)

    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    sets1 = {'mem': (ctx1['xF'], ctx1['yF']),
             'aux': rec(tset, aux, device),
             'eval': rec(tset, fresh, device)}
    sets2 = {'mem2': (ctx2['xF'], ctx2['yF']),
             'aux': sets1['aux'], 'eval': sets1['eval']}

    # ---- request-2 honest family -----------------------------------
    srcs = {'fit': sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt')),
        'cal': sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6],
        'start': sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[:6]}
    assert all(len(v) == 6 for v in srcs.values()), srcs
    fam = {}
    for grp, paths in srcs.items():
        fam[grp] = []
        for i, p in enumerate(paths):
            sd0 = torch.load(p, map_location=device)
            sd = ctx1['fit_full'](keep12, 47401 + i, 1, init=sd0)
            e = ctx2['eval_full'](sd)
            fam[grp].append((p, sd, {m: float(e[m])
                                     for m in METRICS}))
        print(' fam %s done' % grp, flush=True)
    c2 = {m: float(np.median([e[m] for _, _, e in fam['fit']]))
          for m in METRICS}
    sig2 = {m: float(np.median(np.abs(
        np.array([e[m] for _, _, e in fam['fit']]) - c2[m])))
        for m in METRICS}

    def sc2(e):
        return max(abs(e[m] - c2[m]) / max(sig2[m], 1e-12)
                   for m in METRICS)

    q2 = conformal_q([sc2(e) for _, _, e in fam['cal']], 0.20)
    e4_pass = [sc2(e) <= q2 for _, _, e in fam['start']]
    print('A6-2 q20=%.4f | e4 starts pass %d/6' % (q2, sum(e4_pass)),
          flush=True)

    def score_sd2(sd):
        return sc2(ctx2['eval_full'](sd))

    fit2_scores = [score_sets(model, sd, dict(sets1, **sets2))
                   for _, sd, _ in fam['fit']]

    # ---- objects -----------------------------------------------------
    ce1 = ('research_outputs/uw/lfpr-20261009T152330/ckpt/'
           'ce_e3-s1.pt')
    gr1 = ('research_outputs/uw/gate-g1b-20261007T102014/ckpt/'
           'fb_s1.pt')
    base1 = srcs['cal'][1]  # e3-s1 start (same pairing as run_lfpr)
    objs = [('e3-s1', 'base', base1), ('e3-s1', 'ce', ce1),
            ('e3-s1', 'gr', gr1)]

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q2': float(q2), 'e4_pass': [bool(x) for x in e4_pass],
           'rows': []}
    rows = out['rows']

    # G1: stale scoring under audit-2
    for tag, famn, path in objs:
        sd = torch.load(path, map_location=device)
        s6 = score_sd2(sd)
        rows.append({'tag': tag, 'family': famn, 'arm': 'stale',
                     's6_2': s6, 'pass2': bool(s6 <= q2)})
        print(' G1 %s stale s6_2=%.3f pass=%s' % (famn, s6,
                                                  s6 <= q2),
              flush=True)

    # G2: honest request-2 processing + F1 readout before/after
    for tag, famn, path in objs:
        sd = torch.load(path, map_location=device)
        pre = probe(model, ctx1, sd, sets1, fit2_scores, score_sd2)
        row = {'tag': tag, 'family': famn, 'arm': 'proc12',
               'pre_tpr': pre['wp']['neggrad']['0.05']['tpr'],
               'arms': {}}
        for seed in (905, 906):
            sd2 = ctx1['fit_full'](keep12, seed, 1, init=sd)
            post = probe(model, ctx1, sd2, sets1, fit2_scores,
                         score_sd2)
            row['arms']['s%d' % seed] = {
                's6_2': post['s6'], 'pass2': bool(post['s6'] <= q2),
                'tpr': post['wp']['neggrad']['0.05']['tpr'],
                'm1': post['m1']}
            print(' G2 %s s%d pass2=%s F1tpr %.2f->%.2f'
                  % (famn, seed, post['s6'] <= q2,
                     row['pre_tpr'],
                     row['arms']['s%d' % seed]['tpr']), flush=True)
            if famn == 'ce':
                arm = row['arms']['s%d' % seed]
                if arm['pass2'] and not locals().get('_enc_set'):
                    enc_from = sd2
                    _enc_set = True
        rows.append(row)

    # G3: second encoding from the honestly-processed CE model
    if not cli.smoke:
        mem2 = sets2['mem2']
        cx = rec(tset, cons, device)
        enc2 = ce_gen(model, mem2, cx, score_sd2, q2,
                      {'sd': enc_from, 'n': 4 if cli.smoke else 60})
        pre2 = probe(model, ctx2, enc_from, sets2, fit2_scores,
                     score_sd2, mem_key='mem2')
        post2 = probe(model, ctx2, enc2, sets2, fit2_scores,
                      score_sd2, mem_key='mem2')
        f1_resid = probe(model, ctx1, enc2, sets1, fit2_scores,
                         score_sd2)
        row = {'tag': 'e3-s1', 'family': 'ce', 'arm': 'encode2',
               's6_2': post2['s6'], 'pass2': bool(post2['s6'] <= q2),
               'F2_tpr_pre': pre2['wp']['neggrad']['0.05']['tpr'],
               'F2_tpr_fin': post2['wp']['neggrad']['0.05']['tpr'],
               'F1_tpr_resid': f1_resid['wp']['neggrad'][
                   '0.05']['tpr']}
        rows.append(row)
        print(' G3 encode2 pass2=%s F2tpr %.2f->%.2f F1resid %.2f'
              % (row['pass2'], row['F2_tpr_pre'], row['F2_tpr_fin'],
                 row['F1_tpr_resid']), flush=True)

    # preregistered gates
    stale = [r for r in rows if r['arm'] == 'stale']
    g1 = all(not r['pass2'] for r in stale)
    proc = [r for r in rows if r['arm'] == 'proc12'
            and r['family'] in ('ce', 'gr')]
    rets = []
    base_row = [r for r in rows if r['arm'] == 'proc12'
                and r['family'] == 'base'][0]
    for r in proc:
        den = r['pre_tpr'] - base_row['pre_tpr']
        for k, v in r['arms'].items():
            bnum = (base_row['arms'][k]['tpr']
                    - base_row['pre_tpr'])
            if den > 0:
                rets.append((v['tpr'] - base_row['arms'][k]['tpr'])
                            / den)
    g2_med = float(np.median(rets)) if rets else None
    enc_rows = [r for r in rows if r['arm'] == 'encode2']
    g3 = bool(enc_rows and enc_rows[0]['pass2']
              and enc_rows[0]['F2_tpr_fin']
              > enc_rows[0]['F2_tpr_pre'])
    out['gates'] = {'G1_stale_all_rejected': g1,
                    'G2_median_retention': g2_med,
                    'G2_washed': (g2_med is not None and g2_med < 0.5),
                    'G3_compound': g3}
    out['verdict'] = ('%s; %s; %s'
                      % ('stale-rejected' if g1 else 'stale-pass',
                         'washed' if out['gates']['G2_washed']
                         else 'survives',
                         'compound' if g3 else 'no-compound'))
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-eseq' if cli.smoke else 'eseq',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
