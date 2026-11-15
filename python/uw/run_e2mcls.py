"""E2 (prereg docs/plans/2026-11-03-e2mcls-multi-class-prereg.md):
multi-class positive-domain replication. For each of 3 new classes
(dog=5, airplane=0, frog=6): full pipeline = theta0 -> 24 one-epoch
ft on keep_cls -> A6-cls audit (fit 6 / cal 12 / starts 6) ->
ce_gen encoding x6 starts under the class's own audit -> AUC CI.
Gates frozen per class: CLASS-REPLICATED (>=2/6 accepted &
CI-positive) / PARTIAL (1) / NOT-REPLICATED (0).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K, METRICS, conformal_q
from uw.run_lfpr import ce_gen, channels
from uw.run_gm import auc as rank_auc

CLASSES = {5: ('dog', 47651), 0: ('airplane', 47681), 6: ('frog', 47711)}
N_FT = 24
THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt')


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run_class(cli, device, cls, cname, seed0, theta0, tset):
    spec = dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=cls,
                pool_scheme='build_pool', k=K)
    ctx = ez.make_context(spec, device)
    model = ctx['model']
    cat_te = np.where(np.asarray(tset.cifar10.targets) == cls)[0]
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)
    keep = ctx['keep']

    ck = os.path.join(cli.outdir, 'ckpt_%s' % cname)
    os.makedirs(ck, exist_ok=True)

    def ft_path(s):
        return os.path.join(ck, 'ft_s%d.pt' % s)

    def train_or_load(s):
        p = ft_path(s)
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = ctx['fit_full'](keep, s, 1, init=theta0, cosine=False)
        torch.save(sd, p)
        return sd

    seeds = [seed0 + i for i in range(N_FT)]
    sds = [train_or_load(s) for s in seeds]
    for s in seeds:
        print('  %s ft seed %d done' % (cname, s), flush=True)

    def mvec(sd):
        return ctx['eval_full'](sd)

    fit_mm = [mvec(sd) for sd in sds[:6]]
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(sd):
        e = mvec(sd)
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_scores = [sc(sd) for sd in sds[6:18]]
    q = conformal_q(cal_scores, 0.20)
    starts = sds[18:]
    hon_all = [(se, float(sc(sd))) for se, sd in
               zip(seeds, sds)]
    hon_pass = sum(1 for _, v in hon_all if v <= q)
    print('%s: q-hat=%.3f honest pass %d/%d'
          % (cname, q, hon_pass, len(hon_all)), flush=True)

    out = {'class': cname, 'cls': cls, 'q': q,
           'honest_pass': '%d/%d' % (hon_pass, len(hon_all)),
           'rows': []}
    for se, sd0 in zip(seeds[18:], starts):
        s0 = float(sc(sd0))
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = ce_gen(model, mem, cx, sc, q, {'sd': sd0, 'n': 60})
        sf = float(sc(fin))
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        torch.save(fin, os.path.join(ck, 'enc_s%d.pt' % se))
        row = {'start_seed': se, 's6_start': s0, 's6_fin': sf,
               'pass_start': bool(s0 <= q), 'pass_fin': bool(sf <= q),
               'auc_base': base_auc, 'auc_fin': fin_auc,
               'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
        out['rows'].append(row)
        print('  %s s%d: s6 %.3f->%.3f (q %.3f) pass=%s auc '
              '%.3f->%.3f CI[%.3f,%.3f]'
              % (cname, se, s0, sf, q, row['pass_fin'],
                 base_auc, fin_auc, lo, hi), flush=True)

    surv = sum(1 for r in out['rows'] if r['pass_fin'] and r['ci_positive'])
    out['n_survive'] = surv
    out['verdict'] = ('CLASS-REPLICATED' if surv >= 2
                      else 'CLASS-PARTIAL' if surv == 1
                      else 'CLASS-NOT-REPLICATED')
    print('%s VERDICT %s (%d/6)' % (cname, out['verdict'], surv),
          flush=True)
    return out


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    theta0 = torch.load(THETA0, map_location=device)
    agg = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'classes': []}
    for cls, (cname, seed0) in CLASSES.items():
        agg['classes'].append(
            run_class(cli, device, cls, cname, seed0, theta0, tset))
    rep = sum(1 for c in agg['classes']
              if c['verdict'] == 'CLASS-REPLICATED')
    agg['replicated_classes'] = rep
    agg['overall'] = ('%d/3 new classes replicated (+cat = %d/4)'
                      % (rep, rep + 1))
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(agg, fh, indent=1, default=float)
    print('OVERALL', agg['overall'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/e2mcls-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
