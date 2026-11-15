"""A9-z3: held-out honest FPR validation of the frozen one-sided
z-detector (four-model review ruling 2: S13's honest alarm counts were
calibration-set tail counts; an independent operating-point claim needs
honest outputs excluded from threshold AND statistic selection).

Legs:
  cifar  z-cal = ew9F 6 fit refs (frozen); T1 = mean of top-15 z
        (member direction) over the 30 F records; threshold = 95th
        percentile of T1 over the 18 honest e2/e3/e4 ft models
        (reproduced; sanity = finals detection 10/12 and honest 1/18
        must match a9z-20261101T080001). Held-out = FRESH honest ft
        models from theta0, full-epoch keep recipe verbatim from
        run_a9z.ft_arm, seeds 45301.. (disjoint from all prior seeds).
  mob    z-cal = ez1-mob z1 6 refs (frozen); threshold = 95th pct of
        T1 over the 24 honest z2..z5 ft models (reproduced; sanity =
        finals 6/6, honest 2/24 must match a9z2). Held-out = FRESH
        MobileNetV2 honest ft from ez1-mob theta0 (1-ep keep, no
        cosine, spec recipe), seeds 45401.. .

Gate (frozen before run): held-out honest alarms <= 2/12 per leg.
Everything (pass or fail) is reported and enters the paper as the
detector's held-out operating point.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10, CELLS
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels

THETA0 = 'research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt'
FIT_GLOB = ('research_outputs/uw/ew9F-20260927T001555/ckpt/'
            'ft_k30_s*.pt')
HON_GLOB = 'research_outputs/uw/ex3-20260929T101715/ckpt/e%s_ft_s*.pt'
CE_GLOB = 'research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt'
GR = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
MOB = 'research_outputs/uw/ez1-mob-20261002T080210'
EMOB = sorted(glob.glob(
    'research_outputs/uw/emob-20261026T141216/ckpt/ce_*.pt'))

CIFAR_SEEDS = list(range(45301, 45301 + 12))
MOB_SEEDS = list(range(45401, 45401 + 12))


def t1_of(model, sd, mem, mu, sg, k=15):
    z = (channels(model, sd, *mem)['neggrad'] - mu) / (sg + 1e-12)
    return float(np.mean(np.sort(z)[::-1][:k]))


def ft_arm_cifar(device, sd0, trset, yfull, keep, s):
    """Full-epoch honest ft, verbatim from run_a9z.ft_arm."""
    import torch.nn.functional as F
    from tools.models import resnet18
    torch.manual_seed(s)
    m = resnet18(num_classes=10).to(device)
    m.load_state_dict(sd0)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    rs = np.random.RandomState(s + 7)
    order = rs.permutation(len(keep))
    done = 0
    m.train()
    while done < len(keep):
        b = keep[order[done % len(keep):done % len(keep) + 100]]
        x = torch.stack([trset[int(j)][1] for j in b]).to(device)
        y = torch.tensor(yfull[b]).long().to(device)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        done += 100
    return {k2: v.detach().clone() for k2, v in m.state_dict().items()}


def run_cifar(device, n_fresh):
    import torchvision.transforms as _tf
    from tools.cifar_10 import CIFAR10
    ctx = setup_c10(device)
    model = ctx['model']
    trset = CIFAR10(train=True,
                    transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trset.cifar10.targets)
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    keep = np.setdiff1d(np.arange(ctx['n']), fset).astype(int)
    mem = (ctx['xF'], ctx['yF'])

    fit = [torch.load(p, map_location=device)
           for p in sorted(glob.glob(FIT_GLOB))]
    assert len(fit) == 6
    base = np.stack([channels(model, sd, *mem)['neggrad'] for sd in fit])
    mu, sg = base.mean(0), base.std(0)

    hon = [torch.load(p, map_location=device) for p in sum(
        [sorted(glob.glob(HON_GLOB % g)) for g in ('2', '3', '4')], [])]
    assert len(hon) == 18
    honT = [t1_of(model, sd, mem, mu, sg) for sd in hon]
    q95 = float(np.quantile(honT, 0.95))
    hon_det = int(sum(t > q95 for t in honT))

    fins = [torch.load(p, map_location=device)
            for p in sorted(glob.glob(CE_GLOB)) + GR]
    assert len(fins) == 12
    finT = [t1_of(model, sd, mem, mu, sg) for sd in fins]
    fin_det = int(sum(t > q95 for t in finT))
    print('cifar: q95=%.4f cal_honest=%d/18 finals=%d/12 (refs 1/18,'
          ' 10/12)' % (q95, hon_det, fin_det), flush=True)

    sd0 = torch.load(THETA0, map_location=device)
    freshT = []
    seeds = CIFAR_SEEDS[:n_fresh]
    for s in seeds:
        sd = ft_arm_cifar(device, sd0, trset, yfull, keep, s)
        freshT.append(t1_of(model, sd, mem, mu, sg))
        print(' fresh seed %d T1=%.3f' % (s, freshT[-1]), flush=True)
    fresh_det = int(sum(t > q95 for t in freshT))
    return {'q95': q95, 'cal_honest_det': hon_det,
            'finals_det': fin_det, 'seeds': seeds,
            'fresh_T': freshT, 'fresh_det': fresh_det,
            'n_fresh': len(freshT)}


def run_mob(device, n_fresh):
    import uw.run_ez as ez
    ctx = ez.make_context(CELLS['mob'], device)
    model = ctx['model']
    keep = ctx['keep']
    mem = (ctx['xF'], ctx['yF'])

    fit = [torch.load(p, map_location=device) for p in sorted(
        glob.glob(MOB + '/ckpt/z1_ft_s*.pt'))]
    assert len(fit) == 6
    base = np.stack([channels(model, sd, *mem)['neggrad'] for sd in fit])
    mu, sg = base.mean(0), base.std(0)

    hon = [torch.load(p, map_location=device) for p in sorted(
        glob.glob(MOB + '/ckpt/z[2345]_ft_s*.pt'))]
    assert len(hon) == 24
    honT = [t1_of(model, sd, mem, mu, sg) for sd in hon]
    q95 = float(np.quantile(honT, 0.95))
    hon_det = int(sum(t > q95 for t in honT))

    fins = [torch.load(p, map_location=device) for p in EMOB]
    assert len(fins) == 6
    finT = [t1_of(model, sd, mem, mu, sg) for sd in fins]
    fin_det = int(sum(t > q95 for t in finT))
    print('mob: q95=%.4f cal_honest=%d/24 finals=%d/6 (refs 2/24, 6/6)'
          % (q95, hon_det, fin_det), flush=True)

    sd0 = torch.load(MOB + '/ckpt/theta0.pt', map_location=device)
    freshT = []
    seeds = MOB_SEEDS[:n_fresh]
    for s in seeds:
        sd = ctx['fit_full'](keep, s, 1, init=sd0, cosine=False)
        freshT.append(t1_of(model, sd, mem, mu, sg))
        print(' fresh seed %d T1=%.3f' % (s, freshT[-1]), flush=True)
    fresh_det = int(sum(t > q95 for t in freshT))
    return {'q95': q95, 'cal_honest_det': hon_det,
            'finals_det': fin_det, 'seeds': seeds,
            'fresh_T': freshT, 'fresh_det': fresh_det,
            'n_fresh': len(freshT)}


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'n_fresh': cli.n_fresh}
    out['cifar'] = run_cifar(device, cli.n_fresh)
    out['mob'] = run_mob(device, cli.n_fresh)
    gate_c = out['cifar']['fresh_det'] <= 2
    gate_m = out['mob']['fresh_det'] <= 2
    out['gate_heldout_le_2of12'] = {'cifar': bool(gate_c),
                                    'mob': bool(gate_m)}
    out['verdict'] = ('HELDOUT-CONTROLLED' if gate_c and gate_m
                      else 'HELDOUT-EXCEEDS' if gate_c or gate_m
                      else 'HELDOUT-EXCEEDS-BOTH')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--n_fresh', default=12, type=int)
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/a9z3-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
