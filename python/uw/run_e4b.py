"""E4b (prereg docs/plans/2026-11-04-e4-closing-batch-prereg.md, arm B):
Tier-2 z-readout + persistence on the frog finals (strongest class
replication). Reuses e2mcls-20261103T220400/ckpt_frog (fit refs
47651-56 + encoded finals 47729-34). z calibrated against the class's
own fit-6 references; thresholds frozen on the aux set (95th
percentile, primary-protocol analogue); persistence = one epoch of
benign fine-tuning on keep (seeds 47801/47802) applied to finals and
bases. Gates: FROG-Z-POSITIVE (>=2/6 z-TPR increment >= 0.10);
FROG-WASHED (median retention <= 0).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K
from uw.run_lfpr import channels

SRC = ('research_outputs/uw/e2mcls-20261103T220400/ckpt_frog')
FIT = [47711 + i for i in range(6)]
STARTS = [47729 + i for i in range(6)]
FT_SEEDS = (47801, 47802)
CLS = 6


def rec(src_ds, idx, device):
    x = torch.tensor(np.stack([src_ds.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src_ds.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    spec = dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=CLS,
                pool_scheme='build_pool', k=K)
    ctx = ez.make_context(spec, device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == CLS)[0]
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    keep = ctx['keep']
    yfull = ctx['yfull']

    def load(name):
        return torch.load(os.path.join(SRC, name),
                          map_location=device)

    fit_sds = [load('ft_s%d.pt' % s) for s in FIT]
    # per-set reference normalization (primary wp protocol): F, aux,
    # and fresh each use their own records' mu/sigma from the refs
    refF = np.stack([channels(model, sd, *mem)['neggrad']
                     for sd in fit_sds])
    refA = np.stack([channels(model, sd, ax_[0], ax_[1])['neggrad']
                     for sd in fit_sds])
    refX = np.stack([channels(model, sd, fx[0], fx[1])['neggrad']
                     for sd in fit_sds])
    muF, sgF = refF.mean(0), refF.std(0)
    muA, sgA = refA.mean(0), refA.std(0)
    muX, sgX = refX.mean(0), refX.std(0)

    def zset(sd, x, y, mu, sg):
        return (channels(model, sd, x, y)['neggrad'] - mu) / (sg + 1e-12)

    def tau_of(sd):
        za = zset(sd, ax_[0], ax_[1], muA, sgA)
        return float(np.quantile(za, 0.95))

    def tpr(sd):
        z = zset(sd, mem[0], mem[1], muF, sgF)
        return float(np.mean(z >= tau_of(sd)))

    def fpr(sd):
        zx = zset(sd, fx[0], fx[1], muX, sgX)
        return float(np.mean(zx >= tau_of(sd)))

    def ft_arm(sd0, s):
        import torch.nn.functional as F
        torch.manual_seed(s)
        m = ez.make_context(spec, device)['model']
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        rs = np.random.RandomState(s + 7)
        order = rs.permutation(len(keep))
        done = 0
        m.train()
        tr = CIFAR10(train=True,
                     transform=_tf.Compose([_tf.ToTensor()]))
        while done < len(keep):
            b = keep[order[done % len(keep):done % len(keep) + 100]]
            x = torch.tensor(np.stack(
                [tr.cifar10.data[int(j)] / 255.0 for j in b]))\
                .permute(0, 3, 1, 2).float().to(device)
            y = torch.tensor(yfull[b]).long().to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
            done += 100
        return {k: v.detach().clone() for k, v in
                m.state_dict().items()}

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'tau_note': 'per-model aux q95 (primary wp protocol)',
           'rows': []}
    for s in STARTS:
        base_sd = load('ft_s%d.pt' % s)
        fin_sd = load('enc_s%d.pt' % s)
        tb, tf_ = tpr(base_sd), tpr(fin_sd)
        row = {'start': s, 'tpr_base': tb, 'tpr_fin': tf_,
               'inc': tf_ - tb, 'fpr_fin': fpr(fin_sd)}
        if tf_ - tb > 0.02:
            rets = []
            for fs in FT_SEEDS:
                f_ft = ft_arm(fin_sd, fs)
                b_ft = ft_arm(base_sd, fs)
                num = tpr(f_ft) - tpr(b_ft)
                rets.append(num / max(tf_ - tb, 1e-9))
            row['retention'] = rets
        out['rows'].append(row)
        print('frog s%d z-TPR %.3f->%.3f (inc %.3f) FPR %.3f%s'
              % (s, tb, tf_, row['inc'], row['fpr_fin'],
                 ' ret=%s' % [round(r, 2) for r in row['retention']]
                 if 'retention' in row else ''), flush=True)

    pos = sum(1 for r in out['rows'] if r['inc'] >= 0.10)
    rets = [x for r in out['rows'] for x in r.get('retention', [])]
    med = float(np.median(rets)) if rets else None
    out['n_zpos'] = pos
    out['z_verdict'] = ('FROG-Z-POSITIVE' if pos >= 2
                        else 'FROG-Z-WEAK')
    out['median_retention'] = med
    out['persistence_verdict'] = ('FROG-WASHED'
                                  if med is not None and med <= 0
                                  else 'FROG-PERSISTS' if med is not None
                                  else 'N/A')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('Z:', out['z_verdict'], '(%d/6>=0.10) | PERSISTENCE:'
          % pos, out['persistence_verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/frogz-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
