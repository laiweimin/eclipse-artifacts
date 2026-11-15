"""UW E-Gm gate (a): does honest unlearning kill gradient-MIA at
our scale?  Feature: per-sample L2 norm of dL/dx.  Members = F
(30 deleted train records); non-members = 30 same-class test
records (never trained).  AUC on theta0, honest ft-unlearned (e3),
honest retrains (rtnew), and attack finals (orc) as reference.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch


def grad_norms(ctx, sd, x, y):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    norms = []
    for i in range(len(y)):
        g = torch.autograd.grad(loss[i], x, retain_graph=True)[0][i]
        norms.append(float(g.detach().norm()))
    return np.array(norms)


def auc(pos, neg):
    a = np.concatenate([np.array(pos), np.array(neg)])
    lab = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])
    order = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    return float((ranks[lab == 0].sum()
                  - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def run_gm(cli):
    from uw.run_ez import setup_c10
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    test = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
    # members: F (cat-class, deleted); non-members: 30 cat test records
    pool = ctx['pool']
    from uw.run_ew3 import f_deep
    from uw.run_ey import K
    fset = f_deep(pool, K)
    mem_x = ctx['xF'].clone()
    mem_y = ctx['yF'].clone()
    cat_test = np.where(np.asarray(
        test.cifar10.targets) == 3)[0]
    rng = np.random.RandomState(52001)
    nm_idx = np.sort(rng.choice(cat_test, 30, replace=False))
    nm_x, nm_y = [], []
    for i in nm_idx:
        nm_x.append(test.cifar10.data[i] / 255.0)
        nm_y.append(test.cifar10.targets[i])
    nm_x = torch.tensor(np.stack(nm_x)).permute(0, 3, 1, 2)\
        .float().to(device)
    nm_y = torch.tensor(nm_y).long().to(device)
    rows = []

    def eval_sd(name, sd):
        m = grad_norms(ctx, sd, mem_x, mem_y)
        n = grad_norms(ctx, sd, nm_x, nm_y)
        a = auc(m, n)
        rows.append({'model': name, 'auc': a,
                     'mem_med': float(np.median(m)),
                     'nonmed': float(np.median(n))})
        print(' gm', name, 'AUC %.3f (mem %.4f non %.4f)'
              % (a, np.median(m), np.median(n)), flush=True)

    eval_sd('theta0', torch.load(os.path.join(
        'research_outputs/uw/ew3-20260925T113900', 'ckpt',
        'theta0.pt'), map_location=device))
    for p in sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6]:
        eval_sd('ft_' + os.path.basename(p)[:-3],
                torch.load(p, map_location=device))
    for s in (47301, 47302, 47303):
        eval_sd('rtnew_s%d' % s, torch.load(os.path.join(
            sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[
                -1], 'ckpt', 'rtnew_s{}.pt'.format(s)),
            map_location=device))
    for j in (1, 3, 4):
        eval_sd('orc_s%d' % j, torch.load(os.path.join(
            sorted(glob.glob('research_outputs/uw/ey2-2*'))[-1],
            'ckpt', 'orc_e3_O2_s{}.pt'.format(j)), map_location=device))
    with open(os.path.join(cli.outdir, 'gm.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    t0 = rows[0]['auc']
    ft = [r['auc'] for r in rows if r['model'].startswith('ft_')]
    rt = [r['auc'] for r in rows if r['model'].startswith('rtnew')]
    orc = [r['auc'] for r in rows if r['model'].startswith('orc')]
    print('theta0 %.3f | ft med %.3f | rt med %.3f | orc med %.3f'
          % (t0, np.median(ft), np.median(rt), np.median(orc)),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    cli.outdir = cli.outdir or os.path.join(
        'research_outputs', 'uw',
        'gm-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=False)
    run_gm(cli)


if __name__ == '__main__':
    main()
