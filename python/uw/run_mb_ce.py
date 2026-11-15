"""UW M-B-CE: m5 audit-vs-downstream coexistence for the six
CE (first-order) finals of the F request (review ruling 7:
Table IX's m5 evidence covered gradient finals only; the CE
finals' checkpoints exist in lfpr, so this is tabulation).

For each (start, base honest ft, CE final): the audit's m5
(logistic-probe AUC of F vs the 3000-record retain sub-sample,
deviation from center in MAD units) and the downstream -loss
AUC vs fresh non-members — identical definitions to run_mb.py
so the two tables are directly comparable.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ew3 import SUB_SEED, SUB_N, logistic_auc
from uw.run_ew3 import _ce_rows as ce_rows, _margin_rows as mrow
from uw.run_ey import K
from uw.run_ew3 import f_deep


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    rest = np.setdiff1d(cat_te,
                         np.sort(np.random.RandomState(52801).choice(
                             cat_te, 300, replace=False)))
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(ctx['n']), cat_tr), size=SUB_N,
        replace=False))
    sub_x = torch.tensor(np.stack(
        [trset.cifar10.data[i] / 255.0 for i in sub]))\
        .permute(0, 3, 1, 2).float().to(device)
    sub_y = torch.tensor(yfull[sub]).long().to(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    model = ctx['model']

    def logits(sd, x):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            return model(x).cpu().numpy()

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    ce = sorted(glob.glob(
        'research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt'))
    assert len(ce) == 6, len(ce)
    pairs = [('e3-s1', e3[1], ce[0]), ('e3-s3', e3[3], ce[1]),
             ('e3-s4', e3[4], ce[2]), ('e4-s0', e4[0], ce[3]),
             ('e4-s1', e4[1], ce[4]), ('e4-s2', e4[2], ce[5])]
    rows = []
    for tag, sp, fp in pairs:
        sb = torch.load(sp, map_location=device)
        sf = torch.load(fp, map_location=device)
        out = {}
        for nm, sd in (('base', sb), ('fin', sf)):
            zF = logits(sd, mem[0])
            zS = logits(sd, sub_x)
            ceF, ceS = ce_rows(zF, mem[1].cpu().numpy()), \
                ce_rows(zS, sub_y.cpu().numpy())
            mgF, mgS = mrow(zF, mem[1].cpu().numpy()), \
                mrow(zS, sub_y.cpu().numpy())
            m5 = logistic_auc(np.column_stack([ceF, mgF]),
                              np.column_stack([ceS, mgS]))
            out[nm] = {'m5': m5,
                       'm5_dev': abs(m5 - c['m5_adv'])
                       / max(sig['m5_adv'], 1e-12)}
        ds = rank_auc(
            -torch.nn.functional.cross_entropy(
                torch.tensor(logits(sf, mem[0])), mem[1].cpu(),
                reduction='none').numpy(),
            -torch.nn.functional.cross_entropy(
                torch.tensor(logits(sf, fx[0])), fx[1].cpu(),
                reduction='none').numpy())
        row = {'start': tag, 'base': out['base'], 'fin': out['fin'],
               'downstream_fin': ds}
        rows.append(row)
        print(' ce %s m5 %.3f->%.3f (dev %.2f->%.2f) ds %.3f'
              % (tag, out['base']['m5'], out['fin']['m5'],
                 out['base']['m5_dev'], out['fin']['m5_dev'], ds),
              flush=True)
    with open(os.path.join(cli.outdir, 'mb_ce.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('OUT', cli.outdir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/mb-ce-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
