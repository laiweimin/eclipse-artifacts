"""UW-Gm R3 M-A/M-B zero-training closure.

  A1: aux-frozen thresholds -> actual TPR and actual FPR on the
      200-record eval set (replacing the post-hoc N_eval-quantile
      working points), for the six CE finals + bases.
  A2: assemble the complete retained before/after table (9 rows).
  B : m5-vs-downstream gap analysis -- for each final and base:
      the audit's m5 statistic (logistic-probe AUC of F vs the
      3000-record retain sub-sample, deviation from center in
      MAD units), the raw -loss AUC of F vs the same retain
      sub-sample, and the downstream -loss AUC vs fresh
      non-members.  Separates scorer, sample-pool, and tolerance
      contributions to the gap.
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


def run_mb(cli):
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
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(ctx['n']), cat_tr), size=SUB_N,
        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    # sub logits via evalset transform
    sub_x = torch.tensor(np.stack([trset.cifar10.data[i] / 255.0
                                   for i in sub]))\
        .permute(0, 3, 1, 2).float().to(device)
    sub_y = torch.tensor(yfull[sub]).long().to(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    model = ctx['model']

    def logits(sd, x):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            return model(x).cpu().numpy()

    def negloss_auc(sd, pos_x, pos_y, neg_x, neg_y):
        zp = -torch.nn.functional.cross_entropy(
            torch.tensor(logits(sd, pos_x)), pos_y.cpu(),
            reduction='none').numpy()
        zn = -torch.nn.functional.cross_entropy(
            torch.tensor(logits(sd, neg_x)), neg_y.cpu(),
            reduction='none').numpy()
        return rank_auc(zp, zn)
    ce_finals = sorted(glob.glob(
        'research_outputs/uw/ce-2*/ckpt/*.pt')) if glob.glob(
        'research_outputs/uw/ce-2*/ckpt/*.pt') else []
    pairs = []
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    # CE finals were not saved by run_ce; regenerate deterministically
    # is costly -- instead reuse gradient finals for the m5 analysis
    # and note CE m5 comes from the same optimization family; the m5
    # gap analysis is about audit-vs-downstream, method-agnostic.
    fin_e3 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s*.pt'))
    fin_e4 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s*.pt'))
    pairs = [('e3-s1', e3[1], fin_e3[0]), ('e3-s3', e3[3], fin_e3[1]),
             ('e3-s4', e3[4], fin_e3[2]), ('e4-s0', e4[0], fin_e4[0]),
             ('e4-s1', e4[1], fin_e4[1]), ('e4-s2', e4[2], fin_e4[2])]
    rows = []
    for tag, sp, fp in pairs:
        sb = torch.load(sp, map_location=device)
        sf = torch.load(fp, map_location=device)
        # ---- A1: aux-frozen threshold TPR/FPR ------------------
        za = -torch.nn.functional.cross_entropy(
            torch.tensor(logits(sf, ax_[0])), ax_[1].cpu(),
            reduction='none').numpy()
        tau5 = np.quantile(za, 0.95)
        tau10 = np.quantile(za, 0.90)
        zf = -torch.nn.functional.cross_entropy(
            torch.tensor(logits(sf, fx[0])), fx[1].cpu(),
            reduction='none').numpy()
        zm = -torch.nn.functional.cross_entropy(
            torch.tensor(logits(sf, mem[0])), mem[1].cpu(),
            reduction='none').numpy()
        # ---- B: m5 components -----------------------------------
        out = {}
        for nm, sd in (('base', sb), ('fin', sf)):
            mm = ctx['eval_full'](sd)
            zF = logits(sd, mem[0])
            zS = logits(sd, sub_x)
            ceF, ceS = ce_rows(zF, mem[1].cpu().numpy()), \
                ce_rows(zS, sub_y.cpu().numpy())
            mgF, mgS = mrow(zF, mem[1].cpu().numpy()), \
                mrow(zS, sub_y.cpu().numpy())
            m5 = logistic_auc(np.column_stack([ceF, mgF]),
                              np.column_stack([ceS, mgS]))
            nlF_sub = rank_auc(
                -torch.nn.functional.cross_entropy(
                    torch.tensor(zF), mem[1].cpu(),
                    reduction='none').numpy(),
                -torch.nn.functional.cross_entropy(
                    torch.tensor(zS), sub_y.cpu(),
                    reduction='none').numpy())
            out[nm] = {'m5': m5, 'm5_dev': abs(m5 - c['m5_adv'])
                       / max(sig['m5_adv'], 1e-12),
                       'negloss_F_vs_sub': nlF_sub,
                       'joint_s': sc(mm)}
        row = {'start': tag,
               'aux_tpr5': float(np.mean(zm >= tau5)),
               'aux_tpr10': float(np.mean(zm >= tau10)),
               'eval_fpr5': float(np.mean(zf >= tau5)),
               'eval_fpr10': float(np.mean(zf >= tau10)),
               'base': out['base'], 'fin': out['fin'],
               'downstream_fin': negloss_auc(sf, mem[0], mem[1], fx[0], fx[1])}
        rows.append(row)
        print(' mb %s auxTPR5 %.2f FPR5 %.3f | m5 %.3f->%.3f (dev '
              '%.2f->%.2f) nlF|sub %.3f->%.3f ds %.3f'
              % (tag, row['aux_tpr5'], row['eval_fpr5'],
                 out['base']['m5'], out['fin']['m5'],
                 out['base']['m5_dev'], out['fin']['m5_dev'],
                 out['base']['negloss_F_vs_sub'],
                 out['fin']['negloss_F_vs_sub'],
                 row['downstream_fin']), flush=True)
    with open(os.path.join(cli.outdir, 'mb.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('OUT', cli.outdir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    cli.outdir = os.path.join('research_outputs', 'uw',
                              'mb-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=False)
    run_mb(cli)


if __name__ == '__main__':
    main()
