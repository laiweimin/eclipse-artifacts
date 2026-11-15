"""C5: pairwise gradient-alignment encoding attack (prereg card
docs/plans/2026-10-16-c5pair-prereg.md, frozen).

New channel: audit metrics are all per-record; encode a signal
visible only in PAIRWISE queries (gradient cosine similarity
between F records). Arms: B0 CE baseline (reuse), B1 mu=0.1,
B2 mu=0.5. Readout: 60x59/2 pairwise cosine sim, threshold =
q95 of fresh-fresh pairs, TPR on F-F pairs.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import apply_delta, setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_lfpr import channels, rec
from uw.run_r1 import load_audit

MU = (0.1, 0.5)
ITERS = 60


def input_grad_cos(model, sd, x1, y1, x2, y2):
    """Cosine similarity of input gradients for a pair."""
    model.load_state_dict(sd)
    model.eval()

    def grad_of(x, y):
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        loss = F.cross_entropy(out, y)
        g = torch.autograd.grad(loss, xg)[0]
        return g.flatten()

    g1 = grad_of(x1, y1)
    g2 = grad_of(x2, y2)
    return float(F.cosine_similarity(g1, g2, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c5pair-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    ret = np.sort(np.random.RandomState(52701).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh[:30], device)  # 30 fresh for pairs
    cx = rec(tset, cons, device)

    _, _, _, c_aud, sig_aud, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    def pair_encode(sd_start, mu):
        """CE encoding + pairwise gradient alignment reward."""

        def obj(sd, nx, ny):
            model.load_state_dict(sd)
            model.eval()
            rm = -F.cross_entropy(model(mem[0]), mem[1])
            contrast = F.cross_entropy(model(nx), ny)
            # pairwise cosine on a sample of F pairs (10 pairs)
            rs = np.random.RandomState(53600)
            pair_sim = 0.0
            n_pairs = 10
            for _ in range(n_pairs):
                i, j = rs.choice(30, 2, replace=False)
                x1 = mem[0][i:i + 1]
                y1 = mem[1][i:i + 1]
                x2 = mem[0][j:j + 1]
                y2 = mem[1][j:j + 1]

                def g_of(x, y):
                    xg = x.clone().requires_grad_(True)
                    out = model(xg)
                    loss = F.cross_entropy(out, y)
                    return torch.autograd.grad(
                        loss, xg, create_graph=True)[0].flatten()

                g1 = g_of(x1, y1)
                g2 = g_of(x2, y2)
                pair_sim = pair_sim + F.cosine_similarity(
                    g1, g2, dim=0)
            pair_sim = pair_sim / n_pairs
            return rm.mean() - 0.5 * contrast - mu * pair_sim

        cur = sd_start
        lr = LR0
        rs = np.random.RandomState(53600)
        stall = 0
        for _it in range(ITERS):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone()
                 for nm, p in model.named_parameters()
                 if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_sd(cand) <= q20:
                    v = float(obj(cand, nx, ny))
                    v0 = float(obj(cur, nx, ny))
                    if v < v0 - 1e-9:
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

    def pair_readout(sd):
        """Full pairwise cosine matrix on 30 F + 30 fresh."""
        model.load_state_dict(sd)
        model.eval()
        all_x = torch.cat([mem[0], fx[0]], dim=0)
        all_y = torch.cat([mem[1], fx[1]], dim=0)

        def grad_of(i):
            xg = all_x[i:i + 1].clone().requires_grad_(True)
            out = model(xg)
            loss = F.cross_entropy(out, all_y[i:i + 1])
            g = torch.autograd.grad(loss, xg)[0].flatten()
            return g

        grads = [grad_of(i) for i in range(60)]
        G = torch.stack(grads)
        # cosine sim matrix
        Gn = F.normalize(G, dim=1)
        S = (Gn @ Gn.T).cpu().numpy()
        # F-F pairs (upper triangle among 0..29)
        ff = [S[i, j] for i in range(30)
              for j in range(i + 1, 30)]
        # fresh-fresh pairs (30..59)
        nn = [S[i, j] for i in range(30, 60)
              for j in range(i + 1, 60)]
        # F-fresh pairs
        fn = [S[i, j] for i in range(30)
              for j in range(30, 60)]
        tau = float(np.quantile(nn, 0.95))
        return {'ff_med': float(np.median(ff)),
                'nn_med': float(np.median(nn)),
                'fn_med': float(np.median(fn)),
                'ff_tpr_q95': float(np.mean(
                    np.array(ff) > tau)),
                'nn_fpr': float(np.mean(
                    np.array(nn) > tau)),
                'fn_rate': float(np.mean(
                    np.array(fn) > tau))}

    def zread_simple(sd):
        """Per-record z for audit-blindness check."""
        return channels(model, sd, mem[0], mem[1])

    # starts
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    starts = [('s1', e3[1]), ('s2', e3[3]), ('s3', e3[4])]

    rows = []
    # B0 baseline: reuse existing CE finals
    for pth in sorted(glob.glob(
            'research_outputs/uw/lfpr-*/ckpt/ce_e3-s*.pt'))[:3]:
        sd = torch.load(pth, map_location=device)
        pr = pair_readout(sd)
        zr = channels(model, sd, mem[0], mem[1])
        rows.append({'arm': 'B0', 'start':
                     os.path.basename(pth), 'pair': pr,
                     'negloss_med': float(
                         np.median(zr['negloss'])),
                     'pass': bool(score_sd(sd) <= q20)})
        print(' B0 %s pass=%s ff_med %.3f nn_med %.3f '
              'ff_tpr %.2f'
              % (os.path.basename(pth),
                 rows[-1]['pass'], pr['ff_med'],
                 pr['nn_med'], pr['ff_tpr_q95']),
              flush=True)

    # honest reference
    for pth in sorted(glob.glob(
            'research_outputs/uw/ew9F-20260927T001555/'
            'ckpt/ft_k30_s*.pt'))[:3]:
        sd = torch.load(pth, map_location=device)
        pr = pair_readout(sd)
        rows.append({'arm': 'HON', 'start':
                     os.path.basename(pth), 'pair': pr})
        print(' HON %s ff_med %.3f nn_med %.3f'
              % (os.path.basename(pth), pr['ff_med'],
                 pr['nn_med']), flush=True)

    # B1/B2 encoding
    for mu in MU:
        arm = 'B1' if mu == 0.1 else 'B2'
        for sname, sp in starts:
            sd0 = torch.load(sp, map_location=device)
            sdf = pair_encode(sd0, mu)
            torch.save(sdf, os.path.join(
                ck, '%s_%s.pt' % (arm, sname)))
            pr = pair_readout(sdf)
            zr = channels(model, sdf, mem[0], mem[1])
            rows.append({'arm': arm, 'start': sname,
                         'mu': mu, 'pair': pr,
                         'negloss_med': float(
                             np.median(zr['negloss'])),
                         'pass': bool(score_sd(sdf) <= q20)})
            print(' %s %s pass=%s ff_med %.3f nn_med %.3f '
                  'ff_tpr %.2f negloss_med %.2f'
                  % (arm, sname, rows[-1]['pass'],
                     pr['ff_med'], pr['nn_med'],
                     pr['ff_tpr_q95'],
                     rows[-1]['negloss_med']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # gates
    b1 = [r for r in rows if r['arm'] == 'B1']
    b2 = [r for r in rows if r['arm'] == 'B2']
    hon = [r for r in rows if r['arm'] == 'HON']
    gate = {}
    for arm_name, arm_rows in (('B1', b1), ('B2', b2)):
        if not arm_rows:
            continue
        all_pass = all(r['pass'] for r in arm_rows)
        ff_med = np.median([r['pair']['ff_med']
                            for r in arm_rows])
        nn_med = np.median([r['pair']['nn_med']
                            for r in arm_rows])
        hon_ff = np.median([r['pair']['ff_med']
                            for r in hon])
        hon_nn = np.median([r['pair']['nn_med']
                            for r in hon])
        ff_tpr = np.mean([r['pair']['ff_tpr_q95']
                          for r in arm_rows])
        signal = ff_med - nn_med
        hon_signal = hon_ff - hon_nn
        gate[arm_name] = {
            'all_pass': all_pass,
            'ff_minus_nn': float(signal),
            'hon_ff_minus_nn': float(hon_signal),
            'ff_tpr_q95': float(ff_tpr),
            'P_gate': bool(all_pass and signal >= 0.05
                           and ff_tpr >= 0.30)}
        print(' %s P-gate %s (pass=%s ff-nn=%+.3f '
              'hon=%+.3f tpr=%.2f)'
              % (arm_name, gate[arm_name]['P_gate'],
                 all_pass, signal, hon_signal, ff_tpr),
              flush=True)
    with open(os.path.join(outdir, 'gates.json'), 'w') as fh:
        json.dump(gate, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
