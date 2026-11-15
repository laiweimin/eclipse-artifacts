"""A1: band-width dose response (prereg 2026-11-12 path A).

Fixed primary ft-reference audit (fit6/cal12, conformal q at alpha=0.20).
Acceptance half-widths are scaled per component family:
  utility arm  : kappa applies to m1_acc, m2_catacc (forget side at 1)
  forget arm   : kappa applies to m3..m6 (utility at 1)
kappa in {0.5, 0.75, 1.0, 1.5}; 3 e4 starts; gradient-free feasibility
restoration before encoding when the start is infeasible.
Readout: label-blind neggrad AUC on F vs 200 never-trained, 2000-resample CI.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ey import METRICS

UTIL = ('m1_acc', 'm2_catacc')
FORG = ('m3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
import os as _os
KAPPAS = tuple(float(x) for x in _os.environ.get(
    'W_KAPPAS', '0.5,0.75,1.0,1.5').split(','))
ARMS = tuple(_os.environ.get('W_ARMS', 'utility,forget').split(','))


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 40
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    pool = ctx['pool']
    poolset = set(pool.tolist())
    mem = (ctx['xF'], ctx['yF'])
    model = ctx['model']

    _, _, _, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score(sd, ku, kf):
        m = ctx['eval_full'](sd)
        kap = {k: (ku if k in UTIL else kf) for k in METRICS}
        return max(abs(m[k] - c[k]) / max(sig[k] * kap[k], 1e-12)
                   for k in METRICS)

    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset]), 300,
        replace=False))
    cx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in cons])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in cons])
          .long().to(device))
    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(np.array([i for i in cat_te if i not in poolset]),
                     cons), 200, replace=False))
    fx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in fresh])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in fresh])
          .long().to(device))

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]), mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx), ny))

    def restore(sd, ku, kf, cap=12):
        """Gradient-free feasibility repair: random small nudges that
        reduce the scaled score. Used only when the start is infeasible."""
        cur = sd
        s = score(cur, ku, kf)
        rs = np.random.RandomState(54400)
        for _ in range(cap):
            if s <= q20:
                break
            best, bs = None, s
            sig = 2e-3 * (1.0 + 2.0 * _)
            for _t in range(4):
                cand = {k: v + torch.tensor(
                    rs.randn(*v.shape) * sig, device=v.device,
                    dtype=v.dtype) for k, v in cur.items()}
                sc_ = score(cand, ku, kf)
                if sc_ < bs:
                    best, bs = cand, sc_
            if best is None:
                break
            cur, s = best, bs
        return cur, s

    def gen(sd, ku, kf):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53600)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            ce_obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(6):
                cand = apply_delta(cur, g, -lr)
                if score(cand, ku, kf) <= q20:
                    model.load_state_dict(cur)
                    model.zero_grad()
                    if float(ce_obj(cand, nx, ny)) < \
                            float(ce_obj(cur, nx, ny)) - 1e-9:
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

    def auc_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        return rank_auc(zm, zf)

    def ci_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        rs = np.random.RandomState(54500)
        bs = [rank_auc(zm[rs.choice(len(zm), len(zm), True)], zf)
              for _ in range(2000 if not cli.smoke else 50)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rows = []
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-20260929T101715/ckpt/e4_ft_s*.pt'))[:3]
    for sd_p in e4:
        sd0 = torch.load(sd_p, map_location=device)
        base = auc_nl(sd0)
        for arm in ARMS:
            for kap in KAPPAS:
                ku, kf = (kap, 1.0) if arm == 'utility' else (1.0, kap)
                s0 = score(sd0, ku, kf)
                start = sd0
                if s0 > q20:
                    start, s0 = restore(sd0, ku, kf)
                feas = bool(s0 <= q20)
                t0 = time.time()
                if feas:
                    fF = gen(start, ku, kf)
                    gain = auc_nl(fF) - base
                    lo, hi = ci_nl(fF)
                    passed = bool(score(fF, ku, kf) <= q20)
                else:
                    gain, lo, hi, passed = float('nan'), float('nan'), \
                        float('nan'), False
                rows.append({'start': os.path.basename(sd_p), 'arm': arm,
                             'kappa': kap, 'base': base,
                             'start_feasible': feas,
                             'gain': gain, 'ci_lo': lo - base,
                             'ci_hi': hi - base, 'pass': passed,
                             'secs': round(time.time() - t0, 0)})
                print(' %s %-7s k=%.2f feas=%s gain=%s ci=[%s,%s] '
                      'pass=%s (%.0fs)'
                      % (os.path.basename(sd_p), arm, kap, feas,
                         'nan' if gain != gain else round(gain, 3),
                         'nan' if lo != lo else round(lo - base, 3),
                         'nan' if hi != hi else round(hi - base, 3),
                         passed, time.time() - t0), flush=True)

    def curve(arm):
        out = {}
        for kap in KAPPAS:
            g = [r['gain'] for r in rows
                 if r['arm'] == arm and r['kappa'] == kap
                 and r['gain'] == r['gain']]
            out[str(kap)] = float(np.mean(g)) if g else None
        return out

    cu, cf = curve('utility'), curve('forget')

    def mono(c):
        v = [c[str(k)] for k in KAPPAS if c[str(k)] is not None]
        return len(v) == len(KAPPAS) and all(
            v[i] < v[i + 1] for i in range(len(v) - 1))

    def flat(c):
        v = [c[str(k)] for k in (0.75, 1.0, 1.5) if c[str(k)] is not None]
        return len(v) == 3 and (max(v) - min(v)) <= 0.03

    if mono(cu) and not mono(cf):
        verdict = 'UTILITY-SLACK-GOVERNS'
    elif mono(cf) and not mono(cu):
        verdict = 'FORGET-SLACK-GOVERNS'
    elif flat(cu) and flat(cf):
        verdict = 'WIDTH-INDEPENDENT'
    else:
        verdict = 'PARTIAL'
    out = {'q20': float(q20), 'curves': {'utility': cu, 'forget': cf},
           'rows': rows, 'verdict': verdict}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('CURVES utility', {k: None if v is None else round(v, 3)
                             for k, v in cu.items()}, flush=True)
    print('CURVES forget ', {k: None if v is None else round(v, 3)
                             for k, v in cf.items()}, flush=True)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'width-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()
