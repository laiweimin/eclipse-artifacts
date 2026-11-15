"""UW Step-2 three-arm construction comparison under the frozen A6
audit (external verdict 2026-10-09, Step 2; zero new training beyond
gradient-cheap encoding steps, identical optimizer/budget/starts).

Step-1 diagnosis (research_outputs/uw/lfpr-20261009T152330 and
lfprz-*): the low-FPR bottleneck is DECODER SCALE first (raw channels
read TPR@5% ~0 while a reference-calibrated z-decoder reads 3-18
identified records), and second ALLOCATION (12-27 stuck members sit
within 0.5 sigma_z of the z-threshold on lifting finals; mean CE
spends budget on already-identified members and the N_c contrast).

Arms (same six starts, same frozen audit q20, same optimizer
constants as run_ce.gen; one final model per run, fixed return rule):
  A1 mean contrastive CE          -- regenerated finals from run_lfpr
  A2 raw-space tail objective     -- hinge [t5_raw + kappa - r_i]_+ on
                                     all members, r = negloss, threshold
                                     from a 60-record held-out split of
                                     N_c (not in the contrast gradient);
  A3 calibrated-space tail        -- hinge [t5_z + kappa - z_i]_+ ,
                                     z_i = (r_i - mu_i)/sigma_i with
                                     mu/sigma frozen from the provider
                                     reference ensemble (6 honest ft
                                     models), i.e. difficulty-weighted;
                                     hinge itself performs record
                                     selection (no gradient on already
                                     crossed members).
Both A2/A3 keep the -0.5 E_{N_c} CE contrast term of Eq. (1).

Preregistered primary readout (frozen before running A2/A3):
  z_negloss decoder, threshold at beta=5% frozen on the 100-record
  aux set, PAIRED increment (final - base start) in TPR, actual FPR
  measured on the 200-record eval set. Secondary: z_neggrad.
  GATE (per start): paired TPR increment >= +0.10 AND eval FPR@5%
  <= 0.08 AND audit pass. Utility (m1 drop vs base, m2) is reported
  per arm and NOT folded into the gate -- the audit's own bands are
  the system's declared utility constraint. JOINT SUCCESS = gate.
  matched honest selection baseline (max z-TPR over 12 honest
  candidates: 6 e3/e4 starts + 6 fit models).

Outputs: research_outputs/uw/tail-<ts>/{verdict.json, ckpt/}.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal

KAPPA = 0.1          # margin kappa in the hinge (natural units)
BETA = 0.05          # tail target FPR level for construction
N_THR = 60           # N_c split reserved for threshold estimation
BETA_EVAL = (0.05, 0.025, 0.01)


def gen_arm(model, mem, cx_grad, cx_thr, score_sd, q20, arm, start,
            ref_mu, ref_sig, ref_mu_thr, ref_sig_thr, n_iter,
            freeze_t=False):
    """Feasible-direction descent; arm in {'rawtail', 'ztail'}.

    freeze_t=True (bugfix variant, tail2-*): the construction
    threshold is computed ONCE from the start model and held constant.
    Rationale (probe_tail2 finding): ranking-based statistics jump
    discontinuously under tiny feasible steps (s6 moved 4.21->2.98 on
    a ~6e-3-norm step), so recomputing the 60-record quantile every
    evaluation makes the evaluated objective disagree with the
    descended one. The provider commits to a target threshold before
    optimizing; evaluation thresholds remain aux-frozen.
    """
    mg_x, mg_y = mem
    thr_x, thr_y = cx_thr

    def r_scores(sd, x, y):
        # model state is already loaded by the caller (obj); a
        # reload here would bump weight versions between forward
        # and backward and break autograd
        with torch.no_grad():
            out = model(x)
        return -torch.nn.functional.cross_entropy(out, y,
                                                  reduction='none')

    t_const = None
    if freeze_t:
        model.load_state_dict(start)
        model.eval()
        thr0 = r_scores(start, thr_x, thr_y)
        if arm == 'rawtail':
            t_const = float(torch.quantile(thr0, 1.0 - BETA))
        else:
            t_const = float(torch.quantile(
                (thr0 - torch.tensor(ref_mu_thr, device=thr0.device))
                / torch.tensor(ref_sig_thr, device=thr0.device),
                1.0 - BETA))

    def obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        rm = -torch.nn.functional.cross_entropy(model(mg_x),
                                                mg_y,
                                                reduction='none')
        contrast = torch.nn.functional.cross_entropy(model(nx), ny)
        if arm == 'rawtail':
            if freeze_t:
                t = torch.tensor(t_const, device=rm.device)
            else:
                thr = r_scores(sd, thr_x, thr_y)
                t = torch.quantile(thr, 1.0 - BETA)
            hinge = torch.clamp(t.detach() + KAPPA - rm, min=0).mean()
        else:
            z = (rm - torch.tensor(ref_mu, device=rm.device)) \
                / torch.tensor(ref_sig, device=rm.device)
            if freeze_t:
                t = torch.tensor(t_const, device=rm.device)
            else:
                thr = r_scores(sd, thr_x, thr_y)
                t = torch.quantile(
                    (thr - torch.tensor(ref_mu_thr,
                                        device=thr.device))
                    / torch.tensor(ref_sig_thr, device=thr.device),
                    1.0 - BETA)
            hinge = torch.clamp(t.detach() + KAPPA - z, min=0).mean()
        return hinge - 0.5 * contrast

    cur = start
    lr = LR0
    rs = np.random.RandomState(53600)
    stall = 0
    for _it in range(n_iter):
        idx = rs.choice(len(cx_grad[1]), 60, replace=False)
        nx, ny = cx_grad[0][idx], cx_grad[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        obj(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
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


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
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
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    cx = rec(tset, cons, device)
    cx_grad = (cx[0][:len(cons) - N_THR], cx[1][:len(cons) - N_THR])
    cx_thr = (cx[0][len(cons) - N_THR:], cx[1][len(cons) - N_THR:])

    _, _, _, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    # provider reference ensemble (same honest distribution; the
    # provider-known calibration condition of the paper's threat
    # model) -- frozen per-record mu/sigma on members and the
    # threshold split of N_c, negloss channel
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    sets_ref = {'mem': mem, 'thr': cx_thr}
    fit_scores = [{nm: channels(model, sd, x, y, with_grad=False)
                   for nm, (x, y) in sets_ref.items()}
                  for sd in fit_sds]
    global ref_mu, ref_sig, ref_mu_thr, ref_sig_thr
    m_arr = np.stack([fs['mem']['negloss'] for fs in fit_scores])
    t_arr = np.stack([fs['thr']['negloss'] for fs in fit_scores])
    ref_mu, ref_sig = m_arr.mean(0), m_arr.std(0) + 1e-12
    ref_mu_thr, ref_sig_thr = t_arr.mean(0), t_arr.std(0) + 1e-12

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    pairs = [('e3-s1', e3[1]), ('e3-s3', e3[3]), ('e3-s4', e3[4]),
             ('e4-s0', e4[0]), ('e4-s1', e4[1]), ('e4-s2', e4[2])]
    if cli.smoke:
        pairs = pairs[:1]
    ce_src = {os.path.basename(p)[3:-3]: p for p in sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:]}

    fit_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_eval = [{nm: channels(model, sd, x, y)
                 for nm, (x, y) in fit_sets.items()} for sd in fit_sds]

    def zread(sd):
        """z_negloss / z_neggrad working points for one model."""
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in fit_sets.items()}
        out = {}
        for ch in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][ch],
                      [fs['mem'][ch] for fs in fit_eval])
            za = zcal(s['aux'][ch], [fs['aux'][ch] for fs in fit_eval])
            ze = zcal(s['eval'][ch],
                      [fs['eval'][ch] for fs in fit_eval])
            wp = {}
            for b in BETA_EVAL:
                tau = float(np.quantile(za, 1.0 - b))
                wp[str(b)] = {'tau': tau,
                              'tpr': float(np.mean(zm >= tau)),
                              'fpr': float(np.mean(ze >= tau)),
                              'n_id': int(np.sum(zm >= tau))}
            wp['auc'] = float(rank_auc(zm, ze))
            out['z_' + ch] = wp
        return out

    # honest selection baseline: max z_negloss TPR@5% over 12 honest
    # candidates (6 starts + 6 fit models)
    hon = []
    for tag, sp in pairs:
        hon.append(zread(torch.load(sp, map_location=device))
                   ['z_negloss']['0.05']['tpr'])
    for sd in fit_sds:
        hon.append(zread(sd)['z_negloss']['0.05']['tpr'])
    sel_baseline = float(max(hon))

    global sd0
    rows = []
    row_m1_base = {}
    for tag, sp in pairs:
        sd = torch.load(sp, map_location=device)
        row_m1_base[tag] = float(ctx['eval_full'](sd)['m1_acc'])
        base_z = zread(sd)
        finals = {}
        finals['A1_meanCE'] = torch.load(ce_src[tag],
                                         map_location=device)
        n_it = 4 if cli.smoke else 60
        finals['A2_rawtail'] = gen_arm(
            model, mem, cx_grad, cx_thr, score_sd, q20, 'rawtail',
            sd, ref_mu, ref_sig, ref_mu_thr, ref_sig_thr, n_it,
            cli.frozen_thr)
        finals['A3_ztail'] = gen_arm(
            model, mem, cx_grad, cx_thr, score_sd, q20, 'ztail',
            sd, ref_mu, ref_sig, ref_mu_thr, ref_sig_thr, n_it,
            cli.frozen_thr)
        for arm, fin in finals.items():
            torch.save(fin, os.path.join(
                cli.ckpt, '%s_%s.pt' % (arm, tag)))
            fz = zread(fin)
            s6 = float(score_sd(fin))
            mm = ctx['eval_full'](fin)
            m1, m2 = float(mm['m1_acc']), float(mm['m2_catacc'])
            b5 = base_z['z_negloss']['0.05']
            f5 = fz['z_negloss']['0.05']
            # gate utility clause = the system's declared constraint
            # (audit pass); m1/m2 cost is reported alongside, not
            # hidden inside the gate
            gate = (f5['tpr'] - b5['tpr'] >= 0.10
                    and f5['fpr'] <= 0.08
                    and s6 <= q20)
            row = {'start': tag, 'arm': arm, 's6': s6,
                   'pass': bool(s6 <= q20), 'm1': m1, 'm2': m2,
                   'm1_base': row_m1_base[tag],
                   'base_z': base_z, 'fin_z': fz,
                   'gate': bool(gate)}
            rows.append(row)
            print(' %s %-10s s6 %.3f pass=%s m1 %.4f (d%+.4f) | '
                  'zTPR5 %.2f->%.2f (d%+.2f) fpr %.3f | zTPR2.5 '
                  '%.2f->%.2f | zTPR1 %.2f->%.2f | GATE %s'
                  % (tag, arm, s6, row['pass'], m1,
                     m1 - row_m1_base[tag],
                     b5['tpr'], f5['tpr'], f5['tpr'] - b5['tpr'],
                     f5['fpr'],
                     base_z['z_negloss']['0.025']['tpr'],
                     fz['z_negloss']['0.025']['tpr'],
                     base_z['z_negloss']['0.01']['tpr'],
                     fz['z_negloss']['0.01']['tpr'],
                     gate), flush=True)
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': float(q20), 'kappa': KAPPA, 'beta': BETA,
           'sel_baseline_zTPR5': sel_baseline,
           'honest_candidates_zTPR5': [round(h, 3) for h in hon],
           'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    joint = {}
    for arm in ('A1_meanCE', 'A2_rawtail', 'A3_ztail'):
        sub = [r for r in rows if r['arm'] == arm]
        joint[arm] = {'gate': sum(1 for r in sub if r['gate']),
                      'n': len(sub)}
    print('JOINT', joint, 'sel_baseline %.2f' % sel_baseline,
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--frozen-thr', action='store_true',
                    help='bugfix variant: freeze construction '
                         'threshold at the start model (tail2-*)')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') \
        + ('tail2-' if cli.frozen_thr else 'tail-') \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    cli.ckpt = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(cli.ckpt, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()
