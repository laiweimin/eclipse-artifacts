"""UW Step-1 low-FPR failure diagnosis (zero new training).

External verdict 2026-10-09 ('low-FPR membership encoding under the
audit'): before designing any new construction, diagnose WHERE the
current finals' working-point failure comes from.

What this script does (evaluation only, plus deterministic
regeneration of finals that were never persisted):
  A. Regenerate the six CE finals under the IDENTICAL frozen protocol
     of run_ce.py (same starts, same RandomState(53600), same
     optimizer constants, same frozen A6 audit) and SAVE them this
     time. Gradient finals already exist on disk (gate-g1b runs).
     Regeneration AUCs are cross-checked against the archived Table IX
     values; deviation is disclosed, not hidden.
  B. Aux-frozen thresholds at beta in {5, 2.5, 1}% -> actual TPR on
     the 30 members and actual FPR on the 200-record eval set, base vs
     final, for decoders: negloss / margin / conf / neggrad and their
     OUT-calibrated z-scores (difficulty calibration against the six
     fit honest models = the decoder-control arm).
  C. Per-member crossing attribution at each beta: STAY-BELOW /
     NEW (crossed) / CONFIDENT (already above, gained more) / LOST;
     plus score-gain split between base-identifiable and base-hard
     members (does the optimizer only make easy members confident?).
  D. Blocking-tail census: top-scoring non-members on aux/eval and the
     stuck members' distance-to-threshold in aux-sigma units (what a
     tail-targeted objective would need to move).
  E. Audit headroom q20 - s6 at each final (the budget a Step-2
     allocation could spend).

Outputs: research_outputs/uw/lfpr-<ts>/{verdict.json, ckpt/}.
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

BETAS = (0.05, 0.025, 0.01)
TABLE9_CE = {  # archived final AUCs (research_outputs/uw/ce-*)
    'e3-s1': 0.703, 'e3-s3': 0.781, 'e3-s4': 0.544,
    'e4-s0': 0.632, 'e4-s1': 0.607, 'e4-s2': 0.664}


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def ce_gen(model, mem, cx, score_sd, q20, iters):
    """Verbatim protocol of run_ce.gen: feasible-direction CE descent."""
    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx),
                                                          ny))

    cur, lr = None, LR0
    rs = np.random.RandomState(53600)
    return_sd = None
    # run_ce.gen mutates the start sd in place reference-wise; here we
    # take the start as cur directly (same semantics).
    cur = iters['sd']
    stall = 0
    for _it in range(iters['n']):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        ce_obj(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = apply_delta(cur, g, -lr)
            if score_sd(cand) <= q20:
                v = float(ce_obj(cand, nx, ny))
                v0 = float(ce_obj(cur, nx, ny))
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
    return_sd = cur
    return return_sd


def channels(model, sd, x, y, with_grad=True):
    """negloss / margin / conf (+ neggrad) per record, one forward."""
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        out = model(x)
        logp = -torch.nn.functional.cross_entropy(
            out, y, reduction='none')
        p = torch.softmax(out, 1)
        conf = p.gather(1, y[:, None]).squeeze(1)
        zy = out.gather(1, y[:, None]).squeeze(1)
        zc = out.clone()
        zc.scatter_(1, y[:, None], -1e18)
        margin = zy - zc.max(1).values
    ch = {'negloss': logp.cpu().numpy(),
          'margin': margin.cpu().numpy(),
          'conf': conf.cpu().numpy()}
    if with_grad:
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        loss = torch.nn.functional.cross_entropy(out, y,
                                                 reduction='none')
        gn = -np.array([
            float(torch.autograd.grad(loss[i], xg,
                                      retain_graph=True)[0][i]
                  .norm()) for i in range(len(y))])
        ch['neggrad'] = gn
    return ch


def score_sets(model, sd, sets, with_grad=True):
    """sets: dict name -> (x, y). Returns {name: {channel: np.array}}."""
    return {nm: channels(model, sd, x, y, with_grad)
            for nm, (x, y) in sets.items()}


def zcal(raw, fit_raw, eps=1e-12):
    """OUT-only difficulty calibration: z = (r - mean_fit) / std_fit."""
    arr = np.stack(fit_raw)              # (n_fit, n_records)
    mu, sd = arr.mean(0), arr.std(0)
    return (raw - mu) / (sd + eps)


def working_point(mem_s, aux_s, evl_s, beta):
    tau = float(np.quantile(aux_s, 1.0 - beta))
    return {'tau': tau,
            'tpr': float(np.mean(mem_s >= tau)),
            'fpr': float(np.mean(evl_s >= tau)),
            'n_above_aux': int(np.sum(aux_s >= tau))}


def attribution(mem_base, mem_fin, tau_base, tau_fin):
    """Per-member classification base->final."""
    out = []
    for i in range(len(mem_base)):
        b, f = float(mem_base[i]), float(mem_fin[i])
        sb, sf = b >= tau_base, f >= tau_fin
        cls = ('NEW' if (not sb) and sf else
               'LOST' if sb and (not sf) else
               'CONFIDENT' if sb and sf else 'STAY-BELOW')
        out.append({'i': i, 'base': b, 'fin': f,
                    'cls': cls, 'gain': f - b})
    return out


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
    ret = np.sort(np.random.RandomState(52701).choice(
        np.setdiff1d(cat_tr, fset), 30, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    rx = rec(trset, ret, device)
    cx = rec(tset, cons, device)

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    # ---- starts and finals ----------------------------------------
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    fe3 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s*.pt'))
    fe4 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s*.pt'))
    pairs = [('e3-s1', e3[1], fe3[0]), ('e3-s3', e3[3], fe3[1]),
             ('e3-s4', e3[4], fe3[2]), ('e4-s0', e4[0], fe4[0]),
             ('e4-s1', e4[1], fe4[1]), ('e4-s2', e4[2], fe4[2])]
    if cli.smoke:
        pairs = pairs[:1]

    # ---- fit-ensemble checkpoints (for z-calibration) ---------------
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    # verify identity against cache m1_acc (multiset match)
    cache_fit_m1 = sorted(e['m1_acc'] for e in fit)
    got_m1 = sorted(float(ctx['eval_full'](sd)['m1_acc'])
                    for sd in fit_sds)
    m1_match = (len(got_m1) == 6
                and max(abs(a - b) for a, b in
                        zip(cache_fit_m1, got_m1)) < 2e-3)

    sets = {'mem': mem, 'aux': ax_, 'eval': fx, 'ret': rx}
    raw_ch = ('negloss', 'margin', 'conf', 'neggrad')

    # fit-model per-record scores for calibration
    fit_scores = []
    for sd in fit_sds:
        fit_scores.append(score_sets(model, sd, sets))

    rows = []
    for tag, sp, fp in pairs:
        sd = torch.load(sp, map_location=device)
        # --- CE final: regenerate + save (run_ce did not persist) ----
        if cli.smoke:
            ce_fin = ce_gen(model, mem, cx, score_sd, q20,
                            {'sd': sd, 'n': 4})
        else:
            ce_fin = ce_gen(model, mem, cx, score_sd, q20,
                            {'sd': sd, 'n': 60})
        torch.save(ce_fin, os.path.join(cli.ckpt, 'ce_%s.pt' % tag))
        gr_fin = torch.load(fp, map_location=device)

        for fam, fin in (('ce', ce_fin), ('grad', gr_fin)):
            b_sc = score_sets(model, sd, sets)
            f_sc = score_sets(model, fin, sets)
            s6_b, s6_f = float(score_sd(sd)), float(score_sd(fin))
            row = {'start': tag, 'family': fam,
                   's6_base': s6_b, 's6_fin': s6_f,
                   'pass_fin': bool(s6_f <= q20),
                   'headroom': float(q20 - s6_f),
                   'wp': {}, 'attrib': {}, 'blockers': {},
                   'gain_split': {}, 'dist_census': {}}
            # overall m1 via eval_full (already computed above)
            row['m1_base'] = float(ctx['eval_full'](sd)['m1_acc'])
            row['m1_fin'] = float(ctx['eval_full'](fin)['m1_acc'])

            for ch in raw_ch:
                mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
                ab, af = b_sc['aux'][ch], f_sc['aux'][ch]
                eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
                dec = {'base': {}, 'fin': {}}
                for beta in BETAS:
                    dec['base'][str(beta)] = working_point(
                        mb, ab, eb, beta)
                    dec['fin'][str(beta)] = working_point(
                        mf, af, ef, beta)
                dec['auc_base'] = float(rank_auc(mb, eb))
                dec['auc_fin'] = float(rank_auc(mf, ef))
                row['wp'][ch] = dec
                # z-calibrated variant
                zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
                zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
                zab = zcal(ab, [fs['aux'][ch] for fs in fit_scores])
                zaf = zcal(af, [fs['aux'][ch] for fs in fit_scores])
                zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
                zef = zcal(ef, [fs['eval'][ch] for fs in fit_scores])
                zdec = {'base': {}, 'fin': {}}
                for beta in BETAS:
                    zdec['base'][str(beta)] = working_point(
                        zmb, zab, zeb, beta)
                    zdec['fin'][str(beta)] = working_point(
                        zmf, zaf, zef, beta)
                zdec['auc_base'] = float(rank_auc(zmb, zeb))
                zdec['auc_fin'] = float(rank_auc(zmf, zef))
                row['wp']['z_' + ch] = zdec

            # attribution + gain split on primary channels
            for ch in ('negloss', 'neggrad'):
                tau5b = float(np.quantile(b_sc['aux'][ch], 0.95))
                tau5f = float(np.quantile(f_sc['aux'][ch], 0.95))
                att = attribution(b_sc['mem'][ch], f_sc['mem'][ch],
                                  tau5b, tau5f)
                row['attrib'][ch] = att
                above = [a['gain'] for a in att
                         if a['base'] >= tau5b]
                below = [a['gain'] for a in att if a['base'] < tau5b]
                row['gain_split'][ch] = {
                    'n_base_above': len(above),
                    'mean_gain_above': (float(np.mean(above))
                                        if above else None),
                    'n_base_below': len(below),
                    'mean_gain_below': (float(np.mean(below))
                                        if below else None)}
                # stuck members' distance to tau in aux-sigma units
                sig_aux = float(np.std(f_sc['aux'][ch])) + 1e-12
                stuck = [(tau5f - f_sc['mem'][ch][i]) / sig_aux
                         for i in range(30)
                         if f_sc['mem'][ch][i] < tau5f]
                row['dist_census'][ch] = {
                    'n_stuck': len(stuck),
                    'sig_aux': sig_aux,
                    'within_1sig': int(sum(1 for d in stuck if d <= 1)),
                    '1_to_2sig': int(sum(1 for d in stuck
                                         if 1 < d <= 2)),
                    '2_to_4sig': int(sum(1 for d in stuck
                                         if 2 < d <= 4)),
                    'beyond_4sig': int(sum(1 for d in stuck if d > 4))}
                # blocking tail (final model, eval side)
                order = np.argsort(f_sc['eval'][ch])[::-1][:10]
                row['blockers'][ch] = {
                    'eval_top10': [int(fresh[i]) for i in order],
                    'eval_top10_scores':
                        [float(f_sc['eval'][ch][i]) for i in order],
                    'tau5': tau5f,
                    'min_member': float(f_sc['mem'][ch].min()),
                    'gap_member_min_to_tau':
                        float((tau5f - f_sc['mem'][ch].min())
                              / sig_aux)}
            # CE sanity vs archived Table IX
            if fam == 'ce':
                row['auc_archived_table9'] = TABLE9_CE[tag]
                row['auc_regen_neggrad'] = row['wp']['neggrad'][
                    'auc_fin']
                row['regen_dev'] = abs(row['auc_regen_neggrad']
                                       - TABLE9_CE[tag])
            rows.append(row)
            print(' %s %s s6 %.2f->%.2f pass=%s head %.2f | '
                  'auc(ng) %.3f->%.3f tpr5 %.2f->%.2f fpr5 %.3f | '
                  'tpr5z %.2f->%.2f'
                  % (tag, fam, s6_b, s6_f, row['pass_fin'],
                     row['headroom'],
                     row['wp']['neggrad']['auc_base'],
                     row['wp']['neggrad']['auc_fin'],
                     row['wp']['neggrad']['base']['0.05']['tpr'],
                     row['wp']['neggrad']['fin']['0.05']['tpr'],
                     row['wp']['neggrad']['fin']['0.05']['fpr'],
                     row['wp']['z_neggrad']['base']['0.05']['tpr'],
                     row['wp']['z_neggrad']['fin']['0.05']['tpr']),
                  flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'betas': list(BETAS), 'q20': float(q20),
           'fit_m1_match': bool(m1_match),
           'fit_paths': fit_paths,
           'ce_finals_regenerated': True,
           'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT written; fit_m1_match=%s' % m1_match, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'lfpr-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    cli.ckpt = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(cli.ckpt, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()
