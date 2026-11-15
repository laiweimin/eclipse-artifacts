"""UW R2: solidifying experiments after the R1 verdict (frozen prereg
docs/plans/2026-10-02-uw-r2-solidify-preregistration.md).

Arms:
  nocon     : conservative non-oracle commit -- ascend under
              est_s <= lam * est_q (NO2 estimate), lam in
              {0.4, 0.6, 0.8}; per-start true pass + proxy-commit
              (best-O2 final) outcome.
  baselines : (i) last-layer (fc-only) ascent under the true audit;
              (ii) naive F-finetune of passing starts (1/2/5/10
              steps); (iii) O1 finals cross-evaluated on the O2
              subset (archived finals, forward only).
  c100loop  : class-bias baseline + combined per-record score on
              the ez7 CIFAR-100 unit (closed loop, reviewer VIIIb).
  flipref   : flips vs own start / pool median / best honest
              candidate + per-class test-accuracy differential on
              archived passing O2 finals (forward only).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import (GATE, K, conformal_q, gain_of, per_record_margin,
                       record_subsets)
from uw.run_ew3b import LR0, lr_update
from uw.run_ez import (CELLS, apply_delta, make_context, obj_val,
                       run_attack, setup_c10)
from uw.run_ey2 import mk_audit
from uw.run_r1 import CACHE, RUN3, load_audit

E3 = 'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'
EY2 = None  # resolved at run time


def ey2_dir():
    return sorted(glob.glob('research_outputs/uw/ey2-2*'))[-1]


def theta_ctx(ctx):
    import torch
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=ctx['xF'].device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    return sd0, th_m, bot, top


def commit_of(outdir, prefix, obj):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(
        outdir, '{}_{}_s*.json'.format(prefix, obj))))]
    if not recs:
        return None
    best = max(r['best'] for r in recs)
    c = [r for r in recs if r['best'] == best][0]
    return {'committed_best': best,
            'committed_pass': c.get('true_pass', c.get('est_pass')),
            'per_start_pass': [r.get('true_pass', r.get('est_pass'))
                               for r in recs]}


def run_nocon(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    n_starts = 1 if cli.smoke else 3
    ctx = setup_c10(device)
    _, _, _, _, _, sc, q = load_audit()
    q20 = q[0.20]
    ens = json.load(open(CACHE))['ens_mm']
    est = mk_audit(ens['e3'], ens['e4'])
    _, th_m, bot, top = theta_ctx(ctx)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in ('O1', 'O2', 'O3')}
    starts = sorted(glob.glob(E3))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    def est_sd(sd):
        return est[3](ctx['eval_full'](sd))

    out = {'est_q': float(est[2]), 'true_q': float(q20), 'lams': {}}
    for lam in ((0.6,) if cli.smoke else (0.4, 0.6, 0.8)):
        pref = 'lam' + str(lam).replace('.', '')
        res = run_attack(
            ctx['model'], ctx['xF'], ctx['yF'], bot, top,
            ctx['margins_of'], est_sd, lam * est[2], cli.outdir, ck,
            pref, starts, mgs, th_objs, objs=('O2',),
            n_starts=n_starts, iters=iters,
            judged=lambda sd: (sc(ctx['eval_full'](sd)),
                               sc(ctx['eval_full'](sd)) <= q20))
        out['lams'][str(lam)] = {
            'res': res, 'commit': commit_of(cli.outdir, pref, 'O2')}
        print(' lam', lam, 'best_gp', round(res['O2']['best_gain_pass'],
                                            3), 'commit',
              out['lams'][str(lam)]['commit'], flush=True)
    works = False
    for lam, d in out['lams'].items():
        cm = d['commit']
        if cm and cm['committed_pass'] and d['res']['O2'][
                'best_gain_pass'] >= GATE:
            works = works or True
    out['verdict'] = 'CONSERVATIVE_WORKS' if works else 'NOT_RESCUED'
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def run_baselines(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    ctx = setup_c10(device)
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    all_starts = sorted(glob.glob(E3))
    passing = [p for p in all_starts
               if sc(ctx['eval_full'](torch.load(p, map_location=device)))
               <= q20][:3 if not cli.smoke else 1]
    o2s = [obj_val(ctx['margins_of'](torch.load(p, map_location=device)),
                   'O2', bot, top) for p in all_starts]
    med, sel = float(np.median(o2s)), max(o2s)
    out = {'starts': [os.path.basename(p) for p in passing]}
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    bot_t = torch.as_tensor(bot, device=device)

    # (i) last-layer ascent
    def ll_ascent(sd):
        cur, lr = sd, LR0
        best = obj_val(ctx['margins_of'](cur), 'O2', bot, top)
        stall = 0
        for _it in range(iters):
            model.load_state_dict(cur)
            model.eval()
            z = model(xF)
            zy = z.gather(1, yF[:, None]).squeeze(1)
            zc = z.clone()
            zc.scatter_(1, yF[:, None], -1e18)
            loss = (zy - zc.max(1).values)[bot_t].mean()
            model.zero_grad()
            loss.backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters()
                 if p.grad is not None and nm.startswith('fc.')}
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                if sc(ctx['eval_full'](cand)) <= q20:
                    v = obj_val(ctx['margins_of'](cand), 'O2', bot, top)
                    if v > best + 1e-7:
                        cur, best, acc = cand, v, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > 15:
                break
        return cur, best

    ll = []
    for p in passing:
        sd = torch.load(p, map_location=device)
        fin, best = ll_ascent(sd)
        mm = ctx['eval_full'](fin)
        ll.append({'start': os.path.basename(p), 'best': best,
                   'pass': bool(sc(mm) <= q20),
                   'gain': float(gain_of(best, sel, med, th_o2))})
        print(' lastlayer', os.path.basename(p), round(ll[-1]['gain'], 3),
              ll[-1]['pass'], flush=True)
    out['lastlayer'] = ll

    # (ii) naive F-finetune
    fft = []
    for p in passing:
        for steps in ((1, 5) if cli.smoke else (1, 2, 5, 10)):
            sd = ctx['fit_full'](ctx['fset'], 47000 + steps, steps,
                                 init=torch.load(p, map_location=device),
                                 cosine=False)
            mm = ctx['eval_full'](sd)
            v = obj_val(ctx['margins_of'](sd), 'O2', bot, top)
            fft.append({'start': os.path.basename(p), 'steps': steps,
                        'pass': bool(sc(mm) <= q20), 'o2': v,
                        'gain': float(gain_of(v, sel, med, th_o2))})
        print(' fft', os.path.basename(p),
              [(r['steps'], round(r['gain'], 2), r['pass'])
               for r in fft if r['start'] == os.path.basename(p)],
              flush=True)
    out['fft'] = fft

    # (iii) O1 finals cross-evaluated on the O2 subset
    cross = []
    for pth in sorted(glob.glob(os.path.join(
            ey2_dir(), 'ckpt', 'orc_e3_O1_s*.pt'))):
        fin = torch.load(pth, map_location=device)
        mg = ctx['margins_of'](fin)
        v = obj_val(mg, 'O2', bot, top)
        cross.append({'final': os.path.basename(pth),
                      'o2_val': v,
                      'gain_as_o2': float(gain_of(v, sel, med, th_o2)),
                      'F_acc': float((mg > 0).mean())})
    out['o1_cross'] = cross
    print(' o1_cross gains', [round(r['gain_as_o2'], 3)
                              for r in cross], flush=True)
    with open(os.path.join(cli.outdir, 'baselines.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def run_c100loop(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = make_context(CELLS['c100'],
                       torch.device('cuda:{}'.format(cli.gpu)))
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    ez7 = sorted(glob.glob('research_outputs/uw/ez7-c100b-2*'))[-1]
    ck7 = os.path.join(ez7, 'ckpt')

    def mm_of(tag):
        return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(
            ez7, '{}_ft_s*_mm.json'.format(tag))))]

    fit_mm = mm_of('z1')
    cal_mm = mm_of('z2') + mm_of('z3')
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in
         ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in c}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in c)
    q20 = conformal_q([sc(e) for e in cal_mm], 0.20)

    sd0 = torch.load(os.path.join(ck7, 'theta0.pt'), map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    z4 = sorted(glob.glob(os.path.join(ck7, 'z4_ft_s*.pt')))
    passing = [p for p in z4
               if sc(ctx['eval_full'](torch.load(p, map_location=device)))
               <= q20][:3 if not cli.smoke else 1]
    z4mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in z4]
    med = float(np.median([obj_val(m, 'O2', bot, top) for m in z4mg]))
    sel = max(obj_val(m, 'O2', bot, top) for m in z4mg)
    out = {'q20': float(q20), 'starts': [os.path.basename(p)
                                         for p in passing]}

    # (i) class bias (cls = 0 on CIFAR-100)
    bias = []
    for p in passing:
        best_b, best_row = None, None
        start_mg = ctx['margins_of'](torch.load(p, map_location=device))
        for b in ((1.0, 4.0) if cli.smoke else (0.5, 1.0, 2.0, 4.0, 8.0)):
            sd2 = {k: v.clone() for k, v in torch.load(
                p, map_location=device).items()}
            sd2['fc.bias'][0] += b
            mm = ctx['eval_full'](sd2)
            v = obj_val(ctx['margins_of'](sd2), 'O2', bot, top)
            if sc(mm) <= q20:
                best_b, best_row = b, v
        bias.append({'start': os.path.basename(p), 'b_max': best_b,
                     'o2': best_row,
                     'gain': (float(gain_of(best_row, sel, med, th_o2))
                              if best_row is not None else None)})
        print(' c100 bias', os.path.basename(p), best_b,
              bias[-1]['gain'], flush=True)
    out['bias'] = bias

    # (ii) combined per-record score on the c100 unit
    fit_mg = [ctx['margins_of'](torch.load(p, map_location=device))
              for p in sorted(glob.glob(os.path.join(
                  ck7, 'z1_ft_s*.pt')))]
    R = np.stack(fit_mg)
    c_j = np.median(R, axis=0)
    s_j = np.maximum(np.median(np.abs(R - c_j), axis=0), 1e-9)

    def T_of(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    cal_T = [T_of(torch.load(p, map_location=device)) for p in sorted(
        glob.glob(os.path.join(ck7, 'z2_ft_s*.pt')))
        + glob.glob(os.path.join(ck7, 'z3_ft_s*.pt'))]
    qC = conformal_q([max(sc(e), t) for e, t in zip(cal_mm, cal_T)], 0.20)
    test_p = sorted(glob.glob(os.path.join(ck7, 'z4_ft_s*.pt'))) \
        + sorted(glob.glob(os.path.join(ck7, 'z5_ft_s*.pt')))
    honest = float(np.mean([
        max(sc(ctx['eval_full'](torch.load(p, map_location=device))),
            T_of(torch.load(p, map_location=device))) <= qC
        for p in test_p]))
    out['qC'] = float(qC)
    out['honest_combined'] = honest
    th_objs = {'O2': th_o2}

    def sc_C(sd):
        return max(sc(ctx['eval_full'](sd)), T_of(sd))

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    res = run_attack(
        model, xF, yF, bot, top, ctx['margins_of'], sc_C, qC,
        cli.outdir, ck, 'c100comb', passing, z4mg[:len(passing)],
        th_objs, objs=('O2',),
        n_starts=(1 if cli.smoke else 3), iters=(2 if cli.smoke else 60))
    out['combined_attack'] = res
    out['verdict'] = ('C100_COMBINED_LEAKS'
                      if res['O2']['best_gain_pass'] >= GATE
                      else 'C100_COMBINED_CLOSES')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('qC %.3f honest %.3f | VERDICT %s' % (qC, honest,
                                                out['verdict']),
          flush=True)


def run_flipref(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(E3))
    start_mg = {os.path.basename(p): ctx['margins_of'](torch.load(
        p, map_location=device)) for p in starts}
    best_cand = max(starts, key=lambda p: obj_val(
        start_mg[os.path.basename(p)], 'O2', bot, top))
    bc_mg = start_mg[os.path.basename(best_cand)]
    rows = []
    for pth in sorted(glob.glob(os.path.join(
            ey2_dir(), 'ckpt', 'orc_e3_O2_s*.pt'))):
        j = pth.replace('ckpt', '').replace('.pt', '.json')
        rec_path = os.path.join(ey2_dir(), os.path.basename(
            pth).replace('.pt', '.json'))
        rec = json.load(open(rec_path)) if os.path.exists(rec_path) \
            else None
        fin = torch.load(pth, map_location=device)
        mg = ctx['margins_of'](fin)
        j0 = int(os.path.basename(pth)[:-3].rsplit('s', 1)[1])
        st = start_mg[os.path.basename(starts[j0])]
        atk, stc = mg > 0, st > 0
        # per-class test accuracy differential vs own start
        percls = None
        rows.append({
            'final': os.path.basename(pth),
            'true_pass': (rec or {}).get('true_pass',
                                         (rec or {}).get('est_pass')),
            'flips_vs_start': int((atk & ~stc).sum() - (stc & ~atk).sum()),
            'w2c_vs_start': int((atk & ~stc).sum()),
            'c2w_vs_start': int((stc & ~atk).sum()),
            'F_acc_atk': float(atk.mean()),
            'F_acc_start': float(stc.mean()),
            'identity_ok': int((atk & ~stc).sum() - (stc & ~atk).sum())
            == int(atk.sum() - stc.sum()),
            'vs_best_cand_flips': int(((mg > 0) & ~(bc_mg > 0)).sum()),
        })
        print(' flip', rows[-1]['final'], rows[-1]['w2c_vs_start'],
              rows[-1]['c2w_vs_start'], 'id_ok',
              rows[-1]['identity_ok'], flush=True)
    with open(os.path.join(cli.outdir, 'flipref.json'), 'w') as fh:
        json.dump(rows, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('nocon', 'baselines', 'c100loop', 'flipref'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'r2-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('R2 arm', cli.arm, '->', cli.outdir, flush=True)
    dict(nocon=run_nocon, baselines=run_baselines, c100loop=run_c100loop,
         flipref=run_flipref)[cli.arm](cli)


if __name__ == '__main__':
    main()
