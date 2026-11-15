"""C-LSG: Layer-Selective Gradient encoding.
Prereg: r3 ledger section XI.

CE objective with layer-dimension gradient masking: early layers
(feature extractor) get amplified learning rate, late layers
(classifier) get suppressed rate. Single backward pass, no
double-backward. The mechanism is that feature-layer changes
propagate to ALL six metrics simultaneously.

Layer ratio rho_layer = lr_early / lr_late. Sweep {4, 8, 16}.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_lfpr import (channels, rec, score_sets,
                         working_point, zcal)

RHO = 0.5
N_INST = 10
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def classify_param(nm):
    """Return 'early' if parameter is in layer1-3, 'late' if
    layer4/fc, 'other' for BN running stats etc."""
    if 'layer1' in nm or 'layer2' in nm or 'layer3' in nm:
        return 'early'
    if 'layer4' in nm or 'fc' in nm or 'linear' in nm:
        return 'late'
    if 'conv1' in nm or 'bn1' in nm:
        return 'early'
    return 'other'


def gen_lsg(model, mem, cx, ratio_fn, layer_ratio, it):
    """CE objective with layer-selective gradient application."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        lf = F.cross_entropy(model(mem[0]), mem[1])
        ln = F.cross_entropy(model(nx), ny)
        return lf - 0.5 * ln

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    for itn in range(it['n']):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(loss(cur, nx, ny))
        loss(cur, nx, ny).backward()
        g = {}
        for nm, p in model.named_parameters():
            if p.grad is None:
                continue
            cat = classify_param(nm)
            if cat == 'early':
                g[nm] = p.grad.detach().clone() * layer_ratio
            elif cat == 'late':
                g[nm] = p.grad.detach().clone()  # scale 1.0
            else:
                g[nm] = p.grad.detach().clone()
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() - lr * g.get(nm, p.detach()
                     - p.detach())) if nm in g else p
                    for nm, p in cur.items()}
            # handle params not in g (BN running stats etc.)
            cand = {}
            for nm, p in cur.items():
                if nm in g:
                    cand[nm] = p.detach() - lr * g[nm]
                else:
                    cand[nm] = p
            if ratio_fn(cand) <= 1.0:
                v = float(loss(cand, nx, ny))
                if v < v0 - 1e-9:
                    cur, acc = cand, True
                    nacc += 1
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            break
        if nacc == 0 and itn >= 15:
            break
    return cur, nacc


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe)
             - rank_auc(bm[rs.choice(len(fm), len(fm), replace=True)],
                        be)
             for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(
        cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(
        rest, 200, replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    cx = rec(tset, cons, device)
    sets = {'mem': mem, 'aux': ax_, 'eval': fx}

    fit, cal, test, c, sig, sc, q = load_audit()
    X_fit = np.stack([vec(e) for e in fit])
    c_j = X_fit.mean(0)
    S = np.cov(X_fit, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))

    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = json.load(open(cache))['ens_mm']
    hon18 = [vec(e) for g in ('e2', 'e3', 'e4') for e in ens[g]]
    cal12 = [vec(e) for g in ('p1aud', 'e2') for e in ens[g]]
    rs = np.random.RandomState(44111)
    all_inst = []
    for b in range(200):
        idx = rs.choice(len(hon18), 6, replace=True)
        X = np.stack([hon18[i] for i in idx])
        cj = X.mean(0)
        S_ = np.cov(X, rowvar=False)
        Sh_ = (1 - RHO) * S_ + RHO * np.diag(np.diag(S_))
        try:
            Si_ = np.linalg.inv(Sh_)
        except np.linalg.LinAlgError:
            continue
        qJ_ = conformal_q(
            [float((v - cj) @ Si_ @ (v - cj)) for v in cal12], 0.20)
        all_inst.append((cj, Si_, qJ_))
    qJs = [qJ_ for _, _, qJ_ in all_inst]
    q25, q75 = np.percentile(qJs, [25, 75])
    ok = [inst for inst in all_inst if q25 <= inst[2] <= q75]
    K_inst = ok[:N_INST]

    def ratios(sd):
        e = ctx['eval_full'](sd)
        v = vec(e)
        return [float((v - cj_) @ Si_ @ (v - cj_)) / qJ_
                for cj_, Si_, qJ_ in K_inst]

    def ratio_fn(sd):
        return max(ratios(sd))

    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    starts = [('e3-s1', E3[1]), ('e3-s3', E3[3]), ('e3-s4', E3[4]),
              ('e4-s0', E4[0]), ('e4-s1', E4[1]), ('e4-s2', E4[2])]

    def evaluate(tag, sd, fin, lr_val):
        b_sc = score_sets(model, sd, sets)
        f_sc = score_sets(model, fin, sets)
        ch = 'neggrad'
        mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
        eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
        base, fin_auc, lo, hi = rank_ci(mf, mb, ef, eb)
        zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
        zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
        zab = zcal(b_sc['aux'][ch],
                   [fs['aux'][ch] for fs in fit_scores])
        zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
        zaf = zcal(f_sc['aux'][ch],
                   [fs['aux'][ch] for fs in fit_scores])
        ztpr_b = working_point(zmb, zab, zeb, 0.05)
        ztpr_f = working_point(zmf, zaf, zeb, 0.05)
        v_fin = vec(ctx['eval_full'](fin))
        surv = float(np.mean([
            float((v_fin - cj_) @ Si_ @ (v_fin - cj_)) <= qJ_
            for cj_, Si_, qJ_ in all_inst]))
        return {'tag': tag, 'layer_ratio': lr_val,
                'pass': bool(ratio_fn(fin) <= 1.0),
                'auc_base': base, 'auc_fin': fin_auc,
                'ci': [lo, hi],
                'ztpr_base': ztpr_b['tpr'],
                'ztpr_fin': ztpr_f['tpr'],
                'survival_200': surv}

    rows = []
    # Phase 1: layer_ratio sweep on e3-s1 + e4-s1
    if not cli.skip_sweep:
        sweep_starts = [starts[0], starts[4]]
        for lr_val in (4.0, 8.0, 16.0):
            for tag, sp in sweep_starts:
                sd = torch.load(sp, map_location=device)
                fin, nacc = gen_lsg(model, mem, cx, ratio_fn,
                                    lr_val,
                                    {'sd': sd,
                                     'n': 4 if cli.smoke else 60})
                r = evaluate(tag, sd, fin, lr_val)
                r['phase'] = 'sweep'
                rows.append(r)
                print(' SWEEP %-8s lr=%.0f pass=%s surv=%.2f '
                      'auc %.3f->%.3f CI[%.3f,%.3f]'
                      % (tag, lr_val, r['pass'], r['survival_200'],
                         r['auc_base'], r['auc_fin'], r['ci'][0],
                         r['ci'][1]), flush=True)
        passing = [r for r in rows if r['phase'] == 'sweep'
                   and r['pass']]
        best_lr = (max(passing, key=lambda r: r['auc_fin'])
                   ['layer_ratio'] if passing else 4.0)
        print('BEST layer_ratio: %.0f' % best_lr, flush=True)
    else:
        best_lr = cli.best_ratio

    # Phase 2: main arm
    for tag, sp in (starts[:1] if cli.smoke else starts):
        if any(r['tag'] == tag and r['layer_ratio'] == best_lr
               and r.get('phase') == 'sweep' for r in rows):
            continue
        sd = torch.load(sp, map_location=device)
        fin, nacc = gen_lsg(model, mem, cx, ratio_fn,
                            best_lr,
                            {'sd': sd,
                             'n': 4 if cli.smoke else 60})
        r = evaluate(tag, sd, fin, best_lr)
        r['phase'] = 'main'
        rows.append(r)
        print(' MAIN  %-8s lr=%.0f pass=%s surv=%.2f auc %.3f->%.3f '
              'CI[%.3f,%.3f] zTPR %.2f->%.2f'
              % (tag, best_lr, r['pass'], r['survival_200'],
                 r['auc_base'], r['auc_fin'], r['ci'][0],
                 r['ci'][1], r['ztpr_base'], r['ztpr_fin']),
              flush=True)

    main_rows = [r for r in rows if r.get('phase') == 'main'] or \
                [r for r in rows if r.get('phase') == 'sweep'
                 and r['layer_ratio'] == best_lr]
    n_pass = sum(1 for r in main_rows if r['pass'])
    ci_pos = sum(1 for r in main_rows if r['ci'][0] > 0)
    survs = [r['survival_200'] for r in main_rows if r['ci'][0] > 0]
    med_surv = float(np.median(survs)) if survs else 0
    need = 1 if cli.smoke else 4
    # G4: paired vs GR
    cre_dirs = sorted(glob.glob(
        'research_outputs/uw/cre-2*/verdict.json'))
    g4 = None
    g4_diffs = None
    if cre_dirs and not cli.smoke:
        cre = json.load(open(cre_dirs[-1]))
        gr_rows = {r['tag']: r for r in cre.get('rows', [])
                   if r.get('obj') == 'grad'}
        if gr_rows:
            g4_diffs = [r['auc_fin'] - gr_rows[r['tag']]['auc_fin']
                        for r in main_rows if r['tag'] in gr_rows]
            if g4_diffs:
                ci_lb = float(np.percentile(g4_diffs, 5))
                g4 = ci_lb > -0.03
    verdict = ('METHOD-ADVANCE' if n_pass >= (5 if not cli.smoke
                                              else 1)
               and ci_pos >= need and med_surv >= 0.8 and g4
               else 'EQUIVALENT-TO-GR' if n_pass >= need
               and ci_pos >= need and med_surv >= 0.8 else 'FAIL')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'best_layer_ratio': best_lr,
           'gates': {'G1_feasible': n_pass,
                     'G2_ci_positive': ci_pos,
                     'G3_median_survival': med_surv,
                     'G4_vs_GR': g4,
                     'G4_paired_diffs': g4_diffs},
           'verdict': verdict, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT %s (pass=%d/%d, CI>0=%d/%d, surv=%.2f)' %
          (verdict, n_pass, len(main_rows), ci_pos,
           len(main_rows), med_surv), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--best-ratio', default=4.0, type=float)
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-lsg' if cli.smoke else 'lsg',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
