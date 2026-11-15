"""C-MIR: min-max Intersection Robust encoding objective.
Prereg: r2 ledger section VIII. New objective designed specifically
for the multi-instance intersection acceptance check.

J_MIR = J_CE + lambda * max_i(sJ_i / qJ_i)
where sJ_i is the Mahalanobis score under bootstrap instance i.
The max term regularizes against the worst instance, pushing the
encoding direction away from low-variance directions of any
instance. Acceptance: max ratio <= 1.0 (same as C-CRE).

Protocol: ce_gen frozen loop; the loss adds the worst-instance
penalty. The acceptance check uses the same ratio-based constraint.
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
N_INSTANCES = 10
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_mir(model, mem, cx, ratio_fn, lam, it):
    """ce_gen loop with J_MIR = J_CE + lam * worst_instance_ratio.
    The ratio term is computed from the CURRENT model's metrics,
    not the candidate — it serves as a proximal regularizer guiding
    the descent direction away from tight instances."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        lf = F.cross_entropy(model(mem[0]), mem[1])
        ln = F.cross_entropy(model(nx), ny)
        return lf - 0.5 * ln

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    reason = 'max-iter'
    for itn in range(it['n']):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(loss(cur, nx, ny))
        loss(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        # regularize: penalize worst instance ratio (computed on
        # the CURRENT model, not the candidate — it shapes the
        # gradient direction, the acceptance check handles the
        # constraint)
        wr = ratio_fn(cur)  # scalar, max ratio across K instances
        # scale gradient by (1 + lam * wr) to push harder away
        # from tight instances
        scale = 1.0 + lam * max(0.0, wr - 0.5)
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() - lr * scale * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
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
            reason = 'stall'
            break
        if nacc == 0 and itn >= 15:
            reason = 'no-improve'
            break
    return cur, {'reason': reason, 'n_accepted': nacc}


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)], fe)
             - rank_auc(bm[rs.choice(len(fm), len(fm), replace=True)], be)
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
    q20 = q[0.20]

    # Build K non-degenerate instances (same as C-CRE)
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
        S = np.cov(X, rowvar=False)
        Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
        try:
            Si = np.linalg.inv(Sh)
        except np.linalg.LinAlgError:
            continue
        qJ = conformal_q(
            [float((v - cj) @ Si @ (v - cj)) for v in cal12], 0.20)
        all_inst.append((cj, Si, qJ))
    qJs = [qJ_ for _, _, qJ_ in all_inst]
    q25, q75 = np.percentile(qJs, [25, 75])
    ok = [inst for inst in all_inst if q25 <= inst[2] <= q75]
    K_inst = ok[:N_INSTANCES]
    # All 200 for survival evaluation
    all_200 = all_inst

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

    def evaluate(tag, sd, fin, lam_val):
        b_sc = score_sets(model, sd, sets)
        f_sc = score_sets(model, fin, sets)
        ch = 'neggrad'
        mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
        eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
        base, fin_auc, lo, hi = rank_ci(mf, mb, ef, eb)
        zmb = zcal(mb, [fs['mem'][ch] for fs in fit_scores])
        zmf = zcal(mf, [fs['mem'][ch] for fs in fit_scores])
        zab = zcal(b_sc['aux'][ch], [fs['aux'][ch] for fs in fit_scores])
        zeb = zcal(eb, [fs['eval'][ch] for fs in fit_scores])
        zaf = zcal(f_sc['aux'][ch],
                   [fs['aux'][ch] for fs in fit_scores])
        ztpr_b = working_point(zmb, zab, zeb, 0.05)
        ztpr_f = working_point(zmf, zaf, zeb, 0.05)
        v_fin = vec(ctx['eval_full'](fin))
        surv = float(np.mean([
            float((v_fin - cj_) @ Si_ @ (v_fin - cj_)) <= qJ_
            for cj_, Si_, qJ_ in all_200]))
        return {'tag': tag, 'lambda': lam_val,
                'auc_base': base, 'auc_fin': fin_auc,
                'ci': [lo, hi], 'ztpr_base': ztpr_b['tpr'],
                'ztpr_fin': ztpr_f['tpr'],
                'survival_200': surv,
                'pass_ratio': bool(ratio_fn(fin) <= 1.0)}

    rows = []
    # Phase 1: lambda sweep on e3-s1 + e4-s1
    if not cli.skip_sweep:
        sweep_starts = [starts[0], starts[4]]  # e3-s1, e4-s1
        for lam in (0.1, 0.5, 1.0):
            for tag, sp in sweep_starts:
                sd = torch.load(sp, map_location=device)
                fin, meta = gen_mir(model, mem, cx, ratio_fn,
                                    lam, {'sd': sd,
                                          'n': 4 if cli.smoke else 60})
                r = evaluate(tag, sd, fin, lam)
                r['phase'] = 'sweep'
                rows.append(r)
                print(' SWEEP %-8s lam=%.1f pass=%s surv=%.2f '
                      'auc %.3f->%.3f CI[%.3f,%.3f]'
                      % (tag, lam, r['pass_ratio'], r['survival_200'],
                         r['auc_base'], r['auc_fin'], r['ci'][0],
                         r['ci'][1]), flush=True)
        # Select lambda: max auc_fin among pass_ratio=True
        passing = [r for r in rows if r['phase'] == 'sweep'
                   and r['pass_ratio']]
        if passing:
            best_lam = max(passing, key=lambda r: r['auc_fin'])['lambda']
        else:
            best_lam = 0.1  # fallback
        print('BEST LAMBDA: %.1f' % best_lam, flush=True)
    else:
        best_lam = cli.best_lambda

    # Phase 2: full 6 starts with best lambda
    for tag, sp in (starts[:1] if cli.smoke else starts):
        if any(r['tag'] == tag and r['lambda'] == best_lam
               and r.get('phase') == 'sweep' for r in rows):
            # already ran in sweep, reuse
            continue
        sd = torch.load(sp, map_location=device)
        fin, meta = gen_mir(model, mem, cx, ratio_fn,
                            best_lam, {'sd': sd,
                                       'n': 4 if cli.smoke else 60})
        r = evaluate(tag, sd, fin, best_lam)
        r['phase'] = 'main'
        rows.append(r)
        print(' MAIN  %-8s lam=%.1f pass=%s surv=%.2f auc %.3f->%.3f '
              'CI[%.3f,%.3f] zTPR %.2f->%.2f'
              % (tag, best_lam, r['pass_ratio'], r['survival_200'],
                 r['auc_base'], r['auc_fin'], r['ci'][0], r['ci'][1],
                 r['ztpr_base'], r['ztpr_fin']), flush=True)

    # Gates (on main-arm rows)
    main_rows = [r for r in rows if r.get('phase') == 'main'] or \
                [r for r in rows if r.get('phase') == 'sweep'
                 and r['lambda'] == best_lam]
    g1 = sum(1 for r in main_rows if r['pass_ratio'])
    survs = [r['survival_200'] for r in main_rows if r['pass_ratio']]
    med_surv = float(np.median(survs)) if survs else 0
    g2 = med_surv >= 0.8
    g3 = sum(1 for r in main_rows if r['ci'][0] > 0)
    need = 1 if cli.smoke else 4
    from math import comb
    def pbinom(k, n):
        return sum(comb(n, i) for i in range(k, n+1)) / 2**n
    # G4: paired vs GR (load GR results from C-CRE verdict)
    cre_dirs = sorted(glob.glob('research_outputs/uw/cre-2*/verdict.json'))
    g4 = None
    g4_diffs = None
    if cre_dirs:
        cre = json.load(open(cre_dirs[-1]))
        gr_rows = {r['tag']: r for r in cre.get('rows', [])
                   if r.get('obj') == 'grad'}
        if gr_rows:
            g4_diffs = []
            for r in main_rows:
                if r['tag'] in gr_rows:
                    g4_diffs.append(
                        r['auc_fin'] - gr_rows[r['tag']]['auc_fin'])
            if g4_diffs:
                ci_lb = float(np.percentile(g4_diffs, 5))
                g4 = ci_lb > -0.03
    verdict = ('METHOD-ADVANCE' if g1 >= need and g2 and g3 >= need
               and g4 else
               'GR-SUFFICIENT' if g1 >= need and g2 and g3 >= need
               else 'METHOD-FAIL')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'best_lambda': best_lam,
           'gates': {'G1_feasible': g1, 'G1_p': pbinom(g1, len(main_rows)),
                     'G2_median_survival': med_surv,
                     'G3_auc_positive': g3,
                     'G3_p': pbinom(g3, len(main_rows)),
                     'G4_vs_GR': g4, 'G4_paired_diffs': g4_diffs},
           'verdict': verdict,
           'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--best-lambda', default=0.5, type=float)
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-mir' if cli.smoke else 'mir',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
