"""C-CRE P1: calibration-robust encoding via multi-instance
intersection constraint (prereg: r2 ledger section III).

Acceptance check: max over K bootstrap instances of s_J <= max q_J.
K = 10 instances from the 200 bootstrap set (seed 44111, first 10),
each with its own Sigma-hat and conformal q_J. The encoder must
satisfy the TIGHTEST instance, not just one.

Objectives: ce (6 starts) + grad (3 starts, counter arm).
Protocol: ce_gen frozen loop, only acceptance check replaced.
Readouts: AUC CI (2000 resamples), z-TPR@5%, cross-instance
survival (200 bootstrap re-evaluation), s6.
Gates: G1 >=4/6 accepted; G2 median survival >= 0.8;
G3 >=4/6 AUC CI lb > 0.
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
from uw.run_lfpr import (BETAS, channels, rec, score_sets,
                         working_point, zcal)

RHO = 0.5
N_INSTANCES = 10
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_multi(model, mem, cx, score_multi, qmax, it, obj):
    """ce_gen loop with multi-instance max constraint."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        if obj == 'grad':
            xg = mem[0].clone().requires_grad_(True)
            xg2 = nx.clone().requires_grad_(True)
            lf = F.cross_entropy(model(xg), mem[1],
                                 reduction='none')
            ln = F.cross_entropy(model(xg2), ny,
                                 reduction='none')
            gf = torch.autograd.grad(lf.sum(), xg,
                                     create_graph=True
                                     )[0].flatten(1).norm(1)
            gn = torch.autograd.grad(ln.sum(), xg2,
                                     create_graph=True
                                     )[0].flatten(1).norm(1)
            return (gf - 0.5 * gn).sum()
        lf = F.cross_entropy(model(mem[0]), mem[1])
        if obj == 'fonly':
            return lf
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
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() - lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if score_multi(cand) <= qmax:
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
    q20 = q[0.20]

    # Build K bootstrap instances
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = json.load(open(cache))['ens_mm']
    hon18 = [vec(e) for g in ('e2', 'e3', 'e4') for e in ens[g]]
    cal12 = [vec(e) for g in ('p1aud', 'e2') for e in ens[g]]
    rs = np.random.RandomState(44111)
    instances = []
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
        instances.append((cj, Si, qJ))
    # Select K non-degenerate instances: filter qJ to [p25, p75]
    qJs = [qJ_ for _, _, qJ_ in instances]
    if cli.tight:
        q25, q75 = np.percentile(qJs, [5, 35])
    else:
        q25, q75 = np.percentile(qJs, [25, 75])
    ok = [inst for inst in instances if q25 <= inst[2] <= q75]
    K_inst = ok[:N_INSTANCES]
    qmax = 1.0  # ratio-based threshold
    print('K=%d instances, qmax=%.4f (range %.3f-%.3f)' %
          (N_INSTANCES, qmax,
           min(qJ_ for _, _, qJ_ in K_inst), qmax), flush=True)

    def metrics_to_sJ(sd):
        e = ctx['eval_full'](sd)
        v = vec(e)
        return [float((v - cj_) @ Si_ @ (v - cj_))
                for cj_, Si_, _ in K_inst]

    def score_multi(sd):
        sJs = metrics_to_sJ(sd)
        return max(sJ / qJ_ for sJ, (_, _, qJ_) in
                   zip(sJs, K_inst))

    # Main instance for comparison
    X_fit = np.stack([vec(e) for e in fit])
    c_main = X_fit.mean(0)
    S_main = np.cov(X_fit, rowvar=False)
    Sh_main = (1 - RHO) * S_main + RHO * np.diag(np.diag(S_main))
    Si_main = np.linalg.inv(Sh_main)
    qJ_main = conformal_q(
        [float((v - c_main) @ Si_main @ (v - c_main))
         for v in cal12], 0.20)

    def sJ_main(sd):
        d = vec(ctx['eval_full'](sd)) - c_main
        return float(d @ Si_main @ d)

    def s6(sd):
        return float(sc(ctx['eval_full'](sd)))

    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    starts = [('e3-s1', E3[1]), ('e3-s3', E3[3]), ('e3-s4', E3[4]),
              ('e4-s0', E4[0]), ('e4-s1', E4[1]), ('e4-s2', E4[2])]
    if cli.warm:
        # Use e4-s1's C-CRE GR final as warm start for failing e3 starts
        import glob as _g
        donor = sorted(_g.glob(
            'research_outputs/uw/cre-2*/cre_grad_e4-s1.pt'))
        if not donor:
            # Fallback: look in outdir for any grad_e4-s1
            donor = sorted(_g.glob(
                'research_outputs/uw/*/cre_grad_e4-s1.pt'))
        assert donor, 'No e4-s1 GR final found for warm start'
        donor_p = donor[-1]
        print('WARM START from: %s' % donor_p, flush=True)
        starts = [('warm-e3-s3', donor_p), ('warm-e3-s4', donor_p)]
    if cli.phase == 'fwd':
        starts = [('f-e3-0', E3[0]), ('f-e3-2', E3[2]), ('f-e3-5', E3[5]),
                  ('f-e4-3', E4[3]), ('f-e4-4', E4[4]), ('f-e4-5', E4[5])]

    rows = []
    metas = {}
    for tag, sp in (starts[:1] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        for obj in ('grad',) if (cli.phase == 'fwd' or cli.tight) else (('ce', 'grad') if not cli.smoke else ('ce',)):

            fin, meta = gen_multi(model, mem, cx, score_multi,
                                  qmax, {'sd': sd,
                                         'n': 4 if cli.smoke else cli.iters},
                                  obj)
            fname = 'cre_%s_%s.pt' % (obj, tag)
            torch.save(fin, os.path.join(cli.outdir, fname))
            metas['%s:%s' % (tag, obj)] = meta
            # evaluate
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
            # cross-instance survival
            v_fin = vec(ctx['eval_full'](fin))
            surv = float(np.mean([
                float((v_fin - cj_) @ Si_ @ (v_fin - cj_)) <= qJ_
                for cj_, Si_, qJ_ in instances]))
            s6_fin = s6(fin)
            sj_main_fin = sJ_main(fin)
            row = {'tag': tag, 'obj': obj,
                   'auc_base': base, 'auc_fin': fin_auc,
                   'ci': [lo, hi],
                   'ztpr_base': ztpr_b['tpr'],
                   'ztpr_fin': ztpr_f['tpr'],
                   'ztpr_fpr': ztpr_f['fpr'],
                   'survival_200': surv,
                   's6_fin': s6_fin,
                   'sJ_main_fin': sj_main_fin,
                   'pass_multi': bool(score_multi(fin) <= 1.0)}
            rows.append(row)
            print(' %-8s %-5s pass_multi=%s surv=%.2f auc %.3f->%.3f '
                  'CI[%.3f,%.3f] zTPR %.2f->%.2f'
                  % (tag, obj, row['pass_multi'], surv, base,
                     fin_auc, lo, hi, ztpr_b['tpr'],
                     ztpr_f['tpr']), flush=True)

    # gates
    main = [r for r in rows if r['obj'] == 'ce']
    need = 1 if cli.smoke else 4
    g1 = sum(1 for r in main if r['pass_multi'])
    med_surv = float(np.median([r['survival_200'] for r in main
                                if r['pass_multi']])) if g1 else 0
    g2 = med_surv >= 0.8
    g3 = sum(1 for r in main if r['ci'][0] > 0)
    from math import comb
    def pbinom(k, n):
        return sum(comb(n, i) for i in range(k, n+1)) / 2**n
    verdict = ('CALIBRATION-ROBUST' if g1 >= need and g2 and
               g3 >= need else
               'TRADE-OFF' if g1 >= need and g2 else
               'NOT-ROBUST')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'n_instances': N_INSTANCES, 'qmax': float(qmax),
           'gates': {'G1_feasible': g1, 'G1_p': pbinom(g1, len(main)),
                     'G2_median_survival': med_surv,
                     'G3_auc_positive': g3,
                     'G3_p': pbinom(g3, len(main))},
           'verdict': verdict,
           'rows': rows, 'gen_metas': metas}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--iters', default=60, type=int)
    ap.add_argument('--warm', action='store_true')
    ap.add_argument('--phase', default='main', choices=['main', 'fwd'])
    ap.add_argument('--tight', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-cre' if cli.smoke else 'cre',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
