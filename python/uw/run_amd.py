"""C-AMD: Adaptive Multi-Directional encoding.
Prereg: r3 ledger section VIII.

J_AMD = alpha * J_signal + (1-alpha) * J_robust
  J_signal = -(CE on F - 0.5*CE on Nc)  [maximize separation]
  J_robust = differentiable proxy for Sigma-hat top-1 direction
Alpha adapts every 10 iterations based on constraint headroom.
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
ALPHA_INIT = 0.5
ALPHA_MIN, ALPHA_MAX = 0.1, 0.9
ADJUST_EVERY = 10


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_amd(model, mem, cx, ratio_fn, w_top1, it):
    """Adaptive multi-directional gradient ascent."""
    w3, w6 = float(w_top1[2]), float(w_top1[5])
    one_hot_f = F.one_hot(mem[1], 10).float()

    def total_loss(sd, nx, ny, a):
        model.load_state_dict(sd)
        model.eval()
        out_m = model(mem[0])
        ce_m = F.cross_entropy(out_m, mem[1])
        zy = out_m.gather(1, mem[1][:, None]).squeeze(1)
        zo = out_m - one_hot_f * 1e18
        margin = (zy - zo.max(1).values).mean()
        ln = F.cross_entropy(model(nx), ny)
        j_sig = -(ce_m - 0.5 * ln)
        j_rob = -w3 * ce_m + w6 * margin
        return a * j_sig + (1 - a) * j_rob

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    alpha = ALPHA_INIT
    alpha_traj = [alpha]
    for itn in range(it['n']):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(total_loss(cur, nx, ny, alpha))
        total_loss(cur, nx, ny, alpha).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() + lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if ratio_fn(cand) <= 1.0:
                v = float(total_loss(cand, nx, ny, alpha))
                if v > v0 + 1e-9:
                    cur, acc = cand, True
                    nacc += 1
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if (itn + 1) % ADJUST_EVERY == 0 and itn > 0:
            cur_wr = ratio_fn(cur)
            if cur_wr < 0.8:
                alpha = min(alpha + 0.1, ALPHA_MAX)
            elif cur_wr > 0.95:
                alpha = max(alpha - 0.1, ALPHA_MIN)
            alpha_traj.append(round(alpha, 2))
        if stall > 12:
            break
        if nacc == 0 and itn >= 15:
            break
    return cur, {'n_accepted': nacc,
                 'alpha_final': alpha,
                 'alpha_traj': alpha_traj}


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
    X_fit = np.stack([vec(e) for e in fit])
    c_j = X_fit.mean(0)
    S = np.cov(X_fit, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
    evals, evecs = np.linalg.eigh(Sh)
    w_top1 = evecs[:, -1]
    print('top-1 eigval=%.3f' % evals[-1], flush=True)

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
    if cli.phase == 'fwd':
        starts = [('f-e3-0', E3[0]), ('f-e3-2', E3[2]),
                  ('f-e3-5', E3[5]), ('f-e4-3', E4[3]),
                  ('f-e4-4', E4[4]), ('f-e4-5', E4[5])]

    rows = []
    metas = {}
    for tag, sp in (starts[:1] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        fin, meta = gen_amd(model, mem, cx, ratio_fn,
                            w_top1, {'sd': sd,
                                     'n': 4 if cli.smoke else 60})
        torch.save(fin, os.path.join(cli.outdir,
                                     'amd_%s.pt' % tag))
        metas[tag] = meta
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
        r = {'tag': tag,
             'pass': bool(ratio_fn(fin) <= 1.0),
             'auc_base': base, 'auc_fin': fin_auc,
             'ci': [lo, hi],
             'ztpr_base': ztpr_b['tpr'],
             'ztpr_fin': ztpr_f['tpr'],
             'survival_200': surv,
             'alpha_final': meta['alpha_final']}
        rows.append(r)
        print(' %-8s pass=%s surv=%.2f auc %.3f->%.3f CI[%.3f,%.3f] '
              'zTPR %.2f->%.2f a=%.2f'
              % (tag, r['pass'], surv, base, fin_auc, lo, hi,
                 ztpr_b['tpr'], ztpr_f['tpr'],
                 r['alpha_final']), flush=True)

    ci_pos = sum(1 for r in rows if r['ci'][0] > 0)
    n_pass = sum(1 for r in rows if r['pass'])
    need = 1 if cli.smoke else 5
    survs = [r['survival_200'] for r in rows if r['ci'][0] > 0]
    med_surv = float(np.median(survs)) if survs else 0
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
                        for r in rows if r['tag'] in gr_rows]
            if g4_diffs:
                ci_lb = float(np.percentile(g4_diffs, 5))
                g4 = ci_lb > -0.03
    verdict = ('METHOD-ADVANCE' if n_pass >= need and
               ci_pos >= need and med_surv >= 0.8 and g4 else
               'EQUIVALENT-TO-GR' if n_pass >= need and
               ci_pos >= need and med_surv >= 0.8 else 'FAIL')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'gates': {'G1_feasible': n_pass,
                     'G2_ci_positive': ci_pos,
                     'G3_median_survival': med_surv,
                     'G4_vs_GR': g4,
                     'G4_paired_diffs': g4_diffs},
           'verdict': verdict,
           'rows': rows, 'metas': metas}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT %s (pass=%d/%d, CI>0=%d/%d, surv=%.2f)' %
          (verdict, n_pass, len(rows), ci_pos, len(rows),
           med_surv), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--phase', default='main',
                    choices=['main', 'fwd'])
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s-%s' % (
            'smoke-amd' if cli.smoke else 'amd',
            cli.phase, time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
