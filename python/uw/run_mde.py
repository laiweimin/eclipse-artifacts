"""C-MDE: Multi-Directional Encoding — direct 6-metric deviation
vector control along Sigma-hat top-1 eigenvector direction.

Prereg: docs/decisions/2026-11-01-method-improvement-r3-ledger.md
Unlike CE (loss-gradient direction) or GR (double-backward), MDE
explicitly optimizes the 6-metric deviation vector along a chosen
direction w (= top-1 eigenvector of Sigma-hat, where the joint
rule discounts most). This is computationally simpler (no
double-backward) and directionally controllable.

J_MDE = w^T (m(theta) - m(start))
Optimized by gradient ascent on J_MDE (standard backprop, no
second-order). Acceptance: multi-instance ratio constraint.
"""
import argparse, glob, json, os, time
import numpy as np
import torch

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


def gen_mde(model, ctx, mem, cx, ratio_fn, w_vec, m_start, it):
    """Gradient ascent on J_MDE = w^T(m(theta) - m(start)).
    m(theta) requires eval_full forward pass for each candidate —
    but we can't backprop through eval_full (it uses argmax,
    KS distance, etc.). Instead, use a differentiable surrogate:
    the weighted sum of the 6 metric VALUES as a function of
    model parameters. Since metrics m1-m6 are computed from
    model outputs, we approximate J_MDE by a proxy that IS
    differentiable: the margin-based metrics (m3 Floss, m6 fm)
    are smooth; m1/m2 are argmax-based (non-diff); m4/m5 are
    distributional (non-diff). So we use a smooth surrogate:
    J_proxy = -w_3 * CE(mem) - w_6 * mean_margin(mem) etc.

    Practical approach: since m3 (mean CE on F) and m6 (mean margin
    on F) are the smoothly differentiable components of the 6-metric
    vector, and they carry most of the encoding signal, we use:
    J_MDE ≈ -w[2] * CE_F(θ) + w[5] * margin_F(θ)
    where w[2], w[5] are the components of the top-1 eigvec
    corresponding to m3 and m6. This is a first-order objective
    (single backprop) that moves the differentiable metrics in the
    desired direction.

    The acceptance check (ratio_fn) still uses the full eval_full
    metrics, so the encoder is constrained by the true audit scores.
    """
    w3, w6 = float(w_vec[2]), float(w_vec[5])  # m3_Floss, m6_fm

    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        out = model(mem[0])
        ce = torch.nn.functional.cross_entropy(out, mem[1])
        # margin: correct-class logit - max other logit
        zy = out.gather(1, mem[1][:, None]).squeeze(1)
        zo = out.clone()
        zo.scatter_(1, mem[1][:, None], -1e18)
        margin = (zy - zo.max(1).values).mean()
        return -w3 * ce + w6 * margin

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    for itn in range(it['n']):
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(loss(cur, None, None))
        loss(cur, None, None).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() + lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if ratio_fn(cand) <= 1.0:
                v = float(loss(cand, None, None))
                if v > v0 + 1e-9:
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

    # Build Sigma-hat and eigendecompose
    X_fit = np.stack([vec(e) for e in fit])
    c_j = X_fit.mean(0)
    S = np.cov(X_fit, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
    evals, evecs = np.linalg.eigh(Sh)
    # Top-1 eigenvector (highest variance direction)
    w_top1 = evecs[:, -1]  # last = largest eigenvalue
    print('top-1 eigval=%.3f, w=[%s]' %
          (evals[-1], ', '.join('%.3f' % x for x in w_top1)),
          flush=True)

    # Build K bootstrap instances for ratio constraint
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
    for tag, sp in (starts[:1] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        m_start = vec(ctx['eval_full'](sd))
        fin, nacc = gen_mde(model, ctx, mem, cx, ratio_fn,
                            w_top1, m_start,
                            {'sd': sd,
                             'n': 4 if cli.smoke else 60})
        torch.save(fin, os.path.join(cli.outdir,
                                     'mde_%s.pt' % tag))
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
        # deviation direction cosine with top-1
        d = v_fin - m_start
        cos1 = float(np.dot(d, w_top1) / max(
            np.linalg.norm(d) * np.linalg.norm(w_top1), 1e-12))
        r = {'tag': tag, 'n_acc': nacc,
             'pass': bool(ratio_fn(fin) <= 1.0),
             'auc_base': base, 'auc_fin': fin_auc,
             'ci': [lo, hi],
             'ztpr_base': ztpr_b['tpr'],
             'ztpr_fin': ztpr_f['tpr'],
             'survival_200': surv,
             'cos_top1': cos1}
        rows.append(r)
        print(' %-8s pass=%s surv=%.2f auc %.3f->%.3f CI[%.3f,%.3f] '
              'zTPR %.2f->%.2f cos=%.2f'
              % (tag, r['pass'], surv, base, fin_auc, lo, hi,
                 ztpr_b['tpr'], ztpr_f['tpr'], cos1), flush=True)

    ci_pos = sum(1 for r in rows if r['ci'][0] > 0)
    n_pass = sum(1 for r in rows if r['pass'])
    need = 1 if cli.smoke else 4
    survs = [r['survival_200'] for r in rows if r['ci'][0] > 0]
    med_surv = float(np.median(survs)) if survs else 0
    verdict = ('METHOD-ADVANCE' if n_pass >= (5 if not cli.smoke
                                              else 1)
               and ci_pos >= need and med_surv >= 0.8 else
               'EQUIVALENT' if n_pass >= need and ci_pos >= need
               else 'FAIL')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'top1_w': [float(x) for x in w_top1],
           'gates': {'G1_feasible': n_pass, 'G2_ci_positive': ci_pos,
                     'G3_median_survival': med_surv},
           'verdict': verdict, 'rows': rows}
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
            'smoke-mde' if cli.smoke else 'mde',
            cli.phase, time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
