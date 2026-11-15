"""C-MOB-FWD: GR objective under multi-instance intersection
constraint on MobileNetV2 (prospective validation, no selector).

Context: ez1-mob (MobileNetV2, CIFAR-10). Honest family from
z1-z5 ensembles. K=10 bootstrap instances from 24 honest models
(z2-z5), ratio-based intersection. GR objective from run_cre's
gen_multi. 6 starts = audit-passing models of z4/z5.
Gate: >=4/6 CI>0.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

import uw.run_ez as ez
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep as real_f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import channels, rec, score_sets, working_point, zcal

RHO = 0.5
N_INST = 10
MOB = 'research_outputs/uw/ez1-mob-20261002T080210'


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_multi(model, mem, cx, ratio_fn, it):
    """GR objective under multi-instance ratio constraint."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        xg = mem[0].clone().requires_grad_(True)
        xg2 = nx.clone().requires_grad_(True)
        lf = F.cross_entropy(model(xg), mem[1], reduction='none')
        ln = F.cross_entropy(model(xg2), ny, reduction='none')
        gf = torch.autograd.grad(lf.sum(), xg,
                                 create_graph=True
                                 )[0].flatten(1).norm(1)
        gn = torch.autograd.grad(ln.sum(), xg2,
                                 create_graph=True
                                 )[0].flatten(1).norm(1)
        return (gf - 0.5 * gn).sum()

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
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() - lr * g[nm]) if nm in g
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
    # MobileNetV2 context
    ez_cls = ez.CELLS['mob']
    orig_fdeep = ez.f_deep
    cat_tr_10 = None  # not needed for mob (uses full_class pool)
    ctx = ez.make_context(ez_cls, device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    # For mob: pool = full cat class (500 records)
    cat_tr = ctx['pool']
    fset = ctx['fset']  # mob context has fset built in
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
    print('pool=%d F=%d' % (len(cat_tr), len(fset)), flush=True)

    # Load honest family metrics from ez1-mob run
    hon_paths = sorted(glob.glob(
        MOB + '/ckpt/z[2345]_ft_s*.pt'))  # z2-z5 = 24 models
    fit_paths = sorted(glob.glob(
        MOB + '/ckpt/z1_ft_s*.pt'))  # z1 = 6 for z-cal
    assert len(hon_paths) == 24 and len(fit_paths) == 6

    # Compute metric vectors for all honest models
    hon_vecs = []
    for p in hon_paths:
        sd = torch.load(p, map_location=device)
        e = ctx['eval_full'](sd)
        hon_vecs.append(vec(e))
    hon_vecs = np.stack(hon_vecs)
    # Metric vectors for z1 (cal set)
    cal_vecs = []
    for p in fit_paths:
        sd = torch.load(p, map_location=device)
        e = ctx['eval_full'](sd)
        cal_vecs.append(vec(e))
    cal_vecs = np.stack(cal_vecs)

    # Build K=10 bootstrap instances
    rs = np.random.RandomState(44111)
    all_inst = []
    for b in range(200):
        idx = rs.choice(len(hon_vecs), 6, replace=True)
        X = hon_vecs[idx]
        cj = X.mean(0)
        S = np.cov(X, rowvar=False)
        Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
        try:
            Si = np.linalg.inv(Sh)
        except np.linalg.LinAlgError:
            continue
        # Use cal_vecs (z1) as calibration set for conformal q
        qJ = conformal_q(
            [float((v - cj) @ Si @ (v - cj)) for v in cal_vecs],
            0.20)
        all_inst.append((cj, Si, qJ))
    qJs = [q for _, _, q in all_inst]
    q25, q75 = np.percentile(qJs, [25, 75])
    ok = [inst for inst in all_inst if q25 <= inst[2] <= q75]
    K_inst = ok[:N_INST]
    print('K=%d instances selected' % len(K_inst), flush=True)

    def ratios(sd):
        e = ctx['eval_full'](sd)
        v = vec(e)
        return [float((v - cj_) @ Si_ @ (v - cj_)) / qJ_
                for cj_, Si_, qJ_ in K_inst]

    def ratio_fn(sd):
        return max(ratios(sd))

    # z-cal scores for readout
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    # Starts: audit-passing models from z4/z5
    start_paths = sorted(glob.glob(
        MOB + '/ckpt/z4_ft_s*.pt')) + sorted(glob.glob(
        MOB + '/ckpt/z5_ft_s*.pt'))
    # Score all and pick 6 that pass the main instance
    passing = []
    for p in start_paths:
        sd = torch.load(p, map_location=device)
        if ratio_fn(sd) <= 1.0:
            passing.append(p)
    starts = passing[:6]
    print('starts: %d/%d pass' % (len(passing), len(start_paths)),
          flush=True)

    rows = []
    for i, sp in enumerate(starts):
        tag = 'mob-s%d' % i
        sd = torch.load(sp, map_location=device)
        fin, nacc = gen_multi(model, mem, cx, ratio_fn,
                              {'sd': sd,
                               'n': 4 if cli.smoke else 60})
        torch.save(fin, os.path.join(cli.outdir,
                                     'mobJ_%s.pt' % tag))
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
        # survival across 200
        v_fin = vec(ctx['eval_full'](fin))
        surv = float(np.mean([
            float((v_fin - cj_) @ Si_ @ (v_fin - cj_)) <= qJ_
            for cj_, Si_, qJ_ in all_inst]))
        r = {'tag': tag, 'n_acc': nacc,
             'pass': bool(ratio_fn(fin) <= 1.0),
             'auc_base': base, 'auc_fin': fin_auc,
             'ci': [lo, hi],
             'ztpr_base': ztpr_b['tpr'],
             'ztpr_fin': ztpr_f['tpr'],
             'survival_200': surv}
        rows.append(r)
        print(' %-8s pass=%s surv=%.2f auc %.3f->%.3f CI[%.3f,%.3f] '
              'zTPR %.2f->%.2f'
              % (tag, r['pass'], surv, base, fin_auc, lo, hi,
                 ztpr_b['tpr'], ztpr_f['tpr']), flush=True)

    ci_pos = sum(1 for r in rows if r['ci'][0] > 0)
    need = 1 if cli.smoke else 4
    survs = [r['survival_200'] for r in rows if r['ci'][0] > 0]
    verdict = ('PROSPECTIVE-CONFIRMED' if ci_pos >= need else
               'PARTIAL' if ci_pos >= 2 else 'NOT-CONFIRMED')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'n_starts': len(rows), 'ci_positive': ci_pos,
           'signal_start_survivals': survs,
           'verdict': verdict, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT %s (%d/%d CI>0)' % (verdict, ci_pos, len(rows)),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-mobfwd' if cli.smoke else 'mobfwd',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
