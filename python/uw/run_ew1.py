"""UW E-W1: wash capacity under the statistical metric audit.

Frozen preregistration: docs/plans/2026-09-22-uw-ew1-preregistration.md
(commits ed46940 + erratum da3e879). Nothing here changes frozen constants.

Setup (S0-N frozen bed): ridge head on full train (lam=0.1, cat=3 binary);
pools 43501/43502 (60 cat) with k in {3,10,30,60}; extension pool 43801
(2000 cat) with k in {100,300,1000}; 10 sets per (pool, k) via
RandomState(base + j), bases 44100/44200/44300. Exact retrain shift s_rt
via the Woodbury construction (same formulas as run_s0n_step3.validate_pool).
Audit: six metrics from run_s0n_step3.metrics_at; acceptance bands = 95%
quantile of |metric(s_rt + tau_h*u) - metric(s_rt)| over 200 random unit
dirs (RandomState(60000 + si), si = global set counter); tau_h = 0.0038.
Tier A (verdict) = m1..m5; Tier B (report) = + m6_vsRT.

Capacity rho = 1 - ||Xtr*shift|| / ||Xtr*s_rt|| for feasible shifts
(theta' = theta0 + shift); lazy corner = shift 0 (theta' = theta0).
Methods: (1) c_grid over (t_row, t_null) on the row/null decomposition of
Delta = -s_rt, exact feasibility, step 0.05 + golden refinement;
(2) c_refined projected gradient in R^512 (5 inits, <=2000 iters,
feasibility backtracking); (3) u_qp: QP upper bound with m3/m5/m6
linearized at s_rt (m1/m2/m4 relaxed), 6-dim dual ascent with convergence
flag; slight solver infeasibility inflates u (safe direction for KILL).

Verdict (frozen): primary k = lazy-transition k (first k where the lazy
corner trips >=1 Tier-A alarm on >=50% of sets; pools in order 43501,
43502 is replication only, then 43801; none by k=1000 -> degenerate
PASS+flag). PASS: Tier-A capacity (median over sets of max(c_grid,
c_refined)) >= 0.10. KILL: that capacity < 0.10 AND median u_qp < 0.10
AND linearization valid on >=50% sets. Otherwise UNDECIDED.
"""
import json
import os
import time

import numpy as np

from run_s0n_step3 import metrics_at
from s0j_ae.linhead import grad_contributions, ridge_head

LAM = 0.1
CAT_CLS = 3
TAU_H = 0.0038
N_DIRS = 200
BAND_Q = 0.95
SET_BASES = {'43501': 44100, '43502': 44200, '43801': 44300}
POOLS = (('43501', (3, 10, 30, 60)), ('43502', (3, 10, 30, 60)),
         ('43801', (100, 300, 1000)))
POOL_SIZE = {'43501': 60, '43502': 60, '43801': 2000}
N_SETS = 10
TIER_A = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv')
TIER_B = TIER_A + ('m6_vsRT',)
GRID_STEP = 0.05
GOLDEN_ITERS = 20
REFINE_ITERS = 400          # per start; frozen total budget 2000 across
#                            the 5 starts (prereg " <= 2000 iterations")
REFINE_STALL = 50
REFINE_INITS = 5
PASS_T = 0.10
SUB_SEED = 41200


def exact_shift(Xp, yF, G, Hinv, n, idx):
    """Woodbury retrain shift (identical formulas to step3.validate_pool)."""
    HX = Hinv @ Xp.T
    g = G[:, idx].sum(axis=1)
    Hg = Hinv @ g
    Bm = HX[:, idx] / np.sqrt(n)
    Kd = (Xp[idx] @ HX[:, idx]) / n
    w = np.linalg.solve(np.eye(len(idx)) - Kd, (Xp[idx] @ Hg) / np.sqrt(n))
    return Hg + Bm @ w


def decompose(delta, Xf):
    """Split delta into row(X_F) projection + null remainder + rank."""
    coef, *_ = np.linalg.lstsq(Xf.T, delta, rcond=None)
    d_row = Xf.T @ coef
    d_null = delta - d_row
    rank = np.linalg.matrix_rank(Xf, tol=1e-9 * max(1.0, np.abs(Xf).max()))
    return d_row, d_null, int(rank)


class SetCtx:
    """Frozen per-set state: reference shift, bands, feasibility, rho."""

    def __init__(self, bed, Xf, yf, Xp, G, idx, si):
        self.bed = bed
        self.Xf, self.yf = Xf, yf
        self.s_rt = exact_shift(Xp, bed.yF_pool, G, bed.Hinv, bed.n, idx)
        self.m_ref = self.metrics(self.s_rt)
        rng = np.random.RandomState(60000 + si)
        acc = {m: [] for m in TIER_B}
        for _ in range(N_DIRS):
            u = rng.randn(bed.d)
            u /= np.linalg.norm(u)
            mm = self.metrics(self.s_rt + TAU_H * u)
            for m in TIER_B:
                acc[m].append(abs(mm[m] - self.m_ref[m]))
        self.band = {m: float(np.quantile(acc[m], BAND_Q)) for m in TIER_B}
        self.X_srt = bed.Xtr @ self.s_rt
        self.denom = float(np.linalg.norm(self.X_srt))

    def metrics(self, shift):
        shift = np.asarray(shift, dtype=float)
        if shift.ndim == 0:                      # scalar (lazy corner 0.0)
            shift = float(shift) * np.ones(self.bed.d)
        return metrics_at(shift, self.s_rt, self.bed.s0_sub, self.bed.Xsub,
                          self.bed.ysub, self.bed.cat_sub, self.Xf, self.yf,
                          self.bed.theta0)

    def feasible(self, shift, tier=TIER_A):
        mm = self.metrics(shift)
        return all(abs(mm[m] - self.m_ref[m]) <= self.band[m] for m in tier)

    def alarms(self, shift, tier=TIER_A):
        mm = self.metrics(shift)
        return [m for m in tier if abs(mm[m] - self.m_ref[m]) > self.band[m]]

    def rho(self, shift):
        return 1.0 - np.linalg.norm(self.bed.Xtr @ shift) / self.denom

    def rho_from_proj(self, Xshift):
        return 1.0 - np.linalg.norm(Xshift) / self.denom


def golden_refine(ctx, base_dir, best_t):
    """Golden-section on t in [0, 1.2] along the best grid direction."""
    lo, hi = 0.0, 1.2
    gr = (np.sqrt(5.0) - 1) / 2
    Xbase = ctx.X_srt.copy()
    Xd = None

    def prep():
        return ctx.bed.Xtr @ (best_t[0] * base_dir[0]
                              + best_t[1] * base_dir[1])

    Xd = prep()

    def obj(t):
        shift = ctx.s_rt + t * (best_t[0] * base_dir[0]
                                + best_t[1] * base_dir[1])
        if not ctx.feasible(shift):
            return -1.0
        return ctx.rho_from_proj(Xbase + t * Xd)

    best = max(obj(1.0), 0.0)
    x1, x2 = hi - gr * hi, gr * hi
    f1, f2 = obj(x1), obj(x2)
    for _ in range(GOLDEN_ITERS):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = gr * hi + (1 - gr) * lo
            f2 = obj(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = (1 - gr) * hi + gr * lo
            f1 = obj(x1)
    return float(max(best, f1, f2, 0.0))


def c_grid(ctx, d_row, d_null, tier=TIER_A):
    # result-identical short-circuit: the lazy corner (shift 0, grid point
    # (1,1)) is feasible => capacity is exactly 1.0 (rho <= 1 always)
    if ctx.feasible(0.0, tier):
        return 1.0, (1.0, 1.0)
    best_rho, best_t = 0.0, (0.0, 0.0)
    Xdr = ctx.bed.Xtr @ d_row
    Xdn = ctx.bed.Xtr @ d_null
    ts = np.arange(0.0, 1.0 + GRID_STEP / 2, GRID_STEP)
    for tr in ts:
        for tn in ts:
            shift = ctx.s_rt + tr * d_row + tn * d_null
            if ctx.feasible(shift, tier):
                r = ctx.rho_from_proj(ctx.X_srt + tr * Xdr + tn * Xdn)
                if r > best_rho:
                    best_rho, best_t = float(r), (float(tr), float(tn))
    if best_t != (0.0, 0.0):
        best_rho = max(best_rho,
                       golden_refine(ctx, (d_row, d_null), best_t))
    return float(best_rho), best_t


def c_refined(ctx, si):
    """Projected gradient on shift in R^512 with feasibility backtracking."""
    # result-identical short-circuit (see c_grid)
    if ctx.feasible(0.0):
        return 1.0, np.zeros(ctx.bed.d)
    A = ctx.bed.XtX / ctx.bed.n
    lbar = max(np.trace(A), 1e-12)
    starts = [ctx.s_rt.copy()]
    for i in range(REFINE_INITS - 1):
        r = np.random.RandomState(46000 + si * REFINE_INITS + i)
        u = r.randn(ctx.bed.d)
        u /= np.linalg.norm(u)
        starts.append(ctx.s_rt + TAU_H * u)
    best_rho, best_shift = 0.0, ctx.s_rt.copy()
    for s0 in starts:
        if not ctx.feasible(s0):
            continue
        shift = s0.copy()
        cur_rho = ctx.rho(shift)
        eta = 0.5 / lbar
        stall = 0
        for _ in range(REFINE_ITERS):
            g = 2.0 * (A @ shift)
            step = eta
            cand = None
            for _ in range(30):
                trial = shift - step * g
                if ctx.feasible(trial):
                    cand = trial
                    break
                step *= 0.5
            if cand is None:
                break
            new_rho = ctx.rho(cand)
            if new_rho > cur_rho + 1e-9:
                shift, cur_rho = cand, new_rho
                stall = 0
            else:
                stall += 1
                eta *= 0.7
                if stall > REFINE_STALL:
                    break
        if cur_rho > best_rho:
            best_rho, best_shift = cur_rho, shift
            if best_rho >= 0.9995:               # saturation, cannot improve
                break
    return float(best_rho), best_shift


def metric_grads(ctx, shift):
    """Analytic gradients of m3/m5/m6 wrt shift at theta0+shift."""
    bed = ctx.bed
    theta = bed.theta0 + shift
    rF = ctx.Xf @ theta - ctx.yf
    rS = bed.Xsub @ theta - bed.ysub
    gF = 2.0 / len(rF) * (ctx.Xf.T @ rF)
    gS = 2.0 / len(rS) * (bed.Xsub.T @ rS)
    return {'m3_Floss': gF, 'm5_adv': gS - gF, 'm6_vsRT': gF}


def u_qp(ctx):
    """Dual ascent for min s^T A s s.t. |J_i(s - s_rt)| <= band_i (i in
    m3/m5/m6; m1/m2/m4 relaxed). Returns (u, s_star, lin_valid, conv)."""
    bed = ctx.bed
    J = metric_grads(ctx, ctx.s_rt)
    keys = ('m3_Floss', 'm5_adv', 'm6_vsRT')
    G = np.stack([J[k] for k in keys] + [-J[k] for k in keys])
    h = np.array([ctx.band[k] for k in keys] + [ctx.band[k] for k in keys])
    h = h + G @ ctx.s_rt
    A = bed.XtX / bed.n
    Ainv = np.linalg.inv(A + 1e-10 * np.eye(bed.d))
    QA = G @ Ainv @ G.T
    step = 1.0 / max(np.linalg.eigvalsh(QA).max(), 1e-12)
    lam = np.zeros(len(h))
    conv = False
    for _ in range(20000):
        s = -0.5 * (Ainv @ (G.T @ lam))
        gl = G @ s - h
        if np.max(np.abs(np.maximum(gl, 0.0))) < 1e-9:
            conv = True
            break
        lam = np.maximum(0.0, lam + step * gl)
    s_star = -0.5 * (Ainv @ (G.T @ lam))
    u = float(1.0 - np.linalg.norm(bed.Xtr @ s_star) / ctx.denom)
    mm = ctx.metrics(s_star)
    valid = True
    for k in keys:
        pred = float(J[k] @ (s_star - ctx.s_rt))
        act = mm[k] - ctx.m_ref[k]
        if abs(pred - act) > 0.3 * max(abs(act), 1e-12):
            valid = False
    return u, s_star, bool(valid), conv


class Bed:
    def __init__(self, feat_path):
        z = np.load(feat_path)
        self.Xtr, self.ytr, self.Xte, self.yte = (z['Xtr'], z['ytr'],
                                                  z['Xte'], z['yte'])
        self.n, self.d = self.Xtr.shape
        self.ybin = np.where(self.ytr == CAT_CLS, 1.0, -1.0)
        self.theta0, self.Hinv = ridge_head(self.Xtr, self.ybin, LAM)
        self.XtX = self.Xtr.T @ self.Xtr
        cat_te = np.where(self.yte == CAT_CLS)[0]
        rng_sub = np.random.RandomState(SUB_SEED)
        sub = rng_sub.choice(self.Xte.shape[0], size=2000, replace=False)
        self.Xsub, self.ysub_raw = self.Xte[sub], self.yte[sub]
        self.ysub = np.where(self.ysub_raw == CAT_CLS, 1.0, -1.0)
        self.cat_sub = self.ysub_raw == CAT_CLS
        self.s0_sub = self.Xsub @ self.theta0
        self.pools = {}
        for pseed, size in POOL_SIZE.items():
            rng = np.random.RandomState(int(pseed))
            self.pools[pseed] = np.sort(rng.choice(
                np.where(self.ytr == CAT_CLS)[0], size=size, replace=False))
        self._pool_cache = {}

    def set_data(self, pool, k, j):
        """pos = pool-relative positions (step3 convention, feeds
        exact_shift); idx_global = training-row indices (feeds Xf/yf)."""
        base = SET_BASES[pool]
        pool_idx = self.pools[pool]
        pos = np.sort(np.random.RandomState(base + j).choice(
            len(pool_idx), size=k, replace=False))
        idx_global = pool_idx[pos]
        if pool not in self._pool_cache:
            Xp = self.Xtr[pool_idx]
            self._pool_cache[pool] = (Xp, self.ybin[pool_idx],
                                      grad_contributions(
                                          Xp, self.ybin[pool_idx],
                                          self.theta0, self.n))
        Xp, yF_pool, G = self._pool_cache[pool]
        self.yF_pool = yF_pool
        return self.Xtr[idx_global], self.ybin[idx_global], Xp, G, pos


def main():
    run_id = 'ew1-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', run_id)
    os.makedirs(outdir, exist_ok=False)
    bed = Bed('research_outputs/s0j_ae/features_inst-2000000002.npz')
    results = {'run_id': run_id, 'tau_h': TAU_H, 'sets': [], 'aggregates': {}}
    si = 0
    for pool, ks in POOLS:
        for k in ks:
            per = {'lazy_frac': [], 'capA': [], 'capB': [], 'qp': [],
                   'qp_valid': [], 'qp_conv': []}
            for j in range(N_SETS):
                Xf, yf, Xp, G, idx = bed.set_data(pool, k, j)
                ctx = SetCtx(bed, Xf, yf, Xp, G, idx, si)
                delta = -ctx.s_rt
                d_row, d_null, rank = decompose(delta, Xf)
                lazy = len(ctx.alarms(0.0, TIER_A)) > 0
                cgA, bt = c_grid(ctx, d_row, d_null, TIER_A)
                cr, shift_best = c_refined(ctx, si)
                cgB, _ = c_grid(ctx, d_row, d_null, TIER_B)
                capA = float(max(cgA, cr))
                u, s_qp, valid, conv = u_qp(ctx)
                binding = []
                if capA > 0:
                    mm = ctx.metrics(shift_best)
                    binding = [m for m in TIER_A
                               if abs(mm[m] - ctx.m_ref[m])
                               >= 0.9 * ctx.band[m]]
                rec = {'pool': pool, 'k': k, 'j': j, 'rank': rank,
                       'null_dim': bed.d - rank,
                       'row_share_param': float(
                           np.linalg.norm(d_row)
                           / np.linalg.norm(delta)),
                       'lazy_alarm': bool(lazy),
                       'lazy_metrics': ctx.alarms(0.0, TIER_A),
                       'c_grid': cgA, 'c_grid_t': list(bt), 'c_refined': cr,
                       'capA': capA, 'capB_grid': float(cgB),
                       'u_qp': u, 'qp_valid': bool(valid),
                       'qp_conv': bool(conv), 'binding': binding}
                results['sets'].append(rec)
                per['lazy_frac'].append(lazy)
                per['capA'].append(capA)
                per['capB'].append(cgB)
                per['qp'].append(u)
                per['qp_valid'].append(valid)
                per['qp_conv'].append(conv)
                si += 1
                print(pool, k, j, 'lazy', int(lazy), 'capA',
                      round(capA, 4), 'capB', round(cgB, 4), 'u',
                      round(u, 4), flush=True)
            results['aggregates']['{}|k={}'.format(pool, k)] = {
                'lazy_frac': float(np.mean(per['lazy_frac'])),
                'capA_med': float(np.median(per['capA'])),
                'capA_q05': float(np.quantile(per['capA'], 0.05)),
                'capA_q95': float(np.quantile(per['capA'], 0.95)),
                'capB_med': float(np.median(per['capB'])),
                'qp_med': float(np.median(per['qp'])),
                'qp_valid_frac': float(np.mean(per['qp_valid'])),
                'qp_conv_frac': float(np.mean(per['qp_conv'])),
            }

    order = [(p, k) for p, ks in POOLS for k in ks]
    k_star = None
    for p, k in order:
        agg = results['aggregates']['{}|k={}'.format(p, k)]
        if agg['lazy_frac'] >= 0.5:
            k_star = (p, k)
            break
    if k_star is None:
        verdict = {'status': 'PASS_DEGENERATE',
                   'note': 'no lazy transition by k=1000: audit cannot '
                           'resolve honest updates at bed scale; escalate '
                           'to author before E-W2'}
    else:
        p, k = k_star
        agg = results['aggregates']['{}|k={}'.format(p, k)]
        cap = agg['capA_med']
        verdict = {'primary': [p, k], 'capA_med': cap,
                   'qp_med': agg['qp_med'],
                   'qp_valid_frac': agg['qp_valid_frac'],
                   'qp_conv_frac': agg['qp_conv_frac'], 'status': None}
        if cap >= PASS_T:
            verdict['status'] = 'PASS'
        elif (agg['qp_med'] < PASS_T
              and agg['qp_valid_frac'] >= 0.5
              and agg['qp_conv_frac'] >= 0.5):
            verdict['status'] = 'KILL'
        else:
            verdict['status'] = 'UNDECIDED'
    results['verdict'] = verdict
    print('VERDICT:', json.dumps(verdict, indent=1), flush=True)
    with open(os.path.join(outdir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write('# UW E-W1 (frozen prereg ed46940 + da3e879)\n\n'
                 'run: python -m uw.run_ew1\nverdict: {}\n'.format(
                     verdict.get('status')))
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
