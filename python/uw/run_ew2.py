"""UW E-W2: wash capacity under augmented audits + forgery motive.

Frozen preregistration: docs/plans/2026-09-22-uw-ew2-preregistration.md
(commit a49b235). Reuses the E-W1 machinery verbatim (Bed/SetCtx/c_grid/
c_refined/decompose and all frozen constants).

Augmented suites on top of Tier A (m1..m5):
  S1: + m7  = |mean(Xsub @ shift)|            (global continuous margin)
  S2: + m7 + m8, m8 = max over 20 size-100 holdout subpopulations b of
      |mean(Xsub[b] @ shift)|                 (multi-direction margins)
  S3: + m9  = ||X_R (shift - s_rt)||/sqrt|R|  (oracle retain-side ceiling)
  S4: + m10 = | ||H_N(theta0+shift) - b_N|| - ||H_N(theta0+s_rt) - b_N|| |
      (pinned-statistic residual, CRSD-S2 style)
Extra bands: 200 tau_h random dirs, 95% quantile, RandomState(62000+si);
subpopulations for m8: RandomState(51000+si); si = per-set counter in run
order (E-W2's own numbering, frozen here).

Adversarial framing: capacity is recomputed per suite (S1/S2 grid+refined;
S3/S4 grid only, anchors). Motive: quality-recovery share
1 - (U0-Uw)/(U0-UR) with U(theta) = mean(yte_bin * (Xte @ theta)),
Uw from the S2-optimal wash (strongest verdict-relevant suite).

Verdict at primary (43501, k=30, median over 10 sets):
  KILL_CHEAP  : cap(S1) < 0.10 AND cap(S2) < 0.10
  KILL_MOTIVE : quality recovery < 0.25 (or no-utility flag on >= 5 sets)
  PASS        : recovery >= 0.25 AND (cap(S1) >= 0.10 OR cap(S2) >= 0.10)
  UNDECIDED   : otherwise; forced UNDECIDED if cap(S4) >= 0.10 (contradicts
                the CRSD r2 archived result for pinned residual checks)
"""
import json
import os
import time

import numpy as np

from uw.run_ew1 import (BAND_Q, Bed, N_DIRS, SetCtx, TAU_H, TIER_A, c_grid,
                        c_refined, decompose)

S1 = TIER_A + ('m7',)
S2 = TIER_A + ('m7', 'm8')
S3 = TIER_A + ('m9',)
S4 = TIER_A + ('m10',)
PASS_T = 0.10
QRET_T = 0.25
K30_POOLS = ('43501', '43502')
KCURVE = (('43501', 60), ('43801', 100))  # erratum: k=100 needs pool 43801


class AugCtx:
    """Wraps a SetCtx with extra audit statistics; delegates the rest."""

    def __init__(self, ctx, extras, si, suite):
        object.__setattr__(self, 'ctx', ctx)
        object.__setattr__(self, 'extras', extras)
        object.__setattr__(self, 'suite', tuple(suite))
        rng = np.random.RandomState(62000 + si)
        bands, refs = {}, {}
        for name, fn in extras.items():
            acc = []
            refs[name] = float(fn(ctx.s_rt))
            for _ in range(N_DIRS):
                u = rng.randn(ctx.bed.d)
                u /= np.linalg.norm(u)
                acc.append(abs(float(fn(ctx.s_rt + TAU_H * u))
                               - refs[name]))
            bands[name] = float(np.quantile(acc, BAND_Q))
        object.__setattr__(self, 'extra_band', bands)
        object.__setattr__(self, 'extra_ref', refs)

    def __getattr__(self, name):
        return getattr(self.__dict__['ctx'], name)

    def feasible(self, shift, tier=None):
        tier = self.__dict__['suite'] if tier is None else tuple(tier)
        six = tuple(m for m in tier if m in self.ctx.band)
        if six and not self.ctx.feasible(shift, six):
            return False
        for m in tier:
            if m in self.__dict__['extras']:
                v = float(self.__dict__['extras'][m](shift))
                if abs(v - self.__dict__['extra_ref'][m]) \
                        > self.__dict__['extra_band'][m]:
                    return False
        return True


def make_extras(bed, ctx_srt_holder, X_R=None):
    """Extra statistics; m9 needs X_R (retain rows) captured per set."""
    def m7(shift):
        return float(np.mean(bed.Xsub @ np.asarray(shift, dtype=float)))

    rng_holder = {}

    def m8(shift):
        s = np.asarray(shift, dtype=float)
        best = 0.0
        for b in rng_holder['pops']:
            best = max(best, abs(float(np.mean(bed.Xsub[b] @ s))))
        return best

    def m9(shift):
        s = np.asarray(shift, dtype=float)
        return float(np.linalg.norm(X_R @ (s - ctx_srt_holder['s_rt']))
                     / np.sqrt(X_R.shape[0]))

    H_N = bed.XtX / bed.n + 0.1 * np.eye(bed.d)
    b_N = bed.Xtr.T @ bed.ybin / bed.n

    def m10(shift):
        s = np.asarray(shift, dtype=float)
        return float(np.linalg.norm(H_N @ (bed.theta0 + s) - b_N))

    def set_populations(rng):
        rng_holder['pops'] = [rng.choice(bed.Xsub.shape[0], size=100,
                                         replace=False) for _ in range(20)]

    return {'m7': m7, 'm8': m8, 'm9': m9, 'm10': m10}, set_populations


def utility(bed, shift):
    theta = bed.theta0 + shift
    yte_bin = np.where(bed.yte == 3, 1.0, -1.0)
    return float(np.mean(yte_bin * (bed.Xte @ theta)))


def main():
    run_id = 'ew2-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', run_id)
    os.makedirs(outdir, exist_ok=False)
    bed = Bed('research_outputs/s0j_ae/features_inst-2000000002.npz')
    yte_bin = np.where(bed.yte == 3, 1.0, -1.0)
    bed.yte_bin = yte_bin
    cat_te = np.where(bed.yte == 3)[0]
    results = {'run_id': run_id, 'sets': [], 'aggregates': {}}
    si = 0
    plan = [(p, 30) for p in K30_POOLS] + [KCURVE[0], KCURVE[1]]
    for pool, k in plan:
        per = {'S1': [], 'S2': [], 'S3': [], 'S4': [], 'qret': [], 'cret': [],
               'no_util': []}
        for j in range(10):
            Xf, yf, Xp, G, pos = bed.set_data(pool, k, j)
            idx_global = bed.pools[pool][pos]
            mask = np.ones(bed.n, dtype=bool)
            mask[idx_global] = False
            X_R = bed.Xtr[mask]
            ctx = SetCtx(bed, Xf, yf, Xp, G, pos, si)
            holder = {'s_rt': ctx.s_rt}
            extras, set_pop = make_extras(bed, holder, X_R)
            set_pop(np.random.RandomState(51000 + si))
            aug1 = AugCtx(ctx, {n: extras[n] for n in ('m7',)}, si, S1)
            aug2 = AugCtx(ctx, {n: extras[n] for n in ('m7', 'm8')}, si, S2)
            delta = -ctx.s_rt
            d_row, d_null, _ = decompose(delta, Xf)
            g1, _ = c_grid(aug1, d_row, d_null, S1)
            g2, t2 = c_grid(aug2, d_row, d_null, S2)
            rec = {'pool': pool, 'k': k, 'j': j,
                   'S1_grid': g1, 'S2_grid': g2}
            if k == 30:
                aug3 = AugCtx(ctx, {n: extras[n] for n in ('m9',)}, si, S3)
                aug4 = AugCtx(ctx, {n: extras[n] for n in ('m10',)}, si, S4)
                r1, _ = c_refined(aug1, si)
                r2, shift2 = c_refined(aug2, si)
                g3, _ = c_grid(aug3, d_row, d_null, S3)
                g4, _ = c_grid(aug4, d_row, d_null, S4)
                cap1, cap2 = max(g1, r1), max(g2, r2)
                u0, ur = utility(bed, np.zeros(bed.d)), utility(bed, ctx.s_rt)
                uw = utility(bed, shift2)
                no_util = abs(u0 - ur) < 1e-12
                qret = (None if no_util else
                        1.0 - (u0 - uw) / (u0 - ur))
                cat0 = float(np.mean(bed.Xte[cat_te]
                                     @ (bed.theta0)))
                catr = float(np.mean(bed.Xte[cat_te]
                                     @ (bed.theta0 + ctx.s_rt)))
                catw = float(np.mean(bed.Xte[cat_te]
                                     @ (bed.theta0 + shift2)))
                cret = (None if abs(cat0 - catr) < 1e-12 else
                        1.0 - (cat0 - catw) / (cat0 - catr))
                rec.update({'S1': cap1, 'S2': cap2, 'S3': g3, 'S4': g4,
                            'qret': qret, 'cret': cret, 'no_util': no_util,
                            'U0': u0, 'UR': ur, 'UW': uw})
                per['S1'].append(cap1)
                per['S2'].append(cap2)
                per['S3'].append(g3)
                per['S4'].append(g4)
                if qret is not None:
                    per['qret'].append(qret)
                if cret is not None:
                    per['cret'].append(cret)
                per['no_util'].append(no_util)
            else:
                r1, _ = c_refined(aug1, si)
                r2, _ = c_refined(aug2, si)
                rec.update({'S1': max(g1, r1), 'S2': max(g2, r2)})
                per['S1'].append(rec['S1'])
                per['S2'].append(rec['S2'])
            results['sets'].append(rec)
            with open(os.path.join(outdir, 'results.json'), 'w') as fh:
                json.dump(results, fh, indent=1)   # per-set checkpoint
            si += 1
            print(pool, k, j, 'S1', round(rec['S1'], 3), 'S2',
                  round(rec['S2'], 3),
                  '' if k != 30 else 'S3 {} S4 {} qret {}'.format(
                      round(g3, 3), round(g4, 3),
                      'None' if qret is None else round(qret, 3)),
                  flush=True)
        agg = {'S1_med': float(np.median(per['S1'])),
               'S2_med': float(np.median(per['S2']))}
        if k == 30:
            agg.update({'S3_med': float(np.median(per['S3'])),
                        'S4_med': float(np.median(per['S4'])),
                        'qret_med': (float(np.median(per['qret']))
                                     if per['qret'] else None),
                        'cret_med': (float(np.median(per['cret']))
                                     if per['cret'] else None),
                        'no_util_frac': float(np.mean(per['no_util']))})
        results['aggregates']['{}|k={}'.format(pool, k)] = agg

    prim = results['aggregates']['43501|k=30']
    qret = prim['qret_med']
    qret_ok = qret is not None and qret >= QRET_T
    verdict = {'primary': ['43501', 30], 'capS1': prim['S1_med'],
               'capS2': prim['S2_med'], 'capS4': prim['S4_med'],
               'qret': qret, 'status': None}
    if prim['S4_med'] >= PASS_T:
        verdict['status'] = 'UNDECIDED'
        verdict['note'] = 'S4 (pinned residual) failed to close -- ' \
                          'contradicts CRSD r2 archive; forced review'
    elif prim['S1_med'] < PASS_T and prim['S2_med'] < PASS_T:
        verdict['status'] = 'KILL_CHEAP'
    elif ((qret is None and prim['no_util_frac'] >= 0.5)
          or (qret is not None and qret < QRET_T)):
        verdict['status'] = 'KILL_MOTIVE'
    elif qret_ok and (prim['S1_med'] >= PASS_T or prim['S2_med'] >= PASS_T):
        verdict['status'] = 'PASS'
    else:
        verdict['status'] = 'UNDECIDED'
    results['verdict'] = verdict
    print('VERDICT:', json.dumps(verdict, indent=1), flush=True)
    with open(os.path.join(outdir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write('# UW E-W2 (frozen prereg a49b235)\n\n'
                 'run: python -m uw.run_ew2\nverdict: {}\n'.format(
                     verdict['status']))
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
