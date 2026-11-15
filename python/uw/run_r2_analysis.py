"""R2-verdict analysis (2026-09-23 external review of Latex/UW).

Answers three review items from unmodified artifacts, CPU-only:

1. Dispersion-association variants. The paper's rho=0.785 was computed
   over 14 as-run (cell, estimate) pairs that include four
   re-measurements of the same configuration and one superseded
   three-seed estimate, and exclude the CIFAR-100 cells. Recompute the
   rank correlation over clearly defined cell sets (as-run 14; unique
   current estimates; +CIFAR-100; the exact Table VI row set; etc.).

2. Honest-seed-luck baseline. Per six-seed unit: how much normalized
   forget-margin elevation is available by *returning the luckiest
   honest retrain* (no optimization, no process violation), versus the
   wash's paired I1. luck_max = (max_s fm_R^s - med)/E where
   E = fm(theta0) - med; wash elevation = (fm_wash - med)/E.

3. Direct margin-band exposure. If the audit adds a band on fm itself
   (half-width = honest ensemble max deviation), which existing washes
   are caught, and what I1 does the band cap analytically.

Writes research_outputs/uw/r2-<ts>/r2_analysis.json. No training.
"""
import glob
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'research_outputs', 'uw',
                   'r2-' + time.strftime('%Y%m%dT%H%M%S'))
E3 = os.path.join(ROOT, 'research_outputs/uw/ew3-20260925T113900')
E5 = os.path.join(ROOT, 'research_outputs/uw/ew5-20260925T160840')
E6 = sorted(glob.glob(os.path.join(
    ROOT, 'research_outputs/uw/ew6-2*')))[-1]
E8D = sorted(glob.glob(os.path.join(
    ROOT, 'research_outputs/uw/ew8D-2*')))[-1]
E7A = sorted(glob.glob(os.path.join(
    ROOT, 'research_outputs/uw/ew7arch-2*')))[-1]


def load(p):
    return json.load(open(p))


def spearman(x, y):
    """Average-tie Spearman rank correlation (matches scipy default)."""
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() /
                 np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


UNITS = {
    # unit key -> (theta0 fm path spec, honest fm sources, opt sources)
    't44500_k30': dict(
        fm0=lambda: load(os.path.join(E3, 'pt.json'))['per_k']['30']['m6_fm'],
        honest=[(E3, 'rt_k30_s44101.json', ('per_k', '30')),
                (E3, 'rt_k30_s44102.json', ('per_k', '30')),
                (E3, 'rt_k30_s44103.json', ('per_k', '30'))]
        + [(E6, 'u0_t44500_k30_s4520%d.json' % i, ()) for i in (1, 2, 3)],
        opt=[(E6, 'u0_t44500_k30_opt_s4410%d.json' % i) for i in (1, 2, 3)]
        + [(E6, 'u0_t44500_k30_opt_s4520%d.json' % i) for i in (1, 2, 3)]),
    't44600_k30': dict(
        fm0=lambda: load(os.path.join(E5, 'armA_t44600_k30_pt.json'))['m6_fm'],
        honest=[(E5, 'armA_t44600_k30_s4500%d.json' % i, ('mm',))
                for i in (1, 2, 3)]
        + [(E6, 'u1_t44600_k30_s4521%d.json' % i, ()) for i in (1, 2, 3)],
        opt=[(E6, 'u1_t44600_k30_opt_s4500%d.json' % i) for i in (1, 2, 3)]
        + [(E6, 'u1_t44600_k30_opt_s4521%d.json' % i) for i in (1, 2, 3)]),
    't44700_k30': dict(
        fm0=lambda: load(os.path.join(E5, 'armA_t44700_k30_pt.json'))['m6_fm'],
        honest=[(E5, 'armA_t44700_k30_s4510%d.json' % i, ('mm',))
                for i in (1, 2, 3)]
        + [(E6, 'u2_t44700_k30_s4522%d.json' % i, ()) for i in (1, 2, 3)],
        opt=[(E6, 'u2_t44700_k30_opt_s4510%d.json' % i) for i in (1, 2, 3)]
        + [(E6, 'u2_t44700_k30_opt_s4522%d.json' % i) for i in (1, 2, 3)]),
    't44700_k300': dict(
        fm0=lambda: load(os.path.join(E5, 'armA_t44700_k300_pt.json'))['m6_fm'],
        honest=[(E5, 'armA_t44700_k300_s4511%d.json' % i, ('mm',))
                for i in (1, 2, 3)]
        + [(E6, 'u3_t44700_k300_s4523%d.json' % i, ()) for i in (1, 2, 3)],
        opt=[(E6, 'u3_t44700_k300_opt_s4511%d.json' % i) for i in (1, 2, 3)]
        + [(E6, 'u3_t44700_k300_opt_s4523%d.json' % i) for i in (1, 2, 3)]),
}


def get(d, path):
    for p in path:
        d = d[p]
    return d


# approximate-operator and endpoint units (same analysis, honest family =
# the operator's own outputs). E-W9F ft (k30/k1000), E-W10J NegGrad k30,
# E-W10I k2000 endpoint (E reconstructed from the recorded gamma since
# theta0's fm on the k2000 F set was computed at runtime only).
OP_UNITS = {
    'ft_k30': dict(
        base=sorted(glob.glob(os.path.join(
            ROOT, 'research_outputs/uw/ew9F-2*')))[-1],
        honest='ft_k30_s4590%d.json', opt='opt_k30_s4590%d.json',
        fm0=lambda: load(os.path.join(E3, 'pt.json'))['per_k']['30']['m6_fm'],
        util0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['30']['util_margin'],
        cat0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['30']['cat_margin']),
    'ft_k1000': dict(
        base=sorted(glob.glob(os.path.join(
            ROOT, 'research_outputs/uw/ew9F-2*')))[-1],
        honest='ft_k1000_s4591%d.json', opt='opt_k1000_s4591%d.json',
        fm0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['1000']['m6_fm'],
        util0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['1000']['util_margin'],
        cat0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['1000']['cat_margin']),
    'ng_k30': dict(
        base=sorted(glob.glob(os.path.join(
            ROOT, 'research_outputs/uw/ew10J-2*')))[-1],
        honest='ng_k30_s4620%d.json', opt='opt_k30_s4620%d.json',
        fm0=lambda: load(os.path.join(E3, 'pt.json'))['per_k']['30']['m6_fm'],
        util0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['30']['util_margin'],
        cat0=lambda: load(os.path.join(
            E3, 'pt.json'))['per_k']['30']['cat_margin']),
    'rt_k2000': dict(
        base=sorted(glob.glob(os.path.join(
            ROOT, 'research_outputs/uw/ew10I-2*')))[-1],
        honest='rt_k2000_s4610%d.json', opt='opt_k2000_s4610%d.json',
        fm0=None, util0=None, cat0=None,
        gamma_fix=0.06455822502297492),
}


def main():
    v6 = load(os.path.join(E6, 'verdict.json'))
    os.makedirs(OUT, exist_ok=True)
    res = {'units': {}, 'rho_variants': {}, 'band_catch': {}}

    # ---- per-unit honest-luck and margin-band analysis ----------------
    for key, spec in UNITS.items():
        fm0 = float(spec['fm0']())
        fms, labels = [], []
        for base, fn, path in spec['honest']:
            d = load(os.path.join(base, fn))
            fms.append(float(get(d, path)['m6_fm'] if path else d['m6_fm']))
            labels.append(os.path.basename(fn))
        med = float(np.median(fms))
        E = fm0 - med                      # deletion delta (positive expected)
        b_pos = max(fms) - med
        b_neg = med - min(fms)
        wash = []
        for base, fn in spec['opt']:
            d = load(os.path.join(base, fn))
            wash.append({'seed': d['seed'], 'fm': d['log'][-1]['fm'],
                         'I1': d['I1'],
                         'util': d['log'][-1]['metrics']['util_margin'],
                         'cat': d['log'][-1]['metrics']['cat_margin']})
        luck_max = b_pos / E
        best_elev = max((w['fm'] - med) / E for w in wash)
        i1_star = max(w['I1'] for w in wash)
        caught = sum(1 for w in wash if w['fm'] > max(fms))
        # analytic paired-I1 cap if audit bands fm at the honest max:
        # fm_w <= max fm  =>  I1 <= (max_fm - fm_base_s)/(fm0 - fm_base_s)
        cap = max((max(fms) - f) / (fm0 - f) for f in fms)
        res['units'][key] = dict(
            fm0=fm0, fm_seeds=dict(zip(labels, fms)), med=med, E=E,
            b_pos=b_pos, b_neg=b_neg, luck_max=luck_max,
            i1_star=i1_star, best_wash_elevation=best_elev,
            increment_over_luck=best_elev - luck_max,
            fm_band_caught='%d/6' % caught, i1_cap_under_fm_band=cap,
            wash=wash)
        # consistency vs ew6 verdict
        u = v6['units'][key]
        assert abs(u['b_over_E'] - max(b_pos, b_neg) / E) < 1e-6, key
        assert abs(u['i1_star'] - i1_star) < 1e-6, key

    # ---- operator / endpoint units ------------------------------------
    for key, spec in OP_UNITS.items():
        honest = [load(os.path.join(spec['base'], spec['honest'] % i))
                  for i in range(1, 7)]
        fms = [float(e['m6_fm']) for e in honest]
        utils = [float(e['util_margin']) for e in honest]
        cats = [float(e['cat_margin']) for e in honest]
        med = float(np.median(fms))
        b_pos = max(fms) - med
        if spec['fm0'] is not None:
            fm0 = float(spec['fm0']())
            E = fm0 - med
        else:            # k2000: reconstruct E from recorded gamma
            b_fm = max(abs(v - med) for v in fms)
            E = b_fm / spec['gamma_fix']
        wash = []
        for i in range(1, 7):
            d = load(os.path.join(spec['base'], spec['opt'] % i))
            # ew9F/ew10 opt JSONs store I1 only; fm reconstructs exactly
            # from the runner's I1 definition (i1(fm_w, fm_base, fm0)).
            fm_base = fms[i - 1]
            if spec['fm0'] is not None:
                fm_w = fm_base + d['I1'] * (fm0 - fm_base)
            else:
                fm_w = fm_base + d['I1'] * E  # E ~ (fm0 - fm_base) approx
            wash.append({'seed': i, 'fm': fm_w, 'I1': d['I1']})
        luck_max = b_pos / E
        best_elev = max((w['fm'] - med) / E for w in wash)
        caught = sum(1 for w in wash if w['fm'] > max(fms))
        ent = dict(
            fm_seeds=fms, med=med, E=E, luck_max=luck_max,
            i1_star=max(w['I1'] for w in wash),
            best_wash_elevation=best_elev,
            increment_over_luck=best_elev - luck_max,
            fm_band_caught='%d/6' % caught, wash=wash,
            fm_reconstructed=spec['fm0'] is None)
        if spec['util0'] is not None:
            u0, c0 = float(spec['util0']()), float(spec['cat0']())
            umed, cmed = float(np.median(utils)), float(np.median(cats))
            if abs(u0 - umed) > 1e-3:
                ent['util_luck_max'] = (max(utils) - umed) / (u0 - umed)
            if abs(c0 - cmed) > 1e-3:
                ent['cat_luck_max'] = (max(cats) - cmed) / (c0 - cmed)
        res['units'][key] = ent

    # utility/cat-margin luck control for the four retrain units
    for key, spec in UNITS.items():
        u = res['units'][key]
        utils, cats = [], []
        for base, fn, path in spec['honest']:
            d = load(os.path.join(base, fn))
            e = get(d, path) if path else d
            utils.append(float(e['util_margin']))
            cats.append(float(e['cat_margin']))
        if key == 't44500_k30':
            pt = load(os.path.join(E3, 'pt.json'))['per_k']['30']
        else:
            pt = load(os.path.join(E5, 'armA_%s_pt.json' % key))
        umed, cmed = float(np.median(utils)), float(np.median(cats))
        u['util_luck_max'] = (max(utils) - umed) / (pt['util_margin'] - umed)
        u['util_wash_best'] = max(
            (w['util'] - umed) / (pt['util_margin'] - umed)
            for w in u['wash'])
        u['cat_luck_max'] = (max(cats) - cmed) / (pt['cat_margin'] - cmed)
        u['cat_wash_best'] = max(
            (w['cat'] - cmed) / (pt['cat_margin'] - cmed)
            for w in u['wash'])

    # ---- rho variants -------------------------------------------------
    cells_asrun = v6['law_cells']
    c100 = load(os.path.join(E8D, 'verdict.json'))['units']
    c100_cells = [('c100_k%d' % k, c100[str(k)]['b_over_E'],
                   c100[str(k)]['i1_star']) for k in (30, 100, 250)]
    mob = load(os.path.join(E7A, 'cell.json'))

    def rho(cells):
        return spearman([c[1] for c in cells], [c[2] for c in cells])

    # unique current estimates: ew6 four + ew4 k300/k1000 (not re-run in
    # ew6) + three deletion families
    cur = {'ew6_t44500_k30', 'ew6_t44600_k30', 'ew6_t44700_k30',
           'ew6_t44700_k300', 'ew4_44500_k300', 'ew4_44500_k1000',
           'armB_topF', 'armB_botF', 'armB_lossF'}
    cells_cur = [(c['cell'], c['b_over_E'], c['i1_star'])
                 for c in cells_asrun if c['cell'] in cur] + c100_cells
    # the exact row set of Table VI (14 rows incl. superseded)
    tableVI = [(0.212, 0.248), (0.255, 0.179), (0.424, 0.326),
               (0.126, 0.214), (0.154, 0.170), (0.113, 0.142),
               (0.076, 0.004), (0.057, 0.107), (0.441, 0.189),
               (0.169, 0.130), (0.307, 0.205), (0.259, 0.152),
               (0.226, 0.156), (0.063, 0.079)]

    res['rho_variants'] = {
        'as_run_14': rho([(c['cell'], c['b_over_E'], c['i1_star'])
                          for c in cells_asrun]),
        'unique_current_plus_c100_%d' % len(cells_cur):
            rho(cells_cur),
        'unique_current_no_c100': rho(
            [(c['cell'], c['b_over_E'], c['i1_star'])
             for c in cells_asrun if c['cell'] in cur]),
        'tableVI_rows_14': spearman([c[0] for c in tableVI],
                                    [c[1] for c in tableVI]),
        'tableVI_rows_13_no_superseded': spearman(
            [c[0] for c in tableVI[:6] + tableVI[7:]],
            [c[1] for c in tableVI[:6] + tableVI[7:]]),
        'as_run_14_plus_c100_17': rho(
            [(c['cell'], c['b_over_E'], c['i1_star'])
             for c in cells_asrun] + c100_cells),
        'all_18_plus_mobilenet': rho(
            [(c['cell'], c['b_over_E'], c['i1_star'])
             for c in cells_asrun] + c100_cells
            + [('mobilenet', mob['b_over_E'], mob['i1_star'])]),
    }

    # ---- paired utility/cat luck control for the motive cells --------
    # qret(u0, u_R, u_W) = (u_W - u_R)/(u0 - u_R) as in run_ew4; the luck
    # control replaces u_W with the pool's luckiest honest value.
    E4 = sorted(glob.glob(os.path.join(ROOT, 'research_outputs/uw/ew4-2*'))
                )[-1]
    ptall = load(os.path.join(E3, 'pt.json'))
    res['motive_luck'] = {}
    for k, s0 in ((30, 44102), (300, 44112), (1000, 44123)):
        pool = [load(os.path.join(E3, 'rt_k%d_s%d.json' % (k, s)))
                ['per_k'][str(k)] for s in
                {30: (44101, 44102, 44103), 300: (44111, 44112, 44113),
                 1000: (44121, 44122, 44123)}[k]]
        if k == 30:      # widen pool with the ew6 re-measurement seeds
            pool += [load(os.path.join(E6, 'u0_t44500_k30_s4520%d.json' % i))
                     for i in (1, 2, 3)]
        base = next(e for e in pool if
                    abs(e['util_margin'] -
                        load(os.path.join(E3, 'rt_k%d_s%d.json'
                                          % (k, s0)))['per_k']
                        [str(k)]['util_margin']) < 1e-9)
        w = load(os.path.join(E4, 'opt_k%d_s%d_S0.json' % (k, s0)))
        u0, c0 = ptall['per_k'][str(k)]['util_margin'], \
            ptall['per_k'][str(k)]['cat_margin']
        ent = {}
        for nm, key0, wkey, pkey in (
                ('util', 'util_margin', 'util_margin_best', 'util_margin'),
                ('cat', 'cat_margin', 'cat_margin_best', 'cat_margin')):
            den_u = (u0 if nm == 'util' else c0) - base[pkey]
            luck = (max(e[pkey] for e in pool) - base[pkey]) / den_u
            washv = (w[wkey] - base[pkey]) / den_u
            ent[nm] = {'wash': washv, 'luck': luck, 'n_pool': len(pool)}
        res['motive_luck'][str(k)] = ent

    json.dump(res, open(os.path.join(OUT, 'r2_analysis.json'), 'w'),
              indent=1)
    print('== per-unit honest-luck baseline ==')
    for k, u in res['units'].items():
        print('%-14s luck_max=%.3f  I1*=%.3f  best_wash_elev=%.3f  '
              'increment=%.3f  fm_band_caught=%s'
              % (k, u['luck_max'], u['i1_star'], u['best_wash_elevation'],
                 u['increment_over_luck'], u['fm_band_caught']))
        if 'util_luck_max' in u and 'util_wash_best' in u:
            print('   util luck/wash: %.3f / %.3f   cat luck/wash: '
                  '%.3f / %.3f'
                  % (u['util_luck_max'], u['util_wash_best'],
                     u['cat_luck_max'], u['cat_wash_best']))
        elif 'util_luck_max' in u:
            print('   util luck_max: %.3f   cat luck_max: %.3f'
                  % (u['util_luck_max'], u.get('cat_luck_max', float('nan'))))
    print('== rho variants ==')
    for k, v in res['rho_variants'].items():
        print('%-34s %.4f' % (k, v))
    print('== motive paired luck control (wash vs luckiest honest) ==')
    for k, ent in res['motive_luck'].items():
        print('k=%-5s util wash/luck %.3f/%.3f   cat wash/luck %.3f/%.3f '
              '(pool %d)'
              % (k, ent['util']['wash'], ent['util']['luck'],
                 ent['cat']['wash'], ent['cat']['luck'],
                 ent['util']['n_pool']))
    print('OUT', OUT)


if __name__ == '__main__':
    main()
