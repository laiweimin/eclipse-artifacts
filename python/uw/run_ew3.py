"""UW E-W3: deep-scale pilot (frozen prereg, proposal version
docs/plans/2026-09-23-uw-ew3-preregistration.md).

Three questions: Q-FLOOR (does theta0 itself pass the deployable audit at
k=30?), Q-FRONTIER (max rollback c* with theta'_c = theta_R + c(theta_0
- theta_R) inside the Tier-A band, and retained influence I1), Q-TRANSFER
(L arm: E-W1 bed machinery on the deep features, same pool, deep F set in
slot j=0 + 9 random draws, capacity curve vs the real-retrain frontier).

Deep arm stages (resumable, one json per unit):
  pt.json            theta_0 per-k metrics (passthrough floor)
  noise_s{S}.json    6 k=0 honest retrains (band + materiality source)
  bands.json         per-k bands  B_m = max_s |m - median|  over the 6
  rt_k{K}_s{S}.json  9 honest retrains on D \\ F_k (+ ckpt)
  fr_k{K}_s{S}.json  interpolation frontier, 21 c-points x metrics/flags
  ft_k1000_s{S}.json ft-1ep base (+ ckpt);  ftfr_k1000_s{S}.json frontier
  verdict.json       frozen §6 logic

L arm (CPU): ew3L-<ts>/larm.json, 10 sets per k in {30,300,1000}.

Deep recipe mirrors run_etie_strong_baseline.fit exactly (S0-J lineage):
ResNet-18 from scratch, SGD lr=0.01 mom=0.9 wd=5e-4, cosine T_max=40,
batch 100, 40 epochs, seed_all deterministic seeding (seed-level only,
cudnn flags untouched -- same as the lineage).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

CAT_CLS = 3
INST = '2000000002'
THETA0_PATH = ('research_outputs/s0c_cifar/instance-{}/benign_50ep.pt'
               .format(INST))          # historical reference only
THETA0_SEED = 44500          # erratum: same-recipe deploy theta_0 (40ep)
FEAT_PATH = 'research_outputs/s0j_ae/features_inst-2000000002.npz'
POOL_SEED = 43801
POOL_N = 2000
K_GRID = (30, 300, 1000)
NOISE_SEEDS = [44000 + s for s in range(1, 7)]
RT_SEEDS = {k: [44100 + 10 * ki + s for s in (1, 2, 3)]
            for ki, k in enumerate(K_GRID)}
FT_SEEDS = [44300 + s for s in (1, 2, 3)]
C_GRID = [round(0.05 * i, 2) for i in range(21)]
SUB_SEED = 52000
SUB_N = 3000
L_BASE = 44400
EPOCHS = 40
TIER_A = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv')
TIER_B = TIER_A + ('m6_fm',)
MATERIAL_X = 3.0
PASS_I1 = 0.20


# ---------------- pure-numpy metric machinery (unit-tested) ----------------

def ks_stat(a, b):
    """Two-sample KS statistic (no scipy dependency)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / len(a)
    cdf_b = np.searchsorted(b, allv, side='right') / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def auc_mw(scores_pos, scores_neg):
    """Mann-Whitney AUC with tie-averaged ranks."""
    sp, sn = len(scores_pos), len(scores_neg)
    allv = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(allv, kind='mergesort')
    sv = allv[order]
    ranks = np.empty(len(allv), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[:sp].sum()
    return float((r_pos - sp * (sp + 1) / 2.0) / (sp * sn))


def logistic_auc(feats_pos, feats_neg, steps=200, lr=0.1):
    """Hand-rolled logistic regression on standardized [CE, margin]
    features (plus bias), full-batch GD; AUC of the linear score."""
    X = np.vstack([feats_pos, feats_neg]).astype(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd
    X = np.column_stack([X, np.ones(len(X))])
    y = np.concatenate([np.ones(len(feats_pos)), np.zeros(len(feats_neg))])
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        w = w - lr * (X.T @ (p - y)) / n
    s = X @ w
    return auc_mw(s[:len(feats_pos)], s[len(feats_pos):])


def _ce_rows(z, y):
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return lse - z[np.arange(len(y)), y]


def _margin_rows(z, y):
    zy = z[np.arange(len(y)), y]
    zc = z.copy()
    zc[np.arange(len(y)), y] = -1e18
    return zy - zc.max(axis=1)


def metrics_from_logits(z_F, y_F, z_sub, y_sub, z_te, y_te):
    """Six frozen audit metrics + utility readouts from logits arrays."""
    pred_te = z_te.argmax(axis=1)
    m1 = float((pred_te == y_te).mean())
    cat_te = y_te == CAT_CLS
    m2 = float((pred_te[cat_te] == y_te[cat_te]).mean())
    ce_F, ce_S = _ce_rows(z_F, y_F), _ce_rows(z_sub, y_sub)
    mg_F, mg_S = _margin_rows(z_F, y_F), _margin_rows(z_sub, y_sub)
    m5 = logistic_auc(np.column_stack([ce_F, mg_F]),
                      np.column_stack([ce_S, mg_S]))
    return {
        'm1_acc': m1,
        'm2_catacc': m2,
        'm3_Floss': float(ce_F.mean()),
        'm4_KS': ks_stat(ce_F, ce_S),
        'm5_adv': m5,
        'm6_fm': float(mg_F.mean()),
        'util_margin': float(_margin_rows(z_te, y_te).mean()),
        'cat_margin': float(_margin_rows(z_te, y_te)[cat_te].mean()),
    }


def bands_from_noise(noise_dicts):
    """B_m = max_s |m(theta^0_s) - median_s m| (widest defensible band,
    0% FPR on the calibration set -- most wash-favorable, PASS is the
    conservative direction)."""
    out = {}
    keys = noise_dicts[0].keys()
    for m in keys:
        v = np.array([e[m] for e in noise_dicts], dtype=float)
        out[m] = float(np.max(np.abs(v - np.median(v))))
    return out


def accept(mm, centers, bands, tier=TIER_A, scale=1.0):
    viol = [m for m in tier
            if abs(mm[m] - centers[m]) > scale * bands[m]]
    return (not viol), viol


def frontier(cs, pass_flags):
    """Max passing c on the grid; flag honest-rejection (c=0 fails)."""
    if not pass_flags[0]:
        return 0.0, True
    best = cs[0]
    for c, p in zip(cs, pass_flags):
        if p:
            best = c
    return best, False


def i1(fm_c, fm_R, fm_0):
    return float((fm_c - fm_R) / (fm_0 - fm_R))


def material(e_k, b_fm):
    return bool(e_k >= MATERIAL_X * b_fm)


def build_pool(cat_idx):
    return np.sort(np.random.RandomState(POOL_SEED).choice(
        cat_idx, size=POOL_N, replace=False))


def f_deep(pool, k):
    """Frozen deep-arm deletion set: first k of RandomState(43900+k)
    permutation of the pool."""
    return np.sort(np.random.RandomState(43900 + k).permutation(pool)[:k])


def l_sets(pool, k, ki):
    """L-arm sets: slot 0 = the deep frozen F set; slots 1..9 random."""
    out = [f_deep(pool, k)]
    for j in range(1, 10):
        pos = np.sort(np.random.RandomState(
            L_BASE + ki * 10 + j).choice(len(pool), size=k, replace=False))
        out.append(pool[pos])
    return out


def interpolate_sd(sd_R, sd_0, c):
    """theta' = theta_R + c (theta_0 - theta_R); integer buffers (BN
    num_batches_tracked) taken from theta_R."""
    out = {}
    for key, t in sd_R.items():
        t0 = sd_0[key]
        if t.is_floating_point():
            out[key] = t + c * (t0 - t)
        else:
            out[key] = t.clone()
    return out


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------- deep arm (torch; imported lazily) ----------------

def seed_all(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_deep(cli):
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    if cli.outdir:
        outdir = cli.outdir              # resume into an existing run dir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ew3-smoke-' if cli.smoke else 'ew3-'
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device('cuda:{}'.format(cli.gpu))
    epochs = 2 if cli.smoke else EPOCHS
    k_grid = (30, 300) if cli.smoke else K_GRID
    noise_seeds = NOISE_SEEDS[:2] if cli.smoke else NOISE_SEEDS
    rt_seeds = ({k: RT_SEEDS[k][:1] for k in k_grid} if cli.smoke
                else {k: list(RT_SEEDS[k]) for k in k_grid})
    ft_seeds = FT_SEEDS[:1] if cli.smoke else list(FT_SEEDS)
    c_grid = [0.0, 0.5, 1.0] if cli.smoke else C_GRID

    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    # erratum 2026-09-23: audit metrics on stored records are evaluated in
    # their canonical form (ToTensor-only, [0,1]) -- the train transform's
    # random augmentation made every train-side evaluation a fresh draw,
    # contaminating bands/centers with augmentation noise.  Consistent with
    # the project-wide frozen eval semantics (CIFAR = ToTensor-only).
    trainset_eval = CIFAR10(train=True,
                            transform=_tf.Compose([_tf.ToTensor()]))
    assert np.array_equal(np.asarray(trainset_eval.targets),
                          np.asarray(trainset.targets))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)

    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    fsets = {k: f_deep(pool, k) for k in k_grid}
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    # sub excludes the pool (superset of every F_k), so one frozen
    # subsample is valid at every k -- prereg wording "non-F records",
    # union-exclusion interpretation, noted for §8.
    pos_of = {k: np.searchsorted(pool, fsets[k]) for k in k_grid}
    y_pool = yfull[pool]

    with open(os.path.join(outdir, 'config.json'), 'w') as fh:
        json.dump({'command': 'python -m uw.run_ew3 --gpu {}'.format(
            cli.gpu) + (' --smoke' if cli.smoke else ''),
            'inst': INST, 'theta0': THETA0_PATH, 'epochs': epochs,
            'k_grid': list(k_grid), 'noise_seeds': noise_seeds,
            'rt_seeds': rt_seeds, 'ft_seeds': ft_seeds,
            'c_grid': c_grid, 'sub_seed': SUB_SEED, 'sub_n': SUB_N,
            'pool_seed': POOL_SEED, 'pool_sha': sha256_arr(pool),
            'f_sha': {str(k): sha256_arr(v) for k, v in fsets.items()},
            'sub_sha': sha256_arr(sub), 'smoke': bool(cli.smoke),
            'theta0_seed': THETA0_SEED,
            'eval_semantics': ('train-side audit records evaluated in '
                               'canonical ToTensor-only form (erratum '
                               '2026-09-23; the first run evaluated them '
                               'through the augmented train transform, '
                               'making every train-side metric a fresh '
                               'augmentation draw)'),
            'torch': torch.__version__, 'numpy': np.__version__,
            'material_x': MATERIAL_X, 'pass_i1': PASS_I1}, fh, indent=1)

    model = get_model('ResNet18').to(device)

    def collect(loader):
        zs, ys = [], []
        model.eval()
        with torch.no_grad():
            for _id, x, y in loader:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        return torch.cat(zs).numpy(), torch.cat(ys).numpy()

    test_y = {}

    def eval_model(sd):
        """z_te recomputed per model (it changes with sd); only the test
        labels are cached."""
        model.load_state_dict(sd)
        z_te, y_te = collect(testloader)
        test_y['y_te'] = y_te
        idx_ps = np.concatenate([pool, sub])
        loader_ps = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
            shuffle=False)
        zs, ids = [], []
        model.eval()
        with torch.no_grad():
            for gid, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
                ids.append(gid)
        z_ps = torch.cat(zs).numpy()
        ids_ps = torch.cat(ids).numpy()
        row = {int(g): i for i, g in enumerate(ids_ps)}
        rows_pool = np.array([row[int(g)] for g in pool])
        rows_sub = np.array([row[int(g)] for g in sub])
        z_pool, z_sub = z_ps[rows_pool], z_ps[rows_sub]
        y_sub = yfull[sub]
        per_k = {str(k): metrics_from_logits(
            z_pool[pos_of[k]], y_pool[pos_of[k]], z_sub, y_sub, z_te, y_te)
            for k in k_grid}
        util = {m: per_k[str(k_grid[0])][m]
                for m in ('m1_acc', 'm2_catacc', 'util_margin',
                          'cat_margin')}
        return per_k, util

    def fit(keep, seed, epochs_):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_)
        lf = torch.nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        for _ep in range(epochs_):
            m.train()
            for _ids, x, _yy in loader:
                opt.zero_grad()
                lf(m(x.to(device)), torch.as_tensor(yfull[ _ids.tolist() ])
                   .to(device)).backward()
                opt.step()
            sched.step()
        m.eval()
        return m

    def stage_done(name):
        return os.path.exists(os.path.join(outdir, name))

    def load_json(name):
        with open(os.path.join(outdir, name)) as fh:
            return json.load(fh)

    def dump_json(obj, name):
        with open(os.path.join(outdir, name), 'w') as fh:
            json.dump(obj, fh, indent=1)

    def sd_of(m):
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    # ---- stage 0: deploy theta_0, same recipe as the retrain family
    # (erratum 2026-09-23: benign_50ep is a different recipe -- 50ep --
    # so its floor rejections mixed recipe mismatch with deletion
    # response; the prereg questions are within-recipe by construction)
    ck_theta0 = os.path.join(ckpt_dir, 'theta0.pt')
    if os.path.exists(ck_theta0):
        sd0 = torch.load(ck_theta0, map_location=device)
    else:
        m0 = fit(np.arange(n), THETA0_SEED, epochs)
        sd0 = sd_of(m0)
        torch.save(sd0, ck_theta0)
        print('theta0 trained (seed {}, {}ep)'.format(THETA0_SEED, epochs),
              flush=True)

    # ---- stage 1: theta0 passthrough floor
    if not stage_done('pt.json'):
        per_k, util = eval_model(sd0)
        dump_json({'per_k': per_k, 'util': util}, 'pt.json')
        print('theta0 floor done', flush=True)

    # ---- stage 2: noise retrains (k=0 honest stochasticity)
    for s in noise_seeds:
        fn = 'noise_s{}.json'.format(s)
        if stage_done(fn):
            continue
        ck = os.path.join(ckpt_dir, 'noise_s{}.pt'.format(s))
        if os.path.exists(ck):
            sd = torch.load(ck, map_location=device)
        else:
            m = fit(np.arange(n), s, epochs)
            sd = sd_of(m)
            torch.save(sd, ck)
        per_k, util = eval_model(sd)
        dump_json({'seed': s, 'per_k': per_k, 'util': util}, fn)
        print('noise', s, 'm1', round(util['m1_acc'], 4), flush=True)

    # ---- stage 3: bands per k
    noise = [load_json('noise_s{}.json'.format(s))['per_k']
             for s in noise_seeds]
    bands = {str(k): bands_from_noise([e[str(k)] for e in noise])
             for k in k_grid}
    dump_json(bands, 'bands.json')

    # ---- stage 4: honest retrains theta_R per (k, seed)
    for k in k_grid:
        keep_k = np.setdiff1d(np.arange(n), fsets[k])
        for s in rt_seeds[k]:
            fn = 'rt_k{}_s{}.json'.format(k, s)
            if stage_done(fn):
                continue
            ck = os.path.join(ckpt_dir, fn.replace('.json', '.pt'))
            if os.path.exists(ck):
                sd = torch.load(ck, map_location=device)
            else:
                m = fit(keep_k, s, epochs)
                sd = sd_of(m)
                torch.save(sd, ck)
            per_k, util = eval_model(sd)
            dump_json({'k': k, 'seed': s, 'per_k': per_k, 'util': util}, fn)
            print('rt', k, s, 'm1', round(util['m1_acc'], 4), flush=True)

    # ---- stage 5: centers per k (median over the seeds)
    centers = {}
    for k in k_grid:
        entries = [load_json('rt_k{}_s{}.json'.format(k, s))['per_k']
                   [str(k)] for s in rt_seeds[k]]
        centers[str(k)] = {m: float(np.median([e[m] for e in entries]))
                           for m in entries[0]}
    dump_json(centers, 'centers.json')

    # ---- stage 6+7: interpolation frontiers
    for k in k_grid:
        pt = load_json('pt.json')['per_k'][str(k)]
        for s in rt_seeds[k]:
            fn = 'fr_k{}_s{}.json'.format(k, s)
            if stage_done(fn):
                continue
            sd_R = torch.load(os.path.join(
                ckpt_dir, 'rt_k{}_s{}.pt'.format(k, s)), map_location=device)
            rt_rec = load_json('rt_k{}_s{}.json'.format(k, s))
            fm_R = rt_rec['per_k'][str(k)]['m6_fm']
            fm_0 = pt['m6_fm']
            recs = []
            for c in c_grid:
                sd = sd_R if c == 0.0 else interpolate_sd(sd_R, sd0, c)
                per_k, _u = eval_model(sd)
                mm = per_k[str(k)]
                pa, _va = accept(mm, centers[str(k)], bands[str(k)], TIER_A)
                pa_h, _vh = accept(mm, centers[str(k)], bands[str(k)],
                                   TIER_A, 0.5)
                pb, _vb = accept(mm, centers[str(k)], bands[str(k)], TIER_B)
                recs.append({'c': c, 'metrics': mm, 'passA': pa,
                             'passA_half': pa_h, 'passB': pb})
            csA, hres = frontier(c_grid, [r['passA'] for r in recs])
            csH, _ = frontier(c_grid, [r['passA_half'] for r in recs])
            csB, _ = frontier(c_grid, [r['passB'] for r in recs])
            mmA = next(r['metrics'] for r in recs if r['c'] == csA)
            mmB = next(r['metrics'] for r in recs if r['c'] == csB)
            dump_json({'k': k, 'seed': s, 'recs': recs, 'cstar_A': csA,
                       'cstar_A_half': csH, 'cstar_B': csB,
                       'honest_rejected': hres,
                       'I1_A': i1(mmA['m6_fm'], fm_R, fm_0),
                       'I1_B': i1(mmB['m6_fm'], fm_R, fm_0),
                       'm1_cstar_A': mmA['m1_acc'],
                       'cat_margin_cstar_A': mmA['cat_margin']}, fn)
            print('fr', k, s, 'c*A', csA, 'c*B', csB, flush=True)

    # ---- stage 8: ft-1ep + rollback arm (k=1000 only)
    if 1000 in k_grid:
        k = 1000
        keep_k = np.setdiff1d(np.arange(n), fsets[k])
        for s in ft_seeds:
            fn = 'ft_k1000_s{}.json'.format(s)
            ffn = 'ftfr_k1000_s{}.json'.format(s)
            if stage_done(fn) and stage_done(ffn):
                continue
            ck = os.path.join(ckpt_dir, fn.replace('.json', '.pt'))
            if os.path.exists(ck):
                sd_ft = torch.load(ck, map_location=device)
            else:
                seed_all(s)
                m = get_model('ResNet18').to(device)
                m.load_state_dict(sd0)
                opt = torch.optim.SGD(m.parameters(), lr=0.01,
                                      momentum=0.9, weight_decay=5e-4)
                lf = torch.nn.CrossEntropyLoss()
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(trainset, keep_k),
                    batch_size=100, shuffle=True)
                m.train()
                for _ids, x, _yy in loader:
                    opt.zero_grad()
                    lf(m(x.to(device)), torch.as_tensor(
                        yfull[_ids.tolist()]).to(device)).backward()
                    opt.step()
                m.eval()
                sd_ft = sd_of(m)
                torch.save(sd_ft, ck)
            per_k, util = eval_model(sd_ft)
            dump_json({'k': k, 'seed': s, 'per_k': per_k, 'util': util}, fn)
            pt = load_json('pt.json')['per_k'][str(k)]
            rt_med_fm = centers[str(k)]['m6_fm']
            recs = []
            for c in c_grid:
                sd = sd_ft if c == 0.0 else interpolate_sd(sd_ft, sd0, c)
                per_k, _u = eval_model(sd)
                mm = per_k[str(k)]
                pa, _va = accept(mm, centers[str(k)], bands[str(k)], TIER_A)
                pb, _vb = accept(mm, centers[str(k)], bands[str(k)], TIER_B)
                recs.append({'c': c, 'metrics': mm, 'passA': pa, 'passB': pb})
            csA, hres = frontier(c_grid, [r['passA'] for r in recs])
            csB, _ = frontier(c_grid, [r['passB'] for r in recs])
            mmA = next(r['metrics'] for r in recs if r['c'] == csA)
            dump_json({'k': k, 'seed': s, 'recs': recs, 'cstar_A': csA,
                       'cstar_B': csB, 'honest_rejected': hres,
                       'I1_A': i1(mmA['m6_fm'],
                                  float(np.median(
                                      [load_json('rt_k1000_s{}.json'.format(
                                          rs))['per_k']['1000']['m6_fm']
                                       for rs in rt_seeds[1000]])),
                                  pt['m6_fm']),
                       'm1_cstar_A': mmA['m1_acc']}, ffn)
            print('ftfr', s, 'c*A', csA, flush=True)

    # ---- stage 9: verdict (frozen §6)
    pt_all = load_json('pt.json')['per_k']
    per_k_out = {}
    for k in k_grid:
        pt = pt_all[str(k)]
        b_k, c_k = bands[str(k)], centers[str(k)]
        e_k = abs(pt['m6_fm'] - c_k['m6_fm'])
        mat = material(e_k, b_k['m6_fm'])
        frs = [load_json('fr_k{}_s{}.json'.format(k, s))
               for s in rt_seeds[k]]
        i1a = float(np.median([r['I1_A'] for r in frs]))
        i1b = float(np.median([r['I1_B'] for r in frs]))
        hres_frac = float(np.mean([r['honest_rejected'] for r in frs]))
        okA, violA = accept(pt, c_k, b_k, TIER_A)
        okA_h, _ = accept(pt, c_k, b_k, TIER_A, 0.5)
        per_k_out[str(k)] = {
            'E_k': e_k, 'band_fm': b_k['m6_fm'], 'material': mat,
            'floor_pass_full': okA, 'floor_viol_full': violA,
            'floor_pass_half': okA_h,
            'I1_A_med': i1a, 'I1_B_med': i1b,
            'cstar_A_med': float(np.median([r['cstar_A'] for r in frs])),
            'cstar_B_med': float(np.median([r['cstar_B'] for r in frs])),
            'honest_rejected_frac': hres_frac,
            'm1_cstar_med': float(np.median([r['m1_cstar_A'] for r in frs])),
            'm1_center': c_k['m1_acc'],
        }
    mat_ks = [k for k in k_grid if per_k_out[str(k)]['material']]
    hok_ks = [k for k in mat_ks
              if per_k_out[str(k)]['honest_rejected_frac'] < 0.5]
    if not mat_ks:
        status, reason = 'UNDECIDED', 'materiality gate: no k with E>=3B'
    elif any(per_k_out[str(k)]['I1_A_med'] >= PASS_I1 for k in mat_ks):
        if all(per_k_out[str(k)]['I1_B_med'] < PASS_I1 for k in mat_ks):
            status, reason = 'UNDECIDED', (
                'forced rule: Tier-A passes but Tier-B closes at every '
                'material k (conflicts E-W1/E-W2 band-binding finding)')
        else:
            status, reason = 'PASS', (
                'material k with Tier-A I1 >= 0.20: '
                + ','.join(str(k) for k in mat_ks
                           if per_k_out[str(k)]['I1_A_med'] >= PASS_I1))
    elif hok_ks and all(per_k_out[str(k)]['I1_A_med'] < PASS_I1
                        for k in hok_ks):
        status, reason = 'KILL', (
            'all material, honest-accepted k have Tier-A I1 < 0.20')
    else:
        status, reason = 'UNDECIDED', (
            'material k exist with honest-rejected majority; audit '
            'mis-calibrated there, KILL confounded')
    verdict = {'status': status, 'reason': reason,
               'per_k': per_k_out,
               'ft_anchor': ({'I1_A_med': float(np.median(
                   [load_json('ftfr_k1000_s{}.json'.format(s))['I1_A']
                    for s in ft_seeds])),
                   'cstar_A_med': float(np.median(
                       [load_json('ftfr_k1000_s{}.json'.format(s))['cstar_A']
                        for s in ft_seeds]))}
                  if 1000 in k_grid else None),
               'note': 'L arm runs separately (--mode L); transfer check '
                       'is assembled in the prereg §8 record.'}
    dump_json(verdict, 'verdict.json')
    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write('# UW E-W3 deep arm (frozen prereg 2026-09-23)\n\n'
                 'command: python -m uw.run_ew3 --gpu {}{}\nverdict: {}'
                 ' -- {}\n'.format(cli.gpu,
                                   ' --smoke' if cli.smoke else '',
                                   status, reason))
    print('VERDICT:', status, '--', reason, flush=True)
    print('saved', outdir, flush=True)


# ---------------- L arm (CPU, bed machinery) ----------------

def run_L(cli):
    from s0j_ae.linhead import grad_contributions
    from uw.run_ew1 import Bed, SetCtx, TIER_A, TIER_B, c_grid, c_refined, \
        decompose

    tag = 'ew3L-smoke-' if cli.smoke else 'ew3L-'
    outdir = os.path.join('research_outputs', 'uw',
                          tag + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=False)
    bed = Bed(FEAT_PATH)
    pool = bed.pools['43801']
    k_grid = (30,) if cli.smoke else K_GRID
    sets_out = {'run_id': os.path.basename(outdir), 'sets': [],
                'aggregates': {}}
    for ki, k in enumerate(k_grid):
        fs = l_sets(pool, k, ki)
        per = {'lazy': [], 'capA': [], 'capB': []}
        for j, g in enumerate(fs):
            pos = np.searchsorted(pool, g)
            Xp = bed.Xtr[pool]
            if not hasattr(bed, '_ew3_cache'):
                bed.yF_pool = bed.ybin[pool]
                bed._ew3_cache = (Xp, bed.yF_pool,
                                  grad_contributions(Xp, bed.yF_pool,
                                                     bed.theta0, bed.n))
            Xp, yFp, G = bed._ew3_cache
            ctx = SetCtx(bed, bed.Xtr[g], bed.ybin[g], Xp, G, pos,
                         ki * 10 + j)
            d_row, d_null, rank = decompose(-ctx.s_rt, bed.Xtr[g])
            lazy = len(ctx.alarms(0.0, TIER_A)) > 0
            cgA, _ = c_grid(ctx, d_row, d_null, TIER_A)
            cr, _ = c_refined(ctx, ki * 10 + j)
            cgB, _ = c_grid(ctx, d_row, d_null, TIER_B)
            capA = float(max(cgA, cr))
            per['lazy'].append(lazy)
            per['capA'].append(capA)
            per['capB'].append(float(cgB))
            sets_out['sets'].append(
                {'k': k, 'j': j, 'is_deep_F': j == 0, 'rank': rank,
                 'null_dim': bed.d - rank, 'lazy': bool(lazy),
                 'capA': capA, 'capB': float(cgB)})
            print('L', k, j, 'lazy', int(lazy), 'capA', round(capA, 4),
                  flush=True)
        sets_out['aggregates']['k={}'.format(k)] = {
            'lazy_frac': float(np.mean(per['lazy'])),
            'capA_med': float(np.median(per['capA'])),
            'capA_q05': float(np.quantile(per['capA'], 0.05)),
            'capA_q95': float(np.quantile(per['capA'], 0.95)),
            'capB_med': float(np.median(per['capB'])),
        }
    with open(os.path.join(outdir, 'larm.json'), 'w') as fh:
        json.dump(sets_out, fh, indent=1)
    print('saved', outdir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='deep', choices=['deep', 'L'])
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None,
                    help='resume an existing run directory (skips '
                         'completed stage files)')
    cli = ap.parse_args()
    if cli.mode == 'deep':
        run_deep(cli)
    else:
        run_L(cli)


if __name__ == '__main__':
    main()
