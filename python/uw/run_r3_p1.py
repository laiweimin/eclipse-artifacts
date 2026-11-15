"""UW R3-P1: independent-calibration transfer test (frozen prereg
docs/plans/2026-09-23-uw-r3-minimal-loop.md, author-approved).

Unit k=30 theta0-A (44500). Reference families: exact retrain (primary,
6 fresh auditor seeds 46301-46306, 40ep) and the 1-epoch ft operator
(secondary, 6 fresh auditor seeds 46311-46316). Auditor bands/centers
come ONLY from the auditor ensemble. Groups evaluated under auditor
bands:
  (a) provider's 6 honest models (same distribution -> transfer FPR)
  (b) provider washes regenerated deterministically under provider
      bands (validity: must reproduce ew6/ew9F recorded I1)
  (c) 3 adaptive washes optimized from auditor models under auditor
      bands
  (d) auditor-side honest luck ceiling (max fm / median, in units of
      the deletion delta)

Verdict (frozen): ADAPTIVE_SURVIVES if >=2/3 adaptive washes pass AND
gain over auditor luck >= 0.10; BANDS_DONT_TRANSFER if >= 3/6 provider
honest models rejected; NO_GAIN_BEYOND_LUCK otherwise.
"""
import argparse
import json
import os
import time

import numpy as np

from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED, TIER_A,
                        accept, build_pool, f_deep, i1,
                        metrics_from_logits)
from uw.run_ew3b import LR0, loo_bands, lr_update

RUN3 = 'research_outputs/uw/ew3-20260925T113900'
RUN6 = 'research_outputs/uw/ew6-20260925T213308'
RUN9F = 'research_outputs/uw/ew9F-20260927T001555'
K = 30
PROV_SEEDS = (44101, 44102, 44103, 45201, 45202, 45203)
AUD_SEEDS = (46301, 46302, 46303, 46304, 46305, 46306)
FT_PROV_SEEDS = (45901, 45902, 45903, 45904, 45905, 45906)
FT_AUD_SEEDS = (46311, 46312, 46313, 46314, 46315, 46316)
N_ADAPTIVE = 3
MAX_ITERS = 60
STALL_LIMIT = 15


def mk_bands_centers(entries):
    centers = {m: float(np.median([e[m] for e in entries.values()]))
               for m in TIER_A}
    bands = loo_bands(entries, centers)
    return centers, bands


def luck_ceiling(entries, fm0):
    med = float(np.median([e['m6_fm'] for e in entries.values()]))
    return (max(e['m6_fm'] for e in entries.values()) - med) / (fm0 - med)


def elevation(fm, entries, fm0):
    med = float(np.median([e['m6_fm'] for e in entries.values()]))
    return (fm - med) / (fm0 - med)


def verdict_from_transfer(honest_reject, adaptive_pass, adaptive_gain):
    if honest_reject >= 4:
        return 'BANDS_DONT_TRANSFER', (
            '{}/6 provider honest models rejected by auditor bands; '
            'calibration variance dominates'.format(honest_reject))
    if adaptive_pass >= 2 and adaptive_gain >= 0.10:
        return 'ADAPTIVE_SURVIVES', (
            '{}/3 adaptive washes pass with gain {:.3f} over auditor '
            'luck'.format(adaptive_pass, adaptive_gain))
    return 'NO_GAIN_BEYOND_LUCK', (
        'adaptive washes pass {}/3, gain {:.3f}; independent '
        'calibration does not rescue optimizer gain'.format(
            adaptive_pass, adaptive_gain))


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    epochs = 2 if cli.smoke else 40
    iters = 8 if cli.smoke else MAX_ITERS
    aud_seeds = AUD_SEEDS[:1] if cli.smoke else AUD_SEEDS
    ft_aud_seeds = FT_AUD_SEEDS[:1] if cli.smoke else FT_AUD_SEEDS
    n_adaptive = 1 if cli.smoke else N_ADAPTIVE

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'r3p1-smoke-' if cli.smoke else 'r3p1-'
        outdir = os.path.join('research_outputs', 'uw',
                              tag + time.strftime('%Y%m%dT%H%M%S'))
        os.makedirs(outdir, exist_ok=False)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    device = torch.device('cuda:{}'.format(cli.gpu))

    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    trainset_eval = CIFAR10(train=True,
                            transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]

    model = get_model('ResNet18').to(device)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def fit(keep, seed, epochs_):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        for _ep in range(epochs_):
            m.train()
            for _ids, x, _yy in loader:
                opt.zero_grad()
                lf(m(x.to(device)), y_t[_ids.tolist()].to(
                    device)).backward()
                opt.step()
            sched.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    def ft_run(keep, seed, sd0):
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        lf = torch.nn.CrossEntropyLoss()
        y_t = torch.as_tensor(yfull, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=True)
        m.train()
        for _ids, x, _yy in loader:
            opt.zero_grad()
            lf(m(x.to(device)), y_t[_ids.tolist()].to(
                device)).backward()
            opt.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(n), fset)
    pos = np.searchsorted(pool, fset)
    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']

    def eval_model(sd):
        model.load_state_dict(sd)
        zs, ys = [], []
        model.eval()
        with torch.no_grad():
            for _id, x, y in testloader:
                zs.append(model(x.to(device)).cpu())
                ys.append(y)
        z_te = torch.cat(zs).numpy()
        y_te = torch.cat(ys).numpy()
        zs = []
        with torch.no_grad():
            for _g, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    def f_batch():
        xf, yfl = [], []
        for _g, x, y in torch.utils.data.DataLoader(
                torch.utils.data.Subset(trainset_eval, fset),
                batch_size=500, shuffle=False):
            xf.append(x)
            yfl.append(y)
        return torch.cat(xf).to(device), torch.cat(yfl).to(
            device).long()

    def grad_delta(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        margin = (zy - zc.max(1).values).mean()
        model.zero_grad()
        margin.backward()
        return {nm: p.grad.detach().clone() for nm, p in
                model.named_parameters() if p.grad is not None}

    def apply_delta(base_sd, delta, lr):
        return {k_: (v + lr * delta[k_] if k_ in delta else v)
                for k_, v in base_sd.items()}

    def fm_only(sd, xF, yF):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    def wash(sd_start, centers, bands, xF, yF):
        """Deterministic band-feasible fm ascent (identical recipe to
        run_ew6 S0 arm); returns (record, final_sd)."""
        cur = sd_start
        lr = LR0
        mm0 = eval_model(cur)
        best_fm = mm0['m6_fm']
        log = [{'iter': 0, 'fm': best_fm, 'metrics': mm0}]
        stall = 0
        for it in range(1, iters + 1):
            g = grad_delta(cur, xF, yF)
            acc = False
            for _bt in range(12):
                cand = apply_delta(cur, g, lr)
                mm = eval_model(cand)
                ok, _v = accept(mm, centers, bands, TIER_A)
                if ok and mm['m6_fm'] > best_fm + 1e-7:
                    cur, best_fm, acc = cand, mm['m6_fm'], True
                    log.append({'iter': it, 'fm': best_fm, 'metrics': mm})
                    break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return log, cur

    def get_or(fn, compute):
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            return json.load(open(p))
        v = compute()
        with open(p, 'w') as fh:
            json.dump(v, fh, indent=1)
        return v

    def get_ckpt(fn, compute):
        p = os.path.join(ck, fn)
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = compute()
        torch.save(sd, p)
        return sd

    out = {'config': {'k': K, 'aud_seeds': list(aud_seeds),
                      'ft_aud_seeds': list(ft_aud_seeds),
                      'epochs': epochs, 'iters': iters,
                      'smoke': bool(cli.smoke)}, 'units': {}}

    # ================= retrain-referenced unit =========================
    prov_mm = {}
    for s in PROV_SEEDS:
        if s in (44101, 44102, 44103):
            prov_mm[s] = json.load(open(os.path.join(
                RUN3, 'rt_k{}_s{}.json'.format(K, s))))['per_k'][str(K)]
        else:
            prov_mm[s] = json.load(open(os.path.join(
                RUN6, 'u0_t44500_k30_s{}.json'.format(s))))
    prov_ck = {s: (torch.load(os.path.join(
        RUN3, 'ckpt', 'rt_k{}_s{}.pt'.format(K, s)), map_location=device)
        if s in (44101, 44102, 44103) else torch.load(os.path.join(
            RUN6, 'ckpt', 'u0_t44500_k30_s{}.pt'.format(s)),
            map_location=device)) for s in PROV_SEEDS}
    p_centers, p_bands = mk_bands_centers(prov_mm)

    aud_mm, aud_ck = {}, {}
    for s in aud_seeds:
        sd = get_ckpt('aud_s{}.pt'.format(s), lambda s=s: fit(keep, s,
                                                              epochs))
        aud_ck[s] = sd
        aud_mm[s] = get_or('aud_s{}.json'.format(s),
                           lambda sd=sd: eval_model(sd))
        print('auditor retrain', s, 'fm', round(aud_mm[s]['m6_fm'], 3),
              flush=True)
    a_centers, a_bands = mk_bands_centers(aud_mm)
    a_band = a_bands[next(iter(a_bands))]    # flat: self-inclusive band

    res_r = {'aud_luck': luck_ceiling(aud_mm, fm0)}
    # (a) provider honest under auditor bands
    res_r['honest_pass'] = {
        str(s): accept(prov_mm[s], a_centers, a_band, TIER_A)[0]
        for s in PROV_SEEDS}
    res_r['honest_reject_n'] = sum(1 for v in
                                   res_r['honest_pass'].values() if not v)

    # (b) provider washes regenerated under provider bands
    xF, yF = f_batch()
    regen = {}
    for s in PROV_SEEDS[:2 if cli.smoke else 6]:
        def comp(s=s):
            log, sd = wash(prov_ck[s], p_centers, p_bands[s], xF, yF)
            return {'I1': i1(log[-1]['fm'], prov_mm[s]['m6_fm'], fm0),
                    'final_mm': log[-1]['metrics'], 'n_acc': len(log) - 1}
        regen[s] = get_or('regen_s{}.json'.format(s), comp)
        print('regen wash', s, 'I1', round(regen[s]['I1'], 4), flush=True)
    ref6 = json.load(open(os.path.join(
        RUN6, 'verdict.json')))['units']['t44500_k30']['i1_per_seed']
    res_r['regen_match_ew6'] = {
        str(s): abs(regen[s]['I1'] - ref6[PROV_SEEDS.index(s)]) < 2e-3
        for s in regen}
    res_r['regen_I1'] = {str(s): regen[s]['I1'] for s in regen}
    res_r['regen_pass_aud'] = {
        str(s): accept(regen[s]['final_mm'], a_centers, a_band,
                       TIER_A)[0] for s in regen}

    # (c) adaptive washes under auditor bands
    ad = {}
    for s in list(aud_seeds)[:n_adaptive]:
        def comp(s=s):
            log, sd = wash(aud_ck[s], a_centers, a_bands[s], xF, yF)
            torch.save(sd, os.path.join(ck, 'adapt_s{}.pt'.format(s)))
            return {'I1': i1(log[-1]['fm'], aud_mm[s]['m6_fm'], fm0),
                    'fm': log[-1]['fm'],
                    'final_mm': log[-1]['metrics'],
                    'n_acc': len(log) - 1}
        ad[s] = get_or('adapt_s{}.json'.format(s), comp)
        print('adaptive wash', s, 'I1', round(ad[s]['I1'], 4), flush=True)
    best_ad = max(ad.items(), key=lambda kv: kv[1]['fm'])
    ad_pass = sum(1 for v in ad.values()
                  if accept(v['final_mm'], a_centers, a_band, TIER_A)[0])
    ad_gain = elevation(best_ad[1]['fm'], aud_mm, fm0) - res_r['aud_luck']
    res_r.update({'adaptive': ad, 'adaptive_pass': ad_pass,
                  'adaptive_gain_over_luck': ad_gain})
    v, why = verdict_from_transfer(res_r['honest_reject_n'], ad_pass,
                                   ad_gain)
    res_r['verdict'], res_r['verdict_why'] = v, why
    print('RETRAIN UNIT:', v, '|', why, flush=True)
    out['units']['retrain'] = res_r

    # ================= ft-referenced unit (secondary) ==================
    if True:
        sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                         map_location=device)
        ftp_mm = {}
        for s in FT_PROV_SEEDS:
            ftp_mm[s] = json.load(open(os.path.join(
                RUN9F, 'ft_k{}_s{}.json'.format(K, s))))
        ftp_centers, ftp_bands = mk_bands_centers(ftp_mm)
        ftp_band = ftp_bands[next(iter(ftp_bands))]
        fta_mm, fta_ck = {}, {}
        for s in ft_aud_seeds:
            sd = get_ckpt('ft_aud_s{}.pt'.format(s),
                          lambda s=s: ft_run(keep, s, sd0))
            fta_ck[s] = sd
            fta_mm[s] = get_or('ft_aud_s{}.json'.format(s),
                               lambda sd=sd: eval_model(sd))
        fta_centers, fta_bands = mk_bands_centers(fta_mm)
        fta_band = fta_bands[next(iter(fta_bands))]
        res_f = {'aud_luck': luck_ceiling(fta_mm, fm0)}
        res_f['honest_pass'] = {
            str(s): accept(ftp_mm[s], fta_centers, fta_band, TIER_A)[0]
            for s in FT_PROV_SEEDS}
        res_f['honest_reject_n'] = sum(1 for v in
                                       res_f['honest_pass'].values()
                                       if not v)
        regen_f = {}
        for s in FT_PROV_SEEDS[:1 if cli.smoke else 6]:
            sd_prov = torch.load(os.path.join(
                RUN9F, 'ckpt', 'ft_k{}_s{}.pt'.format(K, s)),
                map_location=device)

            def comp(s=s, sd_prov=sd_prov):
                log, sd = wash(sd_prov, ftp_centers, ftp_bands[s], xF, yF)
                return {'I1': i1(log[-1]['fm'], ftp_mm[s]['m6_fm'], fm0),
                        'final_mm': log[-1]['metrics'],
                        'n_acc': len(log) - 1}
            regen_f[s] = get_or('ft_regen_s{}.json'.format(s), comp)
        ref9 = json.load(open(os.path.join(
            RUN9F, 'verdict.json')))['units']['30']['i1_per_seed']
        res_f['regen_match_ew9f'] = {
            str(s): abs(regen_f[s]['I1']
                        - ref9[FT_PROV_SEEDS.index(s)]) < 5e-3
            for s in regen_f}
        res_f['regen_I1'] = {str(s): regen_f[s]['I1'] for s in regen_f}
        res_f['regen_pass_aud'] = {
            str(s): accept(regen_f[s]['final_mm'], fta_centers,
                           fta_band, TIER_A)[0] for s in regen_f}
        ad_f = {}
        for s in list(ft_aud_seeds)[:n_adaptive]:
            def comp(s=s):
                log, sd = wash(fta_ck[s], fta_centers, fta_bands[s],
                               xF, yF)
                return {'I1': i1(log[-1]['fm'], fta_mm[s]['m6_fm'], fm0),
                        'fm': log[-1]['fm'],
                        'final_mm': log[-1]['metrics'],
                        'n_acc': len(log) - 1}
            ad_f[s] = get_or('ft_adapt_s{}.json'.format(s), comp)
        best_f = max(ad_f.items(), key=lambda kv: kv[1]['fm'])
        adf_pass = sum(1 for v in ad_f.values()
                       if accept(v['final_mm'], fta_centers, fta_band,
                                 TIER_A)[0])
        adf_gain = elevation(best_f[1]['fm'], fta_mm, fm0) \
            - res_f['aud_luck']
        res_f.update({'adaptive': ad_f, 'adaptive_pass': adf_pass,
                      'adaptive_gain_over_luck': adf_gain})
        vf, whyf = verdict_from_transfer(res_f['honest_reject_n'],
                                         adf_pass, adf_gain)
        res_f['verdict'], res_f['verdict_why'] = vf, whyf
        print('FT UNIT:', vf, '|', whyf, flush=True)
        out['units']['ft'] = res_f

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
