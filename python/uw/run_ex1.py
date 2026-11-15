"""UW E-X1: operator-family sweep with honest-luck control (frozen prereg
docs/plans/2026-09-28-uw-ex-multi-config-preregistration.md, author-approved
order E-X1 -> E-X3 -> E-X2).

Arms (k=30, theta0 A=44500, F = f_deep(pool,30), all 6 seeds each):
  a: weak retrain  -- 5 epochs from scratch, lr 1e-3 (card-frozen)
  b: SSD-adapt     -- dampen bottom-p% retain-importance weights by d=0.5;
                      seed varies the importance-estimation batch order
  c: scrub-adapt   -- 3 epochs diagonal-Fisher-preconditioned SGD on retain
                      from theta0; seed varies batch order
Controls (archived, no rerun): retrain 40ep (gamma 0.212), ft 1ep (0.389),
NegGrad (0.859).

Estimand per family (frozen): gain = best wash elevation over the family
median (in deletion-delta units) MINUS the family's honest luck ceiling.
MATERIAL_GAIN at gain >= 0.10; revival requires >= 2 independent units
(card section 1). ALL seeds reported.
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
K = 30
ARM_SEEDS = {'a': (46501, 46502, 46503, 46504, 46505, 46506),
             'b': (46601, 46602, 46603, 46604, 46605, 46606),
             'c': (46701, 46702, 46703, 46704, 46705, 46706)}
SSD_PCT = 10          # dampen bottom-p% of retain-importance weights
SSD_DAMP = 0.5
FISHER_BATCHES = 10
SCRUB_EPOCHS = 3
MATERIAL = 0.10
MAX_ITERS = 60
STALL_LIMIT = 15


def gain_of(elev, luck):
    return elev - luck


def verdict_family(gain, elev):
    if gain >= MATERIAL:
        return 'MATERIAL_GAIN', (
            'wash elevation {:.3f} exceeds luck by {:.3f} (revival needs '
            'a second independent unit)'.format(elev, gain))
    return 'NULL', ('elevation {:.3f}, gain {:.3f} < {:.2f}: the family '
                    'caps fm at its own honest luck'.format(elev, gain,
                                                            MATERIAL))


def bottom_mask(imp, pct):
    """Indices of the lowest-pct entries (deterministic tie-break by
    position via argsort kind='stable')."""
    n = int(round(len(imp) * pct / 100.0))
    order = np.argsort(imp, kind='stable')
    return order[:n]


def main():
    import torch
    from forest.ingredients import Ingredient
    from run_s0i_p0 import build_args
    from tools.cifar_10 import CIFAR10
    from tools.models import get_model
    import torchvision.transforms as _tf

    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('a', 'b', 'c'), required=True)
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    iters = 8 if cli.smoke else MAX_ITERS
    seeds = ARM_SEEDS[cli.arm][:(1 if cli.smoke else 6)]
    epochs_a = 2 if cli.smoke else 5
    fisher_batches = 2 if cli.smoke else FISHER_BATCHES
    scrub_epochs = 1 if cli.smoke else SCRUB_EPOCHS

    if cli.outdir:
        outdir = cli.outdir
        os.makedirs(outdir, exist_ok=True)
    else:
        tag = 'ex1-smoke-' if cli.smoke else 'ex1-'
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
    fset = f_deep(pool, K)
    keep = np.setdiff1d(np.arange(n), fset)
    pos = np.searchsorted(pool, fset)
    pt = json.load(open(os.path.join(RUN3, 'pt.json')))['per_k'][str(K)]
    fm0 = pt['m6_fm']

    model = get_model('ResNet18').to(device)

    def seed_all(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def loader_keep(seed, shuffle=True):
        g = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, keep), batch_size=100,
            shuffle=shuffle, generator=g)

    y_t = torch.as_tensor(yfull, dtype=torch.long)

    def fit_weak(seed):
        """Arm a: from-scratch 5ep retrain at lr 1e-3 (card-frozen)."""
        seed_all(seed)
        m = get_model('ResNet18').to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_a)
        lf = torch.nn.CrossEntropyLoss()
        for _ep in range(epochs_a):
            m.train()
            for _ids, x, _yy in loader_keep(seed):
                opt.zero_grad()
                lf(m(x.to(device)), y_t[_ids.tolist()].to(
                    device)).backward()
                opt.step()
            sched.step()
        m.eval()
        return {k_: v.detach().clone() for k_, v in
                m.state_dict().items()}

    def accum_sqgrad(m, loader, nbatches, sd0):
        """Accumulate per-parameter squared grads of retain CE at sd0."""
        m.load_state_dict(sd0)
        m.train()
        acc = {nm: torch.zeros_like(p) for nm, p in m.named_parameters()}
        lf = torch.nn.CrossEntropyLoss()
        seen = 0
        for _ids, x, _yy in loader:
            m.zero_grad()
            lf(m(x.to(device)), y_t[_ids.tolist()].to(device)).backward()
            for nm, p in m.named_parameters():
                acc[nm] += p.grad.detach() ** 2
            seen += 1
            if seen >= nbatches:
                break
        return acc

    def ssd_adapt(seed):
        """Arm b: dampen bottom-pct retain-importance weights by d."""
        sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                         map_location=device)
        imp = accum_sqgrad(model, loader_keep(seed), fisher_batches, sd0)
        sd = {k_: v.clone() for k_, v in sd0.items()}
        with torch.no_grad():
            for nm, p in model.named_parameters():
                flat = imp[nm].flatten().cpu().numpy()
                idx = bottom_mask(flat, SSD_PCT)
                f = p.flatten()
                msk = torch.zeros(f.numel(), dtype=torch.bool,
                                  device=p.device)
                msk[torch.as_tensor(idx, device=p.device)] = True
                f[msk] *= (1.0 - SSD_DAMP)
                sd[nm] = f.view_as(p).detach().clone()
        model.load_state_dict(sd)
        model.eval()
        return {k_: v.detach().clone() for k_, v in
                model.state_dict().items()}

    def scrub_adapt(seed):
        """Arm c: diag-Fisher-preconditioned descent on retain CE from
        theta0 (scrubbing adapted): p <- p - a * g / (Fhat + eps), Fhat
        normalized to mean 1 per tensor; global grad-norm clip 5.0."""
        sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                         map_location=device)
        fisher = accum_sqgrad(model, loader_keep(seed), fisher_batches, sd0)
        precond = {nm: fisher[nm] / (fisher[nm].mean() + 1e-12)
                   for nm in fisher}
        model.load_state_dict(sd0)
        model.train()
        lf = torch.nn.CrossEntropyLoss()
        alpha, eps = 0.01, 0.1
        # v2 (2026-09-29): clip 5.0 made the operator ~identity (grad
        # norm over 11M params >> 5, per-weight step ~0, E degenerated
        # to -0.063); raised to 50.0 as the bounded labeled repair.
        for _ep in range(scrub_epochs):
            for _ids, x, _yy in loader_keep(seed + 1):
                model.zero_grad()
                lf(model(x.to(device)),
                   y_t[_ids.tolist()].to(device)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                with torch.no_grad():
                    for nm, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p -= alpha * p.grad / (precond[nm] + eps)
        model.eval()
        return {k_: v.detach().clone() for k_, v in
                model.state_dict().items()}

    def family_model(arm, seed):
        if arm == 'a':
            return fit_weak(seed)
        if arm == 'b':
            return ssd_adapt(seed)
        if arm == 'c':
            return scrub_adapt(seed)
        raise ValueError(arm)

    idx_ps = np.concatenate([pool, sub])
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset_eval, idx_ps), batch_size=500,
        shuffle=False)

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

    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_eval, fset), batch_size=500,
            shuffle=False):
        xF.append(x)
        yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()

    def grad_delta(sd):
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

    def fm_only(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            z = model(xF)
        zy = z.gather(1, yF[:, None]).squeeze(1)
        zc = z.clone()
        zc.scatter_(1, yF[:, None], -1e18)
        return float((zy - zc.max(1).values).mean())

    # ---- run family -----------------------------------------------------
    fam_mm, fam_ck = {}, {}
    for s in seeds:
        pth = os.path.join(ck, 'arm%s_s%d.pt' % (cli.arm, s))
        fnj = 'arm%s_s%d.json' % (cli.arm, s)
        if os.path.exists(os.path.join(outdir, fnj)):
            fam_mm[s] = json.load(open(os.path.join(outdir, fnj)))
            fam_ck[s] = torch.load(pth, map_location=device)
            continue
        sd = family_model(cli.arm, s)
        torch.save(sd, pth)
        fam_ck[s] = sd
        fam_mm[s] = eval_model(sd)
        with open(os.path.join(outdir, fnj), 'w') as fh:
            json.dump(fam_mm[s], fh, indent=1)
        print('arm', cli.arm, 'seed', s, 'm1',
              round(fam_mm[s]['m1_acc'], 3), 'fm',
              round(fam_mm[s]['m6_fm'], 3), flush=True)

    med = {m: float(np.median([e[m] for e in fam_mm.values()]))
           for m in TIER_A}
    bands = loo_bands(fam_mm, med)
    band = bands[next(iter(bands))]
    fm_med = float(np.median([e['m6_fm'] for e in fam_mm.values()]))
    E = fm0 - fm_med
    b_fm = float(max(abs(e['m6_fm'] - fm_med) for e in fam_mm.values()))
    luck = max(e['m6_fm'] for e in fam_mm.values()) - fm_med
    luck = luck / E if E != 0 else float('inf')

    rows = []
    for s in seeds:
        fnj = 'arm%s_wash_s%d.json' % (cli.arm, s)
        if os.path.exists(os.path.join(outdir, fnj)):
            rec = json.load(open(os.path.join(outdir, fnj)))
        else:
            cur = fam_ck[s]
            lr = LR0
            best_fm = fam_mm[s]['m6_fm']
            log = [{'fm': best_fm}]
            stall = 0
            for it in range(1, iters + 1):
                g = grad_delta(cur)
                acc = False
                for _bt in range(12):
                    cand = apply_delta(cur, g, lr)
                    mm = eval_model(cand)
                    ok, _v = accept(mm, med, band, TIER_A)
                    if ok and mm['m6_fm'] > best_fm + 1e-7:
                        cur, best_fm, acc = cand, mm['m6_fm'], True
                        break
                    lr *= 0.5
                    if lr < 1e-9:
                        break
                lr = lr_update(lr, acc)
                stall = 0 if acc else stall + 1
                if stall > STALL_LIMIT:
                    break
            final = eval_model(cur)
            torch.save(cur, os.path.join(
                ck, 'arm%s_wash_s%d.pt' % (cli.arm, s)))
            rec = {'I1': i1(final['m6_fm'], fam_mm[s]['m6_fm'], fm0),
                   'fm': final['m6_fm'], 'final_mm': final,
                   'n_acc': len(log) - 1}
            with open(os.path.join(outdir, fnj), 'w') as fh:
                json.dump(rec, fh, indent=1)
        elev = (rec['fm'] - fm_med) / E
        rows.append({'seed': s, 'I1': rec['I1'], 'elev': elev,
                     'pass_band': accept(rec['final_mm'], med, band,
                                         TIER_A)[0],
                     'fm_exceeds_family_max':
                         rec['fm'] > max(e['m6_fm'] for e in
                                         fam_mm.values())})
        print('wash', s, 'I1', round(rec['I1'], 4), 'elev',
              round(elev, 3), flush=True)

    best_elev = max(r['elev'] for r in rows)
    gain = gain_of(best_elev, luck)
    v, why = verdict_family(gain, best_elev)
    if E < 0.5:
        v = 'DEGENERATE_UNIT'
        why = ('family deletion delta E={:.3f} is near zero or negative '
               '(honest fm med {:.3f} vs theta0 {:.3f}): all normalized '
               'quantities are vacuous; family reported raw'
               .format(E, fm_med, fm0))
    caught = sum(1 for r in rows if r['fm_exceeds_family_max'])
    out = {'arm': cli.arm, 'seeds': list(seeds), 'gamma': b_fm / max(abs(E),
                                                                    1e-12),
           'luck': luck, 'best_elev': best_elev, 'gain': gain,
           'E': E, 'fm0': fm0, 'fm_med': fm_med,
           'fm_band_caught': caught, 'verdict': v, 'verdict_why': why,
           'rows': rows, 'honest_mm': {str(s): fam_mm[s] for s in seeds}}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('ARM', cli.arm, ':', v, '|', why, '| gamma',
          round(out['gamma'], 3), 'luck', round(luck, 3), flush=True)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
