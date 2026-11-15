"""UW G-series (attack paper gaps G1/G4/G5, frozen in skeleton section 6).

G1: O2 attack under conformal + symmetric fm band (|fm-c| <= b_fit).
G5: O2 attack under conformal + one-sided fm cap (fm <= max_fit).
G4: alpha fine sweep {0.15,0.20,0.25,0.30,0.35} on O2 (fills curve).
All on the ft family frozen split (fit=ew9f, cal=p1aud+e2, starts=e3).
"""
import argparse, glob, json, math, os, time
import numpy as np
from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED,
                        build_pool, f_deep, metrics_from_logits)
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import (K, METRICS, MAX_ITERS, STALL_LIMIT, conformal_q,
                       per_record_margin, record_subsets)
from uw.run_ey2 import mk_audit
RUN3 = 'research_outputs/uw/ew3-20260925T113900'
G4_ALPHAS = (0.15, 0.20, 0.25, 0.30, 0.35)


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
    cli = ap.parse_args()
    iters = 4 if cli.smoke else MAX_ITERS
    outdir = os.path.join('research_outputs', 'uw',
                          ('eg-smoke-' if cli.smoke else 'eg-')
                          + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda:{}'.format(cli.gpu))
    args = build_args(seed=int(INST), dataset='CIFAR10', epochs=40,
                      batchsize=100, eta=0.01)
    ing = Ingredient(args, setup=dict(device=device))
    ing.initialize_attack_setup()
    trainset, testloader = ing.trainset, ing.testloader
    te = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trainset.targets)
    n = len(trainset)
    pool = build_pool(np.where(yfull == CAT_CLS)[0])
    sub = np.sort(np.random.RandomState(SUB_SEED).choice(
        np.setdiff1d(np.arange(n), pool), size=SUB_N, replace=False))
    y_sub = yfull[sub]
    fset = f_deep(pool, K)
    pos = np.searchsorted(pool, fset)
    loader_ps = torch.utils.data.DataLoader(
        torch.utils.data.Subset(te, np.concatenate([pool, sub])),
        batch_size=500, shuffle=False)
    model = get_model('ResNet18').to(device)
    xF, yF = [], []
    for _g, x, y in torch.utils.data.DataLoader(
            torch.utils.data.Subset(te, fset), batch_size=500,
            shuffle=False):
        xF.append(x); yF.append(y)
    xF = torch.cat(xF).to(device)
    yF = torch.cat(yF).to(device).long()
    yF_np = yF.cpu().numpy()

    def eval_full(sd):
        model.load_state_dict(sd)
        zs, ys = [], []
        model.eval()
        with torch.no_grad():
            for _id, x, y in testloader:
                zs.append(model(x.to(device)).cpu()); ys.append(y)
        z_te = torch.cat(zs).numpy(); y_te = torch.cat(ys).numpy()
        zs = []
        with torch.no_grad():
            for _g, x, _y in loader_ps:
                zs.append(model(x.to(device)).cpu())
        z_ps = torch.cat(zs).numpy()
        return metrics_from_logits(z_ps[:len(pool)][pos], yfull[fset],
                                   z_ps[len(pool):], y_sub, z_te, y_te)

    def margins_of(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            return per_record_margin(model(xF).cpu().numpy(), yF_np)

    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens_mm = json.load(open(cache))['ens_mm']
    fit_mm = ens_mm['ew9f']
    cal_mm = [e for t in ('p1aud', 'e2') for e in ens_mm[t]]
    c, sig, q20, sc = mk_audit(fit_mm, cal_mm)
    cal_s = [sc(e) for e in cal_mm]
    fit_fm = [e['m6_fm'] for e in fit_mm]
    fm_med = float(np.median(fit_fm))
    fm_b = float(max(abs(v - fm_med) for v in fit_fm))
    fm_max = float(max(fit_fm))
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = margins_of(sd0)
    bot, top = record_subsets(th_m, k=10)
    bot_t = torch.as_tensor(bot, device=device)
    th_O2 = float(th_m[bot].mean())
    e3_paths = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e3_mg = [margins_of(torch.load(p, map_location=device))
             for p in e3_paths]
    sel_O2 = max(float(mg[bot].mean()) for mg in e3_mg)
    med_O2 = float(np.median([float(mg[bot].mean()) for mg in e3_mg]))
    E_O2 = th_O2 - med_O2

    def ascend(start_sd, feas):
        cur = start_sd
        lr = LR0
        best = float(margins_of(cur)[bot].mean())
        stall = 0
        for it in range(1, iters + 1):
            model.load_state_dict(cur)
            model.eval()
            z = model(xF)
            zy = z.gather(1, yF[:, None]).squeeze(1)
            zc = z.clone()
            zc.scatter_(1, yF[:, None], -1e18)
            m = zy - zc.max(1).values
            model.zero_grad()
            m[bot_t].mean().backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(12):
                cand = {k2: (v + lr * g[k2] if k2 in g else v)
                        for k2, v in cur.items()}
                mm = eval_full(cand)
                if feas(mm):
                    v = float(margins_of(cand)[bot].mean())
                    if v > best + 1e-7:
                        cur, best, acc = cand, v, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > STALL_LIMIT:
                break
        return best, eval_full(cur)

    def gain(best):
        return (best - med_O2) / E_O2 - (sel_O2 - med_O2) / E_O2

    n_starts = 2 if cli.smoke else 3
    results = {}

    def run_arm(name, feas, extra_q=None):
        gs, ps = [], []
        for j, p in enumerate(e3_paths[:n_starts]):
            fn = '{}_s{}.json'.format(name, j)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, mm = ascend(torch.load(p, map_location=device),
                                  feas)
                rec = {'best': best, 'fm': mm['m6_fm'],
                       's': sc(mm)}
                if extra_q:
                    rec['pass'] = sc(mm) <= extra_q
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1, default=float)
            gs.append(gain(rec['best']))
            if 'pass' in rec:
                ps.append(rec['pass'])
        results[name] = {'gains': gs, 'max_gain': max(gs)}
        if ps:
            results[name]['pass_frac'] = float(np.mean(ps))
        print(name, 'max_gain %.3f' % max(gs), flush=True)

    # G1: conformal + symmetric fm band
    run_arm('G1_sym_fm',
            lambda mm: sc(mm) <= q20
            and abs(mm['m6_fm'] - fm_med) <= fm_b, q20)
    # G5: conformal + one-sided fm cap
    run_arm('G5_oneside_fm',
            lambda mm: sc(mm) <= q20 and mm['m6_fm'] <= fm_max, q20)
    # G4: fine alpha sweep
    for alpha in G4_ALPHAS[:2 if cli.smoke else 5]:
        q = conformal_q(cal_s, alpha)
        hon = [e for t in ('p1aud', 'e2', 'e3', 'e4')
               for e in ens_mm[t]]
        run_arm('G4_a{}'.format(alpha), lambda mm, q=q: sc(mm) <= q, q)
        results['G4_a{}'.format(alpha)]['honest_pass'] = float(
            np.mean([sc(e) <= q for e in hon]))

    json.dump({'fit_stats': {'fm_med': fm_med, 'fm_b': fm_b,
                             'fm_max': fm_max, 'q20': float(q20)},
               'sel_O2': sel_O2, 'med_O2': med_O2, 'th_O2': th_O2,
               'results': results},
              open(os.path.join(outdir, 'verdict.json'), 'w'),
              indent=1, default=float)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
