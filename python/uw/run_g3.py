"""UW G3: scale sweep of the tradeoff curve (k=1000 ft family).
Uses ew9F's archived ft_k1000 checkpoints (6 models) as fit+cal,
with the k1000 F-set. Tests whether the coverage-security tradeoff
shape transfers across deletion scale. Zero new training.
"""
import argparse, glob, json, math, os, time
import numpy as np
from uw.run_ew3 import (CAT_CLS, INST, POOL_SEED, SUB_N, SUB_SEED,
                        build_pool, f_deep, metrics_from_logits)
from uw.run_ey import (METRICS, MAX_ITERS, STALL_LIMIT, conformal_q,
                       per_record_margin, record_subsets)
RUN3 = 'research_outputs/uw/ew3-20260925T113900'
K = 1000
ALPHAS = (0.10, 0.20, 0.30, 0.40)


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
                          ('g3-smoke-' if cli.smoke else 'g3-')
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

    # fit+cal from the 6 archived ft_k1000 models (3/3 split)
    ft_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k1000_s*.pt'))
    mms = [eval_full(torch.load(p, map_location=device))
           for p in ft_paths]
    mgs = [margins_of(torch.load(p, map_location=device))
           for p in ft_paths]
    fit_mms, cal_mms = mms[:3], mms[3:]
    fit_mgs, cal_mgs = mgs[:3], mgs[3:]

    c = {m: float(np.median([e[m] for e in fit_mms])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mms]) - c[m]))) for m in METRICS}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in METRICS)

    cal_s = [sc(e) for e in cal_mms]
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = margins_of(sd0)
    bot, top = record_subsets(th_m, k=10)
    th_O2 = float(th_m[bot].mean())
    sel_O2 = max(float(mg[bot].mean()) for mg in mgs)
    med_O2 = float(np.median([float(mg[bot].mean()) for mg in mgs]))
    E = th_O2 - med_O2
    bot_t = torch.as_tensor(bot, device=device)

    def ascend(start_sd, q_lim):
        from uw.run_ew3b import LR0, lr_update
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
                if sc(mm) <= q_lim:
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
        return best, sc(eval_full(cur))

    results = []
    n_starts = 1 if cli.smoke else 3
    for alpha in ALPHAS[:2 if cli.smoke else 4]:
        q = conformal_q(cal_s, alpha)
        hon_pass = float(np.mean([sc(e) <= q for e in mms]))
        gains, passes = [], []
        for j in range(n_starts):
            fn = 'g3_a{}_s{}.json'.format(alpha, j)
            if os.path.exists(os.path.join(outdir, fn)):
                rec = json.load(open(os.path.join(outdir, fn)))
            else:
                best, s_fin = ascend(torch.load(
                    ft_paths[j], map_location=device), q)
                rec = {'best': best, 'pass': s_fin <= q}
                with open(os.path.join(outdir, fn), 'w') as fh:
                    json.dump(rec, fh, indent=1, default=float)
            g = (rec['best'] - med_O2) / E - (sel_O2 - med_O2) / E
            gains.append(g)
            passes.append(rec['pass'])
        results.append({'alpha': alpha, 'q': float(q),
                        'honest_pass': hon_pass,
                        'max_gain': max(gains),
                        'pass_frac': float(np.mean(passes))})
        print('g3 a=%.2f q=%.2f honPass %.2f gain %.3f pass %.2f'
              % (alpha, q, hon_pass, max(gains), np.mean(passes)),
              flush=True)

    json.dump({'results': results, 'sel_O2': sel_O2,
               'med_O2': med_O2, 'th_O2': th_O2, 'k': K},
              open(os.path.join(outdir, 'verdict.json'), 'w'),
              indent=1, default=float)
    print('OUT', outdir, flush=True)


if __name__ == '__main__':
    main()
