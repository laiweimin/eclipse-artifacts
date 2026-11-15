"""Multi-start probe-select-encode protocol validation.
For each of 6 fresh CIFAR starts: run GR encoding for 10 iters
(probe), measure AUC, compare with known 60-iter results.
Tests whether short probes predict full encoding success.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as Fn

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_lfpr import channels, rec, score_sets, zcal

RHO = 0.5
N_INST = 10
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))

# Known 60-iter outcomes from P-FWD (fresh starts, GR)
KNOWN_60 = {'f-e3-0': False, 'f-e3-2': False, 'f-e3-5': True,
            'f-e4-3': True, 'f-e4-4': False, 'f-e4-5': True}


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def gen_gr_probe(model, ctx, mem, cx, ratio_fn, it):
    """GR encoding with iteration count from `it` dict."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        xg = mem[0].clone().requires_grad_(True)
        xg2 = nx.clone().requires_grad_(True)
        lf = Fn.cross_entropy(model(xg), mem[1], reduction='none')
        ln = Fn.cross_entropy(model(xg2), ny, reduction='none')
        gf = torch.autograd.grad(lf.sum(), xg, create_graph=True
                                 )[0].flatten(1).norm(1)
        gn = torch.autograd.grad(ln.sum(), xg2, create_graph=True
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
            cand = {nm: (p.detach() + lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if ratio_fn(cand) <= 1.0:
                v = float(loss(cand, nx, ny))
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
    X_fit = np.stack([vec(e) for e in fit])
    c_j = X_fit.mean(0)
    S = np.cov(X_fit, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))

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

    starts = [('f-e3-0', E3[0]), ('f-e3-2', E3[2]),
              ('f-e3-5', E3[5]), ('f-e4-3', E4[3]),
              ('f-e4-4', E4[4]), ('f-e4-5', E4[5])]

    def measure_auc(sd_model, sd):
        b_sc = score_sets(model, sd, sets)
        f_sc = score_sets(model, sd_model, sets)
        ch = 'neggrad'
        mb, mf = b_sc['mem'][ch], f_sc['mem'][ch]
        eb, ef = b_sc['eval'][ch], f_sc['eval'][ch]
        return float(rank_auc(mf, ef))

    print('%-10s %7s %7s %7s %5s %5s' %
          ('start', 'auc_0', 'auc_10', 'gain', 'known', 'pred'),
          flush=True)
    rows = []
    for tag, sp in starts:
        sd = torch.load(sp, map_location=device)
        auc_0 = measure_auc(sd, sd)  # baseline = ~0.5

        # Probe: 10 iterations
        fin10, nacc10 = gen_gr_probe(model, ctx, mem, cx,
                                     ratio_fn, {'sd': sd, 'n': 10})
        auc_10 = measure_auc(fin10, sd)
        gain = auc_10 - 0.5  # gain over random

        known = KNOWN_60[tag]
        # Simple prediction: gain > 0.05 at probe → predict success
        pred = gain > 0.05
        rows.append({'tag': tag, 'auc_10': auc_10,
                     'gain_10': gain, 'known_60': known,
                     'predicted': pred})
        print('%-10s %7.3f %7.3f %+7.3f %5s %5s' %
              (tag, auc_0, auc_10, gain,
               'Y' if known else 'N',
               'Y' if pred else 'N'), flush=True)

    # Protocol evaluation
    tp = sum(1 for r in rows if r['predicted'] and r['known_60'])
    fp = sum(1 for r in rows if r['predicted'] and not r['known_60'])
    fn = sum(1 for r in rows if not r['predicted'] and r['known_60'])
    tn = sum(1 for r in rows if not r['predicted']
             and not r['known_60'])
    print('\nProtocol: probe@10iters, threshold gain>0.05')
    print('TP=%d FP=%d FN=%d TN=%d' % (tp, fp, fn, tn), flush=True)
    print('Accuracy: %d/6' % (tp + tn), flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rows': rows, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
           'verdict': 'PROTOCOL-VALID' if tp + tn >= 5
           else 'PROTOCOL-WEAK'}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT %s' % out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/probe-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
