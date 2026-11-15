"""#2 z-readout tier under the stronger-m5 audit (prereg 2026-11-10).
Rebuilds the strong-m5 audit (identical to run_sm5), encodes 3 e4
starts under it, then measures the Tier-2 CAL-Z working-point readout
(z-neggrad, thresholds frozen on a never-trained aux set at nominal
5%, actual TPR/FPR on a disjoint eval set) paired base vs final.
Gates: READOUT-DELIVERED (>=2/3 finals pass + z-TPR above base) /
READOUT-ABSENT."""
import argparse, glob, json, os, time
import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_ew3 import metrics_from_logits, _ce_rows, _margin_rows
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import channels


def rows5(z, y):
    ce, mg = _ce_rows(z, y), _margin_rows(z, y)
    zy = z[np.arange(len(y)), y]
    zc = z.copy()
    zc[np.arange(len(y)), y] = -1e18
    ent = zy - (zc.max(axis=1) + np.log(np.exp(
        zc - zc.max(axis=1, keepdims=True)).sum(axis=1) + 1e-30))
    return np.column_stack([ce, mg, ent, zy,
                            zc.max(axis=1)]).astype(np.float32)


def mlp_auc(fp, fn):
    import torch.nn as nn
    from uw.run_ew3 import auc_mw
    X = np.concatenate([fp, fn])
    yv = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    tX = torch.tensor((X - mu) / sd)
    ty = torch.tensor(yv, dtype=torch.float32).unsqueeze(1)
    net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad(); lf(net(tX), ty).backward(); opt.step()
    with torch.no_grad():
        s = net(tX).squeeze(1).numpy()
    return float(auc_mw(s[:len(fp)], s[len(fp):]))


def run_sm5z(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    te = np.arange(len(tset.cifar10.targets))
    cat_te = te[np.asarray(tset.cifar10.targets) == 3]
    cat_tr = ctx['pool']
    poolset = set(cat_tr.tolist())
    fset_set = set(ctx['fset'].tolist())
    ret_train = np.array([i for i in ctx['keep']
                          if i not in poolset and i not in fset_set])
    sub = np.sort(np.random.RandomState(62000).choice(
        ret_train, 3000, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    model = ctx['model']

    def rec_train(idx):
        return (torch.tensor(np.stack([trset.cifar10.data[i] / 255.0
                for i in idx])).permute(0, 3, 1, 2).float().to(device),
                torch.tensor(np.asarray(trset.cifar10.targets)[idx])
                .long().to(device))

    def rec(idx):
        return (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                for i in idx])).permute(0, 3, 1, 2).float().to(device),
                torch.tensor([tset.cifar10.targets[i] for i in idx])
                .long().to(device))

    xS, yS = rec_train(sub)
    sub_np = np.asarray(trset.cifar10.targets)[sub]
    xT = torch.cat([rec(b)[0] for b in np.array_split(te, 20)])
    yT = torch.tensor(np.asarray(tset.cifar10.targets)[te])\
        .long().to(device)
    yF_np = mem[1].cpu().numpy()

    def strong_metrics(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zF = model(mem[0]).cpu().numpy()
            zS = model(xS).cpu().numpy()
            zT = torch.cat([model(b).cpu()
                            for b in torch.split(xT, 1000)]).numpy()
        mm = metrics_from_logits(zF, yF_np, zS, sub_np, zT,
                                 yT.cpu().numpy())
        return {'m1': mm['m1_acc'], 'm2': mm['m2_catacc'],
                'm3': mm['m3_Floss'], 'm4': mm['m4_KS'],
                'm5s': mlp_auc(rows5(zF, yF_np), rows5(zS, sub_np)),
                'm6': mm['m6_fm']}

    KEYS = ('m1', 'm2', 'm3', 'm4', 'm5s', 'm6')
    fit = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    cal = sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt')) + \
        sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt'))
    mm_fit = [strong_metrics(torch.load(p, map_location=device))
              for p in fit]
    c_ = {k: float(np.median([m[k] for m in mm_fit])) for k in KEYS}
    sg = {k: max(float(np.median(np.abs(np.array(
        [m[k] for m in mm_fit]) - c_[k]))), 1e-9) for k in KEYS}

    def sc_strong(sd):
        mm = strong_metrics(sd)
        return max(abs(mm[k] - c_[k]) / sg[k] for k in KEYS)

    cal_sc = sorted(sc_strong(torch.load(p, map_location=device))
                    for p in cal)
    q_s = float(cal_sc[int(np.ceil(0.8 * len(cal_sc))) - 1])
    print(' sm5z q %.3f' % q_s, flush=True)

    fit_refs = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    mu_ref = np.mean([channels(model, torch.load(p, map_location=device),
                               *mem)['neggrad'] for p in fit_refs], 0)
    sg_ref = np.std([channels(model, torch.load(p, map_location=device),
                              *mem)['neggrad'] for p in fit_refs], 0)

    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset]), 300,
        replace=False))
    cx = rec(cons)
    rest_pool = np.setdiff1d(
        np.array([i for i in cat_te if i not in poolset]),
        np.concatenate([sub if sub[0] < 10000 else [], cons]))
    aux = np.sort(np.random.RandomState(52600).choice(rest_pool, 100,
                                                      replace=False))
    evl = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(rest_pool, aux), 200, replace=False))
    xa, ya = rec(aux)
    xe, ye = rec(evl)

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]), mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx), ny))

    def gen(sd):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53600)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            ce_obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if sc_strong(cand) <= q_s:
                    model.load_state_dict(cur)
                    model.zero_grad()
                    if float(ce_obj(cand, nx, ny)) < \
                            float(ce_obj(cur, nx, ny)) - 1e-9:
                        cur, acc = cand, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > 12:
                break
        return cur

    def zstats(x, y):
        R = np.stack([channels(model, torch.load(p, map_location=device),
                               x, y)['neggrad'] for p in fit_refs])
        return R.mean(0), R.std(0)

    def working_point(sd):
        mm, ms = zstats(*mem)
        am, asg = zstats(xa, ya)
        em, es = zstats(xe, ye)
        zm = (channels(model, sd, *mem)['neggrad'] - mm) / (ms + 1e-12)
        za = (channels(model, sd, xa, ya)['neggrad'] - am) / (asg + 1e-12)
        ze = (channels(model, sd, xe, ye)['neggrad'] - em) / (es + 1e-12)
        tau = float(np.quantile(za, 0.95))
        return float((zm >= tau).mean()), float((ze >= tau).mean())

    rows = []
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[:3]
    for p in e4:
        sd = torch.load(p, map_location=device)
        tb, fb = working_point(sd)
        fF = gen(sd)
        tf_, ff = working_point(fF)
        pa = bool(sc_strong(fF) <= q_s)
        rows.append({'start': os.path.basename(p),
                     'base_tpr': tb, 'base_fpr': fb,
                     'fin_tpr': tf_, 'fin_fpr': ff, 'pass': pa})
        print(' sm5z %s TPR %.2f->%.2f FPR %.3f->%.3f pass %s'
              % (os.path.basename(p), tb, tf_, fb, ff, pa), flush=True)
    dl = sum(1 for r in rows if r['pass'] and r['fin_tpr'] > r['base_tpr'])
    v = 'READOUT-DELIVERED' if dl >= 2 else 'READOUT-ABSENT'
    out = {'q_strong': q_s, 'rows': rows, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'sm5z-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_sm5z(cli)


if __name__ == '__main__':
    main()
