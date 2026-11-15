"""Trace A3_ztail optimization on e3-s1 for a few iterations: per
iteration log lr, audit score of candidate, J improvement, accept."""
import glob

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import auc  # noqa: F401
from uw.run_r1 import load_audit
from uw.run_ey import K
from uw.run_ew3b import LR0, lr_update
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec
from uw.run_tail import KAPPA, BETA, N_THR


def main():
    device = torch.device('cuda:1')
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    cx = rec(tset, cons, device)
    cx_grad = (cx[0][:240], cx[1][:240])
    cx_thr = (cx[0][240:], cx[1][240:])
    mem = (ctx['xF'], ctx['yF'])
    _, _, _, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    sref = {nm: np.stack([
        channels(model, sd, x, y, with_grad=False)['negloss']
        for sd in fit_sds])
        for nm, (x, y) in {'mem': mem, 'thr': cx_thr}.items()}
    mu, sg = sref['mem'].mean(0), sref['mem'].std(0) + 1e-12
    mu_t, sg_t = sref['thr'].mean(0), sref['thr'].std(0) + 1e-12

    def obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        rm = -torch.nn.functional.cross_entropy(model(mem[0]), mem[1],
                                                reduction='none')
        contrast = torch.nn.functional.cross_entropy(model(nx), ny)
        with torch.no_grad():
            rt = -torch.nn.functional.cross_entropy(model(cx_thr[0]),
                                                    cx_thr[1],
                                                    reduction='none')
            t = torch.quantile((rt - torch.tensor(mu_t, device=device))
                               / torch.tensor(sg_t, device=device),
                               1.0 - BETA)
        z = (rm - torch.tensor(mu, device=device)) \
            / torch.tensor(sg, device=device)
        hinge = torch.clamp(t + KAPPA - z, min=0).mean()
        return hinge - 0.5 * contrast

    sp = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[1]
    cur = torch.load(sp, map_location=device)
    lr = LR0
    rs = np.random.RandomState(53600)
    print('LR0 = %g  base s6 = %.4f  q20 = %.4f'
          % (LR0, score_sd(cur), q20), flush=True)
    for it in range(6):
        idx = rs.choice(len(cx_grad[1]), 60, replace=False)
        nx, ny = cx_grad[0][idx], cx_grad[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        obj(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        v0 = float(obj(cur, nx, ny))
        for bt in range(8):
            cand = apply_delta(cur, g, -lr)
            s6c = score_sd(cand)
            v = float(obj(cand, nx, ny))
            ok_audit = s6c <= q20
            ok_impr = v < v0 - 1e-9
            if bt < 3 or (ok_audit and ok_impr):
                print(' it%d bt%d lr %.2e s6 %.4f J %.6f->%.6f '
                      'audit=%s impr=%s'
                      % (it, bt, lr, s6c, v0, v, ok_audit, ok_impr),
                      flush=True)
            if ok_audit and ok_impr:
                cur, acc = cand, True
                break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        print(' it%d accepted=%s lr->%.2e' % (it, acc, lr), flush=True)
        if not acc:
            break


if __name__ == '__main__':
    main()
