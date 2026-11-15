"""Probe: why did A3_ztail stall? Loads one start, computes the A3
objective components (active-hinge count, sigma_i scale, gradient
norms of A1/A2/A3 objectives), zero training."""
import glob
import sys

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
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
    cx_thr = (cx[0][len(cons) - N_THR:], cx[1][len(cons) - N_THR:])
    mem = (ctx['xF'], ctx['yF'])
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    sref = {nm: np.stack([
        channels(model, sd, x, y, with_grad=False)['negloss']
        for sd in fit_sds])
        for nm, (x, y) in {'mem': mem, 'thr': cx_thr}.items()}
    mu, sg = sref['mem'].mean(0), sref['mem'].std(0) + 1e-12
    mu_t, sg_t = sref['thr'].mean(0), sref['thr'].std(0) + 1e-12

    sp = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[1]
    sd = torch.load(sp, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        rm = -torch.nn.functional.cross_entropy(model(mem[0]), mem[1],
                                                reduction='none')
        rt = -torch.nn.functional.cross_entropy(model(cx_thr[0]),
                                                cx_thr[1],
                                                reduction='none')
    z = (rm.cpu().numpy() - mu) / sg
    zt = (rt.cpu().numpy() - mu_t) / sg_t
    t = np.quantile(zt, 1.0 - BETA)
    act = (t + KAPPA) > z
    print('sigma_i: min %.4f med %.4f max %.4f | mean(1/sig) %.2f'
          % (sg.min(), np.median(sg), sg.max(),
             np.mean(1.0 / sg)))
    print('t_z(N_c-thr, 60 records) = %.3f | members z: min %.2f med '
          '%.2f max %.2f | active hinge %d/30'
          % (t, z.min(), np.median(z), z.max(), act.sum()))
    print('z-gaps (t+kappa-z) active: min %.3f med %.3f max %.3f'
          % ((t + KAPPA - z)[act].min(),
             np.median((t + KAPPA - z)[act]),
             (t + KAPPA - z)[act].max()))
    # gradient norms per objective (one backprop each)
    for name in ('A1', 'A3'):
        model.load_state_dict(sd)
        model.zero_grad()
        rm2 = -torch.nn.functional.cross_entropy(model(mem[0]), mem[1],
                                                 reduction='none')
        if name == 'A1':
            obj = rm2.mean() - 0.5 * torch.nn.functional.cross_entropy(
                model(cx[0][:60]), cx[1][:60])
        else:
            zt2 = (rm2 - torch.tensor(mu, device=device)) \
                / torch.tensor(sg, device=device)
            hinge = torch.clamp(
                torch.tensor(t + KAPPA, device=device) - zt2,
                min=0).mean()
            obj = hinge - 0.5 * torch.nn.functional.cross_entropy(
                model(cx[0][:60]), cx[1][:60])
        obj.backward()
        gn = float(np.sqrt(sum(
            (p.grad.detach() ** 2).sum().item()
            for p in model.parameters() if p.grad is not None)))
        print('%s objective value %.4f grad-norm %.3f'
              % (name, float(obj), gn))


if __name__ == '__main__':
    sys.exit(main())
