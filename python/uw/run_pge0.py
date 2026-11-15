"""C-PGE P0: zero-training — check if encoded finals carry MORE
signal under augmented evaluation than canonical evaluation.

The audit evaluates in canonical (no augmentation) form. If encoding
produces stronger membership signal under augmented inputs, the
protocol gap exists and can be exploited.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torchvision.transforms as _tf
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import channels, rec

CE_GLOB = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR_ORIG = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
           'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt']
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))


def aug_x(x, seed=0):
    """Apply horizontal flip to each input tensor."""
    torch.manual_seed(seed)
    return torch.flip(x, dims=[3])  # flip width dimension


def aug_crop(x, pad=4, seed=0):
    """Apply random crop shift."""
    torch.manual_seed(seed)
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    # Fixed shift: shift by 2 pixels (deterministic for comparison)
    return xp[:, :, pad+2:pad+2+H, pad+2:pad+2+W]


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    mem = (ctx['xF'], ctx['yF'])

    # Fresh non-members (same 200 as always)
    cons = np.sort(np.random.RandomState(52801).choice(
        cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh_ids = np.sort(np.random.RandomState(52501).choice(
        rest, 200, replace=False))
    fx_x, fx_y = rec(tset, fresh_ids, device)

    # Honest refs for z-cal
    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]

    def neggrad_z(sd, x_mem, x_fresh):
        """Compute neggrad z-scores for member and fresh records."""
        model.load_state_dict(sd)
        model.eval()
        # Members (canonical)
        xg_m = x_mem.clone().requires_grad_(True)
        out_m = model(xg_m)
        lf_m = F.cross_entropy(out_m, mem[1], reduction='none')
        g_m = torch.stack([
            torch.autograd.grad(lf_m[i], xg_m, retain_graph=True
                                )[0][i].norm()
            for i in range(len(mem[1]))]).cpu().numpy()

        # Fresh (canonical)
        xg_f = x_fresh.clone().requires_grad_(True)
        out_f = model(xg_f)
        lf_f = F.cross_entropy(out_f, fx_y, reduction='none')
        g_f = torch.stack([
            torch.autograd.grad(lf_f[i], xg_f, retain_graph=True
                                )[0][i].norm()
            for i in range(len(fx_y))]).cpu().numpy()
        return -g_m, -g_f  # neggrad (lower grad = more member-like)

    def neggrad_z_aug(sd, x_mem, x_fresh, aug_fn):
        """Same but with augmented inputs."""
        return neggrad_z(sd, aug_fn(x_mem), aug_fn(x_fresh))

    # z-cal from honest refs (canonical)
    ref_m = np.stack([neggrad_z(sd, mem[0], fx_x)[0]
                      for sd in fit_sds])
    mu_m, sg_m = ref_m.mean(0), ref_m.std(0) + 1e-12
    ref_f = np.stack([neggrad_z(sd, mem[0], fx_x)[1]
                      for sd in fit_sds])
    mu_f, sg_f = ref_f.mean(0), ref_f.std(0) + 1e-12

    # z-cal for augmented (flip)
    ref_m_flip = np.stack([neggrad_z_aug(sd, mem[0], fx_x,
                                         aug_x)[0]
                           for sd in fit_sds])
    mu_m_flip = ref_m_flip.mean(0)
    sg_m_flip = ref_m_flip.std(0) + 1e-12
    ref_f_flip = np.stack([neggrad_z_aug(sd, mem[0], fx_x,
                                         aug_x)[1]
                           for sd in fit_sds])
    mu_f_flip = ref_f_flip.mean(0)
    sg_f_flip = ref_f_flip.std(0) + 1e-12

    # z-cal for crop shift
    ref_m_crop = np.stack([neggrad_z_aug(sd, mem[0], fx_x,
                                         aug_crop)[0]
                           for sd in fit_sds])
    mu_m_crop = ref_m_crop.mean(0)
    sg_m_crop = ref_m_crop.std(0) + 1e-12
    ref_f_crop = np.stack([neggrad_z_aug(sd, mem[0], fx_x,
                                          aug_crop)[1]
                           for sd in fit_sds])
    mu_f_crop = ref_f_crop.mean(0)
    sg_f_crop = ref_f_crop.std(0) + 1e-12

    # Evaluate all finals under 3 conditions
    finals = ([('ce-%d' % i, p) for i, p in
               enumerate(sorted(glob.glob(CE_GLOB))[:4])] +
              [('gr-%d' % i, p) for i, p in enumerate(GR_ORIG[:4])])

    print('%-8s %8s %8s %8s %8s' %
          ('final', 'canon', 'flip', 'crop', 'gain'), flush=True)
    rows = []
    for tag, path in finals:
        sd = torch.load(path, map_location=device)

        # Canonical
        gm, gf = neggrad_z(sd, mem[0], fx_x)
        auc_c = float(rank_auc(
            (gm - mu_m) / sg_m, (gf - mu_f) / sg_f))

        # Flip
        gm2, gf2 = neggrad_z_aug(sd, mem[0], fx_x, aug_x)
        auc_f = float(rank_auc(
            (gm2 - mu_m_flip) / sg_m_flip,
            (gf2 - mu_f_flip) / sg_f_flip))

        # Crop shift
        gm3, gf3 = neggrad_z_aug(sd, mem[0], fx_x, aug_crop)
        auc_p = float(rank_auc(
            (gm3 - mu_m_crop) / sg_m_crop,
            (gf3 - mu_f_crop) / sg_f_crop))

        gain = max(auc_f, auc_p) - auc_c
        rows.append({'tag': tag, 'canonical': auc_c,
                     'flip': auc_f, 'crop': auc_p, 'gain': gain})
        print('%-8s %8.3f %8.3f %8.3f %+8.3f' %
              (tag, auc_c, auc_f, auc_p, gain), flush=True)

    gains = [r['gain'] for r in rows]
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rows': rows,
           'mean_gain': float(np.mean(gains)),
           'max_gain': float(np.max(gains)),
           'min_gain': float(np.min(gains)),
           'gate': 'GAP-EXISTS' if max(gains) >= 0.05
           else 'NO-GAP'}
    print(json.dumps({k: round(v, 3) if isinstance(v, float)
                      else v for k, v in out.items()
                      if k != 'generated' and k != 'rows'},
                     indent=1), flush=True)
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/pge0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
