"""C-XCE P0: zero-training — check if existing encoded finals
already separate members via dog-class logits.

For each encoded final: compute dog logit z-score for F records
(vs 6 honest reference models' dog logits on same records),
then AUC (member z-scores vs fresh non-member dog z-scores).
"""
import argparse, glob, json, os, time
import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit

CE_GLOB = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR_ORIG = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
           'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
           'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
DOG_CLS = 5  # CIFAR-10 dog class index
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))


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
    rest_cat_te = np.setdiff1d(cat_te, [])
    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(cat_te,
                     np.sort(np.random.RandomState(52801).choice(
                         cat_te, 300, replace=False))),
        200, replace=False))
    mem = (ctx['xF'], ctx['yF'])

    # z-cal: dog logits from 6 honest refs
    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit, cal, test, c, sig, sc, q = load_audit()

    def dog_logits(sd, x):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            out = model(x)
        return out[:, DOG_CLS].cpu().numpy()

    # z-cal arrays
    mem_dog_ref = np.stack([dog_logits(sd, mem[0])
                            for sd in fit_sds])
    mu_mem = mem_dog_ref.mean(0)
    sd_mem = mem_dog_ref.std(0) + 1e-12

    # fresh cat records for non-member comparison
    fx = torch.tensor(np.stack(
        [tset.cifar10.data[i] / 255.0 for i in fresh]
    ).transpose(0, 3, 1, 2)).float().to(device)
    fresh_dog_ref = np.stack([dog_logits(sd, fx)
                              for sd in fit_sds])
    mu_fr = fresh_dog_ref.mean(0)
    sd_fr = fresh_dog_ref.std(0) + 1e-12

    # base start (e3-s1)
    base_sd = torch.load(E3[1], map_location=device)

    def zscore_dog(sd):
        dm = (dog_logits(sd, mem[0]) - mu_mem) / sd_mem
        df = (dog_logits(sd, fx) - mu_fr) / sd_fr
        return dm, df

    # baseline
    bm, bf = zscore_dog(base_sd)
    auc_base = float(rank_auc(bm, bf))
    print('BASE (e3-s1): dog-channel AUC = %.3f' % auc_base, flush=True)

    # encoded finals
    results = []
    for i, p in enumerate(sorted(glob.glob(CE_GLOB))):
        sd = torch.load(p, map_location=device)
        dm, df = zscore_dog(sd)
        auc = float(rank_auc(dm, df))
        results.append(('CE e3-s%d' % (i + 1) if i < 3 else
                        'CE e4-s%d' % (i - 2), auc))
        print(' %s: dog AUC = %.3f' % results[-1], flush=True)
    for i, p in enumerate(GR_ORIG):
        sd = torch.load(p, map_location=device)
        dm, df = zscore_dog(sd)
        auc = float(rank_auc(dm, df))
        tag = ('GR e3-s%d' % (i + 1) if i < 3 else
               'GR e4-s%d' % (i - 2))
        results.append((tag, auc))
        print(' %s: dog AUC = %.3f' % (tag, auc), flush=True)

    aucs = [a for _, a in results]
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'baseline_auc': auc_base,
           'ce_dog_aucs': [a for (t, a) in results if 'CE' in t],
           'gr_dog_aucs': [a for (t, a) in results if 'GR' in t],
           'mean_dog_auc': float(np.mean(aucs)),
           'max_dog_auc': float(np.max(aucs))}
    print(json.dumps({k: round(v, 3) if isinstance(v, float)
                      else [round(x, 3) for x in v]
                      for k, v in out.items()
                      if k != 'generated'}, indent=1), flush=True)
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/xce0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
