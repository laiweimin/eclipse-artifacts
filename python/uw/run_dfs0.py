"""C-DFS P0: per-record encodability analysis (zero training).
For each record in the 2000-record pool, compute features from
the honest ensemble and check which records' membership signals
are easier to encode (higher z-score shift in successful finals
vs failed finals).
"""
import argparse, glob, json, os, time
import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels

E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
GR_FILES = {
    'e3-s1': 'research_outputs/uw/cre-20261101T164359/cre_grad_e3-s1.pt',
    'e3-s3': 'research_outputs/uw/cre-20261101T164359/cre_grad_e3-s3.pt',
    'e3-s4': 'research_outputs/uw/cre-20261101T164359/cre_grad_e3-s4.pt',
    'e4-s0': 'research_outputs/uw/cre-20261101T164359/cre_grad_e4-s0.pt',
    'e4-s1': 'research_outputs/uw/cre-20261101T164359/cre_grad_e4-s1.pt',
    'e4-s2': 'research_outputs/uw/cre-20261101T164359/cre_grad_e4-s2.pt',
}
# CI>0 outcomes from C-CRE GR
CI_POS = {'e3-s1': True, 'e3-s3': False, 'e3-s4': False,
          'e4-s0': True, 'e4-s1': True, 'e4-s2': True}


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    mem = (ctx['xF'], ctx['yF'])

    # Honest ensemble features for F records
    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit_ch = [channels(model, sd, *mem) for sd in fit_sds]
    # z-cal per record (neggrad)
    for ch in ('neggrad',):
        ref = np.stack([fc[ch] for fc in fit_ch])
        mu, sg = ref.mean(0), ref.std(0) + 1e-12

    # Pool records (all 2000 cat train records) — compute z for a
    # sample (computing all 2000 × 6 models is expensive, sample 200)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    trset = CIFAR10(train=True,
                    transform=_tf.Compose([_tf.ToTensor()]))
    rs = np.random.RandomState(54321)
    pool_sample = np.sort(rs.choice(cat_tr, 200, replace=False))
    # Remove F records from pool sample
    pool_sample = np.setdiff1d(pool_sample, fset)

    def rec_x(idxs):
        x = torch.tensor(np.stack(
            [trset.cifar10.data[i] / 255.0 for i in idxs
             ]).transpose(0, 3, 1, 2)).float().to(device)
        y = torch.tensor([trset.cifar10.targets[i]
                          for i in idxs]).long().to(device)
        return x, y

    # For pool records: compute honest z-scores
    px, py = rec_x(pool_sample)
    pool_fit_ch = [channels(model, sd, px, py) for sd in fit_sds]
    pool_ref = np.stack([fc['neggrad'] for fc in pool_fit_ch])
    pool_mu = pool_ref.mean(0)
    pool_sg = pool_ref.std(0) + 1e-12

    # For each GR final: compute per-record z-scores of F records
    print('Computing per-record encodability from 6 GR finals...',
          flush=True)
    success_z = []  # per-record z from CI>0 finals
    fail_z = []     # per-record z from CI<=0 finals
    for tag, path in GR_FILES.items():
        sd = torch.load(path, map_location=device)
        fin_ch = channels(model, sd, *mem)
        z = (fin_ch['neggrad'] - mu) / sg
        if CI_POS[tag]:
            success_z.append(z)
        else:
            fail_z.append(z)

    success_z = np.stack(success_z)  # (n_success, 30)
    fail_z = np.stack(fail_z)         # (n_fail, 30)

    # Per-record encodability: mean z in successful finals
    # minus mean z in failed finals
    enc_resp = success_z.mean(0) - fail_z.mean(0)

    # Pool record features (honest z-scores)
    base_sd = torch.load(E3[1], map_location=device)
    base_pool_ch = channels(model, base_sd, px, py)
    pool_z_base = (base_pool_ch['neggrad'] - pool_mu) / pool_sg

    # Correlation: do records with higher honest z-score have
    # higher encodability?
    from scipy.stats import spearmanr
    # F records: get their base z too
    base_f_ch = channels(model, base_sd, *mem)
    f_z_base = (base_f_ch['neggrad'] - mu) / sg

    rho, pval = spearmanr(f_z_base, enc_resp)
    print('F records: enc_resp range [%.3f, %.3f], mean=%.3f' %
          (enc_resp.min(), enc_resp.max(), enc_resp.mean()), flush=True)
    print('Spearman(base_z, enc_resp) = %.3f (p=%.4f)' %
          (rho, pval), flush=True)

    # Identify top/bottom encodable records
    top_idx = np.argsort(enc_resp)[::-1][:10]
    bot_idx = np.argsort(enc_resp)[:10]
    print('Top-10 encodable records (idx, enc_resp, base_z):')
    for i in top_idx:
        print('  %d: %.3f (%.3f)' % (i, enc_resp[i], f_z_base[i]),
              flush=True)
    print('Bottom-10 (idx, enc_resp, base_z):')
    for i in bot_idx:
        print('  %d: %.3f (%.3f)' % (i, enc_resp[i], f_z_base[i]),
              flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'enc_resp_per_record': enc_resp.tolist(),
           'f_z_base': f_z_base.tolist(),
           'spearman_rho': float(rho), 'spearman_p': float(pval),
           'top10_idx': top_idx.tolist(),
           'bottom10_idx': bot_idx.tolist(),
           'n_success_finals': success_z.shape[0],
           'n_fail_finals': fail_z.shape[0]}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/dfs0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
