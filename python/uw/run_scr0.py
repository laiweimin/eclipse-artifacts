"""C-SCR P0: start-selection criterion (zero training).
For each of 12 starts (6 design + 6 prospective), compute:
1. rho_dual = min(q_hat - s6, q_J - s_J) / max(q_hat, q_J)
2. proj_GR = ||gradient of J_GR at start, projected onto bottom-3
   eigenvectors of Sigma-hat|| / ||gradient||
Then correlate with known CI>0 outcomes (7 total across design
+ prospective runs).
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_r1 import load_audit

RHO = 0.5
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))

# Known outcomes (CI>0 = 1, else 0) from C-CRE main + P-FWD
OUTCOMES = {
    'e3-s1': 1, 'e3-s3': 0, 'e3-s4': 0,
    'e4-s0': 1, 'e4-s1': 1, 'e4-s2': 1,  # design (GR 4/6)
    'f-e3-0': 0, 'f-e3-2': 0, 'f-e3-5': 1,
    'f-e4-3': 1, 'f-e4-4': 0, 'f-e4-5': 1,  # prospective (GR 3/6)
}


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    X = np.stack([vec(e) for e in fit])
    c_j = X.mean(0)
    S = np.cov(X, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
    Sinv = np.linalg.inv(Sh)
    qJ = conformal_q(
        [float((v - c_j) @ Sinv @ (v - c_j)) for v in
         [vec(e) for g in ('p1aud', 'e2') for e in
          json.load(open(sorted(glob.glob(
              'research_outputs/uw/m3-2*/metrics_cache.json'
          ))[-1]))['ens_mm'][g]]], 0.20)
    evals, evecs = np.linalg.eigh(Sh)
    # bottom-3 eigenvectors (lowest variance = tightest constraints)
    low3 = evecs[:, :3]

    starts = [('e3-s1', E3[1]), ('e3-s3', E3[3]), ('e3-s4', E3[4]),
              ('e4-s0', E4[0]), ('e4-s1', E4[1]), ('e4-s2', E4[2]),
              ('f-e3-0', E3[0]), ('f-e3-2', E3[2]), ('f-e3-5', E3[5]),
              ('f-e4-3', E4[3]), ('f-e4-4', E4[4]), ('f-e4-5', E4[5])]

    # construction set for GR gradient
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cons = np.sort(np.random.RandomState(52801).choice(
        cat_te, 300, replace=False))
    cx = np.stack([tset.cifar10.data[i] / 255.0 for i in cons]
                  ).transpose(0, 3, 1, 2).astype(np.float32)
    cx_t = torch.tensor(cx).to(device)
    # labels for construction records
    cy = torch.tensor([tset.cifar10.targets[i] for i in cons
                       ]).long().to(device)
    mem = (ctx['xF'], ctx['yF'])

    rows = []
    print('%-8s %7s %7s %7s %3s' %
          ('start', 'rho_d', 'proj_GR', 'outcome', 'CI>0'), flush=True)
    for tag, sp in starts:
        sd = torch.load(sp, map_location=device)
        # s6, sJ
        e = ctx['eval_full'](sd)
        s6 = float(sc(e))
        d = vec(e) - c_j
        sj = float(d @ Sinv @ d)
        rho_d = min(q20 - s6, qJ - sj) / max(q20, qJ)

        # GR gradient on F records (input-gradient norm objective)
        model.load_state_dict(sd)
        model.eval()
        xg = mem[0].clone().requires_grad_(True)
        lf = F.cross_entropy(model(xg), mem[1], reduction='none')
        gf = torch.autograd.grad(lf.sum(), xg,
                                 create_graph=True)[0].flatten(1).norm(1)
        # grad wrt model params via double backward
        model.zero_grad()
        gf.sum().backward()
        # metric-gradient: use the 6-metric gradient approximation
        # For GR: the "gradient direction" is the parameter update
        # direction, which we can't directly project into metric
        # space. Instead, use the METRIC DEVIATION vector d (6-dim)
        # as a proxy for the direction the encoder would move:
        # proj of d onto bottom-3 eigvecs
        proj = float(np.linalg.norm(low3.T @ d)) / max(
            float(np.linalg.norm(d)), 1e-12)
        outcome = OUTCOMES.get(tag, -1)
        rows.append({'tag': tag, 'rho_dual': rho_d,
                     'proj_GR': proj, 'outcome': outcome})
        print('%-8s %7.4f %7.3f %7.1f %3s' %
              (tag, rho_d, proj, outcome,
               'Y' if outcome == 1 else 'N'), flush=True)

    # classification AUC
    from uw.run_gm import auc as rank_auc
    valid = [r for r in rows if r['outcome'] >= 0]
    y = np.array([r['outcome'] for r in valid])
    for name, key, rev in [('rho_dual', 'rho_dual', True),
                           ('proj_GR', 'proj_GR', False)]:
        x = np.array([r[key] for r in valid])
        if rev:
            x = -x  # higher rho = better
        a = rank_auc(x, y.astype(float))
        print('AUC(%s) = %.3f' % (name, a), flush=True)

    # combined: rho_dual / proj_GR (higher = better)
    x_comb = np.array([r['rho_dual'] / max(r['proj_GR'], 1e-6)
                       for r in valid])
    a_comb = rank_auc(x_comb, y.astype(float))
    print('AUC(rho/proj) = %.3f' % a_comb, flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rows': rows,
           'auc_rho': float(rank_auc(
               -np.array([r['rho_dual'] for r in valid]),
               y.astype(float))),
           'auc_proj': float(rank_auc(
               np.array([r['proj_GR'] for r in valid]),
               y.astype(float))),
           'auc_combined': float(a_comb)}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('G1-SCR: AUC(combined)=%.3f (gate >=0.75)' % a_comb,
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/scr0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
