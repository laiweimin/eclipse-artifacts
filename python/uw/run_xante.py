"""Ex-ante audit-aware theta0 shaping (prereg card:
docs/plans/2026-10-16-xante-audit-aware-training-prereg.md -- frozen
before any run; read it first).

Arms (per card): H honest / P post-hoc encoding are REUSED from the
lfpr run (no retraining); X1 = theta0 with F upsampled 10x; X2 =
clean training + every-10-epoch unlearning-simulation penalty on
differentiable audit proxies (m1soft, m3, m6 vs centers, normalized
by MAD sig_i). Unlearning operator = exact stage-8 recipe (SGD
0.01/0.9/5e-4, 1 epoch on D\\F, batch 100, seed-shuffled).

Readouts: A6 pass (q20), z-decoder TPR@5% (measured FPR) vs clean
ew9F reference, per-record detectability D_z = max|z_i| on F, m1
utility. Gates G1-G4 computed per card (G2 as date-corrected).
X2 simulation proxies declared: BN buffers frozen during sim,
plain-SGD sim steps, m1soft centered on the fit ensemble's own
soft value (scale-correct center).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal

FT_SEEDS = (44101, 44102, 44103)
SIM_STEPS = 50
SIM_BATCH = 32
PEN_EVERY = 10
LAM = 1.0


def fwd_with(model, params, buffers, x):
    sd = dict(buffers)
    sd.update(params)
    return torch.func.functional_call(model, sd, (x,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--lam', default=1.0, type=float,
                    help='v2: penalty weight (v1=1.0)')
    ap.add_argument('--ret', default=0.0, type=float,
                    help='v2: retention weight on m6 (v1=0)')
    ap.add_argument('--arms', default='X1,X2')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'xante-' \
        + ('v2-' if (cli.lam != 1.0 or cli.ret != 0.0)
           else '') + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    ck = os.path.join(outdir, 'ckpt')
    os.makedirs(ck, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    trset = CIFAR10(train=True,
                    transform=_tf.Compose([
                        _tf.RandomHorizontalFlip(),
                        _tf.RandomCrop(32, 4), _tf.ToTensor()]))
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
    n = len(yfull)
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    keep = np.setdiff1d(np.arange(n), fset)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)

    _, _, _, c_aud, sig_aud, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in fit_sds]

    def zread(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in ev_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][chn],
                      [fs['mem'][chn] for fs in fit_scores])
            za = zcal(s['aux'][chn],
                      [fs['aux'][chn] for fs in fit_scores])
            ze = zcal(s['eval'][chn],
                      [fs['eval'][chn] for fs in fit_scores])
            tau = float(np.quantile(za, 0.95))
            out['z_' + chn] = {
                'tpr5': float(np.mean(zm >= tau)),
                'fpr5': float(np.mean(ze >= tau)),
                'auc': float(rank_auc(zm, ze)),
                'd_z': float(np.max(np.abs(zm)))}
        return out

    from tools.models import resnet18

    def fresh_model():
        return resnet18(num_classes=10).to(device)

    epochs = 3 if cli.smoke else 40
    pen_epochs = ([epochs - 1] if cli.smoke else
                  list(range(9, epochs, PEN_EVERY)))
    # penalty eval batches (fixed)
    prs = np.random.RandomState(7051)
    pen_ids = prs.choice(np.setdiff1d(np.arange(n), fset),
                         256, replace=False)
    pen_x = torch.stack([trset[int(i)][1]
                         for i in pen_ids]).to(device)
    pen_y = torch.tensor(
        yfull[pen_ids]).long().to(device)
    # scale-correct m1soft center from the fit ensemble itself
    bufs = {k: v for k, v in model.state_dict().items()
            if k not in dict(model.named_parameters())}
    fit_soft = []
    for sd in fit_sds:
        with torch.no_grad():
            p = torch.softmax(
                fwd_with(model, {k: v for k, v in sd.items()
                                 if k in dict(
                                     model.named_parameters())},
                         bufs, pen_x), 1)
            fit_soft.append(float(p.gather(
                1, pen_y[:, None]).squeeze(1).mean()))
    m1soft_c = float(np.median(fit_soft))
    print('m1soft center (fit ens) %.4f (c m1_acc %.4f)'
          % (m1soft_c, c_aud['m1_acc']), flush=True)

    def train_theta0(shape, seed):
        torch.manual_seed(seed)
        m = fresh_model()
        opt = torch.optim.SGD(m.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=40)
        rs = np.random.RandomState(seed + 2)
        ids = (np.concatenate([np.arange(n)] + [fset] * 9)
               if shape == 'X1' else np.arange(n))
        lf = torch.nn.CrossEntropyLoss()
        for ep in range(epochs):
            order = rs.permutation(len(ids))
            m.train()
            for i in range(0, len(ids), 100):
                b = ids[order[i:i + 100]].astype(int)
                x = torch.stack([trset[int(j)][1]
                                 for j in b]).to(device)
                y = torch.tensor(
                    yfull[b]).long().to(device)
                opt.zero_grad()
                lf(m(x), y).backward()
                opt.step()
            sched.step()
            if shape == 'X2' and ep in pen_epochs:
                # connected unrolled simulation from LIVE params
                params = dict(m.named_parameters())
                bufs_m = {k: v.detach()
                          for k, v in m.state_dict().items()
                          if k not in params}
                sim = dict(params)  # live leaves, no detach
                g = torch.Generator().manual_seed(
                    seed + 100 + ep)
                m.eval()
                for st in range(SIM_STEPS):
                    pos = torch.randint(
                        0, len(keep), (SIM_BATCH,),
                        generator=g).tolist()
                    bidx = keep[pos]
                    x = torch.stack(
                        [trset[int(j)][1]
                         for j in bidx]).to(device)
                    y = torch.tensor(
                        yfull[bidx]).long().to(device)
                    out = fwd_with(m, sim, bufs_m, x)
                    loss = F.cross_entropy(out, y)
                    grads = torch.autograd.grad(
                        loss, list(sim.values()),
                        create_graph=True)
                    sim = {nm: sim[nm] - 0.01 * gr
                           for (nm, _), gr
                           in zip(sim.items(), grads)}
                out_te = fwd_with(m, sim, bufs_m, pen_x)
                p_te = torch.softmax(out_te, 1)
                m1soft = (p_te.gather(
                    1, pen_y[:, None]).squeeze(1)).mean()
                out_f = fwd_with(m, sim, bufs_m, mem[0])
                m3 = F.cross_entropy(out_f, mem[1])
                zy = out_f.gather(
                    1, mem[1][:, None]).squeeze(1)
                zc = out_f.clone()
                zc.scatter_(1, mem[1][:, None], -1e18)
                m6 = (zy - zc.max(1).values).mean()
                m6_dev = (m6 - c_aud['m6_fm']) \
                    / max(sig_aud['m6_fm'], 1e-6)
                pen = cli.lam * (
                    (m1soft - m1soft_c).abs() / 0.01
                    + (m3 - c_aud['m3_Floss']).abs()
                    / max(sig_aud['m3_Floss'], 1e-6)
                    + m6_dev.abs()) - cli.ret * m6_dev
                opt.zero_grad()
                pen.backward()
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0)
                opt.step()
                print('  pen@ep%d %.4f m1s %.3f m3 %.3f m6 %.3f'
                      % (ep, float(pen), float(m1soft),
                         float(m3), float(m6)), flush=True)
                m.train()
        m.eval()
        return {k: v.detach().clone()
                for k, v in m.state_dict().items()}

    def unlearn_ft(sd0, s):
        torch.manual_seed(s)
        m = fresh_model()
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        rs = np.random.RandomState(s + 7)
        order = rs.permutation(len(keep))
        m.train()
        for i in range(0, len(keep), 100):
            b = keep[order[i:i + 100]].astype(int)
            x = torch.stack([trset[int(j)][1]
                             for j in b]).to(device)
            y = torch.tensor(yfull[b]).long().to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        m.eval()
        return {k: v.detach().clone()
                for k, v in m.state_dict().items()}

    rows = []
    shape_seeds = (7101,) if cli.smoke else (7101, 7102, 7103)
    ft_seeds = FT_SEEDS[:1] if cli.smoke else FT_SEEDS

    e36 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6]
    e46 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[:6]
    honest_paths = fit_paths + e36 + e46
    dz_hon = []
    hon_tpr = []
    for p in honest_paths:
        zr = zread(torch.load(p, map_location=device))
        dz_hon.append(max(zr['z_negloss']['d_z'],
                          zr['z_neggrad']['d_z']))
        hon_tpr.append(zr['z_negloss']['tpr5'])
    dz_hon_q95 = float(np.quantile(dz_hon, 0.95))
    hon_tpr_med = float(np.median(hon_tpr))
    print('honest D_z: med %.2f q95 %.2f | honest tpr5 med %.3f'
          % (float(np.median(dz_hon)), dz_hon_q95, hon_tpr_med),
          flush=True)

    p_paths = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:]
    for pth in p_paths:
        sd = torch.load(pth, map_location=device)
        zr = zread(sd)
        rows.append({'arm': 'P', 'final': os.path.basename(pth),
                     's6': float(score_sd(sd)),
                     'pass': bool(score_sd(sd) <= q20),
                     'm1': float(
                         ctx['eval_full'](sd)['m1_acc']),
                     'zr': zr})
        print(' P %s pass=%s tpr5 %.2f d_z %.2f'
              % (os.path.basename(pth), rows[-1]['pass'],
                 zr['z_negloss']['tpr5'],
                 zr['z_negloss']['d_z']), flush=True)

    for shape in tuple(cli.arms.split(',')):
        for s0 in shape_seeds:
            sd0 = train_theta0(shape, s0)
            torch.save(sd0, os.path.join(
                ck, '%s_theta0_s%d.pt' % (shape, s0)))
            pre = zread(sd0)
            print(' %s s%d PRE tpr5 %.2f fpr %.3f d_z %.2f'
                  % (shape, s0, pre['z_negloss']['tpr5'],
                     pre['z_negloss']['fpr5'],
                     pre['z_negloss']['d_z']), flush=True)
            for fs in ft_seeds:
                sdf = unlearn_ft(sd0, fs)
                torch.save(sdf, os.path.join(
                    ck, '%s_ft_s%d_%d.pt' % (shape, s0, fs)))
                zr = zread(sdf)
                row = {'arm': shape, 'theta0': s0, 'ft': fs,
                       's6': float(score_sd(sdf)),
                       'pass': bool(score_sd(sdf) <= q20),
                       'm1': float(
                           ctx['eval_full'](sdf)['m1_acc']),
                       'zr': zr,
                       'pre_tpr5': pre['z_negloss']['tpr5'],
                       'pre_dz': pre['z_negloss']['d_z']}
                rows.append(row)
                print(' %s s%d/%d pass=%s tpr5 %.2f fpr %.3f '
                      'd_z %.2f m1 %.4f'
                      % (shape, s0, fs, row['pass'],
                         zr['z_negloss']['tpr5'],
                         zr['z_negloss']['fpr5'],
                         zr['z_negloss']['d_z'], row['m1']),
                      flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': float(q20),
           'dz_honest_q95': dz_hon_q95,
           'hon_tpr_med': hon_tpr_med,
           'dz_honest': [float(x) for x in dz_hon],
           'rows': rows}
    if not cli.smoke and 'X2' in cli.arms.split(','):
        x2 = [r for r in rows if r['arm'] == 'X2']
        x1 = [r for r in rows if r['arm'] == 'X1']
        p_r = [r for r in rows if r['arm'] == 'P']
        g1 = {'X2_pass': sum(1 for r in x2 if r['pass']),
              'X2_n': len(x2),
              'X1_pass': sum(1 for r in x1 if r['pass']),
              'X1_n': len(x1)}
        tpr_x2 = [r['zr']['z_negloss']['tpr5'] for r in x2]
        fpr_ok = [r['zr']['z_negloss']['fpr5'] <= 0.08
                  for r in x2]
        g2 = {'mean_tpr5': float(np.mean(tpr_x2)),
              'fpr_ok_frac': float(np.mean(fpr_ok)),
              'h_median': hon_tpr_med,
              'gate': bool(np.mean(tpr_x2)
                           >= hon_tpr_med + 0.10
                           and np.mean(fpr_ok) >= 0.5)}
        dz_x2 = [max(r['zr']['z_negloss']['d_z'],
                     r['zr']['z_neggrad']['d_z']) for r in x2]
        dz_p = [max(r['zr']['z_negloss']['d_z'],
                    r['zr']['z_neggrad']['d_z']) for r in p_r]
        g3 = {'X2_med_dz': float(np.median(dz_x2)),
              'P_med_dz': float(np.median(dz_p)),
              'honest_q95': dz_hon_q95,
              'X2_frac_over': float(np.mean(
                  [d > dz_hon_q95 for d in dz_x2])),
              'P_frac_over': float(np.mean(
                  [d > dz_hon_q95 for d in dz_p]))}
        m1_h = float(np.median(
            [float(ctx['eval_full'](torch.load(
                p, map_location=device))['m1_acc'])
             for p in honest_paths]))
        g4 = {'m1_h_med': m1_h,
              'm1_X2_med': float(np.median(
                  [r['m1'] for r in x2])),
              'gate': bool(m1_h - float(np.median(
                  [r['m1'] for r in x2])) <= 0.03)}
        out['gates'] = {'G1': g1, 'G2': g2, 'G3': g3,
                        'G4': g4}
        print('GATES', json.dumps(out['gates'], indent=1),
              flush=True)
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
