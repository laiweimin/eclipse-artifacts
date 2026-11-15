"""#1 z-detector adaptive arm (prereg 2026-11-10 five-gap program).
Encoder optimizes J_CE under the COMPOSITE constraint: pass A6 audit
AND one-sided top-15 z-mean (member direction, CAL-Z vs ew9F fit refs,
threshold = 95th pct over 18 honest e2/e3/e4 ft models, S13/a9z3
protocol) stays below threshold. 3 e4 starts.
Gates: ZDET-EVADED (>=2/3 starts pass A6 + CI-positive + detector
silent) / ZDET-ROBUST (0 CI-positive under composite) / PARTIAL."""
import argparse, glob, json, os, time
import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import channels
from uw.run_r1 import load_audit

FIT_GLOB = 'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'
HON_GLOB = 'research_outputs/uw/ex3-20260929T101715/ckpt/e%s_ft_s*.pt'


def run_zad(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    te = np.arange(len(tset.cifar10.targets))
    cat_te = te[np.asarray(tset.cifar10.targets) == 3]
    cat_tr = ctx['pool']
    poolset = set(cat_tr.tolist())
    mem = (ctx['xF'], ctx['yF'])
    model = ctx['model']

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    fit_refs = sorted(glob.glob(FIT_GLOB))
    mu_ref = np.mean([channels(model, torch.load(p, map_location=device),
                               *mem)['neggrad'] for p in fit_refs], 0)
    sg_ref = np.std([channels(model, torch.load(p, map_location=device),
                              *mem)['neggrad'] for p in fit_refs], 0)

    def t1(sd):
        z = (channels(model, sd, *mem)['neggrad'] - mu_ref) \
            / (sg_ref + 1e-12)
        return float(np.mean(np.sort(z)[::-1][:15]))

    hon = sorted(glob.glob(HON_GLOB % '2')) + \
        sorted(glob.glob(HON_GLOB % '3')) + \
        sorted(glob.glob(HON_GLOB % '4'))
    tau_z = float(np.percentile([t1(torch.load(p, map_location=device))
                                 for p in hon], 95))
    print(' zad tau_z %.4f (from %d honest)' % (tau_z, len(hon)),
          flush=True)

    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset]), 300,
        replace=False))
    cx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in cons])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in cons])
          .long().to(device))

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]), mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx), ny))

    def composite_ok(sd):
        return sc(ctx['eval_full'](sd)) <= q20 and t1(sd) <= tau_z

    def gen(sd):
        cur, lr = sd, LR0
        rs = np.random.RandomState(54200)
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
                if composite_ok(cand):
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

    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(np.array([i for i in cat_te if i not in poolset]),
                     cons), 200, replace=False))
    fx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in fresh])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in fresh])
          .long().to(device))

    def auc_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        return rank_auc(zm, zf)

    def ci_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        rs = np.random.RandomState(54300)
        bs = [rank_auc(zm[rs.choice(len(zm), len(zm), True)], zf)
              for _ in range(2000 if not cli.smoke else 50)]
        return (float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)))

    rows = []
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))[:3]
    for p in e4:
        sd = torch.load(p, map_location=device)
        base = auc_nl(sd)
        fF = gen(sd)
        aF = auc_nl(fF)
        lo, hi = ci_nl(fF)
        pa = bool(sc(ctx['eval_full'](fF)) <= q20)
        pz = bool(t1(fF) <= tau_z)
        rows.append({'start': os.path.basename(p), 'base': base,
                     'gain': aF - base, 'ci_lo': lo - base,
                     'ci_hi': hi - base, 'passA6': pa,
                     'zdet_silent': pz})
        print(' zad %s base %.3f g %.3f ci[%.3f,%.3f] A6 %s zdet %s'
              % (os.path.basename(p), base, aF - base, lo - base,
                 hi - base, pa, pz), flush=True)
    ev = sum(1 for r in rows if r['passA6'] and r['zdet_silent']
             and r['ci_lo'] > 0)
    cp = sum(1 for r in rows if r['ci_lo'] > 0)
    v = ('ZDET-EVADED' if ev >= 2 else
         ('ZDET-ROBUST' if cp == 0 else 'ZDET-PARTIAL'))
    out = {'tau_z': tau_z, 'rows': rows, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'zad-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_zad(cli)


if __name__ == '__main__':
    main()
