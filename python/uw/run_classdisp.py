# -*- coding: utf-8 -*-
"""#5b class-wise dispersion exhibit (prereg 2026-11-10).
For each class (frog=6, dog=5, airplane=0): build the class pool
(train-side records of that class), f_deep-select k=30, train 3
honest retrains (D minus F, 40ep from scratch, run_ez fit_full
recipe), evaluate the 6 ew3 noise ckpts (full-data retrains) and
the 3 new retrains on the class F with metrics_from_logits;
report dispersion ratio (honest max-dev from median / noise
max-dev from median) per metric per class. Gates: report the
table; RATIOS-CONSISTENT if forget-side ratio > 1 in all three
classes, else NOT-UNIFORM."""
import argparse, glob, json, os, time
import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ew3 import metrics_from_logits

CLASSES = {'frog': 6, 'dog': 5, 'airplane': 0}
NOISE_CK = sorted(glob.glob(
    'research_outputs/uw/ew3-20260925T113900/ckpt/noise_s*.pt'))
THETA0 = None  # retrains from scratch per run_ez fit_full


def run_classdisp(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = ctx['yfull']
    n = len(yfull)
    model = ctx['model']

    def logits(sd, idx, src):
        model.load_state_dict(sd)
        model.eval()
        zs = []
        arr = np.asarray(idx)
        nb = max(1, min(50, len(arr)))
        with torch.no_grad():
            for b in np.array_split(arr, nb):
                if len(b) == 0:
                    continue
                x = torch.tensor(np.stack(
                    [src.cifar10.data[i] / 255.0 for i in b])
                    ).permute(0, 3, 1, 2).float().to(device)
                zs.append(model(x).cpu().numpy())
        return np.concatenate(zs)

    out = {'classes': {}}
    for cname, cls in CLASSES.items():
        pool = np.where(yfull == cls)[0]
        pool = pool[:2000]  # class pool cap, frozen
        rng = np.random.RandomState(46000 + cls)
        f = np.sort(rng.choice(pool, 30, replace=False))
        keep = np.setdiff1d(np.arange(n), f)
        te_cls = np.where(np.asarray(
            tset.cifar10.targets) == cls)[0]
        yF = yfull[f]
        # metric eval on (F, class-fresh test 300, class test slice)
        fresh300 = np.sort(np.random.RandomState(
            46010 + cls).choice(te_cls, min(300, len(te_cls)),
                                replace=False))
        yS = np.asarray(tset.cifar10.targets)[fresh300]
        te_all = np.arange(len(tset.cifar10.targets))
        yT = np.asarray(tset.cifar10.targets)

        def mm(sd):
            zF = logits(sd, f, trset)
            zS = logits(sd, fresh300, tset)
            zT = logits(sd, te_all, tset)
            return metrics_from_logits(zF, yF, zS, yS, zT, yT)

        # honest retrains: 3 seeds, from-scratch 40ep on keep
        rt_ms = []
        for s in (47101 + cls, 47102 + cls, 47103 + cls):
            ck = 'research_outputs/uw/classdisp_ckpt/%s_rt_s%d.pt' \
                % (cname, s)
            if os.path.exists(ck):
                sd = torch.load(ck, map_location=device)
            else:
                sd = ctx['fit_full'](keep, s, 40)
                os.makedirs(os.path.dirname(ck), exist_ok=True)
                torch.save(sd, ck)
            rt_ms.append(mm(sd))
            print(' %s rt s%d done' % (cname, s), flush=True)
        noise_ms = [mm(torch.load(p, map_location=device))
                    for p in NOISE_CK]
        KEYS = ('m3_Floss', 'm4_KS', 'm5_adv', 'm6_fm', 'm1_acc')
        rat = {}
        for k in KEYS:
            hv = np.array([m[k] for m in rt_ms])
            nv = np.array([m[k] for m in noise_ms])
            hc, nc = np.median(hv), np.median(nv)
            rat[k] = float(np.max(np.abs(hv - hc))
                           / max(np.max(np.abs(nv - nc)), 1e-9))
        out['classes'][cname] = {'ratios': rat}
        print(' %s ratios: %s' % (cname, {k: round(v, 2)
                                          for k, v in rat.items()}),
              flush=True)
    ok = all(out['classes'][c]['ratios']['m3_Floss'] > 1
             for c in CLASSES)
    out['verdict'] = 'RATIOS-CONSISTENT' if ok else 'NOT-UNIFORM'
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'classdisp-' + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_classdisp(cli)


if __name__ == '__main__':
    main()
