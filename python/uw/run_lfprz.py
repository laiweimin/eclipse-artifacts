"""UW Step-1b: z-space per-member gap census (zero training).

Follow-up to run_lfpr.py: the raw-space census showed members sit
within 1 aux-sigma of tau, but Step-2 allocation must be designed in
the space the decoder will actually use. This script reports, for the
reference-calibrated decoder z_negloss / z_neggrad:

  - per-member z-scores (base vs final) for all 6 starts x 2 families;
  - z-tau at beta=5%/2.5%/1% frozen on the aux set;
  - the number of STUCK members (final z < tau) within 0.5 / 1 / 2
    aux-z-sigma of tau -- the reachable set a record-selective
    construction could target;
  - how many members each final already identifies (z >= tau) with the
    measured eval FPR.

Reads checkpoints only; writes research_outputs/uw/lfprz-<ts>/
verdict.json.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
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

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device) for p in fit_paths]
    sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [  # per fit model: {set: {channel: arr}}
        {nm: channels(model, sd, x, y)
         for nm, (x, y) in sets.items()} for sd in fit_sds]

    src = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:]
    ce_by_tag = {os.path.basename(p)[3:-3]: p for p in src}
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    fe3 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s*.pt'))
    fe4 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s*.pt'))
    pairs = [('e3-s1', e3[1], fe3[0]), ('e3-s3', e3[3], fe3[1]),
             ('e3-s4', e3[4], fe3[2]), ('e4-s0', e4[0], fe4[0]),
             ('e4-s1', e4[1], fe4[1]), ('e4-s2', e4[2], fe4[2])]
    if cli.smoke:
        pairs = pairs[:1]

    rows = []
    for tag, sp, fp in pairs:
        sd = torch.load(sp, map_location=device)
        finals = {'ce': torch.load(ce_by_tag[tag], map_location=device),
                  'grad': torch.load(fp, map_location=device)}
        for fam, fin in finals.items():
            b_sc = {nm: channels(model, sd, x, y)
                    for nm, (x, y) in sets.items()}
            f_sc = {nm: channels(model, fin, x, y)
                    for nm, (x, y) in sets.items()}
            for ch in ('negloss', 'neggrad'):
                zmb = zcal(b_sc['mem'][ch],
                           [fs['mem'][ch] for fs in fit_scores])
                zmf = zcal(f_sc['mem'][ch],
                           [fs['mem'][ch] for fs in fit_scores])
                zab = zcal(b_sc['aux'][ch],
                           [fs['aux'][ch] for fs in fit_scores])
                zaf = zcal(f_sc['aux'][ch],
                           [fs['aux'][ch] for fs in fit_scores])
                zef = zcal(f_sc['eval'][ch],
                           [fs['eval'][ch] for fs in fit_scores])
                tau5 = float(np.quantile(zaf, 0.95))
                sig = float(np.std(zaf)) + 1e-12
                gaps = sorted((tau5 - z) / sig
                              for z in zmf if z < tau5)
                rows.append({
                    'start': tag, 'family': fam, 'ch': 'z_' + ch,
                    'tau5': tau5, 'sig_aux': sig,
                    'n_id_fin': int(np.sum(zmf >= tau5)),
                    'n_id_base': int(np.sum(zmb >= tau5)),
                    'fpr5': float(np.mean(zef >= tau5)),
                    'n_stuck': len(gaps),
                    'reach_0.5sig': int(sum(1 for g in gaps if g <= 0.5)),
                    'reach_1sig': int(sum(1 for g in gaps if g <= 1)),
                    'reach_2sig': int(sum(1 for g in gaps if g <= 2)),
                    'gaps_sorted_top8': [round(g, 2) for g in gaps[:8]],
                    'per_member_z_fin': [round(float(z), 3)
                                         for z in zmf]})
                print(' %s %s %s: id %d->%d (fpr %.3f) stuck %d '
                      'reach .5/1/2s %d/%d/%d top-gaps %s'
                      % (tag, fam, 'z_' + ch,
                         rows[-1]['n_id_base'], rows[-1]['n_id_fin'],
                         rows[-1]['fpr5'], rows[-1]['n_stuck'],
                         rows[-1]['reach_0.5sig'],
                         rows[-1]['reach_1sig'],
                         rows[-1]['reach_2sig'],
                         rows[-1]['gaps_sorted_top8']), flush=True)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT written', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'lfprz-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()
