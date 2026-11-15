"""C2-S2: aggregate member-sensitive blind pool statistics (prereg
card docs/plans/2026-10-16-c2pool-blind-deletion-prereg.md sec.6,
frozen before running).

Same pools/seeds/models as v1. New statistics computed on pool z
vectors: T_topk_mean(k=5/15/30), T_two_sided (max|z|), T_max
(baseline). Thresholds = per-statistic q95 of HONEST CONTROL pools.
Primary readout: member increment = H1-treat detection minus
H1-decoy-control detection. Leakage vs honest empirical nulls
(unconditional + detection-conditional). Full z vectors archived.
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

N_POOL, N_RES = 100, 50
KS = (5, 15, 30)


def stats_of(z):
    out = {'max': float(np.max(z)),
           'two': float(np.max(np.abs(z)))}
    srt = np.sort(z)[::-1]
    for k in KS:
        out['top%d' % k] = float(np.mean(srt[:k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c2pool2-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    decoy_bank = np.setdiff1d(rest, fresh)
    decoy_bank = np.sort(np.random.RandomState(7901).choice(
        decoy_bank, 200, replace=False))
    mem = (ctx['xF'], ctx['yF'])

    def rec_ids(ids):
        x = torch.tensor(np.stack(
            [tset.cifar10.data[i] / 255.0
             for i in ids])).permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([tset.cifar10.targets[i]
                          for i in ids]).long().to(device)
        return x, y

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    pool_ids = np.concatenate([fresh, decoy_bank])
    sets = {'mem': mem, 'pool': rec_ids(pool_ids)}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in sets.items()}
                  for sd in fit_sds]

    def zvecs(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            out[chn] = {
                'mem': zcal(s['mem'][chn],
                            [fs['mem'][chn]
                             for fs in fit_scores]),
                'pool': zcal(s['pool'][chn],
                             [fs['pool'][chn]
                              for fs in fit_scores])}
        return out

    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6] \
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/'
            'e4_ft_s*.pt'))[:6]
    ce = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:]
    g1 = sorted(glob.glob(
        'research_outputs/uw/gate-g1b-20261007T102014/'
        'ckpt/fb_s*.pt')) + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T104042/'
            'ckpt/fb_s*.pt'))
    h1_paths = ce + g1
    h1_names = (['CE:%d' % i for i in range(len(ce))]
                + ['GR:%d' % i for i in range(len(g1))])

    def arm_rows(paths, names, arm):
        rows = []
        for pth, nm in zip(paths, names):
            z = zvecs(torch.load(pth, map_location=device))
            for chn in ('negloss', 'neggrad'):
                zmem, zpool = z[chn]['mem'], z[chn]['pool']
                rs = np.random.RandomState(8100)
                recs = []
                for r in range(N_RES):
                    dec = rs.choice(len(zpool), 70,
                                    replace=False)
                    treat = np.concatenate(
                        [zmem, zpool[dec]])
                    ctrl = zpool[rs.choice(
                        len(zpool), N_POOL, replace=False)]
                    top5 = np.argsort(
                        np.abs(treat))[::-1][:5]
                    ov5 = int(np.sum(top5 < len(zmem)))
                    recs.append((stats_of(treat),
                                 stats_of(ctrl),
                                 int(np.argmax(treat)),
                                 int(np.argmax(np.abs(treat))),
                                 ov5))
                rows.append({'arm': arm, 'model': nm, 'ch': chn,
                             'pools': recs})
            print(' %s %s scored' % (arm, nm), flush=True)
        return rows

    hon = arm_rows(honest_paths,
                   ['HON:%d' % i
                    for i in range(len(honest_paths))], 'H0')
    h1 = arm_rows(h1_paths, h1_names, 'H1')

    # archive compact z vectors (pool z shared per model; mem z)
    archive = {'pool_ids': [int(i) for i in pool_ids],
               'fset': [int(i) for i in fset]}
    for lst, key in ((hon, 'H0'), (h1, 'H1')):
        archive[key] = []
        for r in lst:
            archive[key].append(
                {'model': r['model'], 'ch': r['ch'],
                 'pools': [{'t': t, 'c': c, 'am': am,
                            'am2': am2, 'ov5': ov5}
                           for t, c, am, am2, ov5
                           in r['pools']]})

    STAT_KEYS = ['max', 'two', 'top5', 'top15', 'top30']
    gates = {}
    for chn in ('negloss', 'neggrad'):
        hon_r = [r for r in hon if r['ch'] == chn]
        h1_r = [r for r in h1 if r['ch'] == chn]
        for sk in STAT_KEYS:
            ctrl_null = [c[sk] for r in hon_r
                         for t, c, am, am2, ov5 in r['pools']]
            q95 = float(np.quantile(ctrl_null, 0.95))
            det = lambda rows, side: np.array([
                1 if (t if side == 't' else c)[sk] > q95
                else 0 for r in rows
                for t, c, am, am2, ov5 in r['pools']])
            h0c, h0t = det(hon_r, 'c'), det(hon_r, 't')
            h1c, h1t = det(h1_r, 'c'), det(h1_r, 't')
            # leakage vs honest empirical null (treatment pools,
            # unconditional): argmax-in-F, |z|-argmax-in-F,
            # mean top-5 overlap with F
            def leakof(rows):
                amf = np.mean([1.0 if am < 30 else 0.0
                               for r in rows
                               for t, c, am, am2, ov5
                               in r['pools']])
                am2f = np.mean([1.0 if am2 < 30 else 0.0
                                for r in rows
                                for t, c, am, am2, ov5
                                in r['pools']])
                ov = np.mean([float(ov5) for r in rows
                              for t, c, am, am2, ov5
                              in r['pools']])
                return {'argmax_in_F': float(amf),
                        'absargmax_in_F': float(am2f),
                        'mean_top5_overlap': float(ov)}
            lk0, lk1 = leakof(hon_r), leakof(h1_r)
            gates['%s|%s' % (chn, sk)] = {
                'q95_null': q95,
                'H0_ctrlFPR': float(h0c.mean()),
                'H0_treatDet': float(h0t.mean()),
                'H1_ctrlDet': float(h1c.mean()),
                'H1_treatDet': float(h1t.mean()),
                'member_increment':
                    float(h1t.mean() - h1c.mean()),
                'leak_H0': lk0, 'leak_H1': lk1}
            print(' %s|%s q95 %.2f H0c %.3f H1c %.3f H1t %.3f '
                  'incr %+.3f | leak amF %.2f->%.2f ov5 '
                  '%.2f->%.2f'
                  % (chn, sk, q95, h0c.mean(), h1c.mean(),
                     h1t.mean(),
                     gates['%s|%s' % (chn, sk)][
                         'member_increment'],
                     lk0['argmax_in_F'], lk1['argmax_in_F'],
                     lk0['mean_top5_overlap'],
                     lk1['mean_top5_overlap']), flush=True)

    with open(os.path.join(outdir, 'verdict_raw.json'),
              'w') as fh:
        json.dump(archive, fh)

    # S2-G1 scan
    g1_hits = [k for k, g in gates.items()
               if g['member_increment'] >= 0.15
               and g['H1_treatDet'] >= 0.5
               and g['H0_ctrlFPR'] <= 0.10]
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'gates': gates, 'S2_G1_hits': g1_hits,
           'S2_G1_pass': bool(g1_hits)}
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('S2-G1 pass=%s hits=%s' % (out['S2_G1_pass'],
                                     g1_hits), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
