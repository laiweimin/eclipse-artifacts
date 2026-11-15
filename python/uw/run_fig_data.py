"""Figure data extraction (B4-13): metric-space PCA coordinates
for honest ensemble vs encoded finals. Honest vectors from the
m3-2 metrics cache; finals' vectors recomputed by forward-pass
eval_full (zero training). Output: pgfplots coordinate table.
"""
import glob
import json

import numpy as np
import torch

cache = sorted(glob.glob(
    'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
ens = json.load(open(cache))['ens_mm']
MET = ['m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS',
       'm5_adv', 'm6_fm']


def cvec(e):
    return np.array([e[m] for m in MET], dtype=float)


hon = [cvec(e) for g in ('e2', 'e3', 'e4') for e in ens[g]]
X = np.stack(hon)
c = X.mean(0)
Xc = X - c
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Vt[:2]
var = (S ** 2 / np.sum(S ** 2))[:2]
print('PC variance ratios: %.3f %.3f' % tuple(var))

device = torch.device('cuda:0')
from uw.run_ez import setup_c10
ctx = setup_c10(device)
model = ctx['model']
ce = sorted(glob.glob(
    'research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt'))
gr = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
tags = ['CE e3-s1', 'CE e3-s3', 'CE e3-s4', 'CE e4-s0',
        'CE e4-s1', 'CE e4-s2', 'GR e3-s1', 'GR e3-s3',
        'GR e3-s4', 'GR e4-s0', 'GR e4-s1', 'GR e4-s2']


def proj(sd):
    mm = ctx['eval_full'](sd)
    v = cvec(mm)
    d = (v - c) @ P.T
    return float(d[0]), float(d[1])


fins = []
for tag, p in zip(tags, ce + gr):
    sd = torch.load(p, map_location=device)
    fins.append({'tag': tag, 'pc': proj(sd)})
    print('%-8s %.3f %.3f' % (tag, fins[-1]['pc'][0],
                              fins[-1]['pc'][1]), flush=True)

hon_pts = [[float(a), float(b)]
           for a, b in ((v - c) @ P.T for v in hon)]
out = {'pc_variance': [float(x) for x in var],
       'honest': hon_pts, 'finals': fins}
with open('research_outputs/uw/fig_pca.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print('saved fig_pca.json')
