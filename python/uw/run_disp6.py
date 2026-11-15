# -*- coding: utf-8 -*-
"""#3 dispersion ensemble-size (prereg 2026-11-10): n=3 -> n=6 honest
retrains per scale by RESUMING the original ew3 run with 3 added
seeds per scale (exact recipe/metrics reused, zero reimplementation).
C_GRID patched to (0.0,) so the added seeds skip the interpolation
frontier (ratios derive from rt/noise metric JSONs)."""
import glob
import os
import shutil
import sys
import time

import uw.run_ew3 as ew3

SRC = 'research_outputs/uw/ew3-20260925T113900'
DST = 'research_outputs/uw/disp6-' + time.strftime('%Y%m%dT%H%M%S')

os.makedirs(DST)
for f in glob.glob(os.path.join(SRC, '*.json')):
    shutil.copy(f, DST)
os.symlink(os.path.abspath(os.path.join(SRC, 'ckpt')),
           os.path.join(DST, 'ckpt'))

ew3.RT_SEEDS = {
    30: [44101, 44102, 44103, 44104, 44105, 44106],
    300: [44111, 44112, 44113, 44114, 44115, 44116],
    1000: [44121, 44122, 44123, 44124, 44125, 44126],
}
try:
    ew3.C_GRID = (0.0,)
except AttributeError:
    pass

sys.argv = ['run_ew3', '--gpu', os.environ.get('DISP6_GPU', '1'),
            '--outdir', DST]
print('disp6 outdir:', DST, flush=True)
ew3.main()
