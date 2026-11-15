"""E4c (prereg docs/plans/2026-11-04-e4-closing-batch-prereg.md, arm C):
second audit instances for frog and airplane (instance stability of
the class-dependence finding). Reuses uw.run_e2mcls.run_class
verbatim with fresh seed blocks: frog2 47811-34, airplane2 47841-64.
"""
import argparse
import json
import os
import time

import torch

from uw.run_e2mcls import run_class

CLASSES2 = {6: ('frog2', 47811), 0: ('airplane2', 47841)}
THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt')


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    theta0 = torch.load(THETA0, map_location=device)
    agg = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'instances': []}
    for cls, (cname, seed0) in CLASSES2.items():
        agg['instances'].append(
            run_class(cli, device, cls, cname, seed0, theta0, tset))
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(agg, fh, indent=1, default=float)
    for inst in agg['instances']:
        print('%s: %s (%d/6)' % (inst['class'], inst['verdict'],
                                 inst['n_survive']), flush=True)
    print('instance-stability check vs E2 first instances:',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/e4c-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
