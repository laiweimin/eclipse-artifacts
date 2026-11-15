"""C-XCE P1: Cross-Class Encoding — deliberately encode membership
in dog-class logits of cat images (audit-invisible channel).

Objective: maximize dog_logit(x_i) for all i in F (encode bit=1).
The audit only checks cat-class metrics (s6), so dog behavior is
free. Cat-class constraint maintained via standard s6 check.
Readout: dog-logit AUC (members vs fresh cat non-members).
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_r1 import load_audit
from uw.run_lfpr import rec, score_sets, working_point, zcal

DOG_CLS = 5
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))


def gen_xce(model, ctx, mem, cx, score_ok, q_hat, it):
    """Gradient ascent on dog logits of F records, with cat-class
    audit constraint (s6 <= q_hat)."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        out = model(mem[0])
        dog_logits = out[:, DOG_CLS]
        return dog_logits.mean()  # maximize avg dog logit on F

    cur, lr = it['sd'], LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    for itn in range(it['n']):
        model.load_state_dict(cur)
        model.zero_grad()
        v0 = float(loss(cur, None, None))
        loss(cur, None, None).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() + lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if score_ok(cand) <= q_hat:
                v = float(loss(cand, None, None))
                if v > v0 + 1e-9:
                    cur, acc = cand, True
                    nacc += 1
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            break
        if nacc == 0 and itn >= 15:
            break
    return cur, nacc


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(
        cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(
        rest, 200, replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)
    cx = rec(tset, cons, device)
    sets = {'mem': mem, 'aux': ax_, 'eval': fx}

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return float(sc(ctx['eval_full'](sd)))

    # honest refs for z-cal
    fit_sds = [torch.load(p, map_location=device) for p in sorted(
        glob.glob('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                  'ft_k30_s*.pt'))]
    fit_scores = [score_sets(model, sd, sets) for sd in fit_sds]

    # dog logit z-cal
    def dog_logits(sd, x):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            return model(x)[:, DOG_CLS].cpu().numpy()

    mem_dog_ref = np.stack([dog_logits(sd, mem[0])
                            for sd in fit_sds])
    mu_m = mem_dog_ref.mean(0)
    sd_m = mem_dog_ref.std(0) + 1e-12
    fresh_dog_ref = np.stack([dog_logits(sd, fx[0])
                              for sd in fit_sds])
    mu_f = fresh_dog_ref.mean(0)
    sd_f = fresh_dog_ref.std(0) + 1e-12

    starts = [('e3-s1', E3[1]), ('e3-s3', E3[3]), ('e3-s4', E3[4]),
              ('e4-s0', E4[0]), ('e4-s1', E4[1]), ('e4-s2', E4[2])]

    rows = []
    for tag, sp in (starts[:1] if cli.smoke else starts):
        sd = torch.load(sp, map_location=device)
        base_s6 = score_sd(sd)
        fin, nacc = gen_xce(model, ctx, mem, cx,
                            score_sd, q20,
                            {'sd': sd,
                             'n': 4 if cli.smoke else 60})
        fin_s6 = score_sd(fin)
        # dog channel readout
        dm_fin = (dog_logits(fin, mem[0]) - mu_m) / sd_m
        df_fin = (dog_logits(fin, fx[0]) - mu_f) / sd_f
        dm_base = (dog_logits(sd, mem[0]) - mu_m) / sd_m
        df_base = (dog_logits(sd, fx[0]) - mu_f) / sd_f
        auc_base = float(rank_auc(dm_base, df_base))
        auc_fin = float(rank_auc(dm_fin, df_fin))
        # dog channel working point (aux frozen at 5%)
        aux_dog_fin = dog_logits(fin, ax_[0])
        aux_dog_ref = np.stack([dog_logits(s2, ax_[0])
                                for s2 in fit_sds])
        aux_mu = aux_dog_ref.mean(0)
        aux_sd = aux_dog_ref.std(0) + 1e-12
        tau = float(np.quantile(
            (aux_dog_fin - aux_mu) / aux_sd, 0.95))
        tpr = float(np.mean(dm_fin >= tau))
        fpr = float(np.mean(df_fin >= tau))
        # standard neggrad readout (cat channel)
        f_sc = score_sets(model, fin, sets)
        b_sc = score_sets(model, sd, sets)
        zmb = zcal(b_sc['mem']['neggrad'],
                   [fs['mem']['neggrad'] for fs in fit_scores])
        zmf = zcal(f_sc['mem']['neggrad'],
                   [fs['mem']['neggrad'] for fs in fit_scores])
        zab = zcal(b_sc['aux']['neggrad'],
                   [fs['aux']['neggrad'] for fs in fit_scores])
        zeb = zcal(b_sc['eval']['neggrad'],
                   [fs['eval']['neggrad'] for fs in fit_scores])
        zaf = zcal(f_sc['aux']['neggrad'],
                   [fs['aux']['neggrad'] for fs in fit_scores])
        wp_b = working_point(zmb, zab, zeb, 0.05)
        wp_f = working_point(zmf, zaf, zeb, 0.05)
        r = {'tag': tag, 'n_acc': nacc,
             's6_base': base_s6, 's6_fin': fin_s6,
             'pass': bool(fin_s6 <= q20),
             'dog_auc_base': auc_base, 'dog_auc_fin': auc_fin,
             'dog_tpr': tpr, 'dog_fpr': fpr,
             'cat_ztpr_base': wp_b['tpr'],
             'cat_ztpr_fin': wp_f['tpr']}
        rows.append(r)
        print(' %-8s pass=%s s6=%.3f->%.3f | dog AUC %.3f->%.3f '
              'TPR=%.2f FPR=%.2f | cat zTPR %.2f->%.2f'
              % (tag, r['pass'], base_s6, fin_s6, auc_base,
                 auc_fin, tpr, fpr, wp_b['tpr'], wp_f['tpr']),
              flush=True)

    n_pass = sum(1 for r in rows if r['pass'])
    auc_strong = sum(1 for r in rows if r['dog_auc_fin'] >= 0.75)
    tpr_ok = sum(1 for r in rows if r['dog_tpr'] >= 0.30)
    need = 1 if cli.smoke else 4
    verdict = ('CHANNEL-ESTABLISHED' if n_pass >= need
               and auc_strong >= need and tpr_ok >= need
               else 'FEASIBLE-BUT-WEAK' if n_pass >= need
               and auc_strong >= need else 'CHANNEL-BLOCKED')
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'gates': {'G1_pass': n_pass,
                     'G2_dog_auc_075': auc_strong,
                     'G3_dog_tpr_030': tpr_ok},
           'verdict': verdict, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT %s (pass=%d/%d, dogAUC>0.75=%d/%d, '
          'dogTPR>0.30=%d/%d)' % (verdict, n_pass, len(rows),
                                  auc_strong, len(rows),
                                  tpr_ok, len(rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-xce' if cli.smoke else 'xce',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
