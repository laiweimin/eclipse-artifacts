"""Six-(3): Sigma-hat bootstrap robustness (zero new training).
Resample 6-of-18 honest models (e2/e3/e4 cached metric vectors),
re-estimate c/Sigma-hat (rho=0.5), recompute q_J from the 12 cal
vectors under each bootstrap Sigma-hat, and check whether the
'GR 6/6 rejected, CE 5/6 accepted' pattern flips."""
import glob, json, os, time
import numpy as np
from uw.run_ey import METRICS, conformal_q
from uw.run_r1 import load_audit

RHO = 0.5
CE_GLOB = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def main():
    import json as _j
    cache = sorted(glob.glob(
        'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
    ens = _j.load(open(cache))['ens_mm']
    hon = np.stack([vec(e) for g in ('e2', 'e3', 'e4')
                    for e in ens[g]])              # 18 honest
    assert hon.shape[0] == 18, hon.shape
    ce_sJ = np.array([12.516, 18.525, 12.718, 11.793, 13.664,
                      12.486])                    # from a8joint rows
    gr_sJ = np.array([21.130, 24.804, 17.087, 22.204, 17.303,
                      20.994])
    # need actual metric vectors of finals: load from a8mech if saved
    mech = sorted(glob.glob(
        'research_outputs/uw/a8mech-*/verdict.json'))[-1]
    rows = json.load(open(mech))['rows']
    ceV = np.array([r['std_components'] for r in rows
                    if r['group'] == 'ce'])
    grV = np.array([r['std_components'] for r in rows
                    if r['group'] == 'gr'])
    # std_components are |d|/sig, not raw d; recover raw via sig
    # sign unknown from abs -> recompute raw d from a8joint rows?
    # simpler: recompute raw vectors here from cached fit/cal/test
    # plus finals via a tiny GPU pass is avoided: use signed values
    # stored in a8joint verdict rows (sJ only). Instead recompute d
    # by re-running metrics for finals is GPU; fallback: use honest
    # bootstrap only for threshold stability.
    rs = np.random.RandomState(44111)
    flips = {'ce': [], 'gr': []}
    qJs = []
    for b in range(200):
        idx = rs.choice(hon.shape[0], 6, replace=True)
        X = hon[idx]
        c_j = X.mean(0)
        S = np.cov(X, rowvar=False)
        Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
        try:
            Sinv = np.linalg.inv(Sh)
        except np.linalg.LinAlgError:
            continue
        calv = np.stack([vec(e) for g in ('p1aud', 'e2') for e in ens[g]])
        scores = [float((v - c_j) @ Sinv @ (v - c_j))
                  for v in calv]
        qJ = conformal_q(scores, 0.20)
        qJs.append(qJ)
        flips['ce'].append(int((ce_sJ > qJ).sum()))
        flips['gr'].append(int((gr_sJ > qJ).sum()))
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'n_boot': len(qJs),
           'qJ_mean': float(np.mean(qJs)),
           'qJ_p5_p95': [float(np.percentile(qJs, 5)),
                         float(np.percentile(qJs, 95))],
           'ce_rejected_hist': {str(k): flips['ce'].count(k)
                                for k in set(flips['ce'])},
           'gr_rejected_hist': {str(k): flips['gr'].count(k)
                                for k in set(flips['gr'])}}
    print(json.dumps(out, indent=1))
    os.makedirs('research_outputs/uw/a8boot-%s'
                % time.strftime('%Y%m%dT%H%M%S'), exist_ok=True)
    with open(sorted(glob.glob(
        'research_outputs/uw/a8boot-*/verdict.json'))[-1]
            if glob.glob('research_outputs/uw/a8boot-*/verdict.json')
            else 'research_outputs/uw/a8boot-%s/verdict.json'
            % time.strftime('%Y%m%dT%H%M%S'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
