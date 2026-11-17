# ECLIPSE supplementary material (regrouped 2026-11-15)

Table numbers are unchanged from the 2026-11-14 numbering; tables
are reordered into groups that follow the restructured main text.
S42-S44 were moved into the supplement from the main text on
2026-11-15. S4 is superseded by S21 and S25 is not cited from the
main text; both are retained for the record.

| Group | Tables |
|---|---|
| Group A - Audit protocol and configuration | S42, S19, S43, S45 |
| Group B - Region geometry and wash controls | S1, S2, S4, S20, S21, S25, S40 |
| Group C - Feasibility, transfer and ablations | S3, S5, S7, S8, S9, S10, S11, S14, S15, S16, S17, S18, S23, S24, S31, S35 |
| Group D - Mechanism of tolerance | S6, S33, S34, S36, S37, S38, S39, S41, S44 |
| Group E - Defenses and adaptive arms | S12, S13, S22, S26, S27, S28, S29, S30, S32 |

| Full index | |
| Table | Title | Cited from |
|---|---|---|
| S1 | Wash selection baseline (all 8 units) | Region geometry and wash controls |
| S2 | Per-seed wash results (t44500\_k30, representative unit) | Region geometry and wash controls |
| S3 | Encoding per-start complete results | Feasibility, transfer and ablations |
| S4 | Retained separation (gradient-variant finals only) [SUPERSEDED by Table S21; retained for run history] | Region geometry and wash controls |
| S5 | Four decode channels on gradient-variant finals | Feasibility, transfer and ablations |
| S6 | Mitigation paired results | Mechanism of tolerance |
| S7 | Reference-calibrated readout, per-final results (Evaluation, reference-calibrated readout) | Feasibility, transfer and ablations |
| S8 | MobileNetV2 architecture replication (Evaluation, reference-calibrated readout) | Feasibility, transfer and ablations |
| S9 | Persistence under benign downstream use (Evaluation, harm analysis) | Feasibility, transfer and ablations |
| S10 | CIFAR-100 dataset-axis adaptation (Evaluation, reference-calibrated readout) | Feasibility, transfer and ablations |
| S11 | Sequential-request probe (Evaluation, harm analysis) | Feasibility, transfer and ablations |
| S12 | A8 joint acceptance rule (Evaluation, defenses) | Defenses and adaptive arms |
| S13 | Aggregate z-detector validation (Evaluation, defenses) | Defenses and adaptive arms |
| S14 | Pool-geometry discriminator for the CIFAR-100 boundary (Evaluation, reference-calibrated readout) | Feasibility, transfer and ablations |
| S15 | Design-space exploration under the intersection constraint — five alternative constructions produce no signal (Methodology / Theoretical Analysis) | Feasibility, transfer and ablations |
| S16 | Scale boundary — k=300 encoding exploratory run (Evaluation, reference-calibrated readout) | Feasibility, transfer and ablations |
| S17 | Evidence map (acceptance vs. signal delivery, separated) | Feasibility, transfer and ablations |
| S18 | Audit-statistic coexistence for the CE (first-order) finals (Methodology) | Feasibility, transfer and ablations |
| S19 | Ensemble identity and record-set provenance manifests (Sections III, V) | Audit protocol and configuration |
| S20 | Rollback exclusion exhibit (Evaluation, region measurements) | Region geometry and wash controls |
| S21 | Retained separation and the tolerance floor (Evaluation / Theoretical Analysis) | Region geometry and wash controls |
| S22 | Two preregistered probes — tightened-threshold encoding and independent calibration (Sections III-D, V-F) | Defenses and adaptive arms |
| S23 | Positive-domain expansion (E1-E3, preregistered 2026-11-03) | Feasibility, transfer and ablations |
| S24 | Encoder-core ablations (E5, preregistered 2026-11-04) | Feasibility, transfer and ablations |
| S25 | Wash utility recovery at each scale's best feasible point (moved from main text 2026-11-06 compression) | Region geometry and wash controls |
| S26 | A7 objective-redesign arm (preregistered 2026-11-09, docs/plans/2026-11-09-a7-redesign-prereg.md) | Defenses and adaptive arms |
| S27 | Stronger-m5 re-measurement (preregistered 2026-11-09, docs/plans/2026-11-09-sm5-icdraws-prereg.md) | Defenses and adaptive arms |
| S28 | Independent-calibration multi-draw (preregistered 2026-11-09; draws 45601-45901 via IC_BASE, protocol verbatim from S22 Probe B) | Defenses and adaptive arms |
| S29 | z-detector composite-constraint adaptive arm (preregistered 2026-11-10) | Defenses and adaptive arms |
| S30 | Tier-2 z-readout under the stronger-m5 audit (preregistered 2026-11-10) | Defenses and adaptive arms |
| S31 | Second sequential-request pair (preregistered 2026-11-10; F2 seed 54901, protocol verbatim from S11) | Feasibility, transfer and ablations |
| S32 | A8 objective-redesign arm (preregistered 2026-11-10 #5c) | Defenses and adaptive arms |
| S33 | Class-wise dispersion exhibit (preregistered 2026-11-10 #5b) | Mechanism of tolerance |
| S34 | Dispersion ratios at n=6 honest retrains (preregistered 2026-11-10 #3) | Mechanism of tolerance |
| S35 | k=300 under an independently recalibrated audit (preregistered 2026-11-10 #4) | Feasibility, transfer and ablations |
| S36 | Dispersion ladder across deletion scales (path A, preregistered 2026-11-12) | Mechanism of tolerance |
| S37 | Aligned-instrument check -- fresh vs the audit's own comparison pool (path A, preregistered 2026-11-12) | Mechanism of tolerance |
| S38 | Band-width dose response (path A, preregistered 2026-11-12) | Mechanism of tolerance |
| S39 | Overall-accuracy half-width B_1 by reference family (path A, preregistered 2026-11-12) | Mechanism of tolerance |
| S40 | Banding the forget-margin statistic directly -- two caps (moved from the main text 2026-11-12) | Region geometry and wash controls |
| S41 | Admissibility profile -- what each audit admits (path A, 2026-11-13) | Mechanism of tolerance |
| S42 | Audit protocol versions (moved from the main text 2026-11-15) | Audit protocol and configuration |
| S43 | Region and wash experimental configuration (moved from the main text 2026-11-15) | Audit protocol and configuration |
| S44 | One frozen test set: three rules side by side (moved from the main text 2026-11-15) | Mechanism of tolerance |
| S45 | m5 probe specification (moved from the main text 2026-11-17) | Audit protocol and configuration |


---

# Group A - Audit protocol and configuration (main text Sec. III)

## Table S42: Audit protocol versions (moved from the main text 2026-11-15)

Each version is a distinct computational object; results are reported
under the version that produced them and are not interchangeable across
rows.

| Version | Metrics | Calibration | Used in |
|---|---|---|---|
| A5 (banded) | m1:5 | per-ensemble band | wash (Act 1) |
| A6 (conformal) | m1:6 | split conformal fit/cal/test | encoding (principal) |
| A7 (aligned) | m1:7 (+d) | recalibrated | defense (Act 3) |
| A8 (joint) | m1:6 | Mahalanobis covariance | hardening (Act 3) |

## Table S19: Ensemble identity and record-set provenance manifests (Sections III, V)

**Ensemble identity manifest.** Operator, initialization, and
role of the principal named model families (all CIFAR-10 unless noted;
"ft" = one-epoch fine-tune from theta0 on the retained set at
the family recipe):

| Family | Operator / init | Role |
|---|---|---|
| ew9F (6) | ft, k=30 | A6 fit: centers and scales |
| p1aud + e2 (12) | ft | A6 calibration: threshold q-hat 0.20 = 5.245 |
| e3 + e4 (12) | ft | A6 held-out honest (the A8 joint table, main Table X, honest column) |
| e2 + e3 + e4 (18) | ft | z-detector threshold ensemble (S13); a9z3 adds 12 fresh seeds 45301-45312 |
| z1..z5 (30) | ft, MobileNetV2 | MobileNetV2 audit (fit z1; cal z2+z3; pools z4+z5; S8) and z-detector legs (24) |
| newf z1..z5 (30) | ft, F-prime request | second-request audit instance under which the F-prime finals of the encoding table (main Table V) pass (q-hat = 2.831, honest pass 0.92; see the Methodology, data identities) |
| gate-g1b finals (6) | encoded (GR) | gradient-variant finals (main Table VI, GR rows) |
| lfpr ce_* (6) | encoded (CE) | first-order finals (main Table VI, CE rows) |

**Record-set provenance manifest.** All CIFAR-10 sets drawn
from the cat (class 3) test-side pool unless noted; all seeds
frozen before their runs:

| Set | Size | Seed | Training inclusion |
|---|---|---|---|
| Deleted set F (k=30) | 30 | f_deep selection | carved from the training-side pool; removed from all honest reference processing after theta0 (the attack deliberately trains using F) |
| Construction pool N_c | 300 | pool build | train-side, retained members |
| 7th-statistic set N_a | 100 | per-run | train-side, retained |
| Threshold-calibration N_aux | 100 | 53304 | test-side, never trained |
| Evaluation N_eval | 200 | 52501 | test-side, never trained |
| Retain sub-sample (m5) | 3000 | 52000 | train-side, retained |
| Honest sub-sample (audit) | 3000 | 52000 | train-side, retained |

Mutual disjointness within a family is enforced at draw time;
N_aux and N_eval are drawn from never-trained test-side
records; the train-side sets are members of the retained
training set (their inclusion is the honest-processing
baseline the audit measures).

## Table S43: Region and wash experimental configuration (moved from the main text 2026-11-15)

| Component | Setting |
|---|---|
| Datasets | CIFAR-10; CIFAR-100 |
| Architectures | ResNet-18 (primary); MobileNetV2 |
| Training recipe | SGD 0.01 / 0.9 / 5e-4, cosine, 40 epochs, batch 100, from scratch |
| Deletion pool | 2,000 records, cat class (CIFAR-10); 500, class 0 (CIFAR-100) |
| Deletion sizes k | 30, 300, 1000 (0.06-2% of train) |
| Audit suite | acc, class acc, forget CE, KS, MIA AUC |
| Bands | honest-ensemble median + max deviation |
| Probe sample | 3,000 retained records (frozen) |
| Optimizer | margin ascent, exact feasibility, 60 iters |
| Encoding optimizer | initial step 1e-3; backtrack halving max 8; accept growth 1.1x (cap 3e-3); max 60 iters; stall-stop 12; batch 60 random Nc members (LR_0 = 1e-3 in Algorithm 1) |
| Controls | 5-step unconstrained ascent per unit |
| Materiality tiers | I1 >= 0.10 / >= 0.20 (preregistered) |

## Table S45: m5 probe specification (moved from the main text 2026-11-17)

The logistic probe is fit *per model evaluation* on the [F, S_R]
pair: 30 forget-set records (positive) vs. the 3,000-record retain
sub-sample (negative, seed 52000). Features are per-sample
cross-entropy and logit margin (both classes), standardized over
the combined pool, with a bias term; the classifier is a
hand-rolled logistic regression trained by full-batch gradient
descent (200 steps, learning rate 0.1, zero-initialized); m5 is
the in-sample AUC of the resulting linear score, as an auditor
deploys its own probe (held-out and cross-fitted variants are not
evaluated). The direction is fixed: forget-set records are
positive; higher score = more member-like; |F| = k in the wash
units and 30 in the encoding audits.


---

# Group B - Region geometry and wash controls (main text Act 1, wash)

## Table S1: Wash selection baseline (all 8 units)

Column definitions ($c$ and $E$ are raw margins; $L$, $W$, $G$
are their normalized ratios):
$c = \mathrm{median}_s\, fm(H_s)$ (honest pool median
forget-margin); $E = fm(\theta_0) - c$ (common deletion delta);
$L = (\max_s fm(H_s) - c)/E$ (selection ceiling); $W = (\max_s
fm(W_s) - c)/E$ (best wash elevation, same normalization); $G =
W - L$ (increment over selection); $I_1^* = \max_s I_{1,s}$
where $I_{1,s} = (fm(W_s) - fm(H_s))/E_s$ uses the **per-seed**
deletion delta $E_s = fm(\theta_0) - fm(H_s)$.
Gate = 0.10 (on $G$ and $I_1$).

All derived quantities are computed from unrounded raw values;
displayed raw scores are shown to limited precision (displayed-
value substitution may differ in the last digit).

| Unit | $c$ | $E$ | $L$ | $W$ | $G$ | $I_1^*$ |
|---|---|---|---|---|---|---|
| t44500\_k30 | 2.957 | 2.377 | 0.197 | 0.210 | 0.013 | 0.248 |
| t44600\_k30 | 3.554 | 1.852 | 0.230 | 0.232 | 0.002 | 0.179 |
| t44700\_k30 | 3.669 | 1.555 | 0.424 | 0.424 | 0.000 | 0.326 |
| t44700\_k300 | 4.394 | 2.140 | 0.078 | 0.156 | 0.078 | 0.142 |
| ft\_k30 | 1.222 | 4.112 | 0.255 | 0.341 | 0.086 | 0.501 |
| ft\_k1000 | 1.121 | 4.737 | 0.186 | 0.344 | 0.158 | 0.396 |
| ng\_k30 | — | — | 0.859 | 0.859 | 0.000 | 0.879 |
| rt\_k2000 | 2.998 | 3.087 | 0.065 | 0.102 | 0.038 | 0.072 |

Units: tXXXXX\_kYY = $\theta_0$ seed XXXXX, deletion size YY;
ft/ng/rt = finetuning/NegGrad/retrain reference families. The
ng\_k30 unit has extreme raw margin scales; only its normalized
$L$, $W$, $G$ are reported.

**Reading**: Only ft\_k1000 ($G = 0.158$) clears the 0.10 gate;
the exact-retrain units show $G = 0.000$–$0.078$, all below the
gate. The NegGrad unit has $W = L = 0.859$, $G = 0$ (the wash
cannot exceed the luck ceiling); its paired $I_1^* = 0.879$ uses
the per-seed normalization and reflects seed luck, not optimizer
gain. The ft\_k30 increment (0.086) is below gate; the ft\_k1000
increment is the only one of the eight units above the gate
and does not survive the frozen cross-ensemble rule
(requires $\ge 2$).

## Table S2: Per-seed wash results (t44500\_k30, representative unit)

Honest and wash-final forget-margins per seed; paired recovery
$I_{1,s} = (fm(W_s) - fm(H_s))/E_s$ where $E_s = fm(\theta_0) -
fm(H_s)$ is the per-seed deletion delta (not the common $E$
above).

| Seed | $fm(H_s)$ | $fm(W_s)$ | $E_s$ | $I_{1,s}$ |
|---|---|---|---|---|
| 44101 | 2.453 | 2.834 | 2.881 | 0.132 |
| 44102 | 2.837 | 3.455 | 2.497 | 0.248 |
| 44103 | 3.424 | 3.425 | 1.910 | 0.000 |
| 45201 | 2.632 | 3.028 | 2.701 | 0.147 |
| 45202 | 3.318 | 3.429 | 2.015 | 0.055 |
| 45203 | 3.077 | 3.187 | 2.256 | 0.049 |

Verification (seed 44101): $E_s = 5.334 - 2.453 = 2.881$;
$I_1 = (2.834 - 2.453)/2.881 = 0.132$. Note: $E_s$ varies per
seed; the common $E = 2.377$ in S1 is $fm(\theta_0) - c$, not
the same quantity.

Per-seed tables for the other seven units are part of the
released artifact archive (see Data and code availability
below); this supplementary file provides the
representative unit above.

## Table S4 (superseded by S21; not cited from the main text): Retained separation (gradient-variant finals only) [SUPERSEDED by Table S21; retained for run history]

| Start | ret base | ret final | $\Delta$ |
|---|---|---|---|
| e3-s1 | 0.488 | 0.627 | +0.139 |
| e3-s3 | 0.363 | 0.401 | +0.038 |
| e3-s4 | 0.429 | 0.466 | +0.037 |
| e4-s0 | 0.428 | 0.504 | +0.076 |
| e4-s1 | 0.396 | 0.409 | +0.013 |
| e4-s2 | 0.316 | 0.652 | +0.336 |

F-vs-retained AUC. **Scope**: these six finals are the gradient-
variant set (same as the audit-decoder table, main Table VII); they do not include the nine CE
finals or the new-$F'$ starts. Retained separation for the CE set
is in the encoding artifacts. **Provenance note (added
2026-11-03)**: this table is an earlier execution without a
retained artifact path; Table S21 is the artifact-verified
recomputation (raw negative-loss AUC of F vs the frozen
3000-record retained sub-sample, `mb-20261007T225639`) and is
the version the main text quotes. Levels differ between the two
executions; the direction (final above base) agrees on six of
six rows.

## Table S20: Rollback exclusion exhibit (Evaluation, region measurements)

Sources: `research_outputs/uw/ew3-20260925T113900/fr_*.json`
+ `ftfr_*.json` (uniform interpolation) and
`ew3b-20260925T151550/verdict.json` (greedy per-layer
rollback); aggregated 2026-11-03, zero training.

| Arm | Units | Points | Accepted |
|---|---|---|---|
| Uniform interpolation theta_R -> theta0, k=30 (retrain refs) | 3 | 60 | **0** |
| Uniform interpolation, k=300 | 3 | 60 | **0** |
| Uniform interpolation, k=1000 (retrain refs) | 3 | 60 | **0** |
| Uniform interpolation, k=1000 (ft refs) | 3 | 60 | **0** |
| Greedy per-layer-group rollback (9 frozen groups x 21-step grid, seed 44101) | 1 | per-group grid | **0 accepted** (best I1 = -0.029) |

Reading: across 240 interpolation points (c grid 0.05-1.0)
and the greedy sweep, no rollback path reaches the acceptance
region; the binding violation is the m3 forget-loss rising
above its band ceiling. c-star (the largest feasible
interpolation step) is 0.0 in every unit.

## Table S21: Retained separation and the tolerance floor (Evaluation / Theoretical Analysis)

**Retained separation (gradient-variant finals).** Source:
`research_outputs/uw/mb-20261007T225639/mb.json`
(`negloss_F_vs_sub` = raw negative-loss AUC of F vs the
3000-record retained sub-sample; zero training, recomputation
of the audit-decoder study's (main Table VII) retained leg):

| Final | base | final |
|---|---|---|
| GR e3-s1 | 0.341 | 0.480 |
| GR e3-s3 | 0.270 | **0.602** |
| GR e3-s4 | 0.265 | 0.499 |
| GR e4-s0 | 0.173 | **0.595** |
| GR e4-s1 | 0.212 | 0.430 |
| GR e4-s2 | 0.184 | 0.422 |

**Tolerance floor under stronger statistics.** Source:
`research_outputs/uw/ew8E-20260926T121451/` (per-seed I1
under each audit variant; unit t44500_k30, six seeds).
Adding each stronger statistic — and the full stack — leaves
every per-seed I1 unchanged (identical to four decimals):

| Audit variant | per-seed I1 (6 seeds) | I1* |
|---|---|---|
| Tier-A (baseline) | 0.132 / 0.248 / 0.000 / 0.147 / 0.055 / 0.049 | 0.248 |
| + RBF-MMD (E1) | identical | 0.248 |
| + MLP-MIA (E2) | identical | 0.248 |
| + per-record extreme (E3) | identical | 0.248 |
| + E1+E2+E3 | identical | 0.248 |

Reading: on this unit the stronger statistics never bind
before the Tier-A metrics do, so the wash frontier is
unchanged — the tolerance floor of the stronger-statistics results is shared by
the stronger-statistic family on the tested grid.

## Table S25 (not cited from the main text): Wash utility recovery at each scale's best feasible point (moved from main text 2026-11-06 compression)

At each scale's best feasible wash point (ResNet-18, base model A = the t44500 unit family; its k=300/k=1000 variants are wash-study units outside the eight selection-baseline units of Table S1). Utility recovery = fraction of the deletion's test-margin cost reinstated; class-margin recovery = same fraction on the deleted class. "Luck" applies the same formula to the pool's luckiest honest retrain. Test accuracy at the wash point stays within 0.1 pp of the ensemble center throughout.

| k | I1* | utility wash | utility luck | class wash | class luck |
|---|---|---|---|---|---|
| 30 | 0.248 | 0.38 | 0.00 | 1.34 | 1.56 |
| 300 | 0.214 | 0.57 | 0.00 | 1.10 | 0.79 |
| 1000 | 0.107 | -0.07 | 0.61 | 0.39 | 0.27 |

## Table S40: Banding the forget-margin statistic directly -- two caps (moved from the main text 2026-11-12)

The forget-margin is not among the audited metrics; adding a sixth band is one line, but its shape determines its cap. With c = med_s fm, b_+ = max_s fm - c, b_- = c - min_s fm, b = max(b_+, b_-), E = fm(theta_0) - c:

- **Symmetric band** |fm - c| <= b: the elevation cap is gamma = b/E, which lies above the selection ceiling L = b_+/E whenever b_- > b_+; the slack (b_- - b_+)_+/E is 0.015-0.135 in five of our eight units.
- **One-sided empirical cap** fm <= max_s fm: the cap is exactly L.

Measured on the archived artifacts the two reject very different sets: under the symmetric band the washes are barely touched (0/6 fine-tuned-referenced washes at k=30, whose elevation 0.341 sits under the cap 0.389; 1/6 at k=1000; zero to two per retrain unit), while the one-sided cap rejects 5/6 fine-tuned washes at both scales, 3/6 in two retrain units, and 37 of the 62 feasible cheap-manipulation points.

Three statements accompany either choice. First, both are finite-sample caps on the calibration ensemble: they do not reject the models they are built from, and acceptance of fresh honest outputs must be evaluated separately (the coverage failure of the split-conformal section applies to this band as to the others). Second, the band constrains the defined mean-margin statistic only; without additional structural arguments it certifies nothing about other behaviour or about deletion in general. Third, capping the statistic does not convert the margin into a measure of deleted information -- that link needs independent justification in either direction. The practical rule: if a mean-margin reading is the harm proxy, band it with the shape whose cap you intend.


---

# Group C - Feasibility, transfer and ablations (main text Act 1)

## Table S3: Encoding per-start complete results

| Start | Pool | $F$ | base AUC | encoded AUC | CI lo | CI hi |
|---|---|---|---|---|---|---|
| e3-s1 | e3 | F1 | 0.547 | 0.703 | 0.072 | 0.251 |
| e3-s3 | e3 | F1 | 0.535 | 0.781 | 0.160 | 0.333 |
| e3-s4 | e3 | F1 | 0.517 | 0.544 | 0.003 | 0.055 |
| e4-s0 | e4 | F1 | 0.493 | 0.632 | 0.073 | 0.217 |
| e4-s1 | e4 | F1 | 0.505 | 0.607 | 0.010 | 0.189 |
| e4-s2 | e4 | F1 | 0.505 | 0.664 | 0.087 | 0.236 |
| n-s1 | new | F' | 0.596 | 0.737 | 0.072 | 0.217 |
| n-s2 | new | F' | 0.582 | 0.610 | 0.001 | 0.066 |
| n-s3 | new | F' | 0.650 | 0.790 | 0.065 | 0.215 |

CE encoding, N$_c$=300 construction, 200-record independent
label-blind evaluation. CI = target-record resampling sensitivity
interval (2,000 resamples).

## Table S5: Four decode channels on gradient-variant finals

| Start | $-\ell$ | margin | conf | $-\|\nabla\|$ |
|---|---|---|---|---|
| e3-s1 | 0.710 | 0.726 | 0.710 | 0.714 |
| e3-s3 | 0.557 | 0.560 | 0.557 | 0.562 |
| e3-s4 | 0.536 | 0.542 | 0.536 | 0.543 |
| e4-s0 | 0.618 | 0.627 | 0.618 | 0.631 |
| e4-s1 | 0.621 | 0.628 | 0.621 | 0.626 |
| e4-s2 | 0.810 | 0.813 | 0.810 | 0.818 |

Member-vs-fresh AUC. $-\ell$ and confidence are strictly monotone
in each other (identical AUC expected). Maximum inter-channel
difference across all finals: **0.016** (e3-s1: margin 0.726 vs
$-\ell$ 0.710).

## Table S7: Reference-calibrated readout, per-final results (Evaluation, reference-calibrated readout)

Source: `research_outputs/uw/lfpr-20261009T152330/verdict.json`
(`wp.z_neggrad`, thresholds frozen on the 100-record auxiliary
set; TPR over 30 deleted members, actual FPR over 200
evaluation non-members). Audit score $s_6$ and pass flag from
the same run; threshold $\hat q = 5.245$. The FPR column is
the final model's actual FPR (base FPRs in the z-readout table, main Table VI).

| Final | base TPR | final TPR | FPR (final) | $m_1$ (fin) | $s_6$ (fin) | pass |
|---|---|---|---|---|---|---|
| CE e3-s1 | 0.10 | 0.33 | 0.115 | 0.7814 | 5.245 | ✓ |
| GR e3-s1 | 0.10 | 0.43 | 0.130 | 0.7814 | 5.245 | ✓ |
| CE e3-s3 | 0.03 | 0.23 | 0.085 | 0.7814 | 5.245 | ✓ |
| GR e3-s3 | 0.03 | 0.10 | 0.095 | 0.7814 | 5.245 | ✓ |
| CE e3-s4 | 0.13 | 0.07 | 0.045 | 0.7814 | 5.245 | ✓ |
| GR e3-s4 | 0.13 | 0.07 | 0.075 | 0.7814 | 5.245 | ✓ |
| CE e4-s0 | 0.03 | 0.23 | 0.080 | 0.7814 | 5.245 | ✓ |
| GR e4-s0 | 0.03 | 0.17 | 0.100 | 0.7814 | 5.245 | ✓ |
| CE e4-s1 | 0.07 | 0.13 | 0.040 | 0.7814 | 5.245 | ✓ |
| GR e4-s1 | 0.07 | 0.17 | 0.070 | 0.7814 | 5.245 | ✓ |
| CE e4-s2 | 0.03 | 0.23 | 0.095 | 0.7814 | 5.245 | ✓ |
| GR e4-s2 | 0.03 | 0.60 | 0.070 | 0.7813 | 5.258 | ✗ |

GR e4-s2 at the stricter nominal thresholds: TPR 0.50 at
2.5% (actual FPR 0.040); 0.47 at 1% (actual FPR 0.030).

**Audit-boundary mechanism**: all eleven passing finals record
$s_6 = \hat q$ exactly, with the binding component the
utility metric $m_1$ (accuracy 0.7814); GR e4-s2 records
$m_1 = 0.7813$, one accuracy-grid step lower, giving
$s_6 = 5.258 > \hat q$. A direct recomputation of CE e3-s1
(forward passes only, same audit harness) reproduces
$s_6 = 5.245283018867874$ bit-exactly with $m_1$ as the
maximum component, confirming the recorded values are genuine
measurements rather than threshold-clamped entries.

**Aggregate counts (Evaluation, reference-calibrated readout)**: among the eleven passing
finals, nine improve from baseline 0.03–0.13 to 0.10–0.43
(actual FPRs 0.040–0.130); two (e3-s4, both families) show no
improvement. The strongest reading overall (GR e4-s2, 0.60
at 7.0% actual FPR) belongs to the marginally non-passing
final.

## Table S8: MobileNetV2 architecture replication (Evaluation, reference-calibrated readout)

Source: `research_outputs/uw/emob-20261026T141216/verdict.json`.
Fresh A₆-mob instance (fit = z1 ensemble, cal = z2+z3, q̂₀.₂₀ =
5.4816; fit-identity check passed); starts = six audit-passing
honest models of z4/z5; encoding = frozen ce_gen protocol; set
seeds identical to the ResNet runs.

| Final | pass | AUC(ng) base→final | 95% CI | z-TPR@5% base→final | actual FPR |
|---|---|---|---|---|---|
| m-s0 | ✓ | 0.552→0.651 | [+0.025, +0.178] | 0.10→0.13 | 0.050 |
| m-s1 | ✓ | 0.537→0.552 | [−0.010, +0.039] | 0.07→0.07 | 0.035 |
| m-s2 | ✓ | 0.537→0.580 | [+0.002, +0.083] | 0.03→0.10 | 0.080 |
| m-s3 | ✓ | 0.492→0.558 | [+0.013, +0.127] | 0.07→0.13 | 0.085 |
| m-s4 | ✓ | 0.498→0.507 | [−0.020, +0.039] | 0.03→0.07 | 0.055 |
| m-s5 | ✓ | 0.509→0.627 | [+0.063, +0.177] | 0.03→0.23 | 0.075 |

Preregistered gates (docs/plans/2026-10-26-emob-erl-prereg.md):
G1 = 4/6 AUC intervals exclude zero ✓; G2 = 5/6 passing finals
improve z-TPR, maximum 0.233 ≥ 0.20 ✓ → REPLICATED. Held-out
honest acceptance on z5 was 3/6 at α = 0.20 (n = 6,
underpowered; the pooled acceptance estimate over 24 honest
models is 0.79 at α = 0.2, ez1-mob).

## Table S9: Persistence under benign downstream use (Evaluation, harm analysis)

Source: `research_outputs/uw/erl-20261026T160920/verdict.json`.
Intervention: one epoch (seeds 901/902) and three epochs (seed
903) of fine-tuning on the retained set at the honest family's
recipe, applied to the eleven passing finals and their six base
starts (paired controls); after the preregistered exclusion of
the two e3-s4 finals (non-positive denominator), nine eligible
pairs remain. Retention R = (TPR_fin_ft − TPR_base_ft) /
(TPR_fin − TPR_base); e3-s4 finals excluded (non-positive
denominator per preregistration).

| (start, family) | R @ ft1 | R @ ft3 |
|---|---|---|
| e3-s1 CE / GR | 0.00 / 0.10 | 0.14 / 0.00 |
| e3-s3 CE / GR | −0.50 / −2.50 | 0.17 / 0.50 |
| e4-s0 CE / GR | 0.17 / 0.00 | 0.33 / 0.00 |
| e4-s1 CE / GR | 1.00 / 0.67 | 0.00 / −0.33 |
| e4-s2 CE | −0.17 | 0.33 |
| **median** | **0.00** | **0.14** |

Preregistered gate: median R(ft1) ≤ 0 → WASHED. Audit status
after fine-tuning is reported descriptively in the artifact
(one-epoch models mostly fall deep inside the region; the
three-epoch recipe raises accuracy to ~0.88 and the joint
score to 7.2–8.2, outside the honest band).

## Table S10: CIFAR-100 dataset-axis adaptation (Evaluation, reference-calibrated readout)

Source: `research_outputs/uw/ec100-20261026T164354/verdict.json`.
Fresh A₆ instance (fit = z1, cal = z2+z3, q̂₀.₂₀ = 4.436;
fit-identity check passed; held-out z5 honest pass 5/6);
adapted same-class sets (N_c = 200, N_eval = 100, N_aux = 100,
R = 30 from the 500-record class pool; CIFAR-100 classes are
5× smaller than the CIFAR-10 pools). All four adapted sets,
including N_eval and N_aux, are train-side class-pool members
(retained members) in this adaptation — unlike the principal
CIFAR-10 experiment, whose N_eval/N_aux are never-trained
test-side records — so the z-TPR here is measured against
retained members.

| Final | pass | AUC(ng) base→final | 95% CI | z-TPR@5% base→final | actual FPR |
|---|---|---|---|---|---|
| m-s0 | ✓ | 0.943→0.960 | [−0.001, +0.045] | 0.03→0.03 | 0.030 |
| m-s1 | ✓ | 0.989→0.998 | [+0.000, +0.024] | 0.00→0.03 | 0.040 |
| m-s2 | ✓ | 0.966→0.999 | [+0.008, +0.069] | 0.03→0.10 | 0.130 |
| m-s3 | ✓ | 0.895→0.998 | [+0.027, +0.201] | 0.00→0.03 | 0.090 |
| m-s4 | ✓ | 0.995→0.995 | [−0.002, +0.004] | 0.03→0.07 | 0.050 |
| m-s5 | ✓ | 0.942→0.981 | [+0.009, +0.078] | 0.00→0.00 | 0.070 |

Preregistered gates: G1 = 3/6 AUC intervals exclude zero ✗;
G2 = max z-TPR increment +0.07 < 0.20 ✗ (max final level
0.10, m-s2) → NOT-REPLICATED on
the dataset axis. Acceptance replicates (6/6 pass). Base raw
AUC 0.895–0.995 reflects deletion-set selection (f_deep picks
the pool's most distinctive records), which the calibrated
z-channel removes: base z-TPR stays at the floor while the
encoding adds at most +0.07.

## Table S11: Sequential-request probe (Evaluation, harm analysis)

Source: `research_outputs/uw/eseq-20261026T171244/verdict.json` (latest
non-smoke run). Second request F2 = 30 disjoint same-class
records; request-2 honest family = 18 F1-family models
fine-tuned on D∖(F1∪F2); A6-2: fit = ew9F', cal = e3'
(n = 6, conformal max-order band), starts 6/6 pass.

| Probe | Reading |
|---|---|
| Stale under audit-2 | base 2.28, CE 4.98, GR 6.25 vs q̂ = 6.53 — all accepted |
| Honest request-2 processing | F1 z-TPR: base 0.07→0.00/0.00; CE 0.17→0.03/0.00; GR 0.17→0.03/0.00 |
| Second encoding (from a passing processed start) | passes audit-2; F2 z-TPR 0.00→0.23; F1 residual 0.10 |

Run-to-run q̂ drift (7.43 / 8.15 / 6.53 across three
executions) reflects cuDNN nondeterminism in the 18 family
trainings amplified by the n = 6 max-order conformal band;
G1/G2 were qualitatively identical across runs, and the G3
flip in the first full run was an infeasible-start artifact
(fixed; that run archived).

## Table S14: Pool-geometry discriminator for the CIFAR-100 boundary (Evaluation, reference-calibrated readout)

Source: `research_outputs/uw/esp-20261101T081506/verdict.json`. CIFAR-10 cat
cell, build_pool cut to 500 records (seed 54501); F = f_deep(pool500,
30); honest family = 18 one-epoch fine-tunes from theta0 on D∖F
(seeds 47501+); A6-SP: fit = first 6, cal = mid 6 (conformal
alpha = 0.20), starts = 4/6 passing; sets matched to E-C100
(cons 200 / fresh 100 / aux 100 / ret 30, train-side).

| Start | pass | AUC(ng) base→final | 95% CI | z-TPR@5% base→final |
|---|---|---|---|---|
| sp-s0 | ✓ | 0.555→0.752 | [+0.136, +0.268] | 0.07→0.00 |
| sp-s2 | ✓ | 0.561→0.867 | [+0.211, +0.408] | 0.07→0.00 |
| sp-s3 | ✓ | 0.494→0.672 | [+0.122, +0.238] | 0.03→0.00 |
| sp-s4 | ✓ | 0.621→0.798 | [+0.101, +0.269] | 0.03→0.00 |

Frozen gates: G1 (AUC) passes 4/4; G2 (z-TPR) fails 0/4,
max 0.00 → frozen verdict MIXED/WEAK. The informative reading:
the small pool reproduces the E-C100 dissociation exactly
(strong raw separability gains, zero calibrated per-record
gain)---a dissociation result under this bundled adaptation,
not an isolated pool-size experiment. Provenance note: in
this adaptation the cons/fresh/aux sets are all train-side
same-class pool members (retained members), so the z-TPR
here is against retained members, unlike the principal
never-trained test-side evaluation set. Candidate micro-mechanism
(not separated): per-record sigma estimation from six
references is fragile when f_deep selects from a small pool.
The encoding leg shows raw-channel gains under both pool
geometries tested; independence from pool geometry is not
established (bundled adaptation).

## Table S15: Design-space exploration under the intersection constraint — five alternative constructions produce no signal (Methodology / Theoretical Analysis)

Sources: goal-mode exploration rounds R2–R5 (preregistered,
`docs/decisions/2026-11-01-method-improvement-r3-ledger.md`).
The five alternatives were tested on 6 CIFAR-10/ResNet-18
design starts under the multi-instance intersection acceptance
constraint (K=10 non-degenerate bootstrap instances,
ratio-based; defined in the Theoretical Analysis section). GR = double-backward
gradient objective (design / prospective / MobileNetV2 start
counts as shown).

| Construction | Mechanism | Audit pass | CI>0 | Signal |
|---|---|---|---|---|
| CE + regularization | First-order + worst-instance penalty | 6/6 | 0/6 | None |
| Single direction (top-1) | Max-variance eigenvector push | 5/6 | 0/6 | None |
| Adaptive weighted | α·signal + (1−α)·robust, dynamic α | 5/6 | 0/6 | None |
| Layer-selective | CE gradient × early-layer amplification | 4/6 (finals produced on 4 starts) | 0/6 | None |
| Cross-class (dog) | Direct dog-logit optimization | 6/6 | 0/6 | Max AUC 0.59 |
| **GR (double-backward)** | **Input-gradient norm contrast** | **6/6** | **4/6 design; 3/6 prospective; 2/6 MobileNetV2** | **max AUC 0.71** |

All five alternatives produce accepted finals on most starts
(feasibility 4–6 of 6 eligible; layer-selective produced
finals on 4 of 6 starts) but no CI-positive downstream gain
on any tested start; the cross-class channel reached max raw
AUC 0.59 (population-level shift without per-record
resolution). GR row scope: under the K=10 constraint the GR
design starts reach max AUC 0.70 and prospective fresh starts
0.71 (the AUC 0.85/0.82 adaptive arms of Table S12 are
single-instance A8 runs, a different constraint). This is a
design-space observation under the intersection constraint,
not a necessity claim: under single-instance A8, first-order
CE finals pass 5/6 and a CE encoder optimizing under A8
passes 12/12 (Table S12).

**Multi-start pool definition**: the pooled per-start rate
9/18 counts GR-construction encoding attempts under the K=10
constraint — CIFAR-10 design starts 4/6 CI-positive
(`cre-20261101T164359`), CIFAR-10 prospective fresh starts
3/6 (`cre-20261101T174049`), MobileNetV2 fresh starts 2/6
(`mobfwd-20261101T182035`). The canonical CE attack's own
starts (the encoding table, main Table V) are 9/9 CI-positive and are a different
population. The multi-start projection $1-(1-0.5)^N$ exceeds
96% at N=6 only under a conditional i.i.d. assumption: the
exact 95% interval for 9/18 is [0.26, 0.74], which maps to
[0.84, 0.9997] at N=6, and the tested starts share data,
recipe, and reference infrastructure, so per-start outcomes
may be correlated. Signal, once obtained, is a property of
the encoded model (warm-start experiment: the donor model's
signal persists across start contexts).

## Table S16: Scale boundary — k=300 encoding exploratory run (Evaluation, reference-calibrated readout)

Calibration status note (2026-11-06): this exploratory configuration reuses the principal threshold rather than independently recalibrating a k=300 conformal audit; its acceptance statements are therefore not an independently recalibrated conformal guarantee at that scale.

Source: `research_outputs/uw/c13k300-20261016T211142/verdict.json`.
Independently trained reference models (6) and starts (3) for a
300-record deletion set (10× the main encoding scale). Audit
threshold reused from the main calibration (q̂ = 5.245); the
k=300 honest dispersion inflation (the calibration table, main Table VIII) suggests this may be
conservative.

| Start | pass | s₆ | m₁ | z-TPR@5% (negloss) | z-AUC (negloss) |
|---|---|---|---|---|---|
| k300_s46101 | ✓ | 5.245 | 0.7814 | 0.03 | 0.513 |
| k300_s46102 | ✓ | 5.245 | 0.7814 | 0.05 | 0.498 |
| k300_s46103 | ✓ | 5.245 | 0.7814 | 0.02 | 0.443 |

All finals pass the audit (s₆ = q̂, utility-pinned as at
k=30). The calibrated per-record readout is absent: z-AUC 0.44–0.51
on both channels, z-TPR at the calibrated floor (raw ranking
unmeasured). This is an
exploratory (n=3 starts) negative result: the calibrated
per-record readout did not reproduce at this scale (raw
ranking unmeasured; no general scale law claimed).

## Table S17: Evidence map (acceptance vs. signal delivery, separated)

Scope: each row identifies construction rule, reference family, audit instance, comparison population, readout, and inferential unit; acceptance and signal delivery are reported separately. "No positive interval" means the stated test did not establish a positive gain; it does not establish absence of every possible encoding or readout. The principal experiment targets cat; the class probes measure transfer. Rows added 2026-11-06 (E1-E3/S22 probes): exact-retrain refs (CE/GR) = accepted 6/6 pinned, 0/6 CI-positive; k=100 preregistered = 6/6 pinned, 0/6 CI-positive; k=300 exploratory = pinned, calibrated z-AUC 0.44-0.51, raw unmeasured; independent calibration (five draws) = 12/15 pass true audit, 8/15 strictly CI-positive (two more borderline), **5/15 accepted and strictly positive**, joint success in 3/5 draws (S28); random-F selection = 4/6 CI-positive under a fresh A6 instance (q-hat 10.07); k=300 independently recalibrated = 3/3 pin-and-pass, no signal (S35); frog second instance = 6/6 CI-positive label-blind, working-point not reproduced (z-TPR increments at most +0.067, one negative).

Every cell is a measured result from a main-text table or an
earlier supplementary table (last column); no cell is
extrapolated. Read the table as the complete scope statement of
the demonstrated harm: positive rows show where per-record or
ranking capability exists; negative rows show where it was
measured to vanish or attenuate.

| Capability tier | Scale | Dataset / arch | Persistence | Readout strength | Evidence |
|---|---|---|---|---|---|
| Tier 1 (label-blind) | k=30 | CIFAR-10 / RN18 | pre-fine-tuning | AUC gain +0.027–0.246; 9/9 CI positive; working-point TPR ≤ 0.10 @5% FPR | main Table V; §VI-B |
| Tier 1 | k=30 | CIFAR-10 / MNV2 | pre-fine-tuning | AUC gain 4/6 CI positive (max +0.118) | S8 |
| Tier 1 | k=30 | CIFAR-100 / RN18 | pre-fine-tuning | AUC 3/6 CI positive (calibrated channel: see Tier-2 row) | S10 |
| Tier 1 | k=300 | CIFAR-10 / RN18 | pre-fine-tuning | Raw ranking unmeasured (only the calibrated channel was probed; see Tier-2 row) | S16 |
| Tier 2 (reference-equipped) | k=30 | CIFAR-10 / RN18 | pre-fine-tuning | TPR 0.07–0.43 over the eleven accepted finals (median 0.17 over all eleven; 9/11 improve from base; actual FPR 0.040–0.130); rejected-by-one-grid-step final reads 0.60 at nominal 5% | main Table VI |
| Tier 2 | k=30 | CIFAR-10 / MNV2 | pre-fine-tuning | z-TPR max 0.23 (5/6 improve) — positive but attenuated vs ResNet's 0.43 | S8 |
| Tier 2 | k=30 | CIFAR-100 / RN18 | pre-fine-tuning | z-TPR at floor; increment ≤ +0.07 | S10 |
| Tier 2 | k=300 | CIFAR-10 / RN18 | pre-fine-tuning | Calibrated readout absent | S16 |
| Tier 2 (excess z-TPR retention; reference-based readout) | k=30 | CIFAR-10 / RN18 | after 1 epoch benign ft | median retention 0.00 (WASHED; individual finals vary — one pair retains 1.00); 3 epochs: median 0.14 | S9 |

Scope conditions attached to the positive rows. (i) The
strongest per-record capability is demonstrated at Tier 2,
k=30, CIFAR-10/ResNet-18, before downstream fine-tuning;
the Tier-2 capability replicates positively but attenuated
on MobileNetV2 (max 0.23), fails to replicate on CIFAR-100
(≤ +0.07), and is absent at k=300 (S9–S16). (ii) Tier-1
harm is ranking-level only. (iii) Multi-start selection
raises the probability of obtaining a signal-carrying
artifact (GR × K=10 pool: design 4/6, prospective 3/6,
MobileNetV2 2/6 → 9/18; a conditional i.i.d. projection
gives $1-(0.5)^N$, > 96% at $N{=}6$, with the pooled rate's
exact interval $[0.26, 0.74]$ mapping to $[0.84, 0.9997]$ at
$N{=}6$) but does not extend any row of this table beyond
its stated boundaries. (iv) The aggregate one-sided
z-statistic (S13) detects encoding presence on both
architectures (held-out honest FPR 2/12 per leg, elevated
above the 5% design level) but carries no per-record
information.

## Table S18: Audit-statistic coexistence for the CE (first-order) finals (Methodology)

Source: `research_outputs/uw/mb-ce-20261103T000022/mb_ce.json`
(runner `uw/run_mb_ce.py`; identical m5 definition to the
gradient-final analysis of the audit-decoder table (main Table VII) — logistic-probe AUC of F
vs the 3000-record retain sub-sample, deviation from center in
MAD units; added 2026-11-03). The three
F′-request CE finals were not checkpointed by their encoding
run and are not regenerated here; this table covers the six
F-request CE finals (the main text's "CE finals" claims refer
to these six). Run-identity note: the downstream-AUC column
is a separate execution from the encoding table's (main Table V); cross-run
differences are at most 0.014 with the same ordering (cf. the
analogous note in Table S6).

| Final | m₅ base→final | m₅ deviation (MAD, final) | downstream AUC |
|---|---|---|---|
| CE e3-s1 | 0.661→0.743 | 0.10 | 0.703 |
| CE e3-s3 | 0.719→0.829 | 1.19 | 0.775 |
| CE e3-s4 | 0.734→0.625 | 1.88 | 0.544 |
| CE e4-s0 | 0.820→0.712 | 0.56 | 0.628 |
| CE e4-s1 | 0.784→0.833 | 1.24 | 0.621 |
| CE e4-s2 | 0.814→0.756 | 0.10 | 0.658 |

Reading: for the CE finals the audit's own membership
statistic moves within band in both directions (decreases on
3/6, increases on 3/6) while downstream separability rises on
all six — coexistence without the directional decrease seen
on the gradient finals (main Table VII, 5/6 decreasing). The
audit-relevant statement is the same for both families: m₅
never flags the encoding (all deviations within tolerance,
all finals accepted); the directional claim belongs to the
gradient variant only.

## Table S23: Positive-domain expansion (E1-E3, preregistered 2026-11-03)

Preregistration cards: `docs/plans/2026-11-03-ertr-exact-retrain-encoding-prereg.md`,
`2026-11-03-e2mcls-multi-class-prereg.md`, `2026-11-03-e3ext-prereg.md`.
Runners `uw/run_ertr.py` / `run_e2mcls.py` / `run_e3ext.py`; sources
`ertr-20261103T210519/`, `e2mcls-20261103T220400/`,
`e3k100-20261103T220400/`, `e3mobdog-20261103T233240/`.
Every verdict follows its frozen gate; no post-hoc adjustment.

**A. Reference family (E1).** Exact-retrain references (15 fresh
from-scratch retrains, 6 fit / 6 cal / starts; + 3 existing ew3
retrains as secondary starts), everything else identical to the
primary protocol. q-hat_RT = 5.638, honest pass 18/18:

| Arm | Starts | accepted | CI-positive | AUC gains |
|---|---|---|---|---|
| Exact-retrain refs (cat, k=30) | 6 | **6/6 (pinned at q)** | **0/6** | ~0 (e.g., 0.504 to 0.504) |

**EXACT-RT-NEGATIVE**: acceptance fully survives under exact-retrain
references; the per-record signal does not exist there. The
headline's fine-tuning-referenced scope is thus a measured
boundary, not a framing choice.

**B. Deletion classes (E2).** Full pipeline (theta0, 24 one-epoch
ft, A6-cls 6/12/6, ce_gen x 6 starts) on three new classes:

| Class | q-hat | honest pass | accepted | CI-positive | Verdict |
|---|---|---|---|---|---|
| frog | 16.77 | 19/24 | 6/6 | **4/6** (+0.10..+0.24) | **REPLICATED** |
| dog | 4.46 | 20/24 | 5/6 | 1/6 (+0.155) | PARTIAL |
| airplane | 4.48 | 22/24 | 6/6 | 0/6 (+0.01..+0.08) | NOT (signal) |

Class tally with the primary: **2/4 classes replicate** (cat,
frog); dog partial; airplane shows the acceptance-signal
separation (all finals pass, no signal) -- the same signature as
the CIFAR-100 dataset boundary. Within the fine-tuning-reference
family the class-specific thresholds differ (frog 16.77 > cat
5.25 > dog 4.46 ~ airplane 4.48); because each class is
normalized by separately calibrated centers and scales, a larger
quantile does not by itself establish wider physical dispersion,
so we record the transfer ordering without that inference. The
exact-retrain arm (q-hat 5.638, no signal) shows the reference
family dominates this
within-family pattern.

**C. Scale and architecture x class (E3).**

| Arm | q-hat | honest pass | accepted | CI-positive | Verdict |
|---|---|---|---|---|---|
| k=100 (cat, RN18) | 3.03 | 21/24 | 6/6 (pinned) | 0/6 (+0.01..+0.03) | SCALE100-NOT |
| MobileNetV2 x dog | 2.74 | 20/24 | 5/6 | 1/6 (0.539 to 0.727, [+0.066, +0.310]) | PARTIAL |

The scale ladder is now complete: k=30 strong, k=100 none
(label-blind), k=300 calibrated-readout none -- acceptance
survives at every scale. The architecture x class crossing
attenuates but does not erase the signal (cf. S8: MobileNetV2 x
cat 4/6 CI-positive).

**D. Closing batch (E4, preregistered 2026-11-04; cards
`2026-11-04-e4-closing-batch-prereg.md`; sources
`ertr-gr-20261104T094808/`, `frogz-20261104T100414/`,
`e4c-20261104T091218/`; three empty `frogz-*` directories
from the crashed attempts are retained as records).**

| Arm | Question | Result | Verdict |
|---|---|---|---|
| GR (second-order) under exact-retrain refs | Is E1's no-signal CE-specific? | 6/6 pin-and-pass, 0/6 CI-positive, AUC consistently *decreases* (e.g., 0.497 to 0.417) | **RT-GR-NEGATIVE** (boundary observed for both tested constructions) |
| frog finals, Tier-2 z-readout + persistence | Does the strongest class carry the working-point readout? | z-TPR increments +0.00 to +0.067 (one -0.10), FPR 0.035-0.120; median retention after 1-epoch benign ft = -0.5 | **FROG-Z-WEAK** + **FROG-WASHED** (frog is Tier-1 only) |
| frog second audit instance | Instance stability | 6/6 CI-positive (gains +0.13 to +0.26), q-hat = 11.4 | **REPLICATED** (stronger than instance 1's 4/6) |
| airplane second audit instance | Instance stability | 6/6 pass, 0/6 CI-positive, q-hat = 4.6 | **NOT** (agrees with instance 1) |

**Reading of the expansion (updated).** Across eight new audit
instances and 54 encodings: **accepted endpoints recur across
the tested settings** (non-passing endpoints are also present);
**signal
delivery is conditional** -- on the reference family
(fine-tuning yes, exact-retrain no, for both first- and
second-order constructions), on class (instance-stable: frog
strongly replicated at Tier 1 in two independent audit
instances; airplane signal-absent in two), on readout tier
(the Tier-2 working-point readout is cat-specific among
classes tested), and on scale (k=30 strong, k=100 and k=300
absent). The positive domain of the per-record claim is now
measured along all four axes rather than assumed.

## Table S24: Encoder-core ablations (E5, preregistered 2026-11-04)

Preregistration: `docs/plans/2026-11-04-e5-ablation-prereg.md`.
Runners `uw/run_abl.py` (arms A-C on the frozen primary audit) and
`uw/run_abld.py` (arm D, full fresh pipeline); sources
`abl-20261104T125847/`, `abld-20261104T120910/`. Three
signal-bearing starts (e3-s1, e3-s3, e4-s0) for A-C; readout =
AUC(ng) base->final with member-resampling CIs.

| Arm | Sweep | Mean gains (per value) | CI-positive | Verdict |
|---|---|---|---|---|
| A: contrastive weight lambda | 0.25 / 0.5 / 1.0 / 2.0 | +0.077 / **+0.179** / +0.100 / +0.002 | 0/3, **3/3**, 0/3, 0/3 | **LAMBDA-SENSITIVE** (frozen 0.5 is the optimum; lambda=2.0 kills the attack) |
| B: construction-pool size Nc | 30 / 100 / 300 / 800 | +0.112 / +0.125 / **+0.179** / +0.148 | 2/3, 2/3, **3/3**, 2/3 | **NC-GRADIENT** (mild peak at 300; CE robust down to Nc=30) |
| C: iteration budget | 15 / 30 / 60 / 120 | +0.163 / +0.181 / +0.179 / +0.179 | 3/3, 3/3, 3/3, 3/3 | **ITER-SATURATED** (15 iterations suffice on signal-bearing starts; 30/60/120 gains within 0.002 of each other) |
| D: F-selection rule | random-30 (vs f_deep) | fresh A6 instance, 6 starts | 4/6 | **RANDF-REPLICATED** (attack independent of f_deep distinctiveness selection; q-hat = 10.07 under its own calibration, the widest among the tested instances, which we note without inferring physical dispersion from a separately normalized quantile) |

Protocol notes: arm-B pools of 30/100/300 are resampled subsets
of the primary 300-record construction pool (train-side, retained
members, per the S19 manifest); the 800-record ceiling arm extends
the pool with never-trained cat test-side records excluding the
frozen evaluation non-members, so construction and evaluation sets
remain disjoint but the 800 arm uses a different population than
the primary protocol. For Nc=30 the construction batch adapts to
min(60, |pool|) (full-pool sampling). Arm B complements, not
contradicts, the Nc=30 collapse of the gradient variant under
independent evaluation (Evaluation, label-blind results): different construction and
evaluation protocol. Arm C's saturation on signal-bearing starts
does not provide a start-selection probe (the trajectory is
non-monotonic across starts; Theoretical Analysis).

Reading: of the four frozen encoder choices, the contrastive
balance (lambda = 0.5) is load-bearing with the frozen value at
the sweep optimum; the construction-pool size and iteration
budget are not critical (mild peak at 300; saturation by 15);
and the deletion-set selection rule (f_deep vs random) does not
gate the attack at all.

## Table S31: Second sequential-request pair (preregistered 2026-11-10; F2 seed 54901, protocol verbatim from S11)

Source: research_outputs/uw/seq2-*/verdict.json. All three pair-1 signatures replicate: G1 stale models accepted under the request-2 audit; G2 honest F2-processing erases F1 signal (0.47 -> 0.07/0.10/0.13/0.23 across finals); G3 the passing second encoding delivers no working-point gain on F2 (0.03 -> 0.03, F1 residual 0.10) --- signal start-dependence, consistent with the primary experiments. Verdict: stale-pass; washed; no-compound.

## Table S35: k=300 under an independently recalibrated audit (preregistered 2026-11-10 #4)

Source: research_outputs/uw/k300cal-*/verdict.json. Audit rebuilt at k=300: 18 fresh honest fine-tune references (fit 6 / cal 12), conformal q = 5.503 (vs the principal 5.245 reused in S16). Three gradient-variant encodings under the recalibrated rule.

| Arm | pass | m1 | z-TPR@5% | z-AUC (neggrad) | actual FPR |
|---|---|---|---|---|---|
| enc s46101 | yes | 0.779 | 0.02 | 0.552 | 0.020 |
| enc s46102 | yes | 0.779 | 0.03 | 0.471 | 0.055 |
| enc s46103 | yes | 0.779 | 0.02 | 0.471 | 0.045 |

Honest k=300 z-TPR@5%: median 0.03 (18 refs). Gate: encoded median 0.02 vs honest 0.03, diff -0.01 -> pass (no signal delivery). Reading: acceptance stability and signal absence at k=300 both hold under independent recalibration; the S16 threshold-reuse caveat is closed.


---

# Group D - Mechanism of tolerance (main text Act 2)

## Table S6: Mitigation paired results

| Start | base | gain$_{A_6}$ | gain$_{A_7}$ | drop | pass$_{A_7}$ |
|---|---|---|---|---|---|
| e4-s0 | 0.494 | +0.140 | +0.045 | 0.095 | ✓ |
| e4-s1 | 0.498 | +0.150 | +0.038 | 0.112 | ✓ |
| e4-s2 | 0.504 | +0.121 | +0.050 | 0.071 | ✓ |

Gains are point estimates per start; member-resampling
intervals are not computed (the paired reductions 0.071-0.112
far exceed the cross-run base difference <= 0.011 noted
below, but no interval column is attached).

Same start checkpoints and construction data; only the audit
version ($A_6$ vs $A_7$) differs.

**Run identity note**: The base AUC values here (0.494/0.498/
0.504) come from a separate execution of the base evaluation
from the encoding table (main Table V; 0.493/0.505/0.505). The source of the cross-run
difference has not been individually isolated; both executions
used the same start checkpoints, the same 200-record candidate
set, and the same scoring direction, but were independent
gradient-computation runs. Within each row of this table, the
$A_6$ and $A_7$ arms share the same base measurement, the same
evaluation protocol (model.eval() mode, batch size 500, no
shuffling, identical record ordering on the 200-record candidate
set), and the same start checkpoint. The within-run $A_6/A_7$
comparison is internally paired; the cross-table base difference
is not treated as the same measurement as the encoding table (main Table V).

## Table S33: Class-wise dispersion exhibit (preregistered 2026-11-10 #5b)

Source: research_outputs/uw/classdisp-20261110T115058/verdict.json. Per class (frog/dog/airplane): class pool capped at 2000 train records, k=30 f_deep-style selection (frozen seed), 3 honest from-scratch retrains (40ep, run_ez recipe) vs the 6 ew3 full-data noise retrains evaluated on the class F; ratio = honest max-dev-from-median / noise max-dev-from-median per metric.

| Class | m3 forget-loss | m4 KS | m5 MIA | m6 margin | m1 acc |
|---|---|---|---|---|---|
| frog | 1.52 | 1.00 | 0.17 | 0.17 | 0.67 |
| dog | 2.74 | 0.08 | 0.54 | 0.58 | 0.72 |
| airplane | 18.16 | 0.62 | 0.82 | 1.28 | 0.57 |

Verdict per frozen gates: **RATIOS-CONSISTENT** -- forget-side inflation >1 in all three classes with flat test-side ratios. Note: airplane, whose encoding delivers no signal (S23), shows the largest inflation; the mechanism that grants room and the encoder's ability to use it are different axes. (Cat principal: 4.78 at the same k=30 under the original protocol; ratios here use the noise-ensemble denominator rather than the max-deviation band, so magnitudes are not directly comparable across the two tables.)

## Table S34: Dispersion ratios at n=6 honest retrains (preregistered 2026-11-10 #3)

Source: research_outputs/uw/disp6-20261110T102730/ (ew3 resumed with 3 added seeds per scale; exact recipe and metrics reused; full-data bands from the original 6 noise retrains). Ratio = honest max-dev-from-median / full-data band, per the original main Table VIII definition (pre-restructure numbering).

| k | n=3 (original) | n=6 | m1 test-side (n=6) |
|---|---|---|---|
| 30 | 4.78 | 7.51 | 0.82 |
| 300 | 1.91 | 2.13 | 0.49 |
| 1000 | 2.84 | 3.94 | 0.58 |

Reading: the qualitative signature (forget-side inflation well above 1, test-side flat below 1, non-monotone in k) is unchanged; the small-sample maxima shift upward at n=6, as expected for max-deviation statistics when more seeds are added. The naive-band honest rejections remain (honest-rejected fraction 5/6, 4/6, 3/6 at k=30/300/1000 -- a majority at the first two scales and a tie at the third, not a majority at every scale), consistent with the calibration claim. Disclosure: the artifact's own wash verdict field reads "UNDECIDED" (its material gate requires honest-rejected majority and k=1000 is a 0.50 tie); honest-rejected fraction is 0.833/0.667/0.50 at k=30/300/1000.

## Table S36: Dispersion ladder across deletion scales (path A, preregistered 2026-11-12)

Source: k=30/300/1000 ratios from the calibration table (main Table VIII, ew3 run); k=100 from `disp100-20261112T170216/verdict.json` (same theta0, pool, sub seed 52000, metric pipeline; 6 archived full-data noise retrains vs 3 fresh honest retrains on D minus F(100), seeds 44141-44143). Signal column = label-blind replication at that scale (Table S23).

| k | m3 dispersion ratio | m4 KS | m5 MIA | m6 margin | m1 acc | label-blind signal |
|---|---|---|---|---|---|---|
| 30 | 4.78 (n=3) / 7.51 (n=6) | 1.58 | 1.02 | 1.24 | 0.20 | yes (9/9, 7/9 gate) |
| 100 | **2.51** | 0.37 | 0.48 | 0.36 | 0.26 | no (0/6) |
| 300 | 1.91 | 1.05 | 0.56 | 0.40 | 0.20 | calibrated readout negative; raw unmeasured (S16) |
| 1000 | 2.84 | 0.79 | 0.29 | 0.32 | 0.67 | not measured for encoding |

Reading: the forget-loss ratio does not order signal across scales -- k=30 (4.78 at n=3; 7.51 at n=6) delivers, k=100 (2.51) does not, and k=1000 (2.84) exceeds the k=100 value but not the k=30 ones, with no signal measurement at k=1000. Note that k=100 is also the family whose accuracy allowance (4.45 points, Table S39) exceeds the working family's 4.17 while delivering nothing, so neither the dispersion ratio nor the allowance orders the scale axis; it is reported as open.

## Table S37: Aligned-instrument check -- fresh vs the audit's own comparison pool (path A, preregistered 2026-11-12)

Source: `align-20261112T185915/verdict.json`; CI rescore `align-20261114T121907/verdict.json` (identical configuration and seeds, runner extended only to archive the per-pair gains; every base mean and median reproduces the archived values exactly, deviation 0.0). Zero training: the 12 primary finals and their 6 paired base starts, scored with per-record neggrad on two comparison populations -- fresh 200 never-trained records (primary readout pool) and the 3,000-record retained sub-sample that the audit's own m5 uses. z = per-record calibration against the ew9F fit ensemble computed within each pool. Paired gains are final minus its own base.

| Readout | base mean | median paired gain | cluster-bootstrap 95% CI |
|---|---|---|---|
| fresh pool, raw | 0.491 | +0.122 | [0.067, 0.207] |
| retained pool, raw | 0.271 | +0.175 | [0.035, 0.259] |
| fresh pool, z-calibrated | 0.491 | +0.151 | [0.033, 0.297] |
| **retained pool, z-calibrated** | 0.483 | **+0.312** | **[0.137, 0.377]** |

Verdict per frozen gates: **ALIGNED-SURVIVES** (retained-pool gain >= 2/3 of the fresh-pool gain). Reading: the acceptance--readout gap is not an artifact of comparing deleted records against never-trained rather than retained records; on the audit's own comparison population the calibrated signal is in fact stronger. Raw (uncalibrated) scoring inverts direction on the retained pool (base 0.271), which is why per-record calibration is load-bearing. Uncertainty: 10,000-resample percentile bootstrap (`uw/align_ci.py`, seed 13000) over the 6 start clusters -- the 12 finals derive from 6 independent starts (two finals per start), so clusters, not individual gains, are resampled; all four 95% CIs exclude zero. The record-level bootstrap over the 12 gains (which treats correlated gains as independent) is archived in the rescore verdict and is wider on the retained pool.

## Table S38: Band-width dose response (path A, preregistered 2026-11-12)

Source: `width-20261112T185048/verdict.json` (kappa 0.5-1.5) and `width-20261112T191940/verdict.json` (utility extension, kappa 2 and 3). Acceptance iff |m_i - c_i| <= kappa_fam * q_hat * sigma_i with q_hat = 5.245 and the principal fit6/cal12 calibration; one family scaled at a time, the other held at kappa = 1. Three e4 starts; label-blind neggrad AUC gain, 2000-resample CI.

| kappa | utility arm (m1, m2 scaled) | forget arm (m3-m6 scaled) |
|---|---|---|
| 0.5 | 0.084 (1/3 starts feasible) | 0.121 |
| 0.75 | 0.125 (1/3) | 0.155 |
| 1.0 | 0.154 | 0.155 |
| 1.5 | 0.188 | 0.154 |
| 2.0 | 0.225 | -- |
| 3.0 | 0.253 | -- |

Per-start utility curves: 0.164/0.147/0.151 at kappa=1 rising to 0.195/0.188/0.182 at 1.5, 0.220/0.247/0.209 at 2, 0.249/0.236/0.273 at 3. Per-start forget curves: 0.155/0.148/0.060 at kappa=0.5, then 0.165/0.149/0.151, 0.164/0.148/0.151, 0.164/0.148/0.151 at 0.75/1.0/1.5.

Denominator note: at kappa=0.5 and 0.75 the reported utility gains are conditional on the single feasible start; at kappa>=1 the reported gains average all three starts.

Verdict per frozen gates: **UTILITY-SLACK-GOVERNS**. Two honest qualifications: (i) at kappa<1 the accuracy-tightened audit no longer accepts the honest start itself (start infeasible for 2 of 3 starts at 0.5 and 0.75), so those cells are mechanism probes, not deployable audit configurations; (ii) the forget arm is flat from 0.75 to 1.5 but drops sharply at 0.5 on one start only (0.060), so a forget-side effect cannot be ruled out for that start -- the asymmetry with the utility arm, whose mean is monotone though one start decreases slightly between kappa = 2 and 3 (0.247 to 0.236), is the finding.

## Table S39: Overall-accuracy half-width B_1 by reference family (path A, preregistered 2026-11-12)

Allowance B = q_hat * sigma_1 (in accuracy points), sigma_1 = MAD of the honest reference ensemble's m1, q_hat = that instance's conformal threshold at alpha = 0.20.

| Family | n refs | c1 | sigma_1 | q_hat | B (points) | signal |
|---|---|---|---|---|---|---|
| main fine-tune, k=30 | 6 | 0.8231 | 0.00795 | 5.245 | 4.17 | yes (9/9) |
| from-scratch retrain (E1) | 6 of 15 | 0.8936 | 0.00130 | 5.638 | 0.73 | no (0/6, both constructions) |
| frog fine-tune | 24 | 0.8096 | 0.01640 | 16.77 | 27.50 | yes (4/6) |
| dog fine-tune | 24 | 0.7962 | 0.01825 | 4.46 | 8.14 | partial (1/6) |
| airplane fine-tune | 24 | 0.8018 | 0.01680 | 4.48 | 7.53 | no (0/6) |
| k=100 fine-tune | 24 | 0.8136 | 0.01470 | 3.03 | 4.45 | no (0/6) |

Reading: the allowance separates the two reference families by 5.7x, which is consistent with the retrain-side failure but confounded with start ancestry (E1 encoded from retrain starts, which never saw F) and below the tested range (0.73 points against a minimum tested 2.1, i.e. kappa = 0.5 of the working allowance). It is not sufficient: k=100 (4.45), airplane (7.53) and dog (8.14) all exceed the working 4.17 and deliver nothing or almost nothing, so the class and scale axes stay open. Retrain references reach 88.9% accuracy with only 0.14 points of dispersion, so the retrain-referenced audit grants only a narrow overall-accuracy half-width; the tested encodings do not reproduce the signal under it, and reference family remains confounded with start ancestry.

## Table S41: Admissibility profile -- what each audit admits (path A, 2026-11-13)

Source: `inadmissible-*/verdict.json`. Retrain audit: c, sigma from six E1 retrains (rt_s47601-47606); q = 5.638, the 6th of 6 calibration order statistics from rt_s47607-47612 under the paper's conformal rule (`conformal_q`, ceil((n+1)(1-alpha))). (An earlier diagnostic printed 4.759, the 5th order statistic; that is 15.6% too small and is superseded here -- every band column below is recomputed at q = 5.638, not the value stored in that run's verdict file.) Fine-tune audit: the principal fit6/cal12 calibration (load_audit), q = 5.245. z entries are (m - c)/sigma; band = q * sigma.

| Component | retrain c | retrain sig | retrain band | theta0 | theta0 z | ft start | ft z |
|---|---|---|---|---|---|---|---|
| m1 accuracy | 0.8936 | 0.00130 | +/-0.0073 | 0.8924 | -0.92 | 0.8217 | -55.3 |
| m2 class acc | 0.7835 | 0.01050 | +/-0.0592 | 0.7800 | -0.33 | 0.6470 | -13.0 |
| m3 forget-set loss | 0.8182 | 0.12246 | +/-0.6904 | 0.1025 | -5.84 | 0.9919 | +1.42 |
| m4 KS | 0.4943 | 0.03200 | +/-0.1804 | 0.3640 | -4.07 | 0.5530 | +1.83 |
| m5 MIA AUC | 0.7979 | 0.00822 | +/-0.0464 | 0.7303 | -8.22 | 0.7835 | -1.74 |
| m6 forget margin | 3.2777 | 0.16106 | +/-0.9081 | 5.3334 | **+12.76** | 1.0173 | **-14.03** |

Fine-tune audit for contrast: m1 centre 0.8231, band +/-0.0417; m6 centre 1.2217, band +/-5.4196.

Reading: under this tested retrain-calibrated configuration, both inspected F-aware profiles are rejected on different components (the original model on the deletion statistics m3 -5.8 / m6 +12.8 despite in-band accuracy; the one-epoch fine-tune start on accuracy -55.3 and margin -14.0), while the tested encodings from admitted F-free retrain starts do not establish positive signal. Reference family and start ancestry are confounded, so these are joint empirical outcomes rather than an identified operator-family or ancestry effect. Note also that the fine-tune-referenced m6 band (+/-5.42 around 1.22) spans both profiles, which is why forget-side manipulation is admissible there at all.

## Table S44: One frozen test set: three rules side by side (moved from the main text 2026-11-15)

12 honest models; 21 archived attack checkpoints scored under the
frozen split-conformal audit only. References = honest models used to
build the rule. Honest and attack columns have different denominators
and generation processes; no detection comparison is implied.

| Rule | Refs | Hon.\ rejections @.20 | Atk.\ rejections @.20 | Hon.\ rejections @.10 |
|---|---|---|---|---|
| max-deviation | 6 | 7/12 | -- | -- |
| split-conformal | 6+12 | 4/12 | 5/21 | 0/12 |
| max-dev. (matched) | 18 | 2/12 | -- | -- |


---

# Group E - Defenses and adaptive arms (main text Act 3)

## Table S12: A8 joint acceptance rule (Evaluation, defenses)

Source: `research_outputs/uw/a8joint-20261101T074754/verdict.json` (primary
run, rho = 0.5, GPU-1 score-reproduction device; device-sensitivity
copy on GPU-0 identical in verdict). Preregistration:
`docs/plans/2026-10-26-a8-joint-statistic-prereg.md`. s_J =
(m−c)ᵀ Σ̂⁻¹ (m−c); Σ̂ = (1−ρ)·S_fit + ρ·diag(S_fit) (n = 6,
shrinkage mandatory); threshold = conformal quantile of the 12
calibration models' s_J at α = 0.20 (Σ̂ frozen first; clean split).

| Group | A6 marginal (accept) | A8 joint (accept) |
|---|---|---|
| Held-out honest (12) | 8 | **12** |
| Encoding finals (12) | 11 | **5** |
| — GR family (6) | 5 (GR e4-s2 out, s₆ = 5.258) | **0** (s_J 17.1–24.8 vs q̂_J 16.47) |
| — CE family (6) | 6 | 5 (s_J 11.8–13.7; CE e3-s3 at 18.5 rejected) |
| E-SEQ stale trio | 3 | 2 (stale-GR rejected) |

**Per-final reconciliation with the z-readout table (main Table VI) / S7** (authoritative
record, sources `a8joint-20261101T074754` primary and its
GPU-0 device copy `a8joint-20261101T074926`). The A6 column
above is the primary execution and matches the z-readout table (main Table VI) exactly
(11/12, GR e4-s2 out). All eleven passing finals sit at the
A6 boundary exactly (utility-pinned); three of them exhibited
device-dependent A6 status across the two scoring devices:
GR e4-s2 (5.258 primary / 5.245 on
GPU-0), CE e3-s1, and CE e4-s1 (5.245 primary / 5.258 on
GPU-0) — on the GPU-0 copy the A6 totals are 10/12 (CE 4, GR
6). The A8 verdicts are unchanged across devices on every
final: CE e3-s3 rejected (s_J 18.5) with the other five CE at
11.8–13.7, all six GR at 17.1–24.8. Readouts are paired with
the primary execution throughout (main Table VI).

Shrinkage sensitivity (descriptive, post-result): honest
acceptance 12/12 at every ρ; finals rejected 5/12 @ ρ=0.75,
7/12 @ ρ=0.50 (primary, frozen), 10/12 @ ρ=0.25 — rejection
power grows with joint (off-diagonal) structure, directionally
consistent with the mechanism. Device note: see the per-final
reconciliation above (three grid-step-boundary finals with
device-dependent A6 status; A8 stable on every final). Claim scope: one audit
instance, twelve finals, frozen shrinkage — a mechanism
demonstration, not a deployed defense.

**Bootstrap robustness (added 2026-11-01):** resampling six
honest models from the 18 cached vectors and re-estimating
Sigma-hat and q_J (200 draws, `a8boot-20261101T134206`) gives q_J in
[4.8, 56.4] (5–95%), CE rejection bimodal (0 rejected in
33.5% of draws, all six in 46.5%), GR all-six rejection in
62.5%. Part of the spread is a small-sample pathology
(degenerate resamples collapse dispersion and blow up the
Mahalanobis scale), but the instance-conditionality of the
CE column is real: conclusions are conditional on the
primary (ew9F-referenced) instance.

**Adaptive arms under the joint constraint** (added
2026-11-02; sources `a8e-20261101T134414/verdict.json`
design starts, `a8e-20261101T141813/verdict.json`
prospective fresh starts; preregistered, R1 ledger
`2026-11-01-method-improvement-ledger.md`). Encoders
re-run with the acceptance check replaced by $s_J \le
\hat q_J$ (same $\hat\Sigma$, $c$, and split as the A8
rule above):

| Arm | Starts | pass $A_8$ | CI-positive gain | Note |
|---|---|---|---|---|
| CE-joint (first-order), design | 6 | **6/6** | **5/6** | non-inferior to marginal counterparts (paired diff −0.015 to +0.028; 90% CI LB −0.009 > −0.03); all finals in the A8-only quadrant (pass $A_8$, outside the marginal band) |
| CE-joint, prospective | 6 | **6/6** | 2/6 | gate G1 (≥4/6 improved+accepted) fails → BLOCKED-A8; signal start-dependent |
| Grad-joint (double-backward), design | 3 | 3/3 | 1/3 | e3-s1 AUC 0.547→**0.850**, z-TPR 0.53 (strongest readout in this work) |
| Grad-joint, prospective | 6 | 6/6 | 3/6 | max AUC 0.822 (f-e4-5, CI [+0.252, +0.493]) |
| F-only-joint (control) | 2 | 2/2 | 0/2 | weak, as expected |
| CE-joint, 120 iters (convergence) | 1 | 1/1 | 1/1 | 0.724 vs 0.725 at 60 iters |

Reading: no construction family is closed by the joint
rule. First-order CE is feasible under $A_8$ everywhere
tested (12/12) but its signal under the joint constraint is
start-dependent (5/6 design → 2/6 prospective); the
gradient objective carries the strongest joint-constraint
readouts. The marginally-constructed GR finals rejected by
the frozen $A_8$ (main table) were constructed under the
marginal rule; gradient encoders that know $s_J$ survive.
The five-alternative negative of Table S15 is specific to
the K=10 multi-instance intersection constraint, a
different and stricter acceptance object.

## Table S13: Aggregate z-detector validation (Evaluation, defenses)

Sources: `research_outputs/uw/a9z-20261101T080001/verdict.json` (CIFAR-10,
frozen two-sided gate + post-hoc one-sided) and
`research_outputs/uw/a9z2-20261101T080230/verdict.json` (MobileNetV2
confirmatory). Detector: T = mean of top-15 z over the 30 F
records; z calibrated against the six fit references (ew9F on
CIFAR-10, z1 on MobileNetV2); thresholds = 95th percentile of
the honest ensemble's T (18 models CIFAR-10, 24 MobileNetV2).

| Statistic | Group | Detected |
|---|---|---|
| Two-sided \|z\| (frozen gate) | honest / finals | 1/18 / **1/12** → NEGATIVE |
| One-sided (member dir.), CIFAR-10 | honest / finals | 1/18 / **10/12** |
| One-sided, CIFAR-10 | H0 ft / D1 / D2 / D3 | 1/2 / 0/2 / 0/2 / 0/2 |
| One-sided, MobileNetV2 (confirmatory) | honest / finals | 2/24 / **6/6** |

The frozen two-sided variant lacks power (honest two-sided
tails are heavy at six-reference calibration); the one-sided
member-direction variant---the form the paper's design
language specified---detects encoded finals on both
architectures while firing on no natural non-compliance
model (including the model-level-broken mis-lr arm). The
one-sided CIFAR numbers are post-hoc (variant chosen after
the frozen gate failed); the MobileNetV2 run is the
confirmatory anchor on models never used to design the
statistic (record-set seeds matched to the ResNet setting).
k ∈ {5, 30} gives identical counts.

**Cross-class spillover rows** (added 2026-11-03; source
`c10xclass-20261016T205609/verdict.json`; derivation
re-verified 2026-11-03: per-model mean reference-normalized z
on non-monitored class records, median over encoded models
minus median over honest models):

| Class (non-monitored) | Channel | median z-shift (encoded − honest) |
|---|---|---|
| dog | probe records, neggrad | **−1.85σ** |
| ship | fresh records, neggrad | **−0.79σ** |

Population-level shifts only; per-record resolution on the
cross-class channel is absent (the cross-class *encoding
construction* reaches max raw AUC 0.59 with CI including
zero — Table S15, a different experiment). The
monitoring-scope implication is in the design-directions
subsection (§VI-D) of the main text.

**Held-out honest evaluation (added 2026-11-02;
`a9z3-20261102T235118/verdict.json`).** The 1/18 and 2/24
honest counts above are calibration-set tail counts (the same
ensembles set the threshold). On 12 fresh honest ft models
per architecture (seeds 45301–45312 / 45401–45412, disjoint
from every seed used elsewhere; full-epoch keep recipe
verbatim; excluded from both threshold and statistic
selection), the frozen one-sided rule fires:

| Leg | Threshold | Held-out alarms | Rate |
|---|---|---|---|
| CIFAR-10 | q₉₅ = 1.4293 | 2/12 (T = 1.500, 1.546) | 16.7% (exact 95% CI 2.1–48.4%) |
| MobileNetV2 | q₉₅ = 1.1246 | 2/12 (T = 1.137, 1.188) | 16.7% |

Detection counts unchanged (finals 10/12 CIFAR-10, 6/6
MobileNetV2; both reproduced bit-consistently in the a9z3
recomputation). Reading: the held-out honest false-positive
rate (2/12 on each leg) is above the 5% design level; with
n=12 the confidence interval includes 5%, but the point
estimate indicates the six-reference z-threshold under-covers
on fresh honest draws. The detector should be read as a
pilot (post-hoc variant selection on CIFAR-10, elevated
held-out FPR on both architectures), not a validated
operating point.

## Table S22: Two preregistered probes — tightened-threshold encoding and independent calibration (Sections III-D, V-F)

Preregistration: `docs/plans/2026-11-03-review-probe-ab-prereg.md`
(frozen gates). Runners `uw/run_probe_t.py` /
`uw/run_probe_ic.py`; sources `probe-t-20261103T094439/verdict.json`
and `probe-ic-20261103T094447/verdict.json`. Single-unit
feasibility evidence (three starts each), not a matrix.

**Probe A — encoder under tightened thresholds.** ce_gen frozen
protocol, acceptance threshold replaced by 0.99/0.95 x q-hat:

| Arm | Start | s6 base->final (threshold) | pass | AUC gain | 95% CI |
|---|---|---|---|---|---|
| 0.99q | e3-s1 | 4.214->5.182 (5.193) | Y | +0.155 | [+0.038, +0.272] |
| 0.99q | e4-s0 | 5.006->5.182 (5.193) | Y | +0.138 | [+0.022, +0.254] |
| 0.99q | e4-s1 | 0.993->5.182 (5.193) | Y | +0.095 | [-0.022, +0.213] |
| 0.95q | e3-s1 | 4.214->4.981 (4.983) | Y | +0.155 | [+0.038, +0.272] |
| 0.95q | e4-s0 | 5.006->4.981 (4.983) | Y | +0.136 | [+0.021, +0.251] |
| 0.95q | e4-s1 | 0.993->4.981 (4.983) | Y | +0.089 | [-0.029, +0.207] |

Gate: TIGHT-FOLLOWED (>=1 accepted-and-positive). Reading: the
encoder converges to a common value just below the tightened
threshold on every run (nearest feasible grid value)
and retains CI-positive signal on 2/3 starts per level;
threshold tightening moves the race rather than ending it.
Held-out honest cost of the same tightening (cached scores):
8/12 unchanged at 0.99q, 7/12 at 0.95q.

**Probe B — encoding under independent calibration.** Provider
trains 12 fresh honest ft (seeds 45501-45512, recipe verbatim;
median centers + MAD scales + split-conformal threshold from
its own 6+6 split, mirroring the audit's estimators). The
provider's estimated threshold q-hat_est = 4.582 and the true
q-hat = 5.245; the two scores use different centers and MAD
scales, so the quantile ratio does not establish that the
estimated region is tighter (or contains/is contained by the
true one). The provider's 12 models score [0.93, 5.42] on the
true audit.

| Start | s_est base->final | true s6 of final | pass true audit | AUC gain | 95% CI |
|---|---|---|---|---|---|
| e3-s1 | 1.48->4.582 | 2.419 | Y | +0.020 | [-0.122, +0.165] |
| e3-s3 | 0.61->4.582 | 2.419 | Y | +0.014 | [-0.123, +0.157] |
| e4-s0 | 5.86->4.582 | 2.419 | Y | +0.097 | [-0.033, +0.222] |

Gate: MIXED (3/3 pass, 0/3 CI-positive). Reading: acceptance
survives independent calibration — the provider does not need
the auditor's calibration draw to produce accepted models
(finals land deep inside the true region, true scores 2.419);
signal delivery
did not survive on this draw; the finals' position is
observable, but its attribution is not established
(differently normalized scores). Both the
estimated-true gap direction and the signal outcome are
draw-specific (one provider draw, three starts).

## Table S26: A7 objective-redesign arm (preregistered 2026-11-09, docs/plans/2026-11-09-a7-redesign-prereg.md)

Source: research_outputs/uw/mcr-20261109T224157/verdict.json. Same A7 audit as S6 (q7 = 6.296), same three e4 starts, record-set seeds, and label-blind evaluation as the paired re-optimization experiment; 2000 member-resample CIs. Gates frozen in the card: EVASION = any start >= +0.10 point and CI-positive; ROBUST = all <= +0.05.

| Start | Arm | base AUC | gain | CI | pass A7 |
|---|---|---|---|---|---|
| e4-s0 | R1 anti-statistic penalty | 0.494 | +0.016 | [-0.073, +0.106] | yes |
| e4-s0 | R2 second-order objective | 0.494 | +0.056 | [-0.045, +0.153] | yes |
| e4-s1 | R1 | 0.498 | +0.007 | [-0.088, +0.099] | yes |
| e4-s1 | R2 | 0.498 | +0.062 | [-0.025, +0.142] | yes |
| e4-s2 | R1 | 0.504 | +0.016 | [-0.072, +0.100] | yes |
| e4-s2 | R2 | 0.504 | +0.040 | [-0.047, +0.123] | yes |

Verdict per frozen gates: **A7-PARTIAL** — no redesigned objective reaches the evasion gate (no CI-positive gain on any start); the second-order arm exceeds the +0.05 point bar on two starts without interval support. Reading: the A7 compression survives both tested evasion tiers (re-optimization and these two redesigns); broader redesign families remain untested.

## Table S27: Stronger-m5 re-measurement (preregistered 2026-11-09, docs/plans/2026-11-09-sm5-icdraws-prereg.md)

Source: research_outputs/uw/sm5-20261109T233746/verdict.json. m5s = MLP probe (5-16-1) on per-sample [CE loss, correct margin, entropy, true logit, top-other], in-sample AUC, same protocol shape as the frozen m5; strong audit = m1-m4, m5s, m6 recalibrated on the primary fit6/cal12 honest models; this instance uses its own frozen 3000-record retained probe (train-side, seed 62000). q_strong = 5.220; honest held-out acceptance 8/12.

Tier A (rescoring the 9 available primary-final checkpoints: all six CE and three GR e3-family fb_s1/s3/s4; the three GR e4 checkpoints were not rescored in this execution): all nine rejected (GR fb_s1/s3/s4 at 5.245, CE finals at 5.245/5.258 -- all above q_strong = 5.220). **Attribution caveat:** q_strong is about 0.5% below the original 5.245, and the rescored finals are utility-pinned at the old boundary, so the threshold reduction alone suffices to reject them; this Tier-A run establishes rejection by the upgraded-and-recalibrated rule, not a detection gain isolated to the MLP component. Tier A also uses a fresh retained-probe sample (seed 62000 vs 52000 in the principal setup), so it is not a single-variable probe substitution. Verdict: STRONG-M5-DETECTS remains as a rule-level statement (on the nine rescored).

Run-identity note: Tier B base AUCs are from this execution's own base scoring (within-run paired comparisons); e4-s1's base (0.545) differs from other executions' bases by more than the disclosed cross-run differences.

Tier B: encoding under the strong audit | base AUC | gain | CI (2000 resamples) | pass |
|---|---|---|---|---|
| e4-s0 | 0.494 | +0.162 | [+0.093, +0.231] | yes |
| e4-s1 | 0.545 | +0.129 | [+0.073, +0.182] | yes |
| e4-s2 | 0.507 | +0.141 | [+0.090, +0.193] | yes |

Verdict B: STRONG-ENCODING-WORKS (3/3 CI-positive). Reading: a stronger privacy instrument rejects the frozen constructions at unchanged honest acceptance, and the adaptive encoder re-pins under it with signal intact -- the arms-race signature extends to instrument strength. Scope: one probe family (MLP on output features), one recalibration instance.

## Table S28: Independent-calibration multi-draw (preregistered 2026-11-09; draws 45601-45901 via IC_BASE, protocol verbatim from S22 Probe B)

Per draw: 12 provider-estimated honest ft models -> estimated centers/scales/threshold; 3 encodings (e3-s1/s3/e4-s0) under the ESTIMATED audit; finals submitted to the TRUE audit (q=5.245).

| Draw (seed base) | q_est | true-audit pass | CI-positive finals | gains |
|---|---|---|---|---|
| 45501 (original, S22) | 4.58 | 3/3 | 0/3 | none |
| 45601 | 2.480 | 3/3 | 0/3 strict (+1 borderline, lb ≈ +0.0003) | +0.023..+0.120, CIs cross zero except one borderline |
| 45701 | 2.461 | 3/3 | 2/3 strict (+1 borderline, lb ≈ +0.0003) | +0.130/+0.214/+0.121 |
| 45801 | 14.519 | 1/3 | 3/3 | +0.208/+0.222/+0.187 (two finals rejected at true audit) |
| 45901 | 3.747 | 2/3 | 3/3 | +0.156/+0.162/+0.142 (one rejected) |

Borderline rule (declared): L ≤ 0 not CI-positive; 0 < L < 0.005 positive but borderline (excluded from the strict count); L ≥ 0.005 strict -- an added margin criterion, not the ordinary definition of an interval excluding zero. Aggregate (5 draws, 15 finals): true-audit acceptance 12/15; strictly CI-positive gains 8/15 (two more borderline, both L ≈ +0.0003); **accepted and strictly positive: 5/15** (2 + 1 + 2 across draws 45701/45801/45901; 7/15 counting borderline); joint success in 3/5 draws (4/5 counting borderline). Verdict per frozen gates: **IND-CAL-DELIVERS** -- signal delivery under independently estimated calibration is draw-dependent, not systematically blocked. Heterogeneity note: q_est alone does not predict delivery (draw 45701 delivers at q_est 2.46; draw 45601 does not at 2.48).

## Table S29: z-detector composite-constraint adaptive arm (preregistered 2026-11-10)

Source: research_outputs/uw/zad-*/verdict.json. Constraint: pass A6 AND one-sided top-15 z-mean below tau_z=1.4293 (95th pct over 18 honest e2/e3/e4 ft models, S13/a9z3 protocol); J_CE optimized by feasible-direction descent under the composite rule; 3 e4 starts; 2000 member-resample CIs.

| Start | base AUC | gain | CI | pass A6 | z-detector silent |
|---|---|---|---|---|---|
| e4-s0 | 0.484 | +0.077 | [-0.019, +0.178] | yes | yes |
| e4-s1 | 0.495 | +0.047 | [-0.032, +0.118] | yes | yes |
| e4-s2 | 0.498 | +0.053 | [-0.030, +0.141] | yes | yes |

Verdict per frozen gates: **ZDET-ROBUST** -- the composite constraint is satisfiable (3/3 pass both rules) but no start delivers a CI-positive gain; the detector constraint suppresses signal delivery at the tested tier.

## Table S30: Tier-2 z-readout under the stronger-m5 audit (preregistered 2026-11-10)

Source: research_outputs/uw/sm5z-*/verdict.json. Same strong-m5 audit as Table S27 (q=5.220); 3 e4 starts encoded under it; CAL-Z working-point readout (z-neggrad, thresholds frozen on a never-trained aux set at nominal 5%, actual FPR on a disjoint eval set), paired base vs final.

| Start | TPR base->final | FPR base->final | pass strong audit |
|---|---|---|---|
| e4-s0 | 0.03 -> 0.20 | 0.090 -> 0.045 | yes |
| e4-s1 | 0.10 -> 0.10 | 0.040 -> 0.050 | yes |
| e4-s2 | 0.03 -> 0.13 | 0.070 -> 0.090 | yes |

Verdict per frozen gates: **READOUT-DELIVERED** (2/3 starts pass with TPR above base). The gate is improvement over the paired base, not an interval-based replication: the third start is unchanged, and these readings are point-level, not validated low-FPR individual identification. The Tier-2 working-point readout does survive the upgraded-and-recalibrated audit.

## Table S32: A8 objective-redesign arm (preregistered 2026-11-10 #5c)

Source: research_outputs/uw/a8r-*/verdict.json. Mahalanobis joint rule rebuilt (qJ=16.468, matching a8joint within rounding; shrinkage rho=0.5 frozen); R1/R2 redesign objectives (as in S26) optimized under the joint constraint; 3 e4 starts.

| Start | Arm | gain | CI | pass A8 |
|---|---|---|---|---|
| e4-s0 | R1 | +0.016 | [-0.071, +0.106] | yes |
| e4-s0 | R2 | +0.078 | [-0.019, +0.171] | yes |
| e4-s1 | R1 | +0.009 | [-0.086, +0.102] | yes |
| e4-s1 | R2 | +0.053 | [-0.038, +0.137] | yes |
| e4-s2 | R1 | +0.014 | [-0.075, +0.099] | yes |
| e4-s2 | R2 | +0.045 | [-0.040, +0.127] | yes |

Verdict per frozen gates: A8-PARTIAL (the artifact's printed label A7-PARTIAL is a template remnant) -- no redesigned objective reaches CI-positive delivery; point gains up to +0.078 exceed the +0.05 robustness bar without interval support; all six runs pass the joint rule. For comparison, re-optimization of the plain objective under A8 (12/12 pass, CI-positive on 5/6 design starts) and the gradient-objective arm reported in S12 (AUC 0.850/0.822) both succeed on their own start populations; these studies use different starts, so they do not establish a rank order among A8 evasions.

---

## Appendix: Proof of Proposition 1 (linear-model dispersion mechanism)

Setup as in the proposition: y = Xβ* + ε, ε ~ N(0, σ²I_n), OLS fits, m_F(θ̂) = (1/k)‖y_F − X_Fθ̂‖². Assumptions: σ² > 0, k ≥ 1, and the retained design full rank, G_R = X_RᵀX_R ≻ 0. Write H for the full-data hat matrix, B = I − H_FF and C = −H_FR so the F-block of the full-fit residual is w = Bε_F + Cε_R; idempotence of (I − H) gives B² + CCᵀ = B, hence B is p.s.d.

**Full fit.** m_F(β̂_full) = (1/k)‖w‖² with w = Bε_F + Cε_R, Bε_F ⊥ Cε_R in covariance (Cov(w) = σ²(B² + CCᵀ) = σ²B). For a Gaussian vector w with Cov(w) = σ²B, Var(‖w‖²) = 2σ⁴ tr(B²), so
Var[m_F(β̂_full)] = (2σ⁴/k²)·tr(B²).

**Retrain.** β̂_R − β* = (X_RᵀX_R)⁻¹X_Rᵀε_R; write A = X_F(X_RᵀX_R)⁻¹X_Rᵀ (k×(n−k)). Then y_F − X_Fβ̂_R = ε_F − Aε_R, and
m_F(β̂_R) = (1/k)(ε_Fᵀε_F + ε_RᵀAᵀAε_R − 2ε_FᵀAε_R).
(The sign does not enter the variance: the cross term has mean
zero and variance 4σ⁴tr(AAᵀ) either way.)
The three terms are pairwise uncorrelated (ε_F ⊥ ε_R; the cross term has mean zero). For ε_RᵀAᵀAε_R with AᵀA p.s.d.: Var = 2σ⁴tr((AᵀA)²) = 2σ⁴tr((AAᵀ)²). For the cross term: Var(2ε_FᵀAε_R) = 4σ⁴tr(AAᵀ) = 4σ⁴λ_F^{(R)} by the Gaussian fourth-moment identity. Hence
Var[m_F(β̂_R)] = (2σ⁴/k²)·[k + tr((AAᵀ)²) + 2λ_F^{(R)}].

**Gap.** Subtracting: k − tr(B²) = tr(CCᵀ) + tr(H_FF) (from idempotence B + H_FF = B² + CCᵀ + H_FF; both tr(CCᵀ), tr(H_FF) ≥ 0), hence
Var[m_F(β̂_R)] − Var[m_F(β̂_full)] = (2σ⁴/k²)·[tr(CCᵀ) + tr(H_FF) + tr((AAᵀ)²) + 2λ_F^{(R)}] ≥ (4σ⁴/k²)·λ_F^{(R)} > 0,
so the bound is unchanged (the extra terms are nonnegative). The gap is the sum of (i) the in-fit contraction term tr(CCᵀ), (i′) the in-fit F-leverage tr(H_FF), (ii) a second-order leverage term tr((AAᵀ)²), and (iii) the dominant cross term 2λ_F^{(R)}. In the numerical check below, the three originally displayed terms sum to ≈0.0085 while the measured gap is 0.0112; the residual ≈0.0027 is exactly the newly identified (i′) tr(H_FF) term, with which the four-term identity closes at 0.0112 = the measured gap.

**Test side (a separate condition).** For fixed x*, V_full(x*) = σ²x*ᵀ(XᵀX)⁻¹x* and V_R(x*) = σ²x*ᵀ(X_RᵀX_R)⁻¹x*. Writing G = XᵀX = G_R^{1/2}(I + M)G_R^{1/2} with M = G_R^{-1/2}X_FᵀX_F G_R^{-1/2}, the inverse is G_R^{-1/2}(I+M)^{-1}G_R^{-1/2}, so for 0 ⪯ M ⪯ δI one obtains 0 ≤ V_R(x*) − V_full(x*) ≤ δ·V_full(x*): a **bounded relative change** under δ = ‖M‖ = O(k/n).
The unwhitened ratio ‖X_FᵀX_F‖/‖X_RᵀX_R‖ does **not** imply this condition. Counterexample within the stated premises: p = 2, k = 1, X_R = (n−2) rows of e₁ᵀ plus one row of e₂ᵀ, X_F = e₂ᵀ. Then G_R = diag(n−2, 1), XᵀX = diag(n−2, 2), and at x* = e₂, V_R = σ²·1 while V_full = σ²/2 — a factor-2 gap for every n, even though ‖X_FᵀX_F‖/‖X_RᵀX_R‖ = 1/(n−2) → 0. Test-side invariance therefore requires the whitened condition; it is not a corollary of the forget-side identity. ∎


**Numerical check (generated verification, agreement at the reported precision).** Random Gaussian design n=500, k=30, p=20, sigma=1, T=4000 noise draws: Var[m_F(full)] = 0.0615, Var[m_F(R)] = 0.0727, gap = 0.0112 >= 4*sigma^4*lambda/k^2 = 0.0059 (lambda_F = 1.32); the ratio Var(R)/Var(full) rounds to 1.18, while gap/bound rounds to 1.90. The three originally displayed terms sum to about 0.0085 and the residual 0.0027 coincides with the tr(H_FF) term at this precision; the individual analytic traces and the fixed design are not reproduced here, so the closure 0.0085 + 0.0027 = 0.0112 is a Monte-Carlo agreement, not an exact analytic check.

Scope note: this is an exact statement for the linear-Gaussian model with fixed design; deep-network training adds optimizer stochasticity on top, which the measurement attributes qualitatively to the same constraint-release signature.
## Data and code availability

The source code, preregistration cards, and per-run verdict records are
publicly available at https://github.com/laiweimin/eclipse-artifacts
(the release carries the code layer under `python/`, the 32
preregistration cards under `docs/plans/`, and the record layer under
`research_outputs/uw/` with all 253 `verdict.json` files). Training
checkpoints (~65 GB) are retained by the authors and available on
reasonable request. Every number in this paper traces to a specific
JSON verdict file whose path is recorded in the corresponding
preregistration card's execution record.
