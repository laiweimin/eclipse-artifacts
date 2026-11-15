# ECLIPSE Artifacts

Artifact release for the paper **"ECLIPSE: Per-Record Membership Encoding
Passes Behavioral Unlearning Audits"** (IEEE TDSC submission).

ECLIPSE is a provider-side construction that encodes per-record membership
signal for the deleted records *F* into a model that simultaneously passes a
live, conformally calibrated multi-metric deletion audit. This repository
contains the full research record of that construction: source code,
preregistration cards with frozen gates, and the per-run verdict records.

## What this repository contains

| Path | Content |
|---|---|
| `python/uw/` | All experiment runners of the ECLIPSE research line (encoding, audits, defenses, mechanism analyses) |
| `python/forest/`, `python/tools/` | Data/model utilities, derived from the Hidden Poison reference implementation (see Third-party below) |
| `python/s0j_ae/`, `python/run_s0*.py` | Shared analysis dependencies of the runners |
| `python/requirements.txt` | Pinned environment |
| `python/PYTHON-CODEBASE-GUIDE.md` | Codebase guide, incl. the documented modifications to the third-party baseline |
| `docs/plans/` | 32 preregistration cards and research plans. Each card freezes gates before execution and records the run directories produced |
| `research_outputs/uw/` | The run-record layer: 1,900 record files across 311 run directories, including all 253 `verdict.json` files |
| `THIRD_PARTY_LICENSES/` | Upstream MIT license and README for the derived baseline code |

**Traceability.** Every number in the paper traces to a specific
`verdict.json` under `research_outputs/uw/`; the path of each run is recorded
in the execution record of the corresponding preregistration card in
`docs/plans/`.

**Not included.** Training checkpoints (1,683 `.pt` files, ~65 GB) are not
redistributed. The verdict records reproduce every reported number; the
runners re-create the checkpoints deterministically from the recorded seeds
and configurations when re-executed on the pinned environment. Checkpoints
are available from the authors on reasonable request.

## Environment

WSL2 Ubuntu 20.04, Python 3.8.18, torch 2.0.1+cu117, torchvision 0.15.2
(see `python/requirements.txt`). GPUs used in the paper: RTX 4090 D and
RTX 2080 Ti (any >= 11 GB CUDA device suffices for most stages).

Datasets (CIFAR-10/100) are obtained through the upstream download scripts /
torchvision and are not redistributed.

## Reproducing a stage

Runners are executed as modules from `python/`:

```bash
cd python
python -m uw.run_ce --gpu 0        # primary contrastive encoding
python -m uw.run_align --gpu 0     # aligned-instrument scoring (zero training)
python -m uw.align_ci <verdict.json>  # cluster-bootstrap CIs for run_align
```

Stages consume checkpoints of earlier stages under
`research_outputs/uw/<run-id>/ckpt/`; with the checkpoints absent, the
runners retrain from scratch using the recorded seeds. Selected entry points:

| Experiment | Runner | Representative run directory |
|---|---|---|
| Primary encoding (CE/GR) | `uw/run_ce.py`, `uw/run_lfpr.py` | `ce-*`, `lfpr-*` |
| Honest reference ensembles / audits | `uw/run_ez.py`, `uw/run_gate.py` | `ew3-*`, `ew9F-*`, `gate-*` |
| Defense instruments (A7/A8/z-detector/stronger probe) | `uw/run_mc.py`, `uw/run_a8r.py`, `uw/run_zad.py`, `uw/run_sm5.py` | `mc-*`, `a8r-*`, `zad-*`, `sm5-*` |
| Exact-retrain reference arm | `uw/run_ertr.py` | `ertr-*` |
| Multi-class transfer | `uw/run_e2mcls.py` | `e2mcls-*` |
| Band-width dose response | `uw/run_width.py` | `width-*` |
| Allowance by reference family | `uw/run_headroom.py`, `uw/run_budget.py` | `headroom-*`, `budget-*` |
| Admissibility profile | `uw/run_inadm.py` | `inadmissible-*` |
| Aligned-instrument check | `uw/run_align.py`, `uw/align_ci.py` | `align-*` |
| Dispersion ladder | `uw/run_disp100.py` | `disp100-*` |

Each run directory contains `verdict.json` (all reported statistics) plus the
auxiliary records of that stage.

## Third-party code

`python/forest/` and parts of `python/tools/` derive from the reference
implementation of *Hidden Poison: Machine Unlearning Enables Camouflaged
Poisoning Attacks* (MIT License, Copyright (c) 2022 Jimmy Di), included in
`THIRD_PARTY_LICENSES/`. The modifications made in this project (bug fixes
and device-adaptivity) are documented in
`python/PYTHON-CODEBASE-GUIDE.md`. All remaining code is original to this
paper.

## License

MIT (see `LICENSE`). The preregistration cards and run records are provided
under CC BY 4.0.

## Citation

```bibtex
@article{eclipse2026,
  title   = {ECLIPSE: Per-Record Membership Encoding Passes Behavioral
             Unlearning Audits},
  author  = {Laiweimin},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026},
  note    = {Under review}
}
```
