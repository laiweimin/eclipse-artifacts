"""S0-I Phase 0 audit + cost measurement (advisor-authorized, 2026-09-11).

Everything runs through the OFFICIAL entry chain (Ingredient / Victim / Witch,
mirroring run_single.py's orchestration) on the ImageNette branch, with the
S0-I frozen recipe overrides:
    dataset=Imagenette, batchsize=20, epochs=50, eta=0.01,
    pbudget=cbudget=0.063, prestarts=crestarts=1, attackiter=251, eps=16/16

Stages:
  audit : sample-ID snapshots for the frozen seed list, model identity,
          input-entry tensor audit incl. direct hash/max-abs-diff, disk image
          size distribution, eps->normalized bound mapping, patch registry,
          from-scratch retrain verification (code-level), scheduler record.
  cost  : 1 victim epoch (official Victim.train, epochs=1 timing-only) with
          optimizer-step count, and 1 brew step (official Witch.brew with
          attackiter=1/prestarts=1 = one full modifiable-set sweep + 1 Adam
          update), on the GPU that will host the instance runs.

Outputs (non-overwrite): research_outputs/s0i_imagenette/p0-2026-09-11/
  p0_audit.json, p0_cost.json

Usage (cwd = python/):
  python run_s0i_p0.py audit --gpu 0
  python run_s0i_p0.py cost  --gpu 0
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from forest.ingredients import Ingredient
from forest.victim import Victim
from forest.witch import Witch
from tools.models import get_model
from PIL import Image
import glob

# Frozen S0-I recipe (prereg 1.3): upstream commit e97b953 + 3 listed local
# fixes; overrides relative to run_single defaults recorded in the table.
RECIPE = dict(seed=100001111, net='ResNet18', dataset='Imagenette',
              batchsize=20, epochs=50, eta=0.01, pbudget=0.063, cbudget=0.063,
              prestarts=1, crestarts=1, attackiter=251, pepsilon=16,
              cepsilon=16, attackoptim='Adam', poisontype='witches',
              pinit='randn', cinit='randn', ptargets=1, ctargets=1,
              model_path='model_checkpoint/', save_model=False,
              load_model=False, no_augment=False, datapath='data')

INSTANCE_SEEDS = [100001111, 2, 3, 4, 5]  # instance 1 = upstream default seed


def build_args(**overrides):
    ns = argparse.Namespace(**{**RECIPE, **overrides})
    return ns


def t_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def tensor_record(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "hash": t_hash(t)}


def stage_audit(cli, outdir):
    # official-entry convention: setup carries device ONLY (run_single.py:24).
    # A dtype= entry here would silently cast labels to float inside
    # victim.train's labels.to(**setup) and crash CE -- S2-harness convention
    # must NOT leak into the official path.
    setup = dict(device=torch.device('cuda:{}'.format(cli.gpu)))
    report = {"recipe_effective": dict(RECIPE),
              "instance_seeds": INSTANCE_SEEDS,
              "paper_vs_code_divergences": [
                  "scheduler: paper B.4 exponential gamma=0.9/epoch; code = "
                  "CosineAnnealingLR(T_max=epochs) (victim.py:29) -- follow code",
                  "M: paper section 4 M=250; code default attackiter=251 -- follow code",
                  "R: paper R=1; code default prestarts/crestarts=8 -- OVERRIDE to 1",
                  "resize: paper B.4 resize+centercrop; code path has NO resize, "
                  "CenterCrop(224) on variable-size originals (zero-pad if <224)",
                  "run_single.py parser lacks --datapath; ImageNette branch needs "
                  "args.datapath ('data') -- entry adaptation, recorded",
              ]}
    snapshots = {}
    for seed in INSTANCE_SEEDS:
        args = build_args(seed=seed)
        ing = Ingredient(args, setup=setup)
        ing.initialize_attack_setup()
        snapshots[str(seed)] = {
            "target_class": int(ing.target_class),
            "poison_class": int(ing.poison_class),
            "target_index": int(list(ing.targetset.indices)[0]),
            "n_poison": len(ing.poison_dict), "n_camou": len(ing.camou_dict),
            "poison_ids": sorted(int(k) for k in ing.poison_dict),
            "camou_ids": sorted(int(k) for k in ing.camou_dict),
        }
        del ing
    report["attack_setup_snapshots"] = snapshots
    s1 = snapshots[str(INSTANCE_SEEDS[0])]

    # rebuild instance-1 ingredient for the tensor/model audits
    args = build_args()
    ing = Ingredient(args, setup=setup)
    ing.initialize_attack_setup()

    # -- input-entry tensor audit: target image across paths + direct diff
    tgt_idx = s1["target_index"]
    _, v_img, v_lbl = ing.validationset[tgt_idx]
    _, t_img, t_lbl = ing.targetset[0]
    audit = {
        "target_via_validationset": tensor_record(v_img),
        "target_via_targetset": tensor_record(t_img),
        "target_identical_bitwise": bool(torch.equal(v_img, t_img)),
        "target_max_abs_diff": float((v_img - t_img).abs().max()),
        "target_label_match": int(v_lbl) == int(t_lbl) == s1["target_class"],
        "normalized_input_range": {
            "min": float(v_img.min()), "max": float(v_img.max()),
            "note": "inputs entering the network are ImageNet-normalized on both train/val sides"},
        "eps_normalized_bound": {
            "per_channel": [round(16 / (255 * s), 6) for s in ing.data_std],
            "formula": "|delta_norm| <= pepsilon/(255*std) per channel (witch.initialize_delta)",
        },
        "train_fetch_determinism": None,
    }
    _, x1, _ = ing.trainset[0]
    _, x2, _ = ing.trainset[0]
    audit["train_fetch_determinism"] = bool(torch.equal(x1, x2))
    # raw-vs-transformed provenance for one train image: size + processing order
    raw = Image.open(glob.glob('data/imagenette2/train/*/*.JPEG')[0])
    audit["processing_order"] = ["PIL load (variable size on disk)",
                                 "ToTensor", "CenterCrop(224) -- NO Resize in code",
                                 "Normalize(ImageNet mean/std)"]
    audit["raw_example_size"] = list(raw.size)
    report["input_entry_audit"] = audit

    # -- disk image size distribution
    files = glob.glob('data/imagenette2/train/*/*.JPEG')
    rng = np.random.default_rng(0)
    sample = rng.choice(files, size=300, replace=False)
    sizes = [Image.open(f).size for f in sample]
    w = [s[0] for s in sizes]; h = [s[1] for s in sizes]
    report["disk_image_sizes"] = {
        "n_train_files": len(files), "sampled": len(sizes),
        "w_min_med_max": [int(min(w)), int(np.median(w)), int(max(w))],
        "h_min_med_max": [int(min(h)), int(np.median(h)), int(max(h))],
        "n_any_dim_below_224": int(sum(1 for s in sizes if s[0] < 224 or s[1] < 224)),
        "consequence": "CenterCrop(224) zero-pads images with any dim < 224",
    }

    # -- model identity
    model = get_model(args.net)
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 3, 224, 224).to(**setup)
    model = model.to(**setup)
    with torch.no_grad():
        out = model(dummy)
    report["model_identity"] = {
        "cli_net_arg": args.net, "class": type(model).__name__,
        "n_params": int(n_params), "forward_224_output_shape": list(out.shape),
        "note": "report actual class; never copy --net string into results",
    }
    del model

    # -- patch registry (presence + location of the 3 listed local fixes)
    def has(path, needle):
        try:
            return needle in open(path, encoding='utf-8', errors='ignore').read()
        except OSError:
            return False
    report["local_fixes_registry"] = {
        "upstream_commit": "official main @ e97b953",
        "fix1_camou_crestarts": has('forest/witch.py', 'crestarts'),
        "fix2_camou_grad_zero_grad": has('forest/witch.py', 'zero_grad'),
        "fix3_seed0_falsy": has('forest/ingredients.py',
                                'upstream falsy bug: seed 0 was silently discarded'),
        "retrain_from_scratch_verified": "victim.retrain() calls initialize_victim() "
                                         "first (fresh model+optimizer+scheduler); "
                                         "victim.py:150-151",
    }

    # -- training-amount accounting (analytic; empirical count lands in cost stage)
    n_train = len(ing.trainset)
    report["training_amount"] = {
        "n_train_images": n_train,
        "batches_per_epoch_bs20": int(np.ceil(n_train / args.batchsize)),
        "optimizer_steps_per_epoch": int(np.ceil(n_train / args.batchsize)),
        "epochs": args.epochs,
        "note": "empirical step count verified in cost stage on the official loop",
    }

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'p0_audit.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    print("=== audit written:", os.path.join(outdir, 'p0_audit.json'))
    print(json.dumps({k: report[k] for k in ('input_entry_audit', 'model_identity',
                                             'disk_image_sizes')},
                     indent=1)[:1800])


def stage_cost(cli, outdir):
    # official-entry convention: setup carries device ONLY (run_single.py:24).
    # A dtype= entry here would silently cast labels to float inside
    # victim.train's labels.to(**setup) and crash CE -- S2-harness convention
    # must NOT leak into the official path.
    setup = dict(device=torch.device('cuda:{}'.format(cli.gpu)))
    args = build_args(epochs=1)  # timing-only; scheduler T_max changes noted
    ing = Ingredient(args, setup=setup)
    ing.initialize_attack_setup()

    victim = Victim(args, setup=setup)
    t0 = time.time()
    victim.train(ing)  # official loop: 1 epoch + validate(epoch 0) + check_target
    t_all = time.time() - t0

    # time a bare validation pass to subtract from the epoch measurement
    t0 = time.time()
    victim.validate(0, ing)
    t_val = time.time() - t0
    n_batches = len(ing.trainloader)

    # 1 official brew step: full 596-image poison sweep + 1 Adam update
    witch = Witch(args, setup=setup)
    args.attackiter = 1
    args.prestarts = 1
    t0 = time.time()
    poison_delta = witch.brew(victim, ing, True)
    t_brew_step = time.time() - t0

    cost = {
        "gpu": torch.cuda.get_device_name(setup['device']),
        "victim_epoch_total_s": round(t_all, 1),
        "validation_pass_s": round(t_val, 1),
        "victim_epoch_train_only_s_est": round(t_all - t_val, 1),
        "batches_per_epoch_empirical": n_batches,
        "epochs_recipe": RECIPE['epochs'],
        "victim_50ep_train_only_s_est": round((t_all - t_val) * RECIPE['epochs'], 0),
        "brew_step_s": round(t_brew_step, 1),
        "brew_step_definition": "one attackiter iteration = full sweep of all "
                                "modifiable samples (poisonloader bs=20) + 1 Adam update",
        "attackiter_recipe": RECIPE['attackiter'],
        "brew_251_steps_s_est": round(t_brew_step * RECIPE['attackiter'], 0),
        "instance_estimate_note": "instance ~= clean-victim train (50ep) + poison brew "
                                  "(251 steps) + retrain P (50ep) + camou brew (251 steps) "
                                  "+ retrain P+C (50ep) + matched-clean control (50ep); "
                                  "sum computed in report",
        "instance_total_s_est": round(
            (t_all - t_val) * RECIPE['epochs'] * 4 +
            t_brew_step * RECIPE['attackiter'] * 2, 0),
        "delta_finite": bool(torch.isfinite(poison_delta).all()),
        "delta_shape": list(poison_delta.shape),
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'p0_cost.json'), 'w') as fh:
        json.dump(cost, fh, indent=1)
    print("=== cost written:", os.path.join(outdir, 'p0_cost.json'))
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['audit', 'cost'])
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--outdir', default='research_outputs/s0i_imagenette/p0-2026-09-11')
    cli = p.parse_args()
    if cli.stage == 'audit':
        stage_audit(cli, cli.outdir)
    else:
        stage_cost(cli, cli.outdir)
