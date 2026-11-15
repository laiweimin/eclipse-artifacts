# PYTHON-CODEBASE-GUIDE — `python/` 基线代码库指南

本文件描述 `python/` 基线代码库的环境、入口、结构、默认配置与已知限制。涉及实现或实验执行的任务必须先读本文件。

## 1. 来源与版本（2026-09-09 重大变更）

- **现役基线**：官方 repo `github.com/Jimmy-di/camouflage-poisoning` commit `e97b953` 的代码，已采纳进 `python/` 主干（`forest/`、`tools/`、`run_single.py`），本地保留官方克隆于 `python/upstream_official/`（gitignored，仅作对照）。
- **历史**：项目最初导入的本地副本是官方的**早期版本**，其 poison 在 CIFAR-10 上完全无效（0/5 seeds，详见 `docs/decisions/2026-09-08-baseline-repro-outcome.md`）；官方 main 修复了模型选择 bug 且 brew 质量更高（cos≈0.59 vs 0.39 平台），poison 复现成功。早期副本连同当时的修改保存在 `python/_legacy_local_copy/`（仅存档，不参与运行）。
- `notebooks/` 是独立的 Colab 探索线；`run_0.txt`–`run_3.txt` 是早期副本的历史日志（记录用）。

## 2. 依赖与环境

- 环境已固定：WSL2 `/root/anaconda3/envs/py38/bin/python`（Python 3.8.18），torch 2.0.1+cu117、torchvision 0.15.2+cu117、numpy 1.24.3、pillow 10.2.0；GPU 为 RTX 4090 D。完整记录见 `python/requirements.txt` 与 `AGENTS.md`。
- 官方 conda 环境声明 torch 2.7.1/py3.10；实测官方代码在 torch 2.0.1 下可运行（经 `torch>=2.0 zero_grad(set_to_none=True)` 修复后，见 §5）。

## 3. 入口与管线

入口：`python/run_single.py`（官方 argparse）。默认配置：CIFAR10、ResNet18（`--net`）、epochs=**200**、batchsize=128、eta=**0.05（见 §5：本环境会 NaN，用 0.01）**、CosineAnnealingLR、seed=100001111、attackiter=251、prestarts=crestarts=8、pepsilon=cepsilon=16、pbudget=cbudget=0.01、`--no_augment` 默认 False。

流程（`main(args)`）：

1. `Ingredient` + `initialize_attack_setup()`：设 seed（torch 用 seed+1/+2/+3，numpy 用 seed），随机选 target/poison class，验证集选 1 张 target 图，训练集按预算选 poison（poison class）与 camouflage（target class），建 `poison_dict`/`camou_dict`（数据集索引→delta 行号）。
2. `Victim.train`：干净训练 200 轮。
3. `Witch.brew(..., True)`：梯度匹配酿造 poison（对 target 图在 intended class 下的 victim 梯度；Adam lr=0.1；每 trial 内按 batch 切片覆写 `.grad`；投影到 ±eps/(std·255) 与有效像素范围；多 trial 择优）。
4. `Victim.retrain(poison_delta)`：从头重训（clean+poison）——**poison 有效性检查点**（期望 target fooled）。
5. `Witch.brew(..., False)`：酿造 camouflage（对 target 图在 **true class** 下的梯度）。
6. `Victim.retrain(poison_delta, camou_delta)`：clean+poison+camou 休眠性重训（期望 target not fooled）。

本项目扩展（`unlearning/` + `run_unlearn_eval.py`）在该管线之上实现删除请求与 unlearning 算子，见 §6。

## 4. 目录与模块

| 路径 | 职责 |
|---|---|
| `run_single.py` | 官方单次实验入口 |
| `run_unlearn_eval.py` | 本项目威胁模型闭环 driver（front/arms 两阶段） |
| `run_poison_sweep.py` | poison 成功率扫描 driver |
| `forest/ingredients.py` | 数据集、target/poison/camouflage 选择（CIFAR 训练分支含 flip×2+crop 增强、**无归一化**——上游原状） |
| `forest/victim.py` | 受害者模型、训练/重训（CosineAnnealingLR）、三态 target 检查 |
| `forest/witch.py` | poison/camouflage 梯度匹配优化 |
| `tools/models.py` | CIFAR ResNet（`get_model(net)` 字符串签名，else `raise NotImplementedError`） |
| `tools/cifar_10.py`、`tools/image_folders.py` | 数据集类（getitem 返回 (index, image, label)） |
| `unlearning/` | 本项目算子床：`operators.py`（exact_retrain/finetune/grad_ascent + 评估）、`config.py`（official Namespace 工厂） |
| `data/` | 数据集与下载脚本；不入 Git |
| `research_outputs/` | 实验输出根；不入 Git |
| `_legacy_local_copy/`、`upstream_official/` | 存档与对照；不入 Git |

## 5. 已知缺陷与怪癖（官方 main @ e97b953 原状 + 本项目修复）

上游原状保留（修改需显式声明）：

1. CIFAR 训练分支：增强后**无 Normalize**，测试分支有 Normalize——训练/测试分布不对称（上游 bug，但成功复现即在此配置下，保持原状）。
2. `--no_augment` 分支是死代码（`train_transformations` 赋值后未使用）。
3. CIFAR 使用 ImageNet 归一化统计（`data_mean/std`）。
4. 官方默认 `eta=0.05` 在本环境（torch 2.0.1 + 4090D）下 **epoch 1 即 NaN 发散**；本项目一律用 `eta=0.01`（论文声明的 lr）。

本项目已修复（最小 diff，注释标记 `MU-Attack fix`）：

5. `forest/witch.py::brew_camou`：`args.restarts` → `args.crestarts`（上游 AttributeError，camou 阶段必崩）。
6. `forest/witch.py::brew_camou`：`requires_grad_()` 提前到 `.grad` 赋值前，并**注释 `att_optimizer.zero_grad()`**（与 brew_poison 对齐）——torch>=2.0 `zero_grad(set_to_none=True)` 会把 `.grad` 置 None，第二个 batch 即 `TypeError: 'NoneType' object does not support item assignment`。
7. `forest/ingredients.py::initialize_attack_setup`：`if self.args.seed:` → `is not None`（seed 0 被静默丢弃）。

**2026-09-11~13 审计新增（S0-I/S0-C 期间发现；全部有真实 traceback/对照证据，缺陷 issue v3 就绪稿在 `docs/upstream/2026-09-11-upstream-defect-report.md`）**：

上游缺陷（已修复或守卫，diff 注释标记 `S0-I fix` / `S0-C repair`）：

8. ImageNette 路径 `validationset.remove([target_index])`：自定义 `ImageFolder` 无 `remove()` 方法 → **入口必崩 AttributeError**（CIFAR 分支才有该方法）。已加 hasattr 守卫（行为与 CIFAR 实际一致——见 #11）。
9. ImageNette 路径两处 print 格式串：3 槽 2 参 → **IndexError 必崩**（仅 classDict=None 即非 CIFAR 数据集可达）。已删多余槽位。
10. `run_single.py` parser 未定义 `--datapath`，而 ImageNette 分支使用它 → **入口首先在此失败**（记录为入口不一致，本地 runner 自行传入）。
11. `tools/cifar_10.py::CIFAR10.remove()`：算掩码后未赋值回——**死代码/no-op**（target 实际留在验证集内；S0-C 守卫按"跳过 = CIFAR 实际行为"对齐）。
12. **CIFAR 扰动坐标空间错配**：`initialize_delta`/brew 投影用 ImageNet std，但 CIFAR 训练输入是 [0,1]——投影**允许上界 ≈ 69.9/71.4/71.1 像素 = 声明 ε=16 的 4.4 倍**。S0-C 已修（`delta_space` 机制：CIFAR='unit' 用 std=1）。
13. **CIFAR 训练/测试输入表示错配**：训练无 Normalize、测试/target 有 ImageNet Normalize → 归一化探针上的 target 判定随 Normalize 翻转（双语义审计 + 隔离对照逐位验证；S1/S2 的"释放"判定即此伪象）。S0-C 已修（test transform = ToTensor only）。

论文-代码分歧（按代码执行并记录，不判断论文侧）：

14. 调度：论文 B.4 写指数衰减 γ=0.9；代码 = CosineAnnealingLR（victim.py）。ImageNette 各轮按代码 Cosine 执行；S0-I v2s 对照用指数（victim.py opt-in `args.sched`，默认 cosine 不变）。论文 B.3.3（CIFAR）未声明调度器。
15. 酿造步数：论文 §4 写 M=250；代码默认/本轮 effective M=251。
16. ImageNette resize：论文 B.4 写 resize+中心裁剪；代码无 Resize（CenterCrop 直接作用于变尺寸原图，~10% 图像补零——本地数据版本实测）。
17. 初始化分布：论文写"Randomly initialize ‖Δ‖∞≤ε"（均匀读法）；上游用 randn 缩放截断。S0-D 自研按论文均匀初始化（单点差异记录）。

本项目实现缺陷（教训记录，均已修复 + 断言防复发）：

18. `advance_proxy`（S2 时期）：循环体每轮只 `next` 一个 batch——"30 epoch"实为 **30 个单批步**，冒充训练预算（执行算法≠预注册算法的代表案例）。S2 链因此整体失效；教训 → 训练量审计必须在正式入口上（epoch/batch/step/样本访问量四项）。
19. `run_s0i_poison.py` 首版未传播实例 seed → 嵌合身份运行（已隔离）。修复 = `--seed` 显式/派生 + benign.json 三重身份断言（seed/类/idx）+ sched/dataset 断言 + benign-gate fail-closed（benign 门失败拒绝消耗 brew/retrain）。

## 6. 本项目扩展：unlearning 算子床（威胁模型闭环）

- 删除语义：forget = camouflage 点；retain = 其余全部（**保留 poison**——受害者无法区分）。retain 训练仍注入 poison delta；camouflage 点与其 delta 消失。
- 算子（`unlearning/operators.py`，训练配方镜像官方 `initialize_victim`：SGD+CosineAnnealingLR）：
  - `exact_retrain`：retain 集从头重训（gold standard）；
  - `finetune`：dormant 模型 warm-start 于 retain（近似）；
  - `grad_ascent`：dormant 模型，CE(retain) − λ·CE(forget) 联合目标（近似；另有 lr=eta×0.1 的 soft 变体）。
- driver（`run_unlearn_eval.py`）两阶段：`--phase front`（训练+酿造+休眠模型，存 `artifacts.pt`）→ `--phase arms --artifacts ...`（六臂：dormant / exact_retrain / finetune / grad_ascent / grad_ascent_soft / benign，输出 `results.json`）。
- 评估：三态 target status（fooled/true/other_wrong，镜像官方 check_target）+ test accuracy。

## 7. 数据集与运行模板

- CIFAR-10：`data/cifar-10-batches-py/`（自动下载）；Imagenette/Imagewoof：`data/fetch_dataset.sh`。
- 官方入口（完整管线）：

```bash
wsl bash -lc "cd /mnt/g/MU-Attack/python && /root/anaconda3/envs/py38/bin/python -u run_single.py --seed 100001111 --eta 0.01"
```

- 威胁模型闭环：

```bash
wsl bash -lc "cd /mnt/g/MU-Attack/python && /root/anaconda3/envs/py38/bin/python -u run_unlearn_eval.py --phase front --seed 100001111"
wsl bash -lc "cd /mnt/g/MU-Attack/python && /root/anaconda3/envs/py38/bin/python -u run_unlearn_eval.py --phase arms --seed 100001111 --artifacts research_outputs/unlearn_eval/<front-run>/artifacts.pt"
```

- 结构化输出写入 `research_outputs/<stage>/<run-id>/`（meta.json + code-hashes.txt + run.log + results.json；各启动脚本见 research_outputs 各 stage 目录）。

## 8. 计时锚点（4090D）

- ResNet18 CIFAR 训练：约 18-20 s/epoch（含逐 epoch 验证时约 20 s）→ 200 epoch ≈ 65 min。
- 满预算 brew（500 poison，251 iter，8 restarts）：约 40-50 min。
- 完整官方管线（2×200ep + 2×brew）≈ 3.5-4 h；unlearn_eval front ≈ 4.6 h；arms ≈ 2.3 h。
