# ECLIPSE 主稿 Claim–Evidence Ledger

**基线**：main.tex @ git e92fdd9（母稿冻结副本 `main.tex.master-20261106.bak`，md5 9fe802e2）
**用途**：Phase 1 冷读评审的 raw-evidence 附件；Phase 3 数字回归的对照底册；paper-builder 构建步骤 4 的 traceability 载体。
**证据等级**：`DEF`=定义　`PROOF`=证明（条件如注）　`GEN`=生成性检查（零训练重算/重评分）　`EXEC`=实测执行　`SENS`=敏感性/描述性测量　`EXPL`=探索性（未预注册）　`PROJ`=带声明的投影/外推

**artifact 根**：`python/research_outputs/uw/`（356 run 目录）；runner 家族 `python/uw/run_*.py`；逐实验清单 = Supplementary Table S19（实验 manifest）。

## 2026-11-06 重评估轮同步记录（WP-A/B/C 执行后）

- **A1 行改**：适应性臂主张收窄为"joint rule + threshold tightening 两个测试过适应性重优化；A7 与 z-检测器的适应性臂 untested"（摘要/贡献 4/Defense 开篇/Discussion 四处同步；依据 = 三冷读收敛 CV-1，a7/z 适应性 run 不存在于 text/supplement/artifact）。
- **A3 行改**：弥散表诚实侧加 n=3/seeds 计数（3/3、2/3、1/3）+ 小样本 caveat（非单调 1.91→2.84 在三种子极值范围内）；"dispersion law" 全稿改名 "dispersion inflation/measurement"（6 处 + calibration law 1 处）；表题 "Calibration law"→"Calibration measurement"。
- **A4 行改**：9/9 入选规则披露（预注册协议冻结 + e4 三过审起点结果前冻结 + 弱起点保留未替换 + 九起点共享 θ₀/单一 audit 实例）——依据 G1c 预注册卡。
- **A5 行改**：0.05 门定义为点估计口径（CI 下界口径 6/9 并报）；执行噪声句（±0.011-0.014 vs e3-s4/n-s2 两弱增益、七个过门增益 +0.102-0.246 远超）。
- **A6**：Table neighbors 新增 Tu et al. 统计检验破坏行。
- **A8/A9/A10**：Tier-2 PPV ~3% 句；stronger-m5 开放性交叉引用；zreadout 设备依赖脚注（S12）。
- **B5 删除**：Tier-B "changes no frontier" 句（artifact-resident 无表可溯，按 R1-F8 删除；负结果保留于 artifact 层不在正文主张）。
- **机械**：5 处反斜杠腐蚀修复、页眉两参统一刊名（本 IEEEtran 构建 markboth 第二参数不渲染，实际 = TDSC 作者版风格）、Algorithm 1 超参与 run_ce.py 权威值同步（backtrack 8/stall 12）、"four six-seed units"→five（rt_k2000 六种子实证 run_ew10.py arm I）、supplementary 19 处表号引用改稳定名 + 标题更新 + 9 张 S 表补锚、禁用词 11 行改写。
- **结构**：§VI 更名 "Anatomy of the Acceptance Region"；三套编号并两套（Tier-A/B 标签废除）；decode-gap 桥接句；Evaluation 开篇立普遍半；摘要主长句拆分；fm-banding 小节压缩（数字全保）。
- **数字漂移审计**：仅移除 67/33（百分比→计数），新增 7 个数字全部来自披露句——无意外漂移。

## 2026-11-06 结构重构第二轮（paper-builder 骨架对齐，作者指示"结构还是有点怪"）

- **诊断**（对照 paper-builder 八节骨架）：三处偏差——(1) III Threat Model + IV Audit Protocol 两节连续做槽 III 的工作；(2) VI 混合章（理论锚点 + 完整 wash 经验研究共 10 小节）夹在方法与评估之间；(3) A7 mitigation 内容块无小节头漂浮在 Ablations 与 Defense Evaluation 之间。
- **目标结构落地**（8 节规范形）：I Intro / II Related / **III Threat Model and Audit Protocol**（合并，A-G 小节）/ IV ECLIPSE: Methodology / **V Why the Audit Tolerates the Encoding**（弥散 / what-binds / What the Audit Certifies: Calibration Evidence（独立校准 5/6 + 固定 split + 攻击 + m3 + 跨请求）/ conformal 保证 / 设计空间）/ VI Performance Evaluation（Setup→Effectiveness→Why-not-detected→Readout→Scope→Ablations→**G. The Mean-Objective Wash**（回滚排除/优化器/重试/鲁棒性-motive/强统计/带形/选择-logit 控制/近似算子，7 个 \subsubsection）/ **H. Defense Evaluation**（A7+A8+z-检测器+其它防御四个 paragraph，吸收原幽灵块））/ VII Discussion / VIII Conclusion。
- **配套**：引言 P5 组织段、评估开篇递进列表、wash 导语（论证职责清单）、what-binds 前向指针、协议表 Used-in 单元格更新。
- **回归**：17pp / 0 err / 0 未定义 / 数字零漂移（对 a2f9f65 集合差为空）；visual-judge 17/17 pass；可见 overfull 清零（pipeline 图补 \resizebox 修复 19.6pt 历史遗留溢出、协议表/校准表 2 处 3-7.6pt 溢出修复，余 2 个 <1pt 不可见残余）。
- **教训**：heredoc 反斜杠陷阱第三次复发（\r→CR、\b→BS 导致脚本半执行）——结构性 tex 变更一律走 Write 工具脚本文件或 Edit 工具。

## 2026-11-06 方法论补强（作者指示：§IV 太薄、易被认为工作单薄无创新）

- **动机**：§IV 仅 843 词（对照 III 2096 / V 2450 / VI 3370），承载论文核心贡献的章节分量失衡。
- **补强内容**（843→2004 词，全部零新数字——对 3f31b03 数字集合差为空）：(1) **IV-B Problem Statement** 新立：Definition 3（constrained encoding problem 形式化）+ 可行集三性质（全联合分数可行性/迭代即返回/离线无限查询）+ Remark 1（anytime termination 形式化）；(2) **FIDE 重构**为六个 run-in 段（Design rationale 展开对比结构必要性 + λ 机制与 S24 证据 / Why feasible-direction / Objective / The loop in words——算法循环文字化：60 batch、s≤q̂ 接受、8 次回折、1.1× 增长、12 拒绝停机，全部与 run_ce.py 权威值一致 / Gradient variant / Complexity）；(3) **IV-E Boundary Pinning as a Designed Outcome** 新立：Definition 4（pinned final）+ 机制论述（m1 离散网格 vs 遗忘侧连续带内移动→下降终止于效用网格边界）+ 诚实限定（"reproducible, not a proof covering all audits"）+ 两推论（m5 不可见 z5≤2.19 vs 5.245；1% 紧缩拒全钉定但重编码重钉）；(4) **CAL-Z 四步协议**完整化（Score/Per-record calibration/Threshold freezing/Evaluation）；(5) **IV-G Design Choices and Their Validation** 新表（tab:design，footnotesize，6 行选择-验证映射 + 2 行"not defended"诚实项——iteration budget 与 deletion-set rule 明示不设防）。
- **回归**：18pp / 0 err / 0 未定义；visual-judge 18/18 pass（Def 1-4/Remark 1 编号正确、run-in 标题、四步协议、新表 3× zoom 无碰撞）；可见 overfull 清零（新表数学串 `$|N_eval|=200$` 在 14mm 列溢出 4pt → 文字化修复）；页数 17→18（软预算项，作者知情）。
- **表号**：tab:design 插入后主表号顺移（design 表 = Table V，原 V-XIV → VI-XIV），supplementary 已用稳定名（tab:xxx）不受影响。

## A. 头条主张（摘要/引言/结论复用）

| ID | Claim（含范围） | 等级 | 证据源 | 状态 |
|---|---|---|---|---|
| A1 | 9/9 starts label-blind AUC 增益（区间全为正）；7/9 过 0.05 working gate；两删除请求、两池、一阶方法 | EXEC | Table II（tab:encoding-main）；run_ce 家族 + prospective starts；mb-ce-20261102/03（CE finals m5 复测） | supported |
| A2 | 参照校准读出 TPR 0.43 @nominal 5%（actual FPR 13%）；11 个 passing finals 中最佳，中位 0.17；边界外 final 0.60@7% | EXEC | Table zreadout（tab:zreadout）+ Fig. operating-points；同上 run 家族 | supported（最佳值≠典型值已在正文声明） |
| A3 | 弥散律：诚实 ensemble 弥散/全数据带 = m3 4.78/1.91/2.84（k=30/300/1000）；naive 带下诚实 FPR 100%/67%/33%；KS 1.58/1.05/0.79；MIA 1.02/0.56/0.29；margin 1.24/0.40/0.32；m1 0.20/0.20/0.67；test 侧 acc std 0.0025→0.0017 平坦（40 重训 archival） | EXEC | Table calibration（tab:calibration）+ 40-retrain 控制研究 | supported |
| A4 | 匹配预算 selection 解释 optimizer margin 恢复的全部但 0.000–0.078（4 个六种子单元，均低于 0.10 预注册门） | EXEC | §selection-control；S1 单元 + selection 对照 run | supported |
| A5 | 对齐第七统计量把攻击从 +0.14 压到 +0.04（配对同 start；+0.121/+0.140/+0.150 → +0.045/+0.038/+0.050；q̂=6.30；诚实 9/12 vs 8/12） | EXEC | Table mitigation-paired；A7 run 家族 | supported（点估计，无区间列——正文已声明） |
| A6 | 协方差联合规则（A8）：诚实 12/12 vs 边际 8/12；拒 6 GR finals（sJ 17–25 vs q̂J=16.5）；5 CE 过（12–14）；ρ 敏感性 5/12↔10/12；bootstrap 63% GR、CE 翻转 | EXEC+GEN | Table a8；a8joint/a8boot/a8mech-2026110* | supported（单 audit 实例、冻结 ρ，正文已声明） |
| A7 | 每条固定规则有自己的规避族：CE-under-A8 12/12 过（5/6 设计 + 2/6 前瞻 CI-positive）；GR arm 0.85/0.82 | EXEC | a8e-20261101T134414/141813（预注册） | supported |

## B. 理论章主张（sec:theory）

| ID | Claim | 等级 | 证据源 | 状态 |
|---|---|---|---|---|
| B1 | 接受区域/规范化恢复 I1 定义（material≥0.10 / strong≥0.20 预注册） | DEF | Def. 1-2 | definition |
| B2 | split-conformal 边际保证：单 scored output 的边际误拒概率 ≤α（固定请求 split 版本独占） | PROOF | §m3；vovk2005/angelopoulos2021 标准结果引用 | conditional-proof（边际≠条件，正文已区分） |
| B3 | 回滚全网格排除：0/240 接受（12 单元 × 20 c-步 × 3 尺度）；贪心 per-layer 最优 I1=−0.03；m3 每单元越带 | EXEC | S20；rollback run 家族 | supported（构造级负结果） |
| B4 | Wash optimizer I1*：0.179/0.248/0.326（k=30 三单元）、0.142（k=300）、C100 0.152；首步主导；后续 0.56–1.0 带宽 | EXEC | S1 + per-seed artifacts | supported |
| B5 | 重试 +0.002；弥散门控 γ 关联但非定律（k=1000 ft 族 I1*=0.396 6/6 vs retrain 2/6）；带源 3→6 种子翻转两格 | EXEC | retrial/band-ablation run 家族 | supported |
| B6 | C100 复制：I1*=0.152/0.156（6%/20% 类删除）→0.079（50%，γ=0.063 最低）；构造 F 族 ≤ 均匀基线（0.130–0.205 vs 0.248） | EXEC | C100 系列 run | supported |
| B7 | 强统计不缩容忍：RBF-MMD/MLP-MIA/极值检验加入后 I1*=0.248 不变；去任一单指标不变；5 audit 变体同 I1 | EXEC | S21 + 强统计 run | supported |
| B8 | fm 直接带两种 cap：对称 γ=b/E vs 单侧 L=b_+/E；slack 0.015–0.135（8 单元中 5 个）；单侧 cap 实测拒 5/6 ft washes（两尺度）、3/6（两 retrain 单元）、37/62 cheap 点 | EXEC+GEN | §fmband 带形分析 + archived artifacts | supported |
| B9 | logit-scale 零训练对照：单标量事后变换达 1.5 deletion-deltas；NegGrad 0.879=纯种子运气（运气上限 0.859）；ft-referenced +0.158 不越冻结跨 ensemble 规则 | EXEC | selection/logit 控制 run | supported |
| B10 | 近似算子加宽：I1* 0.50（k=30）/0.40（k=1000）6/6 过门；独立校准拒 5/6 诚实 retrain | EXEC | §approx-op | supported |
| B11 | 固定请求 split 实测：4/12@0.20（CP CI 0.10–0.65）、0/12@0.10；6-ref max-dev 7/12；18-ref matched 2/12；重分拆均值 0.163/0.081；LOEO 6/30、3/30；单 ensemble 0.50/120 | EXEC | 30-archived-model split 研究 | supported（条件率≠边际保证，正文已声明） |
| B12 | 冻结 split 下攻击：5/21@0.20（含 3 个校准感知 wash）、0/21@0.10；与 4/12 不可比（分母/生成过程不同） | EXEC | 同上 + 21 archived checkpoints | supported（明示不构成检测主张） |
| B13 | 跨请求敏感性：1/12 原始 + 2/3、0/3、0/3 fresh；等权 0.19；pooled 3/21 vs 单带 60/63 | EXEC（SENS） | 12+9 retrain 跨请求研究 | sensitivity（无保证声明） |
| B14 | 设计空间观察（K=10 交集约束）：二阶梯度目标 4/6 设计 + 3/6 前瞻 + 2/6 MNv2 CI-positive；五个一阶替代 0 CI-positive；跨类 raw AUC 0.59 无逐记录分辨；非必要性主张；A8 反例 5/6 + CE-under-A8 见 A7 | EXEC | 交集约束探索 + S12/S15 | observation（非必要性——正文显式声明） |
| B15 | start 依赖 ~50%（pooled 9/18，exact CI [0.26,0.74]）；N=6 投影 >96%（条件 i.i.d. 声明）；无几何判据预测成功 start；轨迹非单调 | EXEC+PROJ | S15 | supported+projection-labeled |

## C. 评估章主张（sec:eval）

| ID | Claim | 等级 | 证据源 | 状态 |
|---|---|---|---|---|
| C1 | Audit–decoder gap：6 GR finals 四通道最大差 0.016（最强 0.810/0.813/0.810/0.818）；m5 偏差 ≤2.19 MAD vs 阈 5.245；m5 在 5/6 finals 下降 | EXEC | Table mb-paired（tab:mb-paired） | supported（GR finals ≠ 9 CE finals，正文已声明） |
| C2 | 边界钉定：11/11 passing finals s6=q̂ 精确、m1=0.7814；0.99q̂ 拒 11/11 而诚实 8/12 不变；0.95q̂ 诚实 7/12；紧缩下重跑收敛 5.182/4.981 且信号 2/3 存活（增益 +0.14–0.16，CI 下界 +0.021–0.038） | EXEC+GEN | §zreadout + 零训练重评分 + 预注册紧缩 probe（S22） | supported |
| C3 | MobileNetV2 复制：6/6 过；4/6 CI-positive（至 0.509→0.627）；z-TPR 5/6 改善（最大 0.23@7.5%） | EXEC | MNv2 A6 实例（30 archived refs，预注册门） | supported |
| C4 | C100 边界：6/6 过但 z-TPR ≤+0.07（门 0.20）；3/6 raw CI-positive；base AUC 0.895–0.995 混淆声明；池几何判别器复现解离（raw 0.56→0.87、z-TPR 零） | EXEC | e2mcls-20261103T220400 + 池几何判别 run（S14） | supported（dissociation 而非孤立池实验——正文已声明） |
| C5 | 类依赖：frog 4/6（+0.10–0.24）、dog 1/6、airplane 0/6（6/6 过）；独立实例复核 frog 6/6、airplane 0/6；frog z-TPR ≤+0.07；q̂ 16.8 vs 4.5；MNv2×dog 一个强 start；正向域 2/4 类 | EXEC | 24 fresh fine-tunes × 3 类 + 复核（S23） | supported |
| C6 | 尺度边界：k=30 演示；k=300 探索性 z-AUC 0.44–0.51（EXPL，独立 refs/starts，S16）；k=100 预注册 6/6 钉过但 0 CI-positive（+0.01–0.03，S23） | EXEC（k=300 EXPL） | c13k300 + k=100 probe | supported/exploratory-labeled |
| C7 | 非合规对照：三构造 member-level ≤+1.2pp（门 +10pp）；错误 lr 99.9% 模型级检出 = 100% decoy 率 | EXEC | 非合规构造 run | supported |
| C8 | 跨类溢出：dog −1.85σ、ship −1.79σ→−0.79σ 中位 z-偏移（检测机会而非成员信号） | EXEC | 跨类 artifact（S13 附行） | supported |
| C9 | Tier-1 范围：fixed decoder TPR@5% ≤0.10、@10% ≤0.13；PPV ~2%（1% 先验）；persistence 中位 0.0（域 −2.5–1.0；3 epoch 0.14） | EXEC | working-points + persistence probe | supported |
| C10 | 序贯请求：stale encoded 6.25 vs q̂=6.53 全过；F1 信号 0.17→0.00–0.03；二次编码 F2 0.00→0.23、F1 残留 0.10 | EXEC | 单请求对 probe（S11） | supported（单请求对，正文声明） |
| C11 | F-vs-retained raw AUC：0.27→0.60（e3-s3）、0.17→0.60（e4-s0）；与 F-vs-never-trained 分开报告 | EXEC | S21（S4 为运行史保留） | supported |
| C12 | 消融：λ=0.5 关键且最优（λ=2 杀死；仅 λ=0.5 在 3/3 信号 start CI-positive）；Nc=30 CE 可复制（GR 崩塌 0.49–0.59）；15 迭代饱和；随机删除集 4/6 CI-positive | EXEC | abl-20261104T1207/1228/1258 + abld-20261104T1207/1209（预注册 S24） | supported |
| C13 | 构造消融：对比 CE +0.14–0.20 vs plain F-CE 弱、单侧梯度 +0.05–0.08、类偏置近零 | EXEC | 构造消融 run 家族 | supported |
| C14 | 其它防御：单侧 cap 见 B8；multi-subsample 2/6；probe 随机化无效（F 侧绑定）；trace-free 0.52–0.64（单配置混淆未分） | EXEC | §mitigation 其它防御段 | supported |
| C15 | z-检测器：top-15 z-mean 拒 10/12 finals、0/6 非合规、H0 1/2；MNv2 确认 6/6（2/24 尾部计数）；held-out FPR 2/12=17%（CP CI 2–48%）双架构；单侧为 post-hoc（MNv2 为确认锚）；双侧 1/12 无功率；极值型 1/6 | EXEC | a9z3-20261102T235118 + S13 | supported（pilot 而非验证工作点——正文已声明） |
| C16 | E1 精确重训练参照：6/6 finals 钉过；一阶增益近零、二阶为负（构造无关边界） | EXEC | ertr-20261103T210519 + ertr-gr-20261104T091206/094808（15 fresh from-scratch；S23） | supported |

## D. 方法/协议章声明（sec:method、sec:audit）

| ID | 声明 | 等级 | 源 | 状态 |
|---|---|---|---|---|
| D1 | FIDE 三失败模板：plain F-fit +0.05–0.08 vs 对比 +0.14–0.20；retention 训练 selection 主导；跨类无逐记录分辨 | EXEC | 构造消融（C13 同源） | supported |
| D2 | 可行方向下降的必要性论证（投影梯度/罚函数/拉格朗日三条路失败模式） | DEF/论证 | §fide | argument（非实测对比——评审可质询） |
| D3 | 复杂度 O(T·(k+60+3000)) 前向等价，2–4 分钟/编码（单消费级 GPU）；六种子单元 ~20 分钟 | EXEC | 成本记录 | supported |
| D4 | m5 probe 规格（in-sample AUC、200 步 lr0.1、零初始化、30-vs-3000、方向固定；held-out/cross-fitted 变体未测） | DEF+EXEC | §m5 规格 | defined（未测变体已声明） |
| D5 | 独立校准 probe：3/3 过真审计、0/3 CI-positive 信号（q̂_est=4.58 vs 真值 5.25） | EXEC | S22 | supported（单 draw 声明） |
| D6 | m4 KS 参照分布 = 3,000 冻结 retained 子样本；S12 设备披露边界翻转；S6 独立执行 base-AUC 差 ≤0.011 未解释 | EXEC | §audit + S6/S12 | disclosed-unexplained |

## E. 主张间依赖与已声明边界（评审提示，非缺陷清单）

- 头条数字（A1/A2）限定：provider-known 校准（D5 边界）、ft-referenced 范围（C16 边界）、cat 类 + k=30 主域（C5/C6 边界）、Tier-2 读出（C9 边界）。
- A2 的 0.43 = 12 检查端点最佳值；中位 0.17；边界外 0.60 不计入 passing。
- "first construction"（摘要/结论）主张范围 = live conformally calibrated multi-metric deletion audit 内的 deliberate per-record encoding——与 Table I（tab:neighbors）七族定界绑定。
- 表号顺移历史：期刊版表号 = 能力表 I / encoding II / mitigation / a8 / zreadout / calibration 等，以 \ref 为准。

## F. Ledger 维护规则

- 正文任何数字改动必须同步本 ledger 对应行（AGENTS 纪律）。
- Phase 3 回归对账以本表 A/B/C/D 全部行 + supplementary S1–S24 交叉为范围。
- 新增主张先入 ledger（含等级与源）再入正文。

## 2026-11-06 GPT-6 Pro R01 轮（外部第三轮）

- 26 项发现全部 upheld（docs/decisions/2026-11-06-gpt6pro-r01-consolidation.md）；修复 commit 67e3862。
- **主张行更新**：A1 行加"CE 构造限定"标注（摘要 0.43 = GR 家族）；A2 行范围词更新（0.07-0.43 全接受域/改善子集 0.10-0.43）；C2 行钉定主张降为"主执行 11/12 观测性质"（S23 dog=5/6 反例入册）；B2/Remark 行改条件不变式；D 行 A7 = rescoring 三层区分；V12 修正（F' 独立审计 q-hat=2.831 入正文）；新增数字仅 2.831（S19 在档），零删除。
- **边界表述分层**（V4）：2/4 类 = label-blind 标准；frog CAL-Z 工作点未复制（<=+0.067）；k=300 = 校准负/raw 未测；exact-retrain = 两构造限定；独立校准 = 单次抽样未交付。
- 回归：17pp 0err 0undef；visual 17/17（重写区域放大核验）；禁用词/腐蚀/CR 全零。
- 未决（作者）：availability URL；是否回剪 16pp；R02 或投稿。
