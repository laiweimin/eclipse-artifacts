# UW M3 覆盖性校准预注册卡（2026-09-29，作者选定路径；R4 审阅人 M3 规格）

## 0. 问题与主张目标

R3-P1 证明：n=6 的 max-dev 带对新诚实系综拒收 5/6——该审计的约束是诚实覆盖。
M3 回答：**标准有限样本校准（split conformal）能否在声明误报目标下交付诚实覆盖，
同时保持对已有攻击候选的拒收？** 成功则论文新增一个已验证的正面能力（修复），
而非"标量约束"。

## 1. 冻结要素（先于任何运行；全部沿用审阅人 M3 措辞）

- **联合审计分数**：s(θ) = max_i |m_i(θ) − c_i| / σ_i，i ∈ {m1,m2,m3,m4,m5,fm}
  （六指标，S1 探针子样本）。c = 该请求诚实参照的中心（逐模型 leave-one-out
  中位数）；σ_i = 校准池上该指标残差的池化 MAD（各系综 MAD 的中位数）。
  ——标准 split conformal 于 max 归一化偏差分数；**不命名为新算法**，引用
  Vovk et al. 2005 / Angelopoulos & Bates 2021（入 bib 前逐条核验）。
- **整体误报目标**：α = 0.20（主）；α = 0.10（次，同报）。
- **校准规则**：q = 校准分数的第 ⌈(n+1)(1−α)⌉ 小值；接受 iff s ≤ q。
  n_cal=24, α=0.20 → 第 20 小；α=0.10 → 第 23 小。
- **允许的诚实发布程序**：该参照族的单次诚实运行（ft 族 = 1-epoch ft 单跑；
  重训族 = 40ep 重训单跑）。**选择/优化后的提交不在交换性保证内**——其覆盖
  是经验问题，用已有攻击候选测量（审阅人明示此边界，如实声明）。
- **基线**：现行审计（单系综 n=6 max-dev 带）对同一测试点的拒收率。

## 2. Phase 1（零新训练）：ft 族跨系综覆盖（方法验证）

- 诚实池：5 个独立 ft 系综 × 6 模型（ew9F 45901-06 / r3p1 ft-aud 46311-16 /
  ex3 e2 46801-06 / e3 46811-16 / e4 46821-26）= 30 模型，ckpt 全在档。
- **LOEO**：留一系综作测试（6 模型，逐模型 LOO 中心），其余 4 系综（24 模型）
  构造 σ 池与 q；5 折共 30 个诚实判定 → 经验 FPR + 精确二项 CI。
- 基线：全部 20 个有序系综对的 max-dev 交叉拒收（120 判定）→ 分布。
- 攻击侧：ex3 三系综的 18 个 wash ckpt（构建于各自系综带）按同规则评测
  （σ 池排除其源系综）；另测 P2 型 logit 缩放点（解析变换，无需 ckpt）。
- **判定（冻结）**：COVERAGE_DELIVERED ⇔ 主 α 下诚实 FPR 点估计 ≤ α 且
  基线 FPR − 诚实 FPR ≥ 0.30（修复幅度实质）；IMPROVED_BUT_IMPRECISE ⇔
  方向成立但 CI 跨 α；NOT_DELIVERED ⇔ 无改善。攻击拒收率只描述性报告。

## 3. Phase 2（需批准，~5.3h GPU0）：重训族跨请求覆盖（部署问题）

- 新数据：k=30 / θ₀A 的 **3 个新 F 抽样**（F₂/F₃/F₄，RandomState 47101/47201/
  47301 各取 pool 前 30）× 各 3 个 40ep 重训（seeds 48101-03 / 48201-03 /
  48301-03）= 9 次重训（≈5.3h）。
- 结构同 Phase 1：LOEO 跨 F 抽样（留一 F 作测试）；诚实测试点含 F₁ 上已有
  12 模型按对应 F 的中心计分；攻击候选 = r3p3 的 12 个 wash ckpt。
- 判定同 Phase 1 冻结规则（跨 F 版）。
- 命令：`python -m uw.run_m3 --phase2 --gpu 0`（先冒烟）。

## 4. 汇报与边界

- 同时报告：诚实接受率（+CI）、各攻击类拒收率（+CI）、违例指标计数；
  交换性边界（跨 F/跨 θ₀ 不可外推；自适应提交无保证）如实入文。
- 全部运行入档 `research_outputs/uw/m3-<ts>/`；负结果同样入稿（项目规则）。

## 执行记录（2026-09-29，Phase 1 当日完成）

- **Phase 1（m3-20260929T151856，GPU1 ~7min，零新训练）**：5 系综 LOEO——
  诚实 FPR @α=0.20 = **0.20**（6/30，CI 0.06–0.34）；@α=0.10 = **0.10**
  （3/30，CI 0–0.21）；基线（单系综 max-dev 交叉 120 判定）= **0.50** →
  **COVERAGE_DELIVERED**（冻结判据：FPR≤α 且改善 ≥0.30）。攻击侧（18 个 ex3
  wash ckpt）5/18 拒收 @0.20（描述性；无保证——交换性声明入卡）。
- **入稿**：新 §m3 小节 + 摘要 "Finally, we validate the repair..." 句 +
  指导清单首项（coverage-targeted）+ 开放问题 #1 改为跨 F 抽样（Phase 2）。
  bib 新增 vovk2005algorithmic / angelopoulos2021gentle（web 核验后入库）。
  审计 **282/282**（+9）；judge 四页首轮 1 fail（第 4 次 heredoc 吃 `\r`
  事故——字节级修复后 CLEAN）。
- **Phase 2 待批准**：3 新 F 抽样 × 3 重训 = 9 次 40ep ≈ **5.3h GPU0**，
  超过 5h 门需作者确认（命令与种子见卡 §3）。

## Phase 2 执行记录（2026-09-29 作者批准当日执行，GPU0 ~5.3h）

- **m3p2-20260929T154632**：9 新重训（F₂/F₃/F₄ = RandomState 47101/47201/47301 各取 pool 前 30；seeds 481xx/482xx/483xx 各 3）+ F₁ 存档 12 模型，四抽样 LOEO。
- **判定 COVERAGE_DELIVERED**：跨抽样诚实 FPR @α=0.20 = **0.143**（3/21，CI 0–0.29）；@α=0.10 = **0.00**（0/21）；基线（单抽样 max-dev 跨抽样）= **0.952**（60/63，唯 F1→F3 0/3 幸免）→ 改善 0.81 ≫ 0.30 冻结线。攻击（F₁ 上 9 个 wash/adaptive ckpt）3/9 拒收 @0.20（描述性）。
- **部署问题正面回答**：池化 conformal 校准迁移到新删除请求；现行校准在该迁移下几乎全灭（95%）。
- 入稿：§m3 增 Phase 2 段 + 摘要跨请求子句 + 开放问题 #1 改"Scaling the validated calibration"。审计 **290/290**（+8）；judge 3/3。

## R5 执行记录（2026-09-29 第五轮审阅"修改后重审"，当日闭合纸面与统计项）

- **统计修复**：CP 精确区间实现（二分反演二项 CDF；与审阅人表格逐位吻合 0.077–0.386 等）；**erratum 声明入稿**（此前 "exact binomial" 实为 Wald 截断）。
- **干净 split 实验（m3b-2\*，零 GPU，读缓存）**：固定划分（跑前指定 fit 6/cal 12/test 12）——诚实 FPR 4/12@0.20（CP 0.099–0.651，p=0.21 与 α=0.20 边际保证相容）、0/12@0.10（CP 0–0.265）、同 fit 预算基线 7/12；200 次重洗均值 0.163@0.20 / 0.081@0.10（5–95% 带报告）。LOEO 降为稳健性（CP 修正后 6/30→0.08–0.39）。
- **Phase 2 逐请求分解**：F1 1/12、**F2 2/3**、F3 0/3、F4 0/3；等权 0.188（池化 3/21 低估 F2）。诚实定性："有限请求集上的经验迁移，非部署总体覆盖"。
- **§m3 重写为完整算法**：部署语义（request-support setting）+ 四步算法（fit/scale/calibrate/accept + 阈值实现细节）+ 保证单位（单评分输出边际 ≤α；请求内相关 → 逐请求报告与等权汇总；LOEO/重洗为描述性聚合）+ **三域分离**（五指标域 / 对称·单侧 fm 域 / conformal 联合域不得拼接）。**Algorithm 1 替换为校准协议**；wash 构造移附录 A.1（门控删除——未验证不保留）。
- **Group 3 一致性**：摘要 recovery score 限定（不再称领域 headline）+ 回滚限定 at-resolutions-tested + M3 句改校准框架；§IV-D plateau 措辞与"三单元在范围内"假句删除；retrial 三事件分离（M/6 含最佳候选 vs 阈值事件）；图注 monotone 限定；0.77/59% 句删除；P1 "one of six...single candidate's fm"；§approx 三域分离；讨论 "not separated"+接受≠已遗忘。
- 审计 **305/305**（+15）；judge 6 页（1 复合词换行修复后过）。15 页终态。

## R6 执行记录（2026-09-29 第六轮审阅"修改后重审"，全部零新训练）

- **R5 静默失败修复**：核查发现 R5 轮四处 python 替换未命中（plateau/同概率/图注/headline——replace 不匹配仍打印 ok）；本轮以 Edit 工具全部落地。
- **Algorithm 1 重写为真隔离版**：输入 = disjoint H_fit/H_cal/H_test；step 2 freeze score（c 与 σ **都只从 H_fit**——与 m3b 实现一致，实现本已干净，歧义在论文写法）；H_cal 只定阈；H_test 只评；Guarantee 行写明 over (H_cal, θ_new) jointly、given H_fit 的标准 split-conformal 边际；Cross-request variant 单列为 empirical、no guarantee claimed。指标措辞对齐（five audited + fm jointly）。
- **m3c 补算（m3c-2\*，GPU ~4min）**：(a) **同总预算 18 参照的 max-dev 基线 = 2/12** → 诚实归因入稿："most of the coverage gain comes from using more reference models; at matched budget the conformal rule is not pointwise superior"——conformal 的贡献 = 声明目标 + 有效性保证；(b) **攻击在冻结固定审计下 5/21@0.20（含全部 3 个 adaptive）**、0/21@0.10——与诚实结果同段并列，"控制诚实误拒 ≠ 检测不诚实"明示；(c) 新 draw 等权 0.222 入稿。
- **统计解释修正**：删 p=0.21 句，改"估计该冻结审计的条件误拒率；边际保证允许不同校准抽样的条件率不同，单次结果既不证明也不否定"；LOEO 只留计数（去 CP 区间——决策共享校准数据）；阶次校正水平 2/13≈0.154、1/13≈0.077 与重洗均值并列；200 重洗定性为划分敏感性。
- **残留清理**：门控降为"探索过、不作部署建议"（成本段+附录一致）；两处 "must accept" 改协议口径（selection-level = 需审计定义参照的声明程序）；Phase 2 support 语义（新 draw 三模型兼任中心源与评价集）；附录 inventory 增 M3 两行。
- 审计 **312/312**（+7）；judge 两轮（p7 指标口径 + p6 页序误派澄清）后全过。15 页终态。

## R7 执行记录（2026-09-30 第七轮审阅"修改后重审"，零训练；Algorithm 1 隔离与保证获正式验收）

- **M1 单一算法叙述**：§m3 重构为 "The guaranteed procedure (Algorithm 1)"（保证只附固定请求 split 版）+ "An empirical cross-request variant (no guarantee claimed)"（self-centered 设计明示：新请求模型兼任中心源与评价集）。旧 archive-尺度四步描述删除。
- **M2 统一结果表**：新 Table（Rule/Refs/Hon@.20/Atk@.20/Hon@.10）= max-dev 6→7/12；split-conformal 6+12→4/12、5/21、0/12；matched 18→2/12；分母不同无检测主张的 caveat 入表注；adaptive washes 身份限定（针对旧带构建，非本冻结分的完整自适应测试）。
- **M3 主张同步**：摘要吸收同预算负对照（"contribution is the declared target and validity conditions, not a lower point estimate"）；跨请求降级为 self-centered exploratory sensitivity（0.19 vs 0.95，非样本外迁移）；普遍归因句（"every recovery score..."）撤除改可检验表述；harm statistic → target statistic 全文更名；五 draw 估计量警示（4 个同池增量 vs −0.066 审计极值诊断差分，heterogeneous sensitivities）；开放问题 #1 改 "Scaling the calibrated audit"（独立支持/评价划分为剩余问题）。
- 审计 312/312（数字不变）；judge 4/4。提交 04a11b8。
