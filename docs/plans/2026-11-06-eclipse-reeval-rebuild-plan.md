# ECLIPSE 论文新版 skills 评估 + paper-builder 重构计划

**日期**：2026-11-06　**状态**：待作者审批
**性质**：论文写作/重构任务计划（PRP-20261026 适用；PSR-20261106 路由适用）
**授权边界**：本计划覆盖论文评估与稿件重构。**不授权**新实验、训练、GPU 运行、外部投稿或对外发送任何材料；评估若发现"缺失实验"类问题，一律记录为建议，由作者另行裁决（5 小时门继续适用）。

---

## 0. 背景与动机

论文当前态（entry 282，16pp / 0 err / 9 章 / ECLIPSE 标题 / RQ1-RQ3 / 递进式 Evaluation）已经过：四模型外部评审 + B1-B4 修复、E1-E5 实验扩展、TDSC 结构重构、五段倒金字塔 Introduction、逐章重写、全稿终审扫描。

2026-11-05/06 新版 skills 完成升级（claim/证据驱动、无固定段数/模块数配额、冷读 vs 定向复审分离、评分 = 稿件质量 × 独立置信度而非问题计数），并已通过 PSR-20261106 路由块接入本项目。本轮目的：**用新版评估工具对成稿做一轮未受先验裁决污染的完整评估，再以 paper-builder 的证据链方法做一次定向重构**——不是推倒重写，而是让最强证据线在结构、叙述与篇幅分配上成为全文组织中心。

## 1. 输入冻结与资产清单

| 资产 | 位置 | 状态 |
|---|---|---|
| 主稿源 | `Latex/UW-Combined/main.tex`（13,768 词源码，9 章） | 基线 = 本计划批准时 HEAD |
| 编译产物 | `Latex/UW-Combined/main.pdf`（16pp）+ `render/` 页图 | 评估输入 |
| 补充材料 | `Latex/UW-Combined/supplementary.md`（S1-S24） | 评估输入 |
| 实验产物 | `python/research_outputs/uw*/`（E1-E5、消融、A8E 等全部 run） | 只读证据源 |
| 数字审计 | `python/audit_etie_numbers.py` 模式（对 UW 需扩展，见 Phase 3） | 回归验证 |
| 参考稿件 | `Latex/FDV/`（IEEEtran 模板）+ `reference/TDSC论文/`（24 篇结构对标） | 格式/结构基准 |
| 写作规范 | `docs/TDSC-FU-READER-CENTERED-WRITING-GUIDE.md` + 本项目 writing profile | 全程适用 |

**第一步动作**：`git stash list` 确认干净 + 记录 HEAD hash 作为基线指纹；评估期间主稿冻结（只读）。

## 2. Phase 0 — 证据盘点与 claim-evidence ledger（约 1-2h，主 agent）

paper-builder 构建步骤 1-4 的前置：为 UW-Combined 建立正式 claim-evidence ledger（`docs/plans/2026-11-06-eclipse-claim-evidence-ledger.md`），格式按 paper-builder `references/claim-evidence-ledger.md`：

```
claim + scope → evidence → source/version/location → support status
```

- 逐条登记正文全部定量主张与限定（9/9 AUC、4.8×、TPR 0.43、+0.14→+0.04、A8 12/12、E1-E5、A2E 反例、MobileNetV2 迁移等），每条挂接 artifact 路径与 run id；
- 区分证据等级：definition / proof / generated check / actual execution / population evidence（paper-builder 不变量：四者不可互换表述）；
- supplementary S1-S24 与正文主张的对应关系入册；
- 产出同时是 Phase 1 冷读评审员的"raw evidence"附件与 Phase 3 回归验证的对照底册。

## 3. Phase 1 — 新版 skills 评估轮（冷读，约一个工作会话）

**协议要点**（academic-paper-reviewer v1.9 / 2026-11-05 cs-ai-v2）：
- **冷读**：评审员只拿当前稿 + ledger + 需要的原始证据，**不提供**四模型历史裁决、B1-B4 修复记录与作者期望结论——防止先验污染；
- 评审员对稿件只读；报告独立产出；
- 每条 finding 必须锚定页/节/表/图或证据文件，写明受威胁的主张、推理、严重级与可行修正；
- 四级区分：demonstrated defect / unresolved concern / scope boundary / optional extension；
- 研究比例原则：不因"未测所有数据集/防御/生产加固"自动扣分（论文未作该类主张即不承担该义务）；
- 诚实双向：earned strengths、合理平局、无发现都是合法结论，无批评配额。

**R1-R3：三个冷读 subagent（并行，Agent tool，上下文隔离）**

| # | 评审员 | 输入 | 透镜 |
|---|---|---|---|
| R1 | EIC / venue-fit | main.tex + supplementary + ledger | 宏观论证链、贡献-证据对齐、TDSC 适配、主张强度 |
| R2 | Methodology | 同上 + 关键 artifact 抽查权 | 实验设计、统计口径、消融归因、对照公平性、威胁模型一致性 |
| R3 | Adversarial（devil's advocate） | 同上 | 对抗性压力测试：最强反例、循环证据、不可证伪主张、被审稿人一句话击穿点 |

**R4：pre-submission-reviewer 五维机械扫描**（主 agent 或单 subagent）
- 宏观逻辑 / 写作细节 / 英文语法 / LaTeX 格式 / 图质量 五维 + 禁用 AI 语气词与 em-dash 全文扫描；
- paradigm 判定 = STEM/technical（权重：Introduction 链、贡献-节映射、消融归因、图质量）；
- 该 skill 的固定公式（如摘要五句式）与项目既有 TDSC 惯例冲突时，**项目指令优先**（paper-builder 优先级规则），冲突项标注不执行并说明。

**R5：主 agent 仲裁合成**（academic-paper-reviewer synthesis 协议）
- 只合成实际报告；CRITICAL/MAJOR 逐条回查稿件与证据，记录 upheld / narrowed / rejected 及理由；
- 产出统一裁决文档 `docs/decisions/2026-11-06-newskills-reeval-consolidation.md`：
  - 最强受支持贡献（论文实际赢的是哪场比赛——PRP §4/§6 口径）；
  - 分级问题清单（CRITICAL→MINOR）+ 每条可行修法；
  - 重构输入：哪些 finding 属结构/叙述层（进 Phase 2）、哪些属证据缺口（记录为建议，不自动执行）；
  - 无目标分数、无一致同意停止条件——merit × confidence 双轴表述。

## 4. Phase 2 — paper-builder 定向重构（1-2 个工作会话，规模视 Phase 1 而定）

**方法**：paper-builder v5（2026-11-05 claim-evidence 版）+ PRP 全程约束。重构深度默认 = **证据驱动定向重构**：只有 Phase 1 判定存在论证级问题的部分才动结构；证据已支持的部分保持稳定（paper-builder 规则：局部任务继承既有大纲，不重启全篇规划——除非 Phase 1 判定故事线本身需重建，那也是允许的，PRP §9）。

**决策序列**（每步以 ledger 为据，一次性记录结论不反复）：

1. **价值线重述**：最强受支持价值线 + 与最近邻工作的真实 delta（R5 产出 → paper-builder 步骤 2）。检查当前标题/摘要/引言是否围绕它组织；不是则重排（PRP §1/§11）。
2. **骨架判定**：现 9 章对 paper-builder 默认八节骨架（I Intro / II Related / III Threat Model+Problem / IV Method / V Theory / VI Eval / VII Discussion / VIII Conclusion）——现结构的 `Threat Model` + `Audit Protocol` 两章对应骨架槽 III 的拆分。判定拆分是否仍有论证职责（III 的两个层次：威胁模型=攻击者能力，审计协议=约束对象）；职责重叠则合并，否则保留并记录理由。
3. **软篇幅预算**：先总量后分节（venue-conventions 软区间，TDSC 期刊对标 `reference/TDSC论文/`），为图表/算法/引用留页；现 16pp 与预算的偏差作为分节目标的约束之一——不是 pass/fail。
4. **Evaluation 论证检查**：用 paper-builder 默认 Evaluation 清单逐项标 applicable/理由——每张表每个实验必须承担明确论证职责（证明核心成立/隔离机制/场景价值/排除替代解释，PRP §8）；不承担职责、易引入无关争议的素材降级/移附录/删。
5. **逐节定向重构**（工作序 III→IV→V→VI→II→I→VII/VIII→title/abstract）：
   - 仅重构 Phase 1 判定有问题的节；其余节只做 R4 机械项修复；
   - Introduction 与 title/abstract 最后动（受益于全文定型后的价值线）；
   - 每节完成后过 `writing-style`（禁用词表经 academic-wording-policy 单一规范入口）+ intro-audit 一致性检查（RQ-节映射、贡献 bullet-模块映射）；
   - **数字冻结纪律**：重构中正文任何数字不得脱离 ledger 改写；新表述若引入新数字必须先入 ledger 挂证据。
6. **claim-evidence ledger 同步**：每轮结构改动后受影响 claim 的 support status 与位置更新（项目既有纪律：每改数字必须同步 claim map）。

## 5. Phase 3 — 回归验证与收口（约半天）

1. **编译链**：pdflatex+bibtex 全量编译，0 err / 0 未定义引用；页数与软预算对照；
2. **视觉验收**：渲染页图 → documents:visual-judge 单门验收（既有流程）；
3. **数字回归**：扩展 `audit_etie_numbers.py` 模式为 UW 版对账脚本（读 artifact 对账正文全部数字 vs ledger）；重点 = Phase 2 动过的每一处；
4. **机械项复扫**：R4 的禁用词/em-dash/标签完整性对改动部分定向复扫（targeted re-review：验证实际关闭，非全篇重审）；
5. **supplementary 一致性**：S1-S24 引用与正文表号/数字同步检查；
6. **收口**：更新 claim-evidence ledger 终态 + AGENTS entry + commit（Conventional Commits，实验/文档/稿件分开）。

## 6. 里程碑与时间估计

| 里程碑 | 内容 | 预计 |
|---|---|---|
| M0 | 计划批准 + 基线指纹 | 0.5h |
| M1 | Phase 0 ledger 建成 | 1-2h |
| M2 | R1-R4 评估完成 | 并行，单会话内 |
| M3 | R5 统一裁决文档 | M2 后同会话 |
| M4 | Phase 2 定向重构 | 1-2 会话（视 M3 规模） |
| M5 | Phase 3 回归验证 + 收口 | 半会话 |

全程 CPU/文本工作，无 GPU、无新实验。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 新版 skill 建议与项目既有惯例冲突（固定段式、评分公式等） | 项目指令 + PRP 优先（paper-builder 明文优先级）；冲突项记录不静默执行 |
| 冷读评审员提出的"缺失实验"诱导扩实验 | 本计划明文不授权；全部记为建议入裁决文档，作者另裁 |
| 重构引入数字漂移 | 数字冻结纪律 + ledger 强制同步 + Phase 3 对账脚本 |
| 重构破坏已验证的论证链（9/9、RQ 映射等） | 每节完成即过 intro-audit；Phase 3 标签/引用完整性复扫 |
| 评估轮先验污染 | 冷读协议：R1-R3 不见历史裁决与修复记录 |

## 8. 需作者裁决的决策点

- **D1 评审员配置**：R1-R3 三冷读 + R4 机械扫描为默认；可增删（如加 perspective lens 或减为两个）。
- **D2 重构深度**：默认"证据驱动定向重构"；若你想要"无条件全篇重构"（每节都过 paper-builder 全流程），工作量约 +1 会话且风险升高，需明示。
- **D3 评估产出去向**：M3 裁决文档是否同步起草外部审阅 review-prompt（作者转交 GPT-6 Pro 等）——默认本轮先内部，外部审阅另行决定。
- **D4 supplementary.md**：是否纳入本轮重构范围（默认只做一致性检查不重构）。

---

**批准即执行 M0-M3；M4 开始前以 M3 裁决文档的实际规模复核 Phase 2 估计。**
