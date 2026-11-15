# 预注册卡：五缺口补强程序（作者 2026-11-10 批准全做）

**顺序**：Batch A（#1+#2）→ B（#3）→ C（#4）→ D（#5a/b/c）。总估 ~9-12h GPU（超 5h 门，作者明示授权；逐批启动、批间核 GPU）。全部判定门结果前冻结。

## #1 z-检测器适应性臂（run_zad.py，~1h）
构造：编码器在**复合约束**（过 A6 审计 AND 单侧 top-15 z-mean 低于检测阈值）下优化 J_CE；z 逐记录校准 vs fit 系综（CAL-Z 机器），检测阈值取 18 诚实模型分位（S13 协议）。3 e4 起点。
门：ZDET-EVADED（≥2/3 起点同时过 A6 + CI 正增益 + 检测器不触发）vs ZDET-ROBUST（复合约束下无 CI 正交付）vs PARTIAL。

## #2 stronger-m5 下 z-readout 层（run_sm5z.py，~40min）
构造：复用 run_sm5 强审计；3 e4 起点编码后测 **Tier-2 CAL-Z 工作点读数**（z-neggrad TPR@nominal 5%，阈值冻结 aux，实际 FPR 于 eval——run_lfpr 协议），非仅 AUC。
门：READOUT-DELIVERED（≥2/3 起点过审且 z-TPR 高于 base 到主实验水平）vs READOUT-ABSENT。

## #3 弥散 ensemble-size（run_disp6.py，~2h）
构造：每尺度加 3 个诚实重训（k=30/300/1000 × 新种子），合并 ew3 原 3 → n=6 重算弥散比（诚实带/全数据带）。
门：报告 n=6 比值与 n=3 差异；判 RATIOS-STABLE（±25% 内）vs RATIOS-SHIFT。

## #4 k=300 独立重校准审计（run_k300cal.py，~1.5h）
构造：k=300 删除下自建 fit6/cal12 诚实 ft 系综 + conformal q(k=300)；3-6 编码于其下。
门：报告过审率 + CI 正计数；判 K300-RECAL-ACCEPTS / SIGNAL 状态（对照 S16 复用阈值版）。

## #5 三子项（~4-6h）
- 5a 第二顺序请求对（run_seq2.py，~1h）：换 F2 种子复跑 S11 探针；门 = 复现 stale-pass/wash/不累积三签名与否。
- 5b 类向弥散表（run_classdisp.py，~2.5h）：frog/dog/airplane 各 k=30 诚实重训 3 种子 + 全数据带 → 类向比值表；门 = 报告表 + 是否支持"类间方向一致"。
- 5c A8 重设计臂（run_a8r.py，~1h）：run_mcr 机器换 sc8（Mahalanobis）约束 × R1/R2 重设计目标 × 3 起点；门 = A8-EVADED / A8-ROBUST / PARTIAL（对照其重优化层 12/12 过）。

**产物**：research_outputs/uw/{zad,sm5z,disp6,k300cal,seq2,classdisp,a8r}-<ts>/verdict.json；全部入补充 S29+，材料性结果入正文对应段。
