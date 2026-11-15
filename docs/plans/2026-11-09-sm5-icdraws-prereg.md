# 预注册卡：stronger-m5 重测（路径 2）+ 独立校准多抽样（路径 3）

**日期**：2026-11-09　**门**：合计估 ~2.5-3h GPU（<5h）　**设备**：GPU 0
**动机**：路径 2 回答两轮审稿共同点名的"头条打的是套件最弱仪器"；路径 3 把独立校准 0/3 孤例变成多抽样刻画。

## 路径 2：stronger-m5（run_sm5.py）
**构造**：m5 升级为 MLP 探针（5 特征：CE 损失/正确类 margin/熵/true-logit/最大他类 logit——全部黑盒+损失可及，审计能力约束不越界；MLP 5-16-1，200 步全批 lr0.1，in-sample AUC 同 m5 协议）。强审计 = 同 m1-m4/m6 + m5s，fit6/cal12/test12 同主拆分模型，conformal q α=0.20；本实例自有冻结探针子样本（retained 3000，seed 62000）。
- **Tier A（重评分）**：主 12 终态（gate-g1b GR 6 + lfpr CE 6）在强审计下 pass 率。
- **Tier B（强审计下编码）**：e4 三起点 CE 编码（同 run_ce 超参），pass 率 + 增益 + 2000 重采样 CI。
**门（冻结）**：Tier A STRONG-M5-DETECTS（≥1 终态被拒）vs STRONG-M5-PASSES（12/12 过）；Tier B STRONG-ENCODING-WORKS（≥2/3 CI 正）vs COMPRESSED（0/3 CI 正），中间如实报。

## 路径 3：独立校准多抽样（run_probe_ic.py × 4 新 draw）
原 draw（seeds 45501-45512）已有：3/3 过真审计、0/3 CI 正。新增 4 draw：IC_BASE ∈ {45601, 45701, 45801, 45901}（每 draw 12 个 provider 诚实 ft + 3 编码，协议逐字复用原 runner）。
**门（冻结）**：跨 5 draw 汇总——IND-CAL-DELIVERS（任一 draw ≥1 终态 CI 正且过真审计）vs REMAINS-ABSENT（全部 draw 0 CI 正）；逐 draw 报 q_est/接受率。

**产物**：research_outputs/uw/sm5-<ts>/、ic-<base>-<ts>/ 各 verdict.json。
