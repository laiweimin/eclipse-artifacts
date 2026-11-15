# 预注册卡：路径 A 机制判定（作者 2026-11-12 批准"完成路径 A"）

**动机**：R03 冷审 R01-2 指出"弥散→信号"的因果桥未建立，并给出可加分条件："identify a decision-relevant mechanism under aligned audit information and coverage conditions"。本卡三项实验用于判定信号交付的真实驱动因素。全部结果预先声明：无论正负，按判定门如实入稿。

**设备**：GPU 0（启动前核温度）。**总估**：A2 ~25min + A3 ~20min + A1 ~100min ≈ 2.5h（<5h 门）。

---

## A2 — k=100 弥散比（补全跨尺度映射）
- **设计**：与 ew3 完全同管道（同 theta0、同 pool/f_deep、同 sub seed 52000、同 metrics_from_logits）；F = f_deep(pool,100)；噪声侧 = ew3 已归档 6 个全数据重训 ckpt 在 F(100) 上重评；诚实侧 = 3 个新 from-scratch 重训（seeds 44141-44143，D∖F(100)，40ep）。
- **读数**：每指标 ratio = 诚实 max-dev-from-median / 噪声带。
- **用途**：与已有 k=30(4.78, 信号有)/k=300(1.91, 无)/k=1000(2.84, 未测信号) 拼成跨尺度表，检验"信号存在 ⇔ 弥散比高"的排序一致性。
- **判定门**：ratio(m3)@k=100 > 2.84（高于 k=1000）或介于 1.91–4.78 区间 → 记录为**宽度与信号无单调对应**的证据；不单独作结论，只入表。

## A3 — 对齐仪器检验（分离"池/仪器失配"与"真实残留"）
- **设计**：零训练。对 12 个主终态（6 CE + 6 GR）及其 6 配对 base，用同一 scorer（neggrad）分别测两个比较总体的 AUC：**fresh 池**（200 从未训练，主实验口径）与 **retained 池**（3000 保留子样本，即审计 m5 的参照总体）；同时给 raw 与 z-校准（ew9F 6 参照的逐记录 μ/σ）两种读数。
- **判定门（结果前冻结）**：
  - **POOL-MISMATCH-DOMINATES**：z-校准后 retained 池的配对增益中位数 ≤ 1/3 × fresh 池增益中位数；
  - **ALIGNED-SURVIVES**：retained 池增益中位数 ≥ 2/3 × fresh 池；
  - 其间 = MIXED（如实报两侧数值）。

## A1 — 带宽剂量响应（哪一类 slack 是驱动）
- **设计**：主 ft 参照审计（fit6/cal12，conformal q̂@0.20 固定）为基线；只按**组件族**缩放接受的带半宽：接受 iff |m_i−c_i| ≤ κ_fam·q̂·σ_i。
  - **utility 臂**：κ 只作用于 m1/m2（accuracy 族），forget 四指标固定 κ=1；
  - **forget 臂**：κ 只作用于 m3/m4/m5/m6，utility 固定 κ=1；
  - κ ∈ {0.5, 0.75, 1.0, 1.5}；3 个 e4 起点；起点不可行时先做 ≤40 步的可行性恢复（只降 s），再编码。
- **读数**：每 (臂,κ,起点) 的 label-blind AUC 增益 + 2000 重采样 CI + pass 状态；κ=1.0 为复现对照（应≈主实验）。
- **判定门（结果前冻结）**：
  - **UTILITY-SLACK-GOVERNS**：utility 臂增益随 κ 单调上升且 κ=0.5 与 κ=1.5 的 CI 不重叠；
  - **FORGET-SLACK-GOVERNS**：forget 臂满足同一条件而 utility 臂不满足；
  - **WIDTH-INDEPENDENT**：两臂在 κ∈{0.75,1.0,1.5} 内增益变化 ≤0.03 且 CI 全部重叠；
  - 其它 = PARTIAL（给曲线，不作机制命名）。
- **诚实边界（预先声明）**：κ≠1 的审计不再覆盖诚实模型（κ<1 会拒诚实输出），因此 κ<1 各行是**机制探针**，不是可用审计配置；这一点在入稿时逐行标注。

**产物**：`research_outputs/uw/{disp100,align,width}-<ts>/verdict.json`；结果入补充 S36-S38 与正文对应段。
