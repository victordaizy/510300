# 510300 成分脆弱性 DSV5 增量检验 V1 冻结协议

## 1. 裁决范围

本研究只回答一个预测问题：在 `510300.SH` 自身的 20 日实现风险、5 日负收益和
20 日回撤之外，父项目已经冻结的成分股静态脆弱性 `F`、成分传导 `T_C` 与
`T_C × F`，能否提高对非重叠未来 5 个交易日下行半方差的严格前序预测。

```text
MODEL_ID = 510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1
NEW_ALPHA_FAMILY = FALSE
NEW_ESTIMAND = TRUE
PARENT_FEATURE_FAMILY = 510300_STRESS_TRANSMISSION_HAZARD_V2
PORTFOLIO_EVALUATION = NOT_ALLOWED
POSITION_IMPACT = 0
```

本次不生成仓位、净值、组合收益、Sharpe、订单或券商动作。即使 G3 通过，下一步也
只能另行冻结 `510300_WEEKLY_DSV5_RISK_BUDGET_POLICY_V1`。

## 2. 两份输入意见的优先级

2026-09-05 的第二份审校材料明确指出，初版“周度成分脆弱性风险预算”不能作为全新
Alpha，也不能直接读取仓位和 Sharpe；应缩小为一次预测增量桥接实验。发生冲突时，
第二份审校材料优先。

据此冻结以下解释：

- 主目标从 5 日总方差改为非重叠 `DSV5`；
- 先做 B0→B1，B1 通过后才允许 B2；
- `F` 和三个成分传导分位数直接复用父项目已冻结的持久化值；
- `T_C` 只取 `BREADTH_DROP5`、`TAIL_DIFFUSION5`、
  `COMOVEMENT_ACCEL5` 三个风险分位数的中位数，不含资金利率冲击；
- 不给父特征新增 252 日暖启动，不重算窗口、覆盖门或分位数。否则本次将同时改变
  估计对象和特征合同，不再属于 `NEW_ESTIMAND_NOT_PARENT_RESCUE`。

初稿中的 252 日新暖启动属于未进入本 V1 的另一种规格，不得在看到本次结果后作为
救援版本启动。

## 3. 本地谱系核对

| 旧研究 | 目标与时钟 | 与本 V1 的关系 |
| --- | --- | --- |
| `510300_DOWNSIDE_RISK_BUDGET_V1` | 20 日最大不利幅度为主、20 日下行半方差为辅，直接风险分数 | 目标、期限、模型链均不同；旧结果冻结拒绝 |
| HAR 波动率挑战者 | 未来 20 日总实现波动率 | 期限与风险对象不同；旧结果冻结拒绝 |
| 期权隐含下行风险预算 V1 | 未来 5 日总方差、日度重叠标签、期权增量 | B1/QLIKE 思路相似，但目标、特征和执行时钟不同，且数据门阻断 |
| Stress Transmission V2 | 从次日开盘开始的 BAD10 稀有事件概率 | 本 V1 的父特征家族；G2 未运行，不是相同估计对象 |

谱系结论：

```text
SIMILAR_COMPONENTS_ALREADY_IMPLEMENTED = TRUE
EXACT_END_TO_END_STRATEGY_IMPLEMENTED = FALSE
EXACT_DSV5_B1_TO_B2_INCREMENT_TEST_FOUND = FALSE
NEXT_ACTION = RUN_ONE_FROZEN_PREDICTION_INCREMENT_TEST
```

## 4. 数据和父特征

执行时必须逐字节匹配协议中的输入 SHA-256。父特征只允许读取：

- `F` 及其三个冻结分量；
- `t1_breadth_drop_risk_percentile`；
- `t2_tail_diffusion_risk_percentile`；
- `t3_comovement_accel_risk_percentile`；
- 日期、覆盖状态和可用性字段。

必须复核持久化 `F` 等于三个分量全部有效时的中位数。普通缺失保持 `NO_VIEW`，不做
前值填充、不填 0。行业、宏观、NBS、期权和分钟字段均不得进入本次模型。

## 5. DSV5 标签

在原点 `t` 收盘形成特征，执行日为下一个交易日。第一日是官方未复权开盘到当日
收盘的对数收益；随后四日是相邻总财富收盘之间的对数收益：

\[
r_{t,1}=\log(P_{t+1,close}/P_{t+1,open}),
\]

\[
r_{t,h}=\log((P_{t+h,close}+D_{t+h})/P_{t+h-1,close}),\quad h=2,\ldots,5.
\]

`D` 只包含从执行日起持有后、按登记日收盘取得权利并在除息日确认为应收的每份现金
分红。执行日之前已经形成的权利不归属于该标签中的新进入者。

\[
DSV5_t=\frac{252}{5}\sum_{h=1}^{5}\min(r_{t,h},0)^2.
\]

5 日总方差只作为诊断字段，不得训练、筛选或替代主目标。

## 6. 非重叠原点和无标签 G1

从第一个同时满足父特征完整、B1 历史完整和未来 5 个交易日日期完整的交易日开始，
按上交所交易日序号每隔 5 日形成固定网格。`OFFSET_0` 是主结果，`OFFSET_1` 至
`OFFSET_4` 只作预注册稳健性检查。网格日期若为 `NO_VIEW`，跳过该原点，但不平移、
不重新锚定后续网格。

在读取任何 `DSV5` 数值前必须完成并提交：

```text
主偏移合格原点 >= 225
前80个原点 = 首次训练集
严格前序评价原点 >= 145
评价期 = 3个连续等数量时代
每个时代 >= 48个原点
```

不满足即输出 `NO_VIEW_INSUFFICIENT_NONOVERLAPPING_ORIGINS`，标签保持未读，模型不
运行。禁止改选起始偏移凑样本。

## 7. 模型

B0 为当次模型时点已成熟训练标签的扩展均值。

B1 固定为：

\[
\log\widehat{DSV5}^{B1}=\alpha+\beta_1\log RV20+\beta_2NEG5+\beta_3DD20.
\]

B2 固定为：

\[
\log\widehat{DSV5}^{B2}=\alpha+\beta_1\log RV20+\beta_2NEG5+\beta_3DD20+
\gamma_1T_C+\gamma_2(T_C F).
\]

所有斜率非负，截距不惩罚；QLIKE Ridge 的 `L2=1.0`。标准化均值和标准差只来自
当期成熟训练样本。首次训练使用 80 个非重叠原点，此后每 13 个评价原点重估一次，
中间保持系数不变。训练标签的期限结束日必须不晚于当前原点。

## 8. 严格门与顺序

G2 必须全部满足：

- B1 相对 B0 的总体前序 QLIKE 改善大于 0；
- 13 原点循环移动块 Bootstrap 的单侧 90% 下界大于 0；
- 三个时代至少两个改善；
- 最新时代改善。

G2 失败即冻结为 `REJECTED_FROZEN_PRICE_RISK_NOT_FORECASTABLE`，B2 不得运行。

G3 必须全部满足：

- B2 相对 B1 的总体 QLIKE 改善至少 2%；
- `log(DSV5 + 1e-8)` MSE 改善大于 0；
- 13 原点循环移动块 Bootstrap 单侧 90% 下界大于 0；
- 三个时代至少两个改善且最新时代改善；
- 五个预注册偏移至少四个方向为正；
- 最高预测风险五分位的实际 DSV5 至少为最低五分位的 1.5 倍；
- 平均预测值/平均实际值在 `[0.80, 1.25]`。

G3 失败即冻结为 `REJECTED_FROZEN_CONSTITUENT_FRAGILITY_INCREMENT`。不得加入宏观、
NBS、估值、成交量、技术指标或其他窗口救援。

## 9. QLIKE 口径

拟合目标使用协议给定的等价形式：

\[
\frac{1}{n}\sum_i\left(\frac{DSV5_i+\epsilon}{\widehat{DSV5}_i}
+\log\widehat{DSV5}_i\right)+\lambda\lVert\theta\rVert_2^2,
\quad \epsilon=10^{-8}.
\]

报告中的相对百分比使用与其仅差观测常数的单位不变 QLIKE deviance：

\[
q_i=z_i-\log z_i-1,\qquad
z_i=\frac{DSV5_i+\epsilon}{\widehat{DSV5}_i}.
\]

Bootstrap 使用逐原点 `q_baseline - q_model`，不按年份、危机或结果挑样本。

## 10. 冻结、提交和一次性消费

顺序不可交换：

1. 完成谱系核对、协议、实现和合成测试；
2. 生成冻结 manifest/receipt 并提交本地 Git；
3. 只读日期与可用性，生成 G0/G1 原点审计并再次提交；
4. 原子创建一次性 claim；
5. 才可读取 DSV5、运行 B0/B1，并按门决定是否运行 B2；
6. 输出结果和执行收据后提交，不覆盖父项目任何文件。

claim 一旦创建，本 V1 即被消费；程序故障也不得静默重跑，必须另建版本并说明原因。

本协议不进行仓位或交易评价，也不改变 `DISCOVERY_ONLY / POSITION_IMPACT=0`。
