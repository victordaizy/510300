# 510300 压力传导危险率 V2：G1 事件预测准入 V1

协议 ID：`510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_V1`  
版本：`1.0.0`  
研究状态：`DISCOVERY_ONLY`  
组合收益读取：`NOT_ALLOWED`  
仓位影响：`0`

## 1. 本步只裁决 G1 可识别性

G0 已通过工程、契约和洁净重放。本协议是首次读取已冻结 BAD10 标签之前的
追加冻结；不修改任何父协议、M/F/T 文件、四状态台账或 BAD10 事件账本。

本执行只回答两个问题：

1. 在 B1 与 B2 都可见的共同样本上，是否仍至少存在 30 个可识别独立事件和
   750 个非事件风险日；
2. 在 B1、B2 与 B3 都可见的共同样本上，是否仍至少存在 40 个可识别独立事件
   和 750 个非事件风险日。

本步不拟合 B0、B1、B2 或 B3，不生成概率、阈值、警报、AUC、PR-AUC、Brier、
Log Loss、校准、收益、净值、夏普、回撤、仓位或订单。

## 2. 标签读取边界

冻结脚本只能核对文件身份、父收据和聚合计数，不得读取实际 BAD10 行。
只有本协议 manifest 已写入且再次验证通过后，构建脚本才允许从固定样本账本
读取以下最小必要字段：

```text
origin_date
entry_date
horizon_end_date
bad10
event_id
sample_group_id
bootstrap_event_block_id
calendar_year_block_id
training_weight
split_assignment
```

禁止读取 `minimum_path_return`、未来路径现金流、首次突破路径数值或其他实际未来
路径字段。BAD10 继续固定为下一可交易开盘起未来 10 个交易日内总财富路径损失
不高于 -4%；本步禁止搜索阈值或期限。

## 3. B1 构造

B1 只使用 510300 自身在 t 日收盘时已经实现的历史：

```text
daily_total_shareholder_return_t
= (unadjusted_close_t + cash_dividend_on_ex_date_t)
  / unadjusted_close_(t-1) - 1
```

没有分红事件的日期，分红现金项为 0；这不是对缺失价格或缺失收益填 0。
缺失前收盘价、缺失价格或非有限价格均保持 `NO_VIEW`。三项固定特征为：

- 20 日实现波动率风险分位；
- 20 日回撤风险分位；
- 5 日下行收益风险分位。

所有风险分位只与严格早于 t 的有效历史比较。H00300 不得代替 510300 执行价格
或 B1 输入。

## 4. 共同样本和权重

样本连接键固定为 `origin_date == feature_date`，特征截止时点固定为 t 日 15:00。

- B2 对 B1：只保留 B1 三项有效、B2 状态为 `VIEW_ALLOWED`，且 `T`、`T×F`
  有限的共同日期；
- B3 对 B2：在前述共同样本上进一步要求 B3 为 `VIEW_ALLOWED`，且 `M`、`T×M`
  有限；
- 不可见日期继续保留在审计账本中，模型权重为空并记录明确原因；
- 一个独立事件只要至少有一个正样本原点可见，就计为可识别事件；
- 对每个模型共同样本，同一事件内可见正样本重新归一，使权重之和严格为 1；
- 每个可见非事件风险日权重固定为 1；
- 训练、校准和评价切分继续保持 `UNASSIGNED_PRE_MODEL_FREEZE`。

重新归一只修正因 `NO_VIEW` 排除部分事件原点后的事件权重，不改变事件身份、
BAD10 标签或事件计数定义。

## 5. 顺序停止门

执行顺序固定为：

```text
构造 B1 与共同样本
→ 统计 B2/B3 可识别事件和非事件风险日
→ 裁决 G1
→ 停止
```

如果 B2 共同样本不足 30 个事件或 750 个非事件风险日：

```text
G1 = NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY
NEXT = STOP_V2_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY
```

如果 B2 满足而 B3 不足 40 个事件或 750 个非事件风险日，只允许另行冻结
B2 对 B1 的事件级预测执行；B3 和宏观增量不得运行。

如果 B2 与 B3 均满足，才允许分别另行冻结 B2 对 B1、B3 对 B2 的预测切分、
逐折训练、事件级损失和区块 Bootstrap。任何真实预测都不得复用本执行的未分配
样本直接拟合。

## 6. 权限状态

无论 G1 结果如何，本步都保持：

```text
RESEARCH_STATE = DISCOVERY_ONLY
RETURN_EVALUATION = NOT_ALLOWED
PORTFOLIO_EVALUATION = NOT_ALLOWED
PAPER_OR_SHADOW = NOT_AUTHORIZED
BROKER_CONNECTION = NOT_AUTHORIZED
POSITION_IMPACT = 0
```

`NO_VIEW` 或 G1 不足是完整研究结果，不得通过放宽覆盖门、加入行业、修改 BAD10、
回填缺失值或读取组合表现进行救援。
