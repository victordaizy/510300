# 510300_ASYMMETRIC_STRESS_HAZARD_V1 BAD10 独立事件普查

## 裁决

- 状态：`PASS_G1_LABEL_IDENTIFIABILITY_ONLY_NO_FEATURE_CONSTRUCTION`
- 独立 BAD10 压力事件：`58` / 要求 `30`。
- 独立非事件十日块：`143` / 要求 `120`。
- 合格标签原点：`2813`；其中 BAD10 原点 `463`。
- 标签固定为未来 `10` 个交易日、路径损失小于或等于 `-4.00%`。
- 冻结 manifest：`6e040088e8789dc7eab951c99b4787d03302b463a98557d95a8a5c978d70d5e8`。

G1 通过只允许进入点时来源合同与机制构造阶段；尚未构造任何因子或模型。

## 研究与交易边界

- `RESEARCH_STATE=DISCOVERY_ONLY`
- `MODEL_POSITION_TARGET=UNSET`
- `ORDER_AUTHORIZATION=NOT_AUTHORIZED`
- `ACTUAL_HOLDINGS_STATE=UNKNOWN_OUT_OF_SCOPE`
- `POSITION_IMPACT=0`
- 本次没有构造 M、F、T，没有训练模型，没有选择概率阈值，没有读取组合收益、净夏普或回撤。
- `ABSTAIN` 不表示现金目标，也不覆盖用户实际持仓。

## 标签口径

信号信息截止于 t 日收盘；以 t+1 日 510300 实际开盘价为入场成本；观察未来十个交易日的实际收盘财富。财富包含持有期间取得资格并自除息日起确认的现金分红应收款。不使用 H00300、同日收盘成交或未来最低价。

BAD10 原点按入场日至十日终点的闭区间做传递重叠合并。非事件块按冻结的最早合格原点贪心向前选择，彼此及其与压力事件均不重叠。
