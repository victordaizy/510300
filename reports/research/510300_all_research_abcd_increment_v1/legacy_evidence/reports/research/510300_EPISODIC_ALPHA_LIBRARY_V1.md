# 510300 短周期情景 Alpha 库 V1：冻结后发现性结果

- 冻结清单：`5576c79749c4c07b21338b37145194248324dbae145d30d0fb3893825b4a7079`
- 历史证据标签：`RESEARCH_OBSERVED_DISCOVERY`
- 当前已验证高夏普策略：`NONE`
- 当前路由：`CASH_CNY`，仓位影响为 0
- 实盘交易授权：`FALSE`

## 数据准入

| 分支 | 准入状态 | 收益评价 | 点时资格 |
|---|---|---|---|
| `510300_INFORMED_CREATION_PREMIUM_V1` | `NO_VIEW_DATA_CONTRACT_FAILED` | `NOT_ALLOWED` | `FALSE` |
| `510300_DERIVATIVE_PRESSURE_RELEASE_V1` | `PASS_RESEARCH_OBSERVED_DISCOVERY_ONLY` | `ALLOWED_DISCOVERY_ONLY` | `FALSE` |
| `510300_OPENING_DISCOUNT_RECOVERY_V1` | `PASS_RESEARCH_OBSERVED_DISCOVERY_ONLY` | `ALLOWED_DISCOVERY_ONLY` | `FALSE` |

## 分支裁决

### 510300_INFORMED_CREATION_PREMIUM_V1

- 研究裁决：`NO_VIEW_DATA_CONTRACT_FAILED`
- 生命周期状态：`CANDIDATE`
- 成熟历史事件：`0`
- 平均毛边际：`NOT_AVAILABLE bp`
- 平均压力成本后净边际：`NOT_AVAILABLE bp`
- 保守净夏普：`NOT_AVAILABLE`
- 前向成熟事件：`0`；历史事件不得用于 ACTIVE 晋级。

### 510300_DERIVATIVE_PRESSURE_RELEASE_V1

- 研究裁决：`RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS`
- 生命周期状态：`CANDIDATE`
- 成熟历史事件：`0`
- 平均毛边际：`NOT_AVAILABLE bp`
- 平均压力成本后净边际：`NOT_AVAILABLE bp`
- 保守净夏普：`NOT_AVAILABLE`
- 前向成熟事件：`0`；历史事件不得用于 ACTIVE 晋级。

### 510300_OPENING_DISCOUNT_RECOVERY_V1

- 研究裁决：`RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS`
- 生命周期状态：`CANDIDATE`
- 成熟历史事件：`8`
- 平均毛边际：`-55.6879 bp`
- 平均压力成本后净边际：`-69.6050 bp`
- 保守净夏普：`-0.4859`
- 前向成熟事件：`0`；历史事件不得用于 ACTIVE 晋级。

## 生命周期与路由

三条分支均没有取得基于新前向事件的 `ACTIVE` 资格。治理器因此不能把任何历史信号映射为仓位；当前唯一合法路由是现金。即使某条历史发现性筛选通过，也只能进入零仓位 `SHADOW`。

## 固定边界

`position_mapping_enabled=false`、`order_generation_enabled=false`、`broker_connection_enabled=false`、`live_trading_enabled=false`。完整期权订单失衡分支维持 `BLOCKED_NO_FREE_TRADE_SIGN_DATA`。
