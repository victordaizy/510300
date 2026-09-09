# 市场状态感知策略资格与风险配置器 V1 状态

- 决策时点：`2026-08-27T16:42:35.556706+08:00`
- 清单冻结时点：`2026-08-27T16:42:04.724702+08:00`
- 主状态：`NO_VIEW_CASH_ONLY`
- 研究视图：`NO_VIEW`
- 当前环境：`NO_VIEW`
- 合格策略数：0
- 现金权重：100.00%
- 研究边界：只输出资格与风险预算；不生成 Paper、Shadow、仓位、订单、账户连接或实盘动作。

## 当前环境证据

- 环境证据 ID：`MISSING`
- 可得时间：`MISSING`
- 判定原因：决策时点没有可得环境证据，禁止沿用旧状态

## 组件冻结状态

| 策略 | 组件状态 | 原组件 Paper/实盘授权 | 是否被元策略改写 |
|---|---|---:|---:|
| BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3 | `REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN` | 否 | 否 |
| DIGITAL_ASSET_DUAL_LONG_HORIZON_TREND_V4 | `REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN` | 否 | 否 |
| DIGITAL_ASSET_RELATIVE_STRENGTH_ROTATION_V5 | `REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN` | 否 | 否 |
| DIGITAL_ASSET_MULTI_HORIZON_TREND_VOTE_V6 | `REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN` | 否 | 否 |
| DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V15 | `REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN` | 否 | 否 |

## 策略资格

| 策略 | 当前环境是否允许 | 条件证据 | 通过门数/总门数 | 合格 | 研究权重 |
|---|---:|---|---:|---:|---:|
| BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3 | 否 | `MISSING` | 1/4 | 否 | 0.00% |
| DIGITAL_ASSET_DUAL_LONG_HORIZON_TREND_V4 | 否 | `MISSING` | 1/4 | 否 | 0.00% |
| DIGITAL_ASSET_RELATIVE_STRENGTH_ROTATION_V5 | 否 | `MISSING` | 1/4 | 否 | 0.00% |
| DIGITAL_ASSET_MULTI_HORIZON_TREND_VOTE_V6 | 否 | `MISSING` | 1/4 | 否 | 0.00% |
| DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V15 | 否 | `MISSING` | 1/4 | 否 | 0.00% |

## 成本后风险预算

- 策略总权重：0.00%
- 现金权重：100.00%
- 绝对策略权重换手：0.00%
- 基础分配切换成本估计：0.00%
- 压力分配切换成本估计：0.00%

## 治理核对

- 使用最近赢家排名：`false`
- 使用历史收益决定当前排名：`false`
- 改写组件拒绝状态：`false`
- 将现金视为失败：`false`

本报告不是交易建议或交易授权。缺少合格证据时，100% 现金是系统的正确输出。
