# 510300 未来20日稀有上涨机会预测 V1 结果

- 正式状态：`REJECTED_FROZEN_UP20_RARE_EVENT_FORECAST_V1_TARGET_OR_INFORMATION_BUDGET_GATE_FAILED_NO_RESCUE`
- 资产边界：仅510300或现金；无杠杆、无做空、无衍生品。
- 证据边界：历史因果预序列检验；独立前向样本仍为0。
- 规则边界：四模块、符号约束模型、概率阈值、相位和成本均在结果前冻结。

## 主相位账户结果

| 口径 | 净夏普 | 总收益 | 最大回撤 | 交易次数 |
|---|---:|---:|---:|---:|
| 基准成本 | -0.227 | -14.59% | -24.23% | 4 |
| 压力成本 | -0.230 | -14.79% | -24.38% | 4 |
| 2021年以来压力 | NA | 0.00% | 0.00% | NA |

## 稀有事件信息质量

| 指标 | 结果 | 冻结预算/门 |
|---|---:|---:|
| 捕获UP20 | 1 | ≥14 |
| 误入DOWN20 | 1 | ≤1 |
| 误入RANGE20 | 0 | ≤10 |
| 预测满仓区间 | 2 | ≥14 |
| UP20精确率 | 50.00% | 诊断项 |
| UP20召回率 | 4.35% | 信息预算参考 |
| Brier Skill | -0.029 | >0 |
| Log Loss Skill | -0.034 | >0 |

## 20种相位压力净夏普

| offset | 压力净夏普 | 捕获UP | 误入DOWN | 误入RANGE |
|---:|---:|---:|---:|---:|
| 0 | -0.230 | 1 | 1 | 0 |
| 1 | -0.222 | 0 | 2 | 2 |
| 2 | -0.240 | 1 | 1 | 1 |
| 3 | -0.406 | 1 | 1 | 1 |
| 4 | -0.462 | 1 | 1 | 2 |
| 5 | -0.279 | 1 | 1 | 1 |
| 6 | -0.215 | 1 | 2 | 0 |
| 7 | -0.256 | 0 | 2 | 2 |
| 8 | -0.214 | 0 | 1 | 2 |
| 9 | 0.191 | 0 | 0 | 1 |
| 10 | -0.048 | 0 | 1 | 2 |
| 11 | 0.006 | 1 | 1 | 2 |
| 12 | 0.144 | 0 | 0 | 2 |
| 13 | -0.001 | 1 | 1 | 2 |
| 14 | 0.129 | 2 | 1 | 2 |
| 15 | 0.107 | 1 | 1 | 2 |
| 16 | 0.225 | 0 | 0 | 3 |
| 17 | 0.086 | 1 | 1 | 2 |
| 18 | 0.254 | 2 | 1 | 2 |
| 19 | -0.400 | 1 | 2 | 0 |

## 冻结门判定

- `primary_base_net_sharpe`：`FAIL`
- `primary_stress_net_sharpe`：`FAIL`
- `primary_recent_stress_net_sharpe`：`FAIL`
- `primary_stress_total_return_positive`：`FAIL`
- `minimum_captured_up_blocks`：`FAIL`
- `maximum_false_down_blocks`：`PASS`
- `maximum_false_range_blocks`：`PASS`
- `every_phase_stress_net_sharpe`：`FAIL`
- `primary_brier_skill_vs_causal_base_positive`：`FAIL`
- `primary_log_loss_skill_vs_causal_base_positive`：`FAIL`
- `minimum_primary_predicted_long_blocks`：`FAIL`

## 结论边界

固定四模块UP20预测器没有同时达到历史夏普1.2、稀有事件信息预算与相位稳健性门；V1按预注册规则停止，不以控制项或调参救援。

控制项仅用于解释否决器贡献，不能替代主规则。无论历史是否通过，Paper、Shadow、持仓映射、订单、券商连接与实盘授权均保持关闭。
