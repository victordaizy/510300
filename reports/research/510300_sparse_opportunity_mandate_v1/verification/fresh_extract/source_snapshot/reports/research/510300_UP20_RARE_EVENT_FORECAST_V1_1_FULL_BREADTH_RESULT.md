# 510300 UP20预测 V1.1 全期广度覆盖结果

- 正式状态：`REJECTED_FROZEN_UP20_RARE_EVENT_FORECAST_V1_1_FULL_BREADTH_NO_RESCUE`
- 唯一变化：官方PIT等权广度从2015-01-05开始可用。
- 未变化：四模块、模型、惩罚、概率门、压力线、20日持有期、相位、成本与验收门。
- 父版本负结果已知且保留；本版本不产生Paper、订单或实盘授权。

## 父版本与V1.1

| 版本 | 基准净夏普 | 压力净夏普 | 捕获UP20 | 误入DOWN20 | 预测满仓区间 |
|---|---:|---:|---:|---:|---:|
| V1 | -0.227 | -0.230 | 1 | 1 | 2 |
| V1.1 | 0.320 | 0.313 | 1 | 0 | 2 |

## V1.1完整账户结果

| 指标 | 结果 | 冻结门 |
|---|---:|---:|
| 基准净夏普 | 0.320 | ≥1.2 |
| 压力净夏普 | 0.313 | ≥1.2 |
| 2021年以来压力净夏普 | NA | ≥1.2 |
| 压力总收益 | 9.70% | >0 |
| 压力最大回撤 | -7.32% | 诊断 |
| 捕获UP20 | 1 | ≥14 |
| 误入DOWN20 | 0 | ≤1 |
| 误入RANGE20 | 1 | ≤10 |
| 预测满仓区间 | 2 | ≥14 |
| Brier Skill | -0.019 | >0 |
| Log Loss Skill | -0.024 | >0 |

## 20种相位压力净夏普

| offset | 压力净夏普 | 满仓区间 | 捕获UP | 误入DOWN | 误入RANGE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.313 | 2 | 1 | 0 | 1 |
| 1 | -0.558 | 3 | 0 | 1 | 2 |
| 2 | -0.157 | 4 | 1 | 1 | 2 |
| 3 | 0.385 | 3 | 2 | 0 | 1 |
| 4 | -0.299 | 5 | 2 | 1 | 2 |
| 5 | -0.389 | 3 | 0 | 1 | 2 |
| 6 | -0.216 | 5 | 1 | 3 | 1 |
| 7 | -0.256 | 4 | 0 | 2 | 2 |
| 8 | -0.107 | 3 | 0 | 0 | 3 |
| 9 | 0.461 | 2 | 2 | 0 | 0 |
| 10 | 0.114 | 4 | 1 | 1 | 2 |
| 11 | 0.007 | 5 | 1 | 1 | 3 |
| 12 | -0.118 | 4 | 0 | 2 | 2 |
| 13 | -0.146 | 3 | 0 | 1 | 2 |
| 14 | 0.336 | 5 | 2 | 1 | 2 |
| 15 | -0.113 | 4 | 0 | 1 | 3 |
| 16 | 0.048 | 1 | 0 | 0 | 1 |
| 17 | 0.278 | 3 | 1 | 0 | 2 |
| 18 | 0.514 | 3 | 2 | 0 | 1 |
| 19 | -0.258 | 2 | 1 | 1 | 0 |

## 冻结门

- `primary_base_net_sharpe`：`FAIL`
- `primary_stress_net_sharpe`：`FAIL`
- `primary_recent_stress_net_sharpe`：`FAIL`
- `primary_stress_total_return_positive`：`PASS`
- `minimum_captured_up_blocks`：`FAIL`
- `maximum_false_down_blocks`：`PASS`
- `maximum_false_range_blocks`：`PASS`
- `every_phase_stress_net_sharpe`：`FAIL`
- `primary_brier_skill_vs_causal_base_positive`：`FAIL`
- `primary_log_loss_skill_vs_causal_base_positive`：`FAIL`
- `minimum_primary_predicted_long_blocks`：`FAIL`

## 正确解释

补齐2015年以来官方PIT等权广度仍未使同一V1模型通过；广度历史缺失不能解释父版本失败，V1家族停止且不得调参救援。

独立前向观察数仍为0；Paper、Shadow、持仓映射、订单、券商连接与实盘授权全部关闭。
