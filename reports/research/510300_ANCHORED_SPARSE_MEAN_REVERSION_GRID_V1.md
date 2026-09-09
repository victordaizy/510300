# 510300 有锚稀疏均值回归网格 V1：第一阶段事件信息预算

- 状态：`REJECTED_PHASE_A_NO_GRID_BACKTEST_NO_RESCUE`
- 研究阶段：`DISCOVERY_ONLY / NO_TRADE`
- 组合回测：未运行；第一阶段失败时明确禁止运行。
- 可交易资产仍只有 `510300.SH` 与 `CASH_CNY`；000300 只作观察锚点。

## 输入结论

- 输入审计：`PASS_DISCOVERY_INDEX_PROXY_INPUTS`。
- 分钟特征：291,851 行，1,211 个交易日。
- 锚点是除息调整后的指数隐含价格代理，不是历史 IOPV，也不宣称真实公平价值。
- 000300 原始 high/low 异常只登记，不回填；模型使用的同步 open/close 通过硬门。

## 固定成本

- 单层名义金额：20,000 元。
- 每腿有效佣金率：0.0250%。
- 基础完整往返成本：0.1500%。
- 1.5 倍压力成本：0.2250%。

## 45 分钟主结果

- 全部诊断事件：2,886。
- 流动性冲击且处于候选时段的事件：39。
- 达到第一层成本边界且结果完整的事件：0。
- 基础成本后平均事件收益：NOT_AVAILABLE。
- 压力成本后平均事件收益：NOT_AVAILABLE。
- ETF 自身反转贡献：NOT_AVAILABLE。
- 指数追赶贡献：NOT_AVAILABLE。
- 总收敛：NOT_AVAILABLE。
- 交易日移动块 Bootstrap 95% 下界：NOT_AVAILABLE。
- 正收益日历年数量：0。
- 最大单一正年份贡献占比：NOT_AVAILABLE。

## 硬门裁决

- FAIL：`completed_event_count_at_least_300`
- FAIL：`base_mean_net_return_positive`
- FAIL：`day_block_bootstrap_95pct_lower_bound_positive`
- FAIL：`at_least_four_positive_calendar_years`
- FAIL：`single_positive_year_contribution_not_above_50pct`
- FAIL：`stress_1_5x_cost_mean_positive`
- FAIL：`both_directions_count_and_net_mean_positive`
- FAIL：`etf_reversal_positive_and_dominates_anchor_catchup`
- FAIL：`severity_monotonic_with_minimum_bin_count`
- FAIL：`lower_path_efficiency_stronger_with_minimum_bin_count`
- FAIL：`nearby_threshold_platform_positive_with_minimum_counts`
- FAIL：`latest_two_calendar_years_positive`
- FAIL：`profit_not_mainly_open_or_tail`
- FAIL：`strongest_10pct_trend_days_do_not_consume_ordinary_positive_sum`
- PASS：`phase_a_uses_no_limit_fill_or_same_record_round_trip`

未通过门槛：completed_event_count_at_least_300, base_mean_net_return_positive, day_block_bootstrap_95pct_lower_bound_positive, at_least_four_positive_calendar_years, single_positive_year_contribution_not_above_50pct, stress_1_5x_cost_mean_positive, both_directions_count_and_net_mean_positive, etf_reversal_positive_and_dominates_anchor_catchup, severity_monotonic_with_minimum_bin_count, lower_path_efficiency_stronger_with_minimum_bin_count, nearby_threshold_platform_positive_with_minimum_counts, latest_two_calendar_years_positive, profit_not_mainly_open_or_tail, strongest_10pct_trend_days_do_not_consume_ordinary_positive_sum。

## 停止线

- `STOP_1_COST_ERASES_REVERSAL`：True
- `STOP_2_CONVERGENCE_NOT_ETF_LED`：True
- `STOP_3_PROFIT_MAINLY_OPEN_OR_TAIL`：True
- `STOP_4_TREND_TAIL_LOSS_TOO_LARGE`：True
- `STOP_5_WIDER_GRID_EVENT_BUDGET_INSUFFICIENT`：True
- `STOP_6_FICTITIOUS_SAME_RECORD_FILL`：False
- `STOP_7_RECENT_TWO_YEARS_DISAPPEAR`：True
- `STOP_8_SAME_EXPOSURE_BENCHMARK_NO_EXCESS`：NOT_EVALUATED_PHASE_A

## 最终决定

停止本分支；不运行网格组合回测，不改参数、窗口、成本、代理或子区间救援。

本报告不是收益承诺、仓位建议、订单或实盘授权。研究、Paper/Shadow、券商连接、持仓、订单和实盘状态保持严格分离。
