# 510300模型层重建V1：统一历史诊断

- 评价期：2021-08-12 至 2026-08-12（1211日）
- 证据标签：`HISTORICALLY_CONTAMINATED / TRUE_OOS_NOT_STARTED`
- 研究边界：`RESEARCH_ONLY / NO_POSITION_CHANGE / NO_ORDER`
- 技术数据闸门：`PASS`
- 估值数据闸门：`NO_VIEW_BLOCKED_NON_VINTAGE_HISTORY`
- R5原始重放：`BLOCKED_SOURCE_DRIFT_CURRENT_MARKET_CAP_MISSING`；拆分输入：`PASS_PRESERVED_DERIVED_SIGNAL_SNAPSHOT`

## 结论先行

- 组合研究：`STOPPED_NO_COMBINATION_TEST`。VAL-01与VAL-02均为NO_VIEW，独立估值线未通过数据闸门；按冻结协议禁止组合优化。
- 技术家族多重检验：White p=0.6863，SPA p=0.6845，PBO=55.71%。
- VAL-01/VAL-02只完成公式和覆盖率审计，未生成信号、收益、IC或仓位结论。

## 第一批独立技术模型（2万元执行层，净成本）

| 模型 | CAGR | 年化主动 | IR | Sharpe | 最大回撤 | 曝险 | 交易数 | 双倍成本主动 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TECH_01_TSMOM_63_126_252 | 0.16% | -0.43% | -0.10 | -0.09 | -20.47% | 42.69% | 60 | -1.14% | `REJECT_HISTORICAL` |
| TECH_03_DONCHIAN_55_20 | -2.53% | -3.11% | -0.31 | -0.32 | -18.20% | 30.26% | 24 | -3.45% | `REJECT_HISTORICAL` |
| TECH_03_DONCHIAN_55_20_25_FLOOR | -1.69% | -2.28% | -0.30 | -0.23 | -18.91% | 44.70% | 24 | -2.62% | `REJECT_HISTORICAL` |
| TECH_03_DONCHIAN_100_50_ROBUSTNESS | -1.07% | -1.66% | -0.19 | -0.19 | -14.33% | 33.25% | 10 | -1.80% | `REJECT_HISTORICAL` |
| TECH_05_TREND_PULLBACK | 1.85% | 1.27% | -0.01 | 0.23 | -1.04% | 2.38% | 49 | 0.88% | `REJECT_HISTORICAL` |
| TECH_08_BREADTH_TREND | -1.31% | -1.90% | -0.21 | -0.31 | -15.72% | 30.16% | 73 | -2.77% | `REJECT_HISTORICAL` |

### 理论层与2万元包装差异

| 模型 | 理论CAGR | 执行CAGR | CAGR差 | 理论主动 | 执行主动 | 主动差 | 曝险差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TECH_01_TSMOM_63_126_252 | 0.14% | 0.16% | 0.02% | -0.43% | -0.43% | 0.00% | -1.15% |
| TECH_03_DONCHIAN_55_20 | -2.54% | -2.53% | 0.01% | -3.11% | -3.11% | -0.01% | -0.31% |
| TECH_03_DONCHIAN_55_20_25_FLOOR | -1.64% | -1.69% | -0.05% | -2.21% | -2.28% | -0.07% | -2.68% |
| TECH_03_DONCHIAN_100_50_ROBUSTNESS | -1.12% | -1.07% | 0.04% | -1.68% | -1.66% | 0.02% | -0.52% |
| TECH_05_TREND_PULLBACK | 1.99% | 1.85% | -0.14% | 1.42% | 1.27% | -0.16% | -0.08% |
| TECH_08_BREADTH_TREND | -1.27% | -1.31% | -0.04% | -1.83% | -1.90% | -0.06% | -0.57% |

### 2万元执行与成本

| 模型 | 调仓腿数 | 平均持仓段（日） | 换手/平均权益 | 佣金（元） | 滑点（元） | 总成本（元） | 最低佣金占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TECH_01_TSMOM_63_126_252 | 60 | 74.0 | 23.41 | 300.25 | 221.47 | 521.72 | 98.33% |
| TECH_03_DONCHIAN_55_20 | 24 | 30.9 | 23.45 | 130.85 | 218.09 | 348.94 | 0.00% |
| TECH_03_DONCHIAN_55_20_25_FLOOR | 24 | 1132.0 | 18.17 | 121.02 | 165.89 | 286.91 | 95.83% |
| TECH_03_DONCHIAN_100_50_ROBUSTNESS | 10 | 82.0 | 9.56 | 55.55 | 92.59 | 148.14 | 0.00% |
| TECH_05_TREND_PULLBACK | 49 | 4.4 | 13.78 | 245.00 | 143.24 | 388.24 | 100.00% |
| TECH_08_BREADTH_TREND | 73 | 19.4 | 41.82 | 371.30 | 390.76 | 762.07 | 83.56% |

### 稳定性与多重检验

| 模型 | H1主动 | H2主动 | 正主动年度占比 | 剔除最佳年后年化主动 | 正贡献最大年占比 | 242日滚动为正 | 484日滚动为正 | DSR概率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TECH_01_TSMOM_63_126_252 | 28.72% | -23.95% | 33.33% | -5.42% | 91.71% | 39.59% | 34.07% | 5.50% |
| TECH_03_DONCHIAN_55_20 | 25.52% | -31.93% | 33.33% | -8.29% | 89.67% | 37.01% | 32.97% | 1.99% |
| TECH_03_DONCHIAN_55_20_25_FLOOR | 18.47% | -24.72% | 33.33% | -6.29% | 90.50% | 36.91% | 32.83% | 2.05% |
| TECH_03_DONCHIAN_100_50_ROBUSTNESS | 34.20% | -31.42% | 50.00% | -7.56% | 88.91% | 38.66% | 34.75% | 3.66% |
| TECH_05_TREND_PULLBACK | 47.27% | -27.71% | 50.00% | -3.94% | 63.24% | 51.24% | 50.82% | 8.43% |
| TECH_08_BREADTH_TREND | 28.02% | -28.98% | 33.33% | -5.75% | 63.70% | 44.54% | 37.09% | 3.38% |

White和SPA是全家族门槛，不能解释为某个单模型的显著性；PBO使用8折CSCV的70个对称拆分。所有候选还被`TRUE_OOS_NOT_STARTED`统一阻断。

### 逐模型未通过门槛

- `TECH_01_TSMOM_63_126_252`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, double_cost_active_return_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, sharpe_improvement_at_least_0_15, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct
- `TECH_03_DONCHIAN_55_20`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, double_cost_active_return_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, sharpe_improvement_at_least_0_15, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct
- `TECH_03_DONCHIAN_55_20_25_FLOOR`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, double_cost_active_return_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, sharpe_improvement_at_least_0_15, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct
- `TECH_03_DONCHIAN_100_50_ROBUSTNESS`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, double_cost_active_return_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, sharpe_improvement_at_least_0_15, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct
- `TECH_05_TREND_PULLBACK`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct
- `TECH_08_BREADTH_TREND`：annualized_active_return_at_least_1_5pct, information_ratio_at_least_0_35, both_chronological_halves_positive, double_cost_active_return_positive, remove_best_year_active_return_positive, positive_year_ratio_majority, single_positive_year_contribution_not_over_50pct, sharpe_improvement_at_least_0_15, upside_capture_at_least_70pct, white_reality_check_5pct, hansen_spa_5pct, pbo_at_most_25pct, deflated_sharpe_probability_95pct

## R5八路拆分（相同2万元包装）

| 变体 | CAGR | 年化主动 | IR | Sharpe | 最大回撤 | 平均曝险 | 交易数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R5_000_BUY_HOLD | 0.58% | 0.00% | — | 0.03 | -33.44% | 93.55% | 1 |
| R5_100_VALUATION_ONLY | 1.25% | 0.67% | 0.03 | 0.05 | -32.05% | 71.80% | 41 |
| R5_010_TREND_ONLY | 0.62% | 0.03% | -0.11 | -0.04 | -19.99% | 56.28% | 45 |
| R5_001_RISK_ONLY | 0.53% | -0.05% | 0.05 | 0.03 | -34.18% | 96.52% | 5 |
| R5_110_VALUATION_TREND | 1.41% | 0.82% | 0.06 | 0.06 | -32.38% | 74.71% | 43 |
| R5_101_VALUATION_RISK | 1.23% | 0.65% | 0.02 | 0.04 | -32.04% | 70.52% | 42 |
| R5_011_TREND_RISK | 0.71% | 0.12% | -0.10 | -0.04 | -16.39% | 52.74% | 53 |
| R5_111_FULL | 1.17% | 0.59% | -0.00 | 0.03 | -30.82% | 68.96% | 55 |

### R5归因

Shapley值按八个组合的年化主动收益计算；它是描述性归因，不是因果识别。

- V：0.63%
- T：0.05%
- R：-0.09%

交互项（组合主动收益的离散差分）：

- VxT：0.12%
- VxR：0.03%
- TxR：0.14%
- VxTxR：-0.35%

## 估值线为何是NO_VIEW

- `BLOCKED_VENDOR_VALUATION_NOT_PROVEN_POINT_IN_TIME_VINTAGE`
- `BLOCKED_POINT_IN_TIME_FUNDAMENTAL_HISTORY_SHORTER_THAN_7_YEARS`
- `BLOCKED_NO_5Y_OR_7Y_WARMUP_AT_2021_EVALUATION_START`

日度供应商PE虽可计算公式，但不能证明每个历史截面是当时可见且未经未来修订；点时月度财务仅有60个月，无法在2021-08-12评价起点提供5年/7年暖机。因此任何估值回报或组合结果都会违反点时性闸门。

## 固定仓位与主基准

| 基准 | CAGR | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| FIXED_25PCT | 1.64% | 0.05 | -7.45% |
| FIXED_50PCT | 1.56% | 0.05 | -17.38% |
| FIXED_75PCT | 1.29% | 0.05 | -26.53% |
| FIXED_100PCT | 0.83% | 0.05 | -34.91% |
| 510300含分红买入持有（2万元执行） | 0.58% | 0.03 | -33.44% |
| 510300含分红买入持有（理论） | 0.57% | 0.03 | -34.10% |
| H00300全收益 | 1.39% | 0.08 | -34.40% |

## 解释限制

- 整段历史已被研究过程观察，所有结果只能称历史诊断，不能称样本外Alpha。
- R5八路拆分依赖冻结派生信号快照；当前原始成分市值源漂移导致无法原始重放。
- 多重检验是近似实现；结论同时受真实前向尚未开始这一更强约束阻断。
- 本报告不输出当日观点、目标仓位或订单。
