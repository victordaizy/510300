# 510300 MACRO-01：M2增速加速度 × 长期趋势确认

- 决策：`REJECT_DISCOVERY_STOP_NO_RESCUE`
- 说明：冻结门槛失败：conditional_spread_positive, conditional_bootstrap_lower_positive, hac_m2_effect, chronological_halves, base_strategy_sharpe, stress_strategy_sharpe, base_monthly_excess_profit_factor, stress_monthly_excess_profit_factor, sharpe_improvement_vs_trend_only, base_cagr_vs_static_60, stress_cagr_vs_static_60。本版本停止，禁止改窗口、均线、暴露或样本起点补救。
- 信号数/完整区间：96/95
- 使用M2月份：截至`2026-06`；执行日线，无分钟线。

## 数据边界

- 510300行情：2016-08-12至2026-08-18，交叉源`PASS`。
- M2：2017-01-31至2026-07-31，官方抽查通过8项。
- 点时限制：`formal_point_in_time_evidence=false`。本文件是当前历史版本，不是逐月公告原始快照；只能支持DISCOVERY_ONLY，不能直接授权Paper或实盘。

## 预测层

- 趋势开启条件下，M2加速/未加速样本：36/18。
- 下一完整区间超现金收益均值差：-0.5071%。
- 6个月区块Bootstrap 95%区间：[-4.0991%, 2.1861%]。
- HAC中M2系数：0.002712，单侧p=0.2623。

## 账户回测

| 路径 | CAGR | 超现金夏普 | 最大回撤 | 交易数 |
|---|---:|---:|---:|---:|
| 组合-基础成本 | 3.27% | 0.203 | -18.89% | 15 |
| 组合-压力成本 | 3.15% | 0.194 | -19.12% | 15 |
| 仅趋势-基础成本 | 4.51% | 0.275 | -33.58% | 15 |
| 仅M2-基础成本 | 3.61% | 0.215 | -28.65% | 15 |
| 静态60%-基础成本 | 4.49% | 0.303 | -27.55% | 1 |
| 98.5%持有-基础成本 | 6.12% | 0.337 | -39.45% | 1 |

- 基础/压力月度超额盈利因子（相对静态60%）：0.893/0.882。
- 基础夏普相对仅趋势增量：-0.071。
- 基础/压力CAGR相对静态60%：-1.22%/-1.33%。

## 门槛

- 通过：`complete_signal_intervals`
- 通过：`conditional_group_support`
- 失败：`conditional_spread_positive`
- 失败：`conditional_bootstrap_lower_positive`
- 失败：`hac_m2_effect`
- 失败：`chronological_halves`
- 失败：`base_strategy_sharpe`
- 失败：`stress_strategy_sharpe`
- 失败：`base_monthly_excess_profit_factor`
- 失败：`stress_monthly_excess_profit_factor`
- 失败：`sharpe_improvement_vs_trend_only`
- 失败：`base_cagr_vs_static_60`
- 失败：`stress_cagr_vs_static_60`
- 通过：`positive_subperiods`
- 通过：`year_contribution_concentration`

## 治理结论

本报告不修改现行510300-only系统，不启用仓位映射，不连接券商，不生成订单。通过也只能进入逐月历史公告快照重建；失败则本版本冻结停止。
