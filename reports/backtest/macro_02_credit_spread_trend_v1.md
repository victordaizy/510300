# 510300 MACRO-02：AAA信用利差压缩 × 长期趋势确认

- 决策：`REJECT_DISCOVERY_STOP_NO_RESCUE`
- 说明：冻结门槛失败：conditional_spread_positive, conditional_bootstrap_lower_positive, hac_compression_effect, chronological_halves, base_strategy_sharpe, stress_strategy_sharpe, base_monthly_excess_profit_factor, stress_monthly_excess_profit_factor, sharpe_improvement_vs_trend_only, base_cagr_vs_static_60, stress_cagr_vs_static_60, positive_subperiods。本版本停止，禁止改期限、窗口、均线、暴露或样本起点补救。
- 信号数/完整区间：110/109
- 决策范围：2017-06-30至2026-07-31；只用日线。

## 数据边界

- 510300行情：2016-08-12至2026-08-18，交叉源`PASS`。
- 3年信用利差：2016-08-12至2026-08-18，范围26.65至171.32bp。
- 官方固定日期核对通过3项；既有国债批次重叠2504日。
- TLS验证/未验证行：1498/1006；独立供应商验证=`false`。

## 预测层

- 趋势开启条件下，信用压缩/未压缩样本：35/31。
- 下一完整月度区间超现金收益均值差：-0.8000%。
- 6个月区块Bootstrap 95%区间：[-2.8969%, 1.1901%]。
- HAC每1bp压缩系数：-0.000392，单侧p=0.9271。

## 账户回测

| 路径 | CAGR | 超现金夏普 | 最大回撤 | 交易数 |
|---|---:|---:|---:|---:|
| 组合-基础成本 | 0.94% | 0.003 | -28.04% | 36 |
| 组合-压力成本 | 0.72% | -0.017 | -28.39% | 36 |
| 仅趋势-基础成本 | 2.08% | 0.110 | -40.72% | 18 |
| 仅信用压缩-基础成本 | -0.72% | -0.089 | -30.91% | 39 |
| 静态60%-基础成本 | 3.24% | 0.208 | -25.95% | 1 |
| 98.5%持有-基础成本 | 4.26% | 0.240 | -38.95% | 1 |

- 基础/压力月度超额盈利因子（相对静态60%）：0.750/0.732。
- 基础夏普相对仅趋势增量：-0.107。
- 基础/压力CAGR相对静态60%：-2.30%/-2.52%。

## 门槛

- 通过：`complete_signal_intervals`
- 通过：`conditional_group_support`
- 失败：`conditional_spread_positive`
- 失败：`conditional_bootstrap_lower_positive`
- 失败：`hac_compression_effect`
- 失败：`chronological_halves`
- 失败：`base_strategy_sharpe`
- 失败：`stress_strategy_sharpe`
- 失败：`base_monthly_excess_profit_factor`
- 失败：`stress_monthly_excess_profit_factor`
- 失败：`sharpe_improvement_vs_trend_only`
- 失败：`base_cagr_vs_static_60`
- 失败：`stress_cagr_vs_static_60`
- 失败：`positive_subperiods`
- 通过：`year_contribution_concentration`

## 治理结论

本报告不修改现行510300-only系统，不启用仓位映射，不连接券商，不生成订单。通过也只能补做独立数据源验证与前向Shadow协议；失败则本版本冻结停止。
