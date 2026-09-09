# QDII跨市场折价回归零方差评估修正V2可见期结果

状态：`REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN`

证据身份：`VERSIONED_EVALUATION_CORRECTION_NOT_NEW_HOLDOUT`。同一可见路径仅作版本化评估修正，不是新未见样本。

## 核心结果

- 可见期：2017-01-03 至 2022-12-30。
- H00300全收益CAGR：4.78%。
- 基础/压力策略净CAGR：2.19%/1.54%。
- 基础/压力年化净超额：-2.59%/-3.24%。
- 基础/压力净夏普：0.140/0.039。
- 基础/压力最大回撤：-8.26%/-8.69%。
- 买入/卖出交易数：16/16。
- 最终基础/压力净值：569663.45/548392.00元。

## 零方差处理

- 基础/压力滚动242日不可识别窗口：521/521。
- 基础/压力Bootstrap不可识别样本：0/0。
- 互不重叠242日块总数/零方差块数/合格块数：6/2/0。

## 互不重叠242日块

- 2017-01-03 至 2017-12-27：基础/压力超额 -21.39%/-21.39%，基础/压力夏普 -1.000/-1.000，状态 DEFINED/DEFINED，合格=False。
- 2017-12-28 至 2018-12-25：基础/压力超额 24.21%/24.21%，基础/压力夏普 不可识别/不可识别，状态 UNDEFINED_ZERO_VARIANCE/UNDEFINED_ZERO_VARIANCE，合格=False。
- 2018-12-26 至 2019-12-24：基础/压力超额 -35.75%/-35.99%，基础/压力夏普 -1.360/-1.482，状态 DEFINED/DEFINED，合格=False。
- 2019-12-25 至 2020-12-23：基础/压力超额 -22.87%/-23.35%，基础/压力夏普 0.676/0.589，状态 DEFINED/DEFINED，合格=False。
- 2020-12-24 至 2021-12-22：基础/压力超额 1.61%/1.61%，基础/压力夏普 不可识别/不可识别，状态 UNDEFINED_ZERO_VARIANCE/UNDEFINED_ZERO_VARIANCE，合格=False。
- 2021-12-23 至 2022-12-21：基础/压力超额 24.28%/21.10%，基础/压力夏普 0.239/0.022，状态 DEFINED/DEFINED，合格=False。

## 全部硬门

- `base_annualized_excess_at_least_40pct`：False。
- `stress_annualized_excess_at_least_40pct`：False。
- `base_strategy_sharpe_defined`：True。
- `stress_strategy_sharpe_defined`：True。
- `base_strategy_sharpe_at_least_1_5`：False。
- `stress_strategy_sharpe_at_least_1_5`：False。
- `base_rolling_excess_median_at_least_40pct`：False。
- `stress_rolling_excess_median_at_least_40pct`：False。
- `base_rolling_sharpe_all_windows_defined`：False。
- `stress_rolling_sharpe_all_windows_defined`：False。
- `base_rolling_sharpe_median_at_least_1_5`：False。
- `stress_rolling_sharpe_median_at_least_1_5`：False。
- `base_bootstrap_excess_lower_bound_positive`：False。
- `stress_bootstrap_excess_lower_bound_positive`：False。
- `base_bootstrap_sharpe_all_samples_defined`：True。
- `stress_bootstrap_sharpe_all_samples_defined`：True。
- `base_bootstrap_sharpe_lower_bound_at_least_1`：False。
- `stress_bootstrap_sharpe_lower_bound_at_least_1`：False。
- `minimum_non_overlapping_year_blocks`：True。
- `minimum_target_qualified_year_blocks`：False。
- `data_quality_complete`：True。
- `capacity_pass`：True。

全部可见期门通过：False。

## 决策

冻结拒绝本候选；不打开封存期，不改变经济参数救回，转向新的独立机制

V1失败原样保留；封存复验、Paper、Shadow、订单、券商连接和实盘均未打开。
