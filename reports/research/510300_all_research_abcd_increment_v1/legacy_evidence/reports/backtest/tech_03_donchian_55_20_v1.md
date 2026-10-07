# TECH_03_DONCHIAN_55_20_V1 历史诊断

> 结论：`REJECT_HISTORICAL`。`alpha_pass=false`；这是截至2026-08-14的受污染历史，不是当前买卖指令。

## 核心结果（2万元执行层）

| 方案 | CAGR | 对510300年化主动收益 | IR | Sharpe | 最大回撤 | 平均仓位 | 成交笔数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 55/20基础成本 | 6.47% | 0.12% | -0.07 | 0.43 | -24.11% | 35.65% | 52 |
| 55/20双倍成本 | 6.16% | -0.18% | -0.09 | 0.41 | -25.17% | 35.63% | 52 |
| 100/50稳健性 | 6.74% | 0.16% | -0.06 | 0.43 | -27.56% | 46.10% | 22 |
| 510300含分红买入持有 | 6.34% | 0.00% | — | 0.33 | -44.12% | 90.76% | 1 |
| H00300全收益 | 7.40% | — | — | 0.37 | -46.06% | 100.00% | — |

## 判定

- 历史屏幕：`FAIL`；风险覆盖历史门槛：`FAIL`。
- Sharpe改善：0.10；最大回撤降幅：45.36%；上涨捕获率：40.53%。
- 四段中主动收益为正：3/4；剔除最佳年份后年化主动收益：-2.19%。
- 242日滚动超额中位数/正值比例：0.05% / 50.03%。
- 484日滚动超额中位数/正值比例：3.19% / 55.21%。

## 多重检验

- 累计登记试验3个，实际收益候选2个；White Reality Check p=0.6919。
- Hansen SPA区块自助近似 p=0.6925；Deflated Sharpe概率=12.60%。
- PBO：`NOT_ESTIMABLE`，原因：实际可比较收益候选少于3，CSCV排名退化。
- 相邻参数网格未运行，因为冻结预算禁止继续搜索；固定100/50已完整报告。

## 数据与执行口径

- 数据闸门：`PASS`；日期截止2026-08-14。
- H00300官方本地快照只有收盘；信号OHLC是用`H00300收盘/000300收盘`缩放000300 OHLC得到的冻结代理，不冒充官方H00300 OHLC。
- T日收盘信号、T+1开盘执行；只在状态改变时交易；100份整手、普通调仓至少1000份、每腿最低5元、5bp滑点、T+1。
- 理论层与2万元执行层分开保存；每个信号、执行和闭合周期均有独立证据文件。
- 当前仓位映射、订单生成和券商连接均为关闭状态。

## 历史门槛明细

| 门槛 | 结果 |
|---|---|
| `annualized_net_excess_at_least_1_5pct` | `FAIL` |
| `information_ratio_at_least_0_35` | `FAIL` |
| `positive_predefined_periods_at_least_3` | `PASS` |
| `double_cost_excess_positive` | `FAIL` |
| `fixed_100_50_robustness_excess_positive` | `PASS` |
| `remove_best_year_excess_positive` | `FAIL` |
| `single_positive_year_contribution_not_over_50pct` | `PASS` |
| `white_reality_check_5pct` | `FAIL` |
| `hansen_spa_5pct` | `FAIL` |
| `deflated_sharpe_probability_95pct` | `FAIL` |
