# 510300 全市场换手率可见度 V1

- 项目标识：`510300_AGGREGATE_TURNOVER_VISIBILITY_V1`
- 最终状态：`REJECTED_FROZEN_AGGREGATE_TURNOVER_VISIBILITY_PREDICTIVE_GATE_FAILED_NO_RESCUE`
- 机制门：`FAIL`
- 组合收益是否获准计算：`False`
- RETURN_EVALUATION：`NOT_ALLOWED`
- NET_SHARPE：`NOT_COMPUTED`
- 实盘授权：`false`

## 冻结候选

因子为两市官方月度成交额之和除以对应月末总市值。上交所严格取主板A股加科创板；深交所取官方Stocks合计，含极小的B股/存托凭证口径，该代理在结果后不得更换。

2014至2025深交所数值来自官方年鉴重建；这不是年鉴在历史月末已发布的时间戳证明。真实前瞻运行必须在下一交易日开盘前独立捕获官方输入，缺失即NO_VIEW。

论文固定预测式为 `forecast = -0.019 + 0.258 × turnover`；仓位为 `clip(forecast / (5 × 0.0901²), 0, 1)`。2015年前510300收益不参与拟合或评价。

## 机制门结果

- 完整月度目标：139
- Newey-West 斜率：-0.09357527973795107
- Newey-West t 值：-1.6615132732104878
- 90% 移动块自助区间：[-0.17772244156316366, -0.008784841839697778]

机制门明细：

- `minimum_complete_monthly_targets`：`True`
- `full_sample_slope_positive`：`False`
- `newey_west_one_sided_t`：`False`
- `bootstrap_90pct_lower_positive`：`False`
- `every_structural_period_slope_positive`：`False`

## 组合评价

机制门失败，按冻结协议禁止计算策略收益、夏普率或任何替代仓位。

## 判定边界

历史通过也只能进入独立前瞻确认，不能生成订单、连接券商或改变仓位；本报告始终为研究用途。
