# 沪深300点时估值V2可重建性审计 V1.1

> 价格补采后，`VAL01_RAW_EY_5Y`数据闸门已经通过；标准化EY、EY利差和全部七年版本仍然NO_VIEW。本报告不包含收益、IC或仓位结果。

## 已解除的硬缺口

- 官方权重快照价格：60个月、18000行。
- 最低价格权重覆盖：99.9900%。
- 缺价/未来取价：0/0行。
- 与当前面板重叠收盘价完全一致率：100.0000%。

## 分支状态

- `VAL01_RAW_EY_5Y`：`PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY`。
- `VAL01_NORM_EY_5Y`：`PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED`；仍有9个月不足。
- `VAL02_NORM_EY_SPREAD_5Y`：`PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED`。
- 七年原始/标准化/利差：`BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED`。

## 允许的下一步

现在只允许冻结`VAL01_RAW_EY_5Y`模型卡，包括原始EY公式、亏损公司处理、月度权重可得时点、分位窗口、信号频率和试验登记。模型卡冻结前仍不运行未来收益、IC或仓位回测。
