# 510300期权不利成交价包络V1开发期结果

- 状态：`DEV_ORACLE_VEHICLE_FEASIBLE_NOT_A_STRATEGY`
- 证据：`DEVELOPMENT_ORACLE_ONLY_NO_VALIDATION_OR_HOLDOUT_READ`
- 区间：2021-08-13 至 2023-12-21
- 非重叠周期：52
- 2024验证期与2025后最终留出期：未读取

|情景|策略年化|基准年化|年化超额|期末权益|
|---|---:|---:|---:|---:|
|ALWAYS_CALL|-42.64%|-13.61%|-29.03%|5403.00|
|PERFECT_UNDERLYING_DIRECTION_ORACLE|30.47%|-13.61%|44.08%|37411.00|
|PERFECT_OPTION_PNL_ORACLE|33.55%|-13.61%|47.17%|39525.00|

## 解释

`PERFECT_UNDERLYING_DIRECTION_ORACLE`使用未来十日指数方向，
`PERFECT_OPTION_PNL_ORACLE`直接选择事后损益更高的一侧；二者都不可能实盘，
只回答车辆是否存在足够上界。入场按次日最高成交价、退出按退出日最低成交价并加费用。
由于历史买卖盘仍不可见，本结果不是历史可执行证明，也不授权策略、仓位或订单。

## 门槛

- 完美方向上界年化超额至少20%：True
- 所有实际开仓至少5000元：True
- 可进入不超过10因子的模型研究：True