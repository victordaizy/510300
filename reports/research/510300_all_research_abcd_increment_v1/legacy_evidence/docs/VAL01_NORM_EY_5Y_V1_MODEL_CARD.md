# VAL01_NORM_EY_5Y_V1 冻结模型卡

状态：`RESEARCH_ONLY / SIGNAL_INPUT_FREEZE / HISTORICALLY_CONTAMINATED`  
登记试验：`VAL01_NORM_EY_5Y`  
版本：`1.0.0`  
本阶段收益、IC、仓位、订单与券商连接：`全部禁用`

## 1. 假设与边界

假设是：按当时已披露财务信息计算的公司周期正常化盈利，经当月沪深300官方权重聚合后，正常化盈利收益率越高，随后中长期可交易收益应越高。本模型是慢速估值候选，不承担短线进入、退出或风险覆盖职责。

历史权重、财务和价格在2026年统一回取，虽然严格执行`available_at <= 快照日`，仍缺少逐日不可变供应商存档，因此全部回溯结果标记为`HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS`，不能取得正式`ALPHA_PASS`。

## 2. 公司正常化盈利

每个快照日只使用当日已经可得的最新财务版本。对公司选择最新可同时计算TTM归母利润与TTM收入、且TTM收入为正的报告期。在该报告期向前3年内至少需要4个可计算TTM观测。

固定公式如下：

```text
ProfitComponent_i,t = median(positive TTM parent profits in 3-year window)
MarginComponent_i,t = current TTM revenue × median(TTM parent margin in 3-year window)
NormalizedProfit_i,t = mean(valid positive components)
NormalizedEPS_i,t = NormalizedProfit_i,t / latest valid total shares
NormalizedEY_i,t = NormalizedEPS_i,t / unadjusted close_i,t
```

两项都有效时固定等权；仅一项有效时使用该项。不得改变三年窗口、四个观测下限、正利润规则或两项等权规则，不使用行业标签，不做缩尾或参数搜索。

## 3. 指数聚合与覆盖

快照固定为2016年8月至2026年7月的120个官方月度权重日，每月300只。价格使用快照日或此前最近真实成交日的未复权收盘价，禁止向后取价。

```text
IndexNormalizedEY_t = Σ Weight_i,t × NormalizedEY_i,t / Σ Weight_i,t(valid)
```

价格权重覆盖必须不低于99%，正常化盈利权重覆盖必须不低于90%。低于门槛的月份不得形成信号。

## 4. 五年分位与候选仓位

窗口固定60个连续月度快照并包含当月，使用未四舍五入`float64`值按精确midrank计算：

```text
Q_t = [#(EY_j < EY_t) + 0.5 × #(EY_j = EY_t)] / 60
```

首个允许信号日为2021-07-30，最早执行日为2021-08-02开盘。候选映射预先冻结为`[0,.2)→25%`、`[.2,.4)→50%`、`[.4,.6)→75%`、`[.6,1]→100%`，本阶段不应用该映射。

## 5. 后续预测屏幕

主要目标固定为信号日后首个共同交易日开盘进入、持有242个共同交易日至收盘的510300含分红毛收益；H00300同期限全收益作确认，60/120日只作辅助诊断。成熟样本按时间中点拆为前后两半，两半必须同向。VAL-01与VAL-02的主要HAC检验使用预登记的Holm-Bonferroni家族校正，家族显著性水平0.10。

预测屏幕通过也只允许进入独立策略评估；失败则停止，不得改参数或用仓位回测掩盖失败。

## 6. 禁止事项

不得重开已拒绝的原始EY分支；不得改变窗口、正常化公式、覆盖门槛、分位边界、主要期限或T+1开盘入场规则；不得在预测屏幕前生成历史仓位、当前建议、份额、订单或连接券商。
