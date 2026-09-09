# VAL-04 PB—ROE 残余收益预测诊断 V1

> 状态：`REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST`。历史信号为事后重建，不是严格样本外。

## 242日主要结果

- 成熟样本：49。
- ETF Spearman IC：0.40145151523691297。
- 前半/后半 IC：0.4233455901885288 / -0.1683410231037886。
- HAC 单侧原始 p：0.02854855627032867；Holm 调整 p：0.02854855627032867。
- ETF 最便宜减最贵平均收益：0.26295928149920195。
- H00300 Spearman IC：0.4036996845980463。
- 失败闸门：['both_chronological_half_ics_strictly_positive', 'etf_bucket_mean_returns_nondecreasing']。

|分档|样本|ETF均值|ETF中位数|
|---|---:|---:|---:|
|B1_EXPENSIVE|5|-18.2093%|-19.1750%|
|B2|8|-7.5309%|-6.9787%|
|B3|17|10.7407%|18.5759%|
|B4_CHEAP|19|8.0866%|10.5307%|

## 边界

- 未计算策略净值、交易成本、仓位、当前建议、目标股数或订单。
