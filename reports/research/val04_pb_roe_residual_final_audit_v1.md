# VAL-04 PB—ROE 残余收益研究终局审计 V1

> 状态：`HISTORICAL_REJECTED_FROZEN_NO_STRATEGY_BACKTEST`。

## 分支状态

- 中信行业数据 V1：`NO_VIEW_DATA_GATE_FAILED_MODEL_FIT_FORBIDDEN`。
- 申万点时行业数据 V2：`PASS_MODEL_CARD_MAY_BE_FROZEN`。
- 无收益信号构建：`PASS_PREDICTIVE_PROTOCOL_MAY_BE_FROZEN`。
- 预测屏幕：`REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST`。

## 242日主要证据

- 成熟样本：49。
- 整体 ETF Spearman IC：0.401452。
- 前半/后半 IC：0.423346 / -0.168341。
- HAC 单侧 p / Holm p：0.028549 / 0.028549。
- 最便宜减最贵平均收益：26.2959%。
- 失败闸门：`['both_chronological_half_ics_strictly_positive', 'etf_bucket_mean_returns_nondecreasing']`。

## 结论

整体相关和端点价差为正，但后半样本方向反转，且最便宜档没有继续优于第三档。按预冻结规则必须拒绝；不得运行策略回测或参数救援。

## 治理

- 未计算策略净值或交易成本。
- 未映射当前信号、仓位、目标股数或订单。
- 未连接券商；本报告不是买卖指令。
