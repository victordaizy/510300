# 510300国家统计局10时负向信息漂移V1：预收益G0/G1裁决

生成时间：2026-09-04T22:47:42.219349+08:00

```text
STATE=BLOCKED_G0_SOURCE_ADMISSION
G0_PASS=FALSE
G1_PASS=FALSE
RETURN_EVALUATION=NOT_ALLOWED
EVENT_RETURN_READS=0
MODEL_TRAINING_RUN=FALSE
G2_RUN=FALSE
G3_RUN=FALSE
G4_RUN=FALSE
POSITION_IMPACT=0
```

## 分钟来源

- 来源状态：BLOCKED_STK_MINS_SOURCE_ADMISSION
- 开放日任意分钟覆盖率：100.0000%
- 标准241根覆盖率：100.0000%
- 日级OHLC对账率：93.6596%
- 合格事件四窗口覆盖率：1.0

## 官方事件账本

- 研究期内年度日程事件：106
- 日程标示10:00：102
- 非10:00：4
- 非交易日：1
- 无法证明事前日程版本：13
- 逐事件准入且四窗口完整：88
- 60日前序非事件日预检后可建模：86
- 首36个之后评价事件：50
- 时代：训练36，Era1=20，Era2=20，Era3=10

本阶段只使用官方日程、交易日和分钟完整性布尔字段。没有读取窗口价格、事件收益或标签，也没有训练模型、生成净值、Sharpe、仓位或订单。
