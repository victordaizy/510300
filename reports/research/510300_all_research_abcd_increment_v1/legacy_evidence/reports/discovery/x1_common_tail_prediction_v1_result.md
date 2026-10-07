# 510300 X1共同尾部未来风险预测 V1 历史结果

- 总状态：`HISTORICAL_REJECTED_FROZEN`
- 数据截止：`2026-08-14`
- 历史标签：`HISTORICALLY_CONTAMINATED`
- 预测资格：`NOT_AUTHORIZED`
- 组合、仓位、订单、券商：`DISABLED`

## 样本与原始X1

- 有效信号日：850
- 成熟主评价信号：830
- 走步概率预测：558
- BAD20无条件事件：66 / 830，发生率 0.07951807228915662
- 最高风险组事件：3 / 113，发生率 0.02654867256637168
- BAD20 Lift：0.333869670152856
- Bootstrap 95%区间：[0.0, 0.976489749746052]
- 一日延迟额外Lift保留率：None
- 前半段Lift：0.0
- 后半段Lift：0.6840659340659341

## 概率与R6增量

- X1 Brier：0.05606461205395601；无条件：0.05624579554493558；Skill：0.003221280617051847
- X1 Log Loss：0.23175379345533642；无条件：0.23299395411072468
- R6 Brier：0.09383670366077815；R6+X1：0.08834214409087048
- R6 Log Loss：0.3532938628768059；R6+X1：0.3436514244106932
- R6最高十分位BAD20率：0.017857142857142856；R6+X1：0.07142857142857142
- R6单位避险贡献：0.0010097383970269897；R6+X1：0.0015272333637630817

## 冻结门槛

- FAIL `bad20_lift`
- FAIL `bootstrap_lift_lower_bound`
- PASS `brier_skill`
- PASS `log_loss`
- FAIL `calibration_bias`
- FAIL `one_day_delay_retention`
- FAIL `first_half_direction`
- FAIL `second_half_direction`
- PASS `incremental_brier`
- PASS `incremental_log_loss`
- PASS `incremental_bad20_recognition`
- PASS `incremental_unit_avoidance`

## 边界

结果无论通过或失败均不映射仓位。失败不得修改5日、504日、5%尾部、90%风险组或20日目标补救。
