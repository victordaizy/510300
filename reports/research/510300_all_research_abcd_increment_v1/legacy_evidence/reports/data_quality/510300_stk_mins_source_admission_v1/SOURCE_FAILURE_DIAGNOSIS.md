# 510300 STK_MINS 来源失败诊断

生成时间：2026-09-04T22:55:04.5965832+08:00

```text
PHASE=POST_ADJUDICATION_SOURCE_DIAGNOSIS_ONLY
SOURCE_STATE=BLOCKED_STK_MINS_SOURCE_ADMISSION
FROZEN_GATE_CHANGED=FALSE
STRATEGY_RETURN_READS=0
POSITION_IMPACT=0
```

## 裁决

分钟历史已成功取得，权限、覆盖、主键、交易时刻、成交量与成交额检查均通过。唯一失败的冻结来源检查是日级 OHLC 对账：

- 冻结门槛：至少 99.9%；
- 实际通过：2,201 / 2,350，93.6596%；
- 失败交易日：149；
- 失败范围：2017-01-25 至 2021-03-26；
- 最大绝对价格差：0.020 元。

按 0.001 元绝对容差逐字段诊断：

| 字段 | 超容差交易日 |
| --- | ---: |
| open | 0 |
| high | 47 |
| low | 52 |
| 简单末根分钟线 close | 54 |

字段失败日存在重叠，合计为 149 个不同交易日。

## 官方收盘价规则诊断

上交所 2018 年规则说明：股票引入收盘集合竞价时，基金收盘价形成方式不调整；基金收盘价按最后一笔交易前一分钟全部交易的成交量加权平均价计算。官方链接：

https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20180806_10784917.shtml

以末根分钟线 `amount / volume` 并按三位小数舍入，仅作为诊断替代简单末根 `close`：

- close 超容差日从 54 降至 1，剩余日期为 2019-06-14；
- high 或 low 超容差仍有 95 日；
- high、low 或诊断 close 任一失败共 96 日；
- 诊断口径总通过率为 95.9149%，仍低于 99.9%。

因此，官方规则可以解释大部分简单末根 close 差异，但不能证明分钟 high/low 与 `fund_daily` 同源一致。该分析不构成来源准入，也不修改冻结适配器或门槛。

## 不变边界

```text
GENERAL_SOURCE_PASS=FALSE
STRATEGY_RESEARCH=NOT_ALLOWED
RETURN_EVALUATION=NOT_ALLOWED
SOURCE_DECISION_FINGERPRINT=6e1a63095b4a60feb4014f6fc0fa01af1b571b7b0de3c44cc83d22b0e0cac0a4
```

不得用上述诊断结果替换正式裁决，不得降低对账门槛，不得读取事件收益进行数据源择优。
