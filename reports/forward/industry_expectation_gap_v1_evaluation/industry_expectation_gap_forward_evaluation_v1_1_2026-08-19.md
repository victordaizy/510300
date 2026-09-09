# 行业预期差 V1 前瞻评价状态

- 预测日：`2026-08-18`
- 首个可捕获日：`2026-08-19` 开盘
- 当前观察日：`2026-08-19`
- 主要状态：`ACCUMULATING_NO_PEEK`
- 原指数观点：`NO_VIEW`（不可事后升级）

## 成熟度

| 期限 | 成熟日 | 日历 | 已积累共同交易日 | 状态 | 是否输出部分收益 |
|---:|---|---|---:|---|---|
| 60 | 2026-11-18 | RESOLVED | 1 | ACCUMULATING_NO_PEEK | 否 |
| 120 | 未解析 | UNRESOLVED_CALENDAR | 1 | UNRESOLVED_CALENDAR | 否 |

## 当前数据截止日

- 成分总收益：`2026-08-19`
- ETF含分红：`2026-08-19`
- 共同前瞻日期：`2026-08-19`

## 不可回写层

- 指数方向：`ABSTAINED_AT_FORECAST`
- 市场流动性：`NOT_SCORABLE_MISSING_AT_FORECAST`
- 国家队：`NOT_SCORABLE_UNOBSERVED_AT_FORECAST`

## 样本边界

当前成熟预测起点数为 `0`；校准至少需要 `20` 个，模型比较至少需要 `40` 个。

未成熟期不输出部分收益；成熟后只评价冻结行业分支，不得把原NO_VIEW改写为交易观点。

## 冻结输入恢复

- 执行层：`INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_1`
- 仅恢复原冻结输入身份；预测、规则、阈值均未变化。
- 当前 PCF/IOPV 就绪报告未被替换。
