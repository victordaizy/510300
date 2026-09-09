# 行业预期差 V1 前瞻评价状态

- 预测日：`2026-08-18`
- 首个可捕获日：`2026-08-19` 开盘
- 当前观察日：`2026-08-18`
- 主要状态：`WAITING_FOR_ENTRY`
- 原指数观点：`NO_VIEW`（不可事后升级）

## 成熟度

| 期限 | 成熟日 | 日历 | 已积累共同交易日 | 状态 | 是否输出部分收益 |
|---:|---|---|---:|---|---|
| 60 | 2026-11-18 | RESOLVED | 0 | WAITING_FOR_ENTRY | 否 |
| 120 | 未解析 | UNRESOLVED_CALENDAR | 0 | UNRESOLVED_CALENDAR | 否 |

## 当前数据截止日

- 成分总收益：`2026-08-14`
- ETF含分红：`2026-08-11`
- 共同前瞻日期：`None`

## 不可回写层

- 指数方向：`ABSTAINED_AT_FORECAST`
- 市场流动性：`NOT_SCORABLE_MISSING_AT_FORECAST`
- 国家队：`NOT_SCORABLE_UNOBSERVED_AT_FORECAST`

## 样本边界

当前成熟预测起点数为 `0`；校准至少需要 `20` 个，模型比较至少需要 `40` 个。

未成熟期不输出部分收益；成熟后只评价冻结行业分支，不得把原NO_VIEW改写为交易观点。
