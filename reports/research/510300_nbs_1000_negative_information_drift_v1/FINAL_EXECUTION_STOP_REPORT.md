# 510300 国家统计局 10 时负向信息漂移 V1：执行与停止报告

生成时间：2026-09-04T22:55:04.5965832+08:00

```text
MODEL_ID=510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1
TRADABLE_UNIVERSE=510300.SH+CASH_CNY_ONLY
MINUTE_PERMISSION=PASS_VIA_STK_MINS
RAW_MINUTE_ACQUISITION=PASS_RAW_ACQUISITION_COMPLETE
MINUTE_SOURCE_ADMISSION=BLOCKED_STK_MINS_SOURCE_ADMISSION
G0_PASS=FALSE
G1_PASS=FALSE
FINAL_STATE=BLOCKED_G0_SOURCE_ADMISSION
RETURN_EVALUATION=NOT_ALLOWED
EVENT_RETURN_READS=0
MODEL_TRAINING_RUN=FALSE
G2_RUN=FALSE
G3_RUN=FALSE
G4_RUN=FALSE
SHARPE_CALCULATED=FALSE
ORDER_GENERATED=FALSE
POSITION_IMPACT=0
```

## 一、权限与替代路径结论

没有依赖 `etf_mins` 专属接口继续推进。实际探针证明当前临时权限可通过 Tushare 兼容接口 `stk_mins` 读取 `510300.SH` 的 1 分钟历史：2017-01-03 单日返回 241 行，字段契约完整。`trade_cal` 与 `fund_daily` 探针也成功。

随后按自然月完成 2017-01-01 至 2026-09-04 的全量采集：

| 数据 | 分区 | 行数 |
| --- | ---: | ---: |
| `trade_cal` | 1 | 3,534 |
| `fund_daily` | 1 | 2,350 |
| `stk_mins` 1 分钟 | 117 | 566,350 |
| 合计分区 | 119 | — |

2,350 个开放交易日全部各有 241 根分钟线，满足 `2,350 × 241 = 566,350`。临时凭证未写入报告、清单或结构化数据。

## 二、分钟底座正式准入

通过项：

- 开放交易日任意分钟覆盖：2,350 / 2,350，100%；
- 标准 241 根覆盖：2,350 / 2,350，100%；
- 88 个已准入 NBS 事件的四个关键窗口完整：88 / 88，100%；
- 重复主键、非目标代码、非法 OHLC、负成交量或成交额、非法交易时刻、非开放日记录：全部为 0；
- 时间戳语义：`BAR_END` 证据通过；
- 2026-07-06 前日成交量最大相对误差：5.06017428377e-09；
- 2026-07-06 前日成交额最大相对误差：9.43603539649e-09。

失败项：

- 日级 OHLC 冻结门槛：至少 99.9%；
- 实际：2,201 / 2,350，93.6596%；
- 失败日：149。

该失败足以令 `GENERAL_SOURCE_PASS=FALSE`。官方基金收盘价规则可以解释多数简单末根 close 差异，但诊断替代后仍有 high/low 差异，且诊断通过率仅 95.9149%；因此没有改动冻结口径。

## 三、国家统计局事前事件账本

已保存并哈希国家统计局 2017—2026 年年度发布日程页面。共解析 110 条年度事件，研究截止日内 106 条：

| 分类 | 数量 |
| --- | ---: |
| 日程标示 10:00 | 102 |
| 非 10:00 | 4 |
| 非上交所开放日 | 1 |
| 无法证明事前日程版本 | 13 |
| 最终日程合格且四窗口完整 | 88 |
| 具备此前 60 个非事件日、可进入模型预检 | 86 |

13 条 `EXCLUDED_UNVERIFIABLE_SCHEDULE_REVISION` 来自 2022 年 1—9 月与 2024 年 1—6 月。国家统计局官网目前展示的对应年度表分别是后续修订版本；旧通知目录能证明原日程曾提前发布，但现有官方公开页面不能证明每个历史单元格在事件发生前的原始日期与时刻。按照冻结 PIT 规则，未用搜索摘要或事后修订表替代原始事前证据。

## 四、G0/G1 裁决

G0 失败：

- 分钟来源一般准入失败；
- 具体失败字段为日级 OHLC 对账率；
- 其余分钟覆盖、事件窗口、时间戳和 NBS 原始页面哈希检查通过。

G1 也独立失败：

| 条件 | 门槛 | 实际 | 结果 |
| --- | ---: | ---: | --- |
| 合格 10:00 事件 | 85 | 88 | PASS |
| 四窗口完整事件 | 80 | 88 | PASS |
| 首次训练事件 | 36 | 36 | PASS |
| Era 1 | 15 | 20 | PASS |
| Era 2 | 15 | 20 | PASS |
| Era 3 | 15 | 10 | FAIL |
| 首 36 个之后的评价事件 | 40 | 50 | PASS |

由于冻结规则要求所有门同时通过，流程停止于 G0/G1。没有读取 `P_pre`、`P_reaction`、`P_entry`、`P_exit`，没有构造 `g`、`z`、`X` 或 `Y`，没有训练 B0/B1，没有计算系数、信号、净值或 Sharpe。

## 五、不可变识别信息

| 产物 | 指纹或 SHA-256 |
| --- | --- |
| 分钟全量采集摘要 | `63567adefc6e708aa63e0099d4e142f6ff0384d421fed13409a587a7f560cad1` |
| 分钟来源裁决 | `6e1a63095b4a60feb4014f6fc0fa01af1b571b7b0de3c44cc83d22b0e0cac0a4` |
| 1 分钟 Parquet | `f39a9f66e0c4bc533cb8323b55b223af8ac218d744b9fcfa92ee1a76aebbfefa` |
| 日级质量账本 | `f84c6fed7e76f6221f3f837221e58b3a6797403db52d0a41bd93169c64c7d855` |
| 时间戳语义证据 | `11fdaa6f5d277c26b76acf5630185f58528ca079b10eb3639aff4868264794e4` |
| NBS 页面库存 | `a98135a5ad0bba1d7d56077061625b556fb6d299a69c336b1c9a28b51f2bb10b` |
| 预收益事件账本 | `f32d32749afea831d9f25cd8603a57cef19c09421836e0f140ecd1ea7db59d57` |
| 预收益事件账本 Parquet | `b924b47f748f617790f8cedb2c87f03a62a72cb1b7d158a7912479a967fb0164` |
| G0/G1 最终裁决 | `0573e3a663318e069cf13ba5c25d57a640a695f6179ff4a7ab70801cd7d61d3c` |

## 六、最终边界

本次结果确认 `stk_mins` 是可取得的历史分钟替代路径，但不确认该数据源满足本策略冻结的正式准入合同，也不提供 Alpha、Sharpe、Paper、仓位、订单或实盘授权。

本 V1 不得通过降低 99.9% 对账门、纳入 9:30 事件、缩短 Era 3 门槛、读取收益后挑数据源或修改事件窗口来救援。若未来取得可验证的上交所同口径分钟源，或取得带事前时间戳的国家统计局原始日程快照，应在新的版本化来源修复协议下独立评估；在此之前保持关闭。
