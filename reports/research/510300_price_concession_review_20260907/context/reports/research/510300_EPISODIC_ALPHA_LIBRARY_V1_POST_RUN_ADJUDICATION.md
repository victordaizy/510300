# 510300 短周期情景 Alpha 库 V1：运行后状态裁决附录

本附录是状态语义纠正，不修改冻结协议、252 日窗口、阈值、成本、时钟或任何候选规则，也不覆盖原始结果。父冻结清单内容摘要为 `5576c79749c4c07b21338b37145194248324dbae145d30d0fb3893825b4a7079`。

冻结的统一前向规则为“2026-09-01 或其后的首个上交所交易日”。根据上交所 2026 年休市安排与本地交易日历，实际前向起点解析为 `2026-09-01`；此前所有事件对前向激活计数仍为 0。

## 权威状态

| 项目 | 状态 |
|---|---|
| `510300_INFORMED_CREATION_PREMIUM_V1` | `NO_VIEW_DATA_CONTRACT_FAILED` |
| `510300_DERIVATIVE_PRESSURE_RELEASE_V1` | `NO_VIEW_FEATURE_CONSTRUCTION_FAILED_FROZEN_252D_WINDOW` |
| `510300_OPENING_DISCOUNT_RECOVERY_V1` | `RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS` |
| 当前已验证高夏普策略 | `NONE` |
| 当前合法路由 | `CASH_CNY`，仓位 0% |
| 实盘交易 | `FALSE` |

## 为什么衍生品分支不是“0 个压力事件”

原始运行正确保留了冻结的 252 日完整窗口，却在输出状态上把不可构造误写成了样本不足。近月 IF 合约每次移仓时，`DELTA_FRONT_OI` 按“同一合约、非移仓日”定义必然缺失。2427 行特征中该变量有 2306 个非空值，但任一此前 252 个交易日窗口最多只有 240 个有效值；因此 `DELTA_OI_Z=0` 个有效行，四分量 `Pressure=0` 个有效行。

所以收益评价必须记为 `NOT_ALLOWED`。不得把最少历史值降到 240、改成最后 252 个非缺失观察、改用总持仓或连续合约，也不得把 0 个候选解读为历史上没有极端压力。若以后研究新的移仓连续口径，必须作为新版本在读取其结果前单独冻结；V1 不得被覆盖。

## 创建分支的数据口径

现有份额文件有 3430 行，但历史份额是事后抓取，缺少原始 `available_at`、修订标记和拆分标记；`feature_asof` 不能替代历史可见时刻。IOPV 快照还缺少一档买卖报价。

旧 readiness 报告记录 2 个 legacy 完整日；新分支的严格检查只有 1 日，因为 2026-08-17 虽有 154 个快照，但最后快照停在 13:37:40，未覆盖收盘。两份记录口径不同，均保留，不把旧计数改写成新策略资格。

## 开盘分支的真实发现性观察

1211 个具备完整开盘分钟窗口的交易日中，冻结规则产生 26 个盘前候选、8 个成熟确认事件。8 个事件的平均毛收益为 -55.69bp，14bp 压力成本后平均净收益为 -69.61bp，单侧 95% 均值下界为 -228.05bp，保守净夏普为 -0.486。

这些数字很弱，但冻结裁决规定少于 40 个成熟事件时保持 `RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS`，因此不能在看到负数后把 40 改成 8，也不能反过来宣布永久拒绝。前向成熟事件仍为 0。

## 最终边界

三条分支都没有 `ACTIVE` 资格。`position_mapping_enabled=false`、`order_generation_enabled=false`、`broker_connection_enabled=false`、`live_trading_enabled=false`；生命周期治理器只能输出现金路由。
