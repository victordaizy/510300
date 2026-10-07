# 510300 盘后过时收盘价捕获 V2：G0 状态

- G0：`BLOCKED_G0_NO_FORWARD_SIGNAL_COLLECTION`
- 阻塞项：`SGX_A50_SESSION_SPEC_CURRENT, BROKER_SUPPORTS_CLOSING_PRICE_ORDER_FOR_510300, BROKER_ACCEPTS_BEFORE_1505, BROKER_ACK_AND_PARTIAL_FILL_TIMESTAMPS_AVAILABLE, ACCOUNT_CASH_SWEEP_CONFLICT_ABSENT_OR_DISABLED, A50_PRIMARY_FEED_REALTIME_ENTITLEMENT_PROVEN, A50_EXCHANGE_TIMESTAMP_AVAILABLE, LOCAL_CLOCK_SYNC_PASS`
- 权威前向起点：`None`
- 观察/NO_VIEW/成熟目标：0/0/0
- 模型训练：`False`
- 当前目标仓位：`UNSET`
- 仓位影响：`0`
- 实盘授权：`False`

## 顺序检查

| 检查 | 状态 | 证据摘要 |
|---|---|---|
| `SSE_AFTER_HOURS_RULE_CURRENT` | `PASS` | 上交所2026年修订交易规则页面标记现行有效，2026-07-06生效；第3.7节未列入暂缓实施清单。 |
| `SGX_A50_SESSION_SPEC_CURRENT` | `FAIL` | SGX官方产品页为动态内容且本次访问受限，未取得可内容寻址的现行A50合约时段规格；不得用相似产品或旧规格替代。 |
| `BROKER_SUPPORTS_CLOSING_PRICE_ORDER_FOR_510300` | `NOT_PROVIDED` | 未收到券商账户对510300收盘定价委托的功能回执；未连接券商。 |
| `BROKER_ACCEPTS_BEFORE_1505` | `NOT_PROVIDED` | 未收到券商在15:05前接受该订单类型的带时间戳回执。 |
| `BROKER_ACK_AND_PARTIAL_FILL_TIMESTAMPS_AVAILABLE` | `NOT_PROVIDED` | 未收到订单确认、交易所确认、部分成交与成交时间戳导出样本。 |
| `ACCOUNT_CASH_SWEEP_CONFLICT_ABSENT_OR_DISABLED` | `NOT_PROVIDED` | 未收到15:00后自动现金管理或可用资金占用状态的账户级证据。 |
| `A50_PRIMARY_FEED_REALTIME_ENTITLEMENT_PROVEN` | `NOT_PROVIDED` | 未收到免费或账户内含A50主行情的实时权利证明；未知延迟网页不予准入。 |
| `A50_EXCHANGE_TIMESTAMP_AVAILABLE` | `NOT_PROVIDED` | 未收到具体A50合约的bid/ask或逐笔成交样本及exchange_timestamp字段证明。 |
| `LOCAL_CLOCK_SYNC_PASS` | `FAIL` | Windows时间源为Free-running System Clock，上次成功同步未提供；对time.windows.com请求5个样本均超时，无法证明绝对偏差不超过100ms。 |
| `CODE_AND_MANIFEST_FROZEN_COMMIT_PRESENT` | `PASS` | 冻结协议文件和manifest均已进入独立本地Git提交，提交内容与当前冻结哈希逐文件一致。 |

G0 未全部通过时，禁止权威信号采集、模型估计、收益评价、仓位映射、券商连接和订单生成。
