# 510300 信息传播 Alpha V2

## 研究对象与边界

唯一收益目标和唯一潜在交易标的是 `510300.SH`。`000300.SH`、IF 实际合约和历史点时 Top50 成分股只提供领先信息或对照信息，不是交易标的，也不改变“研究 510300”的目标。

主问题冻结为：

`IFLead_3m + Top50BreadthImpulse_5m → 510300未来5分钟收益`

本阶段只验证条件期望差异，不生成实盘订单，不宣称文献中的价格发现关系必然覆盖交易成本。

## 冻结主检验

- `IFLeadETF_3m = IF过去3分钟收益 - 1.0 × 510300过去3分钟收益`。
- `Top50Breadth` 是历史点时前50大权重股中，过去3分钟收益为正的等权比例；不得称为“沪深300全市场Breadth”。
- `Top50WeightedBreadth` 是相同股票集合的权重比例，单独输出，不与等权口径混名。
- `Top50BreadthImpulse_5m = Top50Breadth(t) - Top50Breadth(t-5分钟)`。
- `PrimaryAlphaScore = z(IFLeadETF_3m) + z(Top50BreadthImpulse_5m)`。
- z-score 按同一分钟位置、只用此前20个交易日计算，至少需要10个历史观测。
- 开发期只拟合一次 P1—P10 边界；验证期和受污染回顾期不重估。
- 主结果固定为 510300 未来5分钟；1、3、10、15分钟一律标为次要探索结果，不用于重新选参数。
- 主指标为 P1—P10 单调性、Spearman IC、P10-P1以及分年度方向一致性。

唯一权威参数文件是 `config/information_propagation_alpha.yaml`。

## 000300 的角色

`000300.SH` 仅回答“IF 的变化是不是已经被现货指数吸收”。V2 同时计算：

- `IF → 000300` 与 `IF → 510300`；
- `Top50 → 000300` 与 `Top50 → 510300`；
- `IFUniqueResidual = IFReturn - β1 × IndexReturn - β2 × ETFReturn`。

残差回归的系数仅用当前交易日前20个交易日估计，至少10个历史交易日。所有最终条件收益和执行收益仍来自 510300。

## 时间戳硬门槛

- ETF、指数、IF和成分股必须随机抽取固定种子的10个共同交易日进行审计。
- 审计记录每日第一根、上午最后一根、下午第一根、最后一根及柱数。
- ETF时间线必须被指数和当日选定IF实际合约完整覆盖；每只被审计成分股的时间戳集合必须与ETF一致。
- ETF/成分股预期标签为 `09:30—11:30` 与 `13:01—15:00`，共241根；不允许用行位移掩盖午休、隔夜或缺柱。
- 即使标签集合一致，也必须人工确认各供应商标签表示同一闭合区间。人工状态不是 `CONFIRMED` 时，正式扫描保持阻塞。
- 信号使用 t 分钟完整 OHLCV，因此最早只能在 t+1 分钟成交，不允许以 t 的收盘价成交。

时间戳审计报告位于 `reports/data_quality/information_propagation_timestamp_alignment.json`，只有状态为 `PASS` 才能运行正式扫描。

## 统计收益与可执行收益

- 统计价格发现：`close(t+h) / close(t) - 1`。
- 下一分钟开盘执行：`close(t+1+h) / open(t+1) - 1`。
- 下一分钟VWAP执行：`close(t+1+h) / VWAP(t+1) - 1`。
- 基础成本：2万元账户，单边佣金取3bp与最低5元折算值之大者，另计单边5bp滑点和往返2.5bp价差；当前合计18.5bp。
- 压力成本固定为往返25bp。

执行结果只是 510300 的下一分钟 markout，不冒充已经实现 T+1 库存状态机的完整策略收益。

## 数据合同与当前状态

| 数据 | 必要字段 | 当前状态 |
|---|---|---|
| 510300 1分钟 | `trade_time, open, close, vol, amount` | 已有291,851行，2021-08-12至2026-08-12 |
| 000300 1分钟 | `trade_time, close` | 已有291,851行、1,211日；由代理通用`stk_mins`返回并完成close交叉审计 |
| IF实际合约1分钟 | `trade_time, contract, close, volume` | 正式全量缺失；当前Token无 `ft_mins` 权限 |
| 历史Top50并集成分股1分钟 | `trade_time, con_code, close, volume, amount` | 已完成117只、33,853,993行 |
| 沪深300历史权重 | `trade_date, con_code, weight` | 已有36,000行月度快照 |

新浪 IF 文件共63,426行、305个离散交易日，只覆盖各代码尾段，永久标为 `VALIDATION_ONLY_NOT_FORMAL_RESEARCH`，不得代替正式五年 IF 数据。

TuShare 的指数分钟历史接口为 `idx_mins`，期货分钟历史接口为 `ft_mins`；两者都是独立权限。当前临时Token的通用`stk_mins`可返回`000300.SH`，但没有官方`idx_mins`权限，已在元数据中保留这一来源差异。`ft_mins`仍无权限，不能用不完整验证集冒充正式期货输入。

## 运行顺序

```powershell
.\.venv\Scripts\python.exe scripts\audit_information_propagation_timestamps.py
.\.venv\Scripts\python.exe scripts\run_information_propagation_scan.py
```

当前两条命令都会以非零状态结束，并明确列出 `futures` 这一阻塞输入。正式IF数据到位后，先完成时间戳人工语义确认，再运行冻结主检验。
