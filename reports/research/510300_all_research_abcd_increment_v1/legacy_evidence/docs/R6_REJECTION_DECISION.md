# R6历史拒绝决策

状态：`HISTORICALLY_REJECTED`
下一阶段：`R6_POSTMORTEM_COMPLETE`

```text
historical_candidates = 25
historical_passed = 0
historical_best_candidate = B_CONT_W756
historical_best_is_approved = false
r6_forward_1_enabled = false
live_position_mapping_enabled = false
order_generation_enabled = false
new_factor_research_enabled = false
machine_learning_research_enabled = false
next_phase = R6_POSTMORTEM_COMPLETE
```

## 归因结论

- 通过全部失败归因继续研究门槛的候选：0个。
- 验证—历史伪OOS Sharpe秩相关：-0.709。
- 10%档与25%小账户超额秩相关：0.932。
- A_VOL_TARGET：验证期族中位成本后动态择时贡献-3.19%，循环平移中位百分位28.2%。
- B_TREND_VOL：验证期族中位成本后动态择时贡献-0.23%，循环平移中位百分位54.1%。
- C_R5_OVERLAY：验证期族中位成本后动态择时贡献-0.40%，循环平移中位百分位50.8%。

## 正式决定

当前趋势、波动和估值状态族在验证期没有成本后动态择时能力。正式终止R6模型族，不启动R7，不研究相同变量的新阈值变体。

R5仅保留为历史基准和工程回归系统；R6 25候选冻结且不得追加；R6_FORWARD_1继续禁用；V3仅按原协议观察；PCF/IOPV继续采集但不参与仓位；分钟信息和MACD不恢复为Alpha；券商连接与订单生成保持禁用。

本结论只说明当前单ETF择时变量族缺乏验证支持。若未来改变为风险控制目标，必须先通过同平均仓位和同波动静态基准；若仍追求长期正超额，需要另立协议讨论扩大投资范围，不能把它包装成R6参数优化。

## 可选失败族观察档案

`R6_FAILED_FAMILY_OBSERVATION` 从2026-08-17起只追加当日新数据，不回填缺失日期；主统计量固定为三族中位数，满242个新交易日之前禁止评价。它不显示候选排名、不批准候选、不映射仓位、不连接券商也不生成订单。

运行入口：`.venv\Scripts\python.exe scripts\run_r6_failed_family_observation.py`。若当日数据尚不可用，状态文件必须明确报告旧数据日期，且不得把旧结果记成今日观察。
