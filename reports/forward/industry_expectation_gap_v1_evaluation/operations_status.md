# 行业预期差前瞻运行状态

- 总状态：`COLLECTING_FORWARD`
- 运行健康：`PASS`
- 生成时间：`2026-08-22T00:19:58.399292+08:00`
- 独立预测原点：1 个
- 成熟原点：0 个
- 非重叠60日块：0 个
- 下一原点闸门：`WAITING_FOR_NEXT_ORIGIN_EVIDENCE`
- 结果输入闸门：`COLLECTING_OUTCOME_INPUT_STALE`
- 共同结果数据截止：`2026-08-19`
- 最近应覆盖交易日：`2026-08-21`

## 当前阻断

- `CURRENT_MONTH_ALREADY_HAS_ORIGIN`
- `SOURCE_REGISTRY_CUTOFF_NOT_ADVANCED`
- `SECTOR_PANEL_MONTH_NOT_ADVANCED`
- `NEW_OFFICIAL_RELEASES_BELOW_MINIMUM`
- `OUTCOME_INPUT_BEHIND_LATEST_CLOSED_TRADING_DAY`

## 治理边界

- `prediction_date` 才是独立时间样本；同日行业行不重复计数。
- 新原点必须进入新月份，并同时取得更新后的来源注册表、行业点时面板和至少两个新官方发布。
- 新原点仍需人工证据包；程序不会复制旧判断或自动生成行业预测。
- 未成熟期不输出部分收益，原 `NO_VIEW` 不得改写。
- 仓位映射、订单生成、券商连接和实盘均关闭。
