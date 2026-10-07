# 510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_ENGINE_V1 — 冻结章程

## 当前裁决

- `CURRENT_FORECAST_SPECIFICATION=REJECTED_FROZEN`
- `WAITING_FOR_EXISTING_LABELS=PASSIVE_MONITORING_ONLY`
- `NEXT_RESEARCH_ACTION=REBUILD_MEASUREMENT_ARCHITECTURE`
- `PORTFOLIO_ACTION=ABSTAIN`
- `POSITION_STATE=POSITION_UNSET`
- 特定 V1 预测实现永久冻结拒绝；结构经济想法仍是未证明，而不是已证明数学不可能。
- V1 不能因后续标签变化恢复资格；任何重启都必须注册为 V2。

## 研究权限

本引擎只做测量与机制辨识。`RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`，`MODEL_POSITION_TARGET=UNSET`。

## 固定执行顺序

1. 完成功效与可识别性审计，量化 1%、2%、5% MSE 改善所需的独立周期、月度 origin 与年份。
2. 建立 60D/120D 总回报成分账本；先证明会计恒等式，再讨论任何模块解释力。
3. CF 仅面向盈利、经营现金流及其 breadth；DR 仅面向估值倍数与 ERP gap；RC 仅面向波动、下行半方差、相关性和流动性冲击。
4. 价格响应模块只测量消息吸收路径，不预测总回报。
5. 只有各自对象的 PIT 测量有效性通过后，才允许另行冻结新一代预测协议。

## 总回报成分账本

账本必须同时给出实际指数链式口径和 origin 固定成分股/权重口径。目标恒等式为：

`TOTAL_RETURN_FACTOR = DIVIDEND_FACTOR × EARNINGS_FACTOR × MULTIPLE_FACTOR × MEMBERSHIP_WEIGHT_FACTOR × TRACKING_FACTOR`

每一项都必须有来源、可用时钟、覆盖率、状态和残差。没有合格的成分股分红档案时，纯 `DIVIDEND_COMPONENT` 必须为 `NO_VIEW`，不得用公司行动差额冒充。

## PIT 硬门槛

- 历史成分与权重版本必须在 origin 当时可用。
- 财务事实使用首次公开披露时间；后修订值不得冒充 PIT。
- 事件与价格反应必须冻结盘前、盘中、盘后时钟。
- 同日/相邻事件必须聚类，避免把同一信息冲击重复计数。
- 任一关键时钟不可验证即输出 `NO_VIEW`。

## 2026-09 起前向记录

只允许追加不可变诊断记录；原记录不得原位修改。标签成熟只更新诊断账本，不恢复 V1、不生成仓位、不进入 Paper/Shadow 或实盘。

协议清单哈希：`c730f54105545936bb7e8a4a19354b395c71dc078aa81347e705c40ee8dd4b3d`。
