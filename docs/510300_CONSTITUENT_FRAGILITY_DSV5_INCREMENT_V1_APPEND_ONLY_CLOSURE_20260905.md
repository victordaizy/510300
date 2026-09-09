# 成分脆弱性DSV5增量V1追加归档裁决

日期：2026-09-05。依据本轮用户正式裁决；此文件为追加证据，父文件保持原字节。

- `510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1 = REJECTED_FROZEN_CONSTITUENT_FRAGILITY_INCREMENT`。
- 成分脆弱性F/T对DSV5的增量预测家族：`CLOSED_NO_RESCUE`。
- `B1_PRICE_PATH_RISK_MODEL = PREDICTIVE_GATE_PASSED / NOT_YET_A_TRADING_STRATEGY`。
- `CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY = NONE`。
- 不重训B2，不修改B1模型，不将F/T包装为过滤或确认条件。
- 本轮用户另行授权独立的B1政策翻译实验，以及该政策的严格前向Shadow登记；该授权只作用于新模型，不改写父V1的组合禁止状态。
- 新模型：`510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1`；父B2使用为false。
- 原始预测、模型、结果和终态四个SHA-256由新模型manifest逐字节绑定，并与父执行回执核对。
- `POSITION_IMPACT=0`；不连接券商、不产生实际订单、不覆盖实际持仓。
