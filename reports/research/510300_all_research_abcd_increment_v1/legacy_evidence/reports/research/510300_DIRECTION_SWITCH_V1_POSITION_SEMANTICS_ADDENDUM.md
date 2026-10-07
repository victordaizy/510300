# 510300 Direction-Switch V1 仓位语义补充说明

状态：`AUTHORITATIVE_GOVERNANCE_CLARIFICATION`

本补充说明不修改 `510300_DIRECTION_SWITCH_V1` 的历史结果、参数、机制结论或任何收益数字，仅收紧拒绝结果的仓位语义。

## 权威语义

```text
RESEARCH_STATUS = DISCOVERY_ONLY
MODEL_POSITION_TARGET = UNSET
MODEL_ACTION = ABSTAIN
NEW_ORDER_ALLOWED = false
EXISTING_HOLDINGS_OVERRIDE_ALLOWED = false
PAPER_SIGNAL_ALLOWED = false
SHADOW_SIGNAL_ALLOWED = false
LIVE_TRADING_AUTHORIZED = false
```

`NO_TRADE` 只表示研究系统不生成新订单，不表示模型建议将已有持仓卖出至现金，也不表示模型建议继续持有或新买入 510300。

- 用户已有 510300 持仓时，本次拒绝结果没有提供卖出或改变仓位的依据。
- 用户没有 510300 持仓时，本次拒绝结果没有提供买入依据。
- 用户真实持仓状态不属于本研究输出，保持未知且不被覆盖。

## IF 机制结论的精确含义

```text
EFFICACY_TESTED = false
REJECTION_REASON = INSUFFICIENT_PRE_REGISTERED_EVENT_PREVALENCE
RESCUE_ALLOWED = false
```

IF 分支因预注册事件数量门失败而失去历史验证资格。该结论不等于已经证明信号无效，但禁止降低门槛、改为连续 IF 分数或以其他参数重新验证同一候选。

## 不变边界

- 原始历史结果与哈希保持不变。
- 组合收益评价仍为 `NOT_ALLOWED_MECHANISM_GATE_FAILED`。
- 不生成 Paper、Shadow、仓位映射、订单、券商连接或实盘动作。
- 本说明不是交易建议，不改变用户现有持仓。
