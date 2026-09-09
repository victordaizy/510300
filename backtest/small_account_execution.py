"""2万元510300账户的粗粒度纸面目标生成。"""

from __future__ import annotations

import math
from typing import Any


def quantize_position(raw_position: float, step: float) -> float:
    """将连续仓位就近映射到固定仓位档。"""

    if not 0 < step <= 1:
        raise ValueError("仓位步长必须位于(0, 1]")
    clipped = min(max(float(raw_position), 0.0), 1.0)
    units = math.floor(clipped / step + 0.5)
    return min(units * step, 1.0)


def build_small_account_paper_target(
    *,
    account_equity_cny: float,
    raw_target_position: float,
    price_cny: float,
    position_step: float,
    lot_size: int,
    minimum_normal_trade_shares: int,
    current_shares: int | None = None,
    available_cash_cny: float | None = None,
    sellable_shares: int | None = None,
    today_bought_shares: int | None = None,
    allow_position_increase: bool = False,
    allow_position_decrease: bool = False,
    commission_rate: float = 0.0003,
    minimum_commission_cny: float = 5.0,
) -> dict[str, Any]:
    """生成不授权下单的小账户纸面目标。"""

    if account_equity_cny <= 0 or price_cny <= 0:
        raise ValueError("账户权益和价格必须为正数")
    if lot_size <= 0 or minimum_normal_trade_shares <= 0:
        raise ValueError("整手数和最小普通调仓份额必须为正数")
    if minimum_normal_trade_shares % lot_size != 0:
        raise ValueError("最小普通调仓份额必须是整手的整数倍")
    if current_shares is not None and current_shares < 0:
        raise ValueError("当前份额不能为负数")
    if available_cash_cny is not None and available_cash_cny < 0:
        raise ValueError("可用现金不能为负数")
    for label, shares in (
        ("可卖旧仓", sellable_shares),
        ("当日新仓", today_bought_shares),
    ):
        if shares is not None and (shares < 0 or shares % lot_size != 0):
            raise ValueError(f"{label}必须是非负整手数")
    if commission_rate < 0 or minimum_commission_cny < 0:
        raise ValueError("佣金参数不能为负数")

    account_state_complete = all(
        value is not None
        for value in (current_shares, available_cash_cny, sellable_shares, today_bought_shares)
    )
    if account_state_complete and sellable_shares + today_bought_shares != current_shares:
        raise ValueError("可卖旧仓与当日新仓之和必须等于当前总份额")

    target_position = quantize_position(raw_target_position, position_step)
    effective_equity = (
        float(available_cash_cny) + int(current_shares) * price_cny
        if account_state_complete
        else account_equity_cny
    )
    target_notional = effective_equity * target_position
    target_shares = math.floor(target_notional / price_cny / lot_size) * lot_size
    effective_position = target_shares * price_cny / effective_equity
    result: dict[str, Any] = {
        "raw_target_position": float(raw_target_position),
        "target_position_grid": target_position,
        "target_shares": int(target_shares),
        "target_notional_cny": float(target_shares * price_cny),
        "effective_target_position": float(effective_position),
        "current_shares": current_shares,
        "available_cash_cny": available_cash_cny,
        "sellable_shares": sellable_shares,
        "today_bought_shares": today_bought_shares,
        "account_state_complete": account_state_complete,
        "account_equity_used_cny": float(effective_equity),
        "account_equity_source": "CURRENT_ACCOUNT_STATE" if account_state_complete else "REFERENCE_EQUITY",
        "allow_position_increase": bool(allow_position_increase),
        "allow_position_decrease": bool(allow_position_decrease),
        "paper_action": "TARGET_ONLY_NO_HOLDINGS",
        "paper_trade_shares": None,
        "signed_paper_trade_shares": None,
        "automatic_ordering_authorized": False,
    }
    if current_shares is None:
        return result

    if not account_state_complete:
        result["paper_action"] = "BLOCKED_INCOMPLETE_ACCOUNT_STATE"
        return result

    delta = int(target_shares - current_shares)
    requested_delta = delta
    if delta == 0:
        result["paper_action"] = "HOLD_AT_TARGET"
        result["paper_trade_shares"] = 0
        result["signed_paper_trade_shares"] = 0
        return result
    if delta > 0 and not allow_position_increase:
        result["paper_action"] = "HOLD_INCREASE_NOT_ALLOWED"
        result["paper_trade_shares"] = 0
        result["signed_paper_trade_shares"] = 0
        return result
    if delta < 0 and not allow_position_decrease:
        result["paper_action"] = "HOLD_DECREASE_NOT_ALLOWED"
        result["paper_trade_shares"] = 0
        result["signed_paper_trade_shares"] = 0
        return result

    if delta > 0:
        cash_after_minimum_commission = max(
            float(available_cash_cny) - minimum_commission_cny,
            0.0,
        )
        affordable = math.floor(
            cash_after_minimum_commission
            / (price_cny * (1.0 + commission_rate))
            / lot_size
        ) * lot_size
        delta = min(delta, affordable)
    else:
        delta = -min(-delta, int(sellable_shares))

    result["requested_signed_trade_shares"] = requested_delta
    result["paper_trade_shares"] = abs(delta)
    result["signed_paper_trade_shares"] = delta
    if delta == 0:
        result["paper_action"] = (
            "BLOCKED_INSUFFICIENT_AVAILABLE_CASH"
            if requested_delta > 0
            else "BLOCKED_T_PLUS_ONE_NO_SELLABLE_SHARES"
        )
    elif abs(delta) < minimum_normal_trade_shares:
        result["paper_action"] = "HOLD_ACCUMULATE_UNECONOMIC_DIFFERENCE"
        result["paper_trade_shares"] = 0
        result["signed_paper_trade_shares"] = 0
    elif delta > 0:
        result["paper_action"] = "PAPER_BUY"
    else:
        result["paper_action"] = "PAPER_SELL"
    return result
