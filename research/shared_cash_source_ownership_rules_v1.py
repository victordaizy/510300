"""一个现金账户的固定来源优先级；不按结果择来源或改变旧退出。"""
from __future__ import annotations

import numpy as np

from research import support_price_acceptance_inputs_v1 as support

POLICIES = ("CORE_SUPPORT_SOURCE_OWNERSHIP", "CORE_PRICE_SOURCE_OWNERSHIP", "CORE_SUPPORT_FIXED_SOURCE_OWNERSHIP")
BASELINE_CORE = "ORIGINAL_CORE_ONLY_ADAPTER"
BASELINE_SUPPORT = "ORIGINAL_SUPPORT_ONLY_ADAPTER"
NAMES = {POLICIES[0]:"CORE优先与支持补充固定归属",POLICIES[1]:"同共同账户仅价格补充",
         POLICIES[2]:"同共同支持进入与固定退出",BASELINE_CORE:"关闭补充原A适配",BASELINE_SUPPORT:"关闭CORE原支持适配"}


def complement_prefix(policy):
    if policy not in (*POLICIES,BASELINE_CORE,BASELINE_SUPPORT):
        raise ValueError("未知共同现金用途。")
    return "price" if policy == POLICIES[1] else "support"


def complement_exit_policy(policy):
    return support.POLICIES[1] if policy == POLICIES[1] else (support.POLICIES[2] if policy == POLICIES[2] else support.POLICIES[0])


def event_role(shares, target, stopped, locked, has_event, policy):
    """接受当日一次消费，原源未知不补充；不提前看明日成交。"""
    if not has_event:
        return "NO_CURRENT_ACCEPTANCE_EVENT"
    if shares > 0:
        return "CONSUMED_BY_EXISTING_OWNER_NO_SAME_OPEN_REBUY"
    if stopped or locked:
        return "CONSUMED_ACCOUNT_STOP_OR_LOCKED_EXIT"
    if policy == BASELINE_CORE:
        return "COMPLEMENT_DISABLED_ORIGINAL_CORE_ADAPTER"
    if not np.isfinite(target):
        return "CONSUMED_UNKNOWN_CORE_NOT_ZERO"
    if target > 0:
        return "CONSUMED_CORE_POSITIVE_PRIORITY"
    if target < 0:
        raise ValueError("原目标不能为负。")
    return "CURRENT_ZERO_CORE_COMPLEMENT_ATTEMPT"
