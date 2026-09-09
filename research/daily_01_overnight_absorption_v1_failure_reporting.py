"""DAILY_01分组支持异常的只计数失败报告。"""

from __future__ import annotations

import pandas as pd


def build_group_support_failure_report(
    labeled: pd.DataFrame,
    *,
    favorable_threshold: float,
    unfavorable_threshold: float,
    minimum_group_observations: int,
    minimum_mature_enter_events: int,
) -> dict:
    """只记录支持度，不计算任何收益差、回归或资产曲线。"""

    candidate = labeled["entry_sign_eligible"] & labeled["absorption_percentile"].notna()
    mature = labeled["target_d20_net_excess"].notna()
    favorable = candidate & mature & labeled["absorption_percentile"].ge(favorable_threshold)
    unfavorable = candidate & mature & labeled["absorption_percentile"].le(unfavorable_threshold)
    mature_enter = labeled["signal_action"].eq("ENTER") & mature
    favorable_count = int(favorable.sum())
    unfavorable_count = int(unfavorable.sum())
    mature_enter_count = int(mature_enter.sum())
    group_support_passed = (
        favorable_count >= minimum_group_observations
        and unfavorable_count >= minimum_group_observations
    )
    if group_support_passed:
        raise RuntimeError("两组支持度均已通过，不允许使用失败报告替代冻结预测检验")
    not_evaluated = "NOT_EVALUATED_AFTER_GROUP_SUPPORT_FAILURE"
    return {
        "status": "REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST",
        "failure_category": "PRE_REGISTERED_GROUP_SUPPORT_MISSING_OR_INSUFFICIENT",
        "failed_stage": "BEFORE_BLOCK_BOOTSTRAP_HAC_OR_STRATEGY_BACKTEST",
        "sample": {
            "total_rows": int(len(labeled)),
            "valid_candidate_rows": int(candidate.sum()),
            "mature_candidate_rows": int((candidate & mature).sum()),
            "favorable_group_observations": favorable_count,
            "unfavorable_group_observations": unfavorable_count,
            "mature_enter_events": mature_enter_count,
        },
        "gate_checks": {
            "minimum_mature_enter_events": mature_enter_count >= minimum_mature_enter_events,
            "minimum_group_observations": False,
            "both_chronological_halves_positive": not_evaluated,
            "d20_top_minus_bottom_positive": not_evaluated,
            "d20_block_bootstrap_lower_bound_positive": not_evaluated,
            "incremental_hac_coefficient_positive": not_evaluated,
            "d5_direction_not_reversed": not_evaluated,
        },
        "return_differences_computed": False,
        "block_bootstrap_computed": False,
        "hac_regression_computed": False,
        "strategy_backtest_computed": False,
        "alpha_pass": False,
        "strategy_backtest_authorized": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
