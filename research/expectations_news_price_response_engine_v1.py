"""510300 预期—消息—价格响应测量架构的纯变换与校验函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


DEFAULT_NEUTRAL_QUADRANT_MAP: dict[str, str] = {
    "CF_IMPROVING_DR_EASING": "CF_UP_DR_EASING",
    "CF_IMPROVING_DR_TIGHTENING": "CF_UP_DR_TIGHTENING",
    "CF_DETERIORATING_DR_EASING": "CF_DOWN_DR_EASING",
    "CF_DETERIORATING_DR_TIGHTENING": "CF_DOWN_DR_TIGHTENING",
}

FORBIDDEN_SEMANTIC_TOKENS: tuple[str, ...] = (
    "CF_IMPROVING",
    "CF_DETERIORATING",
    "MOMENTUM_FUNDAMENTAL_SUPPORT",
    "REVERSAL_ECONOMIC_BASIS",
    "VALUE_SAFETY_MARGIN",
    "TRUE_CONTRACTION_DECLINE",
    "TEMPORARY_LIQUIDITY_SHOCK_CANDIDATE",
)

NEUTRAL_STATE_COLUMNS: tuple[str, ...] = (
    "origin",
    "origin_completion_status",
    "cf_level",
    "cf_direction",
    "cf_acceleration",
    "cf_breadth",
    "cf_concentration",
    "cash_flow_expectation_factor",
    "cf_news",
    "cf_news_sign",
    "cf_state_status",
    "cf_data_status",
    "cross_sectional_expected_return_factor",
    "equity_risk_premium_state",
    "equity_risk_premium_expanding_z",
    "discount_rate_easing_news",
    "discount_rate_easing_news_sign",
    "valuation_concentration",
    "duration_compression",
    "present_value_data_status",
    "rc_asof_date",
    "rc_asof_age_calendar_days",
    "risk_bearing_capacity_factor",
    "risk_capacity_news",
    "risk_capacity_news_sign",
    "rc_capacity_expanding_median",
    "member_median_return_60d",
    "member_positive_breadth_60d",
    "large_weight_minus_median_member_return_60d",
    "industry_negative_breadth_20d",
    "rc_state_status",
    "cf_dr_quadrant",
    "rc_direction",
    "condition_quality_status",
    "condition_evaluation_eligible",
    "rc_interaction_evaluation_eligible",
)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """校验输入列完整，避免缺列时静默生成不完整测量结果。"""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少必需列：{missing}")


def neutralize_condition_id(
    value: Any,
    quadrant_map: Mapping[str, str] | None = None,
) -> Any:
    """将旧象限词汇机械映射为不带方向假设的状态标签。"""

    if pd.isna(value):
        return value
    text = str(value)
    mapping = dict(quadrant_map or DEFAULT_NEUTRAL_QUADRANT_MAP)
    for source in sorted(mapping, key=len, reverse=True):
        text = text.replace(source, mapping[source])
    return text


def _assert_no_forbidden_semantics(frame: pd.DataFrame, label: str) -> None:
    object_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        values = frame[column].dropna().astype(str)
        for token in FORBIDDEN_SEMANTIC_TOKENS:
            if values.str.contains(token, regex=False).any():
                raise ValueError(f"{label}.{column}仍包含被废止的语义标签：{token}")


def build_neutral_state_panel(
    parent_state: pd.DataFrame,
    quadrant_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """从冻结条件状态构造中性、无收益目标的状态面板。"""

    require_columns(parent_state, NEUTRAL_STATE_COLUMNS, "第二阶段条件状态面板")
    mapping = dict(quadrant_map or DEFAULT_NEUTRAL_QUADRANT_MAP)
    result = parent_state.loc[:, list(NEUTRAL_STATE_COLUMNS)].copy()
    result["origin"] = pd.to_datetime(result["origin"], errors="coerce").dt.normalize()
    result["rc_asof_date"] = pd.to_datetime(
        result["rc_asof_date"], errors="coerce"
    ).dt.normalize()
    if result["origin"].isna().any():
        raise ValueError("中性状态面板存在无法解析的 origin")
    if result["origin"].duplicated().any():
        raise ValueError("中性状态面板 origin 不唯一")

    result["state_quadrant"] = result["cf_dr_quadrant"].map(
        lambda value: neutralize_condition_id(value, mapping)
    )
    result["state_cell"] = result.apply(
        lambda row: (
            f"{row['state_quadrant']}__{row['rc_direction']}"
            if row["state_quadrant"] in set(mapping.values())
            else "NO_VIEW_OR_UNCLASSIFIED_STATE_CELL"
        ),
        axis=1,
    )
    result = result.rename(
        columns={
            "condition_quality_status": "state_quality_status",
            "condition_evaluation_eligible": "state_evaluation_eligible",
            "rc_interaction_evaluation_eligible": "joint_state_evaluation_eligible",
        }
    )
    result = result.drop(columns=["cf_dr_quadrant"])
    result["state_atlas_program_id"] = "510300_CONDITIONAL_STATE_ATLAS_V1"
    result["state_role"] = "DESCRIPTIVE_STATE_ONLY"
    result["return_target_present"] = False
    result["return_prediction_allowed"] = False
    result["portfolio_evaluation_allowed"] = False
    result["model_position_target"] = "UNSET"
    result = result.sort_values("origin", kind="stable").reset_index(drop=True)

    _assert_no_forbidden_semantics(result, "中性状态面板")
    allowed_quadrants = set(mapping.values()) | {"NO_VIEW_REQUIRED_CF_OR_DR_STATE"}
    unexpected = sorted(set(result["state_quadrant"].dropna()).difference(allowed_quadrants))
    if unexpected:
        raise ValueError(f"中性状态面板出现未注册象限：{unexpected}")
    if result["return_target_present"].any():
        raise ValueError("中性状态面板意外包含收益目标")
    return result


def build_neutral_atlas(
    parent_atlas: pd.DataFrame,
    canonical_source_views: Sequence[str],
    quadrant_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """去除 archetype 与方向预设，只保留中性条件的历史描述统计。"""

    required = (
        "sample_scope",
        "view_type",
        "condition_id",
        "horizon_market_days",
        "condition_origin_count",
        "observed_origin_count",
        "censored_origin_count",
        "forward_observation_coverage",
        "parent_state_pass_share",
        "distinct_calendar_years",
        "first_observed_origin",
        "last_observed_origin",
        "mean_total_return",
        "median_total_return",
        "positive_rate",
        "loss_rate",
        "q10_total_return",
        "q25_total_return",
        "q75_total_return",
        "q90_total_return",
        "minimum_total_return",
        "maximum_total_return",
        "reliability_status",
        "overlapping_forward_windows",
    )
    require_columns(parent_atlas, required, "第二阶段条件统计")
    mapping = dict(quadrant_map or DEFAULT_NEUTRAL_QUADRANT_MAP)
    allowed_views = set(map(str, canonical_source_views))
    result = parent_atlas.loc[
        parent_atlas["view_type"].astype(str).isin(allowed_views), list(required)
    ].copy()
    result = result.rename(
        columns={
            "view_type": "state_view",
            "condition_id": "state_id",
            "condition_origin_count": "state_origin_count",
        }
    )
    result["state_id"] = result["state_id"].map(
        lambda value: neutralize_condition_id(value, mapping)
    )
    for column in ("first_observed_origin", "last_observed_origin"):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    result["state_atlas_program_id"] = "510300_CONDITIONAL_STATE_ATLAS_V1"
    result["historical_outcome_role"] = "DESCRIPTIVE_ONLY_NOT_A_FORECAST"
    result["directional_hypothesis_embedded"] = False
    result["mechanism_claim_allowed"] = False
    result["return_prediction_allowed"] = False
    result["portfolio_evaluation_allowed"] = False
    result["model_position_target"] = "UNSET"
    result = result.sort_values(
        ["sample_scope", "state_view", "state_id", "horizon_market_days"],
        kind="stable",
    ).reset_index(drop=True)

    forbidden_fields = {"expected_direction", "directional_result"}
    if forbidden_fields.intersection(result.columns):
        raise ValueError("中性 Atlas 仍包含方向预设字段")
    if not set(result["state_view"]).issubset(allowed_views):
        raise ValueError("中性 Atlas 混入未注册视图")
    _assert_no_forbidden_semantics(result, "中性 Atlas")
    return result


def build_permanent_rejection_receipt(
    prediction_status: Mapping[str, Any],
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    """把已冻结预测追加登记为永久拒绝的诊断记录。"""

    if prediction_status.get("status") != (
        "STAGE_3_COMPLETE_PREDICTIVE_VALIDATION_FAILED_OR_NO_VIEW"
    ):
        raise ValueError("第三阶段不是已冻结的失败或 NO_VIEW 状态")
    if prediction_status.get("validation_summary", {}).get(
        "structural_prediction_validated"
    ) is not False:
        raise ValueError("第三阶段状态意外宣称预测通过")
    forecasts = list(prediction_status.get("latest_forecasts", []))
    if not forecasts:
        raise ValueError("第三阶段状态缺少冻结预测诊断记录")

    immutable_forecasts: list[dict[str, Any]] = []
    expected_origin = str(adjudication["existing_forecast_origin"])
    for forecast in forecasts:
        actual = forecast.get("actual")
        actual_status = forecast.get("actual_status")
        if actual is not None or actual_status != "ACTUAL_CENSORED":
            raise ValueError("冻结预测已有非审查范围内的实际值，禁止改写裁决")
        if str(forecast.get("origin")) != expected_origin:
            raise ValueError("冻结预测 origin 与裁决登记不一致")
        record = dict(forecast)
        record["forecast_record_status"] = adjudication[
            "existing_forecast_status"
        ]
        record["trading_status"] = adjudication[
            "existing_forecast_trading_status"
        ]
        record["v1_requalification_allowed"] = False
        immutable_forecasts.append(record)

    return {
        "program_id": "510300_STRUCTURAL_PREDICTION_V1",
        "adjudication_status": "REJECTED_FROZEN_PERMANENT_NO_RESCUE",
        "current_forecast_specification": adjudication[
            "current_forecast_specification"
        ],
        "structural_economic_idea_status": adjudication[
            "structural_economic_idea_status"
        ],
        "waiting_for_existing_labels": adjudication[
            "waiting_for_existing_labels"
        ],
        "next_research_action": adjudication["next_research_action"],
        "portfolio_action": adjudication["portfolio_action"],
        "position_state": adjudication["position_state"],
        "v1_requalification_allowed": False,
        "future_requalification_requires_new_program": adjudication[
            "future_requalification_requires_new_program"
        ],
        "mathematical_impossibility_proven": adjudication[
            "mathematical_impossibility_proven"
        ],
        "accessible_free_information_feasibility_validated": adjudication[
            "accessible_free_information_feasibility_validated"
        ],
        "current_investable_strategy": adjudication[
            "current_investable_strategy"
        ],
        "source_status_payload_sha256": prediction_status[
            "status_payload_sha256"
        ],
        "immutable_diagnostic_forecasts": immutable_forecasts,
        "immutable_diagnostic_forecast_count": len(immutable_forecasts),
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
