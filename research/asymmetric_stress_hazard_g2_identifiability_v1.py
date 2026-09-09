"""510300 非对称压力风险 V1 的无标签 G2 时间可识别性预检。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from research.project_evidence_contract_v1 import EvidenceContractError


PASS_STATUS = (
    "PASS_G2_TEMPORAL_IDENTIFIABILITY_PREFLIGHT_"
    "FEATURE_CONSTRUCTION_MAY_BE_FROZEN"
)
FAIL_STATUS = "NO_VIEW_G2_TEMPORAL_IDENTIFIABILITY_FAILED_STOP_NO_RESCUE"


@dataclass(frozen=True)
class FixedSubperiod:
    """预先固定且不可结果后重切的历史子期。"""

    period_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("G2 可识别性配置必须是对象")
    return payload


def _as_date(value: Any, *, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise EvidenceContractError(f"{field} 含不可解析日期：{value}")
    if isinstance(parsed, pd.DatetimeIndex):
        raise EvidenceContractError(f"{field} 必须是单一日期")
    return pd.Timestamp(parsed).tz_localize(None).normalize()


def parse_fixed_subperiods(config: Mapping[str, Any]) -> list[FixedSubperiod]:
    raw = config.get("fixed_subperiods")
    if not isinstance(raw, list) or len(raw) != 4:
        raise EvidenceContractError("G2 必须恰好冻结四个历史子期")
    periods: list[FixedSubperiod] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise EvidenceContractError(f"第 {index} 个子期不是对象")
        period_id = str(item.get("id", "")).strip()
        start = _as_date(item.get("start"), field=f"{period_id}.start")
        end = _as_date(item.get("end"), field=f"{period_id}.end")
        if not period_id or start > end:
            raise EvidenceContractError(f"子期定义无效：{item}")
        periods.append(FixedSubperiod(period_id, start, end))
    for previous, current in zip(periods, periods[1:]):
        if current.start <= previous.end:
            raise EvidenceContractError("固定子期必须按时间递增且不能重叠")
    return periods


def validate_protocol(config: Mapping[str, Any]) -> None:
    program = config.get("program")
    if not isinstance(program, Mapping):
        raise EvidenceContractError("配置缺少 program")
    expected_program = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_TEMPORAL_IDENTIFIABILITY_PREFLIGHT_V1",
        "version": "1.0.0",
        "protocol_revision": "PRE_FEATURE_PRE_LABEL_G2_TEMPORAL_FREEZE",
        "research_state": "DISCOVERY_ONLY",
        "position_impact": 0,
        "feature_values_may_be_constructed_in_this_stage": False,
        "bad10_label_artifacts_may_be_read_in_this_stage": False,
        "model_training_allowed": False,
        "return_evaluation_allowed": False,
        "portfolio_evaluation_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": program.get(key)}
        for key, expected in expected_program.items()
        if program.get(key) != expected
    }
    if mismatches:
        raise EvidenceContractError(f"G2 预检程序边界漂移：{mismatches}")
    if program.get("executable_assets") != ["510300.SH", "CASH_CNY"]:
        raise EvidenceContractError("候选执行资产边界发生漂移")
    if str(program.get("observation_start")) != "2015-01-05":
        raise EvidenceContractError("观察起点发生漂移")
    if str(program.get("observation_cutoff")) != "2026-08-14":
        raise EvidenceContractError("观察截止日发生漂移")

    periods = parse_fixed_subperiods(config)
    expected_periods = [
        ("P1_2015_2017", "2015-01-05", "2017-12-29"),
        ("P2_2018_2020", "2018-01-02", "2020-12-31"),
        ("P3_2021_2023", "2021-01-04", "2023-12-29"),
        ("P4_2024_CUTOFF", "2024-01-02", "2026-08-14"),
    ]
    actual_periods = [
        (item.period_id, item.start.date().isoformat(), item.end.date().isoformat())
        for item in periods
    ]
    if actual_periods != expected_periods:
        raise EvidenceContractError("四个冻结子期发生漂移")

    preflight = config.get("temporal_identifiability_preflight")
    if not isinstance(preflight, Mapping):
        raise EvidenceContractError("配置缺少 temporal_identifiability_preflight")
    expected_preflight = {
        "internal_history_market_sessions_inclusive": 25,
        "macro_lead_forward_market_sessions": 5,
        "necessary_union_market_sessions_inclusive": 30,
        "minimum_potential_daily_origins_per_subperiod": 20,
        "required_evaluable_subperiods": 3,
        "total_subperiods": 4,
        "failure_status": FAIL_STATUS,
        "pass_status": PASS_STATUS,
        "failure_is_terminal_for_v1": True,
        "repartition_after_result_allowed": False,
        "current_industry_backfill_allowed": False,
        "alternative_constituent_panel_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": preflight.get(key)}
        for key, expected in expected_preflight.items()
        if preflight.get(key) != expected
    }
    if mismatches:
        raise EvidenceContractError(f"G2 时间覆盖门发生漂移：{mismatches}")
    if (
        int(preflight["internal_history_market_sessions_inclusive"])
        + int(preflight["macro_lead_forward_market_sessions"])
        != int(preflight["necessary_union_market_sessions_inclusive"])
    ):
        raise EvidenceContractError("必要来源联合窗口不是固定的 25+5=30 日")

    feature = config.get("frozen_feature_construction")
    if not isinstance(feature, Mapping):
        raise EvidenceContractError("配置缺少冻结特征口径")
    if feature.get("missing_value_rule") != "NO_VIEW_NO_INTERPOLATION":
        raise EvidenceContractError("缺失值规则发生漂移")
    if feature.get("all_three_channels_required_for_each_composite") is not True:
        raise EvidenceContractError("M/F/T 必须各自三通道全部有效")
    percentile = feature.get("causal_percentile")
    if not isinstance(percentile, Mapping):
        raise EvidenceContractError("配置缺少因果分位数规则")
    if percentile.get("formula") != (
        "EMPIRICAL_MIDRANK_AGAINST_STRICTLY_PRIOR_VALID_OBSERVATIONS_ONLY"
    ):
        raise EvidenceContractError("因果分位数公式发生漂移")
    if percentile.get("current_observation_excluded") is not True:
        raise EvidenceContractError("当前观察不得进入自身分位数参照集")
    if percentile.get("minimum_prior_valid_observations") != 1:
        raise EvidenceContractError("因果分位数最小历史发生漂移")

    mechanism = config.get("g2_mechanism_if_temporally_identifiable")
    if not isinstance(mechanism, Mapping):
        raise EvidenceContractError("配置缺少预冻结 G2 机制检验")
    macro = mechanism.get("macro_lead_test")
    internal = mechanism.get("internal_bad10_test")
    if not isinstance(macro, Mapping) or not isinstance(internal, Mapping):
        raise EvidenceContractError("G2 两个机制检验定义不完整")
    if macro.get("subperiods_required_positive") != 3:
        raise EvidenceContractError("宏观领先方向门必须是四期中至少三期")
    if macro.get("subperiod_selection_or_repartition_allowed") is not False:
        raise EvidenceContractError("禁止结果后重切 G2 子期")
    if internal.get("minimum_scoreable_events") != 30:
        raise EvidenceContractError("G2 独立事件覆盖门发生漂移")
    if internal.get("minimum_scoreable_non_event_blocks") != 120:
        raise EvidenceContractError("G2 非事件块覆盖门发生漂移")
    if mechanism.get("model_training_in_g2") is not False:
        raise EvidenceContractError("G2 不得训练概率模型")
    if mechanism.get("portfolio_metrics_in_g2") is not False:
        raise EvidenceContractError("G2 不得读取组合指标")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise EvidenceContractError(f"{name} 缺少列：{missing}")


def prepare_membership_presence(
    frame: pd.DataFrame,
    *,
    expected_index_code: str,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    _require_columns(frame, ["membership_date", "index_code", "symbol"], "点时成员")
    work = frame[["membership_date", "index_code", "symbol"]].copy()
    work["date"] = pd.to_datetime(work.pop("membership_date"), errors="coerce").dt.normalize()
    work["index_code"] = work["index_code"].astype("string").str.strip()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    invalid = (
        work["date"].isna()
        | work["symbol"].isna()
        | work["symbol"].eq("")
        | work["index_code"].ne(expected_index_code)
    )
    if invalid.any():
        raise EvidenceContractError("点时成员存在非法日期、代码或指数身份")
    work = work.loc[work["date"].between(start, cutoff), ["date", "symbol"]]
    if work.duplicated(["date", "symbol"]).any():
        raise EvidenceContractError("点时成员存在 date-symbol 重复")
    if work.empty:
        raise EvidenceContractError("观察区间内没有点时成员")
    return work.sort_values(["date", "symbol"]).reset_index(drop=True)


def prepare_constituent_presence(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["date", "con_code", "total_return_close"], "成分总收益价格")
    work = frame[["date", "con_code", "total_return_close"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work.pop("con_code").astype("string").str.strip().str.upper()
    values = pd.to_numeric(work.pop("total_return_close"), errors="coerce")
    work["constituent_present"] = np.isfinite(values) & values.gt(0)
    if work[["date", "symbol"]].isna().any().any():
        raise EvidenceContractError("成分总收益价格存在非法日期或代码")
    if work.duplicated(["date", "symbol"]).any():
        raise EvidenceContractError("成分总收益价格存在 date-symbol 重复")
    return work[["date", "symbol", "constituent_present"]]


def prepare_industry_presence(
    frame: pd.DataFrame, *, required_mapping_status: str
) -> pd.DataFrame:
    _require_columns(
        frame,
        ["membership_date", "symbol", "industry_l1_code", "mapping_status"],
        "点时行业",
    )
    work = frame[
        ["membership_date", "symbol", "industry_l1_code", "mapping_status"]
    ].copy()
    work["date"] = pd.to_datetime(work.pop("membership_date"), errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    industry = work["industry_l1_code"].astype("string").str.strip()
    status = work["mapping_status"].astype("string").str.strip()
    work["industry_present"] = status.eq(required_mapping_status) & industry.notna() & industry.ne("")
    if work[["date", "symbol"]].isna().any().any():
        raise EvidenceContractError("点时行业存在非法日期或代码")
    if work.duplicated(["date", "symbol"]).any():
        raise EvidenceContractError("点时行业存在 date-symbol 重复")
    return work[["date", "symbol", "industry_present"]]


def build_daily_necessary_coverage(
    membership: pd.DataFrame,
    constituent_presence: pd.DataFrame,
    industry_presence: pd.DataFrame,
    *,
    expected_members_per_session: int,
) -> pd.DataFrame:
    joined = membership.merge(
        constituent_presence,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    ).merge(
        industry_presence,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    joined["constituent_present"] = joined["constituent_present"].fillna(False).astype(bool)
    joined["industry_present"] = joined["industry_present"].fillna(False).astype(bool)
    daily = (
        joined.groupby("date", sort=True)
        .agg(
            pit_member_count=("symbol", "nunique"),
            constituent_present_count=("constituent_present", "sum"),
            industry_present_count=("industry_present", "sum"),
        )
        .reset_index()
    )
    daily["necessary_day_valid"] = (
        daily["pit_member_count"].eq(expected_members_per_session)
        & daily["constituent_present_count"].eq(expected_members_per_session)
        & daily["industry_present_count"].eq(expected_members_per_session)
    )
    return daily


def build_potential_origin_ceiling(
    daily: pd.DataFrame,
    *,
    history_sessions_inclusive: int,
    forward_sessions: int,
) -> pd.DataFrame:
    _require_columns(daily, ["date", "necessary_day_valid"], "日度必要覆盖")
    if history_sessions_inclusive < 1 or forward_sessions < 0:
        raise EvidenceContractError("来源覆盖窗口必须为正")
    work = daily[["date", "necessary_day_valid"]].copy().sort_values("date")
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    if work["date"].isna().any() or work["date"].duplicated().any():
        raise EvidenceContractError("日度必要覆盖日期非法或重复")
    valid = work["necessary_day_valid"].fillna(False).astype(bool).to_numpy(dtype=np.int8)
    dates = work["date"].reset_index(drop=True)
    prefix = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    span = history_sessions_inclusive + forward_sessions
    rows: list[dict[str, Any]] = []
    first_origin_index = history_sessions_inclusive - 1
    last_origin_index = len(work) - forward_sessions - 1
    for origin_index in range(first_origin_index, last_origin_index + 1):
        start_index = origin_index - history_sessions_inclusive + 1
        end_index = origin_index + forward_sessions
        valid_count = int(prefix[end_index + 1] - prefix[start_index])
        rows.append(
            {
                "origin_date": dates.iloc[origin_index],
                "necessary_start_date": dates.iloc[start_index],
                "forward_end_date": dates.iloc[end_index],
                "necessary_valid_session_count": valid_count,
                "necessary_required_session_count": span,
                "potential_origin": valid_count == span,
            }
        )
    columns = [
        "origin_date",
        "necessary_start_date",
        "forward_end_date",
        "necessary_valid_session_count",
        "necessary_required_session_count",
        "potential_origin",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_subperiod_coverage(
    origins: pd.DataFrame,
    periods: Sequence[FixedSubperiod],
    *,
    minimum_potential_origins: int,
) -> list[dict[str, Any]]:
    _require_columns(
        origins,
        ["origin_date", "forward_end_date", "potential_origin"],
        "候选原点上界",
    )
    summaries: list[dict[str, Any]] = []
    for period in periods:
        in_period = origins["origin_date"].between(period.start, period.end) & origins[
            "forward_end_date"
        ].between(period.start, period.end)
        candidates = origins.loc[in_period]
        potential = candidates.loc[candidates["potential_origin"].astype(bool)]
        count = int(len(potential))
        summaries.append(
            {
                "subperiod_id": period.period_id,
                "start": period.start.date().isoformat(),
                "end": period.end.date().isoformat(),
                "calendar_candidate_origin_count": int(len(candidates)),
                "potential_origin_upper_bound_count": count,
                "minimum_required": int(minimum_potential_origins),
                "evaluable_upper_bound": count >= minimum_potential_origins,
                "first_potential_origin": (
                    potential["origin_date"].min().date().isoformat()
                    if count
                    else None
                ),
                "last_potential_origin": (
                    potential["origin_date"].max().date().isoformat()
                    if count
                    else None
                ),
            }
        )
    return summaries


def adjudicate_temporal_identifiability(
    subperiod_summaries: Sequence[Mapping[str, Any]],
    *,
    required_evaluable_subperiods: int,
    total_subperiods: int,
) -> dict[str, Any]:
    if len(subperiod_summaries) != total_subperiods:
        raise EvidenceContractError("子期覆盖摘要数量与冻结总数不一致")
    evaluable = [
        str(item["subperiod_id"])
        for item in subperiod_summaries
        if item.get("evaluable_upper_bound") is True
    ]
    passed = len(evaluable) >= required_evaluable_subperiods
    return {
        "gate_id": "G2_TEMPORAL_IDENTIFIABILITY_PREFLIGHT",
        "passed": passed,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "required_evaluable_subperiods": required_evaluable_subperiods,
        "total_subperiods": total_subperiods,
        "evaluable_subperiod_count_upper_bound": len(evaluable),
        "evaluable_subperiod_ids_upper_bound": evaluable,
        "feature_construction_allowed": passed,
        "g2_mechanism_evaluation_allowed": passed,
        "g3_allowed": False,
        "terminal_for_v1": not passed,
        "rescue_allowed": False,
        "repartition_allowed": False,
    }


def build_preflight_result(
    *,
    subperiod_summaries: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    daily: pd.DataFrame,
    origins: pd.DataFrame,
) -> dict[str, Any]:
    potential = origins.loc[origins["potential_origin"].astype(bool)]
    necessary = daily.loc[daily["necessary_day_valid"].astype(bool)]
    return {
        "coverage_upper_bound": {
            "market_session_count": int(len(daily)),
            "necessary_day_valid_count": int(len(necessary)),
            "first_necessary_day_valid": (
                necessary["date"].min().date().isoformat()
                if not necessary.empty
                else None
            ),
            "last_necessary_day_valid": (
                necessary["date"].max().date().isoformat()
                if not necessary.empty
                else None
            ),
            "potential_origin_upper_bound_count": int(len(potential)),
            "first_potential_origin": (
                potential["origin_date"].min().date().isoformat()
                if not potential.empty
                else None
            ),
            "last_potential_origin": (
                potential["origin_date"].max().date().isoformat()
                if not potential.empty
                else None
            ),
            "subperiods": [dict(item) for item in subperiod_summaries],
            "is_optimistic_upper_bound": True,
        },
        "gate": dict(gate),
        "research_disposition": "CONTINUE_TO_FROZEN_G2" if gate["passed"] else "NO_VIEW",
        "feature_values_constructed": False,
        "bad10_label_artifacts_read": False,
        "return_values_read": False,
        "return_evaluation": "NOT_ALLOWED",
        "g2_mechanism_evaluated": False,
        "g3_allowed": False,
        "model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "paper_shadow_generated": False,
        "order_generated": False,
        "broker_action_performed": False,
        "live_trading_authorized": False,
        "position_impact": 0,
        "rescue_allowed": False,
        "next_allowed_action": (
            "FREEZE_G2_FEATURE_CONSTRUCTION_AND_MECHANISM_RUN"
            if gate["passed"]
            else "NONE_V1_STOPPED_NO_REPARTITION_OR_SOURCE_BACKFILL"
        ),
    }
