"""V2 G1 事件预测准入：B1、共同样本、事件权重与可识别性。

本模块不访问文件，也不拟合模型。真实 BAD10 标签只能由冻结后的独立入口传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.stress_transmission_hazard_v2 import (
    MODEL_FEATURES,
    NO_VIEW,
    UNASSIGNED_SPLIT,
    VIEW_ALLOWED,
    build_b1_price_risk_features,
    validate_event_weighting,
    validate_split_integrity,
)


PROGRAM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2"
EXECUTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_V1"

B1_FEATURE_COLUMNS = list(MODEL_FEATURES["B1"])
SAMPLE_READ_COLUMNS = [
    "origin_date",
    "entry_date",
    "horizon_end_date",
    "bad10",
    "event_id",
    "sample_group_id",
    "bootstrap_event_block_id",
    "calendar_year_block_id",
    "training_weight",
    "split_assignment",
]
EVENT_READ_COLUMNS = ["event_id"]

B1_OUTPUT_COLUMNS = [
    "date",
    "unadjusted_close",
    "previous_unadjusted_close",
    "cash_dividend_per_share_on_ex_date",
    "daily_total_return",
    "realized_vol20",
    "drawdown20",
    "downside_return5",
    *B1_FEATURE_COLUMNS,
    "B1_state",
]

SAMPLE_ELIGIBILITY_COLUMNS = [
    *SAMPLE_READ_COLUMNS,
    *B1_FEATURE_COLUMNS,
    "B1_state",
    "F",
    "M",
    "T",
    "T_x_F",
    "T_x_M",
    "B2_feature_state",
    "B3_feature_state",
    "b1_eligible",
    "b2_vs_b1_eligible",
    "b3_vs_b2_eligible",
    "b2_vs_b1_model_weight",
    "b3_vs_b2_model_weight",
    "b2_vs_b1_state",
    "b2_vs_b1_no_view_reason",
    "b3_vs_b2_state",
    "b3_vs_b2_no_view_reason",
]

EVENT_ELIGIBILITY_COLUMNS = [
    "event_id",
    "total_positive_origin_count",
    "b2_eligible_positive_origin_count",
    "b2_eligible_positive_origin_ratio",
    "b2_identifiable_event",
    "b2_positive_model_weight_sum",
    "b3_eligible_positive_origin_count",
    "b3_eligible_positive_origin_ratio",
    "b3_identifiable_event",
    "b3_positive_model_weight_sum",
]


class G1EventPredictionAdmissionError(RuntimeError):
    """G1 输入、共同样本或事件权重违反冻结契约。"""


@dataclass(frozen=True)
class G1AdmissionArtifacts:
    """G1 准入构建的三个审计表和门禁指标。"""

    b1_feature_panel: pd.DataFrame
    sample_eligibility_ledger: pd.DataFrame
    event_eligibility_ledger: pd.DataFrame
    metrics: dict[str, Any]
    gate_result: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise G1EventPredictionAdmissionError(f"{label}缺少必需列：{missing}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.tz_localize(None).dt.normalize()
    if parsed.isna().any():
        raise G1EventPredictionAdmissionError(f"{label}存在不可解析日期")
    return parsed


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(columns)].head(5).to_dict("records")
        raise G1EventPredictionAdmissionError(f"{label}存在重复键：{examples}")


def _require_finite(values: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise G1EventPredictionAdmissionError(f"{label}必须全部为有限数")


def _assert_industry_free(columns: Sequence[str]) -> None:
    forbidden = ("industry", "sector", "shenwan", "申万", "行业")
    offending = [
        str(column)
        for column in columns
        if any(token in str(column).casefold() for token in forbidden)
    ]
    if offending:
        raise G1EventPredictionAdmissionError(f"G1 核心出现行业字段：{offending}")


def build_etf_total_return_series(
    *,
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    symbol: str,
    observation_start: str,
    observation_cutoff: str,
) -> pd.DataFrame:
    """以未复权收盘价和除息日现金权益构造当日已实现总股东收益。"""

    _require_columns(etf_daily, ["date", "symbol", "close"], "510300 日行情")
    daily = etf_daily[["date", "symbol", "close"]].copy()
    daily["date"] = _normalize_dates(daily["date"], "510300 日行情日期")
    start = pd.Timestamp(observation_start)
    cutoff = pd.Timestamp(observation_cutoff)
    daily = daily.loc[daily["date"].between(start, cutoff, inclusive="both")].copy()
    if daily.empty:
        raise G1EventPredictionAdmissionError("冻结观察区间内没有 510300 日行情")
    observed_symbols = set(daily["symbol"].astype(str))
    if observed_symbols != {symbol}:
        raise G1EventPredictionAdmissionError(
            f"510300 日行情证券身份不一致：{sorted(observed_symbols)}"
        )
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    _require_finite(daily["close"], "510300 未复权收盘价")
    if daily["close"].le(0.0).any():
        raise G1EventPredictionAdmissionError("510300 未复权收盘价必须严格为正")
    _require_unique(daily, ["date"], "510300 日行情")
    daily = daily.sort_values("date", kind="stable").reset_index(drop=True)

    _require_columns(
        dividends,
        [
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
        ],
        "510300 现金分红",
    )
    cash = dividends[
        [
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
        ]
    ].copy()
    if cash.empty:
        raise G1EventPredictionAdmissionError("510300 现金分红账本为空")
    cash_symbols = set(cash["symbol"].astype(str))
    if cash_symbols != {symbol}:
        raise G1EventPredictionAdmissionError(
            f"510300 分红证券身份不一致：{sorted(cash_symbols)}"
        )
    for column in ("record_date", "ex_date", "payment_date"):
        cash[column] = _normalize_dates(cash[column], f"510300 分红 {column}")
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    _require_finite(cash["cash_dividend_per_share"], "510300 每份现金分红")
    if cash["cash_dividend_per_share"].le(0.0).any():
        raise G1EventPredictionAdmissionError("510300 每份现金分红必须严格为正")
    if cash["record_date"].gt(cash["ex_date"]).any():
        raise G1EventPredictionAdmissionError("现金分红登记日晚于除息日")
    if cash["ex_date"].gt(cash["payment_date"]).any():
        raise G1EventPredictionAdmissionError("现金分红除息日晚于支付日")
    _require_unique(
        cash,
        ["symbol", "record_date", "ex_date", "cash_dividend_per_share"],
        "510300 现金分红",
    )
    cash_by_ex_date = cash.groupby("ex_date")["cash_dividend_per_share"].sum()
    mapped_cash = daily["date"].map(cash_by_ex_date)
    daily["cash_dividend_per_share_on_ex_date"] = mapped_cash.where(
        mapped_cash.notna(), 0.0
    )
    daily["previous_unadjusted_close"] = daily["close"].shift(1)
    daily["daily_total_return"] = (
        (daily["close"] + daily["cash_dividend_per_share_on_ex_date"])
        / daily["previous_unadjusted_close"]
        - 1.0
    )
    if not daily["daily_total_return"].iloc[:1].isna().all():
        raise G1EventPredictionAdmissionError("首个交易日必须因缺少前收盘而无收益")
    later = daily["daily_total_return"].iloc[1:]
    _require_finite(later, "首日之后的 510300 日总收益")
    if later.le(-1.0).any():
        raise G1EventPredictionAdmissionError("510300 日总收益不得小于或等于 -100%")
    return daily.rename(columns={"close": "unadjusted_close"})[
        [
            "date",
            "unadjusted_close",
            "previous_unadjusted_close",
            "cash_dividend_per_share_on_ex_date",
            "daily_total_return",
        ]
    ]


def build_b1_feature_panel(total_returns: pd.DataFrame) -> pd.DataFrame:
    """构造 B1，首个无前收盘日不传给累计财富函数。"""

    _require_columns(
        total_returns,
        [
            "date",
            "unadjusted_close",
            "previous_unadjusted_close",
            "cash_dividend_per_share_on_ex_date",
            "daily_total_return",
        ],
        "510300 总收益",
    )
    frame = total_returns.copy()
    frame["date"] = _normalize_dates(frame["date"], "510300 总收益日期")
    _require_unique(frame, ["date"], "510300 总收益")
    valid = frame.loc[frame["daily_total_return"].notna(), ["date", "daily_total_return"]]
    if len(valid) < 21:
        raise G1EventPredictionAdmissionError("510300 有效总收益不足以构造 B1")
    built = build_b1_price_risk_features(valid)
    selected = built[
        [
            "date",
            "realized_vol20",
            "drawdown20",
            "downside_return5",
            *B1_FEATURE_COLUMNS,
            "B1_state",
        ]
    ]
    result = frame.merge(selected, on="date", how="left", validate="one_to_one")
    result["B1_state"] = result["B1_state"].where(
        result["B1_state"].notna(), NO_VIEW
    )
    view = result["B1_state"].eq(VIEW_ALLOWED)
    if result.loc[view, B1_FEATURE_COLUMNS].isna().any().any():
        raise G1EventPredictionAdmissionError("B1 可见日存在缺失特征")
    for column in B1_FEATURE_COLUMNS:
        if not np.isfinite(result.loc[view, column].to_numpy(dtype=float)).all():
            raise G1EventPredictionAdmissionError(f"B1 可见日 {column} 非有限")
    return result[B1_OUTPUT_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def prepare_mft_common_view(mft: pd.DataFrame) -> pd.DataFrame:
    """独立于标签构造 B2/B3 的日级共同样本可见性。"""

    required = ["date", "F", "M", "T", "B2_feature_state", "B3_feature_state"]
    _require_columns(mft, required, "M/F/T 面板")
    _assert_industry_free(mft.columns)
    frame = mft[required].copy()
    frame["date"] = _normalize_dates(frame["date"], "M/F/T 日期")
    _require_unique(frame, ["date"], "M/F/T 面板")
    for column in ("F", "M", "T"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["T_x_F"] = frame["T"] * frame["F"]
    frame["T_x_M"] = frame["T"] * frame["M"]
    for state_column in ("B2_feature_state", "B3_feature_state"):
        invalid_states = set(frame[state_column].dropna().astype(str)).difference(
            {VIEW_ALLOWED, NO_VIEW}
        )
        if invalid_states:
            raise G1EventPredictionAdmissionError(
                f"{state_column} 存在非法状态：{sorted(invalid_states)}"
            )
    b2_view = frame["B2_feature_state"].eq(VIEW_ALLOWED)
    b3_view = frame["B3_feature_state"].eq(VIEW_ALLOWED)
    if frame.loc[b2_view, ["F", "T", "T_x_F"]].isna().any().any():
        raise G1EventPredictionAdmissionError("B2 可见日存在缺失特征")
    if frame.loc[b3_view, ["F", "M", "T", "T_x_F", "T_x_M"]].isna().any().any():
        raise G1EventPredictionAdmissionError("B3 可见日存在缺失特征")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def prepare_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    """只读取并验证事件 ID，不读取首次突破日期或路径字段。"""

    _require_columns(events, EVENT_READ_COLUMNS, "BAD10 事件身份表")
    result = events[EVENT_READ_COLUMNS].copy()
    result["event_id"] = result["event_id"].astype("string")
    if result.empty or result["event_id"].isna().any() or result["event_id"].eq("").any():
        raise G1EventPredictionAdmissionError("BAD10 事件身份为空或缺失")
    _require_unique(result, ["event_id"], "BAD10 事件身份表")
    return result.sort_values("event_id", kind="stable").reset_index(drop=True)


def prepare_bad10_samples(samples: pd.DataFrame, event_ids: pd.DataFrame) -> pd.DataFrame:
    """验证冻结样本账本，不接触实际未来路径幅度。"""

    _require_columns(samples, SAMPLE_READ_COLUMNS, "BAD10 样本账本")
    frame = samples[SAMPLE_READ_COLUMNS].copy()
    for column in ("origin_date", "entry_date", "horizon_end_date"):
        frame[column] = _normalize_dates(frame[column], f"BAD10 {column}")
    _require_unique(frame, ["origin_date"], "BAD10 样本账本")
    frame = frame.sort_values("origin_date", kind="stable").reset_index(drop=True)
    if not frame["origin_date"].lt(frame["entry_date"]).all():
        raise G1EventPredictionAdmissionError("BAD10 执行日必须晚于信息原点")
    if not frame["entry_date"].le(frame["horizon_end_date"]).all():
        raise G1EventPredictionAdmissionError("BAD10 期限日期非法")
    numeric_labels = pd.to_numeric(frame["bad10"], errors="coerce")
    if numeric_labels.isna().any() or not set(numeric_labels.unique()).issubset({0, 1}):
        raise G1EventPredictionAdmissionError("BAD10 标签必须为完整 0/1")
    frame["bad10"] = numeric_labels.astype("int64")
    frame["training_weight"] = pd.to_numeric(frame["training_weight"], errors="coerce")
    _require_finite(frame["training_weight"], "BAD10 原始训练权重")
    if frame["training_weight"].le(0.0).any():
        raise G1EventPredictionAdmissionError("BAD10 原始训练权重必须为正")
    for column in (
        "event_id",
        "sample_group_id",
        "bootstrap_event_block_id",
        "calendar_year_block_id",
        "split_assignment",
    ):
        frame[column] = frame[column].astype("string")
    if not frame["split_assignment"].eq(UNASSIGNED_SPLIT).all():
        raise G1EventPredictionAdmissionError("G1 前样本切分必须全部保持未分配")
    positive = frame["bad10"].eq(1)
    negative = ~positive
    if frame.loc[positive, "event_id"].isna().any():
        raise G1EventPredictionAdmissionError("BAD10 正样本缺少事件 ID")
    if frame.loc[negative, "event_id"].notna().any():
        raise G1EventPredictionAdmissionError("BAD10 非事件日不得绑定事件 ID")
    expected_ids = set(event_ids["event_id"].astype(str))
    observed_ids = set(frame.loc[positive, "event_id"].astype(str))
    if observed_ids != expected_ids:
        raise G1EventPredictionAdmissionError("事件身份表与正样本事件集合不一致")
    validate_event_weighting(
        events=pd.DataFrame(
            {
                "event_id": event_ids["event_id"].astype(str),
                "event_origin_start": pd.NaT,
                "event_origin_end": pd.NaT,
                "event_entry_start": pd.NaT,
                "event_entry_end": pd.NaT,
                "event_horizon_end": pd.NaT,
                "first_breach_date": pd.NaT,
                "positive_origin_count": 0,
                "positive_training_weight_sum": 1.0,
                "split_group_id": event_ids["event_id"].astype(str),
                "bootstrap_event_block_id": event_ids["event_id"].astype(str),
                "calendar_year_block_id": "NOT_READ",
            }
        ),
        samples=frame,
    )
    validate_split_integrity(frame)
    return frame


def build_daily_common_view(
    *, b1: pd.DataFrame, mft: pd.DataFrame
) -> pd.DataFrame:
    """仅由特征状态生成共同样本标记，函数不接收标签。"""

    _require_columns(b1, ["date", *B1_FEATURE_COLUMNS, "B1_state"], "B1 面板")
    mft_view = prepare_mft_common_view(mft)
    b1_view = b1[["date", *B1_FEATURE_COLUMNS, "B1_state"]].copy()
    b1_view["date"] = _normalize_dates(b1_view["date"], "B1 日期")
    _require_unique(b1_view, ["date"], "B1 面板")
    daily = b1_view.merge(mft_view, on="date", how="outer", validate="one_to_one")
    daily = daily.sort_values("date", kind="stable").reset_index(drop=True)
    b1_finite = daily[B1_FEATURE_COLUMNS].notna().all(axis=1)
    daily["b1_eligible"] = daily["B1_state"].eq(VIEW_ALLOWED) & b1_finite
    b2_finite = daily[["T", "T_x_F"]].notna().all(axis=1)
    daily["b2_vs_b1_eligible"] = (
        daily["b1_eligible"]
        & daily["B2_feature_state"].eq(VIEW_ALLOWED)
        & b2_finite
    )
    b3_finite = daily[["M", "T_x_M"]].notna().all(axis=1)
    daily["b3_vs_b2_eligible"] = (
        daily["b2_vs_b1_eligible"]
        & daily["B3_feature_state"].eq(VIEW_ALLOWED)
        & b3_finite
    )
    if (daily["b3_vs_b2_eligible"] & ~daily["b2_vs_b1_eligible"]).any():
        raise G1EventPredictionAdmissionError("B3 共同样本不是 B2 共同样本子集")
    return daily


def _renormalized_model_weight(
    samples: pd.DataFrame, eligibility_column: str
) -> pd.Series:
    eligible = samples[eligibility_column].astype(bool)
    positive = eligible & samples["bad10"].eq(1)
    negative = eligible & samples["bad10"].eq(0)
    weight = pd.Series(np.nan, index=samples.index, dtype="float64")
    counts = samples.loc[positive].groupby("event_id")["origin_date"].transform("count")
    weight.loc[positive] = 1.0 / counts.to_numpy(dtype=float)
    weight.loc[negative] = 1.0
    sums = samples.loc[positive].assign(_weight=weight.loc[positive]).groupby("event_id")[
        "_weight"
    ].sum()
    if not sums.empty and not np.allclose(
        sums.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0.0
    ):
        raise G1EventPredictionAdmissionError(
            f"{eligibility_column} 可见正样本事件权重之和不为 1"
        )
    if not weight.loc[negative].eq(1.0).all():
        raise G1EventPredictionAdmissionError(
            f"{eligibility_column} 可见非事件风险日权重不为 1"
        )
    if weight.loc[~eligible].notna().any():
        raise G1EventPredictionAdmissionError(
            f"{eligibility_column} 的 NO_VIEW 行出现模型权重"
        )
    return weight


def _reason_columns(samples: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    b2_reasons: list[str] = []
    b3_reasons: list[str] = []
    for row in result.itertuples(index=False):
        reasons2: list[str] = []
        if not bool(row.b1_eligible):
            reasons2.append("B1_NO_VIEW")
        if str(row.B2_feature_state) != VIEW_ALLOWED:
            reasons2.append("B2_FEATURE_NO_VIEW")
        if pd.isna(row.T) or pd.isna(row.T_x_F):
            reasons2.append("B2_REQUIRED_VALUE_MISSING")
        b2_reasons.append("VIEW_ALLOWED" if not reasons2 else ";".join(reasons2))

        reasons3 = list(reasons2)
        if str(row.B3_feature_state) != VIEW_ALLOWED:
            reasons3.append("B3_FEATURE_NO_VIEW")
        if pd.isna(row.M) or pd.isna(row.T_x_M):
            reasons3.append("B3_REQUIRED_VALUE_MISSING")
        b3_reasons.append("VIEW_ALLOWED" if not reasons3 else ";".join(reasons3))
    result["b2_vs_b1_state"] = np.where(
        result["b2_vs_b1_eligible"], VIEW_ALLOWED, NO_VIEW
    )
    result["b2_vs_b1_no_view_reason"] = b2_reasons
    result["b3_vs_b2_state"] = np.where(
        result["b3_vs_b2_eligible"], VIEW_ALLOWED, NO_VIEW
    )
    result["b3_vs_b2_no_view_reason"] = b3_reasons
    return result


def _iso_date_or_none(values: pd.Series, *, first: bool) -> str | None:
    valid = pd.to_datetime(values, errors="coerce").dropna()
    if valid.empty:
        return None
    chosen = valid.min() if first else valid.max()
    return pd.Timestamp(chosen).date().isoformat()


def _gate_result(metrics: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = contract["mechanism_discovery"]
    full = contract["full_three_coefficient_model"]
    mechanism_pass = (
        int(metrics["b2_identifiable_event_count"])
        >= int(mechanism["minimum_independent_events"])
        and int(metrics["b2_eligible_non_event_risk_day_count"])
        >= int(mechanism["minimum_non_event_risk_days"])
    )
    full_pass = (
        int(metrics["b3_identifiable_event_count"])
        >= int(full["minimum_independent_events"])
        and int(metrics["b3_eligible_non_event_risk_day_count"])
        >= int(full["minimum_non_event_risk_days"])
    )
    if not mechanism_pass:
        status = str(contract["insufficient_state"])
        next_step = str(contract["if_mechanism_fails_next_step"])
    elif not full_pass:
        status = "PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE"
        next_step = str(contract["if_mechanism_passes_next_step"])
    else:
        status = "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
        next_step = str(contract["if_mechanism_passes_next_step"])
    return {
        "G1_DATA_AND_EVENTS": status,
        "mechanism_discovery_prerequisite_passed": mechanism_pass,
        "full_three_coefficient_model_prerequisite_passed": full_pass,
        "next_allowed_step": next_step,
        "additional_step_if_full_model_passed": (
            str(contract["if_full_model_passes_additional_step"])
            if full_pass
            else None
        ),
    }


def build_g1_admission_artifacts(
    *,
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    mft: pd.DataFrame,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    config: Mapping[str, Any],
) -> G1AdmissionArtifacts:
    """构造 G1 准入账本；最后一步才使用标签做事件计数和权重。"""

    program = config["program"]
    total_returns = build_etf_total_return_series(
        etf_daily=etf_daily,
        dividends=dividends,
        symbol=str(config["inputs"]["etf_unadjusted_daily"]["symbol"]),
        observation_start=str(program["observation_start"]),
        observation_cutoff=str(program["observation_cutoff"]),
    )
    b1 = build_b1_feature_panel(total_returns)
    daily_common = build_daily_common_view(b1=b1, mft=mft)
    mft_dates = pd.DatetimeIndex(_normalize_dates(mft["date"], "M/F/T 日期"))
    b1_dates = pd.DatetimeIndex(b1["date"])
    if not b1_dates.equals(mft_dates):
        raise G1EventPredictionAdmissionError(
            "冻结区间的 510300 B1 交易日与 M/F/T 交易日不完全一致"
        )

    event_ids = prepare_event_ids(events)
    prepared_samples = prepare_bad10_samples(samples, event_ids)
    daily_columns = [
        "date",
        *B1_FEATURE_COLUMNS,
        "B1_state",
        "F",
        "M",
        "T",
        "T_x_F",
        "T_x_M",
        "B2_feature_state",
        "B3_feature_state",
        "b1_eligible",
        "b2_vs_b1_eligible",
        "b3_vs_b2_eligible",
    ]
    ledger = prepared_samples.merge(
        daily_common[daily_columns],
        left_on="origin_date",
        right_on="date",
        how="left",
        validate="one_to_one",
    ).drop(columns=["date"])
    for column in ("b1_eligible", "b2_vs_b1_eligible", "b3_vs_b2_eligible"):
        ledger[column] = ledger[column].where(ledger[column].notna(), False).astype(bool)
    ledger["b2_vs_b1_model_weight"] = _renormalized_model_weight(
        ledger, "b2_vs_b1_eligible"
    )
    ledger["b3_vs_b2_model_weight"] = _renormalized_model_weight(
        ledger, "b3_vs_b2_eligible"
    )
    ledger = _reason_columns(ledger)

    positive = ledger.loc[ledger["bad10"].eq(1)].copy()
    grouped_total = positive.groupby("event_id").size()
    grouped_b2 = positive.loc[positive["b2_vs_b1_eligible"]].groupby("event_id").size()
    grouped_b3 = positive.loc[positive["b3_vs_b2_eligible"]].groupby("event_id").size()
    b2_weight_sums = positive.groupby("event_id")["b2_vs_b1_model_weight"].sum(
        min_count=1
    )
    b3_weight_sums = positive.groupby("event_id")["b3_vs_b2_model_weight"].sum(
        min_count=1
    )
    event_ledger = event_ids.copy()
    event_ledger["total_positive_origin_count"] = (
        event_ledger["event_id"].map(grouped_total).astype("int64")
    )
    event_ledger["b2_eligible_positive_origin_count"] = (
        event_ledger["event_id"].map(grouped_b2).where(
            event_ledger["event_id"].map(grouped_b2).notna(), 0
        ).astype("int64")
    )
    event_ledger["b2_eligible_positive_origin_ratio"] = (
        event_ledger["b2_eligible_positive_origin_count"]
        / event_ledger["total_positive_origin_count"]
    )
    event_ledger["b2_identifiable_event"] = event_ledger[
        "b2_eligible_positive_origin_count"
    ].gt(0)
    event_ledger["b2_positive_model_weight_sum"] = event_ledger["event_id"].map(
        b2_weight_sums
    )
    event_ledger["b3_eligible_positive_origin_count"] = (
        event_ledger["event_id"].map(grouped_b3).where(
            event_ledger["event_id"].map(grouped_b3).notna(), 0
        ).astype("int64")
    )
    event_ledger["b3_eligible_positive_origin_ratio"] = (
        event_ledger["b3_eligible_positive_origin_count"]
        / event_ledger["total_positive_origin_count"]
    )
    event_ledger["b3_identifiable_event"] = event_ledger[
        "b3_eligible_positive_origin_count"
    ].gt(0)
    event_ledger["b3_positive_model_weight_sum"] = event_ledger["event_id"].map(
        b3_weight_sums
    )
    for prefix in ("b2", "b3"):
        identifiable = event_ledger[f"{prefix}_identifiable_event"]
        sums = event_ledger.loc[identifiable, f"{prefix}_positive_model_weight_sum"]
        if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0.0):
            raise G1EventPredictionAdmissionError(f"{prefix.upper()} 事件权重不为 1")
        if event_ledger.loc[~identifiable, f"{prefix}_positive_model_weight_sum"].notna().any():
            raise G1EventPredictionAdmissionError(
                f"{prefix.upper()} 不可识别事件出现模型权重"
            )

    negative = ledger["bad10"].eq(0)
    b2_view = ledger["b2_vs_b1_eligible"]
    b3_view = ledger["b3_vs_b2_eligible"]
    metrics: dict[str, Any] = {
        "total_sample_count": int(len(ledger)),
        "total_independent_event_count": int(len(event_ledger)),
        "total_positive_origin_count": int((~negative).sum()),
        "total_non_event_risk_day_count": int(negative.sum()),
        "b1_eligible_sample_day_count": int(ledger["b1_eligible"].sum()),
        "b2_common_sample_day_count": int(b2_view.sum()),
        "b2_identifiable_event_count": int(event_ledger["b2_identifiable_event"].sum()),
        "b2_eligible_positive_origin_count": int((b2_view & ~negative).sum()),
        "b2_eligible_non_event_risk_day_count": int((b2_view & negative).sum()),
        "b2_first_eligible_origin_date": _iso_date_or_none(
            ledger.loc[b2_view, "origin_date"], first=True
        ),
        "b2_last_eligible_origin_date": _iso_date_or_none(
            ledger.loc[b2_view, "origin_date"], first=False
        ),
        "b3_common_sample_day_count": int(b3_view.sum()),
        "b3_identifiable_event_count": int(event_ledger["b3_identifiable_event"].sum()),
        "b3_eligible_positive_origin_count": int((b3_view & ~negative).sum()),
        "b3_eligible_non_event_risk_day_count": int((b3_view & negative).sum()),
        "b3_first_eligible_origin_date": _iso_date_or_none(
            ledger.loc[b3_view, "origin_date"], first=True
        ),
        "b3_last_eligible_origin_date": _iso_date_or_none(
            ledger.loc[b3_view, "origin_date"], first=False
        ),
    }
    gate = _gate_result(metrics, config["g1_gate_contract"])
    return G1AdmissionArtifacts(
        b1_feature_panel=b1[B1_OUTPUT_COLUMNS].reset_index(drop=True),
        sample_eligibility_ledger=ledger[SAMPLE_ELIGIBILITY_COLUMNS].reset_index(
            drop=True
        ),
        event_eligibility_ledger=event_ledger[EVENT_ELIGIBILITY_COLUMNS].reset_index(
            drop=True
        ),
        metrics=metrics,
        gate_result=gate,
    )
