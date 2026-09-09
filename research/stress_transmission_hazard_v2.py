"""510300 压力传导危险率 V2 的数据契约、特征骨架与事件账本。

本模块只提供协议冻结后的确定性构件。它不会读取项目数据、选择阈值、
运行组合、生成仓位或触发订单。所有真实数据运行必须由单独的一次性入口
在对应研究门通过后显式调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROGRAM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2"
VIEW_ALLOWED = "VIEW_ALLOWED"
NO_VIEW = "NO_VIEW"
UNASSIGNED_SPLIT = "UNASSIGNED_PRE_MODEL_FREEZE"


class StressTransmissionContractError(RuntimeError):
    """V2 输入、时钟或不变量不满足时封闭失败。"""


class ConstituentReturnState(str, Enum):
    """成分股单日总股东收益的四态契约。"""

    TRADED_VALID = "TRADED_VALID"
    OFFICIAL_SUSPENSION = "OFFICIAL_SUSPENSION"
    CORPORATE_ACTION_UNRESOLVED = "CORPORATE_ACTION_UNRESOLVED"
    SUPPLIER_MISSING_OR_CONFLICT = "SUPPLIER_MISSING_OR_CONFLICT"


ACTION_NONE_CONFIRMED = "NONE_CONFIRMED"
ACTION_RESOLVED = "RESOLVED"
ACTION_UNRESOLVED = "UNRESOLVED"
ALLOWED_ACTION_STATES = {
    ACTION_NONE_CONFIRMED,
    ACTION_RESOLVED,
    ACTION_UNRESOLVED,
}
USABLE_RETURN_STATES = {
    ConstituentReturnState.TRADED_VALID.value,
    ConstituentReturnState.OFFICIAL_SUSPENSION.value,
}

CLASSIFIED_RETURN_COLUMNS = [
    "date",
    "symbol",
    "constituent_return_state",
    "daily_total_shareholder_return",
    "return_is_usable",
    "state_reason",
]

EVENT_LEDGER_COLUMNS = [
    "event_id",
    "event_origin_start",
    "event_origin_end",
    "event_entry_start",
    "event_entry_end",
    "event_horizon_end",
    "first_breach_date",
    "positive_origin_count",
    "positive_training_weight_sum",
    "split_group_id",
    "bootstrap_event_block_id",
    "calendar_year_block_id",
]

SAMPLE_LEDGER_COLUMNS = [
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

MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "B0": (),
    "B1": (
        "b1_realized_vol20_risk_percentile",
        "b1_drawdown20_risk_percentile",
        "b1_downside_return5_risk_percentile",
    ),
    "B2": ("T", "T_x_F"),
    "B3": ("T", "T_x_F", "T_x_M"),
}

FORBIDDEN_PREDICTIVE_NAME_TOKENS = (
    "industry",
    "sector",
    "shenwan",
    "申万",
    "行业",
)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise StressTransmissionContractError(f"{label}缺少必需列：{missing}")


def _normalize_dates(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce").dt.normalize()
    return result.astype("datetime64[ns]")


def _normalize_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).astype(
        "datetime64[ns, UTC]"
    )


def _require_unique_key(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if frame.duplicated(list(columns)).any():
        examples = frame.loc[frame.duplicated(list(columns), keep=False), list(columns)]
        raise StressTransmissionContractError(
            f"{label}存在重复键：{examples.head(5).to_dict(orient='records')}"
        )


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    allowed = values.isin([True, False, 1, 0])
    if values.isna().any() or not bool(allowed.all()):
        raise StressTransmissionContractError(f"{label}必须逐行显式为布尔值")
    return values.astype(bool)


def _nonempty_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().ne("")


def assert_predictive_names_are_industry_free(names: Sequence[str]) -> None:
    """禁止任何行业分类字段进入 V2 核心预测输入。"""

    violations: list[str] = []
    for name in names:
        normalized = str(name).casefold()
        if any(token.casefold() in normalized for token in FORBIDDEN_PREDICTIVE_NAME_TOKENS):
            violations.append(str(name))
    if violations:
        raise StressTransmissionContractError(
            f"V2 核心预测字段含被禁止的行业分类语义：{sorted(violations)}"
        )


def causal_midrank(values: pd.Series, *, minimum_prior_observations: int = 1) -> pd.Series:
    """当前值相对严格更早有效观察的经验中秩，绝不使用当前或未来值。"""

    if minimum_prior_observations < 1:
        raise StressTransmissionContractError("因果分位数的最少历史观察必须至少为 1")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, value in numeric.items():
        if not np.isfinite(value):
            continue
        if len(history) >= minimum_prior_observations:
            prior = np.asarray(history, dtype=float)
            result.loc[index] = float(
                (
                    np.count_nonzero(prior < value)
                    + 0.5 * np.count_nonzero(prior == value)
                )
                / len(prior)
            )
        history.append(float(value))
    return result


def required_median(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """仅当所有冻结通道都有效时计算中位数，否则保持缺失。"""

    _require_columns(frame, columns, "复合风险分数")
    valid = frame[list(columns)].notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid] = frame.loc[valid, list(columns)].median(axis=1)
    return result


def classify_constituent_returns(rows: pd.DataFrame) -> pd.DataFrame:
    """按四态契约分类成分股收益，只有有证据的停牌日允许记 0。

    输入中的公司行动经济量全部以行动前每股为单位：

    ``期末财富 = 未复权收盘价 * 行动后/行动前股数比
                  + 现金分配 - 认购现金流出``。

    ``NONE_CONFIRMED`` 必须显式证明当日无公司行动，且三个经济量分别为
    0、1、0；``RESOLVED`` 必须有证据并提供完整经济量；任何未解决行动都
    保持缺失。证券首次出现但没有前收盘价也保持供应商缺失态。
    """

    required = [
        "date",
        "symbol",
        "previous_unadjusted_close",
        "unadjusted_close",
        "source_observed",
        "supplier_conflict",
        "official_suspension",
        "suspension_evidence_id",
        "corporate_action_status",
        "corporate_action_evidence_id",
        "cash_distribution_per_pre_event_share",
        "post_to_pre_share_ratio",
        "subscription_cash_outflow_per_pre_event_share",
    ]
    _require_columns(rows, required, "成分收益四态输入")
    collisions = sorted(set(CLASSIFIED_RETURN_COLUMNS[2:]).intersection(rows.columns))
    if collisions:
        raise StressTransmissionContractError(
            f"输入不得预置分类输出列：{collisions}"
        )

    frame = rows.copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    if frame[["date", "symbol"]].isna().any().any() or frame["symbol"].eq("").any():
        raise StressTransmissionContractError("成分收益四态输入的日期或证券代码非法")
    _require_unique_key(frame, ["date", "symbol"], "成分收益四态输入")

    frame["source_observed"] = _strict_bool(
        frame["source_observed"], "source_observed"
    )
    frame["supplier_conflict"] = _strict_bool(
        frame["supplier_conflict"], "supplier_conflict"
    )
    frame["official_suspension"] = _strict_bool(
        frame["official_suspension"], "official_suspension"
    )
    action_status = (
        frame["corporate_action_status"].astype("string").str.strip().str.upper()
    )
    invalid_actions = sorted(set(action_status.dropna()).difference(ALLOWED_ACTION_STATES))
    if invalid_actions or action_status.isna().any():
        raise StressTransmissionContractError(
            f"公司行动状态必须属于固定枚举：{invalid_actions}"
        )
    frame["corporate_action_status"] = action_status

    numeric_columns = [
        "previous_unadjusted_close",
        "unadjusted_close",
        "cash_distribution_per_pre_event_share",
        "post_to_pre_share_ratio",
        "subscription_cash_outflow_per_pre_event_share",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    state = pd.Series(
        ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value,
        index=frame.index,
        dtype="string",
    )
    reason = pd.Series("SOURCE_MISSING_OR_UNUSABLE", index=frame.index, dtype="string")
    total_return = pd.Series(np.nan, index=frame.index, dtype=float)

    has_action_evidence = _nonempty_text(frame["corporate_action_evidence_id"])
    has_suspension_evidence = _nonempty_text(frame["suspension_evidence_id"])
    action_unresolved = frame["corporate_action_status"].eq(ACTION_UNRESOLVED)

    action_numbers_finite = pd.Series(
        np.isfinite(
            frame[
                [
                    "cash_distribution_per_pre_event_share",
                    "post_to_pre_share_ratio",
                    "subscription_cash_outflow_per_pre_event_share",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1),
        index=frame.index,
    )
    action_numbers_valid = (
        action_numbers_finite
        & frame["cash_distribution_per_pre_event_share"].ge(0.0)
        & frame["post_to_pre_share_ratio"].gt(0.0)
        & frame["subscription_cash_outflow_per_pre_event_share"].ge(0.0)
    )
    none_action_terms_exact = (
        frame["cash_distribution_per_pre_event_share"].eq(0.0)
        & frame["post_to_pre_share_ratio"].eq(1.0)
        & frame["subscription_cash_outflow_per_pre_event_share"].eq(0.0)
    )
    action_checked = (
        frame["corporate_action_status"].isin(
            [ACTION_NONE_CONFIRMED, ACTION_RESOLVED]
        )
        & has_action_evidence
        & action_numbers_valid
        & (
            frame["corporate_action_status"].eq(ACTION_RESOLVED)
            | none_action_terms_exact
        )
    )
    action_problem = action_unresolved | (~action_checked)
    action_problem &= ~frame["supplier_conflict"]
    state.loc[action_problem] = ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value
    reason.loc[action_problem] = "CORPORATE_ACTION_NOT_FULLY_RECONCILED"

    previous_valid = (
        frame["previous_unadjusted_close"].notna()
        & np.isfinite(frame["previous_unadjusted_close"])
        & frame["previous_unadjusted_close"].gt(0.0)
    )
    suspension_valid = (
        frame["official_suspension"]
        & has_suspension_evidence
        & previous_valid
        & action_checked
        & (~frame["supplier_conflict"])
    )
    state.loc[suspension_valid] = ConstituentReturnState.OFFICIAL_SUSPENSION.value
    reason.loc[suspension_valid] = "OFFICIAL_SUSPENSION_MARKED_TO_LAST_VALID_PRICE"
    total_return.loc[suspension_valid] = 0.0

    close_valid = (
        frame["unadjusted_close"].notna()
        & np.isfinite(frame["unadjusted_close"])
        & frame["unadjusted_close"].gt(0.0)
    )
    traded_candidate = (
        frame["source_observed"]
        & (~frame["official_suspension"])
        & (~frame["supplier_conflict"])
        & action_checked
        & previous_valid
        & close_valid
    )
    ending_wealth = (
        frame["unadjusted_close"] * frame["post_to_pre_share_ratio"]
        + frame["cash_distribution_per_pre_event_share"]
        - frame["subscription_cash_outflow_per_pre_event_share"]
    )
    computed_return = ending_wealth / frame["previous_unadjusted_close"] - 1.0
    computed_valid = (
        np.isfinite(computed_return)
        & ending_wealth.gt(0.0)
        & computed_return.gt(-1.0)
    )
    traded_valid = traded_candidate & computed_valid
    state.loc[traded_valid] = ConstituentReturnState.TRADED_VALID.value
    reason.loc[traded_valid] = "UNADJUSTED_PRICE_PLUS_RESOLVED_ACTION_LEDGER"
    total_return.loc[traded_valid] = computed_return.loc[traded_valid]

    invalid_economic_terms = traded_candidate & (~computed_valid)
    state.loc[invalid_economic_terms] = (
        ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value
    )
    reason.loc[invalid_economic_terms] = "RESOLVED_ACTION_TERMS_PRODUCE_INVALID_WEALTH"

    conflict = frame["supplier_conflict"]
    state.loc[conflict] = ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value
    reason.loc[conflict] = "SUPPLIER_CONFLICT_FAIL_CLOSED"
    total_return.loc[conflict] = np.nan

    return_is_usable = state.isin(USABLE_RETURN_STATES) & total_return.notna()
    if total_return.loc[~return_is_usable].notna().any():
        raise StressTransmissionContractError("非可用收益状态意外保留了数值")
    suspension_returns = total_return.loc[
        state.eq(ConstituentReturnState.OFFICIAL_SUSPENSION.value)
    ]
    if not suspension_returns.eq(0.0).all():
        raise StressTransmissionContractError("官方停牌态必须且只能产生 0 收益")

    frame["constituent_return_state"] = state
    frame["daily_total_shareholder_return"] = total_return
    frame["return_is_usable"] = return_is_usable
    frame["state_reason"] = reason
    return frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def prepare_membership(
    membership: pd.DataFrame,
    *,
    expected_index_code: str = "000300",
    expected_members_per_day: int | None = 300,
) -> pd.DataFrame:
    """准入逐日点时成员；不得用当前成员回填历史。"""

    _require_columns(membership, ["date", "symbol"], "点时成员")
    columns = ["date", "symbol"]
    if "weight" in membership.columns:
        columns.append("weight")
    if "index_code" in membership.columns:
        columns.append("index_code")
    frame = membership[columns].copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    if frame[["date", "symbol"]].isna().any().any() or frame["symbol"].eq("").any():
        raise StressTransmissionContractError("点时成员日期或证券代码非法")
    _require_unique_key(frame, ["date", "symbol"], "点时成员")
    if "index_code" in frame.columns:
        index_code = frame["index_code"].astype("string").str.strip()
        if not index_code.eq(expected_index_code).all():
            raise StressTransmissionContractError("点时成员混入非冻结指数")
    if expected_members_per_day is not None:
        counts = frame.groupby("date")["symbol"].nunique()
        if not counts.eq(expected_members_per_day).all():
            raise StressTransmissionContractError(
                "点时成员数量漂移："
                f"expected={expected_members_per_day}, min={int(counts.min())}, "
                f"max={int(counts.max())}"
            )
    if "weight" in frame.columns:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        finite = np.isfinite(frame["weight"].to_numpy(dtype=float))
        if not finite.all() or frame["weight"].lt(0.0).any():
            raise StressTransmissionContractError("点时权重必须为有限非负值")
        if frame.groupby("date")["weight"].sum().le(0.0).any():
            raise StressTransmissionContractError("点时权重每日合计必须为正")
    return frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def build_daily_coverage_ledger(
    *,
    membership: pd.DataFrame,
    classified_returns: pd.DataFrame,
    minimum_member_coverage: float = 0.98,
    minimum_weight_coverage: float = 0.99,
    reliable_point_in_time_weights: bool = False,
    expected_members_per_day: int | None = 300,
) -> pd.DataFrame:
    """构造日度覆盖账本；普通缺失可排除，但不得填 0。"""

    if not 0.0 < minimum_member_coverage <= 1.0:
        raise StressTransmissionContractError("成员覆盖门必须位于 (0, 1]")
    if not 0.0 < minimum_weight_coverage <= 1.0:
        raise StressTransmissionContractError("权重覆盖门必须位于 (0, 1]")
    members = prepare_membership(
        membership,
        expected_members_per_day=expected_members_per_day,
    )
    _require_columns(
        classified_returns,
        [
            "date",
            "symbol",
            "constituent_return_state",
            "daily_total_shareholder_return",
            "return_is_usable",
        ],
        "四态收益",
    )
    returns = classified_returns[
        [
            "date",
            "symbol",
            "constituent_return_state",
            "daily_total_shareholder_return",
            "return_is_usable",
        ]
    ].copy()
    returns["date"] = _normalize_dates(returns["date"])
    returns["symbol"] = returns["symbol"].astype("string").str.strip().str.upper()
    _require_unique_key(returns, ["date", "symbol"], "四态收益")

    joined = members.merge(
        returns,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    numeric_return = pd.to_numeric(
        joined["daily_total_shareholder_return"], errors="coerce"
    )
    usable = (
        joined["return_is_usable"].eq(True)
        & joined["constituent_return_state"].isin(USABLE_RETURN_STATES)
        & numeric_return.notna()
        & np.isfinite(numeric_return)
    )
    joined["usable"] = usable
    grouped = joined.groupby("date", sort=True)
    ledger = grouped.agg(
        point_in_time_member_count=("symbol", "nunique"),
        usable_member_count=("usable", "sum"),
    )
    ledger["usable_member_ratio"] = (
        ledger["usable_member_count"] / ledger["point_in_time_member_count"]
    )
    ledger["member_coverage_passed"] = ledger["usable_member_ratio"].ge(
        minimum_member_coverage
    )

    if reliable_point_in_time_weights:
        if "weight" not in joined.columns:
            raise StressTransmissionContractError(
                "声明点时权重可靠时，成员表必须提供 weight"
            )
        joined["usable_weight"] = joined["weight"].where(joined["usable"])
        total_weight = grouped["weight"].sum()
        usable_weight = joined.groupby("date")["usable_weight"].sum(min_count=1)
        ledger["usable_weight_ratio"] = usable_weight / total_weight
        ledger["weight_coverage_passed"] = ledger["usable_weight_ratio"].ge(
            minimum_weight_coverage
        )
        ledger["weight_coverage_state"] = "EVALUATED_RELIABLE_PIT_WEIGHT"
    else:
        ledger["usable_weight_ratio"] = np.nan
        ledger["weight_coverage_passed"] = True
        ledger["weight_coverage_state"] = "NOT_APPLICABLE_NO_RELIABLE_PIT_WEIGHT"

    ledger["aggregation_state"] = np.where(
        ledger["member_coverage_passed"] & ledger["weight_coverage_passed"],
        VIEW_ALLOWED,
        NO_VIEW,
    )
    ledger["missing_member_count"] = (
        ledger["point_in_time_member_count"] - ledger["usable_member_count"]
    )
    return ledger.reset_index()


def _rolling_compound(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise StressTransmissionContractError("复合收益窗口必须为正整数")
    invalid = returns.le(-1.0) & returns.notna()
    if invalid.any().any():
        raise StressTransmissionContractError("日总股东收益不得小于或等于 -100%")
    log_return = np.log1p(returns)
    count = returns.notna().rolling(window, min_periods=window).sum()
    compounded = np.expm1(log_return.rolling(window, min_periods=window).sum())
    return compounded.where(count.eq(window))


def build_internal_raw_features(
    *,
    membership: pd.DataFrame,
    classified_returns: pd.DataFrame,
    h00300_total_return_close: pd.DataFrame,
    member_coverage_minimum: float = 0.98,
    comovement_member_ratio_minimum: float = 0.90,
    comovement_minimum_valid_observations: int = 15,
    lookback_days: int = 20,
    change_days: int = 5,
    tail_volatility_days: int = 60,
    tail_sigma_multiple: float = 1.5,
    expected_members_per_day: int | None = 300,
) -> pd.DataFrame:
    """构造完全无行业依赖的 F/T 原始通道及覆盖状态。"""

    assert_predictive_names_are_industry_free(
        [
            "BREADTH20",
            "LEADERSHIP_GAP20",
            "COMOVEMENT20",
            "BREADTH_DROP5",
            "TAIL_DIFFUSION5",
            "COMOVEMENT_ACCEL5",
        ]
    )
    if lookback_days != 20 or change_days != 5 or tail_volatility_days != 60:
        raise StressTransmissionContractError("V2 核心窗口已冻结为 20/5/60")
    if tail_sigma_multiple != 1.5:
        raise StressTransmissionContractError("V2 左尾阈值倍数已冻结为 1.5")
    if not 0.0 < comovement_member_ratio_minimum <= 1.0:
        raise StressTransmissionContractError("共同运动成员比例门非法")
    if not 1 <= comovement_minimum_valid_observations <= lookback_days:
        raise StressTransmissionContractError("共同运动有效观察门非法")

    members = prepare_membership(
        membership,
        expected_members_per_day=expected_members_per_day,
    )
    _require_columns(
        classified_returns,
        [
            "date",
            "symbol",
            "constituent_return_state",
            "daily_total_shareholder_return",
            "return_is_usable",
        ],
        "四态收益",
    )
    returns = classified_returns[
        [
            "date",
            "symbol",
            "constituent_return_state",
            "daily_total_shareholder_return",
            "return_is_usable",
        ]
    ].copy()
    returns["date"] = _normalize_dates(returns["date"])
    returns["symbol"] = returns["symbol"].astype("string").str.strip().str.upper()
    _require_unique_key(returns, ["date", "symbol"], "四态收益")
    numeric = pd.to_numeric(returns["daily_total_shareholder_return"], errors="coerce")
    returns["usable_return"] = numeric.where(
        returns["return_is_usable"].eq(True)
        & returns["constituent_return_state"].isin(USABLE_RETURN_STATES)
        & numeric.notna()
        & np.isfinite(numeric)
    )

    dates = pd.DatetimeIndex(sorted(members["date"].unique()))
    symbols = pd.Index(sorted(set(members["symbol"].astype(str))))
    wide = returns.pivot(index="date", columns="symbol", values="usable_return")
    wide = wide.reindex(index=dates, columns=symbols)
    membership_mask = (
        members.assign(is_member=True)
        .pivot(index="date", columns="symbol", values="is_member")
        .reindex(index=dates, columns=symbols)
        .eq(True)
    )
    member_count = membership_mask.sum(axis=1)
    if member_count.le(0).any():
        raise StressTransmissionContractError("点时成员存在空交易日")

    return20 = _rolling_compound(wide, lookback_days)
    return5 = _rolling_compound(wide, change_days)
    volatility60 = wide.rolling(
        tail_volatility_days,
        min_periods=tail_volatility_days,
    ).std(ddof=1)

    current_return20 = return20.where(membership_mask)
    valid20_count = current_return20.notna().sum(axis=1)
    return20_coverage = valid20_count / member_count
    return20_gate = return20_coverage.ge(member_coverage_minimum)
    breadth20 = current_return20.gt(0.0).sum(axis=1) / valid20_count.replace(0, np.nan)
    breadth20 = breadth20.where(return20_gate)
    equal_weight_return20 = current_return20.mean(axis=1, skipna=True).where(
        return20_gate
    )

    _require_columns(h00300_total_return_close, ["date", "close"], "H00300 总收益")
    benchmark = h00300_total_return_close[["date", "close"]].copy()
    benchmark["date"] = _normalize_dates(benchmark["date"])
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    _require_unique_key(benchmark, ["date"], "H00300 总收益")
    if (
        benchmark["close"].isna().any()
        or (~np.isfinite(benchmark["close"])).any()
        or benchmark["close"].le(0.0).any()
    ):
        raise StressTransmissionContractError("H00300 总收益收盘值必须为有限正数")
    benchmark_close = benchmark.set_index("date")["close"].reindex(dates)
    benchmark_return20 = benchmark_close / benchmark_close.shift(lookback_days) - 1.0
    leadership_gap20 = benchmark_return20 - equal_weight_return20

    tail_eligible = return5.notna() & volatility60.notna() & membership_mask
    tail_eligible_count = tail_eligible.sum(axis=1)
    tail_coverage = tail_eligible_count / member_count
    tail_gate = tail_coverage.ge(member_coverage_minimum)
    tail_flag = return5.lt(
        -tail_sigma_multiple * volatility60 * np.sqrt(float(change_days))
    ).where(tail_eligible)
    tail_diffusion5 = tail_flag.sum(axis=1, skipna=True) / tail_eligible_count.replace(
        0, np.nan
    )
    tail_diffusion5 = tail_diffusion5.where(tail_gate)

    daily_member_return = wide.where(membership_mask)
    daily_valid_count = daily_member_return.notna().sum(axis=1)
    daily_coverage = daily_valid_count / member_count
    daily_sum = daily_member_return.sum(axis=1, min_count=1)
    peer_return = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)
    daily_gate = daily_coverage.ge(member_coverage_minimum)
    for symbol in symbols:
        denominator = daily_valid_count - daily_member_return[symbol].notna().astype(int)
        peer = (daily_sum - daily_member_return[symbol]) / denominator.replace(0, np.nan)
        peer_return[symbol] = peer.where(
            daily_gate & membership_mask[symbol] & daily_member_return[symbol].notna()
        )

    comovement20 = pd.Series(np.nan, index=dates, dtype=float)
    comovement_scoreable_ratio = pd.Series(np.nan, index=dates, dtype=float)
    for position, date in enumerate(dates):
        if position < lookback_days - 1:
            continue
        current_symbols = symbols[membership_mask.loc[date].to_numpy(dtype=bool)]
        window_dates = dates[position - lookback_days + 1 : position + 1]
        correlations: list[float] = []
        for symbol in current_symbols:
            pair = pd.DataFrame(
                {
                    "member": wide.loc[window_dates, symbol],
                    "peer": peer_return.loc[window_dates, symbol],
                }
            ).dropna()
            if len(pair) < comovement_minimum_valid_observations:
                continue
            if pair["member"].nunique() < 2 or pair["peer"].nunique() < 2:
                continue
            correlation = float(pair["member"].corr(pair["peer"]))
            if np.isfinite(correlation):
                correlations.append(correlation)
        ratio = len(correlations) / len(current_symbols) if len(current_symbols) else 0.0
        comovement_scoreable_ratio.loc[date] = ratio
        if ratio >= comovement_member_ratio_minimum and correlations:
            comovement20.loc[date] = float(np.median(correlations))

    feature = pd.DataFrame(index=dates)
    feature["point_in_time_member_count"] = member_count
    feature["return20_scoreable_member_count"] = valid20_count
    feature["return20_coverage_ratio"] = return20_coverage
    feature["tail_scoreable_member_count"] = tail_eligible_count
    feature["tail_coverage_ratio"] = tail_coverage
    feature["comovement_scoreable_member_ratio"] = comovement_scoreable_ratio
    feature["breadth20"] = breadth20
    feature["leadership_gap20"] = leadership_gap20
    feature["comovement20"] = comovement20
    feature["breadth_drop5"] = breadth20.shift(change_days) - breadth20
    feature["tail_diffusion5"] = tail_diffusion5
    feature["comovement_accel5"] = comovement20 - comovement20.shift(change_days)
    feature["f1_breadth_risk_percentile"] = causal_midrank(-feature["breadth20"])
    feature["f2_leadership_risk_percentile"] = causal_midrank(
        feature["leadership_gap20"]
    )
    feature["f3_comovement_risk_percentile"] = causal_midrank(
        feature["comovement20"]
    )
    feature["F"] = required_median(
        feature,
        [
            "f1_breadth_risk_percentile",
            "f2_leadership_risk_percentile",
            "f3_comovement_risk_percentile",
        ],
    )
    feature["t1_breadth_drop_risk_percentile"] = causal_midrank(
        feature["breadth_drop5"]
    )
    feature["t2_tail_diffusion_risk_percentile"] = causal_midrank(
        feature["tail_diffusion5"]
    )
    feature["t3_comovement_accel_risk_percentile"] = causal_midrank(
        feature["comovement_accel5"]
    )
    feature["internal_coverage_state"] = np.where(
        return20_gate
        & tail_gate
        & comovement_scoreable_ratio.ge(comovement_member_ratio_minimum),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    return feature.reset_index(names="date")


def causal_latest_observation(
    *,
    market_dates: pd.DatetimeIndex,
    releases: pd.DataFrame,
    value_column: str,
    observation_date_column: str = "observation_date",
    available_at_column: str = "available_at",
) -> pd.DataFrame:
    """按 t 日 15:00 可得时钟选择最新参考期及其当时可见版本。"""

    _require_columns(
        releases,
        [observation_date_column, available_at_column, value_column],
        f"{value_column} 发布账本",
    )
    source = releases[
        [observation_date_column, available_at_column, value_column]
    ].copy()
    source[observation_date_column] = _normalize_dates(source[observation_date_column])
    source[available_at_column] = _normalize_utc(source[available_at_column])
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce")
    source = source.dropna()
    source = source.loc[np.isfinite(source[value_column])].sort_values(
        [observation_date_column, available_at_column], kind="stable"
    )
    if source.empty:
        return pd.DataFrame(
            {
                "date": market_dates,
                value_column: np.nan,
                f"{value_column}_source_observation_date": pd.NaT,
                f"{value_column}_source_available_at": pd.NaT,
            }
        )

    rows: list[dict[str, Any]] = []
    normalized_dates = pd.DatetimeIndex(market_dates).normalize()
    close_times = normalized_dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    close_times = close_times.tz_convert("UTC")
    for market_date, close_time in zip(normalized_dates, close_times):
        eligible = source.loc[
            source[observation_date_column].le(market_date)
            & source[available_at_column].le(close_time)
        ]
        if eligible.empty:
            rows.append(
                {
                    "date": market_date,
                    value_column: np.nan,
                    f"{value_column}_source_observation_date": pd.NaT,
                    f"{value_column}_source_available_at": pd.NaT,
                }
            )
            continue
        latest_observation = eligible[observation_date_column].max()
        chosen = eligible.loc[
            eligible[observation_date_column].eq(latest_observation)
        ].sort_values(available_at_column, kind="stable").iloc[-1]
        rows.append(
            {
                "date": market_date,
                value_column: float(chosen[value_column]),
                f"{value_column}_source_observation_date": pd.Timestamp(
                    chosen[observation_date_column]
                ),
                f"{value_column}_source_available_at": pd.Timestamp(
                    chosen[available_at_column]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_macro_features(
    *,
    market_dates: pd.DatetimeIndex,
    earnings_yield_releases: pd.DataFrame,
    china_10y_releases: pd.DataFrame,
    dr007_releases: pd.DataFrame,
    reverse_repo_7d_releases: pd.DataFrame,
) -> pd.DataFrame:
    """构造 M 与资金冲击；月度信用、汇率和新闻不进入核心。"""

    channels = [
        (earnings_yield_releases, "csi300_earnings_yield"),
        (china_10y_releases, "china_10y_yield"),
        (dr007_releases, "dr007"),
        (reverse_repo_7d_releases, "reverse_repo_7d_rate"),
    ]
    frame = pd.DataFrame({"date": pd.DatetimeIndex(market_dates).normalize()})
    for releases, value_column in channels:
        selected = causal_latest_observation(
            market_dates=pd.DatetimeIndex(frame["date"]),
            releases=releases,
            value_column=value_column,
        )
        frame = frame.merge(selected, on="date", how="left", validate="one_to_one")

    frame["ey_rate_buffer"] = (
        frame["csi300_earnings_yield"] - frame["china_10y_yield"]
    )
    frame["funding_spread"] = frame["dr007"] - frame["reverse_repo_7d_rate"]
    frame["funding_spread_5d_mean"] = frame["funding_spread"].rolling(
        5, min_periods=5
    ).mean()
    frame["funding_spread_20d_mean"] = frame["funding_spread"].rolling(
        20, min_periods=20
    ).mean()
    frame["funding_shock5"] = (
        frame["funding_spread_5d_mean"] - frame["funding_spread_20d_mean"]
    )
    frame["m1_ey_rate_buffer_risk_percentile"] = causal_midrank(
        -frame["ey_rate_buffer"]
    )
    frame["m2_funding_spread_risk_percentile"] = causal_midrank(
        frame["funding_spread_5d_mean"]
    )
    frame["M"] = required_median(
        frame,
        [
            "m1_ey_rate_buffer_risk_percentile",
            "m2_funding_spread_risk_percentile",
        ],
    )
    frame["t4_funding_shock_risk_percentile"] = causal_midrank(
        frame["funding_shock5"]
    )
    frame["macro_state"] = np.where(frame["M"].notna(), VIEW_ALLOWED, NO_VIEW)
    return frame


def build_b1_price_risk_features(etf_returns: pd.DataFrame) -> pd.DataFrame:
    """构造冻结的 ETF 自身价格风险基准 B1。"""

    _require_columns(etf_returns, ["date", "daily_total_return"], "510300 总收益")
    frame = etf_returns[["date", "daily_total_return"]].copy()
    frame["date"] = _normalize_dates(frame["date"])
    frame["daily_total_return"] = pd.to_numeric(
        frame["daily_total_return"], errors="coerce"
    )
    _require_unique_key(frame, ["date"], "510300 总收益")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    invalid = frame["daily_total_return"].le(-1.0) & frame["daily_total_return"].notna()
    if invalid.any():
        raise StressTransmissionContractError("510300 日总收益不得小于或等于 -100%")
    wealth = (1.0 + frame["daily_total_return"]).cumprod(skipna=False)
    frame["realized_vol20"] = frame["daily_total_return"].rolling(
        20, min_periods=20
    ).std(ddof=1)
    frame["drawdown20"] = wealth / wealth.rolling(20, min_periods=20).max() - 1.0
    frame["downside_return5"] = wealth / wealth.shift(5) - 1.0
    frame["b1_realized_vol20_risk_percentile"] = causal_midrank(
        frame["realized_vol20"]
    )
    frame["b1_drawdown20_risk_percentile"] = causal_midrank(-frame["drawdown20"])
    frame["b1_downside_return5_risk_percentile"] = causal_midrank(
        -frame["downside_return5"]
    )
    frame["B1_state"] = np.where(
        frame[list(MODEL_FEATURES["B1"])].notna().all(axis=1),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    return frame


def assemble_feature_panel(
    *,
    internal: pd.DataFrame,
    macro: pd.DataFrame,
    b1: pd.DataFrame,
) -> pd.DataFrame:
    """合并固定通道并生成 B0/B1/B2/B3 所需设计列。"""

    _require_columns(
        internal,
        [
            "date",
            "F",
            "t1_breadth_drop_risk_percentile",
            "t2_tail_diffusion_risk_percentile",
            "t3_comovement_accel_risk_percentile",
            "internal_coverage_state",
        ],
        "内部特征",
    )
    _require_columns(
        macro,
        ["date", "M", "t4_funding_shock_risk_percentile", "macro_state"],
        "宏观特征",
    )
    _require_columns(b1, ["date", *MODEL_FEATURES["B1"], "B1_state"], "B1 特征")
    panel = (
        internal.merge(macro, on="date", how="outer", validate="one_to_one")
        .merge(b1, on="date", how="outer", validate="one_to_one")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    panel["T"] = required_median(
        panel,
        [
            "t1_breadth_drop_risk_percentile",
            "t2_tail_diffusion_risk_percentile",
            "t3_comovement_accel_risk_percentile",
            "t4_funding_shock_risk_percentile",
        ],
    )
    panel["T_x_F"] = panel["T"] * panel["F"]
    panel["T_x_M"] = panel["T"] * panel["M"]
    panel["B2_state"] = np.where(
        panel["internal_coverage_state"].eq(VIEW_ALLOWED)
        & panel[["T", "T_x_F"]].notna().all(axis=1),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["B3_state"] = np.where(
        panel["B2_state"].eq(VIEW_ALLOWED)
        & panel["macro_state"].eq(VIEW_ALLOWED)
        & panel[["T", "T_x_F", "T_x_M"]].notna().all(axis=1),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    assert_predictive_names_are_industry_free(
        [name for columns in MODEL_FEATURES.values() for name in columns]
    )
    return panel


def build_bad10_event_ledger(
    origin_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把重叠 BAD10 正样本合并为事件，并使每个事件训练权重之和为 1。"""

    required = [
        "origin_date",
        "entry_date",
        "horizon_end_date",
        "bad10",
        "first_breach_date",
    ]
    _require_columns(origin_panel, required, "BAD10 原点面板")
    frame = origin_panel[required].copy()
    for column in ("origin_date", "entry_date", "horizon_end_date", "first_breach_date"):
        frame[column] = _normalize_dates(frame[column])
    frame["bad10"] = pd.to_numeric(frame["bad10"], errors="coerce")
    if frame[["origin_date", "entry_date", "horizon_end_date", "bad10"]].isna().any().any():
        raise StressTransmissionContractError("BAD10 原点面板存在不可解析关键字段")
    if not set(frame["bad10"].unique()).issubset({0, 1}):
        raise StressTransmissionContractError("BAD10 标签必须为 0/1")
    _require_unique_key(frame, ["origin_date"], "BAD10 原点面板")
    if not frame["origin_date"].sort_values(kind="stable").is_monotonic_increasing:
        raise StressTransmissionContractError("BAD10 原点日期不可排序")
    if not (frame["origin_date"] < frame["entry_date"]).all():
        raise StressTransmissionContractError("BAD10 执行日必须晚于信息原点")
    if not (frame["entry_date"] <= frame["horizon_end_date"]).all():
        raise StressTransmissionContractError("BAD10 持有区间非法")
    positive = frame.loc[frame["bad10"].eq(1)].sort_values(
        ["entry_date", "origin_date"], kind="stable"
    )
    if positive["first_breach_date"].isna().any():
        raise StressTransmissionContractError("BAD10 正样本缺少首次突破日期")

    groups: list[pd.DataFrame] = []
    current_indices: list[int] = []
    current_end: pd.Timestamp | None = None
    for index, row in positive.iterrows():
        entry_date = pd.Timestamp(row["entry_date"])
        horizon_end = pd.Timestamp(row["horizon_end_date"])
        if current_end is None or entry_date <= current_end:
            current_indices.append(index)
            current_end = horizon_end if current_end is None else max(current_end, horizon_end)
        else:
            groups.append(positive.loc[current_indices].copy())
            current_indices = [index]
            current_end = horizon_end
    if current_indices:
        groups.append(positive.loc[current_indices].copy())

    event_rows: list[dict[str, Any]] = []
    origin_to_event: dict[pd.Timestamp, tuple[str, int]] = {}
    previous_event_end: pd.Timestamp | None = None
    for number, group in enumerate(groups, start=1):
        ordered = group.sort_values("origin_date", kind="stable")
        event_id = f"BAD10_V2_EVENT_{number:04d}"
        event_start = pd.Timestamp(ordered["entry_date"].min())
        event_end = pd.Timestamp(ordered["horizon_end_date"].max())
        if previous_event_end is not None and event_start <= previous_event_end:
            raise StressTransmissionContractError("BAD10 事件合并后仍有重叠")
        previous_event_end = event_end
        count = int(len(ordered))
        first_breach = pd.Timestamp(ordered["first_breach_date"].min())
        year_block = f"CALENDAR_YEAR_{first_breach.year:04d}"
        event_rows.append(
            {
                "event_id": event_id,
                "event_origin_start": pd.Timestamp(ordered["origin_date"].min()),
                "event_origin_end": pd.Timestamp(ordered["origin_date"].max()),
                "event_entry_start": event_start,
                "event_entry_end": pd.Timestamp(ordered["entry_date"].max()),
                "event_horizon_end": event_end,
                "first_breach_date": first_breach,
                "positive_origin_count": count,
                "positive_training_weight_sum": 1.0,
                "split_group_id": event_id,
                "bootstrap_event_block_id": event_id,
                "calendar_year_block_id": year_block,
            }
        )
        for origin_date in ordered["origin_date"]:
            origin_to_event[pd.Timestamp(origin_date)] = (event_id, count)

    events = pd.DataFrame(event_rows, columns=EVENT_LEDGER_COLUMNS)
    event_lookup = events.set_index("event_id") if not events.empty else pd.DataFrame()
    sample_rows: list[dict[str, Any]] = []
    for row in frame.sort_values("origin_date", kind="stable").itertuples(index=False):
        origin_date = pd.Timestamp(row.origin_date)
        if int(row.bad10) == 1:
            event_id, count = origin_to_event[origin_date]
            event_record = event_lookup.loc[event_id]
            sample_group = event_id
            bootstrap_event = event_id
            year_block = str(event_record["calendar_year_block_id"])
            weight = 1.0 / count
        else:
            event_id = pd.NA
            sample_group = f"NON_EVENT_DAY_{origin_date:%Y%m%d}"
            bootstrap_event = sample_group
            year_block = f"CALENDAR_YEAR_{origin_date.year:04d}"
            weight = 1.0
        sample_rows.append(
            {
                "origin_date": origin_date,
                "entry_date": pd.Timestamp(row.entry_date),
                "horizon_end_date": pd.Timestamp(row.horizon_end_date),
                "bad10": int(row.bad10),
                "event_id": event_id,
                "sample_group_id": sample_group,
                "bootstrap_event_block_id": bootstrap_event,
                "calendar_year_block_id": year_block,
                "training_weight": float(weight),
                "split_assignment": UNASSIGNED_SPLIT,
            }
        )
    samples = pd.DataFrame(sample_rows, columns=SAMPLE_LEDGER_COLUMNS)
    validate_event_weighting(events=events, samples=samples)
    validate_split_integrity(samples)
    return events, samples


def validate_event_weighting(
    *,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    tolerance: float = 1e-12,
) -> None:
    """验证每个压力事件的正样本训练权重之和严格为 1。"""

    _require_columns(events, EVENT_LEDGER_COLUMNS, "BAD10 事件账本")
    _require_columns(samples, SAMPLE_LEDGER_COLUMNS, "BAD10 样本账本")
    positive = samples.loc[samples["bad10"].eq(1)]
    if positive["event_id"].isna().any():
        raise StressTransmissionContractError("BAD10 正样本缺少 event_id")
    sums = positive.groupby("event_id")["training_weight"].sum()
    expected_ids = set(events["event_id"].astype(str))
    if set(sums.index.astype(str)) != expected_ids:
        raise StressTransmissionContractError("事件表与正样本事件集合不一致")
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=tolerance, rtol=0.0):
        raise StressTransmissionContractError("存在事件的正样本训练权重之和不为 1")
    negative = samples.loc[samples["bad10"].eq(0), "training_weight"]
    if not negative.eq(1.0).all():
        raise StressTransmissionContractError("非事件风险日训练权重必须固定为 1")


def validate_split_integrity(
    samples: pd.DataFrame,
    *,
    split_column: str = "split_assignment",
) -> None:
    """禁止同一压力事件跨训练、校准与评价区间。"""

    _require_columns(samples, ["bad10", "event_id", split_column], "模型样本账本")
    positive = samples.loc[samples["bad10"].eq(1)].copy()
    assigned = positive.loc[positive[split_column].ne(UNASSIGNED_SPLIT)]
    if assigned.empty:
        return
    split_counts = assigned.groupby("event_id")[split_column].nunique()
    if split_counts.gt(1).any():
        offending = list(split_counts.loc[split_counts.gt(1)].index.astype(str))
        raise StressTransmissionContractError(
            f"同一 BAD10 事件跨越多个样本区间：{offending}"
        )


def assign_group_splits(
    samples: pd.DataFrame,
    assignments: Mapping[str, str],
) -> pd.DataFrame:
    """仅按完整 sample_group_id 分配样本区间。"""

    _require_columns(samples, SAMPLE_LEDGER_COLUMNS, "模型样本账本")
    allowed_splits = {"TRAIN", "CALIBRATION", "EVALUATION"}
    invalid = sorted(set(assignments.values()).difference(allowed_splits))
    if invalid:
        raise StressTransmissionContractError(f"非法样本区间：{invalid}")
    result = samples.copy()
    result["split_assignment"] = result["sample_group_id"].map(assignments).fillna(
        UNASSIGNED_SPLIT
    )
    validate_split_integrity(result)
    return result


@dataclass(frozen=True)
class ConstrainedLogitFit:
    """固定 L2、非负风险系数的逻辑回归结果。"""

    model_id: str
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    l2_penalty: float
    converged: bool
    iterations: int


def build_model_matrix(
    panel: pd.DataFrame,
    *,
    model_id: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """按固定 B0/B1/B2/B3 定义生成设计矩阵，不做变量选择。"""

    if model_id not in MODEL_FEATURES:
        raise StressTransmissionContractError(f"未知固定模型：{model_id}")
    feature_names = MODEL_FEATURES[model_id]
    assert_predictive_names_are_industry_free(feature_names)
    _require_columns(panel, ["date", *feature_names], f"{model_id} 特征面板")
    dates = panel[["date"]].copy()
    if feature_names:
        matrix = panel[list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
    else:
        matrix = np.empty((len(panel), 0), dtype=float)
    return dates, matrix


def fit_constrained_logit(
    *,
    model_id: str,
    matrix: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    l2_penalty: float = 1.0,
) -> ConstrainedLogitFit:
    """拟合截距自由、其余系数非负且 L2 固定为 1.0 的逻辑回归。"""

    if model_id not in MODEL_FEATURES:
        raise StressTransmissionContractError(f"未知固定模型：{model_id}")
    if l2_penalty != 1.0:
        raise StressTransmissionContractError("V2 L2 惩罚系数已冻结为 1.0，禁止搜索")
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y) or len(y) != len(weight):
        raise StressTransmissionContractError("模型矩阵、标签和权重维度不一致")
    if x.shape[1] != len(MODEL_FEATURES[model_id]):
        raise StressTransmissionContractError("模型矩阵列数与固定模型定义不一致")
    if len(y) == 0 or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise StressTransmissionContractError("模型标签必须是非空 0/1 序列")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise StressTransmissionContractError("模型输入不得含缺失或非有限值")
    if not np.isfinite(weight).all() or (weight <= 0.0).any():
        raise StressTransmissionContractError("模型样本权重必须为有限正数")
    total_weight = float(weight.sum())
    weighted_rate = float(np.dot(weight, y) / total_weight)
    clipped_rate = float(np.clip(weighted_rate, 1e-8, 1.0 - 1e-8))
    initial = np.zeros(x.shape[1] + 1, dtype=float)
    initial[0] = np.log(clipped_rate / (1.0 - clipped_rate))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        linear = intercept + x @ coefficients
        probability = np.where(
            linear >= 0.0,
            1.0 / (1.0 + np.exp(-linear)),
            np.exp(linear) / (1.0 + np.exp(linear)),
        )
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        loss = -float(
            np.dot(weight, y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
            / total_weight
        )
        penalty = 0.5 * l2_penalty * float(np.dot(coefficients, coefficients))
        residual = weight * (probability - y) / total_weight
        gradient = np.concatenate(
            ([float(residual.sum())], x.T @ residual + l2_penalty * coefficients)
        )
        return loss + penalty, gradient

    bounds = [(None, None), *[(0.0, None) for _ in range(x.shape[1])]]
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success:
        raise StressTransmissionContractError(
            f"受约束逻辑回归未收敛：{fitted.message}"
        )
    coefficients = tuple(float(value) for value in fitted.x[1:])
    if any(value < -1e-12 for value in coefficients):
        raise StressTransmissionContractError("拟合结果违反非负系数约束")
    return ConstrainedLogitFit(
        model_id=model_id,
        feature_names=MODEL_FEATURES[model_id],
        intercept=float(fitted.x[0]),
        coefficients=coefficients,
        l2_penalty=l2_penalty,
        converged=True,
        iterations=int(fitted.nit),
    )


def predict_probability(
    fit: ConstrainedLogitFit,
    matrix: np.ndarray,
) -> np.ndarray:
    """使用已拟合的固定模型产生概率，不包含阈值或仓位映射。"""

    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(fit.coefficients):
        raise StressTransmissionContractError("预测矩阵列数与模型不一致")
    if not np.isfinite(x).all():
        raise StressTransmissionContractError("预测矩阵不得含缺失或非有限值")
    linear = fit.intercept + x @ np.asarray(fit.coefficients, dtype=float)
    return np.where(
        linear >= 0.0,
        1.0 / (1.0 + np.exp(-linear)),
        np.exp(linear) / (1.0 + np.exp(linear)),
    )
