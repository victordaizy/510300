"""V2 四态历史账本的未复权日线与公司行动对账构件。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.stress_transmission_hazard_v2 import (
    ACTION_NONE_CONFIRMED,
    ACTION_RESOLVED,
    ACTION_UNRESOLVED,
    ConstituentReturnState,
    StressTransmissionContractError,
    build_daily_coverage_ledger,
    classify_constituent_returns,
)


DAILY_PROVIDER_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
LEGACY_ALLOWED_COLUMNS = [
    "con_code",
    "date",
    "pre_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "pct_chg",
    "vol",
    "amount",
    "price_source",
]
DIVIDEND_COLUMNS = [
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
]
PRICE_COLUMNS = ["raw_open", "raw_high", "raw_low", "raw_close", "pre_close"]


@dataclass(frozen=True)
class ReconciliationResult:
    """四态分类输入、分类结果、行动账本和覆盖账本。"""

    daily_observations: pd.DataFrame
    classifier_inputs: pd.DataFrame
    classified_returns: pd.DataFrame
    corporate_action_ledger: pd.DataFrame
    daily_coverage: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise StressTransmissionContractError(f"{label}缺少字段：{missing}")


def _normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _strict_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        sample = frame.loc[frame.duplicated(keys, keep=False), keys].head(10)
        raise StressTransmissionContractError(
            f"{label}存在重复键：{sample.to_dict('records')}"
        )


def normalize_provider_daily(
    frame: pd.DataFrame,
    *,
    source_id: str = "FRESH_TUSHARE_DAILY",
    pct_tolerance: float = 0.011,
) -> pd.DataFrame:
    """规范供应商未复权日线，并标记而不是删除内部冲突。"""

    _require_columns(frame, DAILY_PROVIDER_COLUMNS, "供应商日线")
    work = frame[DAILY_PROVIDER_COLUMNS].copy().rename(
        columns={
            "ts_code": "symbol",
            "trade_date": "date",
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
        }
    )
    work["date"] = _normalize_date(work["date"])
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    numeric_columns = PRICE_COLUMNS + ["change", "pct_chg", "vol", "amount"]
    work[numeric_columns] = work[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if work[["date", "symbol"]].isna().any().any() or work["symbol"].eq("").any():
        raise StressTransmissionContractError("供应商日线证券或日期非法")
    _strict_unique(work, ["symbol", "date"], "供应商日线")

    prices_finite = pd.Series(
        np.isfinite(work[PRICE_COLUMNS].to_numpy(dtype=float)).all(axis=1),
        index=work.index,
    ) & work[PRICE_COLUMNS].gt(0.0).all(axis=1)
    other_finite = pd.Series(
        np.isfinite(work[["pct_chg", "vol", "amount"]].to_numpy(dtype=float)).all(
            axis=1
        ),
        index=work.index,
    ) & work[["vol", "amount"]].ge(0.0).all(axis=1)
    envelope = (
        work["raw_high"].ge(work[["raw_open", "raw_close", "raw_low"]].max(axis=1))
        & work["raw_low"].le(
            work[["raw_open", "raw_close", "raw_high"]].min(axis=1)
        )
    )
    expected_pct = (work["raw_close"] / work["pre_close"] - 1.0) * 100.0
    identity_error = (expected_pct - work["pct_chg"]).abs()
    identity_passed = identity_error.le(pct_tolerance)
    work["daily_internal_conflict"] = ~(
        prices_finite & other_finite & envelope & identity_passed
    )
    work["pct_chg_identity_abs_error"] = identity_error
    work["daily_source"] = source_id
    return work.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def normalize_legacy_daily(
    frame: pd.DataFrame,
    *,
    pct_tolerance: float = 0.011,
) -> pd.DataFrame:
    """只提取旧种子的未复权原始列，拒绝任何旧收益或停牌语义。"""

    _require_columns(frame, LEGACY_ALLOWED_COLUMNS, "旧日线种子")
    provider = frame[LEGACY_ALLOWED_COLUMNS].copy().rename(
        columns={
            "con_code": "ts_code",
            "date": "trade_date",
            "raw_open": "open",
            "raw_high": "high",
            "raw_low": "low",
            "raw_close": "close",
        }
    )
    provider["change"] = pd.to_numeric(provider["close"], errors="coerce") - pd.to_numeric(
        provider["pre_close"], errors="coerce"
    )
    normalized = normalize_provider_daily(
        provider,
        source_id="LEGACY_REVALIDATED_UNADJUSTED_DAILY_COLUMNS_ONLY",
        pct_tolerance=pct_tolerance,
    )
    return normalized


def combine_daily_sources(
    legacy: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    price_tolerance: float = 0.011,
    pct_tolerance: float = 0.011,
) -> pd.DataFrame:
    """合并新旧未复权日线；重叠冲突保留并失败关闭。"""

    old = normalize_legacy_daily(legacy, pct_tolerance=pct_tolerance)
    new = normalize_provider_daily(fresh, pct_tolerance=pct_tolerance)
    compare_columns = PRICE_COLUMNS + ["pct_chg"]
    overlap = old[["symbol", "date", *compare_columns]].merge(
        new[["symbol", "date", *compare_columns]],
        on=["symbol", "date"],
        how="inner",
        suffixes=("_legacy", "_fresh"),
        validate="one_to_one",
    )
    conflict = pd.Series(False, index=overlap.index)
    for column in PRICE_COLUMNS:
        conflict |= (
            overlap[f"{column}_legacy"] - overlap[f"{column}_fresh"]
        ).abs().gt(price_tolerance)
    conflict |= (
        overlap["pct_chg_legacy"] - overlap["pct_chg_fresh"]
    ).abs().gt(pct_tolerance)
    conflict_keys = overlap.loc[conflict, ["symbol", "date"]].assign(
        source_overlap_conflict=True
    )

    old = old.assign(source_priority=1)
    new = new.assign(source_priority=2)
    combined = (
        pd.concat([old, new], ignore_index=True)
        .sort_values(["symbol", "date", "source_priority"], kind="stable")
        .drop_duplicates(["symbol", "date"], keep="last")
        .drop(columns="source_priority")
    )
    combined = combined.merge(
        conflict_keys,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    combined["source_overlap_conflict"] = combined[
        "source_overlap_conflict"
    ].fillna(False).astype(bool)
    combined["supplier_conflict"] = (
        combined["daily_internal_conflict"] | combined["source_overlap_conflict"]
    )
    combined = combined.sort_values(["symbol", "date"], kind="stable").reset_index(
        drop=True
    )
    combined["previous_observed_date"] = combined.groupby("symbol", sort=False)[
        "date"
    ].shift()
    combined["previous_unadjusted_close"] = combined.groupby("symbol", sort=False)[
        "raw_close"
    ].shift()
    return combined


def identify_action_candidate_symbols(
    daily: pd.DataFrame,
    *,
    absolute_tolerance_cny: float = 0.011,
) -> list[str]:
    """只用当日 `pre_close` 与前一实际收盘定位候选证券。"""

    _require_columns(
        daily,
        ["symbol", "pre_close", "previous_unadjusted_close"],
        "合并日线",
    )
    gap = (daily["pre_close"] - daily["previous_unadjusted_close"]).abs()
    return sorted(daily.loc[gap.gt(absolute_tolerance_cny), "symbol"].astype(str).unique())


def prepare_dividend_actions(
    dividends: pd.DataFrame,
    *,
    admitted_process: str = "实施",
) -> pd.DataFrame:
    """把 `dividend` 压成每证券除权日一行，冲突条款不做选择。"""

    _require_columns(dividends, DIVIDEND_COLUMNS, "分红送转")
    work = dividends[DIVIDEND_COLUMNS].copy()
    work["symbol"] = work["ts_code"].astype("string").str.strip().str.upper()
    for column in ("ann_date", "record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date"):
        work[column] = _normalize_date(work[column])
    for column in ("stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        work["div_proc"].astype("string").str.strip().eq(admitted_process)
        & work["symbol"].notna()
        & work["ex_date"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "action_row_count",
                "exact_duplicate_count",
                "conflicting_economic_terms",
                "terms_available_before_ex_date",
                "action_terms_valid",
                "cash_distribution_per_pre_event_share",
                "post_to_pre_share_ratio",
                "subscription_cash_outflow_per_pre_event_share",
                "corporate_action_evidence_id",
                "imp_ann_date",
            ]
        )

    output: list[dict[str, Any]] = []
    signature_columns = ["stk_div", "cash_div_tax"]
    for (symbol, ex_date), group in work.groupby(["symbol", "ex_date"], sort=True):
        signatures = group[signature_columns].drop_duplicates()
        conflicting = len(signatures) > 1
        chosen = group.sort_values(["imp_ann_date", "ann_date"], kind="stable").iloc[0]
        stock_dividend = float(chosen["stk_div"]) if pd.notna(chosen["stk_div"]) else np.nan
        gross_cash = (
            float(chosen["cash_div_tax"])
            if pd.notna(chosen["cash_div_tax"])
            else np.nan
        )
        terms_available = bool(
            pd.notna(chosen["imp_ann_date"])
            and pd.Timestamp(chosen["imp_ann_date"]) < pd.Timestamp(ex_date)
        )
        terms_valid = bool(
            not conflicting
            and np.isfinite(stock_dividend)
            and stock_dividend >= 0.0
            and np.isfinite(gross_cash)
            and gross_cash >= 0.0
            and terms_available
        )
        evidence_payload = {
            "symbol": str(symbol),
            "ex_date": pd.Timestamp(ex_date).date().isoformat(),
            "stk_div": None if not np.isfinite(stock_dividend) else stock_dividend,
            "cash_div_tax": None if not np.isfinite(gross_cash) else gross_cash,
            "imp_ann_date": (
                None
                if pd.isna(chosen["imp_ann_date"])
                else pd.Timestamp(chosen["imp_ann_date"]).date().isoformat()
            ),
            "conflicting": conflicting,
        }
        evidence_hash = hashlib.sha256(
            json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        output.append(
            {
                "symbol": str(symbol),
                "date": pd.Timestamp(ex_date).normalize(),
                "action_row_count": int(len(group)),
                "exact_duplicate_count": int(len(group) - len(group.drop_duplicates())),
                "conflicting_economic_terms": conflicting,
                "terms_available_before_ex_date": terms_available,
                "action_terms_valid": terms_valid,
                "cash_distribution_per_pre_event_share": gross_cash,
                "post_to_pre_share_ratio": 1.0 + stock_dividend,
                "subscription_cash_outflow_per_pre_event_share": 0.0,
                "corporate_action_evidence_id": f"TUSHARE_DIVIDEND_{evidence_hash}",
                "imp_ann_date": chosen["imp_ann_date"],
            }
        )
    result = pd.DataFrame(output)
    _strict_unique(result, ["symbol", "date"], "公司行动压缩账本")
    return result.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def build_classifier_inputs(
    daily: pd.DataFrame,
    dividend_actions: pd.DataFrame,
    *,
    action_tolerance_cny: float = 0.011,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成核心四态分类器输入，并保留每个行动候选的裁决原因。"""

    _require_columns(
        daily,
        [
            "symbol",
            "date",
            "raw_close",
            "pre_close",
            "previous_unadjusted_close",
            "previous_observed_date",
            "supplier_conflict",
            "daily_source",
        ],
        "合并日线",
    )
    action_columns = [
        "symbol",
        "date",
        "action_row_count",
        "exact_duplicate_count",
        "conflicting_economic_terms",
        "terms_available_before_ex_date",
        "action_terms_valid",
        "cash_distribution_per_pre_event_share",
        "post_to_pre_share_ratio",
        "subscription_cash_outflow_per_pre_event_share",
        "corporate_action_evidence_id",
        "imp_ann_date",
    ]
    _require_columns(dividend_actions, action_columns, "公司行动账本")
    frame = daily.merge(
        dividend_actions[action_columns],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    reference_gap = (frame["pre_close"] - frame["previous_unadjusted_close"]).abs()
    action_candidate = frame["previous_unadjusted_close"].notna() & reference_gap.gt(
        action_tolerance_cny
    )
    has_action_row = frame["action_row_count"].notna()
    action_status = pd.Series(ACTION_NONE_CONFIRMED, index=frame.index, dtype="string")
    evidence_id = pd.Series(
        "TUSHARE_DAILY_PRECLOSE_MATCH_PRIOR_RAW_CLOSE_V1",
        index=frame.index,
        dtype="string",
    )
    cash = pd.Series(0.0, index=frame.index, dtype=float)
    ratio = pd.Series(1.0, index=frame.index, dtype=float)
    outflow = pd.Series(0.0, index=frame.index, dtype=float)
    resolution_reason = pd.Series("NO_ACTION_CONFIRMED_BY_DAILY_PRECLOSE", index=frame.index, dtype="string")

    unexpected_action_row = (~action_candidate) & has_action_row & (
        pd.to_numeric(
            frame["cash_distribution_per_pre_event_share"], errors="coerce"
        ).fillna(np.inf).ne(0.0)
        | pd.to_numeric(frame["post_to_pre_share_ratio"], errors="coerce")
        .fillna(np.inf)
        .ne(1.0)
    )
    action_status.loc[unexpected_action_row] = ACTION_UNRESOLVED
    evidence_id.loc[unexpected_action_row] = frame.loc[
        unexpected_action_row, "corporate_action_evidence_id"
    ].fillna("DIVIDEND_ACTION_WITHOUT_DAILY_REFERENCE_CHANGE")
    cash.loc[unexpected_action_row] = np.nan
    ratio.loc[unexpected_action_row] = np.nan
    outflow.loc[unexpected_action_row] = np.nan
    resolution_reason.loc[unexpected_action_row] = "ACTION_TERMS_CONFLICT_WITH_DAILY_PRECLOSE"

    candidate_with_terms = action_candidate & has_action_row
    expected_reference = (
        (
            frame["previous_unadjusted_close"]
            - frame["cash_distribution_per_pre_event_share"]
            + frame["subscription_cash_outflow_per_pre_event_share"]
        )
        / frame["post_to_pre_share_ratio"]
    ).round(2)
    reference_matches = (expected_reference - frame["pre_close"]).abs().le(
        action_tolerance_cny
    )
    resolved = (
        candidate_with_terms
        & frame["action_terms_valid"].eq(True)
        & reference_matches
    )
    action_status.loc[resolved] = ACTION_RESOLVED
    evidence_id.loc[resolved] = frame.loc[resolved, "corporate_action_evidence_id"].astype(
        "string"
    )
    cash.loc[resolved] = frame.loc[
        resolved, "cash_distribution_per_pre_event_share"
    ]
    ratio.loc[resolved] = frame.loc[resolved, "post_to_pre_share_ratio"]
    outflow.loc[resolved] = frame.loc[
        resolved, "subscription_cash_outflow_per_pre_event_share"
    ]
    resolution_reason.loc[resolved] = "IMPLEMENTED_TERMS_MATCH_DAILY_PRECLOSE"

    unresolved_candidate = action_candidate & (~resolved)
    action_status.loc[unresolved_candidate] = ACTION_UNRESOLVED
    evidence_id.loc[unresolved_candidate] = frame.loc[
        unresolved_candidate, "corporate_action_evidence_id"
    ].fillna("DAILY_PRECLOSE_ACTION_CANDIDATE_WITHOUT_MATCHED_TERMS")
    cash.loc[unresolved_candidate] = np.nan
    ratio.loc[unresolved_candidate] = np.nan
    outflow.loc[unresolved_candidate] = np.nan
    resolution_reason.loc[unresolved_candidate] = np.select(
        [
            ~has_action_row.loc[unresolved_candidate],
            frame.loc[unresolved_candidate, "conflicting_economic_terms"].eq(True),
            ~frame.loc[unresolved_candidate, "terms_available_before_ex_date"].eq(True),
            ~frame.loc[unresolved_candidate, "action_terms_valid"].eq(True),
            ~reference_matches.loc[unresolved_candidate],
        ],
        [
            "NO_IMPLEMENTED_DIVIDEND_TERMS_ON_ACTION_DATE",
            "CONFLICTING_DIVIDEND_TERMS",
            "IMPLEMENTATION_TERMS_NOT_STRICTLY_PRIOR",
            "INVALID_OR_INCOMPLETE_DIVIDEND_TERMS",
            "THEORETICAL_REFERENCE_DOES_NOT_MATCH_DAILY_PRECLOSE",
        ],
        default="UNRESOLVED_ACTION",
    )

    inputs = pd.DataFrame(
        {
            "date": frame["date"],
            "symbol": frame["symbol"],
            "previous_unadjusted_close": frame["previous_unadjusted_close"],
            "unadjusted_close": frame["raw_close"],
            "source_observed": True,
            "supplier_conflict": frame["supplier_conflict"].astype(bool),
            "official_suspension": False,
            "suspension_evidence_id": "",
            "corporate_action_status": action_status,
            "corporate_action_evidence_id": evidence_id,
            "cash_distribution_per_pre_event_share": cash,
            "post_to_pre_share_ratio": ratio,
            "subscription_cash_outflow_per_pre_event_share": outflow,
            "daily_source": frame["daily_source"],
            "previous_observed_date": frame["previous_observed_date"],
            "daily_pre_close": frame["pre_close"],
            "action_reference_gap_cny": reference_gap,
            "action_resolution_reason": resolution_reason,
        }
    )
    action_ledger = inputs.loc[
        action_candidate | has_action_row,
        [
            "date",
            "symbol",
            "previous_observed_date",
            "previous_unadjusted_close",
            "daily_pre_close",
            "action_reference_gap_cny",
            "corporate_action_status",
            "corporate_action_evidence_id",
            "cash_distribution_per_pre_event_share",
            "post_to_pre_share_ratio",
            "subscription_cash_outflow_per_pre_event_share",
            "action_resolution_reason",
            "daily_source",
        ],
    ].copy()
    _strict_unique(inputs, ["date", "symbol"], "分类器观测输入")
    return inputs, action_ledger


def normalize_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """规范逐日点时成员，不允许当前成员倒填。"""

    _require_columns(frame, ["membership_date", "index_code", "symbol"], "点时成员")
    work = frame[["membership_date", "index_code", "symbol"]].copy().rename(
        columns={"membership_date": "date"}
    )
    work["date"] = _normalize_date(work["date"])
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    work["index_code"] = work["index_code"].astype("string").str.strip()
    if not work["index_code"].eq("000300").all():
        raise StressTransmissionContractError("点时成员混入非 000300 身份")
    _strict_unique(work, ["date", "symbol"], "点时成员")
    counts = work.groupby("date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise StressTransmissionContractError(
            f"点时成员每日不是 300：min={int(counts.min())}, max={int(counts.max())}"
        )
    return work.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def add_explicit_missing_member_inputs(
    classifier_inputs: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """为无日线的点时成员日追加显式供应商缺失态输入。"""

    members = normalize_membership(membership)
    observed_keys = classifier_inputs[["date", "symbol"]].copy().assign(observed=True)
    missing = members.merge(
        observed_keys,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    missing = missing.loc[missing["observed"].isna(), ["date", "symbol"]]
    if missing.empty:
        return classifier_inputs.sort_values(["date", "symbol"], kind="stable").reset_index(
            drop=True
        )
    additions = pd.DataFrame(
        {
            "date": missing["date"],
            "symbol": missing["symbol"],
            "previous_unadjusted_close": np.nan,
            "unadjusted_close": np.nan,
            "source_observed": False,
            "supplier_conflict": True,
            "official_suspension": False,
            "suspension_evidence_id": "",
            "corporate_action_status": ACTION_UNRESOLVED,
            "corporate_action_evidence_id": "NO_DAILY_ROW_NO_EXCHANGE_OFFICIAL_SUSPENSION_SOURCE",
            "cash_distribution_per_pre_event_share": np.nan,
            "post_to_pre_share_ratio": np.nan,
            "subscription_cash_outflow_per_pre_event_share": np.nan,
            "daily_source": "MISSING_PROVIDER_DAILY_ROW",
            "previous_observed_date": pd.NaT,
            "daily_pre_close": np.nan,
            "action_reference_gap_cny": np.nan,
            "action_resolution_reason": "MISSING_DAILY_ROW_FAIL_CLOSED",
        }
    )
    result = pd.concat([classifier_inputs, additions], ignore_index=True)
    _strict_unique(result, ["date", "symbol"], "含显式缺失的分类器输入")
    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def build_four_state_ledgers(
    *,
    legacy_daily: pd.DataFrame,
    fresh_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    membership: pd.DataFrame,
    price_tolerance: float = 0.011,
    pct_tolerance: float = 0.011,
    member_coverage_minimum: float = 0.98,
) -> ReconciliationResult:
    """执行未复权日线对账、四态分类与日度覆盖，不读取绩效。"""

    daily = combine_daily_sources(
        legacy_daily,
        fresh_daily,
        price_tolerance=price_tolerance,
        pct_tolerance=pct_tolerance,
    )
    actions = prepare_dividend_actions(dividends)
    base_inputs, action_ledger = build_classifier_inputs(
        daily,
        actions,
        action_tolerance_cny=price_tolerance,
    )
    all_inputs = add_explicit_missing_member_inputs(base_inputs, membership)
    classified = classify_constituent_returns(
        all_inputs[
            [
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
        ]
    )
    provenance = all_inputs[
        [
            "date",
            "symbol",
            "daily_source",
            "previous_observed_date",
            "daily_pre_close",
            "action_reference_gap_cny",
            "action_resolution_reason",
        ]
    ]
    classified = classified.merge(
        provenance,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    members = normalize_membership(membership)
    coverage = build_daily_coverage_ledger(
        membership=members,
        classified_returns=classified,
        minimum_member_coverage=member_coverage_minimum,
        reliable_point_in_time_weights=False,
        expected_members_per_day=300,
    )
    official_count = int(
        classified["constituent_return_state"].eq(
            ConstituentReturnState.OFFICIAL_SUSPENSION.value
        ).sum()
    )
    if official_count != 0:
        raise StressTransmissionContractError(
            "未准入交易所官方停牌源时不得产生 OFFICIAL_SUSPENSION"
        )
    unusable = ~classified["return_is_usable"].astype(bool)
    if classified.loc[unusable, "daily_total_shareholder_return"].notna().any():
        raise StressTransmissionContractError("不可用四态不得保留收益数值")
    return ReconciliationResult(
        daily_observations=daily,
        classifier_inputs=all_inputs,
        classified_returns=classified,
        corporate_action_ledger=action_ledger,
        daily_coverage=coverage,
    )
