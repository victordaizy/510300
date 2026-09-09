from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.stress_transmission_hazard_v2 import (
    ACTION_RESOLVED,
    ACTION_UNRESOLVED,
    ConstituentReturnState,
    classify_constituent_returns,
)
from research.stress_transmission_hazard_v2_four_state_ledger_v1 import (
    DIVIDEND_COLUMNS,
    add_explicit_missing_member_inputs,
    build_classifier_inputs,
    combine_daily_sources,
    normalize_membership,
    prepare_dividend_actions,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _legacy(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["raw_open"] = frame.get("raw_open", frame["raw_close"])
    frame["raw_high"] = frame.get("raw_high", frame[["raw_open", "raw_close"]].max(axis=1))
    frame["raw_low"] = frame.get("raw_low", frame[["raw_open", "raw_close"]].min(axis=1))
    frame["vol"] = frame.get("vol", 100.0)
    frame["amount"] = frame.get("amount", 1000.0)
    frame["price_source"] = "tushare.daily"
    return frame[
        [
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
    ]


def _empty_fresh() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def _dividend(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ts_code": "000001.SZ",
        "end_date": "20191231",
        "ann_date": "20200101",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "stk_co_rate": 0.0,
        "cash_div": 0.2,
        "cash_div_tax": 0.2,
        "record_date": "20200105",
        "ex_date": "20200106",
        "pay_date": "20200106",
        "div_listdate": None,
        "imp_ann_date": "20200103",
    }
    row.update(overrides)
    return row


def _two_day_daily() -> pd.DataFrame:
    first_close = 10.0
    second_pre_close = 9.8
    second_close = 10.2
    return combine_daily_sources(
        _legacy(
            [
                {
                    "con_code": "000001.SZ",
                    "date": "2020-01-03",
                    "pre_close": first_close,
                    "raw_close": first_close,
                    "pct_chg": 0.0,
                },
                {
                    "con_code": "000001.SZ",
                    "date": "2020-01-06",
                    "pre_close": second_pre_close,
                    "raw_close": second_close,
                    "pct_chg": (second_close / second_pre_close - 1.0) * 100.0,
                },
            ]
        ),
        _empty_fresh(),
    )


def test_cash_action_resolves_with_prior_announcement_and_reference_match() -> None:
    daily = _two_day_daily()
    actions = prepare_dividend_actions(pd.DataFrame([_dividend()]))
    inputs, ledger = build_classifier_inputs(daily, actions)
    action = ledger.loc[ledger["date"].eq(pd.Timestamp("2020-01-06"))].iloc[0]
    assert action["corporate_action_status"] == ACTION_RESOLVED
    classified = classify_constituent_returns(
        inputs[
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
    result = classified.loc[classified["date"].eq(pd.Timestamp("2020-01-06"))].iloc[0]
    assert result["constituent_return_state"] == ConstituentReturnState.TRADED_VALID.value
    assert result["daily_total_shareholder_return"] == pytest.approx(0.04)


def test_same_day_implementation_announcement_fails_closed() -> None:
    daily = _two_day_daily()
    actions = prepare_dividend_actions(
        pd.DataFrame([_dividend(imp_ann_date="20200106")])
    )
    _, ledger = build_classifier_inputs(daily, actions)
    action = ledger.loc[ledger["date"].eq(pd.Timestamp("2020-01-06"))].iloc[0]
    assert action["corporate_action_status"] == ACTION_UNRESOLVED
    assert action["action_resolution_reason"] == "IMPLEMENTATION_TERMS_NOT_STRICTLY_PRIOR"


def test_conflicting_duplicate_action_terms_fail_closed() -> None:
    actions = prepare_dividend_actions(
        pd.DataFrame([_dividend(cash_div_tax=0.2), _dividend(cash_div_tax=0.3)])
    )
    assert len(actions) == 1
    assert bool(actions.iloc[0]["conflicting_economic_terms"]) is True
    assert bool(actions.iloc[0]["action_terms_valid"]) is False


def test_exact_duplicate_action_terms_may_be_collapsed_without_conflict() -> None:
    row = _dividend()
    actions = prepare_dividend_actions(pd.DataFrame([row, row.copy()]))
    assert len(actions) == 1
    assert bool(actions.iloc[0]["conflicting_economic_terms"]) is False
    assert int(actions.iloc[0]["exact_duplicate_count"]) == 1


def test_cross_source_price_conflict_is_not_silently_selected() -> None:
    legacy = _legacy(
        [
            {
                "con_code": "000001.SZ",
                "date": "2020-01-03",
                "pre_close": 10.0,
                "raw_close": 10.0,
                "pct_chg": 0.0,
            }
        ]
    )
    fresh = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20200103"],
            "open": [10.1],
            "high": [10.1],
            "low": [10.1],
            "close": [10.1],
            "pre_close": [10.1],
            "change": [0.0],
            "pct_chg": [0.0],
            "vol": [100.0],
            "amount": [1000.0],
        }
    )
    combined = combine_daily_sources(legacy, fresh)
    assert bool(combined.iloc[0]["source_overlap_conflict"]) is True
    assert bool(combined.iloc[0]["supplier_conflict"]) is True


def test_missing_member_day_becomes_explicit_supplier_conflict_not_zero() -> None:
    symbols = [f"{value:06d}.SZ" for value in range(1, 301)]
    membership = pd.DataFrame(
        {
            "membership_date": pd.Timestamp("2020-01-06"),
            "index_code": "000300",
            "symbol": symbols,
        }
    )
    base = pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-06"),
            "symbol": symbols[:-1],
            "previous_unadjusted_close": 10.0,
            "unadjusted_close": 10.1,
            "source_observed": True,
            "supplier_conflict": False,
            "official_suspension": False,
            "suspension_evidence_id": "",
            "corporate_action_status": "NONE_CONFIRMED",
            "corporate_action_evidence_id": "DAILY_IDENTITY",
            "cash_distribution_per_pre_event_share": 0.0,
            "post_to_pre_share_ratio": 1.0,
            "subscription_cash_outflow_per_pre_event_share": 0.0,
            "daily_source": "TEST",
            "previous_observed_date": pd.Timestamp("2020-01-03"),
            "daily_pre_close": 10.0,
            "action_reference_gap_cny": 0.0,
            "action_resolution_reason": "NO_ACTION",
        }
    )
    augmented = add_explicit_missing_member_inputs(base, membership)
    missing = augmented.loc[augmented["symbol"].eq(symbols[-1])].iloc[0]
    assert bool(missing["source_observed"]) is False
    assert bool(missing["supplier_conflict"]) is True
    classified = classify_constituent_returns(
        augmented[
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
    classified_missing = classified.loc[classified["symbol"].eq(symbols[-1])].iloc[0]
    assert classified_missing["constituent_return_state"] == (
        ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value
    )
    assert pd.isna(classified_missing["daily_total_shareholder_return"])


def test_membership_requires_exact_point_in_time_300_without_backfill() -> None:
    symbols = [f"{value:06d}.SZ" for value in range(1, 301)]
    membership = pd.DataFrame(
        {
            "membership_date": pd.Timestamp("2020-01-06"),
            "index_code": "000300",
            "symbol": symbols,
        }
    )
    normalized = normalize_membership(membership)
    assert normalized["symbol"].nunique() == 300
    assert normalized["date"].nunique() == 1


def test_execution_config_keeps_performance_and_zero_fill_closed() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    assert config["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["program"]["position_impact"] == 0
    assert config["four_state_execution"]["blanket_zero_fill_allowed"] is False
    assert config["four_state_execution"]["official_exchange_suspension_source_admitted"] is False


def test_implementation_contains_no_blanket_fillna_zero() -> None:
    source = (
        ROOT / "research/stress_transmission_hazard_v2_four_state_ledger_v1.py"
    ).read_text(encoding="utf-8")
    assert ".fillna(0" not in source.replace(" ", "")


def test_frozen_execution_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["execution_id"] == config["program"]["execution_id"]
    assert manifest["return_evaluation"] == "NOT_ALLOWED"
    assert manifest["portfolio_results_read"] is False
