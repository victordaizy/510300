from __future__ import annotations

import numpy as np
import pandas as pd

from research.pit_earnings_information_diffusion_diagnostic_proxy_v1 import (
    anchored_quarterly_walk_forward,
    build_daily_outcome_blind_features,
    build_member_event_panel,
    classification_metrics,
    map_to_next_open,
    prepare_market_features_and_labels,
    select_threshold,
    validate_daily_membership,
    validate_weights,
)


def test_map_to_next_open_is_strictly_later() -> None:
    open_dates = pd.DatetimeIndex(
        pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    )
    notice = pd.Series(pd.to_datetime(["2021-01-03", "2021-01-04", "2021-01-06"]))
    actual = map_to_next_open(notice, open_dates)
    assert actual.iloc[0] == pd.Timestamp("2021-01-04")
    assert actual.iloc[1] == pd.Timestamp("2021-01-05")
    assert pd.isna(actual.iloc[2])


def test_membership_and_weight_structural_validation() -> None:
    symbols = [f"{index:06d}.SZ" for index in range(300)]
    membership = pd.DataFrame(
        {
            "membership_date": [pd.Timestamp("2021-01-04").date()] * 300,
            "index_code": ["000300"] * 300,
            "symbol": symbols,
            "source_contract": ["CSI_OFFICIAL_REPLAY"] * 300,
        }
    )
    membership_summary = validate_daily_membership(membership)
    assert membership_summary["members_min"] == 300
    assert membership_summary["duplicate_date_symbol_rows"] == 0

    weights = pd.DataFrame(
        {
            "con_code": symbols,
            "trade_date": [pd.Timestamp("2020-12-31")] * 300,
            "weight": [100.0 / 300.0] * 300,
            "source": ["proxy"] * 300,
            "retrieved_at": [pd.Timestamp("2026-08-13", tz="Asia/Shanghai")] * 300,
        }
    )
    _, weight_summary = validate_weights(weights)
    assert weight_summary["snapshots"] == 1
    assert weight_summary["revision_vintage_proven"] is False


def test_event_panel_uses_exact_membership_and_strict_prior_weight() -> None:
    facts = pd.DataFrame(
        {
            "announcement_id": ["A1"],
            "ts_code": ["000001.SZ"],
            "announcement_type": ["EARNINGS_FORECAST"],
            "official_publication_date": ["2021-01-03"],
            "known_at": ["2021-01-03T23:59:59+08:00"],
            "financial_period": ["2020-12-31"],
            "cumulative_core_np_lower_yuan": [-20.0],
            "cumulative_core_np_upper_yuan": [-10.0],
            "fact_status": ["PASS_COMPLETE_CUMULATIVE_CORE_PARENT_NET_PROFIT_FACT"],
            "first_public_status": ["PASS_FIRST_NUMERIC_CORE_EVENT"],
            "market_price_read": [False],
            "future_return_read": [False],
        }
    )
    membership = pd.DataFrame(
        {
            "membership_date": [pd.Timestamp("2021-01-04")],
            "symbol": ["000001.SZ"],
            "source_contract": ["CSI_OFFICIAL_REPLAY"],
        }
    )
    weights = pd.DataFrame(
        {
            "con_code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2020-12-31")],
            "weight": [1.5],
            "source": ["proxy"],
            "retrieved_at": [pd.Timestamp("2026-08-13", tz="Asia/Shanghai")],
        }
    )
    events, summary = build_member_event_panel(
        facts,
        membership,
        weights,
        pd.DatetimeIndex(pd.to_datetime(["2021-01-04", "2021-01-05"])),
        maximum_weight_age_days=45,
        first_public_status="PASS_FIRST_NUMERIC_CORE_EVENT",
        fact_status="PASS_COMPLETE_CUMULATIVE_CORE_PARENT_NET_PROFIT_FACT",
    )
    assert len(events) == 1
    assert bool(events.loc[0, "negative_core_fact"])
    assert bool(events.loc[0, "valid_strict_prior_weight"])
    assert events.loc[0, "weight_source_reliability"] == (
        "UNVERIFIED_HISTORICAL_WEIGHT_REVISION_VINTAGE"
    )
    assert summary["member_valid_weight_fact_count"] == 1

    daily = build_daily_outcome_blind_features(
        events, pd.Series(pd.to_datetime(["2021-01-04", "2021-01-05"]))
    )
    assert float(daily.loc[0, "negative_fact_count_5d"]) == 1.0
    assert float(daily.loc[1, "negative_fact_breadth_20d"]) == 1.0
    assert not daily["market_price_read"].any()
    assert not daily["future_return_read"].any()


def test_future_label_matches_open_to_open_dividend_contract() -> None:
    dates = pd.bdate_range("2021-01-04", periods=15)
    opens = 100.0 + np.arange(15, dtype=float)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.5,
            "volume": np.full(15, 1_000_000.0),
            "symbol": ["510300.SH"] * 15,
        }
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [dates[1]],
            "ex_date": [dates[2]],
            "payment_date": [dates[4]],
            "cash_dividend_per_share": [1.0],
        }
    )
    prepared = prepare_market_features_and_labels(market, dividends, horizon=10)
    expected = (opens[11] + 1.0) / opens[1] - 1.0
    assert np.isclose(
        prepared.loc[0, "future_10d_open_to_open_total_return"], expected
    )
    assert prepared.loc[0, "future_10d_label_end_date"] == dates[11]


def test_threshold_selection_and_confusion_metrics() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    threshold, audit = select_threshold(
        labels,
        probabilities,
        grid_start=0.1,
        grid_stop=0.9,
        grid_step=0.1,
        recall_min=0.75,
        fpr_max=0.20,
    )
    metrics = classification_metrics(labels, probabilities >= threshold)
    assert audit["status"] == "PASS_CALIBRATION_RECALL_FPR_CONSTRAINT"
    assert metrics["recall"] >= 0.75
    assert metrics["false_positive_rate"] <= 0.20


def test_walk_forward_enforces_label_end_before_next_stage() -> None:
    rng = np.random.default_rng(510300)
    dates = pd.bdate_range("2016-01-04", periods=1800)
    signal = rng.normal(size=len(dates))
    labels = signal < 0.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "future_10d_label_end_date": dates + pd.offsets.BDay(10),
            "target_10d_negative": labels,
            "baseline": rng.normal(size=len(dates)),
            "earnings": signal,
            "future_10d_open_to_open_total_return": np.where(labels, -0.01, 0.01),
            "next_day_open_to_open_total_return": rng.normal(0.0, 0.01, len(dates)),
        }
    )
    predictions, refits = anchored_quarterly_walk_forward(
        frame,
        baseline_features=["baseline"],
        augmented_features=["baseline", "earnings"],
        oos_start=pd.Timestamp("2021-01-01"),
        initial_training_observations=504,
        calibration_observations=126,
        random_seed=510300,
        threshold_contract={"start": 0.01, "stop": 0.99, "step": 0.01},
        prediction_gates={
            "oos_event_recall_min": 0.75,
            "oos_false_positive_rate_max": 0.20,
        },
    )
    assert not predictions.empty
    passed = refits.loc[refits["status"].eq("PASS_REFIT_DIAGNOSTIC_ONLY")]
    assert not passed.empty
    assert (
        pd.to_datetime(passed["training_last_label_end"])
        < pd.to_datetime(passed["calibration_start"])
    ).all()
    assert (
        pd.to_datetime(passed["calibration_last_label_end"])
        < pd.to_datetime(passed["test_start"])
    ).all()

