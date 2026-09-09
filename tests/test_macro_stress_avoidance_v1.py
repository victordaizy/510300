"""510300宏观压力规避V1的冻结前合成数据测试。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from research.macro_stress_avoidance_v1 import (
    _asof_join,
    _calendar_trailing_quantile,
    _complete_round_trips_by_year,
    _prepare_monthly_stress,
    apply_recovery_hysteresis,
    evaluate_event_gate,
    evaluate_portfolios,
    load_config,
)
from scripts.download_510300_macro_stress_inputs_v1 import (
    TIME_ZONE,
    _parse_pbc_publish_time,
    _parse_tsf_stock_yoy,
)


def test_calendar_quantile_requires_full_year_and_excludes_current() -> None:
    dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-06-01", "2021-12-31", "2022-01-01"]))
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    threshold, eligible = _calendar_trailing_quantile(
        dates, values, years=1, quantile=0.5
    )
    assert eligible.tolist() == [False, False, False, True]
    assert threshold.iloc[-1] == pytest.approx(2.0)


def test_calendar_quantile_starts_from_first_valid_derived_observation() -> None:
    dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-03-01", "2022-01-01", "2022-03-01"]))
    values = pd.Series([np.nan, 1.0, 2.0, 3.0])
    _, eligible = _calendar_trailing_quantile(dates, values, years=1, quantile=0.9)
    assert eligible.tolist() == [False, False, False, True]


def test_recovery_requires_five_days_per_upward_step() -> None:
    raw = pd.Series([0.0, 0.0] + [1.0] * 10)
    target, counter = apply_recovery_hysteresis(
        raw,
        confirmation_days=5,
        upward_step=0.5,
        initial_target=1.0,
    )
    assert target.iloc[0] == 0.0
    assert target.iloc[5] == 0.0
    assert target.iloc[6] == 0.5
    assert target.iloc[10] == 0.5
    assert target.iloc[11] == 1.0
    assert counter.iloc[6] == 0
    assert counter.iloc[11] == 0


def test_recovery_confirmation_resets_when_raw_band_changes() -> None:
    raw = pd.Series([0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    target, _ = apply_recovery_hysteresis(
        raw,
        confirmation_days=5,
        upward_step=0.5,
        initial_target=1.0,
    )
    assert target.iloc[2] == 0.0
    assert target.iloc[6] == 0.0
    assert target.iloc[7] == 0.5


def test_monthly_pressure_uses_exact_t_minus_three_first_release() -> None:
    frame = pd.DataFrame(
        {
            "reference_period": ["2021-01", "2021-02", "2021-03", "2021-04"],
            "available_at": [
                "2021-01-31T09:00:00+08:00",
                "2021-02-28T09:00:00+08:00",
                "2021-03-31T09:00:00+08:00",
                "2021-04-30T09:00:00+08:00",
            ],
            "first_release_value": [51.0, 50.0, 49.0, 48.0],
        }
    )
    growth = _prepare_monthly_stress(frame, kind="growth")
    credit = _prepare_monthly_stress(frame, kind="credit")
    assert pd.isna(growth.loc[2, "stress"])
    assert growth.loc[3, "value_t_minus_3"] == 51.0
    assert growth.loc[3, "stress"] == 1
    assert credit.loc[3, "stress"] == 1


def test_asof_join_rejects_release_after_signal_cutoff_until_next_day() -> None:
    panel = pd.DataFrame(
        {
            "signal_cutoff": pd.to_datetime(
                ["2026-01-05T15:00:00+08:00", "2026-01-06T15:00:00+08:00"],
                utc=True,
            ).tz_convert(TIME_ZONE)
        }
    )
    source = pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2026-01-05T15:01:00+08:00"], utc=True).tz_convert(
                TIME_ZONE
            ),
            "value": [7.5],
        }
    )
    joined = _asof_join(panel, source, prefix="source", source_columns=["value"])
    assert pd.isna(joined.loc[0, "source_value"])
    assert joined.loc[1, "source_value"] == 7.5


def test_pbc_parser_prefers_preserved_first_release_time_over_migration_metadata() -> None:
    html = """
    <html><head><meta name="createDate" content="2026-06-26 18:05:55" /></head>
    <body><!--2025年02月14日-->2025-02-14 17:00:30
    <div id="zoom">初步统计，2025年1月末社会融资规模存量为415.2万亿元，同比增长8%。</div>
    </body></html>
    """.encode("utf-8")
    parsed = _parse_pbc_publish_time(html)
    assert parsed == datetime(2025, 2, 14, 17, 0, 30, tzinfo=TIME_ZONE)
    assert _parse_tsf_stock_yoy(html) == 8.0


def test_event_spacing_uses_at_least_twenty_trading_day_ordinals() -> None:
    config = load_config()
    config["event_study"]["minimum_independent_events"] = 2
    dates = pd.bdate_range("2025-01-02", periods=170)
    score = np.zeros(len(dates), dtype=int)
    score[[10, 20, 40]] = 2
    close = np.linspace(4.0, 4.5, len(dates))
    panel = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "model_eligible_5y": True,
            "stress_score_5y": score,
            "target_position_5y": np.where(score >= 2, 0.5, 1.0),
            "fdr_liquidity_stress_5y": np.where(score >= 2, 1, 0),
            "pmi_stress": np.where(score >= 2, 1, 0),
            "tsf_stress": 0,
            "fx_fx_stress_5y": 0,
        }
    )
    dividends = pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )
    benchmark = pd.DataFrame({"date": dates, "close": np.linspace(4000, 4500, len(dates))})
    report, events, _ = evaluate_event_gate(panel, dividends, benchmark, config)
    assert report["raw_onsets"] == 3
    assert report["independent_onsets"] == 2
    assert events["date"].tolist() == [dates[10], dates[40]]


def test_round_trip_definition_counts_only_completed_cycles() -> None:
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2026-01-02", "2026-01-05"]
            ),
            "target_position": [1.0, 0.5, 1.0, 0.0, 0.5],
        }
    )
    assert _complete_round_trips_by_year(targets) == {"2025": 1}


def test_portfolio_layer_is_forbidden_when_event_gate_fails() -> None:
    with pytest.raises(PermissionError, match="事件门未通过"):
        evaluate_portfolios(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"passed": False},
            load_config(),
        )


def test_conditional_portfolio_branch_runs_on_synthetic_data() -> None:
    config = load_config()
    dates = pd.bdate_range("2024-01-02", periods=500)
    close = 4.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        }
    )
    targets = np.ones(len(dates))
    targets[100:130] = 0.5
    targets[300:330] = 0.0
    panel = pd.DataFrame(
        {
            "date": dates,
            "model_eligible_3y": True,
            "model_eligible_5y": True,
            "target_position_3y": targets,
            "target_position_5y": targets,
        }
    )
    dividends = pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )
    benchmark = pd.DataFrame(
        {"date": dates, "close": 4000.0 * np.exp(np.linspace(0.0, 0.15, len(dates)))}
    )
    report, artifacts = evaluate_portfolios(
        panel,
        market,
        dividends,
        benchmark,
        {"passed": True, "independent_complete_primary_events": 8},
        config,
    )
    assert report["five_year"]["strategy"]["observations"] == 500
    assert report["three_year"]["strategy"]["observations"] == 500
    assert not artifacts["five_year_ledger"].empty
    assert not artifacts["buy_hold_trades"].empty


def test_protocol_preserves_only_510300_cash_and_no_rescue() -> None:
    config = load_config()
    assert config["protocol"]["allowed_assets"] == ["510300.SH", "CASH_CNY"]
    assert config["protocol"]["data_start"] == "2021-01-01"
    assert config["protocol"]["no_parameter_rescue"] is True
    assert config["point_in_time"]["expanding_window_forbidden"] is True
    assert config["governance"]["live_trading_authorized"] is False
