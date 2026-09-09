"""核对跨时区、缺项、不偷看未来和真实进出场。"""
import numpy as np
import pandas as pd
import pytest

from research.volatility_term_inputs_v1 import paired_sources, preopen_frame
from research.event_clock_account_v1 import simulate_event_account


def raw(dates, values):
    return pd.DataFrame({"DATE": pd.to_datetime(dates).strftime("%m/%d/%Y"), "CLOSE": np.asarray(values, float)})


def pair(dates, month, short):
    return paired_sources(raw(dates, month), raw(dates, short), "2026-08-14")


def test_winter_and_summer_completion_times():
    result = pair(["2020-01-02", "2020-07-02"], [20, 20], [15, 15])
    assert result.available_at.tolist() == [pd.Timestamp("2020-01-03 06:00"), pd.Timestamp("2020-07-03 05:00")]


def test_next_morning_includes_overnight_but_excludes_future_us_close():
    result = preopen_frame(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]), pair(["2020-01-02", "2020-01-03"], [20, 20], [15, 25]))
    assert result.source_date.iloc[0] == pd.Timestamp("2020-01-02")
    assert result.available_at.iloc[0] > result.date.iloc[0] + pd.Timedelta(hours=15)
    assert result.target.iloc[0] == 1 and result.target.iloc[1] == 0
    assert result.decision_time.iloc[0] == pd.Timestamp("2020-01-03 09:00")


def test_latest_partial_pair_does_not_backtrack_to_older_complete_pair():
    sources = paired_sources(raw(["2020-01-02", "2020-01-03", "2020-01-06"], [20, 20, 20]), raw(["2020-01-02", "2020-01-06"], [15, 15]), "2020-01-06")
    result = preopen_frame(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]), sources)
    assert len(sources) == 3 and result.source_state.iloc[1] == "MISSING_PAIR"
    assert result.source_date.iloc[1] == pd.Timestamp("2020-01-03") and np.isnan(result.target.iloc[1])
    assert result.target.iloc[2] == 1


def test_equal_volatility_exits_and_age_boundary_is_inclusive():
    sources = pair(["2020-01-02"], [20], [20])
    sources["available_at"] = pd.to_datetime(["2020-01-03 09:00"])
    result = preopen_frame(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-10", "2020-01-13", "2020-01-14"]), sources)
    assert result.target.iloc[0] == 0 and result.target.iloc[1] == 0
    assert result.source_age_days.iloc[1] == 7 and result.source_state.iloc[2] == "STALE"
    assert np.isnan(result.target.iloc[2])


def test_future_values_and_truncated_calendar_cannot_affect_earlier_decisions():
    dates = pd.bdate_range("2020-01-02", periods=8)
    sources = pair(dates, [20] * 8, [15] * 8)
    expected = preopen_frame(dates, sources)
    changed = sources.copy()
    changed.loc[changed.source_date >= "2020-01-08", "vix9d_close"] = 30
    pd.testing.assert_frame_equal(expected.iloc[:3], preopen_frame(dates, changed).iloc[:3])
    short = preopen_frame(dates[:4], sources)
    pd.testing.assert_frame_equal(expected.iloc[:3], short.iloc[:3])
    assert short.source_date.isna().iloc[-1] and short.decision_time.isna().iloc[-1]
    assert short.source_state.iloc[-1] == "NOT_USED_NO_NEXT_EXECUTION"


@pytest.mark.parametrize("kind", ["duplicate", "negative", "infinite"])
def test_invalid_sources_are_rejected(kind):
    source = raw(["2020-01-02", "2020-01-03"], [20, 20])
    if kind == "duplicate":
        source.loc[1, "DATE"] = source.DATE.iloc[0]
    else:
        source.loc[1, "CLOSE"] = -1 if kind == "negative" else np.inf
    with pytest.raises(ValueError):
        paired_sources(source, raw(["2020-01-02", "2020-01-03"], [15, 15]), "2020-01-03")


def test_before_source_stays_unknown():
    result = preopen_frame(pd.bdate_range("2019-12-30", periods=5), pair(["2020-01-02"], [20], [15]))
    assert result.source_state.iloc[0] == "BEFORE_SOURCE" and np.isnan(result.target.iloc[0])


def test_actual_account_uses_prior_close_budget_keeps_missing_day_and_reenters():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"])
    source = pair(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"], [20]*6, [15, np.nan, 25, 15, 15, 15])
    states = preopen_frame(dates, source)
    op, cl = np.array([10, 10.2, 10.2, 10.3, 10.2, 10.3, 10.4]), np.array([10, 10.2, 10.3, 10.2, 10.3, 10.4, 10.4])
    data = pd.DataFrame({"date": dates, "open": op, "close": cl, "high": np.maximum(op, cl), "low": np.minimum(op, cl),
        "previous_close": np.r_[np.nan, cl[:-1]], "dividend": 0., "variance60": .0001})
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(data, div, cfg, cost, "2020-01-03", "期限测试", targets=states.target.to_numpy(), event_mask=np.ones(len(data), bool))
    assert decisions.requested_quantity.iloc[0] == 10000 and 0 < ledger.filled_quantity.iloc[0] < 10000
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-08")]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-10")]
    assert decisions.signal_state.iloc[1] == "NO_VIEW_KEEP_EXISTING_SHARES"
    assert ledger.filled_quantity.iloc[1] == 0 and ledger.shares.iloc[1] == ledger.shares.iloc[0]
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-7
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
