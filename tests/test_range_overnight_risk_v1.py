"""验证区间和隔夜分别进入风险、缺失边界及下一开盘实际交易。"""
import numpy as np
import pandas as pd
import pytest
from research.range_overnight_risk_inputs_v1 import range_overnight_statistics, range_overnight_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "risk_window": 20, "annual_days": 242, "target_volatility": .10,
    "combination": "PARENT143_TIMES_RANGE_PLUS_OVERNIGHT_RISK_MULTIPLIER"}


def update_night(data):
    data["previous_close"] = data.close.shift()
    data["overnight_log"] = np.log((data.open+data.dividend)/data.previous_close)


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=80), "open": 10., "high": 10.1, "low": 9.9,
        "close": 10., "dividend": 0., "variance60": .0004})
    update_night(data)
    indices = np.arange(24, 79)
    parents = {cost: {MODELS[0]: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "origin_index": indices,
        "execution_date": data.date.iloc[indices+1].to_numpy(), "reference_weight": .8 if cost == "BASE" else .6,
        "source_cost": cost, "source_model": MODELS[0]})} for cost in CFG["costs"]}
    return data, parents, data.date.iloc[25]


def test_range_and_overnight_are_both_counted_and_zero_is_known():
    data, parents, start = fixture()
    basic = range_overnight_statistics(data, CFG)
    expected = np.log(10.1/9.9)**2/(4*np.log(2))
    assert np.isclose(basic.range_mean_variance20.iloc[24], expected)
    assert basic.overnight_variance20.iloc[24] == 0
    assert np.isclose(basic.risk_volatility20.iloc[24], np.sqrt(242*expected))
    assert 0 < basic.risk_multiplier.iloc[24] < 1
    changed = data.copy()
    changed.loc[changed.index % 2 == 0, "open"] = 10.05
    update_night(changed)
    overnight = range_overnight_statistics(changed, CFG)
    assert overnight.risk_volatility20.iloc[24] > basic.risk_volatility20.iloc[24]
    assert np.isclose(overnight.range_mean_variance20.iloc[24], basic.range_mean_variance20.iloc[24])
    data[["high", "low"]] = 10.
    flat = range_overnight_statistics(data, CFG)
    assert flat.risk_volatility20.iloc[24] == 0 and flat.risk_multiplier.iloc[24] == 1
    assert flat.risk_multiplier.iloc[:20].isna().all()


def test_dividend_adjusted_intraday_range_and_illegal_quotes():
    data, parents, start = fixture()
    data.loc[24, "dividend"] = .1
    update_night(data)
    s = range_overnight_statistics(data, CFG)
    assert np.isclose(s.adjusted_range_log.iloc[24], np.log(10.2/10.))
    assert data.overnight_log.iloc[24] > 0
    data.loc[24, "high"] = 9.
    with pytest.raises(ValueError):
        range_overnight_statistics(data, CFG)


def test_missing_risk_zero_parent_and_cost_clock_boundaries():
    data, parents, start = fixture()
    data.loc[30, "high"] = np.nan
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(30), "reference_weight"] = 0
    frames, _ = range_overnight_frames(data, parents, CFG, start)
    assert frames["BASE"].target.iloc[30:50].isna().all()
    assert np.isfinite(frames["BASE"].target.iloc[50])
    pd.testing.assert_series_equal(frames["BASE"].risk_multiplier, frames["STRESS"].risk_multiplier)
    assert frames["BASE"].target.iloc[24] != frames["STRESS"].target.iloc[24]
    parents["STRESS"][MODELS[0]]["execution_date"] = parents["STRESS"][MODELS[0]].origin
    with pytest.raises(ValueError):
        range_overnight_frames(data, parents, CFG, start)


def test_future_range_parent_and_prefix_isolation():
    data, parents, start = fixture()
    original, _ = range_overnight_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[60:, "high"] = 12.
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(60), "reference_weight"] = 0
    future, _ = range_overnight_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:60], future["BASE"].iloc[:60])
    prefix = {cost: {model: frame[frame.origin_index.lt(59)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = range_overnight_frames(data.iloc[:60], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:59], short["BASE"].iloc[:59])


def test_actual_partial_adjustment_exit_reentry_and_dividend():
    data, parents, start = fixture()
    parent = parents["BASE"][MODELS[0]]
    parent.loc[parent.origin_index.eq(30), "reference_weight"] = 0
    parent.loc[parent.origin_index.eq(35), "reference_weight"] = .2
    data.loc[28, "dividend"] = .1
    update_night(data)
    dividends = pd.DataFrame({"record_date": [data.date.iloc[27]], "ex_date": [data.date.iloc[28]],
        "payment_date": [data.date.iloc[29]], "cash_dividend_per_share": [.1]})
    frames, _ = range_overnight_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[31], "shares"] == 0 and by_date.loc[data.date.iloc[32], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[36], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[36], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[27], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum()) and ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
