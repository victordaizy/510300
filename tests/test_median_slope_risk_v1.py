"""检查全配对中位数、完整窗口、分红、因果目标和真实进出场。"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import theilslopes
from research.median_slope_risk_inputs_v1 import median_slope_factors, median_slope_risk_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "candidate_models": list(CANDIDATES), "slope_window": 60, "volatility_window": 20, "risk_target": .1}


def fixture():
    close = np.r_[np.linspace(10, 12, 85), np.linspace(11.98, 9, 85), np.linspace(9.02, 13, 85)]
    data = prices(close)
    data["vol20"] = .2
    return data, {cost: {} for cost in CFG["costs"]}, data.date.iloc[65]


def test_all_pair_median_known_lines_and_single_extreme_observation():
    for slope in [.002, -.002, 0.]:
        data = prices(10 * np.exp(slope * np.arange(130)))
        data["vol20"] = .2
        factors = median_slope_factors(data)
        assert factors.median_pair_slope60.iloc[:59].isna().all()
        np.testing.assert_allclose(factors.median_pair_slope60.iloc[59:], slope, atol=1e-14, rtol=0)
        data.loc[59, "wealth"] *= np.exp(20.)
        got = median_slope_factors(data).median_pair_slope60.iloc[59]
        assert got == pytest.approx(slope, abs=1e-14)
        assert got == pytest.approx(theilslopes(np.log(data.wealth.iloc[:60]), np.arange(60)).slope, abs=1e-14)


def test_missing_window_recovers_only_after_sixty_complete_rows_and_invalid_inputs_stop():
    data, parents, start = fixture()
    data.loc[70, "wealth"] = np.nan
    factors = median_slope_factors(data)
    assert factors.median_pair_slope60.iloc[70:130].isna().all() and pd.notna(factors.median_pair_slope60.iloc[130])
    frames, _ = median_slope_risk_frames(data, parents, CFG, start)
    assert frames["BASE"][PRIMARY + "_target"].iloc[70:130].isna().all()
    for column, value in [("wealth", 0.), ("wealth", np.inf), ("vol20", -.1), ("vol20", np.inf)]:
        changed = data.copy()
        changed.loc[100, column] = value
        with pytest.raises(ValueError):
            median_slope_factors(changed)
    changed = data.copy()
    changed.loc[100, "date"] = changed.date.iloc[99]
    with pytest.raises(ValueError):
        median_slope_factors(changed)


def test_positive_risk_target_cap_and_known_nonpositive_direction_exit():
    data, parents, start = fixture()
    data.loc[70:73, "vol20"] = [np.nan, 0., .4, .05]
    frames, _ = median_slope_risk_frames(data, parents, CFG, start)
    target = frames["BASE"][PRIMARY + "_target"]
    assert target.iloc[69] == .5 and target.iloc[70:72].isna().all()
    assert target.iloc[72] == .25 and target.iloc[73] == 1.
    assert target.iloc[155] == 0.
    flat = prices(np.full(90, 10.))
    flat["vol20"] = np.nan
    frames, _ = median_slope_risk_frames(flat, parents, CFG, flat.date.iloc[65])
    assert frames["BASE"][PRIMARY + "_target"].iloc[64:-1].eq(0).all()


def test_cash_dividend_does_not_create_false_negative_slope():
    raw = np.r_[np.full(70, 10.), np.full(40, 9.)]
    dividends = np.zeros(len(raw))
    dividends[70] = 1.
    data = prices(raw, dividends)
    data["vol20"] = .2
    factors = median_slope_factors(data)
    np.testing.assert_allclose(factors.log_wealth, 0., atol=0, rtol=0)
    assert factors.median_pair_slope60.iloc[59:].eq(0).all()


def test_future_prefix_decision_clock_and_both_costs_have_identical_factors():
    data, parents, start = fixture()
    original, _ = median_slope_risk_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[180:, "wealth"] *= 2.
    future, _ = median_slope_risk_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:180], future["BASE"].iloc[:180])
    short, _ = median_slope_risk_frames(data.iloc[:180], parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:179], short["BASE"].iloc[:179])
    pd.testing.assert_frame_equal(original["BASE"].drop(columns="source_cost"), original["STRESS"].drop(columns="source_cost"))
    assert original["BASE"][PRIMARY + "_target"].iloc[:64].isna().all() and pd.isna(original["BASE"][PRIMARY + "_target"].iloc[-1])
    expected = pd.DatetimeIndex(data.date.iloc[64:-1]) + pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original["BASE"].decision_time.iloc[64:-1]).equals(expected)


def test_real_open_entry_exit_unknown_holding_and_dividend_rights():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy()
    raw[72:] -= .1
    dividend = np.zeros(len(raw)); dividend[72] = .1
    data = prices(raw, dividend)
    data.loc[72, "open"] = raw[71] - .1
    data["vol20"] = .2
    data.loc[70, "vol20"] = np.nan
    frames, _ = median_slope_risk_frames(data, parents, CFG, start)
    target = frames["BASE"][PRIMARY + "_target"].to_numpy(float)
    rights = pd.DataFrame({"record_date": [data.date.iloc[71]], "ex_date": [data.date.iloc[72]],
        "payment_date": [data.date.iloc[75]], "cash_dividend_per_share": [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 2
    assert decisions.loc[decisions.origin_index.eq(70), "requested_quantity"].iloc[0] == 0
    assert by_date.loc[data.date.iloc[71], "shares"] > 0
    earned = by_date.loc[data.date.iloc[71], "shares"] * .1
    assert by_date.loc[data.date.iloc[72], "dividend_recognized"] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[75], "dividend_paid"] == pytest.approx(earned)
    zero_origins = np.flatnonzero((target == 0) & np.r_[False, target[:-1] > 0])
    assert len(zero_origins) > 0 and all(by_date.loc[data.date.iloc[t + 1], "shares"] == 0 for t in zero_origins)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
