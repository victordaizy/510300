"""核对平均K线初始化、分红、缺失、时点隔离及真实价格交易。"""
import numpy as np
import pandas as pd
import pytest
from research.heikin_price_state_inputs_v1 import heikin_candles, heikin_price_frames, PRIMARY, CANDIDATES, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "combination": "HEIKIN_STANDALONE_AND_REFERENCE_CONFIRMATION", "candidate_models": list(CANDIDATES), "numeric_tolerance": 1e-12}


def prices(close, dividend=None):
    close = np.asarray(close, float)
    previous = np.r_[np.nan, close[:-1]]
    opening = np.r_[close[0], close[:-1]]
    dividend = np.zeros(len(close)) if dividend is None else np.asarray(dividend, float)
    daily = np.r_[1., (close[1:]+dividend[1:])/previous[1:]]
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)), "open": opening,
        "high": np.maximum(opening, close)+.05, "low": np.minimum(opening, close)-.05, "close": close,
        "previous_close": previous, "dividend": dividend, "wealth": daily.cumprod(), "variance60": .0004})


def fixture():
    data = prices([10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.1, 9.9, 9.8, 9.7, 9.8, 10, 10.2, 10.4, 10.5, 10.6, 10.6, 10.6, 10.5, 10.4])
    indices = np.arange(2, len(data)-1)
    parents = {cost: {MODELS[0]: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": .6 if cost == "BASE" else .4, "source_cost": cost, "source_model": MODELS[0]})} for cost in CFG["costs"]}
    return data, parents, data.date.iloc[3]


def test_first_candle_and_direct_two_day_arithmetic():
    data, _, _ = fixture()
    f = heikin_candles(data)
    assert f.heikin_open.iloc[0] == 1 and f.heikin_close.iloc[0] == 1
    assert f.heikin_open.iloc[1] == 1
    assert np.isclose(f.heikin_close.iloc[1], (10+10.15+9.95+10.1)/40)
    assert np.isclose(f.heikin_open.iloc[2], (f.heikin_open.iloc[1]+f.heikin_close.iloc[1])/2)
    np.testing.assert_allclose(f.economic_close, data.wealth, atol=1e-12)


def test_ex_dividend_does_not_create_false_price_decline():
    data = prices([10, 10, 10, 9, 9, 9], [0, 0, 0, 1, 0, 0])
    data.loc[3, ["open", "high", "low"]] = [9, 9.05, 8.95]
    f = heikin_candles(data)
    np.testing.assert_allclose(f.economic_close, 1, atol=1e-12)
    assert f.heikin_direction.eq(0).all()


def test_missing_propagates_and_known_zero_parent_still_exits():
    data, parents, start = fixture()
    data.loc[6, "high"] = np.nan
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(8), "reference_weight"] = 0
    f = heikin_candles(data)
    assert pd.isna(f.heikin_close.iloc[6]) and f.heikin_open.iloc[7:].isna().all()
    frames, _ = heikin_price_frames(data, parents, CFG, start)
    assert frames["BASE"][PRIMARY+"_target"].iloc[6:-1].isna().all()
    assert frames["BASE"][list(CANDIDATES)[1]+"_target"].iloc[8] == 0
    assert pd.isna(frames["STRESS"][list(CANDIDATES)[1]+"_target"].iloc[8])


def test_illegal_prices_wealth_clock_and_cost_are_rejected():
    data, parents, start = fixture()
    for column, value in [("high", np.inf), ("low", 100.), ("wealth", 20.), ("previous_close", 1.)]:
        changed = data.copy()
        changed.loc[5, column] = value
        with pytest.raises(ValueError):
            heikin_candles(changed)
    parents["BASE"][MODELS[0]]["source_cost"] = "STRESS"
    with pytest.raises(ValueError):
        heikin_price_frames(data, parents, CFG, start)


def test_future_and_prefix_do_not_change_past_states():
    data, parents, start = fixture()
    original, _ = heikin_price_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[15:, ["open", "high", "low", "close"]] *= 2
    changed["previous_close"] = changed.close.shift()
    changed["wealth"] = np.r_[1., ((changed.close+changed.dividend)/changed.previous_close).iloc[1:]].cumprod()
    future, _ = heikin_price_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:15], future["BASE"].iloc[:15])
    prefix = {cost: {model: f[f.origin_index.lt(14)].copy() for model, f in group.items()} for cost, group in parents.items()}
    short, _ = heikin_price_frames(data.iloc[:15], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:14], short["BASE"].iloc[:14])


def test_true_price_accounts_exit_and_reenter_for_both_candidates():
    data, parents, start = fixture()
    frames, _ = heikin_price_frames(data, parents, CFG, start)
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    for model in CANDIDATES:
        target = frames["BASE"][model+"_target"].to_numpy(float)
        ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, model,
            targets=target, event_mask=np.ones(len(data), bool))
        assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 2
        by_date = ledger.set_index("date")
        zero_origins = np.flatnonzero((target == 0) & (np.r_[False, target[:-1] > 0]))
        assert all(by_date.loc[data.date.iloc[t+1], "shares"] == 0 for t in zero_origins if t+1 < len(data))
        np.testing.assert_allclose(ledger.mark.iloc[:-1], data.close.iloc[3:-1], atol=0, rtol=0)
        assert ledger.mark.iloc[-1] == data.open.iloc[-1] and ledger.shares.iloc[-1] == 0
        assert ledger.accounting_error.abs().max() < 1e-6
        assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
