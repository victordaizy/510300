"""核对相邻价方向、完整窗口、经济分红、未知状态和真实进出。"""
import math
import numpy as np
import pandas as pd
import pytest
from research.vortex_risk_inputs_v1 import economic_ohlc, vortex_factors, vortex_risk_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "candidate_models": list(CANDIDATES), "vortex_window": 14, "risk_target": .1}


def fixture():
    close = np.r_[np.linspace(10, 12, 40), np.linspace(11.95, 9, 40), np.linspace(9.05, 13, 40), np.linspace(12.95, 11, 30)]
    data = prices(close)
    data["vol20"] = .2
    return data, {cost: {} for cost in CFG["costs"]}, data.date.iloc[25]


def test_first_complete_window_and_explicit_two_direction_arithmetic():
    data, _, _ = fixture()
    f = vortex_factors(data)
    assert f.vortex_plus.iloc[:14].isna().all() and f.vortex_plus.iloc[14] > f.vortex_minus.iloc[14]
    x = economic_ohlc(data)
    positive, negative, tr = [], [], []
    for t in range(1, 15):
        positive.append(abs(x.economic_high.iloc[t]-x.economic_low.iloc[t-1]))
        negative.append(abs(x.economic_low.iloc[t]-x.economic_high.iloc[t-1]))
        tr.append(max(x.economic_high.iloc[t]-x.economic_low.iloc[t], abs(x.economic_high.iloc[t]-x.economic_close.iloc[t-1]), abs(x.economic_low.iloc[t]-x.economic_close.iloc[t-1])))
    assert f.vortex_plus.iloc[14] == pytest.approx(math.fsum(positive)/math.fsum(tr), abs=1e-12)
    assert f.vortex_minus.iloc[14] == pytest.approx(math.fsum(negative)/math.fsum(tr), abs=1e-12)
    assert f.positive_direction.iloc[65] == 0 and f.positive_direction.iloc[105] == 1


def test_dividend_has_no_false_direction_and_invalid_source_is_rejected():
    data = prices([10.]*20+[9.]*20, [0.]*20+[1.]+[0.]*19)
    data.loc[20, ["open", "high", "low"]] = [9, 9.05, 8.95]
    data["vol20"] = .2
    f = vortex_factors(data)
    np.testing.assert_allclose(f.economic_close, 1., atol=1e-12)
    assert f.positive_direction.iloc[14:].eq(0).all()
    for column, value in [("high", np.inf), ("low", 100.), ("previous_close", 1.), ("wealth", 2.)]:
        changed = data.copy(); changed.loc[25, column] = value
        with pytest.raises(ValueError):
            economic_ohlc(changed)


def test_missing_window_is_not_shortened_and_recovers_after_full_adjacent_window():
    data, parents, start = fixture()
    data.loc[30, "high"] = np.nan
    f = vortex_factors(data)
    assert f.vortex_plus.iloc[30:45].isna().all() and pd.notna(f.vortex_plus.iloc[45])
    frames, _ = vortex_risk_frames(data, parents, CFG, start)
    assert frames["BASE"][PRIMARY+"_target"].iloc[30:45].isna().all()


def test_equality_exits_without_volatility_but_positive_direction_needs_positive_known_volatility():
    flat = prices([10.]*50); flat["vol20"] = np.nan
    parents = {cost: {} for cost in CFG["costs"]}
    frames, _ = vortex_risk_frames(flat, parents, CFG, flat.date.iloc[25])
    assert frames["BASE"][PRIMARY+"_target"].iloc[24:-1].eq(0).all()
    data, parents, start = fixture()
    data.loc[25:27, "vol20"] = [np.nan, 0., .4]
    frames, _ = vortex_risk_frames(data, parents, CFG, start)
    target = frames["BASE"][PRIMARY+"_target"]
    assert target.iloc[24] == .5 and target.iloc[25:27].isna().all() and target.iloc[27] == .25
    no_range = flat.copy(); no_range[["open", "high", "low", "close"]] = 10.; no_range["vol20"] = 0.
    assert vortex_factors(no_range).positive_direction.isna().all()


def test_future_prefix_and_costs_preserve_same_known_target():
    data, parents, start = fixture()
    original, _ = vortex_risk_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[110:, ["open", "high", "low", "close"]] *= 2
    changed["previous_close"] = changed.close.shift()
    changed["wealth"] = np.r_[1., ((changed.close+changed.dividend)/changed.previous_close).iloc[1:]].cumprod()
    future, _ = vortex_risk_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:110], future["BASE"].iloc[:110])
    short, _ = vortex_risk_frames(data.iloc[:110], parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:109], short["BASE"].iloc[:109])
    np.testing.assert_allclose(original["BASE"][PRIMARY+"_target"], original["STRESS"][PRIMARY+"_target"], atol=0, rtol=0, equal_nan=True)


def test_real_next_open_exit_reentry_and_unknown_target_keeps_actual_shares():
    data, parents, start = fixture()
    data.loc[30, "vol20"] = np.nan
    frames, _ = vortex_risk_frames(data, parents, CFG, start)
    targets = frames["BASE"][PRIMARY+"_target"].to_numpy(float)
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY, targets=targets, event_mask=np.ones(len(data), bool))
    assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 2
    assert decisions[decisions.origin_index.eq(30)].iloc[0].requested_quantity == 0
    by_date = ledger.set_index("date")
    zero_origins = np.flatnonzero((targets == 0) & np.r_[False, targets[:-1] > 0])
    assert all(by_date.loc[data.date.iloc[t+1], "shares"] == 0 for t in zero_origins)
    np.testing.assert_allclose(ledger.mark.iloc[:-1], data.close.iloc[25:-1], atol=0, rtol=0)
    assert ledger.mark.iloc[-1] == data.open.iloc[-1] and ledger.shares.iloc[-1] == 0
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
