"""核对边界请求、未知与零、分红权益、受阻退出和时间因果性。"""
import numpy as np
import pandas as pd
import pytest
from research.adaptive_allocation_v1 import Account, target_request
from research.boundary_rebalance_inputs_v1 import boundary_request, boundary_rebalance_frames, PRIMARY, MODELS, CANDIDATES
from research.boundary_rebalance_account_v1 import simulate_boundary_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "candidate_models": list(CANDIDATES), "parent_models": MODELS, "boundary_equality_tolerance": 1e-12}


def account(weight, nav=100000., close=10.):
    shares = int(weight*nav/close)
    return Account(cash=nav-shares*close, shares=shares, purchase_lots=[(0, shares)])


def test_both_sides_trade_to_nearest_boundary_instead_of_center():
    low = account(.2)
    got = boundary_request(low, 10., .6, CFG)
    assert got["rebalance_weight"] == .5 and got["requested_quantity"] == 3000
    assert target_request(low, 10., .6, CFG)["requested_quantity"] == 4000
    high = account(.8)
    got = boundary_request(high, 10., .4, CFG)
    assert got["rebalance_weight"] == .5 and got["requested_quantity"] == -3000
    assert target_request(high, 10., .4, CFG)["requested_quantity"] == -4000
    assert got["reference_weight"] == .4


def test_exact_boundaries_and_roundoff_hold_without_double_filtering():
    for weight in [.4, .5, .6]:
        got = boundary_request(account(weight), 10., .5, CFG)
        assert got["requested_quantity"] == 0 and got["request_rule"] == "HOLD_INSIDE_INTERVAL"
    got = boundary_request(account(.35), 10., .5, CFG)
    assert got["requested_quantity"] == 500 and got["rebalance_weight"] == .4
    # 边界修正只有五个百分点，不能被旧的十个百分点阈值再次阻拦。
    assert target_request(account(.35), 10., .4, CFG)["requested_quantity"] == 0
    near = Account(cash=40000.-5e-8, shares=6000)
    assert boundary_request(near, 10., .5, CFG)["requested_quantity"] == 0


def test_empty_entry_zero_exit_unknown_and_receivable_use_own_nav():
    flat = Account(100000.)
    assert boundary_request(flat, 10., .05, CFG)["requested_quantity"] == 500
    held = account(.05)
    assert boundary_request(held, 10., 0., CFG)["requested_quantity"] == -500
    got = boundary_request(held, 10., np.nan, CFG)
    assert got["requested_quantity"] == 0 and np.isnan(got["reference_weight"])
    with pytest.raises(ValueError):
        boundary_request(held, 10., np.inf, CFG)
    rights = Account(50000., 5000, {0: 20000.}, {}, [(0, 5000)])
    got = boundary_request(rights, 10., .8, CFG)
    assert got["request_origin_nav"] == 120000. and got["requested_quantity"] == 3400


def fixture():
    dates = pd.bdate_range("2020-01-01", periods=12)
    close = np.r_[np.full(5, 10.), np.full(7, 9.9)]
    data = pd.DataFrame({"date": dates, "open": close, "close": close, "previous_close": np.r_[10., close[:-1]],
                         "dividend": np.r_[np.zeros(5), .1, np.zeros(6)], "variance60": .0004})
    data.loc[7, "open"] = 8.91
    dividends = pd.DataFrame({"record_date": [dates[4]], "ex_date": [dates[5]], "payment_date": [dates[7]], "cash_dividend_per_share": [.1]})
    target = np.array([np.nan, .2, .8, .3, .3, np.nan, 0., 0., .4, .4, .4, np.nan])
    return data, dividends, target


def test_parent_cost_identity_complete_clock_and_unknown_are_preserved():
    data, _, target = fixture()
    indices = np.arange(1, 11)
    parents = {cost: {MODELS[0]: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(),
        "execution_date": data.date.iloc[indices+1].to_numpy(), "origin_index": indices,
        "reference_weight": target[indices], "source_cost": cost, "source_model": MODELS[0]})} for cost in CFG["costs"]}
    frames, _ = boundary_rebalance_frames(data, parents, CFG, data.date.iloc[2])
    np.testing.assert_allclose(frames["BASE"][PRIMARY+"_target"], target, atol=0, rtol=0, equal_nan=True)
    parents["BASE"][MODELS[0]]["source_cost"] = "STRESS"
    with pytest.raises(ValueError):
        boundary_rebalance_frames(data, parents, CFG, data.date.iloc[2])


def test_actual_open_boundary_orders_blocked_full_exit_reentry_and_dividend_rights():
    data, dividends, target = fixture()
    ledger, decisions = simulate_boundary_account(data, dividends, CFG, CFG["costs"]["BASE"], data.date.iloc[2], PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[7], "status"] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert by_date.loc[data.date.iloc[8], "shares"] == 0 and by_date.loc[data.date.iloc[9], "filled_quantity"] > 0
    assert decisions[decisions.origin_index.eq(5)].iloc[0].requested_quantity == 0
    assert by_date.loc[data.date.iloc[5], "dividend_recognized"] == pytest.approx(by_date.loc[data.date.iloc[4], "shares"]*.1)
    assert by_date.loc[data.date.iloc[7], "dividend_paid"] == pytest.approx(by_date.loc[data.date.iloc[5], "dividend_recognized"])
    assert decisions.request_rule.eq("LOWER_BOUNDARY").any() and decisions.request_rule.eq("UPPER_BOUNDARY").any()
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))


def test_future_targets_and_terminal_prefix_do_not_change_earlier_real_accounts():
    data, dividends, target = fixture()
    def run(frame, goals):
        return simulate_boundary_account(frame, dividends, CFG, CFG["costs"]["BASE"], data.date.iloc[2], PRIMARY,
            targets=goals, event_mask=np.ones(len(frame), bool))
    full, decisions = run(data, target)
    changed = target.copy(); changed[9:11] = .9
    future, _ = run(data, changed)
    pd.testing.assert_frame_equal(full.iloc[:7], future.iloc[:7])
    short, short_decisions = run(data.iloc[:9], target[:9])
    pd.testing.assert_frame_equal(full.iloc[:6], short.iloc[:-1])
    pd.testing.assert_frame_equal(decisions.iloc[:7], short_decisions)
    with pytest.raises(ValueError, match="本入口仅支持边界目标账户"):
        simulate_boundary_account(data, dividends, CFG, CFG["costs"]["BASE"], data.date.iloc[2], "BUY_HOLD", targets=target, event_mask=np.ones(len(data), bool))
