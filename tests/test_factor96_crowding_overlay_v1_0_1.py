"""检查T06同成员比较、缺失传播、成熟时钟和减半恢复状态。"""
import numpy as np
import pandas as pd

from research import intraday_overnight_increment_v1 as engine
from research.factor96_crowding_overlay_v1_0_1 import (
    financing_features, lag_features, overlay_transition, reduced_quantity, same_member_medians, simulate,
)


def panel(n=60):
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(81)
    returns = pd.DataFrame(rng.normal(0, .01, (n, 4)), index=dates, columns=list("ABCD"))
    members = pd.DataFrame(True, index=dates, columns=returns.columns)
    allowed = pd.Series(True, index=dates)
    return returns, members, allowed


def test_joint_trigger_is_not_either_condition():
    for a, b in [(False, False), (True, False), (False, True)]:
        assert overlay_transition(False, 0, True, a, b, "FULL", True) == (False, 0)
    assert overlay_transition(False, 0, True, True, True, "FULL", True) == (True, 0)
    assert overlay_transition(False, 0, True, True, True, "FULL", False) == (False, 0)


def test_both_conditions_clear_for_two_consecutive_days():
    assert overlay_transition(True, 0, True, False, True, "FULL", True) == (True, 0)
    state = overlay_transition(True, 0, True, False, False, "FULL", True)
    assert state == (True, 1)
    assert overlay_transition(*state, True, False, False, "FULL", True) == (False, 0)


def test_missing_does_not_clear_cut_or_continue_streak():
    assert overlay_transition(True, 1, False, False, False, "FULL", True) == (True, 0)
    assert overlay_transition(False, 0, False, True, True, "FULL", True) == (False, 0)


def test_single_condition_controls_have_own_restore_rule():
    assert overlay_transition(False, 0, True, True, False, "FINANCE_ONLY", True) == (True, 0)
    assert overlay_transition(True, 1, True, False, True, "FINANCE_ONLY", True) == (False, 0)
    assert overlay_transition(True, 1, True, True, False, "INTERNAL_ONLY", True) == (False, 0)


def test_half_is_of_risk_limited_shares_and_rounds_down():
    assert reduced_quantity(1300, True) == 600
    assert reduced_quantity(100, True) == 0
    assert reduced_quantity(1300, False) == 1300
    assert overlay_transition(True, 0, True, True, True, "FULL", True) == (True, 0)


def test_median_uses_identical_members_at_both_dates():
    returns, members, allowed = panel()
    members.loc[returns.index[35], "D"] = False
    members.loc[returns.index[30], "A"] = False
    result, cumulative, common = same_member_medians(returns, members, allowed, minimum=2, expected_members=3)
    assert list(common.columns[common.iloc[35]]) == ["B", "C"]
    assert result.common_members.iloc[35] == 2
    assert result.internal_known.iloc[35]
    assert np.isclose(result.median20_current.iloc[35], np.median(cumulative.loc[returns.index[35], ["B", "C"]]))
    assert np.isclose(result.median20_prior5_same_members.iloc[35], np.median(cumulative.loc[returns.index[30], ["B", "C"]]))


def test_missing_return_cannot_be_zero_filled_but_suspension_zero_is_valid():
    returns, members, allowed = panel()
    returns.loc[returns.index[25], "A"] = np.nan
    returns.loc[returns.index[25], "B"] = 0.
    result, cumulative, common = same_member_medians(returns, members, allowed, minimum=4, expected_members=4)
    assert not result.internal_known.iloc[30]
    assert not common.loc[returns.index[30], "A"] and common.loc[returns.index[30], "B"]
    assert np.isnan(cumulative.loc[returns.index[30], "A"])
    assert result.internal_known.iloc[50]


def test_internal_prefix_is_unchanged_by_future_data():
    returns, members, allowed = panel()
    full, _, _ = same_member_medians(returns, members, allowed, minimum=4, expected_members=4)
    shorter, _, _ = same_member_medians(returns.iloc[:45], members.iloc[:45], allowed.iloc[:45], minimum=4, expected_members=4)
    pd.testing.assert_frame_equal(full.iloc[:45], shorter)


def test_financing_missing_day_invalidates_six_day_window_and_threshold_excludes_today():
    dates = pd.bdate_range("2018-01-02", periods=330)
    market = pd.DataFrame({"date": dates})
    margin = pd.DataFrame({"date": dates, "market_rzye": np.exp(np.arange(330)*.001)})
    price = pd.DataFrame({"wealth": 1.-np.arange(330)*.0001})
    f = financing_features(market, margin, price)
    t = 300
    assert np.isclose(f.F5_q80.iloc[t], f.F5.iloc[t-252:t].quantile(.8))
    missing = financing_features(market, margin.drop(index=299), price)
    assert missing.F5.iloc[299:305].isna().all()
    assert missing.F5.iloc[305] == f.F5.iloc[305]


def test_lag_moves_all_economic_inputs_but_not_current_risk():
    returns, members, allowed = panel()
    internal, _, _ = same_member_medians(returns, members, allowed, minimum=4, expected_members=4)
    price = pd.DataFrame({"date": returns.index, "wealth": 1., "es95": np.arange(len(returns))*.0001,
                          "pressure5": 0., "trend20": 0., "log_rv5_rv60": 0.})
    market = price[["date"]].assign(close=3., dividend=0.)
    financing = pd.DataFrame({"date": returns.index, "F5": np.arange(len(returns)), "finance_known": True, "finance_crowded": True})
    shifted = lag_features(market, price, internal, financing, 2)
    assert shifted.stat_idx.iloc[-1] == len(returns)-3
    assert shifted.stat_date.iloc[-1] == market.date.iloc[-3]
    assert shifted.F5.iloc[-1] == financing.F5.iloc[-3]
    assert shifted.es95.iloc[-1] == price.es95.iloc[-1]


def test_simulated_cut_restores_only_after_two_clear_origins():
    dates = pd.bdate_range("2020-01-02", periods=12)
    market = pd.DataFrame({"date": dates, "open": 3., "close": 3., "previous_close": 3., "dividend": 0.})
    x = pd.DataFrame({"date": dates, "price_long": True, "common_known": True, "finance_known": True, "internal_known": True,
        "finance_crowded": False, "internal_weak": False, "es95": .03, "stat_date": dates-pd.Timedelta(days=1), "stat_idx": np.arange(12)-1,
        "F5": .01, "F5_q80": .005, "r5": -.01, "median20_change5": -.01, "common_members": 300})
    x.loc[1:2, "finance_crowded"] = True
    x.loc[1:2, "internal_weak"] = True
    x.loc[8:, "price_long"] = False
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate(market, x, dividends, "FULL", 20000, "STRESS", dates[1], dates[-1], engine)
    cut = decisions[decisions.origin.eq(dates[1])].iloc[0]
    assert cut.overlay_active and cut.filled_quantity < 0
    waiting = decisions[decisions.origin.eq(dates[3])].iloc[0]
    assert waiting.overlay_active and waiting.clear_streak == 1 and waiting.filled_quantity <= 0
    restore = decisions[decisions.origin.eq(dates[4])].iloc[0]
    assert not restore.overlay_active and restore.filled_quantity > 0 and not restore.initial_entry
    expiry = decisions[decisions.date.eq(dates[6])].iloc[0]
    assert expiry.base_exit_pending and expiry.filled_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7
    assert (ledger.loc[ledger.filled_quantity.lt(0), "filled_quantity"].abs() <= ledger.loc[ledger.filled_quantity.lt(0), "sellable_before"]).all()
