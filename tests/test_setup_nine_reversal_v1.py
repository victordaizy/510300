"""九次价格事件的必要合成测试，不读取研究收益。"""
from fractions import Fraction
import numpy as np
import pandas as pd
from research.setup_nine_reversal_inputs_v1 import economic_gross, setup_nine_factors, SetupNineController
from research.joint_entry_exit_account_v1 import simulate_joint_policy

CFG = {"comparison_days": 4, "event_count": 9}


def frame(prices):
    prices = np.asarray(prices, float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)), "close": prices,
        "previous_close": np.r_[np.nan, prices[:-1]], "dividend": 0.})


def test_exact_dividend_neutrality_and_four_day_telescope():
    assert economic_gross(9.9, 10., .1) == Fraction(1)
    data = frame([10., 10., 9.9, 9.9, 9.9, 9.9]); data.loc[2, "dividend"] = .1
    f, _ = setup_nine_factors(data, CFG)
    assert f.four_day_economic_return.iloc[:4].isna().all()
    assert f.four_day_economic_return.iloc[4:].eq(0).all()
    assert f.direction.iloc[4:].eq(0).all() and f.down_count.iloc[4:].eq(0).all()
    data = frame([100., 101., 99., 103., 102., 104.])
    f, _ = setup_nine_factors(data, CFG)
    assert f.four_day_economic_return.iloc[4] == .02
    assert f.four_day_economic_return.iloc[5] == float(Fraction(104, 101)-1)


def test_first_ninth_event_not_repeated_above_nine_and_opposite_reset():
    f, summary = setup_nine_factors(frame(np.r_[np.arange(100., 80., -1), np.arange(200., 215.)]), CFG)
    assert f.factor_status.iloc[:4].eq("NO_VIEW_WARMUP").all()
    assert f.down_count.iloc[4] == 1 and f.down_count.iloc[12] == 9 and f.down_count.iloc[13] == 10
    assert f.entry_event.iloc[12] and not f.entry_event.iloc[13]
    assert f.down_count.iloc[20] == 0 and f.up_count.iloc[20] == 1
    assert f.up_count.iloc[28] == 9 and f.exit_event.iloc[28] and not f.exit_event.iloc[29]
    assert summary["entry_events"] == 1 and summary["exit_events"] == 1


def test_exact_equality_breaks_sequence_and_new_run_starts_at_one():
    data = frame(np.r_[np.arange(100., 86., -1), 90., np.arange(80., 65., -1)])
    f, _ = setup_nine_factors(data, CFG)
    assert f.down_count.iloc[13] == 10
    assert f.direction.iloc[14] == 0 and f.down_count.iloc[14] == 0 and f.up_count.iloc[14] == 0
    assert f.down_count.iloc[15] == 1 and f.down_count.iloc[23] == 9 and f.entry_event.iloc[23]


def test_future_changes_and_prefix_preserve_prior_factors():
    data = frame(100.+np.sin(np.arange(90)/7.))
    original, _ = setup_nine_factors(data, CFG)
    altered = data.copy(); altered.loc[50:, "close"] *= 3.; altered.loc[50:, "dividend"] = .1
    changed, _ = setup_nine_factors(altered, CFG)
    prefix, _ = setup_nine_factors(data.iloc[:50], CFG)
    pd.testing.assert_frame_equal(original.iloc[:50], changed.iloc[:50])
    pd.testing.assert_frame_equal(original.iloc[:50], prefix)


def test_internal_gap_is_unknown_remainder_not_zero_or_restarted_run():
    data = frame(np.arange(100., 60., -1)); data.loc[20, "close"] = np.nan
    f, summary = setup_nine_factors(data, CFG)
    assert len(f) == len(data) and summary["status"] == "NO_VIEW_SOURCE_GAP_REMAINDER"
    assert f.factor_status.iloc[20:].eq("NO_VIEW_SOURCE_GAP_REMAINDER").all()
    assert f[["down_count", "up_count", "direction"]].iloc[20:].isna().all().all()
    assert f.entry_event.iloc[20:].isna().all()
    assert SetupNineController(f)(25, 0)["policy_action"] is None
    assert SetupNineController(f)(25, 2)["policy_action"] == 0


def test_event_controller_holds_rejects_duplicate_entry_and_preserves_locked_exit():
    f = pd.DataFrame({"down_count": [8., 9., 10., 0., 0., np.nan], "up_count": [0., 0., 0., 9., 10., np.nan],
        "factor_status": "INDICATOR_AVAILABLE"})
    c = SetupNineController(f)
    assert c(0, 0)["policy_action"] == 0 and c(1, 0)["policy_action"] == 1 and c(2, 0)["policy_action"] == 0
    assert c(1, 1)["policy_action"] == 1 and c(3, 1)["policy_action"] == 0 and c(4, 1)["policy_action"] == 1
    assert c(4, 2)["policy_action"] == 0 and c(5, 1)["policy_action"] is None and c(5, 2)["policy_action"] == 0


def test_real_account_no_retry_failed_entry_exit_lock_dividend_and_terminal():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=12), "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    data.loc[2, "open"] = 11.; data.loc[7, "open"] = 9.; data.loc[6, "dividend"] = .1
    div = pd.DataFrame({"record_date": [data.date.iloc[5]], "ex_date": [data.date.iloc[6]], "payment_date": [data.date.iloc[7]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    f = pd.DataFrame({"down_count": [0., 9., 10., 11., 9., 0., 0., np.nan, 9., 0., 0., 0.],
        "up_count": [0., 0., 0., 0., 0., 8., 9., np.nan, 0., 0., 0., 0.], "factor_status": "INDICATOR_AVAILABLE"})
    ledger, decisions, cycles = simulate_joint_policy(data, div, cfg, cost, data.date.iloc[1], SetupNineController(f))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[5], data.date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[8], data.date.iloc[11]]
    assert ledger[ledger.date.eq(data.date.iloc[2])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger[ledger.date.eq(data.date.iloc[3])].iloc[0].requested_quantity == 0
    assert ledger[ledger.date.eq(data.date.iloc[7])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert decisions[decisions.origin_index.eq(7)].iloc[0].decision_status == "LOCKED_EXIT_CONTINUES"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[5])].shares.iloc[0]*.1)
    assert len(cycles) == 2 and cycles.holding_intervals.ge(1).all()
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
