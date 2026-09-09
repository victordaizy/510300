"""必要合成测试：加法口径、均线起点、因果性及次日真实进出场。"""
import numpy as np
import pandas as pd
from research.negative_volume_trend_inputs_v1 import economic_return, negative_volume_factors, NegativeVolumeController
from research.joint_entry_exit_account_v1 import simulate_joint_policy

CFG = {"signal_period": 255, "initial_index": 1000.}


def frame(prices):
    prices = np.asarray(prices, float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)), "close": prices,
        "previous_close": np.r_[np.nan, prices[:-1]], "dividend": 0., "volume": 1000.-np.arange(len(prices))})


def test_dividend_neutrality_and_additive_decrease_gate_not_compounding():
    assert economic_return(9.9, 10., .1) == 0.
    assert economic_return(9.95, 10., .1) == .005
    data = frame([100., 110., 121., 133.1, 146.41])
    data.volume = [10., 9., 8., 9., 9.]
    factors, _ = negative_volume_factors(data, CFG)
    np.testing.assert_allclose(factors.nvi, [1000., 1010., 1020., 1020., 1020.], rtol=0, atol=0)
    assert pd.isna(factors.economic_return.iloc[0]) and pd.isna(factors.selected_percentage_change.iloc[0])
    assert factors.volume_decreased.iloc[1:].tolist() == [True, True, False, False]


def test_255_level_seed_includes_initial_value_then_one_over_128_update():
    data = frame(np.r_[100., np.full(254, 110.), 121.])
    factors, summary = negative_volume_factors(data, CFG)
    seed = (1000.+254*1010.)/255.
    assert factors.signal_mean.iloc[:254].isna().all()
    assert factors.signal_mean.iloc[254] == seed
    assert factors.signal_mean.iloc[255] == seed+(1020.-seed)/128.
    assert summary["available_indicator_days"] == 2
    assert NegativeVolumeController(factors)(253, 0)["policy_action"] is None


def test_future_changes_and_prefix_do_not_change_prior_factors():
    data = frame(100.+np.sin(np.arange(360)/7.))
    original, _ = negative_volume_factors(data, CFG)
    changed = data.copy(); changed.loc[280:, "volume"] *= 1000.
    updated, _ = negative_volume_factors(changed, CFG)
    prefix, _ = negative_volume_factors(data.iloc[:280], CFG)
    pd.testing.assert_frame_equal(original.iloc[:280], updated.iloc[:280])
    pd.testing.assert_frame_equal(original.iloc[:280], prefix)


def test_volume_unit_scaling_preserves_factors_and_actions():
    data = frame(100.+np.cos(np.arange(300)/7.))
    data.volume = np.tile([100., 90., 90., 110.], 75)
    original, _ = negative_volume_factors(data, CFG)
    data.volume *= 1000.
    scaled, _ = negative_volume_factors(data, CFG)
    pd.testing.assert_frame_equal(original, scaled)
    a, b = NegativeVolumeController(original), NegativeVolumeController(scaled)
    for t in range(len(data)):
        for mode in [0, 1, 2]:
            assert a(t, mode)["policy_action"] == b(t, mode)["policy_action"]


def test_observed_zero_volume_and_missing_source_remain_distinct():
    data = frame(np.full(270, 100.)); data.loc[260:261, "volume"] = 0.
    factors, _ = negative_volume_factors(data, CFG)
    assert factors.volume_decreased.iloc[260] and not factors.volume_decreased.iloc[261]
    assert factors.nvi.eq(1000.).all() and factors.signal_mean.iloc[254:].eq(1000.).all()
    data.loc[262, "volume"] = np.nan
    missing, summary = negative_volume_factors(data, CFG)
    assert missing.factor_status.iloc[262:].eq("NO_VIEW_SOURCE_GAP_REMAINDER").all()
    assert missing[["nvi", "signal_mean"]].iloc[262:].isna().all().all() and summary["observations"] == 262
    data = frame(np.full(270, 100.)); data.loc[0, "volume"] = np.nan
    invalid, _ = negative_volume_factors(data, CFG)
    assert invalid.nvi.isna().all()


def test_entry_exit_equality_missing_and_locked_exit():
    factors = pd.DataFrame({"nvi": [1001., 999., 1000., np.nan, 1001.], "signal_mean": 1000.,
        "factor_status": ["INDICATOR_AVAILABLE"]*4+["NO_VIEW_SOURCE_GAP_REMAINDER"]})
    c = NegativeVolumeController(factors)
    assert c(0, 0)["policy_action"] == 1 and c(0, 1)["policy_action"] == 1
    assert c(1, 0)["policy_action"] == 0 and c(1, 1)["policy_action"] == 0
    assert c(2, 0)["policy_action"] == 0 and c(2, 1)["policy_action"] == 1
    assert c(3, 1)["policy_action"] is None and c(3, 2)["policy_action"] == 0
    assert c(4, 0)["policy_action"] is None and c(0, 2)["policy_action"] == 0


def test_complete_account_next_open_exit_lock_reentry_dividend_and_terminal():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10), "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    data.loc[4, "open"] = 9.; data.loc[3, "dividend"] = .1
    div = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]], "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    factors = pd.DataFrame({"nvi": [999., 1001., 1001., 999., np.nan, 999., 1001., 1001., 1001., 1001.],
        "signal_mean": 1000., "factor_status": "INDICATOR_AVAILABLE"})
    ledger, decisions, cycles = simulate_joint_policy(data, div, cfg, cost, data.date.iloc[1], NegativeVolumeController(factors))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[2], data.date.iloc[7]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[5], data.date.iloc[9]]
    assert decisions[decisions.origin_index.eq(4)].iloc[0].decision_status == "LOCKED_EXIT_CONTINUES"
    assert ledger[ledger.date.eq(data.date.iloc[4])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[2])].shares.iloc[0]*.1)
    assert len(cycles) == 2 and cycles.holding_intervals.ge(1).all()
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
