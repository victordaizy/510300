"""核对完整交易成熟边界、收益权益及拒绝一次机会后的状态。"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research.intraday_overnight_increment_v1 import Account
from research.learned_entry_opportunity_v1 import completed_cycle_target, training_rows, fit_entry, EntryOpportunityGate
from research.learned_entry_opportunity_account_v1 import simulate_entry_opportunity
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=12), "open": 10., "close": 10.,
                         "previous_close": 10., "dividend": 0., "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(12, dtype=int), "exit": {1: np.zeros(12, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, dividends, cfg, cost, str(data.date.iloc[1].date()), rule, spec]


def model(value, t=0):
    return {"fit_index": t, "latest_exit_index": t, "fit_origin": f"测试时点{t}", "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 3, "scale": [1.] * 3, "coefficients": [0.] * 3, "feature_clip": 5., "intercept": value}}


def test_label_counts_only_rights_owned_before_actual_exit_even_if_unpaid():
    cycle = {"entry_date": "2020-01-02", "exit_date": "2020-01-06", "entry_quantity": 100, "entry_cost_cny": 1000.}
    dividends = pd.DataFrame({"record_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
                             "cash_dividend_per_share": [.5, .1, .2, .3]})
    target, rights, profit = completed_cycle_target(cycle, dividends, 1005.)
    assert rights == pytest.approx(30.)
    assert profit == pytest.approx(35.)
    assert target == pytest.approx(.035)


def test_training_excludes_whole_unfinished_cycle_and_future_extension():
    samples = pd.DataFrame({"cycle_id": range(1, 13), "exit_index": range(10, 130, 10),
                            "mom20": np.arange(12) * .01, "sma120": np.arange(12) ** 2 * .001,
                            "vol20": .1 + np.arange(12) * .01, "target": np.sin(np.arange(12)) * .1})
    cfg = {"recent_cycles": 4, "feature_clip": 5., "ridge_alpha": 1.}
    before = training_rows(samples.iloc[:8], 65, cfg)
    changed = samples.copy()
    changed.loc[changed.exit_index > 65, ["mom20", "sma120", "vol20", "target"]] = 10000.
    after = training_rows(changed, 65, cfg)
    assert list(before.cycle_id) == [3, 4, 5, 6]
    assert_frame_equal(before, after)
    assert fit_entry(before, cfg) == fit_entry(after, cfg)


def test_future_model_and_missing_inputs_preserve_missing_prediction():
    data = setup()[0]
    gate = EntryOpportunityGate(data, [model(-.02, 4)])
    result = gate(0, Account(100000.), 1, 10000)
    assert result["entry_allowed"] and result["entry_prediction"] is None
    assert result["entry_model_status"] == "NO_VIEW_NO_MATURE_MODEL"
    data.loc[5, "vol20"] = np.nan
    result = gate(5, Account(100000.), 1, 10000)
    assert result["entry_allowed"] and result["entry_prediction"] is None
    assert result["entry_model_status"] == "NO_VIEW_ENTRY_FEATURE_MISSING"


def test_rejected_opportunity_cannot_retry_when_model_turns_positive_without_signal_reset():
    args = setup()
    gate = EntryOpportunityGate(args[0], [model(-.01), model(.01, 2)])
    ledger, decisions, cycles = simulate_entry_opportunity(*args, entry_gate=gate)
    assert not (ledger.filled_quantity > 0).any()
    assert decisions.entry_model_status.notna().sum() == 1
    assert "等待" in decisions.iloc[-1].action


def test_zero_prediction_rejects_then_signal_reset_allows_a_new_opportunity():
    args = setup()
    args[5]["entry"][4] = 0
    gate = EntryOpportunityGate(args[0], [model(0.), model(.01, 2)])
    ledger, decisions, cycles = simulate_entry_opportunity(*args, entry_gate=gate)
    buys = ledger[ledger.filled_quantity > 0]
    assert len(buys) == 1 and buys.date.iloc[0] == args[0].date.iloc[6]
    assert decisions.entry_model_status.notna().sum() == 2


def test_allowed_but_unfilled_order_does_not_consume_opportunity():
    args = setup()
    args[0].loc[1, "open"] = 11.
    gate = EntryOpportunityGate(args[0], [model(.01)])
    ledger, decisions, cycles = simulate_entry_opportunity(*args, entry_gate=gate)
    assert ledger.iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger[ledger.filled_quantity > 0].date.iloc[0] == args[0].date.iloc[2]
    assert decisions.entry_model_status.notna().sum() == 2


def test_always_allowed_entry_preserves_original_account_path_and_exit():
    args = setup()
    args[5]["exit"][1][3] = True
    args[5]["entry"][5] = 0
    original, old_decisions, old_cycles = simulate_rearmed_exit(*args)
    revised, new_decisions, new_cycles = simulate_entry_opportunity(*args, entry_gate=EntryOpportunityGate(args[0], [model(.01)]))
    assert_frame_equal(original, revised)
    assert_frame_equal(old_cycles, new_cycles)
    assert (revised.filled_quantity > 0).sum() == 2
