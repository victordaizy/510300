"""单日区间的股息权益、成熟窗口、费用差与真实次日开盘状态。"""
import numpy as np
import pandas as pd
import pytest
from research.observed_return_state_inputs_v1 import label_frame, estimate_schedule, economic_values, ObservedStateController, MODES
from research.observed_return_state_account_v1 import simulate_observed_state_account
from research.adaptive_allocation_v1 import Account

CFG = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "training_window": 12, "minimum_state_rows": 4}
BASE = {"commission": .0002, "minimum": 5., "slippage": .0005}
EMPTY_DIV = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])


def market(n=35):
    returns = np.where(np.arange(n)%2, .004, -.004)
    close = 10.*np.cumprod(1+returns)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "open": close, "close": close,
                         "previous_close": np.r_[10., close[:-1]], "dividend": 0., "total_simple": returns})


def fixed_model(fit=1):
    return {"fit_index": fit, "latest_exit_index": fit, "latest_maturity_index": fit, "status": "ESTIMATION_COMPLETE",
            "estimates": {"0": {"price_return": -.02, "new_dividend_yield": 0.}, "1": {"price_return": .02, "new_dividend_yield": 0.}, "pooled": {"price_return": 0., "new_dividend_yield": 0.}}}


def test_label_uses_entry_record_rights_and_delays_maturity_until_recognition():
    d = market(12)
    div = pd.DataFrame([{"record_date": d.date.iloc[4], "ex_date": d.date.iloc[6], "payment_date": d.date.iloc[8], "cash_dividend_per_share": .2}])
    labels = label_frame(d, div).set_index("origin_index")
    assert labels.loc[3, "entry_index"] == 4 and labels.loc[3, "exit_index"] == 5
    assert labels.loc[3, "maturity_index"] == 6
    assert labels.loc[3, "new_dividend_yield"] == .2/d.open.iloc[4]
    assert labels.loc[2, "new_dividend_yield"] == 0 and labels.loc[4, "new_dividend_yield"] == 0
    d.loc[7, "open"] = np.nan
    missing = label_frame(d, div)
    assert len(missing) == len(labels) and missing.loc[missing.origin_index.eq(5), "price_return"].isna().all()


def test_means_have_exact_mature_same_window_and_pooled_weighted_identity():
    d = market()
    labels = label_frame(d, EMPTY_DIV)
    estimates = estimate_schedule(d, labels, [{"fit_index": 20}], CFG)[0]
    assert estimates["status"] == "ESTIMATION_COMPLETE" and estimates["first_origin_index"] == 7 and estimates["last_origin_index"] == 18
    assert estimates["latest_maturity_index"] == 20
    rows = labels[labels.origin_index.between(7, 18)]
    for state in [0, 1]:
        selected = rows[rows.observed_state.eq(state)]
        assert estimates["counts"][str(state)] == len(selected)
        np.testing.assert_allclose(estimates["estimates"][str(state)]["price_return"], sum(selected.price_return)/len(selected), atol=1e-15)
    pooled = sum(estimates["counts"][str(s)]*estimates["estimates"][str(s)]["price_return"] for s in [0, 1])/12
    np.testing.assert_allclose(pooled, estimates["estimates"]["pooled"]["price_return"], atol=1e-15)
    late = labels.copy()
    late.loc[late.origin_index.eq(18), "maturity_index"] = 21
    assert estimate_schedule(d, late, [{"fit_index": 20}], CFG)[0]["estimates"] is None


def test_future_prices_do_not_change_past_model_and_missing_state_does_not_drop_rows():
    d = market()
    schedule = [{"fit_index": 20}]
    old = estimate_schedule(d, label_frame(d, EMPTY_DIV), schedule, CFG)
    changed = d.copy()
    changed.loc[21:, ["open", "close"]] *= 100.
    assert estimate_schedule(changed, label_frame(changed, EMPTY_DIV), schedule, CFG) == old
    changed.loc[10, "total_simple"] = np.nan
    missing = estimate_schedule(changed, label_frame(changed, EMPTY_DIV), schedule, CFG)[0]
    assert missing["rows"] == 12 and missing["estimates"] is None


def test_positive_gross_mean_can_fail_entry_cost_while_held_position_can_continue():
    prediction = {"price_return": .001, "new_dividend_yield": 0.}
    flat = economic_values(Account(1010.), 10., prediction, BASE, CFG)
    held = economic_values(Account(10., shares=100, purchase_lots=[(0, 100)]), 10., prediction, BASE, CFG)
    assert flat["economic_quantity"] == 100 and flat["estimated_entry_gain_cny"] < 0
    assert held["estimated_continue_gain_cny"] > 0


def test_unknown_state_preserves_shares_and_future_maturity_is_rejected():
    d = market()
    d.loc[6, "total_simple"] = np.nan
    account = Account(100., shares=1000, purchase_lots=[(1, 1000)])
    controller = ObservedStateController(d, [fixed_model()], BASE, CFG, MODES[0])
    row = controller(6, account)
    assert row["requested_quantity"] == 0 and pd.isna(row["reference_weight"]) and pd.isna(row["predicted_price_return"])
    future = fixed_model()
    future["latest_maturity_index"] = 2
    with pytest.raises(ValueError, match="未来"):
        ObservedStateController(d, [future], BASE, CFG, MODES[0])(5, account)


def test_next_open_entry_and_locked_failed_exit_use_only_old_inventory():
    d = market(10)
    d[["open", "close", "previous_close"]] = 10.
    d["total_simple"] = .001
    d.loc[3, ["close", "total_simple"]] = [9.9, -.01]
    d.loc[4, ["open", "close", "previous_close", "total_simple"]] = [8.9, 9.95, 9.9, .005]
    d.loc[5, ["open", "close", "previous_close"]] = [9.96, 9.96, 9.95]
    controller = ObservedStateController(d, [fixed_model()], BASE, CFG, MODES[0])
    ledger, decisions = simulate_observed_state_account(d, EMPTY_DIV, CFG, BASE, str(d.date.iloc[3].date()), MODES[0], controller)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == d.date.iloc[3]
    blocked = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
    assert blocked.filled_quantity == 0 and blocked.status == "UNFILLED_DIRECTIONAL_LIMIT"
    positive_after_failure = decisions[decisions.origin.eq(d.date.iloc[4])].iloc[0]
    assert positive_after_failure.predicted_price_return > 0 and positive_after_failure.requested_quantity < 0
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == d.date.iloc[5]
    assert ledger.accounting_error.abs().max() < 1e-6


def test_both_modes_share_availability_but_use_different_saved_mean():
    d = market()
    model = fixed_model()
    account = Account(200000.)
    conditioned = ObservedStateController(d, [model], BASE, CFG, MODES[0])(5, account)
    pooled = ObservedStateController(d, [model], BASE, CFG, MODES[1])(5, account)
    assert conditioned["estimate_fit_index"] == pooled["estimate_fit_index"] == 1
    assert conditioned["predicted_price_return"] == .02 and pooled["predicted_price_return"] == 0.
    assert conditioned["requested_quantity"] > 0 and pooled["requested_quantity"] == 0
