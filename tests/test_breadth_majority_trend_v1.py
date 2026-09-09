"""检验独立广度覆盖、缺失状态和完整账户的进出场因果性。"""
import numpy as np
import pandas as pd
import pytest

from research.additional_cycle_exit_account_v1 import simulate_additional_exit
from research.breadth_majority_inputs_v1 import build_factors, rule_from_factors, MajorityExitController
from research.source_aware_cycle_account_v1 import simulate_source_aware_cycle


def inputs(n=40):
    price = np.r_[np.full(20, 10.), np.full(n - 20, 10.2)]
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "wealth": price, "close": price,
                         "open": np.r_[price[0], price[:-1]], "previous_close": np.r_[price[0], price[:-1]],
                         "dividend": 0., "feature_valid": True})
    source = pd.DataFrame({"date": data.date, "breadth20": .6, "return20_scoreable_member_count": 294,
        "return20_coverage_ratio": .98, "point_in_time_member_count": 300,
        "four_state_daily_coverage_state": "VIEW_ALLOWED", "internal_feature_state": "NO_VIEW"})
    return data, source


def account(data, rule, controller=None, spec=None, engine=simulate_source_aware_cycle):
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    spec = spec or {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return engine(data, div, cfg, cost, str(data.date.iloc[1].date()), rule, spec, controller)


def test_raw_breadth_remains_independent_of_unavailable_old_combined_factor():
    data, source = inputs()
    factors = build_factors(data, source)
    assert factors.breadth_valid.all() and factors.internal_feature_state.eq("NO_VIEW").all()
    assert factors.loc[20, "combined_entry"] == 1


@pytest.mark.parametrize("field,value", [("return20_scoreable_member_count", 293),
    ("return20_coverage_ratio", .99), ("four_state_daily_coverage_state", "NO_VIEW"), ("breadth20", np.nan)])
def test_invalid_source_cannot_enter_exit_on_breadth_or_rearm(field, value):
    data, source = inputs()
    source.loc[20, field] = value
    factors = build_factors(data, source)
    row = factors.iloc[20]
    assert not row.breadth_valid and row.breadth_state.startswith("NO_VIEW")
    assert row.combined_entry == 0 and not row.combined_rearm_allowed
    assert not MajorityExitController(factors)(20, {}, 100, 100)["breadth_exit_requested"]


def test_majority_is_strict_and_price_requires_twenty_complete_closes():
    data, source = inputs()
    source.loc[20, "breadth20"] = .5
    factors = build_factors(data, source)
    assert factors.loc[:18, "price_valid"].eq(False).all()
    assert factors.loc[19, "price_weak"] and factors.loc[20, "combined_entry"] == 0
    assert factors.loc[20, "breadth_weak"] and factors.loc[21, "combined_entry"] == 1


def test_missing_breadth_does_not_prevent_known_price_weakness():
    data, source = inputs()
    source.loc[22, "breadth20"] = np.nan
    data.loc[22, ["wealth", "close"]] = 9.9
    factors = build_factors(data, source)
    row = factors.iloc[22]
    assert not row.combined_entry_view and row.combined_rearm_allowed
    decision = MajorityExitController(factors)(22, {}, 100, 100)
    assert decision["price_exit_requested"] and not decision["breadth_exit_requested"]


def test_future_prices_and_future_breadth_do_not_change_prior_signals():
    data, source = inputs()
    prefix = build_factors(data.iloc[:25].copy(), source.iloc[:25].copy())
    data.loc[25:, "wealth"], source.loc[25:, "breadth20"] = 50., .01
    full = build_factors(data, source)
    pd.testing.assert_frame_equal(prefix, full.iloc[:25].reset_index(drop=True))


def test_absent_source_does_not_affect_price_only_ablation():
    data, source = inputs()
    good = build_factors(data, source)
    source["breadth20"] = np.nan
    missing = build_factors(data, source)
    for key in ["entry", "entry_view", "rearm_allowed"]:
        np.testing.assert_array_equal(rule_from_factors(good, False)[key], rule_from_factors(missing, False)[key])


def test_original_account_is_identical_when_new_source_inputs_are_omitted():
    data, source = inputs()
    factors = build_factors(data, source)
    rule = rule_from_factors(factors)
    old_rule = {"entry": rule["entry"], "exit": rule["exit"]}
    new = account(data, old_rule, MajorityExitController(factors))
    old = account(data, old_rule, MajorityExitController(factors), engine=simulate_additional_exit)
    for actual, expected in zip(new, old):
        pd.testing.assert_frame_equal(actual[expected.columns], expected)


def test_missing_after_forced_exit_does_not_rearm_until_known_false():
    data, _ = inputs()
    n = len(data)
    entry, view, rearm = np.ones(n, int), np.ones(n, bool), np.zeros(n, bool)
    entry[3], view[3] = 0, False
    entry[6], rearm[6] = 0, True
    rule = {"entry": entry, "entry_view": view, "rearm_allowed": rearm, "exit": {1: np.zeros(n, bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": 1}}}
    ledger, decisions, cycles = account(data, rule, spec=spec)
    assert cycles.iloc[0].entry_date == data.date.iloc[1] and cycles.iloc[0].exit_date == data.date.iloc[2]
    missing = decisions.loc[decisions.origin_index.eq(3)].iloc[0]
    assert not missing.entry_rearmed and pd.isna(missing.reference_weight) and missing.requested_quantity == 0
    assert decisions.loc[decisions.origin_index.isin([4, 5]), "requested_quantity"].eq(0).all()
    assert cycles.iloc[1].entry_date == data.date.iloc[8]
    assert ledger.accounting_error.abs().max() < 1e-7


def test_breadth_exit_is_next_open_and_blocked_request_survives_recovery_and_missing():
    data, source = inputs()
    source.loc[22, "breadth20"] = .5
    source.loc[23, "breadth20"] = np.nan
    data.loc[23, "open"] = 9.18
    factors = build_factors(data, source)
    ledger, decisions, cycles = account(data, rule_from_factors(factors), MajorityExitController(factors))
    assert cycles.iloc[0].entry_date == data.date.iloc[21]
    assert ledger.loc[ledger.date.eq(data.date.iloc[23]), "status"].iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert cycles.iloc[0].exit_date == data.date.iloc[24]
    assert "上涨比例" in cycles.iloc[0].exit_reasons
    assert decisions.loc[decisions.origin_index.eq(23), "requested_quantity"].iloc[0] < 0


def test_missing_while_holding_does_not_force_exit():
    data, source = inputs()
    source.loc[22:24, "breadth20"] = np.nan
    factors = build_factors(data, source)
    ledger, decisions, cycles = account(data, rule_from_factors(factors), MajorityExitController(factors))
    held = decisions[decisions.origin_index.between(22, 24)]
    assert held.requested_quantity.eq(0).all() and held.breadth_exit_requested.eq(False).all()
    assert ledger.loc[ledger.date.isin(data.date.iloc[22:25]), "shares"].gt(0).all()


def test_unfilled_entry_preserves_permission_but_missing_suspends_retry():
    data, source = inputs()
    data.loc[21, "open"] = 11.22
    source.loc[21, "breadth20"] = np.nan
    factors = build_factors(data, source)
    ledger, decisions, cycles = account(data, rule_from_factors(factors), MajorityExitController(factors))
    assert ledger.loc[ledger.date.eq(data.date.iloc[21]), "status"].iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    missing = decisions[decisions.origin_index.eq(21)].iloc[0]
    assert missing.entry_rearmed and missing.requested_quantity == 0 and pd.isna(missing.reference_weight)
    assert cycles.iloc[0].entry_date == data.date.iloc[23]


def test_unknown_entry_view_cannot_be_combined_with_nonzero_entry_request():
    data, source = inputs()
    rule = rule_from_factors(build_factors(data, source))
    rule["entry_view"] = rule["entry_view"].copy()
    rule["entry_view"][20] = False
    with pytest.raises(ValueError, match="无观点日"):
        account(data, rule)
