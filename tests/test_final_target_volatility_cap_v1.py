"""检验只在超过上限时缩小、既有风险状态与实际进出场。"""
import numpy as np
import pandas as pd
import pytest
from research.final_target_volatility_cap_inputs_v1 import capped_reference_targets, PRIMARY, PARENT, MULTIPLIER, VOLATILITY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"target_volatility": .10, "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1,
    "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def fixture(n=12):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .0004})
    risk = pd.DataFrame({"date": data.date, MULTIPLIER: .5, VOLATILITY: .2})
    indices = np.arange(1, n-1)
    references = {cost: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": .8 if cost == "BASE" else .4, "source_cost": cost, "source_model": PARENT}) for cost in CFG["costs"]}
    return data, risk, references, data.date.iloc[2]


def test_only_above_cap_is_reduced_not_all_positive_targets_multiplied():
    data, risk, references, start = fixture()
    references["BASE"].loc[0, "reference_weight"] = .2
    references["BASE"].loc[1, "reference_weight"] = .5
    references["BASE"].loc[3, "reference_weight"] = 0.
    result, summary = capped_reference_targets(data, risk, references, CFG, start, "BASE")
    assert result.target.iloc[1] == .2 and result.target.iloc[2] == .5 and result.target.iloc[3] == .5 and result.target.iloc[4] == 0.
    assert not result.cap_binding.iloc[1] and not result.cap_binding.iloc[2] and result.cap_binding.iloc[3]
    assert result.target.iloc[[0, -1]].isna().all() and summary["zero_target_origins"] == 1


def test_unknown_parent_and_existing_carried_cap_remain_distinct():
    data, risk, references, start = fixture()
    references["BASE"].loc[2, "reference_weight"] = np.nan
    risk.loc[3:5, VOLATILITY] = np.nan
    risk.loc[3:5, MULTIPLIER] = .3
    result, _ = capped_reference_targets(data, risk, references, CFG, start, "BASE")
    assert pd.isna(result.target.iloc[3]) and result.target.iloc[4] == .3
    assert result.risk_view_status.iloc[4] == "NO_NEW_VOLATILITY_VIEW_CARRIED_CAP"
    risk.loc[4, MULTIPLIER] = np.nan
    with pytest.raises(ValueError):
        capped_reference_targets(data, risk, references, CFG, start, "BASE")


def test_same_risk_cap_keeps_source_cost_and_next_open_identity():
    data, risk, references, start = fixture()
    base, _ = capped_reference_targets(data, risk, references, CFG, start, "BASE")
    stress, _ = capped_reference_targets(data, risk, references, CFG, start, "STRESS")
    assert base.target.iloc[1] == .5 and stress.target.iloc[1] == .4
    references["BASE"].loc[0, "execution_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        capped_reference_targets(data, risk, references, CFG, start, "BASE")
    references["STRESS"]["source_cost"] = "BASE"
    with pytest.raises(ValueError):
        capped_reference_targets(data, risk, references, CFG, start, "STRESS")


def test_invalid_risk_target_and_changed_parameters_are_rejected():
    data, risk, references, start = fixture()
    for invalid in [-.1, 1.1, np.inf]:
        modified = risk.copy(); modified.loc[2, MULTIPLIER] = invalid
        with pytest.raises(ValueError):
            capped_reference_targets(data, modified, references, CFG, start, "BASE")
    references["BASE"].loc[0, "reference_weight"] = 1.1
    with pytest.raises(ValueError):
        capped_reference_targets(data, risk, references, CFG, start, "BASE")
    with pytest.raises(ValueError):
        capped_reference_targets(data, risk, references, {**CFG, "target_volatility": .2}, start, "BASE")


def test_future_and_prefix_do_not_change_prior_targets():
    data, risk, references, start = fixture()
    original, _ = capped_reference_targets(data, risk, references, CFG, start, "BASE")
    new_risk = risk.copy(); new_risk.loc[7:, MULTIPLIER] = .1
    changed = {k: v.copy() for k, v in references.items()}
    changed["BASE"].loc[changed["BASE"].origin_index.ge(7), "reference_weight"] = 0.
    future, _ = capped_reference_targets(data, new_risk, changed, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], future.iloc[:7])
    prefix = {k: v[v.origin_index.lt(7)].copy() for k, v in references.items()}
    short, _ = capped_reference_targets(data.iloc[:8], risk.iloc[:8], prefix, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], short.iloc[:7])
    assert pd.isna(short.target.iloc[-1])


def test_real_partial_exit_full_exit_reentry_and_dividend_use_own_account():
    data, risk, references, start = fixture()
    risk.loc[3:4, MULTIPLIER] = .2
    references["BASE"].loc[references["BASE"].origin_index.eq(5), "reference_weight"] = 0.
    data.loc[3, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    factors, _ = capped_reference_targets(data, risk, references, CFG, start, "BASE")
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=factors.target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[4], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[4], "shares"] > 0
    assert by_date.loc[data.date.iloc[6], "shares"] == 0 and by_date.loc[data.date.iloc[7], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[2], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
