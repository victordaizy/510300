"""核对状态并集、未知输入、信号交接及原完整组合账户。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import normalize_dividends
from research.event_clock_account_v1 import simulate_event_account
from research.signal_union_inputs_v1 import union_targets


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=8), "open": 10., "close": 10.,
                         "previous_close": 10., "dividend": 0., "variance60": .0001})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    return data, div, cfg, cost


def run_fixture(data, div, cfg, cost, a, b):
    return simulate_event_account(data, div, cfg, cost, str(data.date.iloc[1].date()), "并集测试",
                                  targets=union_targets(a, b), event_mask=np.ones(len(data), bool))


def test_all_four_known_states_and_missing_are_distinct():
    result = union_targets([0, 0, 1, 1, np.nan, 1], [0, 1, 0, 1, 1, np.nan])
    np.testing.assert_allclose(result, [0, 1, 1, 1, np.nan, np.nan], equal_nan=True)


@pytest.mark.parametrize("bad", [2., -.1, np.inf])
def test_invalid_expert_state_fails_instead_of_clipping(bad):
    with pytest.raises(ValueError, match="专家状态"):
        union_targets([0, bad], [1, 0])


def test_future_state_changes_leave_past_targets_unchanged():
    original = union_targets([0, 1, 0, 1], [0, 0, 1, 0])
    changed = union_targets([0, 1, 0, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(original[:3], changed[:3])


def test_expert_handoff_keeps_shares_and_exits_after_both_zero():
    args = fixture()
    a = [1, 1, 0, 0, 0, 0, 0, 0]
    b = [0, 1, 1, 1, 0, 0, 0, 0]
    ledger, decisions = run_fixture(*args, a, b)
    assert ledger.iloc[0].filled_quantity > 0
    assert ledger.iloc[1:4].filled_quantity.eq(0).all()
    assert ledger.iloc[4].filled_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7
    assert decisions.iloc[4].execution_date == args[0].date.iloc[5]


def test_missing_retains_actual_shares_and_keeps_target_unknown():
    args = fixture()
    ledger, decisions = run_fixture(*args, [1, np.nan, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0])
    assert decisions.iloc[1].signal_state == "NO_VIEW_KEEP_EXISTING_SHARES"
    assert pd.isna(decisions.iloc[1].reference_weight)
    assert ledger.iloc[1].shares > 0 and ledger.iloc[1].filled_quantity == 0
    assert ledger.iloc[2].shares == 0


def test_small_dividend_cash_does_not_force_daily_addition():
    data, div, cfg, cost = fixture()
    data.loc[2:, ["open", "close"]] = 9.8
    data.loc[3:, "previous_close"] = 9.8
    data.loc[2, "dividend"] = .2
    div = pd.DataFrame([{"record_date": data.date.iloc[1], "ex_date": data.date.iloc[2],
                         "payment_date": data.date.iloc[4], "cash_dividend_per_share": .2}])
    ledger, _ = run_fixture(data, div, cfg, cost, np.ones(8), np.zeros(8))
    assert ledger.iloc[1:-1].filled_quantity.eq(0).all()
    assert ledger.dividend_recognized.sum() == pytest.approx(ledger.iloc[0].shares * .2)
    assert ledger.accounting_error.abs().max() < 1e-7


def test_blocked_exit_is_recomputed_from_latest_combined_target():
    data, div, cfg, cost = fixture()
    data.loc[3, "open"] = 9.
    ledger, decisions = run_fixture(data, div, cfg, cost, [1, 1, 0, 1, 1, 0, 0, 0], np.zeros(8))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert decisions.iloc[3].reference_weight == 1 and decisions.iloc[3].requested_quantity == 0
    assert ledger.iloc[3].shares > 0
    assert ledger.iloc[5].shares == 0


@pytest.mark.parametrize("period", ["evaluation", "earlier_diagnostic"])
@pytest.mark.parametrize("cost_id", ["BASE", "STRESS"])
def test_same_account_engine_reproduces_saved_half_allocation(period, cost_id, tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/research/510300_panic_learned_equal_blend_v1"
    cfg = json.loads((root / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(root / cfg["features"])
    if period == "earlier_diagnostic":
        data = data[data.date <= cfg["earlier_terminal"]].copy()
    state = pd.read_parquet(path / f"{period}_states.parquet")
    div = normalize_dividends(pd.read_csv(root / cfg["dividends"]))
    start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
    ledger, decisions = simulate_event_account(data, div, cfg, cfg["costs"][cost_id], start, "PANIC_LEARNED_HALF",
        targets=.5 * state.panic_state.to_numpy() + .5 * state.learned_state.to_numpy(), event_mask=np.ones(len(data), bool))
    for current, kind in [(ledger, "ledger"), (decisions, "decisions")]:
        current.to_parquet(tmp_path / f"{kind}.parquet", index=False)
        saved = pd.read_parquet(path / period / cost_id / f"PANIC_LEARNED_HALF_{kind}.parquet")
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / f"{kind}.parquet"), saved)
