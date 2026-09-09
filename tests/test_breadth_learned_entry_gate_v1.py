"""检验广度只影响进入、未知回退和旧学习账户保持一致。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import normalize_dividends
from research.breadth_learned_gate_inputs_v1 import gate_rule
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules
from research.source_aware_cycle_account_v1 import simulate_source_aware_cycle
from tests.test_cycle_cusum_exit_v1 import account_fixture


def factors(data, valid=None, majority=None):
    return pd.DataFrame({"date": data.date, "breadth_valid": True if valid is None else valid,
                         "breadth_majority": True if majority is None else majority})


def negative_model():
    return [{"fit_index": 0, "latest_exit_index": 0, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": -.02}}]


def test_only_valid_weak_breadth_can_reject_original_entry():
    args = account_fixture()
    args[5]["entry"][3] = 0
    local = factors(args[0])
    local.loc[[1, 2, 3], "breadth_majority"] = False
    local.loc[2, "breadth_valid"] = False
    rule, trace = gate_rule(args[5], local)
    assert rule["entry"][:4].tolist() == [1, 0, 1, 0]
    assert trace.loc[2, "breadth_gate_state"].startswith("NO_VIEW")
    assert trace.loc[2, "breadth_missing_baseline_fallback"]
    assert not rule["rearm_allowed"][1] and not rule["rearm_allowed"][2] and rule["rearm_allowed"][3]
    np.testing.assert_array_equal(rule["exit"][1], args[5]["exit"][1])


def test_future_breadth_does_not_change_old_gated_rule():
    args = account_fixture()
    local = factors(args[0])
    whole, _ = gate_rule(args[5], local)
    local.loc[5:, "breadth_majority"] = False
    changed, _ = gate_rule(args[5], local)
    np.testing.assert_array_equal(whole["entry"][:5], changed["entry"][:5])


def test_rejected_entry_does_not_consume_permission_and_missing_later_uses_baseline():
    args = account_fixture()
    local = factors(args[0], majority=False)
    local.loc[2, "breadth_valid"] = False
    args[5], _ = gate_rule(args[5], local)
    ledger, decisions, cycles = simulate_source_aware_cycle(*args)
    assert cycles.iloc[0].entry_date == args[0].date.iloc[3]
    assert decisions.loc[decisions.origin_index.lt(2), "requested_quantity"].eq(0).all()
    assert decisions.loc[decisions.origin_index.eq(2), "requested_quantity"].iloc[0] > 0


def test_breadth_does_not_reset_original_rearm_after_learned_exit():
    args = account_fixture()
    args[5]["entry"][6] = 0
    local = factors(args[0])
    local.loc[3:4, "breadth_majority"] = False
    local.loc[5, "breadth_valid"] = False
    args[5], _ = gate_rule(args[5], local)
    ledger, decisions, cycles = simulate_source_aware_cycle(*args, ExitController(args[0], negative_model()))
    assert cycles.iloc[0].entry_date == args[0].date.iloc[1] and cycles.iloc[0].exit_date == args[0].date.iloc[3]
    assert decisions.loc[decisions.origin_index.between(3, 5), "entry_rearmed"].eq(False).all()
    assert cycles.iloc[1].entry_date == args[0].date.iloc[8]
    assert cycles.exit_reasons.str.contains("学习条件").iloc[0]


def test_learned_exit_remains_pending_when_breadth_recovers_or_disappears():
    args = account_fixture()
    args[0].loc[3, "open"] = 9.
    local = factors(args[0])
    local.loc[2, "breadth_majority"] = False
    local.loc[3, "breadth_valid"] = False
    args[5], _ = gate_rule(args[5], local)
    ledger, decisions, cycles = simulate_source_aware_cycle(*args, ExitController(args[0], negative_model()))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert cycles.iloc[0].exit_date == args[0].date.iloc[4]
    assert decisions.loc[decisions.origin_index.eq(3), "requested_quantity"].iloc[0] < 0


def test_disabled_filter_matches_original_engine_with_dividend_and_learning():
    args = account_fixture()
    args[0].loc[2:, ["open", "close"]] = 9.8
    args[0].loc[3:, "previous_close"] = 9.8
    args[0].loc[2, "dividend"] = .2
    args[1] = pd.DataFrame([{"record_date": args[0].date.iloc[1], "ex_date": args[0].date.iloc[2],
                            "payment_date": args[0].date.iloc[5], "cash_dividend_per_share": .2}])
    original = simulate_rearmed_exit(*args, ExitController(args[0], negative_model()))
    args[5], _ = gate_rule(args[5], factors(args[0], majority=False), enabled=False)
    replay = simulate_source_aware_cycle(*args, ExitController(args[0], negative_model()))
    for actual, expected in zip(replay, original):
        pd.testing.assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("period", ["evaluation", "earlier_diagnostic"])
@pytest.mark.parametrize("cost", ["BASE", "STRESS"])
def test_all_missing_optional_breadth_replays_saved_baseline_exactly(period, cost, tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(root / cfg["features"])
    if period == "earlier_diagnostic":
        data = data[data.date <= cfg["earlier_terminal"]].copy()
    start = cfg["earlier_start"] if period == "earlier_diagnostic" else cfg["evaluation_start"]
    models = json.loads((root / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    div = normalize_dividends(pd.read_csv(root / cfg["dividends"]))
    rule, _ = gate_rule(make_rules(data)["D60_INTRA"], factors(data, valid=False, majority=False))
    ledger, decisions, cycles = simulate_source_aware_cycle(data, div, cfg, cfg["costs"][cost], start, rule, cfg["specification"], ExitController(data, models, cfg["confirmation_days"]))
    old = root / "reports/research/510300_rearmed_session_exit_v1" / period / cost
    for actual, kind in [(ledger, "ledger"), (decisions, "decisions")]:
        saved = pd.read_parquet(old / f"REARM_RIDGE_{kind}.parquet")
        replay_file = tmp_path / f"{kind}_replay.parquet"
        actual[saved.columns].to_parquet(replay_file, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(replay_file), saved)
