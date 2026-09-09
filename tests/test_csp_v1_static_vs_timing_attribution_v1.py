"""仅合成数据：归因对照、开盘时钟、保存成交会计与统计匹配。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.conditional_score_policy_v1 import period, simulate_reference
from research.csp_v1_static_vs_timing_attribution_v1 import (
    analytic_bounds, audit_ledger, ideal_paths, joint_bootstrap, opening_inputs_valid, simulate_control,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return json.loads((ROOT / "config/510300_conditional_score_policy_v1.json").read_text(encoding="utf-8"))


def synthetic(growth=False, dividend=False):
    dates = pd.bdate_range("2020-01-01", periods=90)
    close = 4 * 1.035 ** np.arange(90) if growth else np.full(90, 4.0)
    events = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    if dividend:
        events = pd.DataFrame({"record_date": [dates[10]], "ex_date": [dates[11]],
                               "payment_date": [dates[14]], "cash_dividend_per_share": [.1]})
        close[11:] -= .1
    data = pd.DataFrame({"date": dates, "open": close, "close": close,
                         "high": close, "low": close, "volume": 10000.0,
                         "amount": 40000.0, "feature_valid": True, "dividend": 0.0})
    data["previous_close"] = data.close.shift(1).fillna(data.close.iloc[0])
    if dividend:
        data.loc[11, "dividend"] = .1
    for name in ["T", "Q", "B", "D", "R", "v", "G", "u5", "u20", "u60"]:
        data[name] = 0.0
    return period(data, events, str(dates[1].date()), str(dates[-1].date()))


def test_bounds_use_full_precision_and_legal_box():
    bounds = analytic_bounds([-.8907306367448995, .05218839774590267, .10178111370003229,
                              .004114209146956038, .18032294639427904, .02951703835204811])
    assert bounds["constant_target"] == pytest.approx(.1515984498467432, abs=1e-8)
    assert not bounds["score20_exit_reachable"] and not bounds["addition25_cap_can_bind"]
    assert bounds["target_bounds"][0] > .045 and bounds["target_bounds"][1] < .219
    lo, hi = bounds["universal_no_trade_exposure_open_interval"]
    assert lo < hi
    assert max(abs(bounds["target_bounds"][0] - (lo + hi) / 2),
               abs(bounds["target_bounds"][1] - (lo + hi) / 2)) < .1


@pytest.mark.parametrize("cost_name", ["BASE", "STRESS"])
def test_c1_same_frozen_execution_as_original_on_synthetic(cfg, cost_name):
    p = synthetic(growth=True, dividend=True)
    new, trades, _ = simulate_control(p, 30.0, "C1", cfg["costs"][cost_name], cfg)
    old, _, _ = simulate_reference(p, np.full(len(p.data), 30.0), cfg["costs"][cost_name], cfg, "FULL")
    fields = ["equity", "cash", "shares", "dividend_receivable", "net_return", "filled_quantity"]
    np.testing.assert_array_equal(new[fields].to_numpy(), old[fields].to_numpy())
    assert (new.filled_quantity < 0).any()
    assert audit_ledger(p, new, trades, cfg["costs"][cost_name], cfg)["status"] == "PASS_ACCOUNT_IDENTITIES"


def test_opening_predicate_ignores_end_of_day_fields(cfg):
    p = synthetic(growth=True)
    row = p.data.iloc[2].copy()
    row[["volume", "amount", "high", "low", "close"]] = np.nan
    assert opening_inputs_valid(row)
    original, _, _ = simulate_control(p, 30, "C1", cfg["costs"]["BASE"], cfg)
    p.data[["volume", "amount"]] = 0
    changed, _, _ = simulate_control(p, 30, "C1", cfg["costs"]["BASE"], cfg)
    pd.testing.assert_frame_equal(original, changed)


def test_c0_retains_static_shares_and_dividend_cash(cfg):
    p = synthetic(dividend=True)
    ledger, trades, decisions = simulate_control(p, 30, "C0", cfg["costs"]["BASE"], cfg)
    assert len(trades) == 1 and len(decisions) == 1
    assert ledger.shares.nunique() == 1 and ledger.shares.iloc[0] == 8300
    assert ledger.dividend_recognized.sum() == pytest.approx(830.0)
    assert ledger.dividend_paid.sum() == pytest.approx(830.0)
    assert ledger.cash.iloc[-1] - ledger.cash.iloc[0] == pytest.approx(830.0)
    audit_ledger(p, ledger, trades, cfg["costs"]["BASE"], cfg)
    broken = ledger.copy()
    broken.loc[12, "dividend_receivable"] += 10
    with pytest.raises(ValueError, match="会计核对失败"):
        audit_ledger(p, broken, trades, cfg["costs"]["BASE"], cfg)


def test_c0_partial_initial_fill_completes_build(cfg):
    p = synthetic()
    cost = {**cfg["costs"]["BASE"], "minimum": 190000.0}
    ledger, trades, _ = simulate_control(p, 30, "C0", cost, cfg)
    assert 0 < ledger.filled_quantity.iloc[0] < ledger.requested_quantity.iloc[0]
    assert (ledger.requested_quantity.iloc[1:] == 0).all() and len(trades) == 1
    audit_ledger(p, ledger, trades, cost, cfg)


@pytest.mark.parametrize("model", ["C0", "C1"])
def test_pending_open_limit_keeps_previous_close_request(cfg, model):
    p = synthetic()
    p.data.loc[1, "open"] = 4.4
    p.data.loc[1, "feature_valid"] = False
    ledger, trades, _ = simulate_control(p, 30, model, cfg["costs"]["BASE"], cfg)
    assert ledger.filled_quantity.iloc[0] == 0
    assert ledger.filled_quantity.iloc[1] == ledger.requested_quantity.iloc[0]
    assert trades.origin.iloc[0] == trades.origin.iloc[1]
    audit_ledger(p, ledger, trades, cfg["costs"]["BASE"], cfg)


def test_ideal_matching_and_paired_zero_interval(cfg):
    rng = np.random.default_rng(73)
    dates = pd.bdate_range("2020-01-01", periods=80)
    r = rng.normal(.001, .012, 80)
    ledgers = {}
    for cost in ["BASE", "STRESS"]:
        ledgers[f"BUY_HOLD_{cost}"] = pd.DataFrame({"date": dates, "net_return": r, "exposure": .9})
        ledgers[f"FULL_{cost}"] = pd.DataFrame({"date": dates, "net_return": r * .2, "exposure": .18})
    paths, matching = ideal_paths(ledgers, cfg)
    assert len(paths) == 4
    assert all(x["matching_error"] < 1e-12 for x in matching.values())
    diag = {"bootstrap_repetitions": 15, "bootstrap_block_days": 20, "bootstrap_seed": 20260906}
    same = {"A": paths["IDEAL_EXPOSURE_BASE"], "B": paths["IDEAL_EXPOSURE_BASE"].copy()}
    result, indices, _ = joint_bootstrap(same, [("A", "B")], cfg, diag)
    assert indices.shape == (15, 80)
    for item in result["paired_increments"]["A_MINUS_B"].values():
        assert item["point"] == 0 and item["interval95"] == [0, 0]
