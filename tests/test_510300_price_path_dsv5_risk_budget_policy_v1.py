"""真实输入读取前，验证政策边界、账户恒等式与停止门。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import price_path_dsv5_risk_budget_policy_v1 as m

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("q,expected", [(1, 1), (.001, 1), (100, .25),
    (64 / 49, .75), (64 / 25, .5), (64 / 9, .25),
    (np.nextafter(64 / 49, 0), 1), (np.nextafter(64 / 25, 0), .75),
    (np.nextafter(64 / 9, 0), .5)])
def test_policy_exact_ties_and_neighbours(q, expected):
    assert m.weight_from_ratio(q) == expected


@pytest.mark.parametrize("q", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_ratio_has_no_view(q):
    assert m.weight_from_ratio(q) is None


def test_no_view_preserves_legal_target_and_unset_initial():
    assert m.policy_targets([np.nan, 3, np.nan, 1]) == [None, .5, .5, 1]


def synthetic_market():
    dates = pd.bdate_range("2026-01-05", periods=15)
    prices = pd.DataFrame({"date": dates, "open": 4., "close": 4., "symbol": "510300.SH"})
    schedule = pd.DataFrame({"next_tradable_open": dates[[0, 5, 10]],
        "horizon_end_date": dates[[4, 9, 14]], "era_id": ["ERA_1", "ERA_2", "ERA_3"]})
    dividends = pd.DataFrame(columns=["symbol", "record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return prices, dividends, schedule


def test_commission_affordability_no_negative_cash_and_equity_identity():
    p, d, s = synthetic_market()
    l = m.simulate(p, d, s, np.array([1., .25, 1.]), 5)
    assert l.cash.min() >= 0
    assert (l.shares % 100 == 0).all()
    assert l.shares.iloc[0] == 49900
    assert l.same_day_bought_shares.iloc[0] == 49900
    assert l.old_available_shares.iloc[0] == 0
    assert l.old_available_shares.iloc[1] == 49900
    assert (l.sold_shares <= l.old_available_shares).all()
    assert np.allclose(l.equity, l.cash + l.shares * l.close + l.dividend_receivable)
    assert np.isclose(200000 - l.equity.iloc[-1], l.commission.sum() + l.slippage.sum())


def test_minimum_commission_small_account():
    p, d, s = synthetic_market()
    l = m.simulate(p, d, s.iloc[[0]], np.array([.25]), 5, capital=2000)
    assert l.shares.iloc[0] == 100
    assert l.commission.iloc[0] == 5


def test_dividend_entitlement_survives_record_date_sale_and_payment():
    p, d, s = synthetic_market()
    d = pd.DataFrame({"symbol": ["510300.SH"], "record_date": [p.date.iloc[4]],
        "ex_date": [p.date.iloc[5]], "payment_date": [p.date.iloc[8]], "cash_dividend_per_share": [.1]})
    p.loc[5:, ["open", "close"]] = 3.9
    l = m.simulate(p, d, s, np.array([1., .25, .25]), 5)
    entitled = l.shares.iloc[4] * .1
    assert l.sold_shares.iloc[5] > 0
    assert l.dividend_receivable.iloc[5] == entitled
    assert l.dividends.iloc[5] == 0
    assert l.dividends.iloc[8] == entitled
    assert l.dividend_receivable.iloc[8] == 0
    assert np.isclose(l.daily_return.iloc[5], -(l.commission.iloc[5] + l.slippage.iloc[5]) / l.equity.iloc[4])


def test_entry_after_record_date_gets_no_unearned_dividend():
    p, d, s = synthetic_market()
    d = pd.DataFrame({"symbol": ["510300.SH"], "record_date": [p.date.iloc[0] - pd.Timedelta(days=1)],
        "ex_date": [p.date.iloc[0]], "payment_date": [p.date.iloc[3]], "cash_dividend_per_share": [.1]})
    l = m.simulate(p, d, s, np.array([1., 1., 1.]), 5)
    assert l.dividends.sum() == 0
    assert l.dividend_receivable.sum() == 0


def test_missing_origin_holds_target_without_filling_forecasts():
    p, d, s = synthetic_market()
    l = m.simulate(p, d, s, np.array([.5, np.nan, 1.]), 5)
    assert l.target_weight.iloc[5] == .5
    assert l.target_weight.iloc[9] == .5
    assert l.target_weight.iloc[10] == 1


def test_same_day_two_rebalances_rejected():
    p, d, s = synthetic_market()
    s.loc[1, "next_tradable_open"] = s.loc[0, "next_tradable_open"]
    with pytest.raises(m.ContractError):
        m.simulate(p, d, s, np.array([1., .25, 1.]), 5)


def test_slippage_charged_once_and_stress_cost_increases():
    p, d, s = synthetic_market()
    a = m.simulate(p, d, s, np.array([1., .5, 1.]), 5)
    b = m.simulate(p, d, s, np.array([1., .5, 1.]), 10)
    assert b.equity.iloc[-1] < a.equity.iloc[-1]
    assert np.isclose(a.equity.iloc[-1], 200000 - a.commission.sum() - a.slippage.sum())


def test_p0_has_only_initial_buy():
    p, d, s = synthetic_market()
    one = s.iloc[[0]].copy()
    one.loc[:, "horizon_end_date"] = s.horizon_end_date.iloc[-1]
    ledger = m.simulate(p, d, one, np.array([1.]), 5)
    assert len(ledger) == len(p)
    assert (ledger.same_day_bought_shares > 0).sum() == 1
    assert ledger.sold_shares.sum() == 0


def test_g1_insufficient_variation_is_stop_not_economic_loss():
    cfg = m.load_config(ROOT)
    dates = pd.bdate_range("2021-01-04", periods=178 * 5)[::5]
    f = pd.DataFrame({"offset": 0, "origin_date": dates, "next_tradable_open": dates + pd.Timedelta(days=1),
        "horizon_end_date": dates + pd.Timedelta(days=7), "target_weight": 1., "q": 1.,
        "era_id": ["ERA_1"] * 60 + ["ERA_2"] * 59 + ["ERA_3"] * 59})
    result = m.inspect_variation(f, cfg)
    assert not result["passed"]
    assert result["non100_origins"] == 0
    assert result["average_target_weight"] == 1


def test_history_gate_failure_never_reads_market(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "verify", lambda root: ({}, {}))
    monkeypatch.setattr(m, "read_json", lambda path: {"G1": {"passed": False}, "STATE": "NO_VIEW_INSUFFICIENT_POLICY_VARIATION"})
    monkeypatch.setattr(m, "load_market", lambda *args: pytest.fail("不应读取价格"))
    with pytest.raises(m.ContractError, match="G1未通过"):
        m.run_history(tmp_path)
    assert not (tmp_path / m.REPORT / "historical_one_shot_claim.json").exists()


def test_duplicate_one_shot_claim_cannot_overwrite(tmp_path):
    path = tmp_path / "claim.json"
    m.write_new(path, {"consumed": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.write_new(path, {"consumed": False})
    assert path.read_bytes() == original


def test_parquet_projection_never_imports_b2_or_realized_dsv(monkeypatch):
    cfg = m.load_config(ROOT)
    seen = []
    def sentinel(path, *, columns, **kwargs):
        seen.extend(columns)
        raise RuntimeError("列裁剪已经设置")
    monkeypatch.setattr(pd, "read_parquet", sentinel)
    with pytest.raises(RuntimeError, match="列裁剪已经设置"):
        m.import_predictions(ROOT, cfg)
    assert "predicted_B0" in seen and "predicted_B1" in seen
    assert not set(seen) & {"predicted_B2", "DSV5", "F", "T", "T_C", "T_C_X_F"}


def test_placebo_preserves_era_weight_distribution():
    w = np.tile(np.array([.25, .5, .75, 1.]), 45)[:178]
    eras = np.array(["ERA_1"] * 60 + ["ERA_2"] * 59 + ["ERA_3"] * 59)
    out = m.permute_blocks(w, eras, np.random.default_rng(123))
    assert not np.array_equal(out, w)
    for era in np.unique(eras):
        assert np.array_equal(np.sort(out[eras == era]), np.sort(w[eras == era]))


def test_p2_is_causal_and_downside_only():
    p, d, s = synthetic_market()
    dates = pd.bdate_range("2025-01-02", periods=80)
    close = 4 * np.exp(np.cumsum(np.sin(np.arange(80)) * .01))
    p = pd.DataFrame({"date": dates, "close": close})
    a = m.p2_risk(p, d)
    p.loc[50:, "close"] *= 2
    b = m.p2_risk(p, d)
    assert np.allclose(a.iloc[:50], b.iloc[:50], equal_nan=True)
    assert a.iloc[:20].isna().all()
    assert np.isfinite(a.iloc[20:]).all()


def test_entrypoint_direct_file_from_other_cwd(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"run_{m.SLUG}.py"), "--help"],
        cwd=tmp_path, capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    assert "register-forward" in result.stdout


def test_sharpe_uses_net_simple_daily_return_and_252():
    r = np.array([.01, -.01, .02, -.005])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert m.sharpe(r) == pytest.approx(expected)
