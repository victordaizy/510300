"""日周线可用时点、限价空间、T+1和跳空损失的有针对性检验。"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("weekly_points", ROOT / "research/weekly_daily_entry_locations_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def no_dividend():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "payment_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})


def raw_weeks():
    dates = pd.bdate_range("2022-01-03", periods=35)
    lows = np.repeat([4., 3.9, 3.7, 3.8, 3.9, 4., 4.1], 5)
    return pd.DataFrame({"date": dates, "open": lows + .10, "high": lows + .15, "low": lows, "close": lows + .11, "volume": 1000.})


def test_weekly_pivot_needs_two_right_weeks_and_next_week_usage():
    d, w = M.features(raw_weeks(), no_dividend())
    assert w.new_pivot_low.iloc[:4].isna().all()
    assert w.new_pivot_low.iloc[4] == 3.7
    assert d.support_index.iloc[24] != d.support_index.iloc[24]
    assert d.support_index.iloc[25] == 3.7
    assert d.weekly_last_date.iloc[24] == pd.Timestamp("2022-01-28")
    assert d.support_confirm_date.iloc[25] == pd.Timestamp("2022-02-04")


def test_incomplete_week_changes_cannot_modify_earlier_daily_background():
    p = raw_weeks()
    full, _ = M.features(p, no_dividend())
    prefix, _ = M.features(p.iloc[:28], no_dividend())
    columns = ["support_index", "weekly_macd_hist", "weekly_ema20", "daily_macd_hist"]
    assert np.allclose(full[columns].iloc[:28], prefix[columns], equal_nan=True)
    changed = p.copy()
    changed.loc[28:, ["open", "high", "low", "close"]] += 3
    altered, _ = M.features(changed, no_dividend())
    assert np.allclose(full[columns].iloc[:28], altered[columns].iloc[:28], equal_nan=True)


def test_entry_ceiling_is_known_and_next_tick_loses_required_net_ratio():
    cap = M.entry_ceiling(3.90, 4.30)
    assert M.planned_rr(cap, 3.90, 4.30) >= 2
    assert M.planned_rr(cap + .001, 3.90, 4.30) < 2
    assert M.planned_rr(4.10, 3.90, 4.30) < 1
    assert np.isnan(M.entry_ceiling(4.30, 3.90))


def execution_fixture():
    dates = pd.bdate_range("2022-03-01", periods=5)
    d = pd.DataFrame({"date": dates, "open": [4.05, 4.08, 3.80, 3.81, 3.83], "high": [4.10, 4.09, 3.82, 3.83, 3.86], "low": [4.01, 3.94, 3.78, 3.79, 3.82], "close": [4.08, 3.96, 3.81, 3.82, 3.84], "cash_shift": 0., "dividend": 0.})
    for raw, index in [("open", "ao"), ("high", "ah"), ("low", "al"), ("close", "ac")]:
        d[index] = d[raw]
    sig = {"event_id": "TEST", "signal_idx": 0, "signal_date": dates[0], "stop_index": 3.985, "target_index": 4.50, "signal_cash_shift": 0., "max_entry_raw_at_signal": M.entry_ceiling(3.985, 4.50), "weekly_macd_rising": True, "daily_macd_rising": True, "low_volatility": True, "relative_volume_high": True}
    return sig, d


def test_entry_day_stop_waits_until_next_open_and_gap_loss_not_capped():
    sig, d = execution_fixture()
    status, rows = M.label_one(sig, d, no_dividend())
    assert status["entry_idx"] == 1 and status["decision_idx"] == 1 and status["exit_idx"] == 2
    assert status["exit_reference"] == 3.8
    assert status["exit_reason"] == "CLOSE_BELOW_STRUCTURE"
    stress = next(r for r in rows if r["cost"] == "STRESS")
    assert stress["realized_multiple_of_planned_risk"] < -2


def test_opening_gap_over_predeclared_ceiling_cannot_enter_space_policy():
    sig, d = execution_fixture()
    sig["max_entry_raw_at_signal"] = 4.0
    status, rows = M.label_one(sig, d, no_dividend())
    assert status["entry_status"] == "OPEN_PROXY_AVAILABLE"
    assert not status["space_ok"]
    frame = M.policies_from_labels(pd.DataFrame(rows))
    assert set(frame.policy) == {"P0_CONFIRM"}


def test_down_limit_delays_exit_and_dividend_entitlement_is_preserved():
    sig, d = execution_fixture()
    d.loc[2, ["open", "ao"]] = round(d.close.iloc[1] * .9, 3)
    div = pd.DataFrame({"record_date": [d.date.iloc[1]], "ex_date": [d.date.iloc[2]], "payment_date": [d.date.iloc[4]], "cash_dividend_per_share": [.01]})
    d.loc[2, "dividend"] = .01
    d.loc[2, ["open", "ao"]] = round((d.close.iloc[1] - .01) * .9, 3)
    status, rows = M.label_one(sig, d, div)
    assert status["exit_idx"] == 3
    assert all(r["dividend_per_share"] == .01 for r in rows)


def test_nonoverlap_preserves_both_event_rows_but_not_two_positions():
    f = pd.DataFrame([{"policy": "P1_SPACE", "cost": "STRESS", "event_id": "A", "entry_idx": 2, "exit_idx": 8}, {"policy": "P1_SPACE", "cost": "STRESS", "event_id": "B", "entry_idx": 5, "exit_idx": 10}, {"policy": "P1_SPACE", "cost": "STRESS", "event_id": "C", "entry_idx": 9, "exit_idx": 12}])
    out = M.nonoverlap(f)
    assert len(out) == 3
    assert out.sequential_eligible.tolist() == [True, False, True]


def test_realized_payoff_is_not_profit_factor_or_planned_ratio():
    r = M.metrics([.04, .02, -.02, -.01, -.01])
    assert r["win_rate"] == .4
    assert np.isclose(r["payoff_ratio"], 2.25)
    assert np.isclose(r["profit_factor"], 1.5)
