"""检验过程次序、标签成熟时点与现金/T+1/容量/股息边界。"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.intraday_process_increment_v1 import (
    CONFIG, Account, add_labels, build_process, constrained_execution,
    expected_times, forecasts, simulate_account,
)


@pytest.fixture
def cfg():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_same_rebound_different_persistence_and_future_does_not_change_past(tmp_path, cfg):
    dates = pd.bdate_range("2020-01-02", periods=64)
    old = pd.DataFrame({"date": dates, "close_pressure_30m_60d": [np.nan] * 60 + [0.0] * 4})
    old.to_parquet(tmp_path / "old_c.parquet", index=False)
    cfg["inputs"]["old_c_features"] = str(tmp_path / "old_c.parquet")
    frames = []
    for date in dates:
        times = pd.to_datetime([f"{date.date()} {clock}" for clock in expected_times()])
        values = np.full(241, 5.0)
        if date == dates[-1]:
            values[26:31] = 4.98
            values[31:61] = 4.995
        opens = np.r_[5.0, values[:-1]]
        frames.append(pd.DataFrame({"date": date, "clock": times.strftime("%H:%M"), "trade_time": times, "open": opens, "high": np.maximum(opens, values), "low": np.minimum(opens, values), "close": values, "vol": 10000, "amount": values * 10000, "price_valid": True, "amount_bad": False, "strict_amount_bad": False}))
    minutes = pd.concat(frames, ignore_index=True)
    prices = pd.DataFrame({"date": dates, "open": 5., "high": 5., "low": 4.98, "close": 5., "volume": 2410000., "amount": 12000000.})
    for feature in cfg["daily_features"]:
        prices[feature] = 0.0
    stable, stable_events, _ = build_process(minutes, prices, cfg)
    changed = minutes.copy()
    selected = changed.date.eq(dates[-1])
    indexes = np.flatnonzero(selected.to_numpy())
    changed.loc[indexes[36:60], "close"] = 4.985
    vals = changed.loc[indexes, "close"].to_numpy()
    opens = np.r_[5., vals[:-1]]
    changed.loc[indexes, "open"] = opens
    changed.loc[indexes, "high"] = np.maximum(opens, vals)
    changed.loc[indexes, "low"] = np.minimum(opens, vals)
    changed.loc[indexes, "amount"] = vals * 10000
    unstable, unstable_events, _ = build_process(changed, prices, cfg)
    pd.testing.assert_frame_equal(stable.iloc[:-1], unstable.iloc[:-1])
    assert len(stable_events) == len(unstable_events) == 1
    assert stable_events.repair_5.iloc[0] == unstable_events.repair_5.iloc[0]
    assert stable_events.repair_30.iloc[0] == unstable_events.repair_30.iloc[0]
    assert stable_events.repair_persistence.iloc[0] > unstable_events.repair_persistence.iloc[0]


def test_unmatured_labels_and_future_features_cannot_change_past_forecasts(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["model"]["minimum_training_days"] = 20
    cfg["model"]["refit_interval_trading_days"] = 5
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=90)
    data = pd.DataFrame({"date": dates, "regime": "PRE_20260706", "common_valid": True, "quality_state": "AVAILABLE", "event_count": 1, "exit_h2": dates + pd.offsets.BDay(2)})
    for col in cfg["daily_features"] + sorted(set(sum(cfg["feature_groups"].values(), []))):
        data[col] = rng.normal(size=len(data))
    for horizon in (1, 2, 5):
        data[f"return_h{horizon}"] = rng.normal(0, .01, len(data))
    baseline, fits = forecasts(data, cfg)
    cutoff = dates[60]
    altered = data.copy()
    altered.loc[altered.exit_h2.gt(cutoff), "return_h2"] = 5.0
    altered.loc[altered.date.gt(cutoff), cfg["daily_features"]] = 9999.
    candidate, _ = forecasts(altered, cfg)
    pd.testing.assert_frame_equal(baseline.loc[baseline.date.le(cutoff)].reset_index(drop=True), candidate.loc[candidate.date.le(cutoff)].drop(columns=["return_h2"]).assign(return_h2=baseline.loc[baseline.date.le(cutoff), "return_h2"].to_numpy()).reindex(columns=baseline.columns).reset_index(drop=True))
    assert all(pd.Timestamp(f["last_training_label_exit"]) <= f["fit_date"] for f in fits)


def test_capacity_partial_fill_and_t_plus_one(cfg):
    account = Account(10000.)
    cost = cfg["costs"]["BASE"]
    buy = constrained_execution(account, 1000, 4., 4., 0., 1000., 1, cost, cfg)
    assert buy["requested_quantity"] == 1000
    assert buy["filled_quantity"] == 100
    assert buy["status"] == "PARTIALLY_FILLED_CONSTRAINT"
    same_day = constrained_execution(account, -100, 4., 4., 0., 10000., 1, cost, cfg)
    assert same_day["filled_quantity"] == 0
    assert account.shares == 100
    next_day = constrained_execution(account, -100, 4., 4., 0., 10000., 2, cost, cfg)
    assert next_day["filled_quantity"] == -100
    assert account.shares == 0


def test_fixed_exit_failure_keeps_inventory_and_dividend_until_payment(cfg):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    prices = pd.DataFrame({"date": dates, "open": [4., 4., 4.1, 4.2, 4.2], "close": [4., 4., 4.1, 4.2, 4.2], "previous_close": [4., 4., 4., 4.1, 4.2], "dividend": [0., 0., .1, 0., 0.]})
    minutes = pd.DataFrame([{"date": d, "clock": t, "vol": 0. if d == dates[2] and t == "15:00" else 1000., "price_valid": True, "amount_bad": False} for d in dates for t in ["09:30", "15:00"]])
    pred = pd.DataFrame({"date": dates, "prediction_D": [.1, np.nan, np.nan, np.nan, np.nan]})
    dividends = pd.DataFrame({"record_date": [dates[1]], "ex_date": [dates[2]], "payment_date": [dates[4]], "cash_dividend_per_share": [.1]})
    ledger, orders, decisions, cycles = simulate_account(prices, dividends, minutes, pred, 20000, "BASE", "D", cfg)
    assert len(ledger) == 4
    assert ledger.loc[ledger.date.eq(dates[2]), "shares"].iloc[0] == 100
    assert "UNFILLED_NO_VOLUME" in orders.status.tolist()
    assert cycles.exit_date.iloc[0] == dates[3]
    assert cycles.dividend.iloc[0] == pytest.approx(10.)
    assert ledger.equity.iloc[-1] == pytest.approx(20019.5)
    assert ledger.cash.iloc[-1] == pytest.approx(20019.5)
    assert ledger.receivable.iloc[-1] == 0
    assert ledger.accounting_error.abs().max() < 1e-7


def test_label_dividend_requires_record_date_hold_at_close():
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame({"date": dates, "open": 4., "close": 4.})
    process = pd.DataFrame({"date": [dates[0]]})
    divs = pd.DataFrame({"record_date": [dates[1], dates[2]], "ex_date": [dates[2], dates[3]], "payment_date": [dates[3], dates[4]], "cash_dividend_per_share": [.1, .2]})
    labels = add_labels(process, prices, divs)
    assert labels.return_h2.iloc[0] == pytest.approx(.025)
    assert labels.return_h1.iloc[0] == 0
