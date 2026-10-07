"""检验资格隔离、因果时钟和预测与执行的不同状态。"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from research import daily_native_baseline_v1 as native
from research import intraday_process_increment_v1 as old


def configuration():
    cfg = json.loads(native.CONFIG.read_text(encoding="utf-8"))
    parent = json.loads((native.ROOT / cfg["parent_config"]).read_text(encoding="utf-8"))
    return cfg, parent


def prices(count=100):
    i = np.arange(count)
    close = 5 + .001 * i + .01 * np.sin(i / 3)
    return pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=count), "open": close - .003, "high": close + .01, "low": close - .01, "close": close, "volume": 100000, "amount": close * 100000})


def no_dividends():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "payment_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})


def labeled_panel():
    cfg, _ = configuration()
    count = 44
    data = pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=count), "D_input_valid": True, "regime": "PRE_20260706"})
    for index, field in enumerate(cfg["daily_features"]):
        data[field] = np.sin(np.arange(count) / (index + 2)) + .001 * np.arange(count)
    data["return_h2"] = .01 * np.cos(np.arange(count) / 4)
    data["exit_h2"] = data.date.shift(-2)
    data.loc[count - 2:, "return_h2"] = np.nan
    return data


def test_c_and_minute_process_failures_do_not_block_daily_inputs():
    cfg, parent = configuration()
    panel = pd.DataFrame({"date": pd.bdate_range("2026-03-09", periods=3), "daily_price_valid": True, "daily_amount_valid": True})
    for field in cfg["daily_features"]:
        panel[field] = 1.0
    process = pd.DataFrame({"date": panel.date, "process_available": [True, True, False], "B_amount_valid": [True, False, False], "C_amount_valid": False, "common_valid": False, "quality_state": "旧共同门失效"})
    for field in set(sum(parent["feature_groups"].values(), [])):
        process[field] = 1.0
    process["close_pressure_30m_60d"] = np.nan
    result = native.input_qualification(panel, process, parent)
    assert result.D_input_valid.tolist() == [True, True, True]
    assert result.A_input_valid.tolist() == [True, True, False]
    assert result.B_input_valid.tolist() == [True, False, False]
    assert not result.C_input_valid.any()
    assert not result.D_common_input_valid.any()


def test_future_daily_prices_and_optional_amount_do_not_rewrite_history():
    raw = prices()
    before = native.daily_features(raw, no_dividends())
    changed = raw.copy()
    changed.loc[70:, ["open", "high", "low", "close"]] *= 2
    changed.loc[70:, "amount"] = np.nan
    after = native.daily_features(changed, no_dividends())
    pdt.assert_frame_equal(before.iloc[:70], after.iloc[:70])
    assert after.daily_price_valid.all()
    assert not after.loc[70:, "daily_amount_valid"].any()


def test_current_label_not_required_and_future_data_cannot_change_predictions():
    cfg, parent = configuration()
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(minimum_training_days=6, refit_interval_trading_days=5)
    data = labeled_panel()
    before, models, members = native.native_forecasts(data, cfg, parent)
    changed = data.copy()
    changed.loc[30:, cfg["daily_features"]] *= 999
    changed.loc[30:, "return_h2"] = -99
    after, _, _ = native.native_forecasts(changed, cfg, parent)
    pdt.assert_frame_equal(before.iloc[:30], after.iloc[:30])
    assert before.tail(2).prediction_D.notna().all()
    assert not before.current_label_mature_at_decision.any()
    assert all(pd.Timestamp(model["last_training_label_exit"]) <= model["fit_date"] for model in models)
    assert (members.exit_h2 <= members.fit_date).all()


def test_bad_current_input_has_own_reason_and_does_not_reuse_stale_prediction():
    cfg, parent = configuration()
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(minimum_training_days=6, refit_interval_trading_days=5)
    data = labeled_panel()
    data.loc[30, "D_input_valid"] = False
    result, _, _ = native.native_forecasts(data, cfg, parent)
    assert result.loc[30, "prediction_state"] == "INPUT_MISSING_OR_BAD"
    assert pd.isna(result.loc[30, "prediction_D"])
    assert result.loc[31, "prediction_state"] == "PREDICTION_AVAILABLE"


def test_minute_price_volume_and_amount_have_separate_qualification():
    minute = pd.DataFrame({"trade_time": pd.to_datetime(["2026-03-09 09:30", "2026-03-09 15:00"]), "open": 5., "high": 5.01, "low": 4.99, "close": 5., "vol": [10000., 0.], "amount": [np.nan, 0.]})
    result = native.minute_fields(minute)
    assert result.price_valid.all() and result.volume_valid.all()
    assert native.execution_state(result.iloc[0]) == "EXECUTION_AMOUNT_CONSISTENCY_FAILED"
    assert native.execution_state(result.iloc[1]) == "EXECUTION_NO_VOLUME"


def test_missing_execution_keeps_prediction_and_prior_request():
    _, parent = configuration()
    p = native.daily_features(prices(8), no_dividends())
    minute = pd.DataFrame([{"trade_time": date + pd.Timedelta(hours=h, minutes=m), "open": 5., "high": 5.01, "low": 4.99, "close": 5., "vol": 100000., "amount": 500000.} for date in p.date for h, m in [(9, 30), (15, 0)]])
    prediction = pd.DataFrame({"date": p.date.iloc[1:], "prediction_D": .05})
    base_minutes = native.minute_fields(minute)
    _, requests, _, _ = old.simulate_account(p, no_dividends(), base_minutes, prediction, 20000, "BASE", "D", parent)
    minute.loc[minute.trade_time.eq(p.date.iloc[2] + pd.Timedelta(hours=9, minutes=30)), "amount"] = np.nan
    _, missing_requests, decisions, _ = old.simulate_account(p, no_dividends(), native.minute_fields(minute), prediction, 20000, "BASE", "D", parent)
    assert requests.requested_quantity.iloc[0] == missing_requests.requested_quantity.iloc[0] > 0
    assert missing_requests.filled_quantity.iloc[0] == 0
    assert missing_requests.status.iloc[0] == "UNFILLED_EXECUTION_DATA_MISSING"
    assert decisions.prediction.iloc[0] == .05
    assert decisions.state.iloc[0] == "REQUEST_NEXT_OPEN"


def test_current_date_without_local_data_never_inherits_old_view_or_cash_position():
    runtime = pd.DataFrame({"date": pd.to_datetime(["2026-08-12"])})
    result = native.status_for_date(runtime, pd.Timestamp("2026-10-02"))
    assert result["prediction_state"] == "NO_VIEW_NO_CURRENT_LOCAL_RECEIPT"
    assert result["prediction"] is None
    assert result["actual_position"] == "UNKNOWN"
    assert not result["historical_signal_carried_as_current"]
