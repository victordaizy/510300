"""合成案例验证历史修订、时区、月度标签和全账户经济含义。"""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.epu_vintage_increment_v1 import (
    CONFIG, available_clock, asof_state, build_samples, epu_at, fit_model,
    mature_training, month_indices, monthly_label, parse_vintages, predict,
    price_features, simulate_account,
)


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def empty_dividends():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]),
                         "payment_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})


def synthetic_vintages():
    months = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    return pd.DataFrame({"period_start_date": months, "CHNMAINLANDEPU": 100. + np.arange(len(months)),
                         "realtime_start_date": months + pd.offsets.MonthBegin(1) + pd.Timedelta(days=4),
                         "realtime_end_date": pd.NaT})


def test_chicago_calendar_delay_handles_dst_and_never_assumes_china_midnight():
    cfg = config()
    assert available_clock("2026-07-06", cfg) == pd.Timestamp("2026-07-08 13:00", tz="Asia/Shanghai")
    assert available_clock("2026-01-06", cfg) == pd.Timestamp("2026-01-08 14:00", tz="Asia/Shanghai")
    assert available_clock("2026-03-07", cfg) == pd.Timestamp("2026-03-09 13:00", tz="Asia/Shanghai")


def test_revision_becomes_visible_only_at_its_delayed_clock():
    cfg = config()
    raw = synthetic_vintages()
    index = 18
    raw.loc[index, "realtime_end_date"] = pd.Timestamp("2022-02-04")
    new = raw.iloc[[index]].copy()
    new["CHNMAINLANDEPU"], new["realtime_start_date"], new["realtime_end_date"] = 999., pd.Timestamp("2022-02-05"), pd.NaT
    versions = parse_vintages(pd.concat([raw, new], ignore_index=True), cfg)
    before = asof_state(versions, pd.Timestamp("2022-02-07 13:59", tz="Asia/Shanghai"))
    after = asof_state(versions, pd.Timestamp("2022-02-07 14:00", tz="Asia/Shanghai"))
    target = pd.Timestamp("2021-07-01")
    assert before.set_index("observation_month").loc[target, "epu_value"] == 118.
    assert after.set_index("observation_month").loc[target, "epu_value"] == 999.


def test_future_revision_does_not_change_past_epu_features():
    cfg = config()
    raw = synthetic_vintages()
    versions = parse_vintages(raw, cfg)
    decision = pd.Timestamp("2022-02-01 09:00", tz="Asia/Shanghai")
    original = epu_at(versions, decision, cfg)
    changed = raw.copy()
    changed.loc[changed.realtime_start_date.ge("2022-02-01"), "CHNMAINLANDEPU"] = 999999.
    assert epu_at(parse_vintages(changed, cfg), decision, cfg) == original
    assert original["epu_observation_month"] == pd.Timestamp("2021-12-01")


def test_delayed_multi_month_launch_is_not_backfilled_into_old_signals():
    cfg = config()
    raw = synthetic_vintages()
    raw["realtime_start_date"] = pd.Timestamp("2023-02-01")
    versions = parse_vintages(raw, cfg)
    assert epu_at(versions, pd.Timestamp("2022-12-01 09:00", tz="Asia/Shanghai"), cfg)["feature_status"] == "NO_VIEW_EPU_HISTORY_MISSING"
    actual = epu_at(versions, pd.Timestamp("2023-03-01 09:00", tz="Asia/Shanghai"), cfg)
    assert actual["feature_status"] == "PASS" and actual["epu_observation_month"] == pd.Timestamp("2022-12-01")
    assert epu_at(versions, pd.Timestamp("2023-04-03 09:00", tz="Asia/Shanghai"), cfg)["feature_status"] == "NO_VIEW_EPU_TOO_OLD"


def test_only_one_row_per_month_and_label_exit_is_next_month_first_session():
    cfg = config()
    cfg["sample_entry_start"], cfg["evaluation_entry_start"] = "2021-05-01", "2021-06-01"
    dates = pd.bdate_range("2021-01-01", "2021-09-15")
    close = 4 * np.exp(np.sin(np.arange(len(dates))) / 100)
    data = {"prices": pd.DataFrame({"date": dates, "open": close, "high": close + .1, "low": close - .1, "close": close, "volume": 10000.}),
            "dividends": empty_dividends(), "vintages": parse_vintages(synthetic_vintages(), cfg)}
    sample = build_samples(data, price_features(data), cfg, with_labels=False)
    assert len(sample) == 5 and sample.entry_month.nunique() == 5
    assert sample.iloc[0].entry_date == pd.Timestamp("2021-05-03")
    assert sample.iloc[0].exit_date == pd.Timestamp("2021-06-01")
    assert sample.iloc[-1].label_status == "CENSORED_AFTER_CUTOFF"
    assert sample.Y_NET_MONTH.isna().all()


def test_label_does_not_claim_pre_entry_gap_or_old_dividend_entitlement():
    cfg = config()
    dates = pd.bdate_range("2024-01-01", periods=30)
    feature = pd.DataFrame({"date": dates, "open": 4., "close": 4., "dividend": 0.})
    feature.loc[0, "close"] = 2.
    div = pd.DataFrame({"record_date": [dates[0]], "ex_date": [dates[1]], "payment_date": [dates[8]], "cash_dividend_per_share": [.1]})
    row = monthly_label(feature, div, 1, 23, cfg)
    assert row["entry_gap_total_return"] == pytest.approx(1.)
    assert row["label_dividend_per_share"] == 0 and row["Y_NET_MONTH"] < 0


def test_training_rejects_label_maturing_at_or_after_reference_close():
    frame = pd.DataFrame({"exit_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                          "feature_status": "PASS", "label_status": "MATURE", "Y_NET_MONTH": [1., 2., 3.]})
    assert mature_training(frame, "2023-01-02").Y_NET_MONTH.tolist() == [1.]


def test_constant_extra_information_does_not_change_common_prediction():
    cfg = config()
    dates = pd.date_range("2020-01-01", periods=35, freq="MS")
    frame = pd.DataFrame({"x": np.arange(35), "epu": 0., "Y_NET_MONTH": np.sin(np.arange(35)) / 100,
                          "origin": dates, "exit_date": dates, "origin_index": np.arange(35)})
    one, two = fit_model(frame, ["x"], cfg), fit_model(frame, ["x", "epu"], cfg)
    assert predict({"x": 17., "epu": 0.}, one, cfg) == pytest.approx(predict({"x": 17., "epu": 0.}, two, cfg))


def test_monthly_account_uses_standard_deviation_and_preserves_dividend_wealth():
    cfg = config()
    dates = pd.bdate_range("2024-01-01", periods=48)
    feature = pd.DataFrame({"date": dates, "open": 4., "close": 4., "dividend": 0., "RV20": .0001})
    feature.loc[8:, ["open", "close"]] = 3.9
    feature.loc[8, "dividend"] = .1
    feature["previous_close"] = feature.close.shift(1).fillna(4.)
    div = pd.DataFrame({"record_date": [dates[7]], "ex_date": [dates[8]], "payment_date": [dates[11]], "cash_dividend_per_share": [.1]})
    predictions = pd.DataFrame({"entry_index": [1, 24], "exit_index": [24, 45], "entry_month": ["2024-01", "2024-02"],
                                "label_status": "MATURE", "prediction_status": ["PASS", "NO_VIEW_EPU_TOO_OLD"], "prediction_M1": [.01, np.nan]})
    ledger, trades, decisions = simulate_account(feature, div, predictions, cfg["costs"]["STRESS"], cfg, "M1")
    assert decisions.target_weight.iloc[0] == pytest.approx(.1 / np.sqrt(.0001 * 242))
    assert decisions.target_weight.iloc[1] == 0
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0
    assert ledger.dividend_recognized.sum() > 0
    assert ledger.dividend_recognized.sum() == pytest.approx(ledger.dividend_paid.sum())
    assert ledger.equity.iloc[-1] < cfg["initial_capital"] and trades.commission.sum() > 0
    assert ledger.bootstrap_month.iloc[-1] == "2024-02"


def test_calendar_month_blocks_keep_missing_month_positions():
    cfg = config()
    cfg["bootstrap_repetitions"] = 10
    indices = month_indices(43, cfg, 7)
    assert indices.shape == (10, 43)
    assert np.array_equal(indices, month_indices(43, cfg, 7))
    assert np.all(np.diff(indices[:, :3], axis=1) % 43 == 1)
    values = np.arange(43, dtype=float)
    values[12] = np.nan
    assert np.isnan(values[indices]).any()
