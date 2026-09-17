"""验证A/H图表时区、首次可用时钟、未来追加与连续账户的关键边界。"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.ah_premium_increment_v1 import (
    align_source, block_indices, build_features, parse_official_chart, read_config,
)
from research.if_open_interest_increment_v1 import (
    five_day_label, mature_training, ridge_fit, ridge_predict, simulate_account,
)


def empty_dividends():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]),
                         "payment_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})


def test_chart_timestamp_uses_hong_kong_date_not_utc_date():
    timestamp = int(pd.Timestamp("2026-09-11", tz="Asia/Hong_Kong").timestamp() * 1000)
    obj = {"indexCode": "01044.00", "lastUpdate": "2026-09-11 00:00:00", "indexLevels-5y": [[timestamp, 124.10]]}
    frame = parse_official_chart(obj)
    assert frame.date.iloc[0] == pd.Timestamp("2026-09-11")
    assert pd.Timestamp(timestamp, unit="ms", tz="UTC").date().isoformat() == "2026-09-10"
    with pytest.raises(ValueError, match="身份"):
        parse_official_chart({**obj, "indexCode": "00001.00"})


def test_before_available_time_cannot_read_the_source_day_close():
    cfg = read_config()
    source = pd.DataFrame({"date": pd.to_datetime(["2026-09-09", "2026-09-10"]), "close": [100., 200.]})
    decisions = pd.Series(pd.to_datetime(["2026-09-11 07:59", "2026-09-11 09:00"]))
    aligned = align_source(decisions, source, "ah", cfg)
    assert aligned.ah_close.tolist() == [100., 200.]
    assert aligned.ah_available_at.le(decisions).all()


def test_cross_market_holiday_uses_actual_previous_calendar_day_and_rejects_stale():
    cfg = read_config()
    source = pd.DataFrame({"date": pd.to_datetime(["2026-09-25", "2026-09-28", "2026-09-30"]), "close": [100., 101., 102.]})
    decisions = pd.Series(pd.to_datetime(["2026-09-29 09:00", "2026-10-09 09:00"]))
    aligned = align_source(decisions, source, "ah", cfg)
    assert aligned.ah_source_date.iloc[0] == pd.Timestamp("2026-09-28")
    assert aligned.ah_source_ok.tolist() == [True, False]
    assert aligned.ah_age_days.tolist() == [1, 9]


def test_future_external_values_do_not_change_past_predictions_features():
    cfg = read_config()
    dates = pd.bdate_range("2021-01-01", periods=90)
    close = 4 * np.exp(np.sin(np.arange(90) / 3) / 100)
    prices = pd.DataFrame({"date": dates, "open": close, "high": close + .1, "low": close - .1,
                           "close": close, "volume": 100000.})
    ah = pd.DataFrame({"date": dates, "close": 120 + np.sin(np.arange(90))})
    hsi = pd.DataFrame({"date": dates, "close": 20000 + np.arange(90)})
    data = {"prices": prices, "dividends": empty_dividends(), "ah": ah, "hsi": hsi}
    original = build_features(data, cfg)
    changed = deepcopy(data)
    changed["ah"].loc[61:, "close"] = 9000
    changed["hsi"].loc[61:, "close"] = 900000
    alternative = build_features(changed, cfg)
    pd.testing.assert_frame_equal(original.iloc[:60], alternative.iloc[:60])
    appended = {k: (v.iloc[:61] if k != "dividends" else v) for k, v in data.items()}
    prefix = build_features(appended, cfg)
    pd.testing.assert_frame_equal(original.iloc[:60], prefix.iloc[:60])


def test_unmatured_labels_cannot_enter_the_monthly_training():
    samples = pd.DataFrame({"exit_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                            "feature_status": "PASS", "label_status": "MATURE", "Y5_NET_BASE": [.1, .2, .3]})
    assert mature_training(samples, pd.Timestamp("2024-01-02")).Y5_NET_BASE.tolist() == [.1]


def test_zero_new_information_does_not_change_the_common_model_prediction():
    cfg = read_config()
    dates = pd.bdate_range("2021-01-01", periods=30)
    sample = pd.DataFrame({"x": np.arange(30), "ah": 0., "Y5_NET_BASE": np.sin(np.arange(30)) / 100,
                           "origin": dates, "exit_date": dates, "origin_index": np.arange(30)})
    a, b = ridge_fit(sample, ["x"], cfg), ridge_fit(sample, ["x", "ah"], cfg)
    assert ridge_predict({"x": 12., "ah": 0.}, a, cfg) == pytest.approx(ridge_predict({"x": 12., "ah": 0.}, b, cfg))


def test_label_excludes_pre_purchase_gap_and_old_dividend_rights():
    cfg = read_config()
    dates = pd.bdate_range("2024-01-01", periods=16)
    feature = pd.DataFrame({"date": dates, "open": 4., "close": 4., "dividend": 0.})
    feature.loc[0, "close"] = 2.
    dividends = pd.DataFrame({"record_date": [dates[0]], "ex_date": [dates[1]],
                              "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    label = five_day_label(feature, dividends, 0, cfg)
    assert label["entry_gap_total_return"] == pytest.approx(1.)
    assert label["label_dividend_per_share"] == 0
    assert label["Y5_NET_BASE"] < 0
    assert label["exit_index"] - label["entry_index"] == 5


def test_continuous_account_with_dividend_has_exact_wealth_and_final_cash():
    cfg = read_config()
    dates = pd.bdate_range("2024-01-01", periods=18)
    cfg["evaluation_entry_start"] = str(dates[1].date())
    feature = pd.DataFrame({"date": dates, "open": 4., "close": 4., "dividend": 0., "RV20": .0001})
    feature.loc[4:, ["open", "close"]] = 3.9
    feature.loc[4, "dividend"] = .1
    feature["previous_close"] = feature.close.shift(1).fillna(4.)
    dividends = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]],
                              "payment_date": [dates[8]], "cash_dividend_per_share": [.1]})
    predictions = pd.DataFrame({"origin_index": np.arange(18), "prediction_status": "PASS", "prediction_M1": .01})
    ledger, trades, decisions = simulate_account(feature, dividends, predictions, cfg["costs"]["STRESS"], cfg, "M1")
    assert ledger.accounting_error.abs().max() < 1e-6
    assert ledger.shares.iloc[-1] == 0 and ledger.cash.ge(0).all()
    assert ledger.dividend_paid.sum() == pytest.approx(ledger.dividend_recognized.sum())
    assert ledger.dividend_recognized.sum() > 0 and ledger.equity.iloc[-1] < cfg["initial_capital"]
    assert trades.commission.sum() > 0 and decisions.target_weight.le(1).all()


def test_paired_calendar_blocks_preserve_adjacency_and_are_reproducible():
    cfg = read_config()
    cfg["bootstrap_repetitions"] = 10
    indices = block_indices(53, cfg, seed=17)
    assert indices.shape == (10, 53)
    assert np.array_equal(indices, block_indices(53, cfg, seed=17))
    assert np.all(np.diff(indices[:, :20], axis=1) % 53 == 1)
    assert indices.min() >= 0 and indices.max() < 53
