"""验证IF持仓实验的合约、信息时钟、权益和简单账户边界。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.if_open_interest_increment_v1 import (
    account_request, five_day_label, if_daily_features, mature_training,
    read_config, ridge_fit, ridge_predict, simulate_account,
)
from research.intraday_overnight_increment_v1 import Account, execute_order


def market_frame(count=20):
    dates = pd.bdate_range("2020-01-01", periods=count)
    return pd.DataFrame({"date": dates, "open": 4.0, "close": 4.0,
                         "previous_close": 4.0, "dividend": 0.0, "RV20": .0001})


def empty_dividends():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]),
                         "payment_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})


def test_real_contract_prices_do_not_turn_roll_gap_into_return():
    dates = pd.bdate_range("2020-01-01", periods=12)
    rows = []
    for i, date in enumerate(dates):
        rows.append({"date": date, "symbol": "IF2006", "close": 4000.0, "open_interest": 100})
        rows.append({"date": date, "symbol": "IF2001" if i < 6 else "IF2009",
                     "close": 3900.0 if i < 6 else 5000.0, "open_interest": 200})
    expiry = pd.DataFrame({"symbol": ["IF2001", "IF2006", "IF2009"],
                           "expiry_date": pd.to_datetime(["2020-01-31", "2020-06-19", "2020-09-18"])})
    spot = pd.DataFrame({"date": dates, "close": 4000.0})
    result = if_daily_features(pd.DataFrame(rows), expiry, spot, "2020-01-01")
    assert result.IF_REL5.dropna().abs().max() == 0
    assert result.loc[6, "IF_ROLL_BOUNDARY"] == 1
    assert result.IF_TOTAL_OI.eq(300).all()
    assert result.IF_OI_LOG_CHANGE1.dropna().eq(0).all()


def test_appending_future_contract_data_does_not_change_past_features():
    dates = pd.bdate_range("2020-01-01", periods=15)
    daily = pd.DataFrame({"date": dates, "symbol": "IF2006", "close": np.arange(15) + 4000.,
                          "open_interest": np.arange(15) + 100})
    expiry = pd.DataFrame({"symbol": ["IF2006"], "expiry_date": [pd.Timestamp("2020-06-19")]})
    spot = pd.DataFrame({"date": dates, "close": 4000.0})
    prefix = if_daily_features(daily.iloc[:10], expiry, spot.iloc[:10], "2020-01-01")
    full = if_daily_features(daily, expiry, spot, "2020-01-01").iloc[:10]
    pd.testing.assert_frame_equal(prefix, full)


def test_unmatured_and_same_day_exit_labels_are_purged():
    samples = pd.DataFrame({"exit_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                            "feature_status": "PASS", "label_status": "MATURE", "Y5_NET_BASE": [.1, .2, .3]})
    admitted = mature_training(samples, pd.Timestamp("2020-01-02"))
    assert admitted.Y5_NET_BASE.tolist() == [.1]


def test_dividend_buyer_after_record_date_has_no_entitlement():
    cfg = read_config()
    feature = market_frame()
    dividends = pd.DataFrame({"record_date": [feature.date.iloc[0]], "ex_date": [feature.date.iloc[1]],
                              "payment_date": [feature.date.iloc[4]], "cash_dividend_per_share": [.1]})
    label = five_day_label(feature, dividends, 0, cfg)
    assert label["label_dividend_per_share"] == 0
    dividends.loc[0, "record_date"] = feature.date.iloc[1]
    dividends.loc[0, "ex_date"] = feature.date.iloc[2]
    label = five_day_label(feature, dividends, 0, cfg)
    assert label["label_dividend_per_share"] == pytest.approx(.1)


def test_label_excludes_gap_before_purchase_and_uses_five_full_days():
    cfg = read_config()
    feature = market_frame()
    feature.loc[0, "close"] = 2.0
    label = five_day_label(feature, empty_dividends(), 0, cfg)
    assert label["entry_gap_total_return"] == pytest.approx(1.0)
    assert label["Y5_NET_BASE"] < 0
    assert label["exit_index"] - label["entry_index"] == 5
    assert five_day_label(feature, empty_dividends(), len(feature) - 4, cfg)["label_status"] == "CENSORED_AFTER_CUTOFF"


def test_ridge_has_unpenalized_intercept_and_fixed_mean_loss_penalty():
    cfg = read_config()
    dates = pd.bdate_range("2018-01-01", periods=4)
    frame = pd.DataFrame({"x": [-1., 1., -1., 1.], "Y5_NET_BASE": [-1., 3., -1., 3.],
                          "origin": dates, "exit_date": dates, "origin_index": np.arange(4)})
    model = ridge_fit(frame, ["x"], cfg)
    assert model["coefficients"][0] == pytest.approx(1.)
    assert model["coefficients"][1] == pytest.approx(2 / 1.1)
    assert ridge_predict({"x": 0}, model, cfg) == pytest.approx(1.)


def test_zero_increment_columns_leave_baseline_prediction_unchanged():
    cfg = read_config()
    dates = pd.bdate_range("2018-01-01", periods=30)
    frame = pd.DataFrame({"x": np.arange(30), "zero": 0., "Y5_NET_BASE": np.sin(np.arange(30)) / 100,
                          "origin": dates, "exit_date": dates, "origin_index": np.arange(30)})
    m0, m1 = ridge_fit(frame, ["x"], cfg), ridge_fit(frame, ["x", "zero"], cfg)
    assert ridge_predict({"x": 10., "zero": 0.}, m0, cfg) == pytest.approx(ridge_predict({"x": 10., "zero": 0.}, m1, cfg))


def test_share_execution_respects_t_plus_one_and_cash():
    cfg = read_config()
    cost = cfg["costs"]["BASE"]
    account = Account(1000.)
    bought = execute_order(account, 1000, 4., 4., 0., 1, cost, cfg)
    assert bought["filled_quantity"] == 200
    assert account.cash >= 0
    sold_same_day = execute_order(account, -200, 4., 4., 0., 1, cost, cfg)
    assert sold_same_day["filled_quantity"] == 0
    sold_next_day = execute_order(account, -200, 4., 4., 0., 2, cost, cfg)
    assert sold_next_day["filled_quantity"] == -200


def test_risk_budget_uses_etf_volatility_and_never_exceeds_one():
    cfg = read_config()
    account = Account(200000.)
    market = market_frame().iloc[0].copy()
    quantity, weight = account_request(account, market, .01, cfg["costs"]["BASE"], cfg)
    assert 0 < weight < 1 and quantity % 100 == 0
    market["RV20"] = 1e-10
    quantity, weight = account_request(account, market, .01, cfg["costs"]["BASE"], cfg)
    assert weight == 1 and quantity <= 50000
    assert account_request(account, market, -.01, cfg["costs"]["BASE"], cfg)[1] == 0


def test_continuous_account_pays_costs_preserves_dividends_and_liquidates():
    cfg = deepcopy(read_config())
    feature = market_frame(18)
    cfg["evaluation_entry_start"] = str(feature.date.iloc[1].date())
    predictions = pd.DataFrame({"origin_index": np.arange(18), "prediction_status": "PASS", "prediction_M1": .01})
    dividends = pd.DataFrame({"record_date": [feature.date.iloc[3]], "ex_date": [feature.date.iloc[4]],
                              "payment_date": [feature.date.iloc[8]], "cash_dividend_per_share": [.1]})
    feature.loc[4:, ["open", "close"]] = 3.9
    feature["previous_close"] = feature.close.shift(1).fillna(4.)
    feature.loc[4, "dividend"] = .1
    ledger, trades, decisions = simulate_account(feature, dividends, predictions, cfg["costs"]["BASE"], cfg, "M1")
    assert ledger.accounting_error.abs().max() < 1e-6
    assert ledger.shares.iloc[-1] == 0
    assert ledger.cash.ge(0).all()
    assert ledger.dividend_recognized.sum() > 0
    assert ledger.dividend_paid.sum() == pytest.approx(ledger.dividend_recognized.sum())
    assert ledger.equity.iloc[-1] < cfg["initial_capital"]
    assert trades.commission.sum() > 0
    assert decisions.target_weight.le(1).all()
