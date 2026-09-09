"""用可手算账户和合成价格验证本轮时钟、岭目标及分红记账。"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.intraday_overnight_increment_v1 import (
    CONFIG, Account, affordable_quantity, build_features, choose_order, commission,
    evaluate, execute_order, fill_price, fit_ridge, holding_total_return,
    normalize_prices, predict, render_report, return_metrics, sample_schedule, simulate,
)


@pytest.fixture
def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def no_dividends():
    return pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]),
                         "payment_date": pd.to_datetime([]), "cash_dividend_per_share": []})


def synthetic_prices(start="2019-08-01", end="2020-05-29"):
    dates = pd.bdate_range(start, end)
    x = np.arange(len(dates), dtype=float)
    closes = 10 * np.exp(0.001 * x + 0.025 * np.sin(x / 8))
    opens = closes * (1 + 0.002 * np.cos(x / 3))
    return pd.DataFrame({"date": dates, "open": opens, "close": closes,
                         "high": np.maximum(opens, closes) * 1.005,
                         "low": np.minimum(opens, closes) * 0.995, "volume": 1000000})


def test_dividend_wealth_identity_has_no_mechanical_overnight_loss():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                           "open": [10.0, 9.5], "close": [10.0, 9.6]})
    dividends = pd.DataFrame({"ex_date": pd.to_datetime(["2020-01-03"]), "cash_dividend_per_share": [0.5]})
    frame = build_features(prices, dividends, lookback=1)
    assert frame.overnight_log.iloc[1] == pytest.approx(0)
    assert frame.intraday_log.iloc[1] == pytest.approx(np.log(10.1 / 10.0))
    assert frame.identity_error.iloc[1] == pytest.approx(0, abs=1e-14)


def test_future_prices_and_dividends_do_not_change_past_features():
    prices = synthetic_prices()
    original = build_features(prices, no_dividends())
    changed = prices.copy()
    changed.loc[100:, ["open", "close"]] *= 1.3
    div = pd.DataFrame({"ex_date": [prices.date.iloc[120]], "cash_dividend_per_share": [0.8]})
    altered = build_features(changed, div)
    columns = ["M20", "D20", "RV20"]
    pd.testing.assert_frame_equal(original.loc[:99, columns], altered.loc[:99, columns])


def test_entry_gap_excluded_and_exit_ex_date_entitlement_retained():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=8), "open": [10, 12, 12, 12, 12, 11.5, 11.5, 11.5]})
    dividends = pd.DataFrame({"record_date": [data.date.iloc[4]], "ex_date": [data.date.iloc[5]],
                              "payment_date": [data.date.iloc[7]], "cash_dividend_per_share": [0.5]})
    total, distribution = holding_total_return(data, dividends, 1, 6)
    assert total == pytest.approx(0)
    assert distribution == 0.5
    assert holding_total_return(data, dividends, 5, 7) == pytest.approx((0, 0))
    assert holding_total_return(data, dividends, 1, 5) == pytest.approx((0, 0.5))


def test_train_label_maturity_and_five_day_anchor(config):
    config["train_origin_start"] = "2019-10-01"
    data = build_features(synthetic_prices(), no_dividends())
    sample = sample_schedule(data, config, with_labels=True, dividends=no_dividends())
    train = sample.loc[sample.train_eligible]
    evaluation = sample.loc[sample.evaluation_eligible]
    assert train.exit_date.max() <= pd.Timestamp("2019-12-31")
    assert evaluation.origin.iloc[0] == pd.Timestamp("2019-12-31")
    assert (sample.exit_index - sample.entry_index == 5).all()
    assert (sample.origin_index.diff().dropna() == 5).all()
    assert not sample.loc[(sample.origin <= "2019-12-31") & (sample.exit_date > "2019-12-31"), "train_eligible"].any()


def test_ridge_penalty_is_mean_loss_and_intercept_not_penalized():
    n = 40
    x = np.linspace(-2, 2, n)
    y = 0.01 + 0.1 * x + 0.02 * np.sin(x * 4)
    sample = pd.DataFrame({"M20": x, "LOG_RV20": np.cos(x * 3), "Y5": y,
                           "origin": pd.bdate_range("2018-01-02", periods=n),
                           "exit_date": pd.bdate_range("2018-01-10", periods=n)})
    model = fit_ridge(sample, "M0", 0.1)
    z = (sample[["M20", "LOG_RV20"]].to_numpy() - model["mean"]) / model["scale"]
    residual = predict(model, sample) - y
    gradient = 2 * z.T @ residual / n + 0.2 * np.array(model["slopes"])
    np.testing.assert_allclose(gradient, 0, atol=1e-12)
    assert residual.mean() == pytest.approx(0, abs=1e-14)
    duplicate = pd.concat([sample, sample], ignore_index=True)
    doubled = fit_ridge(duplicate, "M0", 0.1)
    np.testing.assert_allclose(model["slopes"], doubled["slopes"], atol=1e-12)


def test_evaluation_labels_cannot_change_frozen_predictions(config):
    config["train_origin_start"] = "2019-10-01"
    data = build_features(synthetic_prices(), no_dividends())
    sample = sample_schedule(data, config, with_labels=True, dividends=no_dividends())
    model = fit_ridge(sample.loc[sample.train_eligible], "M1", 0.1)
    altered = sample.copy()
    altered.loc[~altered.train_eligible, "Y5"] = 100
    other = fit_ridge(altered.loc[altered.train_eligible], "M1", 0.1)
    np.testing.assert_array_equal(predict(model, sample), predict(other, altered))


def test_minimum_commission_tick_and_cash_limit(config):
    cost = config["costs"]["BASE"]
    assert commission(100, 4, cost) == 5
    assert commission(0, 4, cost) == 0
    assert fill_price(4, 1, cost, 0.001) >= 4.002 - 1e-12
    assert fill_price(4, -1, cost, 0.001) <= 3.998 + 1e-12
    assert affordable_quantity(404.9, 4, cost, 100) == 0
    assert affordable_quantity(405, 4, cost, 100) == 100


def test_same_day_purchase_cannot_be_sold_and_next_day_can(config):
    cost = config["costs"]["ZERO_COST_DIAGNOSTIC"]
    account = Account(10000)
    buy = execute_order(account, 100, 10, 10, 0, 1, cost, config)
    assert buy["filled_quantity"] == 100
    sale = execute_order(account, -100, 10, 10, 0, 1, cost, config)
    assert sale["filled_quantity"] == 0
    later = execute_order(account, -100, 10, 10, 0, 2, cost, config)
    assert later["filled_quantity"] == -100
    assert account.value(10) == 10000


def test_hold_wins_score_tie_and_decision_contains_no_future_open(config):
    account = Account(5000, 500, purchase_lots=[(0, 500)])
    cost = config["costs"]["ZERO_COST_DIAGNOSTIC"]
    decision = choose_order(account, 10, 0, 0, cost, config)
    assert decision["action"] == "HOLD"
    assert decision["requested_quantity"] == 0
    assert "open" not in decision


def test_opening_gap_only_clips_precommitted_quantity(config):
    cost = config["costs"]["BASE"]
    account = Account(200000)
    decision = choose_order(account, 10, 0.1, 0.0001, cost, config)
    requested = decision["requested_quantity"]
    fill = execute_order(account, requested, 10.5, 10, 0, 1, cost, config)
    assert fill["requested_quantity"] == requested
    assert 0 < fill["filled_quantity"] < requested
    assert account.cash >= 0


def test_directional_limit_is_unfilled_without_full_day_lookahead(config):
    account = Account(10000)
    result = execute_order(account, 100, 11, 10, 0, 1, config["costs"]["BASE"], config)
    assert result["status"] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert account.cash == 10000 and account.shares == 0


def test_dividend_record_ex_payment_and_terminal_cash_hand_calculation(config):
    dates = pd.bdate_range("2019-12-02", "2020-01-10")
    prices = pd.DataFrame({"date": dates, "open": 10.0, "close": 10.0})
    prices.loc[prices.date >= "2020-01-07", ["open", "close"]] = 9.5
    dividends = pd.DataFrame({"record_date": pd.to_datetime(["2020-01-06"]),
                              "ex_date": pd.to_datetime(["2020-01-07"]),
                              "payment_date": pd.to_datetime(["2020-01-08"]),
                              "cash_dividend_per_share": [0.5]})
    data = build_features(prices, dividends)
    ledger, trades, _ = simulate(data, dividends, {}, config["costs"]["ZERO_COST_DIAGNOSTIC"], config, "BUY_HOLD")
    dated = ledger.set_index("date")
    assert dated.loc["2020-01-06", "dividend_receivable"] == 0
    assert dated.loc["2020-01-07", "dividend_receivable"] == 10000
    assert dated.loc["2020-01-07", "cash"] == 0
    assert dated.loc["2020-01-08", "cash"] == 10000
    assert ledger.equity.iloc[-1] == pytest.approx(200000)
    assert ledger.accounting_error.abs().max() < 1e-8
    assert trades.filled_quantity.tolist() == [20000, -20000]


def test_declared_data_errors_are_not_filled_or_deduplicated():
    data = synthetic_prices()
    with pytest.raises(ValueError, match="重复"):
        normalize_prices(pd.concat([data, data.iloc[[0]]]))
    data.loc[30, "open"] = np.nan
    with pytest.raises(ValueError, match="缺失"):
        normalize_prices(data)


def test_drawdown_includes_initial_capital_and_zero_volatility_is_undefined():
    result = return_metrics(np.array([-0.1, 0.0]), 242)
    assert result["max_drawdown"] == pytest.approx(-0.1)
    assert return_metrics(np.zeros(10), 242)["net_sharpe"] is None


def test_complete_synthetic_experiment_does_not_gate_accounts_on_prediction(config):
    settings = copy.deepcopy(config)
    settings.update({"train_origin_start": "2019-10-01", "data_cutoff": "2020-05-29", "bootstrap_repetitions": 12})
    data = build_features(synthetic_prices(), no_dividends())
    result = evaluate(data, no_dividends(), settings)
    assert len(result["comparison"]) == 9
    assert len(result["ledgers"]) == 9
    assert result["comparison"].maximum_accounting_error.max() < 1e-6
    assert result["comparison"].terminal_liquidated.all()
    assert result["result"]["historical_runs_consumed"] == 1
    assert result["result"]["position_impact"] == 0
    report = render_report(result, settings)
    assert "ZERO_COST_DIAGNOSTIC" in report and "MSE=" in report
    for name, ledger in result["ledgers"].items():
        assert (ledger.cash >= -1e-7).all()
        assert (ledger.shares % 100 == 0).all()
        assert ledger.pnl.sum() == pytest.approx(ledger.equity.iloc[-1] - settings["initial_capital"])
        assert np.prod(1 + ledger.net_return) == pytest.approx(ledger.equity.iloc[-1] / settings["initial_capital"])
