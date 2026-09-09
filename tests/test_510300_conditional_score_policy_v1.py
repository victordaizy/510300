"""只用合成行情核验新政策、时钟和真实账户约束，不接触研究收益。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.conditional_score_policy_v1 import (
    Account, assert_batch_parity, batch_simulate, choose_request, features,
    metrics, period, score, simulate_reference, target_weight,
)
from research.intraday_overnight_increment_v1 import execute_order
from scripts.run_510300_conditional_score_policy_v1 import (
    evaluate, fit, fixed_path_costs, initial_population, load_data, select_configuration,
)


@pytest.fixture
def cfg():
    path = Path(__file__).resolve().parents[1] / "config/510300_conditional_score_policy_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic(count=250, seed=13):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=count)
    returns = rng.normal(0.0003, 0.012, count)
    close = 4 * np.exp(np.cumsum(returns))
    opening = close * np.exp(rng.normal(0, 0.004, count))
    data = pd.DataFrame({"date": dates, "open": opening, "close": close,
                         "high": np.maximum(opening, close) * 1.01,
                         "low": np.minimum(opening, close) * 0.99,
                         "volume": 1000000, "amount": rng.uniform(1e7, 3e7, count)})
    dividends = pd.DataFrame({"record_date": [dates[120]], "ex_date": [dates[121]],
                              "payment_date": [dates[126]], "cash_dividend_per_share": [0.1]})
    data.loc[121:, ["open", "close", "high", "low"]] -= 0.1
    return features(data, dividends), dividends


def test_causal_features_and_dividend_identity():
    data, dividends = synthetic()
    expected = np.log((data.close + data.dividend) / data.close.shift(1))
    np.testing.assert_allclose(data.r.iloc[1:], expected.iloc[1:], atol=1e-13)
    modified = data.copy()
    modified.loc[170:, ["open", "close", "high", "low"]] *= 1.4
    rerun = features(modified, dividends)
    pd.testing.assert_frame_equal(data.loc[:169, ["T", "Q", "B", "D", "R"]], rerun.loc[:169, ["T", "Q", "B", "D", "R"]])
    assert not data.feature_valid.iloc[:65].any()
    assert data.feature_valid.iloc[65:].all()
    assert data.T.size > 0
    for name in ["T", "Q"]:
        assert data[name].dropna().between(-1, 1).all()
    for name in ["B", "D", "R", "G"]:
        assert data[name].dropna().between(0, 1).all()


def test_zero_range_valid_but_missing_or_halted_not_zero():
    data, dividends = synthetic()
    data.loc[90, ["open", "high", "low", "close"]] = 4.0
    data.loc[100, "amount"] = np.nan
    data.loc[110, "volume"] = 0
    result = features(data, dividends)
    assert result.q.iloc[90] == 0
    assert np.isnan(result.q.iloc[100]) and np.isnan(result.q.iloc[110])
    assert not result.feature_valid.iloc[100]
    assert len(result) == len(data)


@pytest.mark.parametrize("value,target", [(20, 0), (35, .25), (50, .5), (65, .75), (80, 1)])
def test_score_mapping(value, target):
    assert float(target_weight(value)) == pytest.approx(target)


def test_band_addition_exit_and_no_view(cfg):
    account = Account(100000, 25000, purchase_lots=[(0, 25000)])
    request = choose_request(account, 4, 74, cfg)
    assert request["target_weight"] == pytest.approx(.9)
    assert request["adjusted_weight"] == pytest.approx(.75)
    assert request["requested_quantity"] == 12500
    assert choose_request(account, 4, 55, cfg)["requested_quantity"] == 0
    assert choose_request(account, 4, 56, cfg)["requested_quantity"] > 0
    assert choose_request(account, 4, 30, cfg)["requested_quantity"] < -16000
    small = Account(190000, 2500, purchase_lots=[(0, 2500)])
    assert choose_request(small, 4, 20, cfg)["requested_quantity"] == -2500
    assert choose_request(small, 4, np.nan, cfg)["requested_quantity"] is None


def test_cash_affordability_and_t_plus_one(cfg):
    account = Account(2000)
    execution = execute_order(account, 1000, 4, 4, 0, 1, cfg["costs"]["BASE"], cfg)
    assert execution["filled_quantity"] == 400
    same_day = execute_order(account, -400, 4, 4, 0, 1, cfg["costs"]["BASE"], cfg)
    assert same_day["filled_quantity"] == 0
    next_day = execute_order(account, -400, 4, 4, 0, 2, cfg["costs"]["BASE"], cfg)
    assert next_day["filled_quantity"] == -400
    assert account.cash >= 0


@pytest.mark.parametrize("model,coefficients", [("FULL", [-.5, 1, .5, .3, 1, .7]), ("SIMPLE", [.3, 1.4, .2, .7]), ("NO_REPAIR", [.7, 1, .5, .3, 1, .7])])
def test_original_engine_batch_daily_parity(cfg, model, coefficients):
    data, dividends = synthetic()
    p = period(data, dividends, str(data.date.iloc[90].date()), str(data.date.iloc[-1].date()))
    values = score(p.data, np.asarray(coefficients), model)
    result = assert_batch_parity(p, values, cfg, model)
    assert result["BASE"]["shares"] == 0
    assert result["STRESS"]["filled_quantity"] == 0


def test_batch_multiple_candidates_and_no_view(cfg):
    data, dividends = synthetic()
    data.loc[140, "feature_valid"] = False
    data.loc[160, "open"] = np.nan
    p = period(data, dividends, str(data.date.iloc[90].date()), str(data.date.iloc[-1].date()))
    parameters = np.array([[-2, 0, 0, 0, 0, 0], [.5, 1, 1, .3, 1, .6], [1.7, .3, .8, .2, .6, 1.0]])
    values = score(p.data, parameters, "FULL")
    all_results = batch_simulate(p, values, cfg, details=True)
    for member in range(3):
        for cost_index, cost in enumerate(("BASE", "STRESS")):
            ledger, _, _ = simulate_reference(p, values[:, member], cfg["costs"][cost], cfg, "FULL")
            expected = ledger[["equity", "cash", "shares", "filled_quantity", "net_return"]].to_numpy(float)
            np.testing.assert_allclose(all_results["path"][:, cost_index * 3 + member], expected, atol=1e-7, rtol=1e-12)
    cash_ledger, _, _ = simulate_reference(p, values[:, 0], cfg["costs"]["BASE"], cfg, "FULL")
    assert metrics(cash_ledger, cfg)["net_sharpe"] is None


def test_dividend_entitlement_and_payment(cfg):
    data, dividends = synthetic()
    p = period(data, dividends, str(data.date.iloc[100].date()), str(data.date.iloc[140].date()))
    values = np.full(len(p.data), 90.0)
    ledger, _, _ = simulate_reference(p, values, cfg["costs"]["BASE"], cfg, "FULL")
    record = ledger.loc[ledger.date == data.date.iloc[120]].iloc[0]
    ex = ledger.loc[ledger.date == data.date.iloc[121]].iloc[0]
    payment = ledger.loc[ledger.date == data.date.iloc[126]].iloc[0]
    assert ex.dividend_recognized == record.shares * .1
    assert ex.dividend_paid == 0
    assert payment.dividend_paid == record.shares * .1
    assert ex.dividend_receivable == record.shares * .1
    assert_batch_parity(p, values, cfg, "FULL")


def test_open_gap_does_not_retarget_and_pending_rolls(cfg):
    data, dividends = synthetic()
    start = 100
    anchor = data.close.iloc[start-1]
    data.loc[start, ["open", "high", "close"]] = round(anchor * 1.1, 3)
    data.loc[start + 1, "open"] = anchor * 1.02
    p = period(data, dividends, str(data.date.iloc[start].date()), str(data.date.iloc[start + 3].date()))
    values = np.array([90.0, np.nan, np.nan, 90.0, 90.0])
    ledger, trades, decisions = simulate_reference(p, values, cfg["costs"]["BASE"], cfg, "FULL")
    assert ledger.filled_quantity.iloc[0] == 0
    assert ledger.requested_quantity.iloc[0] == ledger.requested_quantity.iloc[1]
    assert trades.origin.iloc[0] == trades.origin.iloc[1]
    assert ledger.filled_quantity.iloc[1] == decisions.requested_quantity.iloc[0]
    assert_batch_parity(p, values, cfg, "FULL")


def test_fixed_path_cost_identity(cfg):
    data, dividends = synthetic()
    p = period(data, dividends, str(data.date.iloc[90].date()), str(data.date.iloc[-1].date()))
    values = score(p.data, np.array([.5, 1, .5, .3, 1, .7]), "FULL")
    ledger, _, _ = simulate_reference(p, values, cfg["costs"]["BASE"], cfg, "FULL")
    table, info = fixed_path_costs(p, ledger, cfg)
    assert info["fixed_path_total_extra_cost"] > 0
    np.testing.assert_allclose(table.base_equity - table.fixed_path_stress_equity, table.cumulative_extra_cost, atol=1e-8)


def test_selection_reference_utility_and_tie_break(cfg):
    rows = []
    for gamma in cfg["gamma_grid"]:
        for penalty in cfg["lambda_grid"]:
            for year in cfg["validation_years"]:
                rows.append({"model": "FULL", "gamma": gamma, "lambda": penalty, "validation_year": year,
                             "validation": {cost: {"utility_gamma5": .05, "annualized_two_way_turnover": 5}
                                            for cost in ["BASE", "STRESS"]}})
    selected, all_rows = select_configuration(rows, "FULL", cfg)
    assert selected["gamma"] == 10 and selected["lambda"] == .1
    assert len(all_rows) == 9
    altered = copy.deepcopy(rows)
    for row in altered:
        if row["gamma"] == 10:
            row["validation"]["BASE"]["annualized_two_way_turnover"] = 13
    selected, _ = select_configuration(altered, "FULL", cfg)
    assert selected["gamma"] == 5


def test_population_budget_and_structural_constraints(cfg):
    pop = initial_population(6, 123, 72)
    assert pop.shape == (72, 6)
    assert np.all(pop[:, 1:].sum(axis=1) <= 6)
    assert np.all(pop[:, 3] <= pop[:, 4] / 2)
    np.testing.assert_equal(pop, initial_population(6, 123, 72))
    assert (cfg["optimizer"]["maxiter"] + 1) * len(pop) == cfg["optimizer"]["maximum_candidate_evaluations_per_fit"]


def test_main_data_read_requires_frozen_model(cfg, tmp_path):
    with pytest.raises(ValueError, match="主评价行情读取前缺少模型冻结"):
        load_data(cfg, cfg["data_cutoff"], tmp_path, "main")


def test_synthetic_fit_to_report_pipeline(cfg, tmp_path):
    data, dividends = synthetic(280)
    cfg["optimizer"].update(maxiter=2, population_members=8, maximum_candidate_evaluations_per_fit=24)
    cfg["bootstrap_repetitions"] = 12
    cfg["evaluation_start"] = str(data.date.iloc[190].date())
    cfg["data_cutoff"] = str(data.date.iloc[-1].date())
    cfg["high_sharpe_target"] = 1e9
    training = period(data, dividends, str(data.date.iloc[90].date()), str(data.date.iloc[189].date()))
    models = {}
    for model in ("FULL", "SIMPLE"):
        fitted = fit(training, model, 5, .01, cfg, 2019)
        fitted["selection"] = {"selection_score": .01}
        models[model] = fitted
    locked = {"models": models, "total_candidate_evaluations": sum(v["candidate_evaluations"] for v in models.values())}
    result = evaluate(data, dividends, locked, cfg, tmp_path)
    assert result["main_calendar_days"] == 90
    assert result["acceptance_passed"] is False
    assert len(result["economics"]) == 8
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "研究报告.md").exists()
    assert (tmp_path / "连续账户比较.png").stat().st_size > 1000
    decisions = pd.read_csv(tmp_path / "每日决策表.csv")
    assert len(decisions) == 182
    assert decisions["T"].notna().all()
