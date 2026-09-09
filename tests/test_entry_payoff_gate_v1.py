"""检验完整交易经济标签、整组成熟、进入筛选与原再次进入状态。"""
import numpy as np
import pandas as pd
from research.entry_payoff_gate_inputs_v1 import FEATURES, full_entry_label, choose_rows, fit_model, predict, entry_views
from research.entry_payoff_gate_account_v1 import simulate_entry_payoff_gate
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def account_inputs():
    dates = pd.bdate_range("2020-01-01", periods=12)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": np.ones(len(data), dtype=int), "exit": {1: np.zeros(len(data), dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": .06, "trail": .08, "take": None, "days": 60}}}
    views = pd.DataFrame({"date": dates, "entry_model_status": "ENTRY_MODEL_PREDICTION_AVAILABLE", "predicted_entry_return": .01, "entry_fit_origin": dates[0]})
    return data, div, cfg, cost, str(dates[1].date()), rule, spec, views


def test_positive_entry_gate_exactly_preserves_natural_account_and_dividends():
    args = list(account_inputs())
    dates = args[0].date
    args[0].loc[4, "dividend"] = .05
    args[1] = pd.DataFrame({"record_date": [dates.iloc[3]], "ex_date": [dates.iloc[4]], "payment_date": [dates.iloc[7]], "cash_dividend_per_share": [.05]})
    args[5]["exit"][1][5] = True
    new, decisions, _ = simulate_entry_payoff_gate(*args)
    old, _, _ = simulate_rearmed_exit(*args[:-1], None)
    pd.testing.assert_frame_equal(new, old)
    assert new.dividend_recognized.sum() > 0
    assert new.dividend_paid.sum() == new.dividend_recognized.sum()
    assert (decisions.requested_quantity > 0).sum() == 1


def test_gate_rejection_does_not_reset_original_entry_condition_after_exit():
    args = list(account_inputs())
    args[5]["exit"][1][2] = True
    args[7].loc[3:5, "predicted_entry_return"] = -.02
    ledger, decisions, _ = simulate_entry_payoff_gate(*args)
    assert ledger.filled_quantity.gt(0).sum() == 1
    assert "旧入场条件" in decisions.iloc[-1].action
    args[5]["entry"][6] = 0
    ledger, _, _ = simulate_entry_payoff_gate(*args)
    assert ledger.filled_quantity.gt(0).sum() == 2
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[-1] == args[0].date.iloc[8]


def test_unavailable_model_is_no_new_buy_and_does_not_consume_permission():
    args = list(account_inputs())
    args[7].loc[:3, "predicted_entry_return"] = np.nan
    args[7].loc[:3, "entry_model_status"] = "NO_VIEW_NO_MATURE_ENTRY_MODEL"
    ledger, decisions, _ = simulate_entry_payoff_gate(*args)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == args[0].date.iloc[5]
    assert decisions.reference_weight.iloc[:4].isna().all()
    assert ledger.shares.iloc[:4].eq(0).all()


def test_complete_label_counts_receivable_once_and_delays_extra_rights():
    dates = pd.bdate_range("2020-01-01", periods=4)
    path = pd.DataFrame({"date": dates[:2], "cash": [0., 110.], "shares": [10, 0], "shares_before": [0, 10], "mark": [10., 11.], "dividend_receivable": [0., 2.],
        "equity": [100., 112.], "price_pnl": [0., 10.], "dividend_recognized": [0., 2.], "commission": [0., 0.], "slippage_cost": [0., 0.], "filled_quantity": [10, -10]})
    div = pd.DataFrame({"record_date": [dates[0], dates[0]], "ex_date": [dates[1], dates[3]], "payment_date": [dates[2], dates[3]], "cash_dividend_per_share": [.2, .1]})
    label = full_entry_label(path, div, 100.)
    assert abs(label["target"]-.13) < 1e-12
    assert label["extra_owned_dividend_after_exit"] == 1.
    assert label["economic_maturity_date"] == dates[3]


def test_training_waits_for_whole_group_and_weights_groups_equally():
    rows = pd.DataFrame({"episode_id": [1, 1, 2, 3], "path_id": [1, 2, 3, 4], "training_maturity_date": pd.to_datetime(["2016-03-01", "2016-03-01", "2016-04-01", None])})
    selected, ids = choose_rows(rows, "2016-03-15", 20)
    assert ids == [1] and len(selected) == 2
    np.testing.assert_allclose(selected.sample_weight, .5)
    selected, ids = choose_rows(rows, "2016-04-01", 1)
    assert ids == [2] and selected.sample_weight.iloc[0] == 1


def test_entry_ridge_matches_weighted_normal_equations():
    rng = np.random.default_rng(95)
    x = rng.normal(size=(90, 9))
    x[:, 6] = 50
    weights = np.repeat([1/20, 1/30, 1/40], [20, 30, 40])
    y = rng.normal(scale=.05, size=len(x))
    rows = pd.DataFrame(x, columns=FEATURES)
    rows["target"], rows["sample_weight"] = y, weights
    model = fit_model(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1
    design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -5, 5)]
    beta = np.linalg.solve(design.T@(weights[:, None]*design)+np.diag([0.]+[1.]*9), design.T@(weights*y))
    np.testing.assert_allclose([model["intercept"], *model["coefficients"]], beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose([predict(model, row) for row in x], design@beta, atol=1e-12, rtol=0)


def test_future_model_and_prices_cannot_change_prior_admission():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, **{key: np.ones(10) for key in FEATURES[:-1]}})
    d60 = pd.Series(np.ones(10))
    model = {"mean": [0.]*9, "scale": [1.]*9, "coefficients": [.001]*9, "intercept": .01, "clip": 5.}
    records = [{"fit_index": 2, "status": "FIT_COMPLETE", "latest_training_maturity": str(dates[1].date()), "model": model}]
    before = entry_views(data, d60, records)
    later = [*records, {"fit_index": 7, "status": "FIT_COMPLETE", "latest_training_maturity": str(dates[6].date()), "model": {**model, "intercept": -.1}}]
    changed = data.copy()
    changed.loc[7:, FEATURES[:-1]] *= -1
    after = entry_views(changed, d60, later)
    pd.testing.assert_frame_equal(before.iloc[:7], after.iloc[:7])
    assert before.predicted_entry_return.iloc[:2].isna().all()
