"""用独立完整协方差解、合成时钟和真实账户验证相关误差退出。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research import cycle_serial_error_exit_inputs_v1 as module
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def sample():
    rng = np.random.default_rng(152)
    rows = pd.DataFrame(rng.normal(size=(24, 8)), columns=FEATURES)
    rows["cycle_id"] = np.repeat([1, 2, 3], 8)
    rows["origin_index"] = np.concatenate([np.array([1, 2, 3, 5, 6, 7, 9, 10]) + d for d in [0, 30, 60]])
    rows["exit_index"], rows["sample_weight"] = np.repeat([30, 60, 90], 8), 1 / 8
    rows["target"] = .03 * rows[FEATURES[0]] + .02 * rows[FEATURES[1]] + rng.normal(0, .02, 24) + rows.cycle_id * .1
    cfg = {"feature_clip": 5., "ridge_alpha": 1., "recent_cycles": 20, "minimum_cycles": 3, "minimum_rows": 20, "correlation_bounds": [-.99, .99]}
    return rows, cfg, fit_within_cycle_exit(rows, cfg)


def dense_solution(rows, source, rho):
    z = np.clip((rows[FEATURES].to_numpy() - source["mean"]) / source["scale"], -5, 5)
    ids = sorted(set(rows.cycle_id))
    d = np.column_stack([z, *[rows.cycle_id.eq(c).to_numpy(float) for c in ids]])
    precision = np.zeros((len(rows), len(rows)))
    for c in ids:
        indexes = np.flatnonzero(rows.cycle_id.eq(c))
        origins = rows.origin_index.to_numpy()[indexes]
        correlation = rho ** np.abs(origins[:, None] - origins[None, :])
        precision[np.ix_(indexes, indexes)] = np.linalg.inv(correlation) / len(indexes)
    penalty = np.diag(np.r_[np.ones(8), np.zeros(len(ids))])
    return np.linalg.solve(d.T @ precision @ d + penalty, d.T @ precision @ rows.target.to_numpy())


def test_schur_fit_equals_dense_covariance_with_gaps_first_rows_and_negative_rho():
    rows, _, source = sample()
    for rho in [.7, -.45]:
        got = module.fit_serial_gls(rows, source, rho)
        direct = dense_solution(rows, source, rho)
        np.testing.assert_allclose(got["coefficients"], direct[:8], atol=1e-12, rtol=0)
        np.testing.assert_allclose([g["cycle_intercept"] for g in got["cycle_intercepts"]], direct[8:], atol=1e-12, rtol=0)
        assert got["intercept"] == pytest.approx(direct[8:].mean(), abs=1e-12)
        assert sum(g["rows"] for g in got["cycle_intercepts"]) == len(rows)


def test_zero_correlation_matches_original_objective_and_missing_rows_are_rejected():
    rows, _, source = sample()
    got = module.fit_serial_gls(rows, source, 0.)
    np.testing.assert_allclose(got["coefficients"], source["coefficients"], atol=1e-12, rtol=0)
    assert got["intercept"] == pytest.approx(source["intercept"], abs=1e-12)
    rows.loc[0, FEATURES[2]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        module.fit_serial_gls(rows, source, .4)


def test_correlation_pairs_never_cross_cycle_or_missing_trading_day():
    rows = pd.DataFrame(np.zeros((6, 8)), columns=FEATURES)
    rows["cycle_id"], rows["origin_index"], rows["sample_weight"] = [1, 1, 1, 2, 2, 2], [1, 2, 4, 10, 11, 12], 1 / 3
    rows["target"] = [1., 2., 100., 2., 1., 0.]
    model = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES, "mean": [0.] * 8, "scale": [1.] * 8,
             "coefficients": [0.] * 8, "feature_clip": 5., "cycle_intercepts": [{"cycle_id": c, "cycle_intercept": 0.} for c in [1, 2]]}
    got = module.residual_correlation(rows, model)
    assert got["raw_rho"] == pytest.approx(3 / 3.5) and got["adjacent_pairs"] == 3 and got["nonadjacent_gaps"] == 1
    rows["target"] = 0.
    with pytest.raises(ValueError, match="无法识别"):
        module.residual_correlation(rows, model)


def monthly_fixture():
    rows, cfg, source = sample()
    records = []
    for t in [0, 100, 101, 102]:
        chosen, ids = training_rows(rows, t, cfg)
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=t)
        records.append({"fit_index": t, "fit_origin": str(date.date()), "fit_time": str(date + pd.Timedelta(hours=15, minutes=5)),
                        "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(chosen),
                        "latest_exit_index": 90 if ids else None, "latest_exit_date": "2020-03-31" if ids else None,
                        "status": "FIT_COMPLETE" if ids else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "model": copy.deepcopy(source) if ids else None})
    return rows, cfg, records


def test_identical_input_is_fit_once_and_future_months_do_not_change_prior_records(monkeypatch):
    rows, cfg, records = monthly_fixture()
    original = module.fit_serial_gls
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "fit_serial_gls", counted)
    full, _, _, counts = module.build_monthly_models(rows, records, cfg)
    assert len(calls) == 1 and counts["new_model_fits"] == 1 and counts["reused_monthly_fits"] == 2
    assert [r["parameter_first_fit_index"] for r in full] == [None, 100, 100, 100]
    prefix, _, _, _ = module.build_monthly_models(rows, records[:2], cfg)
    assert full[:2] == prefix and full[0]["model"] is None


def test_identical_failed_input_is_cached_without_retry_or_old_prediction_fallback(monkeypatch):
    rows, cfg, records = monthly_fixture()
    calls = []
    def failed(*args):
        calls.append(1)
        raise np.linalg.LinAlgError("合成求解失败")
    monkeypatch.setattr(module, "fit_serial_gls", failed)
    full, _, _, counts = module.build_monthly_models(rows, records, cfg)
    assert len(calls) == 1 and counts["failed_fits"] == 1
    assert all(r["model"] is None for r in full)
    assert all(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in full[1:])


def constant(value, t=0, latest=None):
    return {"fit_index": t, "parameter_first_fit_index": t, "latest_exit_index": t if latest is None else latest,
            "status": "FIT_COMPLETE", "model": {"kind": module.KIND, "features": FEATURES, "mean": [0.] * 8,
            "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": value}}


def test_fixed_entry_version_missing_state_and_future_clock():
    data = fixture()[0]
    data.loc[2, "mom5"] = np.nan
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = module.SerialErrorExitController(data, [constant(-.01), constant(.05, 2)])
    got = [controller(t, cycle, 100000., 100000.) for t in range(1, 5)]
    assert [g["negative_confirmation_count"] for g in got] == [1, 0, 1, 2]
    assert len({g["fixed_prediction_identity"] for g in got}) == 1
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["continuation_prediction"] == .05
    with pytest.raises(ValueError, match="未来周期"):
        module.SerialErrorExitController(data, [constant(-.1, 0, 8)])(5, cycle, 100000., 100000.)
    with pytest.raises(ValueError, match="首次持仓收盘"):
        module.SerialErrorExitController(data, [constant(-.1)])(6, cycle, 100000., 100000.)


def test_actual_exit_blockage_reentry_and_natural_exit_without_initial_model():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.SerialErrorExitController(args[0], [constant(-.01), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert decisions[decisions.origin_index.eq(7)].iloc[0].continuation_prediction == .03
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.SerialErrorExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert not ledger.terminal_unliquidated.iloc[-1]
