"""合成数据检验成交标签、成熟时钟、嵌套岭回归和日历区块。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.eps_growth_disagreement_increment_v1 import fit_model, predict, mature_training, make_samples, block_draw_indices


def test_same_alpha_objective_and_constant_feature_matches_sklearn():
    train = pd.DataFrame({"x": np.arange(20, dtype=float), "z": np.sin(np.arange(20)), "constant": np.ones(20), "Y60": np.cos(np.arange(20)) / 10})
    columns = ["x", "z", "constant"]
    model = fit_model(train, columns, 10.0)
    reference = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[columns], train.Y60)
    test = {"x": 25.0, "z": -.7, "constant": 1.0}
    np.testing.assert_allclose(predict(test, model), reference.predict(pd.DataFrame([test])[columns])[0], rtol=0, atol=1e-13)
    assert model["scale"][-1] == 1.0


def test_future_features_and_unmatured_labels_cannot_change_prior_fit():
    origins = pd.date_range("2020-01-31", periods=18, freq="ME")
    frame = pd.DataFrame({"origin": origins, "all_features_valid": True, "label_exit_date": origins + pd.Timedelta(days=61), "x": np.arange(18, dtype=float), "Y60": np.sin(np.arange(18)) / 10})
    decision = pd.Timestamp("2021-06-30")
    before = mature_training(frame, decision)
    changed = frame.copy()
    future = changed.label_exit_date.gt(decision) | changed.origin.ge(decision)
    changed.loc[future, ["x", "Y60"]] = 999999.0
    after = mature_training(changed, decision)
    pd.testing.assert_frame_equal(before, after)
    assert fit_model(before, ["x"], 10.0) == fit_model(after, ["x"], 10.0)


def test_label_uses_next_open_and_entitled_dividend_with_sixty_session_exit():
    dates = pd.bdate_range("2020-01-01", periods=100)
    data = pd.DataFrame({"date": dates, "open": 10.0})
    data.loc[10, "open"] = 5.0
    data.loc[71, "open"] = 11.0
    monthly = pd.DataFrame({"origin": [dates[10], dates[50]], "all_features_valid": True})
    dividends = pd.DataFrame({"record_date": [dates[11], dates[71]], "ex_date": [dates[12], dates[72]], "cash_dividend_per_share": [.2, .9]})
    sample = make_samples(monthly, data, dividends, {"horizon": 60})
    assert sample.loc[0, "entry_index"] == 11
    assert sample.loc[0, "exit_index"] == 71
    np.testing.assert_allclose(sample.loc[0, "Y60"], .12, rtol=0, atol=1e-14)
    assert sample.loc[0, "label_dividend_per_share"] == .2
    assert sample.loc[1, "label_status"] == "CENSORED_AFTER_FROZEN_CUTOFF"
    assert pd.isna(sample.loc[1, "Y60"])


def test_calendar_blocks_preserve_missing_position_and_nested_models_add_one_column():
    indices = block_draw_indices(11, 3, 2000, 20260914)
    assert indices.shape == (2000, 11)
    for start in (0, 3, 6):
        assert np.all((indices[:, start + 1] - indices[:, start]) % 11 == 1)
        assert np.all((indices[:, start + 2] - indices[:, start + 1]) % 11 == 1)
    losses = np.arange(11, dtype=float)
    losses[4] = np.nan
    assert np.isnan(losses[indices]).any()
    from research.eps_growth_disagreement_increment_v1 import read, CONFIG
    config = read(CONFIG)
    assert config["models"]["M1"] == config["models"]["M0"] + ["median_absolute_growth_disagreement"]
