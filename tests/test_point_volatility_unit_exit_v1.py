"""风险单位方法的恒等退化、标签成熟及预测恢复测试。"""
import numpy as np
import pandas as pd
import pytest

from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
from research.point_volatility_unit_exit_v1 import fit_volatility_unit, predict_original_return, model_at_entry


def rows():
    rng = np.random.default_rng(123)
    data = pd.DataFrame(rng.normal(size=(120, 8)), columns=FEATURES)
    data["vol20"] = .2
    data["cycle_id"] = np.repeat(np.arange(1, 11), 12)
    data["target"] = rng.normal(0., .04, len(data))
    data["origin_index"] = np.arange(len(data))
    data["exit_index"] = data.cycle_id * 20 + 100
    data["sample_weight"] = 1. / 12
    return data


CFG = {"feature_clip": 5., "ridge_alpha": 1., "recent_cycles": 20}


def test_constant_volatility_restores_original_model_predictions_exactly():
    sample = rows()
    baseline = fit_within_cycle_exit(sample, CFG)
    changed = fit_volatility_unit(sample, CFG)
    for x in sample[FEATURES].iloc[::9].to_numpy(float):
        assert predict_original_return(changed, x) == pytest.approx(within_cycle_prediction(baseline, x), abs=1e-12)


def test_future_cycle_labels_cannot_change_fit_and_version_locks_at_entry():
    sample = rows()
    observed, ids = training_rows(sample, 230, CFG)
    model = fit_volatility_unit(observed, CFG)
    altered = sample.copy()
    altered.loc[altered.exit_index.gt(230), "target"] = 1000.
    altered_rows, altered_ids = training_rows(altered, 230, CFG)
    other = fit_volatility_unit(altered_rows, CFG)
    assert ids == altered_ids
    np.testing.assert_allclose(model["coefficients"], other["coefficients"], atol=0, rtol=0)
    records = [{"fit_index": 100, "status": "FIT_COMPLETE"}, {"fit_index": 200, "status": "FIT_COMPLETE"}, {"fit_index": 300, "status": "FIT_COMPLETE"}]
    assert model_at_entry(records, 250)["fit_index"] == 200
    assert model_at_entry(records[:2], 250) == model_at_entry(records, 250)


def test_nonpositive_volatility_is_rejected_and_original_return_unit_restored():
    sample = rows()
    broken = sample.copy()
    broken.loc[0, "vol20"] = 0.
    with pytest.raises(ValueError, match="正的当时波动"):
        fit_volatility_unit(broken, CFG)
    model = fit_volatility_unit(sample, CFG)
    x = sample[FEATURES].iloc[0].to_numpy(float, copy=True)
    expected = within_cycle_prediction(model, x) * x[-1]
    assert predict_original_return(model, x) == pytest.approx(expected)
    x[-1] = np.nan
    with pytest.raises(ValueError, match="当时正波动"):
        predict_original_return(model, x)
