"""核对年龄权重、整周期等权、惩罚总量与成熟边界。"""
import numpy as np
import pandas as pd
import pytest
from research.recency_weighted_exit_v1 import recency_weights
from research.learned_cycle_exit_v1 import FEATURES, fit_one


def rows():
    data = pd.DataFrame({"cycle_id": [1, 1, 2, 2, 2], "exit_index": [0, 0, 242, 242, 242],
                         "origin_index": [-2, -1, 239, 240, 241], "target": [.01, .02, -.01, .02, .04]})
    for j, feature in enumerate(FEATURES):
        data[feature] = np.arange(5, dtype=float) * (j + 1) + .1
    data["sample_weight"] = 1 / data.groupby("cycle_id").origin_index.transform("count")
    return data


def test_one_year_older_cycle_has_half_weight_without_duplicating_long_cycles():
    weighted, cycles, effective = recency_weights(rows(), 242, 242)
    total = weighted.groupby("cycle_id").sample_weight.sum()
    assert total.loc[2] / total.loc[1] == pytest.approx(2.)
    np.testing.assert_allclose(total.to_numpy(), [2 / 3, 4 / 3])
    assert weighted.sample_weight.sum() == pytest.approx(2.)
    assert effective == pytest.approx(1.8)
    np.testing.assert_allclose(weighted.loc[weighted.cycle_id == 2, "sample_weight"], 4 / 9)


def test_same_exit_age_reproduces_original_equal_cycle_model():
    original = rows()
    original["exit_index"] = 242
    weighted, cycles, effective = recency_weights(original, 484, 242)
    np.testing.assert_allclose(weighted.sample_weight, original.sample_weight)
    cfg = {"ridge_alpha": 1., "feature_clip": 5.}
    a, b = fit_one(original, "RIDGE", cfg), fit_one(weighted, "RIDGE", cfg)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(a[key], b[key])
    assert effective == pytest.approx(2.)


def test_passage_of_time_without_new_cycle_does_not_create_new_relative_information():
    a, cycle_a, eff_a = recency_weights(rows(), 242, 242)
    b, cycle_b, eff_b = recency_weights(rows(), 484, 242)
    np.testing.assert_allclose(a.sample_weight, b.sample_weight)
    assert eff_a == pytest.approx(eff_b)
    np.testing.assert_allclose(cycle_b.raw_weight, cycle_a.raw_weight / 2)


def test_unfinished_cycle_cannot_receive_a_training_weight():
    with pytest.raises((ValueError, AssertionError), match="未来实际退出"):
        recency_weights(rows(), 200, 242)
