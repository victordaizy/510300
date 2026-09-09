"""检验扩展路径训练和预测不能跨过整组的可知时间。"""
import numpy as np
import pandas as pd
import pytest

from research.entry_path_coverage_v1 import fit_record, GroupExitController
from research.learned_cycle_exit_v1 import FEATURES, predict


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=15)})
    rows = []
    for group, path, origin, end, mature, target in [(1, 1, 1, 3, 4, .02), (1, 2, 2, 4, 4, .02),
                                                    (2, 3, 3, 5, 12, -9.), (2, 4, 6, 10, 12, -9.)]:
        rows.append({"episode_id": group, "path_id": path, "origin_index": origin, "exit_index": end, "group_mature_index": mature,
                     "target": target, **dict(zip(FEATURES, [1., .01, -.01, 1., .02, .03, .04, .15]))})
    cfg = {"recent_episodes": 20, "minimum_episodes": 1, "minimum_rows": 2, "feature_clip": 5., "ridge_alpha": 1.}
    return data, pd.DataFrame(rows), cfg


def test_model_cannot_train_on_early_closed_path_of_future_mature_group():
    data, samples, cfg = fixture()
    record, rows = fit_record(samples, 10, data, cfg)
    assert record["training_episodes"] == [1] and record["latest_group_mature_index"] == 4
    assert record["status"] == "FIT_COMPLETE"
    assert predict(record["model"], np.zeros(8)) == pytest.approx(.02)
    samples.loc[samples.episode_id.eq(2), "target"] = 100.
    unchanged, _ = fit_record(samples, 10, data, cfg)
    assert record == unchanged


def test_no_model_when_number_of_groups_insufficient_even_with_many_paths():
    data, samples, cfg = fixture()
    cfg["minimum_episodes"] = 2
    record, rows = fit_record(samples, 10, data, cfg)
    assert record["status"].startswith("NO_VIEW") and record["model"] is None
    assert record["training_path_count"] == 2


def test_controller_rejects_future_group_maturity_in_saved_model():
    data, samples, cfg = fixture()
    record, _ = fit_record(samples, 10, data, cfg)
    GroupExitController(data, [record])
    record["latest_group_mature_index"] = 11
    with pytest.raises(ValueError, match="未来整组"):
        GroupExitController(data, [record])
