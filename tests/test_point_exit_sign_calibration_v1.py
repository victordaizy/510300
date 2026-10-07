"""必要测试：周期权重、未知、入场身份和首次确认事件。"""
import numpy as np
import pandas as pd

from research.point_exit_sign_calibration_v1 import (
    first_confirmed_events, natural_match, prediction_sign, summarize_group,
)


def test_cycle_equal_weight_does_not_count_long_cycle_as_more_evidence():
    frame = pd.DataFrame({"cycle_id": [1, 1, 1, 2], "entry_year": [2020]*4,
                          "prediction": [-.01]*4, "target": [-.03, -.03, -.03, .01]})
    summary = summarize_group(frame)
    assert summary["cycles"] == 2
    assert np.isclose(summary["cycle_equal_mean_target"], -.01)
    assert summary["cycle_equal_negative_target_rate"] == .5
    assert summary["target_mean_interval"]["status"] == "NOT_COMPUTED_INSUFFICIENT_ENTRY_YEARS"


def test_unknown_prediction_and_label_are_not_filled_with_zero():
    frame = pd.DataFrame({"cycle_id": [1, 2, 3], "entry_year": [2020, 2021, 2022],
                          "prediction": [np.nan, -.1, .1], "target": [-.2, np.nan, .2]})
    assert prediction_sign(np.nan) == "NO_VIEW"
    assert prediction_sign(0.) == "NONNEGATIVE"
    summary = summarize_group(frame)
    assert summary["all_rows"] == 3 and summary["eligible_rows"] == 1
    assert summary["cycles"] == 1 and summary["cycle_equal_mean_target"] == .2


def test_reference_cycle_number_cannot_replace_entry_identity():
    cycles = pd.DataFrame({"cycle_id": [1, 2], "entry_index": [10, 20],
                           "entry_date": pd.to_datetime(["2020-01-02", "2020-02-03"])})
    assert natural_match(cycles, 1, "2020-01-02") is None
    assert natural_match(cycles, 10, "2020-02-03") is None
    assert natural_match(cycles, 20, "2020-02-03").cycle_id == 2


def test_only_first_original_confirmed_event_per_cycle_and_cost():
    frame = pd.DataFrame({"period": ["2020_2026"]*5, "cost": ["BASE"]*4+["STRESS"],
                          "cycle_id": [1, 1, 1, 2, 1], "origin_index": [10, 11, 12, 20, 11],
                          "learned_exit_requested": [False, True, True, True, True]})
    events = first_confirmed_events(frame)
    assert sorted(events.origin_index.to_list()) == [11, 11, 20]
    prefix = first_confirmed_events(frame.loc[frame.origin_index.le(11)])
    assert prefix.reset_index(drop=True).equals(events.loc[events.origin_index.le(11)].reset_index(drop=True))
