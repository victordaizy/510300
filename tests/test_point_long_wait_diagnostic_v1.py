"""来源未知不能归入零信号，非重叠日历不能在筛选后改相位。"""
import numpy as np
import pytest

from research.point_long_wait_diagnostic_v1 import classify_reason, fixed_phase


def test_wait_reasons_distinguish_missing_gate_and_actual_positive_target():
    assert classify_reason(False, np.nan, 0., 0., np.nan, 0.) == "NO_VIEW"
    assert classify_reason(False, 0., 0., .8, 0., 0.) == "AUX_REJECTED_BY_GATE"
    assert classify_reason(False, 0., 0., 0., 0., 1.) == "NO_CORE_NO_AUX"
    assert classify_reason(False, .2, .2, 0., 0., 1.) == "POSITIVE_TARGET_BLOCKED"
    assert classify_reason(True, np.nan, np.nan, np.nan, np.nan, np.nan) == "HELD"
    with pytest.raises(ValueError):
        classify_reason(False, 0., 0., .8, 0., 1.)


def test_fixed_calendar_phase_is_unchanged_by_subgroup_selection():
    origins = np.arange(100, 181)
    whole = origins[fixed_phase(origins, 100)]
    subgroup = origins[(origins >= 109) & (origins <= 155)]
    selected = subgroup[fixed_phase(subgroup, 100)]
    assert whole.tolist() == [100, 120, 140, 160, 180]
    assert selected.tolist() == [120, 140]
