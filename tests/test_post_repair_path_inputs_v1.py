"""检验跨确认动量连续性、缺失和未来后缀隔离。"""

import numpy as np
import pandas as pd

from research.point_fresh_repair_order_inputs_v1 import signals
from research.post_repair_path_inputs_v1 import known_paths


def sample(hist=None):
    d = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=8),
        "ac": [9., 11., 12., 9., 9., 11., 12., 9.], "ema20": [10.] * 8,
        "daily_hist": hist if hist is not None else [-1., 1., 1., 1., 1., 1., 1., 1.],
        "up_volume_balance5": [-1., 1., 1., -1., 1., 1., 1., -1.],
        "available": [True] * 8, "atr20": [1.] * 8,
    })
    return d


def test_positive_endpoints_do_not_prove_continuity():
    retained = sample()
    a = known_paths(retained, signals(retained))
    assert a.momentum_path_class.iloc[5] == "MOMENTUM_RETAINED"
    assert a.known_previous_confirmation_index.iloc[5] == 1
    assert a.known_first_price_failure_index.iloc[5] == 3
    interrupted = sample([-1., 1., 1., -1., 1., 1., 1., 1.])
    b = known_paths(interrupted, signals(interrupted))
    assert b.current_daily_hist.iloc[1] > 0 and b.current_daily_hist.iloc[5] > 0
    assert b.momentum_path_class.iloc[5] == "MOMENTUM_REBUILT"


def test_unknown_gap_does_not_prove_retained_path():
    d = sample()
    d.loc[3, "daily_hist"] = np.nan
    r = known_paths(d, signals(d))
    assert r.momentum_path_class.iloc[3] == "NO_VIEW_CONTINUITY"
    assert r.momentum_path_class.iloc[5] == "NO_PREVIOUS_CONFIRMATION"
    assert not r.momentum_retained_through_failure.iloc[5]
    assert r.known_previous_confirmation_index.iloc[5] == -1


def test_path_states_do_not_read_future_suffix():
    d = sample()
    full = known_paths(d, signals(d))
    for n in range(1, len(d) + 1):
        prefix = d.iloc[:n].reset_index(drop=True)
        result = known_paths(prefix, signals(prefix))
        pd.testing.assert_frame_equal(result, full.iloc[:n].reset_index(drop=True), check_dtype=False)
        changed = d.copy()
        changed.loc[n:, ["ac", "daily_hist", "up_volume_balance5"]] = [1., -100., -100.]
        alternate = known_paths(changed, signals(changed))
        pd.testing.assert_frame_equal(alternate.iloc[:n].reset_index(drop=True), result, check_dtype=False)
