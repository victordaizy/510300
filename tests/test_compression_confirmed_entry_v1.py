"""核对准备窗口边界、过去分位和事件去重。"""
import numpy as np
import pandas as pd

from research.compression_confirmed_entry_v1 import build_factors, preparation_window
from research.volatility_compression_change_point import select_event_indices


def sample_data(n=900):
    r = pd.Series(np.sin(np.arange(n) * .73) * (.02 - np.arange(n) / 60000))
    w = np.exp(r.cumsum())
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "total_log": r, "wealth": w,
                         "sma120": .1, "efficiency20": .5, "sma60": .03, "sma20": .01, "mom20": .02,
                         "mom5": .01, "z20": 0., "close_location": .5, "rsi2": 50., "dd20": 0., "feature_valid": True})


CFG = {"annual_days": 242, "quantile_window": 756, "quantile_minimum": 504, "low_quantile": .2,
       "event_cooldown": 20, "preparation_days": 20}


def test_event_day_excluded_and_twentieth_day_included():
    event = pd.Series(False, index=range(40))
    event.iloc[5] = True
    prepared = preparation_window(event, 20)
    assert not prepared.iloc[5]
    assert prepared.iloc[6:26].all()
    assert not prepared.iloc[26:].any()


def test_original_event_selector_requires_new_condition_and_spacing():
    c = pd.Series(False, index=range(70))
    c.iloc[0:3] = True
    c.iloc[20:23] = True
    c.iloc[30:32] = True
    c.iloc[51:54] = True
    assert select_event_indices(c, cooldown=20) == [0, 30, 51]


def test_future_prices_do_not_change_past_preparation_or_entry():
    data = sample_data()
    changed = data.copy()
    changed.loc[700:, "total_log"] = .4
    changed.loc[700:, "wealth"] *= 10
    pd.testing.assert_frame_equal(build_factors(data, CFG).iloc[:700], build_factors(changed, CFG).iloc[:700])


def test_quantile_excludes_current_rv_and_uses_log_returns():
    data = sample_data()
    f = build_factors(data, CFG)
    expected = data.total_log.rolling(20).std(ddof=1) * np.sqrt(242)
    pd.testing.assert_series_equal(f.log_rv20, expected, check_names=False)
    assert f.prior_rv20_low_quantile.iloc[700] == expected.iloc[max(0, 700 - 756):700].dropna().quantile(.2)
    assert not f.preparation_inputs_available.iloc[:543].any()
    assert (f.candidate_entry <= f.ordinary_breakout).all()
