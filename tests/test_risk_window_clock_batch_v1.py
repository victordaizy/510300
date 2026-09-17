"""检查长短准备期、未来隔离和固定区间在未知状态下的衔接。"""
import copy

import numpy as np
import pytest

from research.risk_window_clock_batch_inputs_v1 import CANDIDATES, SETTINGS, paths

CFG = {'candidate_models': list(CANDIDATES), 'candidate_settings': SETTINGS, 'annual_days': 242}


@pytest.mark.parametrize('window', [30, 120])
def test_new_windows_use_completed_sample_and_correct_initial_fallback(window):
    returns = np.r_[np.nan, np.tile([-.01, .01], 70)]
    source = np.full(len(returns), .2)
    got = paths(source, returns, 1, CFG)
    expected = .01*np.sqrt(window/(window-1))*np.sqrt(242)
    model = f'RISK_DAILY_{window}_12'
    target, risk, current, applied = got[model]
    assert np.isnan(risk[window-1])
    assert current[window-1] == 1.
    assert risk[window] == pytest.approx(expected, abs=1e-14)
    assert target[window] == pytest.approx(.2*.12/expected, abs=1e-14)
    assert applied[window] == current[window]


def test_all_registered_paths_keep_past_unchanged_when_future_is_replaced():
    returns = np.r_[np.nan, np.tile([-.01, .01], 80)]
    source = np.full(len(returns), .2)
    source[125] = 0.
    before = paths(source, returns, 1, CFG)
    returns[140:] = 5.
    source[140:] = 0.
    after = paths(source, returns, 1, CFG)
    for model in CANDIDATES:
        for left, right in zip(before[model], after[model]):
            np.testing.assert_allclose(left[:140], right[:140], atol=0, rtol=0, equal_nan=True)


def test_unknown_does_not_restart_episode_and_known_zero_resets():
    returns = np.r_[np.nan, np.tile([-.01, .01], 80)]
    source = np.full(len(returns), .2)
    source[90] = np.nan
    source[130] = 0.
    source[131:] = .2
    target, risk, current, applied = paths(source, returns, 1, CFG)['RISK_EPISODE_60_12']
    assert target[89] == .2 and np.isnan(target[90]) and target[91] == .2
    assert applied[90] == 1. and target[130] == 0.
    assert applied[131] == current[131] and target[131] != .2


def test_duplicate_old_settings_and_unregistered_candidates_are_not_run():
    assert len(SETTINGS) == 14 and 'RISK_DAILY_60_12' not in SETTINGS and 'RISK_EPISODE_60_10' not in SETTINGS
    bad = copy.deepcopy(CFG)
    bad['candidate_settings']['RISK_EPISODE_60_12']['window'] = 45
    with pytest.raises(ValueError, match='登记'):
        paths(np.zeros(160), np.zeros(160), 1, bad)
