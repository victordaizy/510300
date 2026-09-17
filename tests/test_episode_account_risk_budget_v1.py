import numpy as np
import pytest

from research.episode_account_risk_budget_inputs_v1 import fixed_episode_budget


def test_hand_fixed_budget_and_new_episode():
    p = [0., .2, .4, .8, 0., .4, .5]
    m = [1., 2., .2, 8., 1., .5, 4.]
    target, fixed, starts = fixed_episode_budget(p, m)
    np.testing.assert_allclose(target, [0., .4, .8, 1., 0., .2, .25])
    np.testing.assert_allclose(fixed, [np.nan, 2, 2, 2, np.nan, .5, .5], equal_nan=True)
    np.testing.assert_array_equal(starts, [False, True, False, False, False, True, False])


def test_unknown_source_does_not_end_episode():
    target, fixed, starts = fixed_episode_budget([.2, np.nan, .3, 0, .3], [2, 4, 8, 1, .5])
    np.testing.assert_allclose(target, [.4, np.nan, .6, 0, .15], equal_nan=True)
    assert fixed[1] == fixed[2] == 2
    assert starts.sum() == 2


def test_unknown_initial_risk_does_not_get_repaired_mid_episode():
    target, _, _ = fixed_episode_budget([.2, .3, 0, .2], [np.nan, 2, 1, 2])
    np.testing.assert_allclose(target, [np.nan, np.nan, 0, .4], equal_nan=True)
    with pytest.raises(ValueError):
        fixed_episode_budget([1.01], [2.])


def test_future_multiplier_does_not_change_past_or_locked_episode():
    source = [np.nan, .2, .3, .4, 0, .2]
    first = fixed_episode_budget(source, [1, 2, .2, .3, 1, .5])[0]
    changed = fixed_episode_budget(source, [1, 2, 100, 200, 500, 10])[0]
    np.testing.assert_allclose(first[:5], changed[:5], equal_nan=True)
    assert changed[5] != first[5]
