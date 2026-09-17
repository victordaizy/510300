"""必要边界：优先级、未知、输入范围及未来隔离。"""
import numpy as np
import pytest

from research.idle_reversal_opportunity_inputs_v1 import priority_target


def test_positive_parent_has_absolute_priority():
    actual=priority_target([.1,.8,1.,.3],[1.,0.,1.,np.nan])
    np.testing.assert_array_equal(actual,[.1,.8,1.,.3])


def test_known_zero_and_unknown_are_distinct():
    actual=priority_target([0.,0.,0.,np.nan,np.nan],[1.,0.,np.nan,1.,0.])
    np.testing.assert_allclose(actual,[1.,0.,np.nan,np.nan,np.nan],equal_nan=True)


def test_invalid_targets_and_shape_are_rejected():
    for source,reversal in [([1.1],[0.]),([-.1],[0.]),([np.inf],[0.]),([0.],[.5]),([0.,1.],[0.])]:
        with pytest.raises((RuntimeError,AssertionError,ValueError)):
            priority_target(source,reversal)


def test_future_changes_cannot_change_prior_targets():
    source=np.array([0.,.2,np.nan,0.,.7,0.])
    reversal=np.array([1.,0.,1.,0.,1.,1.])
    prefix=priority_target(source[:4],reversal[:4])
    source[4:]=[0.,1.]
    reversal[4:]=[0.,0.]
    np.testing.assert_allclose(priority_target(source,reversal)[:4],prefix,equal_nan=True)
