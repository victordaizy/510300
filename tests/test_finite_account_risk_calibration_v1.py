"""检查两档预算分别计算、起始回退、缺失、未来隔离及候选范围。"""
import copy
import numpy as np
import pytest

from research.finite_account_risk_calibration_inputs_v1 import CANDIDATES, RISK_TARGETS, calibrated_paths

CFG={'candidate_models':list(CANDIDATES),'risk_targets':RISK_TARGETS.copy(),'account_risk_window':60,'annual_days':242}


def test_two_budgets_match_hand_window_and_never_exceed_full_capital():
    returns=np.r_[np.nan,np.tile([-.01,.01],32)]
    source=np.full(len(returns),.5)
    got=calibrated_paths(source,returns,1,CFG)
    risk=np.sqrt(60*.01**2/59)*np.sqrt(242)
    for model,budget in RISK_TARGETS.items():
        target,vol,multiplier=got[model]
        assert vol[60] == pytest.approx(risk,abs=1e-14)
        assert multiplier[60] == pytest.approx(budget/risk)
        assert target[60] == pytest.approx(.5*budget/risk)
    assert got['ACCOUNT_RISK_15'][0][60]>got['ACCOUNT_RISK_12'][0][60]
    capped=calibrated_paths(np.ones(len(source)),returns*.01,1,CFG)
    assert all(x[0][60]==1. for x in capped.values())


def test_incomplete_start_and_zero_risk_keep_parent_in_both_candidates():
    returns=np.r_[np.nan,np.zeros(64)]
    source=np.full(len(returns),.2)
    for target,risk,multiplier in calibrated_paths(source,returns,1,CFG).values():
        np.testing.assert_array_equal(target,np.full(len(returns),.2))
        np.testing.assert_array_equal(multiplier,np.ones(len(returns)))
        assert np.isnan(risk[:60]).all() and risk[60]==0.


def test_missing_window_stays_unknown_but_explicit_parent_zero_exits():
    returns=np.r_[np.nan,np.tile([-.01,.01],33)]
    returns[30]=np.nan
    source=np.full(len(returns),.5)
    source[61],source[62]=0.,np.nan
    for target,risk,multiplier in calibrated_paths(source,returns,1,CFG).values():
        assert np.isnan(target[60]) and target[61]==0. and np.isnan(target[62])
        assert np.isnan(risk[60:]).all() and np.isnan(multiplier[60:]).all()


def test_future_returns_and_future_source_do_not_change_past():
    returns=np.r_[np.nan,np.tile([-.01,.01],33)]
    source=np.full(len(returns),.5)
    before=calibrated_paths(source,returns,1,CFG)
    returns[62:],source[62:]=999.,0.
    after=calibrated_paths(source,returns,1,CFG)
    for model in CANDIDATES:
        for left,right in zip(before[model],after[model]):
            np.testing.assert_allclose(left[:62],right[:62],atol=0,rtol=0,equal_nan=True)


def test_unregistered_budget_or_window_cannot_enter_this_batch():
    bad=copy.deepcopy(CFG)
    bad['risk_targets']['ACCOUNT_RISK_12']=.13
    with pytest.raises(ValueError,match='固定两档'):
        calibrated_paths(np.zeros(70),np.zeros(70),1,bad)
    bad=copy.deepcopy(CFG)
    bad['account_risk_window']=20
    with pytest.raises(ValueError,match='固定两档'):
        calibrated_paths(np.zeros(70),np.zeros(70),1,bad)
