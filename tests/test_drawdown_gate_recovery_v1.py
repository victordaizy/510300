import numpy as np
import pytest
from research.drawdown_gate_recovery_inputs_v1 import PRIMARY,MODELS,CANDIDATES,drawdown_gate_recovery_frames
from tests.test_runs_reference_blend_v1 import fixture
from tests.test_trend_reference_router_v1 import CFG as OLD
CFG={**OLD,'candidate_models':list(CANDIDATES),'parent_models':MODELS,'drawdown_window':60,'drawdown_gate':-.05,'recovery_closes':2}
def test_invalid_recovery_rule_rejected():
 d,p,s=fixture()
 with pytest.raises(ValueError):drawdown_gate_recovery_frames(d,p,{**CFG,'recovery_closes':3},s)
def test_short_prefix_unknown():
 d,p,s=fixture();f,_=drawdown_gate_recovery_frames(d,p,CFG,s);assert f['BASE'][PRIMARY+'_target'].isna().all()
