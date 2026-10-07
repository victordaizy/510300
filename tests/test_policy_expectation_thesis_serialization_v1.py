"""仅核对日期和未知保存；不改变固定金融规则。"""
import json
import numpy as np
import pandas as pd

from research import policy_expectation_thesis_exit_completion_v1_0_1 as adapter


def test_pending_date_and_numpy_values_preserve_value():
    original = {"origin": pd.Timestamp("2023-06-20"), "shares": np.int64(27400), "event": np.bool_(True)}
    normalized = adapter.normalize(original)
    assert normalized == {"origin": "2023-06-20T00:00:00", "shares": 27400, "event": True}
    assert json.loads(json.dumps(normalized, allow_nan=False)) == normalized


def test_unknown_is_null_without_zero_imputation():
    normalized = adapter.normalize({"date": pd.NaT, "risk": np.nan, "missing": pd.NA, "return": 0.0125})
    assert normalized == {"date": None, "risk": None, "missing": None, "return": 0.0125}
    assert json.loads(json.dumps(normalized, allow_nan=False)) == normalized
