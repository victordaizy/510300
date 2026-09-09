"""统一比较保存后的空值表示，同时保留空值和有值区别。"""
import pandas as pd
import pytest
from research.afternoon_entry_v1_output_fix import compare_saved_decisions


def test_saved_nulls_are_compared_without_filling_missing_state(tmp_path):
    left, right = tmp_path / "新.parquet", tmp_path / "原.parquet"
    pd.DataFrame([{}, {"learned_exit_requested": False}, {"learned_exit_requested": True}]).to_parquet(left)
    pd.DataFrame({"learned_exit_requested": [None, False, True]}).to_parquet(right)
    assert compare_saved_decisions(left, right) == 3
    assert pd.read_parquet(left).learned_exit_requested.isna().sum() == 1
    pd.DataFrame({"learned_exit_requested": [False, False, True]}).to_parquet(right)
    with pytest.raises(AssertionError):
        compare_saved_decisions(left, right)
