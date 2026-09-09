"""合成分区验证重取比较不会删样本或把成功下载误当质量通过。"""
import pytest

from research.nbs_v2_common import ContractError
from research.nbs_v2_source_refetch_v1_1 import compare_partition
from tests.test_510300_nbs_v2_source import fixture


def test_identical_native_rows_report_identical():
    _, frame, _, _ = fixture()
    result = compare_partition(frame, frame.copy())
    assert result["all_native_fields_identical"]
    assert result["rows"] == 241


def test_reordered_rows_do_not_create_changes():
    _, frame, _, _ = fixture()
    result = compare_partition(frame, frame.iloc[::-1].copy())
    assert result["all_native_fields_identical"]


def test_source_level_price_defect_stays_visible_after_successful_transport():
    _, frame, _, _ = fixture()
    frame.loc[10, "amount"] += 1
    result = compare_partition(frame, frame.copy())
    assert result["all_native_fields_identical"]
    assert result["positive_bar_vwap_outside_rows"] == 1


def test_supplier_field_correction_is_measured_without_changing_other_fields():
    _, old, _, _ = fixture()
    old.loc[10, "high"] = 3.99
    fresh = old.copy()
    fresh.loc[10, "high"] = 4.01
    old.loc[10, "low"] = 3.98
    fresh.loc[10, "low"] = 3.98
    result = compare_partition(old, fresh)
    assert result["changed_rows_by_field"]["high"] == 1
    assert result["changed_rows_by_field"]["amount"] == 0


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_symbol"])
def test_key_and_symbol_changes_stop(kind):
    _, old, _, _ = fixture()
    fresh = old.copy()
    if kind == "missing":
        fresh = fresh.iloc[:-1]
    elif kind == "duplicate":
        fresh.loc[0, "trade_time"] = fresh.loc[1, "trade_time"]
    else:
        fresh.loc[0, "ts_code"] = "510500.SH"
    with pytest.raises(ContractError):
        compare_partition(old, fresh)
