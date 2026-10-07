"""P05/P06保存源的日期、成熟上限、未知及完整成员必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_p05_p06_saved_forecast_intake_v1 import (
    FIELDS, METADATA, build_field, check_saved_clocks, member_support, validate_saved_object)


def source_fixture():
    dates = pd.bdate_range("2015-01-01", periods=700)
    current = pd.DataFrame({"date": dates, "symbol": "510300.SH"})
    saved = pd.DataFrame({"date": dates, "positive_agreement": np.nan, "disagreement": np.nan,
        "cusum": np.nan, "monitor_paused": False, "model_known": False,
        "inner_count": 0, "training_count": 0, "external_stat_idx": np.arange(700)-1,
        "external_stat_date": pd.Series(dates).shift()})
    ids = list(range(300, 644))
    validation = list(range(500, 600, 5))
    lower = dates[650]-pd.DateOffset(years=2)
    outer = [{"decision_idx": 650, "date": dates[650].isoformat(), "lower": lower.isoformat(),
        "training_indices": ids, "latest_training_exit_idx": 649, "inner_validation_indices": validation}]
    inner = [{"decision_idx": 650, "validation_idx": s, "validation_exit_idx": s+6,
        "training_first_idx": 300, "training_last_idx": s-7, "training_count": s-306,
        "lower": lower.isoformat()} for s in validation]
    saved.loc[650, ["positive_agreement", "disagreement", "cusum", "monitor_paused",
                    "model_known", "inner_count", "training_count"]] = [2/3, .15, 0., False, True, 20, len(ids)]
    return current, saved, outer, inner


def test_exact_date_join_preserves_four_known_fields_but_publication_is_unproved():
    current, saved, outer, inner = source_fixture()
    subset = current.iloc[640:661].reset_index(drop=True)
    field, facts = build_field(subset, saved, outer, inner)
    assert facts["outer_saved_clock_records_checked"] == 1
    assert facts["inner_saved_clock_records_checked"] == 20
    assert field.loc[10, "source_index"] == 650
    np.testing.assert_array_equal(field.loc[10, FIELDS].to_numpy(float), [2/3, .15, 0., 0.])
    assert not field.first_publication_evidence_proved.any()
    assert field.loc[9, FIELDS].isna().all()


def test_unknown_original_model_does_not_turn_false_monitor_state_into_known_zero():
    current, saved, outer, inner = source_fixture()
    saved.loc[650, "model_known"] = False
    field, _ = build_field(current, saved, [], [])
    assert field[FIELDS].isna().all().all()
    missing_date = current.iloc[[0]].copy()
    missing_date["date"] = pd.Timestamp("2014-12-31")
    result, _ = build_field(missing_date, saved, [], [])
    assert result.loc[0, "field_status"] == "NO_VIEW_SOURCE_DATE_NOT_SAVED"
    assert result.loc[0, FIELDS].isna().all()


def test_outer_future_duplicate_or_outdated_training_is_rejected():
    _, saved, outer, inner = source_fixture()
    future = copy.deepcopy(outer)
    future[0]["training_indices"][-1] = 644
    future[0]["latest_training_exit_idx"] = 650
    with pytest.raises(ValueError, match="尚未成熟"):
        check_saved_clocks(saved, future, inner)
    duplicated = copy.deepcopy(outer)
    duplicated[0]["training_indices"][-1] = duplicated[0]["training_indices"][0]
    with pytest.raises(ValueError, match="重复"):
        check_saved_clocks(saved, duplicated, inner)
    outdated = copy.deepcopy(outer)
    outdated[0]["training_indices"][0] = 0
    with pytest.raises(ValueError, match="两年下限"):
        check_saved_clocks(saved, outdated, inner)
    lagged = saved.copy()
    lagged.loc[650, "external_stat_idx"] = 648
    with pytest.raises(ValueError, match="主一日"):
        check_saved_clocks(lagged, outer, inner)


def test_inner_validation_and_training_must_be_mature_and_phase_fixed():
    _, saved, outer, inner = source_fixture()
    future = copy.deepcopy(inner)
    future[-1]["validation_idx"] = 645
    future[-1]["validation_exit_idx"] = 651
    future_outer = copy.deepcopy(outer)
    future_outer[0]["inner_validation_indices"][-1] = 645
    with pytest.raises(ValueError, match="当前尚未成熟"):
        check_saved_clocks(saved, future_outer, future)
    training_future = copy.deepcopy(inner)
    training_future[0]["training_last_idx"] = 494
    with pytest.raises(ValueError, match="训练上限尚未成熟"):
        check_saved_clocks(saved, outer, training_future)
    wrong_phase = copy.deepcopy(inner)
    wrong_phase[0]["validation_idx"] = 501
    wrong_phase[0]["validation_exit_idx"] = 507
    wrong_outer = copy.deepcopy(outer)
    wrong_outer[0]["inner_validation_indices"][0] = 501
    with pytest.raises(ValueError, match="相位错误"):
        check_saved_clocks(saved, wrong_outer, wrong_phase)


def test_incomplete_saved_values_unknown_and_finite_illegal_values_rejected():
    current, saved, outer, inner = source_fixture()
    incomplete = saved.copy()
    incomplete.loc[650, "cusum"] = np.nan
    field, _ = build_field(current, incomplete, outer, inner)
    assert field.loc[650, FIELDS].isna().all()
    assert field.loc[650, "field_status"] == "NO_VIEW_INCOMPLETE_SAVED_FORECAST_FIELDS"
    illegal = saved.copy()
    illegal.loc[650, "positive_agreement"] = 2.
    with pytest.raises(ValueError, match="合法范围"):
        build_field(current, illegal, outer, inner)


def test_future_unknown_source_values_do_not_repaint_and_dates_are_strict():
    current, saved, outer, inner = source_fixture()
    validate_saved_object(saved, current)
    changed = saved.copy()
    changed.loc[680:, ["positive_agreement", "disagreement", "cusum"]] = [1., 9., 20.]
    before, _ = build_field(current, saved, outer, inner)
    after, _ = build_field(current, changed, outer, inner)
    pd.testing.assert_frame_equal(before.iloc[:680], after.iloc[:680], check_exact=True)
    duplicate = saved.copy()
    duplicate.loc[651, "date"] = duplicate.loc[650, "date"]
    with pytest.raises(ValueError, match="唯一递增"):
        build_field(current, duplicate, outer, inner)
    wrong = current.copy()
    wrong.loc[650, "symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        build_field(wrong, saved, outer, inner)
    with pytest.raises(ValueError, match="源标的"):
        validate_saved_object(saved, wrong)


def test_original_members_and_maturity_preserved_without_removing_missing_source():
    current, saved, outer, inner = source_fixture()
    field, _ = build_field(current, saved, outer, inner)
    indexes = [649, 650, 651]
    samples = pd.DataFrame({"cycle_id": [1, 1, 1], "origin_index": indexes,
        "origin": current.date.iloc[indexes].to_numpy(), "exit_index": [660, 660, 660],
        "mature_date": current.date.iloc[[660, 660, 660]].to_numpy()})[METADATA]
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 3,
        "fit_index": 690, "fit_origin": current.date.iloc[690].isoformat(), "model": {"saved": True}}
    support = member_support(samples, [record], field)
    assert support.loc[0, "original_training_rows"] == 3
    assert support.loc[0, "saved_source_available_rows"] == 1
    assert support.loc[0, "saved_source_missing_rows"] == 2
    assert not support.loc[0, "all_original_members_supported"]
    with pytest.raises(ValueError, match="数量变化"):
        member_support(samples.iloc[[1]], [record], field)
    premature = copy.deepcopy(record)
    premature["fit_index"] = 659
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [premature], field)
