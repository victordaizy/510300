"""J02保存残差、15:05输入投影、缺源及完整成员必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_j02_saved_residual_intake_v1 import FIELDS, build_field, check_saved_clocks, member_support, validate_common


def fixture():
    index = pd.bdate_range("2020-01-01", periods=700)
    source = pd.DataFrame({"date": index, "close": 7., "dxy": 100., "us_10y": 2., "cn_10y": 3.,
        "available_at": index.tz_localize("Asia/Shanghai")+pd.Timedelta(days=2, hours=23, minutes=59),
        "bar_end_utc": index.tz_localize("UTC")+pd.Timedelta(days=1), "fx_change5": .001,
        "usd_change5": np.arange(700)*.0001, "spread_change5": np.cos(np.arange(700))*.01,
        "window_crosses_missing_source_year": False})
    day = index[600]
    deadline = day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15, minutes=5)
    latest = int(pd.DatetimeIndex(source.available_at).searchsorted(deadline, side="right")-1)
    ids = np.flatnonzero((np.arange(700) < latest) & (index >= day-pd.DateOffset(years=2)))
    saved = pd.DataFrame([{ "date": day, "decision_at": day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16),
        "model_status": "AVAILABLE", "source_admitted": True, "source_idx": latest, "source_date": index[latest],
        "source_available_at": source.available_at.iloc[latest], "source_age_days": (day-index[latest]).days,
        "training_n": len(ids), "training_start": index[ids[0]], "training_end": index[ids[-1]], "training_max_idx": float(ids[-1]),
        "intercept": 0., "beta_usd": 0., "beta_spread": 0., "residual_fx_change5": 0.,
        "observed_fx_change5": .001, "usd_change5": source.usd_change5.iloc[latest], "spread_change5": source.spread_change5.iloc[latest]}])
    training = [{"decision_date": day.isoformat(), "current_source_idx": latest, "training_indices": ids.tolist()}]
    current = pd.DataFrame({"date": index[599:603], "symbol": "510300.SH"})
    return current, saved, source, training


def test_known_zero_saved_residual_is_preserved_with_1505_algorithm_clock_only():
    current, saved, source, training = fixture()
    field, facts = build_field(current, saved, source, training)
    assert field.loc[1, FIELDS[0]] == 0.
    assert field.loc[1, "saved_source_available_at"] <= field.loc[1, "current_decision_deadline"]
    assert field.loc[1, "saved_original_decision_at"] > field.loc[1, "current_decision_deadline"]
    assert not field.first_publication_evidence_proved.any()
    assert facts["saved_available_model_metadata_checked"] == 1


def test_missing_date_and_original_no_model_are_not_carried_or_filled_zero():
    current, saved, source, training = fixture()
    field, _ = build_field(current, saved, source, training)
    assert field.loc[[0, 2, 3], FIELDS].isna().all().all()
    saved.loc[0, ["source_admitted", "model_status"]] = [False, "NO_VIEW_TRAINING_HISTORY"]
    field, _ = build_field(current, saved, source, [])
    assert field[FIELDS].isna().all().all()
    assert field.loc[1, "field_status"] == "NO_VIEW_ORIGINAL_NO_VIEW_TRAINING_HISTORY"


def test_later_source_or_wrong_available_timestamp_is_rejected():
    _, saved, source, training = fixture()
    bad = saved.copy()
    bad.loc[0, "source_idx"] += 1
    with pytest.raises(ValueError, match="最新已可用源"):
        check_saved_clocks(bad, source, training)
    bad = saved.copy()
    bad.loc[0, "source_available_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="可用时间"):
        check_saved_clocks(bad, source, training)


def test_future_duplicate_or_incomplete_prior_training_cannot_pass_saved_clock():
    _, saved, source, training = fixture()
    for kind in ["future", "duplicate", "removed"]:
        bad = copy.deepcopy(training)
        if kind == "future":bad[0]["training_indices"][-1] = int(saved.source_idx.iloc[0])
        elif kind == "duplicate":bad[0]["training_indices"][-1] = bad[0]["training_indices"][0]
        else:bad[0]["training_indices"] = bad[0]["training_indices"][1:]
        with pytest.raises(ValueError, match="保存训练名单"):
            check_saved_clocks(saved, source, bad)


def test_gap_crossing_window_and_bar_end_after_proxy_clock_are_rejected():
    _, _, source, _ = fixture()
    bad = source.copy();bad.loc[10, "window_crosses_missing_source_year"] = True
    with pytest.raises(ValueError, match="跨缺失年份"):
        validate_common(bad)
    bad.loc[10, ["fx_change5", "usd_change5", "spread_change5"]] = np.nan
    validate_common(bad)
    bad.loc[11, "bar_end_utc"] = bad.available_at.iloc[11].tz_convert("UTC")+pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="日柱结束"):
        validate_common(bad)


def test_ms_and_ns_dates_match_but_wrong_asset_or_source_price_is_rejected():
    current, saved, source, training = fixture()
    current.date = current.date.astype("datetime64[ms]")
    assert build_field(current, saved, source, training)[0].loc[1, FIELDS[0]] == 0.
    current.symbol = "510500.SH"
    with pytest.raises(ValueError, match="标的"):
        build_field(current, saved, source, training)
    source.loc[0, "close"] = 0.
    with pytest.raises(ValueError, match="真实报价"):
        validate_common(source)


def test_original_missing_members_and_no_model_month_preserved_with_maturity_check():
    current, saved, source, training = fixture()
    field, _ = build_field(current, saved, source, training)
    samples = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [1, 2], "origin": current.date.iloc[[1, 2]].to_numpy(),
        "exit_index": [3, 3], "mature_date": current.date.iloc[[3, 3]].to_numpy()})
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 2, "fit_index": 4,
        "fit_origin": (current.date.iloc[-1]+pd.Timedelta(days=1)).isoformat(), "model": {"coef": []}}
    result = member_support(samples, [record, dict(record, model=None)], field)
    assert result.saved_source_available_rows.tolist() == [1, 1]
    assert result.saved_source_missing_rows.tolist() == [1, 1]
    assert result.original_model_available.tolist() == [True, False]
    assert not result.all_original_members_supported.any()
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [dict(record, fit_index=2)], field)
