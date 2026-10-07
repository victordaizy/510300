"""J01股票时钟、精确源日期、单位、未知及完整原成员的必要测试。"""
import numpy as np
import pandas as pd
import pytest

from research.point_j01_local_yield_intake_v1 import FIELDS, build_field, dates, member_support


def fixture():
    index = pd.bdate_range("2015-01-01", periods=45)
    stock = pd.DataFrame({"date": index, "symbol": "510300.SH", "mom20": .1})
    rates = pd.DataFrame({"date": index, "cgb_10y": 3+np.arange(45)*.01,
        "source": "chinabond.via_akshare.bond_china_yield", "retrieved_at": pd.Timestamp("2026-09-01", tz="Asia/Shanghai")})
    return stock, rates


def test_previous_stock_day_and_twenty_actual_intervals_exclude_same_day_final_yield():
    stock, rates = fixture()
    rates.loc[21, "cgb_10y"] = 999
    field = build_field(stock, rates)
    assert field.loc[20, FIELDS].isna().all()
    assert field.loc[21, "latest_source_row"] == 20
    assert field.loc[21, "base_source_row"] == 0
    assert field.loc[21, "latest_algorithm_available_at"] < field.loc[21, "decision_deadline"]
    assert field.loc[21, FIELDS[0]] == pytest.approx(20.)
    assert field.loc[21, FIELDS[1]] == pytest.approx(2.)


def test_missing_exact_date_cannot_be_filled_by_nearby_rate():
    stock, rates = fixture()
    field = build_field(stock, rates.drop(index=20).reset_index(drop=True))
    assert field.loc[21, "field_status"] == "NO_VIEW_EXACT_REQUIRED_YIELD_DATE_MISSING"
    assert field.loc[21, FIELDS].isna().all()
    assert field.loc[22, FIELDS].notna().all()


def test_real_zero_change_remains_zero_and_is_not_first_publication_proof():
    stock, rates = fixture()
    rates.cgb_10y = 3.
    field = build_field(stock, rates)
    np.testing.assert_array_equal(field.loc[21, FIELDS].to_numpy(float), [0., 0.])
    assert not field.first_publication_evidence_proved.any()
    assert field.loc[21, "latest_retrieved_at"] > field.loc[21, "decision_deadline"]


def test_unknown_price_or_yield_stays_unknown_instead_of_becoming_zero():
    stock, rates = fixture()
    stock.loc[21, "mom20"] = np.nan
    rates.loc[21, "cgb_10y"] = np.nan
    field = build_field(stock, rates)
    assert field.loc[21:22, FIELDS].isna().all().all()


def test_same_calendar_values_in_ms_and_ns_match_without_truncating_intraday_time():
    stock, rates = fixture()
    stock.date = stock.date.astype("datetime64[ms]")
    field = build_field(stock, rates)
    assert field.loc[21, FIELDS].notna().all()
    with pytest.raises(ValueError, match="非零时刻"):
        dates(stock.date+pd.Timedelta(hours=1), "测试观察日")
    with pytest.raises(ValueError, match="唯一递增"):
        dates([stock.date.iloc[0], stock.date.iloc[0]], "测试观察日")


def test_wrong_security_or_non_chinese_treasury_source_is_rejected():
    stock, rates = fixture()
    stock.symbol = "510500.SH"
    with pytest.raises(ValueError, match="标的"):
        build_field(stock, rates)
    stock.symbol = "510300.SH"
    rates.source = "US_TREASURY"
    with pytest.raises(ValueError, match="来源身份"):
        build_field(stock, rates)


def test_all_original_members_and_unready_months_preserved_with_maturity_gate():
    stock, rates = fixture()
    field = build_field(stock, rates.drop(index=20).reset_index(drop=True))
    samples = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [21, 22], "origin": stock.date.iloc[[21, 22]].to_numpy(),
        "exit_index": [25, 25], "mature_date": stock.date.iloc[[25, 25]].to_numpy()})
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 2,
        "fit_index": 30, "fit_origin": stock.date.iloc[30].isoformat(), "model": {"coef": []}}
    unknown = dict(record, model=None)
    result = member_support(samples, [record, unknown], field)
    assert result.algorithmic_source_available_rows.tolist() == [1, 1]
    assert result.algorithmic_source_missing_rows.tolist() == [1, 1]
    assert result.original_model_available.tolist() == [True, False]
    assert not result.all_original_members_supported.any()
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [dict(record, fit_index=24)], field)
