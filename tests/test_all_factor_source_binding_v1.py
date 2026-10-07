"""核对原字段、官方简表日期与实际取得上界，防止回填历史。"""
import pandas as pd
import pytest

from research.all_factor_source_binding_v1 import eligible_after_capture, normalize_quote, parse_factsheet


def test_single_target_row_required_without_choosing_favorable_duplicate():
    row = {"indexCode": "000300", "tradeDate": "20260930", "close": 4500., "peg": 14.}
    payload = {"success": True, "code": 200, "data": [row]}
    assert normalize_quote(payload, "20260930")["raw_peg"] == 14.
    with pytest.raises(ValueError):
        normalize_quote({**payload, "data": [row, row]}, "20260930")
    with pytest.raises(ValueError):
        normalize_quote({**payload, "data": [{**row, "indexCode": "000905"}]}, "20260930")
    with pytest.raises(ValueError):
        normalize_quote(payload, "20260929")


def test_original_label_and_statistics_date_both_required():
    text = "沪深300 000300 2026年9月30日 滚动市盈率 14.32 市净率 1.4 股息率 2.7"
    fields = parse_factsheet(text, "20260930")
    assert fields["reported_ttm_pe"] == 14.32
    assert fields["reported_dividend_yield_percent"] == 2.7
    with pytest.raises(ValueError):
        parse_factsheet(text, "20260929")
    with pytest.raises(ValueError):
        parse_factsheet(text.replace("滚动市盈率", "原peg"), "20260930")


def test_actual_capture_cannot_be_used_at_old_statistics_close():
    snapshot = {"available_at": "2026-10-06T13:15:00+08:00",
                "current_field_definition_status": "OFFICIAL_FACTSHEET_LABEL_VERIFIED_CURRENT_VERSION_ONLY"}
    assert not eligible_after_capture(snapshot, "2026-09-30T15:05:00+08:00")
    assert eligible_after_capture(snapshot, "2026-10-08T15:05:00+08:00")
    assert not eligible_after_capture(snapshot, "2026-10-08T15:05:00")


def test_old_earnings_and_valuation_are_not_forward_filled_to_later_year():
    from research.all_factor_source_binding_v1 import bind_history
    cases = pd.DataFrame({"date": pd.to_datetime(["2025-05-12"]),
                          "decision_at": ["2025-05-12T15:05:00+08:00"], "core_explanatory_score": [60]})
    calendar = pd.DataFrame({"date": pd.to_datetime(["2025-05-09", "2025-05-12"])})
    valuation = pd.DataFrame(index=pd.DatetimeIndex(["2019-01-08"]))
    earnings = pd.DataFrame({"population": ["DYNAMIC_300"], "group_type": ["全部"],
                             "measure": ["YTD"], "period": ["2023Q1"]})
    result = bind_history(cases, calendar, valuation, earnings)
    assert result.official_original_pe.isna().all()
    assert result.reconstructed_profit_yoy.isna().all()
    assert not result.new_historical_predictive_factor_admission.any()
