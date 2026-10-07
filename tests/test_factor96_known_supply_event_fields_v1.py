"""检验源文明确字段、实际/名义口径、未知传播与版本隔离。"""
import pytest

from research.factor96_known_supply_event_fields_v1 import extract_fields, share_value


def parse(body, title="限售股上市流通公告", kind="ELIGIBLE_ORIGINAL_DISCLOSURE"):
    return extract_fields(["证券代码600001。"+body], "2025-01-02", title, kind, "600001.SH")


def test_dates_exclude_old_ipo_day():
    r = parse("公司于2012年1月4日首次上市。本次限售股上市流通日期为2025年1月8日。本次限售股上市流通数量为1000股。")
    assert r["scheduled_listing_date"] == "2025-01-08"
    assert r["field_status"] == "CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW"


def test_actual_quantity_beats_nominal():
    r = parse("本次解除限售股份数量为76,613,628股，其中实际可上市流通的数量为17,938,638股。本次限售股份可上市流通日均为2025年1月8日。")
    assert r["selected_quantity"]["shares_decimal"] == "17938638"
    assert r["quantity_values"]["NOMINAL_UNLOCK"]["shares_decimal"] == "76613628"


def test_actual_heading_without_value_does_not_fall_back():
    r = parse("本次限售股上市流通数量为1000股。本次限售股上市流通日为2025年1月8日。下表列示实际可上市流通股份数量，另附表格。")
    assert r["selected_quantity"] is None and r["field_status"].startswith("NO_VIEW_ACTUAL_TRADABLE")


def test_multiple_dates_remain_conflict():
    r = parse("本次限售股上市流通数量为1000股。本次限售股上市流通日为2025年1月8日。本次第二批股份上市流通日为2025年2月8日。")
    assert r["scheduled_listing_date"] is None
    assert r["field_status"] == "NO_VIEW_LISTING_DATE_MISSING_OR_CONFLICT"


def test_conflicting_quantities_do_not_choose_maximum():
    r = parse("本次限售股上市流通日为2025年1月8日。本次限售股上市流通数量为1000股。本次上市流通股份数量为1200股。")
    assert r["selected_quantity"] is None
    assert r["quantity_states"]["ANNOUNCED_LISTING"] == "CONFLICTING_EXPLICIT_VALUES"


@pytest.mark.parametrize("number,unit,value,resolution", [("1.2345", "万股", "12345.0000", "1.0000"), ("1.2", "亿股", "120000000.0", "10000000.0"), ("76,613,628", "股", "76613628", "1")])
def test_decimal_units_and_precision(number, unit, value, resolution):
    assert share_value(number, unit) == (value, resolution)


def test_revision_is_not_a_new_event_and_missing_is_not_zero():
    r = parse("本次限售股上市流通日为2025年1月8日。本次上市流通股份数量为1000股。", kind="REVISION_TITLE_RETAINED_SEPARATELY")
    assert r["field_status"] == "VERSION_CANDIDATE_NOT_NEW_EVENT" and not r["trading_feature_admitted"]
    empty = parse("没有明确数量。")
    assert empty["selected_quantity"] is None


def test_page_boundary_cannot_concatenate_amount():
    r = extract_fields(["证券代码600001。本次限售股上市流通数量为12", "000股。本次上市流通日为2025年1月8日。"], "2025-01-02", "上市流通公告", "ELIGIBLE_ORIGINAL_DISCLOSURE", "600001.SH")
    assert r["selected_quantity"] is None


def test_current_archive_is_not_replaced_by_economic_day():
    r = parse("本次限售股上市流通日为2025年1月3日。本次上市流通数量为1000股。")
    assert r["conservative_known_at"] == "2025-01-04T00:00:00+08:00"
    assert not r["known_before_scheduled_day"]


def test_nominal_only_and_unconfirmed_security_do_not_admit():
    r = parse("本次解除限售股份总数为1000股。本次解除限售股份的拟上市日期为2025年1月8日。")
    assert r["quantity_values"]["NOMINAL_UNLOCK"] is not None and r["selected_quantity"] is None
    assert not r["trading_feature_admitted"]


def test_signature_later_than_archive_delays_clock():
    r = parse("本次限售股上市流通日为2025年1月8日。本次上市流通数量为1000股。测试股份有限公司董事会2025年1月3日")
    assert r["signature_date_candidates"] == ["2025-01-03"]
    assert r["conservative_known_at"] == "2025-01-05T00:00:00+08:00"


def test_old_fundraising_listing_is_not_current_unlock():
    r = parse("本次募集配套资金发行新增股份上市日期为2019年5月7日。本次解除限售股份的上市流通日为2025年1月8日。本次上市流通股份数量为1000股。")
    assert r["scheduled_listing_date"] == "2025-01-08"


def test_subset_tradable_without_actual_word_and_other_future_tranche():
    r = parse("本次解除限售股份数量为1371994股，其中可上市流通股份数量为342998股。本次上市流通日为2025年1月8日。相应其他股份上市流通日为2027年1月8日。")
    assert r["scheduled_listing_date"] == "2025-01-08"
    assert r["selected_quantity"]["shares_decimal"] == "342998"
