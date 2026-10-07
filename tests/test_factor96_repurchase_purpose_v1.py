"""明确勾选用途、条件性注销、未知状态和时点查询。"""
from research.factor96_repurchase_purpose_v1 import LABELS, asof_purpose, purpose_fields


def menu(selected):
    return "回购用途" + "".join(("√" if key in selected else "□") + label for key, label in LABELS.items()) + "累计已回购金额100元"


def test_single_incentive_ignores_later_fallback_cancellation():
    text = menu({"EMPLOYEE_INCENTIVE"}) + "若三年内未实施员工持股计划，剩余股份将依法注销。"
    result = purpose_fields([text])
    assert result["status"] == "EXPLICIT_SINGLE_PURPOSE"
    assert result["selected_purposes"] == ["EMPLOYEE_INCENTIVE"]


def test_multiple_purposes_do_not_invent_allocation():
    result = purpose_fields([menu({"EMPLOYEE_INCENTIVE", "CONVERTIBLE_BOND"})])
    assert result["status"] == "EXPLICIT_MULTIPLE_PURPOSES_NO_ALLOCATION"
    assert result["blocks"][0]["multiple_choice_allocation_unknown"]


def test_private_font_glyph_stays_unknown():
    result = purpose_fields([menu({"CANCEL_CAPITAL"}).replace("√", "\uf052")])
    assert result["status"] == "PARTIAL_OR_UNKNOWN_MENU_RETAINED"
    assert result["selected_purposes"] is None


def test_prose_and_incomplete_menu_are_not_forced_into_categories():
    assert purpose_fields(["回购用途用于员工持股计划，未使用部分注销。"])["selected_purposes"] is None
    assert purpose_fields(["回购用途√减少注册资本□用于员工持股计划或股权激励累计已回购金额10元"])["selected_purposes"] is None


def test_unchecked_menu_is_not_no_purpose():
    assert purpose_fields([menu(set())])["status"] == "PARTIAL_OR_UNKNOWN_MENU_RETAINED"


def test_conflicting_menus_stay_separate():
    result = purpose_fields([menu({"EMPLOYEE_INCENTIVE"}), menu({"CANCEL_CAPITAL"})])
    assert result["status"] == "CONFLICTING_COMPLETE_MENUS"


def test_purpose_asof_cannot_backfill_later_change_or_skip_unknown():
    rows = [{"document_id": "a", "root_id": "x", "known_at": "2025-01-02T23:59:59+08:00", "purpose": purpose_fields([menu({"EMPLOYEE_INCENTIVE"})])},
            {"document_id": "b", "root_id": "x", "known_at": "2025-01-05T23:59:59+08:00", "purpose": purpose_fields(["字段未知"])},
            {"document_id": "c", "root_id": "x", "known_at": "2025-01-08T23:59:59+08:00", "purpose": purpose_fields([menu({"CANCEL_CAPITAL"})])}]
    assert asof_purpose(rows, "x", "2025-01-02T10:00:00+08:00")["status"] == "NO_VIEW"
    assert asof_purpose(rows, "x", "2025-01-04T00:00:00+08:00")["selected_purposes"] == ["EMPLOYEE_INCENTIVE"]
    assert asof_purpose(rows, "x", "2025-01-06T00:00:00+08:00")["selected_purposes"] is None
    assert asof_purpose(rows[:2], "x", "2025-01-04T00:00:00+08:00") == asof_purpose(rows, "x", "2025-01-04T00:00:00+08:00")


def test_same_clock_multiple_documents_require_review():
    rows = [{"document_id": name, "root_id": "x", "known_at": "2025-01-02T23:59:59+08:00", "purpose": purpose_fields([menu({"CANCEL_CAPITAL"})])} for name in ["a", "b"]]
    assert asof_purpose(rows, "x", "2025-01-03T00:00:00+08:00")["status"] == "SAME_CLOCK_MULTIPLE_DOCUMENTS_REQUIRE_REVIEW"
