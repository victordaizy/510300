"""舍入区间检查覆盖整数、负值、缺口和末位精度。"""
from decimal import Decimal
import pytest

from research.forward_eps_report_pe_identity_v1 import compare, rounded_interval


def test_precision_is_preserved_not_fixed_percentage():
    assert rounded_interval("1.830") == (Decimal("1.8295"), Decimal("1.8305"))
    assert rounded_interval("11") == (Decimal("10.5"), Decimal("11.5"))


def test_integer_pe_rounding_can_explain_large_point_difference():
    assert compare("1.83", "11", "19.47")["status"] == "COMPATIBLE_WITH_REPORTED_DISPLAY_PRECISION"


def test_inconsistent_reference_is_not_accepted():
    assert compare("1.83", "11.00", "19.47")["status"] == "REFERENCE_PRICE_OUTSIDE_EPS_PE_ROUNDING_RANGE"


def test_negative_eps_and_pe_have_valid_product_bounds():
    value = compare("-0.41", "-57.5", "23.55")
    assert value["status"] == "COMPATIBLE_WITH_REPORTED_DISPLAY_PRECISION"
    assert Decimal(value["implied_price_lower"]) < Decimal(value["implied_price_upper"])


def test_zero_pe_preserves_unusable_state():
    assert compare("1.00", "0", "12.00")["status"] == "NO_VIEW_NONPOSITIVE_REFERENCE_OR_ZERO_PE"


def test_price_display_precision_also_applies():
    assert compare("1.00", "10.00", "10")["status"] == "COMPATIBLE_WITH_REPORTED_DISPLAY_PRECISION"


def test_non_finite_input_is_not_a_numeric_observation():
    with pytest.raises(ValueError):
        compare("NaN", "10", "20")
