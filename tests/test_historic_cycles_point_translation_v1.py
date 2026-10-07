"""验证点位研究中会改变结论的数学与股息边界。"""
import numpy as np
import pandas as pd

from research.historic_cycles_point_translation_v1 import entitled_dividend, point_reference, saved_fill_exists, statistics


def test_positive_expectation_does_not_imply_user_stricter_gate():
    result = statistics([.03, .03, .03, -.02, -.02])
    assert np.isclose(result["product"], .9)
    assert np.isclose(result["mean"], .01)
    assert np.isclose(result["standard_expectation_loss_units"], .5)
    assert np.isclose(result["profit_factor"], 2.25)
    assert not result["point_estimate_pass"]


def test_break_even_trades_change_loss_probability():
    result = statistics([.04, -.02, 0])
    assert result["flat"] == 1
    assert np.isclose(result["mean"], result["mean_loss"] * (result["product"] - result["q"]))
    assert not np.isclose(result["q"], 1 - result["p"])


def test_no_losses_do_not_generate_infinite_verified_payoff():
    result = statistics([.02, .01])
    assert np.isnan(result["product"])
    assert not result["point_estimate_pass"]


def test_dividend_rights_follow_record_close_not_payment_date():
    dividends = pd.DataFrame({"record_date": pd.to_datetime(["2025-06-17"]),
                              "ex_date": pd.to_datetime(["2025-06-18"]),
                              "cash_dividend_per_share": [.088]})
    assert entitled_dividend(dividends, pd.Timestamp("2025-06-17"), pd.Timestamp("2025-06-18")) == .088
    assert entitled_dividend(dividends, pd.Timestamp("2025-06-16"), pd.Timestamp("2025-06-17")) == 0
    assert entitled_dividend(dividends, pd.Timestamp("2025-06-18"), pd.Timestamp("2025-06-19")) == 0


def test_fixed_nominal_reference_includes_both_cost_sides_and_dividend():
    result = point_reference(4., 4.2, .05)
    expected = (25000 * (4.195 - 4.004 + .05) - 25000 * (4.195 + 4.004) * .0004) / 100000
    assert result["quantity"] == 25000
    assert np.isclose(result["point_net_return"], expected)
    assert np.isclose(result["point_gross_return"], .0625)
    assert result["point_net_return"] < result["point_gross_return"]


def test_partial_actual_fill_is_a_valid_saved_entry():
    row = pd.Series({"status": "PARTIALLY_FILLED_CASH_OR_T_PLUS_ONE", "filled_quantity": 69900, "fill_price": 3.416})
    assert saved_fill_exists(row)
    row["filled_quantity"] = 0
    assert not saved_fill_exists(row)
    row["filled_quantity"] = 100
    row["status"] = "NO_TRADE"
    assert not saved_fill_exists(row)
