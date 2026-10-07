"""验证真实路径归因保留亏损/未完成，且不把信号等同成交。"""
import pandas as pd
import pytest

from research.all_factor_joint_path_attribution_v1 import bridge, cycle_gap, disposition


def saved_account(net, gross, commission, slippage, rows):
    trades = pd.DataFrame(rows, columns=["cycle_id", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return", "source"])
    for column in ["entry_origin", "entry_date", "exit_date"]:
        trades[column] = pd.to_datetime(trades[column])
    daily = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "equity": [200000 + net],
        "price_pnl": [gross], "dividend_accrual": [0.], "commission": [commission], "slippage": [slippage]})
    return {"daily": daily, "orders": pd.DataFrame(), "trades": trades}


def test_loss_and_open_cycle_are_kept_in_two_exact_bridges():
    main = saved_account(-8, -3, 2, 3, [
        [1, "2023-12-27", "2023-12-28", "2024-01-02", "COMPLETE", -10, -.01, "FIRST_PASSAGE"],
        [2, "2024-01-01", "2024-01-02", None, "RIGHT_CENSORED", None, None, "FIRST_PASSAGE"]])
    other = saved_account(14, 20, 2, 4, [
        [1, "2023-12-26", "2023-12-27", "2024-01-02", "COMPLETE", 15, .01, "CORE_WEIGHT"],
        [2, "2023-12-27", "2023-12-28", "2024-01-02", "COMPLETE", -5, -.005, "CORE_WEIGHT"]])
    gaps = cycle_gap(main, other)
    assert len(gaps) == 3
    assert gaps.joint_status.eq("RIGHT_CENSORED").sum() == 1
    totals = bridge(main, other, gaps)
    assert totals["net_wealth_gap_cny"] == -22
    assert totals["minus_control_only_closed_net_component"] == -15
    assert totals["matched_entry_closed_net_difference"] == -5
    assert totals["open_and_residual_net_difference"] == -2
    assert totals["net_fee_savings_cny"] == 1
    assert totals["cycle_bridge_error_cny"] == pytest.approx(0)
    assert totals["gross_fee_bridge_error_cny"] == pytest.approx(0)


def test_rejected_saved_account_identity_is_not_hidden():
    bad = saved_account(5, 20, 1, 2, [])
    good = saved_account(0, 0, 0, 0, [])
    with pytest.raises(RuntimeError, match="没有闭合"):
        bridge(bad, good, cycle_gap(bad, good))


def test_quality_permission_existing_position_and_execution_are_different():
    assert disposition(True, 1000, "FIRST_PASSAGE_KEEP_FIXED_STRUCTURE", 1000, False, True, "") == "QUALITY_TRUE_ALREADY_HOLDING"
    assert disposition(True, 0, "FIRST_PASSAGE_ESTIMATED_QUALITY_ENTRY", 1000, True, True, "") == "NEW_CYCLE_EXECUTED"
    assert disposition(True, 0, "FIRST_PASSAGE_ESTIMATED_QUALITY_ENTRY", 1000, False, True, "CASH_OR_OPEN_RISK_CAP") == "QUALITY_TRUE_EXECUTION_REJECTED"
    assert disposition(True, 0, "MISSING_PRIOR_RISK_ESTIMATE", 0, False, True, "") == "QUALITY_TRUE_MISSING_RISK"


def test_duplicate_real_entry_keys_cannot_silently_pair():
    row = [1, "2023-12-27", "2023-12-28", "2024-01-02", "COMPLETE", 0., 0., "CORE_WEIGHT"]
    duplicate = saved_account(0, 0, 0, 0, [row, row])
    empty = saved_account(0, 0, 0, 0, [])
    with pytest.raises(RuntimeError, match="不能强行配对"):
        cycle_gap(duplicate, empty)
