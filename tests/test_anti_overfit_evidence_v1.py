"""阻止已用历史、事后冻结及零样本被标作独立验证。"""
import numpy as np
import pandas as pd
import pytest

from research.anti_overfit_evidence_inputs_v1 import (
    independent_claim_gate, require_independent_claim, calendar_year_trade_bootstrap,
    point_stats, concentration, paired_block_effect,
)


def evidence():
    return {"candidate_frozen_at": "2026-10-01T23:00:00+08:00",
            "last_outcome_used_for_design": "2026-09-30T15:00:00+08:00",
            "evaluation_first_origin": "2026-10-08T15:05:00+08:00",
            "evaluation_last_origin": "2029-10-08T15:05:00+08:00",
            "actual_decisions_registered_before_execution": True, "full_calendar_coverage": True,
            "same_frozen_version": True, "evaluation_schedule_fixed_before_first_origin": True,
            "required_information_met": True, "uncertainty_gate_passed": True,
            "economic_gate_passed": True, "prospective_completed_cycles": 35,
            "candidate_role": "PREDECLARED_PROSPECTIVE_CANDIDATE", "legacy_terminal_rejection": False}


def test_used_history_cannot_become_independent_by_renaming_holdout_or_green_metrics():
    e = evidence()
    e["evaluation_first_origin"] = "2020-01-02T15:05:00+08:00"
    e.update(point_estimate_pass=True, bootstrap_pass=True, walk_forward=True, label="测试集")
    result = independent_claim_gate(e)
    assert not result["independent_promotion_allowed"]
    assert any("已用于" in x for x in result["reasons"])
    with pytest.raises(ValueError, match="不能宣布"):
        require_independent_claim(e)


def test_terminal_diagnostic_and_zero_complete_cycles_each_block_claim():
    for key, value in [("legacy_terminal_rejection", True), ("candidate_role", "DIAGNOSTIC_ONLY"),
                       ("prospective_completed_cycles", 0), ("evaluation_schedule_fixed_before_first_origin", False),
                       ("full_calendar_coverage", False), ("same_frozen_version", False),
                       ("uncertainty_gate_passed", False), ("economic_gate_passed", False)]:
        e = evidence()
        e[key] = value
        assert not independent_claim_gate(e)["independent_promotion_allowed"]
    assert independent_claim_gate(evidence())["independent_promotion_allowed"]


def test_frozen_time_and_missing_timezone_are_not_accepted_as_timely():
    e = evidence()
    e["candidate_frozen_at"] = e["evaluation_first_origin"]
    assert not independent_claim_gate(e)["independent_promotion_allowed"]
    e = evidence()
    e["candidate_frozen_at"] = "2026-10-01"
    assert not independent_claim_gate(e)["independent_promotion_allowed"]


def toy_trades():
    return pd.DataFrame({"status": ["COMPLETE"] * 4, "net_return": [.5, -.04, .03, -.03],
                         "net_pnl": [5000., -400., 300., -300.],
                         "entry_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2022-01-01", "2022-02-01"]),
                         "exit_date": pd.to_datetime(["2020-01-10", "2020-02-10", "2022-01-10", "2022-02-10"])})


def test_complete_calendar_groups_and_undefined_payoff_are_preserved():
    t = toy_trades()
    report, values = calendar_year_trade_bootstrap(t, 2020, 2022, 500)
    assert report["calendar_groups"] == 3 and report["zero_trade_calendar_groups"] == 1
    assert report["undefined_p_times_b_replications"] > 0
    assert np.isnan(point_stats([.01, .02])[0])
    repeat, values2 = calendar_year_trade_bootstrap(t, 2020, 2022, 500)
    np.testing.assert_allclose(values, values2, equal_nan=True)


def test_profit_concentration_does_not_change_original_ledger_or_create_strategy():
    t = toy_trades()
    before = t.copy(deep=True)
    r = concentration(t)
    pd.testing.assert_frame_equal(t, before)
    assert r["top1_fraction_of_net_profit"] > 1
    assert r["without_largest_cycle_p_times_b"] < 1
    assert "不是新的账户" in r["scope"]


def test_paired_resampling_uses_the_same_time_blocks_for_both_accounts():
    r = np.array([.01, -.015, .004, .008, 0., 0.])
    summary, distribution = paired_block_effect(r, r, block=3, replications=100)
    np.testing.assert_allclose(distribution, 0., atol=1e-15)
    assert summary["sharpe_delta_lower_2_5"] == 0.
