from __future__ import annotations

import pandas as pd
import pytest

from research.daily_01_overnight_absorption_v1_failure_reporting import (
    build_group_support_failure_report,
)


def test_missing_unfavorable_group_is_a_hard_rejection_without_statistics() -> None:
    data = pd.DataFrame(
        {
            "entry_sign_eligible": [True, True, False],
            "absorption_percentile": [0.90, 0.85, 0.10],
            "target_d20_net_excess": [0.01, 0.02, -0.01],
            "signal_action": ["ENTER", "HOLD", "HOLD"],
        }
    )

    report = build_group_support_failure_report(
        data,
        favorable_threshold=0.80,
        unfavorable_threshold=0.20,
        minimum_group_observations=1,
        minimum_mature_enter_events=1,
    )

    assert report["status"] == "REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST"
    assert report["sample"]["favorable_group_observations"] == 2
    assert report["sample"]["unfavorable_group_observations"] == 0
    assert report["block_bootstrap_computed"] is False
    assert report["hac_regression_computed"] is False
    assert report["strategy_backtest_authorized"] is False


def test_failure_report_refuses_to_replace_a_supported_evaluation() -> None:
    data = pd.DataFrame(
        {
            "entry_sign_eligible": [True, True],
            "absorption_percentile": [0.90, 0.10],
            "target_d20_net_excess": [0.01, -0.01],
            "signal_action": ["ENTER", "HOLD"],
        }
    )

    with pytest.raises(RuntimeError, match="两组支持度均已通过"):
        build_group_support_failure_report(
            data,
            favorable_threshold=0.80,
            unfavorable_threshold=0.20,
            minimum_group_observations=1,
            minimum_mature_enter_events=1,
        )
