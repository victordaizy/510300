"""检查财报原始单位、历史可用时间和成分变化，不重新检验旧策略。"""
import numpy as np
import pandas as pd

from research.original_earnings_breadth_v1 import (
    METRICS, aggregate_members, next_session, original_amount, paired_events, symmetric_change,
)


def examples():
    events = pd.DataFrame([
        ["A0", "A", "2024-03-31", "2024-04-29"],
        ["A1", "A", "2025-03-31", "2025-04-29"],
        ["A2", "A", "2025-06-30", "2025-08-25"],
        ["B0", "B", "2024-03-31", "2024-04-29"],
        ["B1", "B", "2025-03-31", "2025-04-29"],
    ], columns=["announcement_id", "ts_code", "report_period", "event_publication_date"])
    values = {"A0": [10, 9, 100, 8, 200], "A1": [20, 18, 110, 16, 210],
              "B0": [20, 18, 100, 16, 200], "B1": [10, 9, 90, 8, 200]}
    facts = pd.DataFrame([{"announcement_id": a, "metric_id": k, "verified_value": v}
                          for a, numbers in values.items() for k, v in zip(METRICS, numbers)])
    sessions = pd.bdate_range("2024-01-01", "2025-12-31")
    return events, facts, sessions


def test_amount_unit_parentheses_and_blank_are_distinct():
    assert original_amount("(3,873)", 1_000_000) == -3_873_000_000
    assert original_amount("2，455，270，965．01", 1) == 2_455_270_965.01
    assert np.isnan(original_amount("[CURRENT_AND_PRIOR_AMOUNT_CELLS_BLANK]", 1))
    assert original_amount("0", 1) == 0


def test_symmetric_growth_handles_losses_without_inverting_improvement():
    result = symmetric_change([10, -5, -20, 0], [-10, -10, -10, 0])
    np.testing.assert_allclose(result, [2, 2 / 3, -2 / 3, 0])


def test_date_only_publication_uses_strictly_next_open_session():
    sessions = pd.DatetimeIndex(["2025-04-25", "2025-04-28", "2025-04-29"])
    values = next_session(pd.Series(pd.to_datetime(["2025-04-25", "2025-04-26", "2025-04-28"])), sessions)
    assert list(values.dt.strftime("%Y-%m-%d")) == ["2025-04-28", "2025-04-28", "2025-04-29"]


def test_new_missing_report_cannot_fallback_to_previous_known_report():
    e, f, sessions = examples()
    paired = paired_events(e, f, sessions)
    members = pd.DataFrame({"membership_date": pd.to_datetime(["2025-04-29", "2025-04-30", "2025-08-26"]), "symbol": ["A"] * 3})
    _, panel = aggregate_members(members, paired, minimum=1)
    assert panel.announcement_id.tolist() == ["A0", "A1", "A2"]
    assert panel.company_valid.tolist() == [False, True, False]
    assert panel.iloc[1].prior_announcement_id == "A0"


def test_membership_change_and_future_append_preserve_point_in_time_features():
    e, f, sessions = examples()
    members = pd.DataFrame({"membership_date": pd.to_datetime(["2025-04-30", "2025-05-01"]), "symbol": ["A", "B"]})
    small, _ = aggregate_members(members, paired_events(e[e.announcement_id != "A2"], f, sessions), minimum=1)
    full, panel = aggregate_members(members, paired_events(e, f, sessions), minimum=1)
    pd.testing.assert_frame_equal(small, full)
    assert panel.ts_code.tolist() == ["A", "B"]
    assert full.profit_improve_share.tolist() == [1, 0]
