"""严格滞后、固定行业身份、未知与轮动数学的必要检验。"""
from dataclasses import replace

import numpy as np
import pandas as pd

from research.broker_fixed_cohort_inputs_v1 import Panel
from research import industry_structure_description_inputs_v1 as source


def fixture():
    dates = pd.bdate_range("2020-01-01", periods=35)
    symbols = pd.Index([f"{i:06d}.SZ" for i in range(300)])
    daily = np.repeat(np.asarray([.0007 + i * .0002 for i in range(6)]), 50)
    logs = np.tile(np.log1p(daily), (35, 1))
    panel = Panel(dates, symbols, logs, np.ones((35, 300), dtype=bool), np.ones(35, dtype=bool),
        pd.DataFrame(logs).rolling(20, min_periods=20).sum().to_numpy())
    members = pd.DataFrame({"symbol": symbols, "industry_key": np.repeat([f"C{i:02d}" for i in range(1, 7)], 50),
        "major_name": np.repeat([f"示例行业{i}" for i in range(1, 7)], 50), "row_source_known": True})
    slot = {"membership_source_date": dates[23], "snapshot_id": "TEST", "snapshot_taxonomy": "CSRC_QUARTERLY",
        "source_age_calendar_days": 50, "row_source_snapshot_eligible": True}
    return panel, members, slot, np.full(35, np.log1p(.001))


def test_current_and_future_prices_cannot_change_current_industry_features_or_ranked_identity():
    panel, members, slot, etf = fixture()
    full, anchor, stats = source.describe(panel, 24, members, slot, etf)
    mutated = panel.logs.copy()
    mutated[24:] += .5
    changed = replace(panel, logs=mutated, return20=pd.DataFrame(mutated).rolling(20, min_periods=20).sum().to_numpy())
    later_etf = etf.copy()
    later_etf[24:] += .4
    shorter, changed_anchor, changed_stats = source.describe(changed, 24, members, slot, later_etf)
    assert full == shorter and anchor == changed_anchor and stats == changed_stats
    assert anchor.source_idx == 23 and full["leader_ids"] == "C06;C05;C04"


def test_post_anchor_leaders_do_not_change_when_followers_become_the_new_price_winners():
    panel, members, slot, etf = fixture()
    features, anchor, stats = source.describe(panel, 24, members, slot, etf)
    modified = panel.logs.copy()
    modified[24:29, :150] = np.log1p(.1)
    changed = replace(panel, logs=modified)
    profile, groups = source.observe_fixed(changed, anchor, 29, etf)
    assert profile["view_allowed"] and profile["relative_session"] == 5
    assert {g["industry_key"] for g in groups if g["leading"]} == {"C04", "C05", "C06"}
    assert profile["follower_fixed_mean_cumulative_return"] > profile["leader_fixed_mean_cumulative_return"]


def test_missing_post_anchor_quotes_stay_unknown_and_are_not_zero_returns():
    panel, members, slot, etf = fixture()
    features, anchor, stats = source.describe(panel, 24, members, slot, etf)
    missing = panel.logs.copy()
    missing[25, 250:252] = np.nan
    profile, groups = source.observe_fixed(replace(panel, logs=missing), anchor, 29, etf)
    group = next(g for g in groups if g["industry_key"] == "C06")
    assert group["known_members"] == 48 and group["unknown_members"] == 2 and not group["group_view_allowed"]
    assert group["positive_member_lower"] == .96 and group["positive_member_upper"] == 1
    assert not profile["view_allowed"] and profile["descriptive_fixed_state"].startswith("NO_VIEW")


def test_five_day_rank_change_is_zero_when_stable_and_known_when_reversed():
    groups = [{"industry_key": f"C{i:02d}", "major_name": f"示例{i}", "member_count": 5,
        "industry_return20": i / 10, "industry_return5": i / 100, "industry_previous5": i / 100} for i in range(1, 5)]
    stable, rows = source.summarize_groups(groups, 0, 0, 0)
    assert stable["rotation_churn5blocks"] == 0
    reversed_groups = [{**g, "industry_return5": (5 - i) / 100} for i, g in enumerate(groups, 1)]
    moved, rows = source.summarize_groups(reversed_groups, 0, 0, 0)
    assert np.isclose(moved["rotation_churn5blocks"], 2 / 3)
    assert moved["leader_ids"] == stable["leader_ids"] == "C04;C03;C02"
