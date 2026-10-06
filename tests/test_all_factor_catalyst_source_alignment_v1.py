"""验证公布钟、实际日历、共同来源重复和目录缺失的语义。"""
import pandas as pd

from research.all_factor_catalyst_source_alignment_v1 import align_nodes, bind_scores, daily_recorded_view


def calendar():
    dates = pd.to_datetime(["2024-09-27", "2024-09-30", "2024-10-08"])
    return pd.DataFrame({"date": dates.astype("datetime64[ms]"),
        "decision_at": [str(d.date()) + "T15:05:00+08:00" for d in dates]})


def nodes(times, sources=None):
    count = len(times)
    return pd.DataFrame({"node_id": ["节点" + str(i) for i in range(count)], "source_available_upper": times,
        "common_source_id": sources or ["原文" + str(i) for i in range(count)],
        "economic_identity": ["经济身份" + str(i) for i in range(count)],
        "information_role": ["ANNOUNCEMENT"] * count,
        "registered_action": ["RECORDED_SUPPORT_INFORMATION"] * count})


def test_exact_cutoff_is_known_and_after_cutoff_waits_for_real_next_session():
    result = align_nodes(calendar(), nodes(["2024-09-27T15:05:00+08:00", "2024-09-27T15:05:01+08:00", "2024-09-30T16:00:00+08:00"]))
    assert list(result.first_joint_origin) == list(pd.to_datetime(["2024-09-27", "2024-09-30", "2024-10-08"]))


def test_duplicate_nodes_do_not_become_independent_sources():
    result = align_nodes(calendar(), nodes(["2024-09-30T09:00:00+08:00", "2024-09-30T11:00:00+08:00"], ["同一原文", "同一原文"]))
    daily = daily_recorded_view(calendar(), result)
    row = daily.iloc[1]
    assert row.recorded_new_nodes == 2
    assert row.recorded_new_common_source_count == 1
    assert not row.source_count_is_confidence
    assert not row.all_policy_absence_known


def test_weekend_and_calendar_gap_are_not_backdated():
    result = align_nodes(calendar(), nodes(["2024-09-29T12:00:00+08:00", "2024-10-04T12:00:00+08:00", None]))
    assert result.first_joint_origin.iloc[0] == pd.Timestamp("2024-09-30")
    assert result.first_joint_origin.iloc[1] == pd.Timestamp("2024-10-08")
    assert result.joint_mapping_status.iloc[2] == "UNKNOWN_PUBLICATION_CLOCK"


def test_missing_catalog_arrival_never_means_no_policy_and_first_row_is_unknown():
    result = align_nodes(calendar(), nodes(["2024-09-26T12:00:00+08:00"]))
    daily = daily_recorded_view(calendar(), result)
    assert pd.isna(daily.recorded_new_nodes.iloc[0])
    assert daily.recorded_known_nodes.iloc[0] == 1
    assert daily.recorded_new_status.iloc[1] == "NO_NEW_RECORD_IN_FIXED_CATALOG_NOT_NO_POLICY"
    assert not daily.all_policy_absence_known.any()


def test_saved_no_view_predictions_are_preserved_by_binding():
    result = align_nodes(calendar(), nodes([None]))
    daily = daily_recorded_view(calendar(), result)
    prediction = pd.DataFrame({"date": pd.to_datetime(calendar().date).astype("datetime64[ns]"),
        "policy": ["原模型"] * 3, "status": ["NO_VIEW", "AVAILABLE", "NO_VIEW"], "score": [None, 25., None]})
    joined = bind_scores(daily, prediction)
    assert len(joined) == 3
    assert joined.status.eq("NO_VIEW").sum() == 2
    assert joined.score.isna().sum() == 2
    assert joined.numeric_catalyst_score.eq("NOT_COMPUTED_NOT_ADMITTED").all()
