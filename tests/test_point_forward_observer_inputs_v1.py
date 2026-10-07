"""登记缺口不能变成前瞻证据；频率包含长等待和自然年度。"""
import numpy as np
import pandas as pd

from research import point_forward_observer_inputs_v1 as study


def example():
    dates = pd.bdate_range("2024-03-04", periods=7)
    signals = pd.DataFrame({"candidate": "多头样例", "origin": dates[:-1], "execution_date": dates[1:],
                            "target": [0., 1., 1., 0., 0., 0.], "core_target": 0., "effective_auxiliary": 0.})
    registry = pd.DataFrame({"candidate": signals.candidate, "origin": signals.origin,
                             "source_data_max_date": signals.origin, "planned_execution_date": signals.execution_date,
                             "actual_generated_at": [d.tz_localize("Asia/Shanghai").isoformat() for d in dates[:-1]+pd.Timedelta(hours=16)],
                             "signal_status": "AVAILABLE", "target": signals.target})
    points = pd.DataFrame({"candidate": ["多头样例"], "entry_date": [dates[2]], "exit_date": [dates[4]],
                           "status": ["COMPLETE"], "point_net_return": [-.03]})
    return dates, signals, registry, points


def test_timely_losing_point_is_counted_and_no_entry_days_are_not_trades():
    dates, signals, registry, points = example()
    result, coverage, metrics = study.classify_points(points, signals, registry, dates[1], dates[-1])
    assert result.observation_class.iloc[0] == "TIMELY_COMPLETE_POINT"
    assert coverage.timely_matching.all() and len(coverage) == 6
    forward = metrics.loc[metrics.cohort.eq("TIMELY_COMPLETE_POINTS")].iloc[0]
    assert forward.n == 1 and forward.losses == 1 and forward["mean"] == -.03
    assert pd.isna(forward.b) and not forward.point_estimate_pass


def test_missing_holding_day_or_late_exit_invalidates_whole_point():
    dates, signals, registry, points = example()
    missing = registry.loc[registry.planned_execution_date.ne(dates[3])]
    result, _, _ = study.classify_points(points, signals, missing, dates[1], dates[-1])
    assert result.observation_class.iloc[0] == "LATE_OR_INCOMPLETE_REGISTRATION"
    registry.loc[registry.planned_execution_date.eq(dates[4]), "actual_generated_at"] = "2024-03-08T09:30:00+08:00"
    result, _, metrics = study.classify_points(points, signals, registry, dates[1], dates[-1])
    assert result.observation_class.iloc[0] == "LATE_OR_INCOMPLETE_REGISTRATION"
    assert metrics.loc[metrics.cohort.eq("TIMELY_COMPLETE_POINTS"), "n"].iloc[0] == 0


def test_missing_flat_day_blocks_continuous_validation_even_if_trade_itself_is_timely():
    dates, signals, registry, points = example()
    registry = registry.loc[registry.planned_execution_date.ne(dates[1])]
    result, _, metrics = study.classify_points(points, signals, registry, dates[1], dates[-1])
    assert result.observation_class.iloc[0] == "TIMELY_COMPLETE_POINT"
    assert not metrics.continuous_forward_coverage.any()
    assert metrics.unregistered_or_invalid_sessions.eq(1).all()


def test_inherited_and_unfinished_points_do_not_enter_completed_forward_metrics():
    dates, signals, registry, points = example()
    inherited, _, _ = study.classify_points(points, signals, registry, dates[3], dates[-1])
    assert inherited.observation_class.iloc[0] == "HISTORICAL_OR_INHERITED"
    points.loc[0, "exit_date"] = pd.NaT
    points.loc[0, "status"] = "RIGHT_CENSORED"
    points.loc[0, "point_net_return"] = np.nan
    result, _, metrics = study.classify_points(points, signals, registry, dates[1], dates[-1])
    assert result.observation_class.iloc[0] == "TIMELY_UNFINISHED_POINT"
    assert metrics.n.eq(0).all()


def test_bulk_backfill_preserves_real_generation_time_and_old_registry():
    dates, signals, registry, points = example()
    old = registry.iloc[:1].copy()
    generated = "2024-03-12T08:00:00+08:00"
    result, extra = study.append_registry(old, signals, points, dates[0], generated,
                                         {"source_data_max_date": dates[-2]})
    pd.testing.assert_frame_equal(result[old.columns].iloc[:1], old)
    assert extra.actual_generated_at.eq(generated).all()
    assert extra.eligibility.iloc[:-1].eq("LATE_BACKFILL_OR_NO_VIEW").all()
    assert extra.eligibility.iloc[-1] == "PREOPEN_RESEARCH_INTENT"
    assert len(result) == 6


def test_flat_intervals_include_initial_terminal_and_zero_trade_year():
    data = pd.DataFrame({"date": pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])})
    points = pd.DataFrame({"candidate": ["多头样例"], "entry_date": [data.date.iloc[2]],
                           "exit_date": [data.date.iloc[4]], "status": ["COMPLETE"]})
    annual, gaps, summary = study.frequency_tables(points, data, "2023-12-28")
    assert annual.completed_points.to_list() == [0, 1]
    assert annual.complete_year.to_list() == [True, False]
    assert gaps.flat_sessions.to_list() == [2, 2]
    assert gaps.calendar_days.to_list() == [5, 1]
    assert gaps.boundary.iloc[-1] == "ONGOING_AT_CUTOFF"
    assert summary.total_flat_sessions.iloc[0] == 4
