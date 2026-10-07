"""固定多头点位的月度追加、真实登记时钟及交易频率统计。"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from research import point_monthly_model_inputs_v1 as training
from research.historic_cycles_point_translation_v1 import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def append_months(data, samples, ordinary, within, cfg31, cfg114):
    """沿原制度追加新月记录；不读写旧研究目录，不重新拟合既有模型。"""
    old_schedule = [r["fit_index"] for r in ordinary]
    require(old_schedule == [r["fit_index"] for r in within], "两类旧模型日历不同。")
    schedule = training.monthly_schedule(data, cfg31["earlier_start"])
    require(schedule[:len(ordinary)] == old_schedule, "行情改变了既有拟合日历。")
    for key in ("recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_clip", "ridge_alpha"):
        require(cfg31[key] == cfg114[key], "两类原训练制度不同：" + key)
    result31, result114 = copy.deepcopy(ordinary), copy.deepcopy(within)
    receipts, membership = [], {}
    for t in schedule[len(ordinary):]:
        rows, ids = training.training_rows(samples, t, cfg31)
        require(rows.empty or rows.exit_index.le(t).all(), "拟合读取未自然成熟的标签。")
        eligible = len(ids) >= cfg31["minimum_cycles"] and len(rows) >= cfg31["minimum_rows"]
        basic = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()),
                 "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                 "status": "FIT_COMPLETE" if eligible else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS",
                 "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
                 "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                 "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None}
        result31.append({"signal": "D60_INTRA", "kind": "RIDGE", **basic,
                         "model": training.fit_one(rows, "RIDGE", cfg31) if eligible else None})
        status, failure, model = basic["status"], None, None
        missing = int((~np.isfinite(rows[training.FEATURES].to_numpy(float)).all(axis=1)).sum())
        if eligible and missing:
            status = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        elif eligible:
            try:
                model = training.fit_within_cycle_exit(rows, cfg114)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", str(error)
        result114.append({**basic, "status": status, "eligible_for_fit": eligible, "failure": failure,
                          "missing_feature_rows": missing, "model": model})
        for kind, record in (("ordinary", result31[-1]), ("within", result114[-1])):
            receipts.append({"kind": kind, **{k: v for k, v in record.items()
                                             if k not in ("model", "training_cycles", "kind", "signal")}})
        membership[basic["fit_origin"]] = rows
    require(result31[:len(ordinary)] == ordinary and result114[:len(within)] == within, "旧模型记录被改写。")
    return result31, result114, pd.DataFrame(receipts), membership


def local_clock(value):
    result = pd.Timestamp(value)
    require(result.tzinfo is not None, "真实登记时间缺少时区。")
    return result.tz_convert("Asia/Shanghai")


def append_registry(registry, signals, points, previous, generated_at, metadata):
    """每个候选和计划日仅登记一次；补历史不能倒填生成时间。"""
    clock = local_clock(generated_at)
    rows = []
    for s in signals.loc[signals.origin.gt(pd.Timestamp(previous))].itertuples():
        duplicate = registry.candidate.eq(s.candidate) & registry.planned_execution_date.eq(s.execution_date)
        require(not duplicate.any(), "既有意向不可覆盖。")
        positions = points.loc[points.candidate.eq(s.candidate) & points.entry_date.le(s.origin)
                               & (points.exit_date.isna() | points.exit_date.gt(s.origin))]
        require(len(positions) <= 1, "固定点位出现重叠持仓。")
        held, known = len(positions) == 1, np.isfinite(s.target)
        action = ("UNKNOWN_KEEP_POSITION" if not known else "HOLD_EXISTING_POINT" if held and s.target > 0
                  else "EXIT_NEXT_OPEN" if held else "ENTER_NEXT_OPEN" if s.target > 0 else "FLAT_WAIT")
        earliest = s.origin.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        deadline = s.execution_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        timely = earliest <= clock < deadline
        rows.append({**metadata, "candidate": s.candidate, "direction": "LONG", "origin": s.origin,
                     "actual_generated_at": clock.isoformat(), "planned_execution_date": s.execution_date,
                     "target": s.target, "core_target": s.core_target, "effective_auxiliary": s.effective_auxiliary,
                     "signal_status": "AVAILABLE" if known else "NO_VIEW", "entry_or_exit_intent": action,
                     "point_state_before": "HOLDING_POINT" if held else "FLAT",
                     "eligibility": "PREOPEN_RESEARCH_INTENT" if timely and known else "LATE_BACKFILL_OR_NO_VIEW",
                     "orders_authorized": False})
    extra = pd.DataFrame(rows)
    result = pd.concat([registry, extra], ignore_index=True) if len(extra) else registry.copy()
    require(not result.duplicated(["candidate", "planned_execution_date"]).any(), "意向重复。")
    return result, extra


def coverage_rows(signals, registry, first_execution, last_close):
    """覆盖所有已发生的开盘，包括空仓日，防止只登记有利交易。"""
    require(not registry.duplicated(["candidate", "planned_execution_date"]).any(), "登记日期重复。")
    lookup = registry.set_index(["candidate", "planned_execution_date"])
    rows = []
    for s in signals.loc[signals.execution_date.between(pd.Timestamp(first_execution), pd.Timestamp(last_close))].itertuples():
        reason = "MISSING_REGISTRATION"
        if (s.candidate, s.execution_date) in lookup.index:
            r = lookup.loc[(s.candidate, s.execution_date)]
            try:
                actual = local_clock(r.actual_generated_at)
                earliest = s.origin.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
                deadline = s.execution_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
                timely = earliest <= actual < deadline
                same = (pd.Timestamp(r.origin) == s.origin and pd.Timestamp(r.source_data_max_date) <= s.origin
                        and r.signal_status == "AVAILABLE" and np.isfinite(r.target) and np.isfinite(s.target)
                        and np.isclose(r.target, s.target, rtol=0, atol=1e-12))
                reason = "TIMELY_MATCHING_RECORD" if timely and same else "LATE_UNKNOWN_OR_DIFFERENT_SIGNAL"
            except (ValueError, TypeError):
                reason = "INVALID_REGISTRATION_CLOCK"
        rows.append({"candidate": s.candidate, "execution_date": s.execution_date,
                     "origin": s.origin, "reason": reason, "timely_matching": reason == "TIMELY_MATCHING_RECORD"})
    return pd.DataFrame(rows, columns=["candidate", "execution_date", "origin", "reason", "timely_matching"])


def classify_points(points, signals, registry, first_execution, last_close):
    coverage = coverage_rows(signals, registry, first_execution, last_close)
    result = points.copy()
    classes = []
    for p in result.itertuples():
        if p.entry_date < pd.Timestamp(first_execution):
            classes.append("HISTORICAL_OR_INHERITED")
            continue
        end = p.exit_date if p.status == "COMPLETE" else pd.Timestamp(last_close)
        period = coverage.loc[coverage.candidate.eq(p.candidate) & coverage.execution_date.between(p.entry_date, end)]
        expected = signals.loc[signals.candidate.eq(p.candidate) & signals.execution_date.between(p.entry_date, end)]
        eligible = (len(period) > 0 and len(period) == len(expected) and period.execution_date.iloc[0] == p.entry_date
                    and period.execution_date.iloc[-1] == end and period.timely_matching.all())
        classes.append(("TIMELY_COMPLETE_POINT" if p.status == "COMPLETE" else "TIMELY_UNFINISHED_POINT")
                       if eligible else "LATE_OR_INCOMPLETE_REGISTRATION")
    result["observation_class"] = classes
    metrics = []
    for candidate in signals.candidate.unique():
        local = result.loc[result.candidate.eq(candidate)]
        cov = coverage.loc[coverage.candidate.eq(candidate)]
        for cohort in ("ALL_HISTORY", "TIMELY_COMPLETE_POINTS"):
            sample = local.loc[local.status.eq("COMPLETE")]
            if cohort == "TIMELY_COMPLETE_POINTS":
                sample = sample.loc[sample.observation_class.eq("TIMELY_COMPLETE_POINT")]
            metrics.append({"candidate": candidate, "cohort": cohort, **statistics(sample.point_net_return),
                            "unregistered_or_invalid_sessions": int((~cov.timely_matching.astype(bool)).sum()),
                            "continuous_forward_coverage": bool(len(cov) and cov.timely_matching.all()),
                            "independent_validation": "NOT_ESTABLISHED"})
    return result, coverage, pd.DataFrame(metrics)


def frequency_tables(points, data, start):
    """按退出年数完整点位；空仓段含起始/末尾，退出日至下次入场日左闭右开。"""
    dates = pd.DatetimeIndex(data.loc[data.date.ge(pd.Timestamp(start)), "date"])
    require(len(dates) > 0, "频率统计缺少交易日。")
    annual, gaps, summary = [], [], []
    for candidate in points.candidate.unique():
        local = points.loc[points.candidate.eq(candidate)].sort_values("entry_date")
        complete = local.loc[local.status.eq("COMPLETE")]
        occupied = np.zeros(len(dates), bool)
        for p in local.itertuples():
            occupied |= (dates >= p.entry_date) & ((dates < p.exit_date) if p.status == "COMPLETE" else True)
        flat_indices = np.flatnonzero(~occupied)
        runs = np.split(flat_indices, np.flatnonzero(np.diff(flat_indices) > 1) + 1)
        own_gaps = []
        for indices in runs:
            if not len(indices):
                continue
            a, b = int(indices[0]), int(indices[-1])
            end = dates[b+1] if b+1 < len(dates) else dates[b]
            record = {"candidate": candidate, "flat_start": dates[a], "last_flat_session": dates[b],
                      "end_date": end, "flat_sessions": len(indices), "calendar_days": int((end-dates[a]).days),
                      "boundary": "ONGOING_AT_CUTOFF" if b == len(dates)-1 else "ENDED_AT_NEXT_ENTRY",
                      "starts_at_study_boundary": a == 0}
            gaps.append(record); own_gaps.append(record)
        counts = []
        for year in range(dates[0].year, dates[-1].year+1):
            full = year < dates[-1].year
            count = int(complete.exit_date.dt.year.eq(year).sum())
            annual.append({"candidate": candidate, "year": year, "completed_points": count, "complete_year": full})
            if full:
                counts.append(count)
        longest = max(own_gaps, key=lambda r: r["flat_sessions"]) if own_gaps else None
        summary.append({"candidate": candidate, "average_full_year_count": float(np.mean(counts)) if counts else np.nan,
                        "zero_trade_full_years": sum(n == 0 for n in counts),
                        "longest_flat_sessions": longest["flat_sessions"] if longest else 0,
                        "longest_flat_calendar_days": longest["calendar_days"] if longest else 0,
                        "longest_flat_start": longest["flat_start"] if longest else pd.NaT,
                        "longest_flat_end": longest["end_date"] if longest else pd.NaT,
                        "total_flat_sessions": int((~occupied).sum()), "total_observed_sessions": len(dates)})
    return pd.DataFrame(annual), pd.DataFrame(gaps), pd.DataFrame(summary)
