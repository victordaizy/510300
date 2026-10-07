"""用官方交易日历补查完整覆盖；按需调用原观察器接续，既有规则和登记保持原状。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES = ("CORE_AUXILIARY_DRAWDOWN_GATE", "LAG_CONFIRMED_RUNS_AUXILIARY")
OBSERVER = ROOT / "reports/research/510300_point_forward_observer_v1"
OUT = ROOT / "reports/research/510300_point_forward_calendar_check_v1"
CALENDAR = ROOT / "data/reference/sse_trade_calendar_2026.csv"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_from_calendar(signals, registry, calendar, first_execution, last_close,
                           calendar_valid_through, candidates=CANDIDATES):
    """先产生应有日期，再核对两张表；两表同时少行也不能缩短分母。"""
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("官方日历含缺失、重复或乱序日期。")
    if dates.tz is not None:
        raise ValueError("交易日须为上海本地日历日期，不接受带时区的日期列。")
    first, last, valid = map(pd.Timestamp, (first_execution, last_close, calendar_valid_through))
    if first > valid or last > valid:
        raise ValueError("官方日历覆盖范围不足，不能截短评价窗口。")
    if first not in dates or dates.get_loc(first) == 0:
        raise ValueError("官方日历缺少评价首个开盘或其上一交易日。")
    if last >= first and last not in dates:
        raise ValueError("行情末日不是已确认的官方交易日。")
    if len(set(candidates)) != len(candidates) or not candidates:
        raise ValueError("候选名单重复或为空。")
    signal_required = {"candidate", "origin", "execution_date", "target"}
    registry_required = {"candidate", "origin", "planned_execution_date", "target",
                         "source_data_max_date", "actual_generated_at", "signal_status"}
    if not signal_required.issubset(signals) or not registry_required.issubset(registry):
        raise ValueError("信号表或真实登记表缺少必要证据列。")
    s, r = signals.copy(), registry.copy()
    for frame, names in ((s, ("origin", "execution_date")),
                         (r, ("origin", "planned_execution_date", "source_data_max_date"))):
        for name in names:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
            if frame[name].isna().any():
                raise ValueError("证据日期缺失或无效：" + name)
    expected = dates[(dates >= first) & (dates <= last)]
    invalid_extras = []
    for frame, key, label in ((s, "execution_date", "信号"), (r, "planned_execution_date", "登记")):
        period = frame.loc[frame[key].between(first, last)]
        if (~period[key].isin(expected) | ~period.candidate.isin(candidates)).any():
            invalid_extras.append(label + "在评价窗口内含非交易日或未登记候选。")
    rows = []
    for day in expected:
        previous = dates[dates.get_loc(day) - 1]
        earliest = previous.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        deadline = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        for candidate in candidates:
            sr = s.loc[s.candidate.eq(candidate) & s.execution_date.eq(day)]
            rr = r.loc[r.candidate.eq(candidate) & r.planned_execution_date.eq(day)]
            reasons = []
            if len(sr) != 1:
                reasons.append("MISSING_SIGNAL" if not len(sr) else "DUPLICATE_SIGNAL")
            if len(rr) != 1:
                reasons.append("MISSING_REGISTRATION" if not len(rr) else "DUPLICATE_REGISTRATION")
            if len(sr) == 1 and len(rr) == 1:
                a, b = sr.iloc[0], rr.iloc[0]
                if a.origin != previous or b.origin != previous:
                    reasons.append("ORIGIN_NOT_PREVIOUS_EXCHANGE_SESSION")
                if b.source_data_max_date > previous:
                    reasons.append("SOURCE_CONTAINS_FUTURE_DATE")
                try:
                    clock = pd.Timestamp(b.actual_generated_at)
                    if pd.isna(clock) or clock.tzinfo is None:
                        raise ValueError("登记缺少真实时区时刻")
                    if not earliest <= clock.tz_convert("Asia/Shanghai") < deadline:
                        reasons.append("LATE_OR_BEFORE_COMPLETED_CLOSE")
                except (ValueError, TypeError):
                    reasons.append("INVALID_ACTUAL_CLOCK")
                try:
                    values = np.asarray([a.target, b.target], dtype=float)
                    same = bool(np.isfinite(values).all() and
                                np.isclose(values[0], values[1], rtol=0, atol=1e-12))
                except (TypeError, ValueError):
                    same = False
                if b.signal_status != "AVAILABLE" or not same:
                    reasons.append("UNKNOWN_OR_DIFFERENT_SIGNAL")
            rows.append({"candidate": candidate, "execution_date": day, "expected_origin": previous,
                         "signal_rows": len(sr), "registration_rows": len(rr),
                         "timely_matching": not reasons, "reason": "|".join(reasons) or "TIMELY_MATCHING_RECORD"})
    columns = ["candidate", "execution_date", "expected_origin", "signal_rows", "registration_rows",
               "timely_matching", "reason"]
    table = pd.DataFrame(rows, columns=columns)
    missing = int((~table.timely_matching.astype(bool)).sum())
    complete = bool(len(expected) and not missing and not invalid_extras)
    return table, {"expected_exchange_sessions": len(expected), "expected_candidate_sessions": len(rows),
                   "invalid_candidate_sessions": missing, "invalid_extra_rows": invalid_extras,
                   "full_calendar_coverage": complete,
                   "status": "NO_NEW_OBSERVATIONS" if not len(expected) else "COMPLETE_CALENDAR_COVERAGE" if complete
                   else "INCOMPLETE_OR_INVALID_COVERAGE", "independent_validation": "NOT_ESTABLISHED"}


def run_check():
    from research.point_forward_observer_v1 import check_program

    check_program()
    state = read_json(OBSERVER / "state.json")
    folder = ROOT / state["version"]
    calendar = pd.read_csv(CALENDAR)
    if set(calendar.calendar_year) != {2026}:
        raise ValueError("本版日历范围声明只覆盖2026年，发现其他年份需另存接续版本。")
    sources = {name: folder / "results" / (name + ".parquet")
               for name in ("完整候选意向", "完整实际登记", "全部自然点位")}
    frames = {name: pd.read_parquet(path) for name, path in sources.items()}
    coverage, result = coverage_from_calendar(frames["完整候选意向"], frames["完整实际登记"],
        calendar.trade_date, state["first_execution"], state["last_known_close"], "2026-12-31")
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    target = OUT / "checks" / now.strftime("%Y%m%d_%H%M%S_%f")
    target.mkdir(parents=True, exist_ok=False)
    coverage.to_parquet(target / "官方日历逐日覆盖.parquet", index=False)
    coverage.to_csv(target / "官方日历逐日覆盖.csv", index=False, encoding="utf-8-sig")
    sources.update({"官方日历": CALENDAR, "观察器协议": OBSERVER / "protocol.json", "本检查程序": Path(__file__)})
    points = frames["全部自然点位"]
    newly_completed = points.loc[points.entry_date.ge(pd.Timestamp(state["first_execution"])) &
                                 points.status.eq("COMPLETE") & points.exit_date.le(pd.Timestamp(state["last_known_close"]))]
    result.update(at=now.isoformat(), source_version=state["version"], first_execution=state["first_execution"], last_close=state["last_known_close"],
        fixed_candidate_count=len(CANDIDATES), raw_newly_completed_point_rows=len(newly_completed),
        point_rows_are_not_independent_samples=True, unchanged_registered_intents=len(frames["完整实际登记"]),
        modified_old_signals=False, modified_old_registration=False, new_accounts=0, new_fits=0,
        periodic_review_role="按每个自然季度最后一个官方交易日整理描述结果；不据季度数字提前宣布通过。",
        independent_acceptance_schedule_status="NOT_PREDECLARED_FOR_ANY_ADMISSIBLE_ACCOUNT_CANDIDATE",
        future_account_validation_status="NOT_ESTABLISHED", goal_achieved=False, orders_authorized=False,
        source_hashes={name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in sources.items()})
    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"检查状态": result["status"], "应有交易日": result["expected_exchange_sessions"],
                      "独立验证": result["independent_validation"], "结果": str(target / "result.json")}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="默认只检查已有记录；--advance先按原规则接纳新完整日线，再检查官方日历覆盖。")
    parser.add_argument("--advance", action="store_true", help="调用原固定观察器接续，再检查；不创建后台任务")
    arguments = parser.parse_args()
    if arguments.advance:
        from research.point_forward_observer_v1 import check_or_update
        check_or_update(True)
    run_check()
