"""固定仓位比较的前瞻登记、账户重建和一次性期末评价。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import point_account_nr7_inputs_v1 as binary
from research import point_weight_information_inputs_v1 as weighted
from research.anti_overfit_evidence_inputs_v1 import calendar_year_trade_bootstrap, paired_block_effect

POLICIES = ("POINT_BINARY", "SAVED_WEIGHT")
COSTS = ("BASE", "STRESS")


def clock(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("实际时间必须带时区，不能倒填成决策日期。")
    return result.tz_convert("Asia/Shanghai")


def registration_status(origin, execution, generated_at, frozen_at, design_last_close):
    origin, execution, last = map(pd.Timestamp, (origin, execution, design_last_close))
    actual, frozen = clock(generated_at), clock(frozen_at)
    close = origin.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    opened = execution.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    if execution <= origin or origin <= last or frozen >= close:
        return "INVALID_INDEPENDENCE_BOUNDARY"
    if not close <= actual < opened:
        return "LATE_OR_BEFORE_COMPLETED_CLOSE"
    return "TIMELY_FROZEN_ACCOUNT_INPUTS"


def account_registration_coverage(calendar, first_execution, last_close, registrations, protocol):
    dates = pd.DatetimeIndex(calendar)
    if dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("官方交易日历缺失、重复或乱序。")
    first, last = pd.Timestamp(first_execution), pd.Timestamp(last_close)
    if first not in dates or dates.get_loc(first) == 0 or (last >= first and last not in dates):
        raise ValueError("官方日历缺少评价边界或其前一交易日。")
    rows = []
    for day in dates[(dates >= first) & (dates <= last)]:
        origin = dates[dates.get_loc(day) - 1]
        records = [r for r in registrations if pd.Timestamp(r["execution_date"]) == day]
        reason = "MISSING_ACCOUNT_REGISTRATION" if not records else "DUPLICATE_ACCOUNT_REGISTRATION"
        if len(records) == 1:
            record = records[0]
            if pd.Timestamp(record["origin"]) != origin:
                reason = "WRONG_PREVIOUS_EXCHANGE_SESSION"
            elif not record.get("snapshot_verified") or not record.get("same_protocol"):
                reason = "SNAPSHOT_OR_PROTOCOL_MISMATCH"
            elif not record.get("matches_current_causal_prefix"):
                reason = "REGISTERED_INPUT_PREFIX_CHANGED"
            else:
                reason = registration_status(origin, day, record["generated_at"], protocol["frozen_at"],
                                             protocol["design_last_close"])
        rows.append({"execution_date": day, "expected_origin": origin, "reason": reason,
                     "timely": reason == "TIMELY_FROZEN_ACCOUNT_INPUTS"})
    return pd.DataFrame(rows, columns=["execution_date", "expected_origin", "reason", "timely"])


def accounts(data, dividends, parents, risks, start):
    """两方案共享来源和风险；无准备日、突破或新入场条件。"""
    if not data.date.ge(pd.Timestamp(start)).any():
        return {}
    empty = pd.DataFrame({"date": data.date, "entry_event": False, "stop_index": np.nan, "setup_date": pd.NaT})
    results = {}
    for cost in COSTS:
        results[(cost, "POINT_BINARY")] = binary.account(data, dividends, empty, parents, risks, "POINT_A", cost, start)
        results[(cost, "SAVED_WEIGHT")] = weighted.weight_account(data, dividends, parents, risks, cost, start)
    return results


def prefix_check(previous, current):
    """新收盘可结束旧持仓，但不能改写已记账的日收益或已完成交易。"""
    old_daily, new_daily = previous["daily"], current["daily"]
    if len(old_daily) > len(new_daily):
        raise ValueError("前瞻账户日期缩短。")
    pd.testing.assert_frame_equal(old_daily.reset_index(drop=True), new_daily.iloc[:len(old_daily)].reset_index(drop=True),
                                  check_exact=True)
    old_complete = previous["trades"].loc[previous["trades"].status.eq("COMPLETE")]
    if len(old_complete):
        new_complete = current["trades"].loc[current["trades"].cycle_id.isin(old_complete.cycle_id)]
        pd.testing.assert_frame_equal(old_complete.reset_index(drop=True), new_complete.reset_index(drop=True),
                                      check_exact=True, check_dtype=False)


def annual_statistics(result, calendar):
    """只有覆盖官方全年开市日期的年份才进入年均次数，起始残年不能混入。"""
    dates = pd.DatetimeIndex(calendar)
    daily, trades = result["daily"], result["trades"]
    complete = trades.loc[trades.status.eq("COMPLETE")]
    rows = []
    for year, group in daily.groupby(daily.date.dt.year):
        expected = dates[dates.year == year]
        full = bool(len(expected) and pd.DatetimeIndex(group.date).equals(expected))
        count = int(complete.exit_date.dt.year.eq(year).sum()) if len(complete) else 0
        rows.append({"year": int(year), "full_official_year": full, "sessions": len(group),
                     "completed_cycles": count, "net_year_or_partial_return": float(np.prod(1 + group.net_return) - 1)})
    return pd.DataFrame(rows)


def measured_metrics(result, calendar):
    value = binary.metrics(result)
    annual = annual_statistics(result, calendar)
    counts = annual.loc[annual.full_official_year, "completed_cycles"]
    value["average_full_year_cycles"] = float(counts.mean()) if len(counts) else np.nan
    value["zero_trade_full_years"] = int(counts.eq(0).sum())
    value["complete_official_years"] = len(counts)
    return value


def phase_status(observed_sessions, coverage, protocol, terminal_exists=False):
    if terminal_exists:
        return "TERMINAL_RESULT_ALREADY_FROZEN"
    if observed_sessions == 0:
        return "WAITING_FOR_FIRST_NEW_CLOSE"
    if len(coverage) != observed_sessions or not coverage.timely.all():
        return "INELIGIBLE_INCOMPLETE_REGISTRATION"
    if observed_sessions < protocol["terminal_sessions"]:
        return "ACCUMULATING_NO_EARLY_ACCEPTANCE"
    if observed_sessions > protocol["terminal_sessions"]:
        raise ValueError("期末评价不得包含预定终点之后的收益。")
    return "SCHEDULED_TERMINAL_EVALUATION"


def evaluate_terminal(results, coverage, parent_coverage_complete, protocol, calendar):
    """全部通过也只支持这个预定新窗口，不能推出零过拟合或未来收益保证。"""
    n = len(results[("STRESS", "SAVED_WEIGHT")]["daily"])
    if n != protocol["terminal_sessions"]:
        raise ValueError("尚未到唯一预定评价终点，不计算通过结果。")
    valid = phase_status(n, coverage, protocol) == "SCHEDULED_TERMINAL_EVALUATION" and parent_coverage_complete
    details, intervals, metrics = [], [], []
    for cost in COSTS:
        base, trial = results[(cost, "POINT_BINARY")], results[(cost, "SAVED_WEIGHT")]
        am, bm = measured_metrics(base, calendar), measured_metrics(trial, calendar)
        for policy, values in (("POINT_BINARY", am), ("SAVED_WEIGHT", bm)):
            metrics.append({"cost": cost, "policy": policy, **values})
        information = (bm["completed_cycles"] >= protocol["minimum_candidate_cycles"] and
                       bm["wins"] >= protocol["minimum_wins_and_losses"] and
                       bm["losses"] >= protocol["minimum_wins_and_losses"])
        economic = (bm["net_cagr"] > max(0., am["net_cagr"]) and np.isfinite(am["net_sharpe"]) and
                    bm["net_sharpe"] > max(0., am["net_sharpe"]) and bm["p_times_b"] > 1 and
                    bm["mean_cycle_net_return"] > 0 and bm["max_drawdown"] <= .1)
        robust = False
        point_uncertainty = {"status": "NOT_COMPUTED_INVALID_COVERAGE_OR_INFORMATION_FLOOR"}
        if valid and information:
            robust = True
            for block in protocol["paired_blocks"]:
                values, _ = paired_block_effect(base["daily"].net_return, trial["daily"].net_return,
                    block=block, replications=protocol["paired_replications"])
                intervals.append({"cost": cost, **values})
                robust &= (values["undefined_sharpe_difference"] == 0 and values["sharpe_delta_lower_2_5"] > 0 and
                           values["cagr_delta_lower_2_5"] > 0)
            first_year = int(trial["daily"].date.iloc[0].year)
            last_year = int(trial["daily"].date.iloc[-1].year)
            point_uncertainty, _ = calendar_year_trade_bootstrap(trial["trades"], first_year, last_year,
                                                               replications=protocol["trade_replications"])
            robust &= (point_uncertainty["undefined_p_times_b_replications"] == 0 and
                       point_uncertainty["p_times_b_lower_2_5"] > 1 and point_uncertainty["mean_cycle_return_lower_2_5"] > 0)
        details.append({"cost": cost, "information_floor_met": bool(information), "economic_requirements_met": bool(economic),
                        "descriptive_uncertainty_screen_met": bool(robust), "trade_uncertainty": point_uncertainty})
    info = all(x["information_floor_met"] for x in details)
    econ = all(x["economic_requirements_met"] for x in details)
    robust = all(x["descriptive_uncertainty_screen_met"] for x in details)
    status = ("INVALID_FORWARD_EVIDENCE" if not valid else "INCONCLUSIVE_INFORMATION_FLOOR_NOT_MET" if not info else
              "REJECTED_FROZEN_ECONOMIC_REQUIREMENTS" if not econ else "INCONCLUSIVE_UNCERTAINTY_SCREEN" if not robust
              else "FIXED_NEW_WINDOW_INCREMENT_SUPPORTED")
    return {"status": status, "fixed_window_supported": bool(valid and info and econ and robust),
            "metrics": metrics, "details": details, "paired_intervals": intervals,
            "intervals_are_exact_coverage_guarantees": False, "absolute_zero_overfitting_established": False,
            "legacy_rejections_reversed": False, "goal_achieved": False}
