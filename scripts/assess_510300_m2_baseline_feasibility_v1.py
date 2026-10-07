"""按冻结协议计算M2同制度历史基线的日历边界，不计算新信号或收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/research/510300_pressure_recovery_v1"
OUT = REPORTS / "baseline_feasibility_20261001"
CONFIG = ROOT / "config/510300_pressure_recovery_v1.json"
CALENDAR = ROOT / "data/reference/sse_trade_calendar_2026.csv"
CALENDAR_METADATA = ROOT / "data/reference/sse_trade_calendar_2026.metadata.json"
GRID = REPORTS / "measurement_correction_20261001/01_五分钟消息测量修正版.parquet"
EVENTS = REPORTS / "local_exploration_20261001/02_固定观察事件表.csv"
CLOCK = REPORTS / "order_lineage_20261001/quote_clock_summary.json"
ADMISSION = REPORTS / "order_lineage_20261001/08_原事件使用状态.csv"
MINUTE = ROOT / "data/raw/510300_free_channels_v1/20261001/neigezhu/data/etf_1m/SH/510300.parquet"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(name: str, value: dict) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_calendar(calendar: pd.DataFrame, metadata: dict) -> list[str]:
    """逐日按工作日与已存官方休市区间重算，避免仅依赖现成日期序号。"""
    if digest(CALENDAR) != metadata["sha256"]:
        raise ValueError("本地日历与其来源记录不一致。")
    actual = calendar.trade_date.astype(str).tolist()
    if actual != sorted(set(actual)):
        raise ValueError("本地日历未排序或包含重复日期。")
    closures = [
        (date.fromisoformat(item["start"]), date.fromisoformat(item["end"]))
        for item in metadata["official_closures"]
    ]
    start = date.fromisoformat(metadata["coverage_start"])
    end = date.fromisoformat(metadata["coverage_end"])
    expected = []
    current = start
    while current <= end:
        if current.weekday() < 5 and not any(low <= current <= high for low, high in closures):
            expected.append(current.isoformat())
        current += timedelta(days=1)
    if actual != expected or len(actual) != metadata["trading_day_count"]:
        raise ValueError("逐日日历重算与已存日期不一致。")
    return actual


def main() -> None:
    if OUT.exists():
        raise SystemExit("本批准入边界已存在，拒绝覆盖。")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(CALENDAR_METADATA.read_text(encoding="utf-8"))
    clock = json.loads(CLOCK.read_text(encoding="utf-8"))
    if clock["source_receive_time_proven"] or clock["original_m1_m2_admission"] != "NO_VIEW":
        raise ValueError("时点准入状态已变化，应重新检查当前输入。")
    days = validate_calendar(pd.read_csv(CALENDAR), metadata)
    required = int(config["measurement"]["baseline_previous_qualified_days"])
    minimum_evaluation_days = int(config["evaluation"]["minimum_independent_days_for_interval"])
    regime_start = config["m1"]["regime_start"]
    if regime_start != "2026-07-06" or config["measurement"]["baseline_slot"] != "exact_HH_MM_within_same_market_regime":
        raise ValueError("冻结协议的制度或同刻定义已变化，不能沿用本次解释。")
    post_days = [day for day in days if day >= regime_start]
    if len(post_days) < required + minimum_evaluation_days:
        raise ValueError("已存日历不足，禁止按普通工作日外推。")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    as_of = now.date().isoformat()
    if not days[0] <= as_of <= days[-1]:
        raise ValueError("当前日期超出日历覆盖，不能给出截至当前的计数。")
    earliest = post_days[required]
    theoretical_interval_date = post_days[required + minimum_evaluation_days - 1]
    grid = pd.read_parquet(GRID)
    grid["trade_date"] = grid.trade_date.astype(str)
    grid["slot"] = pd.to_datetime(grid.decision_at).dt.strftime("%H:%M")
    source_days = sorted(grid.trade_date.unique().tolist())
    fixed = pd.read_csv(EVENTS)
    admission = pd.read_csv(ADMISSION)
    if set(fixed.event_id) != set(admission.event_id) or not admission.formal_m2_status.eq("NO_VIEW").all():
        raise ValueError("原事件集合或准入状态已变化。")
    nominal_valid = grid.measurement_status.eq("SOURCE_QUOTE_MEASUREMENT_ONLY") & grid.log_mid_return_5m_bps.notna()
    minute = pd.read_parquet(MINUTE, columns=["timestamp"])
    minute_days = sorted(pd.to_datetime(minute.timestamp).dt.strftime("%Y-%m-%d").unique().tolist())
    inputs = [CONFIG, CALENDAR, CALENDAR_METADATA, GRID, EVENTS, CLOCK, ADMISSION, MINUTE,
              ROOT / "research/pressure_recovery_v1.py", Path(__file__)]
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": digest(path)} for path in inputs]
    save_json("input_receipt.json", {
        "recorded_at": now.isoformat(),
        "purpose": "冻结M2协议的数据可行性推导；不是新策略注册或收益检验。",
        "rule_basis": "同制度、严格早于当天、相同HH:MM的60个合格交易日。",
        "files": sources,
    })
    daily_rows = []
    for day in source_days:
        available_before = [prior for prior in source_days if regime_start <= prior < day]
        upper = sum(prior < day for prior in post_days)
        daily_rows.append({
            "trade_date": day,
            "same_regime_prior_market_days_upper_bound": upper,
            "required_prior_qualified_days": required,
            "calendar_shortfall_even_if_all_data_available": max(0, required - upper),
            "prior_local_l2_source_dates": len(available_before),
            "prior_proven_qualified_l2_days": 0,
            "calendar_warmup_pass": upper >= required,
            "formal_m2_status": "NO_VIEW",
            "matched_microstructure_increment_status": "NOT_RUN_NO_ADMITTED_M2_EVENTS",
        })
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / "01_逐日基线可行性.csv", index=False, encoding="utf-8-sig")
    event_rows = []
    for event in fixed.itertuples(index=False):
        day = event.trade_date
        slot = pd.Timestamp(event.shock_at).strftime("%H:%M")
        candidates = grid.loc[
            nominal_valid & grid.trade_date.ge(regime_start) & grid.trade_date.lt(day) & grid.slot.eq(slot)
        ]
        upper = sum(prior < day for prior in post_days)
        event_rows.append({
            "event_id": event.event_id,
            "trade_date": day,
            "shock_slot": slot,
            "same_regime_prior_market_days_upper_bound": upper,
            "required_prior_qualified_days": required,
            "nominal_same_slot_previous_l2_days": int(candidates.trade_date.nunique()),
            "proven_qualified_same_slot_days": 0,
            "calendar_warmup_pass": upper >= required,
            "price_return_volatility_quintile_matching": "NOT_RUN_NO_ADMITTED_M2_EVENTS",
            "formal_m2_status": "NO_VIEW",
            "new_trade_or_return": False,
        })
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(OUT / "02_原事件基线与对照状态.csv", index=False, encoding="utf-8-sig")
    calendar_rows = [{
        "trade_date": day,
        "prior_same_regime_market_days": index,
        "calendar_only_warmup_pass": index >= required,
        "date_has_elapsed_as_of_receipt": day <= as_of,
        "local_l2_source_present": day in source_days,
        "local_minute_source_present": day in minute_days,
        "market_day_is_not_proof_of_qualified_observation": True,
    } for index, day in enumerate(post_days)]
    pd.DataFrame(calendar_rows).to_csv(OUT / "03_同制度日历边界.csv", index=False, encoding="utf-8-sig")
    if set(event_frame.event_id) != set(fixed.event_id) or event_frame.calendar_warmup_pass.any():
        raise AssertionError("当前源日期与预期的日历限制不一致，应重新判定结果。")
    if sum(day < earliest for day in post_days) != required:
        raise AssertionError("历史基线错误地包含了当日。")
    summary = {
        "status": "POST_REGIME_M2_CALENDAR_WARMUP_IMPOSSIBLE_FOR_CURRENT_L2_DATES",
        "as_of_date": as_of,
        "same_regime_start": regime_start,
        "required_prior_qualified_days": required,
        "earliest_calendar_possible_m2_date": earliest,
        "earliest_date_is_not_signal_or_data_qualification": True,
        "calendar_possible_evaluation_days_elapsed": sum(earliest <= day <= as_of for day in post_days),
        "minimum_independent_days_for_interval": minimum_evaluation_days,
        "earliest_calendar_possible_thirtieth_evaluation_day": theoretical_interval_date,
        "thirtieth_day_is_only_if_every_prior_evaluation_day_had_an_admissible_independent_event": True,
        "three_source_days": daily_rows,
        "original_events_retained": len(event_frame),
        "original_events_failing_even_calendar_upper_bound": int((~event_frame.calendar_warmup_pass).sum()),
        "minute_rows": len(minute),
        "minute_distinct_days": len(minute_days),
        "minute_last_date": minute_days[-1],
        "minute_days_in_current_regime": sum(day >= regime_start for day in minute_days),
        "minute_days_on_or_after_first_calendar_possible_m2_date": sum(day >= earliest for day in minute_days),
        "matched_increment_status": "NOT_RUN_NO_ADMITTED_M2_EVENTS",
        "microstructure_increment_is_zero": None,
        "qualified_m2_signals": 0,
        "net_expectancy": "NOT_COMPUTED",
        "full_account_sharpe": "NOT_COMPUTED",
        "scope": "仅判定冻结M2规则下2026-07-06后的样本；M1纯让步不因此新增60日门槛。",
        "future_input_requirements": [
            "M2须覆盖评价日前同制度的60个合格同刻观测，评价日期不能早于日历边界。",
            "过去及评价日均须有时间可核验的ETF盘口、同步参考价值及误差说明。",
            "价格对照使用过去信息形成的同刻跌幅与波动分组，不利用后来收益选对照。",
            "收益阶段须有实际或独立验证的成交和次日退出证据。",
        ],
        "new_network_requests": 0,
        "actual_orders": 0,
        "goal_achieved": False,
    }
    save_json("summary.json", summary)
    if not all(digest(ROOT / item["path"]) == item["sha256"] for item in sources):
        raise AssertionError("本轮输入在计算期间发生变化。")
    save_json("verification.json", {
        "calendar_matches_saved_official_closure_intervals": True,
        "calendar_matches_saved_source_hash": True,
        "current_day_excluded_from_baseline": True,
        "original_events_retained": True,
        "source_files_unchanged": True,
        "calendar_dates_not_counted_as_qualified_observations": True,
        "historical_price_proxy_not_substituted_for_l2": True,
    })
    save_json("goal_progress.json", {
        "previous_goal_turn_classification": "PROGRESS",
        "current_goal_turn_classification": "PROGRESS",
        "new_evidence": "三个盘口日期在日历上最多有3、27、44个此前同制度交易日；即使补齐这些日期之前全部数据，也不能满足冻结60日基线。",
        "changed_next_action": "将后续数据要求明确为含60个合格历史日及更晚评价日的连续时段，不能靠重复计算现有17事件完成价格对照或成交后收益检验。",
        "remaining_blocker": "现有本地资料没有可准入的同步估值、实际接收时钟及成交验证，且不包含M2日历基线成熟后的盘口评价日。",
        "goal_achieved": False,
        "goal_should_remain_active": True,
        "next_formal_evaluation_requires_changed_input_evidence": True,
    })
    index = [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(OUT.iterdir()) if path.is_file()]
    pd.DataFrame(index).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    size = sum(path.stat().st_size for path in OUT.iterdir() if path.is_file())
    print(json.dumps({"完成": summary, "新增结果字节": size}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
