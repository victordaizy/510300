"""多头点位的可重复日线接续入口；只运行研究，不下单、不创建定时任务。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import new_daily_input_adapter_v1 as adapter
from research import point_forward_observer_inputs_v1 as logic
from research import point_monthly_model_inputs_v1 as training
from research import point_core_observation_inputs_v1 as engine
from research import point_close_observation_v1 as points_engine
from research.adaptive_allocation_v1 import normalize_dividends, factors
from research.point_core_observation_v1 import CONFIGS, LAYERS, compare_numeric
from research.point_state_reconstruction_v1 import MODELS

OUT = ROOT / "reports/research/510300_point_forward_observer_v1"
SEED = ROOT / "reports/research/510300_point_current_observation_20261001"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
CALENDAR = ROOT / "data/reference/sse_trade_calendar_2026.csv"
START = "2020-01-02"
NAMES = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}
require = logic.require


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    """兼容reports子目录联接到E盘，保存逻辑工作区路径。"""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(OUT.relative_to(ROOT) / path.resolve().relative_to(OUT.resolve()))


def table(folder, name, data):
    target = folder / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target.with_suffix(".parquet"), index=False)
    data.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def next_session(calendar, last):
    dates = pd.DatetimeIndex(calendar)
    future = dates[dates > pd.Timestamp(last)]
    require(len(future) > 0, "官方日历缺少下一交易日，需先更新日历。")
    return future[0]


def configs(folder):
    return {key: read(folder / f"inputs/config/{key}.json") for key in (*CONFIGS, "learned_cycle_exit", "within_cycle_exit")}


def state_commit(value):
    # 同一目录原子替换指针；失败结果留在各自目录，不覆盖上次成功状态。
    temporary = OUT / "state.pending.json"
    common.save_json(temporary, value)
    temporary.replace(OUT / "state.json")


def check_program():
    protocol = read(OUT / "protocol.json")
    for source, expected in protocol["program_hashes"].items():
        require(common.digest(ROOT / source) == expected, "登记后的程序发生变化，需要保留旧版并另行说明：" + source)


def initialize():
    require(not (OUT / "state.json").exists(), "观察器已初始化，使用check或update。")
    target = OUT / "seed"
    require(not target.exists(), "种子目录已存在，保留原内容。")
    paths = ["inputs/candidate_prices.parquet", "inputs/candidate_features.parquet", "inputs/dividends.csv",
             "inputs/candidate_dividend_coverage.json", "inputs/ordinary_models.json", "inputs/within_models.json",
             "inputs/admission_receipt.json", "results/全部自然点位.parquet", "results/完整候选意向.parquet",
             "results/下次开盘研究意向.parquet", "results/training_reference/samples.parquet",
             "results/BASE/full_factors.parquet", "results/STRESS/full_factors.parquet"]
    paths += [f"inputs/config/{key}.json" for key in (*CONFIGS, "learned_cycle_exit", "within_cycle_exit")]
    for name in paths:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SEED / name, destination)
    registry = pd.read_parquet(target / "results/下次开盘研究意向.parquet")
    table(target, "完整实际登记", registry)
    sources = [Path(__file__), Path(logic.__file__), Path(training.__file__), Path(engine.__file__),
               Path(points_engine.__file__), ROOT / "research/reference_observation_accounts_v1.py",
               ROOT / "research/training_reference_observation_v1.py"]
    protocol = {"study": "510300_POINT_FORWARD_OBSERVER_V1", "at": common.now(), "direction": "LONG_ONLY_CURRENT_PHASE",
                "user_instruction": "如果空头困难可以先只做多头，后面补空头", "short_status": "DEFERRED_BY_USER",
                "rules": "两条固定多头候选、原成本、原月度制度；正目标入场、零目标退出，完整自然点位统计。",
                "monthly": "仅追加新月首已完成收盘；原20个自然成熟周期制度，旧模型不改写。",
                "registration": "实际生成时间须在来源收盘15:05之后、下一开盘09:30之前；迟到或缺失一律保留原状。",
                "coverage": "同时检查空仓日与持仓日。单笔及时登记不等于整体连续前瞻验证；缺失日期不能回填为及时。",
                "history": "种子和工程重放属于已使用历史；不得转为独立验证。未来亏损与盈利同样纳入。",
                "gate": "p×实际净B>1且平均净收益>0；年度次数软目标，不继承旧账户夏普条件。",
                "source_limit": "当前接纳器要求旧股息事件账本未变；发现新增分红或来源失败时停止接纳，不忽略股息继续。",
                "calendar_limit": "当前官方日历至2026年末；日历不足时停止，不能用工作日假设交易日。",
                "execution": "人工调用本程序才运行；不创建后台任务或定时任务。", "orders_authorized": False,
                "program_hashes": {relative(p): common.digest(p) for p in sources},
                "seed_hashes": {name: common.digest(target / name) for name in paths}}
    common.save_json(OUT / "protocol.json", protocol)
    state = {"at": common.now(), "version": relative(target), "last_known_close": "2026-09-30",
             "next_exchange_session": "2026-10-08", "first_execution": str(registry.planned_execution_date.min().date()),
             "status": "INITIALIZED", "new_monthly_fits": 0, "orders_authorized": False, "goal_achieved": False}
    write_results(target, state)
    state_commit(state)
    print("多头接续入口已初始化，保留原10月8日两条空仓等待意向。", flush=True)


def write_results(folder, state):
    data = pd.read_parquet(folder / "inputs/candidate_features.parquet")
    points = pd.read_parquet(folder / "results/全部自然点位.parquet")
    signals = pd.read_parquet(folder / "results/完整候选意向.parquet")
    registry = pd.read_parquet(folder / "results/完整实际登记.parquet")
    classified, coverage, metrics = logic.classify_points(points, signals, registry, state["first_execution"], data.date.iloc[-1])
    annual, gaps, frequency = logic.frequency_tables(points, data, START)
    for name, frame in (("点位证据归属", classified), ("前瞻逐日覆盖", coverage), ("分集合点位指标", metrics),
                        ("逐年完成次数", annual), ("全部空仓间隔", gaps), ("交易频率摘要", frequency)):
        table(folder, name, frame)
    selected = metrics.loc[metrics.cohort.eq("TIMELY_COMPLETE_POINTS")]
    summary = {**state, "at": common.now(), "historical_point_comparisons": int(points.status.eq("COMPLETE").sum()),
               "timely_complete_point_comparisons": int(selected.n.sum()),
               "unregistered_or_invalid_candidate_sessions": int(selected.unregistered_or_invalid_sessions.sum()),
               "forward_evidence": "NOT_ESTABLISHED", "short_status": "DEFERRED_BY_USER"}
    common.save_json(folder / "summary.json", summary)
    lines = ["# 510300先做多头：固定候选与后续观察", "",
             "当前阶段集中于多头，空头按用户要求后置。两条候选的历史点值达线，但前瞻优势尚未建立。", "",
             "胜率p按完整点位净盈利比例计算，B为平均净盈利/平均净亏损绝对值。硬门槛为p×B>1且平均净收益>0。标准亏损单位期望为p×B−q，q为亏损比例。", "",
             "固定约十万元名义份额、单边万四且至少5元佣金、千一滑点和不利刻度；这里只研究标的点位。", "",
             "| 候选 | 完成点位 | 净胜率 | 实际净B | p×B | 平均净收益 | 完整年均次数 | 最长空仓交易日 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    hist = metrics.loc[metrics.cohort.eq("ALL_HISTORY")].merge(frequency, on="candidate")
    for r in hist.itertuples():
        lines.append(f"| {NAMES[r.candidate]} | {r.n} | {r.p:.2%} | {r.b:.4f} | {r.product:.4f} | {r.mean:.2%} | {r.average_full_year_count:.2f} | {r.longest_flat_sessions} |")
    lines += ["", "年度次数按退出所在年统计；当前年未结束，单列。最长空仓按交易日收盘无持仓计算，含初始和末端等待；退出日计空仓，下一入场日不计。", "",
              "| 年份 | 回撤限制组合 | 相关确认组合 |", "|---|---:|---:|"]
    for year in sorted(annual.year.unique()):
        row = annual.loc[annual.year.eq(year)].set_index("candidate")
        label = f"{year}（截至{data.date.iloc[-1]:%m-%d}）" if year == data.date.iloc[-1].year else str(year)
        lines.append(f"| {label} | {int(row.loc['CORE_AUXILIARY_DRAWDOWN_GATE', 'completed_points'])} | {int(row.loc['LAG_CONFIRMED_RUNS_AUXILIARY', 'completed_points'])} |")
    lines += [""]
    for r in frequency.itertuples():
        lines.append(f"{NAMES[r.candidate]}最长空仓段：{r.longest_flat_start:%Y-%m-%d}至{r.longest_flat_end:%Y-%m-%d}，{r.longest_flat_sessions}个交易日、{r.longest_flat_calendar_days}个日历日。")
    lines += ["", "两条候选大量点位重合，不能相加为交易次数。没有通过放宽条件、改阈值或删掉亏损增加表面频率。", "",
              f"最新接纳收盘：{state['last_known_close']}；下一交易日：{state['next_exchange_session']}。下列意向保留真实生成时间：", "",
              "| 候选 | 计划日 | 意向 | 实际生成时间 |", "|---|---|---|---|"]
    for r in registry.sort_values("planned_execution_date").groupby("candidate", sort=False).tail(1).itertuples():
        lines.append(f"| {NAMES[r.candidate]} | {r.planned_execution_date:%Y-%m-%d} | {r.entry_or_exit_intent} | {r.actual_generated_at} |")
    lines += ["", f"完整及时登记的新点位比较数：{summary['timely_complete_point_comparisons']}。两候选同一进出日期仍分别保留，不当成独立样本。",
              "逐日覆盖包含空仓日；若缺失、迟到、信号不一致，该日期不能在以后升级为及时记录。未完成点位不计胜率或B；继承历史持仓也不变成新前瞻交易。", "",
              "接续流程先确认交易日与15:05收盘时钟，再接纳完整日线和股息覆盖，按原制度追加新月份模型，重算固定状态并保留旧前缀，最后登记下一开盘意向。",
              "当前没有定时任务。check只检查是否有新完整收盘；update在有新收盘时采集并推进；verify只做历史工程对应，不产生前瞻盈利证据。", "",
              "新增分红事件、来源不一致或缺少官方日历时停止接纳，当前结论保持明确截止日期；不沿用旧价格冒充新日信号。", ""]
    (folder / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def recompute(previous, inputs, destination, next_date, restore_last_month=False):
    data = pd.read_parquet(inputs / "candidate_features.parquet")
    prices = pd.read_parquet(inputs / "candidate_prices.parquet")
    old = pd.read_parquet(previous / "inputs/candidate_features.parquet")
    old_prices = pd.read_parquet(previous / "inputs/candidate_prices.parquet")
    require(len(data) >= len(old), "新行情比既有行情更短。")
    pd.testing.assert_frame_equal(data.iloc[:len(old)].reset_index(drop=True), old, check_exact=True)
    pd.testing.assert_frame_equal(prices.iloc[:len(old_prices)].reset_index(drop=True), old_prices, check_exact=True)
    dividends = normalize_dividends(pd.read_csv(previous / "inputs/dividends.csv"))
    reconstructed, _ = factors(prices, dividends)
    pd.testing.assert_frame_equal(reconstructed, data, check_exact=True)
    cfg = configs(previous)
    reference = training.reference_samples(data, dividends, cfg["learned_cycle_exit"], next_date)
    before = reference["samples"].loc[reference["samples"].mature_date.le(old.date.iloc[-1])].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, pd.read_parquet(previous / "results/training_reference/samples.parquet"), check_exact=True)
    models = {kind: read(previous / f"inputs/{kind}_models.json")["models"] for kind in ("ordinary", "within")}
    retained = {kind: records[:-1] if restore_last_month else records for kind, records in models.items()}
    m31, m114, fits, memberships = logic.append_months(data, reference["samples"], retained["ordinary"], retained["within"], cfg["learned_cycle_exit"], cfg["within_cycle_exit"])
    if restore_last_month:
        require(common.clean(m31) == models["ordinary"] and common.clean(m114) == models["within"], "月度模型工程复现不一致。")
    continuous = engine.continuous_references(data, data, dividends, cfg["continuous_reference_min_variance"], m31, next_date)
    chain = engine.assemble_chain(data, dividends, cfg, m114, START, next_date, continuous)
    checks = []
    for cost, frame in chain["factors"].items():
        saved = pd.read_parquet(previous / f"results/{cost}/full_factors.parquet")
        for column in LAYERS:
            checks.append(compare_numeric(frame[column].iloc[:len(saved)], saved[column], cost + "/" + column))
        table(destination, cost + "/full_factors", frame)
    point_frames, signal_frames, events = [], [], []
    for candidate in MODELS:
        signals = points_engine.candidate_signals(data, chain["factors"]["STRESS"], candidate, START, next_date)
        points, event = points_engine.observe_points(data, dividends, signals, START)
        for frame in (points, event, signals):
            frame["candidate"] = candidate
        point_frames.append(points); signal_frames.append(signals); events.append(event)
    points, signals = pd.concat(point_frames, ignore_index=True), pd.concat(signal_frames, ignore_index=True)
    saved = pd.read_parquet(previous / "results/全部自然点位.parquet")
    columns = ["candidate", "entry_date", "exit_date", "point_net_return"]
    a = points.loc[points.status.eq("COMPLETE") & points.exit_date.le(old.date.iloc[-1]), columns].reset_index(drop=True)
    b = saved.loc[saved.status.eq("COMPLETE"), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for name, frame in reference.items():
        table(destination, "training_reference/" + name, frame)
    for origin, rows in memberships.items():
        table(destination, "monthly_membership/" + origin, rows)
    for kind, records in (("ordinary", m31), ("within", m114)):
        common.save_json(destination / f"inputs/{kind}_models.json", {"models": records})
    table(destination, "新增月度模型", fits)
    for name, frame in (("全部自然点位", points), ("完整候选意向", signals), ("点位进出事件", pd.concat(events, ignore_index=True))):
        table(destination, name, frame)
    return points, signals, {"monthly_records_appended": len(fits), "old_target_maximum_error": max(checks),
                             "complete_old_points_unchanged": len(a), "internal_reference_replays": 7, "fixed_point_replays": 2}


def check_or_update(update):
    check_program()
    state = read(OUT / "state.json")
    previous = ROOT / state["version"]
    calendar = pd.to_datetime(pd.read_csv(CALENDAR).trade_date)
    dates, cutoff, next_day = adapter.completed_dates(calendar, state["last_known_close"], common.now())
    require(pd.Timestamp(cutoff) >= pd.Timestamp(state["last_known_close"]), "实际时钟早于已接纳收盘。")
    result = {"at": common.now(), "last_known_close": state["last_known_close"], "latest_completed_session": cutoff,
              "next_exchange_session": next_day, "new_completed_sessions": len(dates),
              "status": "NO_NEW_COMPLETED_DAILY_BAR" if len(dates) == 0 else "NEW_COMPLETED_BARS_REQUIRE_ADMISSION",
              "network_requests": 0, "orders_authorized": False}
    common.save_json(OUT / "latest_check.json", result)
    if not update or not len(dates):
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    stamp = pd.Timestamp(common.now()).strftime("%Y%m%dT%H%M%S%f")
    destination = OUT / "runs" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    last_accepted_close = state["last_known_close"]
    try:
        source = {"prices": relative(previous / "inputs/candidate_prices.parquet"),
                  "features": relative(previous / "inputs/candidate_features.parquet"),
                  "dividends": relative(previous / "inputs/dividends.csv"),
                  "coverage": relative(previous / "inputs/candidate_dividend_coverage.json")}
        adapter.relative = relative
        receipt = adapter.collect_and_admit({"official_configuration": "config/510300_official_dividend_coverage_refresh_v1.json"}, source, inputs, dates, 1)
        require(receipt["accepted"] and receipt["cutoff"] == cutoff, "日线尚未接纳至本次完整收盘。")
        for name in ("dividends.csv",):
            shutil.copyfile(previous / "inputs" / name, inputs / name)
        shutil.copytree(previous / "inputs/config", inputs / "config")
        require(common.digest(inputs / "candidate_features.parquet") == receipt["features_sha256"], "接纳特征内容不符。")
        require(common.digest(inputs / "dividends.csv") == receipt["dividends_sha256"], "分红账本内容不符。")
        points, signals, details = recompute(previous, inputs, destination, pd.Timestamp(next_day))
        generated_at = common.now()
        old_registry = pd.read_parquet(previous / "results/完整实际登记.parquet")
        registry, extra = logic.append_registry(old_registry, signals, points, state["last_known_close"], generated_at,
            {"source_data_max_date": pd.Timestamp(cutoff), "inputs_receipt": relative(inputs / "admission_receipt.json"),
             "parent_state_receipt": relative(destination / "results/STRESS/full_factors.parquet"),
             "parent_state_hash": common.digest(destination / "results/STRESS/full_factors.parquet"),
             "reference_cost_basis": "STRESS_FIXED_NOTIONAL_100000_CNY"})
        table(destination, "完整实际登记", registry)
        table(destination, "本次实际登记", extra)
        state.update(at=generated_at, version=relative(destination), last_known_close=cutoff, next_exchange_session=next_day,
                     status="COMPLETE_CLOSE_UPDATED", new_monthly_fits=details["monthly_records_appended"], **details)
        summary = write_results(destination, state)
        state_commit(state)
        print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        common.save_json(destination / "failure.json", {"at": common.now(), "status": "NO_VIEW_UPDATE_NOT_COMMITTED",
            "error": str(error), "last_accepted_close": last_accepted_close, "orders_authorized": False})
        raise


def verify():
    """已有9月模型退回一条再按原样本恢复，检验接续实现，不改变研究状态。"""
    check_program()
    state = read(OUT / "state.json")
    previous = OUT / "seed"
    destination = OUT / "engineering_verification"
    require(not destination.exists(), "工程验证已保存，不重复覆盖。")
    destination.mkdir()
    calendar = pd.to_datetime(pd.read_csv(CALENDAR).trade_date)
    data = pd.read_parquet(previous / "inputs/candidate_features.parquet")
    _, signals, details = recompute(previous, previous / "inputs", destination, next_session(calendar, data.date.iloc[-1]), True)
    pd.testing.assert_frame_equal(signals, pd.read_parquet(previous / "results/完整候选意向.parquet"), check_exact=True)
    result = {"at": common.now(), "status": "PASS_ORIGINAL_MONTHLY_APPEND_AND_COMPLETE_POINT_RECOMPUTATION", **details,
              "evidence_class": "ENGINEERING_CHECK_ON_ALREADY_USED_HISTORY", "new_prospective_points": 0,
              "active_state_unchanged": state == read(OUT / "state.json"), "orders_authorized": False}
    common.save_json(destination / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="510300多头点位日线接续，不调用旧账户入口。")
    parser.add_argument("action", choices=("initialize", "check", "update", "verify"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "PROCESS.lock"
    # 防止两次手动调用同时写指针；进程异常终止时保留锁，先检查原因再处理。
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(common.now())
    try:
        if args.action == "initialize":
            initialize()
        elif args.action == "verify":
            verify()
        else:
            check_or_update(args.action == "update")
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
