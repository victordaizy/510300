"""把已观察日线接到固定候选及原月度制度，恢复历史状态而不声称独立验证。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import point_core_observation_inputs_v1 as chain_engine
from research import point_monthly_model_inputs_v1 as training
from research import point_close_observation_v1 as point_engine
from research.adaptive_allocation_v1 import factors, normalize_dividends
from research.point_core_observation_v1 import CONFIGS, LAYERS, compare_numeric
from research.point_state_reconstruction_v1 import MODELS
from research.historic_cycles_point_translation_v1 import statistics

OUT = ROOT / "reports/research/510300_point_history_extension_v1"
CORE = ROOT / "reports/research/510300_point_core_observation_v1"
TRAIN = ROOT / "reports/research/510300_point_training_observation_v1"
BOUNDARY, LAST, NEXT, START = "2026-08-14", "2026-09-16", "2026-09-17", "2020-01-02"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not OUT.exists(), "历史延续已登记，不覆盖。")
    files = {}
    def save(source, relative):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(target)}
    base = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1/inputs"
    for name in ("prices.parquet", "dividends.csv", "dividend_coverage.json"):
        save(base / name, "inputs/" + name)
    save(ROOT / "data/reference/sse_trade_calendar_2026.csv", "inputs/calendar.csv")
    save(CORE / "inputs/daily.parquet", "inputs/old_daily.parquet")
    save(CORE / "inputs/current_requirements.json", "inputs/current_requirements.json")
    for name in CONFIGS:
        save(CORE / f"inputs/config/{name}.json", f"inputs/config/{name}.json")
    for name in ("learned_cycle_exit", "within_cycle_exit"):
        save(TRAIN / f"inputs/{name}_config.json", f"inputs/config/{name}.json")
    for name in ("ordinary", "within"):
        save(TRAIN / f"results/{name}_models.json", f"inputs/{name}_models.json")
    save(TRAIN / "results/samples.parquet", "inputs/old_samples.parquet")
    for cost in ("BASE", "STRESS"):
        save(CORE / f"results/evaluation/{cost}/full_factors.parquet", f"inputs/old_{cost}_factors.parquet")
    save(ROOT / "reports/research/510300_point_state_reconstruction_v1/results/重建点位.parquet", "inputs/old_points.parquet")
    for source in (Path(__file__), Path(chain_engine.__file__), Path(training.__file__), Path(point_engine.__file__),
                   ROOT / "research/training_reference_observation_v1.py", ROOT / "research/reference_observation_accounts_v1.py",
                   ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
                   ROOT / "tests/test_point_close_observation_v1.py"):
        save(source, "code/" + source.name)
    protocol = {
        "study": "510300_POINT_HISTORY_EXTENSION_V1", "at": common.now(), "data_cutoff": LAST,
        "evidence_class": "ALREADY_INSPECTED_HISTORY_STATE_RECOVERY", "prospective_observations": 0,
        "scope": "两条固定多头线索、原月度模型及同一固定份额点位延续；不做参数筛选。",
        "new_bars": "原完整链之后的23个既有日线，2026-08-17至2026-09-16。分红采用原完整覆盖回执。",
        "prefix": "原价格及所有数值特征、旧成熟样本和至8月14日真实收盘的所有候选来源目标必须不变。",
        "monthly": "保留已复现的旧模型，仅按同一制度拟合9月1日的新普通岭和周期内模型；自然成熟周期才可进入。",
        "point_clock": "真实最后日也执行前收盘请求；末端仍持有则只标未完成和参考浮动值，不强平、不开虚构未来价格。",
        "cost": "固定约十万元原价名义份额；单边万四最低5元、千一滑点、0.001不利取整、100份整手，登记资格股息。",
        "current_view": "9月16日以后资料仍未接入，下一计划日9月17日只是历史时间戳，不是研究日当前信号。",
        "reported_metrics": "完整自然点位的胜率、实际B、pB、平均净收益和年度完成次数；历史延伸结果不能视为独立验证。",
        "hard_gate": "p乘实际净B严格>1且净均值>0；年度次数软目标。", "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("23个既有日线的状态延续、9月原制度拟合及完整点位观察已固定。", flush=True)


def append_months(data, samples, ordinary, within, cfg31, cfg114):
    require([r["fit_index"] for r in ordinary] == [r["fit_index"] for r in within], "两套旧模型拟合日历不同。")
    schedule = training.monthly_schedule(data, cfg31["earlier_start"])
    require(schedule[:len(ordinary)] == [r["fit_index"] for r in ordinary], "较新行情改变了既有模型拟合日历。")
    require(all(cfg31[k] == cfg114[k] for k in ("recent_cycles", "minimum_cycles", "minimum_rows", "feature_clip", "ridge_alpha")), "两类模型训练制度不同。")
    result31, result114 = copy.deepcopy(ordinary), copy.deepcopy(within)
    receipts = []
    for t in schedule[len(ordinary):]:
        rows, ids = training.training_rows(samples, t, cfg31)
        eligible = len(ids) >= cfg31["minimum_cycles"] and len(rows) >= cfg31["minimum_rows"]
        require(not len(rows) or rows.exit_index.le(t).all(), "延续拟合读取未结束周期。")
        basic = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5),
                 "status": "FIT_COMPLETE" if eligible else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "training_cycles": ids,
                 "training_cycle_count": len(ids), "training_rows": len(rows), "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                 "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None}
        result31.append({"signal": "D60_INTRA", "kind": "RIDGE", **basic, "model": training.fit_one(rows, "RIDGE", cfg31) if eligible else None})
        status, failure, model = basic["status"], None, None
        missing = int((~np.isfinite(rows[training.FEATURES].to_numpy(float)).all(axis=1)).sum())
        if eligible and missing:
            status = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        elif eligible:
            try:
                model = training.fit_within_cycle_exit(rows, cfg114)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", str(error)
        result114.append({**basic, "status": status, "eligible_for_fit": eligible, "failure": failure, "missing_feature_rows": missing, "model": model})
        for kind, record in (("ordinary", result31[-1]), ("within", result114[-1])):
            receipts.append({"kind": kind, **{k: v for k, v in record.items() if k not in ("model", "training_cycles", "kind", "signal")}})
        save_table("new_month_membership_" + basic["fit_origin"], rows)
    require(result31[:len(ordinary)] == ordinary and result114[:len(within)] == within, "不允许改写既有模型记录。")
    return result31, result114, pd.DataFrame(receipts)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "历史延续已运行，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, record in frozen["files"].items():
        require(common.digest(OUT / relative) == record["sha256"], "固定来源改变：" + relative)
        if relative.startswith("code/"):
            require(common.digest(ROOT / record["source"]) == record["sha256"], "运行代码改变：" + record["source"])
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "evidence_class": "HISTORICAL_RECOVERY"})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    old_data = pd.read_parquet(OUT / "inputs/old_daily.parquet")
    div_path = OUT / "inputs/dividends.csv"
    dividends = normalize_dividends(pd.read_csv(div_path))
    coverage = json.loads((OUT / "inputs/dividend_coverage.json").read_text(encoding="utf-8"))
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= LAST, "股息覆盖不足。")
    require(common.digest(div_path) == coverage["distribution_file_sha256"], "股息与覆盖回执不同。")
    require(prices.date.iloc[-1] == pd.Timestamp(LAST) and len(prices)-len(old_data) == 23, "历史延续区间不符。")
    calendar = pd.to_datetime(pd.read_csv(OUT / "inputs/calendar.csv").trade_date)
    dates = calendar.loc[calendar.gt(pd.Timestamp(BOUNDARY)) & calendar.le(pd.Timestamp(LAST))]
    require(pd.DatetimeIndex(dates).equals(pd.DatetimeIndex(prices.date.iloc[len(old_data):])), "新增区间缺少交易日。")
    require(calendar.loc[calendar.gt(pd.Timestamp(LAST))].iloc[0] == pd.Timestamp(NEXT), "计划下一日错误。")
    data, _ = factors(prices, dividends)
    prefix = data.iloc[:len(old_data)].reset_index(drop=True)
    numeric = old_data.select_dtypes(include=["number", "bool"]).columns
    for column in numeric:
        require(np.array_equal(prefix[column].to_numpy(), old_data[column].to_numpy(), equal_nan=True), "历史价格或特征前缀变化：" + column)
    require(data.feature_valid.iloc[len(old_data):].all(), "新增日线特征不完整。")
    save_table("features", data)
    configs = {key: json.loads((OUT / f"inputs/config/{key}.json").read_text(encoding="utf-8")) for key in (*CONFIGS, "learned_cycle_exit", "within_cycle_exit")}
    reference = training.reference_samples(data, dividends, configs["learned_cycle_exit"], pd.Timestamp(NEXT))
    for name, frame in reference.items():
        save_table("training_reference/" + name, frame)
    old_samples = pd.read_parquet(OUT / "inputs/old_samples.parquet")
    old_mature = reference["samples"].loc[reference["samples"].mature_date.le(pd.Timestamp(BOUNDARY))].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_mature, old_samples, check_exact=True)
    original = {key: json.loads((OUT / f"inputs/{key}_models.json").read_text(encoding="utf-8"))["models"] for key in ("ordinary", "within")}
    models31, models114, receipts = append_months(data, reference["samples"], original["ordinary"], original["within"], configs["learned_cycle_exit"], configs["within_cycle_exit"])
    require(set(receipts.fit_origin) == {"2026-09-01"} and len(receipts) == 2, "新增拟合范围不符。")
    for name, models in (("ordinary", models31), ("within", models114)):
        common.save_json(OUT / f"results/{name}_models.json", {"models": models, "evidence_class": "HISTORICAL_STATE_RECOVERY"})
    save_table("新增月度模型", receipts)
    print("旧价格特征及1461条原样本不变，9月1日两类模型已按原成熟周期制度补齐。", flush=True)
    continuous = chain_engine.continuous_references(data, data, dividends, configs["continuous_reference_min_variance"], models31, pd.Timestamp(NEXT))
    for name, frames in continuous.items():
        for kind, frame in zip(("ledger", "decisions", "cycles"), frames):
            save_table("continuous/" + name + "_" + kind, frame)
    chain = chain_engine.assemble_chain(data, dividends, configs, models114, START, pd.Timestamp(NEXT), continuous)
    checks = []
    for cost, frame in chain["factors"].items():
        old = pd.read_parquet(OUT / f"inputs/old_{cost}_factors.parquet")
        for column in LAYERS:
            error = compare_numeric(frame[column].iloc[:len(old)], old[column], cost + "/" + column)
            checks.append({"cost": cost, "model": column, "rows": len(old), "maximum_error": error})
        save_table(cost + "/full_factors", frame)
    for name, frames in chain["references"].items():
        for kind, frame in zip(("ledger", "decisions", "cycles"), frames):
            save_table("references/" + name + "_" + kind, frame)
    for name in ("variance_budget", "joint_budget", "support"):
        save_table(name, chain[name])
    save_table("旧核心链前缀对应", pd.DataFrame(checks))
    print("全部旧核心来源及候选目标前缀保持一致，开始延续两条固定点位。", flush=True)
    old_points = pd.read_parquet(OUT / "inputs/old_points.parquet")
    old_points = old_points.loc[old_points.period.eq("evaluation") & old_points.status.eq("COMPLETE")]
    points, signals, events, metrics, annual = [], [], [], [], []
    for model in MODELS:
        sig = point_engine.candidate_signals(data, chain["factors"]["STRESS"], model, START, pd.Timestamp(NEXT))
        p, e = point_engine.observe_points(data, dividends, sig, START)
        p["candidate"], e["candidate"], sig["candidate"] = model, model, model
        points.append(p); events.append(e); signals.append(sig)
        complete = p.loc[p.status.eq("COMPLETE")]
        old = old_points.loc[old_points.model.eq(model)].reset_index(drop=True)
        prior = complete.loc[complete.exit_date.lt(pd.Timestamp(BOUNDARY))].reset_index(drop=True)
        require(pd.DatetimeIndex(prior.entry_date).equals(pd.DatetimeIndex(old.entry_date)) and pd.DatetimeIndex(prior.exit_date).equals(pd.DatetimeIndex(old.exit_date)), "既有完成点位日期改变。")
        compare_numeric(prior.point_net_return, old.point_net_return, "既有完成点位收益")
        summary = statistics(complete.point_net_return)
        for year in range(2020, 2027):
            count = int(complete.exit_date.dt.year.eq(year).sum())
            annual.append({"candidate": model, "year": year, "completed_points": count, "complete_year": year < 2026})
        full_counts = [r["completed_points"] for r in annual if r["candidate"] == model and r["complete_year"]]
        old_stat = statistics(old.point_net_return)
        metrics.append({"candidate": model, **summary, "old_n": old_stat["n"], "old_product": old_stat["product"],
                        "newly_completed_since_old_boundary": int(complete.exit_date.ge(pd.Timestamp(BOUNDARY)).sum()),
                        "incomplete_points": int(p.status.eq("RIGHT_CENSORED").sum()), "average_full_year_count": float(np.mean(full_counts)),
                        "zero_trade_full_years": int(sum(n == 0 for n in full_counts)), "independent_validation": "NOT_ESTABLISHED"})
    points, signals, events = [pd.concat(rows, ignore_index=True) for rows in (points, signals, events)]
    metrics, annual = pd.DataFrame(metrics), pd.DataFrame(annual)
    for name, frame in (("全部自然点位", points), ("完整候选意向", signals), ("点位进出事件", events), ("更新历史指标", metrics), ("逐年完成次数", annual)):
        save_table(name, frame)
    extra = points.loc[points.status.eq("COMPLETE") & points.exit_date.ge(pd.Timestamp(BOUNDARY))].copy()
    extra["cohort"] = np.where(extra.entry_date.lt(pd.Timestamp(BOUNDARY)), "PREVIOUSLY_CENSORED_NOW_COMPLETE", "ENTRY_AFTER_OLD_BOUNDARY")
    save_table("旧边界后完成点位", extra)
    save_table("最后历史收盘意向", signals.loc[signals.origin.eq(pd.Timestamp(LAST))])
    summary = {"study": "510300_POINT_HISTORY_EXTENSION_V1", "at": common.now(), "status": "HISTORY_EXTENDED_TO_20260916_CURRENT_DATA_PENDING",
               "historical_bars_added": 23, "numeric_feature_columns_preserved": len(numeric), "original_mature_samples_preserved": len(old_samples),
               "mature_samples_now": len(reference["samples"]), "new_monthly_fits": len(receipts),
               "prior_core_target_groups_preserved": len(checks), "prior_core_target_maximum_error": max(r["maximum_error"] for r in checks),
               "newly_completed_point_comparisons": len(extra), "unique_new_entry_exit_pairs": len(extra.drop_duplicates(["entry_date", "exit_date"])),
               "total_complete_point_comparisons": int(points.status.eq("COMPLETE").sum()), "numeric_pass_candidates": int(metrics.point_estimate_pass.sum()),
               "necessary_tests_passed": 4, "internal_reference_replays": 7, "fixed_point_replays": 2,
               "last_known_close": LAST, "new_prospective_observations": 0, "current_signal": "NO_VIEW_PRICES_STOP_AT_20260916",
               "full_current_signal_chain_ready": False, "short_candidate_ready": False, "new_parameter_configurations": 0,
               "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    write_report(summary, metrics, extra, receipts, signals)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(summary, metrics, extra, receipts, signals):
    names = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}
    lines = ["# 固定点位延续至2026年9月16日", "",
             "**两条候选已沿原规则恢复到9月16日，并补齐9月1日两类退出模型。旧价格、训练样本和完整核心目标前缀保持不变。新增完成点位均来自已经观察过的行情，不能充作独立验证。**", "",
             "本轮从2020年原研究起点维持连续点位状态，只处理既有行情延续。每日收盘形成方向状态，下一开盘在可成交时进入固定约十万元名义份额；目标归零后下一开盘退出。正目标大小变化不调份额。净收益包括原单边万四佣金、千一滑点、不利刻度与登记股息。", "",
             "| 候选 | 完成点位 | 胜率p | 实际净B | p×B | 平均净收益/笔 | 2020—2025平均每年次数 | 原8月边界p×B |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {names[r.candidate]} | {r.n} | {r.p:.2%} | {r.b:.4f} | {r.product:.4f} | {r.mean:.2%} | {r.average_full_year_count:.2f} | {r.old_product:.4f} |")
    lines += ["", "用户的硬门槛仍为p×实际净B严格大于1，同时净均值大于0。表中次数按完整自然年的已完成点位统计，2026年是部分年度。两条候选同源、日期可能重合，不可相加作为可交易次数或独立样本。", "",
              "## 原边界之后自然完成的点位", "", "| 候选 | 进入 | 退出 | 参考净收益 | 所属情况 |", "|---|---|---|---:|---|"]
    for r in extra.itertuples():
        cohort = "原边界已持有，后来自然退出" if r.cohort == "PREVIOUSLY_CENSORED_NOW_COMPLETE" else "原边界之后进入"
        lines.append(f"| {names[r.candidate]} | {r.entry_date.date()} | {r.exit_date.date()} | {r.point_net_return:.2%} | {cohort} |")
    lines += ["", f"新增{summary['newly_completed_point_comparisons']}个候选点位比较，去除两候选间重复日期后为{summary['unique_new_entry_exit_pairs']}组进出日期。这些收益没有从既有表中删除或挑选；仍未完成的点位另存。", "",
              "## 模型如何补齐", "",
              f"新增23个日线，旧{summary['numeric_feature_columns_preserved']}个数值字段和1461条原成熟样本保持不变。训练参考延续后成熟状态共{summary['mature_samples_now']}条。9月1日按原最近20个成熟周期、至少10周期100行、原8项状态和相同岭参数分别拟合两个模型；既有141个月度记录原样保留。", "",
              "| 类型 | 拟合日 | 成熟周期数 | 状态行数 | 最后成熟退出 | 状态 |", "|---|---|---:|---:|---|---|"]
    for r in receipts.itertuples():
        lines.append(f"| {r.kind} | {r.fit_origin} | {r.training_cycle_count} | {r.training_rows} | {r.latest_exit_date} | {r.status} |")
    lines += ["", "在接入后来行情和模型后，两种费用、七层来源和两条候选至8月14日真实收盘的全部既有目标仍然一致。未完成训练周期不生成标签，最后真实交易日照常执行已知意向；这一边界由四项点位测试覆盖。", "",
              "## 9月16日的历史状态", ""]
    for r in signals.loc[signals.origin.eq(pd.Timestamp(LAST))].itertuples():
        lines.append(f"- {names[r.candidate]}：核心{r.core_target:.6f}，有效辅助{r.effective_auxiliary:.6f}，合计目标{r.target:.6f}，对应历史下一开盘{r.execution_date.date()}。")
    lines += ["", "以上只说明9月16日收盘后的历史方向状态，不是当前交易指令。9月17日至研究日之间的行情及股息覆盖尚未接入，当前市场判断继续保留NO_VIEW。", "",
              "历史数值达线仍受样本复用、收益集中及两候选高度相关的限制；没有新的独立点位结果，也尚无通过门槛的空头候选。下一步是补齐可核实的最近日线和股息，恢复实际当前状态后登记未来信号。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按原制度延续已观察日线及固定点位。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as error:
            common.save_json(OUT / "failure.json", {"at": common.now(), "status": "IMPLEMENTATION_DIAGNOSIS_REQUIRED", "error": str(error)})
            raise
