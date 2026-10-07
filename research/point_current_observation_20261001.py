"""把固定候选恢复到最近完整日线，并在下次开盘前保存研究观察意向。"""
from __future__ import annotations

import argparse
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
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_core_observation_v1 import CONFIGS, LAYERS, compare_numeric
from research.point_state_reconstruction_v1 import MODELS
from research.historic_cycles_point_translation_v1 import statistics

OUT = ROOT / "reports/research/510300_point_current_observation_20261001"
HISTORY = ROOT / "reports/research/510300_point_history_extension_v1"
INPUTS = ROOT / "reports/research/510300_point_current_inputs_20261001"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
LAST, NEXT, OLD_LAST, START = "2026-09-30", "2026-10-08", "2026-09-16", "2020-01-02"
CALENDAR_URL = "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not OUT.exists(), "当前收盘观察已经登记，不覆盖。")
    files = {}
    def save(source, relative):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(destination)}
    for name in ("candidate_prices.parquet", "candidate_features.parquet", "candidate_dividend_coverage.json", "admission_receipt.json", "summary.json"):
        save(INPUTS / name, "inputs/" + name)
    for name in ("ordinary", "within"):
        save(HISTORY / f"results/{name}_models.json", f"inputs/{name}_models.json")
    for key in (*CONFIGS, "learned_cycle_exit", "within_cycle_exit"):
        save(HISTORY / f"inputs/config/{key}.json", f"inputs/config/{key}.json")
    save(HISTORY / "inputs/dividends.csv", "inputs/dividends.csv")
    save(HISTORY / "results/training_reference/samples.parquet", "inputs/old_training_samples.parquet")
    for cost in ("BASE", "STRESS"):
        save(HISTORY / f"results/{cost}/full_factors.parquet", f"inputs/old_{cost}_factors.parquet")
    save(HISTORY / "results/全部自然点位.parquet", "inputs/old_points.parquet")
    save(CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    for path in (Path(__file__), Path(chain_engine.__file__), Path(training.__file__), Path(point_engine.__file__)):
        save(path, "code/" + path.name)
    protocol = {
        "study": "510300_POINT_CURRENT_OBSERVATION_20261001", "at": common.now(),
        "scope": "固定两条多头候选，使用至9月30日已经结束的日线及原9月模型恢复状态，并保存10月8日开盘前的研究意向。",
        "history": "9月17日至9月30日为后来补齐的历史，所有进出结果保留，只算历史恢复，不计入预先生成信号集合。",
        "rules": "沿用原参考持仓、风险预算、模型支持、辅助限制及正目标进入/零目标退出；不改阈值、止损、目标或训练制度。",
        "models": "沿用包括9月1日的142条普通岭及142条周期内模型；9月16日至9月30日无新月首，不应再训练或跳过必要月度拟合。",
        "prospective": "只有实际生成时间早于计划开盘的意向才可登记。若研究起点已经持有，标为继承历史点位，未来退出也不能使整笔历史点位成为新前瞻交易。",
        "point_cost": "固定约十万元原价名义份额，原单边万四最低5元、千一滑点和0.001不利取整；不评估期权或投资账户绩效。",
        "technical_context": "保存日MACD、上一完整周MACD、相对量和相对波动，仅解释当时状态，不临时作为新增过滤器。",
        "calendar_notice": CALENDAR_URL, "next_session": NEXT, "new_parameter_configurations": 0,
        "goal_gate": "p乘实际净B严格>1且净均值>0；次数软目标。", "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("最近完整行情、固定候选及10月8日研究意向的登记口径已保存。", flush=True)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "当前观察已经运行，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    amendment = json.loads((OUT / "pre_run_display_correction.json").read_text(encoding="utf-8"))
    for relative, record in frozen["files"].items():
        require(common.digest(OUT / relative) == record["sha256"], "固定来源变化：" + relative)
        if relative.startswith("code/"):
            expected = amendment["corrected_program_sha256"] if relative == "code/" + Path(__file__).name else record["sha256"]
            require(common.digest(ROOT / record["source"]) == expected, "执行代码与登记不同：" + record["source"])
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "orders_authorized": False})
    receipt = json.loads((OUT / "inputs/admission_receipt.json").read_text(encoding="utf-8"))
    require(receipt["accepted"] and receipt["cutoff"] == LAST and receipt["new_trading_days"] == 9, "当前日线接纳回执不符。")
    require(common.digest(OUT / "inputs/candidate_features.parquet") == receipt["features_sha256"], "日线特征回执不符。")
    data = pd.read_parquet(OUT / "inputs/candidate_features.parquet")
    prices = pd.read_parquet(OUT / "inputs/candidate_prices.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    require(common.digest(OUT / "inputs/dividends.csv") == receipt["dividends_sha256"], "股息回执不符。")
    configs = {key: json.loads((OUT / f"inputs/config/{key}.json").read_text(encoding="utf-8")) for key in (*CONFIGS, "learned_cycle_exit", "within_cycle_exit")}
    models = {key: json.loads((OUT / f"inputs/{key}_models.json").read_text(encoding="utf-8"))["models"] for key in ("ordinary", "within")}
    schedule = training.monthly_schedule(data, configs["learned_cycle_exit"]["earlier_start"])
    require(all([r["fit_index"] for r in models[key]] == schedule for key in models), "有新的月度拟合尚未生成，不能沿用过期模型。")
    require(all(models[key][-1]["fit_origin"] == "2026-09-01" for key in models), "模型月份不符。")
    reference = training.reference_samples(data, dividends, configs["learned_cycle_exit"], pd.Timestamp(NEXT))
    for name, frame in reference.items():
        save_table("training_reference/" + name, frame)
    prior_samples = reference["samples"].loc[reference["samples"].mature_date.le(pd.Timestamp(OLD_LAST))].reset_index(drop=True)
    pd.testing.assert_frame_equal(prior_samples, pd.read_parquet(OUT / "inputs/old_training_samples.parquet"), check_exact=True)
    continuous = chain_engine.continuous_references(data, data, dividends, configs["continuous_reference_min_variance"], models["ordinary"], pd.Timestamp(NEXT))
    for name, frames in continuous.items():
        for kind, frame in zip(("ledger", "decisions", "cycles"), frames):
            save_table("continuous/" + name + "_" + kind, frame)
    chain = chain_engine.assemble_chain(data, dividends, configs, models["within"], START, pd.Timestamp(NEXT), continuous)
    checks = []
    for cost, frame in chain["factors"].items():
        old = pd.read_parquet(OUT / f"inputs/old_{cost}_factors.parquet")
        for model in LAYERS:
            error = compare_numeric(frame[model].iloc[:len(old)], old[model], cost + "/" + model)
            checks.append({"cost": cost, "model": model, "rows": len(old), "maximum_error": error})
        save_table(cost + "/full_factors", frame)
    for name, frames in chain["references"].items():
        for kind, frame in zip(("ledger", "decisions", "cycles"), frames):
            save_table("references/" + name + "_" + kind, frame)
    for name in ("variance_budget", "joint_budget", "support"):
        save_table(name, chain[name])
    save_table("既有来源前缀对应", pd.DataFrame(checks))
    print("完整候选已延续至9月30日，旧目标与自然成熟样本前缀均保留。", flush=True)
    old_points = pd.read_parquet(OUT / "inputs/old_points.parquet")
    point_frames, event_frames, signal_frames, metrics, annual, registry = [], [], [], [], [], []
    for model in MODELS:
        sig = point_engine.candidate_signals(data, chain["factors"]["STRESS"], model, START, pd.Timestamp(NEXT))
        points, events = point_engine.observe_points(data, dividends, sig, START)
        points["candidate"], events["candidate"], sig["candidate"] = model, model, model
        point_frames.append(points); event_frames.append(events); signal_frames.append(sig)
        complete = points.loc[points.status.eq("COMPLETE")]
        old = old_points.loc[old_points.candidate.eq(model) & old_points.status.eq("COMPLETE")].reset_index(drop=True)
        prefix = complete.loc[complete.exit_date.le(pd.Timestamp(OLD_LAST))].reset_index(drop=True)
        require(pd.DatetimeIndex(prefix.entry_date).equals(pd.DatetimeIndex(old.entry_date)) and pd.DatetimeIndex(prefix.exit_date).equals(pd.DatetimeIndex(old.exit_date)), "原完成点位日期被改变。")
        compare_numeric(prefix.point_net_return, old.point_net_return, "原完成点位净收益")
        for year in range(2020, 2027):
            annual.append({"candidate": model, "year": year, "completed_points": int(complete.exit_date.dt.year.eq(year).sum()), "complete_year": year < 2026})
        counts = [r["completed_points"] for r in annual if r["candidate"] == model and r["complete_year"]]
        metrics.append({"candidate": model, **statistics(complete.point_net_return), "average_full_year_count": float(np.mean(counts)),
                        "zero_trade_full_years": int(sum(n == 0 for n in counts)), "incomplete_points": int(points.status.eq("RIGHT_CENSORED").sum()),
                        "independent_validation": "NOT_ESTABLISHED"})
        held = points.loc[points.status.eq("RIGHT_CENSORED")]
        require(len(held) <= 1, "固定点位同时出现多笔持仓。")
        last, has_position = sig.iloc[-1], len(held) == 1
        known = np.isfinite(last.target)
        action = "UNKNOWN_KEEP_POSITION" if not known else "HOLD_EXISTING_POINT" if has_position and last.target > 0 else "EXIT_NEXT_OPEN" if has_position else "ENTER_NEXT_OPEN" if last.target > 0 else "FLAT_WAIT"
        eligibility = "CARRIED_HISTORICAL_POSITION_EXCLUDE_NEW_ENTRY_COHORT" if has_position else "PREOPEN_NEW_ENTRY_INTENT" if action == "ENTER_NEXT_OPEN" else "PREOPEN_OBSERVATION_NO_NEW_ENTRY"
        timestamp = common.now()
        require(pd.Timestamp(timestamp) < pd.Timestamp(NEXT, tz="Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30), "意向保存已迟于计划开盘。")
        registry.append({"candidate": model, "direction": "LONG", "origin": last.origin, "source_data_max_date": data.date.iloc[-1],
                         "actual_generated_at": timestamp, "planned_execution_date": pd.Timestamp(NEXT),
                         "parent_state_receipt": str((OUT / "results/STRESS/full_factors.parquet").relative_to(ROOT)),
                         "parent_state_hash": common.digest(OUT / "results/STRESS/full_factors.parquet"),
                         "inputs_receipt": str((INPUTS / "admission_receipt.json").relative_to(ROOT)),
                         "signal_status": "AVAILABLE" if known else "NO_VIEW", "entry_or_exit_intent": action,
                         "point_state_before": "HOLDING_HISTORICAL_POINT" if has_position else "FLAT",
                         "inherited_entry_date": held.entry_date.iloc[0] if has_position else None,
                         "inherited_entry_raw": float(held.entry_raw.iloc[0]) if has_position else None,
                         "inherited_reference_mark_return": float(held.observed_close_mark_return.iloc[0]) if has_position else None,
                         "target": float(last.target) if known else None, "core_target": float(last.core_target),
                         "effective_auxiliary": float(last.effective_auxiliary), "reference_cost_basis": "STRESS_FIXED_NOTIONAL_100000_CNY",
                         "eligibility": eligibility, "unavailable_reason": "" if known else "必需来源状态未知",
                         "orders_authorized": False})
    points, events, signals = [pd.concat(frames, ignore_index=True) for frames in (point_frames, event_frames, signal_frames)]
    metrics, annual, registry = pd.DataFrame(metrics), pd.DataFrame(annual), pd.DataFrame(registry)
    for name, frame in (("全部自然点位", points), ("点位进出事件", events), ("完整候选意向", signals), ("历史点位指标", metrics), ("逐年完成次数", annual), ("下次开盘研究意向", registry)):
        save_table(name, frame)
    registry.to_csv(OUT / "后续点位记录.csv", index=False, encoding="utf-8-sig")
    new_complete = points.loc[points.status.eq("COMPLETE") & points.exit_date.gt(pd.Timestamp(OLD_LAST))].copy()
    new_complete["evidence_class"] = "BACKFILLED_HISTORY_NOT_PREOPEN_SIGNAL"
    save_table("本次补齐后完成点位", new_complete)
    technical, _ = common.features(prices, dividends)
    technical["raw_equivalent_ema20"] = technical.ema20-technical.cash_shift
    selected = ["date", "close", "ema20", "raw_equivalent_ema20", "cash_shift", "daily_dif", "daily_dea", "daily_hist", "daily_hist_rising", "weekly_last_date",
                "weekly_hist", "weekly_hist_rising", "relative_volume", "rv_ratio", "up_volume_balance5", "above_ema20"]
    context = technical[selected].tail(1)
    save_table("最新日周线量价状态", context)
    summary = {"study": "510300_POINT_CURRENT_OBSERVATION_20261001", "at": common.now(),
               "status": "CURRENT_CLOSE_RECONSTRUCTED_PREOPEN_RESEARCH_INTENTS_REGISTERED", "last_known_close": LAST,
               "next_exchange_session": NEXT, "new_history_bars": 9, "new_point_comparisons_completed_in_backfill": len(new_complete),
               "new_unique_entry_exit_pairs": len(new_complete.drop_duplicates(["entry_date", "exit_date"])),
               "mature_training_samples": len(reference["samples"]), "new_monthly_fits": 0,
               "numeric_pass_candidates": int(metrics.point_estimate_pass.sum()), "registered_candidate_intents": len(registry),
               "new_prospective_entries": 0, "new_prospective_completed_points": 0,
               "prospective_new_entry_intents": int(registry.eligibility.eq("PREOPEN_NEW_ENTRY_INTENT").sum()),
               "inherited_historical_positions": int(registry.point_state_before.eq("HOLDING_HISTORICAL_POINT").sum()),
               "internal_reference_replays": 7, "fixed_point_replays": 2, "prior_core_target_maximum_error": max(r["maximum_error"] for r in checks),
               "full_current_signal_chain_ready": True, "current_signal": "REGISTERED_RESEARCH_INTENTS_ONLY",
               "short_candidate_ready": False, "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    write_report(summary, metrics, new_complete, registry, context)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(summary, metrics, new_complete, registry, context):
    names = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}
    actions = {"HOLD_EXISTING_POINT": "延续已有研究点位", "EXIT_NEXT_OPEN": "下一开盘观察退出", "ENTER_NEXT_OPEN": "下一开盘观察进入", "FLAT_WAIT": "空仓等待", "UNKNOWN_KEEP_POSITION": "信息不足，保持原状态"}
    lines = ["# 510300固定点位研究：截至2026年9月30日", "",
             "**日线、股息、原月度模型和完整候选状态已恢复到最近完整收盘；10月8日的研究意向已事先保存。历史达线只说明候选值得继续观察，尚无自然完成的前瞻点位，也没有通过门槛的空头候选。**", "",
             "## 用户门槛与完整历史结果", "",
             "p为完成点位的净盈利比例，B为平均正净收益除以平均负净收益的绝对值。用户硬门槛为p×B>1且平均净收益>0。标准亏损单位期望为p×B−q，q为亏损率；无平手时q=1−p。计划止盈/止损比不替代实际B。", "",
             "以下为2020年1月2日至2026年9月30日，固定份额、原严格费用、完整自然退出结果。每边包含万四佣金且至少5元、千一滑点和不利刻度；股息按登记资格计入。", "",
             "| 候选 | 完成点位 | 净胜率p | 实际净B | p×B | 标准期望/平均亏损单位 | 平均净收益/笔 | 2020—2025年均次数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {names[r.candidate]} | {r.n} | {r.p:.2%} | {r.b:.4f} | {r.product:.4f} | {r.standard_expectation_loss_units:.4f} | {r.mean:.2%} | {r.average_full_year_count:.2f} |")
    lines += ["", "两条候选同源，许多进入和退出日相同，次数及样本不能相加。年度次数为软目标；未完成持仓不计胜率和实际盈亏比。当前数值是经历历史选择后的点估计，尚未建立独立验证。", "",
              "## 这次补齐的历史结果", "",
              "两家行情来源的9月17日至9月30日日线完成对应，旧3479行价格及特征保持不变；基金管理人和上交所公告核对支持股息覆盖至9月30日。后补行情按完整日期接入，不能计为当时提前发出的信号。", ""]
    if len(new_complete):
        lines += ["| 候选 | 进入 | 退出 | 净收益 |", "|---|---|---|---:|"]
        for r in new_complete.itertuples():
            lines.append(f"| {names[r.candidate]} | {r.entry_date.date()} | {r.exit_date.date()} | {r.point_net_return:.2%} |")
    else:
        lines.append("这9个交易日没有新增自然完成点位，既有历史指标因此保持原值；仍持有的点位单列。")
    lines += ["", "本段不含新的月首，继续使用9月1日按原成熟周期制度形成的两类模型。7条必要内部参考重放后，至9月16日的所有旧来源目标保持一致。", "",
              "## 已保存的下一开盘研究意向", "",
              f"最近完整收盘为9月30日。下一交易日为10月8日，依据[上交所2026年中秋及国庆休市公告]({CALENDAR_URL})。以下意向在10月1日实际生成并保存，尚未发生未来开盘成交。", ""]
    for r in registry.itertuples():
        text = f"- {names[r.candidate]}：{actions[r.entry_or_exit_intent]}，核心目标{r.core_target:.6f}、有效辅助{r.effective_auxiliary:.6f}、合计{r.target:.6f}。"
        if r.point_state_before == "HOLDING_HISTORICAL_POINT":
            text += f" 继承的历史进入日为{pd.Timestamp(r.inherited_entry_date).date()}，原始开盘参考价{r.inherited_entry_raw:.3f}元；9月30日参考浮动净值{r.inherited_reference_mark_return:.2%}，未计入完成收益。"
        lines.append(text)
    lines += ["", "已有历史持仓不会在研究登记日被假定平仓、重新开仓。未来若退出，也仍属于继承点位；只有新入场前及时保存、随后自然完成的整笔点位，才进入新的前瞻交易集合。正目标是计算状态，固定点位不随正目标大小调份额。没有期权合约或实际下单动作。", "",
              "## 当前技术与量价背景", ""]
    r = context.iloc[0]
    lines += [f"9月30日收盘{r.close:.3f}元，前向含息EMA20换算至当日原价坐标为{r.raw_equivalent_ema20:.3f}元；日MACD柱{r.daily_hist:.5f}，当日{'回升' if r.daily_hist_rising else '未回升'}。此前完整周最后交易日为{r.weekly_last_date.date()}，周MACD柱{r.weekly_hist:.5f}，{'回升' if r.weekly_hist_rising else '未回升'}。当日成交量/此前20日中位数为{r.relative_volume:.2f}倍，20日波动/此前一年中位数为{r.rv_ratio:.2f}倍。", "",
              "这些指标用于解释已知背景，本轮没有据此追加条件筛掉坏点位。候选由多条已有状态决定，计划盈亏比也不能从MACD或正目标大小直接推得。实际B必须等待自然完成的交易样本计算。", "",
              "研究目标仍未完成：需要及时登记的新点位结果检验优势能否延续，同时空头线索仍未达标。未来资料缺失时恢复NO_VIEW；若固定候选新样本不达线，保留失败，不通过调整阈值或回填历史改善结果。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="最近完整收盘候选状态与未来研究意向登记。")
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
