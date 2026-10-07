"""固定已有点位，解释组合入场来源、每日持有状态与额外再入场的对应。"""
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
from research import point_state_reconstruction_v1 as previous
from research import point_entry_exit_contribution_v1 as timing

common, old = previous.common, previous.old
OUT = ROOT / "reports/research/510300_point_component_path_attribution_v1"
MODELS, PERIODS = previous.MODELS, previous.PERIODS
MODEL_NAMES = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}
CAUSE_NAMES = {"CORE_ONLY": "仅核心", "AUXILIARY_ONLY": "仅辅助", "CORE_AND_AUXILIARY": "核心与辅助同时", "UNKNOWN_HOLD": "未知资料保持", "ZERO_WAIT": "零目标仍待退出"}
COHORT_NAMES = {"ORIGINAL_GUARD_ENTRY": "原价格保护规则也在此日入场", "ADDITIONAL_MODEL_ENTRY": "训练退出规则额外再入场", "DURING_GUARD_HOLD": "原价格保护规则持有期间的其他入场", "OTHER_ENTRY": "其他组合入场"}


def save_table(name, frame):
    destination = OUT / "results" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination.with_suffix(".parquet"), index=False)
    frame.to_csv(destination.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def held_at(points, date):
    """退出开盘不再计持有；末端未完成保留到最后观察开盘。"""
    end = points.exit_date.where(points.status.eq("COMPLETE"), points.last_observation_date)
    before_end = points.exit_date.gt(date) | (points.status.ne("COMPLETE") & end.ge(date))
    current = points.loc[points.entry_date.le(date) & before_end]
    require(len(current) <= 1, "同一条连续规则在该开盘出现多个重叠持仓。")
    return current.iloc[0] if len(current) else None


def current_cause(core, auxiliary, target):
    if not np.isfinite([core, auxiliary, target]).all():
        return "UNKNOWN_HOLD"
    if target == 0:
        return "ZERO_WAIT"
    c, a = core > 0, auxiliary > 0
    require(c or a, "正组合目标没有任何正来源。")
    return "CORE_AND_AUXILIARY" if c and a else "CORE_ONLY" if c else "AUXILIARY_ONLY"


def component_flags(vintage, router, auxiliary, target):
    if not np.isfinite([vintage, router, auxiliary, target]).all():
        return "UNKNOWN"
    active = [name for name, value in (("VINTAGE", vintage), ("ROUTER", router), ("AUXILIARY", auxiliary)) if value > 0]
    return "+".join(active) if active else "ZERO"


def add_components(signals, parents):
    merged = signals.merge(parents[["origin", "execution_date", "VINTAGE_REFERENCE_RISK_parent_target", "MODEL_SUPPORT_REFERENCE_ROUTER_parent_target"]],
                           on=["origin", "execution_date"], how="left", validate="many_to_one")
    merged["vintage_contribution"] = merged.budget131 * merged.VINTAGE_REFERENCE_RISK_parent_target
    merged["router_contribution"] = (1-merged.budget131) * merged.MODEL_SUPPORT_REFERENCE_ROUTER_parent_target
    reconstructed = np.clip(merged.vintage_contribution + merged.router_contribution, 0, 1)
    require(np.allclose(reconstructed, merged.core_target, rtol=1e-11, atol=1e-11, equal_nan=True), "核心两个分量不能对应原目标。")
    require(merged.origin.lt(merged.execution_date).all(), "每日状态使用了执行日或之后的收盘。")
    merged["holding_cause"] = [current_cause(r.core_target, r.effective_auxiliary, r.target) for r in merged.itertuples()]
    merged["component_flags"] = [component_flags(r.vintage_contribution, r.router_contribution, r.effective_auxiliary, r.target) for r in merged.itertuples()]
    return merged


def explain_entry(point, signals, reference):
    signal = signals.loc[point["entry_date"]]
    guards = reference.loc[reference.policy.eq("LONG_GUARDS")]
    learned = reference.loc[reference.policy.eq("LONG_MODEL")]
    same_guard = bool(guards.entry_date.eq(point["entry_date"]).any())
    same_model = bool(learned.entry_date.eq(point["entry_date"]).any())
    guard_active, model_active = held_at(guards, point["entry_date"]), held_at(learned, point["entry_date"])
    cohort = "ORIGINAL_GUARD_ENTRY" if same_guard else "ADDITIONAL_MODEL_ENTRY" if same_model else "DURING_GUARD_HOLD" if guard_active is not None else "OTHER_ENTRY"
    require(point["entry_origin"] == signal.origin and signal.target > 0, "原点位入场与当前来源状态不同。")
    require(point["entry_cause"] == signal.holding_cause, "原入场来源不一致。")
    return {**point, "entry_cohort": cohort, "entry_component_flags": signal.component_flags,
            "entry_vintage_contribution": signal.vintage_contribution, "entry_router_contribution": signal.router_contribution,
            "same_original_guard_entry": same_guard, "same_model_entry": same_model,
            "guard_held_at_entry": guard_active is not None, "model_held_at_entry": model_active is not None,
            "guard_active_entry": guard_active.entry_date if guard_active is not None else pd.NaT,
            "guard_active_exit": guard_active.exit_date if guard_active is not None else pd.NaT,
            "model_active_entry": model_active.entry_date if model_active is not None else pd.NaT,
            "model_active_exit": model_active.exit_date if model_active is not None else pd.NaT}


def interval_path(point, prices, dividends, signals):
    """固定份额逐开盘损益；每段除以本笔原始入场价，可加总且不复利。"""
    start = int(point["entry_idx"])
    if point["status"] == "COMPLETE":
        end = int(point["exit_idx"])
    else:
        hits = np.flatnonzero(prices.date.eq(point["last_observation_date"]))
        require(len(hits) == 1, "未完成点位缺少末端行情。")
        end = int(hits[0])
    require(prices.date.iloc[start] == point["entry_date"], "点位入场索引错位。")
    raw = float(point["entry_raw"])
    eligible = dividends.loc[dividends.record_date.ge(point["entry_date"]) & dividends.record_date.lt(prices.date.iloc[end]) & dividends.ex_date.le(prices.date.iloc[end])]
    cash = eligible.groupby("ex_date").cash_dividend_per_share.sum()
    records = []
    for t in range(start, end):
        date, next_date = prices.date.iloc[t], prices.date.iloc[t+1]
        signal = signals.loc[date]
        require(signal.origin < date, "当期开盘到下一开盘收益包含未来状态。")
        amount = float(cash.get(next_date, 0.))
        increment = (float(prices.open.iloc[t+1]) - float(prices.open.iloc[t]) + amount) / raw
        records.append({"point_id": point["point_id"], "model": point["model"], "period": point["period"], "status": point["status"],
                        "entry_date": point["entry_date"], "date": date, "next_date": next_date, "origin": signal.origin,
                        "entry_cause": point["entry_cause"], "holding_cause": signal.holding_cause, "component_flags": signal.component_flags,
                        "core_target": signal.core_target, "effective_auxiliary": signal.effective_auxiliary,
                        "vintage_contribution": signal.vintage_contribution, "router_contribution": signal.router_contribution,
                        "gross_return_increment": increment, "dividend_per_share_increment": amount})
    path = pd.DataFrame(records)
    require(len(path) > 0, "没有完整观察间隔的持仓。")
    gross = float(path.gross_return_increment.sum())
    ref = old.point_reference(raw, float(prices.open.iloc[end]), float(cash.sum()))
    require(abs(gross-ref["point_gross_return"]) < 1e-12, "开盘区间与整笔毛收益无法相加。")
    target_net = point["point_net_return"] if point["status"] == "COMPLETE" else point["terminal_open_mark_return"]
    require(abs(ref["point_net_return"]-target_net) < 1e-12, "固定点位费用及股息不能重现。")
    summary = {
        "point_id": point["point_id"], "model": point["model"], "period": point["period"], "status": point["status"],
        "entry_date": point["entry_date"], "exit_date": point["exit_date"], "entry_cause": point["entry_cause"],
        "observed_intervals": len(path), "gross_return": gross, "friction_return": ref["point_net_return"]-gross,
        "recomputed_net_return": ref["point_net_return"], "saved_net_or_terminal_mark": target_net,
        "net_recomputation_error": ref["point_net_return"]-target_net,
        "state_changes": int(path.holding_cause.ne(path.holding_cause.shift()).sum()-1),
        "distinct_holding_states": int(path.holding_cause.nunique()),
        "ever_core_active": bool(path.core_target.gt(0).any()), "ever_auxiliary_active": bool(path.effective_auxiliary.gt(0).any()),
        "later_core_after_auxiliary_entry": bool(point["entry_cause"] == "AUXILIARY_ONLY" and path.core_target.gt(0).any()),
        "later_auxiliary_only_after_core_entry": bool(point["entry_cause"] != "AUXILIARY_ONLY" and path.holding_cause.eq("AUXILIARY_ONLY").any()),
    }
    for cause in CAUSE_NAMES:
        selected = path.loc[path.holding_cause.eq(cause)]
        summary[cause + "_intervals"] = len(selected)
        summary[cause + "_gross_return"] = float(selected.gross_return_increment.sum())
    return path, summary


def link_extra_entries(extras, points, signals, prices):
    rows, overlaps = [], []
    for extra in extras.to_dict("records"):
        date, end = extra["entry_date"], extra["exit_date"]
        for model in MODELS:
            current = points.loc[points.model.eq(model) & points.period.eq(extra["period"])]
            active = held_at(current, date)
            state = signals.loc[signals.model.eq(model) & signals.period.eq(extra["period"]) & signals.execution_date.eq(date)].iloc[0]
            relation = "SAME_ENTRY" if active is not None and active.entry_date == date else "ALREADY_HELD" if active is not None else "FLAT_AT_EXTRA_ENTRY"
            rows.append({"model": model, "period": extra["period"], "extra_entry_date": date, "extra_exit_date": end,
                         "extra_model_return": extra["point_net_return"], "relation": relation,
                         "composite_point_id": active.point_id if active is not None else None,
                         "composite_entry_date": active.entry_date if active is not None else pd.NaT,
                         "composite_exit_date": active.exit_date if active is not None else pd.NaT,
                         "composite_status": active.status if active is not None else "FLAT",
                         "composite_entry_cause": active.entry_cause if active is not None else "NONE",
                         "composite_full_point_return": active.point_net_return if active is not None else np.nan,
                         "origin": state.origin, "current_target": state.target, "current_core_target": state.core_target,
                         "current_auxiliary_target": state.effective_auxiliary, "component_flags": state.component_flags})
            for point in current.to_dict("records"):
                point_end = point["exit_date"] if point["status"] == "COMPLETE" else point["last_observation_date"]
                overlap_start, overlap_end = max(date, point["entry_date"]), min(end, point_end)
                if overlap_start < overlap_end:
                    observed = int(prices.date.ge(overlap_start).mul(prices.date.lt(overlap_end)).sum())
                    overlaps.append({"model": model, "period": extra["period"], "extra_entry_date": date, "composite_point_id": point["point_id"],
                                     "composite_entry_date": point["entry_date"], "composite_exit_date": point["exit_date"],
                                     "overlap_start": overlap_start, "overlap_end": overlap_end, "overlap_session_intervals": observed})
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def aggregate(entries, summaries):
    records, state_records = [], []
    complete = entries.loc[entries.status.eq("COMPLETE")]
    for (model, period), group in complete.groupby(["model", "period"], sort=False):
        for column, names in (("entry_cause", CAUSE_NAMES), ("entry_cohort", COHORT_NAMES)):
            for key in names:
                values = group.loc[group[column].eq(key), "point_net_return"]
                if column == "entry_cause" and key in ("UNKNOWN_HOLD", "ZERO_WAIT"):
                    continue
                records.append({"model": model, "period": period, "dimension": column, "group": key, "label": names[key], **old.statistics(values)})
        path_summaries = summaries.loc[summaries.model.eq(model) & summaries.period.eq(period) & summaries.status.eq("COMPLETE")]
        require(len(path_summaries) == len(group), "持有路径缺少完整点位。")
        require(abs(path_summaries.recomputed_net_return.mean()-group.point_net_return.mean()) < 1e-12, "路径与原点位平均净收益不同。")
        for cause in CAUSE_NAMES:
            state_records.append({"model": model, "period": period, "state": cause, "label": CAUSE_NAMES[cause], "all_completed_points": len(group),
                                  "intervals": int(path_summaries[cause + "_intervals"].sum()),
                                  "points_ever_in_state": int(path_summaries[cause + "_intervals"].gt(0).sum()),
                                  "average_contribution_per_original_point": path_summaries[cause + "_gross_return"].mean()})
        for cause, field, label in (("FRICTION", "friction_return", "统一参考摩擦"), ("NET", "recomputed_net_return", "最终净回报")):
            state_records.append({"model": model, "period": period, "state": cause, "label": label, "all_completed_points": len(group),
                                  "intervals": int(path_summaries.observed_intervals.sum()), "points_ever_in_state": len(group),
                                  "average_contribution_per_original_point": path_summaries[field].mean()})
    return pd.DataFrame(records), pd.DataFrame(state_records)


def freeze():
    require(not OUT.exists(), "已有来源路径研究，不覆盖。")
    OUT.mkdir(parents=True)
    files = {}
    def copy(source, relative):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(destination)}
    for name in ("逐日组合来源状态", "重建点位", "入场前状态与结果标签", "点位指标"):
        copy(previous.OUT / f"results/{name}.parquet", f"inputs/{name}.parquet")
    for name in ("source_daily.parquet", "dividends.csv"):
        copy(previous.OUT / "inputs" / name, "inputs/" + name)
    for period in PERIODS:
        copy(previous.OUT / f"inputs/{period}/CORE_decisions.parquet", f"inputs/{period}/CORE_decisions.parquet")
    copy(timing.OUT / "results/连续规则全部点位.parquet", "inputs/原始入场退出点位.parquet")
    copy(timing.OUT / "results/连续模型新增的六个入场.parquet", "inputs/已知六次额外再入场.parquet")
    copy(previous.CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    for path in (Path(__file__), Path(old.__file__), Path(common.__file__), ROOT / "tests/test_point_component_path_attribution_v1.py"):
        copy(path, "code/" + path.name)
    protocol = {
        "study": "510300_POINT_COMPONENT_PATH_ATTRIBUTION_V1", "at": common.now(),
        "question": "两条现有组合的新增次数来自哪些入场与持有状态；上一轮六次额外再进入在组合中如何对应？",
        "known_before_this_study": "既有组合94条完成记录、额外六次再入场及其结果已知；本轮是固定记录的全范围归因，不是独立验证。",
        "periods": PERIODS, "models": list(MODELS),
        "entries": "原守护规则同日进入、训练退出规则额外同日进入、原守护规则持有期间的其他进入、其余组合进入四类；按既有时钟描述，不作为新筛选条件。",
        "held_interval": "从入场开盘含当日到退出开盘不含当日；未完成保留到末端观察开盘。每条规则同一开盘最多一个持有点位。",
        "holding_state": "每日区间状态取该开盘之前保存的收盘目标：仅核心、仅有效辅助、二者同时、未知保持、零目标等待；拆开核心的两个已存分量，仅作来源解释。",
        "return_attribution": "每个实际持有开盘到下一开盘的原价差加有资格除息，除该笔原始入场价；按当期状态分组后能加总为整笔毛回报，摩擦单列。不把正信号权重当实际分配份额。",
        "interpretation": "分组展示收益发生时的信号状态，不识别组件独立因果收益；不模拟删除组件、跳过点位或新策略。末端路径保存但排除胜率和完整点位均值。",
        "extra_entries": "固定上一轮全部六次额外入场，逐一检查两组合当时同日新开、已持有或空仓，并列所有实际重叠区间，不只挑对应收益较好的记录。",
        "quality": "沿用p乘实际净B严格>1且净均值>0；频率软目标。缺少盈利或亏损时B保留未定义。",
        "new_point_replays": 0, "new_full_accounts": 0, "new_model_fits": 0, "parameter_searches": 0,
        "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("两条组合全部点位的入场来源与逐开盘路径对照已固定。", flush=True)


def run():
    require(not (OUT / "summary.json").exists(), "已有完整结果，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, item in frozen["files"].items():
        require(common.digest(OUT / relative) == item["sha256"], "固定来源发生变化：" + relative)
    require(common.digest(Path(__file__)) == frozen["files"]["code/" + Path(__file__).name]["sha256"], "当前实现与固定版本不同。")
    prices = pd.read_parquet(OUT / "inputs/source_daily.parquet")
    dividends = pd.read_csv(OUT / "inputs/dividends.csv", parse_dates=["record_date", "ex_date"])
    signals = pd.read_parquet(OUT / "inputs/逐日组合来源状态.parquet")
    points = pd.read_parquet(OUT / "inputs/重建点位.parquet")
    points["point_id"] = points.model + "|" + points.period + "|" + points.entry_date.dt.strftime("%Y-%m-%d")
    reference = pd.read_parquet(OUT / "inputs/原始入场退出点位.parquet")
    extras = pd.read_parquet(OUT / "inputs/已知六次额外再入场.parquet")
    component_frames = []
    for period in PERIODS:
        parents = pd.read_parquet(OUT / f"inputs/{period}/CORE_decisions.parquet")
        component_frames.append(add_components(signals.loc[signals.period.eq(period)], parents))
    components = pd.concat(component_frames, ignore_index=True)
    entries, paths, summaries = [], [], []
    for (model, period), current in points.groupby(["model", "period"], sort=False):
        daily = components.loc[components.model.eq(model) & components.period.eq(period)].set_index("execution_date")
        refs = reference.loc[reference.period.eq(period)]
        for point in current.to_dict("records"):
            entry = explain_entry(point, daily, refs)
            path, summary = interval_path(point, prices, dividends, daily)
            entries.append(entry)
            paths.append(path)
            summaries.append(summary)
    entries, paths, summaries = pd.DataFrame(entries), pd.concat(paths, ignore_index=True), pd.DataFrame(summaries)
    group_metrics, state_metrics = aggregate(entries, summaries)
    links, overlaps = link_extra_entries(extras, points, components, prices)
    crossing = summaries.loc[summaries.status.eq("COMPLETE")].groupby(["model", "period", "entry_cause"]).agg(
        completed=("point_id", "size"), later_core_after_auxiliary_entry=("later_core_after_auxiliary_entry", "sum"),
        later_auxiliary_only_after_core_entry=("later_auxiliary_only_after_core_entry", "sum"),
        distinct_state_mean=("distinct_holding_states", "mean"), mean_net_return=("recomputed_net_return", "mean")).reset_index()
    for name, frame in (("逐日分量及持有状态", components), ("所有点位入场对应", entries), ("逐开盘持有区间", paths),
                        ("整笔路径与来源切换", summaries), ("入场分组全部指标", group_metrics), ("持有状态收益分解", state_metrics),
                        ("六次额外再入场对应", links), ("六次再入场所有重叠区间", overlaps), ("入场后来源接续汇总", crossing)):
        save_table(name, frame)
    summary = {
        "study": "510300_POINT_COMPONENT_PATH_ATTRIBUTION_V1", "at": common.now(), "status": "FIXED_POINT_COMPONENT_PATH_ATTRIBUTION_COMPLETE",
        "complete_point_records": int(points.status.eq("COMPLETE").sum()), "censored_point_records": int(points.status.ne("COMPLETE").sum()),
        "observed_open_intervals": len(paths), "maximum_net_return_recomputation_error": float(summaries.net_recomputation_error.abs().max()),
        "extra_entry_links": len(links), "extra_entry_relations": links.groupby(["model", "relation"]).size().rename("count").reset_index().to_dict("records"),
        "new_point_replays": 0, "new_full_accounts": 0, "new_model_fits": 0, "parameter_searches": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "orders_authorized": False,
    }
    common.save_json(OUT / "summary.json", summary)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="组合点位入场来源与持有路径归因")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
