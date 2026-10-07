"""固定多头空仓归因：覆盖全部等待日、上涨段及原20日标签，不生成新策略。"""
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
from research import upward_episode_anatomy_v1 as anatomy
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_forward_observer_v1 import NAMES
from research.point_forward_observer_inputs_v1 import require

OUT = ROOT / "reports/research/510300_point_long_wait_diagnostic_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OBSERVER = ROOT / "reports/research/510300_point_forward_observer_v1/seed"
REASONS = {"HELD": "已有多头点位", "NO_VIEW": "来源状态不完整", "POSITIVE_TARGET_BLOCKED": "有正目标但开盘进入受阻",
           "NO_CORE_NO_AUX": "核心与原始辅助都没有正目标", "AUX_REJECTED_BY_GATE": "核心无正目标、辅助被门槛挡住"}
CORE_REASONS = {"ALL_REFERENCE_STATES_ZERO": "三个底层参考均无正目标",
                "REFERENCE_POSITIVE_BUT_ROUTING_OR_BUDGET_ZERO": "底层有正目标、核心预算或路由归零",
                "CORE_POSITIVE": "核心正目标", "NO_VIEW": "核心来源不完整"}


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze():
    require(not OUT.exists(), "空仓归因已经登记，不能覆盖。")
    files = {}
    paths = {"prices.parquet": CURRENT / "inputs/candidate_prices.parquet", "data.parquet": CURRENT / "inputs/candidate_features.parquet",
             "dividends.csv": CURRENT / "inputs/dividends.csv", "factors.parquet": CURRENT / "results/STRESS/full_factors.parquet",
             "signals.parquet": CURRENT / "results/完整候选意向.parquet", "points.parquet": CURRENT / "results/全部自然点位.parquet",
             "events.parquet": CURRENT / "results/点位进出事件.parquet", "gaps.parquet": OBSERVER / "results/全部空仓间隔.parquet",
             "vintage.parquet": CURRENT / "results/references/ENTRY_VINTAGE_STRESS_decisions.parquet",
             "panic.parquet": CURRENT / "results/continuous/PANIC_ONLY_decisions.parquet",
             "ridge.parquet": CURRENT / "results/continuous/REARM_RIDGE_decisions.parquet"}
    for name, source in paths.items():
        target = OUT / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(target.relative_to(OUT))] = {"source": str(source.relative_to(ROOT)), "sha256": anatomy.digest(target)}
    for source in (Path(__file__), Path(anatomy.__file__), ROOT / "tests/test_point_long_wait_diagnostic_v1.py"):
        target = OUT / "code" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(target.relative_to(OUT))] = {"source": str(source.relative_to(ROOT)), "sha256": anatomy.digest(target)}
    protocol = {"study": "510300_POINT_LONG_WAIT_DIAGNOSTIC_V1", "at": anatomy.now(),
                "question": "固定两条多头在全部空仓日为何没有进入，特别是203及350交易日最长等待期间是否存在被错过的上涨段？",
                "scope": "2020-01-02至2026-09-30，日线及此前完整周；空头后置。",
                "partition": "按开盘执行后的实际持有状态分持有/空仓；空仓再按来源未知、有正目标但受阻、原始辅助被门槛挡住、核心与辅助均无正目标分组。",
                "core": "核心为零时，另区分三个底层参考状态全零，或底层有正目标但原预算/路由归零。保留底层原行动及退出原因。",
                "opportunity": "完全复用既有5%上涨及5%回落确认的上涨图谱定义，低高点为事后标签；覆盖全部合格上涨段，未完成段单列。",
                "episode_coverage": "从低点次日开盘至高点当日收盘，逐日看是否实际持有；上涨期间一日也未持有才标为整段未参与。这不是可成交的低买高卖收益。",
                "confirmation_context": "首次收盘涨到5%后，使用该收盘的真实已知信号，解释下一交易日是否空仓及原因；不使用未来确认日作为过去可用信息。",
                "daily_labels": "复用原全部原点的次日开盘至第20持有日收盘标签及严格摩擦。每天标签重叠，只描述结果，不作为策略交易。",
                "nonoverlap": "从既有2015年首个合格原点固定每20交易日取一原点，再截取2020年以后、再分空仓原因；绝不在每组筛选后重选相位。",
                "comparisons": "每组报告原点数、20日参考净均值、正收益比例、涨5%及跌5%比例；另分2020—2023、2024—2026。无策略pB、无最优组选择、无调参。",
                "frequency": "日数和图谱上涨段不当成新增可执行交易次数；本轮不会移除过滤器或复活已冻结失败规则。",
                "evidence": "已研究历史上的描述归因，没有独立验证或因果识别。",
                "new_strategy_replays": 0, "new_model_fits": 0, "orders_authorized": False, "goal_achieved": False}
    anatomy.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": anatomy.digest(OUT / "protocol.json")}
    anatomy.save_json(OUT / "freeze.json", {"at": anatomy.now(), "files": files})
    print("多头空仓归因已固定：全部等待日、原5%上涨段与原20日结果标签。", flush=True)


def classify_reason(held, target, core, raw_aux, effective_aux, gate):
    if held:
        return "HELD"
    if not np.isfinite([target, core, raw_aux, effective_aux, gate]).all():
        return "NO_VIEW"
    if target > 0:
        return "POSITIVE_TARGET_BLOCKED"
    require(target == 0 and core == 0 and effective_aux == 0, "零目标与来源不一致。")
    if raw_aux > 0:
        require(gate == 0, "原辅助为正但不是门槛挡住。")
        return "AUX_REJECTED_BY_GATE"
    require(raw_aux == 0, "原辅助目标非法。")
    return "NO_CORE_NO_AUX"


def fixed_phase(origin_indices, first_eligible, step=20):
    return (np.asarray(origin_indices, dtype=int)-int(first_eligible)) % step == 0


def daily_attribution(data, signals, factors, points, events, references, technical):
    lookup = factors.set_index("date")
    tech = technical.set_index("date")
    refs = {key: value.set_index("origin") for key, value in references.items()}
    rows = []
    for candidate in NAMES:
        local_points = points.loc[points.candidate.eq(candidate)]
        local = signals.loc[signals.candidate.eq(candidate) & signals.execution_date.le(data.date.iloc[-1])]
        for s in local.itertuples():
            held = ((local_points.entry_date <= s.execution_date)
                    & (local_points.exit_date.isna() | (local_points.exit_date > s.execution_date))).any()
            cause = classify_reason(held, s.target, s.core_target, s.aux_target, s.effective_auxiliary, s.gate)
            f, t = lookup.loc[s.origin], tech.loc[s.origin]
            v, p, r = (refs[key].loc[s.origin] for key in ("vintage", "panic", "ridge"))
            values = [v.reference_weight, p.reference_weight, r.reference_weight]
            if not np.isfinite(values+[s.core_target]).all():
                core_cause = "NO_VIEW"
            elif s.core_target > 0:
                core_cause = "CORE_POSITIVE"
            else:
                core_cause = "ALL_REFERENCE_STATES_ZERO" if not any(x > 0 for x in values) else "REFERENCE_POSITIVE_BUT_ROUTING_OR_BUDGET_ZERO"
            if cause == "POSITIVE_TARGET_BLOCKED":
                match = events.loc[events.candidate.eq(candidate) & events.date.eq(s.execution_date)]
                require(match.event.str.startswith("ENTRY_BLOCKED_").any(), "正目标空仓没有实际受阻事件。")
            # 把核心展开到三个底层状态，验证预算/路由归因对应真实核心。
            joint = f.selected_parent == "JOINT_DOWNSIDE_REFERENCE_PAIR"
            b, j, a = f.trend_noise_budget, f.downside_budget, f.panic_budget
            vintage_coefficient = b*f.ordinary_risk_multiplier + (1-b)*(j*f.downside_risk_multiplier if joint else f.ordinary_risk_multiplier)
            panic_coefficient = (1-b)*(1-j)*a if joint else 0.
            ridge_coefficient = (1-b)*(1-j)*(1-a) if joint else 0.
            composed = vintage_coefficient*values[0]+panic_coefficient*values[1]+ridge_coefficient*values[2]
            require(np.isclose(composed, s.core_target, atol=1e-12, rtol=0, equal_nan=True), "核心底层归因不相加。")
            rows.append({"candidate": candidate, "execution_date": s.execution_date, "origin": s.origin, "origin_index": s.origin_index,
                         "held": bool(held), "reason": cause, "core_reason": core_cause, "target": s.target,
                         "core_target": s.core_target, "raw_aux_target": s.aux_target, "effective_auxiliary": s.effective_auxiliary, "gate": s.gate,
                         "vintage_state": values[0], "panic_state": values[1], "ridge_state": values[2],
                         "vintage_coefficient": vintage_coefficient, "panic_coefficient": panic_coefficient, "ridge_coefficient": ridge_coefficient,
                         "vintage_action": v.action, "vintage_exit_reasons": v.exit_reasons, "panic_action": p.action, "ridge_action": r.action,
                         "above_ema20": t.above_ema20, "daily_hist": t.daily_hist, "daily_hist_rising": t.daily_hist_rising,
                         "weekly_last_date": t.weekly_last_date, "weekly_hist": t.weekly_hist, "weekly_hist_rising": t.weekly_hist_rising,
                         "relative_volume": t.relative_volume, "up_volume_balance5": t.up_volume_balance5, "rv_ratio": t.rv_ratio,
                         "drawdown60": f.drawdown60, "lag_direction": f.lag_direction, "runs_direction": f.runs_direction})
    return pd.DataFrame(rows)


def episode_coverage(episodes, daily, data):
    rows = []
    for ep in episodes.loc[episodes.admitted & episodes.bottom_date.ge("2020-01-02")].itertuples():
        for candidate in NAMES:
            span = daily.loc[daily.candidate.eq(candidate) & daily.execution_date.gt(ep.bottom_date) & daily.execution_date.le(ep.peak_date)]
            require(len(span) == ep.rise_sessions, "上涨段覆盖没有对应全部交易日。")
            confirmation = daily.loc[daily.candidate.eq(candidate) & daily.origin.eq(ep.confirm_up_date)]
            c = confirmation.iloc[0] if len(confirmation) else None
            flat = span.loc[~span.held]
            rows.append({"candidate": candidate, "episode_id": ep.episode_id, "status": ep.status,
                         "bottom_date": ep.bottom_date, "peak_date": ep.peak_date, "confirm_up_date": ep.confirm_up_date,
                         "confirm_down_date": ep.confirm_down_date, "retrospective_rise": ep.gross_rise,
                         "rise_sessions": ep.rise_sessions, "held_rising_sessions": int(span.held.sum()), "flat_rising_sessions": len(flat),
                         "entire_rise_unheld": bool(len(span) and not span.held.any()),
                         "flat_no_signal_sessions": int(flat.reason.eq("NO_CORE_NO_AUX").sum()),
                         "flat_gate_sessions": int(flat.reason.eq("AUX_REJECTED_BY_GATE").sum()),
                         "confirm_next_open": c.execution_date if c is not None else pd.NaT,
                         "confirm_next_open_reason": c.reason if c is not None else "NOT_YET_OBSERVED",
                         "confirm_core_reason": c.core_reason if c is not None else "NOT_YET_OBSERVED",
                         **{"confirm_"+key: c[key] if c is not None else np.nan for key in
                            ("above_ema20", "daily_hist", "daily_hist_rising", "weekly_hist", "weekly_hist_rising", "relative_volume", "up_volume_balance5", "rv_ratio")}})
    return pd.DataFrame(rows)


def describe_labels(daily, labels):
    joined = daily.merge(labels, left_on="origin", right_on="date", how="left", validate="many_to_one", suffixes=("", "_label"))
    require(joined.status.notna().all(), "诊断日期没有原结果标签。")
    phase_start = int(labels.idx.min())
    joined["fixed_nonoverlap_origin"] = fixed_phase(joined.idx, phase_start)
    joined["era"] = np.where(joined.origin.lt("2024-01-01"), "2020_2023", "2024_2026")
    results = []
    for candidate in NAMES:
        for era in ("ALL", "2020_2023", "2024_2026"):
            frame = joined.loc[joined.candidate.eq(candidate) & joined.status.eq("MATURE")]
            if era != "ALL":
                frame = frame.loc[frame.era.eq(era)]
            for clock in ("ALL_DAILY_OVERLAPPING", "FIXED_CALENDAR_NONOVERLAP"):
                chosen = frame.loc[frame.fixed_nonoverlap_origin] if clock.startswith("FIXED") else frame
                for reason in REASONS:
                    sample = chosen.loc[chosen.reason.eq(reason)]
                    results.append({"candidate": candidate, "era": era, "clock": clock, "reason": reason,
                                    "origins": len(sample), "mean_net_label": sample.net_reference_return.mean(),
                                    "positive_net_label_fraction": sample.net_reference_return.gt(0).mean() if len(sample) else np.nan,
                                    "rise5_fraction": sample.upside5.mean(), "fall5_fraction": sample.downside5.mean()})
    return joined, pd.DataFrame(results)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本诊断已经运行，不覆盖。")
    for name, item in read(OUT / "freeze.json")["files"].items():
        require(anatomy.digest(OUT / name) == item["sha256"], "诊断来源改变："+name)
        if name.startswith("code"):
            require(anatomy.digest(ROOT / item["source"]) == item["sha256"], "执行程序与固定版本不同。")
    anatomy.save_json(OUT / "RUN_STARTED.json", {"at": anatomy.now(), "new_strategy_replays": 0})
    frames = {key: pd.read_parquet(OUT / f"inputs/{key}.parquet") for key in
              ("data", "prices", "factors", "signals", "points", "events", "gaps", "vintage", "panic", "ridge")}
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    technical, _ = anatomy.features(frames["prices"], dividends)
    daily = daily_attribution(frames["data"], frames["signals"], frames["factors"], frames["points"], frames["events"],
                              {key: frames[key] for key in ("vintage", "panic", "ridge")}, technical)
    episodes = anatomy.upward_episodes(technical)
    coverage = episode_coverage(episodes, daily, frames["data"])
    print("全部实际空仓日已按信号和原门槛归因，继续对应既有上涨段和失败标签。", flush=True)
    labels = anatomy.label_origins(technical, dividends)
    tagged, descriptions = describe_labels(daily, labels)
    reason_counts = daily.groupby(["candidate", "reason", "core_reason"], dropna=False).size().rename("sessions").reset_index()
    action_counts = daily.loc[~daily.held].groupby(["candidate", "vintage_action", "vintage_exit_reasons"], dropna=False).size().rename("sessions").reset_index()
    long_rows = []
    for candidate in NAMES:
        gap = frames["gaps"].loc[frames["gaps"].candidate.eq(candidate)].sort_values("flat_sessions", ascending=False).iloc[0]
        span = daily.loc[daily.candidate.eq(candidate) & daily.execution_date.between(gap.flat_start, gap.last_flat_session)]
        require(len(span) == gap.flat_sessions and not span.held.any(), "最长空仓日与原记录不同。")
        wave = coverage.loc[coverage.candidate.eq(candidate) & coverage.status.eq("COMPLETE_RETROSPECTIVE")
                            & coverage.bottom_date.ge(gap.flat_start) & coverage.peak_date.le(gap.last_flat_session)]
        market = technical.loc[technical.date.between(gap.flat_start, gap.last_flat_session)]
        long_rows.append({**gap.to_dict(), "market_total_close_change": market.total_return_index.iloc[-1]/market.total_return_index.iloc[0]-1,
                          "contained_completed_upwaves": len(wave), "contained_entirely_unheld_upwaves": int(wave.entire_rise_unheld.sum()),
                          **{reason+"_sessions": int(span.reason.eq(reason).sum()) for reason in REASONS},
                          **{reason+"_sessions": int(span.core_reason.eq(reason).sum()) for reason in CORE_REASONS}})
    longest = pd.DataFrame(long_rows)
    for name, frame in (("全部逐日原因", daily), ("空仓底层行动", action_counts), ("全部原因计数", reason_counts),
                        ("全部上涨段覆盖", coverage), ("原点标签与固定相位", tagged), ("按原因的20日标签", descriptions), ("最长等待归因", longest)):
        save_table(name, frame)
    complete = coverage.loc[coverage.status.eq("COMPLETE_RETROSPECTIVE")]
    summary = {"study": "510300_POINT_LONG_WAIT_DIAGNOSTIC_V1", "at": anatomy.now(), "status": "ALL_LONG_WAIT_DAYS_ATTRIBUTED",
               "last_known_close": str(frames["data"].date.iloc[-1].date()), "candidate_day_comparisons": len(daily),
               "flat_candidate_day_comparisons": int((~daily.held).sum()), "unknown_candidate_days": int(daily.reason.eq("NO_VIEW").sum()),
               "completed_upwave_comparisons": len(complete), "entirely_unheld_upwave_comparisons": int(complete.entire_rise_unheld.sum()),
               "longest_wait": longest.to_dict("records"), "new_strategy_replays": 0, "new_model_fits": 0,
               "new_prospective_points": 0, "goal_achieved": False, "orders_authorized": False}
    anatomy.save_json(OUT / "summary.json", summary)
    print(json.dumps(anatomy.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断固定多头等待，保留原规则与失败对照。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()
