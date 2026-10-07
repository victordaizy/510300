"""统一解释全部价格确认、原上涨段及实际价格对照，不新增金融政策。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import post_repair_path_inputs_v1 as path_inputs
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_account_nr7_inputs_v1 import PARENT_A

PRIOR = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OUT = ROOT / "reports/research/510300_post_repair_path_explanation_v1"
CASE_IDS = [18, 37, 42, 55]
CLASS_NAMES = {
    "MOMENTUM_RETAINED": "动量保留", "MOMENTUM_REBUILT": "动量重建",
    "MOMENTUM_NOT_POSITIVE": "动量尚未转正", "NO_PREVIOUS_CONFIRMATION": "无已知上次确认",
    "NO_VIEW_CONTINUITY": "连续性未知",
}
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name: str, frame: pd.DataFrame) -> None:
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p.with_suffix(".parquet"), index=False)
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze() -> None:
    if (OUT / "protocol.json").exists():
        raise ValueError("本固定路径解释已经登记。")
    tests = read(OUT / "tests_receipt.json")
    if tests["passed"] != 3 or tests["exit_code"] != 0:
        raise ValueError("三项路径时钟测试尚未通过。")
    for source in read(PRIOR / "protocol.json")["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("原R161冻结来源已变化，不能拼接当前字段："+source["path"])
    files = [Path(__file__), Path(path_inputs.__file__), ROOT / "tests/test_post_repair_path_inputs_v1.py", OUT / "tests_receipt.json",
             PRIOR / "protocol.json", PRIOR / "summary.json", PRIOR / "saved_result_verification.json",
             PRIOR / "results/全部原点的当时已知修复顺序.parquet", PRIOR / "results/原186事件的顺序状态与旧结果标签.parquet",
             ATLAS / "protocol.json", ATLAS / "results/上涨段全集.parquet", ATLAS / "results/features.parquet",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
             WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet", WEIGHT / "inputs/risks.parquet",
             ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/point_fresh_repair_order_inputs_v1.py",
             ROOT / "research/upward_episode_anatomy_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/point_account_nr7_inputs_v1.py", OUT / "pre_registration_first_test_failure.json",
             OUT / "initial_path_inputs_before_null_fix.py"]
    for period in PERIODS:
        files.extend([CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet",
                      PRIOR / f"results/accounts/{period}/STRESS/PRICE_CONFIRMATION/trades.parquet"])
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_POST_REPAIR_PATH_EXPLANATION_V1", "technical_decision": "TECH.R162",
        "phase": "EXPLANATION_ONLY_NO_NEW_FINANCIAL_POLICY", "user_order": "先解释具体上涨段的量价和指标，再反推能提前识别的点位。",
        "previous_goal_turn_classification": "PROGRESS_NEW_COMPLETE_CASE_DERIVED_POLICY_EVIDENCE",
        "question": "价格修复失效后再次确认时，MACD动量是否一直保留、重建或尚未为正？这些状态是否解释延续、失败及原A覆盖？",
        "known_state": "每个原点只读取此前及当前的原字段。上一合格价格确认记录，随后首次收盘<=EMA20记失效；当前MACD正段出生<=上一确认且未中断则动量保留，之后重新出生则重建，当前非正单列。未知中断既有关系，不以两个正端点证明全段连续。",
        "membership": "全部当前3488日，2015起原价格确认全保留；原186成熟事件必须完整匹配，不以未来成熟筛当前状态，尾部无旧标签单列。",
        "outcomes": "只复用原186固定20日标签及R161纯价格对照保存实际周期；独立结果文件在当时状态计算后才合并。实际周期子集不是新的政策账户，开放周期保留。",
        "grouping": "唯一预先声明的动量路径分组，ALL及原2015-2019/2020-2023/2024-2026一起报告；不做量/周线/波动组合筛选、排名或阈值搜索。",
        "episodes": "保留原61条分段及原admitted/status，正式49上涨段完整展示原A持仓/已知目标覆盖；低高点仅回顾对齐，不进入当时状态。",
        "charts": "原已解释18/37/42/55四案例，价格/日MACD/量/周MACD与波动四面板；低点前10至高点后10日是固定图示范围，不是交易窗口。",
        "new_fits": 0, "new_training_labels": 0, "new_accounts": 0, "new_market_requests": 0,
        "necessary_tests": 3, "no_rescue": "R161严格顺序完整政策终态保持；本轮不是放宽该政策、调退出、改量窗口、改年份或选出最优再确认策略。",
        "independent_validation": "NOT_ESTABLISHED", "source_first_vintage": "NOT_CERTIFIED",
        "goal_achieved": False, "orders_authorized": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files],
    })
    print("全部确认事件的路径解释已固定；不运行新账户或拟合。", flush=True)


def plot_case(data, events, daily_context, episode) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    b, p = int(episode.bottom_idx), int(episode.peak_idx)
    part = data.iloc[max(0, b-10):min(len(data), p+11)].copy()
    context = daily_context.reindex(part.date)
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 11.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1.6, 1.5]})
    axes[0].plot(part.date, part.close, color="#284458", label="原价收盘", linewidth=1.8)
    axes[0].plot(part.date, part.ema20-part.cash_shift, color="#bc8123", label="EMA20（同原价单位）", linewidth=1.4)
    colors = {"MOMENTUM_RETAINED": "#177d62", "MOMENTUM_REBUILT": "#bc8123", "MOMENTUM_NOT_POSITIVE": "#af3b42",
              "NO_PREVIOUS_CONFIRMATION": "#777777", "NO_VIEW_CONTINUITY": "#777777"}
    ev = events.loc[events.date.between(part.date.iloc[0], part.date.iloc[-1])]
    for category, group in ev.groupby("momentum_path_class", sort=False):
        price = data.set_index("date").close.reindex(group.date)
        axes[0].scatter(group.date, price, marker="^", color=colors[category], s=50, label="当时确认："+CLASS_NAMES[category], zorder=5)
    hold = context.A_shares_at_origin.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(part.date, 0, 1, where=hold, color="#777777", alpha=.12, transform=axes[0].get_xaxis_transform(), label="原A实际已持仓")
    for row in ev.itertuples(index=False):
        axes[0].annotate(row.date.strftime("%m-%d"), (row.date, float(data.close.iloc[row.origin_index])),
                         xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("价格 / 元")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].bar(part.date, part.daily_hist, width=1.3, color=np.where(part.daily_hist.ge(0), "#af3b42", "#177d62"), alpha=.75, label="日MACD柱")
    axes[1].plot(part.date, part.daily_dif, color="#bc8123", linewidth=1.1, label="DIF")
    axes[1].plot(part.date, part.daily_dea, color="#284458", linewidth=1.1, label="DEA")
    axes[1].axhline(0, color="#777777", linewidth=.7)
    axes[1].set_ylabel("日MACD")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[2].bar(part.date, part.relative_volume, width=1.3, color="#8798a7", alpha=.65, label="量 / 前20日中位量")
    axes[2].axhline(1, color="#777777", linewidth=.7, linestyle="--")
    axes[2].set_ylabel("相对成交量")
    volume_axis = axes[2].twinx()
    volume_axis.plot(part.date, part.up_volume_balance5, color="#af3b42", linewidth=1.2, label="五日量方向")
    volume_axis.axhline(0, color="#af3b42", linewidth=.6, linestyle=":")
    volume_axis.set_ylim(-1.1, 1.1)
    volume_axis.set_ylabel("五日量方向")
    axes[2].legend(loc="upper left", fontsize=8)
    volume_axis.legend(loc="upper right", fontsize=8)
    axes[3].plot(part.date, part.weekly_hist, color="#284458", linewidth=1.3, label="上一完整周MACD柱")
    axes[3].axhline(0, color="#777777", linewidth=.7)
    axes[3].set_ylabel("周MACD柱")
    risk_axis = axes[3].twinx()
    risk_axis.plot(part.date, part.rv_ratio, color="#bc8123", linewidth=1.1, label="RV20 / 前252日中位值")
    risk_axis.axhline(1, color="#bc8123", linewidth=.6, linestyle=":")
    risk_axis.set_ylabel("波动率比")
    axes[3].legend(loc="upper left", fontsize=8)
    risk_axis.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.axvspan(episode.bottom_date, episode.peak_date, color="#bc8123", alpha=.04)
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"原案例{episode.episode_id}：修复、失效及再次确认的已知量价路径\n阴影上涨段仅作事后对齐；三角是价格确认，未生成新交易", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_修复至再次确认量价全路径.png"
    fig.savefig(OUT / name, dpi=155)
    plt.close(fig)
    return name


def run() -> None:
    protocol = read(OUT / "protocol.json")
    if (OUT / "RUN_STARTED.json").exists():
        raise ValueError("本路径解释已经开始，禁止重复运行。")
    for source in protocol["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("冻结来源改变："+source["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "phase": protocol["phase"], "new_financial_runs": 0})
    data, _, _, parents = load()
    sequence = pd.read_parquet(PRIOR / "results/全部原点的当时已知修复顺序.parquet")
    from research.point_fresh_repair_order_inputs_v1 import signals
    pd.testing.assert_frame_equal(sequence, signals(data), check_exact=True)
    known = path_inputs.known_paths(data, sequence)
    table("全部原点_当时已知修复与再确认", known)
    for date in ["2019-03-27", "2020-04-14", "2020-06-16", "2024-11-20"]:
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        check = path_inputs.known_paths(data.iloc[:n], sequence.iloc[:n])
        pd.testing.assert_frame_equal(check, known.iloc[:n].reset_index(drop=True), check_dtype=False, check_exact=True)

    context_parts, cycles = [], []
    for period in PERIODS:
        daily = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet")
        context = daily[["date", "shares", "exposure"]].rename(columns={"shares": "A_shares_at_origin", "exposure": "A_exposure_at_origin"})
        targets = parents[period][["origin", PARENT_A]].rename(columns={"origin": "date", PARENT_A: "A_known_target_at_origin"})
        context_parts.append(context.merge(targets, on="date", how="left", validate="one_to_one"))
        saved = pd.read_parquet(PRIOR / f"results/accounts/{period}/STRESS/PRICE_CONFIRMATION/trades.parquet")
        saved = saved[["cycle_id", "entry_origin", "entry_date", "exit_date", "status", "net_return", "net_pnl", "holding_sessions"]].copy()
        saved["original_account_period"] = period
        cycles.append(saved)
    daily_context = pd.concat(context_parts, ignore_index=True).set_index("date")
    if not daily_context.index.is_unique:
        raise ValueError("原A两个时期上下文重复。")
    events = known.loc[known.price_confirmation & known.date.ge("2015-01-01")].copy()
    features = data[["date", "close", "daily_dif", "weekly_hist", "weekly_last_date", "relative_volume", "rv_ratio", "breakout20"]]
    events = events.merge(features, on="date", how="left", validate="one_to_one").merge(daily_context.reset_index(), on="date", how="left", validate="one_to_one")
    if events.A_shares_at_origin.isna().any() or events.A_known_target_at_origin.isna().any():
        raise ValueError("当时A库存或目标缺失，不能填空仓。")
    events["A_inventory_or_known_positive_target"] = events.A_shares_at_origin.gt(0) | events.A_known_target_at_origin.gt(0)
    table("全部价格确认_当时路径与A已知覆盖", events)

    old = pd.read_parquet(PRIOR / "results/原186事件的顺序状态与旧结果标签.parquet")
    labelled = events.merge(old[["date", "idx", "status", "net_reference_return", "entry_date", "end_date"]].rename(columns={
        "status": "old_20d_label_status", "entry_date": "old_20d_entry_date", "end_date": "old_20d_end_date"}), on="date", how="left", validate="one_to_one")
    labelled["in_original_186"] = labelled.idx.notna()
    matched = labelled.loc[labelled.in_original_186]
    if len(matched) != 186 or not np.array_equal(matched.idx.to_numpy(int), matched.origin_index.to_numpy(int)):
        raise ValueError("原186成熟价格成员未完整按原点匹配。")
    labelled["old_20d_label_status"] = labelled.old_20d_label_status.fillna("NOT_IN_ORIGINAL_REFERENCE_NO_NEW_LABEL")
    table("原186结果与尾部未知_仅事后解释", labelled)
    actual_cycles = pd.concat(cycles, ignore_index=True)
    annotated_cycles = actual_cycles.merge(events, left_on="entry_origin", right_on="date", how="left", validate="one_to_one")
    if annotated_cycles.momentum_path_class.isna().any():
        raise ValueError("原实际价格周期没有当时合格价格确认。")
    table("原纯价格实际周期_路径归因而非新策略", annotated_cycles)

    groups = []
    for era, (start, end) in ERAS.items():
        label_era = labelled.loc[labelled.date.between(start, end)]
        cycle_era = annotated_cycles.loc[annotated_cycles.entry_origin.between(start, end)]
        for category in path_inputs.CLASSES:
            selected = label_era.loc[label_era.momentum_path_class.eq(category)]
            realised = cycle_era.loc[cycle_era.momentum_path_class.eq(category)]
            for outcome_source, values in [("OLD_20D_REFERENCE_LABEL", selected.loc[selected.in_original_186, "net_reference_return"]),
                                           ("SAVED_PRICE_POLICY_COMPLETE_CYCLE", realised.loc[realised.status.eq("COMPLETE"), "net_return"])]:
                groups.append({"era": era, "momentum_path_class": category, "outcome_source": outcome_source,
                               "all_known_price_events": len(selected), "old_reference_events": int(selected.in_original_186.sum()),
                               "unknown_old_result_events": int((~selected.in_original_186).sum()),
                               "A_inventory_or_positive_target_events": int(selected.A_inventory_or_known_positive_target.sum()),
                               "original_price_policy_open_cycles": int(realised.status.ne("COMPLETE").sum()),
                               **path_inputs.descriptive_returns(values)})
    groups = pd.DataFrame(groups)
    table("动量路径分层_旧标签及原实际周期全部报告", groups)

    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    episode_rows, mappings = [], []
    for ep in episodes.itertuples(index=False):
        local = events.loc[events.origin_index.between(ep.bottom_idx, ep.peak_idx)]
        window = data.date.iloc[int(ep.bottom_idx):int(ep.peak_idx)+1]
        a = daily_context.reindex(window)
        episode_rows.append({**ep._asdict(), "original_A_observed_sessions": int(a.A_shares_at_origin.notna().sum()),
                             "original_A_inventory_sessions": int(a.A_shares_at_origin.gt(0).sum()),
                             "original_A_positive_target_sessions": int(a.A_known_target_at_origin.gt(0).sum()),
                             "price_confirmation_events": len(local),
                             "momentum_retained_confirmation_events": int(local.momentum_path_class.eq("MOMENTUM_RETAINED").sum()),
                             "A_inventory_or_target_confirmation_events": int(local.A_inventory_or_known_positive_target.sum()),
                             "retrospective_alignment_only": True})
        for r in local.itertuples(index=False):
            mappings.append({"episode_id": ep.episode_id, "original_episode_status": ep.status, "original_admitted": ep.admitted,
                             "date": r.date, "origin_index": r.origin_index, "momentum_path_class": r.momentum_path_class,
                             "retrospective_alignment_only": True})
    episode_summary = pd.DataFrame(episode_rows)
    table("原上涨段全集_完整路径及A覆盖回顾", episode_summary)
    table("原上涨段与确认事件_仅事后对齐", pd.DataFrame(mappings))
    charts = [plot_case(data, events, daily_context, episodes.loc[episodes.episode_id.eq(i)].iloc[0]) for i in CASE_IDS]
    case_event_rows = events.loc[events.date.between("2020-03-23", "2020-07-13")]
    status = "COMPLETED_FIXED_ALL_EVENT_PATH_EXPLANATION_NO_NEW_POLICY"
    summary = {"at": now(), "technical_decision": "TECH.R162", "status": status,
               "all_origin_rows": len(known), "all_known_price_events": len(events), "original_reference_events": len(matched),
               "tail_events_without_new_labels": int((~labelled.in_original_186).sum()),
               "original_actual_price_cycles": len(annotated_cycles), "complete_original_price_cycles": int(annotated_cycles.status.eq("COMPLETE").sum()),
               "open_original_price_cycles": int(annotated_cycles.status.ne("COMPLETE").sum()),
               "original_episode_rows": len(episodes), "original_admitted_episode_rows": int(episodes.admitted.sum()),
               "charts": charts, "path_class_counts": events.momentum_path_class.value_counts().to_dict(),
               "A_inventory_or_positive_target_events": int(events.A_inventory_or_known_positive_target.sum()),
               "three_original_eras_reported": True, "fixed_descriptive_groups": groups.to_dict("records"),
               "known_2020_case_events": case_event_rows[["date", "momentum_path_class", "current_daily_hist", "current_volume_balance5", "weekly_hist", "relative_volume", "rv_ratio", "A_shares_at_origin", "A_known_target_at_origin"]].to_dict("records"),
               "necessary_tests_passed": 3, "actual_prefix_checks": 4, "unchanged_frozen_sources": len(protocol["sources"]),
               "new_fits": 0, "new_accounts": 0, "new_training_labels": 0, "new_market_requests": 0,
               "financial_strategy_decision_preserved": "TECH.R161", "actual_prediction_decision_preserved": "TECH.R158",
               "original_exit_decision_preserved": "TECH.R145", "independent_validation": "NOT_ESTABLISHED",
               "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False}
    write_json(OUT / "summary.json", summary)
    print(f"已解释全部{len(events)}确认、原186标签及{len(annotated_cycles)}原实际周期，保留{int(annotated_cycles.status.ne('COMPLETE').sum())}开放周期；四图完成，无新金融运行。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="解释原修复、失效及再次确认路径，不新跑策略。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
