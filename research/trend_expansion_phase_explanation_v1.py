"""解释均线上方趋势展开的日周时差与量价事件，原结果只作诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import trend_expansion_phase_inputs_v1 as phase
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.post_repair_path_explanation_v1 import read, write_json, digest, now
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_trend_expansion_phase_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
CLOSED = ROOT / "reports/research/510300_offline_daily_policy_batch_v2_closeout_v1/results_closeout/reproduction_code/run_next.py"
CASE_IDS = [18, 37, 42, 55]
EVENTS = {"JOINT_PHASE_ONSET": "joint_phase_onset", "RANGE20_BREAK_ONSET": "range20_breakout_onset", "RELATIVE_VOLUME_ONSET": "relative_volume_expansion_onset"}
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}


def table(name, data):
    target = OUT / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target.with_suffix(".parquet"), index=False)
    data.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "protocol.json").exists():
        raise ValueError("本展开解释已登记。")
    tests = read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 3:
        raise ValueError("三个阶段时钟测试未通过。")
    paths = [Path(__file__), Path(phase.__file__), ROOT / "tests/test_trend_expansion_phase_inputs_v1.py", OUT / "tests_receipt.json",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv", WEIGHT / "inputs/risks.parquet",
             WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
             ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/post_repair_path_explanation_v1.py",
             ROOT / "research/post_repair_path_inputs_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
             ROOT / "research/directional_entry_timing_v1.py", ROOT / "research/sequential_patterns_regime_v1.py",
             ROOT / "research/weekly_daily_technical_v1.py", ROOT / "config/weekly_daily_technical_v1.yaml",
             ROOT / "research/compression_confirmed_entry_v1.py", ROOT / "research/breakout_retest_entry_v1.py",
             ROOT / "research/daily_supply_test_v1.py", ROOT / "research/price_volume_coherence_inputs_v1.py", CLOSED]
    paths.extend(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet" for period in PERIODS)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_TREND_EXPANSION_PHASE_EXPLANATION_V1", "technical_decision": "TECH.R163",
        "scope": "先解释价格已经在均线上方的趋势展开，再反推当时可识别点位。",
        "old_review": "直接RANGE20_BREAK为20日新高首日及5日结构/2R/20日到期；压缩与突破回踩各有已完成失败；旧周日T_ONLY为周价格/EMA20斜率与DIF>DEA状态及回踩触发；已终态TREND_NO_SLOPE为财富MA60连续确认，REPAIR_SEQUENCE为z5极端后修复。价量相关为20日Pearson两次确认。此处联合阶段出生与状态失效是不同信息时钟的完整用途提案，有限核对不声称全项目穷尽去重。",
        "phase": "只用原EMA20、日MACD12/26/9的DIF>0、上一完整周MACD柱>0；价格在EMA20上分四日周正/非正象限，价格不在上单列，未知单列。",
        "onset": "当前三项都正且上一原点字段已知、上一三项未全成立时才为联合阶段出生；预热或未知恢复到正状态不伪造出生。",
        "observational_channels": "既有20日突破状态首次进入、既有相对量>=1.5首次进入，全部保留；不是其他候选或组合搜索。",
        "future_complete_policy_intent_fixed_before_old_label_join": "唯一JOINT_PHASE_START：联合阶段出生次真实开尝试；阶段任一条件在已知收盘不再成立则次合法开退出，风险只减，不固定2R或20日到期。不因后面的旧20日诊断改条件或选择另一个政策。",
        "outcome": "仅复用原2846标签（2826成熟/20末期未知），当前尾部无旧标签单列，不新增标签。只作解释，未计算策略收益夏普。",
        "membership": "全部3488日先生成已知状态；当前2015起2855日及所有三类事件都保留，原61分段和49admitted原样；事后低高点只做回顾对齐。",
        "necessary_tests": 3, "new_fits": 0, "new_training_labels": 0, "new_accounts": 0, "new_market_requests": 0,
        "no_rescue": "原R161严格修复及全部旧终态保持；不改EMA/MACD/量/波动窗口、门槛或已有失败退出营救。",
        "independent_validation": "NOT_ESTABLISHED", "source_first_vintage": "NOT_CERTIFIED", "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    })
    print("日周展开阶段和唯一未来完整用途已在结果连接前固定；没有金融运行。", flush=True)


def plot(data, state, context, episode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lo, hi = max(0, int(episode.bottom_idx)-10), min(len(data), int(episode.peak_idx)+11)
    x, s = data.iloc[lo:hi], state.iloc[lo:hi]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.7, 1.6]})
    axes[0].plot(x.date, x.close, label="原价收盘", color="#284458", linewidth=1.7)
    axes[0].plot(x.date, x.ema20-x.cash_shift, label="EMA20（同原价）", color="#bc8123", linewidth=1.2)
    positive = s.joint_phase_active.fillna(False).to_numpy(bool)
    axes[0].fill_between(x.date, 0, 1, where=positive, transform=axes[0].get_xaxis_transform(), color="#2b779d", alpha=.08, label="已知日周同步正阶段")
    a = context.reindex(x.date)
    held = a.A_shares.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(x.date, 0, 1, where=held, transform=axes[0].get_xaxis_transform(), color="#777777", alpha=.18, label="原A实际已持仓")
    birth = s.joint_phase_onset.to_numpy(bool)
    axes[0].scatter(x.date.loc[birth], x.close.loc[birth], color="#af3b42", marker="o", s=48, label="联合阶段出生", zorder=5)
    br = s.range20_breakout_onset.to_numpy(bool)
    axes[0].scatter(x.date.loc[br], x.close.loc[br], color="#177d62", marker="^", s=30, label="既有20日突破首日", zorder=4)
    for date, price in zip(x.date.loc[birth], x.close.loc[birth]):
        axes[0].annotate(date.strftime("%m-%d"), (date, price), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[0].set_ylabel("价格 / 元")
    axes[1].plot(x.date, x.daily_dif, color="#bc8123", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#284458", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#af3b42", "#177d62"), alpha=.3, width=1.2, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.7)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[1].set_ylabel("动量原值")
    axes[2].bar(x.date, x.relative_volume, color="#8798a7", alpha=.65, width=1.2, label="量 / 前20日中位量")
    axes[2].axhline(1, color="#777777", linewidth=.7, linestyle="--")
    axes[2].set_ylabel("相对量")
    other = axes[2].twinx()
    other.plot(x.date, x.rv_ratio, color="#bc8123", linewidth=1.1, label="RV20 / 前252日中位值")
    other.set_ylabel("波动率比")
    axes[2].legend(loc="upper left", fontsize=8)
    other.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#999999", linestyle=":", linewidth=.8)
        ax.grid(alpha=.15)
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"原案例{episode.episode_id}：均线上方的展开时钟\n圆点为当时联合阶段出生，三角为旧突破；竖线为原5%确认，未生成新交易", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_日周展开与量价时钟.png"
    fig.savefig(OUT / name, dpi=155)
    plt.close(fig)
    return name


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise ValueError("本展开解释已开始，不重跑。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("原固定来源变化："+source["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_financial_runs": 0})
    data, _, _, parents = load()
    state = phase.phase_states(data)
    table("全部原点_当时日周展开阶段", state)
    for date in ["2019-01-18", "2020-06-08", "2024-09-30"]:
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        pd.testing.assert_frame_equal(phase.phase_states(data.iloc[:n].reset_index(drop=True)), state.iloc[:n].reset_index(drop=True), check_exact=True)
    context = []
    for period in PERIODS:
        a = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet")
        a = a[["date", "shares", "exposure"]].rename(columns={"shares": "A_shares", "exposure": "A_exposure"})
        target = parents[period][["origin", PARENT_A]].rename(columns={"origin": "date", PARENT_A: "A_known_target"})
        context.append(a.merge(target, on="date", validate="one_to_one", how="left"))
    context = pd.concat(context, ignore_index=True).set_index("date")
    features = data[["date", "close", "daily_hist", "relative_volume", "up_volume_balance5", "rv_ratio", "breakout20"]]
    known = state.loc[state.date.ge("2015-01-01")].merge(features, on="date", validate="one_to_one").merge(context.reset_index(), on="date", how="left", validate="one_to_one")
    table("2015起完整已知阶段与A覆盖", known)
    old = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(old[["date", "idx", "status", "net_reference_return"]].rename(columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    if int(labels.idx.notna().sum()) != len(old) or not np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int), labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)):
        raise ValueError("原全部每日标签没有按原点完整保留。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原每日结果标签及当前尾部_仅解释", labels)
    rows, event_rows = [], []
    for name, column in EVENTS.items():
        event_rows.append(known.loc[known[column]].assign(event_type=name))
        for era, (start, end) in ERAS.items():
            x = labels.loc[labels[column] & labels.date.between(start, end)]
            mature = x.loc[x.old_label_status.eq("MATURE")]
            rows.append({"event_type": name, "era": era, "known_events": len(x), "unknown_old_labels": len(x)-len(mature),
                         **descriptive_returns(mature.net_reference_return)})
    stats = pd.DataFrame(rows)
    table("三类原事件_全时期旧标签诊断", stats)
    event_rows = pd.concat(event_rows, ignore_index=True)
    table("全部展开突破与放量事件_当时原值", event_rows)
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped = []
    for ep in episodes.itertuples(index=False):
        x = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        row = ep._asdict()
        row.update({"after_confirmation_sessions": len(x), "known_joint_positive_sessions": int(x.joint_phase_active.fillna(False).sum()),
                    "joint_phase_onsets_after_confirm": int(x.joint_phase_onset.sum()),
                    "breakout_onsets_after_confirm": int(x.range20_breakout_onset.sum()),
                    "volume_onsets_after_confirm": int(x.relative_volume_expansion_onset.sum()),
                    "A_inventory_sessions_after_confirm": int(x.A_shares.gt(0).sum()), "retrospective_alignment_only": True})
        mapped.append(row)
    mapped = pd.DataFrame(mapped)
    table("原61分段_确认之后完整展开覆盖", mapped)
    case_records, snapshots = [], []
    for i in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(i)].iloc[0]
        x = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10), min(len(data)-1, int(ep.peak_idx)+10))].copy()
        x["original_episode_id"] = i
        x["retrospective_alignment_only"] = True
        snapshots.append(x)
        case_records.extend(x.loc[x.joint_phase_onset].to_dict("records"))
    table("四案例逐日阶段原值_只作回顾对齐", pd.concat(snapshots, ignore_index=True))
    charts = [plot(data, state, context, episodes.loc[episodes.episode_id.eq(i)].iloc[0]) for i in CASE_IDS]
    summary = {"at": now(), "technical_decision": "TECH.R163", "status": "COMPLETED_FIXED_TREND_EXPANSION_EXPLANATION_NO_FINANCIAL_RESULT",
               "all_daily_rows": len(data), "daily_origins_since_2015": len(known), "original_old_label_rows": len(old),
               "original_mature_label_rows": int(old.status.eq("MATURE").sum()), "original_right_censored_labels_preserved": int(old.status.ne("MATURE").sum()),
               "tail_origins_without_new_labels": int(labels.idx.isna().sum()), "joint_phase_onsets": int(known.joint_phase_onset.sum()),
               "range20_onsets": int(known.range20_breakout_onset.sum()), "relative_volume_onsets": int(known.relative_volume_expansion_onset.sum()),
               "original_episodes_preserved": len(mapped), "original_admitted_waves_preserved": int(mapped.admitted.sum()),
               "case_joint_phase_onsets": case_records, "old_label_diagnostics": stats.to_dict("records"), "charts": charts,
               "necessary_tests": 3, "actual_prefix_checks": 3, "frozen_sources": len(protocol["sources"]),
               "new_fits": 0, "new_accounts": 0, "new_training_labels": 0, "new_market_requests": 0,
               "future_complete_policy_intent_fixed": protocol["future_complete_policy_intent_fixed_before_old_label_join"],
               "latest_financial_strategy_decision_preserved": "TECH.R161", "independent_validation": "NOT_ESTABLISHED",
               "overfitting_removed": False, "goal_achieved": False}
    write_json(OUT / "summary.json", summary)
    print(f"已解释{len(known)}原点与{int(known.joint_phase_onset.sum())}联合阶段出生；原每日标签/61段完整保留，四图完成，无金融运行。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="先解释日周展开阶段，不运行新金融账户。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()


if __name__ == "__main__":
    main()
