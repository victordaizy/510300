"""完整日线向上缺口与回补时钟：先固定完整用途，再解释全体路径。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import full_daily_gap_inputs_v1 as rules
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_full_daily_gap_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OLD = ROOT / "reports/research/510300_downtrend_break_explanation_v1"
CASE_IDS = (18, 37, 42, 55)
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "完整日线缺口解释已经固定。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 4 and receipt["exit_code"] == 0, "四个缺口输入测试未通过。")
    for source in receipt["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "缺口输入不对应已通过测试。")
    intraday = read(ROOT / "reports/research/510300_intraday_imbalance_research.json")
    smc = read(ROOT / "reports/research/510300_smc_sweep_fvg_historical_v1/result.json")
    rr3 = read(ROOT / "reports/research/510300_state_mechanism_rr3_v1/result.json")
    old_review = [
        {"source": "research/research_intraday_imbalances.py", "actual_computation_status": intraday["status"],
         "actual_order_flow_status": intraday["pv_lc_status"],
         "definition": "同日连续十五分钟三柱，当前low>前二柱high，规定若干柱未来收盘和当日回补；是假设性日内端点研究，其PASS不是完整账户盈利准入。不能把分钟结果搬到本日线用途。"},
        {"source": "research/smc_sweep_fvg_historical_v1.py", "actual_status": smc["status"],
         "actual_portfolio_status": smc["portfolio_status"],
         "definition": "一分钟先试探昨日低点再收复、突破试探区间、三柱FVG及回测，多个T1/T2/T5端点，主事件门失败、组合NOT_RUN。原拒绝保持。"},
        {"source": "research/state_mechanism_rr3_v1.py", "actual_status": rr3["status"],
         "definition": "趋势回调/冲击修复事前目标失效与压力3:1开盘准入，gap_up字段只是原开盘买价过高取消的执行约束，不以相邻两个日线整日区间分离作为入场及缺口回补退出。"},
        {"source": "research/downtrend_break_inputs_v1.py", "actual_status": read(ROOT / "reports/research/510300_downtrend_break_study_v1/summary.json")["status"],
         "definition": "上一收盘已经确认双高双低下降，收盘首次跨H2，初始旧高并只上移确认低点；R173关闭。本次原字段high/low日内极值用于两个完整日区间分离及回补，不增加该规则的过滤或改变其2/2窗口。"},
    ]
    sources = [Path(__file__), Path(rules.__file__), ROOT / "tests/test_full_daily_gap_inputs_v1.py", OUT / "tests_receipt.json",
               CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv", WEIGHT / "inputs/risks.parquet",
               WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
               ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
               OLD / "results/2015起全部已知突破与量价A覆盖.parquet", OLD / "summary.json",
               ROOT / "reports/research/510300_downtrend_break_budget_attribution_v1/next_point_information_admission_boundary.json",
               ROOT / "reports/research/510300_downtrend_break_study_v1/summary.json",
               ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
               ROOT / "research/post_repair_path_inputs_v1.py", ROOT / "research/point_first_passage_study_v1.py",
               ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/daily_supply_test_v1.py",
               ROOT / "research/weekly_daily_technical_v1.py", ROOT / "research/directional_entry_timing_v1.py",
               ROOT / "research/sequential_patterns_regime_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
               ROOT / "research/research_intraday_imbalances.py", ROOT / "reports/research/510300_intraday_imbalance_research.json",
               ROOT / "research/smc_sweep_fvg_historical_v1.py", ROOT / "reports/research/510300_smc_sweep_fvg_historical_v1/result.json",
               ROOT / "research/state_mechanism_rr3_v1.py", ROOT / "reports/research/510300_state_mechanism_rr3_v1/result.json",
               ROOT / "research/downtrend_break_inputs_v1.py"]
    sources.extend(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet" for period in PERIODS)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_FULL_DAILY_UP_GAP_EXPLANATION_V1", "technical_decision": "TECH.R175",
        "definition": "当前完整日线现金平移low严格大于上一完整日线现金平移high；每对相邻真实交易日单独出生，无幅度/量/MACD/RV过滤，不把开盘跳空但日内已经重叠算完整缺口。low<=上一high表示价格区间接触/回补。向下缺口只观察，不交易。",
        "numeric_clock": "原OHLC与已发生分红皆为.001刻度，转整数tick再累计当时分红；严格相等不是向上缺口，避免浮点除息中性边界伪信号。非刻度或不一致报价NO_VIEW，未知分红坐标保留未知，不回填。",
        "future_complete_policy_intent_fixed_before_any_new_outcome_join": "空仓完整向上缺口出生，次真实开一次尝试，次开现金平移tick<=缺口下沿（上一日high）或坐标未知取消不追入；原下一原点若出现实质新缺口可独立出生。初始失效为上一high；持有中新的完整向上缺口只将线提高到max(原线,新缺口上一high)，当前已知整日low<=线即缺口已回补，次合法开退出。收盘仍在线上也不撤销当日已回补事实。未知自身不退出、原风险只减，无加仓、混A、固定2R/20日或期末清仓。",
        "purpose_distinction": "从相邻整日区间分离与回补建立独立完整入出用途；不是原H2加缺口门，非分钟FVG移植、改变确认窗口、开盘取消阈值或旧退出预测MSE营救。",
        "old_review": old_review, "dedup_limit": "有限完整用途核对，不声称全项目所有缺口表达未试；旧全部实际终态保持。",
        "population": "全部3488日、2015起所有缺口与回补/开放、原61分段/49正式波段/四固定案例及失败；MACD、量和RV只记录当时原值。金融沿原首日前一收盘起决定，预边界原点与期末无次开请求分列，不混入新标签。",
        "outcomes": "原2846结果标签、2826成熟/20删失/9尾部保留，只事后描述；缺口回补路径非账户收益或新训练标签，不以旧20日标签选择政策。",
        "necessary_tests": 4, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, exclusive=True)
    print("R175完整日线缺口及回补的唯一未来用途在结果连接前固定；不交易分钟或复活旧FVG。", flush=True)


def fill_paths(states, events):
    rows = []
    for event in events.itertuples(index=False):
        floor, changes, failure = int(event.gap_lower_ticks), 0, None
        for day in states.iloc[int(event.origin_index)+1:].itertuples(index=False):
            if not day.current_quote_known:
                continue
            if day.gap_event and day.gap_lower_ticks > floor:
                floor, changes = int(day.gap_lower_ticks), changes+1
            if day.known_cash_low_ticks <= floor:
                failure = day
                break
        rows.append({"birth_date": event.date, "birth_index": event.origin_index,
                     "initial_gap_lower_ticks": int(event.gap_lower_ticks), "final_floor_ticks": floor,
                     "floor_updates": changes, "failure_date": failure.date if failure is not None else pd.NaT,
                     "failure_index": int(failure.origin_index) if failure is not None else -1,
                     "status": "KNOWN_DAILY_GAP_FILLED" if failure is not None else "RIGHT_CENSORED",
                     "filled_day_close_above_floor": bool(failure.known_cash_close_ticks > floor) if failure is not None else None,
                     "not_an_account_or_executable_return": True})
    return pd.DataFrame(rows)


def plot(data, states, context, episode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lo, hi = max(0, int(episode.bottom_idx)-10), min(len(data), int(episode.peak_idx)+11)
    x, s = data.iloc[lo:hi], states.iloc[lo:hi]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    axes[0].plot(x.date, x.close, color="#253f57", label="原价收盘", linewidth=1.7)
    axes[0].fill_between(x.date, x.low, x.high, color="#8ca2b0", alpha=.18, label="完整日线高低区间")
    birth = s.gap_event.to_numpy(bool)
    down = s.down_gap_observation.to_numpy(bool)
    axes[0].scatter(x.date.loc[birth], x.close.loc[birth], marker="^", color="#a83550", s=45, zorder=5, label="完整向上缺口")
    axes[0].scatter(x.date.loc[down], x.close.loc[down], marker="v", color="#318878", s=30, zorder=5, label="向下缺口只观察")
    for i in np.flatnonzero(birth):
        origin = int(s.origin_index.iloc[i])
        row = s.iloc[i]
        label = row.date.strftime("%m-%d")
        axes[0].annotate(label, (row.date, x.close.iloc[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        initial = row.gap_lower_ticks*.001 - x.cash_shift.iloc[i]
        axes[0].plot([row.date, data.date.iloc[min(origin+1, len(data)-1)]], [initial, initial], color="#a83550", alpha=.6, linewidth=1)
    held = context.reindex(x.date).A_shares.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(x.date, 0, 1, where=held, transform=axes[0].get_xaxis_transform(), color="#777777", alpha=.13, label="原A实际持仓")
    axes[0].set_ylabel("价格/元")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].plot(x.date, x.daily_dif, color="#b48125", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#253f57", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#a83550", "#318878"), width=1.2, alpha=.35, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.6)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[1].set_ylabel("动量原值")
    axes[2].bar(x.date, x.relative_volume, color="#8ca2b0", width=1.2, alpha=.65, label="量/前20日中位量")
    axes[2].axhline(1, color="#777777", linewidth=.6, linestyle="--")
    other = axes[2].twinx()
    other.plot(x.date, x.rv_ratio, color="#b48125", linewidth=1.1, label="RV20/前252日中位值")
    axes[2].legend(loc="upper left", fontsize=8)
    other.legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("相对量")
    other.set_ylabel("波动率比")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#777777", linestyle=":", linewidth=.8)
        ax.grid(alpha=.15)
    fig.suptitle(f"原案例{episode.episode_id}：整日区间分离、量价与缺口回补时钟\n三角为当日收盘后可知的缺口，竖线仅为原事后波段回顾对齐", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_完整日线缺口与量价.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "固定完整缺口解释已经开始，不重复运行。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "固定解释来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, _ = load()
    states = rules.gap_states(data)
    table("全部原点_完整日线缺口及整数现金坐标", states)
    prefix_checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        prefix = rules.gap_states(data.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix, states.iloc[:n].reset_index(drop=True), check_exact=True)
        prefix_checks.append({"through": date, "rows": n, "status": "EXACT_PAST_DAILY_GAPS_AND_CASH_TICKS"})
    previous = pd.read_parquet(OLD / "results/2015起全部已知突破与量价A覆盖.parquet")
    context = previous[["date", "A_shares", "A_exposure", "A_known_target"]].set_index("date")
    fields = ["date", "open", "high", "low", "close", "volume", "cash_shift", "ema20", "daily_dif", "daily_hist",
              "weekly_hist", "weekly_last_date", "relative_volume", "up_volume_balance5", "rv_ratio"]
    known_all = states.merge(data[fields], on="date", validate="one_to_one").merge(context.reset_index(), on="date", how="left", validate="one_to_one")
    table("全部3488已知缺口与量价_前2015A未知保留", known_all)
    known = known_all.loc[known_all.date.ge("2015-01-01")].copy()
    original = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(original[["date", "idx", "status", "net_reference_return"]].rename(columns={"status": "old_label_status"}),
                         on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(original), "原2846标签没有完整保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int), labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "旧标签与原点错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签与尾部未知_仅解释", labels)
    events = labels.loc[labels.gap_event].copy()
    table("全部完整向上缺口_当时量价及旧标签", events)
    table("全部完整向下缺口_只观察不交易", labels.loc[labels.down_gap_observation])
    paths = fill_paths(states, events)
    table("全部向上缺口至只上移回补_仅路径", paths)
    diagnostics = []
    for era, (start, end) in ERAS.items():
        local = events.loc[events.date.between(start, end)]
        mature = local.loc[local.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "known_events": len(local), "old_labels_unknown": len(local)-len(mature),
                            **descriptive_returns(mature.net_reference_return)})
    table("完整缺口_全时期旧标签描述", pd.DataFrame(diagnostics))
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, snapshots, cases = [], [], []
    for ep in episodes.itertuples(index=False):
        local = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        mapped.append({**ep._asdict(), "full_up_gap_births": int(local.gap_event.sum()),
                       "full_down_gap_observations": int(local.down_gap_observation.sum()),
                       "A_inventory_sessions": int(local.A_shares.gt(0).sum()), "retrospective_alignment_only": True})
    table("原61分段及49正式波段_完整日线缺口覆盖", pd.DataFrame(mapped))
    for identifier in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]
        local = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10), min(len(data)-1, int(ep.peak_idx)+10))].copy()
        local["original_episode_id"], local["retrospective_alignment_only"] = identifier, True
        snapshots.append(local)
        cases.append({"episode_id": identifier, "original_gross_rise": float(ep.gross_rise), "rise_sessions": int(ep.rise_sessions),
                      "events_between_bottom_and_peak": local.loc[local.origin_index.between(ep.bottom_idx, ep.peak_idx) & local.gap_event].to_dict("records"),
                      "events_in_fixed_case_window": local.loc[local.gap_event].to_dict("records"),
                      "original_wave_only_retrospective_alignment": True})
    table("四案例逐日完整量价及整日缺口", pd.concat(snapshots, ignore_index=True))
    boundary = []
    for period, (start, end) in PERIODS.items():
        first = int(np.flatnonzero(data.date.ge(start))[0])
        event = known_all.iloc[first-1]
        boundary.append({"period": period, "original_pre_first_session_origin": event.date,
                         "gap_event": bool(event.gap_event), "A_inventory_is_not_this_initialized_account": True,
                         "inside_old_2015_label_population": bool(event.date >= pd.Timestamp("2015-01-01")),
                         "old_label_or_return_created": False})
    charts = [plot(data, states, context, episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]) for identifier in CASE_IDS]
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R175", "status": "COMPLETED_ALL_FULL_DAILY_GAP_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(data), "origins_since_2015": len(known), "full_up_gaps_since_2015": len(events),
        "full_down_gaps_observation_only": int(known.down_gap_observation.sum()),
        "completed_gap_fill_paths": int(paths.failure_index.ge(0).sum()), "open_gap_fill_paths": int(paths.failure_index.lt(0).sum()),
        "known_filled_but_close_above_floor": int(paths.filled_day_close_above_floor.eq(True).sum()),
        "original_episodes": len(episodes), "original_admitted_waves": int(episodes.admitted.sum()),
        "original_old_labels": len(original), "original_mature_labels": int(original.status.eq("MATURE").sum()),
        "original_censored_labels": int(original.status.ne("MATURE").sum()), "tail_origins_without_new_label": int(labels.idx.isna().sum()),
        "case_records": cases, "old_label_diagnostics": diagnostics, "original_pre_session_boundary_origins": boundary,
        "charts": charts, "actual_prefix_checks": prefix_checks, "necessary_tests": 4, "frozen_sources": len(protocol["sources"]),
        "future_complete_policy_intent_fixed": protocol["future_complete_policy_intent_fixed_before_any_new_outcome_join"],
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"R175：全部3488状态、{len(events)}完整向上缺口、原61/49段及四例解释完成；未运行金融账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日线完整缺口：用途先固定，全体路径和量价后解释。")
    parser.add_argument("command", choices=("freeze", "run"))
    command = parser.parse_args().command
    {"freeze": freeze, "run": run}[command]()
