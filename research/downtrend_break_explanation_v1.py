"""下降结构首次被打破的具体量价路径；唯一完整用途在连接结果前固定。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import downtrend_break_inputs_v1 as rules
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_downtrend_break_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
PREVIOUS = ROOT / "reports/research/510300_confirmed_structure_study_v1"
OLD_STRUCTURE = ROOT / "reports/research/510300_confirmed_structure_explanation_v1"
CASE_IDS = (18, 37, 42, 55)
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "首次下降结构突破解释已登记。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["exit_code"] == 0 and tests["passed"] == 4, "四项突破输入测试未通过。")
    for s in tests["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "突破输入不再对应已通过测试。")
    graph = read(ROOT / "reports/backtest/graph_regime_martin_turtle_v2.json")
    old_review = [
        {"source": "research/confirmed_structure_inputs_v1.py", "actual_status": read(PREVIOUS / "summary.json")["status"],
         "definition": "R170最近两高两低同时抬升的出生及结构失效。原2/2组件复用，本轮为下降结构尚在时首次突破旧高点，持有失效基于被突破高点及只上移确认低点；不是缩短原确认或调整原窗口。"},
        {"source": "research/simple_volume_reversal_v1.py", "actual_status": read(ROOT / "reports/research/510300_simple_volume_reversal_v1/acceptance_outcome.json")["status"],
         "definition": "V3前20日高突破且量>=1.5，前10日低/6%亏损及跟踪/60日等退出；V5当日低破20日低收回且收盘位置>=0.6，均值及3%亏损/5%目标/5日退出；其余急跌、缩量或波动反转，六实际候选目标未达。"},
        {"source": "research/daily_supply_test_v1.py", "actual_status": read(ROOT / "reports/research/510300_daily_supply_test_v1/result.json")["status"],
         "definition": "先20日低收复、10日内较高低点回测、3日内破测试高；缩量缩幅，事件低减0.5ATR失效/原2R目标/20日。固定研究已拒绝。"},
        {"source": "research/graph_regime_martin_turtle_v2.py", "registered_pass_count": graph["registered_pass_count"],
         "actual_decisions": {k: v["decision"] for k, v in graph["variants"].items()},
         "definition": "3/3 OHLC、ATR幅度及价格/DIF配对背离；入场20日突破或震荡底背离，10日低/顶背离、均值/ATR/到期等失效，未定义本次严格收盘上一确认下降高点的首次突破。当前实际报告六轨道FAIL或INSUFFICIENT_EVIDENCE，原结论保持。"},
        {"source": "research/factor96_rapid_structure_v1.py", "worth_followup": read(ROOT / "reports/research/510300_factor96_rapid_structure_v1/result.json")["worth_followup"],
         "definition": "名称虽有structure，实际为IF合约OI集中度及固定三ETF份额迁移代理，不是价格枢轴，不能作为重复价格用途。实际worth_followup为空，原失败保持。"},
    ]
    sources = [Path(__file__), Path(rules.__file__), ROOT / "research/confirmed_structure_inputs_v1.py",
               ROOT / "tests/test_downtrend_break_inputs_v1.py", OUT / "tests_receipt.json",
               OUT / "initial_input_test_failure.json", OUT / "initial_input_failure_explanation.json", OUT / "initial_inputs_before_anchor_dtype_fix.py",
               PREVIOUS / "next_structure_break_explanation_proposal.json", PREVIOUS / "summary.json",
               CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv", WEIGHT / "inputs/risks.parquet",
               WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
               ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
               OLD_STRUCTURE / "results/全部原点_已确认高低结构.parquet",
               ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
               ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/post_repair_path_inputs_v1.py",
               ROOT / "research/point_first_passage_study_v1.py", ROOT / "research/adaptive_allocation_v1.py",
               ROOT / "research/weekly_daily_technical_v1.py", ROOT / "research/directional_entry_timing_v1.py",
               ROOT / "research/sequential_patterns_regime_v1.py", ROOT / "research/daily_supply_test_v1.py",
               ROOT / "research/graph_regime_martin_turtle_v2.py", ROOT / "config/graph_regime_martin_turtle_v2.yaml",
               ROOT / "scripts/run_graph_regime_martin_turtle_v2.py", ROOT / "reports/backtest/graph_regime_martin_turtle_v2.json",
               ROOT / "research/simple_volume_reversal_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
               ROOT / "config/510300_simple_volume_reversal_v1.json", ROOT / "reports/research/510300_simple_volume_reversal_v1/acceptance_outcome.json",
               ROOT / "reports/research/510300_daily_supply_test_v1/result.json", ROOT / "research/factor96_rapid_structure_v1.py",
               ROOT / "reports/research/510300_factor96_rapid_structure_v1/result.json"]
    sources.extend(CONTROL / p / "STRESS/A_SAVED_WEIGHT/daily.parquet" for p in PERIODS)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_DOWNTREND_BREAK_EXPLANATION_V1", "technical_decision": "TECH.R171",
        "definition": "原严格2/2日收盘交替枢轴、当前更极端尾点更新时钟保持。上一原点已经确认至少四个交替点，两高和两低都严格下降；上一收盘<=上一已确认最新高点H2、当日现金平移收盘>同一个H2。只接受同一H2中心/确认身份首次合格越过，未知跨越不补、同锚反复越过只描述。",
        "future_complete_policy_intent_fixed_before_any_new_outcome_join": "只在首次合格下降结构突破且空仓时次真实开一次尝试。初始失效线为信号前一原点已确认、当日刚突破的高点H2；次开现金平移价<=该已知线取消，不追入。持有后只在已知原点用最新已确认低点将失效线取max向上移；已知收盘<=该线次合法开退出，未知本身不退出。风险只减，无加仓、混A、固定2R或20日；期末不人工清仓。",
        "old_review": old_review, "dedup_limit": "有限完整用途核对，不声称全项目穷尽去重；同组件不等于同用途。",
        "population": "全部3488状态、2015起全部首次突破/同锚反复/失效、原61分段/49正式波段及四例；不只展示赢家，不由结果加量/MACD/RV过滤。",
        "outcomes": "原2846每日结果标签原样及当前9尾部未知保留；出生至只上移失效线的收盘路径仅描述，不是资金账户或训练标签。",
        "necessary_tests": 4, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, exclusive=True)
    print("TECH.R171首次下降结构突破、只上移确认失效线及旧完整用途在结果连接前固定。", flush=True)


def close_paths(states, events):
    paths = []
    for event in events.itertuples(index=False):
        stop, failure, stop_updates = float(event.initial_broken_high_stop), None, 0
        for row in states.iloc[int(event.origin_index)+1:].itertuples(index=False):
            if not row.structure_known:
                continue
            low = float(row.latest_confirmed_low_price)
            if np.isfinite(low) and low > stop:
                stop, stop_updates = low, stop_updates+1
            if row.known_cash_adjusted_close <= stop:
                failure = row
                break
        paths.append({"birth_date": event.date, "birth_index": event.origin_index, "initial_stop": event.initial_broken_high_stop,
                      "final_known_stop": stop, "stop_updates": stop_updates,
                      "failure_date": failure.date if failure is not None else pd.NaT,
                      "failure_index": int(failure.origin_index) if failure is not None else -1,
                      "status": "KNOWN_CLOSE_STOP_FAILED" if failure is not None else "RIGHT_CENSORED",
                      "close_path_return_only": float(failure.known_cash_adjusted_close/event.known_cash_adjusted_close-1) if failure is not None else np.nan,
                      "not_account_or_executable_return": True})
    return pd.DataFrame(paths)


def plot(data, states, context, episode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lo, hi = max(0, int(episode.bottom_idx)-10), min(len(data), int(episode.peak_idx)+11)
    x, s = data.iloc[lo:hi], states.iloc[lo:hi]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.5]})
    axes[0].plot(x.date, x.close, color="#253f57", label="原价收盘", linewidth=1.7)
    axes[0].plot(x.date, x.ema20-x.cash_shift, color="#b48125", label="EMA20（同原价）", linewidth=1)
    axes[0].step(x.date, s.prior_H2_price.to_numpy()-x.cash_shift.to_numpy(), where="post", color="#b55069", alpha=.8, label="前收盘已确认最新高点", linewidth=1.1)
    axes[0].step(x.date, s.latest_confirmed_low_price.to_numpy()-x.cash_shift.to_numpy(), where="post", color="#318878", alpha=.7, label="当时最新已确认低点", linewidth=1.1)
    down = s.prior_complete_downtrend.to_numpy(bool)
    held = context.reindex(x.date).A_shares.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(x.date, 0, 1, where=down, transform=axes[0].get_xaxis_transform(), color="#b55069", alpha=.07, label="前收盘已确认下降结构")
    axes[0].fill_between(x.date, 0, 1, where=held, transform=axes[0].get_xaxis_transform(), color="#777777", alpha=.16, label="原A实际持仓")
    birth = s.break_event.to_numpy(bool)
    rising = s.old_rising_structure_onset.to_numpy(bool)
    axes[0].scatter(x.date.loc[birth], x.close.loc[birth], marker="^", color="#a83550", s=50, zorder=5, label="首次突破确认下降高点")
    axes[0].scatter(x.date.loc[rising], x.close.loc[rising], marker="o", color="#318878", s=35, zorder=4, label="原双抬升出生（已失败对照）")
    for date, price in zip(x.date.loc[birth], x.close.loc[birth]):
        axes[0].annotate(date.strftime("%m-%d"), (date, price), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("价格 / 元")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].plot(x.date, x.daily_dif, color="#b48125", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#253f57", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#a83550", "#318878"), width=1.2, alpha=.35, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.6)
    axes[1].set_ylabel("动量原值")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[2].bar(x.date, x.relative_volume, color="#8898a8", width=1.2, alpha=.7, label="量 / 前20日中位量")
    axes[2].axhline(1, color="#777777", linestyle="--", linewidth=.6)
    axes[2].set_ylabel("相对量")
    other = axes[2].twinx()
    other.plot(x.date, x.rv_ratio, color="#b48125", linewidth=1.1, label="RV20 / 前252日中位值")
    other.set_ylabel("波动率比")
    axes[2].legend(loc="upper left", fontsize=8)
    other.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#888888", linestyle=":", linewidth=.8)
        ax.grid(alpha=.15)
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"原案例{episode.episode_id}：下降结构被打破的量价与确认时钟\n三角为前收盘已确认旧高首次突破；圆点为原双抬升，竖线仅回顾对齐", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_下降结构首次突破与量价.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本固定突破解释已开始，不重跑。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "解释来源变化："+s["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, parents = load()
    states, pivots = rules.break_states(data)
    old = pd.read_parquet(OLD_STRUCTURE / "results/全部原点_已确认高低结构.parquet")
    mapped_old = states.rename(columns={"old_rising_structure_onset": "structure_onset", "old_rising_structure_rule_exit": "rule_exit"})
    pd.testing.assert_frame_equal(mapped_old[old.columns], old, check_exact=True)
    table("全部原点_上一确认下降结构与首次突破", states)
    table("全部枢轴确认及原尾点替换", pivots)
    prefix_checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        prefix, arrivals = rules.break_states(data.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix, states.iloc[:n].reset_index(drop=True), check_exact=True)
        pd.testing.assert_frame_equal(arrivals, pivots.loc[pivots.confirmation_index.lt(n)].reset_index(drop=True), check_exact=True)
        prefix_checks.append({"through": date, "rows": n, "status": "EXACT_PAST_STATES_AND_CONSUMED_ANCHORS"})
    context = []
    for period in PERIODS:
        a = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet")
        a = a[["date", "shares", "exposure"]].rename(columns={"shares": "A_shares", "exposure": "A_exposure"})
        target = parents[period][["origin", PARENT_A]].rename(columns={"origin": "date", PARENT_A: "A_known_target"})
        context.append(a.merge(target, on="date", how="left", validate="one_to_one"))
    context = pd.concat(context, ignore_index=True).set_index("date")
    fields = ["date", "close", "cash_shift", "ema20", "daily_dif", "daily_hist", "weekly_hist", "weekly_last_date",
              "relative_volume", "up_volume_balance5", "rv_ratio", "breakout20"]
    known = states.loc[states.date.ge("2015-01-01")].merge(data[fields], on="date", validate="one_to_one").merge(context.reset_index(), on="date", how="left", validate="one_to_one")
    table("2015起全部已知突破与量价A覆盖", known)
    original_labels = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(original_labels[["date", "idx", "status", "net_reference_return"]].rename(columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(original_labels), "原每日标签没有完整保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int), labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "旧标签原点错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签与尾部未知_只解释", labels)
    events = labels.loc[labels.break_event].copy()
    table("全部首次下降结构突破_当时量价及旧标签", events)
    table("全部同锚后续越过_不产生新资格", labels.loc[labels.break_candidate & ~labels.break_event])
    diagnostics = []
    for era, (start, end) in ERAS.items():
        x = events.loc[events.date.between(start, end)]
        mature = x.loc[x.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "known_events": len(x), "old_labels_unknown": len(x)-len(mature), **descriptive_returns(mature.net_reference_return)})
    table("首次突破_全时期旧标签描述", pd.DataFrame(diagnostics))
    paths = close_paths(states, events)
    table("全部首次突破至只上移失效_仅收盘路径", paths)
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, snapshots, cases = [], [], []
    for ep in episodes.itertuples(index=False):
        x = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        row = ep._asdict()
        row.update(first_break_events=int(x.break_event.sum()), old_rising_births=int(x.old_rising_structure_onset.sum()),
                   A_inventory_sessions=int(x.A_shares.gt(0).sum()), retrospective_alignment_only=True)
        mapped.append(row)
    table("原61分段及49正式波段_完整首次突破覆盖", pd.DataFrame(mapped))
    for i in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(i)].iloc[0]
        x = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10), min(len(data)-1, int(ep.peak_idx)+10))].copy()
        x["original_episode_id"], x["retrospective_alignment_only"] = i, True
        snapshots.append(x)
        during = x.loc[x.origin_index.between(ep.bottom_idx, ep.peak_idx) & x.break_event]
        cases.append({"episode_id": i, "original_gross_rise": float(ep.gross_rise), "rise_sessions": int(ep.rise_sessions),
                      "events_between_bottom_and_peak": during.to_dict("records"),
                      "events_in_fixed_case_window": x.loc[x.break_event].to_dict("records"),
                      "original_wave_only_retrospective_alignment": True})
    table("四案例逐日完整量价与上一确认下降结构", pd.concat(snapshots, ignore_index=True))
    charts = [plot(data, states, context, episodes.loc[episodes.episode_id.eq(i)].iloc[0]) for i in CASE_IDS]
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R171", "status": "COMPLETED_ALL_DOWNTREND_BREAK_PATH_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(data), "origins_since_2015": len(known), "confirmed_arrivals": len(pivots), "first_break_events": len(events),
        "same_anchor_recrosses_not_new_events": int((known.break_candidate & ~known.break_event).sum()),
        "completed_close_paths": int(paths.failure_index.ge(0).sum()), "open_close_paths": int(paths.failure_index.lt(0).sum()),
        "original_episodes": len(episodes), "original_admitted_waves": int(episodes.admitted.sum()),
        "original_old_labels": len(original_labels), "original_mature_labels": int(original_labels.status.eq("MATURE").sum()),
        "original_censored_labels": int(original_labels.status.ne("MATURE").sum()), "tail_origins_without_new_label": int(labels.idx.isna().sum()),
        "case_records": cases, "old_label_diagnostics": diagnostics, "charts": charts, "actual_prefix_checks": prefix_checks,
        "necessary_tests": 4, "original_confirmed_structure_rows_unchanged": len(old), "frozen_sources": len(protocol["sources"]),
        "future_complete_policy_intent_fixed": protocol["future_complete_policy_intent_fixed_before_any_new_outcome_join"],
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"TECH.R171：{len(known)}原点、{len(events)}首次突破、61/49原分段及四例全体解释完成，无金融结论。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="首次下降结构突破：先冻结完整用途，再一次解释全体具体路径。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
