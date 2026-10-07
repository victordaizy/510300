"""先解释具体上涨及失败，再核对因果高低点结构；只用原日线。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import confirmed_structure_inputs_v1 as rules
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_confirmed_structure_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
PROPOSAL = ROOT / "reports/research/510300_joint_onset_budget_attribution_v1/next_confirmed_structure_explanation_proposal.json"
CASE_IDS = (18, 37, 42, 55)
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "结构解释已登记，不重写。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["exit_code"] == 0 and tests["passed"] == 4, "四项确认时钟测试未通过。")
    for s in tests["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "输入不再对应已通过测试。")
    proposal = read(PROPOSAL)
    old_review = [
        {"source": "research/point_a04_exit_prediction_v1.py",
         "definition": "严格日收盘2/2最近合法低高点回调比例，作为原退出第九预测字段；不是双高双低抬升完整政策。",
         "actual_status": read(ROOT / "reports/research/510300_point_a04_exit_prediction_v1/prediction_summary.json")["status"]},
        {"source": "research/graph_regime_martin_turtle_v2.py",
         "definition": "OHLC日枢轴3/3、同类更极端替换、1ATR显著幅度与DIF配对背离；低价新低而DIF抬高底背离、价格/DIF确认取最大，15日有效；突破20日高进入海龟，10日低/顶背离等退出；震荡底背离触发分层与均值退出。",
         "comparison": "交替枢轴组件已使用；严格收盘四点同时正向抬升及结构失效完整用途不同。旧2万元/242日/现金1.5%口径不得直接与本账户排名。",
         "actual_status": "NOT_REASSERTED_FROM_CONFIG_OR_CODE_ALONE"},
        {"source": "research/weekly_daily_entry_locations_v1.py",
         "definition": "上个完整周严格2/2支撑，日线触碰后5日内收复，最近周阻力作目标、测试低点减0.5ATR止损，成本后计划2R限入和20日到期；不是日收盘四点结构失效。",
         "actual_status": read(ROOT / "reports/research/510300_weekly_daily_entry_locations_v1/result.json")["status"]},
    ]
    sources = [Path(__file__), Path(rules.__file__), ROOT / "tests/test_confirmed_structure_inputs_v1.py",
               OUT / "tests_receipt.json", OUT / "initial_test_failure.json", OUT / "initial_inputs_before_date_dtype_fix.py",
               PROPOSAL, CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
               WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
               ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
               ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
               ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/post_repair_path_inputs_v1.py",
               ROOT / "research/point_first_passage_study_v1.py", ROOT / "research/adaptive_allocation_v1.py",
               ROOT / "research/weekly_daily_technical_v1.py", ROOT / "research/directional_entry_timing_v1.py",
               ROOT / "research/sequential_patterns_regime_v1.py", ROOT / "research/daily_supply_test_v1.py",
               ROOT / "research/point_a04_exit_prediction_v1.py", ROOT / "research/graph_regime_martin_turtle_v2.py",
               ROOT / "config/graph_regime_martin_turtle_v2.yaml", ROOT / "research/weekly_daily_entry_locations_v1.py",
               ROOT / "reports/research/510300_point_a04_exit_prediction_v1/prediction_summary.json",
               ROOT / "reports/research/510300_weekly_daily_entry_locations_v1/result.json"]
    sources.extend(CONTROL / p / "STRESS/A_SAVED_WEIGHT/daily.parquet" for p in PERIODS)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_CONFIRMED_STRUCTURE_EXPLANATION_V1", "technical_decision": "TECH.R168",
        "scope": "先四个具体上涨/失败路径、原61段/49正式波段，再全部出生和失效；不以赢家选规则。",
        "old_review": old_review, "dedup_limit": "有限完整用途核对，不声称穷尽全项目去重。",
        "definition": proposal["source_clock"] + proposal["online_sequence"] + proposal["proposed_structure"],
        "future_complete_policy_intent_fixed_before_old_label_join": proposal["future_complete_policy_intent_before_any_new_outcome_join"],
        "unknown": "不足4点为已知未成立，报价/现金分红未知为NO_VIEW；恢复时不伪造出生，未知本身不强制退出。",
        "observations": "MACD12/26/9日值、上个完整周柱、相对量、量方向及RV比保持原字段，仅解释，不据结果加过滤。",
        "outcome": "原2846每日标签和61分段原样保留，当前尾部无标签单列；出生至失效的收盘路径是描述，不是实际交易收益。",
        "necessary_tests": 4, "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, exclusive=True)
    print("TECH.R168：日线四点结构、旧完整用途和唯一未来政策在结果连接前固定。", flush=True)


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
    axes[0].plot(x.date, x.close, color="#233c55", label="原价收盘", linewidth=1.7)
    axes[0].plot(x.date, x.ema20-x.cash_shift, color="#b58222", label="EMA20（同原价）", linewidth=1)
    for name, color in [("H2", "#a85665"), ("L2", "#37897d")]:
        axes[0].step(x.date, s[f"{name}_price"].to_numpy()-x.cash_shift.to_numpy(), where="post", color=color,
                     alpha=.75, linewidth=1.1, label=f"当时已确认{name}（确认日才显示）")
    active = s.structure_active.fillna(False).to_numpy(bool)
    axes[0].fill_between(x.date, 0, 1, where=active, transform=axes[0].get_xaxis_transform(), color="#37897d", alpha=.09, label="已知双高双低抬升成立")
    held = context.reindex(x.date).A_shares.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(x.date, 0, 1, where=held, transform=axes[0].get_xaxis_transform(), color="#777777", alpha=.16, label="原A实际持仓")
    birth = s.structure_onset.to_numpy(bool)
    failed = s.rule_exit.to_numpy(bool) & s.structure_active.shift(1).fillna(False).to_numpy(bool)
    axes[0].scatter(x.date.loc[birth], x.close.loc[birth], marker="^", color="#ac4256", s=45, zorder=5, label="当时结构首次成立")
    axes[0].scatter(x.date.loc[failed], x.close.loc[failed], marker="x", color="#202020", s=40, zorder=5, label="已知结构失效")
    for date, price in zip(x.date.loc[birth], x.close.loc[birth]):
        axes[0].annotate(date.strftime("%m-%d"), (date, price), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("价格 / 元")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].plot(x.date, x.daily_dif, color="#b58222", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#233c55", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#ac4256", "#37897d"), width=1.2, alpha=.35, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.6)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[1].set_ylabel("动量原值")
    axes[2].bar(x.date, x.relative_volume, color="#8898a8", width=1.2, alpha=.7, label="量 / 前20日中位量")
    axes[2].axhline(1, color="#777777", linestyle="--", linewidth=.6)
    axes[2].set_ylabel("相对量")
    other = axes[2].twinx()
    other.plot(x.date, x.rv_ratio, color="#b58222", label="RV20 / 前252日中位值", linewidth=1.1)
    other.set_ylabel("波动率比")
    axes[2].legend(loc="upper left", fontsize=8)
    other.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#888888", linestyle=":", linewidth=.8)
        ax.grid(alpha=.15)
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"原案例{episode.episode_id}：上涨及失败的量价与已确认结构\n台阶从确认日显示，三角为当时出生；原5%确认竖线仅回顾对齐", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_量价指标与因果高低结构.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本结构解释已开始，不重跑。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "解释来源改变："+s["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, parents = load()
    states, pivots = rules.structure_states(data)
    table("全部原点_已确认高低结构", states)
    table("全部枢轴到达及当前尾点替换", pivots)
    prefix_checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        prefix, arrivals = rules.structure_states(data.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix, states.iloc[:n].reset_index(drop=True), check_exact=True)
        pd.testing.assert_frame_equal(arrivals, pivots.loc[pivots.confirmation_index.lt(n)].reset_index(drop=True), check_exact=True)
        prefix_checks.append({"through": date, "rows": n, "status": "EXACT_PAST_STATES_AND_PIVOT_ARRIVALS"})
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
    table("2015起全部已知结构与量价A覆盖", known)
    old = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(old[["date", "idx", "status", "net_reference_return"]].rename(columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(old), "原每日标签没有完整保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int), labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "旧标签与当时结构原点错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签与尾部未知_仅解释", labels)
    events = labels.loc[labels.structure_onset].copy()
    table("全部结构出生_当时量价与旧标签", events)
    diagnostics = []
    for era, (start, end) in ERAS.items():
        x = events.loc[events.date.between(start, end)]
        mature = x.loc[x.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "known_births": len(x), "old_labels_unknown": len(x)-len(mature), **descriptive_returns(mature.net_reference_return)})
    table("全部出生_全时期旧标签描述", pd.DataFrame(diagnostics))
    paths = []
    for event in events.itertuples(index=False):
        later = states.loc[states.origin_index.gt(event.origin_index) & states.rule_exit]
        end = later.iloc[0] if len(later) else None
        paths.append({"birth_date": event.date, "birth_index": event.origin_index, "birth_close_coordinate": event.known_cash_adjusted_close,
                      "failure_date": end.date if end is not None else pd.NaT,
                      "failure_index": int(end.origin_index) if end is not None else -1,
                      "failure_status": end.structure_status if end is not None else "RIGHT_CENSORED",
                      "close_path_return_only": float(end.known_cash_adjusted_close/event.known_cash_adjusted_close-1) if end is not None else np.nan,
                      "not_an_account_or_executable_return": True})
    paths = pd.DataFrame(paths)
    table("全部结构出生至已知失效_仅收盘路径", paths)
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, snapshots, cases = [], [], []
    for ep in episodes.itertuples(index=False):
        x = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        row = ep._asdict()
        row.update(after_confirmation_sessions=len(x), structure_active_sessions=int(x.structure_active.fillna(False).sum()),
                   structure_births=int(x.structure_onset.sum()), A_inventory_sessions=int(x.A_shares.gt(0).sum()), retrospective_alignment_only=True)
        mapped.append(row)
    table("原61分段及49正式波段_完整结构覆盖", pd.DataFrame(mapped))
    for i in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(i)].iloc[0]
        x = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10), min(len(data)-1, int(ep.peak_idx)+10))].copy()
        x["original_episode_id"], x["retrospective_alignment_only"] = i, True
        snapshots.append(x)
        during = x.loc[x.origin_index.between(ep.bottom_idx, ep.peak_idx) & x.structure_onset]
        prior = x.loc[x.origin_index.lt(ep.bottom_idx) & x.structure_onset]
        checkpoints = []
        for name, idx in [("事后低点_当时不作为入场低点", int(ep.bottom_idx)), ("原5%确认_仅回顾", int(ep.confirm_up_idx)),
                          ("事后高点_当时不作为退出高点", int(ep.peak_idx))]:
            r = known.loc[known.origin_index.eq(idx)].iloc[0].to_dict()
            r["checkpoint"] = name
            checkpoints.append(r)
        cases.append({"episode_id": i, "original_gross_rise": float(ep.gross_rise), "rise_sessions": int(ep.rise_sessions),
                      "births_between_bottom_and_peak": during.to_dict("records"), "prior_births_in_fixed_window": prior.to_dict("records"),
                      "checkpoints": checkpoints, "retrospective_alignment_only": True})
    table("四案例逐日完整量价与确认结构", pd.concat(snapshots, ignore_index=True))
    charts = [plot(data, states, context, episodes.loc[episodes.episode_id.eq(i)].iloc[0]) for i in CASE_IDS]
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R168", "status": "COMPLETED_ALL_CONFIRMED_STRUCTURE_PATH_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(data), "origins_since_2015": len(known), "confirmed_arrivals": len(pivots), "known_structure_births": len(events),
        "completed_close_paths": int(paths.failure_index.ge(0).sum()), "open_close_paths": int(paths.failure_index.lt(0).sum()),
        "close_path_failure_status_counts": paths.failure_status.value_counts().to_dict(),
        "original_episodes": len(episodes), "original_admitted_waves": int(episodes.admitted.sum()),
        "original_old_labels": len(old), "original_mature_labels": int(old.status.eq("MATURE").sum()),
        "original_censored_labels": int(old.status.ne("MATURE").sum()), "tail_origins_no_new_label": int(labels.idx.isna().sum()),
        "case_records": cases, "old_label_diagnostics": diagnostics, "charts": charts, "actual_prefix_checks": prefix_checks,
        "necessary_tests": 4, "initial_representation_failure_preserved": "initial_test_failure.json",
        "frozen_sources": len(protocol["sources"]), "future_complete_policy_intent_fixed": protocol["future_complete_policy_intent_fixed_before_old_label_join"],
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"TECH.R168：{len(known)}原点、{len(events)}出生、61/49原分段和四例量价结构解释已完成；没有金融收益结论。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="严格确认的四点结构：先冻结解释，再一次完成全体路径。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
