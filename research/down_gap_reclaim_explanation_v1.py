"""先固定向下整日缺口收复完整用途，再解释原上涨、量价及全部替换和失败。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import down_gap_reclaim_inputs_v1 as rules
from research import full_daily_gap_inputs_v1 as coordinate
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, PERIODS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_down_gap_reclaim_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OLD = ROOT / "reports/research/510300_full_daily_gap_explanation_v1"
LIFECYCLE = ROOT / "reports/research/510300_full_range_reversal_lifecycle_attribution_v1"
CARD = LIFECYCLE / "下一不同信息用途卡_向下整日缺口收复_仅提案.md"
CASE_IDS = (18, 37, 42, 55)
STATE_TABLE = "全部原点_最新向下缺口锚及首次严格收复"
KNOWN_TABLE = "全部3488已知收复与量价_前2015A未知保留"
EVENT_TABLE = "全部首次收复_原锚当时量价及旧标签"
CASE_TABLE = "四案例逐日完整量价及缺口收复"
MAP_TABLE = "原61分段及49正式波段_缺口收复两区间覆盖"
INTENT = (
    "HIGH严格低于上一完整日LOW时出生完整向下缺口，原锚下沿为当日HIGH，上沿为昨LOW。"
    "只保留最新未收复锚，新完整向下缺口替换旧锚；出生之后首次已知CLOSE严格越过上沿确认并消耗锚。"
    "空仓收复次真实开一次尝试，开盘整数现金价<=锚下沿或坐标未知取消，不延迟重试。"
    "持有初始固定线为出生锚下沿，不抬线；已知CLOSE<=固定线或出现新完整向下缺口时次合法开退出，"
    "同时发生优先记收盘线失败。未知自身不退出、原风险只减；无加仓、混A、固定获利目标、"
    "时间退出、期末清仓或量/MACD/RV过滤。锚状态从原输入首日连续推进，金融按原两时期独立初始化。")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "收复输入测试已记录，不覆盖。")
    test_file = ROOT / "tests/test_down_gap_reclaim_inputs_v1.py"
    completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", str(test_file)],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = completed.stdout+completed.stderr
    ok = completed.returncode == 0 and "4 passed" in output
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "passed": 4 if ok else 0, "exit_code": completed.returncode, "output": output,
        "new_account_results_read": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(rules.__file__), Path(coordinate.__file__), test_file]],
    }, exclusive=True)
    print(output, end="", flush=True)
    require(ok, "输入测试失败，保留首次证据。")


def freeze():
    require(not (OUT / "protocol.json").exists(), "向下缺口收复解释已固定。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 4 and receipt["exit_code"] == 0, "四项输入测试未通过。")
    for source in receipt["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "输入源码不对应测试。")
    reviews = [
        {"source": "research/gap_recovery_v1.py", "inputs": "research/gap_recovery_inputs_v1.py",
         "result": "reports/research/510300_gap_recovery_v1/result.json",
         "distinction": "旧规则同日OPEN+当日分红<昨C且CLOSE+当日分红>昨C，CLOSE<=当日OPEN退出；本次引用此前完整HIGH<昨LOW的区间及未来原点首次CLOSE越过上沿，不是调其低开阈值/退出。旧历史目标未达标保持。"},
        {"source": "research/simple_volume_reversal_v1.py", "result": "reports/research/510300_simple_volume_reversal_v1/acceptance_outcome.json",
         "distinction": "原V5引用此前二十日收盘低点、收盘位置过滤及均线退出，本次引用实际出生缺口两端及最新锚消耗，无该窗口或过滤。"},
        {"source": "research/daily_supply_test_v1.py", "result": "reports/research/510300_daily_supply_test_v1/result.json",
         "distinction": "原二十日低点收复/回测/缩量、ATR/2R/时限拒绝保持，非本用途。"},
        {"source": "research/sequential_patterns_regime_v1.py",
         "distinction": "原BREAKOUT/RECLAIM/REPAIR引用十日/二十日区间或冲击幅度，本次不改窗口或符号；仅完整代码定义核对，不推断未知结果路径状态。"},
        {"source": "research/full_daily_gap_account_v1.py", "result": "reports/research/510300_full_daily_gap_study_v1/summary.json",
         "distinction": "R177完整向上缺口出生/LOW回补及只抬线失败保持；本次是此前完整向下缺口的后续CLOSE收复，不要求新向上缺口。"},
        {"source": "research/full_range_reversal_account_v1.py", "result": "reports/research/510300_full_range_reversal_study_v1/summary.json",
         "distinction": "R181同日跨昨LOW且CLOSE跨昨HIGH并对称反转退出失败保持；本次此前向下缺口首次收复，不要求该同日反转。"},
    ]
    paths = [Path(__file__), Path(rules.__file__), Path(coordinate.__file__),
             ROOT / "tests/test_down_gap_reclaim_inputs_v1.py", OUT / "tests_receipt.json", CARD,
             LIFECYCLE / "summary.json", LIFECYCLE / "next_information_admission_boundary.json",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
             WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
             OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet", OLD / "summary.json"]
    paths.extend(ROOT / "research" / name for name in (
        "point_fresh_repair_order_study_v1.py", "point_first_passage_study_v1.py", "post_repair_path_inputs_v1.py",
        "adaptive_allocation_v1.py", "weekly_daily_technical_v1.py", "directional_entry_timing_v1.py"))
    for review in reviews:
        paths.append(ROOT / review["source"])
        if "inputs" in review:
            paths.append(ROOT / review["inputs"])
        if "result" in review:
            result = read(ROOT / review["result"])
            review["actual_status"] = result["status"]
            if "historical_point_target_met" in result:
                review["historical_point_target_met"] = result["historical_point_target_met"]
            paths.append(ROOT / review["result"])
        else:
            review["actual_status"] = "CODE_DEFINITION_CHECK_ONLY_NO_RESULT_STATUS_INFERRED"
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_DOWN_GAP_RECLAIM_EXPLANATION_V1", "technical_decision": "TECH.R183",
        "definition": "此前HIGH<昨LOW的最新完整向下缺口，出生以后首次已知CLOSE严格越过原上沿；新向下缺口替换未收复锚，收复消耗，无等待窗口或再穿越重试。",
        "numeric_clock": "复用原.001整数报价及当时已发生分红现金平移；非刻度/不一致/未知报价NO_VIEW，未知分红坐标不得回填；收盘形态无日内极值先后或主动订单流。",
        "future_complete_policy_intent_fixed_before_any_new_outcome_join": INTENT,
        "purpose_distinction": "此前完整向下区间及首次收盘否定，形成连续最新锚状态；不是旧同日低开收复、低点窗口变体或R177/R181的参数营救。",
        "old_review": reviews, "dedup_limit": "有限完整代码/用途及实际结果核对，不声称全项目或全球穷尽新颖。更宽搜索找到旧gap_recovery，已实核并保留其失败。",
        "population": "全部3488原点、所有向下锚的收复/替换/开放、原61分段/49上涨及四固定例，量/MACD/上一完整周/RV原值观察。",
        "outcomes": "原2846标签/2826成熟/20删失/9尾部只事后解释，不创建训练标签、不据结果选锚或规则。",
        "necessary_tests": 4, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("R183最新向下缺口收复完整用途已在任何新结果连接前固定。", flush=True)


def failure_paths(states, events):
    rows = []
    for event in events.itertuples(index=False):
        floor, failure, status = int(event.reclaim_floor_ticks), None, "RIGHT_CENSORED"
        for day in states.iloc[int(event.origin_index)+1:].itertuples(index=False):
            if not day.current_quote_known:
                continue
            if day.known_cash_close_ticks <= floor or day.new_down_gap:
                failure = day
                status = "KNOWN_CLOSE_FIXED_FLOOR_FAILED" if day.known_cash_close_ticks <= floor else "NEW_FULL_DOWN_GAP"
                break
        rows.append({"reclaim_date": event.date, "reclaim_index": event.origin_index,
                     "anchor_id": str(event.reclaim_anchor_id), "fixed_floor_ticks": floor,
                     "failure_date": failure.date if failure is not None else pd.NaT,
                     "failure_index": int(failure.origin_index) if failure is not None else -1,
                     "status": status, "not_an_account_or_executable_return": True})
    return pd.DataFrame(rows)


def plot(data, states, paths, episode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lo, hi = max(0, int(episode.bottom_idx)-10), min(len(data), int(episode.peak_idx)+11)
    x, s = data.iloc[lo:hi], states.iloc[lo:hi]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    axes[0].plot(x.date, x.close, color="#253f57", label="原价收盘")
    axes[0].fill_between(x.date, x.low, x.high, color="#8ca2b0", alpha=.18, label="完整日线区间")
    born, reclaimed = s.new_down_gap.to_numpy(bool), s.reclaim_event.to_numpy(bool)
    axes[0].scatter(x.date.loc[born], x.close.loc[born], marker="v", color="#318878", s=45, label="完整向下缺口出生")
    axes[0].scatter(x.date.loc[reclaimed], x.close.loc[reclaimed], marker="^", color="#a83550", s=60, label="收盘后首次严格收复")
    for i in np.flatnonzero(reclaimed):
        axes[0].annotate(s.date.iloc[i].strftime("%m-%d"), (s.date.iloc[i], x.close.iloc[i]),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    for anchor in paths.itertuples(index=False):
        end = int(anchor.terminal_index) if anchor.terminal_index >= 0 else len(data)-1
        left, right = max(lo, int(anchor.birth_index)), min(hi-1, end)
        if left <= right:
            xx = data.iloc[left:right+1]
            axes[0].fill_between(xx.date, anchor.gap_lower_ticks*.001-xx.cash_shift,
                                 anchor.gap_upper_ticks*.001-xx.cash_shift, color="#b48125", alpha=.14)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[0].set_ylabel("原价/元；淡黄为当时缺口")
    axes[1].plot(x.date, x.daily_dif, color="#b48125", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#253f57", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#a83550", "#318878"),
                width=1.2, alpha=.35, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.6)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[2].bar(x.date, x.relative_volume, color="#8ca2b0", width=1.2, alpha=.65, label="量/前20日中位量")
    axes[2].axhline(1, color="#777777", linewidth=.6, linestyle="--")
    other = axes[2].twinx()
    other.plot(x.date, x.rv_ratio, color="#b48125", label="RV20/前252日中位值")
    axes[2].legend(loc="upper left", fontsize=8)
    other.legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("相对量")
    other.set_ylabel("波动率比")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#777777", linestyle=":", linewidth=.8)
        ax.grid(alpha=.15)
    fig.suptitle(f"原案例{episode.episode_id}：此前完整向下缺口、首次收复与量价\n竖线及底峰只事后对齐；三角收盘后才可知", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_向下缺口收复与量价.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "解释已经开始，不覆盖。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "固定解释来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, _ = load()
    states = rules.reclaim_states(data)
    table(STATE_TABLE, states)
    checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        pd.testing.assert_frame_equal(rules.reclaim_states(data.iloc[:n].reset_index(drop=True)),
                                      states.iloc[:n].reset_index(drop=True), check_exact=True)
        checks.append({"through": date, "rows": n, "status": "EXACT_LATEST_ANCHORS_CONSUMPTION_AND_CASH_TICKS"})
    previous = pd.read_parquet(OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet")
    fields = ["date", "open", "high", "low", "close", "volume", "cash_shift", "ema20", "daily_dif", "daily_hist",
              "weekly_hist", "weekly_last_date", "relative_volume", "up_volume_balance5", "rv_ratio"]
    known_all = states.merge(data[fields], on="date", validate="one_to_one").merge(
        previous[["date", "A_shares", "A_exposure", "A_known_target"]], on="date", how="left", validate="one_to_one")
    table(KNOWN_TABLE, known_all)
    known = known_all.loc[known_all.date.ge("2015-01-01")].copy()
    original = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(original[["date", "idx", "status", "net_reference_return"]].rename(
        columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(original), "原2846标签没有完整保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int),
                           labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "原标签错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签与尾部未知_仅解释", labels)
    events = labels.loc[labels.reclaim_event].copy()
    paths = rules.anchor_paths(states)
    table("全部向下缺口原锚_收复替换及开放不筛选", paths)
    table(EVENT_TABLE, events)
    table("全部完整向下缺口_原出生量价", known_all.loc[known_all.new_down_gap])
    failures = failure_paths(states, events)
    table("全部首次收复至固定线失败或新缺口_仅路径", failures)
    diagnostics = []
    for era, left, right in (("ALL", "2015-01-01", "2026-09-30"), ("2015_2019", "2015-01-01", "2019-12-31"),
                             ("2020_2023", "2020-01-01", "2023-12-31"), ("2024_2026", "2024-01-01", "2026-09-30")):
        local = events.loc[events.date.between(left, right)]
        mature = local.loc[local.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "known_events": len(local), "old_labels_unknown": len(local)-len(mature),
                            **descriptive_returns(mature.net_reference_return)})
    table("全时期旧标签描述_不是实际交易", pd.DataFrame(diagnostics))
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, snapshots, case_records = [], [], []
    for ep in episodes.itertuples(index=False):
        first = known.loc[known.origin_index.between(ep.bottom_idx, ep.peak_idx)]
        confirmed = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        mapped.append({**ep._asdict(), "reclaims_bottom_to_peak": int(first.reclaim_event.sum()),
                       "reclaims_confirm_to_peak": int(confirmed.reclaim_event.sum()),
                       "new_down_gaps_bottom_to_peak": int(first.new_down_gap.sum()), "retrospective_alignment_only": True})
    table(MAP_TABLE, pd.DataFrame(mapped))
    lines = ["# 具体上涨的量价解释与向下缺口收复反推", "",
             "先观察原具体上涨及失败，再检验此前完整向下区间是否被已知收盘否定。TECH.R183完整用途已经在新事件结果连接前固定；本解释没有新金融账户。", "",
             "日MACD和上一完整周指标只描述已发生动量，量比和RV只描述交易量及波动；日线不能识别主动买卖流或证明上涨因果。缺口收复也只是历史价格区间被否定，不保证新趋势。", ""]
    for identifier in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]
        local = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10), min(len(data)-1, int(ep.peak_idx)+10))].copy()
        local["original_episode_id"], local["retrospective_alignment_only"] = identifier, True
        snapshots.append(local)
        selected = local.loc[local.reclaim_event]
        case_records.append({"episode_id": identifier, "original_bottom_date": ep.bottom_date,
                             "original_peak_date": ep.peak_date, "original_gross_rise": float(ep.gross_rise),
                             "all_reclaims_in_fixed_window": selected.to_dict("records"), "not_executable_bottom_peak": True})
        lines.extend([f"## 原案例{identifier}：{ep.bottom_date:%Y-%m-%d}至{ep.peak_date:%Y-%m-%d}", "",
                      f"原分段上涨{ep.gross_rise:.2%}，底峰仅事后定位；原5%确认日{ep.confirm_up_date:%Y-%m-%d}。固定前后各10交易日观察窗的全部收复如下，不按获利筛选。", "",
                      "| 收复确认 | 原缺口出生 | 收盘 | 固定线原价 | 原上沿原价 | 相对量 | 日DIF | 日柱 | 上一完整周柱 | RV比 | 原A份额 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for event in selected.itertuples(index=False):
            birth = known_all.iloc[int(event.reclaim_reference_index)]
            lines.append(f"| {event.date:%Y-%m-%d} | {birth.date:%Y-%m-%d} | {event.close:.3f} | "
                         f"{event.reclaim_floor_ticks*.001-event.cash_shift:.3f} | {event.reclaim_upper_ticks*.001-event.cash_shift:.3f} | "
                         f"{event.relative_volume:.3f} | {event.daily_dif:+.6f} | {event.daily_hist:+.6f} | {event.weekly_hist:+.6f} | "
                         f"{event.rv_ratio:.3f} | {event.A_shares:.0f} |")
            phase = ("原底峰区间之前" if event.origin_index < ep.bottom_idx else "原上涨分段内"
                     if event.origin_index <= ep.peak_idx else "原分段之后")
            momentum = "日柱仍负" if event.daily_hist < 0 else "日柱已非负"
            weekly = "上一完整周柱仍负" if event.weekly_hist < 0 else "上一完整周柱已非负"
            lines.extend(["", f"{event.date:%Y-%m-%d}处于{phase}，收复的是{birth.date:%Y-%m-%d}已经出生的整日下跌缺口；"
                          f"{momentum}、{weekly}。相对量{event.relative_volume:.3f}与RV比{event.rv_ratio:.3f}是当时原值，"
                          "不据其符号或大小新增过滤。真正可尝试时间是下一真实开盘，尚不能把后来的底峰涨幅记作本信号回报。", ""])
        if not len(selected):
            lines.extend(["", "固定窗口没有首次收复，该机制无法在这段内给出新进场；不能改用更旧锚或放宽严格端点补信号。", ""])
        name = plot(data, states, paths, ep)
        lines.extend([f"![原案例{identifier}](./{name})", ""])
    table(CASE_TABLE, pd.concat(snapshots, ignore_index=True))
    boundary = []
    for period, (start, end) in PERIODS.items():
        first = int(np.flatnonzero(data.date.ge(start))[0])
        event = known_all.iloc[first-1]
        boundary.append({"period": period, "original_pre_first_session_origin": event.date,
                         "reclaim_event": bool(event.reclaim_event), "anchor_active_after_origin": bool(event.anchor_active_after_origin),
                         "old_label_or_return_created": False})
    scope_paths = paths.loc[paths.birth_date.ge("2015-01-01")]
    lines.extend(["## 全体与可检验点位", "", f"全部{len(data)}日连续最新锚状态，2015起{len(scope_paths)}个原缺口锚、{len(events)}次首次收复。"
                  f"全历史锚终态{paths.status.value_counts().to_dict()}；所有替换和开放保留。原61分段/49正式上涨及原2846标签/2826成熟/20删失/9尾部保留。", "",
                  INTENT, "", "上述收复、失效路径及旧标签描述不是实际胜率、B或全账户收益。随后只按这一完整用途、原两时期两费用账户一次测量，并与原A/纯价格同日历比较。", "",
                  "旧同日低开收复规则的实际失败保持；本次不改其阈值或日内转弱退出。有限旧完整用途核对不能证明全球新颖。所有已用历史仍DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，不能称去除过拟合。", ""])
    with (OUT / "具体上涨与点位反推.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R183", "status": "COMPLETED_ALL_DOWN_GAP_RECLAIM_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(data), "origins_since_2015": len(known), "all_down_gap_anchors": len(paths),
        "down_gap_anchors_since_2015": len(scope_paths), "all_anchor_terminal_counts": paths.status.value_counts().to_dict(),
        "scope_anchor_terminal_counts": scope_paths.status.value_counts().to_dict(), "reclaims_since_2015": len(events),
        "all_reclaims": int(states.reclaim_event.sum()), "completed_reclaim_paths": int(failures.failure_index.ge(0).sum()),
        "open_reclaim_paths": int(failures.failure_index.lt(0).sum()), "original_episodes": len(episodes),
        "original_admitted_waves": int(episodes.admitted.sum()), "original_old_labels": len(original),
        "original_mature_labels": int(original.status.eq("MATURE").sum()), "original_censored_labels": int(original.status.ne("MATURE").sum()),
        "tail_origins_without_new_label": int(labels.idx.isna().sum()), "case_records": case_records,
        "old_label_diagnostics": diagnostics, "original_pre_session_boundary_origins": boundary,
        "charts": [f"案例{i}_向下缺口收复与量价.png" for i in CASE_IDS], "actual_prefix_checks": checks,
        "necessary_tests": 4, "frozen_sources": len(protocol["sources"]), "future_complete_policy_intent_fixed": INTENT,
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"R183：全部{len(paths)}原锚、{len(events)}当期首次收复及61/49段、四例量价已完成，无新金融账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="先固定向下整日缺口收复，再展开具体上涨与全体失败。")
    parser.add_argument("command", choices=("tests", "freeze", "run"))
    command = parser.parse_args().command
    {"tests": tests, "freeze": freeze, "run": run}[command]()
