"""先固定整日反转完整用途，再解释全体上涨、失败和当时量价。"""
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

from research import full_range_reversal_inputs_v1 as rules
from research import full_daily_gap_inputs_v1 as coordinate
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT, CONTROL, PERIODS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_full_range_reversal_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OLD = ROOT / "reports/research/510300_full_daily_gap_explanation_v1"
CASE_IDS = (18, 37, 42, 55)
STATE_TABLE = "全部原点_整日反转及整数现金坐标"
KNOWN_TABLE = "全部3488已知反转与量价_前2015A未知保留"
EVENT_TABLE = "全部向上整日反转_当时量价及旧标签"
CASE_TABLE = "四案例逐日完整量价及整日反转"
MAP_TABLE = "原61分段及49正式波段_整日反转覆盖"
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}
INTENT = (
    "空仓当日LOW严格低于上一日LOW且CLOSE严格高于上一日HIGH，收盘后确认，次真实开一次尝试。"
    "开盘整数现金平移价<=事件当日LOW或坐标未知取消，不延迟追入。初始失效线是该事件当日LOW。"
    "持有中新的同向整日反转只将线抬至max(旧线,新事件当日LOW)；已知CLOSE<=线或"
    "当日HIGH>上一HIGH且CLOSE<上一LOW的反向整日反转，次合法开退出；两者同时发生时记收盘线失败。"
    "未知自身不退出、原风险只减；无加仓、混A、2R、20日、期末清仓或量/MACD/RV过滤。")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "整日反转输入测试已经记录，不覆盖。")
    test_file = ROOT / "tests/test_full_range_reversal_inputs_v1.py"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", str(test_file)],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = result.stdout+result.stderr
    ok = result.returncode == 0 and "4 passed" in output
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "passed": 4 if ok else 0, "exit_code": result.returncode, "output": output,
        "new_account_results_read": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(rules.__file__), Path(coordinate.__file__), test_file]],
    }, exclusive=True)
    print(output, end="", flush=True)
    require(ok, "输入测试失败，保留首次证据。")


def freeze():
    require(not (OUT / "protocol.json").exists(), "整日反转解释已固定。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 4 and receipt["exit_code"] == 0, "四项必要输入测试未通过。")
    for source in receipt["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "输入源码不对应测试。")
    reviewed = [
        {"source": "research/simple_volume_reversal_v1.py", "result": "reports/research/510300_simple_volume_reversal_v1/acceptance_outcome.json",
         "actual_status": read(ROOT / "reports/research/510300_simple_volume_reversal_v1/acceptance_outcome.json")["status"],
         "distinction": "V5跌破原二十日收盘低点后收回该低点、收盘位置过滤、均线退出；不要求收盘跨上一完整日HIGH，也不含对称HIGH越过/CLOSE跨昨LOW的完整退出。不是只把20改1。"},
        {"source": "research/daily_supply_test_v1.py", "result": "reports/research/510300_daily_supply_test_v1/result.json",
         "actual_status": read(ROOT / "reports/research/510300_daily_supply_test_v1/result.json")["status"],
         "distinction": "此前二十日低点收复与回测/缩量测试、ATR失效和2R/20日退出；原固定失败保持，本次不用这些端点或过滤。"},
        {"source": "research/smc_sweep_fvg_historical_v1.py", "result": "reports/research/510300_smc_sweep_fvg_historical_v1/result.json",
         "actual_status": read(ROOT / "reports/research/510300_smc_sweep_fvg_historical_v1/result.json")["status"],
         "distinction": "分钟扫昨低、收复、突破、三柱FVG及回测，原主事件拒绝、组合NOT_RUN；本次只读整日OHLC，不解释分钟主动订单或极值先后。"},
        {"source": "research/sequential_patterns_regime_v1.py", "actual_status": "CODE_DEFINITION_CHECK_ONLY_NO_RESULT_STATUS_INFERRED",
         "distinction": "BREAKOUT事前低点已试探/此前十日上沿，RECLAIM此前二十日低点，REPAIR冲击区间；完整定义核对未见同日跨昨低且收盘跨昨高并对称退出。未因猜测结果路径缺失判其NOT_RUN。"},
        {"source": "research/full_daily_gap_account_v1.py", "result": "reports/research/510300_full_daily_gap_study_v1/summary.json",
         "actual_status": read(ROOT / "reports/research/510300_full_daily_gap_study_v1/summary.json")["status"],
         "distinction": "R177要求相邻整日区间完全分离、LOW触及缺口下沿退出；新事件整日区间跨越/重叠，收盘越过相反边界及对称反转管理，不是加过滤或改R177回补参数。"},
    ]
    paths = [Path(__file__), Path(rules.__file__), Path(coordinate.__file__),
             ROOT / "tests/test_full_range_reversal_inputs_v1.py", OUT / "tests_receipt.json",
             OUT / "initial_input_tests_failure.json", OUT / "initial_input_tests_before_fixture_repair.py",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
             WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
             OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet", OLD / "summary.json",
             ROOT / "reports/research/510300_full_daily_gap_lifecycle_attribution_v1/summary.json"]
    paths.extend(ROOT / "research" / name for name in (
        "point_fresh_repair_order_study_v1.py", "point_first_passage_study_v1.py", "post_repair_path_inputs_v1.py",
        "adaptive_allocation_v1.py", "daily_supply_test_v1.py", "weekly_daily_technical_v1.py",
        "directional_entry_timing_v1.py", "sequential_patterns_regime_v1.py", "point_account_nr7_inputs_v1.py"))
    for review in reviewed:
        paths.append(ROOT / review["source"])
        if "result" in review:
            paths.append(ROOT / review["result"])
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_FULL_RANGE_REVERSAL_EXPLANATION_V1", "technical_decision": "TECH.R179",
        "definition": "整数现金平移LOW<上一完整日LOW且CLOSE>上一完整日HIGH为同向出生；HIGH>上一HIGH且CLOSE<上一LOW为相反事件，只用于观察及多头失效。严格相等不算。",
        "numeric_clock": "原OHLC及当时已发生分红用.001整数tick；非刻度/不一致/未知报价NO_VIEW，未知分红坐标不回填。收盘才知道形态，不知道日内HIGH/LOW先后或主动买卖流。",
        "future_complete_policy_intent_fixed_before_any_new_outcome_join": INTENT,
        "purpose_distinction": "完整上一日区间两端共同跨越及对称反转管理；不单纯缩短旧收复低点窗口，不营救R177缺口、旧分钟FVG或原退出预测。",
        "old_review": reviewed, "dedup_limit": "有限代码及完整用途核对，不声称全项目或所有历史形态全球新颖。",
        "population": "3488全日线、2015起全部同向/相反出生及失败/开放，原61分段/49正式上涨和四固定案例全部保留。量、MACD、上一完整周、RV只是当时观察。",
        "outcomes": "原2846标签/2826成熟/20删失/9尾部保留，只旧结果解释，不建新训练标签、不据标签选规则。路径失效不是实际完成收益。",
        "necessary_tests": 4, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "pre_result_test_fixture_repair": "首次除息相等测试例9.978+.033实际10.011，并非预期10.010；只将测试CLOSE改9.977，原输入源码不变，首次失败及测试源码保留。修正时新事件结果连接、金融账户及金融结果读取均0。",
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("R179唯一整日反转完整用途已在新结果连接前固定。", flush=True)


def failure_paths(states, events):
    rows = []
    for event in events.itertuples(index=False):
        floor, changes, failure, reason = int(event.reversal_floor_ticks), 0, None, "RIGHT_CENSORED"
        for day in states.iloc[int(event.origin_index)+1:].itertuples(index=False):
            if not day.current_quote_known:
                continue
            if day.bullish_range_reversal and day.reversal_floor_ticks > floor:
                floor, changes = int(day.reversal_floor_ticks), changes+1
            if day.known_cash_close_ticks <= floor or day.bearish_range_reversal:
                failure = day
                reason = "KNOWN_CLOSE_FLOOR_FAILED" if day.known_cash_close_ticks <= floor else "OPPOSING_RANGE_REVERSAL"
                break
        rows.append({"birth_date": event.date, "birth_index": event.origin_index,
                     "initial_reversal_floor_ticks": int(event.reversal_floor_ticks), "final_floor_ticks": floor,
                     "floor_updates": changes, "failure_date": failure.date if failure is not None else pd.NaT,
                     "failure_index": int(failure.origin_index) if failure is not None else -1, "status": reason,
                     "opposing_exit_above_floor": bool(failure is not None and reason == "OPPOSING_RANGE_REVERSAL"),
                     "not_an_account_or_executable_return": True})
    return pd.DataFrame(rows, columns=["birth_date", "birth_index", "initial_reversal_floor_ticks", "final_floor_ticks",
                                      "floor_updates", "failure_date", "failure_index", "status",
                                      "opposing_exit_above_floor", "not_an_account_or_executable_return"])


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
    up, down = s.bullish_range_reversal.to_numpy(bool), s.bearish_range_reversal.to_numpy(bool)
    axes[0].scatter(x.date.loc[up], x.close.loc[up], marker="^", color="#a83550", s=55, label="跨昨低且收盘跨昨高")
    axes[0].scatter(x.date.loc[down], x.close.loc[down], marker="v", color="#318878", s=40, label="相反形态：观察/多头失效")
    for i in np.flatnonzero(up):
        axes[0].annotate(s.date.iloc[i].strftime("%m-%d"), (s.date.iloc[i], x.close.iloc[i]),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    held = context.reindex(x.date).A_shares.fillna(0).gt(0).to_numpy()
    axes[0].fill_between(x.date, 0, 1, where=held, transform=axes[0].get_xaxis_transform(),
                         color="#777777", alpha=.13, label="原A实际库存")
    axes[0].set_ylabel("价格/元")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].plot(x.date, x.daily_dif, color="#b48125", label="日DIF")
    axes[1].plot(x.date, x.weekly_hist, color="#253f57", label="上一完整周MACD柱")
    axes[1].bar(x.date, x.daily_hist, color=np.where(x.daily_hist.ge(0), "#a83550", "#318878"),
                width=1.2, alpha=.35, label="日MACD柱")
    axes[1].axhline(0, color="#777777", linewidth=.6)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[1].set_ylabel("动量原值")
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
    fig.suptitle(f"原案例{episode.episode_id}：整日收盘跨区间、量价与相反失效\n三角仅当日收盘后可知；原波段及竖线只事后对齐", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"案例{episode.episode_id}_整日反转与量价.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "解释已开始，不覆盖或重复。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "固定解释来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, _ = load()
    states = rules.reversal_states(data)
    table(STATE_TABLE, states)
    checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        pd.testing.assert_frame_equal(rules.reversal_states(data.iloc[:n].reset_index(drop=True)),
                                      states.iloc[:n].reset_index(drop=True), check_exact=True)
        checks.append({"through": date, "rows": n, "status": "EXACT_PAST_RANGE_REVERSALS_AND_CASH_TICKS"})
    previous = pd.read_parquet(OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet")
    context = previous[["date", "A_shares", "A_exposure", "A_known_target"]].set_index("date")
    fields = ["date", "open", "high", "low", "close", "volume", "cash_shift", "ema20", "daily_dif", "daily_hist",
              "weekly_hist", "weekly_last_date", "relative_volume", "up_volume_balance5", "rv_ratio"]
    known_all = states.merge(data[fields], on="date", validate="one_to_one").merge(
        context.reset_index(), on="date", how="left", validate="one_to_one")
    table(KNOWN_TABLE, known_all)
    known = known_all.loc[known_all.date.ge("2015-01-01")].copy()
    original = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    labels = known.merge(original[["date", "idx", "status", "net_reference_return"]].rename(
        columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(original), "原2846标签未完整保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int),
                           labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "原标签错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签与尾部未知_仅解释", labels)
    events = labels.loc[labels.bullish_range_reversal].copy()
    table(EVENT_TABLE, events)
    table("全部相反整日反转_只观察及多头失效", labels.loc[labels.bearish_range_reversal])
    paths = failure_paths(states, events)
    table("全部同向出生至收盘失效或相反事件_仅路径", paths)
    diagnostics = []
    for era, (start, end) in ERAS.items():
        local = events.loc[events.date.between(start, end)]
        mature = local.loc[local.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "known_events": len(local), "old_labels_unknown": len(local)-len(mature),
                            **descriptive_returns(mature.net_reference_return)})
    table("整日反转_全时期旧标签描述", pd.DataFrame(diagnostics))
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, snapshots, cases = [], [], []
    for ep in episodes.itertuples(index=False):
        local = known.loc[known.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        mapped.append({**ep._asdict(), "bullish_reversal_births": int(local.bullish_range_reversal.sum()),
                       "bearish_reversal_events": int(local.bearish_range_reversal.sum()),
                       "A_inventory_sessions": int(local.A_shares.gt(0).sum()), "retrospective_alignment_only": True})
    table(MAP_TABLE, pd.DataFrame(mapped))
    for identifier in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]
        local = known.loc[known.origin_index.between(max(0, int(ep.bottom_idx)-10),
                                                    min(len(data)-1, int(ep.peak_idx)+10))].copy()
        local["original_episode_id"], local["retrospective_alignment_only"] = identifier, True
        snapshots.append(local)
        cases.append({"episode_id": identifier, "original_gross_rise": float(ep.gross_rise),
                      "rise_sessions": int(ep.rise_sessions),
                      "events_between_bottom_and_peak": local.loc[local.origin_index.between(ep.bottom_idx, ep.peak_idx)
                                                                     & local.bullish_range_reversal].to_dict("records"),
                      "events_in_fixed_case_window": local.loc[local.bullish_range_reversal].to_dict("records"),
                      "original_wave_only_retrospective_alignment": True})
    table(CASE_TABLE, pd.concat(snapshots, ignore_index=True))
    boundary = []
    for period, (start, end) in PERIODS.items():
        first = int(np.flatnonzero(data.date.ge(start))[0])
        event = known_all.iloc[first-1]
        boundary.append({"period": period, "original_pre_first_session_origin": event.date,
                         "bullish_range_reversal": bool(event.bullish_range_reversal),
                         "old_label_or_return_created": False, "A_inventory_is_not_this_initialized_account": True})
    charts = [plot(data, states, context, episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]) for identifier in CASE_IDS]
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R179",
        "status": "COMPLETED_ALL_FULL_RANGE_REVERSAL_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(data), "origins_since_2015": len(known), "bullish_reversals_since_2015": len(events),
        "bearish_reversals_observation_and_long_exit_only": int(known.bearish_range_reversal.sum()),
        "completed_reversal_paths": int(paths.failure_index.ge(0).sum()),
        "open_reversal_paths": int(paths.failure_index.lt(0).sum()),
        "opposing_exit_above_floor": int(paths.opposing_exit_above_floor.sum()),
        "original_episodes": len(episodes), "original_admitted_waves": int(episodes.admitted.sum()),
        "original_old_labels": len(original), "original_mature_labels": int(original.status.eq("MATURE").sum()),
        "original_censored_labels": int(original.status.ne("MATURE").sum()),
        "tail_origins_without_new_label": int(labels.idx.isna().sum()),
        "case_records": cases, "old_label_diagnostics": diagnostics, "original_pre_session_boundary_origins": boundary,
        "charts": charts, "actual_prefix_checks": checks, "necessary_tests": 4, "frozen_sources": len(protocol["sources"]),
        "future_complete_policy_intent_fixed": INTENT, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0,
        "new_market_requests": 0, "historical_sample_role": "DEVELOPMENT_CALIBRATION",
        "source_first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"R179：{len(events)}同向整日反转、全部失效/开放、原61/49段和四图完成；没有新金融账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整日反转：先固定用途，再解释全部具体量价。")
    parser.add_argument("command", choices=("tests", "freeze", "run"))
    command = parser.parse_args().command
    {"tests": tests, "freeze": freeze, "run": run}[command]()
