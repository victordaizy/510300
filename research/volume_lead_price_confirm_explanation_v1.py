"""先解释具体上涨及全部量领先失败，再按唯一用途反推已知点位。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import volume_lead_price_confirm_inputs_v1 as rules
from research.point_fresh_repair_order_study_v1 import load, CURRENT, WEIGHT
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.post_repair_path_inputs_v1 import descriptive_returns

OUT = ROOT / "reports/research/510300_volume_lead_price_confirm_explanation_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OLD = ROOT / "reports/research/510300_full_daily_gap_explanation_v1"
CARD = ROOT / "docs/510300_VOLUME_LEAD_PRICE_CONFIRM_V1.md"
CASE_IDS = (18, 37, 42, 55)
INTENT = "累计量先严格超过已确认高点中心日的累计量、价格仍低于高点且高于已确认低点；固定准备时低点。准备后的不同日价格严格超过同一高点、量仍超过，消耗确认。准备前后新高点替换旧锚；准备后到/低于固定低点或参考量则消耗失败，不重试。空仓次真实开一次、开盘到/低于固定低点或坐标未知取消。持有固定低点、高点和参考累计量，已知收盘到/低于低点，或量到/低于参考且价格到/低于原高点共同失效，次合法开退出；未知本身不出、原风险只减，无加仓/抬线/目标/时间退出/期末清仓/指标过滤或混A。"
STATE_TABLE = "全部3488已知量先恢复与价格后确认"
KNOWN_TABLE = "全部3488量价MACD波动及原A未知保留"
EVENT_TABLE = "全部确认点位及原标签_仅事后解释"


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "量价顺序解释已经冻结，不覆盖。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 6 and receipt["exit_code"] == 0, "六项必要输入测试未通过。")
    for source in receipt["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "输入代码不对应通过的测试。")
    reviews = [
        ("signed_volume_within_v1", "research/signed_volume_within_inputs_v1.py", "二十日归一化量作为九因子周期内继续价值，不是本次累计量先行与同锚后确认；其BASE局部增量及成本敏感失败保持。"),
        ("volume_weighted_trend_v1", "research/volume_weighted_trend_v1.py", "二十/六十日VWMA均线排列，不是累计份额与已确认高点锚的顺序。"),
        ("negative_volume_trend_v1", "research/negative_volume_trend_inputs_v1.py", "仅缩量日累计价格百分比与255平滑线，不是全部日真实份额累计或同锚先后。"),
        ("price_volume_coherence_v1", "research/price_volume_coherence_inputs_v1.py", "二十日价格收益与量变同期相关及连续两日状态，不是累计量先行；旧完整目标失败保持。"),
        ("session_signed_rank_v1", "research/session_signed_rank_inputs_v1.py", "日内隔夜排序及风险目标，不使用已确认价锚的累计成交量先行。"),
        ("confirmed_structure_study_v1", "research/confirmed_structure_inputs_v1.py", "复用严格2/2枢轴组件，旧双高双低抬升政策失败保持；本次不要求旧抬升状态。"),
        ("downtrend_break_study_v1", "research/downtrend_break_inputs_v1.py", "旧下降高点价格突破与动态低点政策失败保持；本次唯一价锚前必须有累计份额先行、固定准备低点及量价共同失效，不是其价格条件调参。"),
        ("down_gap_reclaim_study_v1", "research/down_gap_reclaim_inputs_v1.py", "R185此前完整向下整日缺口首次收复失败保持；本次不引用整日缺口或给旧收复加成交量过滤。"),
    ]
    review_records, paths = [], [Path(__file__), Path(rules.__file__), CARD,
        ROOT / "research/full_daily_gap_inputs_v1.py", ROOT / "research/confirmed_structure_inputs_v1.py",
        ROOT / "tests/test_volume_lead_price_confirm_inputs_v1.py", OUT / "tests_receipt.json",
        CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
        WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
        ATLAS / "results/上涨段全集.parquet", ATLAS / "results/全部原点结果标签.parquet",
        OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet",
        ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/point_first_passage_study_v1.py",
        ROOT / "research/post_repair_path_inputs_v1.py"]
    for stem, code, distinction in reviews:
        folder = ROOT / ("reports/research/510300_"+stem)
        result_path = folder / ("summary.json" if "study" in stem else "result.json")
        result = read(result_path)
        review_records.append({"code": code, "result": result_path.relative_to(ROOT).as_posix(),
                               "actual_status": result["status"], "distinction": distinction})
        paths.extend([ROOT / code, result_path])
        if (folder / "acceptance_outcome.json").exists():
            outcome = read(folder / "acceptance_outcome.json")
            review_records[-1]["actual_acceptance_status"] = outcome["status"]
            paths.append(folder / "acceptance_outcome.json")
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_VOLUME_LEAD_PRICE_CONFIRM_EXPLANATION_V1", "technical_decision": "TECH.R187",
        "complete_purpose_fixed_before_new_state_or_outcome_join": INTENT,
        "pivot_clock": "原严格左右两日现金收盘枢轴、交替尾部规则，今天新到达枢轴先更新；高点中心日后两日收盘才知。",
        "volume_clock": "原.001现金坐标方向×当天真实整数成交份额连续累计，首日零；任何必要资料未知后累计持续未知，不补数/删缺失/重置。",
        "old_complete_uses_review": review_records,
        "finite_review_limit": "有限完整定义及实际结果核对，不声称全球或全仓库穷尽等价式。旧OBV类归档不被改写为已成功。",
        "source_formula": "https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/",
        "source_formula_boundary": "官方只支持涨加量/跌减量/平不变及分析思路，项目分红方向/2日锚/顺序/失效用途为本项目定义，非真实主动订单流或因果。",
        "population": "全部3488日、全部参考高点与准备/确认/失败/替换/开放、原61分段/49上涨、四固定例18/37/42/55。",
        "old_labels_role": "原2846/2826成熟/20删失/9尾部只事后解释，不创造训练标签，不用未来底峰进出。",
        "necessary_tests": 6, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False,
        "goal_achieved": False, "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("R187累计量先恢复、同锚价格后确认完整用途已在新结果连接前冻结。", flush=True)


def failure_paths(states, events):
    paths = []
    for event in events.itertuples(index=False):
        failure = None
        for day in states.iloc[int(event.origin_index)+1:].itertuples(index=False):
            if not day.current_quote_known:
                continue
            low_failed = day.known_cash_close_ticks <= event.fixed_low_ticks
            joint_failed = bool(day.cumulative_volume_known and day.cumulative_signed_volume <= event.reference_cumulative_volume
                                and day.known_cash_close_ticks <= event.reference_high_ticks)
            if low_failed or joint_failed:
                failure = day
                reason = "FIXED_LOW_FAILED" if low_failed else "REFERENCE_VOLUME_AND_PRICE_FAILED"
                break
        paths.append({"confirmation_date": event.date, "confirmation_index": event.origin_index,
                      "anchor_id": event.reference_anchor_id, "fixed_low_ticks": event.fixed_low_ticks,
                      "reference_high_ticks": event.reference_high_ticks,
                      "reference_cumulative_volume": event.reference_cumulative_volume,
                      "first_failure_date": failure.date if failure is not None else pd.NaT,
                      "first_failure_index": int(failure.origin_index) if failure is not None else -1,
                      "status": reason if failure is not None else "RIGHT_CENSORED",
                      "not_an_account_or_executable_return": True})
    return pd.DataFrame(paths)


def plot(known, paths, episode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lo, hi = max(0, int(episode.bottom_idx)-10), min(len(known), int(episode.peak_idx)+11)
    part = known.iloc[lo:hi]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5, 1.5]})
    axes[0].plot(part.date, part.close, color="#243e55", linewidth=1.4, label="原价收盘及完整日区间")
    axes[0].fill_between(part.date, part.low, part.high, color="#9bafb9", alpha=.15)
    leads, confirmed = part.volume_lead_event, part.price_after_volume_event
    axes[0].scatter(part.date[leads], part.close[leads], color="#c18c25", marker="o", s=36, label="收盘后量先恢复")
    axes[0].scatter(part.date[confirmed], part.close[confirmed], color="#a5334b", marker="^", s=62, label="不同日价格后确认")
    offset = int(part.cumulative_signed_volume.dropna().iloc[0]) if part.cumulative_signed_volume.notna().any() else 0
    volume_line = (part.cumulative_signed_volume.astype(float)-offset)/1e8
    axes[1].plot(part.date, volume_line, color="#516c96", label="累计涨跌成交份额（亿份，统一平移）")
    for anchor in paths.itertuples(index=False):
        end = int(anchor.terminal_index) if anchor.terminal_index >= 0 else len(known)-1
        left, right = max(lo, int(anchor.birth_index)), min(hi-1, end)
        if left > right or pd.isna(anchor.reference_cumulative_volume):
            continue
        active = known.iloc[left:right+1]
        axes[1].plot(active.date, np.full(len(active), (int(anchor.reference_cumulative_volume)-offset)/1e8),
                     color="#c18c25", linestyle="--", linewidth=.8)
        axes[0].plot(active.date, anchor.reference_high_ticks*.001-active.cash_shift,
                     color="#c18c25", linestyle="--", linewidth=.8)
        if anchor.lead_index >= 0:
            prepared = known.iloc[max(lo, int(anchor.lead_index)):right+1]
            axes[0].plot(prepared.date, anchor.fixed_low_ticks*.001-prepared.cash_shift,
                         color="#388779", linestyle=":", linewidth=.9)
        if anchor.lead_index >= 0 and anchor.confirmation_index < 0 and lo <= end < hi and anchor.terminal_index >= 0:
            terminal = known.iloc[end]
            axes[0].scatter([terminal.date], [terminal.close], color="#388779", marker="x", s=45)
    for event in part.loc[leads | confirmed].itertuples(index=False):
        axes[0].annotate(event.date.strftime("%m-%d"), (event.date, event.close),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    axes[2].plot(part.date, part.daily_dif, color="#b38527", label="日DIF")
    axes[2].plot(part.date, part.weekly_hist, color="#243e55", label="上一完整周MACD柱")
    axes[2].bar(part.date, part.daily_hist, color=np.where(part.daily_hist.ge(0), "#a5334b", "#388779"),
                width=1.2, alpha=.35, label="日MACD柱")
    axes[2].axhline(0, color="#777", linewidth=.6)
    axes[3].bar(part.date, part.relative_volume, color="#9bafb9", width=1.2, alpha=.7, label="量/前20日中位量")
    axes[3].axhline(1, color="#777", linewidth=.6, linestyle="--")
    other = axes[3].twinx()
    other.plot(part.date, part.rv_ratio, color="#b38527", label="RV20/前252日中位值")
    other.legend(loc="upper right", fontsize=8)
    other.set_ylabel("波动率比")
    axes[0].set_ylabel("原价/元；黄高点，绿固定低点")
    axes[1].set_ylabel("亿份；黄线为同锚参考量")
    for ax in axes:
        ax.axvline(episode.confirm_up_date, color="#777", linestyle=":", linewidth=.7)
        ax.grid(alpha=.16)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(f"原案例{episode.episode_id}：量先恢复与同锚价格后确认\n底峰及灰色原5%确认竖线仅事后对齐；橙圆准备，红三角确认，绿叉准备失败", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    filename = f"案例{episode.episode_id}_量先恢复与价格后确认.png"
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    return filename


def run():
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "解释冻结来源发生变化。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, _ = load()
    data = data.copy()
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    states, anchors, pivots = rules.volume_lead_states(data)
    table(STATE_TABLE, states)
    table("全部参考高点_准备确认失败替换开放", anchors)
    table("原严格2日确认枢轴到达_未重选极值", pivots)
    checks = []
    for date in ("2015-06-17", "2019-01-18", "2020-06-08", "2024-09-30"):
        n = int(np.flatnonzero(data.date.eq(date))[0])+1
        prefix, _, _ = rules.volume_lead_states(data.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix, states.iloc[:n].reset_index(drop=True), check_exact=True)
        checks.append({"through": date, "rows": n, "status": "EXACT_KNOWN_VOLUME_PIVOT_LEAD_CONFIRMATION_AND_CONSUMPTION"})
    original_known = pd.read_parquet(OLD / "results/全部3488已知缺口与量价_前2015A未知保留.parquet")
    fields = ["date", "open", "high", "low", "close", "volume", "cash_shift", "ema20", "daily_dif", "daily_hist",
              "weekly_hist", "weekly_last_date", "relative_volume", "up_volume_balance5", "rv_ratio", "A_shares", "A_exposure", "A_known_target"]
    known = states.merge(original_known[fields], on="date", validate="one_to_one")
    require(len(known) == len(data), "全部日线未完整连接。")
    table(KNOWN_TABLE, known)
    original = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    scope = known.loc[known.date.ge("2015-01-01")]
    labels = scope.merge(original[["date", "idx", "status", "net_reference_return"]].rename(
        columns={"status": "old_label_status"}), on="date", how="left", validate="one_to_one")
    require(int(labels.idx.notna().sum()) == len(original), "原2846标签没有全部保留。")
    require(np.array_equal(labels.loc[labels.idx.notna(), "idx"].to_numpy(int),
                           labels.loc[labels.idx.notna(), "origin_index"].to_numpy(int)), "原标签索引错位。")
    labels.old_label_status = labels.old_label_status.fillna("NO_OLD_LABEL_NO_NEW_LABEL_CREATED")
    table("原全部每日标签和尾部_只解释", labels)
    events = labels.loc[labels.price_after_volume_event].copy()
    table(EVENT_TABLE, events)
    table("全部量先恢复原点_不筛MACD量比或波动", labels.loc[labels.volume_lead_event])
    failures = failure_paths(states, events)
    table("全部价格确认至原固定失效_仅路径非账户", failures)
    diagnostics = []
    for era, left, right in (("ALL", "2015-01-01", "2026-09-30"), ("2015_2019", "2015-01-01", "2019-12-31"),
                             ("2020_2023", "2020-01-01", "2023-12-31"), ("2024_2026", "2024-01-01", "2026-09-30")):
        local = events.loc[events.date.between(left, right)]
        mature = local.loc[local.old_label_status.eq("MATURE")]
        diagnostics.append({"era": era, "confirmations": len(local), "old_labels_unknown": len(local)-len(mature),
                            **descriptive_returns(mature.net_reference_return)})
    table("全时期旧标签描述_不作为实际胜率收益", pd.DataFrame(diagnostics))
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    mapped, case_rows, records, charts = [], [], [], []
    for ep in episodes.itertuples(index=False):
        first = scope.loc[scope.origin_index.between(ep.bottom_idx, ep.peak_idx)]
        after = scope.loc[scope.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        mapped.append({**ep._asdict(), "leads_bottom_to_peak": int(first.volume_lead_event.sum()),
                       "confirmations_bottom_to_peak": int(first.price_after_volume_event.sum()),
                       "confirmations_confirm_to_peak": int(after.price_after_volume_event.sum()),
                       "retrospective_alignment_only": True})
    map_frame = pd.DataFrame(mapped)
    table("原61分段及49上涨_量领先及确认两区间覆盖", map_frame)
    lines = ["# 具体上涨量价解释：累计量先恢复、价格后确认", "",
             "完整用途在新状态计算和结果连接前固定，下面先展开原案例与所有失败。本文件没有新金融账户，不能报实际交易胜率或夏普。", "",
             "累计线为涨日加真实成交份额、跌日减、平日不变，方向按已发生分红现金平移；量恢复不能当成主动买单、资金净流入或上涨原因。参考高点中心日后两日才可知。日周MACD、相对量和波动均为观察，不选门槛过滤。", ""]
    for identifier in CASE_IDS:
        ep = episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]
        lo, hi = max(0, int(ep.bottom_idx)-10), min(len(known)-1, int(ep.peak_idx)+10)
        local = known.loc[known.origin_index.between(lo, hi)].copy()
        local["original_episode_id"], local["retrospective_alignment_only"] = identifier, True
        case_rows.append(local)
        selected = local.loc[local.volume_lead_event | local.price_after_volume_event]
        terminal = anchors.loc[anchors.lead_index.ge(0) & anchors.terminal_index.between(lo, hi) & anchors.confirmation_index.lt(0)]
        lines.extend([f"## 原案例{identifier}：{ep.bottom_date:%Y-%m-%d}至{ep.peak_date:%Y-%m-%d}", "",
                      f"原底峰涨幅{ep.gross_rise:.2%}、5%确认日{ep.confirm_up_date:%Y-%m-%d}只用于事后定位。原前后各10交易日窗口中的全部准备与确认如下。", "",
                      "| 收盘日期 | 阶段 | 参考高点中心/当时确认 | 当前收盘 | 同锚高价 | 固定低点 | 量超参考/百万份 | 相对量 | 日DIF | 日柱 | 上周柱 | RV比 |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for event in selected.itertuples(index=False):
            center_date = known.date.iloc[int(event.reference_center_index)]
            known_date = known.date.iloc[int(event.reference_confirmation_index)]
            lines.append(f"| {event.date:%Y-%m-%d} | {'量先恢复' if event.volume_lead_event else '价格后确认'} | "
                         f"{center_date:%Y-%m-%d}/{known_date:%Y-%m-%d} | {event.close:.3f} | "
                         f"{event.reference_high_ticks*.001-event.cash_shift:.3f} | {event.fixed_low_ticks*.001-event.cash_shift:.3f} | "
                         f"{(int(event.cumulative_signed_volume)-int(event.reference_cumulative_volume))/1e6:.3f} | "
                         f"{event.relative_volume:.3f} | {event.daily_dif:+.6f} | {event.daily_hist:+.6f} | {event.weekly_hist:+.6f} | {event.rv_ratio:.3f} |")
        if not len(selected):
            lines.extend(["", "这个固定窗口没有量先恢复或价格后确认；该机制在这里没有提前识别点，不能换更旧高点或改变累计窗口补信号。"])
        lines.append("")
        for event in selected.itertuples(index=False):
            phase = "原底峰上涨之前" if event.origin_index < ep.bottom_idx else "原上涨分段内" if event.origin_index <= ep.peak_idx else "原上涨分段之后"
            daily = "日柱负" if event.daily_hist < 0 else "日柱非负"
            weekly = "上一完整周柱负" if event.weekly_hist < 0 else "上一完整周柱非负"
            action = "只是准备，尚无进场确认" if event.volume_lead_event else "价格确认收盘后才知，最早次开尝试"
            lines.extend([f"{event.date:%Y-%m-%d}处于{phase}，{daily}、{weekly}，相对量{event.relative_volume:.3f}、RV比{event.rv_ratio:.3f}。{action}。这些背景不能反过来证明量领先有效，后来的底峰涨幅不是本点位的实际回报。", ""])
        for anchor in terminal.itertuples(index=False):
            lines.extend([f"失败准备也保留：{anchor.lead_date:%Y-%m-%d}准备在{anchor.terminal_date:%Y-%m-%d}以{anchor.status}结束，未产生价格确认。", ""])
        filename = plot(known, anchors, ep)
        charts.append(filename)
        lines.extend([f"![原案例{identifier}](./{filename})", ""])
        records.append({"episode_id": identifier, "bottom_date": ep.bottom_date, "peak_date": ep.peak_date,
                        "gross_rise": float(ep.gross_rise), "known_lead_confirmation_rows": selected.to_dict("records"),
                        "failed_preparation_rows": terminal.to_dict("records"), "not_executable_bottom_peak": True})
    table("四原案例逐日全部量价及量先恢复状态", pd.concat(case_rows, ignore_index=True))
    admitted = map_frame.loc[map_frame.admitted]
    lines.extend(["## 全体、失败与下一步", "",
                  f"全部{len(known)}日、2015起{len(scope)}原点；全历史{len(anchors)}个高点锚、"
                  f"{int(states.volume_lead_event.sum())}次量先恢复、{int(states.price_after_volume_event.sum())}次价格后确认。"
                  f"全部锚终态{anchors.status.value_counts().to_dict()}。49正式上涨中{int(admitted.confirmations_bottom_to_peak.eq(0).sum())}段底峰间无确认，"
                  f"{int(admitted.confirmations_confirm_to_peak.eq(0).sum())}段5%确认至峰间无确认，不能称全部上涨启动的共同规律。", "",
                  INTENT, "", "所有旧标签/固定失效路径只解释，不代表实际交易净收益；持有中减仓、取消次开、重叠信号、股息和订单成本必须在4原口径完整账户中测量。"
                  "下一步只允许这一完整用途的唯一开发测量，旧失败不重跑、不按案例量比或MACD改过滤。", "",
                  "现有历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，尚不能称提高收益夏普或去除过拟合。", "",
                  "基础公式来源：[TradingView官方OBV说明](https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/)。其说明支持基础公式及分析思路，510300有效性由项目实验决定。", ""])
    with (OUT / "具体上涨与点位反推.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    cleaned_records = pd.DataFrame(events).astype(object).where(events.notna(), None).to_dict("records")
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R187", "status": "COMPLETED_ALL_VOLUME_FIRST_PRICE_LATER_EXPLANATION_NO_FINANCIAL_RESULT",
        "all_daily_rows": len(known), "origins_since_2015": len(scope), "all_reference_anchors": len(anchors),
        "all_anchor_terminal_counts": anchors.status.value_counts().to_dict(),
        "all_volume_leads": int(states.volume_lead_event.sum()), "all_price_confirmations": int(states.price_after_volume_event.sum()),
        "volume_leads_since_2015": int(scope.volume_lead_event.sum()), "price_confirmations_since_2015": len(events),
        "original_episodes": len(episodes), "original_admitted_waves": int(episodes.admitted.sum()),
        "waves_without_confirmation_bottom_to_peak": int(admitted.confirmations_bottom_to_peak.eq(0).sum()),
        "waves_without_confirmation_confirm_to_peak": int(admitted.confirmations_confirm_to_peak.eq(0).sum()),
        "old_labels": len(original), "mature_old_labels": int(original.status.eq("MATURE").sum()),
        "censored_old_labels": int(original.status.ne("MATURE").sum()), "tail_origins_without_new_label": int(labels.idx.isna().sum()),
        "case_records": records, "confirmed_event_records": cleaned_records, "old_label_descriptions": diagnostics,
        "charts": charts, "actual_prefix_checks": checks, "necessary_tests": 6,
        "frozen_sources": len(protocol["sources"]), "new_accounts": 0, "new_fits": 0, "new_training_labels": 0,
        "new_market_requests": 0, "historical_sample_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "source_first_vintage": "NOT_CERTIFIED", "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(f"全部{len(known)}日、{len(anchors)}参考锚、{int(states.volume_lead_event.sum())}准备、{len(events)}当期价格确认及四图解释完成；0金融账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定量价先后用途并展开全体及四案例。")
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
