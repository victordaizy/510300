"""券商周期框架的日周线观察：只解释已知状态，不事后命名牛熊或生成账户。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.broker_cycle_sources_v1 import OUT, ROOT, stamp, write_json

INPUTS = {
    "known_daily": ROOT / "reports/research/510300_cnh_macro_first_passage_v1/results/全部3488当时已知技术宏观与定盘偏离_未知保留.parquet",
    "cases": ROOT / "reports/research/510300_volume_lead_price_confirm_explanation_v1/results/四原案例逐日全部量价及量先恢复状态.parquet",
    "episodes": ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet",
}
STATE_NAMES = {
    "ACTIVE_EXPANDING": "活跃增加、波动扩张",
    "ACTIVE_SETTLING": "活跃增加、波动收敛",
    "QUIET_EXPANDING": "活跃减少、波动扩张",
    "QUIET_SETTLING": "活跃减少、波动收敛",
    "UNKNOWN": "观察不足",
}
COLORS = {"ACTIVE_EXPANDING": "#e8ad63", "ACTIVE_SETTLING": "#58a995",
          "QUIET_EXPANDING": "#d26d76", "QUIET_SETTLING": "#94a5b8", "UNKNOWN": "#dddddd"}
KEY_DATES = ("2015-06-24", "2015-06-25", "2015-06-29", "2015-06-30",
             "2019-01-08", "2019-01-14", "2019-01-18", "2020-03-27", "2020-04-01",
             "2020-04-23", "2020-05-28", "2020-05-29", "2020-06-08",
             "2024-09-23", "2024-09-24", "2024-09-26", "2024-09-30")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NaT:
        return None
    return value


def states(frame):
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    if not d.date.is_unique or not d.date.is_monotonic_increasing or not d.symbol.eq("510300.SH").all():
        raise ValueError("研究对象或日历与原冻结输入不一致。")
    economic_return = np.log((d.close + d.dividend) / d.close.shift())
    d["rv20"] = economic_return.rolling(20, min_periods=20).std(ddof=1)
    d["rv60"] = economic_return.rolling(60, min_periods=60).std(ddof=1)
    d["activity20_60"] = d.volume.rolling(20, min_periods=20).mean() / d.volume.rolling(60, min_periods=60).mean()
    d["volatility20_60"] = d.rv20 / d.rv60
    d["phase_known"] = np.isfinite(d[["activity20_60", "volatility20_60"]].to_numpy(float)).all(axis=1)
    active, expanding = d.activity20_60.gt(1.), d.volatility20_60.gt(1.)
    d["observed_state"] = np.select(
        [d.phase_known & active & expanding, d.phase_known & active & ~expanding,
         d.phase_known & ~active & expanding, d.phase_known & ~active & ~expanding],
        ["ACTIVE_EXPANDING", "ACTIVE_SETTLING", "QUIET_EXPANDING", "QUIET_SETTLING"], default="UNKNOWN")
    d["state_name"] = d.observed_state.map(STATE_NAMES)
    d["same_state_run_id"] = d.observed_state.ne(d.observed_state.shift()).cumsum()
    d["known_economic_change5"] = economic_return.rolling(5, min_periods=5).sum()
    d["daily_weekly_context"] = np.select(
        [d.daily_hist_atr.notna() & d.weekly_hist_atr.notna() & d.daily_hist_atr.gt(0) & d.weekly_hist_atr.gt(0),
         d.daily_hist_atr.notna() & d.weekly_hist_atr.notna() & d.daily_hist_atr.gt(0) & d.weekly_hist_atr.le(0),
         d.daily_hist_atr.notna() & d.weekly_hist_atr.notna() & d.daily_hist_atr.le(0) & d.weekly_hist_atr.gt(0),
         d.daily_hist_atr.notna() & d.weekly_hist_atr.notna() & d.daily_hist_atr.le(0) & d.weekly_hist_atr.le(0)],
        ["日周柱均正", "日柱转正、上周柱非正", "日柱非正、上周柱正", "日周柱均非正"], default="日周信息未知")
    known_week = d.weekly_last_date.notna()
    if not d.loc[known_week, "weekly_last_date"].lt(d.loc[known_week, "date"]).all():
        raise ValueError("完整周观察包含本周信息。")
    return d


def table(name, frame):
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f"{name}.parquet", index=False)
    frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def plots(cases, episodes):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for eid, chunk in cases.groupby("original_episode_id", sort=True):
        chunk = chunk.sort_values("date").reset_index(drop=True)
        episode = episodes.set_index("episode_id").loc[eid]
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [2.5, 1.3, 1.5, 1.2]})
        x = np.arange(len(chunk))
        axes[0].plot(x, chunk.ac, color="#183c58", linewidth=1.7, label="当日已知现金股息平移收盘")
        for i, key in enumerate(chunk.observed_state):
            axes[0].axvspan(i-.5, i+.5, color=COLORS[key], alpha=.23, linewidth=0)
        for date, label, color in [(episode.confirm_up_date, "原上涨确认日", "#295d8c")]:
            loc = np.flatnonzero(chunk.date.eq(date))
            if len(loc):
                for ax in axes:
                    ax.axvline(loc[0], color=color, linestyle="--", linewidth=.9)
                axes[0].text(loc[0]+.4, float(chunk.ac.max()), label, color=color, fontsize=9, va="top")
        axes[0].set_ylabel("经济价格 / 元")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[1].plot(x, chunk.activity20_60, label="ETF成交活跃度20/60", color="#277d6d")
        axes[1].plot(x, chunk.volatility20_60, label="收益波动20/60", color="#b47728")
        axes[1].axhline(1., color="#666666", linestyle=":", linewidth=1)
        axes[1].set_ylabel("相对比值")
        axes[1].legend(loc="upper left", fontsize=9, ncol=2)
        axes[2].bar(x, chunk.daily_hist_atr, label="日MACD柱 / ATR", color="#397899", alpha=.7)
        axes[2].plot(x, chunk.weekly_hist_atr, label="上一完整周MACD柱 / 周尺度ATR", color="#9a4d78")
        axes[2].axhline(0., color="#777777", linewidth=.6)
        axes[2].legend(loc="upper left", fontsize=9, ncol=2)
        axes[2].set_ylabel("标准化柱")
        axes[3].plot(x, chunk.funding_gap_pp, label="当时资金利差 / 百分点", color="#42618a")
        axes[3].plot(x, chunk.pmi_orders_level, label="已公布PMI订单减50", color="#879438")
        axes[3].axhline(0., color="#777777", linewidth=.6)
        axes[3].set_ylabel("宏观原字段")
        axes[3].legend(loc="upper left", fontsize=9, ncol=2)
        ticks = np.linspace(0, len(chunk)-1, min(8, len(chunk))).astype(int)
        axes[3].set_xticks(ticks, chunk.date.iloc[ticks].dt.strftime("%Y-%m-%d"), rotation=20)
        for ax in axes:
            ax.grid(axis="y", color="#dddddd", alpha=.6)
        legend = "；".join(f"{STATE_NAMES[key]}" for key in COLORS if key != "UNKNOWN")
        fig.suptitle(f"原案例{eid}：当时可见的参与、波动、日周线与宏观\n背景按四个观察状态着色；不把事后底峰用于状态计算", fontsize=13, y=.985)
        fig.text(.05, .012, "绿色=活跃增/收敛；橙色=活跃增/扩张；红色=活跃减/扩张；灰蓝=活跃减/收敛。未知不补。", fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95))
        path = folder / f"原案例{eid}_事前可见状态.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path.absolute().relative_to(ROOT)))
    return paths


def main():
    protocol_path = OUT / "phase_observation_protocol.json"
    if protocol_path.exists() or (OUT / "phase_observation_summary.json").exists():
        raise RuntimeError("本观察用途已登记或运行，不覆盖旧结果。")
    manifest = {key: {"path": str(path.absolute().relative_to(ROOT)), "sha256": digest(path)} for key, path in INPUTS.items()}
    write_json(protocol_path, {
        "at": stamp(), "registration_decision": "TECH.R209", "result_decision": "TECH.R210",
        "question": "券商市场阶段思路能否用于解释原上涨与反例，如何区分同一活跃波动状态的不同价格及宏观背景？",
        "mechanism": "共同状态观察，不再统一给宏观或放量加分；此处不识别真实牛熊周期，也不运行策略。",
        "state": "ETF20日平均成交份额/60日平均份额与含息日log收益20日/60日样本标准差，各以1为固定分界，交叉四类；相等归不增加/不扩张，未知不分组。",
        "source_difference": "华泰原版使用指数实际换手率及长周期波动；本观察为ETF成交活跃度替代性的独立描述，不是原指标复现。20/60沿用既有标准尺度，不搜索窗口或四类标签。",
        "clock": "当日16:00观察，上一完整周；若之后另立交易用途，下一合法开盘。宏观逐值复用R208已知钟和未知。",
        "selection": "全部3488原日、全部61原分段/49正式上涨、四原案例240行和原17关键日，均不新增或挑选。",
        "future_usage": "既有分段只作事后对齐，不进入计算；不读取新未来收益，不拟合、生成新标签或新账户。",
        "prefix_check": "17原关键日逐一截断输入复算状态，要求过去观察精确相同；完整周钟保留。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "new_fits": 0, "new_accounts": 0, "financial_metrics": "NOT_COMPUTED",
        "inputs": manifest, "code_sha256": digest(Path(__file__))})
    frame = pd.read_parquet(INPUTS["known_daily"])
    d = states(frame)
    if len(d) != 3488 or not frame.volume_unit.eq("share").all() or not frame.amount_unit.eq("CNY").all():
        raise ValueError("原观察数量或成交量额单位不符合来源合同。")
    compared = ["date", "activity20_60", "volatility20_60", "observed_state", "daily_weekly_context"]
    for day in KEY_DATES:
        index = int(np.flatnonzero(d.date.eq(pd.Timestamp(day)))[0])
        prefix = states(frame.iloc[:index+1])
        pd.testing.assert_frame_equal(prefix[compared], d.iloc[:index+1][compared], check_exact=True)
    case_source = pd.read_parquet(INPUTS["cases"])
    episodes = pd.read_parquet(INPUTS["episodes"])
    columns = ["date", "observed_state", "state_name", "phase_known", "same_state_run_id", "activity20_60", "volatility20_60", "rv20", "rv60", "daily_weekly_context", "known_economic_change5"]
    cases = case_source[["date", "original_episode_id", "retrospective_alignment_only"]].merge(d, on="date", how="left", validate="many_to_one")
    confirmations = episodes.merge(d[columns], left_on="confirm_up_date", right_on="date", how="left", validate="many_to_one")
    for path in INPUTS.values():
        key = next(key for key, value in INPUTS.items() if value == path)
        if digest(path) != manifest[key]["sha256"]:
            raise ValueError("观察期间原输入发生变化。")
    table("全部3488原日_参与波动与宏观背景_未知保留", d)
    table("全部61原分段_确认日观察_底峰仅对齐", confirmations)
    table("四原案例240行_共同状态观察", cases)
    key_rows = d.loc[d.date.isin(pd.to_datetime(KEY_DATES))].copy()
    table("原17关键日_共同状态与不同背景", key_rows)
    evaluation = d.loc[d.date.ge(pd.Timestamp("2015-01-05"))]
    summaries = []
    for name, start, end in [("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")]:
        part = d.loc[d.date.between(pd.Timestamp(start), pd.Timestamp(end))]
        runs = part.groupby("same_state_run_id").size()
        summaries.append({"period": name, "rows": len(part), "state_counts": part.state_name.value_counts().to_dict(),
                          "state_runs": len(runs), "median_run_sessions": float(runs.median()),
                          "active_expanding_with_observed_negative_change5": int((part.observed_state.eq("ACTIVE_EXPANDING") & part.known_economic_change5.lt(0)).sum())})
    impacts = 100000. / evaluation.amount
    capacity = {"illustrative_order_notional_cny": 100000., "daily_amount_unit": "CNY",
                "median_order_to_daily_amount": float(impacts.median()), "maximum_order_to_daily_amount": float(impacts.max()),
                "interpretation": "只是规模与日成交额比较，不证明次开盘冲击、成交或盈利；不增加风险上限。",
                "base_roundtrip_rate_100000_before_tick": .0014, "stress_roundtrip_rate_100000_before_tick": .0028,
                "base_roundtrip_rate_3000_before_tick": 10./3000+.001,
                "stress_roundtrip_rate_3000_before_tick": 10./3000+.002}
    figures = plots(cases, episodes)
    summary = {
        "at": stamp(), "status": "COMPLETED_SOURCE_AND_CAUSAL_PHASE_DESCRIPTION_NOT_STRATEGY_VALIDATION",
        "decision": "TECH.R210", "all_calendar_rows": len(d), "evaluation_calendar_rows": len(evaluation),
        "case_rows": len(cases), "episode_rows": len(episodes), "admitted_episodes": int(episodes.admitted.sum()),
        "key_dates": len(key_rows), "prefix_checks": len(KEY_DATES), "figures": figures,
        "all_observation_state_counts": d.state_name.value_counts().to_dict(), "period_description": summaries,
        "admitted_confirmation_states": confirmations.loc[confirmations.admitted.eq(True), "state_name"].fillna("观察未知").value_counts().to_dict(),
        "small_capital_scale": capacity, "new_labels": 0, "new_fits": 0, "new_accounts": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "source_contract": "ETF交易活跃度不能冒称指数换手率；四状态既含涨也含跌，必须进一步区分背景。",
        "next_experiment_status": "MECHANISM_HYPOTHESIS_NOT_FULL_FINANCIAL_REGISTRATION",
        "prior_financial_result_unchanged": "TECH.R208_REJECTED", "inputs": manifest}
    write_json(OUT / "phase_observation_summary.json", clean(summary))
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
