"""一次描述全部进入信息的更新顺序；只复用已知周期，零新交易账户。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_stage_policy_study_v1 as parent
from research import broker_fixed_cohort_observation_v1 as original
from research import entry_information_sequence_inputs_v1 as inputs
from research import industry_structure_description_study_v1 as industry
from research import industry_leader_failure_study_v1 as financial

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_entry_information_sequence_description_v1"
OBSERVED = original.SOURCEFILES["observed"]
INDUSTRY_DAILY = industry.OUT / "results/全部3488逐点行业结构_未知与源龄.parquet"
EVENTS = industry.OUT / "results/全部143原阶段事件_行业锚与原周期上下文.parquet"
KEY_DATES = [*original.KEY_DATES, pd.Timestamp("2017-04-05"), pd.Timestamp("2026-05-11")]
NEGATIVE_CASES = (("2017-04-05", "2017-04-19"), ("2026-05-11", "2026-05-19"))
SOURCE_NAMES = {"orders": "PMI新订单", "policy_rate": "对应政策利率", "funding": "日常资金", "margin": "融资"}


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def financial_paths():
    return [financial.OUT / "accounts" / period / "STRESS" / financial.candidate.POLICIES[0] / "trades.parquet"
        for period in parent.PERIODS]


def source_paths():
    return [Path(__file__), Path(inputs.__file__), ROOT / "tests/test_entry_information_sequence_v1.py",
        ROOT / "docs/510300_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_V1.md", OBSERVED, INDUSTRY_DAILY, EVENTS,
        original.SOURCEFILES["cases"], *financial_paths(), financial.OUT / "summary.json",
        industry.OUT / "summary.json", ROOT / "research/macro_technical_first_passage_inputs_v1.py"]


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("进入信息顺序已登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["passed"] != 6 or tests["exit_code"] != 0 or tests["inputs_sha256"] != parent.digest(Path(inputs.__file__)):
        raise ValueError("六必要测试未通过或描述输入改变。")
    if parent.read(financial.OUT / "summary.json")["decision"] != "TECH.R224":
        raise ValueError("原金融上下文不匹配。")
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": "TECH.R225", "decision": "TECH.R226",
        "purpose": "ALL_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_NOT_FINANCIAL", "all_daily_slots": 3488,
        "evaluation_slots": 2855, "all_original_stage_events": 143, "original_case_rows": 240,
        "key_dates": [str(date.date()) for date in KEY_DATES], "negative_case_windows": [list(pair) for pair in NEGATIVE_CASES],
        "macro_decision_clock": "原decision_time 16:00；不提前来源。行业仍严格前一完整源日。",
        "identity_and_publication": "首次观察身份与上一决定之后到当前决定之间的新公布分开；原known与时钟门；未知不补。",
        "policy_scope": "仅对应已生效利率记录与policy_known_at，未覆盖全部政策宣布；每日资金/融资更新不是政策冲击。",
        "repeat_count": "全部历史顺序中，同PMI/利率身份、同原路线此前事件数；只计过去，非实际交易次数。",
        "contexts": "143原事件及R212/R224已有完整/开放周期仅作既有事后解释，不计算新胜率或未来训练标签。",
        "necessary_tests": 6, "prefix_checks": 19, "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0,
        "new_market_requests": 0, "parameter_grid": False, "first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "financial_metrics": "NOT_COMPUTED",
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in source_paths()]})
    print("R225进入信息顺序固定：3488槽/143事件、240原案例、两个已知反例、19截断；零新账户。", flush=True)


def dates(frame, column="date"):
    frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return frame


def contexts():
    frames = []
    for period, path in zip(parent.PERIODS, financial_paths()):
        frame = pd.read_parquet(path)
        for name in ("entry_origin", "entry_date", "exit_date"):
            dates(frame, name)
        frame["r224_period"] = period
        names = ["entry_origin", "entry_date", "exit_date", "status", "source", "net_pnl", "net_return", "holding_sessions", "r224_period"]
        frame = frame[names].rename(columns={name: "r224_" + name for name in names if name != "entry_origin" and name != "r224_period"})
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.entry_origin.duplicated().any():
        raise ValueError("原R224周期信号身份重复。")
    return result.rename(columns={"entry_origin": "date"})


def source_summaries(frame, scope):
    rows = []
    for name in inputs.SOURCES:
        known = frame[f"{name}_clock_admitted"]
        ages = frame.loc[known, f"{name}_source_age_calendar_days"]
        rows.append({"scope": scope, "source": name, "rows": len(frame), "known": int(known.sum()), "unknown": int((~known).sum()),
            "first_observed": int(frame[f"{name}_first_observed_here"].sum()),
            "new_publication": int(frame[f"{name}_publication_since_previous_decision"].sum()),
            "first_seen_without_new_publication": int(frame[f"{name}_first_observed_without_new_publication"].sum()),
            "future_clock": int(frame[f"{name}_future_clock"].sum()),
            "median_source_age_days": ages.median(), "max_source_age_days": ages.max()})
    return rows


def existing_context_summary(events):
    rows = []
    for source in ("orders", "policy_rate"):
        for route, group in events.groupby("stage_entry_type", dropna=False):
            count = group[f"{source}_earlier_same_source_route_events"]
            statuses = pd.Series(np.where(count.isna(), "来源未知", np.where(count.eq(0), "该来源下该路线首次", "该来源下该路线重复")), index=group.index)
            for status, values in group.groupby(statuses, sort=True):
                original = values.loc[values.original_account_context.eq("COMPLETE")]
                actual = values.loc[values.r224_status.eq("COMPLETE")]
                rows.append({"source": source, "route": route, "source_route_status": status,
                    "original_events": len(values), "original_no_complete_cycle": len(values)-len(original),
                    "original_complete_contexts": len(original), "original_positive_contexts": int(original.original_cycle_net_pnl.gt(0).sum()),
                    "original_negative_contexts": int(original.original_cycle_net_pnl.lt(0).sum()),
                    "original_existing_net_pnl_sum": original.original_cycle_net_pnl.sum(),
                    "r224_complete_contexts": len(actual), "r224_positive_contexts": int(actual.r224_net_pnl.gt(0).sum()),
                    "r224_negative_contexts": int(actual.r224_net_pnl.lt(0).sum()),
                    "r224_existing_net_pnl_sum": actual.r224_net_pnl.sum(),
                    "new_publication_events": int(values[f"{source}_publication_since_previous_decision"].sum()),
                    "role": "原金融已知上下文，不是新进入策略、独立样本或新胜率"})
    return pd.DataFrame(rows)


def draw_case(frame, title, filename, marked_date=None):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(5, 1, figsize=(12, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, 1, 1, 1.1]})
    x = np.arange(len(frame))
    start = float(frame.ac.iloc[0])
    axes[0].plot(x, frame.ac/start*100, label="价格含现金平移（起日100）", color="#244b78")
    axes[0].plot(x, frame.ema20/start*100, label="20日EMA", color="#d97b28", alpha=.85)
    events = frame.stage_entry_type.ne("NONE")
    axes[0].scatter(x[events], frame.loc[events, "ac"]/start*100, marker="^", color="#9d2553", s=28, label="原进入事件")
    axes[0].set_ylabel("价格指数"); axes[0].legend(loc="upper left", fontsize=8)
    axes[1].bar(x, np.exp(frame.log_relative_volume), color="#7997aa", width=.8, label="原相对成交量")
    axes[1].axhline(1, color="#888888", linewidth=.8); axes[1].set_ylabel("相对成交量")
    axes[2].plot(x, frame.daily_hist_atr, label="日MACD柱/原ATR", color="#40755b")
    axes[2].plot(x, frame.weekly_hist_atr, label="上一完整周MACD柱/原ATR", color="#a74c50", linestyle="--")
    axes[2].axhline(0, color="#888888", linewidth=.8); axes[2].set_ylabel("各自标准化MACD")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[3].plot(x, frame.industry_positive5_fraction*100, label="行业5日上涨比例（前一源日）", color="#266f77")
    axes[3].plot(x, frame.rotation_churn5blocks*100, label="两个5块排名变化", color="#a57d2c", linestyle="--")
    axes[3].set_ylim(-5, 105); axes[3].set_ylabel("比例 %"); axes[3].legend(loc="upper left", fontsize=8)
    for y, name in enumerate(inputs.SOURCES):
        known = frame[f"{name}_clock_admitted"]
        fresh = frame[f"{name}_publication_since_previous_decision"]
        first_old = frame[f"{name}_first_observed_without_new_publication"]
        axes[4].scatter(x[known], np.full(int(known.sum()), y), marker=".", color="#bcc5ce", s=22)
        axes[4].scatter(x[fresh], np.full(int(fresh.sum()), y), marker="|", color="#00705b", s=130, label="本区间新公布" if y == 0 else None)
        axes[4].scatter(x[first_old], np.full(int(first_old.sum()), y), marker="x", color="#d08327", s=35, label="首次见到旧公布记录" if y == 0 else None)
    axes[4].set_yticks(range(4), list(SOURCE_NAMES.values())); axes[4].set_ylim(-.6, 3.6)
    axes[4].legend(loc="upper left", fontsize=8)
    if marked_date is not None:
        index = np.flatnonzero(frame.date.eq(pd.Timestamp(marked_date)))
        if len(index) != 1:
            raise ValueError("反例信号日未唯一落入窗口。")
        for ax in axes:
            ax.axvline(index[0], color="#af2350", linestyle=":", alpha=.75)
    ticks = np.unique(np.linspace(0, len(frame)-1, min(8, len(frame)), dtype=int))
    axes[-1].set_xticks(ticks, [str(frame.date.iloc[i].date()) for i in ticks], rotation=25, ha="right")
    for ax in axes:
        ax.grid(alpha=.14)
    fig.suptitle(title+f"｜{len(frame)}原交易日\n当前行宏观决定16:00；行业为前一完整源日；利率记录不覆盖全部政策宣布", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    path = OUT / "figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return relative(path)


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("进入信息顺序已开始，不重复运行。")
    protocol = parent.read(OUT / "protocol.json")
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("冻结描述输入改变：" + source["path"])
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts": 0})
    observed = dates(pd.read_parquet(OBSERVED))
    described = inputs.describe(observed)
    prefix = []
    for date in KEY_DATES:
        shorter = observed.loc[observed.date.le(date)].copy()
        cut = inputs.describe(shorter)
        pd.testing.assert_frame_equal(described.iloc[:len(shorter)].reset_index(drop=True), cut, rtol=0, atol=0)
        prefix.append({"cut": str(date.date()), "all_prior_rows_exact": True})
    daily_industry = dates(pd.read_parquet(INDUSTRY_DAILY))
    fields = [name for name in daily_industry if name not in observed.columns or name == "date"]
    daily_industry = daily_industry[fields].rename(columns={"view_allowed": "industry_view_allowed"})
    if not daily_industry.date.equals(observed.date):
        raise ValueError("原宏观与行业3488日错位。")
    daily = observed.merge(described.drop(columns=["decision_time", "stage_entry_type"]), on="date", validate="one_to_one")
    daily = daily.merge(daily_industry, on="date", validate="one_to_one")
    prior_events = dates(pd.read_parquet(EVENTS))
    meta_fields = [name for name in prior_events if name in ("date", "anchor_id", "anchor_date", "event_type", "anchor_status") or name.startswith("original_")]
    events = daily.loc[daily.date.ge(pd.Timestamp("2015-01-05")) & daily.any_original_stage_event].copy()
    events = events.merge(prior_events[meta_fields], on="date", how="left", validate="one_to_one")
    events = events.merge(contexts(), on="date", how="left", validate="one_to_one")
    if len(events) != 143 or events.anchor_id.isna().any() or len(daily) != 3488:
        raise ValueError("原全部日历或143事件改变。")
    evaluation = daily.loc[daily.date.ge(pd.Timestamp("2015-01-05"))]
    if len(evaluation) != 2855:
        raise ValueError("原研究分母2855改变。")
    source_cases = dates(pd.read_parquet(original.SOURCEFILES["cases"], columns=["date", "original_episode_id"]))
    cases = source_cases.merge(daily, on="date", how="left", validate="one_to_one")
    keys = cases.loc[cases.date.isin(original.KEY_DATES)].copy()
    if len(cases) != 240 or len(keys) != 17:
        raise ValueError("原四案例与17关键日未完整保留。")
    negative_frames, figures = [], []
    case_titles = {18: "2015反弹后继续下跌反例", 37: "2019修复上涨", 42: "2020修复上涨与分类未知", 55: "2024政策重新定价上涨"}
    for case_id, frame in cases.groupby("original_episode_id", sort=True):
        figures.append(draw_case(frame, case_titles[int(case_id)], f"原案例{int(case_id)}_量价行业与信息公布顺序.png"))
    all_contexts = contexts()
    for signal, end in NEGATIVE_CASES:
        i = int(np.flatnonzero(daily.date.eq(pd.Timestamp(signal)))[0])
        start = max(0, i-10)
        frame = daily.iloc[start:].loc[lambda data: data.date.le(pd.Timestamp(end))].copy()
        frame["negative_signal"] = pd.Timestamp(signal)
        negative_frames.append(frame)
        context = all_contexts.loc[all_contexts.date.eq(pd.Timestamp(signal))]
        if len(context) != 1 or context.r224_exit_date.iloc[0] != pd.Timestamp(end) or context.r224_net_pnl.iloc[0] >= 0:
            raise ValueError("原新增亏损反例身份或已知退出日改变。")
        figures.append(draw_case(frame, f"释放资金后新增进入亏损：{signal}", f"新增亏损{signal}_信息与量价顺序.png", signal))
    negatives = pd.concat(negative_frames, ignore_index=True)
    negative_keys = events.loc[events.date.isin(pd.to_datetime([pair[0] for pair in NEGATIVE_CASES]))].copy()
    if len(negative_keys) != 2:
        raise ValueError("两个新增信号不是原143阶段事件。")
    source_rows = source_summaries(daily, "全部3488观察") + source_summaries(evaluation, "原2855研究日历") + source_summaries(events, "原143进入事件")
    for year, frame in evaluation.groupby(evaluation.date.dt.year):
        source_rows.extend(source_summaries(frame, str(int(year))+"原完整或当前部分年"))
    summaries = pd.DataFrame(source_rows)
    repeats = existing_context_summary(events)
    states = events.groupby(["stage_entry_type", "descriptive_structure_state"], dropna=False).size().reset_index(name="all_original_events")
    for name, frame in (("全部3488进入信息身份_公布钟与量价行业", daily), ("全部143进入事件_来源重复与既有周期上下文", events),
            ("原四上涨反例240日_信息公布与量价行业顺序", cases), ("原17关键日_来源钟与行业量价", keys),
            ("两个新增亏损信号_全部进入信息与上下文", negative_keys), ("两个新增亏损窗口_原信号前10槽至已知退出", negatives),
            ("全部来源范围年份_更新与未知分母", summaries), ("全部同来源路线首次重复_既有上下文而非新胜率", repeats),
            ("全部143原路线与行业状态_不删未知", states)):
        table(name, frame)
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("一次描述改变原输入：" + source["path"])
    summary = {"at": parent.original.now(), "registration": "TECH.R225", "decision": "TECH.R226",
        "status": "ALL_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_COMPLETED_NOT_FINANCIAL",
        "daily_rows": len(daily), "evaluation_rows": len(evaluation), "original_events": len(events), "case_rows": len(cases),
        "original_key_rows": len(keys), "negative_signal_rows": len(negative_keys), "negative_window_rows": len(negatives),
        "necessary_tests_passed": 6, "prefix_checks": prefix, "source_files_unchanged": len(protocol["files"]),
        "source_summary": summaries.loc[summaries.scope.isin(["原2855研究日历", "原143进入事件"])].to_dict("records"),
        "repeat_existing_context_summary": repeats.to_dict("records"), "figures": figures,
        "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0, "new_market_requests": 0,
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "latest_financial_decision_unchanged": "TECH.R224", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    print(f"R226进入信息顺序完成：3488槽/143事件、240原案例、{len(negatives)}反例窗口行、19截断、6图；零新账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定登记或一次描述进入信息顺序")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("必须唯一选择--freeze或--run")
    freeze() if args.freeze else run()
