"""一次观察全部行业结构、原阶段事件和原上涨/失效；0新交易账户。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_fixed_cohort_inputs_v1 as prices
from research import broker_fixed_cohort_observation_v1 as original
from research import industry_structure_description_inputs_v1 as inputs
from research import official_industry_full_sequence_intake_v1 as sources

ROOT, parent = original.ROOT, original.parent
OUT = ROOT / "reports/research/510300_industry_structure_description_v1"
SOURCE_ROOT = sources.OUT / "implementation_v1_0_1"
SOURCE_MEMBERS = SOURCE_ROOT / "results/全部实际源成员逐行分类_未知与出处保留.parquet"
SOURCE_CALENDAR = SOURCE_ROOT / "results/全部3488日官方发布钟_成员覆盖_源龄与未知.parquet"
COUNTER_DATES = pd.to_datetime(["2019-06-12", "2019-06-19", "2019-06-20", "2019-07-02", "2025-06-26", "2025-07-03", "2025-07-04"])


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame, csv=True):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    if csv:
        frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def source_paths():
    return [Path(__file__), Path(inputs.__file__), Path(prices.__file__),
        ROOT / "tests/test_industry_structure_description_v1.py", ROOT / "docs/510300_INDUSTRY_STRUCTURE_DESCRIPTION_V1.md",
        SOURCE_MEMBERS, SOURCE_CALENDAR, SOURCE_ROOT / "summary.json", *original.SOURCEFILES.values(),
        ROOT / "reports/research/510300_broker_cohort_failure_v1/post_run_diagnosis.json"]


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("行业结构描述已经登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 4 or tests["inputs_sha256"] != parent.digest(Path(inputs.__file__)):
        raise ValueError("四项必要测试未通过或输入代码改变。")
    prior = parent.read(SOURCE_ROOT / "summary.json")
    if prior["decision"] != "TECH.R220" or prior["actual_member_rows"] != 846900:
        raise ValueError("实际全来源面板不匹配。")
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": "TECH.R221", "decision": "TECH.R222",
        "purpose": "INDUSTRY_STRUCTURE_AND_FIXED_PROPAGATION_DESCRIPTION_NOT_FINANCIAL",
        "all_daily_slots": 3488, "all_stage_events": 143, "event_observation_sessions": list(range(21)),
        "original_case_rows": 240, "original_key_dates": [date.isoformat() for date in original.KEY_DATES],
        "known_exit_counter_dates": [date.isoformat() for date in COUNTER_DATES],
        "source_lag": "前一完整ETF交易日；严格原成员与当时可知分类，股息/报价实际首可得仍未认证。",
        "global_view": "原市场VIEW_ALLOWED且300成员中分类和20历史同时明确至少294，未知不补。",
        "industry_set": "当前固定名单，至少5成员/20历史98%完整；前3按20日相对ETF选取，至少4行业；源龄诊断不过滤。",
        "features": ["领先行业20/5日相对强度", "领先相对优势两个5块变化", "保留行业5日上涨比例", "两个不重叠5块排名变化"],
        "fixed_anchor": "全部原阶段信号与四案例起日，行业/股票/分类/领先身份固定；后锚0至20槽只用当期前一源日。",
        "outcome_context": "原R212完整周期在1/5/20槽只是已有说明性上下文；严格信息早于原退出，不能当成新策略或进入胜率。",
        "parameter_search": "NONE_SINGLE_FIXED_DESCRIPTION", "necessary_tests": 4, "prefix_checks": 17,
        "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0, "new_market_requests": 0,
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in dict.fromkeys(source_paths())]})
    print("R221行业结构描述固定：全部3488槽/143事件、原240案例/17关键日、20历史/5块/前三行业；0账户。", flush=True)


def load():
    q, m, c, observed = original.load_raw()
    for frame, field in ((q, "date"), (m, "membership_date"), (c, "date"), (observed, "date")):
        frame[field] = pd.to_datetime(frame[field]).astype("datetime64[ns]")
    panel = prices.make_panel(observed.date, q, m, c)
    columns = ["date", "membership_source_date", "symbol", "industry_key", "major_name", "row_source_known",
        "row_origin", "section_origin", "major_origin", "pdf_page", "pdf_sha256"]
    members = pd.read_parquet(SOURCE_MEMBERS, columns=columns)
    members["date"] = pd.to_datetime(members.date).astype("datetime64[ns]")
    members["membership_source_date"] = pd.to_datetime(members.membership_source_date).astype("datetime64[ns]")
    calendar = pd.read_parquet(SOURCE_CALENDAR)
    calendar["date"] = pd.to_datetime(calendar.date).astype("datetime64[ns]")
    calendar["membership_source_date"] = pd.to_datetime(calendar.membership_source_date).astype("datetime64[ns]")
    if not calendar.date.equals(observed.date):
        raise ValueError("行业源3488日期与原ETF日历不完全一致。")
    return q, m, c, observed, panel, members, calendar


def identity(panel, anchor):
    return [(group.key, group.name, group.leading, tuple(str(panel.symbols[i]) for i in group.members)) for group in anchor.groups]


def background(row):
    names = ["close", "daily_hist", "weekly_hist", "weekly_last_date", "log_relative_volume", "volatility20_60",
        "pmi_orders_level", "pmi_orders_change", "orders_known", "funding_gap_pp", "funding_gap_change5", "funding_known",
        "financing_net_change5", "margin_known", "stage_entry_type"]
    return {"etf_close" if name == "close" else name: row[name] for name in names}


def fixed_names(row):
    return {"fixed_snapshot_id" if key == "industry_snapshot_id" else "fixed_taxonomy" if key == "industry_taxonomy" else key: value for key, value in row.items()}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("行业结构描述已经开始，不重复运行。")
    protocol = parent.read(OUT / "protocol.json")
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("冻结来源或描述代码改变：" + source["path"])
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts": 0})
    q, m, c, observed, panel, members, calendar = load()
    etf_logs = np.log((observed.close + observed.dividend) / observed.close.shift()).to_numpy(float)
    slots = calendar.set_index("date").to_dict("index")
    locations = members.groupby("date", sort=False).indices
    empty = members.iloc[:0]
    def source_members(i):
        return members.iloc[locations[panel.dates[i]]] if panel.dates[i] in locations else empty
    daily, industries = [], []
    for i, date in enumerate(panel.dates):
        feature, anchor, groups = inputs.describe(panel, i, source_members(i), slots[date], etf_logs)
        daily.append({**feature, **background(observed.iloc[i])})
        industries.extend({"date": date, "industry_snapshot_id": anchor.snapshot_id, "industry_taxonomy": anchor.taxonomy,
            "industry_source_age_days": anchor.source_age_days, **group} for group in groups)
        if i and i % 500 == 0:
            print(f"逐点行业结构已观察{i}/{len(panel.dates)}日。", flush=True)
    prefix = []
    for date in original.KEY_DATES:
        i = int(panel.dates.get_loc(date))
        shorter = prices.make_panel(panel.dates[:i + 1], q.loc[q.date.le(date)], m.loc[m.membership_date.le(date)], c.loc[c.date.le(date)])
        full, anchor, full_groups = inputs.describe(panel, i, source_members(i), slots[date], etf_logs)
        cut, cut_anchor, cut_groups = inputs.describe(shorter, i, source_members(i), slots[date], etf_logs[:i + 1])
        pd.testing.assert_series_equal(pd.Series(full), pd.Series(cut), check_dtype=False, check_names=False, rtol=0, atol=0)
        if identity(panel, anchor) != identity(shorter, cut_anchor) or full_groups != cut_groups:
            raise ValueError("输入截断后的行业身份或已知统计改变。")
        prefix.append({"cut": date, "identity_and_features_exact": True})
    periods = []
    for period, name in (("2015_2019", "earlier_trades"), ("2020_2026", "recent_trades")):
        frame = pd.read_parquet(original.SOURCEFILES[name])
        for column in ("entry_origin", "entry_date", "exit_date"):
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
        frame["period"] = period
        periods.append(frame)
    trades = pd.concat(periods, ignore_index=True)
    if trades.entry_origin.duplicated().any():
        raise ValueError("原周期信号上下文重复。")
    contexts = {row.entry_origin: row for row in trades.itertuples(index=False)}
    candidates = np.flatnonzero(observed.stage_entry_type.ne("NONE") & observed.date.between("2015-01-05", "2026-09-30"))
    if len(candidates) != 143:
        raise ValueError("原143阶段事件范围改变。")
    events, profiles, fixed_industries, identities = [], [], [], []
    for serial, a in enumerate(candidates, 1):
        anchor_id = f"EVENT_{serial:04d}"
        feature, anchor, stats = inputs.describe(panel, int(a), source_members(int(a)), slots[panel.dates[a]], etf_logs)
        context = contexts.get(panel.dates[a])
        meta = {"anchor_id": anchor_id, "anchor_date": panel.dates[a], "event_type": observed.stage_entry_type.iloc[a],
            "anchor_status": anchor.status, "original_account_context": context.status if context is not None else "NO_ACTUAL_CYCLE",
            "original_period": context.period if context is not None else None,
            "original_cycle_net_pnl": context.net_pnl if context is not None else np.nan,
            "original_cycle_net_return": context.net_return if context is not None else np.nan,
            "original_entry_date": context.entry_date if context is not None else pd.NaT,
            "original_exit_date": context.exit_date if context is not None else pd.NaT}
        events.append({**meta, **feature})
        identities.extend(identity_rows(panel, anchor, source_members(int(a)), anchor_id))
        for i in range(int(a), min(int(a) + 21, len(panel.dates))):
            profile, groups = inputs.observe_fixed(panel, anchor, i, etf_logs)
            profile = {**meta, **fixed_names(profile), **background(observed.iloc[i])}
            profile["information_before_original_entry_open"] = bool(context is not None and panel.dates[i] < context.entry_date)
            profile["information_before_original_exit_open"] = bool(context is not None and pd.notna(context.exit_date) and panel.dates[i] < context.exit_date)
            profiles.append(profile)
            fixed_industries.extend({"anchor_id": anchor_id, "anchor_date": panel.dates[a], **group} for group in groups)
    case_source = pd.read_parquet(original.SOURCEFILES["cases"], columns=["date", "original_episode_id"])
    case_source["date"] = pd.to_datetime(case_source.date).astype("datetime64[ns]")
    if len(case_source) != 240:
        raise ValueError("原四案例240行改变。")
    cases = []
    for case_id, group in case_source.groupby("original_episode_id", sort=True):
        dates = group.date.sort_values()
        a = int(panel.dates.get_loc(dates.iloc[0]))
        feature, anchor, stats = inputs.describe(panel, a, source_members(a), slots[panel.dates[a]], etf_logs)
        identities.extend(identity_rows(panel, anchor, source_members(a), f"CASE_{case_id}"))
        for date in dates:
            i = int(panel.dates.get_loc(date))
            profile, groups = inputs.observe_fixed(panel, anchor, i, etf_logs)
            cases.append({"original_episode_id": int(case_id), **daily[i], **fixed_names(profile), "daily_view_allowed": daily[i]["view_allowed"]})
    daily_frame, case_frame, profile_frame = pd.DataFrame(daily), pd.DataFrame(cases), pd.DataFrame(profiles)
    key_frame = case_frame.loc[case_frame.date.isin(original.KEY_DATES)].copy()
    if len(key_frame) != 17:
        raise ValueError("原17关键日未全部保留。")
    context_rows = []
    complete = profile_frame.loc[profile_frame.original_account_context.eq("COMPLETE") & profile_frame.relative_session.isin([1, 5, 20])]
    for keys, values in complete.groupby(["original_period", "relative_session", "descriptive_fixed_state"], dropna=False):
        context_rows.append({"period": keys[0], "relative_session": int(keys[1]), "state": keys[2], "original_completed_cycles": len(values),
            "original_wins": int(values.original_cycle_net_pnl.gt(0).sum()), "original_losses": int(values.original_cycle_net_pnl.lt(0).sum()),
            "information_before_original_exit": int(values.information_before_original_exit_open.sum()),
            "role": "原R212已有上下文，不是新策略胜率或可执行收益"})
    for name, frame, csv in (("全部3488逐点行业结构_未知与源龄", daily_frame, True),
        ("全部当时固定行业_过去强弱与轮动", pd.DataFrame(industries), False),
        ("全部143原阶段事件_行业锚与原周期上下文", pd.DataFrame(events), True),
        ("全部原事件0至20槽_固定行业传播", profile_frame, True),
        ("全部事件固定行业逐组后锚观察", pd.DataFrame(fixed_industries), False),
        ("全部固定锚实际成员_归属与未知", pd.DataFrame(identities), False),
        ("原四案例240行_行业结构与价量宏观", case_frame, True),
        ("原17关键日行业结构_源龄与未知", key_frame, True),
        ("原R216退出反例日期_逐点行业结构", daily_frame.loc[daily_frame.date.isin(COUNTER_DATES)], True),
        ("原周期1_5_20槽说明性上下文", pd.DataFrame(context_rows), True)):
        table(name, frame, csv)
    figures = draw_cases(case_frame)
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("一次观察改变原冻结输入：" + source["path"])
    evaluation = daily_frame.loc[daily_frame.date.ge(pd.Timestamp("2015-01-05"))]
    summary = {"at": parent.original.now(), "registration": "TECH.R221", "decision": "TECH.R222",
        "status": "ALL_INDUSTRY_STRUCTURE_DESCRIPTION_COMPLETED_NOT_FINANCIAL",
        "daily_rows": len(daily_frame), "evaluation_rows": len(evaluation), "known_evaluation_rows": int(evaluation.view_allowed.sum()),
        "daily_state_counts": evaluation.descriptive_structure_state.value_counts().to_dict(),
        "industry_daily_rows": len(industries), "stage_events": len(events), "event_anchor_states": pd.DataFrame(events).anchor_status.value_counts().to_dict(),
        "event_profile_rows": len(profiles), "event_profile_state_counts": profile_frame.descriptive_fixed_state.value_counts().to_dict(),
        "fixed_industry_profile_rows": len(fixed_industries), "identity_rows": len(identities), "case_rows": len(case_frame),
        "key_rows": len(key_frame), "prefix_checks": prefix, "source_files_unchanged": len(protocol["files"]),
        "necessary_tests_passed": 4, "figures": figures, "new_accounts": 0, "new_fits": 0,
        "new_predictive_targets": 0, "new_market_requests": 0, "financial_metrics": "NOT_COMPUTED",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    print(f"R222行业结构完成：{len(daily_frame)}槽/{len(events)}事件/{len(profiles)}观察、240案例/17截断；0新账户。", flush=True)


def identity_rows(panel, anchor, members, anchor_id):
    assignment = {str(panel.symbols[i]): (group.key, group.name, group.leading) for group in anchor.groups for i in group.members}
    records = []
    for row in members.itertuples():
        fixed = assignment.get(row.symbol)
        records.append({"anchor_id": anchor_id, "anchor_date": panel.dates[anchor.anchor_idx], "symbol": row.symbol,
            "source_industry_key": row.industry_key, "source_classification_known": row.row_source_known,
            "fixed_industry_key": fixed[0] if fixed else None, "fixed_major_name": fixed[1] if fixed else None,
            "fixed_leading": fixed[2] if fixed else None, "anchor_status": anchor.status, "source_row_origin": row.row_origin,
            "source_pdf_sha256": row.pdf_sha256})
    return records


def draw_cases(frame):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 9})
    figures = []
    for case_id, values in frame.groupby("original_episode_id", sort=True):
        d = values.sort_values("date")
        fig, axes = plt.subplots(5, 1, figsize=(12.6, 11.4), sharex=True)
        axes[0].plot(d.date, d.etf_close / d.etf_close.iloc[0], color="#414955", label="ETF原价 / 案例首日")
        axes[1].plot(d.date, 100 * d.leader_mean_relative20, color="#267868", label="逐点领先行业过去20日相对ETF / %")
        axes[1].plot(d.date, 100 * d.leader_fixed_excess_over_etf, color="#b67732", label="案例固定领先行业后锚相对ETF / %")
        axes[1].axhline(0, color="#999999", linewidth=.8)
        axes[2].plot(d.date, d.industry_positive5_fraction, color="#725a8b", label="保留行业过去5日上涨比例")
        axes[2].plot(d.date, d.rotation_churn5blocks, color="#b67732", label="两个5日块平均排名变化 / 行业数减1")
        axes[2].set_ylim(-.03, 1.03)
        axes[3].plot(d.date, d.daily_hist, color="#267868", label="当日日线MACD柱")
        axes[3].plot(d.date, d.weekly_hist, color="#ba7836", label="上一完整周MACD柱")
        axes[3].axhline(0, color="#999999", linewidth=.8)
        axes[4].plot(d.date, d.eligible20_member_count / 300, color="#267868", label="分类与20日历史同时可知 / 300")
        axes[4].plot(d.date, d.retained_member_count / 300, color="#725a8b", label="保留行业代表的成员 / 300")
        axes[4].axhline(.98, color="#999999", linewidth=.8, linestyle="--")
        axes[4].set_ylim(-.03, 1.03)
        for axis in axes:
            axis.legend(loc="best", fontsize=8)
            axis.grid(axis="y", alpha=.22)
        axes[0].set_ylabel("ETF价格相对值")
        axes[1].set_ylabel("不同区间相对强度")
        axes[2].set_ylabel("结构描述")
        axes[3].set_ylabel("原MACD柱")
        axes[4].set_ylabel("成员比例")
        unknown = int((~d.daily_view_allowed).sum())
        fig.suptitle(f"原案例{int(case_id)}：行业结构与量价顺序\n行业统计滞后一完整日，源龄逐点保留；{unknown}/60日逐点结构未知；仅作开发解释", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        path = OUT / "figures" / f"原案例{int(case_id)}_行业强弱扩散与轮动.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(relative(path))
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="行业结构与固定行业传播的一次描述。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
