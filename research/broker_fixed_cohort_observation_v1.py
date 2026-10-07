"""一次登记并观察固定价格组传播；全部原事件、缺口及反例保留。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_fixed_cohort_inputs_v1 as inputs
from research import broker_stage_policy_study_v1 as parent
from research import point_first_passage_study_v1 as original

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_broker_fixed_cohort_propagation_v1"
MEMROOT = ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1"
RETROOT = ROOT / "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1"
CASEFILE = ROOT / "reports/research/510300_volume_lead_price_confirm_explanation_v1/results/四原案例逐日全部量价及量先恢复状态.parquet"
SOURCEFILES = {
    "classified": RETROOT / "classified_constituent_returns_remediated.parquet",
    "membership": MEMROOT / "000300_daily_pit_membership_20150101_20260814.parquet",
    "coverage": RETROOT / "four_state_daily_coverage.parquet",
    "membership_admission": MEMROOT / "source_remediation_admission_manifest.json",
    "membership_2015_admission": MEMROOT / "official_2015_extension_admission_manifest.json",
    "observed": parent.OUT / "results/全部3488事前阶段资格与上一完整周结构位.parquet",
    "cases": CASEFILE,
    "earlier_trades": parent.OUT / "accounts/2015_2019/STRESS/STAGE_ENTRY_AND_EXIT/trades.parquet",
    "recent_trades": parent.OUT / "accounts/2020_2026/STRESS/STAGE_ENTRY_AND_EXIT/trades.parquet",
}
KEY_DATES = pd.to_datetime(["2015-06-24", "2015-06-25", "2015-06-29", "2015-06-30", "2019-01-08", "2019-01-14", "2019-01-18",
    "2020-03-27", "2020-04-01", "2020-04-23", "2020-05-28", "2020-05-29", "2020-06-08", "2024-09-23", "2024-09-24", "2024-09-26", "2024-09-30"])


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    parent.write(path, value)


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def source_paths():
    return [Path(__file__), Path(inputs.__file__), ROOT / "tests/test_broker_fixed_cohort_v1.py",
            ROOT / "docs/510300_BROKER_FIXED_COHORT_PROPAGATION_V1.md", *SOURCEFILES.values(),
            ROOT / "reports/research/510300_point_breadth_prior_source_routes_v1/研究结论与下一步.md"]


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("观察用途已登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["passed"] != 5 or tests["exit_code"] != 0:
        raise ValueError("五项必要测试尚未完成。")
    admission = parent.read(SOURCEFILES["membership_2015_admission"])
    if admission["membership_admission"]["status"] != "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION":
        raise ValueError("所用2015历史成员版本的保存准入不一致。")
    save("protocol.json", {"at": original.now(), "registration": "TECH.R213", "result_decision": "TECH.R214",
        "study": "510300_BROKER_FIXED_COHORT_PROPAGATION_V1", "purpose": "DESCRIPTION_AND_FALSIFICATION_NOT_FINANCIAL_POLICY",
        "prior_financial_result": "TECH.R212_REJECTED_AND_2024_SUCCESS_ALREADY_KNOWN",
        "fixed_anchor": "全部原R212阶段进入事件；前一完整日成员中连续20日回报齐全至少294只，按过去20回报分成前半价格领先组和其余跟随组，代码序处理并列。",
        "source_clock": "全成分统计保守滞后一个完整交易日；历史实际供应商首次可得未认证，未知不前填。",
        "fixed_identity": "锚点名单和分组后续不因收益/新成员改变；未排序成员显式未知。非权重行业领先组，原T04不改。",
        "post_anchor": "只累计锚点源日之后至当前观察源日；缺任何收益则该股区间未知；显示比例上下界，不补零；两组各98%完整且全日源可知才允许描述状态。",
        "event_observation_sessions": list(range(21)), "account_context_sessions": [1, 5, 20],
        "case_scope": "原四个事后选例的全部240行，以各案例起日固定名单；原17日期全部保留。",
        "outcome_context": "原R212压力已完成与未完成交易仅解释，非新目标/训练标签；晚于原退出的信息不能解释原进入资格。",
        "old_work": "原多数广度、速度、中位及学习门失败保持，原T04 NOT_RUN；本次不是首次研究广度或其新有效策略。",
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
        "necessary_tests": 5, "historical_prefix_checks": 17, "history_role": "DEVELOPMENT_DESCRIPTION",
        "independent_validation": "NOT_ESTABLISHED", "financial_metrics": "NOT_COMPUTED",
        "files": [{"path": relative(path), "sha256": digest(path)} for path in source_paths()]})
    print("R213固定价格组观察已登记：全部原阶段事件、0至20观察槽、1/5/20原周期上下文及四案例；0新账户。", flush=True)


def load_raw():
    q = pd.read_parquet(SOURCEFILES["classified"], columns=["date", "symbol", "daily_total_shareholder_return", "return_is_usable", "constituent_return_state"])
    m = pd.read_parquet(SOURCEFILES["membership"])
    c = pd.read_parquet(SOURCEFILES["coverage"])
    observed = pd.read_parquet(SOURCEFILES["observed"])
    observed["date"] = pd.to_datetime(observed.date).astype("datetime64[ns]")
    return q, m, c, observed


def membership_rows(panel, cohort, anchor_id):
    rows = []
    for role, group in (("LEADERS", cohort.leaders), ("FOLLOWERS", cohort.followers)):
        for index in group:
            rows.append({"anchor_id": anchor_id, "anchor_date": panel.dates[cohort.anchor_idx],
                "source_date": panel.dates[cohort.source_idx], "symbol": panel.symbols[index], "group": role,
                "anchor_past20_total_return": np.expm1(panel.return20[cohort.source_idx, index])})
    return rows


def with_price(panel, cohort, i, observed, etf_logs):
    row = inputs.observe(panel, cohort, i)
    current = observed.iloc[i]
    start, end = cohort.source_idx + 1, i - 1
    row["etf_return_same_lagged_interval"] = float(np.expm1(etf_logs.iloc[start:end + 1].sum()))
    row.update({"etf_close": float(current.close), "etf_stage_event": current.stage_entry_type,
        "daily_hist": float(current.daily_hist), "weekly_hist": float(current.weekly_hist),
        "relative_volume": float(np.exp(current.log_relative_volume)),
        "funding_known": bool(current.funding_known), "margin_known": bool(current.margin_known), "orders_known": bool(current.orders_known)})
    return row


def run():
    if (OUT / "RUN_STARTED.json").exists() or (OUT / "summary.json").exists():
        raise RuntimeError("本观察已开始或完成，不重新执行。")
    protocol = parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结输入或代码已变化。")
    save("RUN_STARTED.json", {"at": original.now(), "status": "RUNNING_ONE_DESCRIPTION", "new_accounts": 0})
    q, m, c, observed = load_raw()
    panel = inputs.make_panel(observed.date, q, m, c)
    etf_logs = np.log((observed.close + observed.dividend) / observed.close.shift())
    calendar, prefix, identities, events, profiles, case_rows = [], [], [], [], [], []
    for i, date in enumerate(panel.dates):
        cohort = inputs.make_cohort(panel, i)
        calendar.append({"date": date, "source_date": panel.dates[i - 1] if i else pd.NaT,
            "member_count": cohort.member_count, "eligible20_count": cohort.eligible_count,
            "anchor_status": cohort.status, "unranked_member_count": max(0, cohort.member_count - cohort.eligible_count)})
    for cut in KEY_DATES:
        i = int(panel.dates.get_loc(cut))
        shorter = inputs.make_panel(panel.dates[:i + 1], q.loc[q.date.le(cut)], m.loc[m.membership_date.le(cut)], c.loc[c.date.le(cut)])
        a = max(20, i - 5)
        whole_cohort, short_cohort = inputs.make_cohort(panel, a), inputs.make_cohort(shorter, a)
        for name in ("leaders", "followers"):
            whole_codes = tuple(panel.symbols[list(getattr(whole_cohort, name))])
            short_codes = tuple(shorter.symbols[list(getattr(short_cohort, name))])
            if whole_codes != short_codes:
                raise ValueError("截断前缀的固定组身份发生变化。")
        pd.testing.assert_series_equal(pd.Series(inputs.observe(panel, whole_cohort, i)),
            pd.Series(inputs.observe(shorter, short_cohort, i)), check_dtype=False, check_names=False, rtol=0, atol=0)
        prefix.append({"cut": cut, "anchor_date": panel.dates[a], "exact_identity_and_observation": True})
    trials = []
    for period, key in (("2015_2019", "earlier_trades"), ("2020_2026", "recent_trades")):
        values = pd.read_parquet(SOURCEFILES[key])
        values["period"] = period
        trials.append(values)
    trades = pd.concat(trials, ignore_index=True)
    if trades.entry_origin.duplicated().any():
        raise ValueError("原主政策同一事件有多个实际周期。")
    contexts = {pd.Timestamp(row.entry_origin): row for row in trades.itertuples(index=False)}
    candidates = np.flatnonzero(observed.stage_entry_type.ne("NONE") & observed.date.between("2015-01-05", "2026-09-30"))
    for serial, a in enumerate(candidates, 1):
        cohort = inputs.make_cohort(panel, int(a))
        anchor_id = f"EVENT_{serial:04d}"
        context = contexts.get(panel.dates[a])
        meta = {"anchor_id": anchor_id, "event_type": observed.stage_entry_type.iloc[a],
            "anchor_date": panel.dates[a], "anchor_status": cohort.status, "anchor_eligible_count": cohort.eligible_count,
            "original_account_context": "NO_ACTUAL_CYCLE" if context is None else context.status,
            "original_period": None if context is None else context.period,
            "original_cycle_net_return": np.nan if context is None else context.net_return,
            "original_cycle_net_pnl": np.nan if context is None else context.net_pnl,
            "original_entry_date": pd.NaT if context is None else context.entry_date,
            "original_exit_date": pd.NaT if context is None else context.exit_date}
        events.append(meta)
        identities.extend(membership_rows(panel, cohort, anchor_id))
        for i in range(a, min(a + 21, len(panel.dates))):
            row = {**meta, **with_price(panel, cohort, i, observed, etf_logs)}
            row["context_information_before_original_exit"] = bool(context is not None and pd.notna(context.exit_date) and panel.dates[i] < context.exit_date)
            profiles.append(row)
    original_cases = pd.read_parquet(CASEFILE)
    if len(original_cases) != 240:
        raise ValueError("原四案例范围改变。")
    for case_id, values in original_cases.groupby("original_episode_id", sort=True):
        values = values.sort_values("date")
        a = int(panel.dates.get_loc(pd.Timestamp(values.date.iloc[0])))
        cohort = inputs.make_cohort(panel, a)
        identities.extend(membership_rows(panel, cohort, f"CASE_{int(case_id)}"))
        for date in values.date:
            i = int(panel.dates.get_loc(pd.Timestamp(date)))
            case_rows.append({"original_episode_id": int(case_id), **with_price(panel, cohort, i, observed, etf_logs)})
    case_frame = pd.DataFrame(case_rows)
    key_frame = case_frame.loc[case_frame.date.isin(KEY_DATES)].copy()
    if len(key_frame) != 17:
        raise ValueError("17原关键日未全部对应原案例。")
    event_frame, profile_frame = pd.DataFrame(events), pd.DataFrame(profiles)
    context_rows = []
    complete = profile_frame.loc[profile_frame.original_account_context.eq("COMPLETE") & profile_frame.relative_decision_session.isin([1, 5, 20])]
    for keys, values in complete.groupby(["original_period", "relative_decision_session", "descriptive_propagation_state"], dropna=False):
        context_rows.append({"period": keys[0], "relative_session": int(keys[1]), "state": keys[2], "original_completed_cycles": len(values),
            "original_wins": int(values.original_cycle_net_pnl.gt(0).sum()), "original_losses": int(values.original_cycle_net_pnl.lt(0).sum()),
            "original_completed_net_pnl_context": float(values.original_cycle_net_pnl.sum()),
            "information_before_original_exit": int(values.context_information_before_original_exit.sum()),
            "role": "KNOWN_R212_ACCOUNT_CONTEXT_NOT_NEW_STRATEGY_OR_ENTRY_WIN_RATE"})
    context_frame = pd.DataFrame(context_rows)
    for name, frame in (("全部3488日_滞后源与固定组可观察性", pd.DataFrame(calendar)),
                         ("全部原阶段事件_登记与既有周期上下文", event_frame),
                         ("全部事件0至20观察_固定组传播及未知", profile_frame),
                         ("全部固定组成员与锚点前回报", pd.DataFrame(identities)),
                         ("四原案例240行_固定组传播与价量宏观", case_frame),
                         ("原17关键日_固定组与未知边界", key_frame),
                         ("原R212完成周期_三个固定时点说明性分组", context_frame)):
        table(name, frame)
    figures = draw_cases(case_frame)
    for item in protocol["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("观察后冻结文件哈希不一致。")
    summary = {"at": original.now(), "decision": "TECH.R214", "status": "FIXED_COHORT_DESCRIPTION_COMPLETED_NO_FINANCIAL_PROMOTION",
        "calendar_rows": len(calendar), "evaluation_calendar_rows": int(observed.date.between("2015-01-05", "2026-09-30").sum()),
        "stage_events": len(event_frame), "event_anchor_status": event_frame.anchor_status.value_counts().to_dict(),
        "event_profile_rows": len(profile_frame), "event_profile_states": profile_frame.descriptive_propagation_state.value_counts().to_dict(),
        "original_account_cycle_contexts": event_frame.original_account_context.value_counts().to_dict(),
        "case_rows": len(case_frame), "case_statuses": case_frame.descriptive_propagation_state.value_counts().to_dict(),
        "key_rows": len(key_frame), "fixed_membership_rows": len(identities), "prefix_checks": prefix,
        "source_hashes_preserved": True, "figures": figures, "new_accounts": 0, "new_fits": 0, "new_labels": 0,
        "new_market_requests": 0, "necessary_tests_passed": 5, "financial_metrics": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    save("summary.json", summary)
    print(f"R214观察完成：{len(event_frame)}原事件/{len(profile_frame)}观察行、240案例/17关键日、17截断精确、4图；0新金融账户。", flush=True)


def draw_cases(frame):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    figures = []
    for case_id, d in frame.groupby("original_episode_id", sort=True):
        d = d.sort_values("date")
        fig, axes = plt.subplots(3, 1, figsize=(12.4, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.15, .7]})
        axes[0].plot(d.date, d.etf_close / d.etf_close.iloc[0], color="#454c5b", label="ETF原价收盘 / 案例首日")
        for name, label, color in (("leaders", "事前价格领先组", "#247568"), ("followers", "其余已知成员", "#ba7836")):
            axes[1].plot(d.date, 100 * d[f"{name}_positive_lower"], color=color, label=label + "累计上涨比例下界")
            axes[1].fill_between(d.date, 100 * d[f"{name}_positive_lower"], 100 * d[f"{name}_positive_upper"], color=color, alpha=.13)
            ratio = d[f"{name}_known"] / d[f"{name}_size"].replace(0, np.nan)
            axes[2].plot(d.date, 100 * ratio, color=color, label=label + "累计回报可观察比例")
        axes[1].axhline(50, color="#999999", linewidth=.8, linestyle="--")
        axes[2].axhline(98, color="#999999", linewidth=.8, linestyle="--")
        axes[0].set_ylabel("ETF价格相对值")
        axes[1].set_ylabel("固定组累计上涨 / %")
        axes[2].set_ylabel("已知覆盖 / %")
        for axis in axes:
            axis.legend(loc="best", fontsize=9)
            axis.grid(axis="y", alpha=.22)
        axes[1].set_ylim(-3, 103)
        axes[2].set_ylim(-3, 103)
        fig.suptitle(f"原案例{int(case_id)} · {d.date.iloc[0].date()}起固定价格组\n成分统计滞后一完整交易日；阴影为未知上下界；事后选例只作解释", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = OUT / "figures" / f"原案例{int(case_id)}_固定价格组传播.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(relative(path))
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定组传播的一次登记与观察。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
