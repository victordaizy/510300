"""读取已保存共同评分与20账户，解释全部进入路径和财富差；不生成新策略。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_all_factor_joint_path_attribution_v1"
FINANCE = ROOT / "reports/research/510300_all_factor_macro_earnings_account_v1"
PREDICTION = ROOT / "reports/research/510300_all_factor_macro_earnings_joint_v1"
BASE = ROOT / "reports/research/510300_point_second_weight_comparison_v1/inputs/controls"
PRIMARY = "TECH_MACRO_EARNINGS_JOINT"
CONTROLS = ["TECH_COMMON", "TECH_MACRO_COMMON", "TECH_EARNINGS_COMMON", "A_SAVED_WEIGHT"]
POLICIES = CONTROLS[:3] + [PRIMARY]
PERIODS = ["2015_2019", "2020_2026"]
COSTS = ["BASE", "STRESS"]
KEYS = ["entry_origin", "entry_date"]
CONTRACT = FINANCE / "next_diagnostic_contract.json"
SCORES = PREDICTION / "results/全部四模型_共同原点评分及未知.parquet"
CONTEXT = PREDICTION / "results/全部3488共同源视图_不足保留.parquet"
METRICS = FINANCE / "results/全部20账户_四场景同口径比较.parquet"
SUPPORT = ROOT / "reports/research/510300_core_support_complement_description_v1/implementation_v1_0_1/results/全部12支持接受_公布锚价格时钟与宏观原值.parquet"
CONTEXT_FIELDS = [
    "date", "decision_at", "close", "ac", "atr14", "daily_hist_atr", "weekly_hist_atr",
    "momentum5_scaled", "log_relative_volume", "up_volume_balance5", "log_rv_ratio",
    "pmi_orders_level", "pmi_orders_change", "orders_reference_period", "orders_available_at",
    "funding_gap_pp", "funding_gap_change5", "funding_available_at", "financing_net_change5",
    "financing_buy_activity", "margin_available_at", "profit_cash_joint_improve_share",
    "report_age_scaled", "recent_disclosure_share", "earnings_available_at", "fundamental_coverage",
    "source_technical_known", "source_macro_known", "source_earnings_known", "joint_source_known",
]
SCORE_FIELDS = [
    "date", "status", "candidate_quality_pass", "score", "predicted_p_times_b",
    "predicted_net_expectation", "earnings_fields_on_path", "macro_fields_on_path", "fit_index",
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def table(name, frame):
    target = OUT / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target.with_suffix(".parquet"), index=False)
    frame.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def account_folder(period, cost, policy):
    if policy == "A_SAVED_WEIGHT":
        return BASE / period / cost / policy
    return FINANCE / "results/accounts" / period / cost / policy


def load_account(period, cost, policy):
    names = ["daily", "orders", "trades"]
    if policy != "A_SAVED_WEIGHT":
        names += ["decisions", "rejections"]
    saved = {name: pd.read_parquet(account_folder(period, cost, policy) / (name + ".parquet")) for name in names}
    for frame in saved.values():
        for column in ["date", "origin", "execution_date", "entry_origin", "entry_date", "exit_date"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return saved


def account_totals(saved):
    daily, trades = saved["daily"], saved["trades"]
    require(len(daily) > 0 and daily.date.is_unique, "保存账户日历为空或重复。")
    gross = float((daily.price_pnl + daily.dividend_accrual).sum())
    commission, slippage = float(daily.commission.sum()), float(daily.slippage.sum())
    net = float(daily.equity.iloc[-1]) - 200000.
    closed = float(trades.loc[trades.status.eq("COMPLETE"), "net_pnl"].sum())
    error = net - (gross - commission - slippage)
    require(abs(error) < 1e-6, "保存账户毛费净损益没有闭合。")
    return {"gross": gross, "commission": commission, "slippage": slippage, "net": net,
            "closed_net": closed, "open_and_residual_net": net - closed, "closure_error": error,
            "completed_cycles": int(trades.status.eq("COMPLETE").sum()),
            "right_censored_cycles": int(trades.status.ne("COMPLETE").sum())}


def score_relations(prediction, context):
    require(len(prediction) == 13952 and not prediction.duplicated(["date", "policy"]).any(), "共同评分日历改变。")
    main = prediction.loc[prediction.policy.eq(PRIMARY), SCORE_FIELDS].copy()
    main = main.rename(columns={k: "joint_" + k for k in SCORE_FIELDS if k != "date"})
    output = []
    for control in CONTROLS[:3]:
        other = prediction.loc[prediction.policy.eq(control), SCORE_FIELDS].copy()
        other = other.rename(columns={k: "control_" + k for k in SCORE_FIELDS if k != "date"})
        pair = main.merge(other, on="date", how="outer", validate="one_to_one")
        require(len(pair) == 3488 and pair.date.notna().all(), "两模型评分原点没有完整匹配。")
        available = pair.joint_status.eq("AVAILABLE") & pair.control_status.eq("AVAILABLE")
        require(pair.joint_status.eq("AVAILABLE").equals(pair.control_status.eq("AVAILABLE")), "共同池可评分状态不同。")
        qj, qc = pair.joint_candidate_quality_pass.astype(bool), pair.control_candidate_quality_pass.astype(bool)
        pair["quality_relation"] = np.select(
            [~available, qj & qc, qj & ~qc, ~qj & qc],
            ["NO_VIEW", "BOTH_QUALITY", "JOINT_ONLY", "CONTROL_ONLY"], default="NEITHER")
        pair["control"] = control
        pair = pair.merge(context[CONTEXT_FIELDS], on="date", how="left", validate="one_to_one")
        output.append(pair)
    return pd.concat(output, ignore_index=True)


def disposition(quality, shares_before, reason, desired, executed, execution_known, rejection):
    """区分评分许可、持仓状态、真实请求和真实成交，不把它们等同。"""
    if executed:
        return "NEW_CYCLE_EXECUTED"
    if not quality:
        return "NO_QUALITY_ENTRY"
    if shares_before > 0:
        return "QUALITY_TRUE_ALREADY_HOLDING"
    if not execution_known:
        return "QUALITY_TRUE_NO_NEXT_SESSION"
    if reason == "MISSING_PRIOR_RISK_ESTIMATE":
        return "QUALITY_TRUE_MISSING_RISK"
    if rejection:
        return "QUALITY_TRUE_EXECUTION_REJECTED"
    if reason == "FIRST_PASSAGE_ESTIMATED_QUALITY_ENTRY" and desired > 0:
        return "QUALITY_ENTRY_REQUEST_NOT_FILLED"
    return "QUALITY_TRUE_ZERO_QUANTITY_OR_ACCOUNT_GATE"


def decision_routes(saved, score):
    decisions, orders, trades = saved["decisions"].copy(), saved["orders"], saved["trades"]
    require(decisions.origin.is_unique, "账户请求原点重复。")
    selected = score[SCORE_FIELDS].rename(columns={"date": "origin", "status": "score_status"})
    shared_fields = set(selected.columns).intersection(decisions.columns) - {"origin"}
    decisions = decisions.rename(columns={name: "request_saved_" + name for name in shared_fields})
    decisions = decisions.merge(selected, on="origin", how="left", validate="one_to_one")
    require(decisions.score_status.notna().all(), "实际请求缺少保存评分，包括期初与末日。")
    cycle_origins = set(trades.entry_origin)
    buys = orders.loc[orders.side.eq("BUY")].groupby("origin").quantity.sum() if len(orders) else pd.Series(dtype=float)
    sells = orders.loc[orders.side.eq("SELL")].groupby("origin").quantity.sum() if len(orders) else pd.Series(dtype=float)
    rejections = saved["rejections"]
    rejected = rejections.groupby("origin").reason.agg(lambda x: "|".join(sorted(set(x)))) if len(rejections) else pd.Series(dtype=str)
    decisions["actual_buy_quantity"] = decisions.origin.map(buys).fillna(0).astype(int)
    decisions["actual_sell_quantity"] = decisions.origin.map(sells).fillna(0).astype(int)
    decisions["actual_rejection_reasons"] = decisions.origin.map(rejected).fillna("")
    decisions["actual_new_cycle"] = decisions.origin.isin(cycle_origins)
    decisions["quality_entry_request"] = decisions.reason.eq("FIRST_PASSAGE_ESTIMATED_QUALITY_ENTRY") & decisions.shares_before.eq(0) & decisions.desired_shares.gt(0)
    require(decisions.entry_event.astype(bool).equals(decisions.candidate_quality_pass.astype(bool)), "账户进入许可与保存质量门不同。")
    require(not (decisions.actual_new_cycle & ~decisions.candidate_quality_pass).any(), "真实新周期未获原模型质量许可。")
    decisions["execution_disposition"] = [disposition(
        bool(row.candidate_quality_pass), row.shares_before, row.reason, row.desired_shares,
        bool(row.actual_new_cycle), pd.notna(row.execution_date), row.actual_rejection_reasons)
        for row in decisions.itertuples()]
    decisions.loc[~decisions.score_status.eq("AVAILABLE") & ~decisions.actual_new_cycle, "execution_disposition"] = "NO_VIEW_NO_NEW_ENTRY"
    return decisions


def trade_side(saved, prefix):
    trades = saved["trades"].copy()
    require(not trades.duplicated(KEYS).any(), "周期同进入原点和实际日期重复，不能强行配对。")
    if len(saved["orders"]):
        fees = saved["orders"].groupby("cycle_id")[["commission", "slippage"]].sum()
    else:
        fees = pd.DataFrame(columns=["commission", "slippage"])
    trades["actual_cycle_commission"] = trades.cycle_id.map(fees.commission).fillna(0.)
    trades["actual_cycle_slippage"] = trades.cycle_id.map(fees.slippage).fillna(0.)
    trades["closed_net_component"] = trades.net_pnl.where(trades.status.eq("COMPLETE"), 0.).astype(float)
    trades["closed_gross_component"] = np.where(trades.status.eq("COMPLETE"),
        trades.closed_net_component + trades.actual_cycle_commission + trades.actual_cycle_slippage, 0.)
    for column in ["entry_quantity", "exit_reason", "holding_sessions"]:
        if column not in trades:
            trades[column] = np.nan
    keep = KEYS + ["cycle_id", "source", "exit_date", "status", "net_pnl", "net_return", "entry_quantity",
                   "exit_reason", "holding_sessions", "actual_cycle_commission", "actual_cycle_slippage",
                   "closed_net_component", "closed_gross_component"]
    return trades[keep].rename(columns={k: prefix + k for k in keep if k not in KEYS})


def cycle_gap(primary_saved, control_saved):
    matched = trade_side(primary_saved, "joint_").merge(trade_side(control_saved, "control_"),
        on=KEYS, how="outer", validate="one_to_one", indicator=True)
    matched["cycle_relation"] = matched["_merge"].astype(str).map({"left_only": "JOINT_ONLY", "right_only": "CONTROL_ONLY", "both": "MATCHED_ENTRY"})
    matched = matched.drop(columns="_merge")
    for prefix in ["joint_", "control_"]:
        for name in ["closed_net_component", "closed_gross_component"]:
            matched[prefix + name] = matched[prefix + name].fillna(0.)
    matched["closed_net_gap_component"] = matched.joint_closed_net_component - matched.control_closed_net_component
    matched["closed_gross_gap_component"] = matched.joint_closed_gross_component - matched.control_closed_gross_component
    matched["same_actual_exit_date"] = matched.joint_exit_date.eq(matched.control_exit_date)
    matched["same_actual_entry_quantity"] = matched.joint_entry_quantity.eq(matched.control_entry_quantity)
    return matched.sort_values(KEYS).reset_index(drop=True)


def bridge(primary_saved, control_saved, cycles):
    require(pd.DatetimeIndex(primary_saved["daily"].date).equals(pd.DatetimeIndex(control_saved["daily"].date)), "比较账户日历不同。")
    main, other = account_totals(primary_saved), account_totals(control_saved)
    gross_delta = main["gross"] - other["gross"]
    commission_delta, slip_delta = main["commission"] - other["commission"], main["slippage"] - other["slippage"]
    net_delta = main["net"] - other["net"]
    new_only = float(cycles.loc[cycles.cycle_relation.eq("JOINT_ONLY"), "closed_net_gap_component"].sum())
    missing = float(cycles.loc[cycles.cycle_relation.eq("CONTROL_ONLY"), "closed_net_gap_component"].sum())
    shared = float(cycles.loc[cycles.cycle_relation.eq("MATCHED_ENTRY"), "closed_net_gap_component"].sum())
    open_delta = main["open_and_residual_net"] - other["open_and_residual_net"]
    cycle_error = net_delta - (new_only + missing + shared + open_delta)
    fee_error = net_delta - (gross_delta - commission_delta - slip_delta)
    require(abs(cycle_error) < 1e-6 and abs(fee_error) < 1e-6, "全部周期和毛费净财富差未闭合。")
    missing_complete = cycles.cycle_relation.eq("CONTROL_ONLY") & cycles.control_status.eq("COMPLETE")
    new_complete = cycles.cycle_relation.eq("JOINT_ONLY") & cycles.joint_status.eq("COMPLETE")
    return {"net_wealth_gap_cny": net_delta, "gross_pnl_gap_cny": gross_delta,
            "commission_gap_cny": commission_delta, "slippage_gap_cny": slip_delta,
            "net_fee_savings_cny": -commission_delta - slip_delta,
            "joint_only_closed_net_component": new_only, "minus_control_only_closed_net_component": missing,
            "matched_entry_closed_net_difference": shared, "open_and_residual_net_difference": open_delta,
            "missing_control_winners": int((missing_complete & cycles.control_net_pnl.gt(0)).sum()),
            "missing_control_losers": int((missing_complete & cycles.control_net_pnl.lt(0)).sum()),
            "missing_control_winner_net_cny": float(cycles.loc[missing_complete & cycles.control_net_pnl.gt(0), "control_net_pnl"].sum()),
            "missing_control_loser_net_cny": float(cycles.loc[missing_complete & cycles.control_net_pnl.lt(0), "control_net_pnl"].sum()),
            "joint_extra_winners": int((new_complete & cycles.joint_net_pnl.gt(0)).sum()),
            "joint_extra_losers": int((new_complete & cycles.joint_net_pnl.lt(0)).sum()),
            "main": main, "control_totals": other, "cycle_bridge_error_cny": cycle_error,
            "gross_fee_bridge_error_cny": fee_error, "is_single_causal_effect": False}


def attach_routes(cycles, main_routes, other_routes):
    fields = ["origin", "score_status", "candidate_quality_pass", "score", "reason", "shares_before",
              "desired_shares", "actual_new_cycle", "execution_disposition", "actual_rejection_reasons",
              "predicted_p_times_b", "predicted_net_expectation", "macro_fields_on_path", "earnings_fields_on_path"]
    out = cycles.copy()
    for prefix, routes in [("joint_origin_", main_routes), ("control_origin_", other_routes)]:
        if routes is None:
            continue
        selected = routes[fields].rename(columns={"origin": "entry_origin", **{k: prefix + k for k in fields if k != "origin"}})
        out = out.merge(selected, on="entry_origin", how="left", validate="many_to_one")
    return out


def missed_reason(row):
    if row.cycle_relation != "CONTROL_ONLY":
        return "NOT_CONTROL_ONLY"
    if row.joint_origin_score_status != "AVAILABLE":
        return "NO_VIEW_NOT_SCORED"
    if not row.joint_origin_candidate_quality_pass:
        return "JOINT_SCORE_QUALITY_FILTER"
    return row.joint_origin_execution_disposition


def flatten_bridge(value):
    return {k: v for k, v in value.items() if not isinstance(v, dict)}


def freeze():
    declared = read(CONTRACT)
    for source in declared["sources"]:
        require(sha(ROOT / source["path"]) == source["sha256"], "上一轮固定诊断来源改变。")
    paths = [CONTRACT, FINANCE / "summary.json", FINANCE / "protocol.json", METRICS, SCORES, CONTEXT, SUPPORT,
             PREDICTION / "summary.json", ROOT / "tests/test_all_factor_joint_path_attribution_v1.py"]
    for period in PERIODS:
        for cost in COSTS:
            for policy in CONTROLS + [PRIMARY]:
                names = ["daily", "orders", "trades"] + ([] if policy == "A_SAVED_WEIGHT" else ["decisions", "rejections"])
                paths += [account_folder(period, cost, policy) / (name + ".parquet") for name in names]
    sources = [{"path": p.absolute().relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in paths]
    write(OUT / "protocol.json", {
        "study_id": "510300_ALL_FACTOR_JOINT_PATH_ATTRIBUTION_V1", "registered_at": now(),
        "registration": "TECH.R257", "decision": "TECH.R258", "code_sha256": sha(__file__), "sources": sources,
        "prior_results_observed": True, "purpose": declared["purpose"], "hypothesis": "联合模型劣化可能源自评分错拒或新增许可、持仓/风险/成交路径或费用；不预定单一解释。",
        "periods": PERIODS, "costs": COSTS, "primary": PRIMARY, "all_controls": CONTROLS,
        "score_origins": 3488, "saved_accounts": 20, "all_bridges": 16,
        "cycle_match": "同进入原点与真实进入日期外连接；完整、亏损、期末未完成周期均保留；退出和数量差单列。",
        "account_identity": "净财富差=实际毛损益差-佣金差-滑点差；另=新增完整周期-遗漏完整周期+相同进入路径差+开放/残差。",
        "signal_identity": "评分质量门、真实进入请求、已有持仓、风险/现金/执行原因、实际新周期逐原点分开。",
        "news_context_scope": "Sep24公开催化仅用于已知案例解释；12支持点是正例来源，不能把其他日期标无政策。",
        "not_allowed": declared["not_allowed"], "new_fits": 0, "new_accounts": 0, "new_labels": 0,
        "financial_admission": "NOT_ADMITTED_DIAGNOSTIC_ONLY", "independent_validation": "NOT_ESTABLISHED",
        "stop_condition": declared["stop_condition"], "goal_achieved": False, "orders_authorized": False,
    }, exclusive=True)
    print("已冻结全原点、20保存账户与16完整财富差；不新增模型或账户。")


def make_report(summary, bridges, scores, routes, cycles):
    report = ["# 全因素共同评分失误：评分许可、实际账户路径与毛费净差", "",
        "这是已完成R256的诊断，没有重新拟合、重跑账户、改进入/退出参数或选择更好去组对照作为策略。全部3488评分原点、20保存账户、两时期两费用和16比较保留。",
        "", "共同源不足保持NO_VIEW；早期没有可用模型不是经济成功。完整金融结果仍为R256固定拒绝，独立完成点位为0。", "",
        "| 时期 | 成本 | 对照 | 联合净财富差/元 | 实际毛损益差/元 | 费用节省/元 | 新增周期净损益 | 负的遗漏周期净损益 | 同进入路径差 | 开放/残差差 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in bridges:
        report.append(f"| {row['period']} | {row['cost']} | {row['control']} | {row['net_wealth_gap_cny']:.2f} | {row['gross_pnl_gap_cny']:.2f} | {row['net_fee_savings_cny']:.2f} | {row['joint_only_closed_net_component']:.2f} | {row['minus_control_only_closed_net_component']:.2f} | {row['matched_entry_closed_net_difference']:.2f} | {row['open_and_residual_net_difference']:.2f} |")
    report += ["", "毛损益是原账户真实数量的解释分解，不是零费或理想仓位可执行策略。相同进入原点仍有数量、后续减仓和退出差；表中路径差不能归为单一因果。遗漏是相对于给定对照的真实周期，也不是事后保证可获利的机会。", "",
        "## 全部评分许可与真实执行", "",
        "| 对照 | 共同可评分日期 | 双方许可 | 仅联合许可 | 仅对照许可 | 双方不许可 | 未知 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for control in CONTROLS[:3]:
        counts = scores[scores.control.eq(control)].quality_relation.value_counts()
        report.append(f"| {control} | {int(counts.sum()-counts.get('NO_VIEW',0))} | {counts.get('BOTH_QUALITY',0)} | {counts.get('JOINT_ONLY',0)} | {counts.get('CONTROL_ONLY',0)} | {counts.get('NEITHER',0)} | {counts.get('NO_VIEW',0)} |")
    report += ["", "| 近期STRESS模型 | 许可日数 | 空仓真实进入请求 | 新周期成交 | 已持仓许可 | 成交拒绝 |", "|---|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        frame = routes[routes.period.eq("2020_2026") & routes.cost.eq("STRESS") & routes.policy.eq(policy)]
        report.append(f"| {policy} | {int(frame.candidate_quality_pass.sum())} | {int(frame.quality_entry_request.sum())} | {int(frame.actual_new_cycle.sum())} | {int(frame.execution_disposition.eq('QUALITY_TRUE_ALREADY_HOLDING').sum())} | {int(frame.actual_rejection_reasons.ne('').sum())} |")
    report += ["", "## 所有遗漏和新增周期，保留赢家与输家", "",
        "| 近期STRESS对照 | 漏掉的对照盈利周期 | 漏掉的对照亏损周期 | 对照遗漏盈利/元 | 对照遗漏亏损/元 | 联合额外盈利周期 | 联合额外亏损周期 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in bridges:
        if row["period"] == "2020_2026" and row["cost"] == "STRESS":
            report.append(f"| {row['control']} | {row['missing_control_winners']} | {row['missing_control_losers']} | {row['missing_control_winner_net_cny']:.2f} | {row['missing_control_loser_net_cny']:.2f} | {row['joint_extra_winners']} | {row['joint_extra_losers']} |")
    report += ["", "## 2024年上涨段与其后亏损的当时信息", "",
        "Sep24的共同原点评分：纯价量77.638823、宏观价量9.648307、盈利价量77.638823、主三源联合0.311553。它们是未校准的开发估计分，不是可信上涨概率。联合模型用旧PMI订单、盈利/现金流广度与报告龄筛选了这两个上涨进入点。", "",
        "| 原点 | 实际进入→退出 | 对照周期状态 | 对照净损益/元 | 联合当时质量门 | 联合实际处置 |", "|---|---|---|---:|---|---|"]
    frame = cycles[cycles.period.eq("2020_2026") & cycles.cost.eq("STRESS") & cycles.control.eq("TECH_COMMON") & cycles.entry_origin.between("2024-09-24", "2024-11-26")]
    for row in frame.itertuples():
        exit_text = pd.Timestamp(row.control_exit_date).strftime("%Y-%m-%d") if pd.notna(row.control_exit_date) else "未完成"
        pnl = f"{row.control_net_pnl:.2f}" if pd.notna(row.control_net_pnl) else "未计算"
        report.append(f"| {row.entry_origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d}→{exit_text} | {row.control_status} | {pnl} | {row.joint_origin_candidate_quality_pass} | {row.joint_origin_execution_disposition} |")
    report += ["", "央行Sep24宣布降准和政策利率下调；央视记者当日上午09:08:37已有报道，早于15:05决策。原支持档案也有Sep24 11:42:50的公开信息上界。这证明当日有可观察的新增催化；不能单靠旧PMI与财报背景代替它，也不能证明此次上涨由该消息单独造成，更没有事前一致预期就不能命名为‘超预期’。",
        "来源：[央视对当日国务院新闻办公室发布会的报道](https://news.cctv.cn/2024/09/24/ARTIwQJC9xG5fTUjpGt1K4g2240924.shtml)。它是本轮解释用复核，不作为历史原始抓取回执或新模型输入。",
        "", "原12支持接受点只有正例来源，不是全样本政策日历。全因素83槽中的未绑定项保持未知，当前披露的估值/行业权重不能回填历史。",
        "", "## 研究决策与不同用途的下一步", "",
        "假设→固定共同模型劣化可能来自评分筛选、真实资金/风险/执行路径、费用或信息表达缺口。",
        "方法→原保存评分逐日期对应实际请求、成交和全部完整/期末未完成周期；全部16比较双重财富恒等式闭合，不生成新结果策略。",
        "结果→上表区分被评分挡掉的真实对照交易、新增周期、共同进入路径差和费用。诊断接受为开发解释，原完整账户拒绝保持。",
        "接受/拒绝理由→公开催化与旧统计背景是不同信息用途的研究线索；当前未覆盖全部政策日历，不能据一段上涨直接指定新闻权重，也不能调旧树深、叶子、阈值或反转失败信号。",
        "重新验证→新用途须先构建包含成功、失败、无动作与时钟不明记录的完整公开催化台账，按实际公布/可用时间和多报道共同事件去重；未知不记为0。新催化的事前预期/实际落地与价格响应、趋势延续/拥挤风险分开。随后冻结唯一用途、全部同池对照和成本后账户评价，并用真正新样本验证。",
        "当前下一步准入状态：NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN。仅保存研究线索，不新增数值配置；没有自动采集或实盘授权。",
        "", f"诊断完成：{summary['completed_at']}；新拟合/新账户/新收益标签均为0，目标未完成。"]
    (OUT / "全因素共同评分_全部失误与资金费用路径.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run():
    protocol = read(OUT / "protocol.json")
    require(protocol["code_sha256"] == sha(__file__), "诊断登记后代码改变。")
    for source in protocol["sources"]:
        require(sha(ROOT / source["path"]) == source["sha256"], "保存诊断来源改变：" + source["path"])
    write(OUT / "run_started.json", {"at": now(), "one_saved_data_run": True, "new_accounts": 0}, exclusive=True)
    try:
        prediction, context = pd.read_parquet(SCORES), pd.read_parquet(CONTEXT)
        prediction["date"] = pd.to_datetime(prediction.date).astype("datetime64[ns]")
        context["date"] = pd.to_datetime(context.date).astype("datetime64[ns]")
        relations = score_relations(prediction, context)
        table("全部3488原点_三对照许可及未知", relations)
        routes, cycles, bridges, identities = [], [], [], []
        for period in PERIODS:
            for cost in COSTS:
                accounts = {policy: load_account(period, cost, policy) for policy in CONTROLS + [PRIMARY]}
                local_routes = {}
                for policy in POLICIES:
                    selected = prediction[prediction.policy.eq(policy)]
                    path = decision_routes(accounts[policy], selected)
                    path["period"], path["cost"], path["policy"] = period, cost, policy
                    local_routes[policy] = path
                    routes.append(path)
                for policy, account in accounts.items():
                    identities.append({"period": period, "cost": cost, "policy": policy, **account_totals(account)})
                for control in CONTROLS:
                    gaps = cycle_gap(accounts[PRIMARY], accounts[control])
                    part = bridge(accounts[PRIMARY], accounts[control], gaps)
                    part.update(period=period, cost=cost, control=control)
                    bridges.append(part)
                    gaps = attach_routes(gaps, local_routes[PRIMARY], local_routes.get(control))
                    gaps["missed_cycle_route"] = [missed_reason(row) for row in gaps.itertuples()]
                    gaps["period"], gaps["cost"], gaps["control"] = period, cost, control
                    cycles.append(gaps)
                    print(f"{period}/{cost}/{control}：联合净财富差{part['net_wealth_gap_cny']:.2f}元，全部路径闭合。", flush=True)
        all_routes, all_cycles = pd.concat(routes, ignore_index=True), pd.concat(cycles, ignore_index=True)
        table("全部实际决策_质量门持仓风险与成交", all_routes)
        table("全部周期外连接_盈利亏损和未完成保留", all_cycles)
        table("全部16财富差_真实毛费净与周期", pd.DataFrame([flatten_bridge(b) for b in bridges]))
        table("全部20保存账户_双重损益核对", pd.DataFrame(identities))
        relation_counts = relations.groupby(["control", "quality_relation"], dropna=False).size().reset_index(name="origin_count")
        table("全部评分许可差_计数", relation_counts)
        route_counts = all_routes.groupby(["period", "cost", "policy", "execution_disposition"], dropna=False).size().reset_index(name="decision_count")
        table("全部真实执行处置_计数", route_counts)
        missed = all_cycles[all_cycles.cycle_relation.eq("CONTROL_ONLY")].groupby(
            ["period", "cost", "control", "missed_cycle_route", "control_status"], dropna=False).agg(
            cycles=("cycle_relation", "size"), saved_control_closed_net_cny=("control_closed_net_component", "sum")).reset_index()
        table("全部遗漏周期_评分与账户路径归因", missed)
        support = pd.read_parquet(SUPPORT)
        require(len(support) == 12, "原正例来源范围改变。")
        table("原12正例催化公布钟_不能外推无政策", support)
        require(all(sha(ROOT / source["path"]) == source["sha256"] for source in protocol["sources"]), "保存数据在诊断中改变。")
        summary = {"study_id": protocol["study_id"], "decision": "TECH.R258", "completed_at": now(),
            "status": "COMPLETED_SAVED_JOINT_ALL_PATH_ATTRIBUTION_FINANCIAL_REJECTION_RETAINED",
            "score_origins": 3488, "paired_score_rows": len(relations), "saved_accounts": 20,
            "actual_decision_rows": len(all_routes), "outer_join_cycle_rows": len(all_cycles), "full_bridges": len(bridges),
            "all_source_hashes_unchanged": True, "all_twenty_account_identities_closed": True,
            "all_sixteen_gross_fee_and_cycle_bridges_closed": True,
            "maximum_bridge_error_cny": max(abs(b[k]) for b in bridges for k in ["cycle_bridge_error_cny", "gross_fee_bridge_error_cny"]),
            "all_comparisons": bridges, "new_fits": 0, "new_accounts": 0, "new_labels": 0,
            "new_independent_completed_points": 0, "financial_admission": "NOT_ADMITTED_DIAGNOSTIC_ONLY",
            "actual_latest_financial": read(FINANCE / "summary.json")["status"],
            "fixed_configuration_rejected": True, "single_cause_not_established": True,
            "next_different_purpose": "新增公开催化/事前预期/实际落地与价量阶段分开建台账；旧宏观和财报背景不替代新事件。",
            "next_different_purpose_admission": "NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN",
            "new_numeric_configurations": 0, "goal_achieved": False, "orders_authorized": False}
        write(OUT / "summary.json", summary, exclusive=True)
        make_report(summary, bridges, relations, all_routes, all_cycles)
        write(OUT / "run_completed.json", {"at": now(), "terminal": True, "new_accounts": 0}, exclusive=True)
        print("全部3488原点、20保存账户、16毛费净与周期差已闭合；原金融拒绝保持。")
    except Exception as exc:
        write(OUT / "implementation_failure.json", {"at": now(), "terminal": True, "type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="已保存共同评分的全部真实路径归因")
    parser.add_argument("action", choices=["freeze", "run"])
    action = parser.parse_args().action
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
