"""只分解R246已保存账户的剩余缺口；不重跑、拟合或产生交易规则。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import policy_expectation_thesis_exit_completion_v1_0_1 as source
from research.close_broker_stage_policy_v1 import prepared_prepend

ROOT, STATE, parent = source.ROOT, source.STATE, source.parent
OUT = ROOT / "reports/research/510300_policy_thesis_exit_gap_attribution_v1"
REGISTER, DECISION = "TECH.R247", "TECH.R248"
POLICIES = (source.POLICY, source.BASELINE, parent.SAVED_CORE)
REPORT = "完整账户剩余缺口_支持毛优势费用与CORE路径.md"
read, write, digest, rel = source.read, source.write, source.digest, source.rel


def folder(period, cost, policy):
    if policy == source.POLICY:
        return source.OUT / "accounts" / period / cost / policy
    if policy == source.BASELINE:
        return parent.OUT / "accounts" / period / cost / policy
    if policy == parent.SAVED_CORE:
        return parent.previous.parent.original.CONTROL / period / cost / policy
    raise ValueError("未知保存账户。")


def export(frame, name):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists():
        raise FileExistsError("诊断结果禁止覆盖：" + name)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def register():
    state = read(STATE)
    if state["latest_actual_financial_decision"] != "TECH.R246":
        raise ValueError("当前金融基准已变化，不能核错账户。")
    paths = [Path(__file__), source.OUT / "summary.json", source.OUT / "project_state_update_receipt.json",
             ROOT / "reports/research/510300_point_forward_calendar_check_v1/checks/20261006_085151_790969/result.json"]
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            for policy in POLICIES:
                paths.extend(folder(period, cost, policy) / name for name in ("daily.parquet", "orders.parquet", "trades.parquet", "terminal.json"))
    write(OUT / "protocol.json", {"at": source.now(), "registration": REGISTER, "decision": DECISION,
        "status": "SAVED_LEDGER_COMPLETE_GAP_ATTRIBUTION_ONLY", "hypothesis": "R246虽改善一笔退出仍逊于原A；剩余净财富差额可能来自支持本身毛优势、额外摩擦和原CORE实际路径变化。",
        "scope": "两时期两费用，R246/原共同账户/原A全部12已保存账户，所有完成及开放周期；不只抽取2023亏损。",
        "known_before_registration": "已读R246全部指标、一笔实际修改、八原路线状态及R240归因；新完整分解尚未计算，不宣称盲测。",
        "different_from_R240": "只处理R246新增退出后的完整实际数量和剩余对原A缺口；不重复原R240账户或改变终态。",
        "identities": ["每账户净财富变化=全部完成净周期+真实开放损益", "同实际数量毛损益=净损益+实际佣金+实际滑点",
                       "R246净财富减原A=支持净损益+(R246_CORE净损益-原A净损益)", "同数量毛差与实际摩擦差精确闭合"],
        "interpretation": "CORE路径差包含量、风险、时间、机会被吸收及现金反馈；不把残差单独命名资金或风险因果。关闭账户去费只用于代数分解，不能当可执行零成本策略。",
        "next_action_rule": "若近期两费用支持毛损益均非正，优先要求不同的进入前收益证据；若支持毛正但净负则检查摩擦；若支持净正而整体负则优先明确CORE路径/机会冲突。不是新策略或新通过门。",
        "no_rescue": "不调R246进入、退出、期限、风险、费用或持有窗口，不增加训练/账户/标签；不利用已有赢家形成数值新规则。",
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_http": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "frozen": [{"path": rel(p), "sha256": digest(p)} for p in paths],
        "financial_before": {k: state[k] for k in source.facts.FINANCE}, "forward_before": {k: state[k] for k in source.facts.FORWARD}})
    print("R247完整剩余缺口诊断已登记：12保存账户，0金融重跑。", flush=True)


def frozen_exact():
    protocol = read(OUT / "protocol.json")
    for row in protocol["frozen"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("保存输入改变：" + row["path"])
    state = read(STATE)
    for names, key in ((source.facts.FINANCE, "financial_before"), (source.facts.FORWARD, "forward_before")):
        if {k: state[k] for k in names} != protocol[key]:
            raise ValueError("原金融或前瞻合同变化。")
    return protocol


def saved_decomposition(period, cost, policy):
    base = folder(period, cost, policy)
    daily, orders, trades = [pd.read_parquet(base / f"{name}.parquet") for name in ("daily", "orders", "trades")]
    terminal = read(base / "terminal.json")
    if "owner" not in trades:
        if policy != parent.SAVED_CORE:
            raise ValueError("共同账户缺少真实来源归属。")
        trades = trades.copy()
        trades["owner"] = "CORE"
    if orders.cycle_id.isna().any():
        raise ValueError("订单没有所属周期，不能分配成本。")
    if not set(orders.cycle_id.astype(int)) <= set(trades.cycle_id.astype(int)):
        raise ValueError("存在无周期订单。")
    by_cycle = orders.groupby("cycle_id")[["commission", "slippage"]].sum()
    actual_open = trades.loc[trades.status.ne("COMPLETE")]
    if len(actual_open) != int(terminal["open_shares"] > 0):
        raise ValueError("开放周期与真实期末份额不一致。")
    records = []
    for row in trades.to_dict("records"):
        cid = int(row["cycle_id"])
        charges = by_cycle.loc[cid] if cid in by_cycle.index else pd.Series({"commission": 0., "slippage": 0.})
        net = float(row["net_pnl"]) if row["status"] == "COMPLETE" else float(terminal["open_pnl_cny"])
        records.append({"period": period, "cost": cost, "policy": policy, "cycle_id": cid,
            "owner": row["owner"], "source": row.get("source", "CORE_ONLY"), "entry_origin": row["entry_origin"],
            "entry_date": row["entry_date"], "exit_date": row["exit_date"], "status": row["status"],
            "net_pnl_cny": net, "commission_cny": float(charges.commission), "slippage_cny": float(charges.slippage),
            "gross_at_actual_quantities_cny": net+float(charges.commission)+float(charges.slippage),
            "initial_quantity": int(row["entry_quantity"]), "buy_debit_cny": float(row["buy_debit"]),
            "setup_source_ids": row.get("setup_source_ids", ""), "exit_reason": row.get("exit_reason", "")})
    frame = pd.DataFrame(records)
    ending_net = float(daily.equity.iloc[-1])-200000.
    ledger_gross = float((daily.price_pnl+daily.dividend_accrual).sum())
    total_commission, total_slippage = float(daily.commission.sum()), float(daily.slippage.sum())
    errors = [abs(frame.net_pnl_cny.sum()-ending_net), abs(frame.gross_at_actual_quantities_cny.sum()-ledger_gross),
              abs(frame.commission_cny.sum()-total_commission), abs(frame.slippage_cny.sum()-total_slippage)]
    if max(errors) > 1e-6:
        raise AssertionError("净财富、开放损益或真实毛费分解不闭合。")
    by_owner = frame.groupby("owner").agg(net=("net_pnl_cny", "sum"), gross=("gross_at_actual_quantities_cny", "sum"),
        commission=("commission_cny", "sum"), slippage=("slippage_cny", "sum"), cycles=("cycle_id", "size"))
    def owner(key, column):
        return float(by_owner.loc[key, column]) if key in by_owner.index else 0.
    summary = {"period": period, "cost": cost, "policy": policy, "net_wealth_change": ending_net,
        "gross_at_actual_quantities": ledger_gross, "commission": total_commission, "slippage": total_slippage,
        "core_net": owner("CORE", "net"), "core_gross": owner("CORE", "gross"), "support_net": owner("COMPLEMENT", "net"),
        "support_gross": owner("COMPLEMENT", "gross"), "support_commission": owner("COMPLEMENT", "commission"),
        "support_slippage": owner("COMPLEMENT", "slippage"), "support_cycles": int(owner("COMPLEMENT", "cycles")),
        "open_pnl_cny": float(terminal["open_pnl_cny"]), "estimated_future_exit_friction": float(terminal["estimated_future_exit_friction"]),
        "maximum_identity_error": max(errors)}
    return frame, summary


def run():
    frozen_exact()
    write(OUT / "run_started.json", {"at": source.now(), "new_financial_runs": 0})
    cycles, summaries, bridges, core_comparisons = [], [], [], []
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            local = {}
            for policy in POLICIES:
                rows, summary = saved_decomposition(period, cost, policy)
                local[policy] = (rows, summary)
                cycles.append(rows)
                summaries.append(summary)
            new, old, original = [local[policy][1] for policy in POLICIES]
            bridge = {"period": period, "cost": cost, "net_gap_vs_A": new["net_wealth_change"]-original["net_wealth_change"],
                "support_net": new["support_net"], "support_gross": new["support_gross"],
                "core_path_net_difference_vs_A": new["core_net"]-original["core_net"],
                "core_path_gross_difference_vs_A": new["core_gross"]-original["core_gross"],
                "commission_difference_vs_A": new["commission"]-original["commission"],
                "slippage_difference_vs_A": new["slippage"]-original["slippage"],
                "net_increment_vs_R240": new["net_wealth_change"]-old["net_wealth_change"],
                "support_net_change_vs_R240": new["support_net"]-old["support_net"],
                "core_net_change_vs_R240": new["core_net"]-old["core_net"]}
            first = bridge["support_net"]+bridge["core_path_net_difference_vs_A"]
            second = bridge["support_gross"]+bridge["core_path_gross_difference_vs_A"]-bridge["commission_difference_vs_A"]-bridge["slippage_difference_vs_A"]
            if max(abs(first-bridge["net_gap_vs_A"]), abs(second-bridge["net_gap_vs_A"])) > 1e-6:
                raise AssertionError("与原A的完整净财富差额不闭合。")
            bridges.append(bridge)
            for control in (source.BASELINE, parent.SAVED_CORE):
                first_core = local[source.POLICY][0].loc[lambda x: x.owner.eq("CORE")]
                second_core = local[control][0].loc[lambda x: x.owner.eq("CORE")]
                joined = first_core.merge(second_core, on="entry_origin", how="outer", suffixes=("_new", "_control"), indicator=True, validate="one_to_one")
                joined["period"], joined["cost"], joined["control"] = period, cost, control
                core_comparisons.append(joined)
            print(f"{period}/{cost}保存账本精确分解，原A净财富差{bridge['net_gap_vs_A']:+.2f}元。", flush=True)
    export(pd.concat(cycles, ignore_index=True), "全部12账户所有周期及开放_真实毛费来源")
    export(pd.DataFrame(summaries), "全部12账户来源净财富与成本闭合")
    export(pd.DataFrame(bridges), "全部四场景剩余缺口_真实数量毛费与CORE路径")
    export(pd.concat(core_comparisons, ignore_index=True), "全部原CORE进入外连接_不把残差当单一因果")
    recent = [x for x in bridges if x["period"] == "2020_2026"]
    if all(x["support_gross"] <= 0 for x in recent):
        next_action = "SUPPORT_ENTRY_RETURN_INFORMATION_REQUIRED_NOT_FEE_OR_EXIT_PARAMETER_RESCUE"
        reason = "近期两费用支持本身在实际数量下的毛损益均非正；降低费用不能建立进入优势，继续改退出也缺独立理由。下一优先不同、进入前可知的收益证据。"
    elif all(x["support_net"] <= 0 for x in recent):
        next_action, reason = "ACTUAL_FRICTION_DIAGNOSIS_REQUIRED", "先检查真实摩擦占比；不将去费代数当策略。"
    else:
        next_action, reason = "CORE_PATH_AND_OPPORTUNITY_CONFLICT_INFORMATION_REQUIRED", "支持净额与全账户差分别披露，优先核真实CORE路径变化，不能只相加独立周期。"
    summary = {"at": source.now(), "registration": REGISTER, "decision": DECISION,
        "status": "COMPLETED_R246_SAVED_ACCOUNT_GAP_ATTRIBUTION_NOT_NEW_STRATEGY", "saved_accounts": 12,
        "account_identity_checks": 12, "exact_gap_bridges": 4, "all_four_bridges": bridges,
        "next_action": next_action, "reason": reason, "new_accounts": 0, "new_fits": 0,
        "new_labels": 0, "new_http": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "no_counterfactual_account": True, "R246_financial_terminal_unchanged": True,
        "maximum_account_identity_error": max(x["maximum_identity_error"] for x in summaries)}
    write(OUT / "summary.json", summary)
    text = ["# 完整账户剩余缺口：支持毛优势、费用与CORE路径", "", reason,
            "", "这是R246新退出后的全部保存账本诊断，0新账户/训练/标签/网络。原R246冻结终态保持，目标未完成。",
            "", "| 时期 | 费用 | 相对原A净财富差 | 支持净额 | 支持同实际数量毛额 | CORE路径净差 | 相对原A佣金差 | 相对原A滑点差 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in bridges:
        text.append(f"| {row['period']} | {row['cost']} | {row['net_gap_vs_A']:+.2f} | {row['support_net']:+.2f} | {row['support_gross']:+.2f} | {row['core_path_net_difference_vs_A']:+.2f} | {row['commission_difference_vs_A']:+.2f} | {row['slippage_difference_vs_A']:+.2f} |")
    text.extend(["", "来源净差精确恒等式：净财富差=支持净损益+CORE路径净差。毛费精确恒等式：净财富差=支持同数量毛损益+CORE路径毛差−佣金差−滑点差。开放损益包括，期末没有人为清仓。",
                 "", "CORE路径差包含实际数量、风险约束、时间、机会吸收与现金反馈，不能将其单独解释为资金占用因果。全部原CORE进入外连接保留，原周期的独立大赢家不能和共同账户小亏损拼接。",
                 "", "同数量毛额是账本代数分解，不是零费用账户、可执行反事实或提高头寸的授权。各原历史周期已被看过；费用复本不是独立样本，不将此诊断变成筛选赢家的新门槛。",
                 "", "原A、R240、R246及既有分红/贷款利率/HMM/回购策略终态各自保持；当前独立新点位0、已准入待跑新金融0。新完整用途只能基于不同信息/机制、真实实现错误或真正新样本另登记。"])
    (OUT / REPORT).write_text("\n".join(text)+"\n", encoding="utf-8")
    print("全部12保存账本与四完整缺口桥已闭合，下一优先事项由真实毛费来源决定。", flush=True)


def close():
    protocol = frozen_exact()
    summary = read(OUT / "summary.json")
    write(OUT / "close_started.json", {"at": source.now(), "decision": DECISION})
    before = STATE.read_bytes()
    state = read(STATE)
    recent = next(x for x in summary["all_four_bridges"] if x["period"] == "2020_2026" and x["cost"] == "STRESS")
    text = f"""### TECH.R247—R248：R246完整剩余缺口归因（2026-10-06）

假设→新退出后仍逊于原A，须区分支持毛优势、实际摩擦与原CORE路径，才能决定下一研究方向。
方法→冻结R246/R240/原A两时期两费用12保存账户，全完成/开放周期，12毛费净身份与4净差桥闭合；原CORE真实进入外连接，不重跑账户、不构造零费用/理想仓位策略。
结果→近期STRESS相对原A净财富差{recent['net_gap_vs_A']:+.2f}元，其中支持净损益{recent['support_net']:+.2f}、CORE路径净差{recent['core_path_net_difference_vs_A']:+.2f}；支持同实际数量毛损益{recent['support_gross']:+.2f}元。两时期/两费用和全部周期均保存。
接受/拒绝→接受精确来源毛费分解，下一优先{summary['next_action']}：{summary['reason']}拒绝将路径残差当单一现金/风险因果，拒绝把去费代数当有效策略或按已知盈亏改进入/退出。
再验证→0新账户/拟合/标签/采集；最新实际金融仍R246固定拒绝，收益/Sharpe目标、独立和去过拟合未完成。当前待跑新金融0、受阻计数0，本轮实质诊断PROGRESS；原E03 13项与R244份额观察合同保持。真正新进入前信息/新样本需另完整用途；不重跑分红、贷款源/HMM/回购旧终态。
依据：[全部剩余缺口](../{rel(OUT/REPORT)})、[固定诊断](../{rel(OUT/'protocol.json')})、[真实四场景分解](../{rel(OUT/'summary.json')})。"""
    prepared = [(ROOT / "docs" / name, *prepared_prepend(ROOT / "docs" / name, text)) for name in
                ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")]
    state.update({"updated_at": source.now(), "previous_goal_turn_classification": state["goal_turn_classification"],
        "goal_turn_classification": "PROGRESS_R247_R248_COMPLETE_SAVED_ACCOUNT_GAP_ATTRIBUTION",
        "latest_completed_study": rel(OUT), "latest_result": rel(OUT / "summary.json"), "latest_report": rel(OUT / REPORT),
        "latest_research_status": summary["status"], "latest_technical_decision": DECISION,
        "latest_progress": "R246全部12保存账户毛费净与四场景原A缺口闭合，明确进入优势和CORE路径的剩余问题。",
        "current_phase": "COMPLETE_SAVED_ACCOUNT_GAP_ATTRIBUTION_FINISHED_NO_NEW_FINANCIAL_ADMISSION",
        "current_study": "510300_POLICY_THESIS_EXIT_GAP_ATTRIBUTION_V1", "current_phase_trial_accounting": summary,
        "current_goal_turn_actual_work": {"saved_account_decompositions": 12, "exact_gap_bridges": 4, "new_accounts": 0, "new_fits": 0, "new_network": 0},
        "new_accounts_in_current_phase": 0, "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "goal_achieved": False,
        "next_research_question": summary["reason"], "latest_gap_attribution": rel(OUT / "summary.json")})
    with (OUT / "state_before_TECH_R248.json").open("xb") as stream:
        stream.write(before)
    documents = []
    for path, old, new, body in prepared:
        if path.read_bytes() != old:
            raise ValueError("准备后长期事实变化。")
        with (OUT / f"{path.stem}_before_TECH_R248.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if not path.read_bytes().endswith(body):
            raise AssertionError("原正文没有保留。")
        documents.append(rel(path))
    STATE.write_text(__import__('json').dumps(source.normalize(state), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    saved = read(STATE)
    for names, key in ((source.facts.FINANCE, "financial_before"), (source.facts.FORWARD, "forward_before")):
        if {k: saved[k] for k in names} != protocol[key]:
            raise AssertionError("诊断改变了实际金融或原前瞻。")
    write(OUT / "project_state_update_receipt.json", {"at": source.now(), "decision": DECISION, "documents": documents,
        "actual_state_updates": 1, "finance_five_preserved_exact": True, "forward_thirteen_preserved_exact": True,
        "new_accounts": 0, "goal_status": "active", "goal_achieved": False})
    print("R247—R248长期事实一次更新；R246金融5字段和E03前瞻13字段精确保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="R246完整剩余缺口来源分解")
    parser.add_argument("action", choices=("register", "run", "close"))
    {"register": register, "run": run, "close": close}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
