"""记录本轮固定失败与用户不需要交付包的偏好，保留原始严格T11未准入状态。"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_factor96_t11_date_proxy_account_v1 import OUT, clean, digest, now, read, save
import json


def update(path, value):
    temporary = path.with_name(path.name+".t11-update.tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    result = read(OUT / "result.json")
    verified = read(OUT / "saved_verification_receipt.json")
    assert verified["status"] == "PASS_SAVED_O02_CAUSAL_REFERENCES_AND_FULL_ACCOUNT_RECONCILIATION"
    assert result["status"] == "FIXED_DATE_PROXY_TARGET_FAILED"
    assert not (OUT / "program_update_receipt.json").exists()
    program = ROOT / "reports/research/510300_factor96_program_v1"
    files = [program / n for n in ("status.json", "strategy_progress.json", "factor_progress.json")]
    files.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    before = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in files]
    status, strategies, factors, mandate = [read(p) for p in files]
    assert not mandate["delivery_package_required"] and not status["delivery_package_required"]
    identity = "T11_DATE_PROXY_VARIANT"
    assert not any(v["id"] == identity for v in status["additional_fixed_research_questions"])
    relative = OUT.relative_to(ROOT).as_posix()
    note = ("T11日期代理变体120个固定账户完成；主账户20万元压力成本净夏普-0.351761，年化-0.270499%，最大回撤4.703664%，"
            "7次入场6次亏损。23个正向财务日中20个有反应参照，7个反应较小。严格原T11因first_seen及更正链仍NOT_RUN，不调阈值救援。")
    for row in strategies:
        if row["id"] == "T11":
            row.update(current_status="DATE_PROXY_VARIANT_COMPLETE_ORIGINAL_STRICT_NOT_RUN", current_evidence=note,
                       current_evidence_path=relative+"/result.json", date_proxy_variant_result_path=relative+"/result.json",
                       original_seed_status="NOT_RUN_STRICT_CLOCK_AND_CORRECTION_CHAIN_NOT_ADMITTED")
    for row in factors:
        if row["id"] in ("L02", "L04", "O02"):
            row.update(current_status="DATE_PROXY_MEASURED_COMPOSITE_T11_TARGET_FAILED", current_note=note,
                       current_evidence=note, current_evidence_path=relative+"/result.json")
    status["cumulative_admitted_account_scenarios"] += 120
    status["cumulative_executed_account_scenarios"] += 120
    status["additional_fixed_research_questions"].append({"id": identity, "status": result["status"], "result": relative+"/result.json"})
    status["completed_total_fixed_questions"] = status["original_library_completed_questions"]+len(status["additional_fixed_research_questions"])
    status.update(at=now(), latest_round=result["study_id"], latest_result=relative+"/result.json",
        admitted_account_scenarios_this_round=120, invalid_implementation_accounts_this_round=0,
        last_completed_account_experiment=result["study_id"], last_completed_account_result=relative+"/result.json",
        latest_t11_date_proxy_account_result=relative+"/result.json", latest_progress_receipt=relative+"/saved_verification_receipt.json",
        latest_continuation_classification="PROGRESS_FIXED_T11_DATE_PROXY_REJECTED",
        current_research_phase="DATE_PROXY_VARIANT_FAILED_PRESERVED_NO_PARAMETER_RESCUE", last_source_result=note,
        next_candidates=["T12_FREE_FLOAT_AND_REMAINING_EXECUTION_PURPOSE_CHAIN", "T13_EVENT_IDENTITIES_AND_FREE_FLOAT", "T04_HISTORICAL_WEIGHTS"],
        delivery_package_required=False, latest_user_delivery_instruction="不需要交付包", goal_achieved=False, goal_status="active",
        independent_forward_observations=0, orders_authorized=False, qualified_candidates=[])
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates")):
            status[key] = 0
    status["source_field_candidate_kind"] = "本轮复用已冻结价格、分红、财务与风险算法，无新增来源或原文解析"
    mandate.update(current_round=result["study_id"], current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_verification_receipt.json", last_research_result=note,
        latest_continuation_report=relative+"/研究结论.md", latest_continuation_classification="PROGRESS_FIXED_T11_DATE_PROXY_REJECTED",
        research_execution_state="DATE_PROXY_VARIANT_FAILED_PRESERVED_NO_PARAMETER_RESCUE", last_source_result=note,
        goal_status="active", goal_achieved=False, delivery_package_required=False, latest_user_delivery_instruction="不需要交付包")
    summary = pd.read_csv(OUT / "account_summary.csv")
    names = {"FULL": "财务＋较小首日反应", "FINANCIAL_ALL": "全部正向财务日", "FINANCIAL_COMMON": "共同可算范围财务对照",
             "ABOVE_MEDIAN": "较大首日反应对照", "BUY_HOLD_50": "期初半仓买入持有", "CASH": "零息现金"}
    selected = summary.loc[(summary.period == "FULL") & (summary.capital == 200000) & (summary.cost == "STRESS")]
    lines = ["本轮没有实现净夏普1.2。固定的T11日期代理主方案净夏普-0.352，年化收益-0.27%，最大回撤4.70%；20万元期末为194,840.68元，亏损5,159.32元。",
             "用户已说明不需要交付包，本轮只保留研究代码和原始账本，没有制作ZIP。",
             "检验范围2017-01-03至2026-08-14，2335个交易日全部计入。242日年化、现金收益0；压力费用为双边佣金万四、每单最低5元和单边10基点滑点，并按不利价位取整。",
             "| 固定规则 | 入场时钟 | 净夏普 | 年化收益 | 最大回撤 | 入场次数 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in selected.itertuples(index=False):
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"| {names[row.policy]} | 观察后第{row.lag}个开盘 | {sharpe} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.new_entries} |")
    lines.extend(["", "主方案23个正向财务日中20个有足够历史反应参照，7个反应较小，最终入场7次，1盈6亏；7次退出均触发观察日财富低点失效，另有2次按风险预算减仓。佣金362.42元，滑点970.30元。2335日中18日收盘持仓，平均收盘仓位0.23%。",
        "两万元主时钟压力账户净夏普-0.379、年化-0.28%；延迟一日的20万元主方案净夏普-0.423。主方案早期与主期分段同样未通过，全部120个预定账户均未满足联合点目标。",
        "预定20日循环区块、4000次配对重采样中，主方案夏普95%区间为[-1.109, 0.129]，其中3998次有定义；主方案相对共同范围财务对照的夏普差区间[-0.515, 0.624]，不能确认过滤有正增量。这没有校正全历史家族搜索。",
        "11项合成规则测试通过；保存账本已独立核对全部份额、T+1、现金、应收分红、成本和指标。风险缓存延伸149日，旧3307日与2990条训练记录完全一致，每次决策仅使用已成熟的两年标签。",
        "证据边界仍在：公告时刻大多只有名义日期，历史更正和取消链不完整；范围外旧财务字段未全量逐页复核；15个正向财务日只有一家披露公司。这是披露篮子的日期代理研究，不能称为整个指数盈利更新或严格原版T11。",
        "本固定变体登记失败，保留原定义和结果，不调整阈值、退出或窗口救援。原始严格T11仍未准入；合格候选0，独立前向观测0，夏普1.2目标保持未完成。",
        "可复算结果：account_summary.csv、annual_accounts.csv、cycles.csv、o02_first_response_measurement.parquet、accounts各账户账本；保存金额复核记录为saved_verification_receipt.json。"])
    (OUT / "研究结论.md").write_text("\n\n".join(lines[:3])+"\n\n"+"\n".join(lines[3:])+"\n", encoding="utf-8")
    for path, value in zip(files, [status, strategies, factors, mandate]):
        update(path, value)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": now(), "before": before,
        "after": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in files],
        "added_accounts": 120, "cumulative_admitted": status["cumulative_admitted_account_scenarios"],
        "cumulative_executed": status["cumulative_executed_account_scenarios"],
        "original_strict_T11_completed": False, "goal_achieved": False, "delivery_package_created": False})
    print("失败结果与120个账户已登记；目标未完成，本轮没有交付包。", flush=True)


if __name__ == "__main__":
    main()
