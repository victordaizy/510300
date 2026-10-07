"""保存实际回购事实及指数目录覆盖结果，继续保持高夏普研究目标。"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_fact_ledger_v1 as facts
import research.corporate_repurchase_index_catalogue_v1 as catalogue
import research.corporate_repurchase_public_completion_v1 as source
import research.corporate_repurchase_source_probe_v1 as probe

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
COVERAGE = catalogue.OUT / "transport_completion"


def run():
    result = read(facts.OUT / "result.json")
    coverage = read(COVERAGE / "result.json")
    manifest = read(source.OUT / "results/resolved_documents.json")
    assert len(manifest) == result["source_documents"] == 57
    assert all(row["status"] == "PDF_TEXT_SAVED" for row in manifest)
    assert result["new_accounts"] == result["new_fits"] == coverage["new_accounts"] == coverage["new_fits"] == 0
    assert digest(facts.OUT / "code/corporate_repurchase_fact_ledger_v1.py") == read(facts.OUT / "protocol.json")["code_sha256"]
    frame = pd.read_parquet(facts.OUT / "results/facts.parquet")
    actual = frame[frame.cumulative_cents.notna()].copy()
    assert (pd.to_datetime(actual.economic_cutoff) <= pd.to_datetime(actual.known_at).dt.tz_localize(None)).all()
    summaries = {row["scheme"]: row for row in result["scheme_summaries"]}
    expected_cents = {"600519_20240921": 599998596695, "600519_20251106": 299993374957,
        "300750_20231031": 271071390710, "300750_20250407": 438550468790,
        "300750_20260725": 19997882746}
    assert all(summaries[key]["actual_cumulative_cents"] == amount for key, amount in expected_cents.items())
    assert all(row["closed_scheme_future_capacity_cents"] == 0 for row in summaries.values() if row["closed"])
    weights = pd.read_parquet(COVERAGE / "results/reference_weights.parquet")
    selected = weights[weights.con_code.isin(["600519.SH", "300750.SZ"])].groupby("trade_date").weight.sum()
    daily = pd.read_parquet(facts.OUT / "results/known_disclosure_daily.parquet")
    assert daily.current_day_actual_buy_cents.isna().all()
    status = "ISSUER_DEMAND_FACTS_AND_INDEX_SOURCE_COVERAGE_COMPLETED_STRATEGY_PENDING"
    unresolved = read(COVERAGE / "results/company_status.json")
    unresolved = [{"symbol": row["symbol"], "status": row["status"]} for row in unresolved if not row["complete"]]
    links = {r["document_id"]: r["source_url"] for r in manifest}
    catl = summaries["300750_20250407"]
    recent = summaries["300750_20260725"]
    report = f"""本轮已经把“宏观支持会不会真正变成指数内的股票买盘”落实为新的来源和可运行事实账本。完成贵州茅台、宁德时代57份发行人原文的核实，并将公告目录范围扩展到研究期间历史成分并集354家公司。尚未找到满足压力成本后夏普1.2、年化10%、最大回撤10%的合格策略，总目标继续active。此次没有新增策略账户、预测拟合或独立前向观察，不把数据工程结果算成投资收益证据。

这项信息与已有研究的区别。

此前已研究宏观政策、资金利率、融资卖压、ETF份额、成分广度、日内隔夜和盘中价格结构。这次新增的是发行人自己的实际二级市场买入需求：政策工具提供资金可能性，公司方案给出有条件的意图，实际回购才是已经发生的购买。三个层次分别保存，不能相加。

具体的待检验机制是：在价格与宏观背景相近时，已经披露的实际需求增强，能否帮助区分“暂时卖压得到承接”和“仍缺少承接”的下跌。已有实际买入不保证未来继续买，也不保证价格上涨；回购本身可能是公司对下跌的反应。必须比较公告可知之后的结果，不能拿回购发生当月的上涨反过来证明可交易。

两家公司事实账本发现了什么。

57份PDF得到42个信息事件，涉及5个独立方案：6份计划披露、36次执行披露，另15份持股清单、限制性股票注销、债权人通知和价格上限调整不增加实际买入金额。5个方案中4个已完成，1个仍在实施期。36次执行披露中有12次累计额完全没有增加，另有1份明确尚未开始。

如果把每份执行公告中的累计额直接相加，会得到{result['naive_sum_of_cumulative_cny']/1e8:.6f}亿元；逐方案只保留最后累计额，合计为{result['sum_latest_scheme_cumulative_cny']/1e8:.6f}亿元。错误相加会放大约{result['naive_overcount_multiple']:.4f}倍。这个最终累计合计仍含期前基数，不是研究期间新增资金。期前无法分解的27.107139亿元保留为旧基数，能够由同方案相邻披露确认的新增累计变化合计为{result['sum_known_disclosed_increment_cny']/1e8:.6f}亿元，也没有将它平均摊到每日。

宁德时代2025年方案计划40—80亿元，最终实际{catl['actual_cumulative_cents']/1e10:.6f}亿元。到2026-04-07期限届满并宣布完成时，距离上限仍有{catl['unused_cap_cents']/1e10:.6f}亿元。这个未使用上限已经不再是该方案未来的可用需求，账本将完成后的容量记为0。[发行人完成公告]({links['1225082511']})。

宁德时代2026年新方案为200—400亿元，但2026-09-03公告明确截至8月31日尚未实施；9月11日才披露首次实际买入约{recent['actual_cumulative_cents']/1e10:.6f}亿元。正文中提到专项贷款等可能来源，也不能证明贷款已经支用。[尚未实施的进展公告]({links['1225545968']})、[首次实际回购公告]({links['1225561269']})。

茅台2024年方案从首次可观察计划到首次执行相隔103个日历日；2025年方案相隔55日。方案发布后立即把额度算成买盘，会把尚未发生的需求提前使用。宁德时代两份期内新方案相应为18日和48日。这些时间是样本事实，不作为新调优阈值。

来源时间与模型输入。

每份记录区分经济截止日、签章日、目录日期和可匹配的交易所日期。研究可用时点取这些披露日期的较晚者至当日末，最早下一交易日判断；禁止按回购截止日倒填信息。历史目录与原文不能认证当时第一次送达的准确时刻，所以这仍是保守历史重建，不是独立实时观察。

每日状态文件有{result['daily_state_rows']}行，覆盖两家公司{result['daily_state_rows']//2}个交易日；这不是{result['daily_state_rows']}次独立买入。每天实际买入额全部保持未知，只记录到当时为止已经披露的历史。金额使用整数分，同方案才计算增量，新方案不延续旧方案累计数。

指数覆盖已经推进到哪里。

两家公司由研究起点之前的2024-08-30权重前两名选定，合计8.175%；在之后已保存24个月度快照中，其合计权重范围为{selected.min():.3f}%—{selected.max():.3f}%。不能用两家公司替代整个沪深300。因此本轮进一步收集2024-09-01至2026-09-24期间、24份历史快照全部成分的并集354家公司公告目录，原两家公司目录直接复用。

实际取得{coverage['complete_companies']}/354家公司的完整标题查询、共{coverage['documents']}条目录，其中{coverage['candidate_execution_documents']}条被标题规则列为执行披露候选，涉及{coverage['candidate_execution_companies']}家公司。{coverage['companies_without_title_matches']}家公司没有标题命中，这只表示此检索没有返回公告，不证明实际买入为0。未完整取得的公司数量为{len(unresolved)}，其缺失状态保留。标题命中也不等于已经读过原文，更不等于新增资金。

按各公告日期之前最近保存的月度快照参考，{coverage['outside_prior_snapshot_documents']}份公告所属公司当时不在该快照中，不能因为它后来成为成分股，就把此前公告提前并入指数信号。{coverage['after_last_saved_snapshot_documents']}份公告日期晚于最新已保存的2026-07-31权重快照，后续身份覆盖需要补充；本轮没有把旧快照当作经过认证的当日正式名单。

接下来进入策略检验还缺什么。

第一，完整原文事实要扩展到目录内的公司，包括不同回购方案、A/H股份、用途与重复累计的处理，目录数量不能替代这一工作。第二，实际需求需要用当时可得的规模或成交额归一化，不能直接把名义金额增长解释为流动性改善。第三，只把一项预先固定的实际需求增量加入已有价格和宏观基准，用最近两日历年、每日更新的同一训练池比较完整账户，并单独观察失败尾部。只有成本后增量、账户三项目标与之后的新资料验证都成立，才有理由称为候选高夏普策略。

这次没有按未来收益挑公司、筛方案、修改已失败策略或增设有利阈值。事实处理通过金额单位、重复累计、方案隔离、未知期前基数、限制性注销排除、未来披露隔离和不伪造日买入七项实现检查。检查用于保证输入含义，不能证明交易优势。没有制作审核包或用户表格，当前市场视图仍为NO_VIEW，没有生成实盘交易指令。
"""
    report_path = facts.OUT / "本轮研究结论.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    completed = {"at": now(), "study_id": facts.STUDY, "catalogue_study_id": catalogue.STUDY,
        "status": status, "fact_result_sha256": digest(facts.OUT / "result.json"),
        "catalogue_result_sha256": digest(COVERAGE / "result.json"), "report_sha256": digest(report_path),
        "resolved_catalogue_result": (COVERAGE / "result.json").relative_to(ROOT).as_posix(),
        "issuer_originals": 57, "fact_events": 42, "execution_disclosures": 36,
        "unchanged_cumulative_disclosures": 12, "schemes": 5,
        "catalogue_companies": coverage["companies"], "complete_catalogue_companies": coverage["complete_companies"],
        "catalogue_documents": coverage["documents"], "candidate_execution_documents": coverage["candidate_execution_documents"],
        "unresolved_catalogue_companies": unresolved,
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0,
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(facts.OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_NEW_ISSUER_ACTUAL_DEMAND_FACTS_AND_INDEX_CATALOGUE",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=facts.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_corporate_repurchase_round=completed,
        remaining_research_question="将实际回购事实扩展到历史指数成分并确定可用时点，再检验它在宏观与价格基准上的账户增量；当前来源和测量改进尚不构成高夏普策略。",
        goal_metadata_note="本轮取得57份发行人原文，完成5方案实际需求事实，并扩展历史成分并集公告目录，属于实质新增证据；目标未完成。")
    for study in [probe.STUDY, source.STUDY, facts.STUDY, catalogue.STUDY]:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=facts.STUDY, latest_integrated_experiment=facts.STUDY,
        current_protocol=(facts.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(facts.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=status,
        last_research_result=f"新增57份发行人原文、42个事实事件与354家公司历史并集目录；完整目录{coverage['complete_companies']}家。没有新策略收益结果，高夏普总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    marker = "<!-- CORPORATE_REPURCHASE_FACTS_20260926 -->"
    if marker not in text:
        status_path.write_text(f"{marker}\n\n> 2026-09-26 实际回购需求事实完成：57份发行人原文、5个方案；历史成分并集354家公司目录完整取得{coverage['complete_companies']}家。新账户0，目标ACTIVE且未完成；见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + text, encoding="utf-8")
    print("实际回购事实、指数目录覆盖、中文结论和主线进度已保存，高夏普目标继续active。", flush=True)


if __name__ == "__main__":
    run()
