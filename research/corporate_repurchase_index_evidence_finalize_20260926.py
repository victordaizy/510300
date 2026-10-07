"""保存指数回购原文、金额提取和候选链结果，继续总研究目标。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_index_documents_v1 as source
import research.corporate_repurchase_execution_extractor_v1 as extract
import research.corporate_repurchase_chain_diagnostic_v1 as chain

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    documents = read(source.OUT / "result.json")
    extraction = read(extract.OUT / "result.json")
    result = read(chain.OUT / "result.json")
    originals = read(source.OUT / "documents.json")
    facts = read(extract.OUT / "results/parsed_documents.json")
    paths = [source.OUT / "result.json", extract.OUT / "result.json", chain.OUT / "result.json"]
    assert documents["complete_texts"] == documents["documents"] == len(originals) == len(facts) == 2407
    assert extraction["pilot_actual_disclosures"] == extraction["pilot_amounts_matched"] == 36
    assert extraction["additional_inspected_examples_matched"] == 8
    assert sum(result["increment_status_counts"].values()) == result["candidate_chain_rows"]
    assert all(read(path)["new_accounts"] == 0 and read(path)["new_fits"] == 0 for path in paths)
    for module in [source, extract, chain]:
        protocol = read(module.OUT / "protocol.json")
        assert digest(module.OUT / "code" / Path(module.__file__).name) == protocol["code_sha256"]
    sources = {row["document_id"]: row["source_url"] for row in originals}
    counts = extraction["status_counts"]
    increments = result["increment_status_counts"]
    precision_reconciled = sum(row.get("coarse_summary_precision_reconciled", False) for row in facts)
    status = "INDEX_REPURCHASE_ORIGINALS_AND_EXTRACTION_COMPLETED_ACCOUNT_TEST_PENDING"
    report = f"""本轮已完成从两家公司试点到指数历史成分范围的回购原文采集：2,407份选定发行人文件全部取得文本，其中2,365份为本轮新增、42份复用，涉及185家公司。金额提取得到{extraction['extracted_amounts']}条记录，进一步形成{result['candidate_chain_rows']}条候选方案连接记录、{result['candidate_schemes']}个候选方案，涉及{result['candidate_companies']}家公司。高夏普策略目标尚未实现，总目标保持active；本轮没有新增账户、模型拟合或独立前向观察。

这一轮与前一轮的实质变化。

上一轮取得的是354家历史成分并集的3,765条公告目录，本轮实际下载并读取其中全部计划或执行候选原文，不再停留在标题计数。来源窗口为2024-09-01至2026-09-24。185家是该标题范围内有计划或执行候选的公司数，不等于当期完整300家公司都有实际回购，也不证明其余公司实际买入为0。A股范围、方案身份和金额仍需在原文层面判定。

这条研究线关注的是宏观资金能否落实到股票需求。政策安排、授信或计划不自动产生二级市场购买；执行披露补充了已经发生的需求信息。例如海康威视2025-04-01公告载明，截至3月末实际累计支付约12.3548亿元，并说明资金来源包括自有资金及回购专项贷款；公告未拆分两类资金各自的实际支用额。[发行人原文]({sources['1222971674']})。这能支持“有已披露实际购买”的事实，不能单独证明它造成了后续上涨或给510300带来可交易超额收益。

已处理的几个重要口径。

一是月内累计与方案累计。公告可能先写本月累计买入，再写整个方案累计买入，两者不能互换。程序保留金额原文和上下文，只在符合定义的累计或首次执行记录中取数。

二是同一金额的披露精度。有的摘要以万元呈现，正文精确到元；本轮{precision_reconciled}份记录出现这种能够相互对应的精度差异。程序使用原文直接给出的精确值，并保存摘要的粗精度。仅披露粗单位时，“报告数未变”不等于期间绝对没有任何小额购买。

三是旧方案混入累计。例如大华股份2025年11月的一份公告明确包含2023年已回购股份，该累计口径不能直接接在2025年方案的上期数字后求差。[发行人原文]({sources['1224801667']})。这种记录保留为跨方案待核实，不把旧买入重复算成新买盘。

四是同日披露不同截止日。月末尚未开始的进展报告与随后首次回购报告可能在同一天公开。程序在同一保守可用时点内按经济截止日对齐，不用公告ID大小覆盖状态；同一截止日存在冲突仍需核实。

五是董事会与股东会。两个日期在同一句里并列时，不能把较靠近“召开”字样的股东会日期误认成董事会日期；利润分配会议也不属于回购方案批准。董事会日期和预算范围只是候选身份，仍不保证只有一个方案。

六是非成分A股二级市场需求。H股文件、子公司股权回购、美元债券回购、业绩补偿股份和单纯注销文件分别排除或保留为其他用途，不直接计入510300股票组合的新增买盘。PDF分页页码和页脚声明也不能被拼进金额数字，原始文件始终保留。

当前的提取与连接结果。

2,407份原文中，{counts.get('EXTRACTED_AMOUNT_WITH_BOARD_ANCHOR',0)}份提取到了金额及明确会议日期，{counts.get('EXTRACTED_AMOUNT_SCHEME_UNRESOLVED',0)}份虽有金额但会议身份尚未解决；另有{counts.get('NO_VIEW_AMBIGUOUS_AMOUNT',0)}份金额上下文存在歧义、{counts.get('NO_VIEW_ACTUAL_AMOUNT_NOT_IDENTIFIED',0)}份没有识别出可接受金额。计划或其他文件、排除项、跨方案累计和源文日期冲突另行保存，不填零。

在进一步具备唯一预算上下限、金额未超上限等条件的{result['candidate_chain_rows']}条候选链记录中：{increments.get('POSITIVE_REPORTED_CHANGE',0)}次报告累计额增加，{increments.get('UNCHANGED_REPORTED_CUMULATIVE',0)}次报告累计额未变，{increments.get('EXPLICIT_FIRST_REPORTED_EXECUTION',0)}次有明确首购基数，{increments.get('EXPLICIT_ZERO_BASELINE',0)}次明确尚未开始，{increments.get('UNKNOWN_STARTING_BASELINE',0)}次期初基数未知，另{increments.get('POSITIVE_WITHIN_ONE_REPORTING_UNIT',0)}次变化未超过一个披露单位。它们是信息记录，不是911次独立交易，也不是已验证的资金因子。

已提取金额但尚不能形成完整候选身份的{result['identity_or_amount_issue_rows']}条记录单独列出。部分标题或正文只给预算区间、通过议案日期或者旧方案引用，不能为了增加覆盖而猜测身份。另有{counts.get('NO_VIEW_SOURCE_ECONOMIC_DATE_BEFORE_APPROVAL',0)}份记录的经济截止日早于所识别批准日；其中美的一份完成公告正文写出的日期与其列明的批准日顺序不一致，程序保留原文并标记待核实，没有擅自改年份。[原文]({sources['1225058108']})。

来源与时点边界。

每份记录保留原始PDF、逐页文本、来源URL和请求记录。全文完整不等于每份语义都已经核实，更不等于原文覆盖了所有可能的买盘。历史目录与原文没有认证首发实时送达，研究可用时点仍以披露日期的保守日末处理，禁止回填到经济发生日。新采集2,365份原文是新的研究证据，不是2,365个独立前向收益样本。

现有成交额分母已经核对：成分日线共有484,200行，覆盖2019-12-23至2026-08-19，成交额无缺失且已转换成人民币元；其中历史市值列全部为空。因此可以继续检验基于当时成交额的已披露需求强度，不能声称已经得到可靠的回购占流通市值比例。公告之后的成交额不能用于此前判断，2026-08-19之后也不能用旧成交额冒充同日资料。

实现验证与未完成工作。

金额提取与此前逐份核实的36条试点执行披露全部一致，另8个已阅读的原文金额片段保持一致。实现检查覆盖金额单位、月内和方案累计、每股价格与预算、分页数字、并列会议日期、方案隔离、完成后变化、未知基数、同日多截止日和未来披露隔离。这些检查解决程序含义，不能估计全部文档错误率，也不属于外部投资验证。

下一步要把候选身份与原始方案引用逐一对应，处理剩余金额歧义和跨方案累计，然后固定一项已披露需求增量，与宏观和价格基准用同一最近两年训练池、每日更新方式比较完整账户。不能只检验少数回购日的后续反弹，也不能凭“企业在买”直接建立仓位。当前压力成本后夏普1.2、年化10%、最大回撤10%的目标不变。本轮未制作审核包或用户表格，当前市场视图仍为NO_VIEW。
"""
    report_path = chain.OUT / "本轮研究结论.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    completed = {"at": now(), "study_id": chain.STUDY, "source_study_id": source.STUDY,
        "extraction_study_id": extract.STUDY, "status": status,
        "result_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "report_sha256": digest(report_path), "source_documents": 2407, "new_source_documents": 2365,
        "reused_documents": 42, "source_companies": 185,
        "extracted_amounts": extraction["extracted_amounts"], "candidate_chain_rows": result["candidate_chain_rows"],
        "candidate_schemes": result["candidate_schemes"], "candidate_companies": result["candidate_companies"],
        "trading_feature_admitted_documents": 0, "new_accounts": 0, "new_fits": 0,
        "independent_forward_observations": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}
    save(chain.OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_2365_NEW_ISSUER_ORIGINALS_AND_1445_AMOUNT_RECORDS",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=chain.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_corporate_repurchase_index_round=completed,
        remaining_research_question="核实候选方案身份与累计口径后，用已知成交额归一化实际披露需求，检验宏观及价格基准上的完整账户增量；当前未形成合格高夏普策略。",
        goal_metadata_note="上一轮完成事实试点与全历史成分并集目录；本轮新增2365份发行人原文、1445条金额提取及911条候选链，属于实质进展，目标未完成。")
    for study in [source.STUDY, extract.STUDY, chain.STUDY]:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=chain.STUDY, latest_integrated_experiment=chain.STUDY,
        current_protocol=(chain.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(chain.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=status,
        last_research_result="2407份选定原文全部取得，提取1445条金额、连接911条候选记录；未新增策略账户，高夏普总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    previous = status_path.read_text(encoding="utf-8")
    marker = "<!-- CORPORATE_REPURCHASE_INDEX_ORIGINALS_20260926 -->"
    if marker not in previous:
        status_path.write_text(f"{marker}\n\n> 2026-09-26 指数范围2407份选定回购原文全部取得，1445条金额提取、911条候选方案连接记录；新增账户0，目标ACTIVE且未完成；见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    print("指数回购新证据、候选链、中文结论及主线状态已保存，高夏普目标保持active。", flush=True)


if __name__ == "__main__":
    run()
