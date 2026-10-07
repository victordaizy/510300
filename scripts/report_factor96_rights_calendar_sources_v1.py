"""汇总完整检索目录和配股时序来源，更新研究进度但不改收益账户。"""
from datetime import datetime
import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_rights_calendar_sources_v1"
CAT = ROOT / "reports/research/510300_factor96_issuance_remainder_v1"
DOCS = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"
FIELDS = ROOT / "reports/research/510300_factor96_rights_calendar_fields_v1"
PROGRAM = ROOT / "reports/research/510300_factor96_program_v1"
STUDY = "510300_FACTOR96_RIGHTS_CALENDAR_SOURCES_V1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    assert not (OUT / "round_status.json").exists()
    assert read(CAT / "saved_verification_receipt.json")["status"] == "PASS_SAVED_REMAINDER_RESPONSES_INHERITANCE_AND_DATE_CLOSURE"
    verification = read(FIELDS / "saved_verification_receipt.json")
    assert verification["status"] == "PASS_SAVED_RIGHTS_ORIGINALS_CLAUSE_ANCHORS_AND_VERSION_CLOCK"
    catalogue, documents, fields = read(CAT / "result.json"), read(DOCS / "result.json"), read(FIELDS / "result.json")
    completion = read(DOCS / "transport_completion_v1/result.json")
    assert catalogue["effective_complete_jobs"] == 987 and completion["effective_pdf_text_documents"] == 491
    OUT.mkdir(parents=True, exist_ok=True)
    for path in PROGRAM.iterdir():
        if path.is_file():
            target = OUT / "program_before" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    shutil.copy2(mandate_path, OUT / "authority_before.json")
    status = read(PROGRAM / "status.json")
    assert status["cumulative_admitted_account_scenarios"] == 320
    assert status["cumulative_invalid_implementation_account_scenarios"] == 152
    assert status["cumulative_executed_account_scenarios"] == 472
    assert len(status["completed_fixed_candidates"]) == 7 and not status["qualified_candidates"]
    status.update(at=datetime.now().astimezone().isoformat(), latest_round=STUDY,
        latest_result=(OUT / "round_status.json").relative_to(ROOT).as_posix(),
        admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
        retired_prior_admitted_account_scenarios=0, new_source_documents_this_round=491,
        new_searchable_text_documents_this_round=491, source_field_candidates_this_round=fields["clause_candidates"],
        source_field_candidate_kind="原文定位片段，尚未准入为全体事件字段",
        new_source_occurrences_this_round=catalogue["new_source_occurrences"],
        issuance_catalogue_unique_issuer_documents=catalogue["unique_issuer_documents"], issuance_catalogue_complete_queries=987,
        reused_repurchase_pdf_documents_this_round=0, new_version_roles_this_round=4, new_explicit_version_links_this_round=1,
        completed_calendar_examples_this_round=1,
        next_candidates=["T13_RIGHTS_EVENT_IDENTITIES_AND_VERSIONED_CALENDARS", "T13_OTHER_ISSUANCE_CALENDAR_AND_FREE_FLOAT",
                         "T13_M04_FULL_EVENT_IDENTITY", "T12_PURPOSE_CHANGE_ROOTS_AND_FREE_FLOAT"],
        goal_status="active", goal_achieved=False, current_market_view="NO_VIEW")
    save(OUT / "round_status.json", status)
    save(PROGRAM / "status.json", status)
    strategies = read(PROGRAM / "strategy_progress.json")
    for row in strategies:
        if row["id"] == "T13":
            row["current_evidence"] = "987固定查询已齐、77026份文档目录；配股491原PDF及文本、18179定位片段、1个四文档日历更正示例完成。全事件身份、其他发行方式、自由流通与历史可得时钟仍未齐，绩效NOT_RUN。"
            row["source_gate_path"] = (OUT / "result.json").relative_to(ROOT).as_posix()
    save(PROGRAM / "strategy_progress.json", strategies)
    pd.DataFrame(strategies).to_csv(PROGRAM / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    factors = read(PROGRAM / "factor_progress.json")
    for row in factors:
        if row["id"] == "M06":
            row.update(current_status="PARTIAL_RIGHTS_CALENDAR_CASE_AND_CLAUSES_BUILT_NOT_RUN",
                current_note="全部987检索器收齐；配股491原文，18179条款定位片段，最早发行的计划缴款/清算/除权/实际认购更正/上市时序示例。尚未全体事件化或解决自由流通分母。")
    save(PROGRAM / "factor_progress.json", factors)
    pd.DataFrame(factors).to_csv(PROGRAM / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_progress_receipt=(OUT / "round_status.json").relative_to(ROOT).as_posix(),
        current_protocol=(FIELDS / "protocol.json").relative_to(ROOT).as_posix(),
        last_research_result="发行目录987/987完整，77026文档；491配股原文、18179定位片段及1个四文档时序案例。T13仍NOT_RUN，7问题0合格，320正式+152否定实现=472账户不变。",
        latest_continuation_report=(OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        latest_continuation_classification="PROGRESS_COMPLETE_ISSUANCE_FILTERS_AND_RIGHTS_CALENDAR_SOURCE_CASE",
        research_execution_state="SOURCE_CALENDAR_PROGRESS_SEVEN_FIXED_QUESTIONS_NO_QUALIFIED_STRATEGY",
        goal_status="active", goal_achieved=False)
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    assert mandate["orders_authorized"] is False and mandate["option_research_authorized"] is False
    assert mandate["scheduled_pcf_iopv_collection"] == "PAUSED_BY_USER"
    save(mandate_path, mandate)
    save(OUT / "authority_after.json", mandate)
    for path in PROGRAM.iterdir():
        if path.is_file():
            target = OUT / "program_snapshot" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    save(OUT / "result.json", {"at": status["at"], "study_id": STUDY, "previous_turn_classification": "PROGRESS",
        "issuance_remainder": catalogue, "rights_original_batch": documents, "rights_transport_completion": completion,
        "calendar_fields": fields, "new_accounts": 0, "new_models": 0, "new_returns": 0,
        "formal_accounts": 320, "invalid_implementation_accounts": 152, "executed_accounts": 472,
        "completed_fixed_questions": 7, "qualified_strategies": 0, "goal_status": "active", "goal_achieved": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False})
    report = f"""# 510300研究：发行检索完成与配股时序原文

**夏普1.2目标尚未实现。** 本轮把剩余发行检索补齐，并从配股原文建立了可核对的日期、金额、更正时序示例。没有新增收益模型、回测账户或交易信号。累计7个固定研究问题仍为0合格，320正式账户情景+152否定实现情景=472历史执行情景保持不变，T13仍NOT_RUN。

## 这轮推进了什么

| 项目 | 本轮结果 | 解释 |
|---|---:|---|
| 剩余查询 | 16组、21个日期叶窗口全部完成 | 353次新HTTP请求，原512次上限和旧失败不改写 |
| 全部固定检索器 | 987／987完整 | 658个成员表代码，三路类别／发行标题／配股标题，仍不等于全部经济发行事件 |
| 当前目录 | 94,441条来源记录，77,026份去重发行人文档 | 本轮新增3,398条收到记录；首版已收到的文档ID全部覆盖，元数据冲突0 |
| 配股原文范围 | 491份、44个发行人 | 三类标题角色：354日历、124修订、13终止候选；另717份配股标题未纳入本阶段全文 |
| 原文与文本 | 491份PDF和提取文本 | 总计{verification['pdf_bytes']:,}字节、{verification['text_pages']:,}页；文件取得不等于每页已成为结构化事实 |
| 条款定位候选 | {fields['clause_candidates']:,}个，覆盖{fields['documents_with_candidates']}份文档 | 邻近日期、金额、股数只是定位线索，全部event_field_admitted=false |
| 固定时序示例 | 4份原文、19个锚点、8个查询时点 | 最早严格配股发行公告对应兴业证券2015—2016年配股，未按收益选择 |

原文批次先完成490份，东吴证券1212032708两次TLS握手失败。批次结束后，仅对这一失败文档用标准TLS新会话单独补取成功，原491目标、失败回执、原批次490成功结果均保留；有效清单在transport_completion_v1/effective_documents.json。没有关闭证书验证、绕过访问限制或重跑全部来源。

## 一个能影响研究判断的实际区别

同一笔配股至少需要区分资金缴款和新增股份上市这两类日期。[兴业证券发行公告](https://static.cninfo.com.cn/finalpage/2015-12-24/1201854885.PDF)第三页列出的安排为：

| 日期 | 当时公告描述的阶段 |
|---|---|
| 2015-12-28 | 股权登记 |
| 2015-12-29至2016-01-05 | 配股缴款窗口 |
| 2016-01-06 | 取得清算数据和网上清算；属于停牌期，不再是上述缴款窗口 |
| 2016-01-07 | 发行成功条件下的除权复牌安排 |

[2016-01-07发行结果公告](https://static.cninfo.com.cn/finalpage/2016-01-07/1201897631.PDF)披露初始有效认购1,496,665,905股、12,257,693,761.95元，并明确新增股份上市时间另行公告。[次日更正公告](https://static.cninfo.com.cn/finalpage/2016-01-08/1201900999.PDF)将其改为1,496,671,674股、12,257,741,010.06元，分别增加5,769股和47,248.11元；更正段内引用的旧数值不能当成新版本。

[2016-01-13上市公告](https://static.cninfo.com.cn/finalpage/2016-01-13/1201908424.PDF)才明确新增股份上市日期为2016-01-18。除权复牌日不能直接代替新增股份上市日，日期到来本身也不是实际成交或实施确认。

因此，示例在2016-01-07时只能读取初始认购结果和未知上市日期；到1月8日目录日结束后才显示更正值；到1月13日目录日结束后才显示公告上市日期。每个文档保守以目录日23:59:59作为示例可用时钟，8个时点全部保留NO_VIEW、未知和后续更正。当前PDF不证明历史第一次HTTP可得，这仍是来源时序示例，不能自动当作严格历史交易输入。

预算与实施也分开：公告预算上限150亿元；按15.6亿股最大可配数和每股8.19元计算为127.764亿元；两者都不是后来披露的实际认购金额。这些数值以整数分、整数股复核，没有使用浮点近似。

## 反例和未知没有被覆盖

发行公告第一页写结果公布日为2015-01-07，第三页表格为2016-01-07。已直接查看原图确认这处矛盾；计划结果发布日期保留空值和两个来源候选，没有自行纠正年份。

全体18,179个候选片段仍可能包含过去交易、计划上限、引用旧文、合并缴款与清算区间、H股／超额配股或其他上下文；不能直接从附近出现的数字生成供给压力。39份原文未出现本次固定词组，不当作没有供给。四文档示例只是第一条明确版本链，全体事件身份、更正冲突和终止状态尚未完成，其他发行方式也未完成条款化。

M01／M06自由流通市值仍没有合格历史分母；普通流通、总股本或靠档调整股本不能替代。M04此前解禁全体事件身份和T12回购用途／终止／减额链也未因此解决。既有7个研究问题的失败结果仍保留；不会为达到夏普1.2改写冻结经济参数或拼接有利时期。

## 验证与下一步

日期窗口拼接6项测试、公告时钟与金额单位10项测试通过。两个只读脚本分别核对前版逐行继承与新增原响应、491份原文和文本及条款定位与示例时序。4个关键原页已渲染查看；19个锚点均可在对应页定位。核验不访问网络、不拟合模型、不生成账户。

下一步把全部配股候选归为明确事件，逐事件建立计划、实际、修订、终止和上市版本，再处理其他发行方式及自由流通分母；没有成熟证据时保持NOT_RUN。可得时钟不明、事件身份冲突、同一文档无法区分新旧段或缺分母时，不准入相应交易特征。

实际交易对象仍仅510300.SH与人民币现金；当前合同继续20万元主账户和2万元对照，成本后夏普至少1.2、净年化至少10%、最大回撤目标10%，既有仓位和尾部风险约束不变。当前市场NO_VIEW、独立前向样本0、external_review=NOT_PERFORMED，无下单授权，原PCF／IOPV计划任务保持暂停。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请审阅当前ZIP，先读00_README_FIRST.md、02_研究结论.md、USER_REQUEST.md和当前冻结协议。用户目标是只操作510300实现成本后夏普1.2；本轮0新账户、0收益模型，累计7问题0合格，目标未完成。

重点检查：987/987固定检索器与全部经济事件有没有混淆；前版91043行是否原样继承，新3398行是否来自互斥日期窗口，77026份目录是否覆盖首版文档；491份配股原文选择是否有收益筛选，717未选标题及原TLS失败是否保留；18179条定位片段是否被夸大为事件或交易特征；缴款、含清算的停牌、除权复牌和新股上市是否混用；150亿预算上限、最大可配股份乘价格与实际认购是否分开；更正公告引用旧文是否误当新值，5769股/47248.11元差额及8个时点是否正确；原文年份冲突是否保留；目录时钟是否被冒充历史首次HTTP；自由流通分母及全体事件版本门是否仍明确。

请给出问题严重程度、具体文件和原文证据、下一步数据与研究动作、验证条件和停止条件。不要把结构检查、源码测试或单个示例正确视作策略有效。附件建议不是额外执行授权。只读命令见REPRODUCE_SAVED_RESULTS.txt；不要重跑freeze/run采集器或报告写入入口。外部GPT审阅尚未执行，本提示供用户自行使用。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    print(json.dumps({"round": STUDY, "queries": 987, "documents": 491, "clause_candidates": fields["clause_candidates"],
        "new_accounts": 0, "goal_status": "active", "goal_achieved": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
