"""合并一份事前发行公告，保存基础记录及有效版本，并登记T13来源进度。"""
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.factor96_rights_event_chains_v1 import OUT, read, save, digest, now
from research.factor96_rights_event_review_v1 import Document, base_fact, state_at, date_literal
from scripts.record_factor96_remaining_changes_v1 import update


def complete():
    assert not (OUT / "program_update_receipt.json").exists()
    original_files = [OUT/name for name in ("result.json", "reviewed_document_facts.json", "event_chains.json", "publication_boundary_queries.json")]
    original_hashes = {p.name: digest(p) for p in original_files}
    supplement = OUT / "tianqi_notice_supplement_v1"
    source = read(supplement / "document.json")
    doc = Document(source)
    fact = base_fact(doc, "2019_1849")
    fact["updates"] = {"record_date": "2019-12-17", "payment_start": "2019-12-18", "payment_end": "2019-12-24",
                       "issue_price_cents": 875, "plan_max_shares": 342596383,
                       "plan_max_subscription_cents_calculated": 342596383*875}
    for key in ("record_date", "payment_start", "payment_end"):
        fact["evidence"][key] = doc.literal(date_literal(fact["updates"][key]))
    fact["evidence"]["issue_price_cents"] = doc.literal("配股价格为8.75元/股")
    fact["evidence"]["plan_max_shares"] = doc.literal("可配售股份总数为342,596,383股")
    fact["evidence"]["approval_key"] = doc.literal("证监许可[2019]1849号")
    fact["notes"].append("来自已存目录的标题筛选漏项；独立补取该原件，不以结果文档反推更早可得性。目录日末仍为日期代理。")
    save(supplement / "reviewed_fact.json", fact)
    facts = read(OUT / "reviewed_document_facts.json") + [fact]
    groups = defaultdict(list)
    for row in facts:
        if row["event_id"]:
            groups[row["event_id"]].append(row)
    events, boundaries = [], []
    for event_id, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: (x["known_at"], x["document_id"]))
        for date in sorted({r["known_at"] for r in rows}):
            previous_time = (datetime.fromisoformat(date)-timedelta(seconds=1)).isoformat()
            previous, current = state_at(rows, previous_time), state_at(rows, date)
            assert not set(r["document_id"] for r in rows if r["known_at"] == date) & set(previous["source_documents"])
            assert all(r["known_at"] <= date for r in current["field_sources"].values())
            boundaries.append({"event_id": event_id, "boundary": date, "before": previous, "at": current})
        dates = [r["known_at"] for r in rows]
        final = state_at(rows, dates[-1])
        notice_ids = [r["document_id"] for r in rows if r["role"] in ("ISSUANCE_NOTICE", "REVISED_ISSUANCE_NOTICE")]
        assert notice_ids and final["actual_subscribed_shares"] <= final["plan_max_shares"]
        assert final["actual_subscribed_shares"] == final["listing_announced_total_shares"]
        events.append({"event_id": event_id, "symbol": rows[0]["symbol"], "approval_key": rows[0]["approval_key"],
                       "document_ids": [r["document_id"] for r in rows], "issuance_notice_ids": notice_ids,
                       "initial_issuance_notice_present_in_scope": True, "first_source_known_at_in_scope": dates[0],
                       "latest_source_known_at_in_scope": dates[-1], "latest_reviewed_state": final,
                       "full_intermediate_version_coverage": False, "free_float_denominator": None,
                       "strict_M06_pressure": None, "trading_feature_admitted": False})
    assert len(facts) == 127 and len(events) == 36 and len(boundaries) == 112
    tianqi = groups["002466.SZ_A_RIGHTS_2019_1849"]
    tianqi_checks = []
    for at in ("2019-12-12T23:59:59+08:00", "2019-12-13T23:59:58+08:00", "2019-12-13T23:59:59+08:00",
               "2019-12-25T23:59:59+08:00", "2019-12-26T23:59:59+08:00", "2020-01-02T23:59:59+08:00"):
        tianqi_checks.append(state_at(tianqi, at))
    assert tianqi_checks[0]["status"] == tianqi_checks[1]["status"] == "NO_VIEW"
    assert tianqi_checks[2]["payment_end"] == "2019-12-24" and tianqi_checks[2]["actual_subscription_cents"] is None
    assert tianqi_checks[3]["actual_subscribed_shares"] is None
    assert tianqi_checks[4]["actual_subscribed_shares"] == 335111438
    assert tianqi_checks[4]["announced_listing_date"] is None
    assert tianqi_checks[5]["announced_listing_date"] == "2020-01-03"
    save(supplement / "publication_queries.json", tianqi_checks)
    effective = OUT / "effective_v1_0_1"
    save(effective / "reviewed_document_facts.json", facts)
    save(effective / "event_chains.json", events)
    save(effective / "publication_boundary_queries.json", boundaries)
    result = read(OUT / "result.json")
    result.update(at=now(), status="FIXED_SCOPE_36_A_RIGHTS_EVENTS_WITH_ISSUANCE_RESULT_LISTING_REVIEWED",
                  reviewed_documents=127, original_local_documents_reused=126,
                  newly_acquired_source_documents=1, events_with_initial_notice_in_scope=36,
                  issuance_calendar_documents=37, publication_boundary_pairs=112,
                  publication_boundary_queries=224, new_network_requests=1,
                  missing_initial_notice_events_in_scope=[], effective_version="V1.0.1_SOURCE_SUPPLEMENT",
                  base_result_preserved=original_hashes,
                  scope_note="基础126文档中天齐2019缺原件的说明仅对应基础范围；有效版本增加1207160457，保存后的日期代理查询改为从2019-12-13日末已知事前条款。")
    save(effective / "result.json", result)
    flat = [{"发行身份": e["event_id"], "证券代码": e["symbol"], "首份已存来源时钟": e["first_source_known_at_in_scope"],
             "登记日": e["latest_reviewed_state"]["record_date"], "主要缴款开始": e["latest_reviewed_state"]["payment_start"],
             "主要缴款结束": e["latest_reviewed_state"]["payment_end"],
             "额外沪股通缴款日": "|".join(e["latest_reviewed_state"].get("connect_payment_dates", [])),
             "计划最大股数": e["latest_reviewed_state"]["plan_max_shares"],
             "发行价格元": str(Decimal(e["latest_reviewed_state"]["issue_price_cents"])/100),
             "最终已披露认购股数": e["latest_reviewed_state"]["actual_subscribed_shares"],
             "最终已披露认购金额元": str(Decimal(e["latest_reviewed_state"]["actual_subscription_cents"])/100),
             "公告安排上市日": e["latest_reviewed_state"]["announced_listing_date"],
             "严格M06压力": "NOT_COMPUTED", "交易特征准入": False, "文档ID": "|".join(e["document_ids"])} for e in events]
    with (effective/"36组配股发行与日历.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    visual = [
        ("1203573137_p2.png", "特变电工比例由1.5627改为1.56269；股数内部分类改变38股，总计划数量未变。"),
        ("1204527139_p2.png", "广汇能源境内认购结果及沪股通3月29日、4月3日、4月4日缴款安排。"),
        ("1204597968_p2.png", "广汇能源最终1515678586股、3864980394.30元；比首次境内结果增加577480股。"),
        ("1205995906_p3.png", "金风科技上市表格确实误写002202.SH，同页上市地点为深圳证券交易所；原文冲突保留。"),
    ]
    save(OUT / "visual_review_receipt.json", {"at": now(), "review_scope": "四份原PDF各一页已实际查看；不宣称全部PDF逐页目检。",
         "pages": [{"path": str(OUT/"page_previews"/name), "sha256": digest(OUT/"page_previews"/name), "finding": finding} for name, finding in visual],
         "renderer_note": "Poppler报告部分备用字体缺失，生成页面关键中文及数值可读；已按图片核对。"})
    assert original_hashes == {p.name: digest(p) for p in original_files}
    save(effective / "verification_receipt.json", {"at": now(), "status": "PASS_SOURCE_SCOPE_AND_PUBLICATION_BOUNDARIES",
         "base_files_unchanged": original_hashes, "effective_facts": 127, "events": 36,
         "known_at_boundary_pairs": 112, "tianqi_specific_queries": 6,
         "calculated_phenomenon": "文档事实与日期代理版本，不是交易特征有效性或历史首版可得性。",
         "implementation_sha256": digest(Path(__file__)), "new_accounts": 0})
    record_program(result)
    print(json.dumps({"status": result["status"], "documents": 127, "events": 36,
                      "boundary_queries": 224, "requests": 1, "T13": "NOT_RUN", "new_accounts": 0}, ensure_ascii=False))


def record_program(result):
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program/name for name in ("status.json", "strategy_progress.json", "factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in paths]
    status, strategies, factors, mandate = objects
    assert mandate["delivery_package_required"] is False and mandate["orders_authorized"] is False
    protected_keys = ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
                      "cumulative_invalid_implementation_account_scenarios", "last_completed_account_experiment",
                      "last_completed_account_result", "independent_forward_observations", "completed_total_fixed_questions",
                      "qualified_candidates", "orders_authorized")
    protected = {key: status[key] for key in protected_keys}
    before = [{"path": str(p), "sha256": digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix()
    effective = relative + "/effective_v1_0_1"
    note = ("固定126份本地配股文件及1份标题漏项原件，已还原36组A股发行身份、36组事前登记及主要缴款日历、38认购金额版本和36上市安排。"
            "保留兴业更正、特变内部分类、西部代码更正、广汇分段沪股通窗口和待批终止。224次公开边界查询通过；"
            "全体中间版本、其他发行方式、上市股份限售拆分、自由流通与历史首版可得性仍未齐，T13绩效NOT_RUN。")
    for row in strategies:
        if row["id"] == "T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE", current_evidence=note,
                       source_gate_path=effective+"/result.json", current_evidence_path=effective+"/result.json")
    for row in factors:
        if row["id"] == "M06":
            row.update(current_status="36_RIGHTS_EVENT_CHAINS_DATE_PROXY_ONLY_NOT_RUN", current_note=note,
                       current_evidence_path=effective+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=result["study_id"], latest_result=effective+"/result.json",
                  latest_progress_receipt=effective+"/verification_receipt.json", last_source_result=note,
                  latest_continuation_classification="PROGRESS_36_RIGHTS_ISSUANCE_IDENTITIES_AND_CALENDAR_CHAINS",
                  current_research_phase="T13_36_RIGHTS_CALENDARS_REVIEWED_FULL_COVERAGE_AND_FREE_FLOAT_PENDING",
                  new_source_documents_this_round=1, new_searchable_text_documents_this_round=1,
                  reused_rights_pdf_documents_this_round=126, new_archived_source_http_responses_this_round=1,
                  new_reviewed_rights_document_facts_this_round=127, source_field_candidates_this_round=1981,
                  source_field_candidate_kind="126固定文档1981定位片段，经条款复核形成127文件事实（含1补充原件）、36发行链；无新账户。",
                  admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                  new_rights_event_chains_this_round=36, completed_calendar_examples_this_round=36,
                  latest_rights_event_chains=effective+"/event_chains.json",
                  latest_rights_document_facts=effective+"/reviewed_document_facts.json",
                  latest_rights_publication_queries=effective+"/publication_boundary_queries.json",
                  rights_events_with_issuance_result_listing_in_scope=36, rights_initial_notice_gaps_in_scope=0,
                  rights_local_original_pdfs_including_supplement=492,
                  next_independent_source_action="T13_REMAINING_RIGHTS_INTERMEDIATE_VERSIONS_AND_UNSELECTED_TITLE_COVERAGE",
                  goal_status="active", goal_achieved=False, delivery_package_required=False)
    for key, value in protected.items():
        assert status[key] == value
    mandate.update(current_round=result["study_id"], current_protocol=relative+"/protocol.json",
                   latest_progress_receipt=effective+"/verification_receipt.json", last_research_result=note,
                   last_source_result=note, latest_continuation_report=relative+"/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    paragraphs = [
        "只操作510300、成本后净夏普达到1.2的目标仍未实现。本轮完成T13配股供给时序的一项来源工作，没有新增回测，没有制作交付包。合格候选仍为0，独立前向观测仍为0。",
        "从已保存的491份配股相关原文中，在读正文前固定了126份发行、结果、上市、更正、终止及超额配股文件。已逐项核对36组A股配股发行身份，并将发行时可知的登记日、主要缴款日期、价格、最大股数，与以后才可知的认购结果和上市安排分开。有效版本共有127份文件事实、38个认购金额版本、36组上市安排、112组公开边界前后查询（224次）。",
        "多份公告可能属于同次发行，也有同公司不同年度发行。核准文号和A股对象作为身份锚点，保留2017与2019天齐、2017与2020浪潮、2020与2021东吴、2015与2022兴业为不同事件。金风、招商、中信、浙商文件中的H股核准文号没有混入A股发行。国元上市报告中的历史非公开发行核准号也单独排除。",
        "广汇能源2018-03-28披露境内认购1515101106股、3863507820.30元，同时公布沪股通投资者3月29日、4月3日及4月4日的额外缴款安排，并预计4月10日复牌。2018-04-10报告增至1515678586股、3864980394.30元。新增577480股、1472574.00元来自同次发行的境外认购，不能把两个结果当成两次发行，也不能把原境内缴款结束当成全部窗口结束。",
        "兴业2016-01-08更正认购结果仅自该公告代理时钟起生效，先前仍为原值；计划结果发布日期的2015/2016年份冲突继续保留为空。特变2017-05-27把每10股配1.5627更正为1.56269，无限售和限售计划内部各调整38股，总计划505983018股未变。西部更正的是提示公告中086273为082673，原发行公告的代码本已正确，缴款窗口未变。",
        "10份终止或到期文件单独保留。其中三聚环保和国信证券仍需股东大会审议，国投电力撤回申请尚需监管同意；独董意见不是另一笔决策。3份H股IPO超额配股权文件不属于A股配股。中科三环2020年的更正涉及发行条件说明所引监管比例，原文明示不影响配股方案，没有据此改动2022年发行条款。",
        "天齐2019年原件起初不在126份范围内。在旧目录未选标题中找到1207160457《关于配股发行的公告》，发现旧筛选漏掉了标题中的‘的’字。按唯一目标补取1份PDF后，核实目录日2019-12-13、证监许可[2019]1849号、登记日12月17日、缴款12月18日至24日、价格8.75元、最多342596383股。有效版本从12月13日日末起显示该事前安排，12月26日前仍不显示实际认购结果。基础126文档缺口记录和结果均保留，新增有效结果保存在effective_v1_0_1。",
        "38个认购金额版本均与对应股数乘发行价格精确一致，36个上市公告的新增总股数与最后已披露认购总数相符。已实际查看4个关键PDF页面。金风上市文件把代码写作002202.SH，但同页上市地点为深圳证券交易所，此来源冲突显式保留；没有据此改变目录中的002202.SZ身份。",
        "这些是历史来源的日期代理回放，目录日末并不能证明当年第一版文件确实已经可得。尚未核实全部提示公告、说明书及其他中间文件的变更覆盖，也未补齐其他发行方式及每次上市的限售拆分。配股新增总股数不等于当天全部可卖，更不等于实际卖出。严格M06供给压力仍为NOT_COMPUTED，T13保持NOT_RUN。自由流通分母来源仍受已有过期凭据影响，本轮没有重试该接口。",
        "有效研究账户552、无效实现152、累计执行704保持原值，本轮没有新增账户、收益或模型。下一项可独立处理的本地来源工作是其余配股中间文档与未选标题覆盖，避免遗漏第二个类似‘发行的公告’的标题情形。本轮结果不支持夏普达标或实盘结论。",
        "直接查看effective_v1_0_1/36组配股发行与日历.csv。详细证据在该目录reviewed_document_facts.json，分时点查询在publication_boundary_queries.json，研究代码为research/factor96_rights_event_chains_v1.py、research/factor96_rights_event_review_v1.py及两份收尾脚本。",
    ]
    (OUT/"研究进展.md").write_text("\n\n".join(paragraphs)+"\n", encoding="utf-8")
    for p, obj in zip(paths, objects):
        update(p, obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json", {"at": now(), "before": before,
         "after": [{"path": str(p), "sha256": digest(p)} for p in paths],
         "protected_counts_and_authority": protected, "new_accounts": 0,
         "goal_achieved": False, "delivery_package_created": False})


if __name__ == "__main__":
    complete()
