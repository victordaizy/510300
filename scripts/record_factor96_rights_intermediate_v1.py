"""复算本轮已保存来源记录并登记研究进度，保护既有账户结果。"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.factor96_rights_intermediate_v1 import OUT, read, save
from research.factor96_rights_event_chains_v1 import digest, normalized, now
from research.factor96_rights_event_review_v1 import state_at
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    for row in read(OUT / "freeze.json")["files"]:
        assert digest(row["path"]) == row["sha256"], row["path"]
    facts = read(OUT / "reminder_and_administrative_facts.json")
    anchors = 0
    for fact in facts:
        assert digest(fact["raw_path"]) == fact["raw_sha256"]
        assert digest(fact["text_path"]) == fact["text_sha256"]
        pages = {p["page"]: normalized(p["text"]) for p in read(fact["text_path"])["pages"]}
        for field, items in fact["evidence"].items():
            for hit in items:
                assert hit["document_id"] == fact["document_id"]
                assert pages[hit["page"]][hit["normalized_start"]:hit["normalized_end"]] == hit["literal"]
                anchors += 1
        assert set(fact["updates"]) <= set(fact["evidence"])
        assert fact["trading_feature_admitted"] is False
    groups = defaultdict(list)
    combined = read(OUT / "combined_source_facts.json")
    assert len({r["document_id"] for r in combined}) == 316
    for fact in combined:
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    queries = 0
    for pair in read(OUT / "publication_boundary_queries.json"):
        for key in ("before", "at"):
            saved = pair[key]
            assert state_at(groups[pair["event_id"]], saved["as_of"]) == saved
            assert all(s["known_at"] <= saved["as_of"] for s in saved["field_sources"].values())
            assert saved["strict_M06_pressure"] is None and not saved["trading_feature_admitted"]
            queries += 1
    assert queries == 586
    result = read(OUT / "result.json")
    regular = [f for f in facts if f["role"] == "PAYMENT_REMINDER"]
    assert len(regular) == 179 and all(f["calendar_fields_reconciled"] for f in regular)
    assert sum(len(f.get("rejected_table_candidates", [])) for f in regular) == 5
    assert result["new_accounts"] == result["new_returns"] == result["new_models"] == 0
    assert read(OUT / "supplement/result.json")["source_requests"] + read(OUT / "ambiguous_title_supplement/result.json")["requests"] == 19
    save(OUT / "saved_recomputation_receipt.json", {"at": now(), "status": "PASS_SAVED_SOURCE_FIELDS_AND_DATE_PROXY_REPLAY",
         "new_facts_verified": len(facts), "raw_and_text_hash_pairs": len(facts),
         "literal_anchors_verified": anchors, "saved_publication_queries_recomputed": queries,
         "frozen_inputs_unchanged": True, "result_sha256": digest(OUT / "result.json"),
         "new_accounts": 0, "scope": "源文件、摘录和保存的时点回放可复算；不证明历史首版、全覆盖、策略有效或外部复核。"})


def record():
    assert not (OUT / "program_update_receipt.json").exists(), "本轮进度已经登记。"
    verify_saved()
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / name for name in ("status.json", "strategy_progress.json", "factor_progress.json")]
    paths.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in paths]
    status, strategies, factors, mandate = objects
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys = ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
            "cumulative_invalid_implementation_account_scenarios", "last_completed_account_experiment",
            "last_completed_account_result", "independent_forward_observations", "completed_total_fixed_questions",
            "qualified_candidates", "orders_authorized")
    protected = {key: status[key] for key in keys}
    before = [{"path": str(p), "sha256": digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix()
    note = ("剩余365配股文档候选定位及717旧未选标题重分类完成，补取19定向原件。183提示标题中179份常规提示四字段与当时已知主公告一致，"
            "3份沪股通提示保留离散缴款日及授权发生日/披露时钟，1份为筹划复牌；另6份模糊发行标题均属筹划审核。"
            "保留西部首次提示内部代码冲突及次日更正，排除5个跨表行登记日误提取。华安少1份计划提示原件，195份非提示索引文档仍仅候选。"
            "586次保存回放可复算；历史首版、全部发行方式/中间版本、限售拆分及自由流通分母未齐，T13仍NOT_RUN。")
    for row in strategies:
        if row["id"] == "T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE", current_evidence=note,
                       source_gate_path=relative + "/result.json", current_evidence_path=relative + "/result.json")
    for row in factors:
        if row["id"] == "M06":
            row.update(current_status="36_RIGHTS_EVENTS_REMINDER_FIELDS_RECONCILED_DATE_PROXY_ONLY_NOT_RUN", current_note=note,
                       current_evidence_path=relative + "/result.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round="510300_FACTOR96_RIGHTS_INTERMEDIATE_V1", latest_result=relative + "/result.json",
                  latest_progress_receipt=relative + "/saved_recomputation_receipt.json", last_source_result=note,
                  latest_continuation_classification="PROGRESS_RIGHTS_REMINDER_RECONCILIATION_WITH_EXPLICIT_COVERAGE_GAP",
                  current_research_phase="T13_REMINDER_FIELDS_RECONCILED_NONREMINDER_VERSIONS_AND_FREE_FLOAT_PENDING",
                  new_source_documents_this_round=19, new_searchable_text_documents_this_round=19,
                  reused_rights_pdf_documents_this_round=365, new_archived_source_http_responses_this_round=19,
                  new_reviewed_rights_document_facts_this_round=189, source_field_candidates_this_round=5831,
                  source_field_candidate_kind="378文件机器定位5831片段；183提示标题分类、179四字段对照、3沪股通与7行政短文；195文件仍仅候选，未宣称完整人工阅读。",
                  admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                  new_rights_event_chains_this_round=0, completed_calendar_examples_this_round=0,
                  latest_rights_event_chains=relative + "/event_chains_with_reminders.json",
                  latest_rights_document_facts=relative + "/combined_source_facts.json",
                  latest_rights_publication_queries=relative + "/publication_boundary_queries.json",
                  rights_events_with_issuance_result_listing_in_scope=36, rights_initial_notice_gaps_in_scope=0,
                  rights_planned_reminder_original_gaps_in_scope=1, rights_local_original_pdfs_including_supplement=511,
                  next_independent_source_action="T13_LOCAL_195_NONREMINDER_VERSION_AND_PROSPECTUS_SCOPE",
                  goal_status="active", goal_achieved=False, delivery_package_required=False)
    for key, value in protected.items():
        assert status[key] == value
    mandate.update(current_round=status["latest_round"], current_protocol=relative + "/protocol.json",
                   latest_progress_receipt=relative + "/saved_recomputation_receipt.json", last_research_result=note,
                   last_source_result=note, latest_continuation_report=relative + "/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    paragraphs = [
        "只操作510300、成本后完整账户净夏普达到1.2的目标仍未实现。本轮补充T13所需配股公告时序，没有新增账户、收益或模型，没有制作交付包。合格候选与独立前向观测仍均为0。",
        "在正文处理前固定剩余365份本地文档和717条旧未选标题，对378份文档21765页进行机器定位，得到5831段字段候选。补取13份明确实施/提示标题及6份模糊‘发行股票相关事项’原件，19次PDF请求均完成。未把机器扫描称为逐页人工阅读。",
        "183份提示类标题分为179份常规缴款提示、3份广汇沪股通提示、1份新奥筹划期复牌。179份常规提示中的登记日、主要缴款起止、价格和最大股数，经标签定位及异常处理，均与该文件目录代理时钟前已知的发行条款一致。广汇5份文档出现的3月20日登记日候选来自表格行串接，正文和原PDF表格均明确登记日为3月19日；错误候选完整保留。长春的4月19—22日加4月25日分段列示不被截断为22日结束。",
        "西部证券2017-03-30首次提示1203224562第1页写082673、第3页写086273，真实来源内部冲突现已保存，并与次日1203236200更正连接。3月30日时点保持冲突，3月31日公告代理时点后解除；既有3月27日发行公告的正确代码和缴款日历未被改写。宁波1211684963的核准文号跨第2、3页，已实际查看两页后确认2021_2718身份。共查看4张原PDF页面图片，未宣称全部页面目检。",
        "广汇3份沪股通提示分别于2018-03-29、04-03和04-04披露，保留3月29日、4月3日、4月4日三个离散缴款日。3月29日文件报告3月28日取得香港监管授权及完成登记，发生日和本次可得代理时钟分开。三份仍只披露境内1515101106股、3863507820.30元；不能提前使用4月10日才披露的最终认购总量。",
        "新奥2017-04-18复牌来自筹划期董事会方案，股东大会及监管批准仍待取得，不属于2018年已实施配股结束。另6份模糊标题均为筹划或监管阶段：太平洋预算120亿元降至45亿元、会议取消，广汇审核延期及重新排期，东北每10股配2股及基于2015-06-30股本的数量测算。计划预算、测算股数和会议取消均未填为实际发行、实际认购或终止事件。",
        "华安证券说明书1210115278第27页安排5次提示，但现有原件只覆盖2021-06-02、06-04、06-07、06-08，6月3日原件未找到。6条定向网页查询及一次搜狐页面打开未定位新原件，继续保留1份计划提示原件缺口；不能声称当日没有发布。此项计划次数证据仅局部查看说明书，不算全文完成。",
        "本轮形成189份字段/行政记录，合并上轮127份后共316份来源记录，36组发行身份保持不变。293组时钟边界、586次保存回放已复算；189份新增事实的原PDF/文本哈希及摘录位置也已核对。来源可复算不代表已证明历史首版可得、全覆盖或策略有效。",
        "剩余195份非提示文档还只有候选定位，涵盖说明书、方案调整及行政进展等。全体中间版本、其他发行方式、上市限售拆分与自由流通分母仍未齐；严格M06压力仍为NOT_COMPUTED，T13仍为NOT_RUN。自由流通数据接口的过期凭据本轮没有重试。下一步可独立处理这195份已存文件的当前/历史引用及方案变化，不重复已经完成的提示字段对照。",
        "研究账户统计仍为有效552、无效实现152、累计执行704；9个固定研究问题与0个合格候选不变。账户结果没有因这次补数而改变。可直接查看36组配股提示覆盖.csv、result.json及reminder_and_administrative_facts.json；无需交付包。",
    ]
    (OUT / "研究进展.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    save(OUT / "huaan_web_lookup_note.json", {"at": now(), "record_type": "TOOL_OBSERVATION_NOTE_NOT_RAW_HTTP_ARCHIVE",
         "queries": ["site.sse.com.cn 600909 2021-06-03 配股 提示性公告", "site.cninfo.com.cn 华安证券 配股提示性公告 2021年6月3日",
                     '"华安证券股份有限公司配股提示性公告" "2021年6月3日"', '"600909" "2021-06-03" "配股提示"',
                     '"华安证券" "2021-053" 配股', 'site.sse.com.cn "华安证券" "2021-06-03" 提示性'],
         "opened_url": "https://q.stock.sohu.com/cn/600909/bw_17.shtml", "opened_url_result": "CACHE_MISS",
         "result": "未找到6月3日的新原件；已存6月4日原件及说明书仅作为重合结果，未增加来源记录。",
         "raw_http_response_saved": False, "no_nonpublication_inference": True})
    for path, obj in zip(paths, objects):
        update(path, obj)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": now(), "before": before,
         "after": [{"path": str(p), "sha256": digest(p)} for p in paths], "protected_counts_and_authority": protected,
         "new_accounts": 0, "goal_achieved": False, "delivery_package_created": False})
    print(json.dumps({"本轮": status["latest_round"], "新原件": 19, "常规提示字段对照": 179,
                      "回放查询": 586, "T13": "NOT_RUN", "新账户": 0, "合格候选": 0, "交付包": False}, ensure_ascii=False))


if __name__ == "__main__":
    record()
