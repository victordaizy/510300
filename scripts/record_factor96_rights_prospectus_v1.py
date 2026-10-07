"""复算说明书字段及新增持有期来源，并登记本轮研究进度。"""
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.factor96_rights_prospectus_v1 import OUT, PREVIOUS, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import state_at
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    anchors = 0
    for row in read(OUT / "reviewed_current_section_comparisons.json"):
        assert digest(row["raw_path"]) == row["raw_sha256"]
        assert digest(row["text_path"]) == row["text_sha256"]
        pages = {p["page"]: normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for group in (row["field_candidates"], row["source_scope_evidence"]):
            for hits in group.values():
                for hit in hits:
                    assert hit["document_id"] == row["document_id"]
                    assert pages[hit["page"]][hit["start"]:hit["end"]] == hit["literal"]
                    anchors += 1
        assert not any(v in ("DIFFERS_REQUIRE_REVIEW", "MULTIPLE_VALUES_REQUIRE_REVIEW") for v in row["comparisons"].values())
        assert not row["trading_feature_admitted"] and not row["quantity_admitted_as_final_issuance"]
    current = read(OUT / "combined_source_facts_with_restriction_addendum.json")
    original = {r["document_id"]: r for r in read(PREVIOUS / "combined_source_facts.json")}
    assert len(current) == len({r["document_id"] for r in current}) == 318
    for fact in current:
        rid = fact["document_id"]
        if rid in original and rid != "1202299350":
            assert fact == original[rid]
        if rid not in ("1202174218", "1202174219", "1202299350"):
            continue
        assert digest(fact["raw_path"]) == fact["raw_sha256"] and digest(fact["text_path"]) == fact["text_sha256"]
        pages = {p["page"]: normalized(p["text"]) for p in read(fact["text_path"])["pages"]}
        for hits in fact["evidence"].values():
            for hit in hits:
                assert pages[hit["page"]][hit["normalized_start"]:hit["normalized_end"]] == hit["literal"]
                anchors += 1
        if rid in ("1202174218", "1202174219"):
            assert "长春高新超达投资有限公司" in "".join(pages.values())
    groups = defaultdict(list)
    for fact in current:
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    boundaries = []
    for event_id, members in sorted(groups.items()):
        for at in sorted({f["known_at"] for f in members}):
            before = (datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
            a,b = state_at(members,before),state_at(members,at)
            new_ids={f["document_id"] for f in members if f["known_at"]==at}
            assert not new_ids & set(a["source_documents"])
            assert all(x["known_at"]<=at for x in b["field_sources"].values())
            boundaries.append({"event_id":event_id,"boundary":at,"before":a,"at":b})
    assert len(groups)==36 and len(boundaries)==293
    save(OUT / "publication_boundary_queries_with_restriction_addendum.json",boundaries)
    for pair in read(OUT / "publication_boundary_queries_with_restriction_addendum.json"):
        for key in ("before","at"):
            saved=pair[key]
            assert state_at(groups[pair["event_id"]],saved["as_of"])==saved
    for saved in read(OUT / "changchun_restriction_publication_queries.json"):
        assert state_at(groups["000661.SZ_A_RIGHTS_2016_460"],saved["as_of"])==saved
    original_ids=set(original)
    assert {f["document_id"] for f in current}-original_ids=={"1202174218","1202174219"}
    for hit in read(OUT / "holding_restriction_clause_reviews.json"):
        fact=next(f for f in current if f["document_id"]==hit["document_id"]) if hit["document_id"] in {"1202174218","1202174219"} else None
        if fact:
            page=next(p for p in read(fact["text_path"])["pages"] if p["page"]==hit["page"])
            assert normalized(page["text"])[hit["start"]:hit["end"]]==hit["literal"]
    images=["1207348214_p24.png","1207356246_p25.png","1202174218_p19.png","1205988163_p21.png",
            "1201906629_p22.png","1201906629_p23.png","1205988163_p22.png","1202174218_p20.png","1202299350_p3.png"]
    save(OUT / "visual_review_receipt.json",{"at":now(),"scope":"已实际查看9个关键页面；不宣称584页范围或所有文档逐页人工阅读。",
        "pages":[{"path":str(OUT/"page_previews"/name),"sha256":digest(OUT/"page_previews"/name)} for name in images]})
    save(OUT / "saved_recomputation_receipt.json",{"at":now(),"status":"PASS_SAVED_FIELD_SCOPE_AND_RESTRICTION_DATE_REPLAY",
        "prospectus_raw_text_hash_pairs":74,"literal_anchors_verified":anchors,
        "prior_canonical_facts_unchanged":315,"prior_listing_fact_extended":1,"new_commitment_source_facts":2,
        "saved_global_publication_queries_recomputed":586,"specific_changchun_queries_recomputed":6,
        "result_sha256":digest(OUT/"result_v1_0_1.json"),"new_accounts":0,
        "interpretation":"复算支持来源与时点记录，不证明完整可卖量、历史首版或策略有效。"})


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    verify_saved()
    program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/name for name in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths]
    status,strategies,factors,mandate=objects
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios",
          "cumulative_invalid_implementation_account_scenarios","last_completed_account_experiment",
          "last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
          "qualified_candidates","orders_authorized")
    protected={k:status[k] for k in keys}
    before=[{"path":str(p),"sha256":digest(p)} for p in paths]
    relative=OUT.relative_to(ROOT).as_posix()
    note=("195份非提示文件固定后，完成74份说明书当前发行章节字段对照。全部与主公告同目录日期；66份明确日期、6份仅月日、2份仅T日日程，67份数值价格一致。"
          "数量分成50份同主公告测算、10份A/H合计拆分、8份股本基准日测算、4份回购专户处置情景、2份仅比例。"
          "18段持有期条款已读；长春上市公告新增登记无限售38772000股、登记限售13695股，超达实际认配8778110股，原6个月自愿锁定承诺另存，未计算即时可卖量。"
          "仍有121份方案/审核文件及全体实际上市限售、承诺变更、其他发行方式、自由流通分母待齐；T13保持NOT_RUN。")
    for row in strategies:
        if row["id"]=="T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                       source_gate_path=relative+"/result_v1_0_1.json",current_evidence_path=relative+"/result_v1_0_1.json")
    for row in factors:
        if row["id"]=="M06":
            row.update(current_status="PROSPECTUS_SCOPE_AND_ONE_LISTING_RESTRICTION_REVIEWED_NOT_RUN",current_note=note,
                       current_evidence_path=relative+"/result_v1_0_1.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_PROSPECTUS_V1",latest_result=relative+"/result_v1_0_1.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_74_PROSPECTUS_FIELD_COMPARISONS_AND_ONE_ACTUAL_LISTING_SPLIT",
        current_research_phase="T13_PLAN_ADMIN_VERSIONS_LISTING_RESTRICTIONS_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=195,reused_additional_listing_documents_this_round=1,
        new_prospectus_comparison_records_this_round=74,new_canonical_commitment_facts_this_round=2,
        new_reviewed_rights_document_facts_this_round=2,new_extended_listing_facts_this_round=1,
        source_field_candidates_this_round=2784,
        source_field_candidate_kind="195文档候选2784；74当前发行章节三类字段对照、18持有期条款、1既存上市公告的分类及股东认配量；非全篇阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_prospectus_comparisons=relative+"/reviewed_current_section_comparisons.json",
        latest_rights_event_chains=relative+"/event_chains_with_restriction_addendum.json",
        latest_rights_document_facts=relative+"/combined_source_facts_with_restriction_addendum.json",
        latest_rights_publication_queries=relative+"/publication_boundary_queries_with_restriction_addendum.json",
        latest_rights_holding_restriction_review=relative+"/holding_restriction_clause_reviews.json",
        rights_remaining_plan_admin_documents=121,
        next_independent_source_action="T13_121_PLAN_ADMIN_DOCUMENTS_AND_REMAINING_ACTUAL_LISTING_RESTRICTION_SPLITS",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    for key,value in protected.items():
        assert status[key]==value
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、成本后完整账户净夏普达到1.2的目标仍未实现。本轮没有新增账户、收益、模型或网络请求，没有制作交付包；合格候选和独立前向观测仍均为0。",
        "上轮是有进展的一轮：179份提示字段完成对照并补存19份原件。本轮在读取新增正文范围前固定了195份剩余非提示文件，重新区分74份正式说明书/摘要、7份申报稿、5份申请文件更新通知和其他方案、审核、历史募集资金文件。195份文件共定位2784条候选，当前发行章节定位涉及584页范围；这不表示逐页人工读完584页，更不表示读完全部195份。",
        "74份正式说明书/摘要与37份对应发行公告或修订公告均为相同目录日期，覆盖36组既有发行，没有建立更早公开时点。60份在当前发行章节有当次核准号支撑；其余14份仅按同发行人、同目录日主公告作来源对照，没有把历史核准号强行当成当前身份。来源日末仍为日期代理，历史首版未验证。",
        "66份当前章节的完整登记和缴款日期与同日主公告一致；西部、兴业2022和浙商的6份文件日程只写月日，保留无年份字段；长春2份说明书日程只列T日、T+1日等，不填入主公告的完整日期。67份有明确数值价格且相符；7份当前章节仅保留定价原则或未得到明确数值。太平洋摘要缴款区间跨22、23页，隆基摘要价格跨21、22页，均已目检连接；天风2份登记日跨表行误提取候选保留并排除。",
        "数量口径分成五类：50份当前章节的测算数量与主公告计划数量一致；10份涉及A/H合计，A股量、H股量与合计分开；8份使用不同股本基准日；4份东吴说明书列回购专户处置前后两种情景；2份节能风电当前章节给出每10股配3股但未抽取到明确总数量。这些都不是实际认购股数，不覆盖既有认购结果。",
        "隆基按2018-09-30股本测算837504000股，同日主公告837241060股；江苏银行按2020-06-30股本测算3463352690股，主公告3463356846股；华安按2021-03-31股本测算1086314031股，主公告1086315441股；财通按2021-12-20股本测算1076704335股，主公告1076704995股。原文都允许因股本变动调整数量，分别保存基准日和测算值，不能把差额记成实际认购更正。财通比例乘积有不足1股差额，保留原数，不套用统一向下取整。",
        "东吴2020年章节列899129490股（扣除回购专户2901700股）和900000000股（专户股份全部处置）；2021年列1151645218股（扣除41701514股）和1164155672股（全部处置）。同次发行的条件情景分别保留，不累计成两次供给。金风、招商、中信、东方和浙商的A/H合计均与两类股份相加一致，只取A股量作同日公告比较。",
        "当前章节18段持有期条款已全部阅读，涉及9组发行：8段一般不设持有期、8段保留法规或监管例外、2段长春明确超达投资自愿锁定6个月。没有从不设额外持有期推断所有股份随时可卖，也没有从未找到条款推断不存在其他交易限制。",
        "按明确股东锁定条款定向追加读取1份已存上市公告1202299350。其2016-05-06目录代理时点披露：5月9日安排上市新增38785695股，其中登记无限售新增38772000股、登记限售新增13695股（高管锁定）；另披露超达认配8778110股。说明书先前6个月自愿锁定承诺仍单列。上市登记分类与自愿承诺是不同口径；没有推定承诺撤销，没有指定未披露的解禁终日，也没有直接相减生成即时可卖量。",
        "上述2份说明书的持有期记录补入来源账，1份已有上市事实扩展分类字段；之前315份来源事实完全保持原样，合计318份。6次长春专门时点查询证明，实际认配及上市分类没有提前至4月14日，全球36组发行的293组边界、586次保存回放也已复算。实际查看9张关键页面。字段与时点可复算不是全覆盖、外部复核或策略有效性。",
        "剩余121份方案、审核等文件尚待正文语义核对，全体实际上市股份限制及股东承诺版本仍未齐，其他发行方式与自由流通分母也未齐。严格M06压力为NOT_COMPUTED，T13仍NOT_RUN。下一步核对这些既存文件及其余上市公告的实际股份拆分，不重复本轮74份字段对照。",
        "有效研究账户552、无效实现152、累计执行704、固定研究问题9、合格候选0保持不变。直接查看74份说明书字段对照.csv、reviewed_current_section_comparisons.json、changchun_restriction_source_addendum.json和有效结果result_v1_0_1.json。"
    ]
    (OUT/"研究进展.md").write_text("\n\n".join(paragraphs)+"\n",encoding="utf-8")
    for path,obj in zip(paths,objects):
        update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"阶段":status["current_research_phase"],"说明书对照":74,"持有期条款":18,"实际上市分类新增复核":1,
                      "新账户":0,"合格候选":0,"下一步文件":121},ensure_ascii=False))


if __name__=="__main__":
    record()
