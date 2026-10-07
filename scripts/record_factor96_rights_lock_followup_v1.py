"""核对新披露与历史时点回放并登记量化进展，不重跑收益。"""
from copy import deepcopy
from datetime import date, datetime, timedelta
import json
import pandas as pd

from research.factor96_rights_lock_followup_review_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, citic_view, VISUAL_PAGES
from scripts.record_factor96_remaining_changes_v1 import update


def verify():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    assert digest(OUT/"targets.json")==read(OUT/"source_target_freeze.json")["target_sha256"]
    facts=read(OUT/"reviewed_source_facts.json");add=read(OUT/"claim_addenda.json")
    prior=read(PREVIOUS/"combined_source_facts.json");combined=read(OUT/"combined_source_facts.json")
    assert len(facts)==5 and len(add)==1 and len(prior)==453 and len(combined)==458
    assert combined[:453]==prior and combined[453:]==facts and len({f["document_id"] for f in combined})==458
    assert read(OUT/"combined_claim_addenda.json")==read(PREVIOUS/"claim_addenda.json")+add
    anchors=0;pages={}
    for f in facts+add:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        pp={p["page"]:normalized(p["text"]) for p in read(f["text_path"])["pages"]};pages[f["document_id"]]=pp
        assert f["event_id"] is None and f["updates"]=={} and not f["trading_feature_admitted"]
        assert f["effective_sellable_new_shares"] is None
        for es in f["claim_evidence"].values():
            for e in es:
                assert pp[e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
    for r in read(OUT/"read_scope.json"):
        assert len(pages[r["document_id"]])==r["total_pages"]
        for p in r["read_text"]:assert p["literal"]==pages[r["document_id"]][p["page"]]
    byid={f["document_id"]:f for f in facts}
    h=byid["1212493239"]["reviewed_claims"];release=byid["1221139139"]["reviewed_claims"]["announced_release"]
    assert sum(h["a_share_allotment_with_inherited_lock"]["actors"].values())==121480144
    assert sum(release["rights_by_actor"].values())==release["rights_derived_shares"]==121480144
    assert sum(release["total_by_actor"].values())==release["total_shares"]==931347773
    assert release["original_acquisition_shares"]+release["rights_derived_shares"]==release["total_shares"]
    assert h["a_h_registration_classification"]["registered_new_unrestricted_A"]+h["a_h_registration_classification"]["new_H_shares"]==1893770800
    original=byid["1207134172"]["reviewed_claims"]["announced_ipo_release"]
    assert original["controller_shares"]+original["social_security_account_shares"]==original["total_shares"]==919000000
    hc=read(OUT/"huaan_unlock_and_current_scope.json")
    begin=date.fromisoformat(hc["original_announced_unlock"]);anniversary=begin.replace(year=begin.year+2)
    assert anniversary.isoformat()==hc["second_anniversary_calculated"]=="2021-12-06"
    assert begin<date.fromisoformat(hc["2021_rights_record_date"])<anniversary
    assert hc["new_rights_restricted_shares"] is None and hc["new_rights_share_coverage"]=="NOT_ESTABLISHED"
    for p in hc["extra_full_pages_read"]:assert p["literal"]==pages["1210115279"][p["page"]]
    conflict=read(OUT/"preserved_quantity_conflict.json")
    assert sum(conflict["earlier_original_components"])==conflict["earlier_component_sum"]==809867629
    assert conflict["reported_original_consideration_shares"]==h["reported_legacy_consideration_total_disputed"]["reported_shares"]==833786629
    assert conflict["reported_original_consideration_shares"]-conflict["earlier_component_sum"]==conflict["difference"]==23919000
    allfacts={f["document_id"]:f for f in facts+add};queries=read(OUT/"source_publication_queries.json")
    for q in queries:
        f=allfacts[q["document_id"]]
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        for side in ("before","at"):assert q[side]==source_view(f,q[side]["as_of"])
    replay=read(OUT/"citic_holding_timeline.json")
    for state in replay:
        assert state==citic_view(facts,state["as_of"])
        matured=[f for f in facts if f["known_at"]<=state["as_of"]]
        assert state==citic_view(matured,state["as_of"])
        assert state["effective_sellable_new_shares"] is None
    assert replay[2]["identified_rights_tranche_shares"] is None
    assert replay[3]["identified_rights_tranche_shares"] is None and replay[4]["identified_rights_tranche_shares"]==121480144
    assert replay[5]["extended_lock_end"] is None and replay[6]["extended_lock_end"]=="2024-09-10"
    assert replay[7]["announced_release_date"] is None and replay[8]["announced_release_date"]=="2024-09-11"
    assert replay[9]["release_calendar_phase"].endswith("NOT_REAL_SALE_PROOF")
    extension=read(OUT/"extension_target_freeze.json");receipt=read(OUT/"extension_catalogue_receipt.json")
    assert digest(receipt["raw_path"])==receipt["sha256"] and receipt["http_status"]==200
    assert byid["1219266222"]["known_at"]=="2024-03-09T23:59:59+08:00"
    assert extension["target"]["document_id"]=="1219266222"
    result=read(OUT/"result.json");assert result["new_pdf_pages"]==130 and result["new_source_pages_read"]==16
    assert sum(len(f["reviewed_claims"]) for f in facts+add)==result["source_fields_reviewed"]==12
    images=[{"document_id":rid,"page":p,"path":str(OUT/"page_previews"/f"{rid}_p{p}.png"),
             "sha256":digest(OUT/"page_previews"/f"{rid}_p{p}.png")} for rid,p in VISUAL_PAGES]
    output={"at":now(),"status":"PASS_SOURCE_QUANTITIES_AND_NO_FUTURE_DISCLOSURE_BACKFILL",
            "source_literal_anchors_checked":anchors,"source_boundary_views":len(queries)*2,
            "citic_timeline_states_recomputed":len(replay),"future_source_removal_checks":len(replay),
            "read_new_source_pages":16,"read_reused_source_pages":2,"prior_facts_unchanged":453,
            "preserved_conflicts":1,"visual_pages":images,"result_sha256":digest(OUT/"result.json"),
            "review_code_sha256":digest(ROOT/"research/factor96_rights_lock_followup_review_v1.py"),
            "goal_achieved":False,"strategy_validation_performed":False}
    save(OUT/"saved_recomputation_receipt.json",output)
    return output


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    verified=verify();program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_HOLDING_REMAINDER_V1"
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios","cumulative_invalid_implementation_account_scenarios",
          "last_completed_account_experiment","last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
          "qualified_candidates","orders_authorized","latest_rights_event_chains","rights_remaining_plan_admin_documents")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths]
    relative=OUT.relative_to(ROOT).as_posix();result=read(OUT/"result.json")
    note=("新增5份原始披露、130页文本，核读16页新来源及2页已存来源，4张原图确认。"
          "中信2022年3月3日H股结果附带A股明细，越秀两家实际认购39802949与81677195股，共121480144股继承旧锁定；不倒填至2月上市前。"
          "2024年3月9日目录公告明确延锁6个月至9月10日，9月5日公告拟9月11日解禁。"
          "H股结果所述旧收购总数833786629与原组件合计809867629相差23919000，原文并存不订正。"
          "华安2019年12月4日解禁公告确定12月6日可上市及此后两年无减持计划，当前配股新股份的覆盖仍未明确。"
          "来源累计458份，原453份不变，另补充华安主体更名连接。已保存10个时点的约束回放。T13未回测，目标未实现。")
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                                    current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="EXPLICIT_RIGHTS_RESTRICTED_TRANCHE_121480144_KNOWN_MARCH3_2022_NOT_RUN",
                                    current_note=note,current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_LOCK_FOLLOWUP_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_EXPLICIT_RIGHTS_TRANCHE_QUANTITY_AND_EXTENSION_PIT_CHAIN",
        current_research_phase="T13_MIXED_H_TITLE_COVERAGE_AND_REMAINING_HOLDING_GAPS",
        new_source_documents_this_round=5,new_searchable_text_documents_this_round=5,
        new_archived_source_http_responses_this_round=result["raw_http_response_receipts"],
        new_reviewed_rights_document_facts_this_round=5,new_holding_context_claim_fields_this_round=12,
        new_existing_holding_source_addenda_this_round=1,new_source_conflicts_this_round=1,
        source_field_candidates_this_round=5,source_field_candidate_kind="5份约束链直接来源，含H股公告附带A股认购与限售、延锁、解禁和IPO承诺。",
        latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_holding_followup_facts=relative+"/reviewed_source_facts.json",
        latest_rights_holding_claim_addenda=relative+"/combined_claim_addenda.json",
        latest_rights_holding_followup_queries=relative+"/source_publication_queries.json",
        latest_rights_citic_holding_timeline=relative+"/citic_holding_timeline.json",
        latest_rights_huaan_unlock_chain=relative+"/huaan_unlock_and_current_scope.json",
        latest_rights_mixed_title_coverage_gap=relative+"/mixed_title_coverage_gap.json",
        latest_rights_holding_quantity_conflict=relative+"/preserved_quantity_conflict.json",
        rights_local_original_pdfs_including_supplement=status["rights_local_original_pdfs_including_supplement"]+5,
        next_independent_source_action=result["next_source_action"],goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    oldqueue=read(PREVIOUS/"unresolved_holding_links.json")
    queue=deepcopy(oldqueue)
    queue["at"]=now()
    for item in queue["items"]:
        if item["id"]=="CITIC_INHERITED_RIGHTS_LOCK":
            item.update(status="QUANTITY_AND_EXTENSION_SOURCES_FOUND_PARTIAL_VERSION_COVERAGE",
                        need="保留2022年3月3日前确切数量未知；查漏H股/A+H标题中的A股字段，旧收购总数冲突未获正式更正。",
                        new_evidence=["1212493239","1219266222","1221139139"])
        elif item["id"]=="HUAAN_IPO_POSTLOCK_PLAN":
            item.update(status="ORIGINAL_UNLOCK_DATE_FOUND_NEW_RIGHTS_SCOPE_UNKNOWN",
                        need="2019年12月6日起两年的旧承诺与2021年新获配股份的覆盖关系；不从旧限售股量推算新限售量。",
                        new_evidence=["1207134172","1202838760"])
    save(OUT/"unresolved_holding_links.json",queue)
    status["latest_rights_unresolved_holding_links"]=relative+"/unresolved_holding_links.json"
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、完整账户成本后夏普达到1.2的目标仍未实现。达标候选0、独立前向观测0。本轮首次从直接来源识别中信2022配股中121480144股继承旧锁定，保存随后延长和解禁公告；这不是新的回测收益，也没有交付包。",
        "中信2022年3月3日H股配股发行结果公告第2页明确，两家越秀主体A股认购分别39802949股和81677195股，与原股份执行同样限售期，合计121480144股。该文件标题此前被归为H股，正文却含A股量化字段。3月3日之后可读取这项披露，不能倒填到2月15日上市前。",
        "原上市公告与这份H股结果的股本表仍把新增1552021645股A股登记为无限售流通股；这一分类没有覆盖全部承诺约束。H股结果另将旧收购对价总量写成833786629股，与此前说明书两笔265352996及544514633股合计809867629股不符，相差23919000股。原图确认原文如此，差异单独保存，未按经验修正，也未将它作为新配股股数。",
        "2024年3月8日的承诺函随后在3月9日巨潮目录披露，明确将包括配股股份在内的931347773股A股延长限售6个月，原到期2024年3月10日，延至9月10日。另有官网香港披露副本落款3月8日，本轮回放采用已取得目录回执的3月9日日末代理。2024年9月5日公告再明确拟于9月11日解禁，拆分为旧收购809867629股与配股121480144股；不是9.31亿股全部属于2022配股。H股不属于此项限售范围。",
        "华安2019年12月4日解禁公告确定原IPO限售股可上市时间为12月6日，控股股东887020879股、社保转持户31979121股，合计919000000股。首页的两年无减持计划针对所持首次公开发行前限售股，正文另用所持公司股票的一般表述。2021年配股说明书仍列该承诺，但现有材料未明确覆盖当次新获配股份，因此新限售数量仍未知。两周年2021年12月6日是依据日期计算的周年节点，不是另一个原文解禁日期。",
        "2016年招股摘要补充原锁定条件：2012年7月13日起60个月或上市日起36个月孰长，并有特定股价条件下延长12个月的承诺；未推定延长条件曾触发。该摘要以披露URL日期保存，未独立核对目录回执，2019年公告已重新披露相同锁定条款。2021年说明书的主体释义明确安徽国控集团与原国资运营公司为更名前后名称。",
        "取得5份PDF共130页，仅4份短公告全文及招股摘要5页被核读，共16页新来源，另读2页已存说明书、查看4张原图。新增5份来源和1份原来源的主体名称补充，共12项字段；453份旧来源不变，累计458份来源，累计3份条款补充独立保存。10个时点回放确认：数量、延锁日期和拟解禁日期均不会在其来源可得之前出现。",
        "下一步核对已有未选中H股及A+H配股结果、股份变动标题中的A股字段，保留此前浪潮、节能风电及其他持有约束缺口。完整发行覆盖、其他约束和自由流通股分母仍未齐，严格M06未计算，T13尚未回测。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as h:h.write("\n\n".join(paragraphs)+"\n")
    for p,obj in zip(paths,objects):update(p,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"goal_achieved":False,"new_accounts":0,"delivery_package_created":False})
    print(json.dumps({"来源总数":458,"明确配股承诺限售股数":121480144,"原文锚点":verified["source_literal_anchors_checked"],
                      "时点回放":10,"目标实现":False},ensure_ascii=False))


if __name__=="__main__":record()
