"""复核来源字段、时点和阅读边界后登记进展，账户与失败结果保持原值。"""
from collections import Counter
from datetime import datetime, timedelta
import json
import re

import pandas as pd

from research.factor96_rights_holding_remainder_review_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, FULL_READ, VISUAL_PAGES
from research.factor96_rights_holding_remainder_v1 import PATTERN
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"])==f["sha256"]
    rows=read(OUT/"targets.json");reviewed=read(OUT/"reviewed_candidates.json")
    source_rows=read(PREVIOUS/"remaining_holding_candidates.json")["candidates"]
    targets={r["document_id"]:r for r in read(PREVIOUS/"targets.json")}
    pages={}
    for rid in {r["document_id"] for r in reviewed}:
        target=targets[rid]
        assert digest(target["text_path"])==target["text_sha256"]
        pages[rid]={p["page"]:normalized(p["text"]) for p in read(target["text_path"])["pages"]}
    assert len(rows)==len(reviewed)==len(source_rows)==248
    windows=0;characters=0
    for i,(r,out,old) in enumerate(zip(rows,reviewed,source_rows)):
        assert r["pending_index"]==out["pending_index"]==i
        for key in ("document_id","candidate_set","candidate_index","page","start","end","literal"):
            assert r[key]==out[key]==old[key]
        text=pages[r["document_id"]][r["page"]]
        assert text[r["start"]:r["end"]]==r["literal"]
        assert out["reviewed"] and out["full_candidate_read"]==(i in FULL_READ)
        assert out["windows"]==r["windows"]
        for w in out["windows"]:
            assert text[w["start"]:w["end"]]==w["literal"]
            windows+=1;characters+=len(w["literal"])
        for m in re.finditer(PATTERN,r["literal"]):
            assert any(w["start"]<=r["start"]+m.start() and r["start"]+m.end()<=w["end"] for w in out["windows"])
    assert windows==293 and characters==54277
    assert read(OUT/"remaining_holding_candidates.json")["count"]==0
    extra=read(OUT/"extra_pages_read.json")
    for e in extra:assert e["literal"]==pages[e["document_id"]][e["page"]] and e["full_page_read"]
    prior=read(PREVIOUS/"combined_source_facts.json");facts=read(OUT/"reviewed_context_source_facts.json")
    addenda=read(OUT/"claim_addenda.json");combined=read(OUT/"combined_source_facts.json")
    assert len(prior)==445 and len(combined)==453 and combined[:445]==prior and combined[445:]==facts
    assert len({r["document_id"] for r in combined})==453 and len(facts)==8 and len(addenda)==2
    anchors=0
    for f in facts+addenda:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        assert f["event_id"] is None and f["updates"]=={} and not f["trading_feature_admitted"]
        assert f["effective_sellable_new_shares"] is None and f["strict_M06_pressure"] is None
        assert set(f["reviewed_claims"])==set(f["claim_evidence"])
        for es in f["claim_evidence"].values():
            for e in es:
                assert pages[f["document_id"]][e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
    qs=read(OUT/"source_publication_queries.json");by_id={f["document_id"]:f for f in facts+addenda}
    assert len(qs)==10
    for q in qs:
        f=by_id[q["document_id"]]
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        for side in ("before","at"):assert q[side]==source_view(f,q[side]["as_of"])
    comp=read(OUT/"citic_inheritance_vs_listing_classification.json")
    listing=read(OUT/"citic_followup_scope.json")["source"]
    assert digest(listing["text_path"])==listing["text_sha256"] and digest(listing["raw_path"])==listing["raw_sha256"]
    lp={p["page"]:normalized(p["text"]) for p in read(listing["text_path"])["pages"]}
    for page in comp["listing_full_pages"]:assert page["literal"]==lp[page["page"]]
    assert comp["listing_source_claims"]==listing
    assert comp["prospectus_claims"]==by_id["1212175293"]["reviewed_claims"]
    registration=comp["prospectus_claims"]["acquisition_share_registration"]
    assert sum(registration["original_consideration_shares"].values())==registration["sum_original_consideration_shares"]==809867629
    assert registration["registration_completed_date"]=="2020-03-11"
    assert comp["pre_listing_publication_days"]==27 and comp["relationship_known_at"]==listing["known_at"]
    assert comp["actual_allotment_by_actor"] is None and comp["current_rights_effective_sellable_shares"] is None
    assert not comp["zero_effective_restrictions_proved_by_registration"] and not comp["computed_from_holder_difference"]
    for q in read(OUT/"citic_targeted_name_searches.json"):
        assert digest(q["text_path"])==q["text_sha256"]
        assert not [p for p in read(q["text_path"])["pages"] if any(t in normalized(p["text"]) for t in q["terms"])]
    result=read(OUT/"result.json")
    assert sum(len(f["reviewed_claims"]) for f in facts+addenda)==result["source_fields_reviewed"]==12
    assert dict(Counter(r["semantic_class"] for r in reviewed))==result["semantic_counts"]
    assert result["new_accounts"]==result["qualified_candidates"]==result["current_rights_sellable_quantities_computed"]==0
    assert result["extra_full_prospectus_pages_read"]==len(extra)==11
    visual=[{"document_id":rid,"page":p,"path":str(OUT/"page_previews"/f"{rid}_p{p}.png"),
             "sha256":digest(OUT/"page_previews"/f"{rid}_p{p}.png")} for rid,p in VISUAL_PAGES]
    receipt={"at":now(),"status":"PASS_SAVED_WINDOWS_CLAIMS_AND_PUBLICATION_BOUNDARIES",
             "keyword_candidates_checked":248,"keyword_windows_checked":windows,"window_characters_checked":characters,
             "full_candidates_actually_read":16,"new_source_facts":8,"append_only_addenda":2,
             "literal_anchors_checked":anchors,"boundary_views_checked":20,
             "old_source_records_unchanged":445,"source_quantity_sum_checked":1,"current_rights_quantity_computed":0,
             "visual_pages":visual,"result_sha256":digest(OUT/"result.json"),
             "review_code_sha256":digest(ROOT/"research/factor96_rights_holding_remainder_review_v1.py"),
             "meaning":"已核对固定关键词候选，不是全文阅读、完整限售覆盖、策略验证或交易许可。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved();program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_PROSPECTUS_CONSTRAINTS_V1"
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios","cumulative_invalid_implementation_account_scenarios",
          "last_completed_account_experiment","last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
          "qualified_candidates","orders_authorized","latest_rights_event_chains","rights_remaining_plan_admin_documents")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths]
    relative=OUT.relative_to(ROOT).as_posix()
    note=("剩余248段候选的293个关键词窗口已逐一核读，其中16段补读完整候选及11页正文，查看3页原图；固定关键词候选待核数降为0，但未声称全文已读。"
          "新增8份来源及2份旧来源的独立补充，共12项字段，原445份来源不变，累计453份。"
          "中信证券广州证券收购股份48个月锁定明确继承至配股增加股份，2020年3月11日登记与表格2019年事项标签分开，不能因上市公告新增股全部登记无限售就判可卖。"
          "华安IPO后两年无减持计划的履行列为严格履行而非已到期；天齐2015至今语句是2015问询回复，非2017现状。"
          "天风、红塔IPO后减持限制及南山、东方条款按事项和触发条件保存；没有量化当次新增有效可卖股数。T13未运行，目标未实现。")
    queue=[
        {"id":"CITIC_INHERITED_RIGHTS_LOCK","priority":1,"sources":["1212175293","1212332253"],
         "need":"越秀金控及金控有限2022年实际认购数量，以及48个月承诺后续变更和履行的同期来源。"},
        {"id":"HUAAN_IPO_POSTLOCK_PLAN","priority":2,"sources":["1210115279"],
         "need":"安徽国控集团原锁定期届满时点、两年无减持计划覆盖范围与2021配股期间状态。"},
        {"id":"INSPUR_COMMITMENT_ISSUE_IDENTITY","priority":3,"sources":["1207347972"],
         "need":"2016年10月其他承诺中配股后六个月条款对应哪一次配股及后续版本。"},
        {"id":"WINDPOWER_2022_COMMITMENT_IDENTITY","priority":4,"sources":["1213982762","1215110331"],
         "need":"2022年配股原始承诺，不以2021年可转债已履行承诺作为独立确认或撤销证据。"},
        {"id":"OTHER_LEGACY_LOCK_AND_QUANTITY_LINKS","priority":5,
         "sources":["1202123523","1205513864","1211627066","1207348369","1210543901","1202315263"],
         "need":"历史锁定、增持承诺及IPO减持条件与当前获配股份、起止时间和已实施数量之间的证据连接。"},
    ]
    save(OUT/"unresolved_holding_links.json",{"at":now(),"items":queue,"count":len(queue),
         "scope":"候选窗口阅读结束后的事项、时间和数量缺口；不是完整发行覆盖清单。"})
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                                    current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="KEYWORD_CONTEXTS_REVIEWED_RIGHTS_LOCK_INHERITANCE_FOUND_QUANTITY_UNKNOWN",current_note=note,current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_HOLDING_REMAINDER_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_248_CONTEXTS_REVIEWED_EXPLICIT_RIGHTS_INHERITED_LOCK_FOUND",
        current_research_phase="T13_HOLDING_IDENTITY_QUANTITY_AND_FREE_FLOAT_GAPS",
        reused_rights_pdf_documents_this_round=65,new_reviewed_rights_document_facts_this_round=8,
        new_holding_context_claim_fields_this_round=12,new_existing_holding_source_addenda_this_round=2,
        source_field_candidates_this_round=248,source_field_candidate_kind="248段候选的全部关键词窗口核读，16段补读，8份新来源、2份补充、12字段。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_prospectus_context_facts=relative+"/reviewed_context_source_facts.json",
        latest_rights_holding_claim_addenda=relative+"/claim_addenda.json",
        latest_rights_prospectus_context_queries=relative+"/source_publication_queries.json",
        latest_rights_citic_inherited_holding_comparison=relative+"/citic_inheritance_vs_listing_classification.json",
        latest_rights_pending_holding_candidates=relative+"/remaining_holding_candidates.json",
        latest_rights_unresolved_holding_links=relative+"/unresolved_holding_links.json",
        rights_remaining_prospectus_constraint_candidates=0,
        next_independent_source_action="RESOLVE_CITIC_RIGHTS_ALLOTMENT_AND_INHERITED_LOCK_WITH_HUAAN_AND_PRIOR_SCOPE_GAPS",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    result=read(OUT/"result.json")
    paragraphs=[
        "只操作510300、完整账户成本后夏普达到1.2的目标仍未实现。达标候选0，独立前向观测0。本轮完成固定剩余关键词候选的核对，发现中信证券旧收购锁定明确覆盖配股新增股份；即时可卖数量仍未算出，没有新增回测或交付包。",
        "上一轮剩余248段分布于65份说明书及摘要，共151683个规范化字符。本轮逐一阅读全部293个关键词上下文窗口，共54277字符；对16段补读完整候选，并补读11页说明书正文、4页中信上市公告，查看3页原图。固定关键词候选待核数降为0，表示这些候选已按上下文归类，不表示74份PDF全文已读或所有限制已量化。",
        "中信证券2022年1月14日配股说明书第195页明确：越秀金控、金控有限收购广州证券所得对价股份，自登记名下之日起48个月不得转让，监管要求更长时从其要求；配股等原因增加的股份同样遵守。第51页说明登记于2020年3月11日完成，两家原对价股份265352996与544514633股合计809867629股。原对价股数不是2022年实际获配股数，表格的2019年事项标签也不是登记日。",
        "中信证券2月10日上市公告另称新增1552021645股全部为无限售流通股，登记分类与上述承诺分别保留。说明书比该上市公告早27个自然日可得，比较关系只从后者可得时起成立；没有声称这是承诺的首次披露。实际认购数量未由这两份来源直接确立，不以新旧股东表差额或原对价股数乘配股比例代替。已有发行公告和结果公告只做名称检索、无命中，不据此认定未认购。",
        "华安证券2021年5月28日说明书第162页将安徽国控集团‘锁定期满后2年内无减持计划’列为IPO相关承诺。原图确认最后一列为是否及时严格履行，值为是，并非已履行完毕或到期。原锁定终点和当前获配股份覆盖范围继续待核。",
        "中金黄金两份说明书保存的是2015年7月15日起增持安排中的不减持承诺，与后来上市公告的配股后六个月承诺分别保留。天齐2017说明书中‘2015年8月至今处于限售’属于2015年12月问询函回复，不能当成2017现状。南山摘要重述怡力电业既有股份锁定至2020年6月6日。",
        "天风IPO后两年累计减持上限以首次上市日总股本为分母；红塔IPO后两年每年减持上限以公司股份总数为分母，不能混用。天风、东方违反同业竞争承诺时不得转让属于有触发条件的约束，现有来源未证明触发，未当作已经发生的限售。南山集团2016收购前股份12个月约束另作补充。",
        "新增8份来源、2份已有来源的追加条款，共12项字段；445份原记录逐项保持不变，累计453份来源，2份补充单独保存。原36条实施事件和账户结果不变。候选分类为："+json.dumps(result["semantic_counts"],ensure_ascii=False)+"。",
        "下一步优先核对中信越秀两家实际认购与旧承诺变更、华安原锁定届满及范围；浪潮事项归属、节能风电2022配股承诺等缺口仍保留。完整发行覆盖与自由流通股分母仍未齐，严格M06未计算，T13仍未运行。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as h:h.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"核读候选":248,"待读关键词候选":0,"新来源":8,"追加来源":2,"字段":12,
                      "来源总数":453,"原文定位":receipt["literal_anchors_checked"],"目标实现":False},ensure_ascii=False))


if __name__=="__main__":record()
