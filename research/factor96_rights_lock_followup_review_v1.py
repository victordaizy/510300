"""将认购数量、限售承诺与后来解禁分别按来源时点保存。"""
from copy import deepcopy
from datetime import datetime, timedelta
import json

from research.factor96_rights_lock_followup_v1 import OUT, PREVIOUS, ROOT, read, save, digest, now
from research.factor96_rights_event_chains_v1 import normalized
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import claim
from research.factor96_rights_preplans_review_v1 import anchor
from research.factor96_rights_prospectus_constraints_review_v1 import source_view

VISUAL_PAGES=[("1212493239",2),("1219266222",1),("1221139139",3),("1207134172",1)]
READ_PAGES={"1212493239":[1,2],"1221139139":[1,2,3,4],"1207134172":[1,2,3,4],
            "1202838760":[8,13,15,31,32],"1219266222":[1]}


def specs():
    return {
      "1212493239":[
        claim("a_share_allotment_with_inherited_lock",{
            "actors":{"越秀金控":39802949,"金控有限":81677195},"sum_shares_calculated":121480144,
            "scope":"当次A股配股获配股份","same_lock_as_original":True,"base_months":48,
            "calendar_release_date":None,"not_h_share_allotments":True},
            (2,"本次越秀金控、金控有限参与公司A股配股而分别增持的39,802,949股股份、81,677,195股股份,与原股份执行同样的限售期"),
            (2,"该等股份自上述交易发行结束之日起48个月内不进行转让")),
        claim("reported_legacy_consideration_total_disputed",{
            "reported_shares":833786629,"status":"CONFLICT_WITH_EARLIER_PROSPECTUS_COMPONENT_SUM",
            "not_used_as_current_rights_quantity":True,"not_silently_corrected":True},
            (2,"在公司2019年发行股份购买资产交易中合计获得公司833,786,629股A股股份")),
        claim("a_h_registration_classification",{
            "registered_restricted_A_before_and_after":833786629,"registered_new_unrestricted_A":1552021645,
            "new_H_shares":341749155,"total_new_A_H":1893770800,
            "classification_not_effective_sellability":True},
            (1,"有限售条件流通股(A股)833,786,6296.45%-833,786,6295.63%"),
            (1,"1,552,021,645"),(2,"341,749,155"),(2,"1,893,770,800"))],
      "1221139139":[
        claim("holder_name_mapping",{
            "广州越秀金融控股集团股份有限公司":"广州越秀资本控股集团股份有限公司",
            "广州越秀金融控股集团有限公司":"广州越秀资本控股集团有限公司"},
            (1,"广州越秀金融控股集团股份有限公司(现已更名为广州越秀资本控股集团股份有限公司"),
            (1,"广州越秀金融控股集团有限公司(现已更名为广州越秀资本控股集团有限公司")),
        claim("announced_release",{
            "listing_date":"2024-09-11","total_shares":931347773,
            "original_acquisition_shares":809867629,"rights_derived_shares":121480144,
            "rights_by_actor":{"越秀资本":39802949,"广州越秀资本":81677195},
            "total_by_actor":{"越秀资本":305155945,"广州越秀资本":626191828},
            "new_release_announcement_not_actual_sale":True},
            (3,"本次上市流通日期为2024年9月11日"),(3,"本次上市流通的限售股总数为931,347,773股"),
            (3,"参与公司2022年配股认购39,802,949股承诺限售股"),
            (3,"参与公司2022年配股认购81,677,195股承诺限售股"),
            (3,"1向特定对象发行809,867,6292其他注121,480,144合计931,347,773")),
        claim("retrospectively_reported_extension",{
            "commitment_date":"2024-03-08","extension_months":6,"extended_end":"2024-09-10",
            "backfilled_to_commitment_date":False},
            (2,"2024年3月8日,越秀资本出具《关于延长所持有中信证券A股股票限售期的承诺函》"),
            (2,"限售期延长6个月,即至2024年9月10日届满"))],
      "1207134172":[
        claim("announced_ipo_release",{
            "announced_listing_date":"2019-12-06","total_shares":919000000,
            "controller":"安徽省国有资本运营控股集团有限公司","controller_shares":887020879,
            "social_security_account_shares":31979121,"scope":"首次公开发行前限售股",
            "actual_sale_not_proved":True},
            (1,"本次解除限售股份的数量为919,000,000股"),
            (1,"安徽省国有资本运营控股集团有限公司解除限售股份数量为887,020,879股"),
            (1,"全国社会保障基金理事会转持一户解除限售股份数量为31,979,121股"),
            (1,"公司本次限售股份可上市流通日期为2019年12月6日")),
        claim("post_unlock_no_reduction_plan",{
            "actor":"安徽省国有资本运营控股集团有限公司","years_after_unlock":2,
            "frontpage_scope":"所持公司首次公开发行前限售股解禁后",
            "body_scope":"本控股股东所持公司股票锁定期满后",
            "scope_of_2021_rights_new_shares":"NOT_ESTABLISHED", "new_rights_restricted_shares":None,
            "anniversary_calculated":"2021-12-06","anniversary_is_source_literal":False},
            (1,"承诺其所持公司首次公开发行前限售股解禁后两年内无减持计划"),
            (2,"本控股股东所持公司股票锁定期满后两年内无减持计划"))],
      "1202838760":[
        claim("original_ipo_lock_and_intent",{
            "actor_name_in_source":"安徽省国有资产运营有限公司",
            "base_lock_candidates":[{"from":"2012-07-13","months":60},{"from":"股票上市日","months":36}],
            "choose_longer":True,"conditional_extension_months":12,
            "extension_trigger":"上市后六个月内连续20个交易日收盘价低于发行价，或六个月期末收盘价低于发行价",
            "no_reduction_plan_years_after_lock":2,"trigger_activated_not_inferred":True,
            "source_date_basis":"URL日期代理；未单独核对目录回执；2019来源重新披露该承诺"},
            (8,"自2012年7月13日起锁定60个月或自公司股票上市之日起锁定36个月,锁定期以前两者孰长为准"),
            (8,"公司上市后6个月内如公司股票连续20个交易日的收盘价均低于发行价,或者上市后6个月期末收盘价低于发行价,锁定期限自动延长12个月"),
            (8,"本控股股东所持公司股票锁定期满后两年内无减持计划"))],
      "1219266222":[
        claim("extension_terms",{
            "letter_received_date":"2024-03-08","old_end":"2024-03-10","extended_end":"2024-09-10",
            "months":6,"total_A_shares":931347773,"public_date_proxy":"2024-03-09",
            "letter_date_not_used_as_publication_date":True},
            (1,"于2024年3月8日收到公司股东"),(1,"将于2024年3月10日限售期届满"),
            (1,"所持有中信证券931,347,773股A股股票限售期6个月,即至2024年9月10日届满")),
        claim("rights_quantity_and_H_exclusion",{
            "rights_derived_A_shares":121480144,"original_consideration_A_shares":809867629,
            "total_locked_A_shares":931347773,"H_shares_not_subject_to_this_commitment":True},
            (1,"定向发行的809,867,629股A股股票"),
            (1,"参与中信证券A股配股新增取得121,480,144股A股股票"),
            (1,"该部分H股股票不受前述限售承诺影响,为无限售条件流通股"))],
    }


def citic_view(facts, at):
    """只回放此约束链，不替代原36条发行链的其他字段。"""
    formal=next(f for f in read(PREVIOUS/"combined_source_facts.json") if f["document_id"]=="1212175293")
    listing=read(PREVIOUS/"citic_inheritance_vs_listing_classification.json")["listing_source_claims"]
    state={"as_of":at,"event_id":"600030.SH_A_RIGHTS_2021_3729","source_documents":[],
           "inheritance_clause_known":False,"registered_new_unrestricted_A":None,
           "identified_rights_tranche_shares":None,"identified_rights_tranche_by_actor":None,
           "extended_lock_end":None,"announced_release_date":None,"release_calendar_phase":"NO_VIEW",
           "effective_sellable_new_shares":None,"trading_feature_admitted":False}
    if formal["known_at"]<=at:
        state["inheritance_clause_known"]=True;state["source_documents"].append(formal["document_id"])
    if listing["known_at"]<=at:
        state["registered_new_unrestricted_A"]=listing["listing_reported_unrestricted_new_shares"]
        state["source_documents"].append(listing["document_id"])
    byid={f["document_id"]:f for f in facts}
    for rid in ("1212493239","1219266222","1221139139"):
        if rid not in byid or byid[rid]["known_at"]>at:continue
        f=byid[rid];c=f["reviewed_claims"];state["source_documents"].append(rid)
        if rid=="1212493239":
            state["identified_rights_tranche_shares"]=c["a_share_allotment_with_inherited_lock"]["sum_shares_calculated"]
            state["identified_rights_tranche_by_actor"]=deepcopy(c["a_share_allotment_with_inherited_lock"]["actors"])
        elif rid=="1219266222":state["extended_lock_end"]=c["extension_terms"]["extended_end"]
        else:state["announced_release_date"]=c["announced_release"]["listing_date"]
    if state["announced_release_date"]:
        state["release_calendar_phase"]="ANNOUNCED_DATE_AHEAD" if at[:10]<state["announced_release_date"] else "ANNOUNCED_DATE_REACHED_OR_PASSED_NOT_REAL_SALE_PROOF"
    return state


def finalize():
    rows=read(OUT/"documents.json")+[read(OUT/"extension_document.json")]
    assert len(rows)==5 and all(r["status"]=="PDF_TEXT_SAVED" for r in rows)
    facts=[];pages_by_id={};ss=specs();read_scope=[]
    for r in rows:
        rid=r["document_id"];absolute={**r,"raw_snapshot":str(OUT/r["raw_snapshot"]),"text_snapshot":str(OUT/r["text_snapshot"])}
        doc=Document(absolute);pages=dict(doc.pages);pages_by_id[rid]=pages
        f=base_fact(doc,None)
        f.update(reviewed_claims={},claim_evidence={},effective_sellable_new_shares=None,strict_M06_pressure=None,
                 full_body_reviewed=len(READ_PAGES[rid])==len(pages),source_date_basis=r["source_date_basis"],
                 notes=["按此来源日期代理保存；不将后续披露或签署日提前用于旧决策。"])
        for c in ss[rid]:
            f["reviewed_claims"][c["field"]]=c["value"]
            f["claim_evidence"][c["field"]]=[anchor(rid,pages,p,t) for p,t in c["proof"]]
        facts.append(f)
        read_scope.append({"document_id":rid,"total_pages":len(pages),"read_pages":READ_PAGES[rid],
                           "full_document_read":len(READ_PAGES[rid])==len(pages),
                           "read_text":[{"page":p,"literal":pages[p]} for p in READ_PAGES[rid]]})
    prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(prior)==453 and not {f["document_id"] for f in facts}&{f["document_id"] for f in prior}
    huaan=deepcopy(next(f for f in prior if f["document_id"]=="1210115279"))
    hp={p["page"]:normalized(p["text"]) for p in read(huaan["text_path"])["pages"]}
    alias="安徽国控集团、国资运营公司指安徽省国有资本运营控股集团有限公司,原名为安徽省国有资产运营有限公司"
    huaan.update(reviewed_claims={"controller_alias":{"short_names":["安徽国控集团","国资运营公司"],
                        "current_name":"安徽省国有资本运营控股集团有限公司","former_name":"安徽省国有资产运营有限公司"}},
                 claim_evidence={"controller_alias":[anchor(huaan["document_id"],hp,21,alias)]},
                 append_only_claim_addendum=True,notes=["旧名与当前简称的连接来自2021年来源，不倒填为2016年已经发生的更名。"])
    oldadd=read(PREVIOUS/"claim_addenda.json")
    queries=[{"document_id":f["document_id"],"before":source_view(f,(datetime.fromisoformat(f["known_at"])-timedelta(seconds=1)).isoformat()),
              "at":source_view(f,f["known_at"])} for f in facts+[huaan]]
    conflict={"reported_document_id":"1212493239","reported_known_at":"2022-03-03T23:59:59+08:00",
              "reported_original_consideration_shares":833786629,
              "earlier_prospectus_document_id":"1212175293","earlier_original_components":[265352996,544514633],
              "earlier_component_sum":809867629,"difference":23919000,
              "same_as_other_preexisting_restricted_shares_but_no_correction_inferred":True,
              "original_literal_preserved":True,"formal_correction_notice_found":False,
              "current_rights_actor_counts_are_separate_fields":True}
    dates=["2022-01-13T23:59:59+08:00","2022-01-14T23:59:59+08:00","2022-02-15T23:59:59+08:00",
           "2022-03-03T23:59:58+08:00","2022-03-03T23:59:59+08:00",
           "2024-03-09T23:59:58+08:00","2024-03-09T23:59:59+08:00",
           "2024-09-05T23:59:58+08:00","2024-09-05T23:59:59+08:00","2024-09-11T23:59:59+08:00"]
    replay=[citic_view(facts,at) for at in dates]
    huaan_chain={"legacy_source":"1207134172","current_source":"1210115279","relationship_known_at":huaan["known_at"],
                 "original_announced_unlock":"2019-12-06","post_unlock_plan_years":2,
                 "second_anniversary_calculated":"2021-12-06","anniversary_not_reported_expiry_date":True,
                 "2021_rights_record_date":"2021-06-01","record_date_inside_two_year_calendar_interval":True,
                 "new_rights_share_coverage":"NOT_ESTABLISHED","new_rights_restricted_shares":None,
                 "frontpage_preIPO_scope_preserved":True,"source_words_not_upgraded_to_blanket_transfer_ban":True,
                 "controller_alias_evidence":huaan["claim_evidence"]["controller_alias"],
                 "extra_full_pages_read":[{"page":p,"literal":hp[p]} for p in (21,63)]}
    omitted=read(OUT/"local_catalogue_hits.json")["citic"]
    omitted_row=next(r for r in omitted if r["document_id"]=="1212493239")
    title_gap={"document_id":"1212493239","previous_title_role":omitted_row["title_role"],
               "title":omitted_row["title"],"finding":"H股结果公告正文同时含A股配股认购与限售字段。",
               "broader_title_screen_complete":False,
               "next_scope":"核对既有未选中H股及A+H配股结果、股份变动标题中的A股字段；不把H股数量计入A股供给。"}
    receipt_count=len(list((OUT/"receipts").glob("*.json")))+1
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_LOCK_FOLLOWUP_V1",
            "status":"PROGRESS_EXPLICIT_CITIC_121480144_RIGHTS_LOCK_AND_2024_EXTENSION_SOURCE_CHAIN",
            "new_pdf_documents":5,"new_pdf_pages":sum(r["pages"] for r in rows),"fully_read_new_documents":4,
            "new_source_pages_read":sum(len(v) for v in READ_PAGES.values()),"visual_pages_reviewed":4,
            "new_context_source_facts":5,"new_claim_addenda":1,"prior_source_facts_unchanged":453,"combined_source_facts":458,
            "source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts)+1,
            "explicit_current_rights_locked_tranche_shares":121480144,"quantity_source_known_at":"2022-03-03T23:59:59+08:00",
            "quantity_backfilled_to_february_listing":False,"preserved_legacy_quantity_conflicts":1,
            "huaan_original_unlock_date_located":True,"huaan_current_rights_quantity_resolved":False,
            "raw_http_response_receipts":receipt_count,"new_accounts":0,"new_returns":0,"new_models":0,
            "qualified_candidates":0,"independent_forward_observations":0,"effective_sellable_new_quantities_computed":0,
            "full_issuance_coverage":False,"free_float_denominator_available":False,"strict_M06_pressure":"NOT_COMPUTED",
            "T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE","orders_authorized":False,
            "delivery_package_created":False,"goal_achieved":False,
            "next_source_action":"REVIEW_OMITTED_H_AND_AH_RIGHTS_RESULT_TITLES_FOR_A_SHARE_CLAUSES"}
    outputs={"reviewed_source_facts":facts,"combined_source_facts":prior+facts,"claim_addenda":[huaan],
             "combined_claim_addenda":oldadd+[huaan],"source_publication_queries":queries,"citic_holding_timeline":replay,
             "preserved_quantity_conflict":conflict,"huaan_unlock_and_current_scope":huaan_chain,
             "mixed_title_coverage_gap":title_gap,"read_scope":read_scope,"result":result}
    for name,obj in outputs.items():save(OUT/(name+".json"),obj)
    print(json.dumps({"新来源":5,"补充来源":1,"新配股承诺限售股数":121480144,"公开日期代理":"2022-03-03",
                      "来源总数":458,"原文字段":result["source_fields_reviewed"],"回测账户":0},ensure_ascii=False))


if __name__=="__main__":finalize()
