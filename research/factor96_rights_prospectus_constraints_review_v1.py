"""保存已读条款与仍待阅读的候选；旧融资约束不自动赋给当次配股。"""
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

from research.factor96_rights_prospectus_constraints_v1 import OUT, PREVIOUS, PROSPECTUS, ROOT, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import claim
from research.factor96_rights_preplans_review_v1 import anchor

VISUAL_PAGES=[("1207347972",106),("1207347972",107),("1215110331",133),("1215110331",134)]
# 这些文档中优先规则以外的候选也已逐段阅读。
EXTRA_REMAINDER_DOCUMENTS={"1203703598","1203703599","1215110331"}
NOTES={
0:"2013非公开发行、2014转增及旧IPO承诺；另补读第214页确认跨页承诺为2013年非公开发行。",
1:"发行前股份结构。",2:"发行前股份结构及历史受限股。",3:"发行前股份结构及历史受限股。",
5:"2007重组股改与2015年存量结构分开；2012年非公开发行60个月承诺包含衍生股份，是否覆盖当前配股需另核。",
6:"旧2014增持承诺与本次配股6个月承诺分开；后者与此前当前章节已保存条款相同。",
8:"2015年末既有股份结构，不是本次新增股分类。",9:"2015年末既有股份结构，不是本次新增股分类。",
10:"发行前股权结构、已履行IPO和2015定增锁定、2014减持数量限制，与当前全额认购承诺分开。",
13:"2015年7月至2016年1月的不减持承诺，不当作2017年配股后6个月。",
15:"同一旧2015年承诺在修订说明书重述，不产生新承诺。",
16:"发行前股份结构及服务器、机箱专利的机械锁定；专利词不计股份约束。",
17:"发行前股份结构，不是本次新增股分类。",
18:"2015资产重组锁定、旧增持6个月承诺、2017配股全额认购分别保存，不凭认购承诺推出锁定。",
20:"发行前持股、旧集团延长期及2015增持短线交易承诺，与当前配股分开。",
22:"历年股本变动、旧2013非公开发行解禁及发行前结构，不转成当次新增供给。",
25:"宏广计划、2015增持及董监高关联账户约束属于既有持股背景，不能自动判断当前配股适用或已经到期。",
26:"2016重组股东旧股份锁定至2020年6月6日；另有铝价、汇率套保锁定，不等于配股锁定。",
28:"发行前A/H股和原有限售结构，不能替代新获配股份分类。",
31:"2014和2015利润分配或增持承诺，不是本次配股新承诺。",
35:"发行前结构、原股东已减持、旧解除限售及2017增持承诺，与当前认购计划分开。",
36:"2015非公开发行承诺已履行；另一2016年10月其他承诺条目含配股后6个月不减持，对应哪次配股尚未核实。",
38:"发行前股本结构，不把老股限售数量乘当前比例作实际受限新增股。",
39:"发行前结构和IPO的48、36、12个月承诺；当前配股适用性未由这些段落证明。",
40:"发行前结构；2014非公开发行和2015增持锁定报告已履行，与当次配股分开。",
41:"发行前结构及回购专户数量，不是新配股约束。",42:"发行前结构，不是新增股结构。",
43:"发行前结构、2018增持限制和2019计划减持，未给当次配股锁定结论。",
45:"旧H股IPO、股价异常期间不减持及当前全额认购分开；异常状态是否持续不据此判断。",
48:"2016员工持股两期锁定，不是当次配股承诺。",
49:"2007股改、2009增发、2016员工持股和非公开发行、2019董事持股分别保留；旧非公开发行表格36/60个月列差异未用于推导。",
50:"发行前股本结构，不是新增股分类。",
51:"发行前结构、IPO锁定和2019年8月起一年附加锁定；不据此认定2020配股受限股数。",
52:"发行前股本及股东结构。",53:"发行前股本及股东结构。",54:"发行前股东结构。",
55:"发行前股东结构及旧IPO锁定、旧IPO募集资金，不归给当次配股。",
56:"历史转增及可转债、发行前股份结构、2020年非公开发行华侨银行五年锁定分别保存。",
57:"发行前股权结构，与新获配股份分类分开。",58:"发行前结构和回购专户数量。",
59:"发行前结构及已经履行的2014非公开发行60个月承诺。",
60:"发行前结构、股改承诺、收购广州证券相关承诺及新增股份属于不同事项；当前全额认购不等于新增锁定。",
61:"发行前A/H股及有限售结构，不是当前获配股份约束。",64:"发行前股份结构。",
65:"IPO承诺履行风险及锁定届满后两年减持价格/数量约束，当前配股适用性未确认。",
67:"2017非公开发行48个月锁定和违反关联交易承诺后的条件性不得转让，与当前全额认购分开。",
70:"2022年激励股票回购注销，不是本次新增股锁定。",
71:"2014IPO、2020非公开发行、2021可转债承诺、激励回购注销及原股本分开；2021不减持条目报告已于2021年底履行完毕，不能当2022配股承诺的独立佐证。",
72:"2022年末发行前结构、2017境外优先股和旧IPO锁定，未推出新配股限售。",
73:"发行前结构和2017境外优先股，不归为当次配股供给。",
}


def specs():
    return [
      {"index":5,"claims":[
        claim("legacy_private_placement_lock",{"actor":"亚泰集团","origin_year":2012,"reported_subscription_value":"10415.6064",
            "reported_subscription_unit":"万股","original_subscription_shares":104156064,"months":60,
            "trigger":"东北证券非公开发行股份上市之日起","calendar_start":None,"calendar_end":None,
            "derived_shares_clause_present":True,"applies_to_current_rights_allotment":"NOT_ESTABLISHED"},
            (151,"2012年非公开发行新增股份自愿锁定的承诺"),(151,"本次认购取得东北证券10,415.6064万股"),
            (151,"即自东北证券非公开发行股份上市之日起60个月内不上市交易或转让"),
            (151,"在锁定期内,因本次发行的股份而产生的任何股份(包括但不限于股份拆细、派送红股等方式增持的股份)也不转让或上市交易")),
        claim("preissue_restricted_share_snapshot",{"as_of":"2015-09-30","restricted_existing_shares":208312128,"new_rights_shares":False},
            (55,"截至2015年9月30日,公司股本结构如下"),(55,"一、有限售条件股份208,312,12810.64"))]},
      {"index":26,"claims":[
        claim("legacy_acquisition_lock",{"actor":"山东怡力电业有限公司","origin":"2016发行股份购买资产","base_months":36,
            "conditional_extension_months":6,"extension_reported_triggered":True,"reported_release_date":"2020-06-06",
            "applies_to_current_rights_allotment":"NOT_ESTABLISHED"},
            (75,"2016年发行股份购买资产暨关联交易事项中作出重要承诺"),
            (77,"自上市之日起三十六个月内不得转让"),(77,"其持有上市公司股票的锁定期自动延长六个月"),
            (77,"怡力电业持有的上市公司股票锁定期自动延长6个月,解除限售日期为2020年6月6日")),
        claim("preissue_holder_snapshot",{"as_of":"2018-06-30","holder":"山东怡力电业有限公司","existing_shares":2163141993,
            "classification":"限售流通A股","new_rights_shares":False},
            (32,"截至2018年6月30日"),(32,"山东怡力电业有限公司2,163,141,99323.38限售流通A股"))]},
      {"index":36,"claims":[
        claim("ambiguous_plan_scope_commitment",{"actor":"浪潮集团","reported_commitment_month":"2016-10","table_type":"其他承诺",
            "table_validity_label":"长期有效","literal_trigger":"自浪潮信息配股完成后6个月内不减持","months":6,
            "applicable_issue":"UNRESOLVED","calendar_end":None,"new_restricted_shares":None,
            "automatically_assigned_to_2020_rights":False,"backfilled_to_2017_publication":False},
            (106,"浪潮集团其他承诺"),(106,"2016年10月长期有效"),(107,"自浪潮信息配股完成后6个月内不减持"))]},
      {"index":56,"claims":[
        claim("legacy_private_placement_lock",{"actor":"新加坡华侨银行有限公司","reported_commitment_date":"2020-05-15","years":5,
            "trigger":"股份上市之日起","scope":"本次非公开发行所认购的股份","calendar_start":None,"calendar_end":None,
            "applies_to_current_rights_allotment":"NOT_ESTABLISHED"},
            (65,"新加坡华侨银行有限公司股份限售承诺自股份上市之日起5年内不转让本次非公开发行所认购的股份2020年5月15日5年正常履行中"))]},
      {"index":67,"claims":[
        claim("legacy_private_placement_lock",{"actor":"申能集团","origin_year":2017,"months":48,
            "trigger":"2017年非公开发行A股股票并上市后","calendar_end":None,"applies_to_current_rights_allotment":"NOT_ESTABLISHED"},
            (123,"与2017年非公开发行相关的承诺股份限售申能集团"),(123,"在本公司2017年非公开发行A股股票并上市后48个月")),
        claim("conditional_transfer_sanction",{"actor":"申能集团","condition":"违反规范关联交易承诺",
            "restriction":"所持发行人股份不得转让直至相应措施实施完毕","condition_activated":"NOT_VERIFIED",
            "restricted_new_rights_shares":None},
            (123,"申能集团在违反相关承诺发生之日起停止在发行人处取得股东分红,同时持有的发行人股份不得转让,直至按上述承诺采取相应的措施并实施完毕为止"))]},
      {"index":71,"claims":[
        claim("legacy_private_placement_lock",{"actor":"中国节能","reported_commitment_date":"2020-07-28","months":36,
            "scope":"非公开发行认购股份","reported_calendar_start":"2020-09-02","reported_calendar_end":"2023-09-01",
            "reported_fulfilment":"履行中","applies_to_current_rights_allotment":"NOT_ESTABLISHED"},
            (133,"2020年7月28日,中国节能出具《股份限售承诺》"),
            (133,"通过本次非公开发行认购的公司股份自该等股份上市之日起36个月内"),
            (134,"股份限售承诺中国节能2020年7月28日2020年9月2日至2023年9月1日承诺正在履行中")),
        claim("legacy_convertible_non_reduction_commitments",{"origin":"2021年度公开发行可转换公司债券","reported_commitment_date":"2021-04-08",
            "groups":["中国节能及控制关系关联方、一致行动人","全体董事、监事和高级管理人员及一致行动人"],
            "relative_start":"本次发行定价基准日（发行期首日）前六个月","relative_end":"本次发行完成后六个月内",
            "reported_calendar_start":"2020-12-21","reported_calendar_end":"2021-12-25","reported_fulfilment":"履行完毕",
            "applies_to_2022_rights":"NOT_INFERRED","2022_response_revoked_inferred":False},
            (133,"2021年度公开发行可转换公司债券相关承诺"),(133,"2021年4月8日,中国节能出具"),
            (133,"2021年4月8日,发行人全体董事、监事和高级管理人员出具"),
            (133,"自节能风电本次发行定价基准日(发行期首日)前六个月至本次发行完成后六个月内"),
            (134,"中国节能2021年4月8日2020年12月21日至2021年12月25日承诺已履行完毕"),
            (134,"发行人全体董事、监事和高级管理人员2021年4月8日2020年12月21日至2021年12月25日承诺已履行完毕"))]},
    ]


def build_facts():
    targets=read(OUT/"targets.json");source_targets={r["document_id"]:r for r in read(PROSPECTUS/"targets.json")};facts=[]
    for spec in specs():
        row=targets[spec["index"]];src=source_targets[row["document_id"]]["source"]
        doc=Document({**src,"review_role":"PROSPECTUS_LEGACY_OR_AMBIGUOUS_HOLDING_CONTEXT"});f=base_fact(doc,None)
        f.update(reviewed_claims={},claim_evidence={},full_body_reviewed=False,
            effective_sellable_new_shares=None,strict_M06_pressure=None,
            notes=[NOTES[spec["index"]],"在当前来源可得时点保存原文所报告的旧事实，不反向填入早年时点。",
                   "旧承诺可能涉及存量股份；是否覆盖本次配股获配股份尚未成立，不自动相减、归零或判断失效。"])
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            f["reviewed_claims"][c["field"]]=c["value"]
            f["claim_evidence"][c["field"]]=[anchor(row["document_id"],pages,p,t) for p,t in c["proof"]]
        facts.append(f)
    return facts


def source_view(f,at):
    return {"document_id":f["document_id"],"known_at":f["known_at"],"as_of":at,
        "status":"SOURCE_CONTEXT_ONLY" if f["known_at"]<=at else "NO_VIEW",
        "reviewed_claims":deepcopy(f["reviewed_claims"]) if f["known_at"]<=at else {},"event_id":None,
        "effective_sellable_new_shares":None,"trading_feature_admitted":False}


def finalize():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    facts=build_facts();prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(facts)==6 and len(prior)==439 and not {f["document_id"] for f in facts}&{f["document_id"] for f in prior}
    current=read(OUT/"source_candidates.json");front=read(OUT/"frontmatter_candidates.json");tail=read(OUT/"remainder_candidates.json")
    previous_clauses={r["document_id"]:r for r in read(PROSPECTUS/"holding_restriction_clause_reviews.json")}
    reviewed=[];pending=[]
    for i,r in enumerate(current):
        for j,c in enumerate(r["candidates"]):
            reviewed.append({"document_id":r["document_id"],"candidate_set":"CURRENT_SECTION","candidate_index":j,**c,"reviewed":True,
                "semantic_class":previous_clauses[r["document_id"]]["status"],"reused_previously_saved_clause":True,
                "note":"一般持有期与例外同存；长春控股股东承诺已有来源，此次关键词扩展未新增当前章节条款。"})
    front_notes={3:"术语表大小非定义，不是当次发行条款。",10:"客户融资担保物限售风险，不是发行人新股。",11:"客户融资担保物限售风险，不是发行人新股。",
        26:"铝价套期保值锁定，不是股份限售。",27:"铝价套期保值锁定，不是股份限售。",
        34:"原股东已经减持及认购承诺待批，不是新的不减持承诺。",35:"原股东已经减持及认购承诺待批，不是新的不减持承诺。"}
    for i,r in enumerate(front):
        for j,c in enumerate(r["candidates"]):reviewed.append({"document_id":r["document_id"],"candidate_set":"FRONTMATTER","candidate_index":j,
            **c,"reviewed":True,"semantic_class":"NOT_NEW_ISSUE_HOLDING_CLAUSE","note":front_notes[i]})
    for i,r in enumerate(tail):
        for j,c in enumerate(r["candidates"]):
            item={"document_id":r["document_id"],"symbol":r["symbol"],"candidate_set":"REMAINDER","candidate_index":j,**c}
            if c["priority"] or r["document_id"] in EXTRA_REMAINDER_DOCUMENTS:
                reviewed.append({**item,"reviewed":True,"semantic_class":"CONTEXT_REVIEWED_NO_AUTOMATIC_CURRENT_ISSUE_ASSIGNMENT","note":NOTES[i]})
            else:pending.append(item)
    assert len(reviewed)==133 and len(pending)==248 and len(reviewed)+len(pending)==381
    queries=[{"document_id":f["document_id"],"before":source_view(f,(datetime.fromisoformat(f["known_at"])-timedelta(seconds=1)).isoformat()),
              "at":source_view(f,f["known_at"])} for f in facts]
    old_wind=next(f for f in prior if f["document_id"]=="1213982762");new_wind=facts[-1]
    comparison={"response_document_id":old_wind["document_id"],"response_known_at":old_wind["known_at"],
        "response_context":"2022年配股监管回复称本次配股，未在承诺引文中写明出具日期。",
        "response_claims_preserved":deepcopy(old_wind["reviewed_claims"]["holding_commitments"]),
        "prospectus_document_id":new_wind["document_id"],"prospectus_known_at":new_wind["known_at"],
        "prospectus_context":"承诺表明确归于2021年可转债，并写明旧承诺履行完毕。",
        "prospectus_claims":deepcopy(new_wind["reviewed_claims"]["legacy_convertible_non_reduction_commitments"]),
        "relationship_known_at":new_wind["known_at"],"independent_confirmation_of_2022_commitment":False,
        "2022_commitment_revocation_proved":False,"2022_commitment_effective_period_resolved":False,
        "effective_sellable_new_shares":None,"missing_2022_clause_means_revocation":False,
        "next_evidence":"2022年配股对应原始承诺函或明确将主体、事项、有效期连接起来的同期来源。"}
    for name,obj in [("reviewed_context_source_facts",facts),("combined_source_facts",prior+facts),("reviewed_candidates",reviewed),
        ("source_publication_queries",queries),("windpower_response_vs_prospectus_scope",comparison)]:save(OUT/(name+".json"),obj)
    save(OUT/"remaining_holding_candidates.json",{"count":len(pending),"candidates":pending,
        "scope":"固定74份正式说明书全文补扫中的未读候选；非随机样本，尚未判断有无当前约束。"})
    with (OUT/"6份存量与歧义约束来源.csv").open("x",encoding="utf-8-sig",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=["公告ID","证券","可得日期代理","结构化字段","说明","来源URL"]);writer.writeheader()
        for f in facts:writer.writerow({"公告ID":f["document_id"],"证券":f["symbol"],"可得日期代理":f["known_at"],
            "结构化字段":json.dumps(f["reviewed_claims"],ensure_ascii=False),"说明":"；".join(f["notes"]),"来源URL":f["source_url"]})
    targets=read(OUT/"targets.json")
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_PROSPECTUS_CONSTRAINTS_V1","status":"PROGRESS_PRIORITY_CONTEXTS_REVIEWED_248_CANDIDATES_PENDING",
        "documents_scanned":74,"target_pdf_pages":sum(len(read(r["text_path"])["pages"]) for r in targets),
        "full_text_scanned_by_keyword":True,"full_text_read_claimed":False,"current_section_candidates":18,"frontmatter_candidates":7,
        "remainder_candidates":356,"priority_remainder_read":98,"additional_remainder_read":10,
        "candidate_fragments_read":len(reviewed),"candidate_characters_read":sum(len(c["literal"]) for c in reviewed),
        "remaining_candidates":len(pending),"new_context_source_facts":6,"source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts),
        "prior_source_facts_unchanged":439,"combined_source_facts":445,"visual_pages_reviewed":4,"new_publication_queries":12,
        "new_confirmed_current_issue_holding_clauses":0,"ambiguous_inspur_plan_scope":True,"windpower_old_convertible_not_independent_confirmation":True,
        "complete_holding_review":False,"effective_sellable_quantities_computed":0,"full_issuance_coverage":False,
        "free_float_denominator_available":False,"strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_accounts":0,"new_returns":0,"new_models":0,"new_network_requests":0,"qualified_candidates":0,"independent_forward_observations":0,
        "orders_authorized":False,"delivery_package_created":False,"goal_achieved":False,
        "next_source_action":"REVIEW_REMAINING_248_PROSPECTUS_HOLDING_CONTEXTS_AND_RESOLVE_COMMITMENT_IDENTITY"}
    save(OUT/"result.json",result)
    print(json.dumps({"已读片段":133,"待读片段":248,"新增来源":6,"字段":result["source_fields_reviewed"],"确认新当次约束":0},ensure_ascii=False))


if __name__=="__main__":finalize()
