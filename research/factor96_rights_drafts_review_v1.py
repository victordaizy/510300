"""保存7份申报稿的计划版本和一般持有条款；未批准申报不生成实际日历。"""
from copy import deepcopy
from datetime import datetime, timedelta
import csv
import json

from research.factor96_rights_drafts_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, RANGES
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import money, claim
from research.factor96_rights_preplans_review_v1 import anchor

VISUAL_PAGES=[("1216051593",23),("1217530739",20),("1221204347",21),("1221204347",23)]


def specs():
    bank=[
        claim("plan_ratio_per_10",{"value":"3","relation":"SPECIFIED"},(20,"每10股配售3股")),
        claim("plan_quantity_scenario",{"basis_asof":"2022-12-31","basis_shares":48934843657,"total_shares":14680453097,
            "A_shares":10215804204,"H_shares":4464648893,"scope":"A_AND_H_COMBINED","conditional_on_basis":True},
            (20,"截至2022年12月31日的总股本48,934,843,657股"),(20,"本次配售股份数量总计14,680,453,097股"),
            (20,"A股配股股数合计为10,215,804,204股"),(20,"H股配股股数合计为4,464,648,893股")),
        claim("plan_gross_proceeds_cap",{**money("400"),"scope":"A_AND_H_COMBINED"},(20,"本次配股募集资金不超过人民币400亿元(含400亿元)")),
        claim("remaining_approvals",["SSE_REVIEW","CSRC_REGISTRATION"],(19,"本行本次配股尚需获得上交所审核通过及中国证监会同意注册")),
        claim("bank_regulatory_approval_key","银保监复〔2022〕751号",(19,"银保监复〔2022〕751号")),
        claim("analysis_report_shareholder_review",{"status":"PENDING","planned_meeting_date":"2023-04-12"},
            (19,"关于中信银行股份有限公司向原A股股东配售股份的论证分析报告的议案"),(19,"拟提交本行于2023年4月12日召开的")),
        claim("net_proceeds_not_determined",True,(21,"预计募集资金净额为【】亿元")),
        claim("final_price_not_determined",True,(20,"最终配股价格由董事会或其授权人士在发行前根据市场情况与保荐机构(承销商)协商确定")),
        claim("calendar_template",{"anchor":"T","record_date":None,"payment_start_date":None,"payment_end_date":None,
            "payment_relative_start":1,"payment_relative_end":5,"result_relative_day":7,"actual_calendar_admitted":False},
            (22,"【】月【】日(T日)A股配股股权登记"),(22,"(T+1-T+5日)A股配股缴款起止日期"),(22,"(T+7日)刊登A股配股发行结果公告")),
        claim("general_holding_clause",{"general_rule":"NO_HOLDING_PERIOD","regulatory_exceptions":True,"effective_sellable_shares":None},
            (23,"除相关法律法规规定外,本次发行的股票不设持有期限制")),
        claim("resolution_validity",{"months":12,"trigger":"股东大会及A/H类别股东会审议通过","calendar_end":None},
            (22,"自本行股东大会、A股类别股东会和H股类别股东会审议"),(23,"通过之日起12个月内有效")),
        claim("proposed_use","核心一级资本",(21,"将全部用于补充本行的核心一级资本")),
    ]
    output=[{"index":0,"group":"CITIC_BANK_PLAN","claims":bank}]
    for i in range(1,7):
        quantity_page=19 if i<5 else 20
        calendar_page=21 if i<5 else 22
        holding_page=23 if i==6 else 22
        basis_date="2023-03-31" if i==1 else "2023-06-30" if i<5 else "2023-12-31"
        basis=3005171030 if i<5 else 3004071338
        shares=1051809860 if i<5 else 1051424968
        capital=3005171030 if i<5 else 3004071338 if i==5 else 2947095201
        amount,unit=("85","亿元") if i==1 else ("778769.00","万元") if i<5 else ("498000.00","万元")
        display=amount if unit=="亿元" else f"{float(amount):,.2f}"
        ratio_literal="每10股不超过3.5股" if i==1 else "每10股配售3.5股"
        yy,mm,dd=map(int,basis_date.split("-"))
        claims=[
            claim("plan_ratio_per_10",{"value":"3.5","relation":"CAP" if i==1 else "SPECIFIED"},(quantity_page,ratio_literal)),
            claim("plan_quantity_scenario",{"basis_asof":basis_date,"basis_shares":basis,"total_shares":shares,
                "A_shares":shares,"H_shares":None,"scope":"A_ONLY","conditional_on_basis":True},
                (quantity_page,f"截至{yy}年{mm}月{dd}日的总股本{basis:,}股"),(quantity_page,f"{shares:,}股")),
            claim("plan_gross_proceeds_cap",{**money(amount,unit),"scope":"A_ONLY"},(20,f"预计募集资金总额不超过人民币{display}{unit}")),
            claim("remaining_approvals",["SSE_REVIEW","CSRC_REGISTRATION"],(19,"本次发行尚需上交所审核通过并经中国证监会作出予以注册决定后方可实施")),
            claim("issuer_profile_registered_capital",money(str(capital),"元"),(14,f"注册资本:{capital:,}元")),
            claim("net_proceeds_not_determined",True,(20,"预计募集资金净额为【】万元")),
            claim("final_price_not_determined",True,(20,"最终配股价格由董事会或其授权人士根据股东大会的授权,在发行前根据市场情况与保荐机构(主承销商)协商确定")),
            claim("calendar_template",{"anchor":"R","record_date":None,"payment_start_date":None,"payment_end_date":None,
                "payment_relative_start":1,"payment_relative_end":5,"result_relative_day":7,"actual_calendar_admitted":False},
                (calendar_page,"【】年【】月【】日(R日)股权登记日"),(calendar_page,"(R+1日至R+5日)配股缴款起止日期"),
                (22,"(R+7日)刊登发行结果公告")),
            claim("general_holding_clause",{"general_rule":"NO_HOLDING_PERIOD","regulatory_exceptions":True,"effective_sellable_shares":None},
                (holding_page,"除相关法律法规规定外,本次发行的证券不设持有期限制")),
            claim("proposed_use","供应链运营补充流动资金及偿还银行借款",(20 if i<5 else 21,"拟全部用于公司供应链运营业务补充流动资金及偿还银行借款")),
        ]
        if i<6:
            claims.append(claim("resolution_validity",{"months":12,"trigger":"股东大会审议通过","calendar_end":None},
                (20 if i<5 else 21,"自上市公司股东大会审议通过之日起12个月内有效")))
        else:
            claims.append(claim("resolution_validity",{"extension_months":12,"reported_approval_date":"2024-05-06","calendar_end":"2025-05-21",
                "clock_note":"此为2024年9月申报稿所报告的历史决定，仅从本来源可得时点保存。"},
                (19,"2024年5月6日,公司召开2023年年度股东大会审议通过前述相关议案"),
                (21,"有效期自原期限届满之日起延长12个月,即延长至2025年5月21日")))
        if i==5:
            claims.append(claim("validity_extension_shareholder_approval","PENDING",(19,"前述议案尚需公司拟于2024年5月召开的2023年年度股东大会审议通过")))
        output.append({"index":i,"group":"JIANFA_PLAN_2023","claims":claims})
    return output


def build():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    targets=read(OUT/"targets.json");facts=[]
    for spec in specs():
        row=targets[spec["index"]];doc=Document({**row["source"],"review_role":"FILING_DRAFT_OPERATIVE_SOURCE_REVIEW"})
        f=base_fact(doc,None)
        f.update(plan_source_group=spec["group"],reviewed_claims={},claim_evidence={},full_body_reviewed=False,
            operative_clause_review_complete=True,operative_page_range=list(RANGES[row["document_id"]]),
            actual_issuance_calendar=None,actual_issue_price=None,actual_issuance_quantity=None,effective_sellable_new_shares=None,strict_M06_pressure=None,
            notes=["申报稿仅提供条件计划；以目录日末为可得日期代理，未证明历史首次公开。",
                "定价原则、留白日期和相对交易日日程不转成实际价格/日期；一般不设持有期与法规例外一起保存。",
                "已读范围为当次发行概况中的候选及必要跨页原文，不声明全文或经营数据已获验证。"])
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            assert c["field"] not in f["reviewed_claims"]
            f["reviewed_claims"][c["field"]]=c["value"]
            f["claim_evidence"][c["field"]]=[anchor(row["document_id"],pages,p,t) for p,t in c["proof"]]
        facts.append(f)
    return facts


def source_view(fact,at):
    visible=fact["known_at"]<=at
    return {"document_id":fact["document_id"],"as_of":at,"known_at":fact["known_at"],
        "status":"REVIEWED_FILING_DRAFT_DATE_PROXY_ONLY" if visible else "NO_VIEW",
        "reviewed_claims":deepcopy(fact["reviewed_claims"]) if visible else {},"actual_event_id":None,
        "actual_issuance_calendar":None,"effective_sellable_new_shares":None,"strict_M06_pressure":None,"trading_feature_admitted":False}


def comparisons(facts,prior):
    idx={f["document_id"]:f for f in prior};pairs=[]
    for i,rid in [(2,"1217530737"),(3,"1217766946"),(4,"1217834031"),(5,"1219892999"),(6,"1221204348")]:
        f,a=facts[i],idx[rid]
        assert f["symbol"]==a["symbol"] and f["known_at"]==a["known_at"]
        pairs.append({"draft_id":f["document_id"],"admin_id":rid,"relationship_known_at":f["known_at"],
            "source_relationship":"同发行人同日申报更新提示及申报稿；不是两笔发行。","old_admin_unchanged":True,"actual_event_id":None})
    versions=[]
    for a,b in zip(facts[1:-1],facts[2:]):
        ca,cb=a["reviewed_claims"],b["reviewed_claims"]
        keys=sorted(set(ca)|set(cb))
        versions.append({"earlier_source":a["document_id"],"later_source":b["document_id"],"relationship_known_at":b["known_at"],
            "changed_source_fields":{k:{"earlier":ca.get(k),"later":cb.get(k)} for k in keys if ca.get(k)!=cb.get(k)},
            "gross_budget_change_cents":cb["plan_gross_proceeds_cap"]["amount_cents"]-ca["plan_gross_proceeds_cap"]["amount_cents"],
            "conditional_A_quantity_change":cb["plan_quantity_scenario"]["A_shares"]-ca["plan_quantity_scenario"]["A_shares"],
            "new_executed_supply_inferred":False,"missing_field_means_revocation":False})
    old=idx["1215952169"];new=facts[0]
    bank={"preplan_id":old["document_id"],"draft_id":new["document_id"],"relationship_known_at":new["known_at"],
        "budget_unchanged":old["reviewed_claims"]["plan_gross_proceeds"]["amount_cents"]==new["reviewed_claims"]["plan_gross_proceeds_cap"]["amount_cents"],
        "quantity_unchanged":old["reviewed_claims"]["plan_quantity_scenarios"][0]["reported_total_shares"]==new["reviewed_claims"]["plan_quantity_scenario"]["total_shares"],
        "new_general_holding_clause_with_regulatory_exceptions":True,"actual_event_id":None}
    assert bank["budget_unchanged"] and bank["quantity_unchanged"]
    profile={"source_id":facts[6]["document_id"],"known_at":facts[6]["known_at"],
        "registered_capital_cny":2947095201,"quantity_scenario_basis_date":"2023-12-31","quantity_scenario_basis_shares":3004071338,
        "quantity_scenario_reported_A_shares":1051424968,"automatic_rebase_applied":False,
        "interpretation":"发行人基本资料中的注册资本与发行测算采用的历史股本是不同字段；不擅自用前者改写配股数量。"}
    return pairs,versions,bank,profile


def finalize():
    facts=build();prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(facts)==7 and len(prior)==414 and not {f["document_id"] for f in facts}&{f["document_id"] for f in prior}
    pairs,versions,bank,profile=comparisons(facts,prior)
    save(OUT/"reviewed_draft_source_facts.json",facts);save(OUT/"combined_source_facts.json",prior+facts)
    reviewed=[]
    for r in read(OUT/"source_candidates.json"):
        for j,c in enumerate(r["candidates"]):
            reviewed.append({"document_id":r["document_id"],"candidate_index":j,**c,"reviewed":True,
                "review_scope":"当次发行计划、待批条件、留白日历及上市持有条款；机构联系方式等尾部内容不用于研究。"})
    assert len(reviewed)==30
    save(OUT/"reviewed_source_candidates.json",reviewed)
    for name,value in [("same_day_admin_comparisons",pairs),("jianfa_version_comparisons",versions),
        ("citic_bank_preplan_comparison",bank),("jianfa_profile_and_basis_scope",profile)]:save(OUT/(name+".json"),value)
    queries=[]
    for f in facts:
        at=f["known_at"];before=(datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
        queries.append({"document_id":f["document_id"],"before":source_view(f,before),"at":source_view(f,at)})
    save(OUT/"draft_publication_queries.json",queries)
    with (OUT/"7份申报稿发行条款.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["公告ID","证券","可得日期代理","标题","计划来源组","结构化字段","来源URL"]);writer.writeheader()
        for f in facts:writer.writerow({"公告ID":f["document_id"],"证券":f["symbol"],"可得日期代理":f["known_at"],"标题":f["title"],
            "计划来源组":f["plan_source_group"],"结构化字段":json.dumps(f["reviewed_claims"],ensure_ascii=False),"来源URL":f["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_DRAFTS_V1","status":"7_DRAFT_OPERATIVE_SECTIONS_REVIEWED_NO_ACTUAL_CALENDAR_CREATED",
        "documents":7,"target_pdf_pages":2237,"operative_page_span_total":62,"full_pdf_pages_read_claimed":False,
        "candidate_fragments_read":30,"candidate_characters":14686,"source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts),
        "source_plan_groups":2,"jianfa_versions":6,"jianfa_adjacent_version_pairs":5,"same_day_update_notice_pairs":5,
        "new_general_holding_clauses_with_regulatory_exceptions":7,"effective_sellable_quantities_computed":0,
        "pending_review_and_registration_sources":7,"blank_calendar_templates_excluded":7,"visual_pages_reviewed":len(VISUAL_PAGES),
        "prior_source_documents_unchanged":414,"combined_source_documents":421,"new_actual_issuance_calendars":0,
        "new_publication_queries":14,"remaining_response_documents":18,"full_issuance_coverage":False,
        "earlier_holding_clause_full_coverage":False,"free_float_denominator_available":False,"strict_M06_pressure":"NOT_COMPUTED",
        "T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE","new_network_requests":0,"new_accounts":0,"new_returns":0,
        "new_models":0,"qualified_candidates":0,"independent_forward_observations":0,"orders_authorized":False,
        "delivery_package_created":False,"goal_achieved":False,
        "next_source_action":"T13_18_REGULATORY_RESPONSE_REPORTS_AND_EARLIER_HOLDING_COVERAGE"}
    save(OUT/"result.json",result)
    print(json.dumps({"申报稿":7,"来源字段":result["source_fields_reviewed"],"一般持有条款":7,"剩余回复":18,"新增实际日历":0},ensure_ascii=False))


if __name__=="__main__":
    finalize()
