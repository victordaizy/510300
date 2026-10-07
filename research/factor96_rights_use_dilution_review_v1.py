"""复核15份用途和摊薄资料，保留原文冲突及测算假设，不回填实际发行。"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

from research.factor96_rights_use_dilution_v1 import OUT, PREVIOUS, SOURCE, ROOT, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import money, claim
from research.factor96_rights_preplans_review_v1 import anchor

VISUAL_PAGES=[("1203978538",2),("1204532638",4),("1205275253",4),("1209640417",1),
              ("1205272663",2),("1201732322",2),("1209640416",1)]


def share_amount(value, unit="股"):
    scale = Decimal(10000 if unit == "万股" else 1)
    places = max(0,-Decimal(value).as_tuple().exponent)
    nominal = Decimal(value)*scale
    assert nominal == nominal.to_integral_value()
    return {"reported_value":value,"reported_unit":unit,"nominal_shares":int(nominal),
        "reported_decimal_places":places,"unit_resolution_shares":str(scale/(Decimal(10)**places)),
        "interpretation":"按原文单位转换的测算数；万股小数保留原精度，不充当精确的实际认购股数。"}


def allocations(unit, *items):
    return [{"project":name,"proposed_amount":money(value,unit)} for name,value in items]


def specs():
    rows = {}

    def add(i, group, note, *claims):
        rows[i] = {"index":i,"group":group,"note":note,"claims":list(claims)}

    add(0,"CHANGCHUN_2015","10月28日用途来源比所对照的11月13日国资事项公告早；不回填后来的国资批准。",
        claim("plan_gross_proceeds_cap",money("180000","万元"),(2,"本次配股发行募集的资金总额预计不超过180,000万元")),
        claim("proposed_use_components",allocations("万元",("疫苗生产基地I期","40000.00"),("新产品研发","80000.00"),("补充流动资金","60000.00")),
            (2,"单位:万元"),(2,"疫苗生产基地I期项目40,000.00"),(2,"新产品研发投入80,000.00"),(2,"补充流动资金60,000.00")),
        claim("vaccine_project_total_investment",money("64412.33","万元"),(3,"项目总投资64,412.33万元")))
    add(1,"ENN_2017","23亿元是募集预算；37.630587亿元是项目总投资，不能混为融资额。",
        claim("plan_gross_proceeds_cap",money("230000","万元"),(2,"募集资金总额预计不超过人民币230,000万元")),
        claim("proposed_use_components",allocations("万元",("年产20万吨稳定轻烃","230000.00")),
            (2,"将全部用于“年产20万吨稳定轻烃”项目"),(2,"拟以本次配股的募集资金投入230,000.00万元")),
        claim("project_total_investment",money("376305.87","万元"),(2,"项目总投资额376,305.87万元")))
    add(2,"ENN_2017","2017年10月31日、全额认购及23亿元到账都是摊薄测算假设；未创建实施日历。",
        claim("source_plan_proceeds_cap",money("230000","万元"),(1,"拟募集资金总量不超过230,000万元")),
        claim("source_plan_ratio_cap_per_10","3",(1,"每10股配售不超过3股")),
        claim("dilution_assumed_proceeds",money("230000","万元"),(2,"假设本次配股的募集资金总额(不考虑发行费用)为人民币230,000万元")),
        claim("dilution_completion_assumption",{"reported":"2017年10月31日","precision":"DAY","date":"2017-10-31","actual_calendar_admitted":False},
            (2,"假设本次配股方案于2017年10月31日实施完成"),(2,"上述配股完成时间仅用于计算本次发行对摊薄即期回报的影响")),
        claim("dilution_quantity_assumption",{"basis_asof":"2016-12-31","ratio_per_10":"3","basis":share_amount("985785043"),"new":share_amount("295735512")},
            (2,"假设本次股权登记日后的所有股东均参与本次配股"),(2,"配股比例为每10股配3股"),(2,"截至2016年12月31日的总股本985,785,043股"),(2,"本次配售股份数量为295,735,512股")))
    for i,budget,debt in [(3,"625000","159000"),(7,"566000","100000")]:
        add(i,"WANXIANG_2017","使用计划按5项募集投入合计，总投资与拟使用募集资金分开。",
            claim("plan_gross_proceeds_cap",money(budget,"万元"),(1,f"募集资金总额预计不超过人民币{int(budget):,}万元")),
            claim("proposed_use_components",allocations("万元",("3000万套轮毂轴承","200000"),("智慧工厂","184000"),("技术研发","72000"),("偿还债券",debt),("补充流动资金","10000")),
                (1,"拟使用募集资金投入(万元)"),(1,"252,150200,000"),(1,"350,835184,000"),(1,"96,36072,000"),
                (1,f"偿还公司债券159,000{int(debt):,}"),(1,"补充流动资金10,00010,000")),
            claim("project_total_investment",money("868345","万元"),(1,"项目总投资(万元)"),(1,f"合计868,345{int(budget):,}")))
    add(4,"WANXIANG_2017","摊薄说明写62.6亿元，与同日预案和用途报告62.50亿元不一致；保留冲突。股数以万股两位小数报告。",
        claim("dilution_assumed_proceeds",money("62.6"),(2,"本次配股募集资金总额不超过62.6亿元")),
        claim("dilution_completion_assumption",{"reported":"2017年10月31日","precision":"DAY","date":"2017-10-31","actual_calendar_admitted":False},
            (2,"假设本次配股于2017年10月31日实施完成"),(2,"该完成时间仅用于计算本次配股摊薄即期回报对公司每股收益的影响")),
        claim("dilution_quantity_assumption",{"basis_asof":"2017-06-30","ratio_per_10":"3","basis":share_amount("275315.94","万股"),
            "new":share_amount("82594.78","万股"),"post":share_amount("357910.72","万股")},
            (2,"假设本次配股比例为每10股配售3股"),(2,"截至2017年6月30日的总股本275,315.94万股"),
            (2,"本次配售股份数量为82,594.78万股"),(2,"总股本将增加至357,910.72万股")),
        claim("dilution_assumed_price",{"reported_cny_per_share":"7.58","actual_price_admitted":False},(2,"假设配股价格为7.58元/股")))
    for i in (5,10):
        add(i,"DONGWU_2017","65亿元为2017方案计划，后来终止；子项目和顶层用途不相加。",
            claim("plan_gross_proceeds_cap",money("65"),(2,"募集资金总额不超过65亿元")),
            claim("proposed_use_components",allocations("亿元",("子公司投入","15"),("信用交易","20"),("自营业务","25"),("资产管理","5")),
                (2,"加大对子公司投入,加快证券控股集团的建设不超过15亿元"),(2,"进一步扩大信用交易业务规模,优化公司收入结构不超过20亿元"),
                (2,"扩大自营业务规模不超过25亿元"),(2,"大力发展资产管理业务,提高主动管理能力不超过5亿元")))
    rows[5]["claims"].append(claim("subsidiary_allocation_breakdown",allocations("亿元",("东吴香港","10"),("东吴创新资本","5")),
        (3,"对东吴香港投入不超过人民币10亿元"),(3,"对东吴创新资本投入不超过人民币5亿元")))
    rows[10]["claims"].append(claim("Hong_Kong_allocation_breakdown",allocations("亿元",("国新国信东吴海外基金","8.5"),("信息系统","1.5")),
        (3,"国新国信东吴海外基金1不超过人民币8.5亿元"),(3,"募集资金规模不超过人民币1.5亿元")))
    for i,month in [(6,6),(9,9)]:
        add(i,"DONGWU_2017","正文65亿元与摊薄表格万元单位下6,500,000,000数值冲突，原页确认；不自动修正表格单位。",
            claim("source_plan_proceeds_cap",money("65"),(1,"本次配股计划募集资金不超过65亿元人民币")),
            claim("dilution_assumed_proceeds",money("650000","万元"),(2,"本次配股募集资金总额为650,000万元")),
            claim("dilution_completion_assumption",{"reported":f"2018年{month}月末","precision":"MONTH_END","year":2018,"month":month,"date":None,"actual_calendar_admitted":False},
                (2,f"本次配股于2018年{month}月末实施完成"),(2,"该完成时间仅用于计算本次配股对每股收益的影响")),
            claim("dilution_quantity_assumption",{"basis_asof":"2017-12-31","ratio_percent":"30","basis":share_amount("3000000000"),"new":share_amount("900000000")},
                (2,"假设本次配股股份登记日的所有股东均参与此次配售"),(2,"本次配售数量占配售股份前股本总额的30%"),
                (2,"截至2017年12月31日总股本3,000,000,000股"),(2,"本次配售股份数量为900,000,000股")),
            claim("dilution_table_amount_conflict",{"raw_number":"6,500,000,000","raw_unit":"万元","amount_cents":None,
                "status":"UNIT_AND_NUMBER_CONFLICT_WITH_TEXT","automatic_correction":False},
                (4,"本次发行募集资金总额(万元)6,500,000,000")))
    add(8,"WANXIANG_2017","56.60亿元和万股精度数量为假设。原文6.84元假设价格与预算除以股数不符，未作为实际定价。",
        claim("dilution_assumed_proceeds",money("56.60"),(2,"本次配股募集资金总额为56.60亿元")),
        claim("dilution_completion_assumption",{"reported":"2018年10月31日","precision":"DAY","date":"2018-10-31","actual_calendar_admitted":False},
            (2,"假设本次配股于2018年10月31日实施完成"),(2,"该完成时间仅用于计算本次配股摊薄即期回报对公司每股收益的影响")),
        claim("dilution_quantity_assumption",{"basis_asof":"2018-03-31","ratio_per_10":"3","basis":share_amount("2753159454"),
            "new":share_amount("82594.78","万股"),"post":share_amount("357910.73","万股")},
            (2,"截至2018年3月31日的总股数2,753,159,454股"),(2,"每10股配售3股"),(2,"本次配售股份数量为82,594.78万股"),(2,"总股本将增加至357,910.73万股")),
        claim("dilution_assumed_price",{"reported_cny_per_share":"6.84","actual_price_admitted":False},(2,"假设配股价格为6.84元/股")))
    add(11,"GOLDWIND_2018","用途表按整体募集计划列示；A/H划分仅由同日预案另行证明，不能把该总额直接当A股融资额。",
        claim("plan_gross_proceeds_cap",money("474418.30","万元"),(2,"募集资金总额不超过人民币474,418.30万元")),
        claim("proposed_use_components",allocations("万元",("StockyardHill527.5MW","139418.30"),("MooraboolNorth150MW","35000.00"),("补充流动资金","150000.00"),("偿还有息负债","150000.00")),
            (2,"单位:万元"),(2,"518,261.06139,418.30"),(2,"180,339.8135,000.00"),(2,"补充流动资金-150,000.00"),(2,"偿还有息负债-150,000.00")),
        claim("project_total_investments",allocations("万元",("StockyardHill527.5MW","518261.06"),("MooraboolNorth150MW","180339.81")),
            (2,"项目计划总投资额"),(2,"518,261.06139,418.30"),(2,"180,339.8135,000.00")))
    add(12,"SHANXI_2019","60亿元为募集预算，三项用途各不超过20亿元；不是实际到账金额。",
        claim("plan_gross_proceeds_cap",money("60"),(1,"拟募集资金总额为不超过人民币60亿元(含60亿元)")),
        claim("proposed_use_components",allocations("亿元",("资本中介","20"),("债券自营","20"),("子公司增资","20")),
            (1,"资本中介业务不超过20亿元"),(1,"债券自营业务不超过20亿元"),(1,"对子公司增资不超过20亿元")))
    add(13,"ZHONGKE_2020","总投资89000万元，拟用募集资金72000万元；宁波小计39000万元不与四项明细再次累加。",
        claim("plan_gross_proceeds_cap",money("72000.00","万元"),(1,"募集资金总额预计不超过人民币72,000.00万元")),
        claim("proposed_use_components",allocations("万元",("科宁达工业","9492.10"),("科宁达和丰","7929.32"),("科宁达鑫丰","7365.58"),("科宁达日丰","14213.00"),("赣州基地一期","33000.00")),
            (1,"单位:万元"),(1,"9,492.109,492.10"),(1,"7,929.327,929.32"),(1,"7,365.587,365.58"),(1,"14,213.0014,213.00"),(1,"50,000.0033,000.00")),
        claim("project_total_investment",money("89000.00","万元"),(1,"项目投资总额"),(1,"合计89,000.0072,000.00")),
        claim("ningbo_subtotal",money("39000.00","万元"),(1,"小计39,000.0039,000.00")))
    add(14,"ZHONGKE_2020","同日摊薄文本再次明确213040000股，但角色仍是2股比例的假设情景；不替代实际发行。",
        claim("dilution_assumed_proceeds",money("7.2"),(2,"假设本次配股最终募集资金总额为人民币7.2亿元")),
        claim("dilution_completion_assumption",{"reported":"2021年6月","precision":"MONTH","year":2021,"month":6,"date":None,"actual_calendar_admitted":False},
            (2,"假设本次配股于2021年6月完成"),(2,"前述时间仅用于计算本次配股摊薄即期回报对主要财务指标的影响")),
        claim("dilution_quantity_assumption",{"basis_asof":"2020-12-31","ratio_per_10":"2","basis":share_amount("1065200000"),
            "new":share_amount("213040000"),"post":share_amount("1278240000")},
            (1,"假设本次配股比例为每10股配售2股"),(1,"截至2020年12月31日的总股本1,065,200,000股"),
            (1,"本次可配股数量为213,040,000股"),(1,"总股本为1,278,240,000股")))
    assert set(rows) == set(range(15))
    return [rows[i] for i in range(15)]


def build():
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    targets = read(OUT/"targets.json")
    facts=[]
    for spec in specs():
        row=targets[spec["index"]]
        doc=Document({**row["source"],"review_role":"USE_OR_DILUTION_SOURCE_REVIEW"})
        fact=base_fact(doc,None)
        fact.update(plan_source_group=spec["group"],source_subtype=row["body_role_candidate"],reviewed_claims={},claim_evidence={},
            full_body_reviewed=False,operative_clause_review_complete=True,actual_issuance_calendar=None,
            actual_issue_price=None,actual_issuance_quantity=None,strict_M06_pressure=None,
            new_holding_clause_confirmed=False,notes=[spec["note"],"以目录日末为可得日期代理；未确认最早公开。经营预测和项目可行性结论未获独立验证。"])
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            assert c["field"] not in fact["reviewed_claims"]
            fact["reviewed_claims"][c["field"]]=c["value"]
            fact["claim_evidence"][c["field"]]=[anchor(row["document_id"],pages,p,t) for p,t in c["proof"]]
        facts.append(fact)
    return facts


def source_view(fact, at):
    visible=fact["known_at"]<=at
    return {"document_id":fact["document_id"],"as_of":at,"known_at":fact["known_at"],
        "status":"REVIEWED_USE_DILUTION_DATE_PROXY_ONLY" if visible else "NO_VIEW",
        "reviewed_claims":deepcopy(fact["reviewed_claims"]) if visible else {},
        "actual_event_id":None,"actual_issuance_calendar":None,"strict_M06_pressure":None,"trading_feature_admitted":False}


def comparisons(facts, prior):
    idx={f["document_id"]:f for f in prior}
    plan_ids={1:"1203759867",2:"1203759867",3:"1203978536",4:"1203978536",5:"1204532467",6:"1204532467",
        7:"1205272661",8:"1205272661",9:"1205275255",10:"1205275255",11:"1205782943",12:"1207165608",13:"1209640415",14:"1209640415"}
    paired=[]
    for i,rid in plan_ids.items():
        f,p=facts[i],idx[rid]
        assert f["symbol"]==p["symbol"] and f["known_at"]==p["known_at"]
        field="plan_gross_proceeds_cap" if f["source_subtype"]=="PROPOSED_PROCEEDS_FEASIBILITY" else "dilution_assumed_proceeds"
        amount=f["reviewed_claims"][field]["amount_cents"]
        other=p["reviewed_claims"]["plan_gross_proceeds"]["amount_cents"]
        paired.append({"source_id":f["document_id"],"preplan_id":rid,"relationship_known_at":f["known_at"],
            "source_field":field,"source_amount_cents":amount,"preplan_amount_cents":other,
            "numeric_budget_matches":amount==other,"preplan_amount_scope":p["reviewed_claims"]["plan_gross_proceeds"]["scope"],
            "role_equivalence_assumed":False,"prior_preplan_unchanged":True,"actual_event_id":None})
    assert len(paired)==14 and sum(p["numeric_budget_matches"] for p in paired)==13
    conflicts=[]
    mismatch=next(p for p in paired if not p["numeric_budget_matches"])
    conflicts.append({"kind":"SAME_DAY_DILUTION_AND_PREPLAN_BUDGET_CONFLICT",**mismatch,
        "corroborating_use_report_id":"1203978537","difference_cents":mismatch["source_amount_cents"]-mismatch["preplan_amount_cents"],
        "status":"PRESERVED_UNRESOLVED","correction_applied":False})
    for i in (6,9):
        f=facts[i]
        conflicts.append({"kind":"DILUTION_TABLE_UNIT_NUMBER_CONFLICT","source_id":f["document_id"],"known_at":f["known_at"],
            "reported_table_number":"6,500,000,000","reported_table_unit":"万元",
            "consistent_text_amount_cents":f["reviewed_claims"]["dilution_assumed_proceeds"]["amount_cents"],
            "table_amount_admitted_cents":None,"status":"PRESERVED_UNRESOLVED","correction_applied":False})
    price_checks=[]
    for i in (4,8):
        f=facts[i];c=f["reviewed_claims"]
        exact=Decimal(c["dilution_assumed_proceeds"]["amount_cents"])/100/c["dilution_quantity_assumption"]["new"]["nominal_shares"]
        reported=Decimal(c["dilution_assumed_price"]["reported_cny_per_share"])
        check={"source_id":f["document_id"],"known_at":f["known_at"],"reported_hypothetical_price":str(reported),
            "budget_divided_by_reported_nominal_shares":str(exact),"difference_cny_per_share":str(reported-exact),
            "within_half_cent_rounding":abs(reported-exact)<=Decimal("0.005"),"actual_price_admitted":False}
        price_checks.append(check)
        if not check["within_half_cent_rounding"]:
            conflicts.append({"kind":"HYPOTHETICAL_PRICE_NOT_REPRODUCIBLE_FROM_STATED_BUDGET_AND_SHARES",**check,
                "status":"PRESERVED_UNRESOLVED","correction_applied":False})
    earlier={"source_id":facts[0]["document_id"],"source_known_at":facts[0]["known_at"],"compared_later_admin_id":"1201766438",
        "compared_later_admin_known_at":idx["1201766438"]["known_at"],
        "relationship_known_at":idx["1201766438"]["known_at"],
        "numeric_budget_matches":facts[0]["reviewed_claims"]["plan_gross_proceeds_cap"]["amount_cents"]==idx["1201766438"]["reviewed_claims"]["plan_gross_proceeds_cap"]["amount_cents"],
        "earlier_state_asset_approval_inferred":False,"historical_first_publication_verified":False,
        "note":"只证明10月28日来源比所对照的11月13日事项来源更早，后来的国资同意不回填10月。"}
    recovered={"dilution_source_id":facts[14]["document_id"],"preplan_id":"1209640415","known_at":facts[14]["known_at"],
        "dilution_assumption_new_shares":facts[14]["reviewed_claims"]["dilution_quantity_assumption"]["new"]["nominal_shares"],
        "preplan_cap_visually_recovered_shares":idx["1209640415"]["reviewed_claims"]["plan_quantity_scenarios"][0]["reported_total_shares"],
        "role":"NUMERIC_CORROBORATION_CAP_VS_HYPOTHESIS","prior_visual_receipt_unchanged":True,"actual_quantity_admitted":False}
    assert recovered["dilution_assumption_new_shares"]==recovered["preplan_cap_visually_recovered_shares"]==213040000
    terminals=[]
    for indices,rid in [([3,4,7,8],"1206233867"),([5,6,9,10],"1205354373")]:
        end=idx[rid]
        assert all(facts[i]["symbol"]==end["symbol"] and facts[i]["known_at"]<end["known_at"] for i in indices)
        terminals.append({"prior_source_ids":[facts[i]["document_id"] for i in indices],"terminal_source_id":rid,
            "relationship_known_at_no_earlier_than":end["known_at"],"terminal_status":end["administrative_status"],
            "terminal_unchanged":True,"joined_to_executed_event":False,"future_approval_backfilled":False})
    versions=[]
    for ia,ib in [(3,7),(4,8),(5,10),(6,9)]:
        a,b=facts[ia],facts[ib]
        fields=sorted(set(a["reviewed_claims"]) | set(b["reviewed_claims"]))
        versions.append({"earlier_source":a["document_id"],"later_source":b["document_id"],"relationship_known_at":b["known_at"],
            "changed_source_fields":{k:{"earlier":a["reviewed_claims"].get(k),"later":b["reviewed_claims"].get(k)}
                for k in fields if a["reviewed_claims"].get(k)!=b["reviewed_claims"].get(k)},
            "note":"版本声明对照；缺少字段不推断失效，测算情景变化不创造新的实际供给。"})
    return paired,conflicts,price_checks,earlier,recovered,terminals,versions


def candidate_reviews():
    # 每一片段均已阅读；类别只表达来源用途，不承认报告预测已经验证。
    hypothetical={0:[1],2:[1,2,3],3:[3],4:[0,1,2],6:[1,2],7:[3],8:[1,2],9:[1,2],14:[0,1,2,3]}
    general={0:[0,2],1:[1],2:[5],4:[3],6:[4],8:[0,3],9:[5],14:[4]}
    output=[]
    for i,r in enumerate(read(OUT/"source_candidates.json")):
        for j,c in enumerate(r["candidates"]):
            role="CURRENT_PROPOSED_USE_OR_BUDGET_CLAUSE"
            if j in hypothetical.get(i,[]):role="DILUTION_OR_FINANCIAL_SIMULATION_NOT_EXECUTION"
            if j in general.get(i,[]):role="QUALITATIVE_BENEFIT_OR_COMMITMENT_NOT_SUPPLY"
            output.append({"document_id":r["document_id"],"candidate_index":j,**c,"reviewed":True,"review_role":role})
    assert len(output)==55
    return output


def finalize():
    facts=build();prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(facts)==15 and len(prior)==399
    assert not {f["document_id"] for f in facts}&{f["document_id"] for f in prior}
    reviewed=candidate_reviews()
    paired,conflicts,price_checks,earlier,recovered,terminals,versions=comparisons(facts,prior)
    assert len(conflicts)==4
    save(OUT/"reviewed_use_dilution_source_facts.json",facts)
    save(OUT/"combined_source_facts.json",prior+facts)
    save(OUT/"reviewed_source_candidates.json",reviewed)
    assert read(OUT/"additional_holding_candidates.json")==[]
    for name,obj in [("same_day_preplan_comparisons",paired),("preserved_source_conflicts",conflicts),("hypothetical_price_checks",price_checks),
        ("earlier_budget_source_comparison",earlier),("zhongke_numeric_corroboration",recovered),("terminal_source_comparisons",terminals),("version_comparisons",versions)]:
        save(OUT/(name+".json"),obj)
    queries=[]
    for fact in facts:
        at=fact["known_at"];before=(datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
        queries.append({"document_id":fact["document_id"],"before":source_view(fact,before),"at":source_view(fact,at)})
    save(OUT/"use_dilution_publication_queries.json",queries)
    with (OUT/"15份用途与摊薄条款.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["公告ID","证券","可得日期代理","标题","来源分类","计划来源组","结构化字段","解释","来源URL"])
        writer.writeheader()
        for f in facts:
            writer.writerow({"公告ID":f["document_id"],"证券":f["symbol"],"可得日期代理":f["known_at"],"标题":f["title"],
                "来源分类":f["source_subtype"],"计划来源组":f["plan_source_group"],"结构化字段":json.dumps(f["reviewed_claims"],ensure_ascii=False),
                "解释":f["notes"][0],"来源URL":f["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_USE_DILUTION_V1","status":"15_USE_DILUTION_SOURCE_CLAUSES_REVIEWED_WITH_CONFLICTS_PRESERVED",
        "documents":15,"feasibility_reports":9,"dilution_reports":6,"target_pdf_pages":275,"full_pdf_pages_read_claimed":False,
        "candidate_fragments_read":55,"candidate_characters":20075,"candidate_role_counts":dict(Counter(c["review_role"] for c in reviewed)),
        "source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts),"same_day_preplan_pairs":14,"numeric_budget_agreements":13,
        "preserved_source_conflicts":4,"hypothetical_completion_dates_or_periods_excluded":6,"hypothetical_prices_excluded":2,
        "coarse_share_precision_reports":2,"earlier_than_compared_admin_budget_sources":1,"same_day_visual_number_corroborations":1,
        "proposed_use_component_totals":9,"visual_pages_reviewed":len(VISUAL_PAGES),"version_pairs":4,"terminal_comparisons":2,
        "new_current_holding_clauses_confirmed":0,"prior_source_documents_unchanged":399,"combined_source_documents":414,
        "new_actual_issuance_calendars":0,"new_publication_queries":30,"remaining_long_documents":25,
        "full_issuance_coverage":False,"earlier_holding_clause_full_coverage":False,"free_float_denominator_available":False,
        "strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"qualified_candidates":0,"independent_forward_observations":0,
        "orders_authorized":False,"delivery_package_created":False,"goal_achieved":False,
        "next_source_action":"T13_18_REGULATORY_RESPONSES_7_DRAFT_PROSPECTUSES_AND_EARLIER_HOLDING_COVERAGE"}
    save(OUT/"result.json",result)
    print(json.dumps({"文件":15,"字段":result["source_fields_reviewed"],"保留来源冲突":4,"排除假设日期":6,"剩余长文件":25},ensure_ascii=False))


if __name__=="__main__":
    finalize()
