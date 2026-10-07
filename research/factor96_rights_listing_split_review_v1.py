"""复核36份上市公告的原文股份分类；不把公告分类等同于即时可卖量。"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import csv
import json

from research.factor96_rights_listing_split_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import state_at


# 数量单位均为股；这些是已阅读原文的分类，零值必须有全数无限售或明确表格支撑。
CLASSIFICATIONS = [
    ("1201908424",1496671674,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次配股新增上市股份1,496,671,674股,均为无限售条件股份"),
    ("1201949738",1013743887,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加的股份:1,013,743,887股,均为无限售条件股份"),
    ("1202210540",341624458,41662425,3,"SPLIT_EXPLICIT","其中无限售条件流通股增加341,624,458股,有限售条件流通股增加41,662,425股"),
    ("1202299350",38772000,13695,3,"SPLIT_EXPLICIT","其中无限售条件流通股增加38,772,000股,有限售条件流通股(即高管锁定股,含监事乔林先生的获配股份锁定股900股)增加13,695股"),
    ("1202341054",507908392,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配股新增上市股份507,908,392股,均为无限售条件股份"),
    ("1203323591",706270150,0,2,"SPLIT_EXPLICIT","其中无限售条件流通股增加706,270,150股,有限售条件流通股增加0股"),
    ("1203628854",480765103,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加的股份:480,765,103股,均为无限售条件流通股"),
    ("1203752839",288782748,1186709,2,"SPLIT_EXPLICIT","其中无限售条件流通股增加288,782,748股,有限售条件流通股增加1,186,709股"),
    ("1204186050",145655471,114575348,2,"SPLIT_EXPLICIT","其中无限售条件流通股增加145,655,471股,有限售条件流通股增加114,575,348股"),
    ("1204289097",146846271,849930,6,"SPLIT_EXPLICIT","其中限售股份849,930股,无限售条件股份146,846,271股"),
    ("1204428470",243570740,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加的股份(均为无限售条件流通股份):243,570,740股"),
    ("1204641643",1515678586,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加的股份:1,515,678,586股,均为无限售条件流通股"),
    ("1205582178",2699378625,0,5,"TABLE_INCREMENT_RECONCILED","有限售条件的流通股2,163,141,99323.38-2,163,141,99318.10%无限售条件的流通股7,087,960,90276.622,699,378,6259,787,339,52781.90%"),
    ("1205995906",545352788,0,6,"ALL_UNRESTRICTED_EXPLICIT","本次有效认购股数545,352,788股均为无限售流通股"),
    ("1206079405",833419462,0,4,"TABLE_INCREMENT_RECONCILED","有限售条件的流通股9,058,2870.3209,058,2870.25无限售条件的流通股2,781,736,94199.68833,419,4623,615,156,40399.75"),
    ("1207214800",335111438,0,4,"ALL_UNRESTRICTED_EXPLICIT","本次有效认购股数335,111,438股均为无限售流通股"),
    ("1207277683",1228983542,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,228,983,542股,均为无限售条件流通股"),
    ("1207403957",151866908,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配股新增上市股份151,866,908股,本次原无限售条件流通股股东及原有限售条件流通股股东认购的股票均为无限售流通股"),
    ("1207403868",1485967280,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,485,967,280股,均为无限售条件流通股"),
    ("1207418053",880518908,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为880,518,908股,均为无限售条件流通股"),
    ("1208012652",761046394,0,6,"TABLE_INCREMENT_RECONCILED","无限售条件流通股份2,828,725,153100.00%761,046,3943,589,771,547100.00%有限售条件流通股份-----"),
    ("1208080267",1702997123,0,2,"ALL_UNRESTRICTED_EXPLICIT","本次A股配售增加的股份(均为无限售条件流通股份):1,702,997,123股"),
    ("1208538461",1655142470,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,655,142,470股,均为无限售条件流通股"),
    ("1208636972",998330844,0,1,"SPLIT_EXPLICIT","其中,无限售条件流通股份增加998,330,844股,有限售条件流通股份增加0股"),
    ("1209076894",3225083672,0,4,"TABLE_INCREMENT_RECONCILED","有限售条件的流通股114,242,2780.99%-114,242,2780.77%无限售条件的流通股11,430,280,69399.01%3,225,083,67214,655,364,36599.23%"),
    ("1210292954",1076601364,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,076,601,364股,均为无限售条件流通股"),
    ("1210781397",1083382346,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,083,382,346股,均为无限售条件流通股"),
    ("1211875623",595574506,0,5,"TABLE_INCREMENT_RECONCILED","有限售条件的流通股75,855,0921.26%075,855,0921.15%无限售条件的流通股(含高管锁定股)5,932,161,19498.74%595,574,5066,527,735,70098.85%"),
    ("1212032708",1126983743,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,126,983,743股,均为无限售条件流通股"),
    ("1212332253",1552021645,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次A股配售增加的股份总数为1,552,021,645股,均为无限售条件流通股"),
    ("1212510377",150525773,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加的股份总数为150,525,773股,均为无限售条件流通股"),
    ("1212978199",1054713257,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,054,713,257股,均为无限售条件流通股"),
    ("1213301939",1502907061,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次A股配售增加股份总数为1,502,907,061股,均为无限售条件流通股"),
    ("1214448175",1939315620,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,939,315,620股,均为无限售条件流通股"),
    ("1215295719",1462523613,0,1,"ALL_UNRESTRICTED_EXPLICIT","本次配售增加股份总数为1,462,523,613股,均为无限售条件流通股"),
    ("1217194963",4829739185,0,4,"TABLE_INCREMENT_RECONCILED","A股无限售流通股16,714,696,77878.59%4,829,739,18521,544,435,96382.55%H股无限售流通股4,554,000,00021.41%04,554,000,00017.45%限售流通股00.00%000.00%"),
]

NON_REDUCTION = {
    "1201908424":(["福建省财政厅","福建省投资开发集团有限责任公司"],"本公司股份"),
    "1201949738":(["北京华信六合投资有限公司"],"本公司股份"),
    "1202341054":(["中国黄金集团公司"],"本公司股份"),
    "1203628854":(["新疆特变电工集团有限公司","新疆宏联创业投资有限公司"],"本公司股份"),
    "1204289097":(["成都天齐实业(集团)有限公司","张静","李斯龙"],"本公司股份"),
    "1204428470":(["新奥控股投资有限公司","弘创(深圳)投资中心(有限合伙)","廊坊合源投资中心(有限合伙)","河北威远集团有限公司"],"本公司股份"),
    "1204641643":(["新疆广汇实业投资(集团)有限责任公司","新疆广汇实业投资(集团)有限责任公司(宏广定向资产管理计划)","华龙证券-浦发银行-华龙证券金智汇31号集合资产管理计划"],"本公司股份"),
    "1205582178":(["南山集团有限公司","山东怡力电业有限公司"],"本公司股份"),
    "1206079405":(["李振国","李喜燕","李春安"],"本公司股份"),
    "1207418053":(["苏州国际发展集团有限公司","苏州营财投资集团有限公司","苏州信托有限公司"],"获配的本公司股份"),
    "1208080267":(["深圳市招融投资控股有限公司","深圳市集盛投资发展有限公司","中国远洋运输有限公司"],"本公司A股股份"),
}

# 只保留原文限定，不把缺失的额外锁定数量或终止日期补成零。
SPECIAL_SCOPES = {
    "1201949738": [(2,"发行前华信六合所持股份中有225,000,000股为有限售条件流通股","OLD_RESTRICTED_STOCK_NOT_NEW_ISSUE","原225000000股为存量限制，本次新增全部无限售；网上和网下认购不是新增限售分类。")],
    "1202299350": [(3,"高管锁定股在符合","REGISTERED_EXECUTIVE_LOCK_AND_PRIOR_COMMITMENT","既有登记分类及超达6个月自愿锁定承诺均保留，不直接相减生成可卖量。")],
    "1203628854": [(7,"475,092,040","SUBSCRIPTION_CHANNEL_NOT_RESTRICTION","网上、网下是原持股类别对应的认购渠道；不把网下5673063股变成新增限售。")],
    "1204289097": [(6,"988,690,450股为无限售条件流通股(其中1,420,060股为高管锁定股)","REPORTED_UNRESTRICTED_SCOPE_INCLUDES_EXECUTIVE_LOCK","原文将存量高管锁定股放在无限售叙述中；保留报告口径，不重分类或扩大新增股数。")],
    "1205582178": [(5,"648,942,597股为网下配售","SUBSCRIPTION_CHANNEL_NOT_RESTRICTION","网下认购648942597股不等于新增限售；分类表仅无限售增加2699378625股。")],
    "1205995906": [(6,"其中涉及公司高级管理人员认购股份,发行人将按照高级管理人员持股的相关规定","EXECUTIVE_SHARES_SUBJECT_TO_SEPARATE_MANAGEMENT","公告按全部无限售报告，同时保留高管股份登记管理；额外限制数量未知。"),(7,"通过深圳中登对高管认购的股份进行管理","EXECUTIVE_MANAGEMENT_CROSS_PAGE_CONTINUATION","跨页续文和本页重复条款共同确认管理限定，不推成无高管限制。")],
    "1206079405": [(4,"公司累计转股9,209股","CONCURRENT_CONVERSION_SEPARATED","配股分类表排除9209股同期转股；转股计入另一总股本口径，不能并入配股数量。"),(5,"2,449,030股为网下配售","SUBSCRIPTION_CHANNEL_NOT_RESTRICTION","网下配售2449030股不等于新增限售。")],
    "1207214800": [(4,"其中涉及公司高级管理人员认购股份,发行人将按照高级管理人员持股的相关规定","EXECUTIVE_SHARES_SUBJECT_TO_SEPARATE_MANAGEMENT","公告按全部无限售报告，同时保留高管管理；额外限制数量未知。"),(5,"任公司深圳分公司对高管认购的股份进行管理","EXECUTIVE_MANAGEMENT_CROSS_PAGE_CONTINUATION","高管管理条款跨4至5页，分别保留原文锚点。")],
    "1208636972": [(5,"无限售条件流通股份*","REPORTED_UNRESTRICTED_SCOPE_INCLUDES_EXECUTIVE_LOCK","无限售分类带星号，并以含高管锁定股为注，不代表最终可卖分类。"),(5,"含高管锁定股","EXECUTIVE_SCOPE_FOOTNOTE","表格的无限售口径包含高管锁定股，数量未独立给出。"),(5,"346,809,486股股份将于2020年11月2日解除限售并上市流通","OLD_PLACEMENT_UNLOCK_NOT_RIGHTS_NEW_ISSUE","2017年定增股份计划解禁与本次配股分开，不在配股中重复累计。")],
    "1209076894": [(5,"网下有效认购数量为31,107,070股","SUBSCRIPTION_CHANNEL_NOT_RESTRICTION","网下31107070股不是新增限售；表格中存量限售114242278股没有变化。")],
    "1210292954": [(2,"累计转股802股","CONCURRENT_CONVERSION_SEPARATED","发行后总股本含802股同期转股，配股量使用明确披露的1076601364股。")],
    "1211875623": [(5,"无限售条件的流通股(含高管锁定股)","REPORTED_UNRESTRICTED_SCOPE_INCLUDES_EXECUTIVE_LOCK","保留公告含高管锁定股的无限售口径，不能把595574506股全部视为即时可卖。")],
    "1212978199": [(4,"累计转股170股","CONCURRENT_CONVERSION_SEPARATED","发行后总股本包含170股同期转股，配股增加1054713257股，不按总差计算。")],
    "1215295719": [(5,"累计转股3,324股","CONCURRENT_CONVERSION_SEPARATED","发行后总股本含3324股同期转股；配股量为1462523613股。"),(5,"配股股份中向原股东配售的股份同时锁定,不得在二级市场出售或以其他方式转让","INCENTIVE_ALLOTMENT_VOLUNTARY_LOCK","股权激励获配股份按原激励股票限售截止安排继续锁定，具体数量及日历终止日未给出。"),(5,"股权激励对象本次配股获配股份的自愿限售期安排将依据上述规定执行","INCENTIVE_ALLOTMENT_VOLUNTARY_LOCK_IMPLEMENTATION_CLAIM","公告明确本次适用原激励计划锁定要求，不把无限售分类当作解除承诺。")],
    "1217194963": [(4,"截至本公告发布之日,本行H股配股尚未完成","A_H_SEPARATED","这里只计A股新增4829739185股，不提前把未完成H股计入A股供给。")],
}

VISUAL_PAGES = [("1204289097",6),("1205582178",5),("1205995906",6),("1205995906",7),
                ("1208636972",5),("1209076894",4),("1211875623",5),("1215295719",5),("1206079405",4)]


def anchor(pages, rid, page, literal):
    text = pages[page]
    literal = normalized(literal)
    start = text.find(literal)
    assert start >= 0, (rid,page,literal)
    end = start + len(literal)
    return {"document_id":rid,"page":page,"normalized_start":start,"normalized_end":end,
            "literal":literal,"context":text[max(0,start-70):min(len(text),end+100)]}


def prepare_reviews():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"]) == item["sha256"]
    targets = {r["document_id"]:r for r in read(OUT/"targets.json")}
    assert len(CLASSIFICATIONS)==36 and set(targets)=={r[0] for r in CLASSIFICATIONS}
    reviews=[]
    for rid,unrestricted,restricted,page,method,literal in CLASSIFICATIONS:
        row=targets[rid]
        assert digest(row["raw_path"])==row["raw_sha256"] and digest(row["text_path"])==row["text_sha256"]
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        evidence=[anchor(pages,rid,page,literal)]
        total=row["updates"]["listing_announced_total_shares"]
        assert unrestricted>=0 and restricted>=0 and unrestricted+restricted==total
        assert f"{unrestricted:,}" in literal
        if restricted:
            assert f"{restricted:,}" in literal
        exceptions=[]
        for pn,quote,kind,note in SPECIAL_SCOPES.get(rid,[]):
            exceptions.append({"kind":kind,"note":note,"evidence":[anchor(pages,rid,pn,quote)]})
        commitments=[]
        if rid in NON_REDUCTION:
            actors,scope=NON_REDUCTION[rid]
            text=pages[1]
            end=text.index("6个月内不减持")+len("6个月内不减持")+len(scope)
            start=text.rfind("参与本次",0,end)
            assert start>=0 and "上市之日起" in text[start:end] and text[end-len(scope):end]==scope
            proof=[anchor(pages,rid,1,text[start:end])]
            for actor in actors:
                assert normalized(actor) in proof[0]["literal"],(rid,actor)
            commitments.append({"kind":"SIX_MONTH_NON_REDUCTION_COMMITMENT","actors":actors,"months":6,
                "relative_start":"本次配股新增股份上市之日起","share_scope_literal":scope,
                "calendar_end":None,"constrained_new_shares":None,"evidence":proof})
        reviews.append({"document_id":rid,"symbol":row["symbol"],"event_id":row["event_id"],
            "known_at":row["known_at"],"source_url":row["source_url"],"raw_path":row["raw_path"],
            "text_path":row["text_path"],"raw_sha256":row["raw_sha256"],"text_sha256":row["text_sha256"],
            "listing_announced_total_shares":total,"listing_reported_unrestricted_new_shares":unrestricted,
            "listing_reported_restricted_new_shares":restricted,"classification_method":method,
            "classification_evidence":evidence,"scope_exceptions":exceptions,"non_reduction_commitments":commitments,
            "effective_sellable_new_shares":None,"additional_restricted_share_quantity":None,
            "quantity_scope":"ANNOUNCEMENT_REPORTED_CLASSIFICATION_NOT_EFFECTIVE_SELLABILITY",
            "previous_classification_reused":rid=="1202299350","full_body_reviewed":False,
            "historical_holding_commitment_full_version_coverage":False,"trading_feature_admitted":False})
    return reviews


def review_candidates():
    reviewed=[]
    for row in read(OUT/"source_candidates.json"):
        for c in row["candidates"]:
            reviewed.append({**c,"reviewed":True,"review_scope":"新增分类、存量和增量、认购渠道、表格限定；不对公司全文背书。"})
    save(OUT/"reviewed_classification_candidates.json",reviewed)
    groups={
        "SIX_MONTH_NON_REDUCTION_CLAUSE":{0,1,8,10,15,18,20,21,25,32,35},
        "DIRECTOR_MANAGEMENT_CONTINUATION":{19},
        "EXECUTIVE_CLASSIFICATION_OR_MANAGEMENT":{6,17,22,23,27,28,37,39},
        "INCENTIVE_ALLOTMENT_LOCK":{42},
        "SUBSCRIPTION_COMMITMENT_NOT_HOLD_PERIOD":{3,4,7,9,12,14,16,24,29,30,31,34,38,41},
        "DISCLOSURE_ACCURACY_COMMITMENT":{2,5,11,13,33,36},
        "CONCERT_PARTY_AND_CONCURRENT_CONVERSION_CONTEXT":{26},
        "BUSINESS_SCOPE_NOT_SHARE_LOCK":{40},
    }
    all_indexes=[i for group in groups.values() for i in group]
    candidates=read(OUT/"additional_constraint_candidates.json")
    assert len(all_indexes)==len(set(all_indexes))==len(candidates)==43 and set(all_indexes)==set(range(43))
    save(OUT/"reviewed_additional_constraints.json",[{**c,"reviewed":True,"review_index":i,
        "semantic_class":next(kind for kind,indexes in groups.items() if i in indexes),
        "complete_commitment_history":False} for i,c in enumerate(candidates)])


def finalize():
    reviews=prepare_reviews()
    save(OUT/"reviewed_listing_classifications.json",reviews)
    review_candidates()
    original=read(PREVIOUS/"combined_source_facts_with_restriction_addendum.json")
    combined=deepcopy(original)
    current={r["document_id"]:r for r in combined}
    for row in reviews:
        fact=current[row["document_id"]]
        # 增加明确的公告口径字段；保留此前更窄的长春登记字段及原有证据，避免覆盖。
        for key in ("listing_reported_unrestricted_new_shares","listing_reported_restricted_new_shares"):
            assert key not in fact["updates"]
            fact["updates"][key]=row[key]
            fact["evidence"][key]=deepcopy(row["classification_evidence"])
        fact["updates"]["listing_reported_classification_scope"]=row["quantity_scope"]
        fact["evidence"]["listing_reported_classification_scope"]=deepcopy(row["classification_evidence"])
        if "effective_sellable_new_shares" not in fact["updates"]:
            fact["updates"]["effective_sellable_new_shares"]=None
            fact["notes"].append("即时可卖量未计算：没有完整股东限制和承诺版本，未知不是零。")
        if row["non_reduction_commitments"]:
            fact["updates"]["listing_disclosed_non_reduction_commitments"]=[{k:v for k,v in c.items() if k!="evidence"} for c in row["non_reduction_commitments"]]
            fact["evidence"]["listing_disclosed_non_reduction_commitments"]=[a for c in row["non_reduction_commitments"] for a in c["evidence"]]
        if row["scope_exceptions"]:
            fact["updates"]["listing_classification_scope_exceptions"]=[{k:v for k,v in c.items() if k!="evidence"} for c in row["scope_exceptions"]]
            fact["evidence"]["listing_classification_scope_exceptions"]=[a for c in row["scope_exceptions"] for a in c["evidence"]]
        fact["notes"].append("本轮分类来自该上市公告原文；含高管锁定或额外承诺时保留限定，不等同最终中登锁定状态或即时可卖量。")
    save(OUT/"combined_source_facts.json",combined)
    groups=defaultdict(list)
    for fact in combined:
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    events=deepcopy(read(PREVIOUS/"event_chains_with_restriction_addendum.json"))
    for event in events:
        rows=groups[event["event_id"]]
        event["latest_reviewed_state"]=state_at(rows,max(r["known_at"] for r in rows))
    save(OUT/"event_chains.json",events)
    boundaries=[]
    for event_id,rows in sorted(groups.items()):
        for at in sorted({r["known_at"] for r in rows}):
            before=(datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
            boundaries.append({"event_id":event_id,"boundary":at,"before":state_at(rows,before),"at":state_at(rows,at)})
    assert len(boundaries)==293
    save(OUT/"publication_boundary_queries.json",boundaries)
    with (OUT/"36份上市股份分类.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["公告ID","证券","公开日期代理","安排上市日","新增总股数","公告列示无限售新增","公告列示限售新增","取数方法","六个月不减持条款","其他口径限定","即时可卖量","来源URL"])
        writer.writeheader()
        for row in reviews:
            writer.writerow({"公告ID":row["document_id"],"证券":row["symbol"],"公开日期代理":row["known_at"],
                "安排上市日":current[row["document_id"]]["updates"]["announced_listing_date"],
                "新增总股数":row["listing_announced_total_shares"],"公告列示无限售新增":row["listing_reported_unrestricted_new_shares"],
                "公告列示限售新增":row["listing_reported_restricted_new_shares"],"取数方法":row["classification_method"],
                "六个月不减持条款":"已披露" if row["non_reduction_commitments"] else "本次未提取；不代表不存在",
                "其他口径限定":"；".join(x["note"] for x in row["scope_exceptions"]),"即时可卖量":"未知，未计算","来源URL":row["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_LISTING_SPLIT_V1",
        "status":"36_LISTING_REPORTED_CLASSIFICATIONS_AND_CONSTRAINTS_REVIEWED",
        "documents":36,"new_classification_documents_reviewed":35,"reused_classification_documents":1,
        "pdf_pages_in_target_documents":359,"classification_candidates_read":93,"additional_constraint_candidates_read":43,
        "classification_methods":dict(Counter(r["classification_method"] for r in reviews)),
        "positive_reported_restricted_increment_events":sum(r["listing_reported_restricted_new_shares"]>0 for r in reviews),
        "zero_reported_restricted_increment_events":sum(r["listing_reported_restricted_new_shares"]==0 for r in reviews),
        "listing_six_month_non_reduction_claims":len(NON_REDUCTION),"incentive_allotment_lock_claims":1,
        "executive_inclusive_reported_classification_documents":3,"separate_executive_management_documents":2,
        "concurrent_conversion_scope_documents":4,"visual_pages_reviewed":len(VISUAL_PAGES),
        "prior_nonlisting_source_facts_unchanged":282,"existing_listing_facts_extended":36,
        "new_canonical_source_documents":0,"combined_source_facts":len(combined),"events":len(events),
        "publication_boundary_pairs":len(boundaries),"publication_queries":2*len(boundaries),
        "effective_sellable_quantity_computed_events":0,"historical_holding_commitment_full_version_coverage":False,
        "full_actual_listing_restriction_coverage":False,"within_fixed_scope_reported_classification_complete":True,
        "rights_remaining_plan_admin_documents":121,"historical_first_publication_verified":False,
        "strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"qualified_candidates":0,
        "independent_forward_observations":0,"orders_authorized":False,"goal_achieved":False,"delivery_package_created":False,
        "next_source_action":"T13_121_PLAN_ADMIN_VERSION_REVIEW_AND_EARLIER_HOLDING_CLAUSE_COVERAGE"}
    save(OUT/"result.json",result)
    print(json.dumps({"公告分类":36,"正限售增量":result["positive_reported_restricted_increment_events"],
        "公告新增限售为零":result["zero_reported_restricted_increment_events"],"六个月不减持条款":11,
        "股份相加一致":36,"可卖数量已计算":0,"账户新增":0},ensure_ascii=False))


if __name__=="__main__":
    finalize()
