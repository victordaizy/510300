"""保存已逐项核读的关键词窗口、补读条款与事项归属，不推定可卖数量。"""
from copy import deepcopy
from datetime import datetime, timedelta
import json

from research.factor96_rights_holding_remainder_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import claim
from research.factor96_rights_preplans_review_v1 import anchor
from research.factor96_rights_prospectus_constraints_review_v1 import source_view

FULL_READ = {19,21,51,52,68,73,117,118,119,120,175,183,200,232,233,234}
EXTRA_PAGES = {20:[128],26:[75],39:[182,203],53:[161,162],60:[51,195],67:[123,148,150]}
VISUAL_PAGES = [("1210115279",162),("1212175293",195),("1212175293",51)]


def decisions():
    groups = {
        "EXISTING_EQUITY_HISTORY_OR_SNAPSHOT": [1,5,6,7,9,10,13,14,15,17,18,24,29,45,46,47,48,49,50,51,53,54,55,56,57,61,63,64,65,66,93,94,97,98,103,104,110,111,126,127,130,131,134,138,141,142,148,149,151,152,153,154,157,161,162,163,168,169,170,173,174,182,187,188,189,190,192,193,195,197,198,205,210,212,213,217,218,225,226,230,231,238,241,242,246,247],
        "OPERATING_HEDGE": [20,22,33,34,35,36,37,39,40,41,42,43,67,69,70,71,72,74,211],
        "MECHANICAL_PATENT": list(range(77,93)),
        "CONSTRAINT_CONTEXT_SUPPLEMENTED": [19,21,52,68,73,117,118,119,120,175,183,200,232,233,234],
        "REGULATORY_BACKGROUND_OR_OTHER_STOCK_BUSINESS": [0,105,139,140,143,144,145,146,172,177,199,203,204,208,209],
        "ACCOUNTING_INVESTMENT_OR_THIRD_PARTY_BUSINESS": [2,3,4,8,11,12,16,23,25,26,27,28,30,31,32,38,44,58,59,60,62,75,76,95,96,99,100,101,102,106,107,108,109,112,113,114,115,116,121,122,123,124,125,128,129,132,133,135,136,137,147,150,155,156,158,159,160,164,165,166,167,171,176,178,179,180,181,184,185,186,191,194,196,201,202,206,207,214,215,216,219,220,221,222,223,224,227,228,229,235,236,237,239,240,243,244,245],
    }
    notes = {
        "EXISTING_EQUITY_HISTORY_OR_SNAPSHOT": "股本沿革、发行前持股或历史交易/质押状态；窗口未给出本次新增获配股份的可卖数量，不作比例外推。",
        "OPERATING_HEDGE": "原材料、产品或汇率套保中的锁定，不是发行人股份转让限制。",
        "MECHANICAL_PATENT": "机械锁定装置或叶轮系统专利，不是股份限制。",
        "CONSTRAINT_CONTEXT_SUPPLEMENTED": "已补读原候选及必要跨页内容，具体事项和证据见本轮条款记录；不因承诺存在就推定数量或触发状态。",
        "REGULATORY_BACKGROUND_OR_OTHER_STOCK_BUSINESS": "历史监管、国有股划转或证券业务/其他标的减持事项，不是当次配股新约束。",
        "ACCOUNTING_INVESTMENT_OR_THIRD_PARTY_BUSINESS": "投资收益持有期间、税费、第三方股票/客户抵押物或业务安排，与发行人新获配股份限制分开。",
    }
    output = {}
    for category, indices in groups.items():
        for i in indices:
            assert i not in output, i
            output[i] = {"semantic_class": category, "note": notes[category]}
    assert set(output) == set(range(248)), sorted(set(range(248)) - set(output))
    output[52]["note"] = "第128页标题将跨页文字明确连接到2015年12月18日问询函；‘2015年8月至今’不是2017年配股时点的持股限制现状。"
    output[175]["note"] = "表格归于IPO承诺，安徽国控集团锁定期满后两年内无减持计划；严格履行列为是，不等于已经到期。"
    output[200]["note"] = "广州证券收购对价股份48个月锁定明确继承至配股增加股份；条款与2020年3月11日登记日期分别保存，实际获配数量及后来变更仍需核对。"
    return output


def specs():
    gold = {"actor":"中国黄金集团公司", "origin":"2015年7月15日开始的增持安排",
            "initial_increase_shares":285400, "planned_increase_months":12,
            "restriction":"增持实施期间及法定期限内不减持所持有公司股份",
            "calendar_end":None, "new_rights_restricted_shares":None,
            "separate_from_listing_six_month_commitment":True}
    gold_proof = ["2015年7月15日,黄金集团通过上海证券交易所交易系统增持公司股份285,400股",
                  "黄金集团在未来12个月内继续(自本次增持之日算起)增持公司股份",
                  "黄金集团承诺,在增持实施期间及法定期限内不减持所持有的公司股份"]
    return [
        {"index":8,"claims":[claim("legacy_increase_non_reduction_commitment",deepcopy(gold), *[(22,t) for t in gold_proof])]},
        {"index":9,"claims":[claim("legacy_increase_non_reduction_commitment",deepcopy(gold), *[(36,t) for t in gold_proof])]},
        {"index":20,"claims":[claim("historical_inquiry_time_scope", {
            "inquiry_date":"2015-12-18", "inquiry_id":"中小板问询函2015第364号",
            "reported_restriction_since":"2015-08", "through":"问询函回复之日",
            "automatically_extended_to_prospectus_date":False,
            "restriction_at_2017_rights_date_inferred":False},
            (128,"2015年12月18日深交所出具的《关于对天齐锂业股份有限公司的问询函》"),
            (129,"截至问询函回复之日"),
            (129,"2015年8月至今,公司控股股东天齐集团所持有的公司股份9,371.70万股一直处于限售状态"))]},
        {"index":27,"claims":[claim("legacy_existing_holder_release", {
            "holder":"山东怡力电业有限公司", "as_of":"2018-06-30", "existing_shares":2163141993,
            "reported_release_date":"2020-06-06", "current_rights_allotment_coverage":"NOT_ESTABLISHED"},
            (24,"截至2018年6月30日"),(24,"山东怡力电业有限公司2,163,141,99323.38限售流通A股"),
            (24,"怡力电业持有南山铝业的股份全部为限售股,锁定期至2020年6月6日"))]},
        {"index":39,"claims":[
            claim("legacy_ipo_post_lock_reduction_limits", {
                "origin":"首次公开发行相关承诺", "trigger":"所持股份锁定期满", "years":2,
                "denominator":"公司首次上市之日总股本", "cumulative_not_annual":True,
                "groups":[{"actors":["武汉国资","人福医药","湖北省联发"],"cap_percent":"5"},
                          {"actors":["当代科技"],"cap_percent":"3.18"},
                          {"actors":["上海天阖"],"cap_percent":"2.18"},
                          {"actors":["当代明诚"],"cap_percent":"1.05"},
                          {"actors":["三特索道"],"cap_percent":"0.55"}],
                "advance_notice_trading_days":3,"calendar_boundaries":None,
                "restriction_on_current_rights_quantity":"NOT_COMPUTED"},
                (182,"与首次公开发行相关的承诺"),
                (183,"武汉国资、人福医药、湖北省联发承诺方在所持公司股份的锁定期满后两年内,转让股份数量累计不超过公司首次上市之日总股本的5%"),
                (183,"当代科技承诺方在所持公司股份的锁定期满后两年内,转让股份数量累计不超过公司首次上市之日总股本的3.18%"),
                (183,"上海天阖承诺方在所持公司股份的锁定期满后两年内,转让股份数量累计不超过公司首次上市之日总股本的2.18%"),
                (183,"当代明诚承诺方在所持公司股份的锁定期满后两年内,转让股份数量累计不超过公司首次上市之日总股本的1.05%"),
                (183,"三特索道承诺方在所持公司股份的锁定期满后两年内,转让股份数量累计不超过公司首次上市之日总股本的0.55%"),
                (184,"减持公司股份时提前三个交易日公告")),
            claim("conditional_competition_transfer_sanction", {
                "actors":["武汉国资","人福医药","湖北省联发","当代科技","上海天阖","当代明诚","三特索道"],
                "condition":"无合法理由违反避免同业竞争事项或未依法执行相应措施",
                "restriction":"持有的天风证券股份不得转让直至遵守承诺或执行约束措施",
                "source_reported_compliance":True,"activation_proved":False,"current_rights_restricted_shares":None},
                (203,"公司持股5%以上的主要股东及其一致行动人武汉国资、人福医药、湖北省联发、当代科技、上海天阖、当代明诚、三特索道"),
                (204,"本公司/企业无合法理由违反与避免同业竞争有关的事项,或者未依法执行相应措施的"),
                (204,"本公司/企业持有的天风证券股份不得转让,直至本公司/企业依法遵守有关承诺或依法执行有关约束措施"),
                (204,"自作出承诺以来始终严格遵守避免同业竞争承诺"))]},
        {"index":53,"claims":[claim("ipo_post_lock_no_reduction_plan", {
            "actor":"安徽国控集团","origin":"首次公开发行", "years_after_original_lock":2,
            "literal_strength":"无减持计划", "timely_strict_fulfilment_reported":True,
            "fulfilled_and_expired_inferred":False, "original_lock_end":None,"calendar_end":None,
            "current_rights_allotment_coverage":"NOT_ESTABLISHED", "current_rights_restricted_shares":None},
            (162,"与首次公开发行相关的承诺"),
            (162,"其他安徽国控集团锁定期满后2年内无减持计划锁定期满后2年内是是"))]},
        {"index":55,"claims":[claim("ipo_post_lock_reduction_limits", {
            "origin":"首次公开发行", "trigger":"持股锁定期满", "years":2,
            "annual_cap_percent":"1", "denominator":"红塔证券股份总数",
            "advance_notice_trading_days":3,
            "price_floor_group":["合和集团","双维投资","华叶投资","中烟浙江省公司","万兴地产"],
            "floor":"首次发行价格按除权除息事项调整",
            "other_price_rule_group":["云投集团","昆明产投","云南工投"],
            "stricter_future_rule_applies":True,"calendar_boundaries":None,"current_rights_restricted_shares":None},
            (170,"与首次公开发行相关的承诺其他合和集团、双维投资、华叶投资、中烟浙江省公司、万兴地产"),
            (170,"持股锁定期满后两年内,减持价格不低于红塔证券首次公开发行股票的发行价格"),
            (170,"每年减持股份数量不超过红塔证券股份总数的1%"),
            (170,"在实施减持时,将提前3个交易日通过红塔证券予以公告"),
            (170,"若减持时监管部门出台更为严格的减持规定,则应按届时监管部门要求执行"),
            (170,"与首次公开发行相关的承诺其他云投集团、昆明产投、云南工投持股锁定期满后两年内"))]},
        {"index":60,"claims":[
            claim("acquisition_lock_inherits_rights_shares", {
                "actors":["越秀金控","金控有限"], "origin":"发行股份收购广州证券",
                "base_lock_months":48,"trigger":"对价股份登记在承诺方名下之日",
                "longer_regulatory_lock_possible":True,
                "explicit_inherited_causes":["派息","送股","配股","资本公积金转增股本"],
                "timely_strict_fulfilment_reported":True,
                "inherited_rights_lock_clause_explicit":True,
                "actual_new_rights_shares_by_actor":None,"calendar_release_date":None},
                (195,"与发行股份收购广州证券相关的承诺股份限售越秀金控、金控有限"),
                (195,"对价股份登记在越秀金控/金控有限名下之日"),
                (195,"起48个月内不进行转让,除非中国证监会等监管机构提出更长锁定期要求"),
                (195,"如越秀金控、金控有限由于中信证券派息、送股、配股、资本公积金转增股本等原因增持的中信证券股份,亦应遵守上述约定"),
                (195,"自2019年公司发行股份购买资产起至限售期满。是")),
            claim("acquisition_share_registration", {
                "registration_completed_date":"2020-03-11",
                "original_consideration_shares":{"越秀金控":265352996,"金控有限":544514633},
                "sum_original_consideration_shares":809867629,
                "not_current_rights_allotment":True,"table_2019_label_not_registration_date":True},
                (51,"公司于2020年3月11日分别向越秀金控、金控有限发行265,352,996股、544,514,633股股份"),
                (51,"2020年3月11日,公司发行股份购买资产的新增股份登记在中国证券登记结算有限责任公司上海分公司办理完毕"))]},
        {"index":26,"addendum":True,"claims":[claim("legacy_preacquisition_holder_lock", {
            "actor":"南山集团", "origin":"2016发行股份购买资产", "scope":"交易前已持有股份",
            "months":12,"trigger":"收购行为完成", "same_controller_transfer_exception":True,
            "reported_no_breach":True,"current_rights_allotment_coverage":"NOT_ESTABLISHED"},
            (75,"2016年发行股份购买资产暨关联交易事项中作出重要承诺"),
            (75,"南山集团承诺“对本次交易前其所持有的上市公司股份,在本次收购行为完"),
            (76,"成后的12个月内不得转让,在同一实际控制人控制的不同主体之间进行转让不受前述12个月的限制"),
            (76,"南山集团未转让其持有的上市公司股份,无违背承诺之行为"))]},
        {"index":67,"addendum":True,"claims":[claim("current_rights_conditional_competition_transfer_sanction", {
            "actor":"申能集团", "table_context":"与本次配股相关的承诺", "condition":"违反避免同业竞争承诺",
            "restriction":"所持发行人股份不得转让直至相应措施实施完毕",
            "validity":"申能集团作为第一大股东期间", "source_reported_no_competition":True,
            "activation_proved":False, "current_rights_restricted_shares":None,
            "distinct_from_related_transaction_sanction":True},
            (123,"与本次配股相关的承诺避免同业竞争申能集团"),
            (123,"申能集团作为第一大股东期间是"),
            (124,"如违反上述关于避免与发行人同业竞争的承诺,申能集团将在违反相关承诺发生之日起停止在发行人处取得股东分红,同时持有的发行人股份不得转让"),
            (124,"直至按上述承诺采取相应的措施并实施完毕为止"),
            (150,"公司与公司第一大股东申能集团及其直接、间接控制的公司、企业不存在同业竞争关系"))]},
    ]


def finalize():
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    rows=read(OUT/"targets.json");dec=decisions();reviewed=[]
    targets=read(PREVIOUS/"targets.json");target_index={r["document_id"]:r for r in targets}
    source_targets={r["document_id"]:r for r in read(ROOT/"reports/research/510300_factor96_rights_prospectus_v1/targets.json")}
    for r in rows:
        i=r["pending_index"]
        reviewed.append({**r, **dec[i], "reviewed":True,
                         "reading_scope":"FULL_SAVED_CANDIDATE_PLUS_NEEDED_PAGES" if i in FULL_READ else "ALL_KEYWORD_CONTEXT_WINDOWS",
                         "full_candidate_read":i in FULL_READ,"trading_feature_admitted":False})
    facts=[];addenda=[]
    for spec in specs():
        r=targets[spec["index"]];rid=r["document_id"]
        f=base_fact(Document({**source_targets[rid]["source"],"review_role":"HOLDING_CONTEXT_SOURCE"}),None)
        f.update(reviewed_claims={},claim_evidence={},full_body_reviewed=False,effective_sellable_new_shares=None,strict_M06_pressure=None,
                 notes=["只在此来源目录日末代理时点保存所报告事实；不倒填到早年。", "条款、履行、实际获配数量分别判断，未推定即时可卖数量。"])
        pages={p["page"]:normalized(p["text"]) for p in read(r["text_path"])["pages"]}
        for c in spec["claims"]:
            f["reviewed_claims"][c["field"]]=c["value"]
            f["claim_evidence"][c["field"]]=[anchor(rid,pages,p,t) for p,t in c["proof"]]
        if spec.get("addendum"):
            f["append_only_claim_addendum"]=True
            addenda.append(f)
        else:
            facts.append(f)
    prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(prior)==445 and len(facts)==8 and len(addenda)==2
    assert not {f["document_id"] for f in facts}&{f["document_id"] for f in prior}
    assert {f["document_id"] for f in addenda}<={f["document_id"] for f in prior}
    extra=[]
    for i,pns in EXTRA_PAGES.items():
        r=targets[i];pages={p["page"]:normalized(p["text"]) for p in read(r["text_path"])["pages"]}
        for p in pns:extra.append({"document_id":r["document_id"],"page":p,"literal":pages[p],"full_page_read":True})
    queries=[{"document_id":f["document_id"],"addendum":f.get("append_only_claim_addendum",False),
              "before":source_view(f,(datetime.fromisoformat(f["known_at"])-timedelta(seconds=1)).isoformat()),
              "at":source_view(f,f["known_at"])} for f in facts+addenda]
    listing=read(OUT/"citic_followup_scope.json")["source"]
    listing_pages={p["page"]:normalized(p["text"]) for p in read(listing["text_path"])["pages"]}
    citic=next(f for f in facts if f["document_id"]=="1212175293")
    comparison={"prospectus_document_id":citic["document_id"],"prospectus_known_at":citic["known_at"],
                "listing_document_id":listing["document_id"],"relationship_known_at":listing["known_at"],
                "scope":"明确的配股继承约束与上市登记分类分别保留；不是来源数值互相改写。",
                "prospectus_claims":deepcopy(citic["reviewed_claims"]),"listing_source_claims":deepcopy(listing),
                "pre_listing_publication_days":(datetime.fromisoformat(listing["known_at"])-datetime.fromisoformat(citic["known_at"])).days,
                "current_rights_effective_sellable_shares":None,"actual_allotment_by_actor":None,
                "zero_effective_restrictions_proved_by_registration":False,
                "computed_from_holder_difference":False,"actual_new_shares_from_original_amount_times_ratio":False,
                "followup_pages_read":[1,2,4,5],"full_listing_body_read":False,
                "listing_full_pages":[{"page":p,"literal":listing_pages[p]} for p in (1,2,4,5)],
                "next_evidence":"当次越秀两家主体的实际认购披露及承诺变更/履行材料；不将股东表变动自动等同于认购。"}
    targeted=[]
    for rid in ("1212175292","1212286046"):
        path=ROOT/f"reports/research/510300_factor96_rights_issue_documents_v1/text/{rid}.json"
        pages=read(path)["pages"]
        hits=[p["page"] for p in pages if any(t in normalized(p["text"]) for t in ("越秀","金控有限"))]
        assert not hits
        targeted.append({"document_id":rid,"text_path":str(path),"text_sha256":digest(path),"page_count":len(pages),
                         "terms":["越秀","金控有限"],"matched_pages":hits,"scope":"只核对两个字符串，不等于该公司未认购或其他来源不存在。"})
    counts={}
    for r in reviewed:counts[r["semantic_class"]]=counts.get(r["semantic_class"],0)+1
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_HOLDING_REMAINDER_V1",
            "status":"PROGRESS_248_KEYWORD_CANDIDATES_CLASSIFIED_WITH_EXPLICIT_RIGHTS_INHERITANCE_FOUND",
            "remaining_source_candidates_reviewed":248,"target_documents":65,"keyword_windows_read":293,
            "keyword_window_characters_read":54277,"full_saved_candidates_read":len(FULL_READ),
            "extra_full_prospectus_pages_read":len(extra),"full_text_read_claimed":False,
            "pending_keyword_candidates":0,"semantic_counts":counts,
            "new_context_source_facts":len(facts),"append_only_claim_addenda":len(addenda),
            "source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts+addenda),
            "prior_source_facts_unchanged":445,"combined_source_facts":445+len(facts),
            "explicit_rights_share_lock_inheritance_clauses":1,"conditional_transfer_activation_proved":False,
            "current_rights_sellable_quantities_computed":0,"complete_holding_identity_and_quantity_coverage":False,
            "source_publication_boundary_views":len(queries)*2,"visual_pages_reviewed":len(VISUAL_PAGES),
            "new_accounts":0,"new_returns":0,"new_models":0,"new_network_requests":0,
            "qualified_candidates":0,"independent_forward_observations":0,"full_issuance_coverage":False,
            "free_float_denominator_available":False,"strict_M06_pressure":"NOT_COMPUTED",
            "T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
            "orders_authorized":False,"delivery_package_created":False,"goal_achieved":False,
            "next_source_action":"RESOLVE_CITIC_RIGHTS_ALLOTMENT_AND_INHERITED_LOCK_WITH_HUAAN_AND_PRIOR_SCOPE_GAPS"}
    outputs={"reviewed_candidates":reviewed,"reviewed_context_source_facts":facts,"claim_addenda":addenda,
             "combined_source_facts":prior+facts,"extra_pages_read":extra,"source_publication_queries":queries,
             "citic_inheritance_vs_listing_classification":comparison,"citic_targeted_name_searches":targeted,
             "remaining_holding_candidates":{"count":0,"candidates":[],"scope":"固定248段关键词候选全部完成窗口核读；不等于74份PDF全文已读或承诺身份数量齐全。"},
             "result":result}
    for name,obj in outputs.items():save(OUT/(name+".json"),obj)
    print(json.dumps({"关键词候选":248,"新来源":len(facts),"补充来源":len(addenda),"字段":result['source_fields_reviewed'],
                      "明确配股继承条款":1,"可卖数量":None},ensure_ascii=False))


if __name__=="__main__":finalize()
