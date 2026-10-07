"""保存23份已读预案的来源声明、同日补充和版本差异；不生成交易信号。"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

from research.factor96_rights_preplans_v1 import OUT, PREVIOUS, SOURCE, ROOT, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import claim, money

VISUAL_PAGES = [("1209640415",3),("1205272661",3),("1207129939",3),("1215952169",4)]


def specs():
    # 每条均人工指定当次发行章节；表中数量是有条件测算，不能代替实际发行。
    rows = []

    def add(i, group, ratio, relation, rp, basis_asof, basis, quantity, qp, budget, validity_page, validity_literal,
            a_shares=None, h_shares=None, second=None, basis_page=None):
        budget_value, budget_unit, budget_page, budget_literal, budget_relation = budget
        scenario = {"basis_asof":basis_asof,"basis_shares":basis,"reported_total_shares":quantity,
                    "A_shares":quantity if a_shares is None else a_shares,"H_shares":h_shares,
                    "scope":"A_ONLY" if h_shares is None else "A_AND_H_COMBINED",
                    "condition":"按所列历史或当时股本测算，实施前股本变化时调整；不是实际认购量。"}
        qp_proofs = [(basis_page or qp, f"{basis:,}股")]
        if i != 17:
            qp_proofs.append((qp, f"{quantity:,}股"))
        else:
            # 原文本数字缺失，以已查看的PDF原图补充单独证据，不改写原文本。
            qp_proofs.append((qp, "本次配股数量不超过股"))
        if h_shares is not None:
            qp_proofs.extend([(qp,f"{a_shares:,}股"),(qp,f"{h_shares:,}股")])
        scenarios = [scenario]
        if second is not None:
            b,q = second
            scenarios[0]["condition"] = "当前享有配股权利的股本，已经扣除回购专户股份。"
            scenarios.append({"basis_asof":"处置完回购专户股份后假设","basis_shares":b,"reported_total_shares":q,
                "A_shares":q,"H_shares":None,"scope":"A_ONLY","condition":"处置完回购专户所持股份的替代测算，不与第一情景相加。"})
            qp_proofs.extend([(qp,f"{b:,}股"),(qp,f"{q:,}股")])
        ratio_literal = f"每10股配售{'不超过' if relation == 'CAP' else ''}{ratio}股"
        rows.append({"index":i,"group":group,"claims":[
            claim("plan_ratio_per_10",{"reported_value":ratio,"relation":relation},(rp,ratio_literal)),
            claim("plan_quantity_scenarios",scenarios,*qp_proofs),
            claim("plan_gross_proceeds",{**money(budget_value,budget_unit),"relation":budget_relation,
                "scope":scenario["scope"]},(budget_page,budget_literal)),
            claim("resolution_validity",{"months":12,"reported_trigger":validity_literal,"calendar_end":None,
                "interpretation":"保存预案所写起算条件，不从该条款单独确认批准已完成或推算当前有效期。"},(validity_page,validity_literal))]})

    add(0,"WESTERN_PLAN", "2.6","SPECIFIED",2,"现有总股本",2795569620,726848101,2,
        ("50","亿元",3,"募集资金总额不超过人民币50亿元","CAP"),3,"自公司股东大会审议通过之日起12个月内有效")
    add(1,"GUOHAI_2016", "3","SPECIFIED",2,"2016-09-30",4215541972,1264662591,2,
        ("50","亿元",3,"募集资金总额不超过人民币50亿元","CAP"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(2,"ENN_2017", "3","CAP",3,"2016-12-31",985785043,295735512,3,
        ("230000","万元",5,"拟募集资金总额不超过人民币230,000万元(含发行费用)","CAP"),5,"自公司股东大会审议通过之日起12个月内有效")
    add(3,"WANXIANG_2017", "3","CAP",2,"2016年利润分配后",2753159454,825947836,2,
        ("62.50","亿元",3,"预计募集资金总额不超过62.50亿元(含发行费用)","CAP"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(4,"DONGWU_2017", "3","SPECIFIED",3,"截至目前总股本",3000000000,900000000,3,
        ("65","亿元",4,"募集资金总额预计为不超过人民币65亿元","CAP"),5,"自股东大会审议通过之日起12个月内有效")
    add(5,"WANXIANG_2017", "3","SPECIFIED",2,"2018-03-31利润分配后",2753159454,825947836,2,
        ("56.60","亿元",3,"预计募集资金总额为56.60亿元(含发行费用)","EXPECTED_AMOUNT"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(6,"DONGWU_2017", "3","SPECIFIED",2,"截至目前总股本",3000000000,900000000,3,
        ("65","亿元",4,"募集资金总额预计为不超过人民币65亿元","CAP"),6,"自股东大会审议通过之日起12个月内有效")
    add(7,"GUOHAI_2016", "3","SPECIFIED",2,"2018-09-30",4215541972,1264662591,2,
        ("50","亿元",3,"募集资金总额不超过人民币50亿元","CAP"),4,"自公司延长前次股东大会决议有效期的股东大会决议通过之日起12个月内有效")
    add(8,"GOLDWIND_2018", "1.9000","SPECIFIED",3,"2018-12-31",3556203300,675678627,3,
        ("474418.30","万元",4,"募集资金总额不超过人民币474,418.30万元","CAP"),5,"自公司股东大会审议通过之日起12个月内有效",
        a_shares=552167067,h_shares=123511560)
    add(9,"TIANQI_2019", "3","SPECIFIED",2,"本预案修订稿出具日",1141987945,342596383,3,
        ("700000.00","万元",4,"募集资金总额不超过人民币700,000.00万元","CAP"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(10,"TIANFENG_2019", "3","SPECIFIED",2,"2019-06-30",5180000000,1554000000,2,
        ("80","亿元",3,"募集资金总额不超过人民币80亿元","CAP"),4,"自公司股东大会审议通过之日起十二个月内有效")
    add(11,"CMS_2019", "3","SPECIFIED",2,"2019-09-30",6699409329,2009822798,2,
        ("150","亿元",3,"募集资金总额不超过人民币150亿元","CAP"),4,"自公司股东大会、A股类别股东会议、H股类别股东会议审议通过之日起12个月内有效",
        a_shares=1715702444,h_shares=294120354)
    add(12,"KELUN_2019", "1.4","SPECIFIED",3,"预案修订稿出具日扣除回购专户",1430497345,200269628,3,
        ("200000.00","万元",4,"拟募集资金总额不超过人民币200,000.00万元(含发行费用)","CAP"),4,"自公司股东大会审议通过之日起12个月内有效",
        second=(1439786060,201570048))
    add(13,"KELUN_2019", "1.4","SPECIFIED",3,"预案第二次修订稿出具日扣除回购专户",1429106145,200074860,3,
        ("200000.00","万元",4,"拟募集资金总额不超过人民币200,000.00万元(含发行费用)","CAP"),4,"自公司股东大会审议通过之日起12个月内有效",
        second=(1439786060,201570048))
    add(14,"SHANXI_2019", "3","SPECIFIED",2,"2019-09-30",2828725153,848617545,2,
        ("60","亿元",3,"募集资金总额不超过人民币60亿元","CAP"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(15,"GUOYUAN_2019", "3","SPECIFIED",2,"2019-12-31",3365447047,1009634114,2,
        ("55","亿元",3,"募集资金总额不超过人民币55亿元","CAP"),3,"自公司股东大会审议通过之日起12个月内有效")
    add(16,"JIANGSU_BANK_PLAN", "3","SPECIFIED",2,"2020-06-30",11544508967,3463352690,2,
        ("200","亿元",3,"募集资金不超过人民币200亿元(含200亿元)","CAP"),4,"本行股东大会审议通过本次配股方案之日起十二个月")
    add(17,"ZHONGKE_2020", "2","CAP",3,"2020-12-31",1065200000,213040000,3,
        ("72000.00","万元",4,"募集资金总额预计不超过人民币72,000.00万元","CAP"),5,"自公司股东大会审议通过本次配股议案之日起12个月内有效")
    add(18,"CITIC_SECURITIES_PLAN", "1.5","SPECIFIED",2,"2020-12-31",12926776029,1939016404,2,
        ("280","亿元",3,"募集资金总额不超过人民币280亿元","CAP"),3,"自公司股东大会、A股类别股东会、H股类别股东会审议通过之日起12个月内有效",
        a_shares=1597267249,h_shares=341749155)
    add(19,"CAITONG_2021", "3","SPECIFIED",2,"2021-06-30",3589009508,1076702852,2,
        ("80","亿元",3,"拟募集资金总额不超过人民币80亿元","CAP"),4,"自公司股东大会审议通过之日起12个月内有效")
    add(20,"ZHESHANG_BANK_PLAN", "3","SPECIFIED",2,"2022-03-31",21268696778,6380609033,2,
        ("180","亿元",3,"募集资金不超过人民币180亿元(含180亿元)","CAP"),4,"自本公司股东大会、A股类别股东大会、H股类别股东大会审议通过之日起12个月内有效",
        a_shares=5014409033,h_shares=1366200000)
    add(21,"CITIC_BANK_PLAN", "3","SPECIFIED",2,"2022-12-31",48934843657,14680453097,2,
        ("400","亿元",3,"募集资金不超过人民币400亿元(含400亿元)","CAP"),3,"自本行股东大会、A股类别股东会和H股类别股东会审议通过之日起12个月内有效",
        a_shares=10215804204,h_shares=4464648893)
    add(22,"CITIC_BANK_PLAN", "3","SPECIFIED",2,"2022-12-31",48934843657,14680453097,2,
        ("400","亿元",3,"募集资金不超过人民币400亿元(含400亿元)","CAP"),4,"自本行股东大会、A股类别股东会和H股类别股东会审议通过之日起12个月内有效",
        a_shares=10215804204,h_shares=4464648893)

    def extra(i, *claims):
        rows[i]["claims"].extend(claims)

    extra(2, claim("remaining_approvals",["CSRC"],(2,"本次配股尚需获得中国证券监督管理委员会的核准")),
        claim("use", "年产20万吨稳定轻烃项目",(5,"扣除发行费用后的净额将全部用于投资年产20万吨稳定轻烃项目")))
    for i in (3,5):
        extra(i, claim("remaining_approvals",["SHAREHOLDERS","CSRC"],(22,"尚需公司股东大会批准、经证监会核准后方可实施")),
            claim("conditional_issue_window_months_after_future_CSRC_approval",6,(3,"公司将在中国证监会核准本次配股后的6个月内择机向全体股东配售股份")))
    extra(3, claim("proposed_debt_repayment",money("159000","万元"),(3,"拟使用募集资金投入(万元)"),(3,"偿还公司债券159,000159,000")))
    extra(5, claim("proposed_debt_repayment",money("100000","万元"),(3,"拟使用募集资金投入(万元)"),(3,"偿还公司债券159,000100,000")))
    extra(7, claim("preplan_shareholder_approval","PENDING",(2,"尚需提交公司股东大会审议")))
    extra(8, claim("remaining_approvals",["CSRC"],(5,"修订后的本次配股预案需中国证监会核准后方可实施")))
    extra(9, claim("use","偿还购买SQM23.77%股权的部分并购贷款",(4,"净额拟全部用于偿还购买SQM23.77%股权的部分并购贷款")))
    extra(10, claim("remaining_approvals",["CSRC"],(4,"待报中国证监会核准后方可实施")))
    extra(11, claim("remaining_approvals",["CSRC"],(4,"待中国证监会核准后方可实施")),
        claim("A_record_date_basis_excludes_treasury",True,(2,"总股本扣除公司回购专用账户持有的本公司股份后的股份总数为基数")))
    for i in (12,13):
        extra(i, claim("remaining_approvals",["CSRC"],(4,"尚须经中国证监会核准后方可实施")),
            claim("use_components",{"debt_repayment":money("150000.00","万元"),"working_capital":money("50000.00","万元")},
                  (16,"单位:万元"),(16,"偿还有息债务150,000.00"),(16,"补充流动资金50,000.00")))
    extra(14, claim("dilution_completion_assumption_excluded_from_calendar","2020-06-30",
        (27,"假设本次配股于2020年6月30日完成"),(27,"上述时间仅用于计算本次配股摊薄即期回报对主要财务指标的影响")))
    extra(15, claim("dilution_completion_assumption_excluded_from_calendar","2019-12-31",
        (47,"假设本次配股于2019年12月31日完成"),(47,"上述时间仅用于计算本次配股摊薄即期回报对主要财务指标的影响")))
    extra(16, claim("capital_basis_adjustment_includes_convertible_conversion",True,(2,"可转债转股及其他原因导致本行总股本变动")),
        claim("use","核心一级资本",(3,"净额将全部用于补充本行的核心一级资本")))
    extra(17, claim("remaining_approvals",["STATE_ASSET_AUTHORITY","SHAREHOLDERS","CSRC"],
        (6,"尚须取得相关国有资产监督和管理部门的批复,并须经公司股东大会表决通过,及中国证监会核准后方可实施")),
        claim("quantity_number_missing_in_saved_text",True,(3,"本次配股数量不超过股")))
    extra(18, claim("remaining_approvals",["CSRC"],(4,"尚待中国证监会核准后方可实施")))
    extra(19, claim("remaining_approvals",["CSRC"],(4,"尚待中国证监会核准后方可实施")),
        claim("dilution_completion_assumption_excluded_from_calendar","2021-11-30",
            (26,"假设本次配股于2021年11月30日完成"),(26,"上述时间仅用于计算本次配股摊薄即期回报对主要财务指标的影响")),
        claim("dilution_quantity_scenario_separate",{"basis_asof":"2020-12-31","basis_shares":3589000000,"shares":1076700000},
            (25,"假设本次配股比例为每10股配售3股"),(25,"截至2020年12月31日的总股本3,589,000,000股"),(25,"最大可配售数量1,076,700,000股")))
    extra(20, claim("bank_regulatory_approval_key","银保监复〔2022〕163号",(4,"银保监复〔2022〕163号")),
        claim("remaining_approvals",["CSRC_AND_OTHER_COMPETENT_AUTHORITIES"],(4,"本次配股待取得中国证监会等有权部门核准后方可实施")),
        claim("use","核心一级资本",(3,"将全部用于补充本公司的核心一级资本")))
    for i in (21,22):
        extra(i, claim("bank_regulatory_approval_key","银保监复〔2022〕751号",(4,"银保监复〔2022〕751号")),
            claim("use","核心一级资本",(3,"将全部用于补充本行的核心一级资本")))
    extra(21, claim("remaining_approvals",["CSRC_APPROVAL","OTHER_REQUIRED_APPROVALS"],
        (4,"尚待中国证监会核准及取得本次配股涉及的其他必要批准后方可实施")))
    extra(22, claim("remaining_approvals",["SSE_REVIEW","CSRC_REGISTRATION","OTHER_REQUIRED_APPROVALS"],
        (4,"尚需经上海证券交易所审核并获得中国证监会同意注册及取得本次配股涉及的其他必要批准后方可实施")))
    assert len(rows) == 23
    return rows


def anchor(rid, pages, pn, literal):
    literal = normalized(literal)
    pos = pages[pn].find(literal)
    assert pos >= 0, (rid,pn,literal)
    return {"document_id":rid,"page":pn,"normalized_start":pos,"normalized_end":pos+len(literal),"literal":literal}


def build():
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    targets = read(OUT/"targets.json")
    output = []
    for spec in specs():
        row = targets[spec["index"]]
        doc = Document({**row["source"],"review_role":"PREPLAN_OPERATIVE_SOURCE_REVIEW"})
        fact = base_fact(doc,None)
        fact.update(plan_source_group=spec["group"],reviewed_claims={},claim_evidence={},
            full_body_reviewed=False,operative_clause_review_complete=True,holding_restrictions="NOT_ESTABLISHED_BY_THIS_REVIEW",
            actual_record_date=None,actual_payment_calendar=None,actual_issue_price=None,
            effective_sellable_new_shares=None,strict_M06_pressure=None)
        pages = {p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            assert c["field"] not in fact["reviewed_claims"]
            fact["reviewed_claims"][c["field"]] = c["value"]
            fact["claim_evidence"][c["field"]] = [anchor(row["document_id"],pages,p,t) for p,t in c["proof"]]
        if spec["index"] == 17:
            png = OUT/"page_previews/1209640415_p3.png"
            fact["visual_evidence"] = [{"field":"plan_quantity_scenarios[0].reported_total_shares", "value":213040000,
                "document_id":row["document_id"],"page":3,"path":str(png),"sha256":digest(png),
                "transcribed_literal":"本次配股数量不超过213,040,000股", "visually_reviewed":True,
                "reason":"原PDF数字可见，旧文本提取漏掉该数字；保留旧文本不改写。"}]
        fact["notes"] = ["当次预案来源声明，以目录日末作为可得日期代理；不证明历史首次公开。",
            "同一来源组用于比较计划版本，不是已实施事件身份；未将预案数量、募资预算或测算日期当作实际发行。",
            "未在已读候选中确认新的当次持有期条款；未知约束不是无约束，认购承诺不是不减持承诺。"]
        output.append(fact)
    return output


def source_view(fact, at):
    visible = fact["known_at"] <= at
    return {"document_id":fact["document_id"],"as_of":at,"known_at":fact["known_at"],
        "status":"REVIEWED_PREPLAN_SOURCE_DATE_PROXY_ONLY" if visible else "NO_VIEW",
        "reviewed_claims":deepcopy(fact["reviewed_claims"]) if visible else {},
        "actual_event_id":None,"actual_issuance_calendar":None,"strict_M06_pressure":None,"trading_feature_admitted":False}


def candidate_reviews():
    # 列出的片段均已阅读。未列入例外的片段属于当次发行条款和待批准声明。
    duplicate = {0:[2],1:[3],2:[3],3:[3],4:[3],5:[3,4],6:[4,5],7:[3],8:[3],9:[2],
        10:[4],11:[4],12:[3],13:[3],14:[4,6],15:[3],17:[5],18:[4],19:[4,7]}
    historical = {2:[4,5],9:[3,4],10:[5],12:[4],13:[4],14:[7],15:[5],17:[6],18:[5],19:[8],20:[3,4,5],21:[4],22:[4]}
    hypothetical = {14:[5],15:[4],19:[5,6]}
    results = []
    for i,row in enumerate(read(OUT/"source_candidates.json")):
        for j,c in enumerate(row["candidates"]):
            role = "CURRENT_PLAN_CLAUSE"
            if j in duplicate.get(i,[]): role = "REPEATED_USE_DISCUSSION_NOT_ADDITIONAL_SUPPLY"
            if j in historical.get(i,[]): role = "DIVIDEND_HISTORY_OR_DIVIDEND_PROPOSAL_NOT_RIGHTS_TERMS"
            if j in hypothetical.get(i,[]): role = "DILUTION_ASSUMPTION_NOT_ACTUAL_CALENDAR_OR_QUANTITY"
            if i in (21,22) and j == 5: role = "HISTORICAL_PREFERENCE_SHARES_NOT_CURRENT_RIGHTS"
            if i == 17 and j == 7: role = "DILUTION_MEASURE_PENDING_SHAREHOLDER_APPROVAL"
            results.append({"document_id":row["document_id"],"candidate_index":j,**c,"reviewed":True,"review_role":role})
    assert len(results) == 127
    return results


def comparisons(facts, prior):
    idx = {r["document_id"]:r for r in prior}
    admin_ids = {0:"1202686497",1:"1202839814",2:"1203759865",3:"1203978535",4:"1204532468",
        6:"1205275256",7:"1205629671",8:"1205782946",9:"1206375058",10:"1206904222",11:"1207064173",
        14:"1207165605",15:"1207246801",16:"1208463992",17:"1209640414",18:"1210759314",19:"1210958294",
        20:"1213621618",21:"1215889024",22:"1215952175"}
    same_day = []
    for i,rid in admin_ids.items():
        f,a = facts[i],idx[rid]
        assert f["symbol"] == a["symbol"] and f["known_at"] == a["known_at"]
        numeric_missing = a["reviewed_claims"].get("quantity_detail_referenced_not_numeric",False)
        same_day.append({"preplan_id":f["document_id"],"admin_id":rid,"relationship_known_at":f["known_at"],
            "relationship_scope":"同发行人同日方案修订及预案的来源对照；不按公告ID排序覆盖，不创建两次发行。",
            "quantity_not_numeric_in_admin":numeric_missing,
            "new_preplan_quantity_source":f["reviewed_claims"]["plan_quantity_scenarios"],
            "old_admin_fact_unchanged":True,"actual_event_id":None,"trading_feature_admitted":False})
    assert len(same_day) == 20 and sum(r["quantity_not_numeric_in_admin"] for r in same_day) == 9
    version_pairs = []
    for first,second in [(1,7),(3,5),(4,6),(12,13),(21,22)]:
        a,b = facts[first],facts[second]
        assert a["plan_source_group"] == b["plan_source_group"] and a["known_at"] < b["known_at"]
        fields = sorted(set(a["reviewed_claims"]) | set(b["reviewed_claims"]))
        changed = {k:{"earlier":a["reviewed_claims"].get(k),"later":b["reviewed_claims"].get(k)}
                   for k in fields if a["reviewed_claims"].get(k) != b["reviewed_claims"].get(k)}
        version_pairs.append({"plan_source_group":a["plan_source_group"],"earlier_source":a["document_id"],
            "later_source":b["document_id"],"relationship_known_at":b["known_at"],"changed_source_fields":changed,
            "interpretation":"文档字段对照；缺少一字段不等于条款失效，不推断本轮之前的原始首版或实际实施。",
            "actual_event_id":None})
    terminals = []
    for indices,rid,note in [([3,5],"1206233867","万向2017方案同一发行人及825947836股上限与后验核准失效公告相符。"),
                             ([4,6],"1205354373","东吴2017方案与2018终止公告对应，不混入2020配股。"),
                             ([12,13],"1207329075","科伦两份2019预案与2020终止2019方案公告对应。")]:
        end = idx[rid]
        assert all(facts[i]["symbol"] == end["symbol"] and facts[i]["known_at"] < end["known_at"] for i in indices)
        terminals.append({"prior_preplan_ids":[facts[i]["document_id"] for i in indices],"terminal_source_id":rid,
            "relationship_known_at_no_earlier_than":end["known_at"],"terminal_status":end["administrative_status"],
            "note":note,"terminal_fact_unchanged":True,"future_approval_backfilled":False,
            "joined_to_executed_event":False,"trading_feature_admitted":False})
    return same_day,version_pairs,terminals


def finalize():
    facts = build()
    prior = read(PREVIOUS/"combined_source_facts.json")
    assert len(prior) == 376 and len(facts) == 23
    assert not {r["document_id"] for r in prior} & {r["document_id"] for r in facts}
    reviewed = candidate_reviews()
    same_day,versions,terminals = comparisons(facts,prior)
    save(OUT/"reviewed_preplan_source_facts.json",facts)
    save(OUT/"combined_source_facts.json",prior+facts)
    save(OUT/"reviewed_source_candidates.json",reviewed)
    extras = read(OUT/"additional_holding_candidates.json")
    assert len(extras) == 3
    for i,c in enumerate(extras):
        c.update(reviewed=True,review_role="ISSUANCE_WINDOW_NOT_HOLDING_PERIOD" if i<2 else "PAST_SALES_AND_SUBSCRIPTION_PROMISE_NOT_NEW_LOCK")
    save(OUT/"reviewed_additional_holding_candidates.json",extras)
    queries = []
    for fact in facts:
        at = fact["known_at"]
        before = (datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
        queries.append({"document_id":fact["document_id"],"before":source_view(fact,before),"at":source_view(fact,at)})
    save(OUT/"preplan_publication_queries.json",queries)
    save(OUT/"same_day_admin_comparisons.json",same_day)
    save(OUT/"preplan_version_comparisons.json",versions)
    save(OUT/"terminal_source_comparisons.json",terminals)
    with (OUT/"23份配股预案条款.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=["公告ID","证券","可得日期代理","标题","计划来源组","配股比例","数量情景","募资预算","其他来源字段","来源URL"])
        writer.writeheader()
        for f in facts:
            c=f["reviewed_claims"]
            writer.writerow({"公告ID":f["document_id"],"证券":f["symbol"],"可得日期代理":f["known_at"],"标题":f["title"],
                "计划来源组":f["plan_source_group"],"配股比例":json.dumps(c["plan_ratio_per_10"],ensure_ascii=False),
                "数量情景":json.dumps(c["plan_quantity_scenarios"],ensure_ascii=False),
                "募资预算":json.dumps(c["plan_gross_proceeds"],ensure_ascii=False),
                "其他来源字段":json.dumps({k:v for k,v in c.items() if k not in ("plan_ratio_per_10","plan_quantity_scenarios","plan_gross_proceeds")},ensure_ascii=False),
                "来源URL":f["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_PREPLANS_V1",
        "status":"23_PREPLAN_OPERATIVE_SOURCE_CLAUSES_REVIEWED",
        "documents":23,"target_pdf_pages":688,"full_pdf_pages_read_claimed":False,
        "candidate_fragments_read":127,"candidate_characters":45032,"additional_holding_candidates_read":3,
        "candidate_role_counts":dict(Counter(r["review_role"] for r in reviewed)),
        "source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts),"source_plan_groups":18,
        "same_day_admin_preplan_pairs":20,"numeric_notice_gaps_resolved_by_separate_preplan_source":9,
        "preplan_version_pairs":5,"terminal_comparisons":3,"terminal_preplan_documents":6,
        "A_H_combined_documents":6,"treasury_alternative_scenario_documents":2,
        "dilution_completion_dates_excluded":3,"new_current_holding_clauses_confirmed":0,
        "visual_pages_reviewed":len(VISUAL_PAGES),"text_missing_quantities_recovered_from_original_PDF":1,
        "prior_source_documents_unchanged":376,"combined_source_documents":399,
        "new_actual_issuance_calendars":0,"new_preplan_publication_queries":46,"remaining_long_documents":40,
        "earlier_holding_clause_full_coverage":False,"full_issuance_coverage":False,"free_float_denominator_available":False,
        "strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"qualified_candidates":0,
        "independent_forward_observations":0,"orders_authorized":False,"delivery_package_created":False,
        "goal_achieved":False,"next_source_action":"T13_40_FEASIBILITY_DILUTION_RESPONSE_AND_DRAFT_DOCUMENTS_AND_EARLIER_HOLDING_COVERAGE"}
    save(OUT/"result.json",result)
    print(json.dumps({"预案":23,"来源字段":result["source_fields_reviewed"],"补足数字的修订说明":9,
        "原图找回数字":1,"来源总数":399,"剩余长文件":40,"新增账户":0},ensure_ascii=False))


if __name__ == "__main__":
    finalize()
