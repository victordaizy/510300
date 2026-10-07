"""保存18份监管回复的来源条款，区分旧提问、现行答复及经营会计材料。"""
from copy import deepcopy
from datetime import datetime, timedelta
import csv
import json

from research.factor96_rights_responses_v1 import OUT, PREVIOUS, ROOT, SOURCE, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import money, claim
from research.factor96_rights_preplans_review_v1 import anchor

VISUAL_PAGES=[("1213982762",29),("1213982762",30),("1205666143",4)]


def allocation(name,value,unit="亿元"):
    return {"purpose":name,**money(value,unit)}


def quantity(basis,shares,ratio="3",relation="SPECIFIED",basis_asof=None,version="CURRENT_REPLY"):
    return {"basis_shares":basis,"reported_A_shares":shares,"ratio_per_10":ratio,"ratio_relation":relation,
            "basis_asof":basis_asof,"version":version,"conditional_on_basis":True,"actual_issuance_quantity":False}


def specs():
    output=[
      {"index":0,"group":"INSPUR_2017_RESPONSE","claims":[
        claim("response_scope","VIT应收账款坏账计提及发行条件核查",(2,"应收账款全额计提坏账准备")),
        claim("reported_past_no_reduction",{"holder":"浪潮集团","reported_start":"2014-01-01","end":"本回复所称至今","prospective_commitment":False},
              (6,"2014年1月1日至今,浪潮集团不存在减持上市公司股票的情形。"))]},
      {"index":1,"group":"INSPUR_2017_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("31"),(12,"公司本次配股拟募集资金31亿元")),
        claim("proposed_allocations",[allocation("补充流动资金","178000","万元"),allocation("偿还银行贷款","100000","万元"),
              allocation("模块化数据中心研发与产业化","21000","万元"),allocation("全闪存阵列研发与产业化","11000","万元")],
              (87,"1补充流动资金-178,000.002偿还银行贷款-100,000.003模块化数据中心研发与产业化项目50,658.4521,000.00"),
              (88,"4全闪存阵列研发与产业化项目30,105.0011,000.00合计-310,000.00"))]},
      {"index":2,"group":"ENN_2017_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("23"),(2,"本次调整后,公司本次配股募集资金总额预计不超过23亿元")),
        claim("regulatory_question_old_budget",{"amount":money("24"),"role":"SUPERSEDED_QUESTION_NOT_CURRENT_REPLY"},
              (36,"本次配股拟募集资金24亿元,与前次终止的非公开发行方案拟募集资金金额相近")),
        claim("proposed_allocations",[allocation("年产20万吨稳定轻烃项目资本化支出","230000","万元")],
              (32,"方案调整后的募集资金总额为230,000万元,均将用于年产20万吨稳定轻烃项目的资本化支出。")),
        claim("working_capital_use_removed",money("10000","万元"),(32,"补充流动资金10,000万元将不再作为本次配股公开发行的募投项目。")),
        claim("historical_other_financing_excluded",{"type":"NON_PUBLIC_OFFERING","amount":money("24.7"),"reported_termination_date":"2017-04-18","current_rights_financing":False},
              (36,"拟募集资金总额不超过24.7亿元,全部用于收购联信创投100%股权。申请人于2017年4月18日公告终止前次非公开发行股票事项。"))]},
      {"index":3,"group":"NANSHAN_2018_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("500000.00","万元"),(14,"本次配股发行拟募集资金规模不超过500,000.00万元")),
        claim("proposed_allocations",[allocation("项目建设资本性支出","487814.32","万元"),allocation("项目流动资金","12185.68","万元")],
              (14,"项目建设投资尚需487,814.32万元,全部以募集资金投入"),(14,"预计不超过12,185.68万元,该部分投入属于非资本性支出")),
        claim("project_total_not_issuance_budget",money("568539.30","万元"),(14,"本次募投项目总投资568,539.30万元"))]},
      {"index":4,"group":"DONGWU_2017_PLAN_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("65"),(38,"本次配股募集资金总额不超过人民币65亿元")),
        claim("proposed_allocations",[allocation("子公司投入","15"),allocation("信用交易业务","20"),allocation("自营业务","25"),allocation("资产管理业务","5")],
              (2,"1加大对子公司投入,加快证券控股集团的建设不超过15亿元2进一步扩大信用交易业务规模,优化公司收入结构不超过20亿元3扩大自营业务规模不超过25亿元4大力发展资产管理业务,提高主动管理能力不超过5亿元合计不超过65亿元")),
        claim("plan_quantity_scenarios",[quantity(3000000000,900000000)],
              (28,"按每10股配售3股的比例向全体股东配售"),(29,"总股本3,000,000,000股为基数测算,则本次可配售股份数量总计900,000,000股")),
        claim("suballocations_not_added_to_total",[allocation("香港海外基金","8.5"),allocation("香港信息系统","1.5")],
              (2,"本次募集资金拟用于投入国新国信东吴海外基金1不超过人民币8.5亿元"),
              (2,"本次拟用于加强对信息系统的研发与建设投入的募集资金规模不超过人民币1.5亿元")),
        claim("sector_regulatory_opinion_not_issue_approval",{"key":"机构部函【2017】2930号","reported_date":"2017-12-26","validity_years":1,"CSRC_issue_approval_inferred":False},
              (39,"2017年12月26日,中国证监会证券基金机构监管部出具《关于东吴证券配股事项的监管意见书》(机构部函【2017】2930号)"),
              (39,"监管意见书有效期一年"))]},
      {"index":5,"group":"LONGI_2018_PLAN_RESPONSE","claims":[
        claim("plan_quantity_scenarios",[quantity(2791679915,837503974,relation="CAP",basis_asof="2018-06-30",version="PREVIOUS_PLAN_REPORTED"),
              quantity(2791680001,837504000,basis_asof="2018-09-30")],
              (4,"每10股配售不超过3股"),(4,"2018年6月30日的总股本2,791,679,915股为基数测算,本次可配售股份数量不超过837,503,974股"),
              (4,"按照每10股配售3股的比例向全体股东配售"),(4,"2018年9月30日的总股本2,791,680,001股为基数测算,本次可配售股份数量为837,504,000股")),
        claim("regulatory_question_budget",{"amount":money("39"),"role":"QUESTION_AMOUNT_NOT_STANDALONE_EXECUTION_EVIDENCE"},
              (23,"申请人本次配股募集资金39亿元用于宁夏乐叶年产5GW高效单晶电池项目、滁州乐叶年产5GW高效单晶组件项目及补充流动资金")),
        claim("project_proposed_funding_partial",[allocation("宁夏5GW电池","254000","万元"),allocation("滁州5GW组件","106000","万元")],
              (24,"投资总额304,955.000.00/254,000.00"),(26,"合计226,186.00330.31/106,000.00"))]},
      {"index":6,"group":"GOLDWIND_2019_RESPONSE","claims":[
        claim("plan_ratios",{"A_per_10":"1.9000","H_per_10":"1.9000","scope":"A_AND_H_SEPARATELY"},
              (5,"公司按照每10股配售1.9000股的比例向全体A股股东配售;按照每10股配售1.9000股的比例向全体H股股东配售")),
        claim("financial_investment_budget_deduction",money("25581.70","万元"),
              (21,"上述合计金额25,581.70万元"),(21,"公司将该部分金额相应调减本次募集资金总额")),
        claim("proposed_funding_partial",[allocation("补充流动资金","15"),allocation("偿还有息负债","15")],
              (44,"本次募集资金补充流动资金15亿元"),(48,"本次配股募集资金15亿元用于偿还上市公司有息负债"))]},
      {"index":7,"group":"TIANQI_2019_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("700000.00","万元"),(20,"本次配股拟募集资金总额不超过人民币700,000.00万元(含发行费用)")),
        claim("proposed_use","偿还购买SQM23.77%股权的部分并购贷款",(20,"扣除发行费用后的净额拟全部用于偿还购买SQM23.77%股权的部分并购贷款")),
        claim("plan_quantity_scenarios",[quantity(1142052851,342615855,relation="CAP",version="PREVIOUS_PLAN_REPORTED"),quantity(1141987945,342596383)],
              (60,"按照每10股配售不超过3股"),(60,"总股本1,142,052,851股为基数测算,本次配售股份数量不超过342,615,855股"),
              (61,"按照每10股配售3股"),(61,"总股本1,141,987,945股为基数测算,本次可配售股份数量为342,596,383股")),
        claim("historical_incentive_cancellation_not_new_lock",{"shares":64906,"reported_completion":"2019-06-03","new_rights_restricted_shares_inferred":False},
              (61,"实施回购注销激励对象已获授但尚未解锁的限制性股票64,906股后,2019年6月3日,限制性股票回购注销完成")),
        claim("validity_amendment_shareholder_approval",{"status":"PENDING","months":12,"automatic_extension_removed":True},
              (61,"本次配股的决议自公司2018年度股东大会审议通过之日起12个月内有效"),(61,"本议案尚需提交股东大会审议"))]},
      {"index":8,"group":"TIANQI_2019_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("700000.00","万元"),(93,"本次配股拟募集资金总额不超过人民币700,000.00万元")),
        claim("proposed_use","偿还购买SQM23.77%股权的部分并购贷款",(93,"扣除发行费用后的净额拟全部用于偿还购买SQM23.77%股权的部分并购贷款")),
        claim("plan_quantity_scenarios",[quantity(None,342596383)],
              (94,"按照每10股配售3股的比例向全体股东配售"),(94,"本次可配售股份数量为342,596,383股")),
        claim("price_not_determined",True,(94,"在配股正式实施前,根据市场情况与主承销商协商确定合适的配股价格")),
        claim("conditional_full_subscription_commitment",{"holders":["天齐集团","张静","李斯龙"],"reported_commitment_date":"2019-04-11",
              "conditions":["股东大会审议通过","中国证监会核准"],"actual_payment":False},
              (93,"2019年4月11日,公司控股股东天齐集团及其一致行动人张静、李斯龙出具"),
              (93,"将在本次配股方案获得天齐锂业股东大会审议通过,并报经中国证券监督管理委员会核准后履行上述承诺")),
        claim("reported_internal_decision_sequence",{"latest_shareholder_meeting":"2019-07-19","regulatory_issue_approval_inferred":False},
              (92,"2019年7月19日召开的2019年第二次临时股东大会审"),(93,"议通过,履行了必要的内部决策程序"))]},
      {"index":9,"group":"GUOHAI_RESPONSE_2019","claims":[
        claim("plan_gross_proceeds",money("50"),(40,"本次配股募集资金总额不超过人民币50亿元")),
        claim("controller_conditional_subscription",{"holder":"广西投资集团有限公司","reported_commitment_date":"2018-11-22","actual_payment":False},
              (32,"2018年11月22日,发行人控股股东、实际控制人广西投资集团有限公司出具认购承诺函"),
              (32,"以现金方式全额认购根据本次配股方案确定的我司可配售的股份")),
        claim("other_holder_subscription_approval_pending",{"holder":"株洲市国有资产投资控股集团有限公司","status":"STATE_ASSET_PROCEDURE_PENDING"},
              (33,"株洲市国有资产投资控股集团有限公司正在履行出具承诺函的相关国资审批手续")),
        claim("past_divestitures_not_prospective_lock",["广西梧州索芙特美容保健品有限公司","广州市靓本清超市有限公司"],
              (33,"广西梧州索芙特美容保健品有限公司、广州市靓本清超市有限公司已全部减持所持有的公司股份")),
        claim("underwriting_failure_threshold",{"subscription_fraction":"0.70","condition":"原股东认购未达到可配售数量的70%则失败","actual_result":False},
              (33,"如果代销期限届满,原股东认购股票的数量未达到可配售数量的70%,则本次配股发行失败"))]},
    ]
    for i,p,q in [(10,15,14),(11,65,64)]:
        output.append({"index":i,"group":"CAPITAL_2019_RESPONSE","claims":[
            claim("plan_gross_proceeds",money("530000.00","万元"),(p,"单位:万元"),(p,"本次募集资金规模530,000.00")),
            claim("proposed_use","补充流动资金和偿还银行借款",(q,"本次配股募集资金拟用于补充流动资金和偿还银行借款")),
            claim("quasi_financial_investment_restriction_not_share_lock",{"trigger":"募集资金使用完毕前或募集资金到位36个月内",
                "restricted_activity":"新增对类金融业务的资金投入","share_holding_constraint":False},
                (q,"公司承诺在本次配股募集资金使用完毕前或募集资金到位36个月内,不再新增对类金融业务的资金投入"))]})
    output[-1]["claims"].append(claim("past_increment_plans_not_current_rights_lock",{"years":[2015,2016],"current_rights_constraint":False},
        (119,"截至2015年12月31日,该次增持计划已实施完毕"),(119,"截至2016年12月21日,该次增持计划已实施完毕")))
    for i in (12,15):
        output.append({"index":i,"group":"CAPITAL_PREPARATION_ACCOUNTING_RESPONSE","claims":[
            claim("response_scope","发审委会议准备函财务会计专项说明",(1,"有关财务会计问题的专项说明")),
            claim("peer_financing_excluded",{"issuer":"绿色动力","symbol":"601330.SH","IPO_amount":money("3.82"),
                "private_plan_amount":money("23.90"),"capital_rights_amount":False},
                (10,"绿色动力601330.SH"),(10,"2018年IPO募集资金3.82亿元;2019年公告非公开发行股票预案,拟募集资金23.90亿元"))]})
    for i,p in ((13,39),(14,46)):
        output.append({"index":i,"group":"CAPITAL_PREPARATION_RESPONSE","claims":[
            claim("financial_investment_commitment_not_share_lock",{"scope":"发行前不投入与主营业务无关的财务性投资","share_holding_constraint":False},
                (p,"公司承诺发行前不投入与主营业务无关的财务性投资")),
            claim("peer_financing_excluded",{"issuer":"绿色动力","symbol":"601330.SH","capital_rights_amount":False},
                (11,"绿色动力601330.SH"),(11,"2018年IPO募集资金3.82亿元;2019年公告非公开发行股票预案,拟募集资金23.90亿元"))]})
    output.append({"index":16,"group":"HUAAN_2021_RESPONSE","claims":[
        claim("regulatory_question_budget",{"amount":money("40"),"role":"QUESTION_AMOUNT_NOT_STANDALONE_EXECUTION_EVIDENCE"},
              (27,"申请人本次配股拟募集资金不超过40亿元")),
        claim("proposed_allocations",[allocation("资本中介业务","20"),allocation("投资与交易业务","10"),allocation("信息技术和风控体系","5"),allocation("其他营运资金","5")],
              (30,"不超过20亿元募集资金用于资本中介业务"),(31,"不超过10亿元募集资金用于投资与交易业务"),
              (32,"不超过5亿元用于发展信息技术和风控体系建设"),(33,"不超过5亿元募集资金用于其他营运资金安排")),
        claim("sector_regulatory_opinion_not_issue_approval",{"key":"机构部函[2020]3184号","CSRC_issue_approval_inferred":False},
              (41,"机构部函[2020]3184号"),(41,"证监会证券基金机构监管部对公司申请配股事项无异议"))]})
    output.append({"index":17,"group":"WINDPOWER_2022_RESPONSE","claims":[
        claim("plan_gross_proceeds",money("40"),(4,"公司本次配股募集资金总额不超过人民币40亿元")),
        claim("proposed_use","补充流动资金及偿还有息借款",(4,"扣除发行费用后拟全部用于补充流动资金及偿还有息借款")),
        claim("holding_commitments",[
            {"holders":"持股5%以上的股东、控股股东中国节能及其一致行动人中节能资本",
             "related_scope":"与承诺方具有控制关系的关联方和一致行动人","securities":"所持公司股票及其他具有股权性质的证券",
             "prohibition":"不减持且不安排减持计划","start_trigger":"本次发行定价基准日（发行期首日）前六个月",
             "end_trigger":"本次发行完成后六个月内","calendar_start":None,"calendar_end":None,"restricted_new_shares":None},
            {"holders":"公司全体董事、监事及高级管理人员","related_scope":"本人的一致行动人",
             "securities":"所持公司股票及其他具有股权性质的证券","prohibition":"不减持且不安排减持计划",
             "start_trigger":"本次发行定价基准日（发行期首日）前六个月","end_trigger":"本次发行完成后六个月内",
             "calendar_start":None,"calendar_end":None,"restricted_new_shares":None}],
             (29,"公司持股5%以上的股东、控股股东中国节能及其一致行动人中节能资本"),
             (29,"自节能风电本次发行定价基准日(发行期首日)前六个月至本次发行完成后六个月内,本公司承诺本公司及与本公司具有控制关系的关联方和一致行动人将不减持所持节能风电股票及其他具有股权性质的证券,亦不安排任何减持计划"),
             (29,"公司全体董事、监事及高级管理人员出具的关于本次配股发行认购及减持承诺"),
             (29,"自公司本次发行定价基准日(发行期首日)前六个月至本次发行完成后六个月内,本人承诺本人及本人的一致行动人将不减持所持公司股票及其他具有股权性质的证券,亦不安排任何减持计划")),
        claim("question_summary_and_quoted_commitment_scope_difference",{"question_and_summary":"前后各六个月不再进行相关股票买卖",
              "quoted_commitment":"不减持、不安排减持计划","automatic_equivalence_or_broader_ban_inferred":False},
              (29,"承诺本次配股的前后各六个月不再进行相关股票买卖"),(30,"承诺本次配股的前后各六个月不再进行相关股票买卖"),
              (29,"将不减持所持公司股票及其他具有股权性质的证券,亦不安排任何减持计划")),
        claim("reported_earlier_disclosure_not_dated",{"earlier_disclosure_claimed":True,"verified_first_publication":None},
              (29,"公司已公开披露上述承诺"))]})
    return sorted(output,key=lambda r:r["index"])


EXCLUSIONS={
    0:"坏账计提议案待年度股东大会不等于配股方案待批；历史没有减持不是未来锁定承诺。",
    1:"2014、2016非公开发行及已用资金属于历史融资；六个月为坏账账龄；项目总投资不等于拟用募集资金。",
    2:"旧24亿监管提问与新23亿回复按版本区分；24.7亿属已终止的非公开发行；持有期间系Santos投资会计。",
    3:"568539.30万是项目总投资；氧化铝产量和建设日程不是发行股数或日历；铝价、外汇锁定及金融资产持有期不是配股锁定。",
    4:"2016非公开发行、客户质押物积成电子和天泽信息减持，不是本次东吴发行；香港8.5亿和1.5亿是子项，不重复加总。",
    5:"IPO锁定、激励股票及2015增持承诺不同于当前配股约束；2014非公开发行和2017可转债使用情况不计本次融资。",
    6:"财务性投资观察期不是股东锁定；此回复明确扣减25581.70万元，本身未匹配到调减后总额，不能以缺失覆盖前文。",
    7:"旧比例上限和旧数量为修订前版本；64906股是激励注销，六个月是财务性投资观察期，锁定销量不是股东约束。",
    8:"全额认购为有条件承诺；70亿元偿债模拟不是到账；六个月重新规范承诺为监管引文，不生成股票解禁日。",
    9:"36个月行政处罚观察期不是锁定；两股东已经减持为历史状态，认购承诺待批不等于资金已缴。",
    10:"36个月约束对象是类金融业务资金投入，不是持股；53亿元为拟募集规模，财务投资60406.82万元不当作配股额。",
    11:"旧增持2015、2016承诺不是当次配股锁定；锁定债权还款项目不是股份锁定；36个月财务投入禁限不当股权约束。",
    12:"命中金额来自绿色动力、中国天楹等同业融资比较，不能赋给首创；本轮候选未提供当次发行额不等于零。",
    13:"同业融资金额排除；六个月系财务投资观察窗；当次发行金额未在已读候选明确，不覆盖既有53亿元来源。",
    14:"2019年末财务性投资修订不能变成新融资；同业金额排除；借款发生窗口不是股份持有窗口。",
    15:"同业融资比较不归属于首创；修订会计说明不代表新增一轮配股；当次发行字段没有观察到不设为零。",
    16:"40亿元提问额与回复各用途分别保存；机构监管意见不视为发行核准；未来营业收入和IT预算预测不计实际现金。",
    17:"30%非公开发行规则在回复中明确不适用；实际承诺不减持与提问/总结不买卖的文字差别保留；未知限制股数不作零。",
}


def build():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    targets=read(OUT/"targets.json");facts=[]
    assert [s["index"] for s in specs()]==list(range(18))
    for spec in specs():
        row=targets[spec["index"]];doc=Document({**row["source"],"review_role":"REGULATORY_RESPONSE_OPERATIVE_SOURCE_REVIEW"})
        f=base_fact(doc,None)
        f.update(plan_source_group=spec["group"],reviewed_claims={},claim_evidence={},full_body_reviewed=spec["index"]==0,
            candidate_clause_review_complete=True,actual_issuance_calendar=None,actual_issue_price=None,actual_issuance_quantity=None,
            effective_sellable_new_shares=None,strict_M06_pressure=None,
            notes=["目录日末仅为可得日期代理，正文引用更早的事实或承诺不倒推首次公开。",EXCLUSIONS[spec["index"]],
                   "本次保存已读候选及补充原页；不声明1373页全文或经营会计意见已验证。"])
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            assert c["field"] not in f["reviewed_claims"]
            f["reviewed_claims"][c["field"]]=c["value"]
            f["claim_evidence"][c["field"]]=[anchor(row["document_id"],pages,p,t) for p,t in c["proof"]]
        facts.append(f)
    return facts


def source_view(fact,at):
    return {"document_id":fact["document_id"],"as_of":at,"known_at":fact["known_at"],
        "status":"REVIEWED_RESPONSE_DATE_PROXY_ONLY" if fact["known_at"]<=at else "NO_VIEW",
        "reviewed_claims":deepcopy(fact["reviewed_claims"]) if fact["known_at"]<=at else {},
        "actual_event_id":None,"effective_sellable_new_shares":None,"strict_M06_pressure":None,"trading_feature_admitted":False}


def relationships(facts,prior):
    old={f["document_id"]:f for f in prior};pairs=[]
    for i,rid in [(0,"1203410179"),(1,"1203470048"),(2,"1203759860"),(4,"1205275256"),(5,"1205666142"),
                  (6,"1205804810"),(8,"1206556835"),(9,"1206732520"),(10,"1207003866"),(11,"1207003866"),
                  (12,"1207195236"),(13,"1207195236"),(14,"1207447771"),(15,"1207447771"),(16,"1209162003")]:
        f,a=facts[i],old[rid]
        assert f["known_at"]==a["known_at"] and f["symbol"]==a["symbol"]
        pairs.append({"response_id":f["document_id"],"admin_id":rid,"relationship_known_at":f["known_at"],
            "meaning":"同日提示与回复正文并列，不计为两次融资。","old_source_unchanged":True})
    wind=facts[17];notice=old["1215110334"];listing=old["1215295719"]
    holding={"response_id":wind["document_id"],"response_known_at":wind["known_at"],"announcement_id":notice["document_id"],
        "announcement_known_at":notice["known_at"],"listing_id":listing["document_id"],"listing_known_at":listing["known_at"],
        "earlier_than_saved_issuance_notice_days":(datetime.fromisoformat(notice["known_at"])-datetime.fromisoformat(wind["known_at"])).days,
        "earlier_than_saved_listing_days":(datetime.fromisoformat(listing["known_at"])-datetime.fromisoformat(wind["known_at"])).days,
        "relationship_known_at":listing["known_at"],"retrospective_comparison_only":True,"first_ever_disclosure_verified":False,
        "new_response_holding_groups":2,"listing_classification":deepcopy(listing["updates"]),
        "effective_sellable_new_shares":None,"automatic_subtraction_from_listing_total":False,"actual_event_identity_backfilled":False,
        "meaning":"七月承诺与十二月上市分类及激励锁定分别保存；数量、重叠和失效资料不齐，不能推出即时可卖量。"}
    terminal=old["1205354373"]
    reject={"response_id":facts[4]["document_id"],"terminal_document_id":terminal["document_id"],
        "relationship_known_at":terminal["known_at"],"plan":"东吴2017配股方案","old_termination_preserved":True,
        "response_inferred_issuance_success":False,"separate_from_dongwu_2020":True}
    versions=[]
    for i,j in [(0,1),(7,8),(12,15),(13,14)]:
        versions.append({"earlier_response":facts[i]["document_id"],"later_response":facts[j]["document_id"],
            "relationship_known_at":facts[j]["known_at"],"new_issuance_inferred":False,"missing_field_means_revocation":False})
    gold=old["1205782946"]["reviewed_claims"]
    return pairs,holding,reject,versions


def finalize():
    facts=build();prior=read(PREVIOUS/"combined_source_facts.json")
    assert len(prior)==421 and not {f["document_id"] for f in prior}&{f["document_id"] for f in facts}
    pairs,holding,reject,versions=relationships(facts,prior)
    initial=read(OUT/"source_candidates.json");supp=read(OUT/"supplement_candidates.json");reviewed=[]
    for kind,rows in (("INITIAL",initial),("SUPPLEMENT",supp)):
        for i,row in enumerate(rows):
            for j,c in enumerate(row["candidates"]):
                reviewed.append({"document_id":row["document_id"],"candidate_set":kind,"candidate_index":j,**c,
                    "reviewed":True,"scope_note":EXCLUSIONS[i]})
    assert len(reviewed)==144
    queries=[]
    for f in facts:
        at=f["known_at"];before=(datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
        queries.append({"document_id":f["document_id"],"before":source_view(f,before),"at":source_view(f,at)})
    for name,obj in [("reviewed_response_source_facts",facts),("combined_source_facts",prior+facts),("reviewed_source_candidates",reviewed),
                     ("same_day_admin_comparisons",pairs),("earlier_windpower_holding_source",holding),("dongwu_2017_terminal_context",reject),
                     ("response_version_relationships",versions),("response_publication_queries",queries)]:save(OUT/(name+".json"),obj)
    save(OUT/"remaining_response_documents.json",{"count":0,"documents":[],"scope":"上一轮固定的18份监管回复，非全市场发行文件覆盖。"})
    with (OUT/"18份监管回复条款.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["公告ID","证券","可得日期代理","标题","计划来源组","结构化字段","来源URL"]);writer.writeheader()
        for f in facts:writer.writerow({"公告ID":f["document_id"],"证券":f["symbol"],"可得日期代理":f["known_at"],"标题":f["title"],
            "计划来源组":f["plan_source_group"],"结构化字段":json.dumps(f["reviewed_claims"],ensure_ascii=False),"来源URL":f["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_RESPONSES_V1","status":"18_RESPONSE_SOURCE_CLAUSES_REVIEWED_EARLIER_HOLDING_SOURCE_ADDED",
        "documents":18,"target_pdf_pages":1373,"full_pdf_pages_read_claimed":False,"initial_candidates":137,"supplement_candidates":7,
        "candidate_fragments_read":144,"candidate_characters":sum(len(c["literal"]) for c in reviewed),
        "source_fields_reviewed":sum(len(f["reviewed_claims"]) for f in facts),"same_day_update_notice_pairs":len(pairs),
        "version_relationship_pairs":len(versions),"earlier_holding_source_documents":1,"earlier_holding_groups":2,"visual_pages_reviewed":3,
        "prior_source_documents_unchanged":421,"combined_source_documents":439,"remaining_fixed_response_documents":0,
        "new_actual_issuance_calendars":0,"new_publication_queries":36,"full_issuance_coverage":False,
        "earlier_holding_clause_full_coverage":False,"effective_sellable_quantities_computed":0,"free_float_denominator_available":False,
        "strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"qualified_candidates":0,"independent_forward_observations":0,
        "orders_authorized":False,"delivery_package_created":False,"goal_achieved":False,
        "next_source_action":"T13_HOLDING_SOURCE_COVERAGE_AND_ISSUANCE_DENOMINATOR_GAPS"}
    save(OUT/"result.json",result)
    print(json.dumps({"回复":18,"条款字段":result["source_fields_reviewed"],"较早持有约束来源":1,"固定剩余文件":0,"当前来源":439},ensure_ascii=False))


if __name__=="__main__":finalize()
