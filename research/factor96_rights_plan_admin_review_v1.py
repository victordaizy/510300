"""按已读公告建立计划及行政事项来源账，不把提议和历史回顾当作新发行。"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

from research.factor96_rights_plan_admin_v1 import OUT, PREVIOUS, SOURCE, read, save, digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact

VISUAL_PAGES=[("1202686497",1),("1202839814",2),("1203642857",2),("1203759865",1),
              ("1205782946",2),("1205782946",3),("1206375058",2),("1214794484",2)]


def money(value, unit="亿元"):
    multiplier={"元":100,"万元":1000000,"亿元":10000000000}[unit]
    cents=Decimal(value)*multiplier
    assert cents==cents.to_integral_value()
    return {"reported_value":value,"reported_unit":unit,"amount_cents":int(cents)}


def claim(field, value, *proof):
    return {"field":field,"value":value,"proof":proof}


def specs():
    rows=[]

    def add(index, status, note, *claims):
        rows.append({"index":index,"status":status,"note":note,"claims":claims})

    add(0,"FEEDBACK_EXTENSION_REQUEST","申请延后回复，未确认监管已同意延期；公告可得日为9月26日。",
        claim("requested_reply_deadline","2015-12-05",(1,"申请将反馈意见回复期限延长60日至2015年12月5日前回复")),
        claim("extension_request_submitted","2015-09-25",(1,"公司于2015年9月25日向中国证监会递交了延期回复的申请")),
        claim("regulatory_issue_approval","PENDING",(1,"公司本次配股事宜能否获得核准仍存在不确定性")))
    add(1,"FEEDBACK_EXTENSION_REQUEST","预计回复期限不能视为获准发行或新缴款窗口。",
        claim("expected_reply_deadline","2015-10-23",(1,"预计将不晚于2015年10月23日向中国证监会递交反馈意见回复")),
        claim("extension_request_submitted","2015-09-24",(1,"公司于2015年9月24日向中国证监会提交了延期回复的申请")),
        claim("regulatory_issue_approval","PENDING",(1,"尚需获得中国证监会核准")))
    add(2,"VALIDITY_EXTENSION_PROPOSAL","董事会提出延长6个月，股东大会尚未审议；45亿元是拟募资上限。",
        claim("proposed_validity_end","2016-07-29",(1,"拟将本次配股相关决议和授权的有效期自前次有效期到期之日起延长6个月,至2016年7月29日")),
        claim("shareholder_approval","PENDING",(4,"待提交公司2015年第二次临时股东大会审议")),
        claim("plan_ratio_per_10","3",(2,"按每10股配售3股的比例向全体股东配售")),
        claim("plan_gross_proceeds_cap",money("45"),(2,"本次配股募集资金总额拟不超过人民币45亿元")))
    add(3,"STATE_ASSET_APPROVAL_WITH_OTHER_APPROVALS_PENDING","吉林省国资委原则同意用途调整；仍待股东大会及证监会，募资上限不变。",
        claim("state_asset_approval_key","吉国资发产权[2015]77号",(1,"吉国资发产权[2015]77号")),
        claim("plan_gross_proceeds_cap",money("180000","万元"),(1,"募集资金总额预计不超过人民币180,000万元")),
        claim("use_revision_wan_cny",{"vaccine":40000,"working_capital_before":140000,"R_and_D_after":80000,"working_capital_after":60000},
              (1,"补充流动资金140,000.00"),(1,"新产品研发投入80,000.00"),(1,"补充流动资金60,000.00"),(1,"期项目40,000.00")),
        claim("remaining_approvals",["SHAREHOLDERS","CSRC"],(2,"尚需经公司2015年第三次临时股东大会审议通过后,报请中国证券监督管理委员会核准")))
    add(4,"PLAN_RATIO_AND_VALIDITY_REVISED","西部由每10股不超过3股明确为2.6股；固定股本测算数量变化，自动延长条款被删除。",
        claim("plan_ratio_revision",{"before_cap":"3","after":"2.6","base":10},(1,"按照每10股配售不超过3股"),(1,"按照每10股配售2.6股")),
        claim("plan_quantity_revision",{"basis_shares":2795569620,"before_cap_shares":838670886,"after_scenario_shares":726848101},
              (1,"2,795,569,620股"),(1,"不超过838,670,886股"),(1,"配售股份数量为726,848,101股")),
        claim("automatic_validity_extension_removed",True,(2,"则本决议有效期自动延长至配股实施完成日"),(2,"(十)本次配股决议的有效期限与本次配股有关的决议自公司股东大会审议通过之日起12个月内有效。三、财务会计信息")))
    add(5,"PLAN_RATIO_AND_VALIDITY_REVISED","国海2016方案明确3股比例并更新基准日；全额认购承诺不是持有期承诺。",
        claim("plan_year",2016,(1,"2016年度配股公开发行证券预案")),
        claim("plan_ratio_revision",{"before_cap":"3","after":"3","base":10},(2,"按照每10股配售不超过3股"),(2,"按照每10股配售3股")),
        claim("plan_quantity_scenario",{"basis_date":"2016-09-30","basis_shares":4215541972,"shares":1264662591},(2,"截至2016年9月30日的总股本4,215,541,972股"),(2,"本次配售股份数量为1,264,662,591股")),
        claim("automatic_validity_extension_removed",True,(3,"则本决议有效期自动延长至配股实施实成日"),(3,"(十)本次配股决议的有效期限本次配股的决议自公司股东大会审议通过之日起12个月内有效。三、财务会计信息")))
    add(6,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","附件披露拟募资36.29亿元，当前仍待核准；没有背书其经营预测。",
        claim("regulatory_issue_approval","PENDING",(1,"公司本次配股事项尚需获得中国证监会的核准")),
        claim("plan_gross_proceeds_cap",money("36.29"),(2,"本次配股拟募集资金总额不超过36.29亿元")),
        claim("reported_use_assumption",{"working_capital":money("19"),"debt_repayment":money("17.29")},(6,"其中19亿元用于补充流动资金,17.29亿元偿还银行贷款和中期票据")))
    add(7,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","补充回复拟报送，不代表发审委通过。",
        claim("regulatory_issue_approval","PENDING",(1,"尚需中国证监会的核准")),
        claim("reply_submission","TO_BE_SUBMITTED",(1,"公司将按照要求将上述反馈意见的补充回复说明及时上报中国证监会")))
    add(8,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","进一步补充修订回复；仍待核准，所引用早期回复日期不提前可得时点。",
        claim("regulatory_issue_approval","PENDING",(1,"尚需中国证监会的核准")),
        claim("reply_submission","TO_BE_SUBMITTED",(1,"公司将按照要求将上述反馈意见回复修订材料及时报送中国证监会")))
    add(9,"PLAN_QUANTITY_REBASED_WITH_SOURCE_CONFLICT","直接保存配股上限及股本基准；原页每1股送2股文字与股本变化不一致，不自行修正分红比例。",
        claim("plan_year",2017,(1,"公司2017年度配股发行方案")),
        claim("plan_quantity_revision",{"before_cap_shares":688289863,"after_cap_shares":825947836,"basis_shares_after":2753159454},
              (2,"由不超过688,289,863股调整为不超过825,947,836股"),(2,"以总股本2,753,159,454股为基数测算")),
        claim("plan_ratio_cap_per_10","3",(2,"按照每10股配售不超过3股")),
        claim("profit_distribution_ratio_unresolved",True,(2,"每1股派发1.00元现金(含税),送红股2股(含税)")))
    add(10,"FEEDBACK_REPLY_WITH_BUDGET_REVISION","新奥修订回复同时报告募资调减至23亿元；与同日调整公告并列，不计两次供给。",
        claim("plan_year",2017,(1,"调整公司2017年度配股公开发行证券方案")),
        claim("plan_gross_proceeds_cap",money("23"),(1,"公司配股募集资金总额预计不超过23亿元")),
        claim("working_capital_project_removed",True,(1,"补充流动资金项目不再作为本次配股公开发行的募投项目")),
        claim("regulatory_issue_approval","PENDING",(1,"尚需中国证监会的核准")))
    add(11,"PLAN_BUDGET_REDUCED","新奥募资上限24亿元改23亿元，去掉补流用途；计划量不等于实际融资额。",
        claim("plan_year",2017,(1,"公司2017年度配股公开发行证券方案")),
        claim("gross_proceeds_cap_revision",{"before":money("240000","万元"),"after":money("230000","万元")},
              (1,"拟募集资金总额不超过人民币240,000万元"),(1,"拟募集资金总额不超过人民币230,000万元")),
        claim("revised_use","年产20万吨稳定轻烃项目",(1,"扣除发行费用后的净额将全部用于投资年产20万吨稳定轻烃项目。")))
    add(12,"PLAN_AMENDMENT_DETAIL_NOTICE","与同日募资调减和回复构成同次变更的不同来源，不另建发行。",
        claim("plan_year",2017,(1,"2017年度配股公开发行证券预案")),
        claim("regulatory_issue_approval","PENDING",(1,"本次配股尚需获得中国证券监督管理委员会的核准")),
        claim("use_change_referenced",True,(2,"根据调整后的募集资金规模和用途修订该部分的内容")))
    add(13,"PLAN_AMENDMENT_DETAIL_NOTICE","万向修改募资规模用途，但本公告未给数值；不从同日长预案自动填入。",
        claim("plan_year",2017,(1,"公司2017年度配股公开发行证券预案")),
        claim("budget_and_use_changed_numeric_not_in_notice",True,(1,"对本次募集资金规模及用途进行了调整")))
    add(14,"FEEDBACK_EXTENSION_REQUEST","东吴2017方案回复延期申请；与后来的2020配股保持不同身份。",
        claim("expected_reply_deadline","2018-04-30",(1,"预计将在2018年4月30日前向中国证监会递交反馈意见回复")),
        claim("extension_request_submitted","2018-02-08",(1,"公司于2018年2月8日向中国证监会提交了延期回复的申请")),
        claim("regulatory_issue_approval","PENDING",(1,"尚需获得中国证监会核准")))
    add(15,"PLAN_RATIO_AND_USE_REVISED","东吴2017方案明确3股比例、9亿股测算，65亿元上限不变；后来终止记录另保留。",
        claim("plan_year",2017,(1,"2017年度配股公开发行证券预案")),
        claim("plan_ratio_revision",{"before_cap":"3","after":"3","base":10},(1,"按每10股配售不超过3股"),(2,"按每10股配售3股")),
        claim("plan_quantity_scenario",{"basis_shares":3000000000,"shares":900000000},(2,"总股本3,000,000,000股为基数测算,则本次可配售股份数量总计900,000,000股")),
        claim("plan_gross_proceeds_cap",money("65"),(3,"本次配股募集资金总额不超过人民币65亿元")))
    add(16,"PLAN_USE_DETAIL_REVISED","东吴2017方案第二次修订只进一步分解用途，65亿元不变，不能重启后来已终止的方案。",
        claim("plan_year",2017,(1,"2017年度配股公开发行证券预案(第二次修订稿)")),
        claim("plan_gross_proceeds_cap",money("65"),(2,"本次配股募集资金总额预计为不超过人民币65亿元")),
        claim("overseas_use_breakdown",{"fund_cap":money("8.5"),"IT_cap":money("1.5")},(3,"投入国新国信东吴海外基金1不超过人民币8.5亿元"),(3,"信息系统的研发与建设投入的募集资金规模不超过人民币1.5亿元")))
    add(17,"VALIDITY_EXTENSION_PROPOSAL","国海提请续期至待开股东大会通过日起12个月；历史发审委通过不等于已取得发行批复。",
        claim("plan_year",2016,(1,"2016年第一次临时股东大会")),
        claim("proposed_validity_months_from_future_vote",12,(2,"拟将本次配股股东大会决议的有效期延长至审议上述议案的股东大会决议通过之日起12个月内有效")),
        claim("shareholder_approval","PENDING",(2,"尚需提交公司2018年第三次临时股东大会审议")),
        claim("reported_committee_pass_date","2016-12-14",(1,"已于2016年12月14日获中国证监会主板发行审核委员会审核通过")))
    add(18,"PLAN_BASIS_AND_VALIDITY_REVISED","国海股份基准日从2016-09-30更新至2018-09-30，股本和计划数量未变；本修订尚待股东大会。",
        claim("plan_year",2016,(1,"2016年度配股公开发行证券预案")),
        claim("plan_quantity_scenario",{"basis_date":"2018-09-30","basis_shares":4215541972,"shares":1264662591},
              (3,"截至2018年9月30日的总股本4,215,541,972股"),(3,"本次配售股份数量为1,264,662,591股")),
        claim("shareholder_approval","PENDING",(1,"本议案尚需提交股东大会审议")),
        claim("proposed_validity_rule","延长前次决议的股东大会通过日起12个月",(3,"自公司延长前次股东大会决议有效期的股东大会决议通过之日起12个月内有效")))
    add(19,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","隆基2018方案回复修订，仍待核准。",
        claim("regulatory_issue_approval","PENDING",(1,"尚需取得中国证监会的核准")),
        claim("referenced_previous_reply_date","2018-12-06",(1,"2018年12月6日,公司根据要求对反馈意见的回复进行了公开披露")))
    add(20,"SUPPORTING_OPINION_NOT_NEW_DECISION","独董赞同金风方案调整，不是独立新方案或监管批准。",
        claim("opinion","SUPPORT",(1,"我们同意公司本次调整配股募集资金总额和确定配股比例事项")))
    add(21,"PLAN_BUDGET_REDUCED_AND_A_H_RATIO_SET","金风募资上限50亿改47.44183亿；A/H股份分开，按2018-12-31股本测算，受后续股本变化条件约束。",
        claim("gross_proceeds_cap_revision",{"before":money("500000.00","万元"),"after":money("474418.30","万元")},
              (1,"拟配股募集资金总额不超过人民币500,000.00万元"),(2,"拟配股募集资金总额不超过人民币474,418.30万元")),
        claim("plan_A_H_quantity_scenario",{"basis_date":"2018-12-31","basis_shares":3556203300,"total_shares":675678627,"A_shares":552167067,"H_shares":123511560},
              (3,"若以2018年12月31日公司总股本3,556,203,300股为基础测算"),(3,"总计675,678,627股,其中A股可配股数量总计552,167,067股,H股可配股数量总计123,511,560股")),
        claim("plan_ratio_per_10_A_and_H","1.9000",(3,"按照每10股配售1.9000股"),(3,"A股和H股配股比例相同")),
        claim("additional_shareholder_vote","NOT_REQUIRED_BY_NOTICE",(3,"上述事项无需另行提交公司股东大会审议")))
    add(22,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","金风修订回复，尚需核准；不从已报送推定批准。",
        claim("regulatory_issue_approval","PENDING",(2,"公司本次配股发行尚需中国证监会核准")))
    add(23,"AUTOMATIC_VALIDITY_EXTENSION_REMOVAL_PROPOSAL","天齐拟删除自动延长至实施完成的条款，保留12个月，尚待股东大会。",
        claim("plan_year",2019,(1,"公司2019年度配股公开发行证券")),
        claim("automatic_extension_removal_proposed",True,(1,"则决议有效期自动延长至本次配股实施完成日"),(1,"调整为“本次配股的决议自公司股东大会审议通过之日起12个月内有效。”")),
        claim("shareholder_approval","PENDING",(1,"上述调整事项尚需提请公司股东大会审议通过")),
        claim("regulatory_issue_approval","PENDING",(1,"尚未取得中国证券监督管理委员会核准")))
    add(24,"PLAN_QUANTITY_REVISED_WITH_SPLIT_APPROVAL_SCOPES","天齐数量调整已有授权，有效期修改另待表决，分别记录。",
        claim("plan_year",2019,(1,"公司2019年度配股公开发行证券")),
        claim("plan_quantity_revision",{"before_basis_shares":1142052851,"before_cap_shares":342615855,"after_basis_shares":1141987945,"after_scenario_shares":342596383},
              (1,"总股本1,142,052,851股为基数测算,本次配售股份数量不超过342,615,855股"),(1,"总股本1,141,987,945股为基数测算,本次可配售股份数量为342,596,383股")),
        claim("plan_ratio_revision",{"before_cap":"3","after":"3","base":10},(1,"按照每10股配售不超过3股"),(1,"按照每10股配售3股")),
        claim("quantity_change_shareholder_vote","NOT_REQUIRED_BY_PRIOR_AUTHORITY",(2,"已经得到公司股东大会的授权,无需提交股东大会审议")),
        claim("validity_change_shareholder_vote","PENDING",(2,"董事会对本次配股决议的有效期限的修订尚需提交股东大会审议")))
    add(25,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","天齐将初步半年报数据换为正式半年报数据；不当作发行金额修订。",
        claim("reply_financial_vintage_updated",True,(1,"对公司2019年半年度的主要财务数据和指标采用了初步核算的数据"),(1,"现将《<告知函>的回复》中有关半年报的财务数据进行更新")),
        claim("regulatory_issue_approval","PENDING",(1,"尚需取得中国证监会核准")))
    add(26,"FEEDBACK_FINANCIAL_UPDATE_NOTICE","国海根据半年报更新反馈回复并拟报送，具体长回复尚待审。",
        claim("financial_report_period","2019H1",(1,"结合2019年半年度报告对反馈意见回复内容进行了更新")),
        claim("reply_submission","TO_BE_SUBMITTED",(1,"公司将及时向中国证监会报送更新后的反馈意见回复")))
    add(27,"PLAN_AMENDMENT_DETAIL_NOTICE","天风提示数量已明确，但本公告未列具体数值；不从后来的实际发行反填。",
        claim("plan_year",2019,(1,"2019年度配股公开发行证券预案")),
        claim("quantity_detail_referenced_not_numeric",True,(1,"明确了本次配股的具体发行数量")))
    add(28,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","首创反馈回复进一步补充，仍待核准。",
        claim("regulatory_issue_approval","PENDING",(1,"尚需中国证监会核准")),
        claim("reply_submission","TO_BE_SUBMITTED",(1,"及时报送中国证监会")))
    add(29,"PLAN_AMENDMENT_DETAIL_NOTICE","招商明确比例数量的数值在另份预案，本公告仅给修订范围。",
        claim("plan_year",2019,(1,"公司2019年度配股公开发行证券预案")),
        claim("quantity_detail_referenced_not_numeric",True,(1,"主要对本次配股具体配售比例和数量进行了明确")))
    add(30,"VALIDITY_EXTENSION_PROPOSAL_WITH_RETROSPECTIVE_CLAIM","国海2019再提续期；引用2018股东大会确认过去持续有效的说法，只从本公告时点保存，不倒填2017年状态。",
        claim("plan_year",2016,(1,"2016年第一次临时股东大会")),
        claim("proposed_validity_months_from_future_vote",12,(2,"拟将本次配股股东大会决议的有效期延长至审议上述议案的股东大会决议通过之日起12个月内有效")),
        claim("shareholder_approval","PENDING",(3,"上述事项尚需提交公司2019年第二次临时股东大会审议")),
        claim("prior_validity_continuity_claim","2018股东大会追认此前决议一直有效，原会议公告未在本轮补齐",(1,"2018年12月10日,公司召开2018年第三次临时股东大会"),(2,"确定公司2016年第一次临时股东大会审议通过的配股相关的所有决议在该次股东大会审议通过后至今一直有效")),
        claim("reported_committee_pass_date","2019-11-08",(2,"已于2019年11月8日获中国证监会主板发行审核委员会审核通过")))
    add(31,"PLAN_AMENDMENT_DETAIL_NOTICE","山西提示比例数量、子公司增资用途明确，实际数值在另份预案中。",
        claim("plan_year",2019,(1,"公司2019年度配股公开发行预案")),
        claim("quantity_detail_referenced_not_numeric",True,(1,"主要对本次配股具体配售比例和数量及募投项目情况进行了明确")))
    add(32,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","首创告知函回复修订拟报送，核准及时间均未确定。",
        claim("regulatory_issue_approval","PENDING",(1,"公司本次配股公开发行证券事项尚需中国证监会核准")),
        claim("reply_submission","TO_BE_SUBMITTED",(1,"及时报送中国证监会")))
    add(33,"PLAN_AMENDMENT_DETAIL_NOTICE","国元明确认购承诺、比例数量，但本公告不列具体数值，不能代替完整预案。",
        claim("plan_year",2019,(1,"2019年度配股公开发行预案")),
        claim("quantity_detail_referenced_not_numeric",True,(1,"明确了本次配股的具体配售比例和数量")),
        claim("controller_cash_subscription_commitment_referenced",True,(1,"明确了控股股东承诺以现金方式全额认购本次配股方案中的可配售股份")))
    add(34,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","首创正文称4月1日披露，目录为4月2日；本轮不将可得日提前。",
        claim("regulatory_issue_approval","PENDING",(1,"公司本次配股公开发行证券事项尚需中国证监会核准")),
        claim("body_referenced_release_date","2020-04-01",(1,"具体内容详见2020年4月1日公司刊登在上海证券交易所网站")))
    add(35,"VALIDITY_EXTENSION_PROPOSAL","山西董事会拟续12个月；原文所列起止日照录，不自行消除边界日期重叠。",
        claim("reported_old_validity",{"start":"2019-05-17","end":"2020-05-16"},(1,"即2019年5月17日至2020年5月16日")),
        claim("proposed_new_validity",{"start":"2020-05-16","end":"2021-05-15","months":12},(1,"有效期延长12个月(即2020年5月16日至2021年5月15日)")),
        claim("shareholder_approval","PENDING",(2,"上述事项尚需提交公司2020年第三次临时股东大会审议")))
    add(36,"SUPPORTING_OPINION_NOT_NEW_DECISION","山西独董同意提请股东大会续期，不是股东大会已经通过。",
        claim("opinion","SUPPORT_PROPOSAL",(1,"独立董事同意公司关于提请股东大会延长公司配股公开发行决议有效期及授权有效期的相关事项")))
    add(37,"VALIDITY_EXTENSION_PROPOSAL_WITH_APPROVAL_REPORTED","河钢已获配股批复的历史事实与拟续期分开，未发行完成；后来终止仍保留。",
        claim("plan_year",2019,(1,"公司2019年度配股公开发行方案")),
        claim("reported_regulatory_approval_key","2020_33",(1,"证监许可【2020】33号")),
        claim("reported_approval_received_date","2020-01-17",(1,"于2020年1月17日收到中国证券监督管理委员会")),
        claim("proposed_extension_months_after_previous_expiry",12,(1,"延长至自前次决议有效期届满之日起十二个月")),
        claim("shareholder_approval","PENDING",(1,"上述议案尚需提交公司2020年第一次临时股东大会审议")))
    add(38,"UPDATED_SELF_ELIGIBILITY_STATEMENT","中科三环为发行资格自查，页4非公开发行30%规则引用不是本次配股数量调整。",
        claim("issuer_self_assessed_eligible",True,(1,"公司董事会对实际经营情况和相关事项进行逐项对照检查,认为公司具备申请配股的资格和条件")),
        claim("quoted_rights_quantity_cap_percent",30,(3,"拟配售股份数量不超过本次配售股份前股本总额的30%")),
        claim("quoted_nonpublic_issuance_quantity_cap_percent",30,(4,"上市公司申请非公开发行股票的,拟发行的股份数量原则上不得超过本次发行前总股本的30%")))
    add(39,"HISTORICAL_PROCEEDS_REALLOCATION_NOT_NEW_ISSUE","巨化处理2013年已募集资金结余，公告中的历史发行量不新增供给事件。",
        claim("historical_issue_year",2013,(1,"公司2013年配股发行股票节余募集资金")),
        claim("proposed_surplus_working_capital",money("14385.39","万元"),(1,"节余募集资金14,385.39万元")),
        claim("additional_shareholder_vote","NOT_REQUIRED_BY_NOTICE",(1,"故本事项无需提交股东大会审议")),
        claim("scope_is_historical_proceeds_only",True,(1,"公司本次发行募集资金投资项目已全部实施完毕")))
    add(40,"PLAN_AMENDMENT_DETAIL_NOTICE","江苏银行提示比例数量明确及半年报更新，但不在该公告中填入预案数值。",
        claim("quantity_detail_referenced_not_numeric",True,(1,"对本次配股的具体配股比例及数量进行了明确")),
        claim("financial_report_period","2020H1",(1,"根据公司2020年半年度报告更新公司相关财务信息")))
    add(41,"HISTORICAL_PROCEEDS_REALLOCATION_NOT_NEW_ISSUE","隆基2020公告拟调整2018方案的募集资金结余用途，仍待股东大会；不记为2020新配股。",
        claim("historical_issue_approval_key","2019_202",(1,"证监许可[2019]202号")),
        claim("proposed_reallocation",{"new_project":money("120000","万元"),"ningxia_working_capital":money("1698.82","万元"),"chuzhou_working_capital":money("17685.90","万元")},
              (4,"结余募集资金120,000万元"),(4,"剩余的1,698.82万元"),(4,"结余募集资金17,685.90万元")),
        claim("shareholder_approval","PENDING",(1,"尚需提交公司股东大会审议")))
    add(42,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","华安回复进一步修订，核准未确认。",
        claim("regulatory_issue_approval","PENDING",(1,"公司本次配股发行证券尚需获得中国证监会核准")))
    add(43,"PROJECT_STRUCTURE_REVISED_BUDGET_UNCHANGED","中科三环因用地规划调整项目内部结构，总投资及拟使用募集资金不变。",
        claim("plan_year",2020,(1,"公司2020年度配股公开发行证券")),
        claim("project_budget_unchanged",True,(1,"项目投资总额及募集资金拟投入金额不变")),
        claim("project_structure_changed",True,(1,"根据新的用地规划对该项目投资内容进行内部结构调整")))
    add(44,"PLAN_AMENDMENT_DETAIL_NOTICE","中信证券经营管理层按授权明确比例数量；具体数值不在本公告，认购承诺也不等于锁定。",
        claim("quantity_detail_referenced_not_numeric",True,(1,"进一步明确本次配股的具体配售比例及数量")),
        claim("largest_holder_subscription_commitment_referenced",True,(1,"明确了第一大股东承诺以现金方式全额认购公司本次配股方案确定的可获配股份")))
    add(45,"FEEDBACK_REPLY_REVISED_PENDING_CSRC","中信A股及H股均待核准，目录8月27日不回填落款8月26日。",
        claim("regulatory_A_and_H_approval","BOTH_PENDING",(1,"公司本次A股配股和H股配股事项尚需获得中国证监会核准")))
    add(46,"PLAN_AMENDMENT_DETAIL_NOTICE","财通公告只指向具体比例数量，没有给数值；保持未知。",
        claim("plan_year",2021,(1,"2021年度配股公开发行预案")),
        claim("quantity_detail_referenced_not_numeric",True,(1,"主要对本次配股的具体配股比例及数量进行了明确")))
    add(47,"PLAN_AMENDMENT_DETAIL_NOTICE","浙商更新比例数量及内外部审批流程说明，具体值仍待对应预案。",
        claim("quantity_detail_referenced_not_numeric",True,(1,"明确了本次配股的具体配售比例及数量")),
        claim("approval_procedures_updated_not_approval_proof",True,(1,"更新了本次配股履行的公司内部决策和外部审批程序")))
    add(48,"VALIDITY_EXTENSION_PROPOSAL_WITH_A_H_QUANTITIES","浙商拟续至2023-11-22，同时报告获授权人士已确定数量；末段仍提请三类股东大会，不简化为整体批准。",
        claim("proposed_validity_end","2023-11-22",(2,"自前次有效期届满后延长12个月,即延长至2023年11月22日")),
        claim("shareholder_approval","PENDING_GENERAL_AND_A_H",(2,"上述事项尚需提交本公司股东大会及A股类别股东大会、H股类别股东大会审议")),
        claim("plan_A_H_quantity_scenario",{"basis_date":"2022-06-30","basis_shares":21268696778,"total_shares":6380609033,"A_shares":5014409033,"H_shares":1366200000},
              (2,"截至2022年6月30日的总股本21,268,696,778股为基数测算"),(2,"本次可配售股份数量总计6,380,609,033股,其中A股可配售股份数量为5,014,409,033股,H股可配售股份数量为1,366,200,000股")),
        claim("plan_gross_A_H_proceeds_cap",money("180"),(1,"批准本公司配股发行证券募集资金不超过人民币180亿元")),
        claim("plan_ratio_per_10_A_and_H","3",(2,"本次配股的具体配售比例为每10股配售3股,A股和H股配股比例相同")))
    add(49,"PLAN_AMENDMENT_DETAIL_NOTICE","中信银行预案修订说明，不等于已注册或已实施，数量需查原预案。",
        claim("quantity_detail_referenced_not_numeric",True,(1,"明确了本次配股的具体配售比例和数量")),
        claim("financial_report_period","2022Q3",(1,"根据本行2022年第三季度报告更新本行相关财务信息")))
    add(50,"REGULATORY_ROUTE_WORDING_REVISED","此3页文件是修订说明，改核准表述为审核注册流程，没有实际登记日或注册决定。",
        claim("record_date_still_to_be_determined",True,(2,"本次配股股权登记日将在中国证券监督管理委员会(以下简称“中国证监会”)作出予以注册决定后另行确定")),
        claim("issuance_route_wording","SSE_REVIEW_AND_CSRC_REGISTRATION_REQUIRED",(2,"本次配股经上海证券交易所审核并获得中国证监会同意注册后,在注册决定有效期内择机向全体股东配售股份")),
        claim("body_role_corrected","AMENDMENT_DESCRIPTION_NOT_FULL_PLAN",(2,"《中信银行股份有限公司配股方案》的主要修订情况")))
    for index,period,page in [(51,None,1),(52,None,1),(53,"2023H1",2),(55,"2023FY",1),(56,"2024H1",1)]:
        items=[claim("regulatory_approvals","SSE_REVIEW_AND_CSRC_REGISTRATION_PENDING",(page,"尚需通过上交所审核,并获得中国证券监督管理委员会(以下简称“中国证监会”)做出同意注册的决定后方可实施"))]
        if period:
            quote={"2023H1":"2023年半年度财务数据更新版","2023FY":"2023年度财务数据更新版","2024H1":"2024年半年度财务数据更新版"}[period]
            items.append(claim("application_financial_vintage",period,(page,quote)))
        add(index,"APPLICATION_UPDATE_NOT_IMPLEMENTED","建发仅更新申报材料；本公告明确上交所审核和证监会注册仍未完成，不能生成发行日历。",*items)
    add(54,"HISTORICAL_PROCEEDS_REALLOCATION_NOT_NEW_ISSUE","隆基2024处理同一2018配股结余，原资金账及后来用途变更不重复计入新供给。",
        claim("historical_issue_approval_key","2019_202",(1,"证监许可[2019]202号")),
        claim("proposed_surplus_working_capital",money("27557.47","万元"),(3,"节余募集资金总额27,557.47万元永久性补充流动资金")),
        claim("additional_shareholder_vote","NOT_REQUIRED_BY_NOTICE",(4,"本次事项无需提交公司股东大会审议")))
    add(57,"HISTORICAL_PROCEEDS_REALLOCATION_NOT_NEW_ISSUE","中科三环2024拟将2022实际配股结余补流，尚待股东大会，不新计发行股份。",
        claim("historical_issue_approval_key","2021_3203",(1,"证监许可〔2021〕3203号")),
        claim("proposed_surplus_working_capital",money("3872.07","万元"),(1,"节余募集资金3,872.07万元")),
        claim("shareholder_approval","PENDING",(1,"因此本事项尚需提交股东大会审议")))
    assert len(rows)==58 and {r["index"] for r in rows}==set(range(58))
    rows.sort(key=lambda r:r["index"])
    return rows


def build():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"])==item["sha256"]
    targets=read(OUT/"targets.json")
    output=[]
    for spec in specs():
        row=targets[spec["index"]]
        source=row["source"]
        doc=Document({**source,"review_role":"PLAN_ADMIN_NOTICE_REVIEW"})
        fact=base_fact(doc,None)
        assert fact["event_id"] is None and fact["known_at"]==row["known_at"]
        fact.update(administrative_status=spec["status"],reviewed_claims={},claim_evidence={},
                    full_body_reviewed=False,operative_clause_review_complete=True)
        fact["notes"].append(spec["note"])
        fact["notes"].append("计划及行政事项来源声明；没有创建实际发行日历、交易特征或补填历史首版时刻。")
        pages={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for c in spec["claims"]:
            field=c["field"]
            assert field not in fact["reviewed_claims"]
            hits=[]
            for pn,literal in c["proof"]:
                literal=normalized(literal);start=pages[pn].find(literal)
                assert start>=0,(spec["index"],row["document_id"],pn,literal)
                hits.append({"document_id":row["document_id"],"page":pn,"normalized_start":start,
                    "normalized_end":start+len(literal),"literal":literal})
            assert hits
            fact["reviewed_claims"][field]=c["value"]
            fact["claim_evidence"][field]=hits
        output.append(fact)
    return output


def source_view(fact, at):
    visible=fact["known_at"]<=at
    return {"document_id":fact["document_id"],"as_of":at,"known_at":fact["known_at"],
        "status":"REVIEWED_ADMIN_SOURCE_DATE_PROXY_ONLY" if visible else "NO_VIEW",
        "administrative_status":fact["administrative_status"] if visible else None,
        "reviewed_claims":deepcopy(fact["reviewed_claims"]) if visible else {},
        "actual_event_id":None,"actual_issuance_calendar":None,"strict_M06_pressure":None,
        "trading_feature_admitted":False}


def finalize():
    facts=build()
    assert len(facts)==58 and len({r["document_id"] for r in facts})==58
    save(OUT/"reviewed_admin_source_facts.json",facts)
    reviewed=[]
    for row in read(OUT/"source_candidates.json"):
        for c in row["candidates"]:
            reviewed.append({**c,"reviewed":True,"review_scope":"事项及计划条款复核；经营测算、监管意见和长回复内容未获独立验证。"})
    assert len(reviewed)==140
    save(OUT/"reviewed_source_candidates.json",reviewed)
    previous=read(PREVIOUS/"combined_source_facts.json")
    assert not {r["document_id"] for r in previous}&{r["document_id"] for r in facts}
    save(OUT/"combined_source_facts.json",previous+facts)
    queries=[]
    for fact in facts:
        at=fact["known_at"]
        before=(datetime.fromisoformat(at)-timedelta(seconds=1)).isoformat()
        queries.append({"document_id":fact["document_id"],"before":source_view(fact,before),"at":source_view(fact,at)})
    save(OUT/"admin_publication_queries.json",queries)
    legacy={r["document_id"]:r for r in previous}
    # 此处只是后验来源对照。关系不向更早时点开放，也不据此回填未来核准号。
    cases=[
        ("DONGWU_PLAN_2017",["1204532468","1205275256"],"1205354373","两份修订公告和终止公告均明确2017年度配股方案，与2020年实施事件不同。"),
        ("WANXIANG_PLAN_QUANTITY_COMPARISON",["1203642857","1203978535"],"1206233867","同发行人、配股新上限825947836股与失效公告核准股数相符；保留后验对照，不证明早期已知后来的核准身份。"),
        ("HEGANG_PLAN_2019",["1207875095"],"1208713385","均明确2019年度方案和2020_33号核准，先拟延长后终止。"),
    ]
    index={r["document_id"]:r for r in facts}
    comparisons=[]
    for name,ids,terminal,note in cases:
        end=legacy[terminal]
        assert all(index[rid]["symbol"]==end["symbol"] and index[rid]["known_at"]<end["known_at"] for rid in ids)
        comparisons.append({"comparison_id":name,"prior_source_ids":ids,"terminal_source_id":terminal,
            "relationship_known_at_no_earlier_than":end["known_at"],"terminal_status":end["administrative_status"],
            "note":note,"terminal_fact_unchanged":True,"earlier_active_status_inferred":False,
            "joined_to_executed_event":False,"trading_feature_admitted":False})
    save(OUT/"terminal_source_comparisons.json",comparisons)
    with (OUT/"58份方案与审核事项.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["公告ID","证券","可得日期代理","标题","事项分类","事项说明","结构化来源字段","实际发行日历","来源URL"])
        writer.writeheader()
        for r in facts:
            writer.writerow({"公告ID":r["document_id"],"证券":r["symbol"],"可得日期代理":r["known_at"],"标题":r["title"],
                "事项分类":r["administrative_status"],"事项说明":r["notes"][0],
                "结构化来源字段":json.dumps(r["reviewed_claims"],ensure_ascii=False,separators=(",",":")),
                "实际发行日历":"未创建；计划或行政事项不能替代实施公告","来源URL":r["source_url"]})
    result={"at":now(),"study_id":"510300_FACTOR96_RIGHTS_PLAN_ADMIN_V1",
        "status":"58_PLAN_AND_ADMIN_NOTICE_OPERATIVE_CLAUSES_REVIEWED",
        "previous_goal_turn_classification":"PROGRESS_36_LISTING_REPORTED_SPLITS_AND_CONSTRAINTS",
        "documents":58,"target_pdf_pages":147,"candidate_fragments_read":140,"candidate_characters":69953,
        "source_fields_reviewed":sum(len(r["reviewed_claims"]) for r in facts),
        "administrative_status_counts":dict(Counter(r["administrative_status"] for r in facts)),
        "historical_proceeds_documents_excluded_from_new_supply":4,"application_updates_not_implemented":5,
        "quantity_amendment_notices_without_numeric_values":9,"visual_pages_reviewed":len(VISUAL_PAGES),
        "source_literal_conflicts_preserved":1,"source_role_reclassified_from_plan_to_amendment_notice":1,
        "new_canonical_source_documents":58,"prior_source_documents_unchanged":318,"combined_source_documents":376,
        "executed_rights_event_chains_unchanged":36,"new_actual_issuance_calendars":0,
        "new_admin_publication_query_pairs":58,"new_admin_publication_queries":116,
        "terminal_source_comparisons":3,"terminals_revived":0,
        "remaining_plan_admin_long_documents":63,"earlier_holding_clause_full_coverage":False,
        "full_issuance_coverage":False,"free_float_denominator_available":False,
        "strict_M06_pressure":"NOT_COMPUTED","T13":"NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,
        "qualified_candidates":0,"independent_forward_observations":0,"orders_authorized":False,
        "delivery_package_created":False,"goal_achieved":False,
        "next_source_action":"T13_63_LONG_PLAN_REPORT_DOCUMENTS_AND_EARLIER_HOLDING_CLAUSE_COVERAGE"}
    result["quantity_amendment_notices_without_numeric_values"]=sum(r["reviewed_claims"].get("quantity_detail_referenced_not_numeric",False) for r in facts)
    save(OUT/"result.json",result)
    print(json.dumps({"事项文件":58,"字段":result["source_fields_reviewed"],"后续长文件":63,
        "历史资金再用途":4,"申报更新未实施":5,"无数值数量说明":result["quantity_amendment_notices_without_numeric_values"],
        "新增实施日历":0,"新账户":0},ensure_ascii=False))


if __name__=="__main__":
    finalize()
