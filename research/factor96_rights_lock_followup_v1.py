"""限定中信与华安两条持有约束的数据补证范围。"""
from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, now

OUT=ROOT/"reports/research/510300_factor96_rights_lock_followup_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_rights_holding_remainder_v1"


def freeze():
    save(OUT/"protocol.json",{
        "at":now(),"study_id":"510300_FACTOR96_RIGHTS_LOCK_FOLLOWUP_V1",
        "previous_goal_turn_classification":"PROGRESS_248_CONTEXTS_REVIEWED_AND_EXPLICIT_RIGHTS_INHERITANCE_FOUND",
        "scope":["中信证券2022年配股：越秀金控、金控有限实际认购数量及旧收购锁定的变更或后续履行",
                 "华安证券2021年配股：安徽国控原IPO锁定期届满及随后两年无减持计划的实际范围"],
        "search_rule":"先查已有发行公告目录与已下载原文，再定向检索发行人、相关股东或交易所的原始披露；网页摘要只定位，不替代原文。",
        "clock_rule":"实际公开日期与经济事实日期分开，后来的解禁或年度报告不得倒填到配股之前；仅有目录日则继续采用日末代理。",
        "quantity_rule":"必须有原始披露明确认购或衍生受限股数；股东表差额、原股数乘配股比例不自行升级为实际认购。",
        "source_rule":"新增文件及文本留存内容哈希与取得时间；核对相关正文与表格，不声称全文通读。",
        "result_rule":"已有来源及账户不覆盖，不更新历史收益或放宽分母门槛，不改变失败结果。",
        "new_accounts":0,"new_models":0,"orders_authorized":False,"delivery_package_required":False,
    })
    ps=[OUT/"protocol.json",PREVIOUS/"combined_source_facts.json",PREVIOUS/"claim_addenda.json",
        PREVIOUS/"citic_inheritance_vs_listing_classification.json",PREVIOUS/"unresolved_holding_links.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in ps]})
    print("已固定中信与华安两条来源补证范围，旧结果不变。")


if __name__=="__main__":freeze()
