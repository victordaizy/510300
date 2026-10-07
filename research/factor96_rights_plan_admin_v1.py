"""固定剩余配股方案资料，先复核58份事项公告，再处理63份长文件。"""
from pathlib import Path
from collections import Counter
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT=ROOT/"reports/research/510300_factor96_rights_plan_admin_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_rights_listing_split_v1"
PROSPECTUS=ROOT/"reports/research/510300_factor96_rights_prospectus_v1"
SOURCE=ROOT/"reports/research/510300_factor96_rights_issue_documents_v1"
SHORT_ROLES={"REGULATORY_RESPONSE_NOTICE","RESOLUTION_VALIDITY_NOTICE","PLAN_CHANGE_OR_CONDITION_NOTICE",
             "INDEPENDENT_DIRECTOR_OPINION","HISTORICAL_PROCEEDS_USE","APPLICATION_DOCUMENT_UPDATE_NOTICE"}


def freeze():
    all_rows=read(PROSPECTUS/"remaining_121_plan_admin_documents.json")["documents"]
    targets=[r for r in all_rows if r["body_role_candidate"] in SHORT_ROLES or r["document_id"]=="1215952175"]
    ids={r["document_id"] for r in targets}
    remaining=[r for r in all_rows if r["document_id"] not in ids]
    assert len(all_rows)==121 and len(targets)==58 and len(remaining)==63
    save(OUT/"protocol.json",{"at":now(),"study_id":"510300_FACTOR96_RIGHTS_PLAN_ADMIN_V1",
        "previous_goal_turn":"PROGRESS_36_LISTING_REPORTED_SPLITS_AND_CONSTRAINTS",
        "scope":"固定剩余121份资料；本轮核对58份事项公告的计划调整、审核状态、期限和历史资金用途，63份长文件保留待审。",
        "selection_rule":"原有六类事项公告加1215952175（标题为修订说明，原误归预案版本），不按收益、最终发行成败或金额筛选。",
        "plan_rule":"原计划、调整后计划、实际实施分别保存；同日不同文件并列，不按公告ID强行确定先后；未获批准不等于终止。",
        "clock_rule":"目录日末为可得日期代理；正文中更早会议、回复或提交日期不能替代公告可得时点；历史首版未知。",
        "quantity_rule":"万股、亿股、万元、亿元分开；预计上限不是实际认配；A股和A/H合计分开；条件股本基准、比例、用途金额分开。",
        "identity_rule":"未有同一发行的充分证据时保留独立计划或行政记录，不用未来核准号回填过去身份，不创建实施日历。",
        "scope_rule":"审核回复提示不是对长回复内容的背书；募集资金结余再用途不是新增配股；董事会提议延期不能自动变成股东大会已批准。",
        "existing_terminals":"既有终止及失效记录原样保留，过去计划变更不能重启后来已终止事件。",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",targets)
    save(OUT/"remaining_63_long_documents.json",{"at":now(),"documents":remaining,"count":63,"status":"BODY_REVIEW_PENDING"})
    files=[PROSPECTUS/"remaining_121_plan_admin_documents.json",PREVIOUS/"combined_source_facts.json",
           PREVIOUS/"event_chains.json",OUT/"protocol.json",OUT/"targets.json",OUT/"remaining_63_long_documents.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in files]})
    print("已固定58份事项公告及剩余63份长文件。")


def extract():
    for r in read(OUT/"freeze.json")["files"]:
        assert digest(r["path"])==r["sha256"]
    outputs=[]
    pattern=r"调整|调减|修订|配售比例|配股比例|发行数量|募集资金总额|募集资金规模|有效期|延期|延长|终止|撤回|不减持|限售|锁定|持有期|尚需|尚须|尚待|仍需|仍须|核准|批准|受理|回复|注册|结项|节余|结余|补充流动资金|分红|利润分配"
    for row in read(OUT/"targets.json"):
        source=row["source"]
        assert digest(SOURCE/source["raw_snapshot"])==source["raw_sha256"] and digest(row["text_path"])==source["text_sha256"]
        candidates=[]
        pages=read(row["text_path"])["pages"]
        for page in pages:
            t=normalized(page["text"])
            spans=[]
            for m in re.finditer(pattern,t):
                a,b=max(0,m.start()-110),min(len(t),m.end()+240)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:
                candidates.append({"document_id":row["document_id"],"page":page["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        outputs.append({"document_id":row["document_id"],"symbol":row["symbol"],"title":row["title"],"known_at":row["known_at"],
            "pages":len(pages),"candidates":candidates,"candidate_chars":sum(len(c["literal"]) for c in candidates),
            "raw_path":str(SOURCE/source["raw_snapshot"]),"text_path":row["text_path"]})
    save(OUT/"source_candidates.json",outputs)
    print("事项文件",len(outputs),"候选片段",sum(len(r["candidates"]) for r in outputs),"候选字符",sum(r["candidate_chars"] for r in outputs))
    print("文件页数分布",dict(Counter(r["pages"] for r in outputs)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","extract"]);args=p.parse_args()
    freeze() if args.stage=="freeze" else extract()
