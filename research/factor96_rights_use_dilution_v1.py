"""固定15份配股用途及摊薄报告，区分募集计划与测算假设。"""
from collections import Counter
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT = ROOT/"reports/research/510300_factor96_rights_use_dilution_v1"
PREVIOUS = ROOT/"reports/research/510300_factor96_rights_preplans_v1"
SOURCE = ROOT/"reports/research/510300_factor96_rights_issue_documents_v1"


def freeze():
    rows = read(PREVIOUS/"remaining_40_long_documents.json")["documents"]
    roles = {"PROPOSED_PROCEEDS_FEASIBILITY","DILUTION_RISK_OR_MEASURES"}
    targets = [r for r in rows if r["body_role_candidate"] in roles]
    remaining = [r for r in rows if r["body_role_candidate"] not in roles]
    assert len(targets) == 15 and len(remaining) == 25
    save(OUT/"protocol.json",{"at":now(),"study_id":"510300_FACTOR96_RIGHTS_USE_DILUTION_V1",
        "previous_goal_turn_classification":"PROGRESS_23_PREPLAN_CLAUSES_NINE_NUMERIC_GAPS_AND_FIVE_VERSION_PAIRS",
        "scope":"固定余40份长文件中的全部9份用途可行性报告与6份摊薄说明，复核当次募集预算、用途分项、数量/日期假设、批准条件及持有期线索。",
        "selection_rule":"按既有正文分类全取15份；不按最终实施、股价或收益筛选，其余18份回复及7份申报稿保留待审。",
        "source_rule":"已有原PDF和文本不修改，关键表格按需要查看原页；候选条款阅读不代表全文或经营测算验证。",
        "clock_rule":"目录日末为可得日期代理；会议、签署、批准、预测日期独立，不替代来源可得日期；首次公开未确认。",
        "amount_rule":"总投资、拟用募集资金、实际融资、税后收益预测分开；亿元/万元/元及股/万股分开；A/H合计不当成A股金额。",
        "assumption_rule":"摊薄测算中的假设完成日、股数、利润和到账金额不进入实际发行日历或实际供给，不合成不存在的执行事件。",
        "identity_rule":"同日同发行人文件仅作计划来源对照，不重复计量；历史终止/失效记录不覆盖，不回填未来核准号。",
        "holding_rule":"核准后发行窗口、董监高薪酬承诺和全额认购承诺不等于新股锁定；未检出不证明无约束。",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,
        "orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",targets)
    save(OUT/"remaining_25_long_documents.json",{"at":now(),"documents":remaining,"count":25,
        "role_counts":dict(Counter(r["body_role_candidate"] for r in remaining)),"status":"BODY_REVIEW_PENDING"})
    paths=[PREVIOUS/"remaining_40_long_documents.json",PREVIOUS/"combined_source_facts.json",
           OUT/"protocol.json",OUT/"targets.json",OUT/"remaining_25_long_documents.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in paths]})
    print("已固定9份用途报告、6份摊薄说明，剩余25份长文件。")


def extract():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"]) == item["sha256"]
    pattern = r"募集资金总额|募集资金规模|募集资金不超过|每10股|每十股|假设.{0,30}配股|配股.{0,30}完成|尚需|尚须|尚待|仍需|持有期|不减持|限售|锁定|六个月|6个月"
    output=[]
    for row in read(OUT/"targets.json"):
        source = row["source"]
        assert digest(SOURCE/source["raw_snapshot"]) == source["raw_sha256"]
        assert digest(row["text_path"]) == source["text_sha256"]
        pages=read(row["text_path"])["pages"]
        candidates=[]
        for page in pages:
            t=normalized(page["text"]);spans=[]
            for m in re.finditer(pattern,t):
                a,b=max(0,m.start()-100),min(len(t),m.end()+300)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:
                candidates.append({"page":page["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        output.append({"document_id":row["document_id"],"symbol":row["symbol"],"known_at":row["known_at"],
            "title":row["title"],"body_role":row["body_role_candidate"],"pages":len(pages),"candidates":candidates,
            "candidate_chars":sum(len(c["literal"]) for c in candidates)})
    save(OUT/"source_candidates.json",output)
    print("文件",len(output),"总页数",sum(r["pages"] for r in output),
          "候选片段",sum(len(r["candidates"]) for r in output),"候选字符",sum(r["candidate_chars"] for r in output))


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["freeze","extract"]);args=parser.parse_args()
    freeze() if args.stage=="freeze" else extract()
