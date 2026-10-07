"""固定7份申报稿的当次发行概况，保留未知日期和监管待办。"""
from collections import Counter
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT=ROOT/"reports/research/510300_factor96_rights_drafts_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_rights_use_dilution_v1"
SOURCE=ROOT/"reports/research/510300_factor96_rights_issue_documents_v1"
RANGES={"1216051593":(18,23),"1217076017":(14,22),"1217530739":(14,22),"1217766947":(14,22),
        "1217834018":(14,22),"1219892998":(14,23),"1221204347":(14,23)}


def freeze():
    rows=read(PREVIOUS/"remaining_25_long_documents.json")["documents"]
    targets=[r for r in rows if r["body_role_candidate"]=="FILING_DRAFT_PROSPECTUS"]
    remaining=[r for r in rows if r["body_role_candidate"]!="FILING_DRAFT_PROSPECTUS"]
    assert len(targets)==7 and len(remaining)==18 and {r["document_id"] for r in targets}==set(RANGES)
    save(OUT/"protocol.json",{"at":now(),"study_id":"510300_FACTOR96_RIGHTS_DRAFTS_V1",
        "scope":"剩余25份中全部7份申报稿；按目录及发行概况标题定位，中信银行PDF18-23页、建发前四版14-22页及2024两版14-23页。",
        "source_rule":"只复核当次发行批准条件、数量及股本基准、募资预算、日期留白、有效期和上市持有条款，不宣称2237页全文或经营数据已验证。",
        "clock_rule":"目录日末为可得日期代理，会议和被引用批复日不提前来源公开时点，历史首次公开未知。",
        "draft_rule":"申报稿不是发行公告；方括号日期留白、相对R日历和定价原则不能填成实际日期或价格；SSE审核/CSRC注册待办不得升级为已获准。",
        "quantity_rule":"计划上限、总股本、回购股份扣除、A/H合计与A股分别保存；资本调整前后的情景不累加；数据更新不是多次融资。",
        "holding_rule":"一般不设持有期与法规例外同时保留，不计算有效可售数量，不擅自推定例外覆盖范围或解锁日。",
        "prior_failed_research":"此前账户及终止事件不修改。",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",targets)
    save(OUT/"remaining_18_response_documents.json",{"at":now(),"documents":remaining,"count":18,
        "role_counts":dict(Counter(r["body_role_candidate"] for r in remaining)),"status":"BODY_REVIEW_PENDING"})
    paths=[PREVIOUS/"remaining_25_long_documents.json",PREVIOUS/"combined_source_facts.json",OUT/"protocol.json",OUT/"targets.json",OUT/"remaining_18_response_documents.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in paths]})
    print("已固定7份申报稿发行概况，保留18份监管回复待审。")


def extract():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    pattern=r"每10股|每十股|募集资金总额|募集资金不超过|拟募集资金|本次募集|尚需|尚须|尚待|核准|同意注册|有效期|持有期|不减持|限售|锁定|发行日程|股权登记日|发行价格|配股价格"
    output=[]
    for row in read(OUT/"targets.json"):
        source=row["source"]
        assert digest(SOURCE/source["raw_snapshot"])==source["raw_sha256"] and digest(row["text_path"])==source["text_sha256"]
        pages=read(row["text_path"])["pages"];a,z=RANGES[row["document_id"]];candidates=[]
        for p in pages:
            if not a<=p["page"]<=z:continue
            t=normalized(p["text"]);spans=[]
            for m in re.finditer(pattern,t):
                x,y=max(0,m.start()-110),min(len(t),m.end()+340)
                if spans and x<=spans[-1][1]:spans[-1][1]=max(y,spans[-1][1])
                else:spans.append([x,y])
            for x,y in spans:candidates.append({"page":p["page"],"start":x,"end":y,"literal":t[x:y],"reviewed":False})
        output.append({"document_id":row["document_id"],"symbol":row["symbol"],"known_at":row["known_at"],"title":row["title"],
            "total_pdf_pages":len(pages),"operative_page_range":[a,z],"candidates":candidates,"candidate_chars":sum(len(c["literal"]) for c in candidates)})
    save(OUT/"source_candidates.json",output)
    print("申报稿",len(output),"候选片段",sum(len(r["candidates"]) for r in output),"候选字符",sum(r["candidate_chars"] for r in output))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","extract"]);a=p.parse_args()
    freeze() if a.stage=="freeze" else extract()
