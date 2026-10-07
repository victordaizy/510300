"""固定74份正式配股说明书当前发行章节，补提持有及减持约束。"""
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT=ROOT/"reports/research/510300_factor96_rights_prospectus_constraints_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_rights_responses_v1"
PROSPECTUS=ROOT/"reports/research/510300_factor96_rights_prospectus_v1"
PATTERN=r"减持|限售|锁定|持有期|转让限制|不得转让|禁止转让|不得出售|不得卖出|不予流通"


def freeze():
    rows=read(PROSPECTUS/"reviewed_current_section_comparisons.json")
    sections=read(PROSPECTUS/"current_issue_sections_v1_0_1.json")
    assert len(rows)==len(sections)==74
    assert {r["document_id"] for r in rows}=={r["document_id"] for r in sections}
    save(OUT/"protocol.json",{"at":now(),"study_id":"510300_FACTOR96_RIGHTS_PROSPECTUS_CONSTRAINTS_V1",
        "previous_goal_turn_classification":"PROGRESS_18_RESPONSE_SOURCE_CLAUSES_AND_EARLIER_HOLDING_SOURCE",
        "scope":"固定74份正式说明书的本次发行章节，584页范围，补提一般持有、减持及特定股东承诺；先完成章节，不声称全文约束覆盖。",
        "reason":"既有说明书研究保存18条一般持有期语句；上市公告另见减持和高管约束，需检查其发行阶段的已披露来源。",
        "candidate_pattern":PATTERN,"context_before":140,"context_after":500,
        "clock_rule":"仍只采用目录日末代理；同日摘要和全文不算独立发行；与后续上市公告的关系仅在双方来源均可得时比较。",
        "semantic_rule":"区分一般发行持有期、控股股东/董监高承诺、旧发行和原有股份限制、监管引文、认购承诺；缺失不等于零或撤销。",
        "quantity_rule":"不从登记无限售数量机械扣除承诺对象持股或认购量；具体重叠、期限和版本不全则有效可卖量未知。",
        "existing_state_rule":"既有439份来源、36条实施事件、终止/失效和账户结果不覆盖。新增的是来源层条款，不自行用事后核准号回填身份。",
        "financial_result_read":False,"new_accounts":0,"new_models":0,"new_network_requests":0,
        "orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",rows)
    paths=[OUT/"protocol.json",OUT/"targets.json",PROSPECTUS/"current_issue_sections_v1_0_1.json",
           PROSPECTUS/"holding_restriction_clause_reviews.json",PREVIOUS/"combined_source_facts.json",
           ROOT/"reports/research/510300_factor96_rights_listing_split_v1/reviewed_listing_classifications.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in paths]})
    print("已固定74份说明书的当前发行章节。")


def extract():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    sections={r["document_id"]:r for r in read(PROSPECTUS/"current_issue_sections_v1_0_1.json")}
    rows=[]
    for target in read(OUT/"targets.json"):
        assert digest(target["raw_path"])==target["raw_sha256"] and digest(target["text_path"])==target["text_sha256"]
        pages={p["page"]:normalized(p["text"]) for p in read(target["text_path"])["pages"]}
        candidates=[]
        for part in sections[target["document_id"]]["section_pages"]:
            t=pages[part["page"]];assert t[part["start"]:part["end"]]==part["text"]
            spans=[]
            for m in re.finditer(PATTERN,part["text"]):
                a=max(part["start"],part["start"]+m.start()-140)
                b=min(part["end"],part["start"]+m.end()+500)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:candidates.append({"page":part["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        rows.append({"document_id":target["document_id"],"symbol":target["symbol"],"known_at":target["known_at"],
            "comparison_event_id":target["comparison_event_id"],"same_day_issuance_notice_id":target["same_day_issuance_notice_id"],
            "current_issue_pages":target["current_issue_pages"],"candidates":candidates})
    save(OUT/"source_candidates.json",rows)
    print("文件",len(rows),"命中文件",sum(bool(r["candidates"]) for r in rows),"片段",sum(len(r["candidates"]) for r in rows),
          "字符",sum(len(c["literal"]) for r in rows for c in r["candidates"]))
    for i,r in enumerate(rows):
        if r["candidates"]:print(i,r["document_id"],r["symbol"],len(r["candidates"]))


def frontmatter():
    """章节内未新增命中，按正文结构补扫其前的声明、重大事项提示等内容。"""
    rule={"at":now(),"scope":"74份目标中本次发行概况章节起点之前的全部正文；不含当前章节及其后正文。",
          "pattern":PATTERN,"context_before":160,"context_after":600,
          "reason":"当前章节仍仅命中已保存的18段一般持有期条款，需按文件结构补查前置提示；不以是否实施、收益或约束内容选文件。"}
    save(OUT/"frontmatter_rule.json",rule)
    sections={r["document_id"]:r for r in read(PROSPECTUS/"current_issue_sections_v1_0_1.json")}
    rows=[]
    for target in read(OUT/"targets.json"):
        section=sections[target["document_id"]];first=section["section_pages"][0];cs=[];page_count=0
        for page in read(target["text_path"])["pages"]:
            if page["page"]>first["page"]:break
            full=normalized(page["text"]);t=full[:first["start"]] if page["page"]==first["page"] else full
            if not t:continue
            page_count+=1;spans=[]
            for m in re.finditer(PATTERN,t):
                a,b=max(0,m.start()-160),min(len(t),m.end()+600)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:cs.append({"page":page["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        rows.append({"document_id":target["document_id"],"symbol":target["symbol"],"known_at":target["known_at"],
            "comparison_event_id":target["comparison_event_id"],"prefix_page_span":page_count,"candidates":cs})
    save(OUT/"frontmatter_candidates.json",rows)
    print("前置页范围",sum(r["prefix_page_span"] for r in rows),"命中文件",sum(bool(r["candidates"]) for r in rows),
          "片段",sum(len(r["candidates"]) for r in rows),"字符",sum(len(c["literal"]) for r in rows for c in r["candidates"]))
    for i,r in enumerate(rows):
        if r["candidates"]:print(i,r["document_id"],r["symbol"],len(r["candidates"]),sum(len(c["literal"]) for c in r["candidates"]))


def remainder():
    """继续扫描当前发行章节之后的正文，保留全部候选和统一优先阅读线索。"""
    rule={"at":now(),"scope":"74份目标中本次发行概况章节终点之后的全部正文，补齐固定文件文本扫描范围。",
        "pattern":PATTERN,"context_before":220,"context_after":650,
        "priority_pattern":r"本次配股|本次发行|此次配股|本次认购|六个月|6个月|控股股东.{0,35}承诺|自愿.{0,15}锁定",
        "priority_rule":"优先阅读当前发行或六个月承诺候选，其余候选全部保留；优先规则不代表已验证或可排除。",
        "reason":"当前章节18段和前置7段未发现新约束，按结构覆盖余下正文，避免把未检查页当成没有披露。"}
    save(OUT/"remainder_rule.json",rule)
    sections={r["document_id"]:r for r in read(PROSPECTUS/"current_issue_sections_v1_0_1.json")}
    rows=[]
    for target in read(OUT/"targets.json"):
        last=sections[target["document_id"]]["section_pages"][-1];cs=[];page_count=0
        for page in read(target["text_path"])["pages"]:
            if page["page"]<last["page"]:continue
            t=normalized(page["text"]);offset=last["end"] if page["page"]==last["page"] else 0
            if not t[offset:]:continue
            page_count+=1;spans=[]
            for m in re.finditer(PATTERN,t[offset:]):
                a,b=max(offset,offset+m.start()-220),min(len(t),offset+m.end()+650)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:cs.append({"page":page["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False,
                "priority":bool(re.search(rule["priority_pattern"],t[a:b]))})
        rows.append({"document_id":target["document_id"],"symbol":target["symbol"],"known_at":target["known_at"],
            "comparison_event_id":target["comparison_event_id"],"remainder_page_span":page_count,"candidates":cs})
    save(OUT/"remainder_candidates.json",rows)
    print("余下页范围",sum(r["remainder_page_span"] for r in rows),"命中文件",sum(bool(r["candidates"]) for r in rows),
        "片段",sum(len(r["candidates"]) for r in rows),"字符",sum(len(c["literal"]) for r in rows for c in r["candidates"]),
        "优先片段",sum(c["priority"] for r in rows for c in r["candidates"]))
    for i,r in enumerate(rows):
        if r["candidates"]:print(i,r["document_id"],r["symbol"],len(r["candidates"]),sum(c["priority"] for c in r["candidates"]))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","extract","frontmatter","remainder"]);args=p.parse_args()
    {"freeze":freeze,"extract":extract,"frontmatter":frontmatter,"remainder":remainder}[args.stage]()
