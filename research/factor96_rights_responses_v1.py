"""固定最后18份配股监管回复，提取当次发行和持有约束线索。"""
from collections import Counter
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT=ROOT/"reports/research/510300_factor96_rights_responses_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_rights_drafts_v1"
SOURCE=ROOT/"reports/research/510300_factor96_rights_issue_documents_v1"


def freeze():
    rows=read(PREVIOUS/"remaining_18_response_documents.json")["documents"]
    assert len(rows)==18 and all(r["body_role_candidate"]=="REGULATORY_RESPONSE_REPORT" for r in rows)
    save(OUT/"protocol.json",{"at":now(),"study_id":"510300_FACTOR96_RIGHTS_RESPONSES_V1",
        "previous_goal_turn_classification":"PROGRESS_22_USE_DILUTION_AND_DRAFT_SOURCE_DOCUMENTS",
        "scope":"固定最后18份监管回复，1373页；核对当次配股拟募资额、比例数量、版本变化及持有约束候选，不以最终实施或收益筛选。",
        "source_rule":"只读已有PDF和文本，先保存固定关键词候选，再按当前发行/历史发行/监管提问/经营测算区分；关键缺失和歧义回到原页。",
        "review_boundary":"仅称相关条款和补充原页已经阅读，不声称全部1373页逐页核读，不验证会计师意见或经营预测。",
        "clock_rule":"目录日末为可得日期代理；正文引用更早的承诺、批复、回复日期不提前时点；历史首次公开未证实。",
        "amount_rule":"监管提问可能复述旧方案，必须与当次答复及修订说明分开；项目总投资、拟用募集资金、实际融资及历史融资不混用。",
        "holding_rule":"当前配股承诺、旧非公开发行限售、激励限售、短线交易和减持募集资金禁止用途分开；实际受限股数及解锁日未明则保持未知。",
        "approval_rule":"回复或保荐核查意见不是监管同意，董事会提议不等于股东大会已批准，不创建实施日历。",
        "prior_state_rule":"已保存终止、失效、源文件冲突和失败账户不变，不用未来核准号回填早期计划身份。",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,"orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",rows)
    paths=[PREVIOUS/"remaining_18_response_documents.json",PREVIOUS/"combined_source_facts.json",OUT/"protocol.json",OUT/"targets.json"]
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in paths]})
    print("已固定18份监管回复。")


def extract():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    pattern=(r"募集资金总[额量]|(?:拟|配股)募集.{0,12}资金|本次.{0,12}(?:配股|发行).{0,40}(?:不超过|配售数量|发行数量)|"
             r"配股比例|每10股(?:配售|配股)|配股价格|不(?:会|再|得)?减持|六个月内|6个月内|限售|锁定|持有期|转让限制")
    output=[]
    for row in read(OUT/"targets.json"):
        src=row["source"]
        assert digest(SOURCE/src["raw_snapshot"])==src["raw_sha256"]
        assert digest(row["text_path"])==src["text_sha256"]
        pages=read(row["text_path"])["pages"];candidates=[]
        for p in pages:
            t=normalized(p["text"]);spans=[]
            for m in re.finditer(pattern,t):
                a,b=max(0,m.start()-100),min(len(t),m.end()+340)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:candidates.append({"page":p["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        output.append({"document_id":row["document_id"],"symbol":row["symbol"],"known_at":row["known_at"],"title":row["title"],
            "pages":len(pages),"candidates":candidates,"candidate_chars":sum(len(c["literal"]) for c in candidates)})
    save(OUT/"source_candidates.json",output)
    print("文件",len(output),"候选片段",sum(len(r["candidates"]) for r in output),"候选字符",sum(r["candidate_chars"] for r in output))
    for i,r in enumerate(output):print(i,r["document_id"],len(r["candidates"]),r["candidate_chars"])


def supplement():
    """补扫初始表达式未覆盖的减持表述，保留补充规则和独立候选。"""
    rule={"at":now(),"pattern":r"减持|不得转让|限售期|锁定期",
          "reason":"首份补充回复存在'不存在减持'，初始表达式未覆盖该写法；统一补扫18份，避免将未匹配当成不存在约束。",
          "context_before":200,"context_after":450,"exclude_matches_inside_initial_candidate":True}
    save(OUT/"supplement_rule.json",rule)
    original={r["document_id"]:r for r in read(OUT/"source_candidates.json")}
    result=[]
    for row in read(OUT/"targets.json"):
        old=original[row["document_id"]]["candidates"];candidates=[]
        for p in read(row["text_path"])["pages"]:
            t=normalized(p["text"]);spans=[]
            for m in re.finditer(rule["pattern"],t):
                if any(c["page"]==p["page"] and c["start"]<=m.start()<c["end"] for c in old):continue
                a,b=max(0,m.start()-200),min(len(t),m.end()+450)
                if spans and a<=spans[-1][1]:spans[-1][1]=max(b,spans[-1][1])
                else:spans.append([a,b])
            for a,b in spans:candidates.append({"page":p["page"],"start":a,"end":b,"literal":t[a:b],"reviewed":False})
        result.append({"document_id":row["document_id"],"candidates":candidates})
    save(OUT/"supplement_candidates.json",result)
    print("补充候选片段",sum(len(r["candidates"]) for r in result))
    for i,r in enumerate(result):print(i,r["document_id"],len(r["candidates"]))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","extract","supplement"]);a=p.parse_args()
    {"freeze":freeze,"extract":extract,"supplement":supplement}[a.stage]()
