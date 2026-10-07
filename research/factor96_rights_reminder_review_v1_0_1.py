"""修正跨表行取值，独立保留V1候选；不利用收益选择或修改公告条款。"""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import re

from research.factor96_rights_intermediate_v1 import OUT, BASE, read, save
from research.factor96_rights_event_chains_v1 import normalized, now
from research.factor96_rights_event_review_v1 import state_at
from research.factor96_rights_reminder_review_v1 import parse, hit, FULL_DATE, FIELDS

PRECISE_RANGE = re.compile(r"(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日(?:\([RT]\+1日?\))?日?起?(?:至|—|－|–|-)(?:(20[0-9]{2})年)?(?:([0-9]{1,2})月)?([0-9]{1,2})日")


def parse_refined(row):
    values = parse(row)
    values["record_date"] = []
    values["payment_window"] = []
    values["payment_segment"] = []
    record_patterns = [r"股权登记日(?:[(/]?[RT]日[)/]?)?[(:为指,]*"+FULL_DATE,
                       FULL_DATE+r"\((?:[RT]日|股权登记日)[^)]*\)",
                       FULL_DATE+r"股权登记日"]
    for page in read(row["text_path"])["pages"]:
        text = normalized(page["text"])
        for pattern in record_patterns:
            for match in re.finditer(pattern,text):
                value = date(*map(int,match.groups()[:3])).isoformat()
                values["record_date"].append(hit(row["document_id"],page["page"],text,match,value))
        for match in re.finditer(r"(?:可配售股份总计|可配股数量总计为)([0-9][0-9,]*)股",text):
            value = int(match.group(1).replace(",",""))
            values["plan_max_shares"].append(hit(row["document_id"],page["page"],text,match,value))
        for match in PRECISE_RANGE.finditer(text):
            sy,sm,sd,ey,em,ed=match.groups()
            start,end=date(int(sy),int(sm),int(sd)),date(int(ey or sy),int(em or sm),int(ed))
            if end<start:
                continue
            tail=text[match.end():match.end()+55]
            prefix=text[max(0,match.start()-30):match.start()]
            if re.search(r"^[^0-9]{0,4}[RT]\+6",tail):
                continue
            if not (re.search(r"[RT]\+5",tail) or "配股缴款起止" in tail or
                    any(x in prefix for x in ("缴款时间","缴款期限","缴款起止","认购时间"))):
                continue
            field="payment_segment" if re.match(r"、[0-9]{1,2}月[0-9]{1,2}日",tail) else "payment_window"
            values[field].append(hit(row["document_id"],page["page"],text,match,[start.isoformat(),end.isoformat()]))
    return values


def main():
    drafts=read(OUT/"reminder_comparison_draft.json")
    rows=[]
    for row in drafts:
        evidence=parse_refined(row)
        unique={key:list(dict.fromkeys(tuple(h["value"]) if isinstance(h["value"],list) else h["value"] for h in entries))
                for key,entries in evidence.items()}
        expected=row["prior_known_source_state"]
        comparisons={}
        for field in FIELDS:
            prior=(expected.get("payment_start"),expected.get("payment_end")) if field=="payment_window" else expected.get(field)
            found=unique[field]
            if prior is None or field=="payment_window" and None in prior:
                state="NO_PRIOR_ADMITTED_SOURCE_CLAIM"
            elif not found:
                state="NOT_EXTRACTED"
            elif len(found)>1:
                state="MULTIPLE_VALUES_REQUIRE_REVIEW"
            elif found[0]==prior:
                state="MATCHES_PRIOR_KNOWN_CLAIM"
            else:
                state="DIFFERS_FROM_PRIOR_KNOWN_CLAIM"
            comparisons[field]=state
        rows.append({**row,"parsed_values":unique,"parsed_evidence":evidence,"comparisons":comparisons})
    save(OUT/"reminder_comparison_refined_v1_0_1.json",rows)
    save(OUT/"parser_refinement_receipt.json",{"at":now(),"original_draft_preserved":True,
         "source_semantic_repairs":["登记日不跨正常交易/停牌表格行，补读日期在标签之前的格式",
                                    "缴款范围不跨另一完整日期；连续日期加单日的表格片段单列",
                                    "补定位可配售股份总计及可配股数量总计为两种同义标签"],
         "uses_future_returns":False,"new_accounts":0})
    for field in FIELDS:
        print(field,dict(Counter(r["comparisons"][field] for r in rows)))
    print("仍需特别复核的条目")
    for row in rows:
        if any(value!="MATCHES_PRIOR_KNOWN_CLAIM" for value in row["comparisons"].values()):
            print(row["document_id"],row["symbol"],row["parsed_values"],row["comparisons"])


if __name__=="__main__":
    main()
