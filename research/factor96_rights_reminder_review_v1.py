"""提示公告的明确条款定位与既有已知条款对照，异常保留供原文复核。"""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import re

from research.factor96_rights_intermediate_v1 import OUT, BASE, read, save
from research.factor96_rights_event_chains_v1 import normalized, now
from research.factor96_rights_event_review_v1 import state_at

FULL_DATE = r"(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日"
RANGE = re.compile(r"(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日.{0,22}?(?:起至|至|—|－|–|-)(?:(20[0-9]{2})年)?(?:([0-9]{1,2})月)?([0-9]{1,2})日")
FIELDS = ("record_date", "payment_window", "issue_price_cents", "plan_max_shares")


def hit(doc_id, page, text, match, value):
    return {"document_id": doc_id, "page": page, "start": match.start(), "end": match.end(),
            "literal": match.group(0), "context": text[max(0,match.start()-75):match.end()+110], "value": value}


def parse(row):
    values = {key: [] for key in (*FIELDS, "rights_code")}
    for page in read(row["text_path"])["pages"]:
        text = normalized(page["text"])
        patterns = {
            "record_date": r"股权登记日[^0-9]{0,16}" + FULL_DATE,
            "issue_price_cents": r"(?:配股价格|发行价格)[^0-9]{0,16}([0-9]+(?:\.[0-9]+)?)元/股",
            "plan_max_shares": r"可配(?:售)?(?:A股)?(?:股份|股票)?(?:数量|总数|总额|总数量)(?:共计|总计|为|共|:){0,3}([0-9][0-9,]*)股",
            "rights_code": r"配股代码[为:‘’“”\"']{0,5}([0-9]{6})",
        }
        for field, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                if field == "record_date":
                    value = date(*map(int, match.groups())).isoformat()
                elif field == "issue_price_cents":
                    cents = Decimal(match.group(1))*100
                    assert cents == cents.to_integral_value()
                    value = int(cents)
                elif field == "plan_max_shares":
                    value = int(match.group(1).replace(",", ""))
                else:
                    value = match.group(1)
                values[field].append(hit(row["document_id"], page["page"], text, match, value))
        for match in RANGE.finditer(text):
            sy, sm, sd, ey, em, ed = match.groups()
            start = date(int(sy), int(sm), int(sd))
            end = date(int(ey or sy), int(em or sm), int(ed))
            if end < start:
                continue
            tail = text[match.end():match.end()+42]
            prefix = text[max(0,match.start()-30):match.start()]
            # 明确到R/T+6的停牌、清算区间不能成为缴款窗口。
            if re.search(r"^[^0-9]{0,4}[RT]\+6",tail):
                continue
            labelled_five = bool(re.search(r"[RT]\+5",tail))
            labelled_payment = (any(x in prefix for x in ("缴款时间", "缴款期限", "缴款起止", "认购时间"))
                                or "配股缴款起止" in tail)
            if not (labelled_five or labelled_payment):
                continue
            values["payment_window"].append(hit(row["document_id"],page["page"],text,match,[start.isoformat(),end.isoformat()]))
    return values


def draft():
    index = read(OUT/"intermediate_document_index.json")
    groups = defaultdict(list)
    for fact in read(BASE/"reviewed_document_facts.json"):
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    rows = []
    for row in index:
        if row["title_class"] != "PAYMENT_OR_RESUMPTION_REMINDER_TITLE":
            continue
        candidates = parse(row)
        unique = {key: list(dict.fromkeys(tuple(h["value"]) if isinstance(h["value"],list) else h["value"] for h in hits))
                  for key,hits in candidates.items()}
        events = row["candidate_event_ids"]
        event = events[0] if len(events)==1 else None
        if row["document_id"] == "1211684963":
            event = "002142.SZ_A_RIGHTS_2021_2718"
        if row["document_id"] in ("1204534489","1204577532","1204587123"):
            event = "600256.SH_A_RIGHTS_2018_157"
        expected = state_at(groups[event],row["known_at"]) if event else {}
        comparisons = {}
        for field in FIELDS:
            expected_value = (expected.get("payment_start"),expected.get("payment_end")) if field=="payment_window" else expected.get(field)
            found = unique[field]
            if expected_value is None or field=="payment_window" and None in expected_value:
                comparisons[field] = "NO_PRIOR_ADMITTED_SOURCE_CLAIM"
            elif not found:
                comparisons[field] = "NOT_EXTRACTED"
            elif len(found)>1:
                comparisons[field] = "MULTIPLE_VALUES_REQUIRE_REVIEW"
            elif found[0] == expected_value:
                comparisons[field] = "MATCHES_PRIOR_KNOWN_CLAIM"
            else:
                comparisons[field] = "DIFFERS_FROM_PRIOR_KNOWN_CLAIM"
        rows.append({**row,"review_event_id":event,"parsed_values":unique,"parsed_evidence":candidates,
                     "prior_known_source_state":expected,"comparisons":comparisons,
                     "source_semantics_review_complete":False,"trading_feature_admitted":False})
    save(OUT/"reminder_comparison_draft.json",rows)
    print("实际提示标题记录",len(rows))
    for field in FIELDS:
        print(field,dict(Counter(r["comparisons"][field] for r in rows)))
    for row in rows:
        if any(v!="MATCHES_PRIOR_KNOWN_CLAIM" for v in row["comparisons"].values()):
            print(row["document_id"],row["symbol"],row["title"],row["parsed_values"],row["comparisons"])


if __name__ == "__main__":
    draft()
