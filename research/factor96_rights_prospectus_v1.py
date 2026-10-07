"""固定195份非提示文档，拆分说明书、方案版本及行政通知；仅处理来源。"""
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, SOURCE, read, save, digest, normalized, now
from research.factor96_rights_intermediate_v1 import OUT as PREVIOUS
from research.factor96_rights_reminder_review_v1 import FULL_DATE, hit
from research.factor96_rights_reminder_review_v1_0_1 import PRECISE_RANGE

OUT = ROOT / "reports/research/510300_factor96_rights_prospectus_v1"


def prospectus_candidates(row, pages):
    values = {key: [] for key in ("record_date", "payment_window", "issue_price_cents", "plan_max_shares", "rights_code", "invalid_candidates")}
    patterns = {
        "record_date": [r"股权登记日(?:[(/]?[RT]日[)/]?)?[(:为指,]*" + FULL_DATE,
                        FULL_DATE + r"\((?:[RT]日|股权登记日)[^)]*\)", FULL_DATE + r"股权登记日"],
        "issue_price_cents": [r"(?:配股价格|发行价格)[^0-9]{0,16}([0-9]+(?:\.[0-9]+)?)元/股"],
        "plan_max_shares": [r"(?:可配(?:售)?(?:A股)?(?:股份|股票)?(?:数量|总数|总额|总数量)|可配售股份总计|可配股数量总计为)(?:共计|总计|为|共|:){0,3}([0-9][0-9,]*)股"],
        "rights_code": [r"配股代码[为:‘’“”\"']{0,5}([0-9]{6})"],
    }
    for page in pages:
        text = normalized(page["text"])
        for field, expressions in patterns.items():
            for expression in expressions:
                for match in re.finditer(expression, text):
                    try:
                        if field == "record_date":
                            value = date(*map(int, match.groups()[:3])) .isoformat()
                        elif field == "issue_price_cents":
                            value = Decimal(match.group(1))*100
                            if value != value.to_integral_value():
                                raise ValueError("价格不是整分")
                            value = int(value)
                        elif field == "plan_max_shares":
                            value = int(match.group(1).replace(",", ""))
                        else:
                            value = match.group(1)
                    except ValueError as exc:
                        values["invalid_candidates"].append({**hit(row["document_id"], page["page"], text, match, None), "reason": str(exc), "field": field})
                        continue
                    values[field].append(hit(row["document_id"], page["page"], text, match, value))
        for match in PRECISE_RANGE.finditer(text):
            sy, sm, sd, ey, em, ed = match.groups()
            prefix, tail = text[max(0, match.start()-40):match.start()], text[match.end():match.end()+65]
            if not (re.search(r"[RT]\+5", tail) or "配股缴款起止" in tail or
                    any(x in prefix for x in ("缴款时间", "缴款期限", "缴款起止", "认购时间"))):
                continue
            if re.search(r"^[^0-9]{0,4}[RT]\+6", tail) or re.match(r"、[0-9]{1,2}月[0-9]{1,2}日", tail):
                continue
            try:
                start, end = date(int(sy), int(sm), int(sd)), date(int(ey or sy), int(em or sm), int(ed))
                if end < start:
                    raise ValueError("结束日期早于开始日期")
            except ValueError as exc:
                values["invalid_candidates"].append({**hit(row["document_id"], page["page"], text, match, None), "reason": str(exc), "field": "payment_window"})
                continue
            values["payment_window"].append(hit(row["document_id"], page["page"], text, match, [start.isoformat(), end.isoformat()]))
    return values


def classify(title):
    if "说明书" in title and "申报稿" in title:
        return "FILING_DRAFT_PROSPECTUS"
    if "说明书" in title and "提示性公告" in title:
        return "APPLICATION_DOCUMENT_UPDATE_NOTICE"
    if "说明书" in title:
        return "IMPLEMENTATION_PROSPECTUS"
    if "独立意见" in title:
        return "INDEPENDENT_DIRECTOR_OPINION"
    if "募集资金" in title and any(w in title for w in ("节余", "结余", "结项")):
        return "HISTORICAL_PROCEEDS_USE"
    if "反馈" in title or "准备工作" in title:
        return "REGULATORY_RESPONSE_NOTICE" if "公告" in title else "REGULATORY_RESPONSE_REPORT"
    if "预案" in title and ("修订稿" in title or "申报稿" in title):
        return "PROPOSED_PLAN_VERSION"
    if "可行性" in title:
        return "PROPOSED_PROCEEDS_FEASIBILITY"
    if "摊薄" in title or "填补" in title:
        return "DILUTION_RISK_OR_MEASURES"
    if "有效期" in title:
        return "RESOLUTION_VALIDITY_NOTICE"
    if "修订" in title or "调整" in title or "更新后" in title:
        return "PLAN_CHANGE_OR_CONDITION_NOTICE"
    return "OTHER_SOURCE_REQUIRING_BODY_REVIEW"


def freeze():
    source = PREVIOUS / "remaining_candidate_only_documents.json"
    index = read(source)["documents"]
    originals = {r["document_id"]: r for r in read(PREVIOUS / "remaining_local_documents.json")}
    rows = [{**row, "source": originals[row["document_id"]], "body_role_candidate": classify(row["title"])} for row in index]
    assert len(rows) == 195
    counts = dict(Counter(row["body_role_candidate"] for row in rows))
    assert counts["IMPLEMENTATION_PROSPECTUS"] == 74
    protocol = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_PROSPECTUS_V1",
        "previous_goal_turn": "PROGRESS_179_REMINDER_FIELD_COMPARISONS_AND_19_ORIGINALS",
        "scope": "上一轮遗留195份本地非提示文档，不按收益或字段差异选样；其中74份正式说明书/摘要优先核对实施日历，其他121份分类处理。",
        "roles": counts, "selection": "所有含说明书、非申报稿且非申请文件更新提示公告的74份全文/摘要/修订版。",
        "identity": "优先当次发行概况的A股核准文号；同公司旧核准文号仅为候选，不能自动匹配当前发行。无唯一明确身份则保留未知。",
        "fields": ["登记日", "缴款起止", "价格", "计划股数", "比例", "无限售和限售拆分", "计划结果日", "募集说明书版本"],
        "availability": "文档目录日末为日期代理，不承诺历史第一版可得。说明书早于实施公告的已明确安排可保存为当日来源，后来的实施公告仅作一致性核对。",
        "amendments": "方案变更、延期审查、股东大会待批、历史募集资金使用分开；计划或预算不替代实际发行。",
        "review_scope": "机器候选、字段核对和全文人工阅读分别标记，不把搜索命中或相同数值当作完整语义复核。",
        "stop": "歧义不按后来结果解消；字段缺失不填零；失败账户保持冻结。",
        "new_network_requests": 0, "new_accounts": 0, "new_returns": 0, "new_models": 0,
        "orders_authorized": False, "delivery_package_required": False}
    save(OUT / "protocol.json", protocol)
    save(OUT / "targets.json", rows)
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p), "sha256": digest(p)} for p in
         (source, OUT / "protocol.json", OUT / "targets.json", PREVIOUS / "combined_source_facts.json")]})
    print("已固定", len(rows), "份文件", counts)


def extract():
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(item["path"]) == item["sha256"]
    reports = []
    for row in read(OUT / "targets.json"):
        pages = read(row["text_path"])["pages"]
        assert digest(row["text_path"]) == row["source"]["text_sha256"]
        clauses = []
        for page in pages:
            text = normalized(page["text"])
            pattern = r"本次发行概况|本次配股.{0,6}(?:概况|情况)|证监许可|缴款(?:起止|时间|日期)|配股价格|无限售|有限售|更正|调整为|延长至|尚需.{0,12}(?:审议|核准)"
            for hit in re.finditer(pattern, text):
                a, b = max(0, hit.start()-100), min(len(text), hit.end()+240)
                clauses.append({"page": page["page"], "start": a, "end": b, "cue": hit.group(), "text": text[a:b]})
        entry = {"document_id": row["document_id"], "symbol": row["symbol"], "title": row["title"],
                 "known_at": row["known_at"], "role_candidate": row["body_role_candidate"], "clauses": clauses,
                 "full_body_read": False, "trading_feature_admitted": False}
        if row["body_role_candidate"] == "IMPLEMENTATION_PROSPECTUS":
            values = prospectus_candidates(row, pages)
            entry["field_candidates"] = values
            entry["unique_values"] = {k: list(dict.fromkeys(tuple(h["value"]) if isinstance(h["value"], list) else h["value"] for h in v)) for k,v in values.items()}
        reports.append(entry)
    save(OUT / "field_candidates.json", reports)
    print("已定位", len(reports), "文档，", sum(len(r["clauses"]) for r in reports), "条线索。")
    print("正式说明书候选已保存；歧义及未抽取值须逐项复核，不据此声明实施条款。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else extract()
