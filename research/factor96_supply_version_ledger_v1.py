"""拆分T13的32份版本候选，并保存明确引用与按当时可见版本的延期案例。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"reports/research/510300_factor96_known_supply_sources_v1"
OUT = ROOT/"reports/research/510300_factor96_supply_version_ledger_v1"
STUDY = "510300_FACTOR96_SUPPLY_VERSION_LEDGER_V1"
LINKS = [
    ("1201019982", "1200998313", 1, "2015年5月12日", "PARTIAL_ALLOCATION_SUPPLEMENT", "补充文件说明权益分派衔接及部分股数重叠，禁止两份总量直接相加。"),
    ("1201929797", "1201926176", 1, "应更正为", "ISSUER_CORRECTION_OF_OPINION", "2015年1月21日改为2016年1月21日，修订对象是核查意见，不新增发行人供给事件。"),
    ("1204032504", "1204030648", 1, "2017-038", "HOLDER_IDENTITY_SUPPLEMENT", "股东更名补充，不新增本次限售上市数量。"),
    ("1204139386", "1204132799", 1, "2017-048", "SUBTOTAL_CORRECTION_TOTAL_UNCHANGED", "其他首发限售股分项改数，合计数量不变。"),
    ("1205700275", "1205688138", 1, "2018-118", "CAPITAL_STRUCTURE_CORRECTION", "更正股本结构表，不新增另一笔153946037股。"),
    ("1205995258", "1205992913", 1, "2019-014", "CAPITAL_STRUCTURE_CORRECTION", "更正股本结构表，原上市变动量507700000股未改。"),
    ("1206158876", "1206126296", 1, "2019-022", "CAPITAL_STRUCTURE_CORRECTION", "更正股本结构表，原上市变动量4564607股未改。"),
    ("1207939596", "1207938885", 1, "于6月18日披露了", "CAPITAL_STRUCTURE_CORRECTION", "总股本变动项更正为零；档案日为6月19日，不回填为6月18日可用。"),
    ("1207281952", "1206469829", 2, "2019-061", "SAME_TRANCHE_DATE_EXTENSION", "同一2016年登记批次，由2020年1月26日延至2021年1月26日。"),
    ("1209076663", "1207281952", 2, "2020-009", "SAME_TRANCHE_OPEN_ENDED_EXTENSION", "仍未完成补偿，新的解禁日期不确定，不能当作窗口结束或供给归零。"),
]
SCHEDULE = [
    {"document_id": "1206469829", "archive_date": "2019-07-24", "known_at": "2019-07-26T00:00:00+08:00", "prior_scheduled_date": "2019-07-26", "scheduled_unlock_date": "2020-01-26", "state": "DATED_CONDITIONAL_EXTENSION"},
    {"document_id": "1207281952", "archive_date": "2020-01-23", "known_at": "2020-01-25T00:00:00+08:00", "prior_scheduled_date": "2020-01-26", "scheduled_unlock_date": "2021-01-26", "state": "DATED_CONDITIONAL_EXTENSION"},
    {"document_id": "1209076663", "archive_date": "2021-01-09", "known_at": "2021-01-11T00:00:00+08:00", "prior_scheduled_date": "2021-01-26", "scheduled_unlock_date": None, "state": "OPEN_ENDED_CONTINGENT_NOT_ZERO_SUPPLY"},
]


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def document_role(title, first_page, symbol, scanned_manual=False):
    title, first_page = normalized(title), normalized(first_page)
    # 发行人对核查意见的更正仍是发行人更正文，不能先按“核查”二字排为意见。
    if title.endswith("更正公告"):
        return "ISSUER_CORRECTION" if symbol[:6] in first_page else "NO_VIEW_ISSUER_ID"
    if any(term in title for term in ["核查意见", "法律意见", "独立财务顾问"]):
        return "SUPPORTING_OPINION" if first_page or scanned_manual else "NO_VIEW_SOURCE_TEXT"
    if symbol[:6] not in first_page:
        return "NO_VIEW_ISSUER_ID"
    if "延期上市流通" in title:
        return "ISSUER_SCHEDULE_EXTENSION"
    if "补充" in title:
        return "ISSUER_SUPPLEMENT"
    if re.search("暂缓授予|暂缓部分", title) and "解除限售" in first_page:
        return "ISSUER_DEFERRED_GRANT_UNLOCK"
    return "NO_VIEW_UNRESOLVED_ROLE"


def visible_schedule(records, decision_time):
    stamp = pd.Timestamp(decision_time)
    if stamp.tzinfo is None:
        raise ValueError("判断时间必须带时区")
    known = [r for r in records if pd.Timestamp(r["known_at"]) <= stamp]
    return max(known, key=lambda r: pd.Timestamp(r["known_at"])) if known else None


def prepare():
    assert not (OUT/"protocol.json").exists()
    documents = read(SOURCE/"documents.json")
    fields = {r["document_id"]: r for r in read(SOURCE/"event_fields_v1/document_fields.json")}
    versions = [r for r in documents if r["title_kind"] == "REVISION_TITLE_RETAINED_SEPARATELY"]
    ids = {r["document_id"] for r in versions} | {x[1] for x in LINKS}
    selected = [r for r in documents if r["document_id"] in ids]
    assert len(versions) == 32 and len(selected) == 39
    for folder in ["inputs/text", "inputs/raw", "code"]:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    for row in selected:
        for kind in ["text", "raw"]:
            original = SOURCE/row[f"{kind}_path"]
            target = OUT/"inputs"/kind/(row["document_id"]+original.suffix)
            shutil.copy2(original, target)
            assert digest(target) == row[f"{kind}_sha256"]
    save(OUT/"inputs/documents.json", selected)
    save(OUT/"inputs/prior_fields.json", [fields[r["document_id"]] for r in selected])
    save(OUT/"inputs/prior_source_result.json", read(SOURCE/"result.json"))
    save(OUT/"inputs/prior_delivery_receipt.json", read(SOURCE/"delivery_receipt.json"))
    save(OUT/"inputs/scanned_manual_supplement.json", read(SOURCE/"source_evidence/manual_image_supplement.json"))
    save(OUT/"protocol.json", {"at": now(), "study_id": STUDY,
        "scope": "固定32份版本标题候选及7份已有被引用发行人原公告；只拆分角色、验证10条明确引用并保存一条三阶段延期历史。",
        "scope_limit": "没有完成其余公告的独立事件身份、全部字段冲突、原目录外发行人文件或完整M06日历。",
        "source_protocol_sha256": digest(SOURCE/"protocol.json"),
        "prior_outputs": "原分类、字段和ZIP不改写；17份暂缓授予原事件文件只恢复文档角色，不手填可交易数量。",
        "links": LINKS, "schedule": SCHEDULE,
        "time": "继续使用档案日加2自然日的保守研究假设，非首次HTTP时点证明；新版本只影响可见时间之后。",
        "aggregation": "辅助意见不新增供给；更正不与原数量相加；同发行人同上市日不自动合并不同授予计划。",
        "stop": "发生未知批次分配或无确定延期日期时保留未知，不用最后版本覆盖历史或按零供给计算。",
        "T13": "NOT_RUN", "M06_free_float_denominator": "NOT_ESTABLISHED", "new_returns": 0,
        "new_accounts": 0, "goal_achieved": False, "orders_authorized": False})
    shutil.copy2(Path(__file__), OUT/"code"/Path(__file__).name)
    shutil.copy2(ROOT/"tests/test_factor96_supply_version_ledger_v1.py", OUT/"code/test_factor96_supply_version_ledger_v1.py")
    save(OUT/"freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]})
    print("32份版本候选及7份原公告已冻结，未修改原提取结果。", flush=True)


def run():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(OUT/item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(OUT/"code"/Path(__file__).name)
    save(OUT/"run_started.json", {"at": now(), "freeze_sha256": digest(OUT/"freeze.json")})
    documents = read(OUT/"inputs/documents.json")
    source = {r["document_id"]: r for r in documents}
    pages = {key: read(OUT/"inputs/text"/(key+".json")) for key in source}
    fields = {r["document_id"]: r for r in read(OUT/"inputs/prior_fields.json")}
    rows = []
    for item in documents:
        if item["title_kind"] != "REVISION_TITLE_RETAINED_SEPARATELY":
            continue
        key = item["document_id"]
        first = pages[key][0]
        role = document_role(item["title"], first, item["symbol"], key == "1207877835")
        assert not role.startswith("NO_VIEW"), key
        row = {**{k: item[k] for k in ["document_id", "symbol", "archive_date", "title", "title_kind", "source_url", "raw_sha256", "text_sha256"]},
               "document_role": role, "role_evidence_page": 1, "role_evidence": normalized(first)[:1100],
               "prior_field_status": fields[key]["field_status"], "prior_fields_unchanged": True,
               "trading_feature_admitted": False, "independent_event_admitted": False}
        rows.append(row)
    save(OUT/"document_roles.json", rows)
    pd.DataFrame(rows).to_csv(OUT/"32份公告角色台账.csv", index=False, encoding="utf-8-sig")
    links = []
    for updated, original, page, reference, relation, note in LINKS:
        assert source[updated]["symbol"] == source[original]["symbol"]
        assert pd.Timestamp(source[original]["archive_date"]) <= pd.Timestamp(source[updated]["archive_date"])
        text = normalized(pages[updated][page-1])
        needle = normalized(reference)
        assert needle in text, (updated, reference)
        start = text.index(needle)
        links.append({"updated_document_id": updated, "referenced_document_id": original,
                      "symbol": source[updated]["symbol"], "page": page, "literal_reference": needle,
                      "context": text[max(0,start-130):start+len(needle)+150], "relation": relation, "note": note,
                      "counts_as_additional_supply": False, "historical_original_rewritten": False})
    save(OUT/"explicit_version_links.json", links)
    schedule = []
    for row in SCHEDULE:
        assert row["archive_date"] == source[row["document_id"]]["archive_date"]
        combined = normalized("".join(pages[row["document_id"]]))
        assert "2,249,297,094" in combined
        if row["scheduled_unlock_date"]:
            date = pd.Timestamp(row["scheduled_unlock_date"])
            assert f"{date.year}年{date.month}月{date.day}日" in combined
        else:
            assert "待后续基础控股完成业绩承诺补偿后" in combined
        schedule.append({**row, "event_identity": "600515.SH_2016-07-26_海航基础控股发行股份购买资产批次",
                         "nominal_shares": "2249297094", "actual_sales_known": False, "known_clock_assumption": True})
    save(OUT/"hainan_schedule_versions.json", schedule)
    queries = ["2019-07-25", "2019-07-26", "2020-01-24", "2020-01-25", "2021-01-10", "2021-01-11"]
    timeline = []
    for day in queries:
        at = day+"T15:05:00+08:00"
        visible = visible_schedule(schedule, at)
        timeline.append({"decision_time": at, "visible_document": visible["document_id"] if visible else None,
                         "scheduled_unlock_date": visible["scheduled_unlock_date"] if visible else None,
                         "state": visible["state"] if visible else "NO_VIEW_BEFORE_FIRST_SOURCE"})
    save(OUT/"asof_schedule_checks.json", timeline)
    pd.DataFrame(timeline).to_csv(OUT/"延期版本按当时可见时间.csv", index=False, encoding="utf-8-sig")
    save(OUT/"same_issuer_same_day_distinct_plans.json", {"documents": ["1215755330", "1215755334"], "symbol": "002410.SZ",
        "announced_listing_date": "2023-02-08", "separate_plan_years": [2020, 2021], "nominal_quantity_in_first_page": ["233400", "88000"],
        "same_date_is_not_identity": True, "automatic_merge_performed": False,
        "note": "广联达2020年计划第二期与2021年计划第一期，同发行人同上市日仍是不同计划；所列数量为首页名义解禁量，不补入自动实际可流通字段。"})
    result = {"at": now(), "study_id": STUDY, "status": "FIXED_VERSION_ROLES_AND_EXPLICIT_LINKS_COMPLETE_FULL_EVENT_LEDGER_PENDING",
        "role_documents": len(rows), "original_documents_referenced": 7,
        "role_counts": dict(Counter(row["document_role"] for row in rows)), "explicit_links": len(links),
        "schedule_versions": len(schedule), "asof_examples": len(timeline), "new_accounts": 0, "new_returns": 0,
        "all_events_deduplicated": False, "full_M06_calendar_established": False, "M06_free_float_denominator_established": False,
        "T13": "NOT_RUN", "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED"}
    save(OUT/"result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    arguments = parser.parse_args()
    {"prepare": prepare, "run": run}[arguments.action]()
