"""提取T13原文明示日期和数量候选；不猜表格、不合并版本、不生成交易信号。"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd

from research.factor96_known_supply_sources_v1 import OUT as SOURCE, digest, now, read, save

ROOT = Path(__file__).resolve().parents[1]
OUT = SOURCE/"event_fields_v1"
STUDY = "510300_FACTOR96_KNOWN_SUPPLY_EVENT_FIELDS_V1"
DATE = r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
QUANTITY = r"(?P<number>\d[\d,]*(?:\.\d+)?)(?P<unit>亿股|万股|股)"
DATE_PATTERNS = [
    r"(?:本次|此次)[^。；;\d]{0,35}?(?:上市流通|流通上市)(?:的)?(?:日期|时间|日)(?:均)?(?:为|是|将为)?[:：]?"+DATE,
    r"(?:本次|此次)[^。；;\d]{0,10}?(?:解除限售|限售股份|限售股)[^。；;\d]{0,16}?(?:拟上市|上市)日期(?:为|是)?[:：]?"+DATE,
    r"(?:本次|此次)[^。；;\d]{0,35}?(?:将于|定于|拟于)"+DATE+r"(?:\([^)]{0,12}\))?(?:起)?(?:可)?上市流通",
]
QUANTITY_PATTERNS = {
    "ACTUAL_TRADABLE": [r"(?:实际(?:可)?|其中(?:实际)?可)(?:上市流通|流通上市|流通)(?:的)?(?:股份)?(?:数量|股数|总数|总额)?(?:为|是|合计为|共计|合计|共)?[:：]?"+QUANTITY],
    "ANNOUNCED_LISTING": [r"(?:本次|此次)[^。；;\d]{0,24}?(?:上市流通|流通上市)[^。；;\d]{0,22}?"+QUANTITY],
    "NOMINAL_UNLOCK": [r"(?:本次|此次)[^。；;\d]{0,15}?解除限售(?:(?!实际|可上市|上市流通|[。；;\d]).){0,20}?"+QUANTITY],
}


def normalize(page):
    """保留页边界，不能把页码或下一页数字拼进一个数量。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", page))


def share_value(number, unit):
    value = Decimal(number.replace(",", ""))
    scale = {"股": Decimal(1), "万股": Decimal(10000), "亿股": Decimal(100000000)}[unit]
    resolution = Decimal(10)**value.as_tuple().exponent*scale
    return str(value*scale), str(resolution)


def consistent_quantity(candidates):
    if not candidates:
        return None, "MISSING"
    # 严格同值才归一，不把约数相近当精确份额相等。
    values = {Decimal(x["shares_decimal"]) for x in candidates}
    if len(values) != 1:
        return None, "CONFLICTING_EXPLICIT_VALUES"
    item = min(candidates, key=lambda x: Decimal(x["resolution_shares_decimal"]))
    return {"shares_decimal": item["shares_decimal"], "resolution_shares_decimal": item["resolution_shares_decimal"]}, "UNIQUE_EXPLICIT_VALUE"


def extract_fields(pages, archive_date, title, title_kind, symbol):
    normalized = [normalize(page) for page in pages]
    dates, quantities = [], {kind: [] for kind in QUANTITY_PATTERNS}
    for page_no, text in enumerate(normalized, 1):
        seen_dates, seen_quantities = set(), set()
        for pattern in DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    date = pd.Timestamp(int(match["year"]), int(match["month"]), int(match["day"]))
                except ValueError:
                    continue
                key = (match.start(), date)
                if key in seen_dates:
                    continue
                seen_dates.add(key)
                dates.append({"page": page_no, "date": date.date().isoformat(), "context": text[max(0, match.start()-35):match.end()+45]})
        for kind, patterns in QUANTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    key = (kind, match.start(), match.end())
                    if key in seen_quantities:
                        continue
                    seen_quantities.add(key)
                    value, resolution = share_value(match["number"], match["unit"])
                    quantities[kind].append({"page": page_no, "shares_decimal": value, "resolution_shares_decimal": resolution,
                        "reported_number": match["number"], "reported_unit": match["unit"],
                        "context": text[max(0, match.start()-35):match.end()+50]})
    unique_dates = sorted({d["date"] for d in dates})
    values, states = {}, {}
    for kind, candidates in quantities.items():
        values[kind], states[kind] = consistent_quantity(candidates)
    has_actual_qualifier = any(re.search(r"(?:实际(?:可)?|其中(?:实际)?可)(?:上市流通|流通上市|流通)", p) for p in normalized)
    chosen_kind = "ACTUAL_TRADABLE" if has_actual_qualifier else "ANNOUNCED_LISTING"
    quantity = values[chosen_kind]
    date = unique_dates[0] if len(unique_dates) == 1 else None
    signature = []
    tail = normalized[-1][-600:] if normalized else ""
    for match in re.finditer(DATE, tail):
        before = tail[max(0, match.start()-90):match.start()]
        if re.search(r"(?:股份有限公司|有限公司董事会|公司董事会|董事会)$", before):
            try:
                value = pd.Timestamp(int(match["year"]), int(match["month"]), int(match["day"]))
                signature.append(value.date().isoformat())
            except ValueError:
                continue
    signature = sorted(set(signature))
    archive = pd.Timestamp(archive_date).normalize()
    signed = pd.Timestamp(signature[0]) if len(signature) == 1 else None
    clock_base = max(archive, signed) if signed is not None else archive
    known_at = (clock_base+pd.Timedelta(days=2)).tz_localize("Asia/Shanghai")
    status = "CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW"
    if title_kind == "REVISION_TITLE_RETAINED_SEPARATELY":
        status = "VERSION_CANDIDATE_NOT_NEW_EVENT"
    elif any(w in title for w in ["核查", "法律意见", "独立财务顾问"]):
        status = "SUPPORTING_OPINION_NOT_NEW_EVENT"
    elif len(unique_dates) != 1:
        status = "NO_VIEW_LISTING_DATE_MISSING_OR_CONFLICT"
    elif quantity is None:
        status = "NO_VIEW_"+chosen_kind+"_"+states[chosen_kind]
    elif not normalized or symbol[:6] not in normalized[0]:
        status = "NO_VIEW_FIRST_PAGE_SECURITY_ID_UNCONFIRMED"
    elif signed is not None and abs((signed-archive).days) > 30:
        status = "NO_VIEW_SIGNATURE_ARCHIVE_DATE_CONFLICT"
    return {"field_status": status, "scheduled_listing_date": date, "listing_date_candidates": dates,
            "quantity_candidates": quantities, "quantity_states": states, "quantity_values": values,
            "selected_quantity_kind": chosen_kind, "selected_quantity": quantity,
            "actual_tradable_qualifier_present": has_actual_qualifier,
            "signature_date_candidates": signature, "archive_date": archive.date().isoformat(),
            "conservative_known_at": known_at.isoformat(), "clock_is_historical_assumption": True,
            "known_before_scheduled_day": bool(date is not None and known_at.tz_localize(None) < pd.Timestamp(date)),
            "event_identity_confirmed": False, "correction_chain_resolved": False,
            "actual_sales_observed": False, "trading_feature_admitted": False}


def freeze():
    assert not (OUT/"protocol.json").exists()
    tests = read(OUT/"prefreeze_test_receipt.json")
    assert tests["exit_code"] == 0
    (OUT/"code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), OUT/"code"/Path(__file__).name)
    shutil.copy2(ROOT/"tests/test_factor96_known_supply_event_fields_v1.py", OUT/"code/test_factor96_known_supply_event_fields_v1.py")
    protocol = {"at": now(), "study_id": STUDY, "source_protocol_sha256": digest(SOURCE/"protocol.json"),
        "scope": "逐文提取原文明确表述的上市流通日期、实际可流通/拟上市数量、名义解禁数量及单位精度。只是字段候选，不是完整供给预测或T13交易信号。",
        "date": "只读明确属于本次/此次解禁的上市流通日期、时间、拟上市日期或本次将于上市语句直接相连的年月日；排除发行背景和其他锁定批次；多个本次值保留冲突，不按与公告最近择优。",
        "quantity": "全文存在实际可流通或其中可流通措辞时，优先且必须有该口径明确数值；缺少时不以名义解禁值填入。无实际措辞才保留公告上市流通数量。名义量另存且不混入实际量。",
        "precision": "用Decimal从原数值和股/万股/亿股直接换算，保存报告精度；不同明确值不四舍五入合并。",
        "pages": "每页单独正规化和匹配，禁止跨页拼接数字。表格未可靠解析时留未知，不按最大数字或最后合计猜测。",
        "versions": "更正/补充/取消候选始终不作为新事件；普通文档也暂不宣称事件身份和版本链已完成。",
        "clock": "分别保存档案日、最后页明示署期和经济上市日；保守可用时间为档案日与已识别署期较晚者加2自然日00:00。这是历史假设，未证明首次送达。",
        "source_scope": "2015至2025年历史成员并集只决定下载范围；候选后续还需在决策日按当时成员取交集。未建立完整M06发行/缴款日历。",
        "no_backtest": True, "new_returns_loaded": False, "new_accounts": 0, "new_models": 0,
        "goal_achieved": False, "goal_status": "active", "orders_authorized": False,
        "code_sha256": digest(Path(__file__)), "prefreeze_tests_sha256": digest(OUT/"prefreeze_test_receipt.json")}
    save(OUT/"protocol.json", protocol, True)
    save(OUT/"freeze.json", {"at": now(), "before_field_batch_and_new_outcomes": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print("T13字段提取合同已固定，不将字段候选当作已准入交易事件。", flush=True)


def run():
    assert digest(Path(__file__)) == read(OUT/"protocol.json")["code_sha256"]
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(OUT/item["path"]) == item["sha256"]
    source = read(SOURCE/"documents.json")
    assert len(source) == read(SOURCE/"protocol.json")["target_documents"]
    save(OUT/"run_started.json", {"at": now(), "source_documents_sha256": digest(SOURCE/"documents.json"),
                                   "freeze_sha256": digest(OUT/"freeze.json")}, True)
    rows = []
    for item in source:
        row = {k: item[k] for k in ["document_id", "symbol", "title", "title_kind", "archive_date", "source_url", "status"]}
        if item["status"] != "PDF_TEXT_SAVED":
            row.update(field_status="NO_VIEW_SOURCE_TEXT", trading_feature_admitted=False)
        else:
            assert digest(SOURCE/item["text_path"]) == item["text_sha256"]
            row.update(raw_sha256=item["raw_sha256"], text_sha256=item["text_sha256"],
                       **extract_fields(read(SOURCE/item["text_path"]), item["archive_date"], item["title"], item["title_kind"], item["symbol"]))
        rows.append(row)
    save(OUT/"document_fields.json", rows, True)
    flat = []
    for row in rows:
        number = row.get("selected_quantity")
        flat.append({k: row.get(k) for k in ["document_id", "symbol", "title", "field_status", "archive_date", "scheduled_listing_date",
            "conservative_known_at", "known_before_scheduled_day", "selected_quantity_kind"]} |
            {"shares_decimal": number["shares_decimal"] if number else None,
             "resolution_shares_decimal": number["resolution_shares_decimal"] if number else None,
             "trading_feature_admitted": False})
    table = pd.DataFrame(flat)
    table.to_parquet(OUT/"event_field_candidates.parquet", index=False)
    table.to_csv(OUT/"公告日期与数量候选.csv", index=False, encoding="utf-8-sig")
    candidates = table[table.field_status.eq("CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW")]
    result = {"at": now(), "study_id": STUDY, "documents": len(rows), "field_status_counts": table.field_status.value_counts().to_dict(),
              "explicit_field_candidates": len(candidates), "candidate_companies": candidates.symbol.nunique(),
              "known_before_scheduled_day_candidates": int(candidates.known_before_scheduled_day.fillna(False).sum()),
              "all_event_identities_confirmed": False, "all_correction_chains_resolved": False, "full_M06_calendar_established": False,
              "admitted_trading_features": 0, "new_accounts": 0, "new_models": 0, "new_return_labels": 0,
              "independent_forward_observations": 0, "goal_status": "active", "goal_achieved": False, "orders_authorized": False}
    save(OUT/"result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
