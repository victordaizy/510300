"""只读核对T13目录选择、原PDF和字段候选；不生成账户或新增随机样本。"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def verify_freeze(root):
    frozen, started = read(root/"freeze.json"), read(root/"run_started.json")
    assert digest(root/"freeze.json") == started["freeze_sha256"]
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(started["at"])
    for row in frozen["files"]:
        assert digest(root/row["path"]) == row["sha256"], row["path"]
    return len(frozen["files"])


def verify_selection(root):
    catalogue = pd.read_parquet(root/"inputs/announcement_catalogue.parquet")
    membership = pd.read_parquet(root/"inputs/membership.parquet")
    dates = pd.to_datetime(membership.membership_date)
    symbols = set(membership.loc[dates.between("2015-01-01", "2025-12-31"), "symbol"])
    codes = {s[:6] for s in symbols}
    archive = pd.to_datetime(catalogue.announcement_date)
    eligible = catalogue.secCode.isin(codes) & archive.between("2015-01-01", "2025-12-31")
    eligible &= catalogue.title_kind.isin(["ELIGIBLE_ORIGINAL_DISCLOSURE", "REVISION_TITLE_RETAINED_SEPARATELY"])
    expected = catalogue.loc[eligible].copy()
    targets = pd.read_parquet(root/"targets.parquet")
    assert expected.announcementId.is_unique and targets.document_id.is_unique
    assert set(expected.announcementId.astype(str)) == set(targets.document_id)
    assert len(targets) == read(root/"protocol.json")["target_documents"] == 2192
    assert targets.symbol.nunique() == 526
    assert targets.title_kind.value_counts().to_dict() == {"ELIGIBLE_ORIGINAL_DISCLOSURE": 2160, "REVISION_TITLE_RETAINED_SEPARATELY": 32}
    originals = expected.set_index(expected.announcementId.astype(str))
    for row in targets.itertuples():
        prior = originals.loc[row.document_id]
        for col in catalogue.columns:
            assert str(getattr(row, col)) == str(prior[col]), (row.document_id, col)
        assert row.symbol in symbols and row.symbol[:6] == str(prior.secCode)
        assert row.source_url == "https://static.cninfo.com.cn/"+prior.adjunctUrl
    provenance = read(root/"source_evidence/catalogue_provenance.json")
    raw = {}
    for item in provenance:
        path = root/item["snapshot"]
        assert digest(path) == item["sha256"]
        raw[item["original"]] = {str(row["announcementId"]): row for row in read(path)["announcements"]}
    for row in targets.itertuples():
        original = raw[row.raw_response_path][row.document_id]
        for col in ["secCode", "announcementId", "announcementTitle", "announcementTime", "adjunctUrl"]:
            assert str(original[col]) == str(getattr(row, col)), (row.document_id, col)
        day = pd.to_datetime(original["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai").date()
        assert day == pd.Timestamp(row.archive_date).date()
    return targets, len(provenance)


def verify_documents(root, targets):
    documents = read(root/"documents.json")
    assert len(documents) == len(targets) == len({d["document_id"] for d in documents})
    by_id = targets.set_index("document_id")
    pages, raw_bytes = {}, 0
    for ordinal, item in enumerate(documents, 1):
        key = item["document_id"]
        target = by_id.loc[key]
        assert item == read(root/"documents"/(key+".json"))
        assert item["source_url"] == target.source_url and item["symbol"] == target.symbol
        assert item["title_kind"] == target.title_kind and item["title"] == target.clean_title
        assert pd.Timestamp(item["archive_date"]) == target.archive_date
        if item.get("raw_path"):
            raw = root/item["raw_path"]
            assert digest(raw) == item["raw_sha256"]
            raw_bytes += raw.stat().st_size
            if item["status"] in ["PDF_TEXT_SAVED", "PDF_NO_USABLE_TEXT"]:
                with raw.open("rb") as stream:
                    assert stream.read(4) == b"%PDF"
        if item["reused"]:
            prior = next(x for x in read(root/"source_evidence/old_four_pdf_cases.json")["facts"] if x["announcementId"] == key)
            assert item["raw_sha256"] == prior["source_pdf_sha256"]
        elif item.get("receipt_path"):
            receipt = read(root/item["receipt_path"])
            assert receipt["document_id"] == key and receipt["source_url"] == item["source_url"]
            assert pd.Timestamp(receipt["at"]) <= pd.Timestamp(receipt["completed_at"])
            if item["status"] in ["PDF_TEXT_SAVED", "PDF_NO_USABLE_TEXT"]:
                assert receipt["http_status"] == 200 and receipt["status"] == "PDF_DOWNLOADED"
                assert receipt["sha256"] == item["raw_sha256"] and receipt["bytes"] == (root/item["raw_path"]).stat().st_size
        if item.get("text_path"):
            text_path = root/item["text_path"]
            assert digest(text_path) == item["text_sha256"]
            raw_pages = read(text_path)
            assert len(raw_pages) == item["pages"] and sum(map(len, raw_pages)) == item["text_characters"]
            assert bool(item["symbol"][:6] in raw_pages[0]) == item["first_page_security_code_seen"]
            if item["status"] == "PDF_TEXT_SAVED":
                assert len("".join(raw_pages).strip()) >= 50
                pages[key] = [normalized(page) for page in raw_pages]
            else:
                assert item["status"] == "PDF_NO_USABLE_TEXT" and len("".join(raw_pages).strip()) < 50
        if ordinal % 500 == 0:
            print(f"已核对{ordinal}份原PDF、逐页文本和下载回执。", flush=True)
    result = read(root/"result.json")
    assert dict(Counter(d["status"] for d in documents)) == result["statuses"]
    assert result["documents"] == len(documents) and result["complete_texts"] == len(pages)
    assert result["companies"] == len({x["symbol"] for x in documents})
    assert result["reused_documents"] == sum(x["reused"] for x in documents)
    assert result["new_accounts"] == result["new_models"] == result["new_return_labels"] == 0
    assert result["goal_achieved"] is False and result["orders_authorized"] is False
    return documents, pages, raw_bytes


def verify_fields(root, documents, pages):
    folder = root/"event_fields_v1"
    freeze_files = verify_freeze(folder)
    records = read(folder/"document_fields.json")
    source = {row["document_id"]: row for row in documents}
    assert len(records) == len(source) == len({x["document_id"] for x in records})
    assert read(folder/"run_started.json")["source_documents_sha256"] == digest(root/"documents.json")
    contexts, confirmed = 0, 0
    table = pd.read_parquet(folder/"event_field_candidates.parquet").set_index("document_id")
    for row in records:
        key = row["document_id"]
        assert not row["trading_feature_admitted"]
        assert row["field_status"] == table.loc[key, "field_status"]
        if key not in pages:
            assert row["field_status"] == "NO_VIEW_SOURCE_TEXT"
            continue
        assert row["raw_sha256"] == source[key]["raw_sha256"] and row["text_sha256"] == source[key]["text_sha256"]
        assert not row["event_identity_confirmed"] and not row["correction_chain_resolved"] and not row["actual_sales_observed"]
        values = sorted({item["date"] for item in row["listing_date_candidates"]})
        assert row["scheduled_listing_date"] == (values[0] if len(values) == 1 else None)
        for item in row["listing_date_candidates"]:
            context = item["context"]
            assert context in pages[key][item["page"]-1]
            stamp = pd.Timestamp(item["date"])
            pattern = rf"{stamp.year}年0?{stamp.month}月0?{stamp.day}日"
            assert re.search(pattern, context), key
            contexts += 1
        for kind, candidates in row["quantity_candidates"].items():
            numbers = set()
            for item in candidates:
                assert item["context"] in pages[key][item["page"]-1]
                assert item["reported_number"]+item["reported_unit"] in item["context"]
                multiplier = {"股": Decimal(1), "万股": Decimal(10000), "亿股": Decimal(100000000)}[item["reported_unit"]]
                number = Decimal(item["reported_number"].replace(",", ""))
                shares = number*multiplier
                resolution = Decimal(10)**number.as_tuple().exponent*multiplier
                assert shares == Decimal(item["shares_decimal"])
                assert resolution == Decimal(item["resolution_shares_decimal"])
                numbers.add(shares)
                contexts += 1
            expected_state = "MISSING" if not candidates else ("UNIQUE_EXPLICIT_VALUE" if len(numbers) == 1 else "CONFLICTING_EXPLICIT_VALUES")
            assert row["quantity_states"][kind] == expected_state
            saved = row["quantity_values"][kind]
            if len(numbers) != 1:
                assert saved is None
            else:
                assert Decimal(saved["shares_decimal"]) == next(iter(numbers))
                assert Decimal(saved["resolution_shares_decimal"]) == min(Decimal(x["resolution_shares_decimal"]) for x in candidates)
        selected_kind = "ACTUAL_TRADABLE" if row["actual_tradable_qualifier_present"] else "ANNOUNCED_LISTING"
        actual_present = any(re.search(r"(?:实际(?:可)?|其中(?:实际)?可)(?:上市流通|流通上市|流通)", p) for p in pages[key])
        assert row["actual_tradable_qualifier_present"] == actual_present
        assert row["selected_quantity_kind"] == selected_kind
        assert row["selected_quantity"] == row["quantity_values"][selected_kind]
        archive = pd.Timestamp(row["archive_date"])
        signatures = row["signature_date_candidates"]
        clock_base = max(archive, pd.Timestamp(signatures[0])) if len(signatures) == 1 else archive
        expected_clock = (clock_base+pd.Timedelta(days=2)).tz_localize("Asia/Shanghai")
        assert pd.Timestamp(row["conservative_known_at"]) == expected_clock
        expected_before = row["scheduled_listing_date"] is not None and expected_clock.tz_localize(None) < pd.Timestamp(row["scheduled_listing_date"])
        assert row["known_before_scheduled_day"] == expected_before
        if source[key]["title_kind"] == "REVISION_TITLE_RETAINED_SEPARATELY":
            assert row["field_status"] == "VERSION_CANDIDATE_NOT_NEW_EVENT"
        if row["field_status"] == "CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW":
            assert len(values) == 1 and row["selected_quantity"] is not None
            assert row["symbol"][:6] in pages[key][0]
            assert source[key]["title_kind"] == "ELIGIBLE_ORIGINAL_DISCLOSURE"
            assert table.loc[key, "shares_decimal"] == row["selected_quantity"]["shares_decimal"]
            confirmed += 1
    result = read(folder/"result.json")
    assert dict(Counter(r["field_status"] for r in records)) == result["field_status_counts"]
    assert result["explicit_field_candidates"] == confirmed
    selected = table[table.field_status.eq("CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW")]
    assert result["candidate_companies"] == selected.symbol.nunique()
    assert result["known_before_scheduled_day_candidates"] == int(selected.known_before_scheduled_day.fillna(False).sum())
    assert not table.trading_feature_admitted.any()
    assert result["admitted_trading_features"] == result["new_accounts"] == result["new_models"] == 0
    assert not result["goal_achieved"] and not result["full_M06_calendar_established"]
    return {"field_records": len(records), "explicit_field_candidates": confirmed, "source_contexts_checked": contexts,
            "field_freeze_files": freeze_files}


def verify(root):
    freeze_files = verify_freeze(root)
    targets, catalogue_raw_n = verify_selection(root)
    documents, pages, raw_bytes = verify_documents(root, targets)
    fields = verify_fields(root, documents, pages)
    launch = read(root/"launch_failure_01.json")
    assert launch["new_requests"] == launch["new_accounts"] == 0
    return {"status": "PASS_SAVED_SUPPLY_SOURCE_AND_FIELD_RECOMPUTATION", "source_freeze_files": freeze_files,
            "target_documents": len(targets), "catalogue_raw_responses": catalogue_raw_n,
            "complete_pdf_texts": len(pages), "raw_pdf_bytes": raw_bytes, **fields,
            "new_accounts": 0, "new_models": 0, "new_downloads": 0, "network_requests": 0,
            "external_review": "NOT_PERFORMED", "independent_forward_validation": False,
            "scope": "核对固定选取、原始目录、PDF及文本哈希、字段上下文与单位和时钟；不证明字段提取无遗漏、所有语义正确、事件身份或更正链完整，不构成T13策略验证。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
