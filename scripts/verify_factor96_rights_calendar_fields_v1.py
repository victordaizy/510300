"""独立核对配股原文闭合、条款定位、整数金额与公告时序；只读不联网。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def verify(root):
    source = root.parent / "510300_factor96_rights_issue_documents_v1"
    for folder in [source, root]:
        frozen, started = read(folder / "freeze.json"), read(folder / "run_started.json")
        assert pd.Timestamp(frozen["at"]) < pd.Timestamp(started["at"])
        assert digest(folder / "freeze.json") == started["freeze_sha256"]
        for item in frozen["files"]:
            location = source if item.get("scope") == "source" else folder
            assert digest(location / item["path"]) == item["sha256"], item
    targets = read(source / "targets.json")
    original = read(source / "documents.json")
    documents = read(root / "inputs/documents.json")
    assert len(targets) == len(original) == len(documents) == 491
    catalogue = pd.read_parquet(source / "inputs/prior_catalogue.parquet")
    selected = catalogue[catalogue.title.str.contains("配股") & catalogue.title_role.isin(
        ["ISSUER_CALENDAR_DOCUMENT_CANDIDATE", "AMENDMENT_CANDIDATE", "TERMINATION_OR_REJECTION_CANDIDATE"])]
    assert set(selected.document_id) == {r["document_id"] for r in targets}
    states = read(source / "inputs/prior_query_states.json")
    assert len([s for s in states if s["query"] == "rights_title" and s["complete"]]) == 329
    completion = source / "transport_completion_v1"
    protocol = read(completion / "protocol.json")
    assert protocol["original_manifest_sha256"] == digest(source / "documents.json")
    assert protocol["original_result_sha256"] == digest(source / "result.json")
    assert documents == read(completion / "effective_documents.json")
    old_by_id, new_by_id = {r["document_id"]: r for r in original}, {r["document_id"]: r for r in documents}
    assert sum(r["status"] == "PDF_TEXT_SAVED" for r in original) == 490
    assert sum(r["status"] == "PDF_TEXT_SAVED" for r in documents) == 491
    for key, row in old_by_id.items():
        if key != "1212032708":
            assert row == new_by_id[key]
    for attempt in [1, 2]:
        failed = read(source / "receipts" / f"1212032708_a{attempt}.json")
        assert failed["status"] == "REQUEST_FAILED" and failed["error_type"] == "SSLError"
    pages, total_pages, total_chars, total_bytes = {}, 0, 0, 0
    for row in documents:
        raw, text_file = source / row["raw_snapshot"], source / row["text_snapshot"]
        assert digest(raw) == row["raw_sha256"] and raw.stat().st_size == row["bytes"]
        assert digest(text_file) == row["text_sha256"]
        with raw.open("rb") as stream:
            assert stream.read(5) == b"%PDF-"
        receipt = read(source / row["receipt_snapshot"])
        prefix = "transport_completion_v1/" if row["raw_snapshot"].startswith("transport_completion_v1/") else ""
        assert prefix + receipt["raw_path"] == row["raw_snapshot"]
        assert receipt["status"] == "HTTP_OK" and receipt["http_status"] == 200
        assert receipt["url"] == row["source_url"] and receipt["sha256"] == row["raw_sha256"]
        text = read(text_file)["pages"]
        assert len(text) == row["pages"] and [p["page"] for p in text] == list(range(1, len(text) + 1))
        assert sum(len(p["text"]) for p in text) == row["text_characters"]
        for page in text:
            pages[(row["document_id"], page["page"])] = normalized(page["text"])
        total_pages += len(text)
        total_chars += row["text_characters"]
        total_bytes += row["bytes"]
    candidates = read(root / "clause_candidates.json")
    cue_patterns = {
        "PAYMENT_OR_CLEARING": r"配股缴款|认购缴款|缴款起止", "LISTING": r"新增股份上市|新增股票上市|获配股票上市|上市时间|上市流通日",
        "RECORD_DATE": r"股权登记日", "PRICE": r"配股价格|发行价格", "ACTUAL_SUBSCRIPTION": r"有效认购股份|有效认购资金|有效认购数量|认购金额",
        "PROCEEDS": r"募集资金", "REVISION_OR_UNKNOWN": r"现更正为|公告原文为|另行公告|终止本次配股|终止公司配股"}
    expected_locations = Counter()
    for (key, page_number), text in pages.items():
        for kind, pattern in cue_patterns.items():
            for hit in re.finditer(pattern, text):
                expected_locations[(key, page_number, kind, max(0, hit.start() - 65), min(len(text), hit.end() + 220))] += 1
    actual_locations = Counter()
    for row in candidates:
        key = (row["document_id"], row["page"])
        text = pages[key]
        start, end = row["normalized_start"], row["normalized_end"]
        evidence = row["evidence"]
        assert text[start:end] == evidence and row["event_field_admitted"] is False
        assert re.fullmatch(cue_patterns[row["cue_kind"]], row["cue"])
        actual_locations[(key[0], key[1], row["cue_kind"], start, end)] += 1
        for date in row["date_literals"]:
            assert evidence[date["start"]:date["end"]] == date["literal"]
            numbers = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date["literal"])
            try:
                expected = datetime(*map(int, numbers.groups())).date().isoformat()
            except ValueError:
                expected = None
            assert date["value"] == expected
        for field in ["money_literals", "share_literals"]:
            for number in row[field]:
                assert evidence[number["start"]:number["end"]] == number["literal"]
                numeric = number["literal"][:-len(number["unit"])].replace(",", "")
                scale = Decimal(100000000) if number["unit"].startswith("亿") else Decimal(10000) if number["unit"].startswith("万") else Decimal(1)
                assert Decimal(number["normalized_number"]) == Decimal(numeric) * scale
    assert actual_locations == expected_locations
    evidence = read(root / "case_evidence.json")
    assert evidence == read(root / "inputs/case_evidence_specifications.json")
    by_field = {}
    for row in evidence:
        text = pages[(row["document_id"], row["page"])]
        assert text[row["normalized_start"]:row["normalized_end"]] == row["evidence"]
        assert re.fullmatch(row["pattern"], row["evidence"])
        by_field[row["field"]] = row
    def last_number(field):
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", by_field[field]["evidence"])
        return Decimal(numbers[-1].replace(",", ""))
    price = int(last_number("ISSUE_PRICE") * 100)
    old_shares, new_shares = int(last_number("INITIAL_SHARES")), int(last_number("CORRECTED_SHARES"))
    old_cents, new_cents = int(last_number("INITIAL_MONEY") * 100), int(last_number("CORRECTED_MONEY") * 100)
    assert old_shares * price == old_cents and new_shares * price == new_cents
    assert by_field["CORRECTED_SHARES"]["normalized_start"] > by_field["CORRECTION_BOUNDARY"]["normalized_start"]
    assert by_field["CORRECTED_MONEY"]["normalized_start"] > by_field["CORRECTION_BOUNDARY"]["normalized_start"]
    versions = read(root / "case_versions.json")
    assert versions == read(root / "inputs/case_versions_specifications.json")
    assert [v["document_id"] for v in versions] == ["1201854885", "1201897631", "1201900999", "1201908424"]
    assert "临2016-003" in pages[("1201897631", 1)] and "临2016-003" in by_field["EXPLICIT_OLD_DOCUMENT_REFERENCE"]["evidence"]
    assert versions[0]["updates"]["plan_max_shares"] == int(last_number("MAX_SHARES"))
    assert versions[0]["updates"]["issue_price_cents"] == price
    assert versions[0]["updates"]["plan_shares_times_price_cents"] == int(last_number("MAX_SHARES")) * price
    assert versions[0]["updates"]["stated_proceeds_budget_cap_cents"] == int(last_number("PROCEEDS_BUDGET_CAP")) * 100000000 * 100
    assert versions[1]["updates"]["actual_subscribed_shares"] == old_shares and versions[1]["updates"]["actual_subscription_cents"] == old_cents
    assert versions[2]["updates"]["actual_subscribed_shares"] == new_shares and versions[2]["updates"]["actual_subscription_cents"] == new_cents
    assert versions[3]["updates"]["listing_announced_shares"] == int(last_number("LISTING_SHARES")) == new_shares
    expected_dates = {"record_date": ("RECORD_DATE", ["2015-12-28"]), "payment_start": ("PAYMENT_WINDOW", ["2015-12-29", "2016-01-05"]),
        "scheduled_clearing_date": ("CLEARING_DATE", ["2016-01-06"]), "scheduled_exrights_resumption_date": ("EXRIGHTS_RESUMPTION", ["2016-01-07"])}
    for key, (field, expected) in expected_dates.items():
        actual_dates = [datetime(*map(int, match)).date().isoformat() for match in re.findall(r"(\d{4})年(\d+)月(\d+)日", by_field[field]["evidence"])]
        assert actual_dates == expected and versions[0]["updates"][key] == expected[0]
    assert versions[0]["updates"]["payment_end"] == "2016-01-05"
    assert versions[0]["updates"]["scheduled_result_publication_date"] is None
    assert versions[0]["updates"]["result_date_conflict"] == ["2015-01-07", "2016-01-07"]
    assert "2015年1月7日" in by_field["CONFLICTING_RESULT_YEAR"]["evidence"]
    assert versions[3]["updates"]["announced_listing_date"] == "2016-01-18" and "2016年1月18日" in by_field["LISTING_DATE"]["evidence"]
    snapshots = read(root / "case_asof_snapshots.json")
    assert len(snapshots) == 8
    for version in versions:
        assert version["known_at"] == new_by_id[version["document_id"]]["catalogue_date"] + "T23:59:59+08:00"
    for snapshot in snapshots:
        eligible = [v for v in versions if pd.Timestamp(v["known_at"]) <= pd.Timestamp(snapshot["as_of"])]
        assert snapshot["source_documents"] == [v["document_id"] for v in eligible]
        if not eligible:
            assert snapshot["status"] == "NO_VIEW"
            continue
        expected = {"actual_subscribed_shares": None, "actual_subscription_cents": None, "announced_listing_date": None}
        for version in eligible:
            expected.update(version["updates"])
        for key, value in expected.items():
            assert snapshot[key] == value
        day = pd.Timestamp(snapshot["as_of"]).date().isoformat()
        phase = "ANNOUNCED_WINDOW_NOT_STARTED" if day < expected["payment_start"] else "ANNOUNCED_PAYMENT_WINDOW_DATE" if day <= expected["payment_end"] else "ANNOUNCED_PAYMENT_WINDOW_ENDED"
        assert snapshot["payment_phase"] == phase and snapshot["trading_feature_admitted"] is False
    result = read(root / "result.json")
    assert result["clause_candidates"] == len(candidates) and result["case_evidence_anchors"] == len(evidence) == 19
    assert result["documents_with_candidates"] == len({r["document_id"] for r in candidates})
    assert result["correction_delta_shares"] == new_shares - old_shares == 5769
    assert result["correction_delta_cents"] == new_cents - old_cents == 4724811
    assert result["new_accounts"] == result["new_returns"] == result["new_network_requests"] == 0
    return {"status": "PASS_SAVED_RIGHTS_ORIGINALS_CLAUSE_ANCHORS_AND_VERSION_CLOCK", "pdf_text_documents": len(documents),
        "pdf_bytes": total_bytes, "text_pages": total_pages, "text_characters": total_chars,
        "clause_candidates": len(candidates), "case_documents": 4, "case_anchors": 19, "asof_snapshots": 8,
        "new_accounts": 0, "network_requests": 0, "external_review": "NOT_PERFORMED",
        "scope": "核对冻结原PDF和文本、原文候选定位、单个案例数值与时钟；不代表全部日历已结构化或策略有效。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    output = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
