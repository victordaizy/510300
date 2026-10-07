"""从冻结逐页文本独立核对用途勾选、已披露实施比例和保守时钟。"""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify(root):
    frozen, start = read(root / "freeze.json"), read(root / "run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(start["at"])
    assert digest(root / "freeze.json") == start["freeze_sha256"]
    for item in frozen["files"]:
        assert digest(root / item["path"]) == item["sha256"], item["path"]
    source = {r["document_id"]: r for r in read(root / "inputs/documents.json")}
    events = {r["document_id"]: r for r in read(root / "inputs/execution_events.json")}
    nodes = {r["document_id"]: r for r in read(root / "inputs/plan_nodes.json")}
    metas = {r["document_id"]: r for r in read(root / "inputs/prior_document_metadata.json")}
    changes = set(read(root / "inputs/change_notice_ids.json"))
    saved = read(root / "document_purpose_fields.json")
    labels = {"CANCEL_CAPITAL": "减少注册资本", "EMPLOYEE_INCENTIVE": "用于员工持股计划或股权激励",
              "CONVERTIBLE_BOND": "用于转换公司可转债", "VALUE_SUPPORT": "为维护公司价值及股东权益"}
    checked, unchecked = set("√✓✔☑"), set("□☐")
    assert len(saved) == len(source) == 1244 and len(events) == 948 and len(changes) == 54
    blocks_checked = 0
    for row in saved:
        key = row["document_id"]
        original = source[key]
        for field in ["raw", "text", "receipts"]:
            assert digest(root / original[field + "_snapshot"]) == original[field + "_sha256"]
        raw_pages = read(root / original["text_snapshot"])
        pages = ["".join(unicodedata.normalize("NFKC", p).split()) for p in raw_pages]
        expected_locations = []
        for page_no, page in enumerate(pages, 1):
            expected_locations.extend((page_no, m.end()) for m in re.finditer("回购用途", page))
        blocks = row["purpose"]["blocks"]
        assert [(b["page"], b["normalized_start"]) for b in blocks] == expected_locations
        for block in blocks:
            text = pages[block["page"] - 1]
            a, z = block["normalized_start"], block["normalized_end"]
            assert text[a:z] == block["evidence"]
            stop = re.search("累计已回购|累计回购股|实际回购价格|回购方案|预计回购|一、|二、", text[a:a + 450])
            expected_end = a + stop.start() if stop else min(len(text), a + 450)
            assert z == expected_end
            evidence = block["evidence"]
            markers, unknown = {}, []
            for name, label in labels.items():
                count = evidence.count(label)
                pos = evidence.find(label)
                if count != 1 or pos <= 0 or evidence[pos - 1] not in checked | unchecked:
                    unknown.append(name)
                else:
                    markers[name] = evidence[pos - 1] in checked
            selections = sorted(k for k, value in markers.items() if value)
            assert block["marks"] == markers and block["unresolved_labels"] == unknown
            assert block["selected_labels"] == selections
            assert block["complete_four_label_menu"] == (len(markers) == 4 and not unknown and bool(selections))
            assert block["multiple_choice_allocation_unknown"] == (len(selections) > 1)
            assert block["alternative_or_combined_wording"] == bool(re.search("二者之一|二者皆有|之一或|一种或多种", evidence))
            blocks_checked += 1
        complete = [b for b in blocks if b["complete_four_label_menu"]]
        alternatives = {tuple(b["selected_labels"]) for b in complete}
        purposes = None
        if not blocks:
            status = "NO_STANDARD_PURPOSE_HEADER"
        elif not complete:
            status = "PARTIAL_OR_UNKNOWN_MENU_RETAINED"
        elif len(alternatives) > 1:
            status = "CONFLICTING_COMPLETE_MENUS"
        elif any(b["marks"] and not b["complete_four_label_menu"] for b in blocks):
            status = "COMPLETE_AND_PARTIAL_MENUS_REQUIRE_REVIEW"
        else:
            purposes = list(next(iter(alternatives)))
            status = "EXPLICIT_MULTIPLE_PURPOSES_NO_ALLOCATION" if len(purposes) > 1 else "EXPLICIT_SINGLE_PURPOSE"
        assert row["purpose"]["status"] == status and row["purpose"]["selected_purposes"] == purposes
        candidates = [x for x in [metas.get(key), nodes.get(key), events.get(key)] if x]
        clocks = {pd.Timestamp(x["known_at"]).isoformat() for x in candidates if x.get("known_at")}
        assert row["known_clock_candidates"] == sorted(clocks)
        assert row["known_at"] == (max((pd.Timestamp(v) for v in clocks)).isoformat() if clocks else None)
        root_ids = {x["root_id"] for x in [nodes.get(key), events.get(key)] if x and x.get("root_id")}
        assert row["root_id"] == (next(iter(root_ids)) if len(root_ids) == 1 else None)
        assert row["is_prior_direct_execution"] == (key in events)
        assert row["is_selected_plan_node"] == (key in nodes)
        assert row["is_change_title_candidate"] == (key in changes)
        event = events.get(key)
        budget = event.get("budget_terms", {}) if event else {}
        floor = budget.get("floor_cents") if budget.get("unambiguous_pair") else None
        expected_ratio = Decimal(event["cumulative_cents"]) / Decimal(floor) if event and floor and floor > 0 else None
        assert row["reported_plan_floor_cents"] == floor
        assert row["reported_cumulative_cents"] == (event["cumulative_cents"] if event else None)
        if expected_ratio is None:
            assert row["reported_execution_to_floor_ratio"] is None
        else:
            assert abs(row["reported_execution_to_floor_ratio"] - float(expected_ratio)) < 1e-12
        assert row["prior_change_status"] == (event.get("change_status") if event else None)
        assert row["prior_usable_positive_disclosure"] == (event.get("usable_positive_disclosure", False) if event else False)
        assert row["trading_feature_admitted"] is False and row["historical_first_publication_verified"] is False
        assert row["purpose"]["trading_feature_admitted"] is False
        assert row["purpose"]["fallback_cancellation_used_as_current_purpose"] is False
    execution = [r for r in saved if r["is_prior_direct_execution"]]
    assert read(root / "execution_purpose_fields.json") == execution
    assert read(root / "change_notice_fields.json") == [r for r in saved if r["is_change_title_candidate"]]
    result = read(root / "result.json")
    assert result["purpose_statuses"] == dict(Counter(r["purpose"]["status"] for r in saved))
    assert result["execution_purpose_statuses"] == dict(Counter(r["purpose"]["status"] for r in execution))
    assert result["execution_ratios_available"] == sum(r["reported_execution_to_floor_ratio"] is not None for r in execution)
    assert result["change_titles_without_existing_root_link"] == sum(r["is_change_title_candidate"] and r["root_id"] is None for r in saved)
    assert result["new_accounts"] == result["new_returns"] == result["new_network_requests"] == 0
    assert result["T12"] == "NOT_RUN" and result["full_purpose_termination_chain_established"] is False
    return {"status": "PASS_SAVED_REPURCHASE_CHECKBOX_PURPOSE_AND_RATIO_RECOMPUTATION", "documents": len(saved),
            "execution_records": len(execution), "purpose_header_blocks": blocks_checked, "change_title_candidates": len(changes),
            "new_accounts": 0, "new_network_requests": 0, "external_review": "NOT_PERFORMED",
            "scope": "从1244份冻结原文及文本核对明确勾选、未知、多用途、已披露实施比例和继承时钟；不证明完整用途变更链、自由流通市值或T12策略有效。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    output = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
