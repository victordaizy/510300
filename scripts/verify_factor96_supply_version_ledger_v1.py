"""独立只读核对32份文档角色、原文引用和三阶段延期可见时间。"""
from __future__ import annotations
import argparse
from collections import Counter
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
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def verify(root):
    frozen, start = read(root/"freeze.json"), read(root/"run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(start["at"])
    assert digest(root/"freeze.json") == start["freeze_sha256"]
    for item in frozen["files"]:
        assert digest(root/item["path"]) == item["sha256"], item["path"]
    documents = {r["document_id"]: r for r in read(root/"inputs/documents.json")}
    fields = {r["document_id"]: r for r in read(root/"inputs/prior_fields.json")}
    assert len(documents) == 39
    pages = {}
    for key, row in documents.items():
        assert digest(root/"inputs/raw"/(key+".pdf")) == row["raw_sha256"]
        assert digest(root/"inputs/text"/(key+".json")) == row["text_sha256"]
        pages[key] = [normalize(p) for p in read(root/"inputs/text"/(key+".json"))]
    roles = read(root/"document_roles.json")
    assert len(roles) == 32 and len({r["document_id"] for r in roles}) == 32
    expected_counts = {"ISSUER_DEFERRED_GRANT_UNLOCK": 17, "SUPPORTING_OPINION": 4,
                       "ISSUER_CORRECTION": 6, "ISSUER_SCHEDULE_EXTENSION": 3, "ISSUER_SUPPLEMENT": 2}
    assert dict(Counter(r["document_role"] for r in roles)) == expected_counts
    for row in roles:
        key, role = row["document_id"], row["document_role"]
        original = documents[key]
        assert original["title_kind"] == "REVISION_TITLE_RETAINED_SEPARATELY"
        assert row["prior_field_status"] == fields[key]["field_status"]
        assert not row["trading_feature_admitted"] and not row["independent_event_admitted"]
        assert row["role_evidence"] == pages[key][0][:1100]
        if role == "ISSUER_DEFERRED_GRANT_UNLOCK":
            assert re.search("暂缓授予|暂缓部分", row["title"])
            assert row["symbol"][:6] in pages[key][0] and "解除限售" in pages[key][0]
            assert "核查意见" not in row["title"] and "更正" not in row["title"]
        elif role == "ISSUER_CORRECTION":
            assert row["title"].endswith("更正公告") and row["symbol"][:6] in pages[key][0]
        elif role == "ISSUER_SUPPLEMENT":
            assert "补充" in row["title"]
        elif role == "ISSUER_SCHEDULE_EXTENSION":
            assert "延期上市流通" in row["title"] and row["symbol"] == "600515.SH"
        else:
            assert "核查意见" in row["title"] and not row["title"].endswith("更正公告")
    links = read(root/"explicit_version_links.json")
    assert len(links) == 10
    for row in links:
        child, parent = documents[row["updated_document_id"]], documents[row["referenced_document_id"]]
        assert child["symbol"] == parent["symbol"] == row["symbol"]
        assert parent["archive_date"] <= child["archive_date"]
        assert row["context"] in pages[row["updated_document_id"]][row["page"]-1]
        assert row["literal_reference"] in row["context"]
        assert not row["counts_as_additional_supply"] and not row["historical_original_rewritten"]
    versions = read(root/"hainan_schedule_versions.json")
    assert [r["document_id"] for r in versions] == ["1206469829", "1207281952", "1209076663"]
    assert [r["scheduled_unlock_date"] for r in versions] == ["2020-01-26", "2021-01-26", None]
    for row in versions:
        assert pd.Timestamp(row["known_at"]) == (pd.Timestamp(row["archive_date"])+pd.Timedelta(days=2)).tz_localize("Asia/Shanghai")
        assert row["nominal_shares"] == "2249297094" and not row["actual_sales_known"]
    assert "待后续基础控股完成业绩承诺补偿后" in "".join(pages["1209076663"])
    checks = read(root/"asof_schedule_checks.json")
    assert [r["visible_document"] for r in checks] == [None, "1206469829", "1206469829", "1207281952", "1207281952", "1209076663"]
    for row in checks:
        available = [v for v in versions if pd.Timestamp(v["known_at"]) <= pd.Timestamp(row["decision_time"])]
        latest = available[-1] if available else None
        assert row["scheduled_unlock_date"] == (latest["scheduled_unlock_date"] if latest else None)
    assert checks[-1]["state"] == "OPEN_ENDED_CONTINGENT_NOT_ZERO_SUPPLY"
    distinct = read(root/"same_issuer_same_day_distinct_plans.json")
    assert distinct["automatic_merge_performed"] is False
    for key, year in zip(distinct["documents"], distinct["separate_plan_years"], strict=True):
        assert str(year)+"年" in documents[key]["title"]
        assert "2023年2月8日" in pages[key][0]
    result = read(root/"result.json")
    assert result["role_counts"] == expected_counts and result["explicit_links"] == len(links)
    assert not result["all_events_deduplicated"] and not result["goal_achieved"]
    assert result["new_accounts"] == result["new_returns"] == 0
    return {"status": "PASS_SAVED_VERSION_ROLE_REFERENCE_AND_ASOF_RECOMPUTATION", "source_documents": len(documents),
            "role_documents": len(roles), "explicit_links": len(links), "schedule_versions": len(versions),
            "asof_cases": len(checks), "new_accounts": 0, "network_requests": 0,
            "scope": "只覆盖冻结32份候选角色、10条引用及一条三阶段延期链，不表示全部事件去重、字段提取或T13绩效完成。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    value = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False))
