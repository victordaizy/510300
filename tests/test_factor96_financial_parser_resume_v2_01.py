"""续算边界：保留已完成和失败结果，只补没有结果的本地解析。"""
import json
from pathlib import Path

import pytest

from research import factor96_financial_parser_resume_v2_01 as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_sources(tmp_path):
    (tmp_path / "parsed_documents").mkdir()
    targets = [{"sha256": c * 64, "announcement_id": str(i)} for i, c in enumerate("abc")]
    for i, target in enumerate(targets):
        write(tmp_path / "fetch_receipts" / (target["sha256"] + ".json"), {
            "expected_sha256": target["sha256"], "announcement_id": target["announcement_id"],
            "status": "REUSED_SAME_HASH_PDF" if i < 2 else "NO_VIEW_REQUEST_ERROR"})
    return targets


def parsed(target, status="PARSED"):
    return {"announcement_id": target["announcement_id"], "official_pdf_sha256": target["sha256"],
            "parser_version": recovery.PARSER_VERSION, "status": status, "metrics": []}


def test_only_missing_successful_source_is_planned(tmp_path):
    targets = setup_sources(tmp_path)
    path = tmp_path / "parsed_documents" / (targets[0]["sha256"] + ".json")
    write(path, parsed(targets[0]))
    before = path.read_bytes()
    result = recovery.plan_missing(targets, tmp_path)
    assert result["missing_targets"] == [targets[1]]
    assert len(result["completed"]) == 1 and result["unknown_source_hashes"] == [targets[2]["sha256"]]
    assert path.read_bytes() == before


def test_existing_parse_error_is_preserved_without_retry(tmp_path):
    targets = setup_sources(tmp_path)
    write(tmp_path / "parsed_documents" / (targets[0]["sha256"] + ".json"), parsed(targets[0], "NO_VIEW_PARSE_ERROR"))
    assert recovery.plan_missing(targets, tmp_path)["missing_targets"] == [targets[1]]


def test_partial_json_stops_without_overwrite(tmp_path):
    targets = setup_sources(tmp_path)
    path = tmp_path / "parsed_documents" / (targets[0]["sha256"] + ".json")
    path.write_text('{"unfinished":', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        recovery.plan_missing(targets, tmp_path)
    assert path.read_bytes() == before


def test_wrong_source_identity_stops(tmp_path):
    targets = setup_sources(tmp_path)
    result = parsed(targets[0])
    result["official_pdf_sha256"] = "wrong"
    write(tmp_path / "parsed_documents" / (targets[0]["sha256"] + ".json"), result)
    with pytest.raises(AssertionError):
        recovery.plan_missing(targets, tmp_path)


def test_actual_worker_boundary_rejects_existing_result(tmp_path, monkeypatch):
    targets = setup_sources(tmp_path)
    write(tmp_path / "parsed_documents" / (targets[0]["sha256"] + ".json"), parsed(targets[0]))
    monkeypatch.setattr(recovery.batch, "OUT", tmp_path)
    invoked = []
    monkeypatch.setattr(recovery.batch, "parse_document", lambda target: invoked.append(target))
    with pytest.raises(AssertionError):
        recovery.parse_missing(targets[0])
    assert not invoked


def test_actual_worker_uses_original_parser_for_missing_only(tmp_path, monkeypatch):
    targets = setup_sources(tmp_path)
    monkeypatch.setattr(recovery.batch, "OUT", tmp_path)
    invoked = []
    monkeypatch.setattr(recovery.batch, "parse_document", lambda target: invoked.append(target) or {"status": "PARSED"})
    monkeypatch.setattr(recovery.batch, "fetch_document", lambda target: pytest.fail("续算不得请求来源"))
    assert recovery.parse_missing(targets[1]) == {"status": "PARSED"}
    assert invoked == [targets[1]]
