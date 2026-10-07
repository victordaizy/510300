"""已失败、已中断和已缓存来源的接管，不允许重复HTTP或覆盖原文件。"""
import hashlib
import json

import pytest

from research import factor96_financial_parser_batch_v2 as batch


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir()
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "OUT", out)
    def forbidden_network(*args, **kwargs):
        pytest.fail("已有来源尝试不得重复请求")
    monkeypatch.setattr(batch.requests, "get", forbidden_network)
    payload = b"%PDF-1.7\nlocal-saved-document\n"
    checksum = hashlib.sha256(payload).hexdigest()
    target = {"announcement_id": "case", "sha256": checksum, "url": "https://example.invalid/source.pdf",
        "expected_bytes": len(payload), "pdf_relative_path": "saved.pdf", "prior_receipt_snapshot": "original.json",
        "handoff": "REUSE_COMPLETED_V1_FETCH"}
    return tmp_path, out, target, payload


def test_failed_prior_fetch_stays_unknown_without_network(handoff):
    root, out, target, payload = handoff
    (out / "original.json").write_text(json.dumps({"status": "NO_VIEW_HTTP_OR_NOT_PDF"}), encoding="utf-8")
    result = batch.fetch_document(target)
    assert result["status"] == "NO_VIEW_HTTP_OR_NOT_PDF" and result["logical_http_requests"] == 0
    assert not (root / "saved.pdf").exists()


def test_interrupted_prior_attempt_is_not_retried(handoff):
    root, out, target, payload = handoff
    target["handoff"] = "NO_RETRY_INTERRUPTED_V1_ATTEMPT"
    result = batch.fetch_document(target)
    assert result["status"] == "NO_VIEW_V1_ATTEMPT_INTERRUPTED" and result["logical_http_requests"] == 0


def test_same_hash_original_is_reused_without_rewriting(handoff):
    root, out, target, payload = handoff
    pdf = root / "saved.pdf"
    pdf.write_bytes(payload)
    stamp = pdf.stat().st_mtime_ns
    (out / "original.json").write_text(json.dumps({"status": "DOWNLOADED_SAME_HASH_PDF", "sha256": target["sha256"], "bytes": len(payload)}), encoding="utf-8")
    result = batch.fetch_document(target)
    assert result["status"] == "REUSED_SAME_HASH_PDF" and result["logical_http_requests"] == 0
    assert pdf.read_bytes() == payload and pdf.stat().st_mtime_ns == stamp


def test_changed_local_original_cannot_be_admitted(handoff):
    root, out, target, payload = handoff
    (root / "saved.pdf").write_bytes(payload[:-1] + b"X")
    (out / "original.json").write_text(json.dumps({"status": "DOWNLOADED_SAME_HASH_PDF", "sha256": target["sha256"], "bytes": len(payload)}), encoding="utf-8")
    with pytest.raises(AssertionError):
        batch.fetch_document(target)
    assert not (out / "fetch_receipts" / (target["sha256"] + ".json")).exists()
