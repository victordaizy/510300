"""单独补取批次中唯一两次TLS失败的原公告，保留原失败和原清单。"""
from pathlib import Path
import json

from research import factor96_rights_issue_documents_v1 as source


def main():
    original = source.OUT
    documents = source.read(original / "documents.json")
    failed = [r for r in documents if r["status"] != "PDF_TEXT_SAVED"]
    assert len(failed) == 1 and failed[0]["document_id"] == "1212032708"
    for attempt in [1, 2]:
        receipt = source.read(original / "receipts" / f"1212032708_a{attempt}.json")
        assert receipt["status"] == "REQUEST_FAILED" and receipt["error_type"] == "SSLError"
    folder = original / "transport_completion_v1"
    folder.mkdir(exist_ok=False)
    source.save(folder / "protocol.json", {"at": source.now(), "target": "1212032708", "reason": "前批次其他490份成功，唯一文档两次标准TLS握手失败；批次已经终止，单独新会话最多两次补取。",
        "selection": "全部失败文档，非收益或内容选择", "source_url": failed[0]["source_url"],
        "tls_verification": True, "source_collector_sha256": source.digest(Path(source.__file__)),
        "original_manifest_sha256": source.digest(original / "documents.json"),
        "original_result_sha256": source.digest(original / "result.json"),
        "new_accounts": 0, "original_manifest_modified": False}, True)
    (folder / "completion_code.py").write_bytes(Path(__file__).read_bytes())
    assert source.digest(Path(source.__file__)) == source.digest(original / "collector_code.py")
    source.OUT = folder
    row = source.extract(source.download(failed[0]))
    for field in ["raw_snapshot", "text_snapshot", "receipt_snapshot"]:
        if row.get(field):
            row[field] = "transport_completion_v1/" + row[field]
    row["initial_batch_status"] = failed[0]["status"]
    source.save(folder / "document.json", row, True)
    effective = [row if old["document_id"] == row["document_id"] else old for old in documents]
    source.save(folder / "effective_documents.json", effective, True)
    result = {"at": source.now(), "status": "SINGLE_TRANSPORT_GAP_COMPLETED" if row["status"] == "PDF_TEXT_SAVED" else "SINGLE_TRANSPORT_GAP_RETAINED",
        "document_id": row["document_id"], "document_status": row["status"], "effective_pdf_text_documents": sum(r["status"] == "PDF_TEXT_SAVED" for r in effective),
        "original_batch_pdf_text_documents": 490, "original_failures_retained": True, "new_accounts": 0,
        "new_returns": 0, "external_review": "NOT_PERFORMED"}
    source.save(folder / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
