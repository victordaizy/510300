"""仅保存限量公开字段说明/目录元数据；不安装外部包、不调用收费接口。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_pressure_free_source_feasibility_v1.json"


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / config["output_directory"]
    if out.exists():
        raise SystemExit("当前批次目录已存在，拒绝覆盖公开来源回执。")
    out.mkdir(parents=True)
    documents = out / "public_documents"
    documents.mkdir()
    inputs = [CONFIG, Path(__file__)] + [ROOT / name for name in config["upstream"]]
    write_json(out / "registration.json", {"registered_at": now(), "study_id": config["study_id"],
                                           "status": "SOURCE_FEASIBILITY_REGISTERED_NO_RETURN_LABELS",
                                           "configuration": config,
                                           "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in inputs]})
    receipts, summaries = [], []
    total = 0
    scope = config["scope"]
    for key, url in config["public_documents"].items():
        receipt = {"source_id": key, "url": url, "started_at": now(), "authenticated": False,
                   "fee_usd": 0, "bytes_received": 0, "http_status": None}
        chunks = []
        try:
            remaining = scope["maximum_public_document_bytes_total"] - total
            if remaining <= 0:
                raise RuntimeError("本批次8MiB公开文档预算已用完。")
            limit = min(scope["maximum_public_document_bytes_each"], remaining)
            with requests.get(url, timeout=scope["http_timeout_seconds"], stream=True,
                              headers={"User-Agent": "510300 non-commercial source research", "Accept-Encoding": "identity"}) as response:
                receipt["http_status"] = response.status_code
                receipt["content_type"] = response.headers.get("Content-Type")
                receipt["last_modified"] = response.headers.get("Last-Modified")
                for chunk in response.iter_content(chunk_size=65536):
                    receipt["bytes_received"] += len(chunk)
                    total += len(chunk)
                    if receipt["bytes_received"] > limit:
                        raise RuntimeError("该响应超过冻结文档预算，已终止读取。")
                    chunks.append(chunk)
                receipt["complete_response"] = True
            body = b"".join(chunks)
            suffix = ".json" if "json" in (receipt["content_type"] or "") else ".txt"
            target = documents / (key + suffix)
            target.write_bytes(body)
            receipt.update(path=str(target), sha256=digest(target), saved_bytes=len(body), status="PUBLIC_RESPONSE_SAVED")
            if receipt["http_status"] == 200 and suffix == ".json":
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, list):
                    summaries.append({"source_id": key, "items": [{k: row.get(k) for k in ["id", "path", "type", "size", "gated"]}
                                                                   for row in payload if isinstance(row, dict)]})
                elif "items" in payload:
                    summaries.append({"source_id": key, "total_count": payload.get("total_count"),
                                      "items": [{k: row.get(k) for k in ["full_name", "html_url", "description", "size", "updated_at"]}
                                                for row in payload["items"]]})
                else:
                    summaries.append({"source_id": key, "repo_id": payload.get("id"), "revision": payload.get("sha"),
                                      "gated": payload.get("gated"), "siblings": payload.get("siblings")})
        except (requests.RequestException, RuntimeError, ValueError) as error:
            receipt.update(status="PUBLIC_DOCUMENT_REQUEST_FAILED_OR_LIMITED", error=str(error), complete_response=False)
            if chunks:
                target = documents / (key + ".partial.txt")
                target.write_bytes(b"".join(chunks))
                receipt.update(path=str(target), sha256=digest(target), saved_bytes=target.stat().st_size)
        receipt["finished_at"] = now()
        receipts.append(receipt)
        write_json(out / "public_document_receipts.json", {"total_bytes_received": total, "receipts": receipts})
        print(f"公开来源{len(receipts)}/{len(config['public_documents'])}：{key}，HTTP {receipt['http_status']}，{receipt['status']}。", flush=True)
    write_json(out / "public_metadata_summary.json", {"status": "PUBLIC_METADATA_ONLY_NOT_MARKET_DATA", "sources": summaries})
    write_json(out / "probe_status.json", {"finished_at": now(), "status": "PUBLIC_SOURCE_DOCUMENTS_SAVED_AWAITING_CONTRACT_REVIEW",
                                          "requests": len(receipts), "successful_http_200": sum(x["http_status"] == 200 for x in receipts),
                                          "bytes_received": total, "market_data_rows": 0, "strategy_returns": "NOT_COMPUTED", "fee_usd": 0})
    print("元数据探查完成；尚未准入新市场样本。", flush=True)


if __name__ == "__main__":
    main()
