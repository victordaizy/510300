"""V2接管已终止的来源批次：复用原文，已尝试URL不再请求，保留V1语义否定。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd
import requests

from research.factor96_earnings_cashflow_measurement_v1 import digest, now, save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2"
PRIOR = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"
RAW = ROOT / "data/raw/cninfo/factor96_financial_parser_scope_v2"
STUDY = "510300_FACTOR96_FINANCIAL_PARSER_SCOPE_V2"
ACCEPTED = {"REUSED_SAME_HASH_PDF", "DOWNLOADED_SAME_HASH_PDF"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze():
    assert not (OUT / "batch_freeze.json").exists()
    termination = read(PRIOR / "batch_termination_for_scope_v2.json")
    assert termination["status"] == "TERMINATED_INVALID_PARSER_BATCH_FOR_V2_SOURCE_HANDOFF"
    assert not (PRIOR / "batch_complete.json").exists()
    assert "22 passed" in (OUT / "repaired_bounded_memory_pytest.log").read_text(encoding="utf-8")
    assert "4 passed" in (OUT / "handoff_pytest.log").read_text(encoding="utf-8")
    for item in read(PRIOR / "batch_freeze.json")["files"]:
        assert digest(PRIOR / item["path"]) == item["sha256"]
    targets = read(PRIOR / "batch_targets.json")
    copies = {}
    prior_attempts = 0
    for target in targets:
        checksum = target["sha256"]
        receipt = PRIOR / "fetch_receipts" / (checksum + ".json")
        attempt = PRIOR / "fetch_attempts" / (checksum + ".json")
        assert not receipt.exists() or attempt.exists(), "来源回执缺少请求前尝试记录"
        if attempt.exists():
            prior_attempts += int(target["local_reuse"] is None)
        record = None
        if receipt.exists():
            try:
                record = read(receipt)
            except json.JSONDecodeError:
                copies[OUT / "source_handoff/incomplete_receipts" / receipt.name] = receipt
        if record is not None:
            target["handoff"] = "REUSE_COMPLETED_V1_FETCH"
            target["prior_fetch_status"] = record["status"]
            target["prior_receipt_snapshot"] = "source_handoff/receipts/" + receipt.name
            copies[OUT / target["prior_receipt_snapshot"]] = receipt
        elif attempt.exists():
            target["handoff"] = "NO_RETRY_INTERRUPTED_V1_ATTEMPT"
            target["prior_fetch_status"] = "NO_VIEW_V1_ATTEMPT_INTERRUPTED"
            target["prior_receipt_snapshot"] = "source_handoff/attempts/" + attempt.name
            copies[OUT / target["prior_receipt_snapshot"]] = attempt
        else:
            target["handoff"] = "UNATTEMPTED_ORIGINAL_PLAN"
            target["prior_fetch_status"] = None
            target["pdf_relative_path"] = (RAW / "pdf" / (checksum + ".pdf")).relative_to(ROOT).as_posix()
    assert len(targets) == 2112 and sum(len(t["target_metrics"]) for t in targets) == 2529
    for path in [Path(__file__), ROOT / "research/factor96_financial_row_parser_v2.py",
                 ROOT / "tests/test_factor96_financial_parser_scope_v2.py", ROOT / "tests/test_factor96_financial_parser_repair_v1.py",
                 ROOT / "tests/test_factor96_financial_parser_handoff_v2.py"]:
        copies[OUT / "code" / path.name] = path
    for name in ["batch_termination_for_scope_v2.json", "parser_semantic_invalidation_addendum.json", "batch_freeze.json", "scope_v1_red_pytest.log"]:
        copies[OUT / "source_handoff" / name] = PRIOR / name
    for name in ["verified_facts_before.parquet", "original_facts_before.parquet", "confirmed_contradictions.json", "summary_path_fields_before.parquet"]:
        copies[OUT / "inputs" / name] = PRIOR / "inputs" / name
    for path in (PRIOR / "inputs/pdf").glob("*.pdf"):
        copies[OUT / "inputs/pdf" / path.name] = path
    for target, source in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copy2(source, target)
    handoff_counts = pd.Series([t["handoff"] for t in targets]).value_counts().to_dict()
    new_requests = sum(t["handoff"] == "UNATTEMPTED_ORIGINAL_PLAN" and not t["local_reuse"] for t in targets)
    assert prior_attempts + new_requests <= 2106
    save(OUT / "batch_targets.json", targets)
    protocol = {"at": now(), "study_id": STUDY, "target_documents": 2112, "target_fields": 2529,
        "reason": "V1真实原文反例确认独立报表状态和旧表列头泄漏；终止该批次并保留原文件、全部已请求响应及语义否定。",
        "scope": "沿用原冻结2529字段/2112PDF；V2对全部已取得且同哈希原文重新解析，不按金额变化或收益挑选修复样本。",
        "handoff_counts": handoff_counts, "prior_logical_http_attempts": prior_attempts,
        "new_logical_http_requests_maximum": new_requests, "combined_original_plan_http_bound": 2106,
        "failed_or_interrupted_old_requests": "保持未知，不重新请求；终止时已有尝试记录但无最终回执者也不重试。",
        "reused_sources": "已完成原响应按字节、PDF头和SHA核对后复用；原文路径保持，旧解析结果不作为V2字段结果。",
        "new_requests": "原清单中从未尝试的URL各至多请求一次，4并发，连接10秒/读取35秒；无自动重试。",
        "parser": "V2识别无合并前缀的财务报表标题并切断合并状态；识别本年/当年列；新表头重置旧列，只有显式续表可继承。",
        "page_scope": "沿用前三页加原字段页邻页，目标字段缺失则一次全文表格路径；原文金额冲突和缺列头保持未知。",
        "memory": "4个隔离解析进程；表格候选保存后释放当前页缓存，不累计年报全文页对象。",
        "validation": "原七金额反例以及白云山合并/单体边界通过；22项回归通过仍不代表全体来源已人工确认。",
        "formula": "单季、TTM、同季历史窗、披露时钟及200日报告年龄全部保持；V1原重算因新反证未运行，不能称V1已成功。",
        "prior_protocol_timing": "V2规则在发现V1批次中的来源反例之后制定；不能声称对该反例事前未知。",
        "new_accounts": 0, "returns_read": False, "T11": "NOT_RUN", "goal_achieved": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "batch_protocol.json", protocol)
    frozen = list(copies) + [OUT / "batch_targets.json", OUT / "batch_protocol.json", OUT / "repaired_pytest.log", OUT / "repaired_bounded_memory_pytest.log", OUT / "handoff_pytest.log"]
    save(OUT / "batch_freeze.json", {"at": now(), "prior_batch_freeze_sha256": digest(PRIOR / "batch_freeze.json"),
        "files": [{"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)} for path in frozen]})
    print(json.dumps({"handoff_counts": handoff_counts, "prior_logical_attempts": prior_attempts, "new_requests_max": new_requests}, ensure_ascii=False), flush=True)


def fetch_document(target):
    checksum = target["sha256"]
    destination = ROOT / target["pdf_relative_path"]
    receipt_path = OUT / "fetch_receipts" / (checksum + ".json")
    assert not receipt_path.exists()
    receipt = {"at": now(), "announcement_id": target["announcement_id"], "expected_sha256": checksum,
        "url": target["url"], "handoff": target["handoff"], "logical_http_requests": 0, "status": "STARTED"}
    save(OUT / "fetch_attempts" / (checksum + ".json"), receipt)
    if target["handoff"] != "UNATTEMPTED_ORIGINAL_PLAN":
        receipt["original_receipt_snapshot"] = target["prior_receipt_snapshot"]
        if target["handoff"] == "NO_RETRY_INTERRUPTED_V1_ATTEMPT":
            receipt["status"] = "NO_VIEW_V1_ATTEMPT_INTERRUPTED"
        else:
            prior = read(OUT / target["prior_receipt_snapshot"])
            if prior["status"] not in ACCEPTED:
                receipt["status"] = prior["status"]
            else:
                assert destination.stat().st_size == target["expected_bytes"] == prior["bytes"]
                assert digest(destination) == checksum == prior["sha256"]
                with destination.open("rb") as stream:
                    assert stream.read(4) == b"%PDF"
                receipt.update({"status": "REUSED_SAME_HASH_PDF", "sha256": checksum, "bytes": destination.stat().st_size,
                    "pdf_relative_path": target["pdf_relative_path"]})
        receipt["finished_at"] = now()
        save(receipt_path, receipt)
        return receipt
    assert not destination.exists()
    if target["local_reuse"]:
        source = ROOT / target["local_reuse"]
        assert source.stat().st_size == target["expected_bytes"] and digest(source) == checksum
        shutil.copy2(source, destination)
        receipt.update({"status": "REUSED_SAME_HASH_PDF", "sha256": checksum, "bytes": destination.stat().st_size,
            "pdf_relative_path": target["pdf_relative_path"], "finished_at": now()})
        save(receipt_path, receipt)
        return receipt
    receipt["logical_http_requests"] = 1
    temporary = RAW / "partial" / (checksum + ".building")
    size, calculated, head = 0, hashlib.sha256(), b""
    try:
        with requests.get(target["url"], headers={"User-Agent": "Mozilla/5.0", "Accept": "application/pdf"}, stream=True, timeout=(10, 35)) as response:
            receipt.update({"http_status": response.status_code, "final_url": response.url,
                "headers": {k: v for k, v in response.headers.items() if k.lower() in {"date", "etag", "last-modified", "content-type", "content-length"}},
                "redirects": [{"url": r.url, "status": r.status_code} for r in response.history]})
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    if not head:
                        head = chunk[:8]
                    stream.write(chunk)
                    size += len(chunk)
                    calculated.update(chunk)
            receipt.update({"bytes": size, "sha256": calculated.hexdigest(), "pdf_header": head.startswith(b"%PDF")})
            if response.status_code != 200 or not head.startswith(b"%PDF"):
                receipt["status"] = "NO_VIEW_HTTP_OR_NOT_PDF"
            elif size != target["expected_bytes"] or calculated.hexdigest() != checksum:
                receipt["status"] = "NO_VIEW_DIFFERENT_FROM_ARCHIVED_PDF"
            else:
                temporary.replace(destination)
                receipt.update({"status": "DOWNLOADED_SAME_HASH_PDF", "pdf_relative_path": target["pdf_relative_path"]})
    except Exception as error:
        receipt.update({"status": "NO_VIEW_REQUEST_ERROR", "error_type": type(error).__name__, "error": str(error),
            "partial_bytes": size, "partial_sha256": calculated.hexdigest()})
    receipt["finished_at"] = now()
    save(receipt_path, receipt)
    return receipt


def parse_document(target):
    from research.factor96_financial_row_parser_v2 import extract_official_pdf_facts
    started = time.monotonic()
    try:
        content = (ROOT / target["pdf_relative_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == target["sha256"]
        arguments = {"period_type": target["period_type"], "report_period": target["report_period"]}
        result = extract_official_pdf_facts(content, table_scope_pages=set(target["table_scope_pages"]), **arguments)
        fallback = not set(target["target_metrics"]).issubset({m["metric_id"] for m in result["metrics"]})
        if fallback:
            prior_decisions = result["decisions"]
            result = extract_official_pdf_facts(content, **arguments)
            result["primary_scope_decisions"] = prior_decisions
        result.update({"announcement_id": target["announcement_id"], "status": "PARSED", "full_document_table_fallback": fallback,
            "target_metrics": target["target_metrics"], "seconds": time.monotonic() - started, "finished_at": now(),
            "pdf_relative_path": target["pdf_relative_path"]})
    except Exception as error:
        result = {"announcement_id": target["announcement_id"], "status": "NO_VIEW_PARSE_ERROR", "metrics": [],
            "error_type": type(error).__name__, "error": str(error), "seconds": time.monotonic() - started,
            "official_pdf_sha256": target["sha256"], "finished_at": now(), "pdf_relative_path": target["pdf_relative_path"]}
    save(OUT / "parsed_documents" / (target["sha256"] + ".json"), result)
    return {"sha256": target["sha256"], "status": result["status"], "seconds": result["seconds"],
        "target_fields_recovered": len(set(target["target_metrics"]) & {m["metric_id"] for m in result["metrics"]})}


def run():
    assert not (OUT / "batch_started.json").exists()
    for row in read(OUT / "batch_freeze.json")["files"]:
        assert digest(OUT / row["path"]) == row["sha256"], row["path"]
    for name in [Path(__file__).name, "factor96_financial_row_parser_v2.py"]:
        assert digest(ROOT / "research" / name) == digest(OUT / "code" / name)
    targets = read(OUT / "batch_targets.json")
    for path in [RAW / "pdf", RAW / "partial", OUT / "fetch_attempts", OUT / "fetch_receipts", OUT / "parsed_documents"]:
        path.mkdir(parents=True, exist_ok=True)
    save(OUT / "batch_started.json", {"at": now(), "pid": os.getpid(), "batch_freeze_sha256": digest(OUT / "batch_freeze.json"), "target_documents": len(targets)})
    fetched, parsed, parse_futures = [], [], []
    with ThreadPoolExecutor(max_workers=4) as download_pool, ProcessPoolExecutor(max_workers=4) as parse_pool:
        downloads = {download_pool.submit(fetch_document, target): target for target in targets}
        for future in as_completed(downloads):
            receipt = future.result()
            fetched.append(receipt)
            if receipt["status"] in ACCEPTED:
                parse_futures.append(parse_pool.submit(parse_document, downloads[future]))
            if len(fetched) % 40 == 0 or len(fetched) == len(targets):
                print(f"V2来源处理{len(fetched)}/{len(targets)}，解析完成{sum(f.done() for f in parse_futures)}。", flush=True)
        save(OUT / "download_complete.json", {"at": now(), "documents": len(fetched),
            "logical_http_requests": sum(r["logical_http_requests"] for r in fetched),
            "status_counts": pd.Series([r["status"] for r in fetched]).value_counts().to_dict(),
            "matched_pdf_bytes": sum(r.get("bytes", 0) for r in fetched if r["status"] in ACCEPTED)})
        for future in as_completed(parse_futures):
            parsed.append(future.result())
            if len(parsed) % 40 == 0 or len(parsed) == len(parse_futures):
                print(f"V2三字段解析完成{len(parsed)}/{len(parse_futures)}。", flush=True)
    save(OUT / "batch_complete.json", {"at": now(), "documents": len(targets), "download_receipts": len(fetched),
        "parsed_documents": len(parsed), "parsed_status_counts": pd.Series([r["status"] for r in parsed]).value_counts().to_dict(),
        "recovered_target_fields": sum(r["target_fields_recovered"] for r in parsed), "new_accounts": 0, "returns_read": False})
    print("V2来源处理完成，可按冻结原公式构建独立字段版本。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
