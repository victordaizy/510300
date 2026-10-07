"""有限核查官方历史权重归档；不读取收益、不生成交易或账户。

初始国证与 Wayback 请求原样保存在本轮目录。此脚本继续执行事先登记的
Common Crawl 年末批次检查，每个批次最多一次请求；已有回执不重新请求。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_weight_float_source_v1"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(OUT).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def collect_commoncrawl() -> None:
    catalogue = OUT / "sources/commoncrawl_index_list.json"
    available = read(catalogue)
    selected = []
    for year in range(2015, 2026):
        group = sorted((v for v in available if v["id"].startswith(f"CC-MAIN-{year}-")),
                       key=lambda v: v["id"])
        if group:
            selected.append(group[-1])
    assert len(selected) == 11
    params = [("url", "csindex.com.cn"), ("matchType", "domain"),
              ("output", "json"), ("filter", r"url:.*000300closeweight\.xls.*"),
              ("filter", "status:200"), ("collapse", "digest")]
    protocol = {"registered_at": now(), "catalogue": identity(catalogue),
                "selected": selected, "params": params,
                "maximum_attempts_per_query": 1, "minimum_interval_seconds": 2,
                "stop_on_status": [401, 403, 429],
                "no_match_meaning": "仅本次指定批次和文件匹配未命中，不代表全部历史不存在",
                "acquisition_role": "只检索官方原文件的归档地址；归档服务不作为权重编制来源",
                "prices_read": False, "new_account_evaluations": 0}
    registered = OUT / "commoncrawl_protocol.json"
    if registered.exists():
        old = read(registered)
        assert old["selected"] == selected and old["params"] == [list(p) for p in params]
    else:
        write_new(registered, protocol)
    receipt_dir = OUT / "commoncrawl_receipts"
    receipt_dir.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    halted = False
    for batch in selected:
        receipt = receipt_dir / (batch["id"] + ".json")
        if receipt.exists():
            old = read(receipt)
            halted = halted or old.get("status_code") in (401, 403, 429)
            continue
        record = {"batch": batch["id"], "at": now(), "url": batch["cdx-api"],
                  "params": params, "network_attempts": 0}
        if halted:
            record["status"] = "NOT_RUN_AFTER_SOURCE_ACCESS_OR_RATE_LIMIT_RESPONSE"
        else:
            record["network_attempts"] = 1
            try:
                response = session.get(batch["cdx-api"], params=params, timeout=(10, 35))
                payload = response.content
                raw = OUT / "sources" / (batch["id"] + ".jsonl")
                with raw.open("xb") as stream:
                    stream.write(payload)
                record.update(status_code=response.status_code, final_url=response.url,
                              content_type=response.headers.get("Content-Type"), source=identity(raw))
                if response.status_code == 200:
                    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
                    record.update(status="QUERY_RETURNED", records=rows, record_count=len(rows))
                elif response.status_code == 404 and "No Captures found" in response.text:
                    record.update(status="NO_CAPTURE_IN_FIXED_QUERY", record_count=0)
                else:
                    record.update(status="SOURCE_RESPONSE_NOT_ADMITTED", body_excerpt=response.text[:300])
                halted = response.status_code in (401, 403, 429)
            except (requests.RequestException, ValueError) as error:
                record.update(status="SOURCE_REQUEST_FAILED", error=type(error).__name__ + ": " + str(error))
        write_new(receipt, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if not halted:
            time.sleep(2)
    receipts = [read(receipt_dir / (batch["id"] + ".json")) for batch in selected]
    result = {"finished_at": now(), "registered_queries": len(selected),
              "attempted_queries": sum(v["network_attempts"] for v in receipts),
              "queries_with_hits": sum(v.get("record_count", 0) > 0 for v in receipts),
              "capture_records": sum(v.get("record_count", 0) for v in receipts),
              "unrun_queries": sum(v["network_attempts"] == 0 for v in receipts),
              "source_access_stop": halted, "receipts": [identity(receipt_dir / (v["id"] + ".json")) for v in selected]}
    write_new(OUT / "commoncrawl_result.json", result)
    print("有限归档检索结束：" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="官方历史权重的有限归档核查")
    parser.add_argument("action", choices=["commoncrawl"])
    args = parser.parse_args()
    if args.action == "commoncrawl":
        collect_commoncrawl()
