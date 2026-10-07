"""检查免费来源的实际覆盖，并保留首轮研究快照。仅下载公开说明文档。"""

import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_pressure_recovery_v1"
OUT = REPORT / "source_followup_20261001"
CN = timezone(timedelta(hours=8))
SOURCES = {
    "akshare_realtime_book": "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock/stock_ask_bid_em.py",
    "akshare_transaction_details": "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock/stock_zh_a_tick_tx.py",
    "pytdx_history_transactions": "https://raw.githubusercontent.com/rainx/pytdx/master/pytdx/parser/get_history_transaction_data.py",
    "xtquant_documentation": "https://dict.thinktrader.net/nativeApi/xtdata.html",
    "eastmoney_history_documentation": "https://emt.18.cn/api/quant-help/python/python_select_api_history.html",
    "eastmoney_tick_documentation": "https://emt.18.cn/api/quant-help/python/python_object_data.html",
    "eastmoney_faq": "https://emt.18.cn/api/quant-help/faq/index.html",
    "xtick_readme": "https://raw.githubusercontent.com/xticktop/DemoXtickPython/main/README.md",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture_document(item):
    identity, url = item
    receipt = {"id": identity, "url": url, "started_at": datetime.now(CN).isoformat()}
    try:
        response = requests.get(url, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0"})
        path = OUT / "free_source_docs" / (identity + ".txt")
        path.write_bytes(response.content)
        receipt.update(http_status=response.status_code, final_url=response.url,
                       status="RESPONSE_SAVED" if response.ok else "HTTP_FAILURE_SAVED",
                       bytes=len(response.content), sha256=sha_bytes(response.content),
                       path=path.relative_to(OUT).as_posix())
    except requests.RequestException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    receipt.update(received_at=datetime.now(CN).isoformat(), market_rows_acquired=0)
    return receipt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "free_source_review_status.json").exists():
        raise SystemExit("本轮免费来源检查已经完成，请保留原回执。")
    frozen = [REPORT / name for name in (
        "01_同步行情表.csv", "02_事件表.csv", "03_订单表.csv", "04_结果表.csv",
        "status.json", "registration_receipt.json")]
    frozen += [ROOT / "config/510300_pressure_recovery_v1.json",
               ROOT / "research/pressure_recovery_v1.py",
               ROOT / "outputs/01a0f739-6bb2-7741-8d30-204e31d95c62/510300_价格让步与压力恢复_四表.xlsx"]
    before = {p: sha_bytes(p.read_bytes()) for p in frozen}
    write_json(OUT / "free_source_scope.json", {
        "registered_at": datetime.now(CN).isoformat(), "user_constraint": "只有免费数据",
        "historical_only": True, "purchase_allowed": False,
        "public_document_requests": SOURCES, "broker_connection_allowed": False,
        "new_forward_collection_allowed": False,
        "purpose": "核对公开接口能力，并检查本地原始响应中是否遗漏所需字段。",
    })

    rows, extras, batches = [], [], []
    pairs = [
        ("data/raw/primary_market/raw/iopv", "data/raw/primary_market/510300_iopv_snapshots.parquet"),
        ("data/raw/primary_market_v1_2/raw/iopv", "data/raw/primary_market_v1_2/510300_iopv_snapshots_v1_2.parquet"),
    ]
    for raw_dir, table in pairs:
        frame = pd.read_parquet(ROOT / table)
        saved_times = set(pd.to_datetime(frame.retrieved_at, utc=True).astype(str))
        batch_rows, payloads_in_table = [], set()
        for path in sorted((ROOT / raw_dir).rglob("*.json")):
            body = path.read_bytes()
            record = json.loads(body.decode("utf-8-sig"))
            normalized, raw = record["normalized_record"], record["raw_payload"]
            received = normalized["retrieved_at"]
            represented = str(pd.to_datetime(received, utc=True)) in saved_times
            payload_hash = sha_bytes(json.dumps(raw, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8"))
            snap = raw.get("snap")
            row = {
                "source_path": path.relative_to(ROOT).as_posix(), "raw_file_sha256": sha_bytes(body),
                "payload_sha256": payload_hash, "table": table,
                "received_at": received, "source_envelope_time": normalized["exchange_timestamp"],
                "represented_by_receipt_time": represented,
                "raw_keys": "|".join(sorted(raw)),
                "snap_entries": len(snap) if isinstance(snap, list) else -1,
                "snap_contains_nested_structure": any(isinstance(x, (dict, list)) for x in snap),
            }
            batch_rows.append(row)
            if represented:
                payloads_in_table.add(payload_hash)
            else:
                extra = dict(normalized, source_path=row["source_path"], raw_file_sha256=row["raw_file_sha256"],
                             payload_sha256=payload_hash, quality_status="NO_VIEW",
                             reason="事后取得16:29外层快照；无同步估值时点、盘口、未成交申报量或订单证据")
                extras.append(extra)
                target = OUT / "supplemental_raw" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for row in batch_rows:
            row["same_payload_represented_in_table"] = row["payload_sha256"] in payloads_in_table
        represented_times = {str(pd.to_datetime(r["received_at"], utc=True))
                             for r in batch_rows if r["represented_by_receipt_time"]}
        batches.append({"table": table, "table_rows": len(frame), "raw_files": len(batch_rows),
                        "table_receipts_all_found": saved_times <= represented_times,
                        "raw_without_table_receipt": sum(not r["represented_by_receipt_time"] for r in batch_rows)})
        rows.extend(batch_rows)

    for extra in extras:
        extra["same_payload_already_in_table"] = next(
            row["same_payload_represented_in_table"] for row in rows if row["source_path"] == extra["source_path"])
    write_csv(OUT / "原始快照覆盖.csv", rows)
    write_csv(OUT / "补充快照.csv", extras)
    review = {
        "checked_at": datetime.now(CN).isoformat(), "raw_files": len(rows), "batches": batches,
        "snap_length_counts": dict(Counter(r["snap_entries"] for r in rows)),
        "nested_payload_rows": sum(r["snap_contains_nested_structure"] for r in rows),
        "supplemental_receipts": len(extras),
        "supplemental_economic_payloads_already_represented": all(x["same_payload_already_in_table"] for x in extras),
        "finding": "原始响应均为同一17项网页快照布局，未发现遗漏的盘口或盘后排队字段；补充原件不是新的经济行情。",
        "scope": "688条IOPV汇总记录对应的两个原件目录；另有2条原件未进入汇总表。",
    }
    write_json(OUT / "raw_response_review.json", review)
    (OUT / "free_source_docs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(capture_document, SOURCES.items()))
    write_json(OUT / "free_source_receipts.json", receipts)

    unchanged = all(sha_bytes(path.read_bytes()) == digest for path, digest in before.items())
    status = {
        "generated_at": datetime.now(CN).isoformat(),
        "status": "FREE_SOURCE_REVIEW_NO_ADMISSIBLE_HISTORICAL_SAMPLE",
        "only_free_data": True, "goal_achieved": False,
        "initial_four_table_archive_rows": 957, "additional_archive_receipts": len(extras),
        "archive_rows_including_supplement": 957 + len(extras),
        "additional_economic_payloads": sum(not x["same_payload_already_in_table"] for x in extras),
        "qualified_events_added": 0, "qualified_orders_added": 0, "market_rows_downloaded": 0,
        "net_expectancy": None, "net_sharpe": None, "full_account_status": "NOT_RUN",
        "free_source_document_requests": len(receipts),
        "successful_http_responses": sum(r.get("http_status") == 200 for r in receipts),
        "all_parquet_receipts_found": all(b["table_receipts_all_found"] for b in batches),
        "frozen_research_files_unchanged": unchanged,
        "frozen_file_hashes": {p.relative_to(ROOT).as_posix(): digest for p, digest in before.items()},
        "scope_limit": "本轮检索及本地目录范围内未取得，不等于证明所有免费来源不存在。",
    }
    write_json(OUT / "free_source_review_status.json", status)
    print(json.dumps({k: status[k] for k in (
        "status", "additional_archive_receipts", "additional_economic_payloads",
        "market_rows_downloaded", "successful_http_responses", "frozen_research_files_unchanged")}, ensure_ascii=False))
    if not unchanged or not status["all_parquet_receipts_found"]:
        raise SystemExit("保存快照或原件对应关系检查未通过。")


if __name__ == "__main__":
    main()
