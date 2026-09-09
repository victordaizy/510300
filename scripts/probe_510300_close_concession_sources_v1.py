"""只读核查510300公开来源，先保存请求清单，再保存原始响应和时间证据。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_close_concession_source_feasibility_v1"
CN = timezone(timedelta(hours=8))
ALLOWED_HOSTS = {"etf.sse.com.cn", "www.sse.com.cn", "query.sse.com.cn", "yunhq.sse.com.cn",
                 "star.sse.com.cn", "www.huatai-pb.com", "www.ht-pb.com.cn", "gu.qq.com",
                 "qt.gtimg.cn", "web.ifzq.gtimg.cn", "web.sqt.gtimg.cn", "imgcache.qq.com", "big5.sse.com.cn"}


def now() -> str:
    return datetime.now(CN).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def capture(job: dict, directory: Path) -> dict:
    identity = job["id"]
    url = job["url"]
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in ALLOWED_HOSTS:
        raise ValueError("仅接入已列明的HTTPS公开来源")
    result = {"id": identity, "purpose": job["purpose"], "requested_url": url,
              "request_started_at": now(), "tls_verified": True,
              "market_session_observation": "SOURCE_FEASIBILITY_ONLY_NOT_A_STRATEGY_SAMPLE"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
               "Accept": "*/*", "Accept-Encoding": "identity"}
    if job.get("referer"):
        headers["Referer"] = job["referer"]
    start = time.monotonic()
    raw = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=ssl.create_default_context()) as response:
            result.update({"status_code": response.status, "resolved_url": response.url,
                "server_date_header": response.headers.get("Date"),
                "content_type": response.headers.get("Content-Type"),
                "content_encoding": response.headers.get("Content-Encoding"),
                "cache_control": response.headers.get("Cache-Control"),
                "last_modified_header": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag")})
            raw = response.read(12_000_001)
            if len(raw) > 12_000_000:
                raise ValueError("本次公开来源响应超过12MB单文件上限")
        result["status"] = "RAW_RESPONSE_CAPTURED_CONTENT_NOT_YET_ADMITTED"
    except urllib.error.HTTPError as exc:
        raw = exc.read(1_000_000)
        result.update({"status": "HTTP_ERROR", "status_code": exc.code, "error": str(exc)})
    except (OSError, urllib.error.URLError, ValueError) as exc:
        result.update({"status": "SOURCE_REQUEST_FAILED", "error_type": type(exc).__name__,
                       "error": str(exc)})
    result["received_at"] = now()
    result["elapsed_seconds_monotonic"] = time.monotonic() - start
    if raw is not None:
        target = directory / (identity + ".raw")
        target.write_bytes(raw)
        result.update({"raw_path": target.relative_to(OUT).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    else:
        result.update({"raw_path": None, "bytes": None, "sha256": None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="510300公开来源分批只读核查")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.batch):
        raise ValueError("批次名称仅使用小写字母、数字和下划线")
    jobs = json.loads(args.requests.read_text(encoding="utf-8-sig"))
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 12:
        raise ValueError("每批需提供1至12个明确请求")
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9_]+", i) for i in ids):
        raise ValueError("请求编号重复或格式不合法")
    directory = OUT / "source_batches" / args.batch
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"purpose": "FIRST_MECHANISM_PUBLIC_SOURCE_FEASIBILITY",
        "batch": args.batch, "recorded_before_requests_at": now(),
        "probe_script_sha256": sha(Path(__file__).read_bytes()), "jobs": jobs,
        "executable_assets": ["510300.SH", "CASH_CNY"],
        "future_return_reads": 0, "strategy_backtest": False, "broker_connections": 0,
        "old_forward_ledger_writes": 0, "position_impact": 0,
        "clock_offset_proven": False,
        "clock_note": "本机请求时钟仅为采集回执，未经独立校时不能证明交易所延迟",
        "late_response_is_not_historical_availability_proof": True}
    (directory / "request_manifest.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(lambda job: capture(job, directory), jobs))
    (directory / "receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
