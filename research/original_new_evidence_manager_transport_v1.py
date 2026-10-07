"""对同一官方产品页使用默认客户端补取，保留旧代理失败，不改变来源接纳或策略规则。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.new_daily_input_adapter_v1 import read, location, relative, accept_source, run
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.official_dividend_coverage_refresh_v1 import check_http_clock, compare_manager, ledger_rows, parse_manager

OUT = ROOT / "reports/research/510300_original_frozen_new_evidence_20260925"


def main():
    settings_path = OUT / "settings.json"
    settings = read(settings_path)
    source = settings["initial_source"]
    directory = OUT / "2026-09-24"
    official = read(location(settings["official_configuration"]))
    previous = read(directory / "manager_attempt2.json")
    require(pd.Timestamp(now()) - pd.Timestamp(previous["finished_at"]) >= pd.Timedelta(minutes=5),
            "未到同一失败来源五分钟重试间隔")
    destination = directory / "manager_default_client"
    destination.mkdir(exist_ok=False)
    receipt_path = destination / "receipt.json"
    request = {"url": official["manager_url"], "params": None, "referer": "https://www.huatai-pb.com/"}
    receipt = {"source": "manager", "request": request, "attempt": 3, "started_at": now(),
               "tls_verified": True, "request_count": 1, "cost_cny": 0,
               "receipt_file": relative(receipt_path), "transport_change": "只将浏览器标识改为Requests默认标识，地址与内容接纳规则不变",
               "previous_failures": [relative(directory / f"manager_attempt{i}.json") for i in [1, 2]]}
    try:
        response = requests.get(request["url"], headers={"Referer": request["referer"]},
                                timeout=(8, 20), verify=True, allow_redirects=False)
        body = response.content
        raw_path = destination / "response.raw"
        raw_path.write_bytes(body)
        receipt.update(retrieved_at=now(), http_status=response.status_code, final_url=response.url,
                       headers=dict(response.headers), actual_request_headers=dict(response.request.headers),
                       raw_bytes=len(body), raw_file=relative(raw_path), raw_sha256=digest(raw_path))
        response.raise_for_status()
        require(response.status_code == 200 and len(body) <= official["maximum_response_bytes"], "官方响应不是合格正文")
        check_http_clock(receipt["headers"], pd.Timestamp(receipt["retrieved_at"]).to_pydatetime(),
                         pd.Timestamp("2026-09-24").date(), official)
        rows = parse_manager(body, "510300")
        compare_manager(rows, ledger_rows(location(source["dividends"])))
        write_json(destination / "parsed_dividends.json", rows, exclusive=True)
        receipt.update(status="HTTP_OK_NOT_YET_ADMITTED", unchanged_manager_event_count=len(rows))
    except Exception as error:
        receipt.update(status="EXTERNAL_FREE_SOURCE_OR_VALIDATION_FAILED", error_type=type(error).__name__, error=str(error))
    receipt["finished_at"] = now()
    write_json(receipt_path, receipt, exclusive=True)
    if receipt["status"] != "HTTP_OK_NOT_YET_ADMITTED":
        print("同一官方来源补充尚未完成：" + receipt["error"], flush=True)
        return
    accept_source(directory, receipt)
    print("已从同一官方产品页取得完整分红表，事件与旧账本一致；继续原固定账户。", flush=True)
    result = run(settings_path, attempt=3)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
