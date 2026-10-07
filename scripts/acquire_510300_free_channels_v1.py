"""取得公开数据集元信息和510300免费分钟原件；访问受限文件仅记录正常响应。"""

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/510300_free_channels_v1/20261001"
CN = timezone(timedelta(hours=8))
REPOS = {
    "venvoo": "venvoo/china-a-share-l2-level2-limit-order-book-tick-data",
    "alphat04": "alphat04/Tick-by-Tick-Orders-China",
    "neigezhu": "neigezhu/china-etf-1min-ohlcv",
}
LIMIT = 64 * 1024 * 1024


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(job):
    label, url, destination = job
    result = {"id": label, "url": url, "started_at": datetime.now(CN).isoformat(),
              "authenticated": False}
    target = OUT / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"原件已存在，禁止覆盖：{target}")
    try:
        with requests.get(url, timeout=(8, 30), stream=True,
                          headers={"User-Agent": "Mozilla/5.0"}) as response:
            result["http_status"] = response.status_code
            length = response.headers.get("Content-Length")
            if length and int(length) > LIMIT:
                raise ValueError("本次单文件上限64MiB，未下载超过上限的文件。")
            body = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                body.extend(chunk)
                if len(body) > LIMIT:
                    raise ValueError("响应超过本次单文件上限，停止读取。")
            if not response.ok:
                target = target.with_name(target.name + ".http_error.txt")
            target.write_bytes(body)
            result.update(path=target.relative_to(ROOT).as_posix(), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest(),
                          status="SAVED" if response.ok else "ACCESS_OR_HTTP_ERROR_SAVED")
    except (requests.RequestException, ValueError) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    result["received_at"] = datetime.now(CN).isoformat()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "receipts.json").exists():
        raise SystemExit("本批已有完成回执，请读取现有原件。")
    receipts, inventories, jobs = [], {}, []
    for alias, repo in REPOS.items():
        receipt = fetch((alias + "_metadata", "https://huggingface.co/api/datasets/" + repo,
                         alias + "/dataset_info.json"))
        receipts.append(receipt)
        if receipt["status"] != "SAVED":
            continue
        info = json.loads((ROOT / receipt["path"]).read_text(encoding="utf-8"))
        revision = info["sha"]
        paths = [item["rfilename"] for item in info.get("siblings", [])]
        inventory = {"repo": repo, "revision": revision, "gated": info.get("gated"),
                     "files": len(paths), "source": receipt["path"]}
        if alias == "venvoo":
            streams = defaultdict(set)
            expected = {"行情.parquet", "逐笔委托.parquet", "逐笔成交.parquet"}
            for path in paths:
                parts = path.split("/")
                if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isdigit() and parts[1] in expected:
                    streams[parts[0]].add(parts[1])
            complete = sorted(day for day, found in streams.items() if found == expected)
            inventory.update(complete_three_file_dates=len(complete), first_date=min(complete),
                             last_date=max(complete), incomplete_dates={k: sorted(expected-v) for k,v in streams.items() if v != expected},
                             directory_completeness_is_session_completeness=False)
            requested = ["manifests.parquet"]
        elif alias == "alphat04":
            archives = sorted(path for path in paths if path.endswith(".7z"))
            inventory.update(archive_count=len(archives), first_archive=archives[0], last_archive=archives[-1])
            requested = []
        else:
            requested = ["README.md", "LICENSE", "metadata/coverage_by_instrument.csv",
                         "metadata/summary.json", "metadata/market_calendar.csv",
                         "metadata/missing_intervals.csv", "data/etf_1m/SH/510300.parquet"]
        inventories[alias] = inventory
        for path in requested:
            if path not in paths:
                raise ValueError(f"清单内没有请求文件：{alias}/{path}")
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path)}"
            jobs.append((alias + "/" + path, url, alias + "/" + path))
    save_json(OUT / "request_manifest.json", {"registered_at": datetime.now(CN).isoformat(),
              "scope": "只下载目标证券分钟原件与来源清单；尚不计算策略收益。", "jobs": jobs})
    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts.extend(executor.map(fetch, jobs))
    save_json(OUT / "receipts.json", receipts)
    save_json(OUT / "repository_inventory.json", inventories)
    market_file = OUT / "neigezhu/data/etf_1m/SH/510300.parquet"
    if market_file.is_file():
        frame = pd.read_parquet(market_file)
        times = pd.to_datetime(frame["timestamp"])
        days = times.dt.date.astype(str)
        daily = frame.groupby(days).size()
        daily.rename("rows").to_csv(OUT / "510300_minute_daily_rows.csv", encoding="utf-8-sig")
        summary = {
            "status": "DOWNLOADED_PRICE_DATA_NOT_MICROSTRUCTURE_ADMITTED",
            "rows": len(frame), "trading_dates": int(days.nunique()),
            "start": str(times.min()), "end": str(times.max()), "columns": list(frame.columns),
            "symbols": sorted(frame.symbol.astype(str).unique().tolist()),
            "exchanges": sorted(frame.exchange.astype(str).unique().tolist()),
            "duplicate_keys": int(frame.duplicated(["exchange", "symbol", "timestamp"]).sum()),
            "nulls": frame.isna().sum().to_dict(), "day_rows_min": int(daily.min()),
            "day_rows_max": int(daily.max()), "median_day_rows": float(daily.median()),
            "positive_price_rows": int((frame[["open", "high", "low", "close"]] > 0).all(axis=1).sum()),
            "ohlc_inconsistent_rows": int(((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
                                           (frame.low > frame[["open", "close", "high"]].min(axis=1))).sum()),
            "negative_volume_rows": int((frame.volume < 0).sum()),
            "time_contract": "原作者声明UTC+8的无时区分钟标签；开始/结束标记及原供应商未独立证实。",
            "m1_m2_admitted": False, "strategy_returns_computed": False,
        }
        save_json(OUT / "510300_minute_profile.json", summary)
    print(json.dumps({"inventories": inventories,
                      "responses": [{k:r.get(k) for k in ("id", "http_status", "status", "bytes")} for r in receipts]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
