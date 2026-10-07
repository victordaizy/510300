"""保存腾讯最近已结束交易日的分笔原件；仅做覆盖检查，不生成交易信号。"""

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001"
PROBE = BASE / "tencent_probe"
OUT = BASE / "tencent_20260930"
SYMBOL = "sh510300"
DAY = "20260930"
CN = timezone(timedelta(hours=8))
URL = "https://stock.gtimg.cn/data/index.php"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_timeline(body):
    match = re.fullmatch(r'v_detail_time_sh510300=\[(\d{8}),"([^"]*)"\];?', body.strip())
    if not match or match[1] != DAY:
        raise ValueError("目录日期或结构与登记的已结束交易日不符。")
    bounds = [tuple(item.split("~")) for item in match[2].split("|")]
    if not 1 <= len(bounds) <= 100 or any(len(item) != 2 for item in bounds):
        raise ValueError("分笔分页目录异常。")
    return bounds


def parse_page(page, body):
    match = re.fullmatch(r'v_detail_data_sh510300=\[(\d+),"([^"]*)"\];?', body.strip())
    if not match or int(match[1]) != page or not match[2]:
        raise ValueError(f"第{page}页结构异常或为空。")
    rows = []
    for item in match[2].split("|"):
        seq, clock, price, change, volume, amount, side = item.split("/")
        datetime.strptime(clock, "%H:%M:%S")
        values = [float(v) for v in (price, change, volume, amount)]
        if not all(math.isfinite(v) for v in values) or side not in {"B", "S", "M"}:
            raise ValueError("分笔字段不合格。")
        if values[0] <= 0 or values[2] < 0 or values[3] < 0:
            raise ValueError("价格或量额出现不合理值。")
        rows.append({"date": "2026-09-30", "symbol": "510300.SH", "page": page,
                     "source_sequence": int(seq), "time": clock, "price_cny": values[0],
                     "change_cny": values[1], "volume_lots_source": values[2],
                     "amount_cny_source": values[3], "side_source": side})
    return rows


def fetch(job):
    name, url, params = job
    target = OUT / (name + ".txt")
    receipt = {"name": name, "url": url, "params": params,
               "started_at": datetime.now(CN).isoformat()}
    try:
        with requests.get(url, params=params, timeout=(8, 20), stream=True) as response:
            receipt["http_status"] = response.status_code
            body = bytearray()
            for chunk in response.iter_content(16384):
                body.extend(chunk)
                if len(body) > 16384:
                    raise ValueError("单页超过16KiB预算，停止保存。")
        target.write_bytes(body)
        receipt.update(path=target.relative_to(ROOT).as_posix(), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(),
                       status="SAVED" if response.status_code == 200 else "HTTP_ERROR")
    except (requests.RequestException, ValueError) as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
    receipt["received_at"] = datetime.now(CN).isoformat()
    return receipt


def quote_fields(body):
    match = re.fullmatch(r'v_sh510300="([^"]*)";', body.strip())
    if not match:
        raise ValueError("报价结构不符。")
    fields = match[1].split("~")
    if fields[2] != "510300" or not fields[30].startswith(DAY):
        raise ValueError("报价证券或日期不符。")
    return fields[30], float(fields[35].split("/")[2]), fields[1]


def main():
    if OUT.exists():
        raise SystemExit("本批目录已存在，读取已保存结果，不覆盖原件。")
    bounds = parse_timeline((PROBE / "timeline.txt").read_bytes().decode("gbk"))
    first_rows = parse_page(0, (PROBE / "page_000.txt").read_bytes().decode("gbk"))
    before_time, before_amount, name = quote_fields((PROBE / "quote.txt").read_bytes().decode("gbk"))
    OUT.mkdir()
    save_json(OUT / "request_manifest.json", {
        "registered_at": datetime.now(CN).isoformat(), "date": DAY, "symbol": SYMBOL,
        "purpose": "只核验已结束交易日分笔及盘后覆盖，不计算策略收益。",
        "input_probe": PROBE.relative_to(ROOT).as_posix(), "listed_pages": len(bounds),
        "raw_budget_bytes": (len(bounds) + 2) * 16384,
        "is_level2_order_stream": False, "is_forward_collection": False,
    })
    jobs = [(f"page_{page:03d}", URL,
             {"appn": "detail", "action": "data", "c": SYMBOL, "p": page})
            for page in range(1, len(bounds))]
    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(fetch, jobs))
    receipts.extend([fetch(("timeline_after", URL, {"appn": "detail", "action": "timeline", "c": SYMBOL})),
                     fetch(("quote_after", "https://qt.gtimg.cn/q=sh510300", {}))])
    save_json(OUT / "receipts.json", receipts)
    if any(r["status"] != "SAVED" for r in receipts):
        raise SystemExit("部分请求失败，原件与回执已保存，未生成完整数据结论。")
    after_bounds = parse_timeline((OUT / "timeline_after.txt").read_bytes().decode("gbk"))
    after_time, after_amount, _ = quote_fields((OUT / "quote_after.txt").read_bytes().decode("gbk"))
    rows, boundary_mismatches = list(first_rows), []
    if (first_rows[0]["time"], first_rows[-1]["time"]) != bounds[0]:
        boundary_mismatches.append(0)
    for page in range(1, len(bounds)):
        parsed = parse_page(page, (OUT / f"page_{page:03d}.txt").read_bytes().decode("gbk"))
        if (parsed[0]["time"], parsed[-1]["time"]) != bounds[page]:
            boundary_mismatches.append(page)
        rows.extend(parsed)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "510300_20260930_source_ticks.parquet", index=False)
    regular = frame[frame.time <= "15:00:59"]
    after = frame[(frame.time >= "15:05:00") & (frame.time <= "15:30:59")]
    seq = frame.source_sequence
    missing = sorted(set(range(0, int(seq.max()) + 1)) - set(seq))
    regular_amount = float(regular.amount_cny_source.sum())
    profile = {
        "status": "SAVED_SOURCE_AGGREGATED_TICKS_NOT_FILL_EVIDENCE",
        "name": name, "rows": len(frame), "pages": len(bounds),
        "date": "2026-09-30", "start": frame.time.min(), "end": frame.time.max(),
        "regular_rows": len(regular), "after_hours_rows": len(after),
        "after_hours_first": None if after.empty else after.time.min(),
        "after_hours_last": None if after.empty else after.time.max(),
        "after_hours_price_values": sorted(after.price_cny.unique().tolist()),
        "after_hours_amount_cny_source": float(after.amount_cny_source.sum()),
        "after_hours_volume_lots_source": float(after.volume_lots_source.sum()),
        "first_sequence": int(seq.min()), "last_sequence": int(seq.max()),
        "missing_source_sequences": missing, "duplicate_source_sequences": int(seq.duplicated().sum()),
        "source_sequence_strictly_increasing": bool((seq.diff().dropna() > 0).all()),
        "source_times_nondecreasing": frame.time.is_monotonic_increasing,
        "timeline_unchanged": bounds == after_bounds, "page_boundary_mismatches": boundary_mismatches,
        "quote_source_time_before": before_time, "quote_source_time_after": after_time,
        "quote_amount_before": before_amount, "quote_amount_after": after_amount,
        "regular_amount_sum": regular_amount, "regular_amount_difference": regular_amount - after_amount,
        "regular_amount_relative_difference": (regular_amount - after_amount) / after_amount,
        "source_side_is_independently_validated_aggressor": False,
        "level2_order_stream": False, "exchange_message_completeness_established": False,
        "queue_or_fill_evidence": False, "m1_m2_admitted": False, "strategy_returns_computed": False,
        "date_binding": "分笔正文不含日期；取数前后腾讯分页目录与报价日期均为20260930。",
        "point_in_time": "本轮于20261001事后接收，不能证明9月30日15:06前已可用。",
    }
    save_json(OUT / "profile.json", profile)
    print(json.dumps(profile, ensure_ascii=False))


if __name__ == "__main__":
    main()
