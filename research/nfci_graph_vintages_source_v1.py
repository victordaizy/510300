"""从ALFRED官方图表CSV分批取得明确版本列，重建NFCI及信用/风险分项。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SERIES = ("NFCI", "NFCICREDIT", "NFCIRISK")
OUT = ROOT / "reports/research/510300_nfci_graph_vintages_source_v1"
RAW = ROOT / "data/raw/macro/510300_nfci_graph_vintages_source_v1"
URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse(body, pairs, batch):
    frame = pd.read_csv(io.BytesIO(body))
    expected = [s + "_" + d.replace("-", "") for s, d in pairs]
    if frame.columns.to_list() != ["observation_date", *expected]:
        raise ValueError("响应列未逐项匹配请求的序列和版本日期")
    frame["observation_date"] = pd.to_datetime(frame.observation_date)
    if frame.observation_date.duplicated().any() or not frame.observation_date.is_monotonic_increasing:
        raise ValueError("源观察日期重复或非递增")
    rows = []
    for (series, date), column in zip(pairs, expected):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        local = pd.Series(numeric.to_numpy(), index=pd.DatetimeIndex(frame.observation_date)).dropna()
        if len(local) < 5 or local.index.max() > pd.Timestamp(date):
            raise ValueError("版本列没有足够历史，或包含版本日以后的观察")
        latest = local.index.max()
        older = latest - pd.Timedelta(days=28)
        if older not in local.index:
            raise ValueError("版本内缺少严格四周前的观察，不能用下一条代替")
        availability = (pd.Timestamp(date).tz_localize("America/Chicago") + pd.DateOffset(days=1)).tz_convert("Asia/Shanghai")
        rows.append({"series": series, "vintage_date": pd.Timestamp(date), "available_at": availability,
                     "observation_date": latest, "four_week_observation_date": older,
                     "value": float(local.loc[latest]), "value_four_weeks_ago_same_vintage": float(local.loc[older]),
                     "change4": float(local.loc[latest] - local.loc[older]),
                     "nonmissing_historical_values": len(local), "source_batch": batch, "source_column": column})
    return rows


def request_batch(snapshot, number, dates):
    pairs = [(series, date) for date in dates for series in SERIES]
    params = {"id": ",".join(s for s, _ in pairs), "vintage_date": ",".join(d for _, d in pairs)}
    record = {"batch": number, "pairs": pairs, "url": URL, "params": params,
              "requested_at": now(), "tls_verify": True}
    try:
        response = requests.get(URL, params=params, timeout=(12, 45), headers={"User-Agent": "Mozilla/5.0"})
        body = response.content
        path = snapshot / "raw" / f"batch_{number:03d}.csv"
        path.write_bytes(body)
        record.update({"received_at": now(), "response_url": response.url, "http_status": response.status_code,
                       "content_type": response.headers.get("Content-Type"), "http_date": response.headers.get("Date"),
                       "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                       "raw_path": path.relative_to(ROOT).as_posix()})
        response.raise_for_status()
        rows = parse(body, pairs, number)
        record.update(status="SAVED_AND_VERSION_COLUMNS_CHECKED", parsed_rows=len(rows))
        return rows
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        return []
    finally:
        save(snapshot / "receipts" / f"batch_{number:03d}.json", record)


def collect():
    if (OUT / "result.json").exists():
        raise RuntimeError("本次版本数据已取得，请读取保存结果；不覆盖或重复下载。")
    pointer = json.loads((ROOT / "reports/research/510300_nfci_public_vintages_source_v1/latest_source_snapshot.json").read_text(encoding="utf-8"))
    old = ROOT / pointer["snapshot"]
    selections = {s: json.loads((old / s / "selected_vintages.json").read_text(encoding="utf-8")) for s in SERIES}
    start = max(min(v) for v in selections.values())
    dates = sorted({d for v in selections.values() for d in v if start <= d <= "2026-09-25"})
    snapshot = RAW / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f_0800")
    (snapshot / "raw").mkdir(parents=True, exist_ok=False)
    (snapshot / "receipts").mkdir()
    batches = [dates[i:i + 12] for i in range(0, len(dates), 12)]
    protocol = {"study_id": "510300_NFCI_GRAPH_VINTAGES_SOURCE_V1", "at": now(),
                "official_url": URL, "series": list(SERIES), "requested_vintages": dates,
                "vintage_calendar_source": pointer["snapshot"], "first_common_vintage": start,
                "batch_vintages": 12, "workers": 2, "requests": len(batches),
                "within_vintage_features": "每版最新观察、同版严格四周前观察和两者差值。",
                "availability": "版本日America/Chicago午夜加一个自然日，转Asia/Shanghai。",
                "first_public_http_proof": False,
                "evidence_scope": "官方档案回放，历史HTTP实际到达时刻未认证；今日修订值不回填过去原点。",
                "previous_failed_method_preserved": pointer, "new_accounts": 0, "new_model_fits": 0}
    save(snapshot / "protocol.json", protocol)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    save(OUT / "running_snapshot.json", {"snapshot": snapshot.relative_to(ROOT).as_posix(), "status": "RUNNING", "at": now()})
    rows = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(request_batch, snapshot, n, batch): n for n, batch in enumerate(batches)}
        for completed, future in enumerate(as_completed(futures), 1):
            part = future.result()
            rows.extend(part)
            print(f"NFCI历史版本批次完成 {completed}/{len(batches)}，本批取得 {len(part)} 条版本状态。", flush=True)
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((snapshot / "receipts").glob("*.json"))]
    failed = [r["batch"] for r in receipts if r["status"] != "SAVED_AND_VERSION_COLUMNS_CHECKED"]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["vintage_date", "series"]).reset_index(drop=True)
        assert not frame.duplicated(["series", "vintage_date"]).any()
        frame.to_parquet(snapshot / "vintage_features.parquet", index=False)
    complete = not failed and len(frame) == 3 * len(dates)
    result = {"at": now(), "status": "COMPLETE_OFFICIAL_ARCHIVAL_VINTAGE_FEATURES" if complete else "PARTIAL_ARCHIVE_NOT_ADMITTED",
              "snapshot": snapshot.relative_to(ROOT).as_posix(), "series": list(SERIES),
              "requested_vintages": len(dates), "version_rows": len(frame), "failed_batches": failed,
              "first_vintage": dates[0], "last_vintage": dates[-1], "new_model_fits": 0,
              "new_accounts": 0, "independent_market_observations": 0, "goal_achieved": False}
    if complete:
        wide = frame.pivot(index="vintage_date", columns="series", values="observation_date")
        assert wide.nunique(axis=1).eq(1).all(), "三个序列的最新观察周不一致"
        result["latest_observation_date"] = frame.observation_date.max().date().isoformat()
        # 单批保存数据再次解析，对每个请求的最后五周及可用性逐值核对。
        replay = []
        for receipt in receipts:
            replay.extend(parse((ROOT / receipt["raw_path"]).read_bytes(), receipt["pairs"], receipt["batch"]))
        rebuilt = pd.DataFrame(replay).sort_values(["vintage_date", "series"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame, rebuilt, check_exact=True)
        result["saved_raw_recomputation"] = "PASS_ALL_REQUESTED_VERSION_COLUMNS"
    save(snapshot / "result.json", result)
    save(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分批取得官方图表版本CSV并保留失败档案。")
    parser.add_argument("--collect", required=True, action="store_true")
    parser.parse_args()
    collect()
