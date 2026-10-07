"""构建 V1.1 非收益点时输入暂存层，并下载深交所官方简称变更整表。"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import requests
import urllib3
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_1_data_contract.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def at_close(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize() + pd.Timedelta(hours=15)


def build_security_master(contract: dict) -> dict:
    spec = contract["sources"]["security_master"]
    source = ROOT / spec["path"]
    frame = pd.read_parquet(source)
    frame = frame.loc[frame["exchange"].isin(["SSE", "SZSE"])].copy()
    frame["security_type"] = "ORDINARY_A_SHARE"
    frame["status_from"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["status_to"] = pd.to_datetime(frame["delist_date"], errors="coerce")
    frame["available_at"] = at_close(frame["status_from"])
    frame["source"] = "local_tushare_stock_basic_historical_snapshot"
    output = output_path(spec["staging_path"])
    columns = [
        "ts_code", "exchange", "market", "name", "security_type", "list_date",
        "delist_date", "status_from", "status_to", "available_at", "source",
    ]
    frame[columns].sort_values("ts_code").to_parquet(output, index=False)
    return {"path": spec["staging_path"], "rows": len(frame), "sha256": sha256_file(output)}


def build_share_counts(contract: dict) -> dict:
    spec = contract["sources"]["share_count_snapshots"]
    frame = pd.read_parquet(ROOT / spec["path"], columns=["ts_code", "trade_date", "total_share", "float_share"])
    frame = frame.loc[frame["ts_code"].str.endswith((".SH", ".SZ"), na=False)].copy()
    frame = frame.rename(columns={"trade_date": "effective_date"})
    frame["effective_date"] = pd.to_datetime(frame["effective_date"]).dt.normalize()
    frame["available_at"] = at_close(frame["effective_date"])
    frame["total_a_shares"] = pd.to_numeric(frame["total_share"], errors="coerce") * 10_000
    frame["tradable_a_shares"] = pd.to_numeric(frame["float_share"], errors="coerce") * 10_000
    frame["source"] = "tushare.daily_basic_historical_trade_date_snapshot"
    frame = frame.dropna(subset=["effective_date", "total_a_shares", "tradable_a_shares"])
    frame = frame.loc[(frame["total_a_shares"] > 0) & (frame["tradable_a_shares"] > 0)]
    output = output_path(spec["staging_path"])
    columns = ["ts_code", "effective_date", "available_at", "total_a_shares", "tradable_a_shares", "source"]
    frame[columns].sort_values(["ts_code", "effective_date"]).to_parquet(output, index=False)
    return {
        "path": spec["staging_path"],
        "rows": len(frame),
        "symbols": int(frame["ts_code"].nunique()),
        "min_date": frame["effective_date"].min().date().isoformat(),
        "max_date": frame["effective_date"].max().date().isoformat(),
        "sha256": sha256_file(output),
    }


def build_industry(contract: dict) -> dict:
    spec = contract["sources"]["industry_intervals"]
    frame = pd.read_parquet(ROOT / contract["taxonomy"]["source_path"])
    frame = frame.loc[frame["ts_code"].str.endswith((".SH", ".SZ"), na=False)].copy()
    frame["valid_from"] = pd.to_datetime(frame["in_date"], errors="coerce")
    frame["valid_to"] = pd.to_datetime(frame["out_date"], errors="coerce")
    frame["available_at"] = at_close(frame["valid_from"])
    frame["source"] = frame["source"].astype(str) + ":申万一级"
    frame = frame.rename(columns={"l1_name": "industry_l1", "l1_code": "industry_l1_code"})
    frame = frame.dropna(subset=["ts_code", "industry_l1", "valid_from"])
    output = output_path(spec["staging_path"])
    columns = ["ts_code", "industry_l1", "industry_l1_code", "valid_from", "valid_to", "available_at", "source"]
    frame[columns].sort_values(["ts_code", "valid_from"]).to_parquet(output, index=False)
    return {
        "path": spec["staging_path"], "rows": len(frame),
        "symbols": int(frame["ts_code"].nunique()), "industries": int(frame["industry_l1"].nunique()),
        "sha256": sha256_file(output),
    }


def build_daily_market_manifest_and_calendar(contract: dict) -> tuple[dict, dict]:
    spec = contract["sources"]["daily_market"]
    datasets = []
    all_dates: list[pd.Series] = []
    for relative in spec["paths"]:
        path = ROOT / relative
        parquet = pq.ParquetFile(path)
        dates = parquet.read(columns=["date"]).to_pandas()["date"]
        all_dates.append(pd.to_datetime(dates).dt.normalize())
        datasets.append({
            "path": relative,
            "rows": parquet.metadata.num_rows,
            "columns": parquet.schema_arrow.names,
            "sha256": sha256_file(path),
        })
    manifest_path = output_path(spec["manifest_path"])
    manifest_path.write_text(json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unique_dates = pd.DatetimeIndex(pd.concat(all_dates, ignore_index=True).dropna().unique()).sort_values()
    calendar_rows = []
    for exchange in ("SSE", "SZSE"):
        part = pd.DataFrame({"exchange": exchange, "date": unique_dates, "is_open": True})
        part["available_at"] = at_close(part["date"])
        part["source"] = "observed_dates_from_tushare_daily_panels"
        calendar_rows.append(part)
    calendar = pd.concat(calendar_rows, ignore_index=True)
    calendar_path = output_path(contract["sources"]["calendar"]["staging_path"])
    calendar.to_parquet(calendar_path, index=False)
    return (
        {"path": spec["manifest_path"], "datasets": datasets, "sha256": sha256_file(manifest_path)},
        {
            "path": contract["sources"]["calendar"]["staging_path"], "rows": len(calendar),
            "open_dates": len(unique_dates), "min_date": unique_dates.min().date().isoformat(),
            "max_date": unique_dates.max().date().isoformat(), "sha256": sha256_file(calendar_path),
        },
    )


def build_board_rules(contract: dict) -> dict:
    payload = {
        "schema_version": "A_SHARE_BOARD_EXECUTION_RULES_V1",
        "boards": ["SSE_MAIN", "SSE_STAR", "SZSE_MAIN", "SZSE_CHINEXT"],
        "lot_rules": {
            "SSE_MAIN": {"buy_lot": 100}, "SZSE_MAIN": {"buy_lot": 100},
            "SZSE_CHINEXT": {"buy_lot": 100}, "SSE_STAR": {"minimum_buy_shares": 200, "increment": 1},
        },
        "price_limit_periods": [
            {"board": "SSE_MAIN", "from": "1996-12-16", "normal": 0.10, "st": 0.05},
            {"board": "SZSE_MAIN", "from": "1996-12-16", "normal": 0.10, "st": 0.05},
            {"board": "SSE_STAR", "from": "2019-07-22", "normal": 0.20, "first_listing_days_without_limit": 5},
            {"board": "SZSE_CHINEXT", "from": "2009-10-30", "to": "2020-08-23", "normal": 0.10},
            {"board": "SZSE_CHINEXT", "from": "2020-08-24", "normal": 0.20, "first_listing_days_without_limit": 5},
            {"board": "SSE_MAIN", "from": "2023-04-10", "first_listing_days_without_limit": 5},
            {"board": "SZSE_MAIN", "from": "2023-04-10", "first_listing_days_without_limit": 5},
        ],
        "execution_priority": ["is_suspended", "volume_shares_positive", "raw_open_present", "price_limit_clamp"],
        "source_notes": [
            "上交所科创板交易特别规定", "深交所创业板交易特别规定",
            "沪深交易所全面注册制主板新股前五日规则",
        ],
    }
    path = output_path(contract["sources"]["board_rules"]["staging_path"])
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"path": contract["sources"]["board_rules"]["staging_path"], "sha256": sha256_file(path)}


def status_from_name(name: object) -> str:
    text = str(name).upper().replace(" ", "")
    if "退" in text:
        return "DELISTING_BOARD"
    if "ST" in text:
        return "ST"
    return "NORMAL"


def download_szse_status(contract: dict, master_path: Path) -> dict:
    spec = contract["sources"]["szse_name_changes"]
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(
        spec["official_url"],
        params={"SHOWTYPE": "xlsx", "CATALOGID": spec["catalog_id"], "TABKEY": spec["tab_key"], "random": "0.428571"},
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.szse.cn/market/product/stock/list/"},
        timeout=60,
        verify=False,
    )
    response.raise_for_status()
    changes = pd.read_excel(io.BytesIO(response.content), dtype=str)
    expected = {"变更日期", "证券代码", "变更前简称", "变更后简称"}
    missing = expected - set(changes.columns)
    if missing:
        raise ValueError(f"深交所简称变更整表缺少字段：{sorted(missing)}")
    changes["变更日期"] = pd.to_datetime(changes["变更日期"], errors="coerce")
    changes["ts_code"] = changes["证券代码"].astype(str).str.zfill(6) + ".SZ"
    changes = changes.rename(
        columns={"变更日期": "change_date", "变更前简称": "before_name", "变更后简称": "after_name"}
    )
    changes = changes.dropna(subset=["change_date"]).sort_values(["ts_code", "change_date"])
    master = pd.read_parquet(master_path)
    master = master.loc[master["exchange"].eq("SZSE")].copy()
    rows: list[dict] = []
    grouped = {code: group for code, group in changes.groupby("ts_code")}
    for item in master.itertuples(index=False):
        code = str(item.ts_code)
        start = pd.Timestamp(item.list_date).normalize()
        end = pd.Timestamp(item.delist_date).normalize() if pd.notna(item.delist_date) else pd.NaT
        events = grouped.get(code)
        current_name = str(item.name)
        if events is None or events.empty:
            name = current_name
            rows.append({"ts_code": code, "status": status_from_name(name), "security_name": name,
                         "valid_from": start, "valid_to": end, "available_at": start + pd.Timedelta(hours=15),
                         "source": "szse_official_name_change_table_no_change_event"})
            continue
        name = str(events.iloc[0]["before_name"])
        interval_start = start
        for event in events.itertuples(index=False):
            event_date = pd.Timestamp(event.change_date).normalize()
            if event_date > interval_start:
                rows.append({"ts_code": code, "status": status_from_name(name), "security_name": name,
                             "valid_from": interval_start, "valid_to": event_date - pd.Timedelta(days=1),
                             "available_at": interval_start + pd.Timedelta(hours=15),
                             "source": "szse_official_name_change_table"})
            name = str(event.after_name)
            interval_start = max(interval_start, event_date)
        rows.append({"ts_code": code, "status": status_from_name(name), "security_name": name,
                     "valid_from": interval_start, "valid_to": end,
                     "available_at": interval_start + pd.Timedelta(hours=15),
                     "source": "szse_official_name_change_table"})
    intervals = pd.DataFrame(rows).sort_values(["ts_code", "valid_from"])
    output = output_path(spec["staging_path"])
    intervals.to_parquet(output, index=False)
    return {
        "path": spec["staging_path"], "download_bytes": len(response.content),
        "change_rows": len(changes), "interval_rows": len(intervals),
        "symbols": int(intervals["ts_code"].nunique()), "sha256": sha256_file(output),
        "download_sha256": hashlib.sha256(response.content).hexdigest(),
    }


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    receipt = {
        "contract_id": contract["contract_id"],
        "collected_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "return_columns_read_for_metrics": False,
    }
    receipt["security_master"] = build_security_master(contract)
    receipt["share_counts"] = build_share_counts(contract)
    receipt["industry"] = build_industry(contract)
    receipt["daily_market"], receipt["calendar"] = build_daily_market_manifest_and_calendar(contract)
    receipt["board_rules"] = build_board_rules(contract)
    master_path = ROOT / contract["sources"]["security_master"]["staging_path"]
    try:
        receipt["szse_status"] = download_szse_status(contract, master_path)
    except Exception as exc:
        receipt["szse_status"] = {"status": "DOWNLOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    receipt["sse_status"] = {
        "status": "MISSING_OFFICIAL_HISTORICAL_INTERVAL_SOURCE",
        "path": contract["sources"]["sse_status_intervals"]["staging_path"],
        "historical_current_name_backfill_used": False,
    }
    receipt_path = output_path(contract["outputs"]["collection_receipt"])
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"状态": "暂存层构建完成", "回执": str(receipt_path.relative_to(ROOT)), "上海状态": receipt["sse_status"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
