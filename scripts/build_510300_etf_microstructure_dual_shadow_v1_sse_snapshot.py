"""用上交所官方份额和融资融券端点补全双影子V1快照。

输入为既有只追加快照。行情、H00300和双来源净值保持该快照原值；份额与
融资融券逐交易日从上交所官方端点重新采集。重叠日期必须与原TuShare记录
一致，随后才生成新的带哈希快照。该程序不改动冻结输入或已有快照。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
SNAPSHOT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "snapshots"
)
SSE_QUERY_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_MARGIN_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SSE_SHARE_SQL_ID = "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"
HEADERS = {
    "Referer": "https://www.sse.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    final_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("上交所端点返回的JSON顶层不是对象")
            return payload
        except Exception as exc:
            final_error = exc
            if attempt < attempts:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"上交所端点请求失败：{type(final_error).__name__}: {final_error}")


def _fetch_share(
    session: requests.Session, date: pd.Timestamp
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _request_json(
        session,
        SSE_QUERY_URL,
        {
            "isPagination": "true",
            "pageHelp.pageSize": "10000",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
            "pageHelp.endPage": "1",
            "sqlId": SSE_SHARE_SQL_ID,
            "STAT_DATE": date.strftime("%Y-%m-%d"),
        },
    )
    records = payload.get("result") or []
    target = [record for record in records if str(record.get("SEC_CODE")) == "510300"]
    if len(target) != 1:
        raise RuntimeError(f"{date.date()}上交所ETF份额未返回唯一510300记录")
    source = target[0]
    if str(source.get("STAT_DATE")) != date.strftime("%Y-%m-%d"):
        raise RuntimeError(f"{date.date()}上交所ETF份额日期不匹配")
    share_10k = float(source["TOT_VOL"])
    if not np.isfinite(share_10k) or share_10k <= 0:
        raise RuntimeError(f"{date.date()}上交所ETF总份额无效")
    record = {
        "date": date,
        "ts_code": "510300.SH",
        "fund_share_10k": share_10k,
        "fund_shares": share_10k * 10_000.0,
        "fund_type": "ETF",
        "market": "SH",
        "source": f"sse.{SSE_SHARE_SQL_ID}",
    }
    return record, payload


def _fetch_margin(
    session: requests.Session,
    date: pd.Timestamp,
    close: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _request_json(
        session,
        SSE_MARGIN_URL,
        {
            "isPagination": "true",
            "tabType": "mxtype",
            "detailsDate": date.strftime("%Y%m%d"),
            "stockCode": "510300",
            "beginDate": "",
            "endDate": "",
            "pageHelp.pageSize": "5000",
            "pageHelp.pageCount": "50",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
            "pageHelp.endPage": "21",
        },
    )
    records = payload.get("result") or []
    target = [record for record in records if str(record.get("stockCode")) == "510300"]
    if len(target) != 1:
        raise RuntimeError(f"{date.date()}上交所融资融券未返回唯一510300记录")
    source = target[0]
    if str(source.get("opDate")) != date.strftime("%Y%m%d"):
        raise RuntimeError(f"{date.date()}上交所融资融券日期不匹配")
    numeric = {
        name: float(source[name])
        for name in ["rzye", "rzmre", "rzche", "rqyl", "rqchl", "rqmcl"]
    }
    if any(not np.isfinite(value) or value < 0 for value in numeric.values()):
        raise RuntimeError(f"{date.date()}上交所融资融券字段无效")
    rqye = numeric["rqyl"] * float(close)
    record = {
        "date": date,
        "ts_code": "510300.SH",
        "rzye": numeric["rzye"],
        "rqye": rqye,
        "rzmre": numeric["rzmre"],
        "rqyl": numeric["rqyl"],
        "rzche": numeric["rzche"],
        "rqchl": numeric["rqchl"],
        "rqmcl": numeric["rqmcl"],
        "rzrqye": numeric["rzye"] + rqye,
        "financing_net_buy_cny": numeric["rzmre"] - numeric["rzche"],
        "source": "sse.queryMargin.do;rqye=rqyl*510300_close",
    }
    return record, payload


def _load_base_snapshot(
    base_dir: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = base_dir / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ValueError("基础快照项目编号不匹配")
    frames: dict[str, pd.DataFrame] = {}
    for name in ["etf", "benchmark", "nav", "fund_share", "margin"]:
        record = manifest["datasets"][name]
        path = PROJECT_ROOT / record["file"]
        if not path.resolve().is_relative_to(base_dir.resolve()):
            raise ValueError(f"基础快照文件越出快照目录：{path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"基础快照文件哈希漂移：{name}")
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frames[name] = frame
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest, frames


def _overlap_audit(
    official_shares: pd.DataFrame,
    official_margin: pd.DataFrame,
    base_shares: pd.DataFrame,
    base_margin: pd.DataFrame,
) -> dict[str, Any]:
    share_overlap = official_shares.merge(
        base_shares[["date", "fund_shares"]],
        on="date",
        how="inner",
        suffixes=("_sse", "_base"),
    )
    share_difference = (
        share_overlap["fund_shares_sse"] - share_overlap["fund_shares_base"]
    ).abs()
    if not share_overlap.empty and not share_difference.eq(0.0).all():
        raise ValueError("上交所份额与基础快照重叠记录不一致")

    margin_columns = ["rzye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl"]
    margin_overlap = official_margin.merge(
        base_margin[["date", *margin_columns, "rqye", "rzrqye"]],
        on="date",
        how="inner",
        suffixes=("_sse", "_base"),
    )
    maximum_difference: dict[str, float] = {}
    for column in margin_columns:
        difference = (
            margin_overlap[f"{column}_sse"] - margin_overlap[f"{column}_base"]
        ).abs()
        maximum_difference[column] = float(difference.max()) if not difference.empty else 0.0
        if not difference.empty and not difference.eq(0.0).all():
            raise ValueError(f"上交所融资融券字段{column}与基础快照不一致")
    for column in ["rqye", "rzrqye"]:
        difference = (
            margin_overlap[f"{column}_sse"] - margin_overlap[f"{column}_base"]
        ).abs()
        maximum_difference[column] = float(difference.max()) if not difference.empty else 0.0
        if not difference.empty and not difference.le(1e-4).all():
            raise ValueError(f"上交所推导字段{column}与基础快照差异超容差")
    return {
        "share_overlap_rows": int(len(share_overlap)),
        "share_maximum_absolute_difference": (
            float(share_difference.max()) if not share_difference.empty else None
        ),
        "margin_overlap_rows": int(len(margin_overlap)),
        "margin_maximum_absolute_difference": maximum_difference,
        "rqye_derivation": "上交所融券余量乘以同日510300收盘价；重叠区与TuShare融券余额容差1e-4内一致",
    }


def _dataset_summary(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "columns": list(frame.columns),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="用上交所官方端点补全双影子V1快照")
    parser.add_argument("--base-snapshot-dir", required=True)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    retrieved_at = datetime.now(TIMEZONE)
    run_id = arguments.run_id or retrieved_at.strftime("%Y%m%dT%H%M%S%z") + "_sse"
    base_dir = Path(arguments.base_snapshot_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    base_manifest, frames = _load_base_snapshot(base_dir.resolve())
    target_dir = SNAPSHOT_ROOT / run_id
    if target_dir.exists():
        raise FileExistsError(f"目标快照目录已存在，拒绝覆盖：{target_dir}")
    target_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = target_dir / "raw_official_sse"
    raw_dir.mkdir(parents=True, exist_ok=False)

    trading = frames["etf"][["date", "close"]].copy()
    if trading.empty:
        raise ValueError("基础快照没有510300交易日")
    share_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    raw_hashes: list[dict[str, Any]] = []
    with requests.Session() as session:
        for _, market_row in trading.iterrows():
            date = pd.Timestamp(market_row["date"])
            share, share_raw = _fetch_share(session, date)
            margin, margin_raw = _fetch_margin(session, date, float(market_row["close"]))
            share_rows.append(share)
            margin_rows.append(margin)
            for kind, raw in [("fund_share", share_raw), ("margin", margin_raw)]:
                raw_path = raw_dir / f"{kind}_{date.strftime('%Y%m%d')}.json"
                _atomic_json(raw, raw_path)
                raw_hashes.append(
                    {
                        "kind": kind,
                        "date": date.date().isoformat(),
                        "file": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "sha256": _sha256(raw_path),
                    }
                )
            time.sleep(0.15)

    official_shares = pd.DataFrame(share_rows)
    official_margin = pd.DataFrame(margin_rows)
    for frame in [official_shares, official_margin]:
        frame["retrieved_at"] = retrieved_at.isoformat()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    overlap = _overlap_audit(
        official_shares,
        official_margin,
        frames["fund_share"],
        frames["margin"],
    )
    frames["fund_share"] = official_shares
    frames["margin"] = official_margin

    paths = {
        "etf": target_dir / "510300_daily.parquet",
        "benchmark": target_dir / "H00300_total_return_daily.parquet",
        "nav": target_dir / "510300_nav_daily.parquet",
        "fund_share": target_dir / "510300_fund_share_daily.parquet",
        "margin": target_dir / "510300_margin_detail_daily.parquet",
    }
    for name, path in paths.items():
        _atomic_parquet(frames[name], path)
    dates = [date.date().isoformat() for date in trading["date"]]
    payload = {
        "project_id": PROJECT_ID,
        "status": "PASS_COMPLETE_ROWS_AVAILABLE",
        "run_id": run_id,
        "retrieved_at_asia_shanghai": retrieved_at.isoformat(),
        "requested_start": trading["date"].min().date().isoformat(),
        "requested_end": trading["date"].max().date().isoformat(),
        "frozen_files_untouched": True,
        "base_snapshot": {
            "run_id": base_manifest["run_id"],
            "manifest_sha256": base_manifest["manifest_sha256"],
        },
        "signal_calendar": {
            "etf_trading_dates": dates,
            "complete_signal_dates": dates,
            "missing_by_source": {"nav": [], "fund_share": [], "margin": []},
        },
        "execution_market": {"missing_h00300_dates_vs_etf": []},
        "collection_audit": {
            "official_sources": {
                "fund_share": f"sse.{SSE_SHARE_SQL_ID}",
                "margin": "sse.queryMargin.do",
            },
            "official_raw_response_count": int(len(raw_hashes)),
            "official_raw_responses": raw_hashes,
            "overlap_crosscheck": overlap,
        },
        "datasets": {
            name: _dataset_summary(frames[name], paths[name]) for name in paths
        },
        "boundaries": {
            "research_snapshot_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    manifest_path = target_dir / "snapshot_manifest.json"
    _atomic_json(payload, manifest_path)
    print(
        json.dumps(
            {**payload, "snapshot_manifest": str(manifest_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
