from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FX_DIR = PROJECT_ROOT / "data" / "raw" / "a_share_hs_rmb_fxv_inputs_v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "global_liquidity_shock_v0"

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_PAGE_URL = "https://www.cboe.com/tradable_products/vix/vix_historical_data"
CFETS_SOURCE_PAGE = "https://www.chinamoney.com.cn/english/cfxerw/"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_vix(max_attempts: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            VIX_URL,
            headers={"User-Agent": "510300-research/1.0 (+point-in-time audit)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
            if len(payload) < 10_000:
                raise ValueError(f"VIX 下载内容异常短：{len(payload)} 字节")
            if not payload.lstrip(b"\xef\xbb\xbf").startswith(b"DATE,OPEN,HIGH,LOW,CLOSE"):
                raise ValueError("VIX CSV 表头不符合预期")
            return payload, content_type
        except Exception as exc:  # 网络失败需要保留最后一个明确错误
            last_error = exc
            if attempt < max_attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"VIX 官方 CSV 下载失败，共尝试 {max_attempts} 次") from last_error


def parse_vix(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    expected = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    if list(frame.columns) != expected:
        raise ValueError(f"VIX CSV 字段漂移：{list(frame.columns)}")

    frame = frame.rename(columns={column: column.lower() for column in expected})
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="raise")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        duplicated = frame.loc[frame["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"VIX CSV 存在重复日期：{duplicated.head(10).tolist()}")
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("VIX CSV 存在空 OHLC")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("VIX CSV 存在非正 OHLC")
    high_consistent = frame["high"] >= frame[["open", "close", "low"]].max(axis=1)
    low_consistent = frame["low"] <= frame[["open", "close", "high"]].min(axis=1)
    frame["ohlc_consistent"] = high_consistent & low_consistent

    frame["symbol"] = "VIX"
    frame["source"] = "cboe.official.daily_prices"
    frame["decision_time_rule"] = "仅可用于其美国收盘之后的中国决策时点"
    return frame


def parse_cfets() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source_files = sorted(SOURCE_FX_DIR.glob("chinamoney_ccpr_*.json"))
    if not source_files:
        raise FileNotFoundError(f"未找到 CFETS 原始文件：{SOURCE_FX_DIR}")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in source_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        request_currencies = [
            item.strip() for item in str(payload["request"]["currency"]).split(",") if item.strip()
        ]
        if not request_currencies:
            raise ValueError(f"CFETS 文件未声明币种：{path.name}")

        file_rows = 0
        for page in payload.get("pages", []):
            page_currencies = page.get("data", {}).get("searchlist") or request_currencies
            if list(page_currencies) != request_currencies:
                raise ValueError(f"CFETS 分页币种顺序漂移：{path.name}")
            for record in page.get("records", []):
                values = record.get("values", [])
                if len(values) != len(request_currencies):
                    raise ValueError(
                        f"CFETS 数值数量与币种数量不一致：{path.name} {record.get('date')}"
                    )
                row: dict[str, Any] = {"date": record["date"]}
                row.update(dict(zip(request_currencies, values, strict=True)))
                rows.append(row)
                file_rows += 1

        declared_count = int(payload.get("raw_record_count", file_rows))
        if file_rows != declared_count:
            raise ValueError(
                f"CFETS 完整记录数不一致：{path.name}，解析 {file_rows}，声明 {declared_count}"
            )
        provenance.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "retrieved_at": payload.get("retrieved_at"),
                "source": payload.get("source"),
                "first_date": payload.get("first_date"),
                "last_date": payload.get("last_date"),
                "records": file_rows,
                "complete_records": int(payload.get("complete_record_count", file_rows)),
            }
        )

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    numeric_columns = [column for column in frame.columns if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column].replace({"---": pd.NA, "--": pd.NA, "": pd.NA}), errors="coerce"
        )

    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        duplicated = frame.loc[frame["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"CFETS 合并数据存在重复日期：{duplicated.head(10).tolist()}")
    required_complete_columns = ["USD/CNY", "EUR/CNY", "100JPY/CNY", "GBP/CNY", "SGD/CNY"]
    incomplete_required = {
        column: int(frame[column].isna().sum())
        for column in required_complete_columns
        if frame[column].isna().any()
    }
    if incomplete_required:
        raise ValueError(f"CFETS 主要中间价存在空值：{incomplete_required}")

    rename_map = {
        "USD/CNY": "usd_cny",
        "EUR/CNY": "eur_cny",
        "100JPY/CNY": "jpy100_cny",
        "GBP/CNY": "gbp_cny",
        "SGD/CNY": "sgd_cny",
        "CNY/KRW": "cny_krw",
    }
    missing = sorted(set(rename_map) - set(frame.columns))
    if missing:
        raise ValueError(f"CFETS 合并数据缺少冻结币种：{missing}")
    frame = frame[["date", *rename_map]].rename(columns=rename_map)
    frame["source"] = "cfets.official.central_parity"
    frame["publication_rule"] = "中国外汇交易中心于每个工作日银行间市场开盘前公布"
    return frame, provenance


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    vix_payload, content_type = download_vix()
    vix_raw_path = OUTPUT_DIR / "VIX_History.csv"
    atomic_write_bytes(vix_raw_path, vix_payload)
    vix_frame = parse_vix(vix_payload)
    vix_parquet_path = OUTPUT_DIR / "vix_daily.parquet"
    atomic_write_parquet(vix_parquet_path, vix_frame)

    cfets_frame, cfets_provenance = parse_cfets()
    cfets_parquet_path = OUTPUT_DIR / "cfets_rmb_central_parity_daily.parquet"
    atomic_write_parquet(cfets_parquet_path, cfets_frame)

    metadata = {
        "schema_version": "0.1.0",
        "dataset_id": "GLOBAL_LIQUIDITY_SHOCK_INPUTS_V0",
        "status": "DISCOVERY_INPUT_ONLY",
        "retrieved_at_utc": retrieved_at,
        "vix": {
            "source_url": VIX_URL,
            "source_page": VIX_PAGE_URL,
            "content_type": content_type,
            "raw_file": str(vix_raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "raw_sha256": sha256_bytes(vix_payload),
            "parquet_file": str(vix_parquet_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "parquet_sha256": sha256_file(vix_parquet_path),
            "rows": int(len(vix_frame)),
            "ohlc_inconsistent_rows": int((~vix_frame["ohlc_consistent"]).sum()),
            "research_field": "close",
            "first_date": vix_frame["date"].min().strftime("%Y-%m-%d"),
            "last_date": vix_frame["date"].max().strftime("%Y-%m-%d"),
            "point_in_time_rule": "对中国交易日 t，只能连接严格早于 t 中国收盘时点的 VIX 观测。",
        },
        "cfets": {
            "source_page": CFETS_SOURCE_PAGE,
            "input_files": cfets_provenance,
            "parquet_file": str(cfets_parquet_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "parquet_sha256": sha256_file(cfets_parquet_path),
            "rows": int(len(cfets_frame)),
            "missing_by_column": {
                column: int(cfets_frame[column].isna().sum())
                for column in cfets_frame.columns
                if int(cfets_frame[column].isna().sum()) > 0
            },
            "first_date": cfets_frame["date"].min().strftime("%Y-%m-%d"),
            "last_date": cfets_frame["date"].max().strftime("%Y-%m-%d"),
            "point_in_time_rule": "同日中间价在银行间市场开盘前公布，可用于同日收盘决策。",
        },
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    atomic_write_json(metadata_path, metadata)

    print(
        "全球流动性冲击输入已整理："
        f"VIX {len(vix_frame)} 行（{metadata['vix']['first_date']} 至 {metadata['vix']['last_date']}），"
        f"CFETS {len(cfets_frame)} 行（{metadata['cfets']['first_date']} 至 {metadata['cfets']['last_date']}）。"
    )
    print(f"元数据：{metadata_path}")


if __name__ == "__main__":
    main()
