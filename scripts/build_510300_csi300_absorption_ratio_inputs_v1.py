"""构建510300沪深300PCA吸收率V1纯因子输入。

本程序只读取000300文件的日期列作为交易日历，不读取指数开高低收，
不读取510300行情，不计算任何候选市场结果或组合收益。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from csi300_absorption_ratio_timing_v1 import (  # noqa: E402
    ContractError,
    load_config,
    project_path,
    sha256_file,
)
from scripts.download_000300_constituent_daily_sina import (  # noqa: E402
    AMOUNT_URL,
    HFQ_URL,
    HIST_URL,
    _fetch_text,
    _parse_downloaded_payloads,
    _to_sina_symbol,
)


AUDIT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "510300_csi300_absorption_ratio_timing_v1_inputs.json"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _raw_paths(root: Path, symbol: str) -> dict[str, Path]:
    stem = symbol.replace(".", "_")
    return {
        "history": root / "raw" / f"{stem}.history.txt",
        "amount": root / "raw" / f"{stem}.amount.txt",
        "factor": root / "raw" / f"{stem}.factor.txt",
    }


def _parsed_paths(root: Path, symbol: str) -> tuple[Path, Path]:
    stem = symbol.replace(".", "_")
    return root / "parsed" / f"{stem}.parquet", root / "parsed" / f"{stem}.json"


def _raw_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def _raw_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _normalize_member_flags(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized.dropna().unique()).difference(allowed))
    if unexpected:
        raise ContractError(f"{label}存在无法识别的成分标记：{unexpected[:5]}")
    return normalized.isin({"true", "1"})


def load_sanitized_membership(config: dict[str, Any]) -> pd.DataFrame:
    specification = config["inputs"]["historical_membership"]
    frame = pd.read_parquet(project_path(specification["path"]))
    missing = sorted(set(specification["required_columns"]).difference(frame.columns))
    if missing:
        raise ContractError(f"历史成分区间缺少字段：{missing}")
    frame["opt_in"] = pd.to_datetime(frame["opt_in"], errors="coerce")
    frame["opt_out"] = pd.to_datetime(frame["opt_out"], errors="coerce")
    unresolved = frame.loc[frame["opt_in"].isna(), ["symbol", "opt_out"]].copy()
    observed = [
        {
            "symbol": str(row.symbol),
            "opt_out": pd.Timestamp(row.opt_out).date().isoformat(),
        }
        for row in unresolved.sort_values(["symbol", "opt_out"]).itertuples(index=False)
    ]
    if observed != config["data_contract"]["expected_unresolved_null_opt_in_intervals"]:
        raise ContractError("历史成分空纳入日期记录不符合冻结的4条未决区间")
    frame = frame.loc[frame["opt_in"].notna()].copy()
    for correction in config["data_contract"][
        "pre_factor_membership_interval_corrections"
    ]:
        mask = (
            frame["symbol"].astype(str).eq(str(correction["symbol"]))
            & frame["opt_in"].eq(pd.Timestamp(correction["original_opt_in"]))
            & frame["opt_out"].eq(pd.Timestamp(correction["original_opt_out"]))
        )
        if int(mask.sum()) != 1:
            raise ContractError(
                f"{correction['symbol']}待纠错成员区间不是唯一精确匹配"
            )
        frame.loc[mask, "opt_out"] = pd.Timestamp(correction["corrected_opt_out"])
    if frame[["symbol", "opt_in"]].isna().any().any():
        raise ContractError("清理后的历史成分仍有关键空值")
    if frame.duplicated(["symbol", "opt_in", "opt_out"]).any():
        raise ContractError("历史成分区间重复")
    return frame


def _load_panel_member_map(
    path: Path,
    label: str,
) -> dict[pd.Timestamp, list[str]]:
    frame = pd.read_parquet(path, columns=["date", "con_code", "is_index_member"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["con_code"].isna().any():
        raise ContractError(f"{label}日期或证券代码无效")
    frame["is_index_member"] = _normalize_member_flags(frame["is_index_member"], label)
    active = frame.loc[frame["is_index_member"], ["date", "con_code"]].copy()
    return {
        pd.Timestamp(date): sorted(group["con_code"].astype(str).unique())
        for date, group in active.groupby("date", sort=True)
    }


def _interval_members(membership: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    active = membership.loc[
        (membership["opt_in"] <= date)
        & (membership["opt_out"].isna() | (membership["opt_out"] > date)),
        "symbol",
    ]
    return sorted(active.astype(str).unique())


def build_daily_membership(
    calendar: pd.DatetimeIndex,
    membership: pd.DataFrame,
    external_map: dict[pd.Timestamp, list[str]],
    current_map: dict[pd.Timestamp, list[str]],
    config: dict[str, Any],
) -> tuple[dict[pd.Timestamp, list[str]], dict[pd.Timestamp, str]]:
    output: dict[pd.Timestamp, list[str]] = {}
    sources: dict[pd.Timestamp, str] = {}
    segments = config["data_contract"]["membership_segments"]
    for date in calendar:
        current = pd.Timestamp(date)
        if current <= pd.Timestamp(segments["HISTORICAL_INTERVALS"]["last_date"]):
            members = _interval_members(membership, current)
            minimum = int(segments["HISTORICAL_INTERVALS"]["minimum_active_members"])
            maximum = int(segments["HISTORICAL_INTERVALS"]["maximum_active_members"])
            if not minimum <= len(members) <= maximum:
                raise ContractError(
                    f"{current.date()}历史区间成分数{len(members)}不在{minimum}至{maximum}"
                )
            source = "HISTORICAL_INTERVALS"
        elif current <= pd.Timestamp(segments["EXTERNAL_POINT_IN_TIME_PANEL"]["last_date"]):
            if current not in external_map:
                raise ContractError(f"{current.date()}缺少外部点时成分面板")
            members = external_map[current]
            expected = int(segments["EXTERNAL_POINT_IN_TIME_PANEL"]["exact_active_members"])
            if len(members) != expected:
                raise ContractError(f"{current.date()}外部点时成分数不是{expected}")
            source = "EXTERNAL_POINT_IN_TIME_PANEL"
        else:
            if current not in current_map:
                raise ContractError(f"{current.date()}缺少当前点时成分面板")
            members = current_map[current]
            expected = int(segments["CURRENT_POINT_IN_TIME_PANEL"]["exact_active_members"])
            if len(members) != expected:
                raise ContractError(f"{current.date()}当前点时成分数不是{expected}")
            source = "CURRENT_POINT_IN_TIME_PANEL"
        output[current] = members
        sources[current] = source
    return output, sources


def load_calendar_dates_only(config: dict[str, Any]) -> pd.DatetimeIndex:
    specification = config["inputs"]["csi300_price_index"]
    frame = pd.read_parquet(project_path(specification["path"]), columns=["date"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ContractError("交易日历日期无效或重复")
    start = pd.Timestamp(config["dates"]["constituent_price_history_start"])
    end = pd.Timestamp(config["dates"]["factor_calculation_end"])
    dates = pd.DatetimeIndex(
        frame.loc[frame["date"].between(start, end), "date"].sort_values().unique()
    )
    if len(dates) < 500 + 250:
        raise ContractError("交易日历不足750日，无法完成冻结暖机")
    if dates[0] > start or dates[-1] < end:
        raise ContractError("交易日历没有覆盖冻结因子区间")
    return dates


def load_reused_archive_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ContractError(f"复用源ZIP的CRC失败：{bad}")
        if "source_index.json" not in archive.namelist():
            raise ContractError("复用源ZIP缺少source_index.json")
        document = json.loads(archive.read("source_index.json").decode("utf-8"))
    records = {
        str(record["symbol"]): record
        for record in document.get("records", [])
        if record.get("symbol")
    }
    if not records:
        raise ContractError("复用源ZIP清单没有证券记录")
    return records, document


def validate_official_2013_membership_anchor(
    membership: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["inputs"]["official_2013_membership_snapshot"]
    path = project_path(specification["path"])
    frame = pd.read_excel(path, sheet_name="Index Constituents Data")
    required = {"Date", "Index Code", "Constituent Code", "Exchange"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"2013官方成员快照缺少字段：{missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    snapshot_date = pd.Timestamp(specification["snapshot_date"])
    index_codes = (
        frame["Index Code"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )
    rows = frame.loc[frame["Date"].eq(snapshot_date) & index_codes.eq("000300")].copy()
    official_codes = set(
        rows["Constituent Code"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )
    expected = int(specification["expected_constituents"])
    if len(official_codes) != expected:
        raise ContractError(f"2013官方成员快照不是{expected}只")
    local_codes = {
        symbol[:6] for symbol in _interval_members(membership, snapshot_date)
    }
    if local_codes != official_codes:
        raise ContractError(
            f"纠错后历史区间与2013官方快照集合不一致：缺失{sorted(official_codes-local_codes)}，多余{sorted(local_codes-official_codes)}"
        )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "snapshot_date": snapshot_date.date().isoformat(),
        "constituent_count": len(official_codes),
        "set_difference_count": 0,
        "status": "PASS_EXACT_SET_MATCH_AFTER_600357_CORRECTION",
    }


def _write_raw_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _download_raw_symbol(
    symbol: str,
    paths: dict[str, Path],
    attempts: int,
) -> None:
    sina_symbol = _to_sina_symbol(symbol)
    history = _fetch_text(HIST_URL.format(sina_symbol), attempts=attempts)
    amount = _fetch_text(AMOUNT_URL.format(sina_symbol, sina_symbol), attempts=attempts)
    factor = _fetch_text(HFQ_URL.format(sina_symbol), attempts=attempts)
    _write_raw_atomic(paths["history"], history)
    _write_raw_atomic(paths["amount"], amount)
    _write_raw_atomic(paths["factor"], factor)


def _parsed_checkpoint_valid(
    parsed_path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
) -> bool:
    if not parsed_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_hashes") != expected["source_hashes"]:
            return False
        if metadata.get("start") != expected["start"] or metadata.get("end") != expected["end"]:
            return False
        if metadata.get("parsed_sha256") != sha256_file(parsed_path):
            return False
        frame = pd.read_parquet(parsed_path, columns=["date", "con_code", "total_return_close"])
    except Exception:
        return False
    return bool(
        not frame.empty
        and frame["date"].notna().all()
        and frame["con_code"].notna().all()
        and frame["total_return_close"].notna().all()
        and (pd.to_numeric(frame["total_return_close"], errors="coerce") > 0).all()
    )


def _parse_and_checkpoint(
    symbol: str,
    raw_paths: dict[str, Path],
    parsed_root: Path,
    start: str,
    end: str,
    source_group: str,
) -> dict[str, Any]:
    source_hashes = _raw_hashes(raw_paths)
    parsed_path, metadata_path = _parsed_paths(parsed_root, symbol)
    expected = {"source_hashes": source_hashes, "start": start, "end": end}
    if _parsed_checkpoint_valid(parsed_path, metadata_path, expected):
        frame = pd.read_parquet(parsed_path, columns=["date"])
        return {
            "symbol": symbol,
            "status": "REUSED_PARSED_CHECKPOINT",
            "source_checkpoint_group": source_group,
            "raw_paths": raw_paths,
            "raw_hashes": source_hashes,
            "parsed_path": parsed_path,
            "parsed_sha256": sha256_file(parsed_path),
            "rows": int(len(frame)),
            "first_date": pd.to_datetime(frame["date"]).min().date().isoformat(),
            "last_date": pd.to_datetime(frame["date"]).max().date().isoformat(),
        }
    parsed = _parse_downloaded_payloads(
        symbol,
        raw_paths["history"].read_text(encoding="utf-8"),
        raw_paths["amount"].read_text(encoding="utf-8"),
        raw_paths["factor"].read_text(encoding="utf-8"),
        start,
        end,
    )
    parsed = parsed[["date", "con_code", "total_return_close"]].copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    parsed["total_return_close"] = pd.to_numeric(parsed["total_return_close"], errors="coerce")
    if parsed[["date", "con_code", "total_return_close"]].isna().any().any():
        raise ContractError(f"{symbol}解析结果有关键空值")
    if (parsed["total_return_close"] <= 0).any():
        raise ContractError(f"{symbol}解析结果有非正总收益价格")
    if parsed["date"].duplicated().any():
        raise ContractError(f"{symbol}解析结果日期重复")
    parsed = parsed.sort_values("date").reset_index(drop=True)
    _atomic_parquet(parsed_path, parsed)
    metadata = {
        "symbol": symbol,
        "source_checkpoint_group": source_group,
        "source_hashes": source_hashes,
        "start": start,
        "end": end,
        "rows": int(len(parsed)),
        "first_date": parsed["date"].min().date().isoformat(),
        "last_date": parsed["date"].max().date().isoformat(),
        "parsed_sha256": sha256_file(parsed_path),
    }
    _atomic_json(metadata_path, metadata)
    return {
        "symbol": symbol,
        "status": "PARSED",
        "source_checkpoint_group": source_group,
        "raw_paths": raw_paths,
        "raw_hashes": source_hashes,
        "parsed_path": parsed_path,
        "parsed_sha256": metadata["parsed_sha256"],
        "rows": int(len(parsed)),
        "first_date": metadata["first_date"],
        "last_date": metadata["last_date"],
    }


def process_symbol(
    symbol: str,
    *,
    reused_root: Path,
    incremental_root: Path,
    parsed_root: Path,
    reused_archive_records: dict[str, dict[str, Any]],
    start: str,
    end: str,
    attempts: int,
) -> dict[str, Any]:
    reused_paths = _raw_paths(reused_root, symbol)
    incremental_paths = _raw_paths(incremental_root, symbol)
    download_status = "NOT_REQUIRED"
    if _raw_complete(reused_paths):
        archive_record = reused_archive_records.get(symbol)
        if archive_record is None:
            raise ContractError(f"{symbol}复用原始响应不在封存ZIP清单中")
        observed = _raw_hashes(reused_paths)
        expected = {
            key: str(archive_record.get("sha256", {}).get(key))
            for key in ("history", "amount", "factor")
        }
        if observed != expected:
            raise ContractError(f"{symbol}复用原始响应与封存ZIP清单哈希不一致")
        raw_paths = reused_paths
        source_group = "REUSED_FROZEN_DISPERSION_ARCHIVE"
    else:
        raw_paths = incremental_paths
        source_group = "INCREMENTAL_ABSORPTION_RATIO_ARCHIVE"
        if not _raw_complete(raw_paths):
            _download_raw_symbol(symbol, raw_paths, attempts)
            download_status = "DOWNLOADED"
        else:
            download_status = "REUSED_INCREMENTAL_RAW_CHECKPOINT"
    result = _parse_and_checkpoint(
        symbol,
        raw_paths,
        parsed_root,
        start,
        end,
        source_group,
    )
    result["download_status"] = download_status
    return result


def acquire_and_parse_symbols(
    symbols: list[str],
    config: dict[str, Any],
    reused_archive_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    specification = config["data_contract"]["raw_source"]
    reused_root = project_path(specification["reused_checkpoint_root"])
    incremental_root = project_path(specification["incremental_checkpoint_root"])
    parsed_root = incremental_root
    workers = int(specification["request_workers"])
    attempts = int(specification["request_retry_limit"])
    start = str(specification["start"])
    end = str(specification["end"])
    results: dict[str, dict[str, Any]] = {}
    print(f"吸收率输入需处理{len(symbols)}只点时相关成分，支持断点续传。", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_symbol,
                symbol,
                reused_root=reused_root,
                incremental_root=incremental_root,
                parsed_root=parsed_root,
                reused_archive_records=reused_archive_records,
                start=start,
                end=end,
                attempts=attempts,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                results[symbol] = {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            if completed % 25 == 0 or completed == len(futures):
                failures = sum(row["status"] == "FAILED" for row in results.values())
                downloads = sum(row.get("download_status") == "DOWNLOADED" for row in results.values())
                print(
                    f"成分处理进度{completed}/{len(futures)}，新增下载{downloads}，失败{failures}",
                    flush=True,
                )
    return results


def build_incremental_archive(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    incremental = [
        row
        for row in records.values()
        if row.get("source_checkpoint_group") == "INCREMENTAL_ABSORPTION_RATIO_ARCHIVE"
    ]
    index = {
        "schema_version": "1.0.0",
        "provider": "Sina endpoints exposed by AkShare module constants",
        "history_url_template": HIST_URL,
        "amount_url_template": AMOUNT_URL,
        "factor_url_template": HFQ_URL,
        "tls_certificate_verification": True,
        "records": [
            {
                "symbol": row["symbol"],
                "raw_sha256": row["raw_hashes"],
                "parsed_sha256": row["parsed_sha256"],
                "rows": row["rows"],
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "download_status": row["download_status"],
            }
            for row in sorted(incremental, key=lambda value: value["symbol"])
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(
            "source_index.json",
            json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        for row in sorted(incremental, key=lambda value: value["symbol"]):
            stem = row["symbol"].replace(".", "_")
            for name, raw_path in row["raw_paths"].items():
                archive.write(raw_path, arcname=f"raw/{stem}.{name}.txt")
            archive.write(row["parsed_path"], arcname=f"parsed/{stem}.parquet")
    os.replace(temporary, path)
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        members = len(archive.infolist())
    if bad is not None:
        raise ContractError(f"增量源ZIP的CRC失败：{bad}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "members": int(members),
        "symbol_count": int(len(incremental)),
        "crc_status": "PASS",
    }


def combine_total_return_prices(
    records: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, row in sorted(records.items()):
        frame = pd.read_parquet(
            row["parsed_path"], columns=["date", "con_code", "total_return_close"]
        )
        frame["source_checkpoint_group"] = row["source_checkpoint_group"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["total_return_close"] = pd.to_numeric(
        combined["total_return_close"], errors="coerce"
    )
    if combined[["date", "con_code", "total_return_close"]].isna().any().any():
        raise ContractError("合并后的总收益价格包含关键空值")
    if (combined["total_return_close"] <= 0).any():
        raise ContractError("合并后的总收益价格包含非正值")
    if combined.duplicated(["date", "con_code"]).any():
        raise ContractError("合并后的总收益价格日期证券键重复")
    return combined.sort_values(["date", "con_code"]).reset_index(drop=True)


def exponential_covariance_absorption_ratio(
    returns: np.ndarray,
    *,
    half_life: int,
    eigenvector_count: int,
) -> float:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 5:
        raise ContractError("吸收率收益矩阵形状无效")
    if not np.isfinite(values).all():
        raise ContractError("吸收率收益矩阵包含非有限值")
    ages = np.arange(values.shape[0] - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / float(half_life))
    weights /= weights.sum()
    means = weights @ values
    centered = values - means
    covariance = (centered * np.sqrt(weights)[:, None]).T @ (
        centered * np.sqrt(weights)[:, None]
    )
    covariance = (covariance + covariance.T) / 2.0
    trace = float(np.trace(covariance))
    count = int(values.shape[1])
    top = int(eigenvector_count)
    if trace <= 0 or top < 1 or top >= count:
        raise ContractError("吸收率协方差迹或特征值数量无效")
    largest = eigvalsh(
        covariance,
        subset_by_index=[count - top, count - 1],
        check_finite=False,
        driver="evr",
    )
    ratio = float(np.maximum(largest, 0.0).sum() / trace)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0 + 1e-10:
        raise ContractError("PCA吸收率不在有效区间")
    return min(max(ratio, 0.0), 1.0)


def compute_daily_factor(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    daily_members: dict[pd.Timestamp, list[str]],
    membership_sources: dict[pd.Timestamp, str],
    config: dict[str, Any],
) -> pd.DataFrame:
    symbols = sorted({symbol for members in daily_members.values() for symbol in members})
    expected_symbols = int(
        config["data_contract"]["expected_relevant_symbol_count_after_corrections"]
    )
    if len(symbols) != expected_symbols:
        raise ContractError(
            f"纠错后点时相关证券数量从{expected_symbols}漂移为{len(symbols)}"
        )
    matrix = prices.pivot(index="date", columns="con_code", values="total_return_close")
    matrix = matrix.reindex(index=calendar, columns=symbols).sort_index().ffill()
    log_returns = np.log(matrix).diff()
    window = int(config["factor"]["covariance_window_trading_days"])
    half_life = int(config["factor"]["exponential_half_life_trading_days"])
    minimum_count = int(config["data_contract"]["minimum_valid_member_count"])
    minimum_coverage = float(config["data_contract"]["minimum_valid_member_coverage"])
    column_positions = {symbol: index for index, symbol in enumerate(symbols)}
    values = log_returns.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    factor_dates = calendar[window:]
    for completed, date in enumerate(factor_dates, start=1):
        calendar_position = window + completed - 1
        members = daily_members[pd.Timestamp(date)]
        positions = np.array([column_positions[symbol] for symbol in members], dtype=int)
        trailing = values[calendar_position - window + 1 : calendar_position + 1, positions]
        valid_mask = np.isfinite(trailing).all(axis=0)
        valid_returns = trailing[:, valid_mask]
        valid_count = int(valid_mask.sum())
        active_count = int(len(members))
        coverage = float(valid_count / active_count) if active_count else 0.0
        eigenvectors = int(math.floor(valid_count / 5)) if valid_count else 0
        if valid_count >= minimum_count and coverage >= minimum_coverage:
            ratio = exponential_covariance_absorption_ratio(
                valid_returns,
                half_life=half_life,
                eigenvector_count=eigenvectors,
            )
        else:
            ratio = np.nan
        rows.append(
            {
                "date": pd.Timestamp(date),
                "active_member_count": active_count,
                "valid_member_count": valid_count,
                "valid_member_coverage": coverage,
                "eigenvector_count": eigenvectors,
                "absorption_ratio": ratio,
                "membership_source_segment": membership_sources[pd.Timestamp(date)],
            }
        )
        if completed % 100 == 0 or completed == len(factor_dates):
            observed = pd.DataFrame(rows)
            valid_rows = int(observed["absorption_ratio"].notna().sum())
            minimum_observed = float(observed["valid_member_coverage"].min())
            print(
                f"吸收率计算进度{completed}/{len(factor_dates)}，有效{valid_rows}，最低覆盖{minimum_observed:.4f}",
                flush=True,
            )
    factor = pd.DataFrame(rows)
    short = int(config["factor"]["short_average_trading_days"])
    long = int(config["factor"]["long_average_trading_days"])
    factor["absorption_ratio_mean_15"] = factor["absorption_ratio"].rolling(
        short, min_periods=short
    ).mean()
    factor["absorption_ratio_mean_250"] = factor["absorption_ratio"].rolling(
        long, min_periods=long
    ).mean()
    factor["absorption_ratio_std_250"] = factor["absorption_ratio"].rolling(
        long, min_periods=long
    ).std(ddof=int(config["factor"]["long_standard_deviation_ddof"]))
    denominator = factor["absorption_ratio_std_250"].where(
        factor["absorption_ratio_std_250"] > 0
    )
    factor["standardized_absorption_ratio_shift"] = (
        factor["absorption_ratio_mean_15"] - factor["absorption_ratio_mean_250"]
    ) / denominator
    return factor


def _source_inventory_records(
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": row["symbol"],
            "source_checkpoint_group": row["source_checkpoint_group"],
            "download_status": row["download_status"],
            "parse_status": row["status"],
            "rows": row["rows"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "raw_sha256": row["raw_hashes"],
            "parsed_sha256": row["parsed_sha256"],
        }
        for row in sorted(records.values(), key=lambda value: value["symbol"])
    ]


def build_inputs() -> dict[str, Any]:
    config = load_config()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    calendar = load_calendar_dates_only(config)
    membership = load_sanitized_membership(config)
    official_2013_anchor = validate_official_2013_membership_anchor(membership, config)
    external_map = _load_panel_member_map(
        project_path(config["inputs"]["external_membership_panel"]["path"]),
        "外部点时成分面板",
    )
    current_map = _load_panel_member_map(
        project_path(config["inputs"]["current_membership_panel"]["path"]),
        "当前点时成分面板",
    )
    covariance_window = int(config["factor"]["covariance_window_trading_days"])
    first_possible_factor_date = pd.Timestamp(calendar[covariance_window])
    membership_calendar = calendar[calendar >= first_possible_factor_date]
    daily_members, membership_sources = build_daily_membership(
        membership_calendar, membership, external_map, current_map, config
    )
    symbols = sorted({symbol for members in daily_members.values() for symbol in members})
    reused_archive_path = project_path(config["inputs"]["reused_source_archive"]["path"])
    reused_archive_records, reused_archive_index = load_reused_archive_index(
        reused_archive_path
    )
    print(
        f"点时相关证券共{len(symbols)}只；复用封存ZIP记录{len(reused_archive_records)}只。",
        flush=True,
    )
    records = acquire_and_parse_symbols(symbols, config, reused_archive_records)
    failures = [row for row in records.values() if row["status"] == "FAILED"]
    if failures:
        failure_payload = {
            "schema_version": "1.0.0",
            "project_id": config["protocol"]["project_id"],
            "status": "EXTERNAL_FREE_SOURCE_FAILED",
            "checked_at_asia_shanghai": now,
            "required_symbol_count": len(symbols),
            "failure_count": len(failures),
            "failures": failures,
            "candidate_factor_values_read_before_freeze": False,
            "candidate_market_return_outcomes_read_or_computed_before_freeze": False,
            "candidate_portfolio_returns_read_or_computed_before_freeze": False,
            "live_trading_authorized": False,
        }
        _atomic_json(AUDIT_PATH, failure_payload)
        raise ContractError(f"{len(failures)}只证券原始响应下载或解析失败")

    incremental_archive = build_incremental_archive(
        project_path(config["inputs"]["incremental_source_archive"]["path"]), records
    )
    prices = combine_total_return_prices(records)
    price_path = project_path(config["inputs"]["constituent_total_return_close"]["path"])
    _atomic_parquet(price_path, prices)
    factor = compute_daily_factor(
        prices, calendar, daily_members, membership_sources, config
    )
    required_from = pd.Timestamp(
        config["data_contract"]["required_complete_signal_on_every_trading_day_from"]
    )
    required_dates = calendar[calendar >= required_from]
    required = factor.loc[factor["date"].isin(required_dates)].copy()
    if len(required) != len(required_dates):
        raise ContractError("冻结信号区间没有逐交易日因子行")
    required_numeric = [
        "absorption_ratio",
        "absorption_ratio_mean_15",
        "absorption_ratio_mean_250",
        "absorption_ratio_std_250",
        "standardized_absorption_ratio_shift",
    ]
    if required[required_numeric].isna().any().any():
        missing_dates = required.loc[
            required[required_numeric].isna().any(axis=1), "date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise ContractError(f"冻结信号区间因子不完整：{missing_dates[:10]}")
    if not required["valid_member_count"].ge(
        int(config["data_contract"]["minimum_valid_member_count"])
    ).all():
        raise ContractError("冻结信号区间有效成分数不足285")
    if not required["valid_member_coverage"].ge(
        float(config["data_contract"]["minimum_valid_member_coverage"])
    ).all():
        raise ContractError("冻结信号区间有效成分覆盖率不足95%")
    complete_shift = factor.loc[
        factor["standardized_absorption_ratio_shift"].notna(), "date"
    ]
    if complete_shift.empty:
        raise ContractError("没有形成完整标准化吸收率变化")
    first_complete = pd.Timestamp(complete_shift.min())
    if first_complete > pd.Timestamp(
        config["dates"]["latest_allowed_first_complete_standardized_shift"]
    ):
        raise ContractError("首个完整标准化变化晚于2014-12-31")
    factor_path = project_path(config["inputs"]["daily_absorption_ratio_factor"]["path"])
    _atomic_parquet(factor_path, factor)

    reused_archive_summary = {
        "path": reused_archive_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(reused_archive_path),
        "bytes": int(reused_archive_path.stat().st_size),
        "source_index_record_count": int(len(reused_archive_index.get("records", []))),
        "crc_status": "PASS",
    }
    inventory_path = project_path(config["inputs"]["source_inventory"]["path"])
    inventory = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": "PASS",
        "checked_at_asia_shanghai": now,
        "provider": "Sina endpoints exposed by AkShare module constants",
        "tls_certificate_verification": True,
        "required_symbol_count": int(len(symbols)),
        "reused_frozen_archive_symbol_count": int(
            sum(
                row["source_checkpoint_group"] == "REUSED_FROZEN_DISPERSION_ARCHIVE"
                for row in records.values()
            )
        ),
        "incremental_archive_symbol_count": int(incremental_archive["symbol_count"]),
        "reused_archive": reused_archive_summary,
        "incremental_archive": incremental_archive,
        "records": _source_inventory_records(records),
    }
    _atomic_json(inventory_path, inventory)

    source_segments = {
        name: int(count)
        for name, count in factor["membership_source_segment"].value_counts().items()
    }
    audit = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": "PASS",
        "checked_at_asia_shanghai": now,
        "candidate_factor_values_read_before_freeze": True,
        "candidate_market_return_outcomes_read_or_computed_before_freeze": False,
        "candidate_portfolio_returns_read_or_computed_before_freeze": False,
        "calendar_access": {
            "path": config["inputs"]["csi300_price_index"]["path"],
            "columns_read": ["date"],
            "forbidden_columns_not_read": config["data_contract"][
                "factor_builder_forbidden_columns_before_freeze"
            ],
            "rows": int(len(calendar)),
            "first_date": calendar.min().date().isoformat(),
            "last_date": calendar.max().date().isoformat(),
        },
        "membership": {
            "sanitized_interval_rows": int(len(membership)),
            "unresolved_null_opt_in_rows_excluded_without_imputation": 4,
            "pre_factor_interval_corrections": config["data_contract"][
                "pre_factor_membership_interval_corrections"
            ],
            "price_only_warmup_start": calendar.min().date().isoformat(),
            "first_possible_factor_and_required_membership_date": first_possible_factor_date.date().isoformat(),
            "official_2013_anchor": official_2013_anchor,
            "relevant_symbol_count": int(len(symbols)),
            "daily_active_member_minimum": int(
                min(len(members) for members in daily_members.values())
            ),
            "daily_active_member_maximum": int(
                max(len(members) for members in daily_members.values())
            ),
            "factor_source_segment_rows": source_segments,
        },
        "source_archives": {
            "reused": reused_archive_summary,
            "incremental": incremental_archive,
        },
        "constituent_prices": {
            "path": price_path.relative_to(ROOT).as_posix(),
            "rows": int(len(prices)),
            "symbols": int(prices["con_code"].nunique()),
            "first_date": prices["date"].min().date().isoformat(),
            "last_date": prices["date"].max().date().isoformat(),
            "sha256": sha256_file(price_path),
            "bytes": int(price_path.stat().st_size),
        },
        "factor": {
            "path": factor_path.relative_to(ROOT).as_posix(),
            "rows": int(len(factor)),
            "first_date": factor["date"].min().date().isoformat(),
            "last_date": factor["date"].max().date().isoformat(),
            "first_complete_standardized_shift_date": first_complete.date().isoformat(),
            "required_signal_rows": int(len(required)),
            "minimum_required_valid_member_count": int(required["valid_member_count"].min()),
            "minimum_required_valid_member_coverage": float(
                required["valid_member_coverage"].min()
            ),
            "sha256": sha256_file(factor_path),
            "bytes": int(factor_path.stat().st_size),
        },
        "source_inventory": {
            "path": inventory_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(inventory_path),
            "bytes": int(inventory_path.stat().st_size),
        },
        "historical_availability_status": "HISTORICAL_ARCHIVE_RECONSTRUCTION_NOT_TIMESTAMP_PROOF",
        "return_evaluation": "NOT_ALLOWED_BEFORE_FREEZE",
        "live_trading_authorized": False,
    }
    _atomic_json(AUDIT_PATH, audit)
    return audit


def main() -> int:
    try:
        audit = build_inputs()
    except Exception as exc:
        payload = {
            "status": "NO_VIEW_DATA_OR_PROTOCOL_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_market_return_outcomes_read_or_computed_before_freeze": False,
            "candidate_portfolio_returns_read_or_computed_before_freeze": False,
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "project_id": audit["project_id"],
        "status": audit["status"],
        "relevant_symbol_count": audit["membership"]["relevant_symbol_count"],
        "factor_rows": audit["factor"]["rows"],
        "first_complete_standardized_shift_date": audit["factor"][
            "first_complete_standardized_shift_date"
        ],
        "minimum_required_valid_member_coverage": audit["factor"][
            "minimum_required_valid_member_coverage"
        ],
        "candidate_market_return_outcomes_read_or_computed_before_freeze": False,
        "candidate_portfolio_returns_read_or_computed_before_freeze": False,
        "return_evaluation": "NOT_ALLOWED_BEFORE_FREEZE",
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
