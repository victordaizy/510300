"""构建510300横截面收益离散度V1的纯输入，不计算未来波动或组合收益。"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo
import zipfile

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from csi300_return_dispersion_volatility_v1 import (  # noqa: E402
    CONFIG_PATH,
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
    / "510300_csi300_return_dispersion_volatility_inputs_v1.json"
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


def _checkpoint_paths(root: Path, symbol: str) -> dict[str, Path]:
    stem = symbol.replace(".", "_")
    return {
        "parsed": root / "parsed" / f"{stem}.parquet",
        "history": root / "raw" / f"{stem}.history.txt",
        "amount": root / "raw" / f"{stem}.amount.txt",
        "factor": root / "raw" / f"{stem}.factor.txt",
    }


def _checkpoint_is_complete(paths: dict[str, Path]) -> bool:
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return False
    try:
        frame = pd.read_parquet(paths["parsed"], columns=["date", "con_code", "total_return_close"])
    except Exception:
        return False
    return bool(
        not frame.empty
        and frame["date"].notna().all()
        and frame["con_code"].notna().all()
        and frame["total_return_close"].notna().all()
    )


def _download_early_symbol(
    symbol: str,
    *,
    start: str,
    end: str,
    checkpoint_root: Path,
    attempts: int,
) -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_root, symbol)
    if _checkpoint_is_complete(paths):
        parsed = pd.read_parquet(paths["parsed"], columns=["date"])
        return {
            "symbol": symbol,
            "status": "REUSED_CHECKPOINT",
            "rows": int(len(parsed)),
            "first_date": str(pd.to_datetime(parsed["date"]).min().date()),
            "last_date": str(pd.to_datetime(parsed["date"]).max().date()),
            "paths": {name: path for name, path in paths.items()},
        }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    sina_symbol = _to_sina_symbol(symbol)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            history_text = _fetch_text(HIST_URL.format(sina_symbol), attempts=1)
            amount_text = _fetch_text(
                AMOUNT_URL.format(sina_symbol, sina_symbol), attempts=1
            )
            factor_text = _fetch_text(HFQ_URL.format(sina_symbol), attempts=1)
            parsed = _parse_downloaded_payloads(
                symbol,
                history_text,
                amount_text,
                factor_text,
                start,
                end,
            )
            temporary_raw: dict[str, Path] = {}
            for name, content in (
                ("history", history_text),
                ("amount", amount_text),
                ("factor", factor_text),
            ):
                temporary = paths[name].with_suffix(paths[name].suffix + ".tmp")
                temporary.write_text(content, encoding="utf-8")
                temporary_raw[name] = temporary
            temporary_parsed = paths["parsed"].with_suffix(".parquet.tmp")
            parsed.to_parquet(temporary_parsed, index=False, engine="pyarrow")
            for name, temporary in temporary_raw.items():
                os.replace(temporary, paths[name])
            os.replace(temporary_parsed, paths["parsed"])
            return {
                "symbol": symbol,
                "status": "DOWNLOADED",
                "rows": int(len(parsed)),
                "first_date": str(pd.to_datetime(parsed["date"]).min().date()),
                "last_date": str(pd.to_datetime(parsed["date"]).max().date()),
                "paths": {name: path for name, path in paths.items()},
            }
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    return {
        "symbol": symbol,
        "status": "FAILED",
        "error_type": type(error).__name__ if error is not None else "UNKNOWN",
        "error": str(error) if error is not None else "未知下载失败",
        "paths": {name: path for name, path in paths.items()},
    }


def download_early_components(
    membership: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], Path]:
    specification = config["data_contract"]["early_download"]
    start = pd.Timestamp(specification["start"])
    end = pd.Timestamp(specification["end"])
    relevant = membership.loc[
        (membership["opt_in"] <= end)
        & (membership["opt_out"].isna() | (membership["opt_out"] > start))
    ]
    symbols = sorted(relevant["symbol"].astype(str).unique())
    if not symbols:
        raise ContractError("早期区间没有相关沪深300成分")
    checkpoint_root = project_path(specification["checkpoint_root"])
    workers = int(specification["request_workers"])
    attempts = int(specification["request_retry_limit"])
    print(f"早期离散度输入需处理{len(symbols)}只历史成分，支持断点续传。", flush=True)
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_early_symbol,
                symbol,
                start=str(start.date()),
                end=str(end.date()),
                checkpoint_root=checkpoint_root,
                attempts=attempts,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:  # pragma: no cover - 并发任务最终保护
                results[symbol] = {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "paths": _checkpoint_paths(checkpoint_root, symbol),
                }
            if completed % 25 == 0 or completed == len(futures):
                successes = sum(
                    row["status"] in {"DOWNLOADED", "REUSED_CHECKPOINT"}
                    for row in results.values()
                )
                failures = sum(row["status"] == "FAILED" for row in results.values())
                print(
                    f"早期成分进度{completed}/{len(futures)}，成功{successes}，失败{failures}",
                    flush=True,
                )
    return results, checkpoint_root


def acquire_csi300_price_index(config: dict[str, Any]) -> pd.DataFrame:
    specification = config["data_contract"]["csi300_index"]
    provider_symbol = specification["provider_symbol"]
    raw = ak.stock_zh_index_daily(symbol=provider_symbol)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ContractError(f"沪深300价格指数接口缺少字段：{missing}")
    frame = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    start = pd.Timestamp(config["dates"]["constituent_history_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    frame = frame.loc[frame["date"].between(start, end)].copy().reset_index(drop=True)
    if frame.empty:
        raise ContractError("沪深300价格指数下载结果为空")
    if frame["date"].iloc[0] > pd.Timestamp(specification["required_first_date_on_or_before"]):
        raise ContractError("沪深300价格指数起点过晚")
    if frame["date"].iloc[-1] < pd.Timestamp(specification["required_last_date_on_or_after"]):
        raise ContractError("沪深300价格指数终点过早")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("沪深300价格指数含非正价格")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ContractError("沪深300价格指数最高价不满足OHLC约束")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ContractError("沪深300价格指数最低价不满足OHLC约束")
    frame["symbol"] = specification["output_symbol"]
    frame["source"] = specification["source_label"]
    return frame[["date", "symbol", "open", "high", "low", "close", "volume", "source"]]


def _month_end_dates(index_daily: pd.DataFrame) -> dict[str, pd.Timestamp]:
    frame = index_daily[["date"]].copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    return {
        str(month): pd.Timestamp(value)
        for month, value in frame.groupby("month", sort=True)["date"].max().items()
    }


def _active_members(membership: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    active = membership.loc[
        (membership["opt_in"] <= date)
        & (membership["opt_out"].isna() | (membership["opt_out"] > date)),
        "symbol",
    ].astype(str)
    return sorted(active.unique())


def _panel_active_members(frame: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    required = {"date", "con_code", "is_index_member"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"点时成分面板缺少字段：{missing}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ContractError("点时成分面板日期存在空值或无法解析值")
    flags = frame["is_index_member"]
    if not pd.api.types.is_bool_dtype(flags.dtype):
        normalized = flags.astype(str).str.strip().str.lower()
        allowed = {"true", "false", "1", "0"}
        unexpected = sorted(set(normalized.dropna().unique()).difference(allowed))
        if unexpected:
            raise ContractError(f"点时成分标记存在无法识别值：{unexpected[:5]}")
        flags = normalized.isin({"true", "1"})
    else:
        flags = flags.fillna(False)
    members = frame.loc[dates.eq(pd.Timestamp(date)) & flags, "con_code"]
    if members.isna().any():
        raise ContractError(f"{pd.Timestamp(date).date()}点时成分证券代码存在空值")
    return sorted(members.astype(str).unique())


def _panel_active_members_by_date(frame: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    required = {"date", "con_code", "is_index_member"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"点时成分面板缺少字段：{missing}")
    data = frame[["date", "con_code", "is_index_member"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ContractError("点时成分面板日期存在空值或无法解析值")
    flags = data["is_index_member"]
    if not pd.api.types.is_bool_dtype(flags.dtype):
        normalized = flags.astype(str).str.strip().str.lower()
        allowed = {"true", "false", "1", "0"}
        unexpected = sorted(set(normalized.dropna().unique()).difference(allowed))
        if unexpected:
            raise ContractError(f"点时成分标记存在无法识别值：{unexpected[:5]}")
        flags = normalized.isin({"true", "1"})
    else:
        flags = flags.fillna(False)
    active = data.loc[flags, ["date", "con_code"]]
    if active["con_code"].isna().any():
        raise ContractError("点时成分证券代码存在空值")
    return {
        pd.Timestamp(date): sorted(subset["con_code"].astype(str).unique())
        for date, subset in active.groupby("date", sort=False)
    }


def _price_series_by_symbol(frame: pd.DataFrame) -> dict[str, pd.Series]:
    required = {"date", "con_code", "total_return_close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"成分行情缺少字段：{missing}")
    data = frame[["date", "con_code", "total_return_close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["total_return_close"] = pd.to_numeric(
        data["total_return_close"], errors="coerce"
    )
    data = data.dropna(subset=["date", "con_code", "total_return_close"])
    if (data["total_return_close"] <= 0).any():
        raise ContractError("成分后复权收盘价含非正数")
    data = data.sort_values(["con_code", "date"]).drop_duplicates(
        ["con_code", "date"], keep="last"
    )
    return {
        str(symbol): subset.set_index("date")["total_return_close"].sort_index()
        for symbol, subset in data.groupby("con_code", sort=False)
    }


def compute_monthly_dispersion_segment(
    price_frame: pd.DataFrame,
    membership: pd.DataFrame | None,
    month_ends: dict[str, pd.Timestamp],
    *,
    first_month: str,
    last_month: str,
    source_segment: str,
    member_source: str,
    minimum_active_members: int,
    maximum_active_members: int,
    minimum_valid_count: int,
    minimum_coverage: float,
) -> pd.DataFrame:
    if member_source not in {"HISTORICAL_INTERVALS", "PANEL_IS_INDEX_MEMBER"}:
        raise ContractError(f"未知月末成员来源：{member_source}")
    if member_source == "HISTORICAL_INTERVALS" and membership is None:
        raise ContractError("历史区间成员模式缺少成员区间表")
    if minimum_active_members <= 0 or maximum_active_members < minimum_active_members:
        raise ContractError("月末成员数量边界无效")
    prices = _price_series_by_symbol(price_frame)
    panel_members_by_date = (
        _panel_active_members_by_date(price_frame)
        if member_source == "PANEL_IS_INDEX_MEMBER"
        else None
    )
    rows: list[dict[str, Any]] = []
    periods = pd.period_range(first_month, last_month, freq="M")
    for period in periods:
        current_month = str(period)
        previous_month = str(period - 1)
        if current_month not in month_ends or previous_month not in month_ends:
            raise ContractError(f"{source_segment}缺少{previous_month}或{current_month}月末交易日")
        current_end = month_ends[current_month]
        previous_end = month_ends[previous_month]
        if member_source == "HISTORICAL_INTERVALS":
            assert membership is not None
            members = _active_members(membership, current_end)
        else:
            assert panel_members_by_date is not None
            members = panel_members_by_date.get(pd.Timestamp(current_end), [])
        active_member_count = len(members)
        if not minimum_active_members <= active_member_count <= maximum_active_members:
            raise ContractError(
                f"{current_month}月末点时成员数为{active_member_count}，"
                f"不在{minimum_active_members}至{maximum_active_members}范围内"
            )
        returns: list[float] = []
        for symbol in members:
            series = prices.get(symbol)
            if series is None:
                continue
            before = series.loc[:previous_end]
            current = series.loc[:current_end]
            if before.empty or current.empty:
                continue
            start_price = float(before.iloc[-1])
            end_price = float(current.iloc[-1])
            if not math_is_finite_positive(start_price) or not math_is_finite_positive(end_price):
                continue
            value = end_price / start_price - 1.0
            if math_is_finite(value):
                returns.append(float(value))
        valid_count = len(returns)
        coverage = valid_count / active_member_count
        if valid_count < minimum_valid_count or coverage < minimum_coverage:
            raise ContractError(
                f"{current_month}有效成分收益仅{valid_count}/{active_member_count}，覆盖率{coverage:.6f}"
            )
        values = np.asarray(returns, dtype=float)
        dispersion = float(np.std(values, ddof=1))
        if not math_is_finite_positive(dispersion):
            raise ContractError(f"{current_month}横截面离散度无效")
        rows.append(
            {
                "factor_month": current_month,
                "previous_month_end_date": previous_end,
                "month_end_date": current_end,
                "active_member_count": active_member_count,
                "valid_return_count": valid_count,
                "valid_return_coverage": coverage,
                "equal_weight_mean_monthly_return": float(values.mean()),
                "cross_sectional_return_dispersion": dispersion,
                "cross_sectional_return_dispersion_squared": dispersion**2,
                "minimum_valid_member_return": float(values.min()),
                "maximum_valid_member_return": float(values.max()),
                "factor_source_segment": source_segment,
            }
        )
    return pd.DataFrame(rows)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def math_is_finite_positive(value: float) -> bool:
    return math_is_finite(value) and float(value) > 0.0


def _load_successful_early_checkpoints(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, result in sorted(results.items()):
        if result["status"] not in {"DOWNLOADED", "REUSED_CHECKPOINT"}:
            continue
        path = result["paths"]["parsed"]
        frame = pd.read_parquet(
            path, columns=["date", "con_code", "total_return_close"]
        )
        if frame.empty:
            continue
        if not frame["con_code"].astype(str).eq(symbol).all():
            raise ContractError(f"早期检查点{symbol}包含错误证券代码")
        frames.append(frame)
    if not frames:
        raise ContractError("没有任何成功的早期成分行情检查点")
    return pd.concat(frames, ignore_index=True)


def _source_index(
    results: dict[str, dict[str, Any]],
    checkpoint_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for symbol, result in sorted(results.items()):
        row: dict[str, Any] = {
            "symbol": symbol,
            "status": result["status"],
            "rows": result.get("rows"),
            "first_date": result.get("first_date"),
            "last_date": result.get("last_date"),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
        }
        hashes: dict[str, str] = {}
        archive_names: dict[str, str] = {}
        for name, path in result["paths"].items():
            if path.exists():
                hashes[name] = sha256_file(path)
                archive_names[name] = path.relative_to(checkpoint_root).as_posix()
        row["sha256"] = hashes
        row["archive_names"] = archive_names
        records.append(row)
    return {
        "schema_version": "1.0.0",
        "provider": "Sina endpoints exposed by AkShare module constants",
        "history_url_template": HIST_URL,
        "amount_url_template": AMOUNT_URL,
        "factor_url_template": HFQ_URL,
        "tls_certificate_verification": True,
        "records": records,
    }


def build_source_archive(
    archive_path: Path,
    results: dict[str, dict[str, Any]],
    checkpoint_root: Path,
) -> dict[str, Any]:
    source_index = _source_index(results, checkpoint_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "source_index.json",
            json.dumps(source_index, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
        )
        for result in sorted(results.values(), key=lambda row: row["symbol"]):
            for path in sorted(result["paths"].values(), key=lambda value: value.as_posix()):
                if path.exists():
                    archive.write(path, arcname=path.relative_to(checkpoint_root).as_posix())
    os.replace(temporary, archive_path)
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
    if bad_member is not None:
        raise ContractError(f"早期成分源归档CRC失败：{bad_member}")
    return {
        "member_count": int(len(names)),
        "crc_passed": True,
        "source_index_sha256": _sha256_bytes(
            (
                json.dumps(source_index, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        ),
    }


def _panel_snapshot(
    frame: pd.DataFrame,
    *,
    path: Path,
    label: str,
) -> dict[str, Any]:
    required = {
        "date",
        "con_code",
        "is_index_member",
        "total_return_close",
        "price_source",
        "adjustment_source",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"{label}缺少字段：{missing}")
    data = frame[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    active = data.loc[data["is_index_member"].fillna(False)].copy()
    active_counts = active.groupby("date")["con_code"].nunique()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "unique_symbols": int(data["con_code"].nunique()),
        "active_rows": int(len(active)),
        "minimum_active_members_per_day": int(active_counts.min()),
        "maximum_active_members_per_day": int(active_counts.max()),
        "price_sources": sorted(data["price_source"].dropna().astype(str).unique().tolist()),
        "adjustment_sources": sorted(
            data["adjustment_source"].dropna().astype(str).unique().tolist()
        ),
    }


def build_inputs() -> dict[str, Any]:
    config = load_config()
    inputs = config["inputs"]
    membership_path = project_path(inputs["historical_membership"]["path"])
    membership_raw = pd.read_parquet(membership_path)
    required_membership = set(inputs["historical_membership"]["required_columns"])
    missing_membership = sorted(required_membership.difference(membership_raw.columns))
    if missing_membership:
        raise ContractError(f"历史成员缺少字段：{missing_membership}")
    membership_raw["opt_in"] = pd.to_datetime(membership_raw["opt_in"], errors="coerce")
    membership_raw["opt_out"] = pd.to_datetime(membership_raw["opt_out"], errors="coerce")
    if membership_raw["symbol"].isna().any():
        raise ContractError("历史成员存在证券代码空值")
    contract = config["data_contract"]
    if contract["historical_membership_null_opt_in_policy"] != (
        "EXCLUDE_UNRESOLVED_SOURCE_ROWS_NO_IMPUTATION"
    ):
        raise ContractError("历史成员空纳入日期处理政策漂移")
    unresolved = membership_raw.loc[
        membership_raw["opt_in"].isna(), ["symbol", "opt_out"]
    ].copy()
    unresolved_records = sorted(
        [
            {
                "symbol": str(row.symbol),
                "opt_out": str(pd.Timestamp(row.opt_out).date())
                if pd.notna(row.opt_out)
                else None,
            }
            for row in unresolved.itertuples(index=False)
        ],
        key=lambda row: (row["symbol"], str(row["opt_out"])),
    )
    expected_unresolved = sorted(
        [
            {"symbol": str(row["symbol"]), "opt_out": str(row["opt_out"])}
            for row in contract["expected_unresolved_null_opt_in_intervals"]
        ],
        key=lambda row: (row["symbol"], row["opt_out"]),
    )
    if unresolved_records != expected_unresolved:
        raise ContractError(
            "历史成员空纳入日期记录与冻结前已审计的4条上游未决区间不一致"
        )
    membership = membership_raw.loc[membership_raw["opt_in"].notna()].copy()

    index_daily = acquire_csi300_price_index(config)
    index_path = project_path(inputs["csi300_price_index"]["path"])
    _atomic_parquet(index_path, index_daily)
    month_ends = _month_end_dates(index_daily)

    early_results, checkpoint_root = download_early_components(membership, config)
    early_prices = _load_successful_early_checkpoints(early_results)

    external_path = project_path(inputs["existing_2014_2021_panel"]["path"])
    late_path = project_path(inputs["existing_2021_2026_sina_panel"]["path"])
    external_panel = pd.read_parquet(external_path)
    late_panel = pd.read_parquet(late_path)
    external_snapshot = _panel_snapshot(
        external_panel, path=external_path, label="2014至2021成分面板"
    )
    late_snapshot = _panel_snapshot(
        late_panel, path=late_path, label="2021至2026新浪备份成分面板"
    )
    external_audit_path = project_path(inputs["existing_2014_2021_panel_audit"]["path"])
    external_audit = json.loads(external_audit_path.read_text(encoding="utf-8"))
    expected_external_hash = (external_audit.get("hashes") or {}).get("panel")
    if expected_external_hash != external_snapshot["sha256"]:
        raise ContractError("2014至2021成分面板与既有审计哈希不一致")
    late_audit_path = project_path(inputs["existing_2021_2026_sina_panel_audit"]["path"])
    late_audit = json.loads(late_audit_path.read_text(encoding="utf-8"))
    if late_audit.get("status") != inputs["existing_2021_2026_sina_panel_audit"][
        "required_status"
    ]:
        raise ContractError("2021至2026新浪备份成分面板既有审计状态不是PASS")
    if late_audit.get("output_sha256") != late_snapshot["sha256"]:
        raise ContractError("2021至2026新浪备份成分面板与既有审计哈希不一致")

    early_minimum_members = int(contract["early_active_member_count_minimum"])
    early_maximum_members = int(contract["early_active_member_count_maximum"])
    post_2015_members = int(contract["post_2015_panel_active_members_per_month_end"])
    minimum_valid_count = int(contract["minimum_valid_member_count"])
    minimum_coverage = float(contract["minimum_valid_member_coverage"])
    segments = contract["factor_segment_boundaries"]
    early_factor = compute_monthly_dispersion_segment(
        early_prices,
        membership,
        month_ends,
        first_month=segments["EARLY_SINA_RECONSTRUCTION"]["first_month"],
        last_month=segments["EARLY_SINA_RECONSTRUCTION"]["last_month"],
        source_segment="EARLY_SINA_RECONSTRUCTION",
        member_source="HISTORICAL_INTERVALS",
        minimum_active_members=early_minimum_members,
        maximum_active_members=early_maximum_members,
        minimum_valid_count=minimum_valid_count,
        minimum_coverage=minimum_coverage,
    )
    external_factor = compute_monthly_dispersion_segment(
        external_panel,
        membership,
        month_ends,
        first_month=segments["EXISTING_SINA_EXTERNAL_PANEL"]["first_month"],
        last_month=segments["EXISTING_SINA_EXTERNAL_PANEL"]["last_month"],
        source_segment="EXISTING_SINA_EXTERNAL_PANEL",
        member_source="PANEL_IS_INDEX_MEMBER",
        minimum_active_members=post_2015_members,
        maximum_active_members=post_2015_members,
        minimum_valid_count=minimum_valid_count,
        minimum_coverage=minimum_coverage,
    )
    late_factor = compute_monthly_dispersion_segment(
        late_panel,
        membership,
        month_ends,
        first_month=segments["EXISTING_SINA_BACKUP_PANEL"]["first_month"],
        last_month=segments["EXISTING_SINA_BACKUP_PANEL"]["last_month"],
        source_segment="EXISTING_SINA_BACKUP_PANEL",
        member_source="PANEL_IS_INDEX_MEMBER",
        minimum_active_members=post_2015_members,
        maximum_active_members=post_2015_members,
        minimum_valid_count=minimum_valid_count,
        minimum_coverage=minimum_coverage,
    )
    factor = pd.concat(
        [early_factor, external_factor, late_factor], ignore_index=True
    ).sort_values("factor_month").reset_index(drop=True)
    expected_months = pd.period_range(
        contract["expected_first_factor_month"],
        contract["expected_last_factor_month"],
        freq="M",
    ).astype(str)
    if factor["factor_month"].tolist() != expected_months.tolist():
        raise ContractError("三段因子拼接后月份不连续或端点不符")
    if len(factor) != int(contract["expected_factor_months"]):
        raise ContractError("三段因子拼接后的行数不符合冻结契约")
    if factor["factor_month"].duplicated().any():
        raise ContractError("三段因子拼接后月份重复")
    factor_path = project_path(inputs["monthly_dispersion_factor"]["path"])
    _atomic_parquet(factor_path, factor)

    archive_path = project_path(inputs["early_component_source_archive"]["path"])
    archive_snapshot = build_source_archive(
        archive_path, early_results, checkpoint_root
    )
    archive_snapshot.update(
        {
            "path": archive_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(archive_path),
            "bytes": int(archive_path.stat().st_size),
        }
    )

    successful = [
        row
        for row in early_results.values()
        if row["status"] in {"DOWNLOADED", "REUSED_CHECKPOINT"}
    ]
    failed = [row for row in early_results.values() if row["status"] == "FAILED"]
    factor_segments = {
        str(name): {
            "rows": int(len(subset)),
            "first_month": str(subset["factor_month"].min()),
            "last_month": str(subset["factor_month"].max()),
            "minimum_valid_return_count": int(subset["valid_return_count"].min()),
            "minimum_valid_return_coverage": float(
                subset["valid_return_coverage"].min()
            ),
        }
        for name, subset in factor.groupby("factor_source_segment", sort=True)
    }
    early_active_month_end_counts = {
        month: len(_active_members(membership, date))
        for month, date in month_ends.items()
        if segments["EARLY_SINA_RECONSTRUCTION"]["first_month"] <= month
        <= segments["EARLY_SINA_RECONSTRUCTION"]["last_month"]
    }
    if (
        not early_active_month_end_counts
        or min(early_active_month_end_counts.values()) != early_minimum_members
        or max(early_active_month_end_counts.values()) != early_maximum_members
    ):
        raise ContractError("早期正式因子区间点时成员月末数量边界不符合预审计结果")
    audit = {
        "schema_version": "1.0.0",
        "report_id": "510300_CSI300_RETURN_DISPERSION_VOLATILITY_INPUTS_V1",
        "status": "PASS",
        "checked_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "candidate_future_volatility_outcomes_computed": False,
        "candidate_portfolio_returns_computed": False,
        "etf_price_values_read": False,
        "h00300_values_read": False,
        "input_builder_scope": "仅构建成分横截面离散度与保存000300价格指数原始输入；不计算指数收益、未来实现方差、510300收益或夏普率",
        "pre_freeze_contract_corrections": [
            {
                "issue": "现有000300_constituent_daily_status.json绑定旧的579879行新浪面板，而当前000300_constituent_daily.parquet为484200行的后续面板",
                "action": "识别该状态文件实际绑定的旧新浪面板备份；不把它用于当前Tushare文件，改为与哈希完全匹配的pre_tushare_backfill备份共同作为2021至2026统一因子输入",
                "candidate_outcomes_read_before_correction": False,
                "portfolio_returns_read_before_correction": False,
            },
            {
                "issue": "上游历史成员CSV含4条纳入日期空缺且与同证券其他区间存在歧义的记录；早期月末可审计实际集合为300至301只",
                "action": "不插值、不伪造日期，明确排除4条未决区间；2005至2014使用剩余点时区间的300至301只实际集合，2015年后改由行情面板自身is_index_member标记提供每日300只集合",
                "candidate_outcomes_read_before_correction": False,
                "portfolio_returns_read_before_correction": False,
            },
            {
                "issue": "当前Tushare活跃成员面板在2021-12仅有275只月末成员同时具备上月末价格，低于冻结的95%覆盖率",
                "action": "不放宽覆盖率、不使用调入日至月末伪整月收益；整个2021-08至2026-07区段统一改用已审计新浪全历史备份面板，其2021-12为300/300且保持月内单一数据源",
                "rejected_tushare_panel_sha256": "e9809195a5bb5de1a19bcddd334382c5968eb702bde2b1e8d19cafdb3a4ed0e7",
                "admitted_sina_backup_sha256": "7731d3482bc898562046eb17991320a8b3e6fe86ab91aaa9e6afd210a97a65c9",
                "candidate_outcomes_read_before_correction": False,
                "portfolio_returns_read_before_correction": False,
            }
        ],
        "membership": {
            "path": membership_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(membership_path),
            "raw_rows": int(len(membership_raw)),
            "admitted_rows": int(len(membership)),
            "excluded_unresolved_null_opt_in_rows": int(len(unresolved_records)),
            "excluded_unresolved_intervals": unresolved_records,
            "unique_symbols": int(membership["symbol"].nunique()),
            "first_opt_in": str(membership["opt_in"].min().date()),
            "last_opt_in": str(membership["opt_in"].max().date()),
            "early_formal_month_end_count": int(len(early_active_month_end_counts)),
            "early_minimum_active_members": int(min(early_active_month_end_counts.values())),
            "early_maximum_active_members": int(max(early_active_month_end_counts.values())),
            "post_2015_member_source": "PANEL_IS_INDEX_MEMBER",
            "governance_label": "POINT_IN_TIME_MEMBERSHIP_EQUAL_WEIGHT_NOT_OFFICIAL_HISTORICAL_WEIGHT",
        },
        "early_component_download": {
            "requested_symbols": int(len(early_results)),
            "successful_symbols": int(len(successful)),
            "failed_symbols": int(len(failed)),
            "failed_examples": [
                {
                    "symbol": row["symbol"],
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
                for row in failed[:25]
            ],
            "successful_parsed_rows": int(len(early_prices)),
            "source_archive": archive_snapshot,
        },
        "existing_panels": {
            "external_2014_2021": external_snapshot,
            "sina_backup_2021_2026": late_snapshot,
            "sina_backup_audit_path": late_audit_path.relative_to(ROOT).as_posix(),
            "sina_backup_audit_sha256": sha256_file(late_audit_path),
            "current_tushare_panel_rejected_for_factor_due_to_2021_12_coverage": True,
        },
        "csi300_price_index": {
            "path": index_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(index_path),
            "bytes": int(index_path.stat().st_size),
            "rows": int(len(index_daily)),
            "first_date": str(index_daily["date"].min().date()),
            "last_date": str(index_daily["date"].max().date()),
            "source": str(index_daily["source"].iloc[0]),
            "returns_computed": False,
        },
        "monthly_dispersion_factor": {
            "path": factor_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(factor_path),
            "bytes": int(factor_path.stat().st_size),
            "rows": int(len(factor)),
            "first_month": str(factor["factor_month"].iloc[0]),
            "last_month": str(factor["factor_month"].iloc[-1]),
            "minimum_valid_return_count": int(factor["valid_return_count"].min()),
            "minimum_valid_return_coverage": float(
                factor["valid_return_coverage"].min()
            ),
            "segments": factor_segments,
            "future_outcomes_joined": False,
        },
        "config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "historical_availability_status": contract["historical_availability_status"],
        "live_or_forward_use": contract["forward_availability_gate"],
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
            "candidate_future_volatility_outcomes_computed": False,
            "candidate_portfolio_returns_computed": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "status": audit["status"],
        "factor_rows": audit["monthly_dispersion_factor"]["rows"],
        "factor_first_month": audit["monthly_dispersion_factor"]["first_month"],
        "factor_last_month": audit["monthly_dispersion_factor"]["last_month"],
        "minimum_valid_return_coverage": audit["monthly_dispersion_factor"][
            "minimum_valid_return_coverage"
        ],
        "early_successful_symbols": audit["early_component_download"][
            "successful_symbols"
        ],
        "early_failed_symbols": audit["early_component_download"]["failed_symbols"],
        "candidate_future_volatility_outcomes_computed": False,
        "candidate_portfolio_returns_computed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
