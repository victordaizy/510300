"""补采沪深300官方权重快照所需的历史未复权价格。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "point_in_time_valuation_price_backfill.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    forbidden = (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    )
    if enabled := [field for field in forbidden if config["protocol"].get(field)]:
        raise ValueError(f"价格补采配置错误，禁止项被启用：{enabled}")
    return config


def normalize_daily(
    raw: pd.DataFrame,
    symbol: str,
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """保留供应商未复权价格语义并写入请求边界。"""

    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"{symbol}日线缺少字段：{sorted(missing)}")
    data = raw.loc[raw["ts_code"].astype(str).eq(symbol), list(required)].copy()
    if data.empty:
        raise ValueError(f"{symbol}在请求区间没有日线")
    data["date"] = pd.to_datetime(
        data["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    for column in ("open", "high", "low", "close", "vol", "amount"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["date"].between(request_start, request_end)].copy()
    if data.empty or data[["date", "close"]].isna().any().any():
        raise ValueError(f"{symbol}日线日期或收盘价无效")
    if data[["open", "high", "low", "close"]].le(0).any().any():
        raise ValueError(f"{symbol}存在非正OHLC")
    if (
        data["high"].lt(data[["open", "close", "low"]].max(axis=1))
        | data["low"].gt(data[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError(f"{symbol}存在OHLC关系异常")
    result = pd.DataFrame(
        {
            "date": data["date"],
            "con_code": symbol,
            "raw_open": data["open"],
            "raw_high": data["high"],
            "raw_low": data["low"],
            "raw_close": data["close"],
            "volume": data["vol"] * 100.0,
            "amount_cny": data["amount"] * 1000.0,
            "source": "tushare_proxy.daily",
            "retrieved_at": str(retrieved_at),
            "request_start": request_start,
            "request_end": request_end,
        }
    )
    if result.duplicated(["date", "con_code"]).any():
        raise ValueError(f"{symbol}存在重复交易日")
    return result.sort_values("date").reset_index(drop=True)


def cache_covers_request(
    data: pd.DataFrame,
    symbol: str,
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
) -> bool:
    required = {"date", "con_code", "raw_close", "request_start", "request_end"}
    if not required.issubset(data.columns) or data.empty:
        return False
    starts = pd.to_datetime(data["request_start"], errors="coerce")
    ends = pd.to_datetime(data["request_end"], errors="coerce")
    return bool(
        data["con_code"].astype(str).eq(symbol).all()
        and starts.notna().all()
        and ends.notna().all()
        and starts.min() <= request_start
        and ends.max() >= request_end
        and pd.to_numeric(data["raw_close"], errors="coerce").gt(0).all()
    )


def required_missing_symbols(
    weights: pd.DataFrame,
    prior_coverage: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[str], list[pd.Timestamp]]:
    """从上轮失败月份精确提取需要补采的证券集合。"""

    threshold = float(config["scope"]["minimum_snapshot_price_weight_coverage"])
    coverage = prior_coverage.copy()
    coverage["date"] = pd.to_datetime(coverage["date"], errors="coerce")
    failed_dates = sorted(
        coverage.loc[
            coverage["date"].between(
                pd.Timestamp(config["scope"]["weight_window_start"]),
                pd.Timestamp(config["scope"]["missing_snapshot_end"]),
            )
            & coverage["price_weight_coverage"].lt(threshold),
            "date",
        ].unique()
    )
    weight_data = weights.copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"], errors="coerce")
    symbols = sorted(
        weight_data.loc[weight_data["trade_date"].isin(failed_dates), "con_code"]
        .astype(str)
        .unique()
    )
    return symbols, [pd.Timestamp(value) for value in failed_dates]


def normalize_current_history(current: pd.DataFrame) -> pd.DataFrame:
    """从当前面板恢复停牌行对应的最后真实成交日。"""

    data = current.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["is_suspended"] = data["is_suspended"].fillna(False).astype(bool)
    data = data.sort_values(["con_code", "date"]).reset_index(drop=True)
    data["price_trade_date"] = data["date"].where(~data["is_suspended"])
    data["price_trade_date"] = data.groupby("con_code", sort=False)["price_trade_date"].ffill()
    return data[
        ["date", "con_code", "raw_close", "price_trade_date", "is_suspended", "price_source", "retrieved_at"]
    ].rename(columns={"price_source": "source"})


def build_snapshot_prices(
    weights: pd.DataFrame,
    downloaded_history: pd.DataFrame,
    current_history: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """用真实成交日或仅向前最后成交价构造60个月快照价格。"""

    weight_data = weights.copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"], errors="coerce")
    weight_data = weight_data.loc[
        weight_data["trade_date"].between(
            pd.Timestamp(config["scope"]["weight_window_start"]),
            pd.Timestamp(config["scope"]["weight_window_end"]),
        )
    ].copy()
    early_end = pd.Timestamp(config["scope"]["missing_snapshot_end"])
    download = downloaded_history.copy()
    download["date"] = pd.to_datetime(download["date"], errors="coerce")
    download = download.sort_values(["con_code", "date"])
    download_groups = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in download.groupby("con_code", sort=False)
    }
    current = normalize_current_history(current_history)
    current_groups = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in current.groupby("con_code", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weight_data.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        for item in snapshot_weights[["con_code", "weight"]].itertuples(index=False):
            symbol = str(item.con_code)
            if snapshot_date <= early_end:
                history = download_groups.get(symbol)
                if history is None:
                    price_row = None
                else:
                    eligible = history.loc[history["date"].le(snapshot_date)]
                    price_row = eligible.iloc[-1] if not eligible.empty else None
                is_suspended = bool(price_row is not None and price_row["date"] < snapshot_date)
                price_trade_date = price_row["date"] if price_row is not None else pd.NaT
                raw_close = price_row["raw_close"] if price_row is not None else np.nan
                source = (
                    "tushare_proxy.daily_last_trade_forward_fill_snapshot"
                    if is_suspended
                    else "tushare_proxy.daily"
                )
                retrieved_at = price_row["retrieved_at"] if price_row is not None else pd.NaT
            else:
                history = current_groups.get(symbol)
                exact = history.loc[history["date"].eq(snapshot_date)] if history is not None else pd.DataFrame()
                price_row = exact.iloc[-1] if not exact.empty else None
                is_suspended = bool(price_row["is_suspended"]) if price_row is not None else False
                price_trade_date = price_row["price_trade_date"] if price_row is not None else pd.NaT
                raw_close = price_row["raw_close"] if price_row is not None else np.nan
                source = str(price_row["source"]) if price_row is not None else None
                retrieved_at = price_row["retrieved_at"] if price_row is not None else pd.NaT
            rows.append(
                {
                    "date": snapshot_date,
                    "con_code": symbol,
                    "weight_pct": float(item.weight),
                    "raw_close": float(raw_close) if pd.notna(raw_close) else np.nan,
                    "price_trade_date": pd.Timestamp(price_trade_date) if pd.notna(price_trade_date) else pd.NaT,
                    "price_age_calendar_days": (
                        int((snapshot_date - pd.Timestamp(price_trade_date)).days)
                        if pd.notna(price_trade_date)
                        else np.nan
                    ),
                    "is_suspended_or_stale": is_suspended,
                    "source": source,
                    "retrieved_at": str(retrieved_at) if pd.notna(retrieved_at) else None,
                    "point_in_time_rule": "LAST_UNADJUSTED_CLOSE_AT_OR_BEFORE_SNAPSHOT",
                }
            )
    result = pd.DataFrame(rows).sort_values(["date", "con_code"]).reset_index(drop=True)
    if result.duplicated(["date", "con_code"]).any():
        raise ValueError("快照价格表存在重复证券日期")
    if (result["price_trade_date"] > result["date"]).any():
        raise ValueError("快照价格使用了未来成交日")
    return result


def validate_snapshot_prices(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    counts = data.groupby("date")["con_code"].nunique()
    valid = data["raw_close"].notna() & data["raw_close"].gt(0)
    coverage = (
        data.assign(_valid_weight=np.where(valid, data["weight_pct"], 0.0))
        .groupby("date")["_valid_weight"]
        .sum()
        / 100.0
    )
    threshold = float(config["scope"]["minimum_snapshot_price_weight_coverage"])
    future_count = int((data["price_trade_date"] > data["date"]).sum())
    passed = bool(
        len(counts) == 60
        and counts.eq(300).all()
        and coverage.ge(threshold).all()
        and future_count == 0
    )
    stale = data.loc[data["is_suspended_or_stale"].fillna(False)]
    return {
        "status": "PASS" if passed else "BLOCKED",
        "snapshot_count": int(len(counts)),
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "minimum_constituents": int(counts.min()),
        "maximum_constituents": int(counts.max()),
        "minimum_price_weight_coverage": float(coverage.min()),
        "maximum_price_weight_coverage": float(coverage.max()),
        "failed_snapshot_count": int(coverage.lt(threshold).sum()),
        "missing_price_row_count": int((~valid).sum()),
        "stale_price_row_count": int(len(stale)),
        "maximum_price_age_calendar_days": (
            int(stale["price_age_calendar_days"].max()) if not stale.empty else 0
        ),
        "future_price_row_count": future_count,
    }


def cross_check_overlap(downloaded: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """只比较两源都标记为真实成交的重叠未复权收盘价。"""

    left = downloaded[["date", "con_code", "raw_close"]].copy()
    right = current[["date", "con_code", "raw_close", "is_suspended"]].copy()
    for frame in (left, right):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    right = right.loc[~right["is_suspended"].fillna(False).astype(bool)]
    merged = left.merge(right, on=["date", "con_code"], suffixes=("_download", "_current"))
    difference = (merged["raw_close_download"] - merged["raw_close_current"]).abs()
    return {
        "overlap_row_count": int(len(merged)),
        "first_overlap_date": str(merged["date"].min().date()) if not merged.empty else None,
        "last_overlap_date": str(merged["date"].max().date()) if not merged.empty else None,
        "maximum_absolute_close_difference": float(difference.max()) if not merged.empty else None,
        "exact_match_ratio": float(difference.eq(0).mean()) if not merged.empty else None,
    }


def render_markdown(report: dict[str, Any]) -> str:
    check = report["snapshot_validation"]
    overlap = report["cross_source_overlap"]
    return "\n".join(
        [
            "# 沪深300点时估值价格补采报告",
            "",
            f"> 状态：`{report['status']}`。本报告只审计官方权重快照所需的未复权价格，不计算收益、IC或仓位。",
            "",
            "## 补采范围",
            "",
            f"- 缺口月份：{report['failed_month_count']}个。",
            f"- 证券数：{report['required_symbol_count']}只。",
            f"- 请求区间：{report['download_start']}至{report['download_end']}。",
            f"- 下载/复用缓存：{report['downloaded_symbol_count']}/{report['reused_cache_symbol_count']}只。",
            f"- 因跨起点长期停牌追加更早历史：{report['supplemental_prehistory_symbol_count']}只。",
            "- 价格口径：Tushare daily未复权收盘价；停牌快照只使用此前最后成交价。",
            "",
            "## 60个月快照验收",
            "",
            f"- 快照数：{check['snapshot_count']}；总行数：{check['row_count']}。",
            f"- 每期成分：{check['minimum_constituents']}—{check['maximum_constituents']}只。",
            f"- 最低价格权重覆盖：{check['minimum_price_weight_coverage']:.4%}。",
            f"- 缺价行：{check['missing_price_row_count']}；未来价格行：{check['future_price_row_count']}。",
            f"- 停牌/陈旧价格行：{check['stale_price_row_count']}；最大陈旧{check['maximum_price_age_calendar_days']}个自然日。",
            "",
            "## 独立重叠核验",
            "",
            f"- 与当前成分面板真实成交行重叠：{overlap['overlap_row_count']}行。",
            f"- 收盘价完全一致比例：{overlap['exact_match_ratio']:.4%}。",
            f"- 最大绝对差：{overlap['maximum_absolute_close_difference']}元。",
            "",
            "## 治理边界",
            "",
            "- 未覆盖或改写当前`000300_constituent_daily.parquet`。",
            "- 令牌未写入任何产物。",
            "- 本产物只解除五年原始EY的价格缺口；标准化盈利早期历史仍需单独补采。",
            "",
        ]
    )


def write_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    json_path = ROOT / config["artifacts"]["report_json"]
    markdown_path = ROOT / config["artifacts"]["report_markdown"]
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_json.replace(json_path)
    temp_md = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temp_md.write_text(render_markdown(report), encoding="utf-8")
    temp_md.replace(markdown_path)
