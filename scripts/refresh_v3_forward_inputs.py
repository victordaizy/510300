"""刷新V3_FORWARD_1当日所需输入；任何必需刷新失败时禁止生成信号。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v3_forward_validation import (
    BOND_FILE,
    ETF_FILE,
    FINANCIALS_FILE,
    FORWARD_CONSTITUENT_CLOSE_FILE,
    INDEX_FILE,
    TOTAL_RETURN_FILE,
    VALUATION_FILE,
    WEIGHTS_FILE,
)
from scripts.download_000300_daily import fetch_index
from scripts.download_000300_valuation import fetch_valuation
from scripts.download_510300_daily import fetch_from_sina
from scripts.download_china_government_bond_yields import normalize_curve
from scripts.download_csi300_point_in_time_financials import (
    API_FIELDS,
    build_point_in_time_events,
    get_token,
)
from scripts.download_h00300_total_return import normalize_columns
from scripts.quality_check_daily import write_daily_metadata


STATUS_FILE = ROOT / "paper" / "v3_forward_1_input_refresh_status.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _merge_by_keys(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _retry(call: Callable[[], pd.DataFrame], label: str, attempts: int = 3) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None or result.empty:
                raise ValueError("返回空数据")
            return result
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label}刷新失败：{type(error).__name__}: {error}")


def _fetch_sina_quote(source_symbol: str, asof: pd.Timestamp) -> list[str]:
    response = requests.get(
        SINA_QUOTE_URL.format(symbols=source_symbol),
        headers=SINA_HEADERS,
        timeout=(10, 35),
    )
    response.raise_for_status()
    text = response.content.decode("gbk", errors="replace")
    match = re.search(
        rf'var hq_str_{re.escape(source_symbol)}="(?P<values>[^"]*)";', text
    )
    if match is None:
        raise ValueError(f"新浪当日行情缺失：{source_symbol}")
    values = match.group("values").split(",")
    if len(values) < 32:
        raise ValueError(f"新浪当日行情字段不足：{source_symbol}")
    quote_date = pd.Timestamp(values[30])
    if quote_date.normalize() != asof:
        raise ValueError(
            f"{source_symbol}行情日期为{quote_date.date()}，不是{asof.date()}"
        )
    return values


def refresh_market_series(asof: pd.Timestamp, retrieved_at: datetime) -> dict[str, str]:
    start = "2016-08-12"
    end = str(asof.date())
    index = _retry(lambda: fetch_index("000300.SH", start, end), "000300")
    index["retrieved_at"] = retrieved_at.isoformat()
    _atomic_parquet(index.sort_values("date").reset_index(drop=True), INDEX_FILE)

    etf = _retry(lambda: fetch_from_sina("510300.SH", start, end), "510300")
    if pd.to_datetime(etf["date"]).max().normalize() < asof:
        values = _fetch_sina_quote("sh510300", asof)
        numeric = [pd.to_numeric(values[index], errors="coerce") for index in (1, 4, 5, 3, 8, 9)]
        if any(pd.isna(value) for value in numeric) or any(float(value) <= 0 for value in numeric[:4]):
            raise ValueError("510300当日行情兜底存在无效OHLC")
        quote = pd.DataFrame(
            [
                {
                    "date": asof,
                    "open": float(numeric[0]),
                    "high": float(numeric[1]),
                    "low": float(numeric[2]),
                    "close": float(numeric[3]),
                    "volume": float(numeric[4]),
                    "amount": float(numeric[5]),
                    "symbol": "510300.SH",
                    "source": "sina.hq.batch_quote",
                    "volume_unit": "share",
                    "amount_unit": "CNY",
                }
            ]
        )
        etf = _merge_by_keys(etf, quote, ["date"])
    etf["retrieved_at"] = retrieved_at.isoformat()
    etf = etf.sort_values("date").reset_index(drop=True)
    _atomic_parquet(etf, ETF_FILE)
    write_daily_metadata(
        etf,
        ETF_FILE,
        ETF_FILE.with_suffix(".metadata.json"),
        requested_start=start,
        requested_end=end,
        evaluation_start="2021-08-12",
        historical_evaluation_end="2026-08-12",
    )

    total = _retry(
        lambda: ak.stock_zh_index_hist_csindex(
            symbol="H00300",
            start_date="20160812",
            end_date=asof.strftime("%Y%m%d"),
        ),
        "H00300",
    )
    total = normalize_columns(total)
    required = ["date", "symbol", "name", "close", "pct_change"]
    if missing := set(required).difference(total.columns):
        raise ValueError(f"H00300缺少字段：{sorted(missing)}")
    total["date"] = pd.to_datetime(total["date"], errors="coerce")
    for column in total.columns.difference(["date", "symbol", "name"]):
        total[column] = pd.to_numeric(total[column], errors="coerce")
    total = total.dropna(subset=["date", "close"]).sort_values("date")
    total["source"] = "csindex.via_akshare.stock_zh_index_hist_csindex"
    total["retrieved_at"] = retrieved_at.isoformat()
    _atomic_parquet(total.reset_index(drop=True), TOTAL_RETURN_FILE)
    return {
        "000300_latest": str(pd.to_datetime(index["date"]).max().date()),
        "510300_latest": str(pd.to_datetime(etf["date"]).max().date()),
        "H00300_latest": str(pd.to_datetime(total["date"]).max().date()),
    }


def refresh_valuation_and_bonds(asof: pd.Timestamp, retrieved_at: datetime) -> dict[str, str]:
    valuation = _retry(fetch_valuation, "000300估值")
    valuation = valuation.loc[pd.to_datetime(valuation["date"]).le(asof)].copy()
    _atomic_parquet(valuation.sort_values("date").reset_index(drop=True), VALUATION_FILE)

    bond_start = asof - pd.Timedelta(days=45)
    raw_bonds = _retry(
        lambda: ak.bond_china_yield(
            start_date=bond_start.strftime("%Y%m%d"),
            end_date=asof.strftime("%Y%m%d"),
        ),
        "中国国债收益率",
    )
    new_bonds = normalize_curve(raw_bonds, retrieved_at)
    old_bonds = pd.read_parquet(BOND_FILE) if BOND_FILE.exists() else pd.DataFrame()
    bonds = _merge_by_keys(old_bonds, new_bonds, ["date"])
    _atomic_parquet(bonds, BOND_FILE)
    return {
        "valuation_latest": str(pd.to_datetime(valuation["date"]).max().date()),
        "bond_latest": str(pd.to_datetime(bonds["date"]).max().date()),
    }


def _tushare_client():
    import tushare as ts

    pro = ts.pro_api(get_token())
    api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
    if not api_url.startswith(("https://", "http://")):
        raise ValueError("TUSHARE_API_URL必须是HTTP或HTTPS地址")
    pro._DataApi__http_url = api_url
    return pro


def refresh_weights_and_financials(
    asof: pd.Timestamp, retrieved_at: datetime
) -> dict[str, object]:
    old_weights = pd.read_parquet(WEIGHTS_FILE)
    old_events = pd.read_parquet(FINANCIALS_FILE)
    try:
        pro = _tushare_client()
    except RuntimeError as exc:
        if "TUSHARE_TOKEN" not in str(exc) and "TS_TOKEN" not in str(exc):
            raise
        weight_retrieved = pd.to_datetime(
            old_weights["retrieved_at"], errors="coerce", utc=True
        ).dt.tz_convert(TIMEZONE)
        financial_retrieved = pd.to_datetime(
            old_events["retrieved_at"], errors="coerce", utc=True
        ).dt.tz_convert(TIMEZONE)
        weight_refresh_date = weight_retrieved.max().tz_localize(None).normalize()
        financial_refresh_date = financial_retrieved.max().tz_localize(None).normalize()
        if weight_refresh_date != asof or financial_refresh_date != asof:
            raise RuntimeError(
                "自动环境缺少TUSHARE_TOKEN/TS_TOKEN，且权重或财务并非目标日刷新；禁止使用陈旧输入"
            ) from exc
        latest_weight_date = pd.to_datetime(old_weights["trade_date"]).max()
        return {
            "refresh_mode": "REUSED_ALREADY_REFRESHED_TODAY",
            "weight_latest": str(latest_weight_date.date()),
            "weight_symbol_count": int(
                old_weights.loc[
                    pd.to_datetime(old_weights["trade_date"]).eq(latest_weight_date),
                    "con_code",
                ].nunique()
            ),
            "financial_latest_available_at": str(
                pd.to_datetime(old_events["available_at"]).max().date()
            ),
            "credential_available_in_process": False,
        }
    month_start = asof.to_period("M").start_time
    weights_part = pro.index_weight(
        index_code="399300.SZ",
        start_date=month_start.strftime("%Y%m%d"),
        end_date=asof.strftime("%Y%m%d"),
    )
    if weights_part is not None and not weights_part.empty:
        weights_part["trade_date"] = pd.to_datetime(weights_part["trade_date"], errors="coerce")
        weights_part["weight"] = pd.to_numeric(weights_part["weight"], errors="coerce")
        weights_part = weights_part.loc[weights_part["trade_date"].le(asof)].dropna(
            subset=["trade_date", "con_code", "weight"]
        )
        weights_part["source"] = "tushare.index_weight"
        weights_part["retrieved_at"] = retrieved_at
        snapshot = weights_part.loc[
            weights_part["trade_date"].eq(weights_part["trade_date"].max())
        ]
        if not 250 <= snapshot["con_code"].nunique() <= 350:
            raise ValueError("当月官方权重成分数异常")
        if not 98.0 <= snapshot["weight"].sum() <= 102.0:
            raise ValueError("当月官方权重之和异常")
        old_weights = _merge_by_keys(
            old_weights, weights_part, ["trade_date", "con_code"]
        )
        _atomic_parquet(old_weights, WEIGHTS_FILE)

    latest_weight_date = pd.to_datetime(old_weights["trade_date"]).max()
    latest_symbols = set(
        old_weights.loc[
            pd.to_datetime(old_weights["trade_date"]).eq(latest_weight_date), "con_code"
        ].astype(str)
    )
    completed_quarters = pd.date_range(end=asof, periods=5, freq="QE")
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in API_FIELDS}
    for period in completed_quarters:
        period_text = period.strftime("%Y%m%d")
        for api_name, fields in API_FIELDS.items():
            part = _retry(
                lambda api_name=api_name, fields=fields, period_text=period_text: pro.query(
                    api_name, period=period_text, fields=",".join(fields)
                ),
                f"{api_name}/{period_text}",
            )
            pieces[api_name].append(part.loc[part["ts_code"].astype(str).isin(latest_symbols)])
            time.sleep(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")))
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in pieces.items()}
    events = build_point_in_time_events(
        combined["income_vip"],
        combined["balancesheet_vip"],
        combined["fina_indicator_vip"],
        retrieved_at,
    )
    events = _merge_by_keys(
        old_events, events, ["con_code", "report_period", "available_at"]
    )
    _atomic_parquet(events, FINANCIALS_FILE)
    return {
        "weight_latest": str(latest_weight_date.date()),
        "weight_symbol_count": len(latest_symbols),
        "financial_latest_available_at": str(
            pd.to_datetime(events["available_at"]).max().date()
        ),
    }


def _sina_symbol(con_code: str) -> str:
    code, exchange = con_code.split(".")
    return f"{exchange.lower()}{code}"


def refresh_constituent_closes(asof: pd.Timestamp, retrieved_at: datetime) -> dict[str, object]:
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    latest_date = weights.loc[weights["trade_date"].le(asof), "trade_date"].max()
    symbols = sorted(
        weights.loc[weights["trade_date"].eq(latest_date), "con_code"].astype(str).unique()
    )
    prior = (
        pd.read_parquet(FORWARD_CONSTITUENT_CLOSE_FILE)
        if FORWARD_CONSTITUENT_CLOSE_FILE.exists()
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for start in range(0, len(symbols), 100):
        batch = symbols[start : start + 100]
        source_symbols = [_sina_symbol(symbol) for symbol in batch]
        response = requests.get(
            SINA_QUOTE_URL.format(symbols=",".join(source_symbols)),
            headers=SINA_HEADERS,
            timeout=(10, 35),
        )
        response.raise_for_status()
        text = response.content.decode("gbk", errors="replace")
        parsed = {
            match.group("symbol"): match.group("values").split(",")
            for match in re.finditer(
                r'var hq_str_(?P<symbol>[a-z]{2}\d{6})="(?P<values>[^"]*)";', text
            )
        }
        for con_code, source_symbol in zip(batch, source_symbols, strict=True):
            values = parsed.get(source_symbol)
            if values is None or len(values) < 32:
                raise ValueError(f"新浪成分股快照缺失：{con_code}")
            quote_date = pd.Timestamp(values[30])
            if quote_date.normalize() != asof:
                raise ValueError(f"{con_code}快照日期为{quote_date.date()}，不是{asof.date()}")
            current = pd.to_numeric(values[3], errors="coerce")
            previous_close = pd.to_numeric(values[2], errors="coerce")
            suspended = pd.isna(current) or float(current) <= 0
            close = float(previous_close if suspended else current)
            if not close > 0:
                raise ValueError(f"{con_code}没有有效当前价或昨收价")
            rows.append(
                {
                    "date": asof,
                    "con_code": con_code,
                    "raw_close": close,
                    "is_suspended": bool(suspended),
                    "source": "sina.hq.batch_quote",
                    "retrieved_at": retrieved_at.isoformat(),
                }
            )
    snapshot = pd.DataFrame(rows)
    if snapshot["con_code"].nunique() != len(symbols):
        raise ValueError("成分股收盘快照覆盖不完整")
    combined = _merge_by_keys(prior, snapshot, ["date", "con_code"])
    _atomic_parquet(combined, FORWARD_CONSTITUENT_CLOSE_FILE)
    return {
        "constituent_close_date": str(asof.date()),
        "constituent_close_count": int(snapshot["con_code"].nunique()),
        "suspended_count": int(snapshot["is_suspended"].sum()),
    }


def refresh_all(asof: pd.Timestamp | None = None) -> dict[str, object]:
    target = pd.Timestamp(asof or datetime.now(TIMEZONE).date()).normalize()
    retrieved_at = datetime.now(TIMEZONE)
    status: dict[str, object] = {
        "asof": str(target.date()),
        "retrieved_at": retrieved_at.isoformat(),
        "status": "RUNNING",
        "steps": {},
    }
    try:
        market = refresh_market_series(target, retrieved_at)
        status["steps"]["market"] = {"status": "SUCCESS", **market}
        target_text = str(target.date())
        if any(
            market[key] != target_text
            for key in ("000300_latest", "510300_latest", "H00300_latest")
        ):
            status["status"] = "SKIPPED_NON_TRADING_OR_PROVIDER_NOT_READY"
            status["message"] = "三条核心收盘序列尚未共同到达目标日，不生成信号。"
            return status
        remaining_steps = [
            ("valuation_and_bonds", lambda: refresh_valuation_and_bonds(target, retrieved_at)),
            ("weights_and_financials", lambda: refresh_weights_and_financials(target, retrieved_at)),
            ("constituent_closes", lambda: refresh_constituent_closes(target, retrieved_at)),
        ]
        for name, call in remaining_steps:
            status["steps"][name] = {"status": "SUCCESS", **call()}
        status["status"] = "SUCCESS"
        return status
    except Exception as exc:
        status["status"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(
            json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新V3_FORWARD_1前瞻输入")
    parser.add_argument("--date", help="目标日期YYYY-MM-DD，默认上海本地日期")
    args = parser.parse_args()
    try:
        result = refresh_all(pd.Timestamp(args.date) if args.date else None)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"前瞻输入刷新失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
