"""从新浪下载历史沪深300成分股后复权日线，支持断点续传。"""

from __future__ import annotations

import importlib
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
MARKET_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
OUTPUT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "constituents" / "daily_checkpoints"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_constituent_daily_status.json"

SINA_MODULE = importlib.import_module(ak.stock_zh_a_daily.__module__)
HIST_URL = SINA_MODULE.zh_sina_a_stock_hist_url
AMOUNT_URL = SINA_MODULE.zh_sina_a_stock_amount_url
HFQ_URL = SINA_MODULE.zh_sina_a_stock_hfq_url
NODE_DECODER_SOURCE = (
    SINA_MODULE.hk_js_decode
    + r"""
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => input += chunk);
process.stdin.on("end", () => {
  try {
    process.stdout.write(JSON.stringify(d(input.trim())));
  } catch (error) {
    process.stderr.write(String(error && error.stack ? error.stack : error));
    process.exitCode = 1;
  }
});
"""
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
    )
}
REQUIRED_COLUMNS = {
    "date",
    "con_code",
    "total_return_open",
    "total_return_high",
    "total_return_low",
    "total_return_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "outstanding_share",
    "total_market_cap_cny",
    "volume",
    "amount",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_sina_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    return f"{exchange.lower()}{code}"


def _fetch_text(url: str, attempts: int = 4) -> str:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=(10, 35))
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("响应正文为空")
            return response.text
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"HTTP 下载失败：{type(error).__name__}: {error}")


def _extract_encrypted_history(response_text: str) -> str:
    if "=" not in response_text:
        raise ValueError("新浪历史行情响应缺少赋值符")
    payload = response_text.split("=", 1)[1].split(";", 1)[0].strip().strip('"')
    if not payload:
        raise ValueError("新浪历史行情密文为空")
    return payload


def _decode_history_with_node(payload: str) -> list[dict[str, object]]:
    result = subprocess.run(
        ["node", "-e", NODE_DECODER_SOURCE],
        input=payload,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node 解密失败：{result.stderr.strip()[:500]}")
    decoded = json.loads(result.stdout)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("Node 解密结果不是非空行情数组")
    return decoded


def _decode_javascript_fragment(response_text: str, left: str, right: str) -> object:
    start = response_text.find(left)
    end = response_text.rfind(right)
    if start < 0 or end < start:
        raise ValueError("新浪 JavaScript 响应缺少预期的数据边界")
    fragment = response_text[start : end + 1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return SINA_MODULE.demjson.decode(fragment)


def _parse_downloaded_payloads(
    symbol: str,
    history_text: str,
    amount_text: str,
    factor_text: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    history_rows = _decode_history_with_node(_extract_encrypted_history(history_text))
    data = pd.DataFrame(history_rows)
    if "date" not in data.columns:
        raise ValueError("解密后的历史行情缺少 date 字段")
    data = data.drop(columns=["prevclose", "postVol", "postAmt"], errors="ignore")
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_localize(None)
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    missing_columns = sorted(set(numeric_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"解密后的历史行情缺少字段：{missing_columns}")
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", *numeric_columns]).sort_values("date")
    data = data.drop_duplicates("date", keep="last")

    if "(null)" in amount_text or "=null" in amount_text.replace(" ", ""):
        amount = pd.DataFrame(columns=["date", "outstanding_share"])
    else:
        amount_rows = _decode_javascript_fragment(amount_text, "[", "]")
        amount = pd.DataFrame(amount_rows)
        if amount.shape[1] < 2:
            raise ValueError("新浪股本响应字段不足")
        if not {"date", "amount"}.issubset(amount.columns):
            amount = amount.iloc[:, :2].copy()
            amount.columns = ["date", "amount"]
        amount["date"] = pd.to_datetime(amount["date"], errors="coerce", utc=True).dt.tz_localize(None)
        amount["outstanding_share"] = pd.to_numeric(amount["amount"], errors="coerce") * 10_000
        amount = amount.dropna(subset=["date", "outstanding_share"])
        amount = amount[["date", "outstanding_share"]].sort_values("date").drop_duplicates("date", keep="last")

    factor_payload = _decode_javascript_fragment(factor_text, "{", "}")
    factor_rows = factor_payload.get("data", []) if isinstance(factor_payload, dict) else []
    factors = pd.DataFrame(factor_rows)
    if factors.empty or factors.shape[1] < 2:
        raise ValueError("新浪后复权因子不可用")
    date_column = "d" if "d" in factors.columns else factors.columns[0]
    value_column = "f" if "f" in factors.columns else factors.columns[1]
    factors = factors.rename(columns={date_column: "date", value_column: "hfq_factor"})
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce", utc=True).dt.tz_localize(None)
    factors["hfq_factor"] = pd.to_numeric(factors["hfq_factor"], errors="coerce")
    factors = factors.dropna(subset=["date", "hfq_factor"])
    factors = factors[["date", "hfq_factor"]].sort_values("date").drop_duplicates("date", keep="last")

    if amount.empty:
        data["outstanding_share"] = np.nan
    else:
        data = pd.merge_asof(data, amount, on="date", direction="backward")
    data = pd.merge_asof(data, factors, on="date", direction="backward")
    data = data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if data.empty:
        raise ValueError("筛选项目日期后没有历史行情")
    if data["hfq_factor"].isna().any():
        raise ValueError("后复权因子无法覆盖项目日期")

    data["con_code"] = symbol
    data["raw_open"] = data["open"]
    data["raw_high"] = data["high"]
    data["raw_low"] = data["low"]
    data["raw_close"] = data["close"]
    data["total_return_open"] = data["raw_open"] * data["hfq_factor"]
    data["total_return_high"] = data["raw_high"] * data["hfq_factor"]
    data["total_return_low"] = data["raw_low"] * data["hfq_factor"]
    data["total_return_close"] = data["raw_close"] * data["hfq_factor"]
    data["total_market_cap_cny"] = data["raw_close"] * data["outstanding_share"]
    data["market_cap_asof_date"] = data["date"]
    data["is_suspended"] = False
    data["price_source"] = "sina_history_node_decoder"
    data["adjustment_source"] = "sina_hfq_factor"
    keep = [
        "date",
        "con_code",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "is_suspended",
        "outstanding_share",
        "total_market_cap_cny",
        "market_cap_asof_date",
        "volume",
        "amount",
        "price_source",
        "adjustment_source",
    ]
    data = data[keep].replace([np.inf, -np.inf], np.nan)
    price_columns = [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
    ]
    if data[price_columns].isna().any().any() or (data[price_columns] <= 0).any().any():
        raise ValueError("原始价格或总收益价格存在空值、零值或负值")
    if (data["raw_high"] < data[["raw_open", "raw_close", "raw_low"]].max(axis=1)).any():
        raise ValueError("原始最高价不满足OHLC约束")
    if (data["raw_low"] > data[["raw_open", "raw_close", "raw_high"]].min(axis=1)).any():
        raise ValueError("原始最低价不满足OHLC约束")
    return data.reset_index(drop=True)


def _download_one(
    symbol: str,
    start: str,
    end: str,
    attempts: int = 3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    checkpoint = CHECKPOINT_DIR / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists() and not force_refresh:
        cached = pd.read_parquet(checkpoint)
        if not cached.empty and REQUIRED_COLUMNS.issubset(cached.columns):
            return cached
    sina_symbol = _to_sina_symbol(symbol)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            history_text = _fetch_text(HIST_URL.format(sina_symbol))
            amount_text = _fetch_text(AMOUNT_URL.format(sina_symbol, sina_symbol))
            factor_text = _fetch_text(HFQ_URL.format(sina_symbol))
            data = _parse_downloaded_payloads(
                symbol,
                history_text,
                amount_text,
                factor_text,
                start,
                end,
            )
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            data.to_parquet(checkpoint, index=False)
            return data
        except Exception as exc:
            error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{symbol} 下载失败：{type(error).__name__}: {error}")


def _add_suspension_rows(data: pd.DataFrame, intervals: pd.DataFrame, market_dates: pd.Series) -> pd.DataFrame:
    market_dates = pd.Series(pd.to_datetime(market_dates).astype("datetime64[ns]"))
    start = pd.Timestamp(market_dates.min())
    end = pd.Timestamp(market_dates.max())
    pieces: list[pd.DataFrame] = []
    for symbol, symbol_intervals in intervals.groupby("symbol"):
        raw = data.loc[data["con_code"].eq(symbol)].sort_values("date").copy()
        raw["date"] = pd.to_datetime(raw["date"]).astype("datetime64[ns]")
        if raw.empty:
            continue
        active_dates: list[pd.Timestamp] = []
        for row in symbol_intervals.itertuples(index=False):
            interval_end = (
                end + pd.Timedelta(days=1)
                if pd.isna(row.opt_out)
                else min(pd.Timestamp(row.opt_out), end + pd.Timedelta(days=1))
            )
            interval_start = max(pd.Timestamp(row.opt_in), start)
            active_dates.extend(
                pd.to_datetime(
                    market_dates.loc[(market_dates >= interval_start) & (market_dates < interval_end)]
                ).tolist()
            )
        active_dates = sorted(set(active_dates))
        if not active_dates:
            continue
        first_panel_date = min(pd.Timestamp(raw["date"].min()), min(active_dates))
        last_panel_date = max(pd.Timestamp(raw["date"].max()), max(active_dates))
        symbol_dates = market_dates.loc[market_dates.between(first_panel_date, last_panel_date)]
        panel = pd.DataFrame({"date": symbol_dates.to_numpy(dtype="datetime64[ns]")})
        panel = panel.merge(raw, on="date", how="left", validate="one_to_one")
        panel["con_code"] = symbol
        panel["is_index_member"] = panel["date"].isin(active_dates)
        missing_bar = panel["total_return_close"].isna()
        fill_columns = [
            "total_return_open",
            "total_return_high",
            "total_return_low",
            "total_return_close",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "outstanding_share",
            "total_market_cap_cny",
            "market_cap_asof_date",
        ]
        history = raw[["date", *fill_columns]].sort_values("date")
        panel = pd.merge_asof(
            panel.drop(columns=fill_columns).sort_values("date"),
            history,
            on="date",
            direction="backward",
        )
        panel["con_code"] = symbol
        panel["is_suspended"] = missing_bar.to_numpy()
        panel["volume"] = panel["volume"].fillna(0.0)
        panel["amount"] = panel["amount"].fillna(0.0)
        missing_before_history = panel["total_return_close"].isna()
        panel.loc[missing_before_history, "price_source"] = "missing_before_first_provider_history"
        panel.loc[missing_before_history, "adjustment_source"] = "missing_before_first_provider_history"
        panel["price_source"] = panel["price_source"].fillna("sina_forward_fill_suspension")
        panel["adjustment_source"] = panel["adjustment_source"].fillna("sina_hfq_factor_forward_fill")
        pieces.append(panel)
    if not pieces:
        raise ValueError("没有生成成分股成员日线")
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "con_code"]).reset_index(drop=True)


def main() -> int:
    for path in (SETTINGS_FILE, MEMBERSHIP_FILE, MARKET_FILE):
        if not path.exists():
            raise FileNotFoundError(f"成分股日线下载缺少输入：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    intervals = pd.read_parquet(MEMBERSHIP_FILE)
    intervals["opt_in"] = pd.to_datetime(intervals["opt_in"])
    intervals["opt_out"] = pd.to_datetime(intervals["opt_out"])
    project_start = pd.Timestamp(settings["project"]["start_date"])
    project_end = pd.Timestamp(settings["project"]["end_date"])
    warmup_start = project_start - pd.Timedelta(days=120)
    relevant = intervals.loc[
        (intervals["opt_in"] <= project_end)
        & (intervals["opt_out"].isna() | (intervals["opt_out"] > warmup_start))
    ].copy()
    symbols = sorted(relevant["symbol"].unique())
    print(f"需要下载 {len(symbols)} 只历史成分股，已有断点会自动复用。", flush=True)
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_download_one, symbol, str(warmup_start.date()), str(project_end.date())): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = str(exc)
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"进度：{completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}",
                    flush=True,
                )
    if failures:
        sample = dict(list(failures.items())[:10])
        raise RuntimeError(f"仍有 {len(failures)} 只成分股下载失败：{sample}")

    downloaded = pd.concat(results.values(), ignore_index=True)
    market = pd.read_parquet(MARKET_FILE)
    market["date"] = pd.to_datetime(market["date"])
    market_dates = (
        market.loc[market["date"].between(warmup_start, project_end), "date"]
        .sort_values()
        .reset_index(drop=True)
    )
    panel = _add_suspension_rows(downloaded, relevant, market_dates)
    duplicates = int(panel[["date", "con_code"]].duplicated().sum())
    if duplicates:
        raise ValueError(f"成分股成员日线存在 {duplicates} 个重复证券日期")
    active_counts = panel.loc[panel["is_index_member"]].groupby("date")["con_code"].nunique()
    if active_counts.min() != 300 or active_counts.max() != 300:
        raise ValueError(f"成员日线每日证券数异常：{active_counts.min()} 至 {active_counts.max()}")
    research_active = panel.loc[
        panel["date"].between(project_start, project_end) & panel["is_index_member"]
    ]
    research_price_columns = [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
    ]
    if research_active[research_price_columns].isna().any().any():
        missing = research_active.loc[
            research_active[research_price_columns].isna().any(axis=1), ["date", "con_code"]
        ]
        raise ValueError(f"正式研究区间存在成员行情缺失：{missing.head(10).to_dict('records')}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "symbol_count": len(symbols),
        "row_count": int(len(panel)),
        "active_member_row_count": int(panel["is_index_member"].sum()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "min_constituents_per_day": int(active_counts.min()),
        "max_constituents_per_day": int(active_counts.max()),
        "suspended_rows": int(panel["is_suspended"].sum()),
        "warmup_member_rows_missing_before_provider_history": int(
            panel.loc[
                (panel["date"] < project_start)
                & panel["is_index_member"]
                & panel["total_return_close"].isna()
            ].shape[0]
        ),
        "market_cap_missing_rows": int(panel["total_market_cap_cny"].isna().sum()),
        "market_cap_missing_symbol_count": int(
            panel.loc[panel["total_market_cap_cny"].isna(), "con_code"].nunique()
        ),
        "price_source": "新浪历史日线（独立 Node 解密）",
        "total_return_price": "未复权开高低收分别乘新浪后复权因子",
        "market_cap": "未复权收盘价乘新浪流通股本；属于流通市值近似，不是中证计算用调整市值",
        "governance_label": "POINT_IN_TIME_MEMBERSHIP_EQUAL_WEIGHT_NOT_OFFICIAL_INDEX_WEIGHT",
        "source_endpoints": {
            "history": HIST_URL,
            "share_amount": AMOUNT_URL,
            "back_adjustment_factor": HFQ_URL,
        },
        "membership_sha256": _sha256(MEMBERSHIP_FILE),
        "market_calendar_sha256": _sha256(MARKET_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
