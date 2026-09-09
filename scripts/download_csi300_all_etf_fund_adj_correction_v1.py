"""用Tushare fund_daily的除权昨收修复全ETF总收益链，不改变原始成交价。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tushare as ts
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_csi300_all_etf_momentum_v1 import CONFIG_FILE, credentials  # noqa: E402

RAW_STATUS = ROOT / "reports" / "data_quality" / "csi300_all_etf_momentum_v1_status.json"
DAILY_DIR = ROOT / "data" / "raw" / "all_etf_momentum_v1r" / "fund_daily_checkpoints"
CORRECTED_PANEL = ROOT / "data" / "raw" / "all_etf_momentum_v1r" / "etf_total_return_panel_tushare_adj.parquet"
CORRECTION_STATUS = ROOT / "reports" / "data_quality" / "csi300_all_etf_momentum_v1r_status.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_daily(api: object, symbol: str, start: str, end: str) -> pd.DataFrame:
    checkpoint = DAILY_DIR / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists():
        data = pd.read_parquet(checkpoint)
        if not data.empty and {"ts_code", "date", "pre_close", "ts_close", "pct_chg"}.issubset(data.columns):
            return data
    error: Exception | None = None
    for attempt in range(4):
        try:
            data = api.fund_daily(
                ts_code=symbol,
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                fields="ts_code,trade_date,pre_close,open,high,low,close,pct_chg,vol,amount",
            )
            if data is None or data.empty:
                raise ValueError("fund_daily返回空表")
            data = data.rename(
                columns={
                    "trade_date": "date",
                    "close": "ts_close",
                    "open": "ts_open",
                    "high": "ts_high",
                    "low": "ts_low",
                }
            )
            data["date"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="coerce")
            numeric = ["pre_close", "ts_open", "ts_high", "ts_low", "ts_close", "pct_chg", "vol", "amount"]
            data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
            data = data.dropna(subset=["date", "pre_close", "ts_close", "pct_chg"]).sort_values("date").drop_duplicates("date", keep="last")
            if data.empty or data[["pre_close", "ts_close"]].le(0).any().any():
                raise ValueError("fund_daily没有有效正价格")
            DAILY_DIR.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".parquet.tmp")
            data.to_parquet(temporary, index=False)
            temporary.replace(checkpoint)
            return data
        except Exception as exc:
            error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{type(error).__name__}: {error}")


def linked_factor(frame: pd.DataFrame) -> np.ndarray:
    """由交易所除权昨收递推，使当日收益等于close/pre_close-1。"""

    raw_close = frame["raw_close"].to_numpy(float)
    pre_close = frame["pre_close"].to_numpy(float)
    factor = np.ones(len(frame), dtype=float)
    if len(frame) > 1:
        factor[1:] = np.cumprod(raw_close[:-1] / pre_close[1:])
    return factor


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    raw_status = json.loads(RAW_STATUS.read_text(encoding="utf-8"))
    inputs, periods = contract["inputs"], contract["periods"]
    raw_panel_path, master_path = ROOT / inputs["panel"], ROOT / inputs["master"]
    if sha256(raw_panel_path) != raw_status["hashes"]["panel"] or sha256(master_path) != raw_status["hashes"]["master"]:
        raise RuntimeError("原始母表或行情面板哈希变化")
    master = pd.read_parquet(master_path)
    raw = pd.read_parquet(raw_panel_path)
    secret, endpoint = credentials()
    ts.set_token(secret)
    api = ts.pro_api()
    api._DataApi__http_url = endpoint

    histories: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    symbols = master["ts_code"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(download_daily, api, symbol, periods["warmup_start"], periods["evaluation_end"]): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                histories[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = str(exc)
            if completed % 25 == 0 or completed == len(futures):
                print(f"除权昨收进度 {completed}/{len(futures)}，成功 {len(histories)}，失败 {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"fund_daily不完整：失败{len(failures)}只，样例={dict(list(failures.items())[:10])}")

    pieces = []
    for symbol in symbols:
        daily = histories[symbol].sort_values("date").copy()
        required_ohlc = ["ts_open", "ts_high", "ts_low", "ts_close", "pre_close"]
        if daily[required_ohlc].isna().any().any() or daily[required_ohlc].le(0).any().any():
            raise ValueError(f"{symbol}的Tushare原始OHLC或除权昨收无效")
        frame = pd.DataFrame(
            {
                "date": daily["date"],
                "con_code": symbol,
                "raw_open": daily["ts_open"],
                "raw_high": daily["ts_high"],
                "raw_low": daily["ts_low"],
                "raw_close": daily["ts_close"],
                "pre_close": daily["pre_close"],
                "volume": daily["vol"].fillna(0.0) * 100.0,
                "amount": daily["amount"].fillna(0.0) * 1000.0,
            }
        ).reset_index(drop=True)
        factor = linked_factor(frame)
        frame["linked_adjustment_factor"] = factor
        for stem in ("open", "high", "low", "close"):
            frame[f"total_return_{stem}"] = frame[f"raw_{stem}"] * factor
        frame["is_suspended"] = frame["volume"].le(0.0)
        frame["price_source"] = "tushare.fund_daily"
        frame["adjustment_source"] = "tushare.fund_daily_pre_close_link"
        pieces.append(frame)
    corrected = pd.concat(pieces, ignore_index=True).sort_values(["date", "con_code"]).reset_index(drop=True)
    if corrected[["date", "con_code"]].duplicated().any():
        raise ValueError("Tushare修复面板存在重复证券日期")
    cross = corrected[["date", "con_code", "raw_close"]].merge(
        raw[["date", "con_code", "raw_close"]].rename(columns={"raw_close": "sina_raw_close"}),
        on=["date", "con_code"], how="outer", indicator=True,
    )
    overlap = cross["_merge"].eq("both")
    raw_close_mismatch_rows = int(
        cross.loc[overlap, "raw_close"].sub(cross.loc[overlap, "sina_raw_close"]).abs().gt(0.0011).sum()
    )
    tushare_only_rows = int(cross["_merge"].eq("left_only").sum())
    sina_only_rows = int(cross["_merge"].eq("right_only").sum())
    raw_return = corrected.groupby("con_code")["raw_close"].pct_change(fill_method=None)
    total_return = corrected.groupby("con_code")["total_return_close"].pct_change(fill_method=None)
    large_raw = raw_return.abs().gt(0.25)
    remaining_extreme = total_return.abs().gt(0.30)
    known = {
        ("159567.SZ", pd.Timestamp("2025-08-11")),
        ("515880.SH", pd.Timestamp("2026-02-03")),
        ("159381.SZ", pd.Timestamp("2026-06-03")),
        ("159901.SZ", pd.Timestamp("2014-09-01")),
        ("510180.SH", pd.Timestamp("2013-12-23")),
        ("510500.SH", pd.Timestamp("2015-04-15")),
    }
    known_returns = {}
    for symbol, date in sorted(known):
        symbol_rows = corrected.loc[corrected["con_code"].eq(symbol)].sort_values("date")
        symbol_returns = symbol_rows.set_index("date")["total_return_close"].pct_change(fill_method=None)
        known_returns[f"{symbol}@{date.date()}"] = float(symbol_returns.loc[date])
    if any(abs(value) > 0.15 for value in known_returns.values()):
        raise ValueError(f"已知拆并事件未修正：{known_returns}")
    if remaining_extreme.any():
        sample = corrected.loc[
            remaining_extreme.to_numpy(),
            ["date", "con_code", "raw_close", "pre_close", "total_return_close", "linked_adjustment_factor"],
        ].head(20)
        raise ValueError(f"修正后仍有{int(remaining_extreme.sum())}个单日超过30%的跳变：{sample.to_dict('records')}")

    CORRECTED_PANEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = CORRECTED_PANEL.with_suffix(".parquet.tmp")
    corrected.to_parquet(temporary, index=False)
    temporary.replace(CORRECTED_PANEL)
    factor_changes = corrected.groupby("con_code")["linked_adjustment_factor"].nunique().gt(1)
    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "correction_only_formula_unchanged": True,
        "fund_daily_symbol_count": len(histories),
        "panel_rows": int(len(corrected)),
        "independent_sina_overlap_rows": int(overlap.sum()),
        "independent_sina_close_mismatch_rows_over_0_0011": raw_close_mismatch_rows,
        "tushare_only_rows": tushare_only_rows,
        "sina_only_rows_excluded": sina_only_rows,
        "symbols_with_factor_changes": int(factor_changes.sum()),
        "raw_moves_over_25pct": int(large_raw.sum()),
        "corrected_moves_over_30pct": int(remaining_extreme.sum()),
        "known_split_corrected_returns": known_returns,
        "sources": {
            "raw_ohlc": "tushare.fund_daily",
            "total_return_link": "tushare.fund_daily pre_close",
            "independent_crosscheck": "新浪历史日线",
        },
        "hashes": {
            "raw_panel": sha256(raw_panel_path),
            "master": sha256(master_path),
            "corrected_panel": sha256(CORRECTED_PANEL),
        },
    }
    CORRECTION_STATUS.parent.mkdir(parents=True, exist_ok=True)
    CORRECTION_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
