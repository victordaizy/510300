"""下载哈希训练组母表与2014—2023日线；绝不下载盲测组收益。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import tushare as ts
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.download_h00300_total_return import normalize_columns  # noqa: E402

CONFIG_FILE = ROOT / "config" / "a_share_hash_holdout_alpha_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_bucket(code: str) -> int:
    return hashlib.sha256(code.encode("utf-8")).digest()[0] % 5


def fetch_master(api: object, contract: dict) -> pd.DataFrame:
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    chunks = []
    for status in contract["universe"]["stock_statuses"]:
        part = None
        error: Exception | None = None
        for attempt in range(4):
            try:
                part = api.stock_basic(exchange="", list_status=status, fields=fields)
                break
            except Exception as exc:
                error = exc
                time.sleep(2.0 * (attempt + 1))
        if part is None and error is not None:
            raise RuntimeError(f"stock_basic状态{status}连续失败：{type(error).__name__}: {error}") from error
        if part is not None and not part.empty:
            chunks.append(part)
        time.sleep(0.6)
    if not chunks:
        raise RuntimeError("stock_basic未返回股票母表")
    master = pd.concat(chunks, ignore_index=True).drop_duplicates("ts_code", keep="first")
    master = master.loc[master["exchange"].isin(contract["universe"]["exchanges"])].copy()
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce")
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce")
    master = master.loc[master["list_date"].notna()].copy()
    master["split_bucket"] = master["ts_code"].astype(str).map(split_bucket)
    master["split_group"] = np.where(master["split_bucket"].eq(int(contract["split"]["holdout_remainder"])), "HOLDOUT", "TRAIN")
    return master.sort_values("ts_code").reset_index(drop=True)


def download_daily(api: object, symbol: str, start: str, end: str, directory: Path) -> pd.DataFrame:
    checkpoint = directory / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists():
        data = pd.read_parquet(checkpoint)
        if not data.empty and {"date", "con_code", "raw_open", "pre_close", "total_return_close", "amount"}.issubset(data.columns):
            return data
    error: Exception | None = None
    for attempt in range(4):
        try:
            data = api.daily(ts_code=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""), fields="ts_code,trade_date,pre_close,open,high,low,close,pct_chg,vol,amount")
            if data is None or data.empty:
                raise ValueError("daily返回空表")
            data = data.rename(columns={"ts_code": "con_code", "trade_date": "date", "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"})
            data["date"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="coerce")
            numeric = ["pre_close", "raw_open", "raw_high", "raw_low", "raw_close", "pct_chg", "vol", "amount"]
            data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
            data = data.dropna(subset=["date", "pre_close", "raw_open", "raw_high", "raw_low", "raw_close"]).sort_values("date").drop_duplicates("date", keep="last")
            if data.empty or data[["pre_close", "raw_open", "raw_high", "raw_low", "raw_close"]].le(0).any().any():
                raise ValueError("daily原始价格无效")
            raw_close = data["raw_close"].to_numpy(float)
            pre_close = data["pre_close"].to_numpy(float)
            factor = np.ones(len(data), dtype=float)
            if len(data) > 1:
                factor[1:] = np.cumprod(raw_close[:-1] / pre_close[1:])
            data["linked_adjustment_factor"] = factor
            for stem in ("open", "high", "low", "close"):
                data[f"total_return_{stem}"] = data[f"raw_{stem}"] * factor
            data["volume"] = data["vol"].fillna(0.0) * 100.0
            data["amount"] = data["amount"].fillna(0.0) * 1000.0
            data["is_suspended"] = data["volume"].le(0.0)
            data["price_source"] = "tushare.daily"
            data["adjustment_source"] = "tushare.daily_pre_close_link"
            directory.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".parquet.tmp")
            data.to_parquet(temporary, index=False)
            temporary.replace(checkpoint)
            return data
        except Exception as exc:
            error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{type(error).__name__}: {error}")


def download_benchmark(start: str, end: str, path: Path) -> None:
    data = ak.stock_zh_index_hist_csindex(symbol="H00300", start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    data = normalize_columns(data)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end)), ["date", "symbol", "name", "close"]].dropna().drop_duplicates("date").sort_values("date")
    data["source"] = "akshare.stock_zh_index_hist_csindex"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths, periods = contract["paths"], contract["periods"]
    secret, endpoint = credentials()
    ts.set_token(secret)
    endpoints = [endpoint]
    alternate = "https://tt.xiaodefa.cn" if endpoint == "https://fast.xiaodefa.cn" else "https://fast.xiaodefa.cn"
    if alternate not in endpoints:
        endpoints.append(alternate)
    master = None
    api = None
    selected_endpoint = None
    errors = []
    for candidate in endpoints:
        candidate_api = ts.pro_api()
        candidate_api._DataApi__http_url = candidate
        try:
            master = fetch_master(candidate_api, contract)
            api = candidate_api
            selected_endpoint = candidate
            print(f"股票母表接口节点：{candidate}", flush=True)
            break
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if master is None or api is None:
        raise RuntimeError(f"两个已审核代理节点均不可用：{errors}")
    master_path = ROOT / paths["master"]
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(master_path, index=False)
    training = master.loc[
        master["split_group"].eq("TRAIN")
        & master["list_date"].le(pd.Timestamp(periods["training_data_end"]))
        & (master["delist_date"].isna() | master["delist_date"].ge(pd.Timestamp(periods["training_signal_start"])))
    ].copy()
    holdout_count = int(master["split_group"].eq("HOLDOUT").sum())
    print(f"沪深母表{len(master)}只：训练组可下载{len(training)}只，永久盲测组{holdout_count}只（收益尚未下载）", flush=True)
    directory = ROOT / paths["training_checkpoint_directory"]
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_daily, api, code, periods["training_warmup_start"], periods["training_data_end"], directory): code for code in training["ts_code"].astype(str)}
        for completed, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                failures[code] = str(exc)
            if completed % 50 == 0 or completed == len(futures):
                print(f"训练数据 {completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"训练组数据不完整：失败{len(failures)}只，样例={dict(list(failures.items())[:10])}")
    panel = pd.concat(results.values(), ignore_index=True).sort_values(["date", "con_code"]).reset_index(drop=True)
    if panel[["date", "con_code"]].duplicated().any():
        raise ValueError("训练面板存在重复键")
    abnormal = panel.groupby("con_code")["total_return_close"].pct_change(fill_method=None).abs().gt(0.30)
    age = panel.groupby("con_code").cumcount()
    unexplained = abnormal & age.gt(20)
    if unexplained.any():
        sample = panel.loc[unexplained.to_numpy(), ["date", "con_code", "raw_close", "pre_close", "total_return_close"]].head(20)
        raise ValueError(f"训练面板上市20日后仍有{int(unexplained.sum())}个单日超过30%的收益：{sample.to_dict('records')}")
    panel_path = ROOT / paths["training_panel"]
    panel.to_parquet(panel_path, index=False)
    benchmark_path = ROOT / paths["training_benchmark"]
    download_benchmark(periods["training_warmup_start"], periods["training_data_end"], benchmark_path)
    status = {
        "status": "PASS", "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "api_endpoint": selected_endpoint,
        "master_count": int(len(master)), "training_master_count": int(len(training)), "holdout_master_count": holdout_count,
        "training_downloaded_symbols": int(panel["con_code"].nunique()), "training_panel_rows": int(len(panel)),
        "holdout_returns_downloaded": False, "unexplained_returns_over_30pct": int(unexplained.sum()),
        "split_counts": master["split_bucket"].value_counts().sort_index().to_dict(),
        "hashes": {"master": sha256(master_path), "training_panel": sha256(panel_path), "training_benchmark": sha256(benchmark_path)},
    }
    status_path = ROOT / paths["training_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
