"""下载五年沪深300历史成员的Tushare L2分类资金流。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "flow" / "component_moneyflow_checkpoints"
OUTPUT_FILE = ROOT / "data" / "raw" / "flow" / "csi300_component_moneyflow_daily.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "csi300_component_moneyflow_status.json"

AMOUNT_COLUMNS = [
    "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
    "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
    "net_mf_amount",
]
VOLUME_COLUMNS = [
    "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
    "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol", "net_mf_vol",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_token() -> str:
    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError("未设置TUSHARE_TOKEN或TS_TOKEN")
    return token


def normalize_moneyflow(data: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", *AMOUNT_COLUMNS, *VOLUME_COLUMNS}
    if missing := required - set(data.columns):
        raise ValueError(f"moneyflow缺少字段：{sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result[AMOUNT_COLUMNS + VOLUME_COLUMNS] = result[AMOUNT_COLUMNS + VOLUME_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    result = result.dropna(subset=["ts_code", "date"]).sort_values(["ts_code", "date"])
    result = result.drop_duplicates(["ts_code", "date"], keep="last")
    result["large_extra_large_net_amount_10k_cny"] = (
        result["buy_lg_amount"] + result["buy_elg_amount"]
        - result["sell_lg_amount"] - result["sell_elg_amount"]
    )
    result["small_net_amount_10k_cny"] = result["buy_sm_amount"] - result["sell_sm_amount"]
    return result[
        [
            "date", "ts_code", *AMOUNT_COLUMNS, *VOLUME_COLUMNS,
            "large_extra_large_net_amount_10k_cny", "small_net_amount_10k_cny",
        ]
    ].reset_index(drop=True)


def _download_with_retry(call, label: str, attempts: int = 4) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None:
                raise ValueError("返回None")
            return result
        except Exception as exception:
            error = exception
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label}下载失败：{type(error).__name__}: {error}")


def main() -> int:
    token = ""
    try:
        token = get_token()
        settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
        start = pd.Timestamp(settings["project"]["start_date"])
        end = pd.Timestamp(settings["project"]["end_date"])
        retrieved_at = datetime.now(ZoneInfo(settings["project"]["timezone"]))
        weights = pd.read_parquet(WEIGHTS_FILE)
        weights["trade_date"] = pd.to_datetime(weights["trade_date"])
        symbols = sorted(
            weights.loc[weights["trade_date"].between(start - pd.Timedelta(days=45), end), "con_code"]
            .dropna().astype(str).unique()
        )
        api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
        interval = max(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")), 0.5)
        pro = ts.pro_api(token)
        pro._DataApi__http_url = api_url
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        pieces: list[pd.DataFrame] = []
        for index, symbol in enumerate(symbols, start=1):
            checkpoint = CHECKPOINT_DIR / f"{symbol.replace('.', '_')}.parquet"
            if checkpoint.exists():
                part = pd.read_parquet(checkpoint)
            else:
                part = _download_with_retry(
                    lambda symbol=symbol: pro.query(
                        "moneyflow",
                        ts_code=symbol,
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                    ),
                    symbol,
                )
                part.to_parquet(checkpoint, index=False)
                time.sleep(interval)
            if not part.empty:
                pieces.append(part)
            if index % 25 == 0 or index == len(symbols):
                print(f"成分股资金流下载进度 {index}/{len(symbols)}", flush=True)
        if not pieces:
            raise ValueError("所有成分股moneyflow均为空")
        data = normalize_moneyflow(pd.concat(pieces, ignore_index=True))
        data["source"] = "tushare.moneyflow"
        data["retrieved_at"] = retrieved_at
        covered_symbols = data["ts_code"].nunique()
        coverage = covered_symbols / max(len(symbols), 1)
        if coverage < 0.95:
            raise ValueError(f"成分股资金流覆盖率仅{coverage:.2%}")
        if data[["ts_code", "date"]].duplicated().any():
            raise ValueError("成分股资金流存在重复股票日期")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(OUTPUT_FILE, index=False)
        report = {
            "status": "PASS",
            "checked_at": retrieved_at.isoformat(),
            "api_host": api_url.split("//", 1)[-1].split("/", 1)[0],
            "credential_note": "报告不保存或回显Token。",
            "requested_symbol_count": len(symbols),
            "covered_symbol_count": int(covered_symbols),
            "symbol_coverage": float(coverage),
            "row_count": int(len(data)),
            "first_date": str(data["date"].min().date()),
            "last_date": str(data["date"].max().date()),
            "amount_unit": "万元",
            "source_semantics": "Tushare moneyflow按L2订单分类提供小/中/大/特大单及净主力金额；不是ETF自身资金流。",
            "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
            "output_sha256": _sha256(OUTPUT_FILE),
        }
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        message = str(error).replace(token, "[REDACTED]") if token else str(error)
        print(f"成分股资金流下载失败：{type(error).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
