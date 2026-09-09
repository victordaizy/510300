"""采集V2流动性美债重构所需证券历史，不计算策略结果。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.us_liquid_treasury_multi_asset_absolute_momentum_v2 import load_contract
from scripts.download_us_leveraged_multi_asset_history_v1 import (
    AcquisitionContractError,
    atomic_json,
    atomic_parquet,
    build_product_master,
    sha256_file,
)


INHERITED_PANEL = ROOT / "data/raw/us_leveraged_multi_asset_v1/daily_panel.parquet"
INHERITED_TICKERS = ["UPRO", "TQQQ", "UGL"]


def _output_paths(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    return {
        "product_master": ROOT / inputs["product_master"],
        "daily_panel": ROOT / inputs["daily_panel"],
        "visible_panel": ROOT / inputs["visible_panel"],
        "sealed_replication_panel": ROOT / inputs["sealed_replication_panel"],
    }


def _existing_pass_is_intact(
    status_path: Path,
    paths: dict[str, Path],
    receipt_path: Path,
) -> bool:
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY":
        return False
    if status.get("tlt_source") != "YAHOO_FINANCE_CHART_V8":
        return False
    hashes = status.get("hashes", {})
    for key, path in paths.items():
        if not path.is_file() or hashes.get(key) != sha256_file(path):
            return False
    if not receipt_path.is_file() or hashes.get("tlt_raw_receipt") != sha256_file(
        receipt_path
    ):
        return False
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return True


def _split_ratio(event: dict[str, Any]) -> float:
    numerator = event.get("numerator")
    denominator = event.get("denominator")
    if numerator is not None and denominator is not None:
        ratio = float(numerator) / float(denominator)
    else:
        text = str(event.get("splitRatio", "")).replace(":", "/")
        if "/" not in text:
            raise AcquisitionContractError(f"TLT拆分事件缺少有效比率：{event}")
        left, right = text.split("/", 1)
        ratio = float(left) / float(right)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise AcquisitionContractError(f"TLT拆分比率无效：{event}")
    return ratio


def normalize_tlt_yahoo_chart(
    payload: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """把Yahoo Chart原始OHLCV、分红和拆分规范为V1执行面板。"""

    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise AcquisitionContractError(f"Yahoo TLT业务响应失败：{chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise AcquisitionContractError("Yahoo TLT响应结果数量异常")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or []
    if not timestamps or len(quotes) != 1:
        raise AcquisitionContractError("Yahoo TLT响应缺少时间戳或OHLCV")
    quote = quotes[0]
    required = ("open", "high", "low", "close", "volume")
    if any(len(quote.get(name, [])) != len(timestamps) for name in required):
        raise AcquisitionContractError("Yahoo TLT的OHLCV长度不一致")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert(None)
            .normalize(),
            **{name: quote[name] for name in required},
        }
    )
    panel = panel.loc[panel["date"].between(start, end)].copy()
    for name in required:
        panel[name] = pd.to_numeric(panel[name], errors="coerce")
    if panel.empty or panel[list(required)].isna().any().any():
        raise AcquisitionContractError("Yahoo TLT历史为空或包含无法解析的OHLCV")
    if panel["date"].duplicated().any():
        raise AcquisitionContractError("Yahoo TLT历史日期重复")
    if (panel[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise AcquisitionContractError("Yahoo TLT历史包含非正价格")
    if panel["volume"].lt(0.0).any():
        raise AcquisitionContractError("Yahoo TLT历史包含负成交量")
    if (panel["high"] < panel[["open", "close", "low"]].max(axis=1)).any():
        raise AcquisitionContractError("Yahoo TLT历史最高价关系错误")
    if (panel["low"] > panel[["open", "close", "high"]].min(axis=1)).any():
        raise AcquisitionContractError("Yahoo TLT历史最低价关系错误")

    event_root = result.get("events", {})
    dividend_by_date: dict[pd.Timestamp, float] = {}
    for event in (event_root.get("dividends") or {}).values():
        date = (
            pd.to_datetime(int(event["date"]), unit="s", utc=True)
            .tz_convert(None)
            .normalize()
        )
        dividend_by_date[date] = dividend_by_date.get(date, 0.0) + float(
            event["amount"]
        )
    split_by_date: dict[pd.Timestamp, float] = {}
    for event in (event_root.get("splits") or {}).values():
        date = (
            pd.to_datetime(int(event["date"]), unit="s", utc=True)
            .tz_convert(None)
            .normalize()
        )
        split_by_date[date] = split_by_date.get(date, 1.0) * _split_ratio(event)
    trading_dates = set(panel["date"])
    unmatched = sorted((set(dividend_by_date) | set(split_by_date)) - trading_dates)
    if unmatched:
        rendered = [date.date().isoformat() for date in unmatched[:5]]
        raise AcquisitionContractError(f"Yahoo TLT公司行动无法映射交易日：{rendered}")
    panel["split_ratio_at_open"] = panel["date"].map(split_by_date).fillna(1.0)
    panel["qfq_factor"] = panel["split_ratio_at_open"].cumprod()
    panel["cash_distribution_per_post_event_share_usd"] = (
        panel["date"].map(dividend_by_date).fillna(0.0)
    )
    panel["adjust"] = (
        panel["cash_distribution_per_post_event_share_usd"] * panel["qfq_factor"]
    ).cumsum()
    panel["factor_effective_date"] = panel["date"]
    panel["split_adjusted_close"] = panel["close"] * panel["qfq_factor"]
    gross = (
        panel["split_adjusted_close"]
        + panel["cash_distribution_per_post_event_share_usd"] * panel["qfq_factor"]
    ) / panel["split_adjusted_close"].shift(1)
    gross.iloc[0] = 1.0
    if (~np.isfinite(gross)).any() or gross.le(0.0).any():
        raise AcquisitionContractError("Yahoo TLT单证券总收益指数增长因子无效")
    panel["signal_total_return_index"] = 100.0 * gross.cumprod()
    panel["raw_dollar_turnover_usd"] = panel["close"] * panel["volume"]
    panel["ticker"] = "TLT"
    panel = panel.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
            "volume": "raw_volume",
        }
    )
    panel["source_raw"] = "query1.finance.yahoo.com/v8/finance/chart/TLT"
    panel["source_factor"] = "Yahoo Chart dividends and splits events"
    columns = [
        "ticker",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "factor_effective_date",
        "qfq_factor",
        "adjust",
        "split_ratio_at_open",
        "cash_distribution_per_post_event_share_usd",
        "split_adjusted_close",
        "signal_total_return_index",
        "source_raw",
        "source_factor",
    ]
    return panel[columns].reset_index(drop=True)


def fetch_tlt_yahoo_chart(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    period1 = int(
        datetime.combine(start.date(), datetime.min.time(), timezone.utc).timestamp()
    )
    exclusive_end = end.date() + timedelta(days=1)
    period2 = int(
        datetime.combine(exclusive_end, datetime.min.time(), timezone.utc).timestamp()
    )
    endpoint = "https://query1.finance.yahoo.com/v8/finance/chart/TLT"
    url = endpoint + "?" + urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_tlt_yahoo_chart(payload, start=start, end=end), payload


def build_reconstituted_panel(
    inherited_panel: pd.DataFrame,
    tlt_panel: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    """组合三个原样继承产品和新采集TLT，并执行主键与日期边界检查。"""

    periods = contract["historical_partition"]
    start = pd.Timestamp(periods["acquisition_start"])
    end = pd.Timestamp(periods["formula_family_replication_end"])
    inherited = inherited_panel.loc[
        inherited_panel["ticker"].isin(INHERITED_TICKERS)
    ].copy()
    inherited["date"] = pd.to_datetime(inherited["date"]).dt.normalize()
    inherited = inherited.loc[inherited["date"].between(start, end)]
    tlt = tlt_panel.copy()
    tlt["date"] = pd.to_datetime(tlt["date"]).dt.normalize()
    tlt = tlt.loc[tlt["date"].between(start, end)]
    panel = pd.concat([inherited, tlt], ignore_index=True, sort=False)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    fixed = set(contract["universe"]["fixed_tickers"])
    if panel.empty or set(panel["ticker"].unique()) != fixed:
        raise AcquisitionContractError("V2合并价格面板未覆盖全部固定产品")
    if panel[["ticker", "date"]].duplicated().any():
        raise AcquisitionContractError("V2合并价格面板存在重复主键")
    if panel["date"].min() < start or panel["date"].max() > end:
        raise AcquisitionContractError("V2合并价格面板越过冻结日期范围")
    return panel


def main() -> int:
    contract = load_contract()
    periods = contract["historical_partition"]
    start = pd.Timestamp(periods["acquisition_start"])
    end = pd.Timestamp(periods["formula_family_replication_end"])
    visible_end = pd.Timestamp(periods["visible_end"])
    sealed_start = pd.Timestamp(periods["formula_family_replication_start"])
    status_path = ROOT / contract["inputs"]["source_status"]
    receipt_path = ROOT / contract["inputs"]["tlt_raw_receipt"]
    paths = _output_paths(contract)
    if _existing_pass_is_intact(status_path, paths, receipt_path):
        return 0
    try:
        if not INHERITED_PANEL.is_file():
            raise FileNotFoundError(f"缺少V1继承证券面板：{INHERITED_PANEL}")
        inherited = pd.read_parquet(INHERITED_PANEL)
        print("正在采集 Yahoo TLT 原始日线与公司行动事件", flush=True)
        tlt, receipt = fetch_tlt_yahoo_chart(start=start, end=end)
        atomic_json(receipt_path, receipt)
        panel = build_reconstituted_panel(inherited, tlt, contract)
        visible = panel.loc[panel["date"].le(visible_end)].copy()
        sealed = panel.loc[panel["date"].between(sealed_start, end)].copy()
        if visible.empty or sealed.empty:
            raise AcquisitionContractError("V2价格面板的可见期或封存期为空")
        if visible["date"].max() >= sealed["date"].min():
            raise AcquisitionContractError("V2价格可见期与封存期日期重叠")
        product_master = build_product_master(contract)
        frames = {
            "product_master": product_master,
            "daily_panel": panel,
            "visible_panel": visible,
            "sealed_replication_panel": sealed,
        }
        for key, path in paths.items():
            atomic_parquet(path, frames[key])

        inherited_sources = {
            "v1_daily_panel": INHERITED_PANEL,
            "official_fx_full": ROOT / contract["inputs"]["fx_full"],
            "official_fx_visible": ROOT / contract["inputs"]["visible_fx"],
            "official_fx_sealed": ROOT / contract["inputs"]["sealed_replication_fx"],
            "visible_benchmark": ROOT / contract["inputs"]["visible_benchmark"],
            "sealed_benchmark": ROOT / contract["inputs"]["sealed_replication_benchmark"],
        }
        missing = [str(path) for path in inherited_sources.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"V2缺少继承输入：{missing}")
        status = {
            "schema_version": "1.0.0",
            "status": "PASS_INPUT_ACQUISITION_ONLY",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": contract["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "security_total_return_index_computed": True,
            "security_price_inputs_reconstituted_before_outcome": True,
            "tlt_source": "YAHOO_FINANCE_CHART_V8",
            "inherited_tickers": INHERITED_TICKERS,
            "newly_acquired_ticker": "TLT",
            "fixed_tickers": contract["universe"]["fixed_tickers"],
            "panel_rows": int(len(panel)),
            "visible_panel_rows": int(len(visible)),
            "sealed_panel_rows": int(len(sealed)),
            "rows_by_ticker": {
                str(key): int(value)
                for key, value in panel.groupby("ticker", observed=True).size().items()
            },
            "visible_date_range": [
                visible["date"].min().date().isoformat(),
                visible["date"].max().date().isoformat(),
            ],
            "sealed_date_range": [
                sealed["date"].min().date().isoformat(),
                sealed["date"].max().date().isoformat(),
            ],
            "split_event_rows": int(panel["split_ratio_at_open"].ne(1.0).sum()),
            "cash_distribution_event_rows": int(
                panel["cash_distribution_per_post_event_share_usd"].gt(0.0).sum()
            ),
            "sources": {
                "inherited_security_panel": INHERITED_PANEL.relative_to(ROOT).as_posix(),
                "tlt_raw": "Yahoo Finance Chart v8原始OHLCV",
                "tlt_factor": "Yahoo Finance Chart v8现金分配与拆分事件",
                "fx": "中国外汇交易中心人民币汇率中间价历史查询",
                "benchmark": contract["inputs"]["visible_benchmark"],
            },
            "hashes": {
                **{key: sha256_file(path) for key, path in paths.items()},
                "tlt_raw_receipt": sha256_file(receipt_path),
                **{
                    f"inherited_{key}": sha256_file(path)
                    for key, path in inherited_sources.items()
                },
            },
        }
        atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_INPUT_ACQUISITION",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": contract["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json(status_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
