"""构建沪深300估值与波动率市场状态特征。

每一行以当日收盘为信息截止点；交易层只能在下一交易日开盘执行。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
VALUATION_FILE = PROJECT_ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "000300_market_state_daily.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "research" / "000300_market_state_summary.json"


def rolling_percentile_last(series: pd.Series) -> float:
    """返回窗口最后一个值在历史窗口中的百分位，范围为 0–1。"""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.rank(method="average", pct=True).iloc[-1])


def build_market_state_features(
    index_daily: pd.DataFrame,
    valuation_daily: pd.DataFrame,
    minimum_history: int = 1200,
) -> pd.DataFrame:
    index_data = index_daily.copy()
    valuation = valuation_daily.copy()
    index_data["date"] = pd.to_datetime(index_data["date"])
    valuation["date"] = pd.to_datetime(valuation["date"])
    columns = ["date", "pe_static", "pe_ttm", "pb"]
    missing = [column for column in columns if column not in valuation]
    if missing:
        raise ValueError(f"估值数据缺少字段：{missing}")
    data = index_data.merge(valuation[columns], on="date", how="left", validate="one_to_one")
    data = data.sort_values("date").reset_index(drop=True)

    data["return_1d"] = data["close"].pct_change()
    log_return = np.log(data["close"] / data["close"].shift(1))
    for window in [5, 20, 60, 120]:
        data[f"rv_{window}"] = log_return.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(242)
    downside = log_return.where(log_return < 0, 0.0)
    data["downside_rv_20"] = downside.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(242)
    park_variance = np.log(data["high"] / data["low"]) ** 2 / (4.0 * np.log(2.0))
    data["parkinson_rv_20"] = np.sqrt(park_variance.rolling(20, min_periods=20).mean() * 242)
    data["vol_term_20_120"] = data["rv_20"] / data["rv_120"]
    data["trend_ma20_over_ma60"] = data["close"].rolling(20).mean() / data["close"].rolling(60).mean() - 1.0
    data["trend_close_over_ma120"] = data["close"] / data["close"].rolling(120).mean() - 1.0
    data["earnings_yield"] = 1.0 / data["pe_ttm"]

    percentile_window = 242 * 5
    percentile_min = min(minimum_history, percentile_window)
    for column in ["pe_ttm", "pe_static", "pb"]:
        data[f"{column}_percentile_5y"] = data[column].rolling(
            percentile_window, min_periods=percentile_min
        ).apply(rolling_percentile_last, raw=False)

    feature_columns = [
        "pe_static", "pe_ttm", "pb", "earnings_yield", "pe_ttm_percentile_5y",
        "pe_static_percentile_5y", "pb_percentile_5y", "rv_5", "rv_20", "rv_60", "rv_120",
        "downside_rv_20", "parkinson_rv_20", "vol_term_20_120",
        "trend_ma20_over_ma60", "trend_close_over_ma120",
    ]
    for column in feature_columns:
        data[f"signal_{column}"] = data[column]
    data["signal_asof_date"] = data["date"]
    return data


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    index_data = pd.read_parquet(INDEX_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    data = build_market_state_features(index_data, valuation, minimum_history=1200)
    start = pd.Timestamp(config["project"]["start_date"])
    end = pd.Timestamp(config["project"]["end_date"])
    output = data.loc[data["date"].between(start, end)].copy()
    if output.empty or output["date"].duplicated().any():
        raise ValueError("市场状态特征为空或日期重复")
    if (output["signal_asof_date"] != output["date"]).any():
        raise ValueError("信号信息截止日必须等于信号形成日")
    required_signals = [
        "signal_pe_ttm", "signal_pb", "signal_rv_20", "signal_rv_120",
        "signal_pe_ttm_percentile_5y", "signal_pb_percentile_5y",
    ]
    if output[required_signals].isna().any().any():
        raise ValueError("回测窗口内必要市场状态特征存在空值")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_FILE, index=False)
    summary = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": int(len(output)),
        "actual_first_date": str(output["date"].min().date()),
        "actual_last_date": str(output["date"].max().date()),
        "signal_timing": "t日收盘形成，最早t+1开盘执行",
        "valuation_warmup_start": config["project"]["feature_warmup_start"],
        "features": required_signals + [
            "signal_rv_5", "signal_rv_60", "signal_downside_rv_20",
            "signal_parkinson_rv_20", "signal_vol_term_20_120",
            "signal_trend_ma20_over_ma60", "signal_trend_close_over_ma120",
        ],
        "not_included": [
            "historical_constituent_weights", "sector_breadth", "PV", "LC", "FVG"
        ],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
