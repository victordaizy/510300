"""分析 510300 五年价格回报、分红总回报、隔夜与盘中贡献。"""

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
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "510300_daily_total_return.parquet"
REPORT_DIR = PROJECT_ROOT / "reports" / "research"
REPORT_FILE = REPORT_DIR / "510300_return_anatomy.json"
TABLE_FILE = REPORT_DIR / "510300_return_anatomy_daily.parquet"


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def add_return_columns(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """合并除息现金并生成不会把分红误判为亏损的回报字段。"""
    data = prices.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    events = dividends[["ex_date", "cash_dividend_per_share"]].copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events = events.groupby("ex_date", as_index=False)["cash_dividend_per_share"].sum()
    data = data.merge(events, left_on="date", right_on="ex_date", how="left")
    data["cash_dividend_per_share"] = data["cash_dividend_per_share"].fillna(0.0)
    data = data.drop(columns=["ex_date"])

    data["prev_close"] = data["close"].shift(1)
    data["price_return"] = data["close"] / data["prev_close"] - 1.0
    data["dividend_yield"] = data["cash_dividend_per_share"] / data["prev_close"]
    data["total_return"] = (data["close"] + data["cash_dividend_per_share"]) / data["prev_close"] - 1.0
    data["overnight_price_return"] = data["open"] / data["prev_close"] - 1.0
    data["overnight_total_contribution"] = (
        data["open"] + data["cash_dividend_per_share"] - data["prev_close"]
    ) / data["prev_close"]
    data["intraday_return"] = data["close"] / data["open"] - 1.0
    data["intraday_contribution"] = (data["close"] - data["open"]) / data["prev_close"]
    data["decomposition_error"] = (
        data["total_return"] - data["overnight_total_contribution"] - data["intraday_contribution"]
    )
    data["range_return"] = data["high"] / data["low"] - 1.0
    data["year"] = data["date"].dt.year
    return data


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def summarize(returns: pd.Series, periods_per_year: int = 242) -> dict:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {key: None for key in [
            "cumulative_return", "annualized_return", "annualized_volatility",
            "positive_ratio", "mean_return", "median_return", "max_drawdown",
        ]} | {"observations": 0}
    cumulative = float((1.0 + clean).prod() - 1.0)
    years = len(clean) / periods_per_year
    annualized = float((1.0 + cumulative) ** (1.0 / years) - 1.0) if cumulative > -1 and years > 0 else None
    return {
        "observations": int(len(clean)),
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": float(clean.std(ddof=1) * np.sqrt(periods_per_year)) if len(clean) > 1 else None,
        "positive_ratio": float((clean > 0).mean()),
        "mean_return": float(clean.mean()),
        "median_return": float(clean.median()),
        "max_drawdown": max_drawdown(clean),
    }


def main() -> int:
    config = load_config()
    prices = pd.read_parquet(DATA_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    data = add_return_columns(prices, dividends)
    if data["decomposition_error"].abs().max() > 1e-12:
        raise RuntimeError("总回报的隔夜/盘中贡献分解不守恒")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(TABLE_FILE, index=False)
    data.to_parquet(PROCESSED_FILE, index=False)
    summary = {
        "generated_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "symbol": config["symbols"]["etf"],
        "requested_range": [config["project"]["start_date"], config["project"]["end_date"]],
        "actual_range": [data["date"].min().date().isoformat(), data["date"].max().date().isoformat()],
        "observations": int(len(data)),
        "dividend_events": int((data["cash_dividend_per_share"] > 0).sum()),
        "return_units": "decimal, 0.01 means 1%",
        "method": {
            "price_return": "close_t / close_t-1 - 1",
            "total_return": "(close_t + ex_date_cash_dividend_t) / close_t-1 - 1",
            "decomposition": "total_return = overnight_total_contribution + intraday_contribution",
        },
        "all_periods": {
            "price_return": summarize(data["price_return"]),
            "total_return": summarize(data["total_return"]),
            "overnight_total_contribution": summarize(data["overnight_total_contribution"]),
            "intraday_return": summarize(data["intraday_return"]),
        },
        "by_year": {},
    }
    for year, group in data.groupby("year", sort=True):
        summary["by_year"][str(year)] = {
            "price_return": summarize(group["price_return"]),
            "total_return": summarize(group["total_return"]),
            "overnight_total_contribution": summarize(group["overnight_total_contribution"]),
            "intraday_return": summarize(group["intraday_return"]),
        }
    REPORT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"处理后总回报数据：{PROCESSED_FILE}")
    print(f"研究报告：{REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
