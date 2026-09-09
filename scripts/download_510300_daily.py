"""下载510300特征预热期至指定截止日的原始日线数据。

优先使用AKShare新浪入口，因为当前网络环境访问东方财富入口不稳定。
脚本不会修改已有原始文件，而是先写入临时文件，校验通过后再替换目标文件。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quality_check_daily import write_daily_metadata


CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """读取项目配置。"""
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fetch_from_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从新浪入口获取ETF完整日线，再按项目日期范围筛选。"""
    sina_symbol = f"sh{symbol[:6]}" if symbol.endswith(".SH") else f"sh{symbol}"
    data = ak.fund_etf_hist_sina(symbol=sina_symbol)
    if data is None or data.empty:
        raise RuntimeError("新浪入口返回空数据")

    data = data.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    required_columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"新浪入口缺少字段: {missing_columns}")

    result = data[required_columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    for column in required_columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    result = result.loc[result["date"].between(start, end)].copy()
    result["symbol"] = symbol
    result["source"] = "akshare.fund_etf_hist_sina"
    result["volume_unit"] = "share"
    result["amount_unit"] = "CNY"
    return result


def validate(data: pd.DataFrame, start_date: str, end_date: str) -> None:
    """执行下载后的基础质量检查。"""
    required = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    if data.empty:
        raise ValueError("日期范围内没有返回数据")
    if data["date"].duplicated().any():
        raise ValueError("存在重复交易日期")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("日期没有按升序排列")
    price_columns = ["open", "high", "low", "close"]
    if data[price_columns].isna().any().any():
        raise ValueError("价格字段存在空值")
    if (data[price_columns] <= 0).any().any():
        raise ValueError("价格字段存在非正数")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("最高价低于开盘价、收盘价或最低价")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("最低价高于开盘价、收盘价或最高价")
    if (data[["volume", "amount"]] < 0).any().any():
        raise ValueError("成交量或成交额存在负数")
    if data["date"].min() < pd.Timestamp(start_date) or data["date"].max() > pd.Timestamp(end_date):
        raise ValueError("返回日期超出配置范围")


def main() -> int:
    config = load_config()
    project = config["project"]
    symbols = config["symbols"]
    runtime = config["runtime"]
    symbol = symbols["etf"]
    start_date = project.get("feature_warmup_start", project["start_date"])
    end_date = project["end_date"]

    raw_dir = PROJECT_ROOT / runtime["raw_dir"] / "market"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file = raw_dir / "510300_daily_raw.parquet"
    metadata_file = raw_dir / "510300_daily_raw.metadata.json"
    temp_file = output_file.with_suffix(".parquet.tmp")

    retrieved_at = datetime.now(ZoneInfo(project["timezone"])).isoformat()
    print(f"下载范围: {start_date} 至 {end_date}")
    print("数据源: AKShare 新浪ETF历史日线入口")

    try:
        data = fetch_from_sina(symbol, start_date, end_date)
        data = data.sort_values("date").reset_index(drop=True)
        data["retrieved_at"] = retrieved_at
        validate(data, start_date, end_date)
        data.to_parquet(temp_file, index=False, engine="pyarrow")
        temp_file.replace(output_file)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink()
        print(f"下载或校验失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    metadata = write_daily_metadata(
        data,
        output_file,
        metadata_file,
        requested_start=start_date,
        requested_end=end_date,
        evaluation_start=project["start_date"],
        historical_evaluation_end=project["end_date"],
    )

    print(f"保存文件: {output_file}")
    print(f"保存元信息: {metadata_file}")
    print(f"实际日期: {metadata['actual_first_date']} 至 {metadata['actual_last_date']}")
    print(f"交易日数量: {metadata['row_count']}")
    print(f"文件SHA-256: {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
