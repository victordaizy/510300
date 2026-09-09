"""下载沪深300指数（000300）最近五年的原始日线数据。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """读取项目配置。"""
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fetch_index(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从新浪入口获取沪深300指数日线。"""
    source_symbol = "sh000300"
    data = ak.stock_zh_index_daily(symbol=source_symbol)
    if data is None or data.empty:
        raise RuntimeError("指数入口返回空数据")

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"指数入口缺少字段: {missing}")

    result = data[required].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    for column in required[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[result["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
    result["symbol"] = symbol
    result["source"] = "akshare.stock_zh_index_daily"
    result["volume_unit"] = "source_defined"
    return result


def validate(data: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> None:
    """执行指数日线基础校验。"""
    if data.empty:
        raise ValueError("日期范围内没有指数数据")
    if data["date"].duplicated().any():
        raise ValueError("存在重复指数交易日期")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("指数日期没有按升序排列")
    if data["symbol"].nunique() != 1 or data["symbol"].iloc[0] != symbol:
        raise ValueError("指数代码不一致")
    prices = ["open", "high", "low", "close"]
    if data[prices].isna().any().any() or (data[prices] <= 0).any().any():
        raise ValueError("指数价格存在空值或非正数")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("指数最高价逻辑不一致")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("指数最低价逻辑不一致")
    if (data["volume"] < 0).any():
        raise ValueError("指数成交量存在负数")
    if data["date"].min() < pd.Timestamp(start_date) or data["date"].max() > pd.Timestamp(end_date):
        raise ValueError("指数日期超出配置范围")


def sha256_file(path: Path) -> str:
    """计算文件SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = load_config()
    project = config["project"]
    symbol = config["symbols"]["index"]
    start_date = project["start_date"]
    end_date = project["end_date"]
    raw_dir = PROJECT_ROOT / config["runtime"]["raw_dir"] / "market"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / "000300_daily_raw.parquet"
    metadata_file = raw_dir / "000300_daily_raw.metadata.json"
    temp_file = output_file.with_suffix(".parquet.tmp")
    retrieved_at = datetime.now(ZoneInfo(project["timezone"])).isoformat()

    print(f"下载范围: {start_date} 至 {end_date}")
    print("数据源: AKShare 新浪指数日线入口")
    try:
        data = fetch_index(symbol, start_date, end_date).sort_values("date").reset_index(drop=True)
        data["retrieved_at"] = retrieved_at
        validate(data, symbol, start_date, end_date)
        data.to_parquet(temp_file, index=False, engine="pyarrow")
        temp_file.replace(output_file)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink()
        print(f"下载或校验失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "symbol": symbol,
        "start_date_requested": start_date,
        "end_date_requested": end_date,
        "actual_first_date": data["date"].min().date().isoformat(),
        "actual_last_date": data["date"].max().date().isoformat(),
        "row_count": int(len(data)),
        "source": "akshare.stock_zh_index_daily",
        "retrieved_at": retrieved_at,
        "volume_unit": "source_defined",
        "file": output_file.as_posix(),
    }
    metadata["sha256"] = sha256_file(output_file)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"保存文件: {output_file}")
    print(f"保存元信息: {metadata_file}")
    print(f"实际日期: {metadata['actual_first_date']} 至 {metadata['actual_last_date']}")
    print(f"交易日数量: {metadata['row_count']}")
    print(f"文件SHA-256: {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
