"""构建510300收盘前30分钟成交压力日频特征，不读取未来收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "intraday_close_pressure_direction_challenger.yaml"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_intraday_close_pressure_daily.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "510300_intraday_close_pressure_daily_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_close_pressure_features(
    minute: pd.DataFrame,
    lookback_days: int = 60,
    expected_records_per_day: int = 241,
) -> pd.DataFrame:
    """只用信号日15:00及以前数据构造预登记特征。"""

    required = {"trade_time", "open", "close", "amount"}
    missing = required.difference(minute.columns)
    if missing:
        raise ValueError(f"分钟数据缺少字段：{sorted(missing)}")
    data = minute.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    data = data.sort_values("trade_time").reset_index(drop=True)
    if data.empty or data["trade_time"].duplicated().any():
        raise ValueError("分钟数据为空或存在重复时间戳")
    data["date"] = data["trade_time"].dt.normalize()
    data["time"] = data["trade_time"].dt.strftime("%H:%M:%S")
    counts = data.groupby("date").size()
    if not counts.eq(expected_records_per_day).all():
        bad = counts.loc[~counts.eq(expected_records_per_day)].head().to_dict()
        raise ValueError(f"存在非{expected_records_per_day}条记录的交易日：{bad}")

    required_times = {"09:30:00", "14:30:00", "15:00:00"}
    observed_times = set(data["time"].unique())
    if not required_times.issubset(observed_times):
        raise ValueError(f"分钟序列缺少关键时点：{sorted(required_times - observed_times)}")

    grouped = data.groupby("date", sort=True)
    first_open = grouped["open"].first().astype(float)
    daily_close = grouped["close"].last().astype(float)
    daily_amount = grouped["amount"].sum().astype(float)
    close_1430 = (
        data.loc[data["time"].eq("14:30:00")]
        .set_index("date")["close"]
        .astype(float)
        .reindex(first_open.index)
    )
    late = data.loc[data["time"].gt("14:30:00")]
    late_counts = late.groupby("date").size().reindex(first_open.index)
    if not late_counts.eq(30).all():
        raise ValueError("收盘窗口必须在每个交易日严格包含14:31至15:00的30条记录")
    late_amount = late.groupby("date")["amount"].sum().astype(float).reindex(first_open.index)

    result = pd.DataFrame(
        {
            "date": first_open.index,
            "signal_asof": first_open.index + pd.Timedelta(hours=15),
            "daily_open": first_open.to_numpy(),
            "close_1430": close_1430.to_numpy(),
            "daily_close": daily_close.to_numpy(),
            "daily_amount": daily_amount.to_numpy(),
            "late_amount_30m": late_amount.to_numpy(),
        }
    )
    result["late_return_30m"] = result["daily_close"] / result["close_1430"] - 1.0
    result["late_amount_share"] = result["late_amount_30m"] / result["daily_amount"]
    result["late_amount_share_prior_60d_median"] = (
        result["late_amount_share"]
        .shift(1)
        .rolling(lookback_days, min_periods=lookback_days)
        .median()
    )
    result["late_amount_share_normalized_60d"] = (
        result["late_amount_share"]
        / result["late_amount_share_prior_60d_median"].replace(0.0, np.nan)
    )
    result["close_pressure_30m_60d"] = (
        result["late_return_30m"] * result["late_amount_share_normalized_60d"]
    )
    result["feature_available"] = result["close_pressure_30m_60d"].notna()
    return result.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    minute_file = ROOT / config["data"]["minute_file"]
    quality_file = ROOT / config["data"]["minute_quality_file"]
    quality = json.loads(quality_file.read_text(encoding="utf-8"))
    if quality.get("coverage_status") != "PASS":
        raise RuntimeError("一分钟数据覆盖质量门未通过")
    minute = pd.read_parquet(minute_file)
    features = build_close_pressure_features(
        minute,
        lookback_days=int(config["feature"]["lookback_trading_days"]),
        expected_records_per_day=int(config["data"]["expected_records_per_day"]),
    )
    if int(features["feature_available"].sum()) != len(features) - 60:
        raise RuntimeError("预登记60日历史窗口的可用行数不符合预期")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)
    report = {
        "status": "PASS",
        "scope": "510300收盘前30分钟成交压力日频特征；不含任何未来收益或模型结果",
        "contains_outcomes": False,
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "challenger_id": config["research"]["challenger_id"],
        "input_file": minute_file.relative_to(ROOT).as_posix(),
        "input_sha256": _sha256(minute_file),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
        "row_count": int(len(features)),
        "feature_available_count": int(features["feature_available"].sum()),
        "first_date": str(features["date"].min().date()),
        "last_date": str(features["date"].max().date()),
        "first_feature_date": str(features.loc[features["feature_available"], "date"].min().date()),
        "late_window_records_per_day": 30,
        "lookback_days": 60,
        "signal_timing": "特征在信号日15:00记录收盘后可得；最早于下一交易日开盘执行",
        "source_caveats": quality.get("warnings", []),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
