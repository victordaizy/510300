"""从中国债券信息网下载1年和10年国债收益率曲线。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
OUTPUT_FILE = ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_daily.parquet"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_checkpoints"
REPORT_FILE = ROOT / "reports" / "data_quality" / "china_government_bond_yields_status.json"
CURVE_NAME_COLUMN = "曲线名称"
DATE_COLUMN = "日期"
ONE_YEAR_COLUMN = "1年"
TEN_YEAR_COLUMN = "10年"
GOVERNMENT_CURVE_NAME = "中债国债收益率曲线"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """生成小于一年的闭区间，符合上游接口限制。"""

    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.normalize()
    normalized_end = end.normalize()
    while cursor <= normalized_end:
        window_end = min(cursor + pd.Timedelta(days=89), normalized_end)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def normalize_curve(data: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    """筛选中债国债曲线并统一字段和单位。"""

    required = {CURVE_NAME_COLUMN, DATE_COLUMN, ONE_YEAR_COLUMN, TEN_YEAR_COLUMN}
    if missing := required - set(data.columns):
        raise ValueError(f"收益率曲线缺少字段：{sorted(missing)}")
    result = data.loc[
        data[CURVE_NAME_COLUMN].eq(GOVERNMENT_CURVE_NAME),
        [DATE_COLUMN, ONE_YEAR_COLUMN, TEN_YEAR_COLUMN],
    ].copy()
    result.columns = ["date", "cgb_1y", "cgb_10y"]
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result[["cgb_1y", "cgb_10y"]] = result[["cgb_1y", "cgb_10y"]].apply(
        pd.to_numeric, errors="coerce"
    )
    result = result.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if result.empty:
        raise ValueError("未取得中债国债收益率曲线")
    if (result[["cgb_1y", "cgb_10y"]] <= 0).any().any():
        raise ValueError("国债收益率存在零值或负值，需人工核查单位和数据源")
    result["source"] = "chinabond.via_akshare.bond_china_yield"
    result["retrieved_at"] = retrieved_at
    return result.reset_index(drop=True)


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    timezone = ZoneInfo(settings["project"]["timezone"])
    start = pd.Timestamp(settings["project"]["feature_warmup_start"])
    end = pd.Timestamp(settings["project"]["end_date"])
    retrieved_at = datetime.now(timezone)
    pieces: list[pd.DataFrame] = []
    for index, (window_start, window_end) in enumerate(build_download_windows(start, end), start=1):
        checkpoint = CHECKPOINT_DIR / f"{window_start:%Y%m%d}_{window_end:%Y%m%d}.parquet"
        if checkpoint.exists():
            normalized = pd.read_parquet(checkpoint)
        else:
            print(f"下载中债国债曲线 {window_start.date()} 至 {window_end.date()}", flush=True)
            error: Exception | None = None
            for attempt in range(4):
                try:
                    raw = ak.bond_china_yield(
                        start_date=window_start.strftime("%Y%m%d"),
                        end_date=window_end.strftime("%Y%m%d"),
                    )
                    normalized = normalize_curve(raw, retrieved_at)
                    break
                except Exception as exception:
                    error = exception
                    time.sleep(2.0 * (attempt + 1))
            else:
                raise RuntimeError(
                    f"收益率窗口{window_start.date()}至{window_end.date()}下载失败："
                    f"{type(error).__name__}: {error}"
                )
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(checkpoint, index=False)
        pieces.append(normalized)
        if index > 1:
            time.sleep(0.5)
    data = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if data["date"].min() > start + pd.Timedelta(days=10):
        raise ValueError("国债收益率首日距合同起点超过10天")
    if end - data["date"].max() > pd.Timedelta(days=10):
        raise ValueError("国债收益率末日距合同终点超过10天")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "checked_at": retrieved_at.isoformat(),
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "cgb_1y_min": float(data["cgb_1y"].min()),
        "cgb_1y_max": float(data["cgb_1y"].max()),
        "cgb_10y_min": float(data["cgb_10y"].min()),
        "cgb_10y_max": float(data["cgb_10y"].max()),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
