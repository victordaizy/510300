"""下载期权平价所需的中债短端国债收益率曲线，不覆盖既有宏观数据。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START_DATE = pd.Timestamp("2019-12-23")
END_DATE = pd.Timestamp("2026-08-14")
OUTPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "macro"
    / "china_government_bond_short_curve_v1.parquet"
)
CHECKPOINT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "macro"
    / "china_government_bond_short_curve_v1_checkpoints"
)
REPORT_FILE = (
    ROOT
    / "reports"
    / "data_quality"
    / "china_government_bond_short_curve_v1_status.json"
)
CURVE_NAME_COLUMN = "曲线名称"
DATE_COLUMN = "日期"
TENOR_COLUMNS = {
    "3月": "cgb_3m",
    "6月": "cgb_6m",
    "1年": "cgb_1y",
}
GOVERNMENT_CURVE_NAME = "中债国债收益率曲线"
SOURCE = "chinabond.via_akshare.bond_china_yield"


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """按上游接口限制生成不超过 90 个自然日的闭区间。"""

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
    """提取国债 3 月、6 月和 1 年期限，并保留百分数报价。"""

    required = {CURVE_NAME_COLUMN, DATE_COLUMN, *TENOR_COLUMNS.keys()}
    if missing := required - set(data.columns):
        raise ValueError(f"收益率曲线缺少字段：{sorted(missing)}")
    source_columns = [DATE_COLUMN, *TENOR_COLUMNS.keys()]
    result = data.loc[
        data[CURVE_NAME_COLUMN].eq(GOVERNMENT_CURVE_NAME), source_columns
    ].copy()
    result = result.rename(
        columns={DATE_COLUMN: "date", **TENOR_COLUMNS}
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    value_columns = list(TENOR_COLUMNS.values())
    result[value_columns] = result[value_columns].apply(pd.to_numeric, errors="coerce")
    result = (
        result.dropna(subset=["date", *value_columns])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    if result.empty:
        raise ValueError("未取得中债短端国债收益率曲线")
    if (result[value_columns] <= 0).any().any():
        raise ValueError("国债收益率存在零值或负值，需核查数据单位")
    result["source"] = SOURCE
    result["retrieved_at"] = retrieved_at
    return result.reset_index(drop=True)


def download_window(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """下载单个窗口；失败时有限重试并保留明确错误。"""

    error: Exception | None = None
    for attempt in range(1, 5):
        try:
            raw = ak.bond_china_yield(
                start_date=window_start.strftime("%Y%m%d"),
                end_date=window_end.strftime("%Y%m%d"),
            )
            return normalize_curve(raw, retrieved_at)
        except Exception as exception:  # pragma: no cover - 取决于外部服务
            error = exception
            print(
                f"窗口下载第 {attempt} 次失败：{type(exception).__name__}: {exception}",
                flush=True,
            )
            time.sleep(2.0 * attempt)
    raise RuntimeError(
        f"收益率窗口 {window_start.date()} 至 {window_end.date()} 下载失败："
        f"{type(error).__name__}: {error}"
    )


def main() -> int:
    """完成下载、拼接、质量核验和不可混淆的独立落盘。"""

    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    pieces: list[pd.DataFrame] = []
    windows = build_windows(START_DATE, END_DATE)
    for index, (window_start, window_end) in enumerate(windows, start=1):
        checkpoint = CHECKPOINT_DIR / f"{window_start:%Y%m%d}_{window_end:%Y%m%d}.parquet"
        if checkpoint.exists():
            normalized = pd.read_parquet(checkpoint)
            print(
                f"复用检查点 {index}/{len(windows)}："
                f"{window_start.date()} 至 {window_end.date()}",
                flush=True,
            )
        else:
            print(
                f"下载 {index}/{len(windows)}："
                f"{window_start.date()} 至 {window_end.date()}",
                flush=True,
            )
            normalized = download_window(window_start, window_end, retrieved_at)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(checkpoint, index=False)
        pieces.append(normalized)
        if index < len(windows):
            time.sleep(0.25)

    data = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if data["date"].min() > START_DATE + pd.Timedelta(days=10):
        raise ValueError("国债曲线首日距目标起点超过 10 天")
    if END_DATE - data["date"].max() > pd.Timedelta(days=10):
        raise ValueError("国债曲线末日距目标终点超过 10 天")
    if data["date"].duplicated().any():
        raise ValueError("国债曲线日期不唯一")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_FILE, index=False)
    value_columns = list(TENOR_COLUMNS.values())
    report = {
        "status": "PASS",
        "purpose": "510300 期权平价隐含现货价的期限匹配折现率",
        "checked_at": retrieved_at.isoformat(),
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "tenor_percent_ranges": {
            column: {
                "minimum": float(data[column].min()),
                "maximum": float(data[column].max()),
            }
            for column in value_columns
        },
        "source": SOURCE,
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
