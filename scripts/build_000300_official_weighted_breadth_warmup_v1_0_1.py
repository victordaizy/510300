"""补齐评价期前五个交易日的官方权重广度预热行，不读取候选收益。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from build_official_weighted_breadth import build_official_weighted_breadth  # noqa: E402


WEIGHTS = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
CONSTITUENTS = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
ETF_CALENDAR = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
BASE_BREADTH = ROOT / "data" / "features" / "000300_official_weighted_breadth_daily.parquet"
OUTPUT = (
    ROOT
    / "data"
    / "features"
    / "000300_official_weighted_breadth_daily_v1_0_1_warmup.parquet"
)
REPORT = (
    ROOT
    / "reports"
    / "data_quality"
    / "000300_official_weighted_breadth_warmup_v1_0_1.json"
)
WARMUP_START = pd.Timestamp("2021-08-05")
EVALUATION_START = pd.Timestamp("2021-08-12")
EVALUATION_END = pd.Timestamp("2026-08-12")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    calendar = pd.read_parquet(ETF_CALENDAR, columns=["date"])
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    trading_dates = calendar.loc[
        calendar["date"].between(WARMUP_START, EVALUATION_START, inclusive="left"),
        "date",
    ]
    warmup = build_official_weighted_breadth(
        pd.read_parquet(WEIGHTS),
        pd.read_parquet(CONSTITUENTS),
        trading_dates,
        maximum_weight_age_days=35,
    )
    if warmup.empty or warmup["date"].duplicated().any():
        raise ValueError("预热版官方权重广度为空或日期重复")
    expected_dates = pd.DatetimeIndex(trading_dates.drop_duplicates().sort_values())
    actual_dates = pd.DatetimeIndex(warmup["date"])
    if not actual_dates.equals(expected_dates):
        missing = expected_dates.difference(actual_dates)
        extra = actual_dates.difference(expected_dates)
        raise ValueError(f"预热版日期不完整：missing={list(missing)}, extra={list(extra)}")
    coverage_columns = ["ma20_weight_coverage", "ma60_weight_coverage"]
    if warmup[coverage_columns].min().min() < 0.95:
        raise ValueError("仅用于计算五日变化的预热行覆盖率低于95%")
    if (warmup["weight_snapshot_date"] > warmup["date"]).any():
        raise ValueError("预热版官方权重广度使用未来权重")
    if warmup["weight_snapshot_age_calendar_days"].max() > 35:
        raise ValueError("预热版官方权重快照年龄超过35个自然日")
    if (warmup["component_count"] != 300).any():
        raise ValueError("预热版官方权重广度并非每日300只")
    base = pd.read_parquet(BASE_BREADTH)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base["weight_snapshot_date"] = pd.to_datetime(
        base["weight_snapshot_date"]
    ).dt.normalize()
    if len(base) != 1211 or base["date"].duplicated().any():
        raise ValueError("既有审计通过的评价期官方权重广度行数或日期不正确")
    combined = pd.concat([warmup, base], ignore_index=True).sort_values("date")
    if combined["date"].duplicated().any():
        raise ValueError("预热行与既有评价期广度发生日期重叠")
    change_sources = [
        "official_weighted_advancer_share_1d",
        "official_weighted_positive_momentum_share_20d",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
    ]
    original_changes = base.set_index("date")[[f"{column}_change_5d" for column in change_sources]]
    for column in change_sources:
        combined[f"{column}_change_5d"] = combined[column].diff(5)
    evaluation = combined.loc[
        combined["date"].between(EVALUATION_START, EVALUATION_END)
    ].copy()
    if len(evaluation) != 1211:
        raise ValueError(f"评价期行数不是1211：{len(evaluation)}")
    if evaluation[coverage_columns].min().min() < 0.99:
        raise ValueError("评价期官方权重广度覆盖率低于99%")
    change_column = "official_weighted_above_ma20_share_change_5d"
    if evaluation[change_column].isna().any():
        raise ValueError("补齐后评价期B20五日变化仍有缺失")
    overlap_dates = original_changes.dropna().index.intersection(evaluation["date"])
    recalculated = evaluation.set_index("date").loc[
        overlap_dates, original_changes.columns
    ]
    maximum_existing_change_difference = float(
        (recalculated - original_changes.loc[overlap_dates]).abs().max().max()
    )
    if maximum_existing_change_difference > 1e-12:
        raise ValueError("补预热行意外改变了既有非缺失的五日广度变化")
    atomic_parquet(OUTPUT, evaluation)
    report = {
        "status": "PASS_DATA_REMEDIATION_ONLY_NO_CANDIDATE_RETURN_READ",
        "generated_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "purpose": "只补评价期前五个交易日的官方权重广度预热行",
        "warmup_start": WARMUP_START.date().isoformat(),
        "evaluation_start": EVALUATION_START.date().isoformat(),
        "evaluation_end": EVALUATION_END.date().isoformat(),
        "warmup_build_row_count": len(warmup),
        "row_count": len(evaluation),
        "evaluation_row_count": len(evaluation),
        "first_date": evaluation["date"].min().date().isoformat(),
        "last_date": evaluation["date"].max().date().isoformat(),
        "warmup_minimum_ma20_weight_coverage": float(
            warmup["ma20_weight_coverage"].min()
        ),
        "warmup_minimum_ma60_weight_coverage": float(
            warmup["ma60_weight_coverage"].min()
        ),
        "minimum_ma20_weight_coverage": float(
            evaluation["ma20_weight_coverage"].min()
        ),
        "minimum_ma60_weight_coverage": float(
            evaluation["ma60_weight_coverage"].min()
        ),
        "maximum_weight_snapshot_age_calendar_days": int(
            evaluation["weight_snapshot_age_calendar_days"].max()
        ),
        "future_weight_snapshot_rows": int(
            (evaluation["weight_snapshot_date"] > evaluation["date"]).sum()
        ),
        "evaluation_b20_change_missing_rows": int(evaluation[change_column].isna().sum()),
        "maximum_existing_change_difference": maximum_existing_change_difference,
        "input_hashes": {
            WEIGHTS.relative_to(ROOT).as_posix(): sha256_file(WEIGHTS),
            CONSTITUENTS.relative_to(ROOT).as_posix(): sha256_file(CONSTITUENTS),
            ETF_CALENDAR.relative_to(ROOT).as_posix(): sha256_file(ETF_CALENDAR),
            BASE_BREADTH.relative_to(ROOT).as_posix(): sha256_file(BASE_BREADTH),
        },
        "output_file": OUTPUT.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(OUTPUT),
        "candidate_outcomes_computed": False,
        "live_trading_authorized": False,
    }
    atomic_json(REPORT, report)
    return report


def main() -> int:
    report = build()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
