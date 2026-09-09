"""用最近已知的沪深300官方月度权重构造日频点时广度。"""

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
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
CONSTITUENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_raw.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_official_weighted_breadth_daily.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_official_weighted_breadth_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_average(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    valid = values.notna() & weights.notna() & weights.gt(0)
    coverage = float(weights.loc[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage
    return float(np.average(values.loc[valid], weights=weights.loc[valid])), coverage


def build_official_weighted_breadth(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    trading_dates: pd.Series,
    maximum_weight_age_days: int = 45,
) -> pd.DataFrame:
    """每个交易日仅使用该日及以前最近的官方权重快照。"""

    weight_data = weights.copy()
    price_data = constituent_daily.copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"])
    price_data["date"] = pd.to_datetime(price_data["date"])
    weight_data["weight"] = pd.to_numeric(weight_data["weight"], errors="coerce") / 100.0
    price_data = price_data.sort_values(["con_code", "date"]).reset_index(drop=True)
    price_data["return_1d"] = price_data.groupby("con_code", sort=False)["total_return_close"].pct_change()
    price_data["return_20d"] = price_data.groupby("con_code", sort=False)["total_return_close"].pct_change(20)
    price_data["ma20"] = price_data.groupby("con_code", sort=False)["total_return_close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    price_data["ma60"] = price_data.groupby("con_code", sort=False)["total_return_close"].transform(
        lambda values: values.rolling(60, min_periods=60).mean()
    )
    price_by_date = {date: frame for date, frame in price_data.groupby("date", sort=False)}
    snapshots = sorted(pd.Timestamp(value) for value in weight_data["trade_date"].unique())
    snapshot_lookup = {date: frame for date, frame in weight_data.groupby("trade_date", sort=False)}
    rows: list[dict[str, object]] = []
    for date in sorted(pd.to_datetime(trading_dates).dropna().unique()):
        signal_date = pd.Timestamp(date)
        eligible = [snapshot for snapshot in snapshots if snapshot <= signal_date]
        if not eligible:
            continue
        weight_date = eligible[-1]
        age = int((signal_date - weight_date).days)
        if age > maximum_weight_age_days or signal_date not in price_by_date:
            continue
        snapshot = snapshot_lookup[weight_date][["con_code", "weight"]]
        panel = snapshot.merge(
            price_by_date[signal_date][
                ["con_code", "return_1d", "return_20d", "total_return_close", "ma20", "ma60"]
            ],
            on="con_code",
            how="left",
            validate="one_to_one",
        )
        panel["advancer_1d"] = panel["return_1d"].gt(0).where(panel["return_1d"].notna())
        panel["positive_momentum_20d"] = panel["return_20d"].gt(0).where(panel["return_20d"].notna())
        panel["above_ma20"] = panel["total_return_close"].gt(panel["ma20"]).where(panel["ma20"].notna())
        panel["above_ma60"] = panel["total_return_close"].gt(panel["ma60"]).where(panel["ma60"].notna())
        advancer, return_1d_coverage = _weighted_average(panel["advancer_1d"], panel["weight"])
        momentum, return_20d_coverage = _weighted_average(panel["positive_momentum_20d"], panel["weight"])
        above_ma20, ma20_coverage = _weighted_average(panel["above_ma20"], panel["weight"])
        above_ma60, ma60_coverage = _weighted_average(panel["above_ma60"], panel["weight"])
        weighted_return, _ = _weighted_average(panel["return_1d"], panel["weight"])
        valid_return = panel["return_1d"].notna() & panel["weight"].gt(0)
        normalized_weight = panel.loc[valid_return, "weight"] / panel.loc[valid_return, "weight"].sum()
        dispersion = (
            float(
                np.sqrt(
                    np.sum(
                        normalized_weight
                        * (panel.loc[valid_return, "return_1d"] - weighted_return) ** 2
                    )
                )
            )
            if valid_return.any()
            else np.nan
        )
        rows.append(
            {
                "date": signal_date,
                "weight_snapshot_date": weight_date,
                "weight_snapshot_age_calendar_days": age,
                "component_count": int(len(snapshot)),
                "weight_sum": float(snapshot["weight"].sum()),
                "return_1d_weight_coverage": return_1d_coverage,
                "return_20d_weight_coverage": return_20d_coverage,
                "ma20_weight_coverage": ma20_coverage,
                "ma60_weight_coverage": ma60_coverage,
                "official_weighted_advancer_share_1d": advancer,
                "official_weighted_positive_momentum_share_20d": momentum,
                "official_weighted_above_ma20_share": above_ma20,
                "official_weighted_above_ma60_share": above_ma60,
                "official_weighted_return_1d": weighted_return,
                "official_weighted_return_dispersion_1d": dispersion,
                "official_weight_hhi": float((snapshot["weight"] ** 2).sum()),
                "official_top10_weight_share": float(snapshot["weight"].nlargest(10).sum()),
                "valid_for_direction_model": bool(
                    min(return_1d_coverage, return_20d_coverage, ma20_coverage, ma60_coverage) >= 0.95
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for column in [
        "official_weighted_advancer_share_1d",
        "official_weighted_positive_momentum_share_20d",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
    ]:
        result[f"{column}_change_5d"] = result[column].diff(5)
    return result


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    index = pd.read_parquet(INDEX_FILE)
    index["date"] = pd.to_datetime(index["date"])
    start = pd.Timestamp(settings["project"]["start_date"])
    end = pd.Timestamp(settings["project"]["end_date"])
    trading_dates = index.loc[index["date"].between(start, end), "date"]
    result = build_official_weighted_breadth(
        pd.read_parquet(WEIGHTS_FILE),
        pd.read_parquet(CONSTITUENT_FILE),
        trading_dates,
    )
    if result.empty or result["date"].duplicated().any():
        raise ValueError("官方权重广度为空或日期重复")
    coverage_columns = [
        "return_1d_weight_coverage",
        "return_20d_weight_coverage",
        "ma20_weight_coverage",
        "ma60_weight_coverage",
    ]
    if result[coverage_columns].min().min() < 0.95:
        raise ValueError(f"官方权重广度最低覆盖率不足95%：{result[coverage_columns].min().to_dict()}")
    if result["weight_snapshot_age_calendar_days"].max() > 45:
        raise ValueError("官方权重快照年龄超过45个自然日")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "row_count": int(len(result)),
        "first_date": str(result["date"].min().date()),
        "last_date": str(result["date"].max().date()),
        "minimum_coverages": {column: float(result[column].min()) for column in coverage_columns},
        "maximum_weight_snapshot_age_calendar_days": int(result["weight_snapshot_age_calendar_days"].max()),
        "valid_row_share": float(result["valid_for_direction_model"].mean()),
        "governance": "每个交易日只使用当日及以前最近的官方月度权重，不向快照日前回填。",
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
