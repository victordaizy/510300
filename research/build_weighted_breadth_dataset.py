"""基于历史点时权重构建沪深300中性内部结构特征。"""

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
CONTRACT_FILE = ROOT / "config" / "index_structure_data_contract.yaml"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
DAILY_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_weighted_breadth_daily.parquet"
INDUSTRY_OUTPUT_FILE = ROOT / "data" / "features" / "000300_industry_contribution_daily.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "000300_weighted_breadth_dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段：{missing}")


def _weighted_share(frame: pd.DataFrame, condition: pd.Series, available: pd.Series) -> tuple[float | None, float]:
    eligible = available.fillna(False).astype(bool)
    denominator = float(frame.loc[eligible, "weight_decimal"].sum())
    if denominator <= 0:
        return None, 0.0
    numerator = float(frame.loc[eligible & condition.fillna(False), "weight_decimal"].sum())
    return numerator / denominator, denominator


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid].astype(float)))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float | None:
    mean = _weighted_mean(values, weights)
    if mean is None:
        return None
    valid = values.notna() & weights.notna() & (weights > 0)
    variance = np.average((values.loc[valid].astype(float) - mean) ** 2, weights=weights.loc[valid].astype(float))
    return float(np.sqrt(max(float(variance), 0.0)))


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "con_code", "total_return_close", "is_suspended"}
    _require_columns(daily, required, "成分股日线")
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["con_code"] = data["con_code"].astype(str)
    data["total_return_close"] = pd.to_numeric(data["total_return_close"], errors="coerce")
    if data[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股日线存在重复证券日期")
    if data["total_return_close"].isna().any() or (data["total_return_close"] <= 0).any():
        raise ValueError("成分股总收益价格存在空值、零值或负值")
    data["is_suspended"] = data["is_suspended"].astype(bool)
    for value_column, asof_column in (
        ("total_market_cap_cny", "market_cap_asof_date"),
        ("industry_l1", "industry_asof_date"),
    ):
        has_value = value_column in data
        has_asof = asof_column in data
        if has_value != has_asof:
            raise ValueError(f"{value_column}与{asof_column}必须同时提供")
        if has_asof:
            data[asof_column] = pd.to_datetime(data[asof_column])
            if data.loc[data[value_column].notna(), asof_column].isna().any():
                raise ValueError(f"{value_column}存在值但缺少信息截止日期")
            if (data[asof_column] > data["date"]).any():
                raise ValueError(f"{value_column}存在未来信息日期")
        if value_column == "total_market_cap_cny" and has_value:
            data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
            if (data.loc[data[value_column].notna(), value_column] <= 0).any():
                raise ValueError("总市值必须为正数")
        if value_column == "industry_l1" and has_value and "industry_source" not in data:
            raise ValueError("提供行业映射时必须同时提供industry_source")
    data.sort_values(["con_code", "date"], inplace=True)
    grouped = data.groupby("con_code", sort=False)["total_return_close"]
    data["return_1d"] = grouped.pct_change(fill_method=None)
    data["return_20d"] = grouped.pct_change(20, fill_method=None)
    data["ma20"] = grouped.transform(lambda series: series.rolling(20, min_periods=20).mean())
    data["ma60"] = grouped.transform(lambda series: series.rolling(60, min_periods=60).mean())
    return data.sort_values(["date", "con_code"]).reset_index(drop=True)


def build_weighted_breadth(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    maximum_snapshot_age_days: int = 45,
    minimum_price_coverage: float = 0.95,
    minimum_feature_coverage: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成中性内部结构特征；本函数不赋予预测方向。"""

    _require_columns(weights, {"index_code", "con_code", "trade_date", "weight"}, "历史权重")
    point_in_time_weights = weights.copy()
    point_in_time_weights["trade_date"] = pd.to_datetime(point_in_time_weights["trade_date"])
    point_in_time_weights["con_code"] = point_in_time_weights["con_code"].astype(str)
    point_in_time_weights["weight"] = pd.to_numeric(point_in_time_weights["weight"], errors="coerce")
    if point_in_time_weights[["trade_date", "con_code"]].duplicated().any():
        raise ValueError("历史权重存在重复快照证券")
    if point_in_time_weights["weight"].isna().any() or (point_in_time_weights["weight"] < 0).any():
        raise ValueError("历史权重存在空值或负值")
    snapshot_sums = point_in_time_weights.groupby("trade_date")["weight"].sum()
    if not snapshot_sums.between(98.0, 102.0).all():
        raise ValueError("历史权重快照权重和不在98%至102%之间")
    point_in_time_weights["weight_decimal"] = point_in_time_weights["weight"] / 100.0
    point_in_time_weights.sort_values(["trade_date", "weight"], ascending=[True, False], inplace=True)
    daily = _prepare_daily(constituent_daily)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    research_dates = sorted(date for date in daily["date"].unique() if start <= date <= end)
    snapshots = np.array(sorted(point_in_time_weights["trade_date"].unique()), dtype="datetime64[ns]")
    if len(snapshots) == 0:
        raise ValueError("历史权重没有快照")
    summary_rows: list[dict] = []
    industry_rows: list[dict] = []
    daily_by_date = {pd.Timestamp(date): frame for date, frame in daily.groupby("date", sort=False)}
    for date_value in research_dates:
        date = pd.Timestamp(date_value)
        snapshot_index = int(np.searchsorted(snapshots, np.datetime64(date), side="right") - 1)
        if snapshot_index < 0:
            continue
        snapshot_date = pd.Timestamp(snapshots[snapshot_index])
        snapshot_age = int((date - snapshot_date).days)
        snapshot = point_in_time_weights.loc[
            point_in_time_weights["trade_date"].eq(snapshot_date),
            ["con_code", "weight_decimal"],
        ].copy()
        snapshot["weight_rank"] = snapshot["weight_decimal"].rank(method="first", ascending=False)
        day = daily_by_date.get(date, pd.DataFrame(columns=daily.columns))
        cross = snapshot.merge(day, on="con_code", how="left", validate="one_to_one")
        price_available = cross["total_return_close"].notna()
        price_coverage = float(cross.loc[price_available, "weight_decimal"].sum())
        advancer_share, return_coverage = _weighted_share(cross, cross["return_1d"] > 0, cross["return_1d"].notna())
        momentum_share, momentum_coverage = _weighted_share(cross, cross["return_20d"] > 0, cross["return_20d"].notna())
        above_ma20, ma20_coverage = _weighted_share(
            cross, cross["total_return_close"] > cross["ma20"], cross[["total_return_close", "ma20"]].notna().all(axis=1)
        )
        above_ma60, ma60_coverage = _weighted_share(
            cross, cross["total_return_close"] > cross["ma60"], cross[["total_return_close", "ma60"]].notna().all(axis=1)
        )
        top10_weight = float(cross.loc[cross["weight_rank"] <= 10, "weight_decimal"].sum())
        row = {
            "date": date,
            "weight_snapshot_date": snapshot_date,
            "weight_snapshot_age_calendar_days": snapshot_age,
            "component_count": int(len(snapshot)),
            "price_coverage_weight": price_coverage,
            "return_1d_coverage_weight": return_coverage,
            "return_20d_coverage_weight": momentum_coverage,
            "ma20_coverage_weight": ma20_coverage,
            "ma60_coverage_weight": ma60_coverage,
            "weighted_advancer_share_1d": advancer_share,
            "weighted_positive_momentum_share_20d": momentum_share,
            "weighted_above_ma20_share": above_ma20,
            "weighted_above_ma60_share": above_ma60,
            "weighted_return_1d": _weighted_mean(cross["return_1d"], cross["weight_decimal"]),
            "weighted_return_dispersion_1d": _weighted_std(cross["return_1d"], cross["weight_decimal"]),
            "index_weight_hhi": float((cross["weight_decimal"] ** 2).sum()),
            "top10_weight_share": top10_weight,
            "valid_for_price_breadth": bool(
                snapshot_age <= maximum_snapshot_age_days
                and price_coverage >= minimum_price_coverage
                and min(return_coverage, momentum_coverage, ma20_coverage, ma60_coverage) >= minimum_feature_coverage
            ),
        }
        if "total_market_cap_cny" in cross:
            cap_valid = cross["total_market_cap_cny"].notna()
            cap_coverage = float(cross.loc[cap_valid, "weight_decimal"].sum())
            cross.loc[cap_valid, "market_cap_rank_pct"] = cross.loc[cap_valid, "total_market_cap_cny"].rank(pct=True)
            small = cap_valid & (cross["market_cap_rank_pct"] <= 1.0 / 3.0)
            large = cap_valid & (cross["market_cap_rank_pct"] > 2.0 / 3.0)
            small_advancer, _ = _weighted_share(cross, cross["return_1d"] > 0, small & cross["return_1d"].notna())
            large_advancer, _ = _weighted_share(cross, cross["return_1d"] > 0, large & cross["return_1d"].notna())
            row.update(
                {
                    "market_cap_coverage_weight": cap_coverage,
                    "small_cap_index_weight_share": float(cross.loc[small, "weight_decimal"].sum()),
                    "large_cap_index_weight_share": float(cross.loc[large, "weight_decimal"].sum()),
                    "small_cap_advancer_share_1d": small_advancer,
                    "large_cap_advancer_share_1d": large_advancer,
                }
            )
        if "industry_l1" in cross:
            industry_valid = cross["industry_l1"].notna() & cross["return_1d"].notna()
            industry_frame = cross.loc[industry_valid].copy()
            daily_industries: list[dict] = []
            for industry, group in industry_frame.groupby("industry_l1", sort=True):
                industry_weight = float(group["weight_decimal"].sum())
                contribution = float((group["weight_decimal"] * group["return_1d"]).sum())
                industry_return = None if industry_weight <= 0 else contribution / industry_weight
                advancers, _ = _weighted_share(group, group["return_1d"] > 0, group["return_1d"].notna())
                item = {
                    "date": date,
                    "weight_snapshot_date": snapshot_date,
                    "industry_l1": str(industry),
                    "industry_weight": industry_weight,
                    "industry_return_1d": industry_return,
                    "weighted_return_contribution_1d": contribution,
                    "industry_advancer_share_1d": advancers,
                    "component_count": int(len(group)),
                }
                industry_rows.append(item)
                daily_industries.append(item)
            if daily_industries:
                industries = pd.DataFrame(daily_industries)
                row["industry_coverage_weight"] = float(industries["industry_weight"].sum())
                row["positive_industry_weight_share_1d"] = float(
                    industries.loc[industries["industry_return_1d"] > 0, "industry_weight"].sum()
                    / industries["industry_weight"].sum()
                )
                row["industry_return_dispersion_1d"] = _weighted_std(
                    industries["industry_return_1d"], industries["industry_weight"]
                )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("date").reset_index(drop=True)
    industry = pd.DataFrame(industry_rows)
    if not industry.empty:
        industry.sort_values(["date", "industry_l1"], inplace=True)
        industry.reset_index(drop=True, inplace=True)
    return summary, industry


def main() -> int:
    for path in (WEIGHTS_FILE, DAILY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"指数内部结构输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    gates = contract["quality_gates"]
    summary, industry = build_weighted_breadth(
        pd.read_parquet(WEIGHTS_FILE),
        pd.read_parquet(DAILY_FILE),
        settings["project"]["start_date"],
        settings["project"]["end_date"],
        maximum_snapshot_age_days=int(contract["historical_weights"]["maximum_snapshot_age_calendar_days"]),
        minimum_price_coverage=float(gates["minimum_weighted_price_coverage"]),
        minimum_feature_coverage=float(gates["minimum_weighted_feature_coverage"]),
    )
    if summary.empty:
        raise ValueError("Weighted Breadth结果为空")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUTPUT_FILE, index=False)
    if not industry.empty:
        industry.to_parquet(INDUSTRY_OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "row_count": int(len(summary)),
        "valid_price_breadth_days": int(summary["valid_for_price_breadth"].sum()),
        "first_date": str(summary["date"].min().date()),
        "last_date": str(summary["date"].max().date()),
        "weights_sha256": _sha256(WEIGHTS_FILE),
        "constituent_daily_sha256": _sha256(DAILY_FILE),
        "contract_sha256": _sha256(CONTRACT_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "hypothesis_status": contract["governance"]["hypothesis_status"],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
