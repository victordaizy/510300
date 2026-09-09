"""用最近已知官方沪深300权重聚合逐股L2分类资金流。"""

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
MONEYFLOW_FILE = ROOT / "data" / "raw" / "flow" / "csi300_component_moneyflow_daily.parquet"
PRICE_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_official_weighted_moneyflow_daily.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_official_weighted_moneyflow_status.json"


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


def build_official_weighted_moneyflow(
    weights: pd.DataFrame,
    moneyflow: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    maximum_weight_age_days: int = 45,
) -> pd.DataFrame:
    weight_data = weights.copy()
    flow = moneyflow.copy()
    price = prices[["date", "con_code", "amount"]].copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"])
    flow["date"] = pd.to_datetime(flow["date"])
    price["date"] = pd.to_datetime(price["date"])
    weight_data["weight"] = pd.to_numeric(weight_data["weight"], errors="coerce") / 100.0
    flow = flow.merge(
        price,
        left_on=["date", "ts_code"],
        right_on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    flow["net_mf_intensity"] = flow["net_mf_amount"] * 10000.0 / flow["amount"]
    flow["large_extra_large_net_intensity"] = (
        flow["large_extra_large_net_amount_10k_cny"] * 10000.0 / flow["amount"]
    )
    flow["small_net_intensity"] = flow["small_net_amount_10k_cny"] * 10000.0 / flow["amount"]
    flow_by_date = {date: frame for date, frame in flow.groupby("date", sort=False)}
    snapshots = sorted(pd.Timestamp(value) for value in weight_data["trade_date"].unique())
    snapshot_lookup = {date: frame for date, frame in weight_data.groupby("trade_date", sort=False)}
    trading_dates = sorted(date for date in flow_by_date if start <= date <= end)
    rows: list[dict[str, object]] = []
    for date in trading_dates:
        eligible = [snapshot for snapshot in snapshots if snapshot <= date]
        if not eligible:
            continue
        snapshot_date = eligible[-1]
        age = int((date - snapshot_date).days)
        if age > maximum_weight_age_days:
            continue
        snapshot = snapshot_lookup[snapshot_date][["con_code", "weight"]]
        panel = snapshot.merge(
            flow_by_date[date][
                [
                    "ts_code", "net_mf_intensity", "large_extra_large_net_intensity",
                    "small_net_intensity",
                ]
            ],
            left_on="con_code",
            right_on="ts_code",
            how="left",
            validate="one_to_one",
        )
        net_mf, coverage = _weighted_average(panel["net_mf_intensity"], panel["weight"])
        large, large_coverage = _weighted_average(
            panel["large_extra_large_net_intensity"], panel["weight"]
        )
        small, small_coverage = _weighted_average(panel["small_net_intensity"], panel["weight"])
        positive_net, _ = _weighted_average(
            panel["net_mf_intensity"].gt(0).where(panel["net_mf_intensity"].notna()),
            panel["weight"],
        )
        positive_large, _ = _weighted_average(
            panel["large_extra_large_net_intensity"].gt(0).where(
                panel["large_extra_large_net_intensity"].notna()
            ),
            panel["weight"],
        )
        valid = panel["net_mf_intensity"].notna() & panel["weight"].gt(0)
        normalized = panel.loc[valid, "weight"] / panel.loc[valid, "weight"].sum()
        dispersion = float(
            np.sqrt(
                np.sum(normalized * (panel.loc[valid, "net_mf_intensity"] - net_mf) ** 2)
            )
        )
        rows.append(
            {
                "date": date,
                "weight_snapshot_date": snapshot_date,
                "weight_snapshot_age_calendar_days": age,
                "component_count": int(len(snapshot)),
                "moneyflow_weight_coverage": coverage,
                "large_flow_weight_coverage": large_coverage,
                "small_flow_weight_coverage": small_coverage,
                "official_weighted_net_mf_intensity_1d": net_mf,
                "official_weighted_large_extra_large_intensity_1d": large,
                "official_weighted_small_intensity_1d": small,
                "official_weighted_positive_net_mf_share_1d": positive_net,
                "official_weighted_positive_large_flow_share_1d": positive_large,
                "official_weighted_net_mf_dispersion_1d": dispersion,
            }
        )
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    result["official_weighted_net_mf_intensity_5d"] = result[
        "official_weighted_net_mf_intensity_1d"
    ].rolling(5, min_periods=5).mean()
    result["official_weighted_large_extra_large_intensity_5d"] = result[
        "official_weighted_large_extra_large_intensity_1d"
    ].rolling(5, min_periods=5).mean()
    result["official_weighted_positive_net_mf_share_5d"] = result[
        "official_weighted_positive_net_mf_share_1d"
    ].rolling(5, min_periods=5).mean()
    result["official_weighted_positive_large_flow_share_5d"] = result[
        "official_weighted_positive_large_flow_share_1d"
    ].rolling(5, min_periods=5).mean()
    return result


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    start = pd.Timestamp(settings["project"]["start_date"])
    end = pd.Timestamp(settings["project"]["end_date"])
    result = build_official_weighted_moneyflow(
        pd.read_parquet(WEIGHTS_FILE),
        pd.read_parquet(MONEYFLOW_FILE),
        pd.read_parquet(PRICE_FILE),
        start,
        end,
    )
    coverage_columns = [
        "moneyflow_weight_coverage", "large_flow_weight_coverage", "small_flow_weight_coverage"
    ]
    if result.empty or result["date"].duplicated().any():
        raise ValueError("官方权重资金流聚合为空或日期重复")
    if result[coverage_columns].min().min() < 0.95:
        raise ValueError(f"官方权重资金流最低覆盖率不足95%：{result[coverage_columns].min().to_dict()}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "row_count": int(len(result)),
        "first_date": str(result["date"].min().date()),
        "last_date": str(result["date"].max().date()),
        "minimum_coverages": {column: float(result[column].min()) for column in coverage_columns},
        "maximum_weight_snapshot_age_calendar_days": int(
            result["weight_snapshot_age_calendar_days"].max()
        ),
        "unit_audit": "分类金额乘10000后与个股成交额（元）比较；全样本分类买卖额均值/成交额中位数约1。",
        "governance": "只使用当日及以前最近官方月度权重；逐股资金流强度先除以同日成交额再按指数权重聚合。",
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
