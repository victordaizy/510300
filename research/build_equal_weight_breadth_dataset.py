"""基于历史点时成员构建沪深300等权内部宽度特征。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.build_weighted_breadth_dataset import build_weighted_breadth


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round3.yaml"
MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
DAILY_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_equal_weight_breadth_daily.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "000300_equal_weight_breadth_dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_equal_weight_snapshots(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "con_code", "is_index_member"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"成分股连续日线缺少字段：{missing}")
    members = daily.loc[daily["is_index_member"], ["date", "con_code"]].copy()
    members["date"] = pd.to_datetime(members["date"])
    if members[["date", "con_code"]].duplicated().any():
        raise ValueError("点时成员存在重复证券日期")
    counts = members.groupby("date")["con_code"].transform("count")
    if counts.min() != 300 or counts.max() != 300:
        raise ValueError(f"点时成员数不是每日300只：{counts.min()}至{counts.max()}")
    members["index_code"] = "000300.SH"
    members["trade_date"] = members["date"]
    members["weight"] = 100.0 / counts
    return members[["index_code", "con_code", "trade_date", "weight"]]


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, MEMBERSHIP_FILE, DAILY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"等权宽度输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    daily = pd.read_parquet(DAILY_FILE)
    weights = _build_equal_weight_snapshots(daily)
    summary, industry = build_weighted_breadth(
        weights,
        daily,
        settings["project"]["start_date"],
        settings["project"]["end_date"],
        maximum_snapshot_age_days=0,
        minimum_price_coverage=0.95,
        minimum_feature_coverage=0.90,
    )
    if not industry.empty:
        raise ValueError("本轮没有点时行业分类，等权宽度构建器不应生成行业结果")
    rename = {
        "weighted_advancer_share_1d": "equal_advancer_share_1d",
        "weighted_positive_momentum_share_20d": "equal_positive_momentum_share_20d",
        "weighted_above_ma20_share": "equal_above_ma20_share",
        "weighted_above_ma60_share": "equal_above_ma60_share",
        "weighted_return_1d": "equal_return_1d",
        "weighted_return_dispersion_1d": "equal_return_dispersion_1d",
    }
    summary.rename(columns=rename, inplace=True)
    summary["small_minus_large_advancer_share_1d"] = (
        summary["small_cap_advancer_share_1d"] - summary["large_cap_advancer_share_1d"]
    )
    summary["valid_for_cap_breadth"] = (
        summary["market_cap_coverage_weight"].ge(0.95)
        & summary["small_cap_advancer_share_1d"].notna()
        & summary["large_cap_advancer_share_1d"].notna()
    )
    summary["breadth_weighting_semantics"] = "POINT_IN_TIME_MEMBERSHIP_EQUAL_WEIGHT_NOT_OFFICIAL_INDEX_WEIGHT"
    summary["market_cap_semantics"] = "SINA_CIRCULATING_MARKET_CAP_APPROXIMATION_NOT_TOTAL_MARKET_CAP"

    factor_columns = [item["column"] for item in registry["factor_definitions"]]
    missing_factors = sorted(set(factor_columns).difference(summary.columns))
    if missing_factors:
        raise ValueError(f"第三轮登记因子未生成：{missing_factors}")
    if summary[factor_columns].isna().any().any():
        null_counts = summary[factor_columns].isna().sum()
        raise ValueError(f"第三轮登记因子存在空值：{null_counts[null_counts > 0].to_dict()}")
    if not summary["valid_for_price_breadth"].all():
        raise ValueError("存在未通过价格宽度覆盖门槛的研究日")
    if not summary["valid_for_cap_breadth"].all():
        raise ValueError("存在未通过流通市值分层覆盖门槛的研究日")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "registration_timing": registry["registration_timing"],
        "row_count": int(len(summary)),
        "first_date": str(summary["date"].min().date()),
        "last_date": str(summary["date"].max().date()),
        "component_count_min": int(summary["component_count"].min()),
        "component_count_max": int(summary["component_count"].max()),
        "minimum_price_coverage_weight": float(summary["price_coverage_weight"].min()),
        "minimum_market_cap_coverage_weight": float(summary["market_cap_coverage_weight"].min()),
        "valid_price_breadth_days": int(summary["valid_for_price_breadth"].sum()),
        "valid_cap_breadth_days": int(summary["valid_for_cap_breadth"].sum()),
        "factor_columns": factor_columns,
        "weighting_semantics": "POINT_IN_TIME_MEMBERSHIP_EQUAL_WEIGHT_NOT_OFFICIAL_INDEX_WEIGHT",
        "market_cap_semantics": "SINA_CIRCULATING_MARKET_CAP_APPROXIMATION_NOT_TOTAL_MARKET_CAP",
        "membership_sha256": _sha256(MEMBERSHIP_FILE),
        "constituent_daily_sha256": _sha256(DAILY_FILE),
        "registry_sha256": _sha256(REGISTRY_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
