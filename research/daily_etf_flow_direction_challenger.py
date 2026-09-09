"""检验510300日频申赎与融资行为能否稳定预测未来20日方向。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.primary_flow_direction_challenger import walk_forward_flow_direction
from research.short_horizon_direction_volatility import evaluate_direction


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "daily_etf_flow_direction_challenger.yaml"
REGISTERED_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
FUND_SHARE_FILE = ROOT / "data" / "raw" / "flow" / "510300_fund_share_daily_tushare.parquet"
MARGIN_FILE = ROOT / "data" / "raw" / "flow" / "510300_margin_detail_daily_tushare.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_daily_etf_flow_direction_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_daily_etf_flow_direction_report.json"

DAILY_FLOW_FEATURES = [
    "fund_share_change_5d_pct",
    "financing_net_buy_5d_to_turnover",
    "financing_balance_change_5d_pct",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_daily_flow_features(
    registered: pd.DataFrame,
    fund_share: pd.DataFrame,
    margin: pd.DataFrame,
) -> pd.DataFrame:
    """从真实披露日向后合并份额，并按交易日构造5日流量强度。"""

    base = registered.copy()
    shares = fund_share[["date", "fund_shares"]].copy().rename(columns={"date": "share_snapshot_date"})
    financing = margin[
        ["date", "rzye", "financing_net_buy_cny"]
    ].copy()
    base["date"] = pd.to_datetime(base["date"]).astype("datetime64[ns]")
    shares["share_snapshot_date"] = pd.to_datetime(shares["share_snapshot_date"]).astype("datetime64[ns]")
    financing["date"] = pd.to_datetime(financing["date"]).astype("datetime64[ns]")
    data = pd.merge_asof(
        base.sort_values("date"),
        shares.sort_values("share_snapshot_date"),
        left_on="date",
        right_on="share_snapshot_date",
        direction="backward",
        allow_exact_matches=True,
    )
    data = data.merge(financing, on="date", how="left", validate="one_to_one")
    data["share_snapshot_age_calendar_days"] = (
        data["date"] - data["share_snapshot_date"]
    ).dt.days
    data["fund_share_change_5d_pct"] = data["fund_shares"].pct_change(5, fill_method=None)
    data["financing_net_buy_5d_to_turnover"] = (
        data["financing_net_buy_cny"].rolling(5, min_periods=5).sum()
        / data["etf_amount"].rolling(5, min_periods=5).sum()
    )
    data["financing_balance_change_5d_pct"] = data["rzye"].pct_change(5, fill_method=None)
    data["direction_label"] = data["exec_total_return_20d_net"].gt(0).where(
        data["exec_total_return_20d_net"].notna()
    )
    return data.reset_index(drop=True)


def walk_forward_daily_flow(
    data: pd.DataFrame,
    horizon: int = 20,
    minimum_training_samples: int = 504,
    refit_interval: int = 5,
) -> pd.DataFrame:
    """复用严格标签可得性实现，并显式传入本轮预注册日频特征。"""

    return walk_forward_flow_direction(
        data,
        horizon=horizon,
        minimum_training_samples=minimum_training_samples,
        refit_interval=refit_interval,
        feature_columns=DAILY_FLOW_FEATURES,
    )


def main() -> int:
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    if registry["status"] != "PRE_REGISTERED_BEFORE_OUTCOME_TEST":
        raise ValueError("日频资金流候选必须先登记再检验")
    data = prepare_daily_flow_features(
        pd.read_parquet(REGISTERED_FILE),
        pd.read_parquet(FUND_SHARE_FILE),
        pd.read_parquet(MARGIN_FILE),
    )
    forecasts = walk_forward_daily_flow(data)
    if forecasts.empty:
        raise ValueError("没有生成日频申赎/融资方向预测")
    evaluation = evaluate_direction(forecasts)
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    midpoint = len(realized) // 2
    halves = {
        "first_half": evaluate_direction(realized.iloc[:midpoint].copy()),
        "second_half": evaluate_direction(realized.iloc[midpoint:].copy()),
    }
    coefficient_signs = {
        feature: {
            "expected_sign": "positive",
            "positive_share": float(
                (forecasts[f"standardized_coefficient_{feature}"] > 0).mean()
            ),
            "latest_standardized_coefficient": float(
                forecasts.iloc[-1][f"standardized_coefficient_{feature}"]
            ),
        }
        for feature in DAILY_FLOW_FEATURES
    }
    gate = registry["acceptance_gate"]
    checks = {
        "minimum_realized_forecasts": evaluation["realized_forecast_count"] >= gate["minimum_realized_forecasts"],
        "overall_auc": evaluation["auc"] >= gate["minimum_overall_auc"],
        "overall_brier_skill": evaluation["brier_skill_score"] > gate["minimum_overall_brier_skill"],
        "each_half_auc": all(item["auc"] >= gate["minimum_each_half_auc"] for item in halves.values()),
        "each_half_brier_skill": all(
            item["brier_skill_score"] > gate["minimum_each_half_brier_skill"] for item in halves.values()
        ),
        "calibration_slope": gate["calibration_slope_low"]
        <= evaluation["calibration_slope"]
        <= gate["calibration_slope_high"],
        "expected_coefficient_signs": all(
            item["positive_share"] >= gate["minimum_expected_sign_share_each_feature"]
            for item in coefficient_signs.values()
        ),
        "non_overlapping_auc": evaluation["non_overlapping_cohort_auc_above_half_share"]
        >= gate["minimum_non_overlapping_auc_above_half_share"],
        "top_minus_bottom_return": evaluation["top_minus_bottom_mean_net_return"] > 0,
    }
    all_passed = bool(all(checks.values()))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_FILE, index=False)
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    latest = forecasts.iloc[-1]
    report = {
        "status": "DAILY_ETF_FLOW_DIRECTION_ALL_GATES_PASSED" if all_passed else "DAILY_ETF_FLOW_DIRECTION_REJECTED",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "registry_status_before_test": registry["status"],
        "research_family_sequence": registry["research_family_sequence"],
        "features": registry["features"],
        "model": registry["model"],
        "acceptance_gate": gate,
        "gate_checks": checks,
        "evaluation": evaluation,
        "chronological_halves": halves,
        "coefficient_signs": coefficient_signs,
        "latest_forecast": {
            "signal_date": str(pd.Timestamp(latest["signal_date"]).date()),
            "direction_probability_positive_20d_net": float(latest["direction_probability_positive_20d_net"]),
        },
        "trading_use_authorized": False,
        "excluded_data": registry["excluded_data"],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
        "registry_sha256": _sha256(REGISTRY_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
