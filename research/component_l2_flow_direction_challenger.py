"""检验官方权重聚合成分股L2资金流是否提供稳定20日方向信息。"""

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
REGISTRY_FILE = ROOT / "config" / "component_l2_flow_direction_challenger.yaml"
REGISTERED_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
FLOW_FILE = ROOT / "data" / "features" / "000300_official_weighted_moneyflow_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_component_l2_flow_direction_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_component_l2_flow_direction_report.json"

L2_FEATURES = [
    "official_weighted_net_mf_intensity_5d",
    "official_weighted_large_extra_large_intensity_5d",
    "official_weighted_positive_net_mf_share_5d",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_l2_model_data(registered: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    base = registered.copy()
    component_flow = flow[["date", *L2_FEATURES]].copy()
    base["date"] = pd.to_datetime(base["date"])
    component_flow["date"] = pd.to_datetime(component_flow["date"])
    data = base.merge(component_flow, on="date", how="inner", validate="one_to_one")
    data["direction_label"] = data["exec_total_return_20d_net"].gt(0).where(
        data["exec_total_return_20d_net"].notna()
    )
    data["share_snapshot_date"] = data["date"]
    data["share_snapshot_age_calendar_days"] = 0
    return data.sort_values("date").reset_index(drop=True)


def main() -> int:
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    if registry["status"] != "PRE_REGISTERED_BEFORE_OUTCOME_TEST":
        raise ValueError("L2资金流候选必须先登记再检验")
    data = prepare_l2_model_data(
        pd.read_parquet(REGISTERED_FILE), pd.read_parquet(FLOW_FILE)
    )
    forecasts = walk_forward_flow_direction(data, feature_columns=L2_FEATURES)
    if forecasts.empty:
        raise ValueError("没有生成成分股L2资金流方向预测")
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
        for feature in L2_FEATURES
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
        "status": "COMPONENT_L2_FLOW_DIRECTION_ALL_GATES_PASSED" if all_passed else "COMPONENT_L2_FLOW_DIRECTION_REJECTED",
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
        "multiple_research_families_warning": "这是本轮新增数据中的第3个方向研究族，必须结合研究族数量解释结果。",
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
