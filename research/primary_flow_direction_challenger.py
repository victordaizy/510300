"""检验ETF申赎、溢折价和IF期现基差能否提供独立20日方向信息。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.short_horizon_direction_volatility import _direction_model, evaluate_direction


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "primary_flow_direction_challenger.yaml"
REGISTERED_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
FUTURES_FILE = ROOT / "data" / "raw" / "futures" / "IF0_daily_raw.parquet"
NAV_FILE = ROOT / "data" / "raw" / "fund" / "510300_nav_daily_raw.parquet"
SHARES_FILE = ROOT / "data" / "raw" / "fund" / "510300_monthly_shares_sse.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_primary_flow_direction_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_primary_flow_direction_report.json"

FLOW_FEATURES = [
    "if_basis_median_5d",
    "etf_premium_bps_median_5d",
    "latest_monthly_share_change_pct",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_flow_features(
    registered: pd.DataFrame,
    futures: pd.DataFrame,
    nav: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """所有资金流特征只从其真实日期向后使用。"""

    base = registered.copy()
    future = futures[["date", "close"]].rename(columns={"close": "if_close"}).copy()
    fund_nav = nav[["date", "close_premium_bps"]].copy()
    fund_shares = shares[["date", "share_change_pct"]].rename(
        columns={"date": "share_snapshot_date", "share_change_pct": "latest_monthly_share_change_pct"}
    ).copy()
    for frame, column in (
        (base, "date"),
        (future, "date"),
        (fund_nav, "date"),
        (fund_shares, "share_snapshot_date"),
    ):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    data = base.merge(future, on="date", how="left", validate="one_to_one")
    data = data.merge(fund_nav, on="date", how="left", validate="one_to_one")
    data = pd.merge_asof(
        data.sort_values("date"),
        fund_shares.sort_values("share_snapshot_date"),
        left_on="date",
        right_on="share_snapshot_date",
        direction="backward",
        allow_exact_matches=True,
    )
    data["share_snapshot_age_calendar_days"] = (
        data["date"] - data["share_snapshot_date"]
    ).dt.days
    data["if_basis_close"] = data["if_close"] / data["index_close"] - 1.0
    data["if_basis_median_5d"] = data["if_basis_close"].rolling(5, min_periods=3).median()
    data["etf_premium_bps_median_5d"] = data["close_premium_bps"].rolling(5, min_periods=3).median()
    data["latest_monthly_share_change_pct"] = data["latest_monthly_share_change_pct"].clip(-0.3, 0.3)
    data["direction_label"] = data["exec_total_return_20d_net"].gt(0).where(
        data["exec_total_return_20d_net"].notna()
    )
    return data.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def walk_forward_flow_direction(
    data: pd.DataFrame,
    horizon: int = 20,
    minimum_training_samples: int = 504,
    refit_interval: int = 5,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = data.sort_values("date").reset_index(drop=True)
    features = FLOW_FEATURES if feature_columns is None else feature_columns
    model = None
    fit_index: int | None = None
    base_rate = np.nan
    training_count = 0
    latest_training_label_end = pd.NaT
    standardized_coefficients: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_index in range(len(frame)):
        last_train_signal = signal_index - horizon
        if last_train_signal < 0:
            continue
        train_mask = frame.index.to_series().le(last_train_signal)
        train_mask &= frame[features].notna().all(axis=1) & frame["direction_label"].notna()
        train_indices = frame.index[train_mask]
        if len(train_indices) < minimum_training_samples:
            continue
        if frame.loc[[signal_index], features].isna().any(axis=None):
            continue
        if fit_index is None or signal_index - fit_index >= refit_interval:
            target = frame.loc[train_indices, "direction_label"].astype(int)
            if target.nunique() < 2:
                continue
            model = _direction_model().fit(frame.loc[train_indices, features], target)
            fit_index = signal_index
            base_rate = float(target.mean())
            training_count = int(len(train_indices))
            latest_training_label_end = pd.to_datetime(
                frame.loc[train_indices, f"label_end_date_{horizon}d"]
            ).max()
            coefficient_values = model.named_steps["logistic"].coef_[0]
            standardized_coefficients = {
                feature: float(value)
                for feature, value in zip(features, coefficient_values, strict=True)
            }
        if model is None or fit_index is None:
            continue
        probability = float(model.predict_proba(frame.loc[[signal_index], features])[0, 1])
        current = frame.loc[signal_index]
        rows.append(
            {
                "signal_date": pd.Timestamp(current["date"]),
                "model_fit_date": pd.Timestamp(frame.loc[fit_index, "date"]),
                "latest_training_label_end_date": latest_training_label_end,
                "training_sample_count": training_count,
                "direction_probability_positive_20d_net": probability,
                "expanding_base_rate_probability": base_rate,
                "actual_direction_positive_20d_net": current["direction_label"],
                "actual_total_return_20d_net": current["exec_total_return_20d_net"],
                "label_end_date": current[f"label_end_date_{horizon}d"],
                "is_realized": bool(pd.notna(current["direction_label"])),
                "share_snapshot_date": current["share_snapshot_date"],
                "share_snapshot_age_calendar_days": current["share_snapshot_age_calendar_days"],
                **{
                    f"standardized_coefficient_{feature}": value
                    for feature, value in standardized_coefficients.items()
                },
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def main() -> int:
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    if registry["status"] != "PRE_REGISTERED_CHALLENGER_BEFORE_OUTCOME_TEST":
        raise ValueError("资金流挑战模型必须在结果检验前登记")
    data = prepare_flow_features(
        pd.read_parquet(REGISTERED_FILE),
        pd.read_parquet(FUTURES_FILE),
        pd.read_parquet(NAV_FILE),
        pd.read_parquet(SHARES_FILE),
    )
    forecasts = walk_forward_flow_direction(data)
    if forecasts.empty:
        raise ValueError("没有生成资金流方向预测")
    evaluation = evaluate_direction(forecasts)
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    midpoint = len(realized) // 2
    chronological_halves = {
        "first_half": evaluate_direction(realized.iloc[:midpoint].copy()),
        "second_half": evaluate_direction(realized.iloc[midpoint:].copy()),
    }
    coefficient_sign_diagnostics = {
        feature: {
            "expected_sign": next(
                item["expected_sign"] for item in registry["features"] if item["column"] == feature
            ),
            "positive_share": float(
                (forecasts[f"standardized_coefficient_{feature}"] > 0).mean()
            ),
            "latest_standardized_coefficient": float(
                forecasts.iloc[-1][f"standardized_coefficient_{feature}"]
            ),
        }
        for feature in FLOW_FEATURES
    }
    mechanism_sign_gate_passed = bool(
        all(item["positive_share"] >= 0.60 for item in coefficient_sign_diagnostics.values())
    )
    temporal_brier_stability_passed = bool(
        all(
            item["brier_skill_score"] > 0
            for item in chronological_halves.values()
        )
    )
    probability_calibration_gate_passed = bool(0.5 <= evaluation["calibration_slope"] <= 1.5)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_FILE, index=False)
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    latest = forecasts.iloc[-1]
    report = {
        "status": (
            "PRIMARY_FLOW_DIRECTION_ALL_GATES_PASSED"
            if evaluation["direction_gate_passed"]
            and mechanism_sign_gate_passed
            and temporal_brier_stability_passed
            and probability_calibration_gate_passed
            else "PRIMARY_FLOW_SCREENING_METRICS_PASSED_GOVERNANCE_GATES_FAILED"
            if evaluation["direction_gate_passed"]
            else "PRIMARY_FLOW_DIRECTION_CHALLENGER_REJECTED"
        ),
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "registry_status_before_test": registry["status"],
        "features": registry["features"],
        "model": registry["model"],
        "acceptance_gate": registry["acceptance_gate"],
        "evaluation": evaluation,
        "post_screening_diagnostics": {
            "chronological_halves": chronological_halves,
            "coefficient_signs": coefficient_sign_diagnostics,
            "governance_gates": {
                "mechanism_sign_gate_passed": mechanism_sign_gate_passed,
                "temporal_brier_stability_passed": temporal_brier_stability_passed,
                "probability_calibration_gate_passed": probability_calibration_gate_passed,
                "all_post_screening_governance_gates_passed": bool(
                    mechanism_sign_gate_passed
                    and temporal_brier_stability_passed
                    and probability_calibration_gate_passed
                ),
                "note": "这些是首轮筛选后的通用治理检查，不追溯冒充预注册绩效门槛。",
            },
            "calibration_warning": "整体校准斜率显著低于1，原始概率过度自信；完成先前pseudo-OOS校准前不得映射仓位。",
        },
        "latest_forecast": {
            "signal_date": str(pd.Timestamp(latest["signal_date"]).date()),
            "direction_probability_positive_20d_net": float(latest["direction_probability_positive_20d_net"]),
            "share_snapshot_date": str(pd.Timestamp(latest["share_snapshot_date"]).date()),
            "share_snapshot_age_calendar_days": int(latest["share_snapshot_age_calendar_days"]),
        },
        "trading_use_authorized": False,
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
