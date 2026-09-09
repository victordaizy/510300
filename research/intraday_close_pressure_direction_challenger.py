"""按预登记方案检验510300收盘压力对随后两日净收益方向的预测力。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "intraday_close_pressure_direction_challenger.yaml"
FEATURE_FILE = ROOT / "data" / "features" / "510300_intraday_close_pressure_daily.parquet"
FORECAST_FILE = ROOT / "data" / "features" / "510300_intraday_close_pressure_2d_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_intraday_close_pressure_2d_report.json"
FEATURE_COLUMN = "close_pressure_30m_60d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commission(notional: float, rate: float, minimum: float) -> float:
    return max(minimum, notional * rate)


def _net_roundtrip_return(
    entry: float,
    exit_price: float,
    dividend_per_share: float,
    initial_cash: float,
    lot_size: int,
    commission_rate: float,
    minimum_commission: float,
    slippage_bps: float,
    stamp_duty_rate: float,
) -> float:
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0 or exit_price <= 0:
        return np.nan
    slippage = slippage_bps / 10_000.0
    buy_price = entry * (1.0 + slippage)
    sell_price = exit_price * (1.0 - slippage)
    lots = int(initial_cash // (buy_price * lot_size))
    while lots > 0:
        shares = lots * lot_size
        notional = shares * buy_price
        if notional + _commission(notional, commission_rate, minimum_commission) <= initial_cash + 1e-9:
            break
        lots -= 1
    if lots <= 0:
        return np.nan
    shares = lots * lot_size
    buy_notional = shares * buy_price
    cash = initial_cash - buy_notional - _commission(buy_notional, commission_rate, minimum_commission)
    sell_notional = shares * sell_price
    ending_cash = (
        cash
        + sell_notional
        - _commission(sell_notional, commission_rate, minimum_commission)
        - sell_notional * stamp_duty_rate
        + shares * dividend_per_share
    )
    return float(ending_cash / initial_cash - 1.0)


def attach_two_day_outcomes(
    features: pd.DataFrame,
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    target_config: dict,
) -> pd.DataFrame:
    """信号t收盘，t+1开盘买入，t+2收盘卖出并计入持有期分红。"""

    feature_data = features.copy()
    price_data = daily.copy()
    feature_data["date"] = pd.to_datetime(feature_data["date"]).dt.normalize()
    price_data["date"] = pd.to_datetime(price_data["date"]).dt.normalize()
    price_data = price_data.sort_values("date").reset_index(drop=True)
    data = feature_data.merge(
        price_data[["date", "open", "close"]],
        on="date",
        how="inner",
        validate="one_to_one",
    ).sort_values("date").reset_index(drop=True)
    data["entry_date"] = data["date"].shift(-1)
    data["exit_date"] = data["date"].shift(-2)
    data["entry_open"] = data["open"].shift(-1)
    data["exit_close"] = data["close"].shift(-2)

    dividend_data = dividends.copy()
    dividend_data["record_date"] = pd.to_datetime(dividend_data["record_date"]).dt.normalize()
    dividend_by_record_date = dividend_data.groupby("record_date")["cash_dividend_per_share"].sum()
    dividend_lookup = dividend_by_record_date.to_dict()
    data["holding_dividend_per_share"] = [
        float(
            sum(
                amount
                for record_date, amount in dividend_lookup.items()
                if pd.notna(entry_date)
                and pd.notna(exit_date)
                and entry_date <= record_date <= exit_date
            )
        )
        for entry_date, exit_date in zip(data["entry_date"], data["exit_date"], strict=True)
    ]
    data["exec_total_return_2d_gross"] = (
        (data["exit_close"] + data["holding_dividend_per_share"]) / data["entry_open"] - 1.0
    )
    data["exec_total_return_2d_net"] = [
        _net_roundtrip_return(
            float(entry),
            float(exit_price),
            float(dividend),
            float(target_config["initial_cash_cny"]),
            int(target_config["lot_size"]),
            float(target_config["commission_rate"]),
            float(target_config["minimum_commission_cny"]),
            float(target_config["slippage_bps_per_leg"]),
            float(target_config["stamp_duty_rate"]),
        )
        if pd.notna(entry) and pd.notna(exit_price)
        else np.nan
        for entry, exit_price, dividend in zip(
            data["entry_open"],
            data["exit_close"],
            data["holding_dividend_per_share"],
            strict=True,
        )
    ]
    data["direction_label"] = data["exec_total_return_2d_net"].gt(0).where(
        data["exec_total_return_2d_net"].notna()
    )
    data["label_end_date"] = data["exit_date"]
    return data.replace([np.inf, -np.inf], np.nan)


def _model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=c_value, solver="lbfgs", max_iter=1000)),
        ]
    )


def walk_forward_forecasts(
    data: pd.DataFrame,
    minimum_training_samples: int = 504,
    refit_interval: int = 5,
    c_value: float = 0.1,
) -> pd.DataFrame:
    """扩展窗口预测；只有标签结束日不晚于当前信号日的样本可训练。"""

    frame = data.sort_values("date").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []
    fitted: Pipeline | None = None
    last_fit_index: int | None = None
    baseline_probability = np.nan
    training_count = 0
    latest_label_end = pd.NaT
    coefficient = np.nan
    for signal_index, current in frame.iterrows():
        signal_date = pd.Timestamp(current["date"])
        train_mask = (
            frame[FEATURE_COLUMN].notna()
            & frame["direction_label"].notna()
            & frame["label_end_date"].notna()
            & frame["label_end_date"].le(signal_date)
            & frame.index.to_series().lt(signal_index)
        )
        train_indices = frame.index[train_mask]
        if len(train_indices) < minimum_training_samples or pd.isna(current[FEATURE_COLUMN]):
            continue
        should_refit = last_fit_index is None or signal_index - last_fit_index >= refit_interval
        if should_refit:
            outcome = frame.loc[train_indices, "direction_label"].astype(int)
            if outcome.nunique() < 2:
                continue
            fitted = _model(c_value).fit(frame.loc[train_indices, [FEATURE_COLUMN]], outcome)
            baseline_probability = float(outcome.mean())
            training_count = int(len(train_indices))
            latest_label_end = pd.to_datetime(frame.loc[train_indices, "label_end_date"]).max()
            coefficient = float(fitted.named_steps["logistic"].coef_[0, 0])
            last_fit_index = signal_index
        if fitted is None or last_fit_index is None:
            continue
        probability = float(fitted.predict_proba(frame.loc[[signal_index], [FEATURE_COLUMN]])[0, 1])
        rows.append(
            {
                "signal_date": signal_date,
                "model_fit_date": pd.Timestamp(frame.loc[last_fit_index, "date"]),
                "latest_training_label_end_date": latest_label_end,
                "training_sample_count": training_count,
                "standardized_feature_coefficient": coefficient,
                "probability_positive_2d_net": probability,
                "expanding_base_rate_probability": baseline_probability,
                "actual_direction_positive_2d_net": current["direction_label"],
                "actual_total_return_2d_net": current["exec_total_return_2d_net"],
                "label_end_date": current["label_end_date"],
                "is_realized": bool(pd.notna(current["direction_label"])),
            }
        )
    return pd.DataFrame(rows)


def _calibration_slope(probability: pd.Series, outcome: pd.Series) -> float:
    clipped = probability.clip(1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).to_numpy().reshape(-1, 1)
    fitted = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, outcome.astype(int))
    return float(fitted.coef_[0, 0])


def _metric_block(data: pd.DataFrame) -> dict[str, float | int | None]:
    if len(data) < 10 or data["actual_direction_positive_2d_net"].nunique() < 2:
        return {"observations": int(len(data)), "auc": None, "brier_skill_score": None}
    outcome = data["actual_direction_positive_2d_net"].astype(int)
    probability = data["probability_positive_2d_net"]
    baseline = data["expanding_base_rate_probability"]
    model_brier = float(brier_score_loss(outcome, probability))
    baseline_brier = float(brier_score_loss(outcome, baseline))
    return {
        "observations": int(len(data)),
        "auc": float(roc_auc_score(outcome, probability)),
        "brier_score": model_brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_score": float(1.0 - model_brier / baseline_brier),
    }


def evaluate_forecasts(forecasts: pd.DataFrame, gates: dict) -> dict[str, object]:
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    overall = _metric_block(realized)
    if overall["auc"] is None:
        raise ValueError("已实现预测不足以计算方向指标")
    outcome = realized["actual_direction_positive_2d_net"].astype(int)
    probability = realized["probability_positive_2d_net"]
    calibration_slope = _calibration_slope(probability, outcome)
    quintile_labels = pd.qcut(probability, q=5, labels=False, duplicates="drop")
    quintiles = (
        realized.assign(probability_quintile=quintile_labels)
        .groupby("probability_quintile", observed=True)
        .agg(
            observations=("actual_direction_positive_2d_net", "size"),
            mean_probability=("probability_positive_2d_net", "mean"),
            realized_hit_rate=("actual_direction_positive_2d_net", "mean"),
            mean_net_return=("actual_total_return_2d_net", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    top_bottom = float(quintiles[-1]["mean_net_return"] - quintiles[0]["mean_net_return"])
    midpoint = len(realized) // 2
    halves = [_metric_block(realized.iloc[:midpoint]), _metric_block(realized.iloc[midpoint:])]
    cohorts = [_metric_block(realized.iloc[offset::2]) for offset in range(2)]
    cohort_aucs = [float(item["auc"]) for item in cohorts if item["auc"] is not None]
    coefficients = realized["standardized_feature_coefficient"].dropna()
    coefficient_median = float(coefficients.median())
    coefficient_positive_share = float((coefficients > 0).mean())

    checks = {
        "minimum_realized_forecasts": len(realized) >= int(gates["minimum_realized_forecasts"]),
        "overall_auc": float(overall["auc"]) >= float(gates["primary_metrics"]["auc_minimum"]),
        "overall_brier_skill_positive": float(overall["brier_skill_score"]) > 0.0,
        "calibration_slope": float(gates["primary_metrics"]["calibration_slope_minimum"]) <= calibration_slope <= float(gates["primary_metrics"]["calibration_slope_maximum"]),
        "top_minus_bottom_return_positive": top_bottom > 0.0,
        "coefficient_expected_sign": coefficient_median > 0.0 and coefficient_positive_share >= 0.60,
        "both_halves_auc": all(item["auc"] is not None and float(item["auc"]) >= float(gates["temporal_stability"]["chronological_half_auc_minimum"]) for item in halves),
        "both_halves_brier_skill_positive": all(item["brier_skill_score"] is not None and float(item["brier_skill_score"]) > 0.0 for item in halves),
        "non_overlapping_auc_median": len(cohort_aucs) == 2 and float(np.median(cohort_aucs)) >= float(gates["overlap_diagnostic"]["median_auc_minimum"]),
        "non_overlapping_auc_above_half_share": len(cohort_aucs) == 2 and float(np.mean(np.asarray(cohort_aucs) > 0.5)) >= float(gates["overlap_diagnostic"]["auc_above_half_share_minimum"]),
    }
    return {
        "realized_forecast_count": int(len(realized)),
        "positive_base_rate": float(outcome.mean()),
        **overall,
        "log_loss": float(log_loss(outcome, probability)),
        "accuracy_at_0_5": float(((probability >= 0.5).astype(int) == outcome).mean()),
        "calibration_slope": calibration_slope,
        "top_minus_bottom_mean_net_return": top_bottom,
        "standardized_coefficient_median": coefficient_median,
        "standardized_coefficient_positive_share": coefficient_positive_share,
        "chronological_halves": halves,
        "non_overlapping_offset_cohorts": cohorts,
        "probability_quintiles": quintiles,
        "checks": checks,
        "all_gates_passed": bool(all(checks.values())),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    feature_status_file = ROOT / "reports" / "data_quality" / "510300_intraday_close_pressure_daily_status.json"
    feature_status = json.loads(feature_status_file.read_text(encoding="utf-8"))
    if feature_status.get("status") != "PASS" or feature_status.get("contains_outcomes") is not False:
        raise RuntimeError("收盘压力特征质量门未通过")
    features = pd.read_parquet(FEATURE_FILE)
    daily = pd.read_parquet(ROOT / config["data"]["daily_file"])
    dividends = pd.read_csv(ROOT / config["data"]["dividend_file"])
    model_data = attach_two_day_outcomes(features, daily, dividends, config["target"])
    forecasts = walk_forward_forecasts(
        model_data,
        minimum_training_samples=int(config["model"]["minimum_training_samples"]),
        refit_interval=int(config["model"]["refit_interval_trading_days"]),
        c_value=float(config["model"]["logistic_c"]),
    )
    if forecasts.empty:
        raise RuntimeError("没有生成收盘压力方向预测")
    evaluation = evaluate_forecasts(forecasts, config["evaluation"])
    FORECAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECAST_FILE, index=False, engine="pyarrow")
    realized = forecasts.loc[forecasts["is_realized"]]
    if (realized["latest_training_label_end_date"] > realized["signal_date"]).any():
        raise RuntimeError("发现训练标签结束日晚于预测信号日的前视污染")
    report = {
        "status": "ALL_GATES_PASSED" if evaluation["all_gates_passed"] else "REJECTED_GOVERNANCE_GATES_FAILED",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "challenger_id": config["research"]["challenger_id"],
        "pre_registration_file": CONFIG_FILE.relative_to(ROOT).as_posix(),
        "feature_file": FEATURE_FILE.relative_to(ROOT).as_posix(),
        "forecast_file": FORECAST_FILE.relative_to(ROOT).as_posix(),
        "forecast_sha256": _sha256(FORECAST_FILE),
        "design": {
            "signal": "t日14:31至15:00收益乘以该窗口成交额占比相对过去60日中位数",
            "target": "t+1开盘买入、t+2收盘卖出，含持有期分红并扣佣金和双边5bp滑点后的净收益方向",
            "t_plus_one_compliant": True,
            "estimator": "单特征、固定C=0.1的标准化L2 Logistic",
            "minimum_training_samples": int(config["model"]["minimum_training_samples"]),
            "refit_interval_trading_days": int(config["model"]["refit_interval_trading_days"]),
        },
        "evaluation": evaluation,
        "latest_research_forecast": {
            "signal_date": str(pd.Timestamp(forecasts.iloc[-1]["signal_date"]).date()),
            "probability_positive_2d_net": float(forecasts.iloc[-1]["probability_positive_2d_net"]),
            "is_realized": bool(forecasts.iloc[-1]["is_realized"]),
        },
        "trading_use_authorized": False,
        "limitations": [
            "全部五年历史已经用于研究，只能称为伪样本外，不能替代未来前向验证。",
            "这是项目中的新增研究家族，整体多重检验风险继续上升。",
            "即使方向门通过，也必须先完成0至100%仓位映射、连续组合回测与成本压力测试。",
            "分钟数据来自第三方代理，且提供方未明确trade_time属于bar起点还是终点；本研究只在15:00全部记录完成后生成信号。",
        ],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
