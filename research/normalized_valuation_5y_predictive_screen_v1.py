"""评估 VAL01/VAL02 五年正常化估值预测屏幕，不生成策略仓位。"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.normalized_valuation_5y_models_v1 import (
    canonical_hash,
    sha256_file,
)
from research.val01_raw_ey_5y_v1_forward_evaluation import hac_rank_slope


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "normalized_valuation_5y_predictive_screen_v1.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("预测屏幕配置不是映射")
    return config


def verify_input_hashes(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "dataset": name,
                "file": contract["file"],
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "FAILED",
        "files": rows,
    }


def frozen_bucket(percentile: float) -> str:
    if percentile < 0.20:
        return "B1_EXPENSIVE"
    if percentile < 0.40:
        return "B2"
    if percentile < 0.60:
        return "B3"
    return "B4_CHEAP"


def safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def spearman(frame: pd.DataFrame, predictor: str, outcome: str) -> float | None:
    valid = frame[[predictor, outcome]].dropna()
    if len(valid) < 2:
        return None
    return safe_number(valid.corr(method="spearman").iloc[0, 1])


def audit_frozen_labels(labels: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    contract = config["label_contract"]
    required = {
        "signal_observation_date",
        "horizon_trading_days",
        "label_status",
        "etf_total_return",
        "h00300_total_return",
    }
    if missing := required - set(labels.columns):
        raise ValueError(f"冻结标签表缺少字段：{sorted(missing)}")
    forbidden = [
        column
        for column in labels.columns
        if "percentile" in column.lower()
        or "earnings_yield" in column.lower()
        or "ey_spread" in column.lower()
    ]
    data = labels.copy()
    data["signal_observation_date"] = pd.to_datetime(
        data["signal_observation_date"], errors="coerce"
    ).dt.normalize()
    maturity: dict[str, dict[str, int]] = {}
    counts_match = True
    for horizon in contract["horizons_trading_days"]:
        horizon_int = int(horizon)
        subset = data.loc[data["horizon_trading_days"].eq(horizon_int)]
        matured = int(subset["label_status"].eq("MATURED").sum())
        censored = int(
            subset["label_status"].eq("RIGHT_CENSORED_DATA_CUTOFF").sum()
        )
        maturity[str(horizon_int)] = {
            "matured": matured,
            "right_censored": censored,
        }
        counts_match &= matured == int(contract["expected_matured_counts"][horizon_int])
        counts_match &= censored == int(
            contract["expected_right_censored_counts"][horizon_int]
        )
    status = bool(
        len(data) == int(contract["expected_label_rows"])
        and data["signal_observation_date"].nunique()
        == int(contract["expected_signal_dates"])
        and not forbidden
        and counts_match
    )
    return {
        "status": "PASS" if status else "BLOCKED",
        "row_count": int(len(data)),
        "signal_date_count": int(data["signal_observation_date"].nunique()),
        "forbidden_signal_columns": forbidden,
        "maturity": maturity,
        "counts_match_freeze": counts_match,
    }


def compute_target_diagnostic(
    joined: pd.DataFrame,
    predictor: str,
    outcome: str,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = joined.loc[
        joined["horizon_trading_days"].eq(horizon)
        & joined["label_status"].eq("MATURED"),
        ["signal_observation_date", predictor, outcome],
    ].dropna().sort_values("signal_observation_date").reset_index(drop=True)
    frame["bucket"] = frame[predictor].map(frozen_bucket)
    bucket_order = ["B1_EXPENSIVE", "B2", "B3", "B4_CHEAP"]
    buckets: list[dict[str, Any]] = []
    for bucket in bucket_order:
        values = frame.loc[frame["bucket"].eq(bucket), outcome]
        buckets.append(
            {
                "bucket": bucket,
                "observations": int(len(values)),
                "mean_return": safe_number(values.mean()),
                "median_return": safe_number(values.median()),
                "positive_return_ratio": (
                    safe_number(values.gt(0).mean()) if len(values) else None
                ),
            }
        )
    means = [row["mean_return"] for row in buckets]
    monotonic = bool(
        all(value is not None for value in means)
        and all(float(means[index + 1]) >= float(means[index]) for index in range(3))
    )
    mean_spread = (
        float(means[-1]) - float(means[0])
        if means[0] is not None and means[-1] is not None
        else None
    )
    medians = [row["median_return"] for row in buckets]
    median_spread = (
        float(medians[-1]) - float(medians[0])
        if medians[0] is not None and medians[-1] is not None
        else None
    )
    midpoint = len(frame) // 2
    first = frame.iloc[:midpoint]
    second = frame.iloc[midpoint:]
    lags = int(config["diagnostic_definition"]["hac_monthly_max_lags"][horizon])
    return {
        "outcome": outcome,
        "horizon_trading_days": horizon,
        "matured_observations": int(len(frame)),
        "first_observation_date": (
            str(frame["signal_observation_date"].min().date()) if len(frame) else None
        ),
        "last_observation_date": (
            str(frame["signal_observation_date"].max().date()) if len(frame) else None
        ),
        "spearman_ic": spearman(frame, predictor, outcome),
        "hac_rank_regression": hac_rank_slope(frame[predictor], frame[outcome], lags),
        "chronological_halves": {
            "first": {
                "observations": int(len(first)),
                "spearman_ic": spearman(first, predictor, outcome),
            },
            "second": {
                "observations": int(len(second)),
                "spearman_ic": spearman(second, predictor, outcome),
            },
        },
        "buckets": buckets,
        "bucket_mean_returns_nondecreasing": monotonic,
        "cheapest_minus_expensive_mean_return": safe_number(mean_spread),
        "cheapest_minus_expensive_median_return": safe_number(median_spread),
    }


def compute_model_diagnostics(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    model_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    model = config["models"][model_id]
    predictor = model["predictor_column"]
    ready = model["ready_column"]
    signal_values = signals.loc[
        signals[ready].fillna(False), ["date", predictor]
    ].copy()
    signal_values["date"] = pd.to_datetime(
        signal_values["date"], errors="coerce"
    ).dt.normalize()
    signal_values = signal_values.rename(columns={"date": "signal_observation_date"})
    label_data = labels.copy()
    label_data["signal_observation_date"] = pd.to_datetime(
        label_data["signal_observation_date"], errors="coerce"
    ).dt.normalize()
    label_dates = set(label_data["signal_observation_date"].dropna())
    signal_dates = set(signal_values["signal_observation_date"].dropna())
    if signal_dates != label_dates:
        raise ValueError(
            f"{model_id}信号日与冻结标签日不一致："
            f"信号独有{sorted(signal_dates-label_dates)}，标签独有{sorted(label_dates-signal_dates)}"
        )
    joined = label_data.merge(
        signal_values, on="signal_observation_date", how="inner", validate="many_to_one"
    )
    diagnostics: dict[str, Any] = {}
    for horizon in config["label_contract"]["horizons_trading_days"]:
        horizon_int = int(horizon)
        diagnostics[str(horizon_int)] = {
            "etf_total_return": compute_target_diagnostic(
                joined, predictor, "etf_total_return", horizon_int, config
            ),
            "h00300_total_return": compute_target_diagnostic(
                joined, predictor, "h00300_total_return", horizon_int, config
            ),
        }
    return diagnostics


def holm_bonferroni(p_values: dict[str, float | None]) -> dict[str, Any]:
    """计算Holm调整p值和逐步拒绝结果；缺失p值按1处理。"""

    prepared = {
        model: (float(value) if value is not None and math.isfinite(float(value)) else 1.0)
        for model, value in p_values.items()
    }
    ordered = sorted(prepared.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    rows: list[dict[str, Any]] = []
    for index, (model, value) in enumerate(ordered):
        multiplier = m - index
        running = max(running, multiplier * value)
        adjusted_value = min(1.0, running)
        adjusted[model] = adjusted_value
        rows.append(
            {
                "rank": index + 1,
                "model_id": model,
                "raw_p_value": value,
                "holm_multiplier": multiplier,
                "holm_adjusted_p_value": adjusted_value,
            }
        )
    return {"ordered_tests": rows, "adjusted_p_values": adjusted}


def frozen_gate_checks(
    primary: dict[str, Any],
    confirmation: dict[str, Any],
    holm_adjusted_p: float,
    config: dict[str, Any],
) -> dict[str, bool]:
    gate = config["primary_predictive_gate"]
    bucket_counts = [row["observations"] for row in primary["buckets"]]
    first_ic = primary["chronological_halves"]["first"]["spearman_ic"]
    second_ic = primary["chronological_halves"]["second"]["spearman_ic"]
    return {
        "minimum_matured_observations": primary["matured_observations"]
        >= int(gate["minimum_matured_observations"]),
        "minimum_observations_per_bucket": min(bucket_counts)
        >= int(gate["minimum_observations_per_bucket"]),
        "etf_spearman_ic_strictly_positive": primary["spearman_ic"] is not None
        and primary["spearman_ic"] > 0,
        "holm_adjusted_hac_p_at_most_family_alpha": holm_adjusted_p
        <= float(gate["familywise_alpha"]),
        "both_chronological_half_ics_strictly_positive": first_ic is not None
        and second_ic is not None
        and first_ic > 0
        and second_ic > 0,
        "etf_bucket_mean_returns_nondecreasing": primary[
            "bucket_mean_returns_nondecreasing"
        ],
        "etf_cheapest_minus_expensive_mean_positive": primary[
            "cheapest_minus_expensive_mean_return"
        ]
        is not None
        and primary["cheapest_minus_expensive_mean_return"] > 0,
        "etf_cheapest_minus_expensive_median_positive": primary[
            "cheapest_minus_expensive_median_return"
        ]
        is not None
        and primary["cheapest_minus_expensive_median_return"] > 0,
        "h00300_spearman_confirmation_positive": confirmation["spearman_ic"]
        is not None
        and confirmation["spearman_ic"] > 0,
        "h00300_cheapest_minus_expensive_mean_positive": confirmation[
            "cheapest_minus_expensive_mean_return"
        ]
        is not None
        and confirmation["cheapest_minus_expensive_mean_return"] > 0,
    }


def build_family_reports(
    model_diagnostics: dict[str, dict[str, Any]],
    label_audit: dict[str, Any],
    hash_audit: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primary_horizon = str(config["label_contract"]["primary_horizon_trading_days"])
    p_values = {
        model_id: diagnostics[primary_horizon]["etf_total_return"]
        ["hac_rank_regression"]
        ["one_sided_p_value"]
        for model_id, diagnostics in model_diagnostics.items()
    }
    holm = holm_bonferroni(p_values)
    model_reports: dict[str, dict[str, Any]] = {}
    for model_id, diagnostics in model_diagnostics.items():
        primary = diagnostics[primary_horizon]["etf_total_return"]
        confirmation = diagnostics[primary_horizon]["h00300_total_return"]
        adjusted = holm["adjusted_p_values"][model_id]
        checks = frozen_gate_checks(primary, confirmation, adjusted, config)
        passed = bool(
            hash_audit["status"] == "PASS"
            and label_audit["status"] == "PASS"
            and all(checks.values())
        )
        status = (
            config["primary_predictive_gate"]["pass_status"]
            if passed
            else config["primary_predictive_gate"]["fail_status"]
        )
        failed = [name for name, value in checks.items() if not value]
        model_reports[model_id] = {
            "project_id": f"{model_id}_PREDICTIVE_SCREEN",
            "version": config["protocol"]["version"],
            "generated_at": datetime.now(
                ZoneInfo(config["protocol"]["timezone"])
            ).isoformat(),
            "status": status,
            "historical_evidence_label": (
                "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS"
            ),
            "hash_audit": hash_audit,
            "label_audit": label_audit,
            "diagnostics": diagnostics,
            "primary_predictive_gate": {
                "horizon_trading_days": int(primary_horizon),
                "target": "etf_total_return",
                "raw_hac_one_sided_p_value": p_values[model_id],
                "holm_adjusted_p_value": adjusted,
                "checks": checks,
                "passed": passed,
                "failed_checks": failed,
                "stop_reason": None if passed else "PRIMARY_PREDICTIVE_GATE_FAILED",
            },
            "governance": {
                "frozen_forward_labels_reused": True,
                "forward_labels_regenerated": False,
                "predictive_ic_calculated": True,
                "historical_strategy_return_calculated": False,
                "historical_position_mapping_performed": False,
                "current_position_mapping_performed": False,
                "target_shares_generated": False,
                "orders_generated": False,
                "broker_connection_performed": False,
                "alpha_pass": False,
            },
        }
    passed_models = [
        model for model, report in model_reports.items()
        if report["primary_predictive_gate"]["passed"]
    ]
    family_report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "status": (
            "ALL_REGISTERED_MODELS_PASS"
            if len(passed_models) == len(model_reports)
            else "PARTIAL_REGISTERED_MODELS_PASS"
            if passed_models
            else "ALL_REGISTERED_MODELS_REJECTED"
        ),
        "holm_bonferroni": {
            "family_alpha": config["primary_predictive_gate"]["familywise_alpha"],
            **holm,
        },
        "model_status": {
            model: report["status"] for model, report in model_reports.items()
        },
        "passed_models": passed_models,
        "rejected_models": [
            model for model in model_reports if model not in passed_models
        ],
        "strategy_evaluation_authorized_models": passed_models,
        "governance": {
            "model_results_reported_together": True,
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
        },
    }
    return model_reports, family_report


def render_model_report(report: dict[str, Any]) -> str:
    gate = report["primary_predictive_gate"]
    primary = report["diagnostics"][str(gate["horizon_trading_days"])][
        "etf_total_return"
    ]
    return "\n".join(
        [
            f"# {report['project_id']}",
            "",
            f"> 状态：`{report['status']}`。历史为事后重建，不是严格样本外。",
            "",
            f"- 242日成熟样本：{primary['matured_observations']}。",
            f"- Spearman IC：{primary['spearman_ic']}。",
            f"- HAC单侧原始p值：{gate['raw_hac_one_sided_p_value']}。",
            f"- Holm调整p值：{gate['holm_adjusted_p_value']}。",
            f"- 失败闸门：{gate['failed_checks']}。",
            "- 未计算策略净值、仓位、当前建议、份额或订单。",
            "",
        ]
    )


def render_family_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# VAL01/VAL02 正常化估值预测屏幕家族报告",
            "",
            f"> 状态：`{report['status']}`。",
            "",
            f"- 通过模型：{report['passed_models']}。",
            f"- 拒绝模型：{report['rejected_models']}。",
            f"- Holm检验：{report['holm_bonferroni']['ordered_tests']}。",
            "- 只有通过模型允许另行冻结策略评估协议；本报告本身未运行策略回测。",
            "",
        ]
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_reports_and_manifest(
    model_reports: dict[str, dict[str, Any]],
    family_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    report_paths = {
        "val01_report_json": ROOT / artifacts["val01_report_json"],
        "val01_report_markdown": ROOT / artifacts["val01_report_markdown"],
        "val02_report_json": ROOT / artifacts["val02_report_json"],
        "val02_report_markdown": ROOT / artifacts["val02_report_markdown"],
        "family_report_json": ROOT / artifacts["family_report_json"],
        "family_report_markdown": ROOT / artifacts["family_report_markdown"],
    }
    val01 = model_reports["VAL01_NORM_EY_5Y_V1"]
    val02 = model_reports["VAL02_NORM_EY_SPREAD_5Y_V1"]
    atomic_text(
        report_paths["val01_report_json"],
        json.dumps(val01, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_text(report_paths["val01_report_markdown"], render_model_report(val01))
    atomic_text(
        report_paths["val02_report_json"],
        json.dumps(val02, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_text(report_paths["val02_report_markdown"], render_model_report(val02))
    atomic_text(
        report_paths["family_report_json"],
        json.dumps(family_report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_text(
        report_paths["family_report_markdown"], render_family_report(family_report)
    )
    protocol_manifest = ROOT / artifacts["protocol_manifest"]
    protocol = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREDICTIVE_SCREEN_PROTOCOL":
        raise RuntimeError("预测屏幕协议未冻结")
    payload: dict[str, Any] = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_PREDICTIVE_SCREEN_RESULTS_NO_STRATEGY_BACKTEST",
        "frozen_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "protocol_manifest": {
            "file": protocol_manifest.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(protocol_manifest),
            "manifest_content_sha256": protocol["manifest_content_sha256"],
        },
        "input_files": {
            name: {
                "file": contract["file"],
                "sha256": contract["sha256"],
            }
            for name, contract in config["data_contracts"].items()
        },
        "output_files": {
            name: {
                "file": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in report_paths.items()
        },
        "family_status": family_report["status"],
        "model_status": family_report["model_status"],
        "strategy_evaluation_authorized_models": family_report[
            "strategy_evaluation_authorized_models"
        ],
        "governance": family_report["governance"],
        "meaning": "冻结预测关系筛选结果；未运行仓位策略、成本回测或当前信号映射。",
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    manifest = ROOT / artifacts["result_manifest"]
    atomic_text(manifest, json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "file": manifest.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(manifest),
        "manifest_content_sha256": payload["manifest_content_sha256"],
    }
