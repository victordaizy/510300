"""构建 VAL-04 PB—ROE 残余便宜度信号，不读取未来收益。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.normalized_valuation_5y_models_v1 import apply_frozen_percentile
from research.val01_raw_ey_5y_v1 import _next_trading_day_map
from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "val04_pb_roe_residual_v1.yaml"
FORBIDDEN_OUTPUT_TOKENS = (
    "future_return",
    "forward_return",
    "target_position",
    "target_share",
    "order_quantity",
    "predictive_ic",
)


@dataclass(frozen=True)
class SignalBuildResult:
    company_residuals: pd.DataFrame
    signal_inputs: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("VAL-04 模型配置不是映射")
    return config


def verify_input_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / str(contract["file"])
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "dataset": name,
                "file": str(contract["file"]),
                "expected_sha256": str(contract["sha256"]),
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "FAILED",
        "files": rows,
    }


def fit_snapshot_residuals(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按冻结公式拟合单月公司截面，返回公司残差和数值审计。"""

    required = {
        "date",
        "con_code",
        "snapshot_weight",
        "industry_l1",
        "log_price_to_book",
        "ttm_roe",
        "ttm_revenue_growth_yoy",
        "complete_case",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"VAL-04 公司面板缺少字段：{missing}")
    data = frame.loc[frame["complete_case"].fillna(False)].copy()
    for column in (
        "snapshot_weight",
        "log_price_to_book",
        "ttm_roe",
        "ttm_revenue_growth_yoy",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    data = data.dropna(
        subset=[
            "snapshot_weight",
            "industry_l1",
            "log_price_to_book",
            "ttm_roe",
            "ttm_revenue_growth_yoy",
        ]
    ).copy()
    if data.empty:
        raise ValueError("VAL-04 单月完整样本为空")
    sector = pd.get_dummies(data["industry_l1"].astype(str), dtype=float).sort_index(axis=1)
    data["roe_clipped"] = data["ttm_roe"].clip(-0.5, 0.8)
    data["revenue_growth_clipped"] = data["ttm_revenue_growth_yoy"].clip(-1.0, 1.0)
    numeric = data[["roe_clipped", "revenue_growth_clipped"]].astype(float)
    design = pd.concat([sector, numeric], axis=1)
    matrix = design.to_numpy(dtype=float)
    target = data["log_price_to_book"].to_numpy(dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        matrix, target, rcond=None
    )
    fitted = matrix @ coefficients
    residual = target - fitted
    data["fitted_log_price_to_book"] = fitted
    data["pb_roe_growth_residual"] = residual
    data["company_residual_cheapness"] = -residual
    total_weight = float(data["snapshot_weight"].sum())
    if total_weight <= 0:
        raise ValueError("VAL-04 完整样本权重和非正")
    weighted_cheapness = float(
        np.average(
            data["company_residual_cheapness"],
            weights=data["snapshot_weight"],
        )
    )
    target_mean = float(target.mean())
    total_sum_squares = float(np.square(target - target_mean).sum())
    residual_sum_squares = float(np.square(residual).sum())
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else math.nan
    )
    coefficient_map = dict(zip(design.columns, coefficients, strict=True))
    date = pd.Timestamp(data["date"].iloc[0])
    audit = {
        "date": date,
        "complete_company_count": int(len(data)),
        "complete_case_weight_coverage": total_weight,
        "active_industry_count": int(data["industry_l1"].nunique()),
        "design_column_count": int(matrix.shape[1]),
        "design_rank": int(rank),
        "design_residual_degrees_of_freedom": int(len(data) - rank),
        "minimum_singular_value": float(np.min(singular_values)),
        "residual_mean": float(residual.mean()),
        "residual_sum": float(residual.sum()),
        "r_squared": float(r_squared),
        "roe_coefficient": float(coefficient_map["roe_clipped"]),
        "revenue_growth_coefficient": float(
            coefficient_map["revenue_growth_clipped"]
        ),
        "weighted_pb_roe_residual_cheapness": weighted_cheapness,
    }
    output_columns = [
        "date",
        "con_code",
        "snapshot_weight",
        "industry_l1",
        "log_price_to_book",
        "roe_clipped",
        "revenue_growth_clipped",
        "fitted_log_price_to_book",
        "pb_roe_growth_residual",
        "company_residual_cheapness",
    ]
    return data[output_columns].copy(), audit


def build_signal_inputs(
    constituent_panel: pd.DataFrame,
    snapshot_audit: pd.DataFrame,
    index_daily: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拟合 120 个冻结横截面并生成不含收益的 60 月分位。"""

    audit = snapshot_audit.copy()
    audit["date"] = pd.to_datetime(audit["date"], errors="coerce").dt.normalize()
    if not audit["output"].eq("DATA_READY").all():
        failed = audit.loc[audit["output"].ne("DATA_READY"), "date"]
        raise RuntimeError(f"V2 数据闸门未全部通过：{failed.astype(str).tolist()}")
    panel = constituent_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    company_frames: list[pd.DataFrame] = []
    signal_rows: list[dict[str, Any]] = []
    for date, frame in panel.groupby("date", sort=True):
        companies, row = fit_snapshot_residuals(frame)
        company_frames.append(companies)
        signal_rows.append(row)
    company_residuals = pd.concat(company_frames, ignore_index=True)
    signals = pd.DataFrame(signal_rows).sort_values("date").reset_index(drop=True)
    expected = int(config["universe_and_time"]["expected_snapshot_count"])
    if len(signals) != expected:
        raise ValueError(f"VAL-04 信号快照预期{expected}，实际{len(signals)}")
    signals["input_status"] = "PASS"
    window = int(config["percentile_definition"]["window_months"])
    signals = apply_frozen_percentile(
        signals,
        "weighted_pb_roe_residual_cheapness",
        "input_status",
        "pb_roe_residual_percentile_60m",
        window,
    )
    next_days = _next_trading_day_map(signals["date"], index_daily)
    signals["signal_observation_date"] = signals["date"]
    signals["earliest_execution_date"] = signals["date"].map(next_days)
    signals["signal_available_after"] = "SNAPSHOT_DATE_CLOSE"
    signals["earliest_execution_time"] = "NEXT_INDEX_TRADING_DAY_OPEN"
    signals["historical_evidence_label"] = (
        "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS"
    )
    if not signals["earliest_execution_date"].gt(signals["date"]).all():
        raise ValueError("VAL-04 存在不晚于观察日的最早执行日")
    leaked = [
        column
        for column in [*signals.columns, *company_residuals.columns]
        if any(token in column.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if leaked:
        raise ValueError(f"VAL-04 信号产物包含禁止字段：{leaked}")
    return company_residuals, signals


def build_report(
    company_residuals: pd.DataFrame,
    signals: pd.DataFrame,
    hash_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    percentile = config["percentile_definition"]
    ready_column = "pb_roe_residual_percentile_60m_ready"
    ready = signals.loc[signals[ready_column]].copy()
    first = ready.iloc[0] if not ready.empty else None
    checks = {
        "input_hashes_pass": hash_audit["status"] == "PASS",
        "snapshot_count": len(signals)
        == int(config["universe_and_time"]["expected_snapshot_count"]),
        "company_snapshot_count": company_residuals["date"].nunique() == len(signals),
        "minimum_complete_case_coverage": float(
            signals["complete_case_weight_coverage"].min()
        )
        >= float(config["index_signal"]["minimum_complete_case_weight_coverage"]),
        "full_rank_every_snapshot": signals["design_rank"].eq(
            signals["design_column_count"]
        ).all(),
        "residual_orthogonality_numeric_tolerance": signals["residual_mean"]
        .abs()
        .max()
        <= 1e-10,
        "ready_count": len(ready) == int(percentile["expected_ready_observation_count"]),
        "first_ready_date": first is not None
        and pd.Timestamp(first["date"])
        == pd.Timestamp(percentile["expected_first_ready_observation_date"]),
        "first_execution_date": first is not None
        and pd.Timestamp(first["earliest_execution_date"])
        == pd.Timestamp(percentile["expected_first_earliest_execution_date"]),
    }
    passed = bool(all(checks.values()))
    status = (
        config["next_stage_gate"]["pass_status"]
        if passed
        else config["next_stage_gate"]["fail_status"]
    )
    return {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "status": status,
        "hash_audit": hash_audit,
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": [name for name, value in checks.items() if not value],
        "signal_summary": {
            "snapshot_count": int(len(signals)),
            "company_residual_row_count": int(len(company_residuals)),
            "first_snapshot": str(signals["date"].min().date()),
            "last_snapshot": str(signals["date"].max().date()),
            "minimum_complete_case_weight_coverage": float(
                signals["complete_case_weight_coverage"].min()
            ),
            "minimum_complete_company_count": int(
                signals["complete_company_count"].min()
            ),
            "minimum_design_residual_degrees_of_freedom": int(
                signals["design_residual_degrees_of_freedom"].min()
            ),
            "maximum_absolute_residual_mean": float(
                signals["residual_mean"].abs().max()
            ),
            "raw_signal_minimum": float(
                signals["weighted_pb_roe_residual_cheapness"].min()
            ),
            "raw_signal_median": float(
                signals["weighted_pb_roe_residual_cheapness"].median()
            ),
            "raw_signal_maximum": float(
                signals["weighted_pb_roe_residual_cheapness"].max()
            ),
            "ready_observation_count": int(len(ready)),
            "first_ready_observation_date": (
                str(ready["date"].min().date()) if len(ready) else None
            ),
            "first_earliest_execution_date": (
                str(ready["earliest_execution_date"].min().date())
                if len(ready)
                else None
            ),
        },
        "governance": {
            "future_returns_read": False,
            "predictive_ic_calculated": False,
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
            "parent_data_feasibility_files_mutated": False,
        },
        "meaning": "信号输入通过不代表预测有效，不是仓位或买卖指令。",
    }


def build_val04_signal(
    constituent_panel: pd.DataFrame,
    snapshot_audit: pd.DataFrame,
    index_daily: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> SignalBuildResult:
    config = config or load_config()
    hash_audit = verify_input_hashes(config)
    if hash_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 模型输入哈希漂移")
    company, signals = build_signal_inputs(
        constituent_panel, snapshot_audit, index_daily, config
    )
    report = build_report(company, signals, hash_audit, config)
    return SignalBuildResult(company, signals, report)


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["signal_summary"]
    return "\n".join(
        [
            "# VAL-04 PB—ROE 残余便宜度信号输入报告 V1",
            "",
            f"> 状态：`{report['status']}`。本报告没有读取未来收益或生成仓位。",
            "",
            "## 信号构建",
            "",
            f"- 快照：{summary['snapshot_count']}；公司残差行：{summary['company_residual_row_count']}。",
            f"- 日期：{summary['first_snapshot']} 至 {summary['last_snapshot']}。",
            f"- 最低完整样本权重覆盖：{summary['minimum_complete_case_weight_coverage']:.4%}。",
            f"- 最少完整公司：{summary['minimum_complete_company_count']}。",
            f"- 最小残差自由度：{summary['minimum_design_residual_degrees_of_freedom']}。",
            f"- 最大残差均值绝对值：{summary['maximum_absolute_residual_mean']:.3e}。",
            "",
            "## 60月分位",
            "",
            f"- 可用观测：{summary['ready_observation_count']}。",
            f"- 首个观察日：{summary['first_ready_observation_date']}。",
            f"- 首个最早执行日：{summary['first_earliest_execution_date']}。",
            "",
            "## 治理边界",
            "",
            "- 未读取未来收益，未计算 IC。",
            "- 未计算策略收益，未生成仓位、股数或订单。",
            "- 通过只允许另行冻结预测协议。",
        ]
    ) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_artifacts(
    result: SignalBuildResult, config: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    company_path = ROOT / artifacts["company_residuals"]
    signal_path = ROOT / artifacts["signal_inputs"]
    json_path = ROOT / artifacts["signal_report_json"]
    markdown_path = ROOT / artifacts["signal_report_markdown"]
    for path in (company_path, signal_path, json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.company_residuals.to_parquet(company_path, index=False)
    result.signal_inputs.to_parquet(signal_path, index=False)
    _atomic_text(
        json_path, json.dumps(result.report, ensure_ascii=False, indent=2) + "\n"
    )
    _atomic_text(markdown_path, render_markdown(result.report))
    inventory: dict[str, Any] = {}
    for name, path in (
        ("company_residuals", company_path),
        ("signal_inputs", signal_path),
        ("signal_report_json", json_path),
        ("signal_report_markdown", markdown_path),
    ):
        inventory[name] = {
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": result.report["status"],
        "generated_at": result.report["generated_at"],
        "registered_trial_id": config["protocol"]["registered_trial_id"],
        "artifacts": inventory,
        "input_hash_audit": result.report["hash_audit"],
        "signal_summary": result.report["signal_summary"],
        "governance": result.report["governance"],
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    manifest_path = ROOT / artifacts["model_manifest"]
    _atomic_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "inventory": inventory,
        "model_manifest": {
            "file": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_content_sha256"],
        },
    }


__all__ = [
    "SignalBuildResult",
    "build_report",
    "build_signal_inputs",
    "build_val04_signal",
    "fit_snapshot_residuals",
    "load_config",
    "render_markdown",
    "verify_input_hashes",
    "write_artifacts",
]
