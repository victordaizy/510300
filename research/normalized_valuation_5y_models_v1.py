"""构建 VAL01/VAL02 五年正常化估值信号，不读取未来收益。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.point_in_time_valuation_v2_post_acquisition import combined_price_input
from research.val01_norm_ey_5y_post_acquisition_audit import (
    build_normalized_metric_audit_panel,
    independent_valuation_cross_check,
)
from research.val01_raw_ey_5y_v1 import (
    _next_trading_day_map,
    empirical_midrank_percentile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "normalized_valuation_5y_models_v1.yaml"
FORBIDDEN_OUTPUT_TOKENS = (
    "future_return",
    "forward_return",
    "target_position",
    "target_share",
    "order_quantity",
    "predictive_ic",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("正常化估值模型配置不是映射")
    return config


def verify_input_hashes(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        if not path.is_file():
            rows.append(
                {
                    "dataset": name,
                    "file": contract["file"],
                    "expected_sha256": contract["sha256"],
                    "actual_sha256": None,
                    "matches": False,
                }
            )
            continue
        actual = sha256_file(path)
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


def apply_frozen_percentile(
    panel: pd.DataFrame,
    value_column: str,
    status_column: str,
    output_column: str,
    window_months: int,
) -> pd.DataFrame:
    """对通过输入闸门的连续月度值计算精确经验midrank。"""

    result = panel.copy()
    source = pd.to_numeric(result[value_column], errors="coerce").where(
        result[status_column].eq("PASS")
    )
    observation_counts: list[int] = []
    percentiles: list[float] = []
    for index in range(len(result)):
        start = max(0, index - window_months + 1)
        window = source.iloc[start : index + 1]
        observation_counts.append(int(window.notna().sum()))
        if len(window) == window_months and window.notna().all():
            percentiles.append(empirical_midrank_percentile(window))
        else:
            percentiles.append(np.nan)
    result[f"{output_column}_window_observations"] = observation_counts
    result[output_column] = percentiles
    result[f"{output_column}_ready"] = result[output_column].notna()
    return result


def build_common_panel(
    weights: pd.DataFrame,
    extended_financials: pd.DataFrame,
    snapshot_prices: pd.DataFrame,
    current_constituent_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    government_bond_yields: pd.DataFrame,
    coverage: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """构建120个月共同正常化估值面板，不读取未来收益。"""

    shared = config["shared_signal_definition"]
    start = pd.Timestamp(shared["snapshot_start"])
    end = pd.Timestamp(shared["snapshot_end"])
    prices = combined_price_input(snapshot_prices, current_constituent_daily)
    normalized = build_normalized_metric_audit_panel(
        weights,
        extended_financials,
        prices,
        index_daily,
        start,
        end,
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    coverage_data = coverage.copy()
    coverage_data["date"] = pd.to_datetime(
        coverage_data["date"], errors="coerce"
    ).dt.normalize()
    coverage_columns = [
        "date",
        "weight_sum",
        "price_weight_coverage",
        "ttm_metric_weight_coverage",
        "normalized_metric_weight_coverage",
        "cgb_10y_available_exact_date",
    ]
    panel = normalized.merge(
        coverage_data[coverage_columns], on="date", how="left", validate="one_to_one"
    )
    bonds = government_bond_yields[["date", "cgb_10y"]].copy()
    bonds["date"] = pd.to_datetime(bonds["date"], errors="coerce").dt.normalize()
    bonds["cgb_10y"] = pd.to_numeric(bonds["cgb_10y"], errors="coerce")
    bonds = bonds.drop_duplicates("date", keep="last")
    panel = panel.merge(bonds, on="date", how="left", validate="one_to_one")
    panel["cgb_10y_decimal"] = panel["cgb_10y"] / 100.0
    panel["normalized_ey_spread"] = (
        panel["weighted_normalized_earnings_yield"] - panel["cgb_10y_decimal"]
    )
    minimum_price = float(shared["minimum_price_weight_coverage"])
    minimum_normalized = float(
        shared["minimum_normalized_earnings_weight_coverage"]
    )
    common_pass = (
        panel["price_weight_coverage"].ge(minimum_price)
        & panel["normalized_earnings_weight_coverage"].ge(minimum_normalized)
        & panel["normalized_metric_weight_coverage"].ge(minimum_normalized)
    )
    panel["val01_input_status"] = np.where(common_pass, "PASS", "BLOCKED_COVERAGE")
    val02_pass = (
        common_pass
        & panel["cgb_10y"].notna()
        & panel["cgb_10y_available_exact_date"].fillna(False).astype(bool)
    )
    panel["val02_input_status"] = np.where(
        val02_pass, "PASS", "BLOCKED_COVERAGE_OR_EXACT_DATE_BOND"
    )
    expected_count = int(shared["expected_snapshot_count"])
    if len(panel) != expected_count:
        raise ValueError(f"共同面板快照数预期{expected_count}，实际{len(panel)}")
    if not panel["date"].is_monotonic_increasing or panel["date"].duplicated().any():
        raise ValueError("共同面板日期不唯一或未递增")
    if not panel["constituent_count"].eq(
        int(shared["expected_constituents_per_snapshot"])
    ).all():
        raise ValueError("共同面板存在非300只成分快照")
    next_days = _next_trading_day_map(panel["date"], index_daily)
    panel["signal_observation_date"] = panel["date"]
    panel["earliest_execution_date"] = panel["date"].map(next_days)
    panel["signal_available_after"] = "SNAPSHOT_DATE_CLOSE"
    panel["earliest_execution_time"] = "NEXT_INDEX_TRADING_DAY_OPEN"
    window = int(config["percentile_definition"]["window_months"])
    panel = apply_frozen_percentile(
        panel,
        "weighted_normalized_earnings_yield",
        "val01_input_status",
        "norm_ey_percentile_60m",
        window,
    )
    panel = apply_frozen_percentile(
        panel,
        "normalized_ey_spread",
        "val02_input_status",
        "norm_ey_spread_percentile_60m",
        window,
    )
    panel["historical_evidence_label"] = (
        "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS"
    )
    if not panel["earliest_execution_date"].gt(panel["date"]).all():
        raise ValueError("存在不晚于信号观察日的执行日")
    leaked = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if leaked:
        raise ValueError(f"共同面板包含禁止字段：{leaked}")
    return panel.sort_values("date").reset_index(drop=True)


def project_model_signals(
    common_panel: pd.DataFrame, model_id: str, config: dict[str, Any]
) -> pd.DataFrame:
    """物理隔离两个模型的正式信号输入列。"""

    shared_columns = [
        "date",
        "signal_observation_date",
        "earliest_execution_date",
        "signal_available_after",
        "earliest_execution_time",
        "constituent_count",
        "weight_sum",
        "price_weight_coverage",
        "normalized_earnings_weight_coverage",
        "normalized_metric_weight_coverage",
        "normalization_company_count",
        "known_financial_event_count",
        "historical_evidence_label",
    ]
    if model_id == "VAL01_NORM_EY_5Y_V1":
        model_columns = [
            "weighted_normalized_earnings_yield",
            "val01_input_status",
            "norm_ey_percentile_60m_window_observations",
            "norm_ey_percentile_60m",
            "norm_ey_percentile_60m_ready",
        ]
    elif model_id == "VAL02_NORM_EY_SPREAD_5Y_V1":
        model_columns = [
            "weighted_normalized_earnings_yield",
            "cgb_10y",
            "cgb_10y_decimal",
            "cgb_10y_available_exact_date",
            "normalized_ey_spread",
            "val02_input_status",
            "norm_ey_spread_percentile_60m_window_observations",
            "norm_ey_spread_percentile_60m",
            "norm_ey_spread_percentile_60m_ready",
        ]
    else:
        raise KeyError(f"未知模型：{model_id}")
    result = common_panel[shared_columns + model_columns].copy()
    result["model_id"] = model_id
    return result


def build_model_report(
    signals: pd.DataFrame,
    model_id: str,
    config: dict[str, Any],
    hash_audit: dict[str, Any],
    official_pe: pd.DataFrame,
    vendor_pe: pd.DataFrame,
) -> dict[str, Any]:
    model = config["models"][model_id]
    percentile_column = model["predictor_percentile_column"]
    ready_column = f"{percentile_column}_ready"
    status_column = (
        "val01_input_status"
        if model_id == "VAL01_NORM_EY_5Y_V1"
        else "val02_input_status"
    )
    ready = signals.loc[signals[ready_column]].copy()
    percentile = config["percentile_definition"]
    expected_first = pd.Timestamp(percentile["expected_first_ready_observation_date"])
    expected_execution = pd.Timestamp(
        percentile["expected_first_earliest_execution_date"]
    )
    first = ready.iloc[0] if not ready.empty else None
    structure_pass = bool(
        len(signals) == int(config["shared_signal_definition"]["expected_snapshot_count"])
        and signals["constituent_count"].eq(300).all()
        and signals[status_column].eq("PASS").all()
        and len(ready) == int(percentile["expected_ready_observation_count"])
        and first is not None
        and pd.Timestamp(first["date"]) == expected_first
        and pd.Timestamp(first["earliest_execution_date"]) == expected_execution
    )
    warmup_end = expected_first
    normalized_for_cross_check = signals.loc[
        signals["date"].le(warmup_end),
        ["date", "weighted_normalized_earnings_yield"],
    ].copy()
    normalized_for_cross_check["component_normalized_pe"] = np.where(
        normalized_for_cross_check["weighted_normalized_earnings_yield"].gt(0),
        1.0 / normalized_for_cross_check["weighted_normalized_earnings_yield"],
        np.nan,
    )
    cross_check = independent_valuation_cross_check(
        normalized_for_cross_check, official_pe, vendor_pe
    )
    status = (
        "PASS_SIGNAL_INPUTS_NO_RETURN_LABELS"
        if hash_audit["status"] == "PASS" and structure_pass
        else "BLOCKED_SIGNAL_INPUT_GATE"
    )
    return {
        "project_id": model_id,
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "status": status,
        "registered_trial_id": model["registered_trial_id"],
        "hash_audit": hash_audit,
        "structure_gate_passed": structure_pass,
        "signal_summary": {
            "row_count": int(len(signals)),
            "first_observation_date": str(signals["date"].min().date()),
            "last_observation_date": str(signals["date"].max().date()),
            "input_pass_month_count": int(signals[status_column].eq("PASS").sum()),
            "percentile_ready_month_count": int(len(ready)),
            "first_percentile_ready_date": (
                str(ready["date"].min().date()) if not ready.empty else None
            ),
            "first_earliest_execution_date": (
                str(ready["earliest_execution_date"].min().date())
                if not ready.empty
                else None
            ),
            "minimum_price_weight_coverage": float(
                signals["price_weight_coverage"].min()
            ),
            "minimum_normalized_earnings_weight_coverage": float(
                signals["normalized_earnings_weight_coverage"].min()
            ),
            "minimum_percentile": float(ready[percentile_column].min()),
            "maximum_percentile": float(ready[percentile_column].max()),
            "exact_date_bond_month_count": (
                int(signals["cgb_10y_available_exact_date"].sum())
                if "cgb_10y_available_exact_date" in signals
                else None
            ),
        },
        "independent_cross_check": cross_check,
        "cross_check_role": "DIAGNOSTIC_ONLY_NOT_SIGNAL_INPUT_NOT_DECISION_GATE",
        "governance": {
            "future_return_labels_generated": False,
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "raw_ey_rejected_branch_reopened": False,
        },
    }


def render_report(report: dict[str, Any]) -> str:
    summary = report["signal_summary"]
    return "\n".join(
        [
            f"# {report['project_id']} 信号输入闸门",
            "",
            f"> 状态：`{report['status']}`。本报告不包含未来收益、IC、仓位或订单。",
            "",
            f"- 月度快照：{summary['row_count']}。",
            f"- 分位可用月份：{summary['percentile_ready_month_count']}。",
            f"- 首个分位观察日：{summary['first_percentile_ready_date']}。",
            f"- 最早执行日：{summary['first_earliest_execution_date']}。",
            f"- 最低价格权重覆盖：{summary['minimum_price_weight_coverage']:.4%}。",
            f"- 最低正常化盈利权重覆盖：{summary['minimum_normalized_earnings_weight_coverage']:.4%}。",
            "- 独立PE交叉核验只作尺度诊断，不是信号输入或收益放行门槛。",
            "- 历史为事后重建，不是严格样本外。",
            "",
        ]
    )


def _atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_signal_artifacts(
    common_panel: pd.DataFrame,
    val01_signals: pd.DataFrame,
    val02_signals: pd.DataFrame,
    val01_report: dict[str, Any],
    val02_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    paths = {
        "common_panel": ROOT / artifacts["common_panel"],
        "val01_signal_inputs": ROOT / artifacts["val01_signal_inputs"],
        "val02_signal_inputs": ROOT / artifacts["val02_signal_inputs"],
        "val01_report_json": ROOT / artifacts["val01_report_json"],
        "val01_report_markdown": ROOT / artifacts["val01_report_markdown"],
        "val02_report_json": ROOT / artifacts["val02_report_json"],
        "val02_report_markdown": ROOT / artifacts["val02_report_markdown"],
    }
    _atomic_parquet(common_panel, paths["common_panel"])
    _atomic_parquet(val01_signals, paths["val01_signal_inputs"])
    _atomic_parquet(val02_signals, paths["val02_signal_inputs"])
    val01_report["artifacts"] = {
        "signal_inputs": artifacts["val01_signal_inputs"],
        "signal_inputs_sha256": sha256_file(paths["val01_signal_inputs"]),
        "common_panel": artifacts["common_panel"],
        "common_panel_sha256": sha256_file(paths["common_panel"]),
    }
    val02_report["artifacts"] = {
        "signal_inputs": artifacts["val02_signal_inputs"],
        "signal_inputs_sha256": sha256_file(paths["val02_signal_inputs"]),
        "common_panel": artifacts["common_panel"],
        "common_panel_sha256": sha256_file(paths["common_panel"]),
    }
    _atomic_text(
        json.dumps(val01_report, ensure_ascii=False, indent=2, allow_nan=False),
        paths["val01_report_json"],
    )
    _atomic_text(render_report(val01_report), paths["val01_report_markdown"])
    _atomic_text(
        json.dumps(val02_report, ensure_ascii=False, indent=2, allow_nan=False),
        paths["val02_report_json"],
    )
    _atomic_text(render_report(val02_report), paths["val02_report_markdown"])
    return {
        name: {
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def write_model_manifests(
    artifact_inventory: dict[str, Any],
    val01_report: dict[str, Any],
    val02_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    protocol_manifest = ROOT / config["artifacts"]["protocol_manifest"]
    protocol = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_SIGNAL_PROTOCOL_NO_RETURN_EVALUATION":
        raise RuntimeError("正常化估值信号协议尚未冻结")
    protocol_sha = sha256_file(protocol_manifest)
    outputs: dict[str, dict[str, Any]] = {}
    definitions = (
        (
            "VAL01_NORM_EY_5Y_V1",
            val01_report,
            "val01_manifest",
            ("common_panel", "val01_signal_inputs", "val01_report_json", "val01_report_markdown"),
        ),
        (
            "VAL02_NORM_EY_SPREAD_5Y_V1",
            val02_report,
            "val02_manifest",
            ("common_panel", "val02_signal_inputs", "val02_report_json", "val02_report_markdown"),
        ),
    )
    for model_id, report, manifest_key, artifact_keys in definitions:
        if report["status"] != "PASS_SIGNAL_INPUTS_NO_RETURN_LABELS":
            raise RuntimeError(f"{model_id}信号输入闸门未通过")
        payload: dict[str, Any] = {
            "project_id": model_id,
            "version": config["protocol"]["version"],
            "status": "FROZEN_SIGNAL_INPUTS_NO_RETURN_EVALUATION",
            "frozen_at": datetime.now(
                ZoneInfo(config["protocol"]["timezone"])
            ).isoformat(),
            "registered_trial_id": config["models"][model_id]["registered_trial_id"],
            "protocol_manifest": {
                "file": protocol_manifest.relative_to(ROOT).as_posix(),
                "sha256": protocol_sha,
                "manifest_content_sha256": protocol["manifest_content_sha256"],
            },
            "artifacts": {key: artifact_inventory[key] for key in artifact_keys},
            "signal_summary": report["signal_summary"],
            "governance": report["governance"],
            "meaning": "只冻结无收益信号输入；不表示预测有效，不生成仓位或订单。",
        }
        payload["manifest_content_sha256"] = canonical_hash(payload)
        manifest_path = ROOT / config["artifacts"][manifest_key]
        _atomic_text(
            json.dumps(payload, ensure_ascii=False, indent=2), manifest_path
        )
        outputs[model_id] = {
            "file": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_content_sha256": payload["manifest_content_sha256"],
        }
    return outputs
