"""运行 IF + Breadth → 510300 分钟级信息传播扫描。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.information_propagation_alpha import (
    LeadLagParameters,
    build_information_propagation_features,
    evaluate_primary_test,
    summarize_outcome_deciles,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_alpha.yaml"


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.is_dir():
        dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
        return dataset.to_table(columns=columns).to_pandas()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
        return data if columns is None else data[columns]
    raise ValueError(f"不支持的数据格式：{path}")


def _file_readiness(path: Path, required_columns: list[str]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "exists": path.exists(),
        "required_columns": required_columns,
    }
    if not path.exists():
        item["status"] = "MISSING"
        return item
    try:
        if path.is_dir():
            dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
            columns = dataset.schema.names
            missing = sorted(set(required_columns).difference(columns))
            item.update(
                {
                    "status": "READY" if not missing else "SCHEMA_MISMATCH",
                    "row_count": int(dataset.count_rows()),
                    "columns": columns,
                    "missing_columns": missing,
                    "partition_file_count": len(dataset.files),
                }
            )
            return item
        data = _read_table(path)
    except Exception as exc:
        item.update({"status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"})
        return item
    missing = sorted(set(required_columns).difference(data.columns))
    item.update(
        {
            "status": "READY" if not missing else "SCHEMA_MISMATCH",
            "row_count": int(len(data)),
            "columns": data.columns.tolist(),
            "missing_columns": missing,
        }
    )
    if "trade_time" in data.columns and not data.empty:
        timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
        item["first_trade_time"] = None if timestamps.isna().all() else timestamps.min().isoformat()
        item["last_trade_time"] = None if timestamps.isna().all() else timestamps.max().isoformat()
        item["trading_day_count"] = int(timestamps.dt.normalize().nunique())
    return item


def load_config() -> dict[str, Any]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data_config = config["data"]
    schema = config["schema"]
    input_paths = {
        "etf": _resolve(data_config["etf_1m"]),
        "index": _resolve(data_config["index_1m"]),
        "futures": _resolve(data_config["if_contract_1m"]),
        "components": _resolve(data_config["component_1m"]),
        "weights": _resolve(data_config["historical_weights"]),
    }
    readiness = {
        name: _file_readiness(input_paths[name], list(schema[name]["required_columns"]))
        for name in input_paths
    }
    report: dict[str, Any] = {
        "experiment_id": config["experiment"]["id"],
        "experiment_status": config["experiment"]["status"],
        "inputs": readiness,
        "governance_note": config["research_split"]["governance_note"],
    }
    validation_path_text = data_config.get("if_validation_1m")
    if validation_path_text:
        validation_path = _resolve(validation_path_text)
        report["supplemental_validation_input"] = _file_readiness(
            validation_path,
            ["trade_time", "contract", "close", "volume", "is_continuous"],
        )
        report["supplemental_validation_input"]["research_eligible"] = False
    if any(item["status"] != "READY" for item in readiness.values()):
        report["status"] = "BLOCKED_MISSING_OR_INVALID_INPUT"
        report["blocking_inputs"] = [
            name for name, item in readiness.items() if item["status"] != "READY"
        ]
        return 2, report

    timestamp_audit_path = _resolve(data_config["timestamp_audit_report"])
    if not timestamp_audit_path.exists():
        report["status"] = "BLOCKED_TIMESTAMP_AUDIT_NOT_RUN"
        report["timestamp_audit_report"] = str(timestamp_audit_path)
        return 2, report
    try:
        timestamp_audit = json.loads(timestamp_audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["status"] = "BLOCKED_TIMESTAMP_AUDIT_UNREADABLE"
        report["timestamp_audit_error"] = f"{type(exc).__name__}: {exc}"
        return 2, report
    report["timestamp_audit_status"] = timestamp_audit.get("status")
    if timestamp_audit.get("status") != "PASS":
        report["status"] = "BLOCKED_TIMESTAMP_AUDIT_NOT_PASS"
        return 2, report

    parameters_config = config["parameters"]
    parameters = LeadLagParameters(
        lookback_minutes=int(parameters_config["lookback_minutes"]),
        breadth_impulse_minutes=int(parameters_config["breadth_impulse_minutes"]),
        horizons_minutes=tuple(int(value) for value in parameters_config["horizons_minutes"]),
        leader_count=int(parameters_config["leader_count"]),
        minimum_component_weight_coverage=float(parameters_config["minimum_component_weight_coverage"]),
        beta=float(parameters_config["beta"]),
        normalization_sessions=int(parameters_config["normalization_sessions"]),
        normalization_minimum_sessions=int(parameters_config["normalization_minimum_sessions"]),
        decile_count=int(parameters_config["decile_count"]),
        residual_estimation_sessions=int(parameters_config["residual_estimation_sessions"]),
        residual_minimum_sessions=int(parameters_config["residual_minimum_sessions"]),
        execution_delay_minutes=int(parameters_config["execution_delay_minutes"]),
    )
    features = build_information_propagation_features(
        _read_table(input_paths["etf"]),
        _read_table(input_paths["futures"]),
        _read_table(
            input_paths["components"],
            ["trade_time", "con_code", "close", "volume", "amount"],
        ),
        _read_table(input_paths["weights"]),
        parameters,
        index=_read_table(input_paths["index"]),
    )
    overlapping = features.dropna(subset=["if_lead", "leader_return"])
    overlapping_sessions = int(overlapping["session_date"].nunique())
    report["overlap"] = {
        "row_count": int(len(overlapping)),
        "trading_day_count": overlapping_sessions,
        "minimum_required_sessions": int(parameters_config["minimum_overlapping_sessions"]),
    }
    if overlapping_sessions < int(parameters_config["minimum_overlapping_sessions"]):
        report["status"] = "BLOCKED_INSUFFICIENT_OVERLAP"
        return 2, report

    split = config["research_split"]
    periods = {
        name: (str(bounds[0]), str(bounds[1]))
        for name, bounds in split.items()
        if name != "governance_note"
    }
    development_start, development_end = periods["development"]
    cost_config = config["execution_costs"]
    account = float(cost_config["account_cny"])
    commission_per_side = max(
        float(cost_config["commission_rate_per_side"]),
        float(cost_config["minimum_commission_cny_per_side"]) / account,
    )
    base_cost = (
        2 * commission_per_side
        + 2 * float(cost_config["base_slippage_bps_per_side"]) / 10_000
        + float(cost_config["base_round_trip_spread_bps"]) / 10_000
    )
    stress_cost = float(cost_config["stress_total_round_trip_cost_bps"]) / 10_000
    outcome_columns: list[str] = []
    for horizon in parameters.horizons_minutes:
        statistical = f"etf_markout_close_{horizon}m"
        execution_open = f"etf_execution_next_open_{horizon}m"
        execution_vwap = f"etf_execution_next_vwap_{horizon}m"
        base_net = f"{execution_open}_net_base"
        stress_net = f"{execution_open}_net_stress"
        features[base_net] = features[execution_open] - base_cost
        features[stress_net] = features[execution_open] - stress_cost
        outcome_columns.extend([statistical, execution_open, execution_vwap, base_net, stress_net])
    decile_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    for signal in config["signals"]:
        deciles, yearly = summarize_outcome_deciles(
            features,
            signal_column=str(signal),
            outcome_columns=outcome_columns,
            development_start=development_start,
            development_end=development_end,
            evaluation_periods=periods,
            quantile_count=parameters.decile_count,
        )
        deciles["signal"] = signal
        yearly["signal"] = signal
        decile_frames.append(deciles)
        yearly_frames.append(yearly)
    all_deciles = pd.concat(decile_frames, ignore_index=True)
    all_yearly = pd.concat(yearly_frames, ignore_index=True)

    primary_config = config["primary_test"]
    primary_horizon = int(primary_config["horizon_minutes"])
    primary_outcomes = [
        str(primary_config["statistical_outcome"]),
        str(primary_config["executable_outcome"]),
        f"{primary_config['executable_outcome']}_net_base",
        f"{primary_config['executable_outcome']}_net_stress",
    ]
    primary_result = evaluate_primary_test(
        features,
        signal_column=str(primary_config["signal"]),
        outcome_columns=primary_outcomes,
        development_start=development_start,
        development_end=development_end,
        evaluation_periods=periods,
        quantile_count=parameters.decile_count,
    )
    primary_result.update(
        {
            "target_asset": "510300.SH",
            "benchmark_index_role": "000300.SH仅作对照，不是收益或交易标的",
            "primary_horizon_minutes": primary_horizon,
            "base_round_trip_cost_bps": base_cost * 10_000,
            "stress_round_trip_cost_bps": stress_cost * 10_000,
        }
    )

    _atomic_parquet(features, _resolve(data_config["output_features"]))
    _atomic_csv(all_deciles, _resolve(data_config["output_deciles"]))
    _atomic_csv(all_yearly, _resolve(data_config["output_yearly_extremes"]))
    _atomic_json(primary_result, _resolve(data_config["output_primary_test"]))
    report.update(
        {
            "status": "PASS_RESEARCH_SCAN_COMPLETE",
            "feature_row_count": int(len(features)),
            "decile_row_count": int(len(all_deciles)),
            "yearly_extreme_row_count": int(len(all_yearly)),
            "target_asset": "510300.SH",
            "base_round_trip_cost_bps": base_cost * 10_000,
            "stress_round_trip_cost_bps": stress_cost * 10_000,
        }
    )
    return 0, report


def main() -> int:
    config = load_config()
    code, report = run(config)
    report_path = _resolve(config["data"]["readiness_report"])
    _atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
