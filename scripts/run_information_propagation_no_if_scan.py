"""运行000300 + Top50Breadth → 510300的冻结无IF分钟扫描。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.information_propagation_alpha import (
    LeadLagParameters,
    build_no_if_information_propagation_features,
    evaluate_primary_test,
    summarize_outcome_deciles,
)
from scripts.run_information_propagation_scan import (
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _file_readiness,
    _read_table,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_no_if.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data_config = config["data"]
    input_paths = {
        "etf": _resolve(data_config["etf_1m"]),
        "index": _resolve(data_config["index_1m"]),
        "breadth": _resolve(data_config["top50_breadth_1m"]),
    }
    required_columns = {
        "etf": ["trade_time", "open", "close", "vol", "amount"],
        "index": ["trade_time", "close"],
        "breadth": [
            "trade_time", "leader_return", "top50_breadth",
            f"top50_breadth_impulse_{config['parameters']['breadth_impulse_minutes']}m",
        ],
    }
    readiness = {
        name: _file_readiness(path, required_columns[name])
        for name, path in input_paths.items()
    }
    report: dict[str, Any] = {
        "experiment_id": config["experiment"]["id"],
        "experiment_status": config["experiment"]["status"],
        "target_asset": "510300.SH",
        "index_role": "仅作领先信息对照，不是收益或交易标的",
        "inputs": readiness,
        "governance_note": config["research_split"]["governance_note"],
    }
    if any(item["status"] != "READY" for item in readiness.values()):
        report["status"] = "BLOCKED_MISSING_OR_INVALID_INPUT"
        report["blocking_inputs"] = [
            name for name, item in readiness.items() if item["status"] != "READY"
        ]
        return 2, report
    gates = {
        "timestamp_audit": _resolve(data_config["timestamp_audit_report"]),
        "breadth_build": _resolve(data_config["breadth_build_report"]),
    }
    for gate_name, gate_path in gates.items():
        if not gate_path.exists():
            report["status"] = f"BLOCKED_{gate_name.upper()}_MISSING"
            return 2, report
        gate = _read_json(gate_path)
        report[f"{gate_name}_status"] = gate.get("status")
        if gate.get("status") != "PASS":
            report["status"] = f"BLOCKED_{gate_name.upper()}_NOT_PASS"
            return 2, report

    raw_parameters = config["parameters"]
    parameters = LeadLagParameters(
        lookback_minutes=int(raw_parameters["lookback_minutes"]),
        breadth_impulse_minutes=int(raw_parameters["breadth_impulse_minutes"]),
        horizons_minutes=tuple(int(value) for value in raw_parameters["horizons_minutes"]),
        leader_count=int(raw_parameters["leader_count"]),
        minimum_component_weight_coverage=float(
            raw_parameters["minimum_component_weight_coverage"]
        ),
        beta=float(raw_parameters["beta"]),
        normalization_sessions=int(raw_parameters["normalization_sessions"]),
        normalization_minimum_sessions=int(raw_parameters["normalization_minimum_sessions"]),
        decile_count=int(raw_parameters["decile_count"]),
        execution_delay_minutes=int(raw_parameters["execution_delay_minutes"]),
    )
    features = build_no_if_information_propagation_features(
        etf=_read_table(input_paths["etf"]),
        index=_read_table(input_paths["index"]),
        breadth=_read_table(input_paths["breadth"]),
        parameters=parameters,
    )
    primary_signal = str(config["primary_test"]["signal"])
    overlap = features.dropna(subset=[primary_signal])
    overlapping_sessions = int(overlap["session_date"].nunique())
    report["overlap"] = {
        "row_count": int(len(overlap)),
        "trading_day_count": overlapping_sessions,
        "minimum_required_sessions": int(raw_parameters["minimum_overlapping_sessions"]),
    }
    if overlapping_sessions < int(raw_parameters["minimum_overlapping_sessions"]):
        report["status"] = "BLOCKED_INSUFFICIENT_OVERLAP"
        return 2, report

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
    outcomes: list[str] = []
    for horizon in parameters.horizons_minutes:
        statistical = f"etf_markout_close_{horizon}m"
        executable = f"etf_execution_next_open_{horizon}m"
        executable_vwap = f"etf_execution_next_vwap_{horizon}m"
        net_base = f"{executable}_net_base"
        net_stress = f"{executable}_net_stress"
        features[net_base] = features[executable] - base_cost
        features[net_stress] = features[executable] - stress_cost
        outcomes.extend([statistical, executable, executable_vwap, net_base, net_stress])

    split = config["research_split"]
    periods = {
        name: (str(bounds[0]), str(bounds[1]))
        for name, bounds in split.items() if name != "governance_note"
    }
    development_start, development_end = periods["development"]
    decile_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    for signal in config["signals"]:
        deciles, yearly = summarize_outcome_deciles(
            features=features,
            signal_column=str(signal),
            outcome_columns=outcomes,
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
    executable = str(primary_config["executable_outcome"])
    primary_result = evaluate_primary_test(
        features=features,
        signal_column=primary_signal,
        outcome_columns=[
            str(primary_config["statistical_outcome"]),
            executable,
            f"{executable}_net_base",
            f"{executable}_net_stress",
        ],
        development_start=development_start,
        development_end=development_end,
        evaluation_periods=periods,
        quantile_count=parameters.decile_count,
    )
    primary_result.update(
        {
            "experiment_id": config["experiment"]["id"],
            "target_asset": "510300.SH",
            "base_round_trip_cost_bps": base_cost * 10_000,
            "stress_round_trip_cost_bps": stress_cost * 10_000,
        }
    )
    base_net_outcome = f"{executable}_net_base"
    statistical_outcome = str(primary_config["statistical_outcome"])
    period_metrics = primary_result["periods"]
    statistical_relationship_detected = all(
        period_metrics[period][statistical_outcome].get("p10_minus_p1_bps", 0) > 0
        and period_metrics[period][statistical_outcome].get(
            "bucket_monotonicity_spearman", 0
        ) > 0.7
        for period in periods
    )
    base_cost_gate_passed = all(
        period_metrics[period][base_net_outcome].get("p10_mean_bps", float("-inf")) > 0
        for period in periods
    )
    primary_result["decision"] = {
        "statistical_relationship_detected": statistical_relationship_detected,
        "base_cost_gate_passed": base_cost_gate_passed,
        "verdict": "PASS_FOR_FURTHER_TRADING_RESEARCH"
        if statistical_relationship_detected and base_cost_gate_passed
        else "REJECTED_AS_STANDALONE_TRADING_ALPHA_COST_GATE",
        "interpretation": (
            "价格发现排序关系与可交易Alpha分开判断；统一扣减成本不改变P10-P1，"
            "成本门槛使用P10绝对成本后均值。"
        ),
    }
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
            "base_round_trip_cost_bps": base_cost * 10_000,
            "stress_round_trip_cost_bps": stress_cost * 10_000,
        }
    )
    return 0, report


def main() -> int:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    code, report = run(config)
    _atomic_json(report, _resolve(config["data"]["readiness_report"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
