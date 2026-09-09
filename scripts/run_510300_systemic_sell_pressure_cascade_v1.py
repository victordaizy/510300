"""运行510300全A股系统性卖压扩散次日退出V1。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_feasibility_v1 import (  # noqa: E402
    load_and_audit_inputs as load_v1_inputs,
    load_config as load_v1_config,
    validate_manifest as validate_v1_manifest,
)
from research.systemic_sell_pressure_cascade_v1 import (  # noqa: E402
    CONFIG_PATH,
    build_report,
    calibrate_and_apply_signal,
    evaluate_partition,
    load_and_build_daily_features,
    load_config,
    load_master,
    render_markdown,
    replication_gate_checks,
    validate_manifest,
    visible_gate_checks,
)


def main() -> int:
    config = load_config(CONFIG_PATH)
    artifacts = config["artifacts"]
    protected = [
        ROOT / artifacts["report_json"],
        ROOT / artifacts["report_markdown"],
        ROOT / artifacts["output_directory"],
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"冻结候选产物已存在，禁止覆盖：{existing}")

    manifest = validate_manifest(ROOT)
    inputs = config["inputs"]
    v1_config = load_v1_config(ROOT / inputs["v1_config"])
    v1_manifest = validate_v1_manifest(ROOT, ROOT / inputs["v1_manifest"])
    shared_scope_keys = [
        "execution_asset",
        "allowed_holdings",
        "allowed_target_states",
        "initial_capital_cny",
        "long_only",
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_allowed",
        "signal_time",
        "execution_time",
        "full_state_reinvests_paid_cash_dividends",
    ]
    if any(config["scope"][key] != v1_config["scope"][key] for key in shared_scope_keys):
        raise ValueError("候选scope偏离已冻结V1核心范围")
    for section in ("objective", "costs"):
        if config[section] != v1_config[section]:
            raise ValueError(f"候选{section}偏离已冻结V1")
    market, dividends, market_audit = load_v1_inputs(ROOT, v1_config)
    master = load_master(ROOT, config)
    partitions = config["partitions"]

    visible_features, visible_feature_audit = load_and_build_daily_features(
        ROOT,
        inputs["visible_panel"],
        master,
        required_split_group=config["universe"]["visible_required_split_group"],
        feature_start=partitions["visible_feature_start"],
        feature_end=partitions["visible_feature_end"],
        config=config,
    )
    visible_scored, visible_calibration = calibrate_and_apply_signal(
        visible_features,
        calibration_start=partitions["visible_score_calibration_start"],
        calibration_end=partitions["visible_score_calibration_end"],
        signal_start=partitions["visible_validation_signal_start"],
        signal_end=partitions["visible_validation_signal_end"],
        config=config,
    )
    visible_evaluation = evaluate_partition(
        market,
        dividends,
        visible_scored,
        signal_start=partitions["visible_validation_signal_start"],
        execution_start=partitions["visible_validation_execution_start"],
        execution_end=partitions["visible_validation_execution_end"],
        scenarios=["BASE", "STRESS"],
        config=config,
    )
    visible_gates = visible_gate_checks(
        visible_evaluation,
        visible_calibration,
        config,
    )

    replication_payload = None
    replication_scored = None
    replication_evaluation = None
    if visible_gates["all_visible_gates_pass"]:
        replication_features, replication_feature_audit = load_and_build_daily_features(
            ROOT,
            inputs["replication_panel"],
            master,
            required_split_group=config["universe"]["replication_required_split_group"],
            feature_start=partitions["replication_feature_start"],
            feature_end=partitions["replication_feature_end"],
            config=config,
        )
        replication_scored, replication_calibration = calibrate_and_apply_signal(
            replication_features,
            calibration_start=partitions["replication_score_calibration_start"],
            calibration_end=partitions["replication_score_calibration_end"],
            signal_start=partitions["replication_signal_start"],
            signal_end=partitions["replication_signal_end"],
            config=config,
        )
        replication_evaluation = evaluate_partition(
            market,
            dividends,
            replication_scored,
            signal_start=partitions["replication_signal_start"],
            execution_start=partitions["replication_execution_start"],
            execution_end=partitions["replication_execution_end"],
            scenarios=["STRESS"],
            config=config,
        )
        replication_gates = replication_gate_checks(
            replication_evaluation,
            replication_calibration,
            config,
        )
        replication_payload = {
            "feature_audit": replication_feature_audit,
            "calibration": replication_calibration,
            "stress_summary": replication_evaluation["paths"]["STRESS"]["summary"],
            "tail_capture": replication_evaluation["tail_capture"],
            "gates": replication_gates,
        }

    report = build_report(
        config,
        manifest,
        v1_manifest,
        market_audit,
        visible_feature_audit,
        visible_calibration,
        visible_evaluation,
        visible_gates,
        replication_payload,
    )

    output_directory = ROOT / artifacts["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=False)
    report_path = ROOT / artifacts["report_json"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    visible_scored.to_parquet(ROOT / artifacts["visible_daily_features"], index=False)
    visible_scored.loc[
        (visible_scored["date"] >= partitions["visible_validation_signal_start"])
        & (visible_scored["date"] <= partitions["visible_validation_signal_end"])
    ].to_parquet(ROOT / artifacts["visible_signals"], index=False)
    for scenario, prefix in (("BASE", "visible_base"), ("STRESS", "visible_stress")):
        visible_evaluation["paths"][scenario]["ledger"].to_parquet(
            ROOT / artifacts[f"{prefix}_ledger"],
            index=False,
        )
        visible_evaluation["paths"][scenario]["trades"].to_parquet(
            ROOT / artifacts[f"{prefix}_trades"],
            index=False,
        )
    if replication_payload is not None and replication_scored is not None and replication_evaluation is not None:
        replication_scored.to_parquet(
            ROOT / artifacts["replication_daily_features"],
            index=False,
        )
        replication_scored.loc[
            (replication_scored["date"] >= partitions["replication_signal_start"])
            & (replication_scored["date"] <= partitions["replication_signal_end"])
        ].to_parquet(ROOT / artifacts["replication_signals"], index=False)
        replication_evaluation["paths"]["STRESS"]["ledger"].to_parquet(
            ROOT / artifacts["replication_stress_ledger"],
            index=False,
        )
        replication_evaluation["paths"]["STRESS"]["trades"].to_parquet(
            ROOT / artifacts["replication_stress_trades"],
            index=False,
        )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ROOT / artifacts["report_markdown"]).write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "状态": report["status"],
                "目标达成": report["goal_achieved"],
                "可见空仓日": visible_calibration["cash_signal_days"],
                "压力年化净超额": report["visible"]["stress_summary"]["annualized_excess"],
                "压力滚动超额中位数": report["visible"]["stress_summary"][
                    "rolling_242d_excess_median"
                ],
                "复验面板已读取": report["replication_panel_read"],
                "报告": artifacts["report_markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
