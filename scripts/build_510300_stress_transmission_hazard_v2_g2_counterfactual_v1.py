"""按冻结协议执行 V2 反事实 G2 结构增量检验。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
)
from research.stress_transmission_hazard_v2_g2_counterfactual_v1 import (
    G2CounterfactualArtifacts,
    G2CounterfactualError,
    build_g2_counterfactual_artifacts,
)
from scripts.build_510300_stress_transmission_hazard_v2_four_state_ledger_v1 import (
    _write_parquet_new,
    _write_text_new,
)
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g2_counterfactual_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
    verify_registered_inputs,
)


TABLE_OUTPUTS = {
    "predictions": "predictions",
    "model_vintages": "model_vintages",
    "era_metrics": "era_metrics",
    "bootstrap_distribution": "bootstrap_distribution",
}


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _project_path(str(path)) for name, path in config["outputs"].items()
    }


def load_and_build(config: Mapping[str, Any]) -> tuple[G2CounterfactualArtifacts, pd.DataFrame]:
    verify_registered_inputs(config)
    inputs = config["inputs"]

    def read(name: str) -> pd.DataFrame:
        contract = inputs[name]
        return pd.read_parquet(
            _project_path(str(contract["path"])),
            columns=[str(column) for column in contract["read_columns"]],
        )

    samples = read("g1_sample_eligibility")
    events = read("g1_event_eligibility")
    model = config["model_contract"]
    split = config["prequential_split_contract"]
    gate = config["g2_gate_contract"]
    artifacts = build_g2_counterfactual_artifacts(
        samples=samples,
        events=events,
        historical_cutoff=config["program"]["historical_cutoff"],
        era_contracts=config["fixed_eras"],
        first_vintage_year=int(split["first_possible_model_vintage_year"]),
        last_vintage_year=int(split["last_model_vintage_year"]),
        l2_penalty=float(model["l2_penalty"]),
        max_iterations=int(model["optimizer_max_iterations"]),
        ftol=float(model["optimizer_ftol"]),
        gtol=float(model["optimizer_gtol"]),
        probability_clip=float(model["probability_clip"]),
        bootstrap_repetitions=int(gate["bootstrap_repetitions"]),
        bootstrap_seed=int(gate["bootstrap_seed"]),
        bootstrap_lower_quantile=float(gate["lower_quantile"]),
        required_positive_eras=int(gate["same_direction_eras_required"]),
    )
    return artifacts, events


def _validate_actual_parent_counts(
    artifacts: G2CounterfactualArtifacts,
    events: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    expected = config["expected_actual_parent_state"]
    sample = artifacts.result["sample"]
    checks = {
        "独立事件": (
            sample["total_independent_event_count"],
            expected["total_independent_event_count"],
        ),
        "正样本原点": (
            sample["total_positive_origin_count"],
            expected["total_positive_origin_count"],
        ),
        "非事件风险日": (
            sample["total_non_event_risk_day_count"],
            expected["total_non_event_risk_day_count"],
        ),
        "B2 可识别事件": (
            int(events["b2_identifiable_event"].astype(bool).sum()),
            expected["b2_identifiable_event_count"],
        ),
    }
    for label, (actual, frozen) in checks.items():
        if int(actual) != int(frozen):
            raise EvidenceContractError(
                f"真实父 G1 {label}不一致：actual={actual}, frozen={frozen}"
            )
    if artifacts.result["actual_G1_DATA_AND_EVENTS"] != expected["G1_DATA_AND_EVENTS"]:
        raise EvidenceContractError("反事实 G2 意外改写真实 G1")
    if artifacts.result["formal_g2_status"] != "NOT_ADMISSIBLE_ACTUAL_G1_REMAINS_NO_VIEW":
        raise EvidenceContractError("反事实 G2 意外产生正式准入")


def _format_optional(value: Any, digits: int = 8) -> str:
    if value is None or pd.isna(value):
        return "NO_VIEW"
    return f"{float(value):.{digits}f}"


def _build_report(artifacts: G2CounterfactualArtifacts) -> str:
    result = artifacts.result
    gate = result["gate_checks"]
    sample = result["sample"]
    metrics = result["model_metrics"]
    era_lines = [
        "| 时代 | 状态 | 行数 | 事件 | 非事件日 | B1-B2 Log Loss 改善 | 正方向 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["era_results"]:
        era_lines.append(
            f"| `{row['era_id']}` | `{row['prediction_state']}` | {row['row_count']} | "
            f"{row['positive_event_count']} | {row['negative_risk_day_count']} | "
            f"{_format_optional(row['log_loss_improvement_B1_minus_B2'])} | "
            f"{'是' if row['same_direction_positive'] else '否'} |"
        )
    metric_lines = [
        "| 模型 | Log Loss | Brier | PR-AUC | ROC-AUC（仅描述） | 校准截距 | 校准斜率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B0", "B1", "B2"):
        item = metrics[model]
        metric_lines.append(
            f"| {model} | {_format_optional(item['event_weighted_log_loss'])} | "
            f"{_format_optional(item['event_weighted_brier_score'])} | "
            f"{_format_optional(item['event_weighted_pr_auc'])} | "
            f"{_format_optional(item['event_weighted_roc_auc_description_only'])} | "
            f"{_format_optional(item['calibration_intercept'])} | "
            f"{_format_optional(item['calibration_slope'])} |"
        )
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2：反事实 G2 结构增量 V1",
            "",
            "## 结论先行",
            "",
            f"- 反事实 G2：`{result['counterfactual_g2_status']}`。",
            f"- 正式状态：`{result['formal_g2_status']}`。",
            "- 真实 G1 仍为 `NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY`；本次只按用户指令假设开门，没有补造事件或特征。",
            f"- B1-B2 事件加权 Log Loss 改善：`{result['event_weighted_log_loss_improvement_B1_minus_B2']:.10f}`。",
            f"- 正方向时代：`{result['positive_direction_era_count']}/4`（要求至少 3/4）。",
            f"- 事件/日历年块 Bootstrap 单侧 90% 下界：`{result['bootstrap']['lower_bound']:.10f}`。",
            "",
            "## 三项冻结门",
            "",
            f"- 总体改善严格大于 0：`{str(gate['overall_event_weighted_log_loss_improvement_strictly_positive']).lower()}`。",
            f"- 至少 3/4 时代同方向：`{str(gate['at_least_three_of_four_eras_same_direction']).lower()}`。",
            f"- Bootstrap 单侧 90% 下界严格大于 0：`{str(gate['event_and_calendar_year_block_bootstrap_one_sided_90pct_lower_bound_strictly_positive']).lower()}`。",
            "",
            "## 历史 prequential 样本",
            "",
            f"- 父共同样本：{sample['common_sample_day_count']} 日，{sample['common_identifiable_event_count']} 个事件，{sample['common_non_event_risk_day_count']} 个非事件风险日。",
            f"- 真正取得无前瞻训练预测：{sample['scored_prediction_row_count']} 行，{sample['scored_positive_event_count']} 个事件，{sample['scored_negative_risk_day_count']} 个非事件风险日。",
            f"- 可用年度模型版本：{sample['view_allowed_model_vintage_count']}。训练标签均在模型年份开始前到期。",
            "",
            "## 模型描述指标",
            "",
            *metric_lines,
            "",
            "AUC 只作描述，不参与 G2 通过裁决。",
            "",
            "## 四时代方向",
            "",
            *era_lines,
            "",
            "## 边界",
            "",
            "- B1/B2 使用完全相同的父 G1 共同样本和权重；所有斜率非负，L2 固定 1.0。",
            "- 每个年度模型只使用该年 1 月 1 日前已经到期的标签；没有使用未来自然年或未到期标签。",
            "- 读取了历史 BAD10 指示值，但没有读取最小路径收益或其他实际未来路径数值。",
            "- 没有选择概率阈值，没有运行 G3/G4，没有读取或生成组合收益、净值、夏普、回撤。",
            "- 没有生成 Paper/Shadow、经纪商、仓位或订单动作，`POSITION_IMPACT=0`。",
            f"- 下一步：`{result['next_allowed_step']}`。",
            "",
        ]
    )


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = _output_paths(config)
    owned = [*TABLE_OUTPUTS.keys(), "result", "report", "build_receipt"]
    existing = [str(outputs[name]) for name in owned if outputs[name].exists()]
    if existing:
        raise EvidenceContractError(f"反事实 G2 构建输出已存在，禁止覆盖：{existing}")

    artifacts, events = load_and_build(config)
    _validate_actual_parent_counts(artifacts, events, config)
    print("反事实 G2 模型与门禁计算完成，正在原子写入结果。", flush=True)
    table_evidence: dict[str, dict[str, Any]] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        frame = getattr(artifacts, attribute_name)
        _write_parquet_new(frame, outputs[output_name])
        persisted = pd.read_parquet(outputs[output_name])
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            persisted.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
        table_evidence[output_name] = {
            **file_evidence(outputs[output_name], project_root=ROOT),
            "row_count": int(len(persisted)),
            "persisted_semantic_sha256": frame_semantic_sha256(persisted),
        }
    atomic_write_json_new(outputs["result"], artifacts.result)
    _write_text_new(_build_report(artifacts), outputs["report"])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G2_COUNTERFACTUAL_BUILD_V1",
        "created_at": now,
        "status": "PASS_COUNTERFACTUAL_G2_BUILD_PENDING_CLEAN_REPLAY",
        "counterfactual_g2_status": artifacts.result["counterfactual_g2_status"],
        "formal_g2_status": artifacts.result["formal_g2_status"],
        "frozen_manifest": file_evidence(
            _project_path(config["freeze_contract"]["manifest_output"]),
            project_root=ROOT,
        ),
        "inputs": {
            name: file_evidence(_project_path(contract["path"]), project_root=ROOT)
            for name, contract in config["inputs"].items()
        },
        "outputs": {
            **table_evidence,
            "result": file_evidence(outputs["result"], project_root=ROOT),
            "report": file_evidence(outputs["report"], project_root=ROOT),
        },
        "result_payload_sha256": canonical_sha256(artifacts.result),
        "actual_g1_state_overwritten": False,
        "historical_bad10_label_read": True,
        "actual_future_path_value_read": False,
        "observation_after_historical_cutoff_read": False,
        "future_calendar_year_training_used": False,
        "unmatured_label_training_used": False,
        "counterfactual_models_trained": True,
        "probability_threshold_selected": False,
        "formal_model_admitted": False,
        "portfolio_or_performance_artifact_read": False,
        "paper_or_shadow_authorized": False,
        "broker_connection_authorized": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["build_receipt"], receipt)
    print(
        f"反事实 G2 构建完成：{artifacts.result['counterfactual_g2_status']}；"
        "真实 G1 未改写，未运行组合或交易。"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        build(args.config)
        return 0
    except (
        EvidenceContractError,
        G2CounterfactualError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"反事实 G2 构建失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
