"""冻结并运行 510300 结构信号功效与可识别性审计。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.structural_signal_power_identifiability_audit_v1 import (  # noqa: E402
    build_nonoverlap_diagnostics,
    build_paired_loss_panel,
    build_power_requirements,
    build_sharpe_identifiability,
    realized_annualized_volatility,
)


PROGRAM_ID = "510300_STRUCTURAL_SIGNAL_POWER_AND_IDENTIFIABILITY_AUDIT_V1"
CONFIG_PATH = ROOT / "config/510300_structural_signal_power_and_identifiability_audit_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/structural_signal_power_identifiability_audit_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("功效审计配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    _atomic_bytes_new(path, (payload + "\n").encode("utf-8"))


def atomic_text_new(path: Path, value: str) -> None:
    _atomic_bytes_new(path, value.encode("utf-8"))


def atomic_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def validate_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = config["parent_contract"]
    engine_manifest = json.loads(
        project_path(parent["engine_manifest"]).read_text(encoding="utf-8")
    )
    engine_status = json.loads(
        project_path(parent["engine_status"]).read_text(encoding="utf-8")
    )
    if engine_manifest.get("manifest_payload_sha256") != parent[
        "expected_engine_manifest_payload_sha256"
    ]:
        raise ValueError("测量架构修正清单哈希不一致")
    if engine_status.get("status_payload_sha256") != parent[
        "expected_engine_status_payload_sha256"
    ]:
        raise ValueError("测量架构状态哈希不一致")
    if engine_status.get("status") != parent["expected_engine_status"]:
        raise ValueError("测量架构状态不允许功效审计")
    if engine_status.get("return_prediction_allowed") is not False:
        raise ValueError("测量架构意外开放收益预测")

    source = config["source_contract"]
    prediction_status = json.loads(
        project_path(source["rejected_prediction_status"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if prediction_status.get("status_payload_sha256") != source[
        "rejected_prediction_status"
    ]["expected_payload_sha256"]:
        raise ValueError("冻结拒绝预测状态哈希不一致")
    if prediction_status.get("validation_summary", {}).get(
        "structural_prediction_validated"
    ) is not False:
        raise ValueError("冻结预测意外通过验证")
    file_checks = (
        (
            source["prediction_rows"]["path"],
            source["prediction_rows"]["expected_sha256"],
            "冻结预测记录",
        ),
        (
            source["total_return_index"]["path"],
            source["total_return_index"]["expected_sha256"],
            "总回报指数",
        ),
    )
    for relative, expected, label in file_checks:
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label}缺失或哈希不一致")
    return engine_status, prediction_status


def _registered_inputs(config: dict[str, Any]) -> dict[str, str]:
    parent = config["parent_contract"]
    source = config["source_contract"]
    return {
        parent["engine_manifest"]: "FROZEN_MEASUREMENT_ARCHITECTURE_PROTOCOL",
        parent["engine_status"]: "FROZEN_MEASUREMENT_ARCHITECTURE_STATUS",
        source["rejected_prediction_status"]["path"]: "FROZEN_REJECTED_FORECAST_STATUS",
        source["prediction_rows"]["path"]: "PAIRED_OOS_LOSS_SOURCE",
        source["total_return_index"]["path"]: "VOLATILITY_SCALE_ONLY",
    }


def _output_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(value)
        for key, value in config["artifacts"].items()
        if key != "protocol_manifest"
    ]


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"功效审计冻结清单已存在：{manifest_path}")
    existing = [str(path) for path in _output_paths(config) if path.exists()]
    if existing:
        raise RuntimeError(f"冻结前发现同名功效审计输出：{existing}")
    engine_status, prediction_status = validate_inputs(config)
    registered: dict[str, Any] = {}
    for relative, role in _registered_inputs(config).items():
        path = project_path(relative)
        registered[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_POWER_CALCULATION",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered,
        "engine_status_payload_sha256": engine_status["status_payload_sha256"],
        "prediction_status_payload_sha256": prediction_status[
            "status_payload_sha256"
        ],
        "effect_sizes_hac_blocks_and_sharpe_formula_frozen": True,
        "current_specification_can_recover": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json_new(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("功效审计协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_hash(payload) != expected_hash:
        raise ValueError("功效审计清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("功效审计配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("功效审计配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"功效审计实现文件发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"功效审计输入发生漂移或缺失：{relative}")
    validate_inputs(config)
    return manifest


def _build_report(
    manifest: dict[str, Any],
    paired: pd.DataFrame,
    power: pd.DataFrame,
    nonoverlap: pd.DataFrame,
    volatility: dict[str, Any],
    sharpe: dict[str, Any],
) -> str:
    observed = power.drop_duplicates(["target_id", "baseline_model"])
    lines = [
        "# 510300_STRUCTURAL_SIGNAL_POWER_AND_IDENTIFIABILITY_AUDIT_V1",
        "",
        "## 裁决",
        "",
        "本审计不恢复已拒绝的预测 V1，也不进行组合回测。它回答的是：若真实改善只有 1%、2% 或 5%，现有月度 origin 的噪声与标签重叠需要多少样本才能识别。",
        "",
        f"- 协议哈希：`{manifest['manifest_payload_sha256']}`",
        f"- 配对损失记录：`{len(paired)}` 行；功效情景：`{len(power)}` 行；不重叠相位：`{len(nonoverlap)}` 行。",
        "- `CURRENT_SPECIFICATION_CAN_RECOVER=false`",
        "- `PORTFOLIO_ACTION=ABSTAIN`，`POSITION_STATE=POSITION_UNSET`",
        "",
        "## 当前冻结模型的观测损失",
        "",
        "| 目标 | 基准 | 共同 origin | 候选 MSE | 基准 MSE | 相对改善 | 当前胜出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in observed.itertuples(index=False):
        lines.append(
            "| {target} | {baseline} | {count} | {candidate:.6f} | {base:.6f} | {improvement:.2%} | {beats} |".format(
                target=row.target_id,
                baseline=row.baseline_model,
                count=int(row.observed_origin_count),
                candidate=float(row.candidate_mse),
                base=float(row.baseline_mse),
                improvement=float(row.observed_relative_mse_improvement),
                beats="是" if row.observed_candidate_beats_baseline else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 1%、2%、5% MSE 改善的样本需求",
            "",
            "下表采用 `max(HAC 调整月度 origin, 按 horizon 抽取的不重叠 origin)` 作为保守需求。60D 使用 3 个月 block，120D 使用 6 个月 block。",
            "",
            "| 目标 | 基准 | 改善 | IID 独立周期 | HAC 月度 origin | 不重叠月度 origin | 保守总年数 | 额外年数 | 状态 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in power.itertuples(index=False):
        lines.append(
            "| {target} | {baseline} | {effect:.0%} | {iid} | {hac} | {nonoverlap} | {years:.1f} | {additional:.1f} | {status} |".format(
                target=row.target_id,
                baseline=row.baseline_model,
                effect=float(row.hypothetical_relative_mse_improvement),
                iid=int(row.required_independent_cycles_iid),
                hac=int(row.required_monthly_origins_hac),
                nonoverlap=int(row.required_monthly_origins_nonoverlap),
                years=float(row.conservative_total_calendar_years),
                additional=float(row.conservative_additional_calendar_years),
                status=row.identifiability_status,
            )
        )
    lines.extend(
        [
            "",
            "## Sharpe 0.29 → 1.20 的经济与统计量级",
            "",
            f"- H00300 在 `{volatility['start_date']}` 至 `{volatility['end_date']}` 的年化波动率尺度为 `{volatility['annualized_volatility']:.2%}`。这只是标的尺度，不是策略收益。",
            f"- 同波动率下，Sharpe 增量 `0.91` 对应至少 `{sharpe['required_gross_annual_excess_return_increment_at_same_volatility']:.2%}` 的毛年化超额收益增量；成本尚未计入。",
            f"- 高斯 IID 近似的保守下界为 `{sharpe['iid_gaussian_years_using_target_standard_error']}` 个独立年；若方差因依赖性放大 2 倍或 3 倍，下界分别为 `{sharpe['dependence_scenarios'][1]['required_calendar_years_lower_bound']}` 年和 `{sharpe['dependence_scenarios'][2]['required_calendar_years_lower_bound']}` 年。",
            "- 当前不存在通过验证的可投资策略收益序列，所以不能执行经验 Sharpe 检验，也不能把上述下界解释为可实现性。",
            "",
            "## 结论",
            "",
            "- 未证明数学不可能。",
            "- 已证明当前 V1 在已观察样本上没有获得所需预测改善，且小幅改善的可靠识别通常需要远超十年的保守样本。",
            "- 因此等待标签仅保留为被动诊断；主动研究转向总回报成分账本及 CF/DR/RC 各自对象的测量有效性。",
            "- 不生成仓位、订单、NAV 或 Sharpe 绩效声明。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(config)
    source = config["source_contract"]
    predictions = pd.read_parquet(project_path(source["prediction_rows"]["path"]))
    total_return = pd.read_parquet(project_path(source["total_return_index"]["path"]))
    power_config = config["forecast_power"]
    program = config["program"]
    paired = build_paired_loss_panel(
        predictions,
        target_specs=power_config["targets"],
        candidate_model=power_config["candidate_model"],
        baseline_models=list(power_config["baseline_models"]),
        evaluation_start=str(program["evaluation_start"]),
        evaluation_end=str(program["evaluation_end"]),
    )
    nonoverlap = build_nonoverlap_diagnostics(
        paired, target_specs=power_config["targets"]
    )
    power = build_power_requirements(
        paired,
        target_specs=power_config["targets"],
        relative_effects=[
            float(value)
            for value in power_config["hypothetical_relative_mse_improvements"]
        ],
        z_one_sided_alpha=float(power_config["z_one_sided_alpha"]),
        z_target_power=float(power_config["z_target_power"]),
        origins_per_year=int(power_config["origins_per_year"]),
        short_horizon_years=float(power_config["short_horizon_years"]),
        acceptable_years=float(power_config["acceptable_identification_years"]),
    )
    sharpe_config = config["sharpe_identifiability"]
    volatility = realized_annualized_volatility(
        total_return,
        date_column=source["total_return_index"]["date_column"],
        value_column=source["total_return_index"]["value_column"],
        start=str(program["evaluation_start"]),
        end=str(program["evaluation_end"]),
        annualization_days=int(sharpe_config["annualization_days"]),
    )
    sharpe = build_sharpe_identifiability(
        reference_sharpe=float(sharpe_config["reference_sharpe"]),
        target_sharpe=float(sharpe_config["target_sharpe"]),
        realized_volatility=float(volatility["annualized_volatility"]),
        z_one_sided_alpha=float(power_config["z_one_sided_alpha"]),
        z_target_power=float(power_config["z_target_power"]),
        dependence_multipliers=[
            float(value)
            for value in sharpe_config["dependence_scenario_multipliers"]
        ],
    )

    artifacts = config["artifacts"]
    paths = {name: project_path(relative) for name, relative in artifacts.items()}
    pending = [path for name, path in paths.items() if name != "protocol_manifest" and path.exists()]
    if pending:
        raise RuntimeError(f"功效审计输出已存在，禁止覆盖：{pending}")
    atomic_parquet_new(paths["paired_loss_panel"], paired)
    atomic_parquet_new(paths["power_requirements"], power)
    atomic_parquet_new(paths["nonoverlap_diagnostics"], nonoverlap)

    status_counts = {
        str(key): int(value)
        for key, value in power["identifiability_status"].value_counts().items()
    }
    infeasible = power.loc[
        power["identifiability_status"].eq("NOT_IDENTIFIABLE_WITHIN_TEN_YEARS"),
        [
            "target_id",
            "baseline_model",
            "hypothetical_relative_mse_improvement",
            "conservative_total_calendar_years",
            "conservative_additional_calendar_years",
        ],
    ]
    observed = power.drop_duplicates(["target_id", "baseline_model"])[
        [
            "target_id",
            "baseline_model",
            "observed_origin_count",
            "baseline_mse",
            "candidate_mse",
            "observed_relative_mse_improvement",
            "observed_candidate_beats_baseline",
        ]
    ]
    phase_summary = (
        nonoverlap.groupby(["target_id", "baseline_model"], as_index=False)
        .agg(
            phase_count=("phase", "count"),
            phases_candidate_beats_baseline=("candidate_beats_baseline", "sum"),
            minimum_relative_mse_improvement=("relative_mse_improvement", "min"),
            maximum_relative_mse_improvement=("relative_mse_improvement", "max"),
        )
    )
    audit = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "POWER_AND_IDENTIFIABILITY_AUDIT_COMPLETE_NO_MODEL_REQUALIFICATION",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "evaluation_scope": {
            "start": str(program["evaluation_start"]),
            "end": str(program["evaluation_end"]),
            "candidate_model": power_config["candidate_model"],
            "baseline_models": list(power_config["baseline_models"]),
        },
        "observed_loss_comparisons": dataframe_records(observed),
        "power_requirements": dataframe_records(power),
        "nonoverlap_phase_summary": dataframe_records(phase_summary),
        "realized_volatility_scale": volatility,
        "sharpe_identifiability": sharpe,
        "identifiability_status_counts": status_counts,
        "targets_not_identifiable_within_ten_years": dataframe_records(infeasible),
        "current_specification_can_recover": False,
        "mathematical_impossibility_proven": False,
        "waiting_for_existing_labels": "PASSIVE_MONITORING_ONLY",
        "next_research_action": "BUILD_510300_TOTAL_RETURN_COMPONENT_LEDGER_V1",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    audit["payload_sha256"] = canonical_hash(audit)
    atomic_json_new(paths["audit_json"], audit)
    report = _build_report(
        manifest, paired, power, nonoverlap, volatility, sharpe
    )
    atomic_text_new(paths["final_report"], report)

    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "POWER_AND_IDENTIFIABILITY_AUDIT_COMPLETE_NO_MODEL_REQUALIFICATION",
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "paired_loss_panel": {
            "path": paths["paired_loss_panel"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["paired_loss_panel"]),
            "rows": len(paired),
        },
        "power_requirements": {
            "path": paths["power_requirements"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["power_requirements"]),
            "rows": len(power),
            "identifiability_status_counts": status_counts,
            "not_identifiable_within_ten_years_count": int(len(infeasible)),
        },
        "nonoverlap_diagnostics": {
            "path": paths["nonoverlap_diagnostics"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["nonoverlap_diagnostics"]),
            "rows": len(nonoverlap),
        },
        "sharpe_identifiability": sharpe,
        "current_specification_can_recover": False,
        "mathematical_impossibility_proven": False,
        "accessible_free_information_feasibility_validated": False,
        "waiting_for_existing_labels": "PASSIVE_MONITORING_ONLY",
        "next_research_action": "BUILD_510300_TOTAL_RETURN_COMPONENT_LEDGER_V1",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "current_investable_strategy": "NONE",
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "nav_calculated": False,
        "sharpe_calculated": False,
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json_new(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 功效与可识别性审计")
    parser.add_argument("--phase", choices=["freeze", "audit", "all", "verify"], default="all")
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"audit", "all"}:
        run_audit(config, manifest)
    if args.phase == "verify":
        print("功效审计协议与全部冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
