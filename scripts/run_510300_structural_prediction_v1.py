"""运行 510300 结构预测 V1 的冻结、状态构造和滚动样本外评估。"""

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

from backtest.valuation_fvg_engine import build_valuation_signals  # noqa: E402
from research.structural_prediction_v1 import (  # noqa: E402
    attach_price_features_and_targets,
    build_increment_results,
    build_prediction_state_panel,
    evaluate_predictions,
    feature_columns_by_model,
    normalize_dates,
    validation_summary,
    walk_forward_predictions,
)


PROGRAM_ID = "510300_STRUCTURAL_PREDICTION_V1"
CONFIG_PATH = ROOT / "config/510300_structural_prediction_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/structural_prediction_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("结构预测配置的 PROGRAM_ID 不匹配")
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


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    atomic_bytes(path, (payload + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _validate_parent(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = config["parent_contract"]
    manifest = json.loads(project_path(contract["manifest"]).read_text(encoding="utf-8"))
    status = json.loads(
        project_path(contract["authoritative_status"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        project_path(contract["condition_state_receipt"]).read_text(encoding="utf-8")
    )
    checks = [
        (
            manifest.get("manifest_payload_sha256"),
            contract["expected_parent_manifest_payload_sha256"],
            "第二阶段协议哈希",
        ),
        (
            status.get("status_payload_sha256"),
            contract["expected_parent_status_payload_sha256"],
            "第二阶段状态哈希",
        ),
        (
            receipt.get("receipt_payload_sha256"),
            contract["expected_parent_condition_receipt_payload_sha256"],
            "第二阶段条件回执哈希",
        ),
        (status.get("status"), contract["expected_parent_status"], "第二阶段状态"),
    ]
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"{label}与第三阶段配置不一致")
    condition_path = project_path(contract["condition_state_panel"])
    if sha256_file(condition_path) != contract["expected_parent_condition_panel_sha256"]:
        raise ValueError("第二阶段条件面板哈希不一致")
    if status.get("portfolio_evaluation_allowed") is not False:
        raise ValueError("第二阶段意外开放了组合评估")
    return manifest, status, receipt


def _registered_input_roles(config: dict[str, Any]) -> dict[str, str]:
    parent = config["parent_contract"]
    source = config["source_contract"]
    return {
        parent["manifest"]: "FROZEN_STAGE_2_PROTOCOL",
        parent["authoritative_status"]: "FROZEN_STAGE_2_STATUS",
        parent["condition_state_receipt"]: "FROZEN_STAGE_2_STATE_RECEIPT",
        parent["condition_state_panel"]: "OUTCOME_BLIND_STRUCTURAL_STATE",
        source["parent_present_value_panel"]["path"]: (
            "OUTCOME_BLIND_PRESENT_VALUE_DIAGNOSTICS"
        ),
        source["old_r5_valuation"]["raw_path"]: (
            "POINT_IN_TIME_FIXED_OLD_R5_BASELINE_INPUT"
        ),
        source["old_r5_valuation"]["config_path"]: "FIXED_OLD_R5_BASELINE_CONFIG",
        source["old_r5_valuation"]["implementation_path"]: (
            "FIXED_OLD_R5_BASELINE_IMPLEMENTATION"
        ),
        source["total_return_locked_until_prediction_state_freeze"]["path"]: (
            "LOCKED_OUTCOME_NOT_VALUE_READ"
        ),
    }


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"冻结清单已存在，禁止静默覆盖：{manifest_path}")
    parent_manifest, parent_status, parent_receipt = _validate_parent(config)
    registered_inputs: dict[str, Any] = {}
    for relative, role in _registered_input_roles(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"第三阶段冻结输入缺失：{path}")
        registered_inputs[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_PREDICTION_STATE_BUILD_AND_OUTCOME_VALUE_READ",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered_inputs,
        "parent_manifest_payload_sha256": parent_manifest[
            "manifest_payload_sha256"
        ],
        "parent_status_payload_sha256": parent_status["status_payload_sha256"],
        "parent_condition_receipt_payload_sha256": parent_receipt[
            "receipt_payload_sha256"
        ],
        "targets_models_features_hyperparameters_frozen": True,
        "h00300_outcome_values_read": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "sharpe_target_1_2_achieved": False,
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("第三阶段协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_payload_sha256", None)
    if canonical_hash(without_hash) != expected_hash:
        raise ValueError("第三阶段协议清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("第三阶段配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("第三阶段配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"第三阶段实现文件发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"第三阶段冻结输入发生漂移或缺失：{relative}")
    _validate_parent(config)
    return manifest


def _feature_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    feature_config = config["features"]
    group_names = [
        "old_r5_valuation",
        "price_realized_risk",
        "cf",
        "dr",
        "rc",
    ]
    groups = {name: list(feature_config[name]) for name in group_names}
    groups["interactions"] = list(feature_config["interactions"].keys())
    return groups


def build_and_freeze_prediction_states(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    parent_path = project_path(config["parent_contract"]["condition_state_panel"])
    pv_path = project_path(
        config["source_contract"]["parent_present_value_panel"]["path"]
    )
    old_contract = config["source_contract"]["old_r5_valuation"]
    old_config = yaml.safe_load(
        project_path(old_contract["config_path"]).read_text(encoding="utf-8")
    )
    condition = pd.read_parquet(parent_path)
    present_value = pd.read_parquet(pv_path)
    old_raw = pd.read_parquet(project_path(old_contract["raw_path"]))
    old_signals = build_valuation_signals(old_raw, old_config["valuation"])
    state = build_prediction_state_panel(
        condition,
        present_value,
        old_signals,
        old_r5_asof_tolerance_calendar_days=int(
            old_contract["asof_tolerance_calendar_days"]
        ),
    )
    state["origin"] = normalize_dates(state["origin"])
    state = state.loc[
        state["origin"].between(
            pd.Timestamp(config["program"]["state_history_start"]),
            pd.Timestamp(config["program"]["state_origin_cutoff"]),
        )
    ].reset_index(drop=True)
    feature_groups = _feature_groups(config)
    model_features = feature_columns_by_model(
        feature_groups, config["models"]
    )
    state_only_features = sorted(
        {
            column
            for model_id, columns in model_features.items()
            for column in columns
            if column not in feature_groups["price_realized_risk"]
        }
    )
    missing = sorted(set(state_only_features).difference(state.columns))
    if missing:
        raise ValueError(f"预测状态面板缺少冻结特征：{missing}")
    if bool(state["h00300_values_read"].any()):
        raise ValueError("预测状态冻结阶段出现 H00300 收益读取标记")

    state_path = project_path(config["artifacts"]["prediction_state_panel"])
    atomic_parquet(state_path, state)
    primary_pass = state["condition_quality_status"].eq(
        config["features"]["quality_policy"][
            "primary_admitted_condition_status"
        ]
    )
    receipt = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "PREDICTION_STATES_FROZEN_BEFORE_H00300_OUTCOME_VALUE_READ",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "prediction_state_panel": {
            "path": state_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(state_path),
            "rows": int(len(state)),
            "first_origin": str(state["origin"].min().date()),
            "last_origin": str(state["origin"].max().date()),
        },
        "primary_pass_state_origins": int(primary_pass.sum()),
        "excluded_partial_or_no_view_origins": int((~primary_pass).sum()),
        "old_r5_baseline_status_counts": {
            str(key): int(value)
            for key, value in state["old_r5_baseline_status"].value_counts().items()
        },
        "model_feature_columns": model_features,
        "h00300_outcome_values_read": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    atomic_json(
        project_path(config["artifacts"]["prediction_state_receipt"]), receipt
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def verify_prediction_state_freeze(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    receipt_path = project_path(config["artifacts"]["prediction_state_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("第三阶段预测状态尚未冻结")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_hash = receipt.get("receipt_payload_sha256")
    without_hash = dict(receipt)
    without_hash.pop("receipt_payload_sha256", None)
    if canonical_hash(without_hash) != expected_hash:
        raise ValueError("预测状态回执自身哈希失败")
    if (
        receipt.get("protocol_manifest_payload_sha256")
        != manifest["manifest_payload_sha256"]
    ):
        raise ValueError("预测状态回执未绑定当前协议")
    state_path = project_path(config["artifacts"]["prediction_state_panel"])
    if sha256_file(state_path) != receipt["prediction_state_panel"]["sha256"]:
        raise ValueError("冻结预测状态面板发生漂移")
    if receipt.get("h00300_outcome_values_read") is not False:
        raise ValueError("预测状态回执的 H00300 读取状态不合法")
    return receipt


def _percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2%}"


def _number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _metric_table(metrics: pd.DataFrame) -> list[str]:
    lines = [
        "| 目标 | 模型 | n | 主损失 | Spearman/AUC | 符号准确率/Brier Skill | 可靠性 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in metrics.iterrows():
        continuous = row["target_type"] == "CONTINUOUS_RETURN"
        auxiliary = (
            _number(row.get("spearman_correlation"))
            if continuous
            else _number(row.get("roc_auc"))
        )
        secondary = (
            _percent(row.get("sign_accuracy"))
            if continuous
            else _number(row.get("brier_skill_vs_historical_mean"))
        )
        lines.append(
            f"| {row['target_id']} | {row['model_id']} | "
            f"{int(row['observed_prediction_count'])} | {_number(row.get('primary_loss'), 6)} | "
            f"{auxiliary} | {secondary} | {row['reliability_status']} |"
        )
    return lines


def _increment_table(increments: pd.DataFrame) -> list[str]:
    lines = [
        "| 比较 | 目标 | 候选 | 参照 | 损失改善 | 结果 |",
        "|---|---|---|---|---:|---|",
    ]
    for _, row in increments.iterrows():
        lines.append(
            f"| {row['comparison_id']} | {row['target_id']} | {row['candidate_model']} | "
            f"{row['reference_model']} | {_percent(row.get('relative_loss_improvement'))} | {row['result']} |"
        )
    return lines


def _latest_forecasts(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    generated = predictions.loc[predictions["prediction"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for (target_id, model_id), group in generated.groupby(
        ["target_id", "model_id"], sort=True
    ):
        latest = group.sort_values("origin").iloc[-1]
        rows.append(
            {
                "target_id": str(target_id),
                "model_id": str(model_id),
                "origin": str(pd.Timestamp(latest["origin"]).date()),
                "prediction": float(latest["prediction"]),
                "actual": (
                    None if pd.isna(latest["actual"]) else float(latest["actual"])
                ),
                "actual_status": str(latest["actual_status"]),
                "training_observation_count": int(
                    latest["training_observation_count"]
                ),
            }
        )
    return rows


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state_receipt: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    increments: pd.DataFrame,
    validation: dict[str, Any],
    latest_forecasts: list[dict[str, Any]],
) -> str:
    primary_scope = config["evaluation"]["primary_scope"]
    primary_metrics = metrics.loc[
        metrics["sample_scope"].eq(primary_scope)
    ].sort_values(["target_id", "model_id"])
    primary_increments = increments.sort_values(
        ["comparison_type", "comparison_id", "target_id"]
    )
    generated = int(predictions["prediction"].notna().sum())
    observed = int(
        (predictions["prediction"].notna() & predictions["actual"].notna()).sum()
    )
    latest_final = [
        row
        for row in latest_forecasts
        if row["model_id"] == config["evaluation"]["final_model"]
    ]

    lines = [
        "# 510300 结构预测 V1",
        "",
        "## 结论先行",
        "",
        f"第三阶段冻结滚动样本外预测已经完成，最终验证状态为 `{validation['validation_status']}`。",
        "",
        f"- 预测记录中已生成预测 {generated} 条，其中目标已观察 {observed} 条。",
        f"- CF 增量：`{validation['module_increment_statuses']['CF_INCREMENT']}`；DR 增量：`{validation['module_increment_statuses']['DR_INCREMENT']}`；RC 增量：`{validation['module_increment_statuses']['RC_INCREMENT']}`；交互增量：`{validation['module_increment_statuses']['INTERACTION_INCREMENT']}`。",
        "- 只有 CF、DR、RC、交互以及最终模型相对三个基准在三个目标上全部 PASS，才允许宣布结构预测验证通过；没有因本轮结果放宽门槛。",
        "- 本阶段仍不生成仓位、净值或夏普率。预测通过与可交易净夏普率 1.2 不是同一结论。",
        "",
        "## 协议与样本边界",
        "",
        f"- 协议清单哈希：`{manifest['manifest_payload_sha256']}`。",
        f"- 预测状态回执哈希：`{state_receipt['receipt_payload_sha256']}`。",
        f"- 状态从 2015 年开始滚动；父状态 PASS origin={state_receipt['primary_pass_state_origins']}，PARTIAL/NO_VIEW origin={state_receipt['excluded_partial_or_no_view_origins']}，后者不进入主模型。",
        "- 每个预测只使用在预测 origin 之前已经到期的标签；60D/120D 标签即使来自更早 origin，只要尚未到期也不能进入训练。",
        "- 固定最少 36 个成熟训练样本；连续目标使用固定 Ridge(alpha=10)，回撤概率使用固定 L2 Logistic(C=0.10)，不调参、不筛特征。",
        "- 20D 回撤定义继承第一阶段 -10% 回撤激活线：未来 1—20 个交易日的最低全收益相对 origin 收益不高于 -10%。",
        "",
        f"## {primary_scope} 样本外指标",
        "",
        *_metric_table(primary_metrics),
        "",
        "## 模块增量与最终模型对基准",
        "",
        *_increment_table(primary_increments),
        "",
        "## 最新研究预测（不是仓位）",
        "",
        "| 目标 | origin | 最终模型预测 | 实际状态 | 训练样本 |",
        "|---|---|---:|---|---:|",
    ]
    for row in latest_final:
        value = (
            _percent(row["prediction"])
            if row["target_id"] != "20D_DRAWDOWN_PROBABILITY"
            else _percent(row["prediction"])
        )
        lines.append(
            f"| {row['target_id']} | {row['origin']} | {value} | "
            f"{row['actual_status']} | {row['training_observation_count']} |"
        )
    lines.extend(
        [
            "",
            "这些数值只用于检验冻结模型。没有将预期收益或回撤概率映射为 0%—100% 仓位。",
            "",
            "## 可靠性限制",
            "",
            "- 2021+ 是用户指定主评估范围；若模型因为 36 期训练和标签到期要求而更晚才产生预测，实际首个 OOS origin 以结果表为准。",
            "- 月度 60D/120D 标签重叠，虽然训练只使用成熟标签，但 OOS 误差仍不独立；这里不报告常规独立样本显著性。",
            "- 回撤概率只有在 OOS 至少 5 个事件和 20 个非事件时才具备 PASS 评估资格；事件不足保持 NO_VIEW。",
            "- 旧 R5 只作为冻结比较基准，不恢复其历史策略资格；其 PE/PB 数据也不是官方不可变首发档案。",
            "- 财务修订、历史权重代理、信用利差和 ETF 份额发布时钟等父项目限制全部继续有效。",
            "",
            "## 是否进入仓位阶段",
            "",
            (
                "冻结预测已通过全部门槛，但仍需另行预注册唯一一套 510300/现金仓位映射，当前没有授权或执行。"
                if validation["structural_prediction_validated"]
                else "冻结预测没有通过全部门槛，因此按母模型规则停止仓位映射；不能用参数、窗口、标签或模型搜索救援本轮失败。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_structural_prediction(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state_receipt: dict[str, Any],
) -> dict[str, Any]:
    state = pd.read_parquet(
        project_path(config["artifacts"]["prediction_state_panel"])
    )
    total_return_path = project_path(
        config["source_contract"][
            "total_return_locked_until_prediction_state_freeze"
        ]["path"]
    )
    total_return = pd.read_parquet(total_return_path)
    return_horizons = [
        int(specification["horizon_market_days"])
        for specification in config["targets"].values()
        if specification["target_type"] == "CONTINUOUS_RETURN"
    ]
    drawdown = config["targets"]["20D_DRAWDOWN_PROBABILITY"]
    modeling = attach_price_features_and_targets(
        state,
        total_return,
        observation_cutoff=config["program"]["outcome_observation_cutoff"],
        return_horizons_market_days=return_horizons,
        drawdown_horizon_market_days=int(drawdown["horizon_market_days"]),
        drawdown_threshold=float(drawdown["threshold"]),
    )
    feature_groups = _feature_groups(config)
    predictions = walk_forward_predictions(
        modeling,
        target_specs=config["targets"],
        model_specs=config["models"],
        feature_groups=feature_groups,
        walk_forward=config["walk_forward"],
    )
    model_ids = list(config["models"].keys())
    metrics = evaluate_predictions(
        predictions,
        target_specs=config["targets"],
        model_ids=model_ids,
        sample_scopes=config["evaluation"]["sample_scopes"],
        reliability_gate=config["evaluation"]["reliability_gate"],
    )
    increments = build_increment_results(
        metrics,
        target_specs=config["targets"],
        primary_scope=config["evaluation"]["primary_scope"],
        module_comparisons=config["evaluation"]["module_comparisons"],
        final_model=config["evaluation"]["final_model"],
        final_baselines=config["evaluation"]["final_baselines"],
        minimum_loss_improvement=float(
            config["evaluation"]["incremental_loss_improvement_minimum"]
        ),
        continuous_auxiliary_gate=config["evaluation"][
            "continuous_auxiliary_gate"
        ],
        binary_auxiliary_gate=config["evaluation"]["binary_auxiliary_gate"],
    )
    validation = validation_summary(
        increments,
        module_comparison_ids=list(
            config["evaluation"]["module_comparisons"].keys()
        ),
        final_model=config["evaluation"]["final_model"],
        final_baselines=config["evaluation"]["final_baselines"],
        target_ids=list(config["targets"].keys()),
    )
    latest_forecasts = _latest_forecasts(predictions)

    prediction_path = project_path(config["artifacts"]["prediction_rows"])
    metrics_path = project_path(config["artifacts"]["evaluation_metrics"])
    increments_path = project_path(
        config["artifacts"]["module_increment_results"]
    )
    atomic_parquet(prediction_path, predictions)
    atomic_parquet(metrics_path, metrics)
    atomic_parquet(increments_path, increments)

    evaluation_payload = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "FROZEN_WALK_FORWARD_PREDICTION_EVALUATED",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "prediction_state_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "metrics": dataframe_records(metrics),
        "increment_results": dataframe_records(increments),
        "validation_summary": validation,
        "latest_forecasts": latest_forecasts,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    evaluation_payload["payload_sha256"] = canonical_hash(evaluation_payload)
    atomic_json(project_path(config["artifacts"]["evaluation_json"]), evaluation_payload)

    report = build_report(
        config,
        manifest,
        state_receipt,
        predictions,
        metrics,
        increments,
        validation,
        latest_forecasts,
    )
    atomic_text(project_path(config["artifacts"]["final_report"]), report)

    status_value = (
        "STAGE_3_COMPLETE_PREDICTIVE_VALIDATION_PASS_NO_PORTFOLIO"
        if validation["structural_prediction_validated"]
        else "STAGE_3_COMPLETE_PREDICTIVE_VALIDATION_FAILED_OR_NO_VIEW"
    )
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": status_value,
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "prediction_state_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "prediction_rows": {
            "path": prediction_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(prediction_path),
            "rows": int(len(predictions)),
            "generated_predictions": int(predictions["prediction"].notna().sum()),
            "generated_with_observed_actual": int(
                (
                    predictions["prediction"].notna()
                    & predictions["actual"].notna()
                ).sum()
            ),
        },
        "evaluation_metrics": {
            "path": metrics_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(metrics_path),
            "rows": int(len(metrics)),
            "reliability_counts": {
                str(key): int(value)
                for key, value in metrics["reliability_status"].value_counts().items()
            },
        },
        "module_increment_results": {
            "path": increments_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(increments_path),
            "rows": int(len(increments)),
            "result_counts": {
                str(key): int(value)
                for key, value in increments["result"].value_counts().items()
            },
        },
        "validation_summary": validation,
        "latest_forecasts": latest_forecasts,
        "next_stage_status": (
            "ELIGIBLE_FOR_SEPARATE_POSITION_PROTOCOL_NOT_AUTHORIZED"
            if validation["structural_prediction_validated"]
            else "STOP_NO_POSITION_MAPPING_STRUCTURE_MODEL_NOT_VALIDATED"
        ),
        "parent_limitations_propagated": config["limitations"],
        "strategy_validated": False,
        "portfolio_evaluation_allowed": False,
        "nav_calculated": False,
        "sharpe_calculated": False,
        "sharpe_target_1_2_achieved": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json(project_path(config["artifacts"]["authoritative_status"]), status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 结构预测 V1")
    parser.add_argument(
        "--phase",
        choices=["freeze", "build-states", "evaluate", "all"],
        default="all",
    )
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"build-states", "all"}:
        build_and_freeze_prediction_states(config, manifest)
    if args.phase in {"evaluate", "all"}:
        receipt = verify_prediction_state_freeze(config, manifest)
        evaluate_structural_prediction(config, manifest, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
