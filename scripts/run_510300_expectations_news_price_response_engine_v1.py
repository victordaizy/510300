"""冻结并建立 510300 预期—消息—价格响应测量架构。"""

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

from research.expectations_news_price_response_engine_v1 import (  # noqa: E402
    build_neutral_atlas,
    build_neutral_state_panel,
    build_permanent_rejection_receipt,
)


PROGRAM_ID = "510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_ENGINE_V1"
CONFIG_PATH = ROOT / "config/510300_expectations_news_price_response_engine_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/expectations_news_price_response_engine_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("测量架构配置的 PROGRAM_ID 不匹配")
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


def _validate_conditional_parent(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(project_path(contract["manifest"]).read_text(encoding="utf-8"))
    status = json.loads(
        project_path(contract["authoritative_status"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        project_path(contract["condition_state_receipt"]).read_text(encoding="utf-8")
    )
    checks = (
        (
            manifest.get("manifest_payload_sha256"),
            contract["expected_manifest_payload_sha256"],
            "第二阶段协议哈希",
        ),
        (
            status.get("status_payload_sha256"),
            contract["expected_status_payload_sha256"],
            "第二阶段状态哈希",
        ),
        (
            receipt.get("receipt_payload_sha256"),
            contract["expected_receipt_payload_sha256"],
            "第二阶段状态回执哈希",
        ),
        (status.get("status"), contract["expected_status"], "第二阶段状态"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"{label}与测量架构冻结配置不一致")
    file_checks = (
        (
            contract["condition_state_panel"],
            contract["expected_state_panel_sha256"],
            "第二阶段状态面板",
        ),
        (
            contract["condition_map"],
            contract["expected_condition_map_sha256"],
            "第二阶段条件统计",
        ),
    )
    for relative, expected, label in file_checks:
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"{label}文件哈希不一致")
    if status.get("portfolio_evaluation_allowed") is not False:
        raise ValueError("第二阶段意外开放组合评估")
    return manifest, status, receipt


def _validate_prediction_parent(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(project_path(contract["manifest"]).read_text(encoding="utf-8"))
    status = json.loads(
        project_path(contract["authoritative_status"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        project_path(contract["prediction_state_receipt"]).read_text(encoding="utf-8")
    )
    checks = (
        (
            manifest.get("manifest_payload_sha256"),
            contract["expected_manifest_payload_sha256"],
            "第三阶段协议哈希",
        ),
        (
            status.get("status_payload_sha256"),
            contract["expected_status_payload_sha256"],
            "第三阶段状态哈希",
        ),
        (
            receipt.get("receipt_payload_sha256"),
            contract["expected_receipt_payload_sha256"],
            "第三阶段状态回执哈希",
        ),
        (status.get("status"), contract["expected_status"], "第三阶段状态"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"{label}与永久拒绝配置不一致")
    file_checks = (
        (
            contract["prediction_rows"],
            contract["expected_prediction_rows_sha256"],
            "第三阶段预测记录",
        ),
        (
            contract["evaluation_metrics"],
            contract["expected_evaluation_metrics_sha256"],
            "第三阶段评估指标",
        ),
        (
            contract["module_increment_results"],
            contract["expected_increment_results_sha256"],
            "第三阶段模块增量结果",
        ),
    )
    for relative, expected, label in file_checks:
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"{label}文件哈希不一致")
    if status.get("strategy_validated") is not False:
        raise ValueError("第三阶段状态意外宣称策略通过")
    if status.get("portfolio_evaluation_allowed") is not False:
        raise ValueError("第三阶段意外开放组合评估")
    return manifest, status, receipt


def validate_parents(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, conditional_status, _ = _validate_conditional_parent(
        config["parent_contracts"]["conditional_map"]
    )
    _, prediction_status, _ = _validate_prediction_parent(
        config["parent_contracts"]["rejected_prediction"]
    )
    return conditional_status, prediction_status


def _registered_inputs(config: dict[str, Any]) -> dict[str, str]:
    conditional = config["parent_contracts"]["conditional_map"]
    prediction = config["parent_contracts"]["rejected_prediction"]
    return {
        conditional["manifest"]: "FROZEN_STAGE_2_PROTOCOL",
        conditional["authoritative_status"]: "FROZEN_STAGE_2_STATUS",
        conditional["condition_state_receipt"]: "FROZEN_STAGE_2_STATE_RECEIPT",
        conditional["condition_state_panel"]: "SOURCE_FOR_NEUTRAL_STATE_REEXPRESSION",
        conditional["condition_map"]: "SOURCE_FOR_NEUTRAL_DESCRIPTIVE_ATLAS",
        prediction["manifest"]: "FROZEN_REJECTED_STAGE_3_PROTOCOL",
        prediction["authoritative_status"]: "FROZEN_REJECTED_STAGE_3_STATUS",
        prediction["prediction_state_receipt"]: "FROZEN_STAGE_3_STATE_RECEIPT",
        prediction["prediction_rows"]: "IMMUTABLE_DIAGNOSTIC_FORECAST_LEDGER",
        prediction["evaluation_metrics"]: "REJECTED_MODEL_DIAGNOSTIC_EVIDENCE",
        prediction["module_increment_results"]: "REJECTED_INCREMENT_EVIDENCE",
    }


def _output_paths(config: dict[str, Any]) -> list[Path]:
    artifacts = config["artifacts"]
    return [
        project_path(value)
        for key, value in artifacts.items()
        if key != "protocol_manifest"
    ]


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"冻结清单已存在，禁止静默覆盖：{manifest_path}")
    existing_outputs = [str(path) for path in _output_paths(config) if path.exists()]
    if existing_outputs:
        raise RuntimeError(f"冻结前发现同名输出，拒绝覆盖：{existing_outputs}")
    conditional_status, prediction_status = validate_parents(config)
    registered: dict[str, Any] = {}
    for relative, role in _registered_inputs(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"测量架构冻结输入缺失：{path}")
        registered[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_MEASUREMENT_ARCHITECTURE_BUILD",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered,
        "stage_2_status_payload_sha256": conditional_status[
            "status_payload_sha256"
        ],
        "stage_3_status_payload_sha256": prediction_status[
            "status_payload_sha256"
        ],
        "current_forecast_specification": "REJECTED_FROZEN",
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
        raise FileNotFoundError("测量架构协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_hash(payload) != expected_hash:
        raise ValueError("测量架构协议清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("测量架构配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("测量架构配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"测量架构实现文件发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"测量架构冻结输入发生漂移或缺失：{relative}")
    validate_parents(config)
    return manifest


def _build_forward_schema(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    monitoring = config["forward_monitoring"]
    return {
        "program_id": PROGRAM_ID,
        "schema_id": "510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_FORWARD_LEDGER_SCHEMA_V1",
        "schema_version": "1.0.0",
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "start_date": str(monitoring["start_date"]),
        "mode": monitoring["mode"],
        "append_only": True,
        "amendment_policy": "NEW_VERSIONED_RECORD_ONLY_NO_IN_PLACE_MUTATION",
        "record_key": ["recorded_at", "origin", "object_id", "vintage_id"],
        "required_fields": [
            "recorded_at",
            "origin",
            "object_id",
            "module_id",
            "measurement_value",
            "measurement_unit",
            "vintage_id",
            "source_path_or_url",
            "source_sha256_or_version",
            "availability_time",
            "clock_status",
            "label_maturity_time",
            "label_status",
            "diagnostic_status",
        ],
        "allowed_module_ids": ["CF", "DR", "RC", "PRICE_RESPONSE"],
        "initial_record_count": 0,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
    }


def _build_atlas_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state: pd.DataFrame,
    atlas: pd.DataFrame,
) -> str:
    quality_counts = state["state_quality_status"].value_counts().to_dict()
    quadrant_counts = state["state_quadrant"].value_counts().to_dict()
    reliability_counts = atlas["reliability_status"].value_counts().to_dict()
    latest = state.iloc[-1]
    complete = state.loc[
        state["state_evaluation_eligible"].fillna(False)
        & state["origin_completion_status"].eq(
            "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"
        )
    ].iloc[-1]
    return "\n".join(
        [
            "# 510300_CONDITIONAL_STATE_ATLAS_V1",
            "",
            "## 裁决",
            "",
            "本产物是第二阶段冻结状态的中性重表达，只用于历史状态描述。它不是收益预测、因果识别、择时规则或组合授权。",
            "",
            f"- 协议哈希：`{manifest['manifest_payload_sha256']}`",
            f"- 状态面板：`{len(state)}` 个 origin；Atlas：`{len(atlas)}` 行。",
            f"- 状态质量计数：`{json.dumps(quality_counts, ensure_ascii=False, sort_keys=True)}`",
            f"- 中性象限计数：`{json.dumps(quadrant_counts, ensure_ascii=False, sort_keys=True)}`",
            f"- 描述可靠性计数：`{json.dumps(reliability_counts, ensure_ascii=False, sort_keys=True)}`",
            "- 旧 archetype 视图、方向预设字段及方向裁决字段均未进入本 Atlas。",
            "",
            "## 中性标签",
            "",
            "- `CF_UP_DR_EASING`",
            "- `CF_UP_DR_TIGHTENING`",
            "- `CF_DOWN_DR_EASING`",
            "- `CF_DOWN_DR_TIGHTENING`",
            "",
            "这些标签只编码相邻冻结 origin 之间的测量变化，不包含未来收益方向。",
            "",
            "## 最新状态",
            "",
            f"- 最新记录：`{latest['origin'].date()}`，`{latest['state_quadrant']}`，`{latest['rc_direction']}`，质量为 `{latest['state_quality_status']}`。",
            f"- 最新完整且可描述记录：`{complete['origin'].date()}`，`{complete['state_quadrant']}`，`{complete['rc_direction']}`。",
            "- `RETURN_PREDICTION_ALLOWED=false`",
            "- `PORTFOLIO_EVALUATION_ALLOWED=false`",
            "- `MODEL_POSITION_TARGET=UNSET`",
            "",
            "## 使用边界",
            "",
            "历史收益统计仅是已发生结果的条件描述。重叠 60D/120D 窗口不独立，不能被重述为预测胜率、交易信号或夏普率证据。",
            "",
        ]
    )


def _build_engine_charter(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_ENGINE_V1 — 冻结章程",
            "",
            "## 当前裁决",
            "",
            "- `CURRENT_FORECAST_SPECIFICATION=REJECTED_FROZEN`",
            "- `WAITING_FOR_EXISTING_LABELS=PASSIVE_MONITORING_ONLY`",
            "- `NEXT_RESEARCH_ACTION=REBUILD_MEASUREMENT_ARCHITECTURE`",
            "- `PORTFOLIO_ACTION=ABSTAIN`",
            "- `POSITION_STATE=POSITION_UNSET`",
            "- 特定 V1 预测实现永久冻结拒绝；结构经济想法仍是未证明，而不是已证明数学不可能。",
            "- V1 不能因后续标签变化恢复资格；任何重启都必须注册为 V2。",
            "",
            "## 研究权限",
            "",
            "本引擎只做测量与机制辨识。`RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`，`MODEL_POSITION_TARGET=UNSET`。",
            "",
            "## 固定执行顺序",
            "",
            "1. 完成功效与可识别性审计，量化 1%、2%、5% MSE 改善所需的独立周期、月度 origin 与年份。",
            "2. 建立 60D/120D 总回报成分账本；先证明会计恒等式，再讨论任何模块解释力。",
            "3. CF 仅面向盈利、经营现金流及其 breadth；DR 仅面向估值倍数与 ERP gap；RC 仅面向波动、下行半方差、相关性和流动性冲击。",
            "4. 价格响应模块只测量消息吸收路径，不预测总回报。",
            "5. 只有各自对象的 PIT 测量有效性通过后，才允许另行冻结新一代预测协议。",
            "",
            "## 总回报成分账本",
            "",
            "账本必须同时给出实际指数链式口径和 origin 固定成分股/权重口径。目标恒等式为：",
            "",
            "`TOTAL_RETURN_FACTOR = DIVIDEND_FACTOR × EARNINGS_FACTOR × MULTIPLE_FACTOR × MEMBERSHIP_WEIGHT_FACTOR × TRACKING_FACTOR`",
            "",
            "每一项都必须有来源、可用时钟、覆盖率、状态和残差。没有合格的成分股分红档案时，纯 `DIVIDEND_COMPONENT` 必须为 `NO_VIEW`，不得用公司行动差额冒充。",
            "",
            "## PIT 硬门槛",
            "",
            "- 历史成分与权重版本必须在 origin 当时可用。",
            "- 财务事实使用首次公开披露时间；后修订值不得冒充 PIT。",
            "- 事件与价格反应必须冻结盘前、盘中、盘后时钟。",
            "- 同日/相邻事件必须聚类，避免把同一信息冲击重复计数。",
            "- 任一关键时钟不可验证即输出 `NO_VIEW`。",
            "",
            "## 2026-09 起前向记录",
            "",
            "只允许追加不可变诊断记录；原记录不得原位修改。标签成熟只更新诊断账本，不恢复 V1、不生成仓位、不进入 Paper/Shadow 或实盘。",
            "",
            f"协议清单哈希：`{manifest['manifest_payload_sha256']}`。",
            "",
        ]
    )


def build_architecture(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    conditional_status, prediction_status = validate_parents(config)
    conditional_contract = config["parent_contracts"]["conditional_map"]
    parent_state = pd.read_parquet(project_path(conditional_contract["condition_state_panel"]))
    parent_atlas = pd.read_parquet(project_path(conditional_contract["condition_map"]))
    atlas_config = config["conditional_state_atlas"]
    state = build_neutral_state_panel(
        parent_state,
        quadrant_map=atlas_config["neutral_quadrant_labels"],
    )
    atlas = build_neutral_atlas(
        parent_atlas,
        canonical_source_views=atlas_config["canonical_source_views"],
        quadrant_map=atlas_config["neutral_quadrant_labels"],
    )
    if len(state) != int(conditional_status["condition_state_panel"]["rows"]):
        raise ValueError("中性状态面板行数与冻结第二阶段不一致")
    expected_atlas_rows = int(
        parent_atlas["view_type"].isin(atlas_config["canonical_source_views"]).sum()
    )
    if len(atlas) != expected_atlas_rows:
        raise ValueError("中性 Atlas 行数与注册视图不一致")

    rejection = build_permanent_rejection_receipt(
        prediction_status,
        config["adjudication"],
    )
    rejection["created_at"] = now_iso()
    rejection["measurement_engine_manifest_payload_sha256"] = manifest[
        "manifest_payload_sha256"
    ]
    rejection["receipt_payload_sha256"] = canonical_hash(rejection)

    atlas_json = {
        "program_id": "510300_CONDITIONAL_STATE_ATLAS_V1",
        "version": "1.0.0",
        "status": "NEUTRAL_DESCRIPTIVE_STATE_ATLAS_FROZEN",
        "created_at": now_iso(),
        "measurement_engine_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "source_stage_2_status_payload_sha256": conditional_status[
            "status_payload_sha256"
        ],
        "state_rows": len(state),
        "atlas_rows": len(atlas),
        "source_archetype_rows_excluded": int(
            parent_atlas["view_type"].eq("ARCHETYPE").sum()
        ),
        "neutral_quadrant_labels": list(
            atlas_config["neutral_quadrant_labels"].values()
        ),
        "state_quality_counts": {
            str(key): int(value)
            for key, value in state["state_quality_status"].value_counts().items()
        },
        "atlas_reliability_counts": {
            str(key): int(value)
            for key, value in atlas["reliability_status"].value_counts().items()
        },
        "descriptive_statistics": dataframe_records(atlas),
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    atlas_json["payload_sha256"] = canonical_hash(atlas_json)
    forward_schema = _build_forward_schema(config, manifest)
    forward_schema["schema_payload_sha256"] = canonical_hash(forward_schema)
    atlas_report = _build_atlas_report(config, manifest, state, atlas)
    engine_charter = _build_engine_charter(config, manifest)

    artifacts = config["artifacts"]
    paths = {name: project_path(relative) for name, relative in artifacts.items()}
    pending = [path for name, path in paths.items() if name != "protocol_manifest" and path.exists()]
    if pending:
        raise RuntimeError(f"测量架构输出已存在，禁止覆盖：{pending}")
    atomic_json_new(paths["permanent_rejection_receipt"], rejection)
    atomic_parquet_new(paths["neutral_state_panel"], state)
    atomic_parquet_new(paths["neutral_atlas"], atlas)
    atomic_json_new(paths["neutral_atlas_json"], atlas_json)
    atomic_text_new(paths["neutral_atlas_report"], atlas_report)
    atomic_text_new(paths["engine_charter"], engine_charter)
    atomic_json_new(paths["forward_ledger_schema"], forward_schema)

    latest = state.iloc[-1]
    latest_complete = state.loc[
        state["state_evaluation_eligible"].fillna(False)
        & state["origin_completion_status"].eq(
            "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"
        )
    ].iloc[-1]
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "MEASUREMENT_ARCHITECTURE_ACTIVE_NO_RETURN_PREDICTION",
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "current_forecast_specification": "REJECTED_FROZEN",
        "waiting_for_existing_labels": "PASSIVE_MONITORING_ONLY",
        "next_research_action": "REBUILD_MEASUREMENT_ARCHITECTURE",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "permanent_rejection_receipt": {
            "path": paths["permanent_rejection_receipt"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["permanent_rejection_receipt"]),
            "payload_sha256": rejection["receipt_payload_sha256"],
            "immutable_diagnostic_forecast_count": rejection[
                "immutable_diagnostic_forecast_count"
            ],
        },
        "conditional_state_atlas": {
            "program_id": "510300_CONDITIONAL_STATE_ATLAS_V1",
            "status": "NEUTRAL_DESCRIPTIVE_STATE_ATLAS_FROZEN",
            "state_panel_path": paths["neutral_state_panel"].relative_to(ROOT).as_posix(),
            "state_panel_sha256": sha256_file(paths["neutral_state_panel"]),
            "state_rows": len(state),
            "atlas_path": paths["neutral_atlas"].relative_to(ROOT).as_posix(),
            "atlas_sha256": sha256_file(paths["neutral_atlas"]),
            "atlas_rows": len(atlas),
            "source_archetype_rows_excluded": int(
                parent_atlas["view_type"].eq("ARCHETYPE").sum()
            ),
            "latest_state": {
                "origin": latest["origin"].date().isoformat(),
                "state_quadrant": latest["state_quadrant"],
                "rc_direction": latest["rc_direction"],
                "state_quality_status": latest["state_quality_status"],
            },
            "latest_complete_eligible_state": {
                "origin": latest_complete["origin"].date().isoformat(),
                "state_quadrant": latest_complete["state_quadrant"],
                "rc_direction": latest_complete["rc_direction"],
                "state_quality_status": latest_complete["state_quality_status"],
            },
        },
        "engine_charter": {
            "path": paths["engine_charter"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["engine_charter"]),
        },
        "forward_ledger_schema": {
            "path": paths["forward_ledger_schema"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["forward_ledger_schema"]),
            "start_date": str(config["forward_monitoring"]["start_date"]),
            "initial_record_count": 0,
        },
        "workstream_status": {
            "510300_STRUCTURAL_SIGNAL_POWER_AND_IDENTIFIABILITY_AUDIT_V1": "NEXT_FROZEN_WORKSTREAM",
            "510300_TOTAL_RETURN_COMPONENT_LEDGER_V1": "PENDING_POWER_AUDIT",
            "CF_OWN_OBJECT_MEASUREMENT_VALIDITY": "PENDING_LEDGER",
            "DR_OWN_OBJECT_MEASUREMENT_VALIDITY": "PENDING_LEDGER",
            "RC_OWN_OBJECT_MEASUREMENT_VALIDITY": "PENDING_LEDGER",
            "EXPECTATION_NEWS_PRICE_RESPONSE_CHAIN": "PENDING_OWN_OBJECT_VALIDITY",
        },
        "mathematical_impossibility_proven": False,
        "accessible_free_information_feasibility_validated": False,
        "current_investable_strategy": "NONE",
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json_new(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 测量架构冻结与构建")
    parser.add_argument("--phase", choices=["freeze", "build", "all", "verify"], default="all")
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"build", "all"}:
        build_architecture(config, manifest)
    if args.phase == "verify":
        print("测量架构协议与全部冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
