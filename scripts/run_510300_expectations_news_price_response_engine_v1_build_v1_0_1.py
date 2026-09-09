"""追加式修正：从第二阶段状态回执读取中性面板的冻结行数。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_510300_expectations_news_price_response_engine_v1 as base  # noqa: E402


PROGRAM_ID = "510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_ENGINE_V1"
VERSION = "1.0.1"
RUNNER_PATH = Path(__file__).resolve()
AMENDMENT_MANIFEST_PATH = (
    ROOT
    / "config/510300_expectations_news_price_response_engine_v1_manifest_v1_0_1.json"
)


def _load_receipt(config: dict[str, Any]) -> dict[str, Any]:
    relative = config["parent_contracts"]["conditional_map"][
        "condition_state_receipt"
    ]
    receipt = json.loads(base.project_path(relative).read_text(encoding="utf-8"))
    expected = config["parent_contracts"]["conditional_map"][
        "expected_receipt_payload_sha256"
    ]
    if receipt.get("receipt_payload_sha256") != expected:
        raise ValueError("第二阶段状态回执哈希不一致")
    state_metadata = receipt.get("condition_state_panel")
    if not isinstance(state_metadata, dict) or int(state_metadata.get("rows", 0)) <= 0:
        raise ValueError("第二阶段状态回执缺少有效的 condition_state_panel.rows")
    return receipt


def freeze_amendment(config: dict[str, Any]) -> dict[str, Any]:
    if AMENDMENT_MANIFEST_PATH.exists():
        raise RuntimeError(f"修正清单已存在，禁止静默覆盖：{AMENDMENT_MANIFEST_PATH}")
    base_manifest = base.verify_protocol(config)
    receipt = _load_receipt(config)
    manifest = {
        "program_id": PROGRAM_ID,
        "version": VERSION,
        "protocol_revision": "BUILD_ROW_COUNT_RECEIPT_LOOKUP_CORRECTION_ONLY",
        "status": "FROZEN_APPEND_ONLY_BUILD_CORRECTION",
        "frozen_at": base.now_iso(),
        "superseded_build_manifest_path": base.project_path(
            config["artifacts"]["protocol_manifest"]
        ).relative_to(ROOT).as_posix(),
        "superseded_build_manifest_payload_sha256": base_manifest[
            "manifest_payload_sha256"
        ],
        "superseded_build_status": "FROZEN_PROTOCOL_BUILD_ABORTED_BEFORE_OUTPUTS",
        "correction_scope": {
            "error": "CONDITION_STATE_ROW_COUNT_WAS_READ_FROM_STATUS_INSTEAD_OF_STATE_RECEIPT",
            "correct_source": config["parent_contracts"]["conditional_map"][
                "condition_state_receipt"
            ],
            "frozen_state_rows": int(receipt["condition_state_panel"]["rows"]),
            "research_definition_changed": False,
            "registered_inputs_changed": False,
            "neutral_label_mapping_changed": False,
        },
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): base.sha256_file(RUNNER_PATH),
            base.RUNNER_PATH.relative_to(ROOT).as_posix(): base.sha256_file(
                base.RUNNER_PATH
            ),
            base.MODULE_PATH.relative_to(ROOT).as_posix(): base.sha256_file(
                base.MODULE_PATH
            ),
        },
        "base_config_sha256": base.sha256_file(base.CONFIG_PATH),
        "base_config_canonical_sha256": base.canonical_hash(config),
        "stage_2_state_receipt_payload_sha256": receipt[
            "receipt_payload_sha256"
        ],
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    manifest["manifest_payload_sha256"] = base.canonical_hash(manifest)
    base.atomic_json_new(AMENDMENT_MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_amendment(config: dict[str, Any]) -> dict[str, Any]:
    if not AMENDMENT_MANIFEST_PATH.is_file():
        raise FileNotFoundError("V1.0.1 构建修正清单不存在")
    manifest = json.loads(AMENDMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if base.canonical_hash(payload) != expected_hash:
        raise ValueError("V1.0.1 构建修正清单自身哈希失败")
    base_manifest = base.verify_protocol(config)
    if base_manifest["manifest_payload_sha256"] != manifest[
        "superseded_build_manifest_payload_sha256"
    ]:
        raise ValueError("基础冻结清单与 V1.0.1 修正清单不一致")
    if base.sha256_file(base.CONFIG_PATH) != manifest["base_config_sha256"]:
        raise ValueError("基础配置在修正冻结后发生漂移")
    if base.canonical_hash(config) != manifest["base_config_canonical_sha256"]:
        raise ValueError("基础配置语义在修正冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if base.sha256_file(base.project_path(relative)) != expected:
            raise ValueError(f"V1.0.1 构建实现发生漂移：{relative}")
    receipt = _load_receipt(config)
    if receipt["receipt_payload_sha256"] != manifest[
        "stage_2_state_receipt_payload_sha256"
    ]:
        raise ValueError("第二阶段状态回执在修正冻结后发生漂移")
    return manifest


def build_with_correct_receipt_source(
    config: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    receipt = _load_receipt(config)
    original_validate = base.validate_parents

    def corrected_validate(
        current_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        conditional_status, prediction_status = original_validate(current_config)
        corrected_status = deepcopy(conditional_status)
        corrected_status["condition_state_panel"] = deepcopy(
            receipt["condition_state_panel"]
        )
        return corrected_status, prediction_status

    base.validate_parents = corrected_validate
    try:
        return base.build_architecture(config, amendment)
    finally:
        base.validate_parents = original_validate


def main() -> int:
    parser = argparse.ArgumentParser(description="运行测量架构 V1.0.1 追加式构建修正")
    parser.add_argument("--phase", choices=["freeze", "build", "all", "verify"], default="all")
    args = parser.parse_args()
    config = base.load_config()
    if args.phase == "freeze":
        freeze_amendment(config)
        return 0
    if args.phase == "all" and not AMENDMENT_MANIFEST_PATH.exists():
        freeze_amendment(config)
    amendment = verify_amendment(config)
    if args.phase in {"build", "all"}:
        build_with_correct_receipt_source(config, amendment)
    if args.phase == "verify":
        print("V1.0.1 构建修正清单及冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
