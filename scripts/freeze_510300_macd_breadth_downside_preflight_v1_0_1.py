"""冻结仅修复广度预热缺口的V1_0_1协议。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from macd_breadth_downside_preflight_v1_0_1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_macd_breadth_downside_preflight_v1.yaml",
    "config/510300_macd_breadth_downside_preflight_v1_manifest.json",
    "research/macd_breadth_downside_preflight_v1.py",
    "config/510300_macd_breadth_downside_preflight_v1_0_1.yaml",
    "research/macd_breadth_downside_preflight_v1_0_1.py",
    "scripts/build_000300_official_weighted_breadth_warmup_v1_0_1.py",
    "scripts/freeze_510300_macd_breadth_downside_preflight_v1_0_1.py",
    "scripts/run_510300_macd_breadth_downside_preflight_v1_0_1.py",
    "tests/test_510300_macd_breadth_downside_preflight_v1_0_1.py",
]


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze() -> dict[str, Any]:
    config = load_config()
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"V1_0_1冻结清单已经存在，禁止覆盖：{MANIFEST_PATH}")
    result_paths = [
        _project_path(config["paths"][key])
        for key in ("result_json", "result_markdown", "feature_table", "event_table")
    ]
    preexisting = [path.relative_to(ROOT).as_posix() for path in result_paths if path.exists()]
    if preexisting:
        raise RuntimeError(f"V1_0_1冻结前已经存在结果：{preexisting}")

    tracked: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结文件：{relative}")
        tracked[relative] = sha256_file(path)

    inputs: dict[str, str] = {}
    for specification in config["inputs"].values():
        relative = specification["path"]
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        inputs[relative] = sha256_file(path)
    remediation_report = config["correction"]["input_replacement"][
        "remediation_report"
    ]
    inputs[remediation_report] = sha256_file(_project_path(remediation_report))

    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "state": "FROZEN_BEFORE_FIRST_RESULT",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
        "correction_boundaries": {
            "prior_candidate_outcomes_read": False,
            "evaluation_window_changed": False,
            "feature_definition_changed": False,
            "event_definition_changed": False,
            "direction_changed": False,
            "gate_changed": False,
            "bootstrap_changed": False,
            "only_input_warmup_remediated": True,
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)
    receipt_path = _project_path(config["paths"]["freeze_receipt"])
    receipt = {
        "project_id": manifest["project_id"],
        "status": "FROZEN_BEFORE_FIRST_RESULT",
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": manifest["config_sha256"],
        "tracked_file_count": len(tracked),
        "input_file_count": len(inputs),
        "result_preexisted_at_freeze": False,
        "prior_candidate_outcomes_read": False,
        "live_trading_authorized": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    receipt = freeze()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
