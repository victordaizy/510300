"""冻结 VAL01/VAL02 正常化估值模型卡与无收益信号协议。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.normalized_valuation_5y_models_v1 import (
    canonical_hash,
    load_config,
    sha256_file,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/normalized_valuation_5y_models_v1.yaml",
    "docs/VAL01_NORM_EY_5Y_V1_MODEL_CARD.md",
    "docs/VAL02_NORM_EY_SPREAD_5Y_V1_MODEL_CARD.md",
    "research/normalized_valuation_5y_models_v1.py",
    "scripts/build_normalized_valuation_5y_signals_v1.py",
    "scripts/freeze_normalized_valuation_5y_models_v1_protocol.py",
    "tests/test_normalized_valuation_5y_models_v1.py",
)


def _run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_normalized_valuation_5y_models_v1.py",
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"协议冻结测试失败：\n{result.stdout}\n{result.stderr}")
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    config = load_config()
    missing = [path for path in FROZEN_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"协议文件缺失：{missing}")
    for prohibited in (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    ):
        if config["protocol"].get(prohibited) is not False:
            raise RuntimeError(f"协议治理开关未关闭：{prohibited}")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("协议冻结输入哈希不一致")
    tests = _run_tests()
    payload: dict[str, object] = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_SIGNAL_PROTOCOL_NO_RETURN_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo(config["protocol"]["timezone"])).isoformat(),
        "registered_trials": config["trial_registration"]["registered_trials_consumed"],
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "input_hash_audit": audit,
        "test_result": tests,
        "governance": {
            "future_return_labels_generated": False,
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "raw_ey_rejected_branch_reopened": False,
        },
        "meaning": "冻结两个独立模型卡与无收益信号实现；不表示信号输入或预测屏幕已经通过。",
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["protocol_manifest"]
    _atomic_json(manifest, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "manifest": manifest.relative_to(ROOT).as_posix(),
                "manifest_content_sha256": payload["manifest_content_sha256"],
                "test_result": tests,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
