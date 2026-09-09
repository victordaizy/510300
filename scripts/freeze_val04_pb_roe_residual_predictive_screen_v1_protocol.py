"""在读取真实收益值前冻结 VAL-04 预测屏幕。"""

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

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)
from research.val04_pb_roe_residual_predictive_screen_v1 import (
    load_config,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/val04_pb_roe_residual_predictive_screen_v1.yaml",
    "docs/VAL04_PB_ROE_RESIDUAL_PREDICTIVE_SCREEN_V1_PROTOCOL.md",
    "research/val04_pb_roe_residual_predictive_screen_v1.py",
    "scripts/freeze_val04_pb_roe_residual_predictive_screen_v1_protocol.py",
    "scripts/run_val04_pb_roe_residual_predictive_screen_v1.py",
    "tests/test_val04_pb_roe_residual_predictive_screen_v1.py",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run_tests() -> dict[str, Any]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_val04_pb_roe_residual_predictive_screen_v1.py",
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
        raise RuntimeError(f"VAL-04 预测协议测试失败：\n{result.stdout}\n{result.stderr}")
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def main() -> int:
    config = load_config()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"VAL-04 预测冻结文件缺失：{missing}")
    if config["protocol"]["historical_strategy_return_calculation_enabled"]:
        raise RuntimeError("预测协议错误开启策略收益")
    gate = config["primary_predictive_gate"]
    if gate["familywise_hac_method"] != "HOLM_BONFERRONI":
        raise RuntimeError("预测显著性校正不是 Holm-Bonferroni")
    if float(gate["familywise_alpha"]) != 0.05 or int(gate["family_size"]) != 1:
        raise RuntimeError("VAL-04 家族显著性口径漂移")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("VAL-04 预测输入哈希漂移")
    tests = run_tests()
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_PREDICTIVE_SCREEN_PROTOCOL",
        "frozen_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "frozen_files": {
            relative: {
                "sha256": sha256_file(ROOT / relative),
                "size_bytes": (ROOT / relative).stat().st_size,
            }
            for relative in FROZEN_FILES
        },
        "input_hash_audit": audit,
        "primary_target": config["label_contract"]["primary_target"],
        "primary_horizon_trading_days": config["label_contract"][
            "primary_horizon_trading_days"
        ],
        "primary_predictive_gate": gate,
        "trial_and_multiplicity": config["trial_and_multiplicity"],
        "test_result": tests,
        "governance": {
            "real_label_values_read": False,
            "predictive_ic_calculated": False,
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
        },
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    output = ROOT / config["artifacts"]["protocol_manifest"]
    atomic_text(output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "状态": payload["status"],
                "协议清单": str(output),
                "内容哈希": payload["manifest_content_sha256"],
                "测试": tests,
                "真实收益值已读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
