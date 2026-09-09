"""冻结 VAL01/VAL02 正常化估值预测屏幕协议。"""

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

from research.normalized_valuation_5y_models_v1 import canonical_hash, sha256_file
from research.normalized_valuation_5y_predictive_screen_v1 import (
    load_config,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/normalized_valuation_5y_predictive_screen_v1.yaml",
    "docs/NORMALIZED_VALUATION_5Y_PREDICTIVE_SCREEN_V1_PROTOCOL.md",
    "research/normalized_valuation_5y_predictive_screen_v1.py",
    "scripts/run_normalized_valuation_5y_predictive_screen_v1.py",
    "scripts/freeze_normalized_valuation_5y_predictive_screen_v1_protocol.py",
    "tests/test_normalized_valuation_5y_predictive_screen_v1.py",
)


def run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_normalized_valuation_5y_predictive_screen_v1.py",
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
        raise RuntimeError(f"预测协议测试失败：\n{result.stdout}\n{result.stderr}")
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
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
        raise FileNotFoundError(f"预测协议文件缺失：{missing}")
    if config["protocol"]["historical_strategy_return_calculation_enabled"]:
        raise RuntimeError("预测协议错误开启策略收益")
    if config["protocol"]["historical_position_mapping_enabled"]:
        raise RuntimeError("预测协议错误开启历史仓位")
    if config["primary_predictive_gate"]["familywise_hac_method"] != "HOLM_BONFERRONI":
        raise RuntimeError("家族校正方法不是Holm-Bonferroni")
    if float(config["primary_predictive_gate"]["familywise_alpha"]) != 0.10:
        raise RuntimeError("家族显著性水平不是0.10")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("预测协议输入哈希不一致")
    tests = run_tests()
    payload: dict[str, object] = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_PREDICTIVE_SCREEN_PROTOCOL",
        "frozen_at": datetime.now(ZoneInfo(config["protocol"]["timezone"])).isoformat(),
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "input_hash_audit": audit,
        "familywise_gate": {
            "method": "HOLM_BONFERRONI",
            "alpha": 0.10,
            "family_size": 2,
        },
        "test_result": tests,
        "governance": {
            "real_label_values_read": False,
            "predictive_ic_calculated": False,
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
        },
        "meaning": "冻结预测屏幕实现和家族校正规则；尚未读取真实收益或计算IC。",
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["protocol_manifest"]
    atomic_json(manifest, payload)
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
