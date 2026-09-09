"""冻结 VAL-04 唯一模型卡与无收益信号协议。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)
from research.val04_pb_roe_residual_v1 import load_config, verify_input_hashes


FROZEN_FILES = (
    "config/val04_pb_roe_residual_v1.yaml",
    "docs/VAL04_PB_ROE_RESIDUAL_V1_MODEL_CARD.md",
    "research/val04_pb_roe_residual_v1.py",
    "scripts/freeze_val04_pb_roe_residual_v1_protocol.py",
    "scripts/build_val04_pb_roe_residual_v1_signals.py",
    "tests/test_val04_pb_roe_residual_v1.py",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    config = load_config()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"VAL-04 模型冻结文件缺失：{missing}")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("VAL-04 模型输入哈希漂移")
    feasibility_manifest = json.loads(
        (
            ROOT
            / config["data_contracts"]["feasibility_v2_result_manifest"]["file"]
        ).read_text(encoding="utf-8")
    )
    if feasibility_manifest.get("status") != "PASS_MODEL_CARD_MAY_BE_FROZEN":
        raise RuntimeError("VAL-04 V2 数据闸门未授权模型冻结")
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_SIGNAL_PROTOCOL_NO_RETURN_EVALUATION",
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
        "cross_sectional_model": config["cross_sectional_model"],
        "index_signal": config["index_signal"],
        "percentile_definition": config["percentile_definition"],
        "trial_registration": config["trial_registration"],
        "governance": config["governance"],
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    output = ROOT / config["artifacts"]["protocol_manifest"]
    atomic_text(output, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "状态": manifest["status"],
                "协议清单": str(output),
                "内容哈希": manifest["manifest_content_sha256"],
                "注册试验": config["protocol"]["registered_trial_id"],
                "未来收益读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
