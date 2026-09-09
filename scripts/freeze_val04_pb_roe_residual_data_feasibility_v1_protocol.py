"""冻结 VAL-04 数据可行性协议；不读取未来收益。"""

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
    load_config,
    sha256_file,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/val04_pb_roe_residual_data_feasibility_v1.yaml",
    "docs/VAL04_PB_ROE_RESIDUAL_DATA_FEASIBILITY_V1_PROTOCOL.md",
    "research/val04_pb_roe_residual_data_feasibility_v1.py",
    "scripts/freeze_val04_pb_roe_residual_data_feasibility_v1_protocol.py",
    "scripts/audit_val04_pb_roe_residual_data_feasibility_v1.py",
    "tests/test_val04_pb_roe_residual_data_feasibility_v1.py",
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
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    input_audit = verify_input_hashes(config)
    if input_audit["status"] != "PASS":
        raise RuntimeError("输入哈希漂移，禁止冻结 VAL-04 数据协议")
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_DATA_FEASIBILITY_PROTOCOL_NO_RETURN_READING",
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
        "input_hash_audit": input_audit,
        "frozen_candidate_definition": config["frozen_candidate_definition"],
        "quality_gates": config["quality_gates"],
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
                "未来收益读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
