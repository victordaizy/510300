"""在首次生成V2历史结果前冻结510300短周期识别精度前沿。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_accuracy_frontier_v2 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_binary_state_accuracy_frontier_v2.yaml",
    "docs/510300_BINARY_STATE_ACCURACY_FRONTIER_V2_SPEC.md",
    "research/binary_state_accuracy_frontier_v2.py",
    "scripts/run_510300_binary_state_accuracy_frontier_v2.py",
    "scripts/freeze_510300_binary_state_accuracy_frontier_v2.py",
    "tests/test_510300_binary_state_accuracy_frontier_v2.py",
    "config/510300_binary_state_feasibility_v1.yaml",
    "config/510300_binary_state_feasibility_v1_manifest.json",
    "research/binary_state_feasibility_v1.py",
]


def main() -> int:
    config = load_config(CONFIG_PATH)
    if MANIFEST_PATH.exists():
        raise FileExistsError("V2冻结清单已存在，禁止覆盖")
    artifacts = config["artifacts"]
    protected = [
        ROOT / artifacts["report_json"],
        ROOT / artifacts["report_markdown"],
        ROOT / artifacts["output_directory"],
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"V2历史产物已出现，禁止事后冻结：{existing}")
    source = config["source_contract"]
    data_paths = [
        source["etf_market"],
        source["benchmark_total_return"],
        source["cash_distributions"],
    ]
    data_files = {relative: sha256_file(ROOT / relative) for relative in data_paths}
    tracked_files = {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}
    manifest = {
        "schema_version": "2.0.0",
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "implementation_frozen": True,
        "historical_output_absent_at_freeze": True,
        "tracked_files": tracked_files,
        "data_files": data_files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "状态": "FROZEN_BEFORE_FIRST_V2_HISTORICAL_RESULT",
                "清单": str(MANIFEST_PATH.relative_to(ROOT)),
                "冻结文件数": len(tracked_files),
                "冻结数据文件数": len(data_files),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
