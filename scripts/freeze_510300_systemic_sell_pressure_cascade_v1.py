"""在首次读取候选收益前冻结全A股系统性卖压扩散V1。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.systemic_sell_pressure_cascade_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_systemic_sell_pressure_cascade_v1.yaml",
    "docs/510300_SYSTEMIC_SELL_PRESSURE_CASCADE_V1_SPEC.md",
    "research/systemic_sell_pressure_cascade_v1.py",
    "scripts/run_510300_systemic_sell_pressure_cascade_v1.py",
    "scripts/freeze_510300_systemic_sell_pressure_cascade_v1.py",
    "tests/test_510300_systemic_sell_pressure_cascade_v1.py",
    "config/510300_binary_state_feasibility_v1.yaml",
    "config/510300_binary_state_feasibility_v1_manifest.json",
    "research/binary_state_feasibility_v1.py",
    "config/510300_binary_state_accuracy_frontier_v2_manifest.json",
    "reports/research/510300_binary_state_accuracy_frontier_v2.json",
]


def main() -> int:
    config = load_config(CONFIG_PATH)
    if MANIFEST_PATH.exists():
        raise FileExistsError("候选冻结清单已存在，禁止覆盖")
    artifacts = config["artifacts"]
    protected = [
        ROOT / artifacts["report_json"],
        ROOT / artifacts["report_markdown"],
        ROOT / artifacts["output_directory"],
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"候选历史产物已出现，禁止事后冻结：{existing}")
    inputs = config["inputs"]
    data_paths = [
        inputs["visible_panel"],
        inputs["replication_panel"],
        inputs["stock_master"],
        inputs["etf_market"],
        inputs["benchmark_total_return"],
        inputs["cash_distributions"],
    ]
    data_files = {relative: sha256_file(ROOT / relative) for relative in data_paths}
    tracked_files = {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}
    manifest = {
        "schema_version": "1.0.0",
        "candidate_id": config["protocol"]["candidate_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "implementation_frozen": True,
        "historical_output_absent_at_freeze": True,
        "replication_panel_content_not_interpreted_by_freeze": True,
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
                "状态": "FROZEN_BEFORE_FIRST_VISIBLE_OUTCOME",
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
