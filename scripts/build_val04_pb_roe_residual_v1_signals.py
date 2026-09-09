"""按冻结协议构建 VAL-04 无收益信号。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)
from research.val04_pb_roe_residual_v1 import (
    build_val04_signal,
    load_config,
    write_artifacts,
)


def verify_protocol(config: dict) -> dict:
    path = ROOT / config["artifacts"]["protocol_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_content_sha256")
    if canonical_hash(payload) != expected:
        raise RuntimeError("VAL-04 模型协议清单自哈希不一致")
    if payload.get("status") != "FROZEN_SIGNAL_PROTOCOL_NO_RETURN_EVALUATION":
        raise RuntimeError("VAL-04 信号协议未冻结")
    drift = []
    for relative, contract in payload["frozen_files"].items():
        target = ROOT / relative
        actual = sha256_file(target) if target.is_file() else None
        if actual != contract["sha256"]:
            drift.append(relative)
    if drift:
        raise RuntimeError(f"VAL-04 冻结信号文件漂移：{drift}")
    return {"status": "PASS", "content_sha256": expected}


def main() -> int:
    config = load_config()
    protocol_audit = verify_protocol(config)
    contracts = config["data_contracts"]
    result = build_val04_signal(
        pd.read_parquet(ROOT / contracts["constituent_feasibility_panel"]["file"]),
        pd.read_parquet(ROOT / contracts["snapshot_feasibility_audit"]["file"]),
        pd.read_parquet(ROOT / contracts["index_daily_calendar"]["file"]),
        config,
    )
    artifacts = write_artifacts(result, config)
    print(
        json.dumps(
            {
                "状态": result.report["status"],
                "协议审计": protocol_audit,
                "信号摘要": result.report["signal_summary"],
                "失败检查": result.report["failed_checks"],
                "产物": artifacts,
                "治理": result.report["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
