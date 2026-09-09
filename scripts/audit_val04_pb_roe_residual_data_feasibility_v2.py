"""运行 VAL-04 申万点时行业 V2 全窗口数据闸门。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    build_feasibility,
    canonical_hash,
    load_config,
    sha256_file,
    verify_input_hashes,
    write_artifacts,
)


CONFIG_FILE = ROOT / "config" / "val04_pb_roe_residual_data_feasibility_v2.yaml"


def verify_protocol_manifest(config: dict) -> dict:
    path = ROOT / config["artifacts"]["protocol_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_content_sha256")
    actual = canonical_hash(payload)
    if actual != expected:
        raise RuntimeError("VAL-04 V2 协议清单自哈希不一致")
    if payload.get("status") != "FROZEN_DATA_FEASIBILITY_PROTOCOL_NO_RETURN_READING":
        raise RuntimeError("VAL-04 V2 数据协议尚未冻结")
    drift = []
    for relative, contract in payload["frozen_files"].items():
        target = ROOT / relative
        current = sha256_file(target) if target.is_file() else None
        if current != contract["sha256"]:
            drift.append(relative)
    if drift:
        raise RuntimeError(f"VAL-04 V2 冻结文件漂移：{drift}")
    return {
        "status": "PASS",
        "content_sha256": expected,
        "supplier_audit": payload["supplier_audit"],
    }


def main() -> int:
    config = load_config(CONFIG_FILE)
    protocol_audit = verify_protocol_manifest(config)
    input_audit = verify_input_hashes(config)
    if input_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 V2 输入哈希漂移")
    contracts = config["data_contracts"]

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(ROOT / contracts[name]["file"])

    industry = read("sw_industry_intervals")
    required_usage = config["industry_source_acceptance"][
        "required_classification_usage"
    ]
    if not industry["classification_usage"].eq(required_usage).all():
        raise RuntimeError("申万行业记录包含非点时分类")

    def progress(number: int, total: int, date: pd.Timestamp) -> None:
        if number == 1 or number % 12 == 0 or number == total:
            print(f"已完成 {number}/{total} 个 V2 点时快照：{date.date()}", flush=True)

    result = build_feasibility(
        read("historical_weights"),
        read("extended_financial_archive"),
        read("historical_snapshot_prices"),
        read("current_constituent_daily"),
        industry,
        config,
        progress=progress,
    )
    result.report["parent_v1_status"] = config["protocol"]["parent_result"]
    result.report["taxonomy"] = "SW_L1_POINT_IN_TIME"
    result.report["supplier_audit"] = protocol_audit["supplier_audit"]
    result.report["governance"]["parent_v1_files_mutated"] = False
    inventory = write_artifacts(result, config)
    print(
        json.dumps(
            {
                "状态": result.report["status"],
                "协议审计": protocol_audit,
                "窗口": result.report["window"],
                "覆盖摘要": result.report["coverage_summary"],
                "失败类别": result.report["failure_categories"],
                "产物": inventory,
                "治理": result.report["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
