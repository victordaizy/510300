"""运行 VAL-04 全 120 月无收益数据闸门。"""

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


def verify_protocol_manifest(config: dict) -> dict:
    path = ROOT / config["artifacts"]["protocol_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_content_sha256")
    actual = canonical_hash(payload)
    if actual != expected:
        raise RuntimeError("VAL-04 数据协议清单自哈希不一致")
    if payload.get("status") != "FROZEN_DATA_FEASIBILITY_PROTOCOL_NO_RETURN_READING":
        raise RuntimeError("VAL-04 数据协议尚未冻结")
    drift = []
    for relative, contract in payload["frozen_files"].items():
        path_to_file = ROOT / relative
        current = sha256_file(path_to_file) if path_to_file.is_file() else None
        if current != contract["sha256"]:
            drift.append(relative)
    if drift:
        raise RuntimeError(f"VAL-04 冻结协议文件漂移：{drift}")
    return {"status": "PASS", "content_sha256": expected}


def main() -> int:
    config = load_config()
    protocol_audit = verify_protocol_manifest(config)
    input_audit = verify_input_hashes(config)
    if input_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 输入哈希漂移，输出 NO_VIEW 前停止")
    contracts = config["data_contracts"]

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(ROOT / contracts[name]["file"])

    def progress(number: int, total: int, date: pd.Timestamp) -> None:
        if number == 1 or number % 12 == 0 or number == total:
            print(f"已完成 {number}/{total} 个点时快照：{date.date()}", flush=True)

    result = build_feasibility(
        read("historical_weights"),
        read("extended_financial_archive"),
        read("historical_snapshot_prices"),
        read("current_constituent_daily"),
        read("citic_industry_intervals"),
        config,
        progress=progress,
    )
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
