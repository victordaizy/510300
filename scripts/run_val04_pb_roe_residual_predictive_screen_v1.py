"""运行冻结后的 VAL-04 预测屏幕。"""

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
from research.val04_pb_roe_residual_predictive_screen_v1 import (
    load_config,
    run_screen,
    verify_input_hashes,
    write_results,
)


def verify_protocol(config: dict) -> dict:
    path = ROOT / config["artifacts"]["protocol_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_content_sha256")
    if canonical_hash(payload) != expected:
        raise RuntimeError("VAL-04 预测协议自哈希不一致")
    if payload.get("status") != "FROZEN_PREDICTIVE_SCREEN_PROTOCOL":
        raise RuntimeError("VAL-04 预测协议未冻结")
    drift = []
    for relative, contract in payload["frozen_files"].items():
        target = ROOT / relative
        actual = sha256_file(target) if target.is_file() else None
        if actual != contract["sha256"]:
            drift.append(relative)
    if drift:
        raise RuntimeError(f"VAL-04 预测冻结文件漂移：{drift}")
    return {"status": "PASS", "content_sha256": expected}


def main() -> int:
    config = load_config()
    protocol_audit = verify_protocol(config)
    input_audit = verify_input_hashes(config)
    if input_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 预测输入哈希漂移")
    contracts = config["data_contracts"]
    signals = pd.read_parquet(ROOT / contracts["val04_signal_inputs"]["file"])
    labels = pd.read_parquet(ROOT / contracts["frozen_forward_labels"]["file"])
    model, family = run_screen(signals, labels, config)
    manifest = write_results(model, family, config)
    print(
        json.dumps(
            {
                "模型状态": model["status"],
                "家族状态": family["status"],
                "协议审计": protocol_audit,
                "主要闸门": model["primary_predictive_gate"],
                "242日ETF诊断": model["diagnostics"]["242"]["etf_total_return"],
                "242日H00300确认": model["diagnostics"]["242"]["h00300_total_return"],
                "结果清单": manifest,
                "治理": model["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
