"""运行 VAL01/VAL02 正常化估值预测屏幕。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.normalized_valuation_5y_predictive_screen_v1 import (
    audit_frozen_labels,
    build_family_reports,
    compute_model_diagnostics,
    load_config,
    verify_input_hashes,
    write_reports_and_manifest,
)


def main() -> int:
    config = load_config()
    protocol_manifest = ROOT / config["artifacts"]["protocol_manifest"]
    protocol = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREDICTIVE_SCREEN_PROTOCOL":
        raise RuntimeError("预测屏幕协议未冻结，禁止读取真实收益标签")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("预测屏幕输入哈希漂移")
    contracts = config["data_contracts"]
    labels = pd.read_parquet(ROOT / contracts["frozen_forward_labels"]["file"])
    label_audit = audit_frozen_labels(labels, config)
    if label_audit["status"] != "PASS":
        raise RuntimeError("冻结标签表结构或成熟数量不一致")
    model_diagnostics = {}
    for model_id, signal_contract in (
        ("VAL01_NORM_EY_5Y_V1", "val01_signal_inputs"),
        ("VAL02_NORM_EY_SPREAD_5Y_V1", "val02_signal_inputs"),
    ):
        signals = pd.read_parquet(ROOT / contracts[signal_contract]["file"])
        model_diagnostics[model_id] = compute_model_diagnostics(
            signals, labels, model_id, config
        )
    reports, family = build_family_reports(
        model_diagnostics, label_audit, audit, config
    )
    manifest = write_reports_and_manifest(reports, family, config)
    print(
        json.dumps(
            {
                "family_status": family["status"],
                "model_status": family["model_status"],
                "holm_bonferroni": family["holm_bonferroni"],
                "primary_gates": {
                    model: report["primary_predictive_gate"]
                    for model, report in reports.items()
                },
                "result_manifest": manifest,
                "governance": family["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
