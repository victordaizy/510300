"""运行 VAL01/VAL02 五年正常化估值无收益信号构建。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.normalized_valuation_5y_models_v1 import (
    build_common_panel,
    build_model_report,
    load_config,
    project_model_signals,
    verify_input_hashes,
    write_model_manifests,
    write_signal_artifacts,
)


def main() -> int:
    config = load_config()
    protocol_manifest = ROOT / config["artifacts"]["protocol_manifest"]
    protocol = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_SIGNAL_PROTOCOL_NO_RETURN_EVALUATION":
        raise RuntimeError("信号协议未冻结，禁止运行真实输入")
    audit = verify_input_hashes(config)
    if audit["status"] != "PASS":
        raise RuntimeError("信号输入哈希漂移")
    contracts = config["data_contracts"]
    read = lambda name: pd.read_parquet(ROOT / contracts[name]["file"])
    common = build_common_panel(
        read("historical_weights"),
        read("extended_financial_archive"),
        read("snapshot_prices"),
        read("current_constituent_daily"),
        read("index_daily"),
        read("government_bond_yields"),
        read("post_acquisition_coverage"),
        config,
    )
    val01 = project_model_signals(common, "VAL01_NORM_EY_5Y_V1", config)
    val02 = project_model_signals(common, "VAL02_NORM_EY_SPREAD_5Y_V1", config)
    official = read("official_pe_cross_check")
    vendor = read("vendor_pe_cross_check")
    val01_report = build_model_report(
        val01, "VAL01_NORM_EY_5Y_V1", config, audit, official, vendor
    )
    val02_report = build_model_report(
        val02, "VAL02_NORM_EY_SPREAD_5Y_V1", config, audit, official, vendor
    )
    inventory = write_signal_artifacts(
        common, val01, val02, val01_report, val02_report, config
    )
    manifests = write_model_manifests(
        inventory, val01_report, val02_report, config
    )
    print(
        json.dumps(
            {
                "val01_status": val01_report["status"],
                "val02_status": val02_report["status"],
                "val01_summary": val01_report["signal_summary"],
                "val02_summary": val02_report["signal_summary"],
                "manifests": manifests,
                "governance": {
                    "future_returns": False,
                    "ic": False,
                    "positions": False,
                    "orders": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
