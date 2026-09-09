"""生成VAL01未来收益标签并执行冻结预测屏幕。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_raw_ey_5y_v1_forward_evaluation import (
    build_forward_labels,
    build_predictive_report,
    load_config,
    verify_input_hashes,
    write_artifacts,
)


def main() -> int:
    config = load_config()
    hashes = verify_input_hashes(config)
    if hashes["status"] != "PASS":
        raise RuntimeError(json.dumps(hashes, ensure_ascii=False, indent=2))

    def parquet(name: str) -> pd.DataFrame:
        return pd.read_parquet(ROOT / config["data_contracts"][name]["file"])

    dividends = pd.read_csv(ROOT / config["data_contracts"]["dividends"]["file"])
    coverage = json.loads(
        (ROOT / config["data_contracts"]["dividend_coverage"]["file"]).read_text(
            encoding="utf-8"
        )
    )
    signals = parquet("frozen_signal_inputs")
    labels, label_audit = build_forward_labels(
        signals,
        parquet("etf_market"),
        parquet("price_index"),
        parquet("total_return_index"),
        dividends,
        coverage,
        config,
    )
    if label_audit["status"] != "PASS":
        raise RuntimeError(json.dumps(label_audit, ensure_ascii=False, indent=2))
    report = build_predictive_report(signals, labels, label_audit, hashes, config)
    write_artifacts(labels, report, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
