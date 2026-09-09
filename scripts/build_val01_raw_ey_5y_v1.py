"""运行VAL01_RAW_EY_5Y_V1无收益标签信号输入构建。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_raw_ey_5y_v1 import (
    build_report,
    build_signal_inputs,
    load_config,
    verify_input_hashes,
    write_artifacts,
)


def main() -> int:
    config = load_config()
    hashes = verify_input_hashes(config)
    if hashes["status"] != "PASS":
        raise RuntimeError(json.dumps(hashes, ensure_ascii=False, indent=2))
    upstream = json.loads(
        (ROOT / config["data_contracts"]["upstream_v1_1_report"]["file"]).read_text(
            encoding="utf-8"
        )
    )
    if upstream["branch_status"]["VAL01_RAW_EY_5Y"] != "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY":
        raise RuntimeError("上游V1.1没有放行VAL01_RAW_EY_5Y数据分支")

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(ROOT / config["data_contracts"][name]["file"])

    last_announced = {"bucket": 0}

    def progress(number: int, total: int, date: pd.Timestamp) -> None:
        bucket = number // 12
        if number == 1 or number == total or bucket > last_announced["bucket"]:
            last_announced["bucket"] = bucket
            print(f"已完成{number}/{total}个月：{date.date()}", flush=True)

    signals = build_signal_inputs(
        read("historical_weights"),
        read("point_in_time_financials"),
        read("snapshot_prices"),
        read("current_constituent_daily"),
        read("index_trading_calendar"),
        config,
        progress=progress,
    )
    report = build_report(
        signals,
        read("vendor_valuation_cross_check"),
        read("official_pe_cross_check"),
        config,
        hashes,
    )
    write_artifacts(signals, report, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS_SIGNAL_INPUTS_NO_RETURN_LABELS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
