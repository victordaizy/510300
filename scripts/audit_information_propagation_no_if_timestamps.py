"""审计无IF实验中510300、000300与成分股的一分钟时间标签。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.information_propagation_timestamp_audit import audit_timestamp_alignment
from scripts.audit_information_propagation_timestamps import (
    _read_component_sample,
    _read_table,
    _write_json,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_no_if.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    audit_config = config["timestamp_audit"]
    paths = {
        "etf": _resolve(data_config["etf_1m"]),
        "index": _resolve(data_config["index_1m"]),
        "components": _resolve(data_config["component_1m"]),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        report = {"status": "BLOCKED_MISSING_INPUT", "blocking_inputs": missing}
        _write_json(report, _resolve(data_config["timestamp_audit_report"]))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    etf = _read_table(paths["etf"], ["trade_time"])
    index = _read_table(paths["index"], ["trade_time"])
    etf_days = set(pd.to_datetime(etf["trade_time"]).dt.normalize().unique())
    index_days = set(pd.to_datetime(index["trade_time"]).dt.normalize().unique())
    common_days = sorted(etf_days.intersection(index_days))
    sample_count = int(audit_config["sample_trading_days"])
    if len(common_days) < sample_count:
        report = {
            "status": "FAIL_INSUFFICIENT_COMMON_DAYS",
            "common_trading_day_count": len(common_days),
        }
        _write_json(report, _resolve(data_config["timestamp_audit_report"]))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    rng = np.random.default_rng(int(audit_config["deterministic_seed"]))
    sampled_days = sorted(rng.choice(common_days, size=sample_count, replace=False))
    selected_day_set = {pd.Timestamp(day) for day in sampled_days}
    etf = etf.loc[pd.to_datetime(etf["trade_time"]).dt.normalize().isin(selected_day_set)]
    index = index.loc[pd.to_datetime(index["trade_time"]).dt.normalize().isin(selected_day_set)]
    components = _read_component_sample(
        paths["components"], [pd.Timestamp(day) for day in sampled_days]
    )
    report = audit_timestamp_alignment(
        etf=etf,
        index=index,
        futures=None,
        components=components,
        sample_trading_days=sample_count,
        deterministic_seed=int(audit_config["deterministic_seed"]),
        required_anchor_times=tuple(str(value) for value in audit_config["required_anchor_times"]),
        expected_etf_component_bars_per_day=int(
            audit_config["expected_etf_component_bars_per_day"]
        ),
        manual_semantics_status=str(audit_config["manual_semantics_status"]),
    )
    report["manual_semantics_note"] = str(audit_config["manual_semantics_note"])
    report["experiment_id"] = config["experiment"]["id"]
    output_path = _resolve(data_config["timestamp_audit_report"])
    _write_json(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
