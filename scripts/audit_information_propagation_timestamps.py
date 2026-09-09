"""执行 510300 信息传播研究的分钟时间戳硬审计。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.information_propagation_alpha import select_causal_futures_contract
from research.information_propagation_timestamp_audit import audit_timestamp_alignment


CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_alpha.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.is_dir():
        dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
        return dataset.to_table(columns=columns).to_pandas()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
        return data if columns is None else data[columns]
    raise ValueError(f"不支持的数据格式：{path}")


def _read_component_sample(path: Path, sampled_days: list[pd.Timestamp]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    expression = None
    for day in sampled_days:
        lower = pd.Timestamp(day).to_datetime64()
        upper = (pd.Timestamp(day) + pd.Timedelta(days=1)).to_datetime64()
        day_expression = (ds.field("trade_time") >= lower) & (ds.field("trade_time") < upper)
        expression = day_expression if expression is None else expression | day_expression
    return dataset.to_table(columns=["trade_time", "con_code"], filter=expression).to_pandas()


def run(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data_config = config["data"]
    paths = {
        "etf": _resolve(data_config["etf_1m"]),
        "index": _resolve(data_config["index_1m"]),
        "futures": _resolve(data_config["if_contract_1m"]),
        "components": _resolve(data_config["component_1m"]),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return 2, {
            "status": "BLOCKED_MISSING_INPUT",
            "target_asset": "510300.SH",
            "index_role": "仅作价格发现对照，不是收益或交易标的",
            "blocking_inputs": missing,
            "paths": {name: str(path) for name, path in paths.items()},
        }

    etf = _read_table(paths["etf"], ["trade_time"])
    index = _read_table(paths["index"], ["trade_time"])
    futures = _read_table(paths["futures"], ["trade_time", "contract", "volume"])
    selected_futures = select_causal_futures_contract(futures)
    date_sets = []
    for frame in [etf, index, selected_futures]:
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        date_sets.append(set(timestamps.dt.normalize().dropna().unique()))
    common_days = sorted(set.intersection(*date_sets))
    audit_config = config["timestamp_audit"]
    sample_count = int(audit_config["sample_trading_days"])
    if len(common_days) < sample_count:
        return 2, {
            "status": "FAIL_INSUFFICIENT_COMMON_DAYS",
            "common_trading_day_count": len(common_days),
            "required_sample_trading_days": sample_count,
        }
    rng = np.random.default_rng(int(audit_config["deterministic_seed"]))
    sampled_days = sorted(rng.choice(common_days, size=sample_count, replace=False))
    components = _read_component_sample(paths["components"], [pd.Timestamp(day) for day in sampled_days])
    selected_day_set = {pd.Timestamp(day) for day in sampled_days}
    etf = etf.loc[pd.to_datetime(etf["trade_time"]).dt.normalize().isin(selected_day_set)]
    index = index.loc[pd.to_datetime(index["trade_time"]).dt.normalize().isin(selected_day_set)]
    selected_futures = selected_futures.loc[
        pd.to_datetime(selected_futures["trade_time"]).dt.normalize().isin(selected_day_set),
        ["trade_time"],
    ]
    result = audit_timestamp_alignment(
        etf=etf,
        index=index,
        futures=selected_futures,
        components=components,
        sample_trading_days=sample_count,
        deterministic_seed=int(audit_config["deterministic_seed"]),
        required_anchor_times=tuple(str(value) for value in audit_config["required_anchor_times"]),
        expected_etf_component_bars_per_day=int(
            audit_config["expected_etf_component_bars_per_day"]
        ),
        manual_semantics_status=str(audit_config["manual_semantics_status"]),
    )
    return (0 if result["status"] == "PASS" else 2), result


def main() -> int:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    code, report = run(config)
    _write_json(report, _resolve(config["data"]["timestamp_audit_report"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
