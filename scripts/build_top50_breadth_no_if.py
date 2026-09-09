"""用DuckDB流式扫描117只成分股分区，构建点时Top50分钟内部扩散。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.information_propagation_alpha import _causal_weight_map


CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_no_if.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    parameters = config["parameters"]
    etf_path = _resolve(data_config["etf_1m"])
    weights_path = _resolve(data_config["historical_weights"])
    components_path = _resolve(data_config["component_1m"])
    output_path = _resolve(data_config["top50_breadth_1m"])
    report_path = _resolve(data_config["breadth_build_report"])
    required_paths = [etf_path, weights_path, components_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        report = {"status": "BLOCKED_MISSING_INPUT", "missing_paths": missing}
        _atomic_json(report, report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    etf = pd.read_parquet(etf_path, columns=["trade_time"])
    session_dates = pd.to_datetime(etf["trade_time"]).dt.normalize().drop_duplicates()
    weights = pd.read_parquet(weights_path, columns=["trade_date", "con_code", "weight"])
    leaders = _causal_weight_map(
        session_dates=session_dates,
        weights=weights,
        leader_count=int(parameters["leader_count"]),
        symbol_column="con_code",
        weight_date_column="trade_date",
        weight_column="weight",
    )
    leaders["session_date"] = pd.to_datetime(leaders["session_date"])
    leaders["weight_snapshot_date"] = pd.to_datetime(leaders["trade_date"])
    leaders = leaders[["session_date", "weight_snapshot_date", "con_code", "leader_weight"]]
    expected_rows = int(leaders["session_date"].nunique() * int(parameters["leader_count"]))
    if len(leaders) != expected_rows:
        report = {
            "status": "FAIL_INCOMPLETE_CAUSAL_LEADER_MAP",
            "actual_rows": int(len(leaders)),
            "expected_rows": expected_rows,
        }
        _atomic_json(report, report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    parquet_glob = (components_path / "*.parquet").as_posix().replace("'", "''")
    lookback = int(parameters["lookback_minutes"])
    impulse = int(parameters["breadth_impulse_minutes"])
    minimum_coverage = float(parameters["minimum_component_weight_coverage"])
    minimum_count = int(np.ceil(int(parameters["leader_count"]) * minimum_coverage))
    temp_directory = PROJECT_ROOT / "tmp" / "duckdb_top50_breadth"
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory='{temp_directory.as_posix().replace(chr(39), chr(39) * 2)}'")
    connection.execute("SET threads=4")
    connection.register("leader_map_frame", leaders)
    query = f"""
        WITH selected AS (
            SELECT
                CAST(c.trade_time AS TIMESTAMP) AS trade_time,
                CAST(c.trade_time AS DATE) AS session_date,
                l.weight_snapshot_date,
                c.con_code,
                CAST(c.close AS DOUBLE) AS close,
                CAST(l.leader_weight AS DOUBLE) AS leader_weight,
                LAG(CAST(c.trade_time AS TIMESTAMP), {lookback}) OVER (
                    PARTITION BY c.con_code, CAST(c.trade_time AS DATE)
                    ORDER BY c.trade_time
                ) AS lag_time,
                LAG(CAST(c.close AS DOUBLE), {lookback}) OVER (
                    PARTITION BY c.con_code, CAST(c.trade_time AS DATE)
                    ORDER BY c.trade_time
                ) AS lag_close
            FROM read_parquet('{parquet_glob}', union_by_name=true) AS c
            INNER JOIN leader_map_frame AS l
                ON c.con_code = l.con_code
               AND CAST(c.trade_time AS DATE) = CAST(l.session_date AS DATE)
        ), returns AS (
            SELECT *,
                CASE
                    WHEN trade_time = lag_time + INTERVAL {lookback} MINUTE
                    THEN close / lag_close - 1.0
                    ELSE NULL
                END AS component_return
            FROM selected
        ), aggregated AS (
            SELECT
                trade_time,
                FIRST(session_date) AS session_date,
                FIRST(weight_snapshot_date) AS weight_snapshot_date,
                SUM(CASE WHEN component_return IS NOT NULL THEN leader_weight ELSE 0.0 END)
                    AS leader_weight_coverage,
                SUM(CASE WHEN component_return IS NOT NULL THEN component_return * leader_weight ELSE 0.0 END)
                    AS weighted_return_sum,
                SUM(CASE WHEN component_return > 0 THEN leader_weight ELSE 0.0 END)
                    AS positive_weight_sum,
                COUNT(component_return) AS available_component_count,
                SUM(CASE WHEN component_return > 0 THEN 1 ELSE 0 END) AS positive_component_count
            FROM returns
            GROUP BY trade_time
        ), breadth AS (
            SELECT
                *,
                CASE WHEN leader_weight_coverage >= {minimum_coverage}
                    THEN weighted_return_sum / leader_weight_coverage ELSE NULL END AS leader_return,
                CASE WHEN available_component_count >= {minimum_count}
                    THEN positive_component_count / available_component_count ELSE NULL END AS top50_breadth,
                CASE WHEN leader_weight_coverage >= {minimum_coverage}
                    THEN positive_weight_sum / leader_weight_coverage ELSE NULL END AS top50_weighted_breadth
            FROM aggregated
        )
        SELECT
            current.trade_time,
            current.session_date,
            current.weight_snapshot_date,
            current.leader_weight_coverage,
            current.available_component_count,
            current.leader_return,
            current.top50_breadth,
            current.top50_weighted_breadth,
            current.top50_breadth - previous.top50_breadth
                AS top50_breadth_impulse_{impulse}m,
            current.top50_weighted_breadth - previous.top50_weighted_breadth
                AS top50_weighted_breadth_impulse_{impulse}m
        FROM breadth AS current
        LEFT JOIN breadth AS previous
          ON previous.trade_time = current.trade_time - INTERVAL {impulse} MINUTE
        ORDER BY current.trade_time
    """
    result = connection.execute(query).df()
    connection.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    report = {
        "status": "PASS",
        "target_asset": "510300.SH",
        "breadth_name": "Top50Breadth",
        "leader_map_rows": int(len(leaders)),
        "output_rows": int(len(result)),
        "trading_day_count": int(pd.to_datetime(result["session_date"]).nunique()),
        "first_trade_time": pd.Timestamp(result["trade_time"].min()).isoformat(),
        "last_trade_time": pd.Timestamp(result["trade_time"].max()).isoformat(),
        "valid_primary_impulse_rows": int(result[f"top50_breadth_impulse_{impulse}m"].notna().sum()),
        "output_path": str(output_path),
    }
    _atomic_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
