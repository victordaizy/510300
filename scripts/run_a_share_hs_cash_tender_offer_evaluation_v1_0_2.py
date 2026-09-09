from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_cash_tender_offer_evaluation_v1_0_2 import (  # noqa: E402
    PROTOCOL_ID,
    build_event_clock,
    calculate_acceptance_metrics,
    evaluate_acceptance_gates,
    evaluate_event_ledger,
    validate_static_config,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_PATH = (
    ROOT
    / "config/a_share_hs_official_unconditional_full_cash_tender_spread_v1_"
    "evaluation_v1_0_2.json"
)
PROTOCOL_DOC = (
    ROOT
    / "docs/A_SHARE_HS_OFFICIAL_UNCONDITIONAL_FULL_CASH_TENDER_SPREAD_V1_"
    "EVALUATION_V1_0_2_PROTOCOL.md"
)
MODULE_PATH = ROOT / "research/a_share_hs_cash_tender_offer_evaluation_v1_0_2.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_a_share_hs_cash_tender_offer_evaluation_v1_0_2.py"
PROTOCOL_STATUS = (
    "FROZEN_TENDER_EVALUATION_IMPLEMENTATION_AND_INPUTS_"
    "BEFORE_EVENT_MARKET_PRICE_READ"
)
RESULTS_STATUS = "FROZEN_TENDER_EVALUATION_RESULTS_NO_PARAMETER_RESCUE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_static_config(config)
    return config


def project_path(config: dict[str, Any], section: str, key: str) -> Path:
    return ROOT / str(config[section][key])


def verify_manifest_tree(path: Path, expected_status: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"上游冻结清单不存在：{relative(path)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise RuntimeError(
            f"上游冻结状态错误：{relative(path)} {manifest.get('status')}"
        )
    raw_files = manifest.get("files") or {}
    if isinstance(raw_files, dict):
        tracked = {str(key): str(value) for key, value in raw_files.items()}
    elif isinstance(raw_files, list):
        tracked = {
            str(item["path"]): str(item["sha256"])
            for item in [*raw_files, *(manifest.get("references") or [])]
        }
    else:
        raise RuntimeError(f"上游冻结清单files结构非法：{relative(path)}")
    failures: list[dict[str, str | None]] = []
    for item_path, expected_hash in tracked.items():
        item = ROOT / item_path
        actual = sha256_file(item) if item.exists() else None
        if actual != expected_hash:
            failures.append(
                {
                    "path": item_path,
                    "expected": str(expected_hash),
                    "actual": actual,
                }
            )
    if failures:
        raise RuntimeError(
            "上游冻结文件哈希失败：" + json.dumps(failures, ensure_ascii=False)
        )
    return {
        "status": "PASS_UPSTREAM_MANIFEST_TREE_VERIFIED",
        "manifest": relative(path),
        "manifest_sha256": sha256_file(path),
        "checked_file_count": len(tracked),
    }


def verify_upstream(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    parent = verify_manifest_tree(
        ROOT / inputs["parent_protocol_manifest"],
        "FROZEN_AFTER_SOURCE_AND_PDF_FEASIBILITY_"
        "BEFORE_ANY_TENDER_EVENT_MARKET_PRICE_OR_OUTCOME_READ",
    )
    final = verify_manifest_tree(
        ROOT / inputs["final_adjudication_archive_manifest"],
        "FROZEN_FINAL_ADJUDICATION_BEFORE_MARKET_PRICE_AND_OBSERVED_SPREAD_READ",
    )
    lifecycle = verify_manifest_tree(
        ROOT / inputs["lifecycle_review_archive_manifest"],
        "FROZEN_LIFECYCLE_ADJUDICATION_REVIEW_ARCHIVE_"
        "BEFORE_FINAL_EVENT_ADJUDICATION",
    )
    return {
        "status": "PASS_TENDER_EVALUATION_UPSTREAM_VERIFIED",
        "parent_protocol": parent,
        "final_adjudication": final,
        "lifecycle_review": lifecycle,
        "tender_event_market_price_read": False,
        "tender_event_future_return_read": False,
    }


def validate_parquet_metadata(config: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, int] = {}
    for key, contract in config["expected_parquet"].items():
        path = project_path(config, "inputs", key)
        if not path.exists():
            raise RuntimeError(f"冻结输入不存在：{relative(path)}")
        parquet = pq.ParquetFile(path)
        actual_rows = int(parquet.metadata.num_rows)
        required = set(contract["required_columns"])
        actual_columns = set(parquet.schema_arrow.names)
        if actual_rows != int(contract["rows"]):
            raise RuntimeError(
                f"Parquet行数漂移：{key} {actual_rows}!={contract['rows']}"
            )
        missing = sorted(required - actual_columns)
        if missing:
            raise RuntimeError(f"Parquet字段缺失：{key} {missing}")
        rows[key] = actual_rows
    return {
        "status": "PASS_TENDER_EVALUATION_PARQUET_METADATA_VALIDATED",
        "row_counts": rows,
        "market_values_read": False,
        "future_returns_read": False,
    }


def protocol_file_paths(config: dict[str, Any]) -> list[Path]:
    inputs = config["inputs"]
    paths = [CONFIG_PATH, PROTOCOL_DOC, MODULE_PATH, RUNNER_PATH, TEST_PATH]
    paths.extend(ROOT / str(value) for value in inputs.values())
    unique = {path.resolve(): path for path in paths}
    return [unique[key] for key in sorted(unique, key=lambda item: item.as_posix())]


def validate_plan(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    existing_results = [
        str(value)
        for key, value in artifacts.items()
        if key not in {"protocol_manifest", "results_archive_manifest"}
        and (ROOT / str(value)).exists()
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_TENDER_EVALUATION_PLAN_VALIDATED",
        "static_validation": validate_static_config(config),
        "upstream_verification": verify_upstream(config),
        "metadata_validation": validate_parquet_metadata(config),
        "existing_result_artifacts": existing_results,
        "protocol_manifest_exists": project_path(
            config, "artifacts", "protocol_manifest"
        ).exists(),
        "tender_event_market_price_read": False,
        "tender_event_future_return_read": False,
    }


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config, "artifacts", "protocol_manifest")
    if manifest_path.exists():
        raise RuntimeError(f"盲评协议已冻结，禁止覆盖：{relative(manifest_path)}")
    plan = validate_plan(config)
    if plan["existing_result_artifacts"]:
        raise RuntimeError("结果产物在协议冻结前已经存在")
    files = {relative(path): sha256_file(path) for path in protocol_file_paths(config)}
    core = canonical_json(files)
    manifest = {
        "protocol_id": f"{PROTOCOL_ID}_FREEZE",
        "status": PROTOCOL_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "files": files,
        "content_sha256": hashlib.sha256(core.encode("utf-8")).hexdigest(),
        "plan_validation": plan,
        "h00300_input_completion": {
            "reason": "V1.0.1全收益OHLC开盘严重缺失；V1.0.2用冻结的沪深300价格指数当日日内比率桥接H00300开盘",
            "open_bridge_formula": "H00300_CLOSE_T_TIMES_CSI300_PRICE_OPEN_T_DIV_CSI300_PRICE_CLOSE_T",
            "failure_status": "NO_VIEW_GLOBAL_INPUT_CONTRACT_FAILURE",
        },
        "event_set_changed": False,
        "thresholds_changed": False,
        "costs_changed": False,
        "tender_event_market_price_read": False,
        "tender_event_future_return_read": False,
        "observed_spread_filter_applied": False,
        "immutable_rule": "冻结后实现、输入、候选、阈值、成本、基准和门槛只能通过新版本变更",
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config, "artifacts", "protocol_manifest")
    if not manifest_path.exists():
        raise RuntimeError("盲评协议冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str | None]] = []
    if manifest.get("status") != PROTOCOL_STATUS:
        failures.append(
            {
                "path": relative(manifest_path),
                "expected": PROTOCOL_STATUS,
                "actual": str(manifest.get("status")),
            }
        )
    for item_path, expected_hash in (manifest.get("files") or {}).items():
        item = ROOT / item_path
        actual = sha256_file(item) if item.exists() else None
        if actual != expected_hash:
            failures.append(
                {"path": item_path, "expected": expected_hash, "actual": actual}
            )
    if failures:
        raise RuntimeError(
            "盲评协议冻结哈希失败：" + json.dumps(failures, ensure_ascii=False)
        )
    return {
        "status": "PASS_TENDER_EVALUATION_PROTOCOL_FREEZE_VERIFIED",
        "checked_file_count": len(manifest["files"]),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": manifest["content_sha256"],
        "upstream_verification": verify_upstream(config),
        "metadata_validation": validate_parquet_metadata(config),
        "tender_event_market_price_read_before_freeze": False,
        "tender_event_future_return_read_before_freeze": False,
        "benchmark_input_values_read_for_v1_0_1_failure_diagnosis": True,
    }


def load_small_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    keys = (
        "final_event_outcomes",
        "final_corporate_actions",
        "lifecycle_review_archive",
        "security_master",
        "security_status_intervals",
        "trading_calendar",
    )
    return {
        key: pd.read_parquet(project_path(config, "inputs", key)) for key in keys
    }


def validate_h00300_completion(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = pd.read_parquet(
        project_path(config, "inputs", "h00300_close_reference"),
        columns=["date", "close"],
    ).rename(columns={"close": "h00300_close"})
    price_index = pd.read_parquet(
        project_path(config, "inputs", "csi300_price_index_ohlc_bridge"),
        columns=["date", "open", "close"],
    ).rename(
        columns={
            "open": "csi300_price_open",
            "close": "csi300_price_close",
        }
    )
    for frame in (reference, price_index):
        frame["date"] = pd.to_datetime(
            frame["date"], errors="coerce"
        ).dt.normalize()
    if reference["date"].isna().any() or reference["date"].duplicated().any():
        raise RuntimeError("H00300收盘参考日期无效或重复")
    if price_index["date"].isna().any() or price_index["date"].duplicated().any():
        raise RuntimeError("沪深300价格指数桥接日期无效或重复")
    reference["h00300_close"] = pd.to_numeric(
        reference["h00300_close"], errors="coerce"
    )
    for column in ("csi300_price_open", "csi300_price_close"):
        price_index[column] = pd.to_numeric(price_index[column], errors="coerce")
    if (
        reference["h00300_close"].isna().any()
        or reference["h00300_close"].le(0.0).any()
    ):
        raise RuntimeError("H00300收盘参考存在空值或非正值")
    if (
        price_index[["csi300_price_open", "csi300_price_close"]].isna().any().any()
        or (
            price_index[["csi300_price_open", "csi300_price_close"]] <= 0.0
        ).any().any()
    ):
        raise RuntimeError("沪深300价格指数桥接存在空值或非正值")
    overlap = reference.merge(price_index, on="date", how="inner")
    if len(overlap) != len(reference):
        raise RuntimeError(
            f"H00300开盘桥接未覆盖全部参考日期：{len(overlap)}/{len(reference)}"
        )
    output = pd.DataFrame(
        {
            "date": overlap["date"],
            "open": (
                overlap["h00300_close"]
                * overlap["csi300_price_open"]
                / overlap["csi300_price_close"]
            ),
            "close": overlap["h00300_close"],
        }
    )
    if (
        output[["open", "close"]].isna().any().any()
        or (output[["open", "close"]] <= 0.0).any().any()
    ):
        raise RuntimeError("H00300桥接结果存在空值或非正值")
    close_error = (
        output["close"] - overlap["h00300_close"]
    ).abs().max()
    tolerance = float(
        config["benchmarks"]["h00300_close_preservation_relative_tolerance"]
    )
    if float(close_error) > tolerance:
        raise RuntimeError(
            f"H00300桥接未保持原收盘：max_absolute_error={close_error}"
        )
    return output, {
        "status": "PASS_H00300_PRICE_INDEX_OPEN_BRIDGE_VALIDATED",
        "reference_rows": len(reference),
        "price_index_rows": len(price_index),
        "overlap_rows": len(overlap),
        "first_bridge_date": str(output["date"].min().date()),
        "last_bridge_date": str(output["date"].max().date()),
        "formula": (
            "H00300_CLOSE_T_TIMES_CSI300_PRICE_OPEN_T_DIV_CSI300_PRICE_CLOSE_T"
        ),
        "maximum_close_preservation_absolute_error": float(close_error),
        "close_preservation_tolerance": tolerance,
    }
def extract_market_inputs(
    config: dict[str, Any], event_clock: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_path = project_path(config, "inputs", "unified_daily_market")
    peer_path = project_path(config, "inputs", "industry_peer_return_panel")
    clock = event_clock[
        [
            "event_id",
            "ts_code",
            "report_market_date",
            "entry_date",
            "cash_availability_date",
        ]
    ].copy()
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        connection.register("event_clock", clock)
        event_market_rows = connection.execute(
            f"""
            SELECT
                c.event_id,
                CASE
                    WHEN CAST(m.date AS DATE) = CAST(c.report_market_date AS DATE)
                    THEN 'REPORT' ELSE 'ENTRY'
                END AS market_role,
                m.con_code,
                CAST(m.date AS DATE) AS date,
                m.raw_open,
                m.raw_high,
                m.raw_low,
                m.raw_close,
                m.pct_chg,
                m.amount,
                m.total_return_open,
                m.total_return_close,
                m.is_suspended,
                m.observed_traded_row
            FROM read_parquet('{sql_path(market_path)}') AS m
            INNER JOIN event_clock AS c
              ON m.con_code = c.ts_code
             AND CAST(m.date AS DATE) IN (
                 CAST(c.report_market_date AS DATE), CAST(c.entry_date AS DATE)
             )
            ORDER BY c.event_id, market_role
            """
        ).fetch_df()
        industry_entry_stats = connection.execute(
            f"""
            WITH subject AS (
                SELECT
                    c.event_id,
                    c.ts_code,
                    CAST(c.entry_date AS DATE) AS entry_date,
                    p.industry_l1_code
                FROM event_clock AS c
                LEFT JOIN read_parquet('{sql_path(peer_path)}') AS p
                  ON p.con_code = c.ts_code
                 AND CAST(p.date AS DATE) = CAST(c.entry_date AS DATE)
            ), peer_rows AS (
                SELECT
                    s.event_id,
                    s.industry_l1_code,
                    p.con_code AS peer_code,
                    m.total_return_open,
                    m.total_return_close
                FROM subject AS s
                LEFT JOIN read_parquet('{sql_path(peer_path)}') AS p
                  ON CAST(p.date AS DATE) = s.entry_date
                 AND p.industry_l1_code = s.industry_l1_code
                 AND p.con_code <> s.ts_code
                LEFT JOIN read_parquet('{sql_path(market_path)}') AS m
                  ON m.con_code = p.con_code
                 AND CAST(m.date AS DATE) = s.entry_date
            )
            SELECT
                event_id,
                MIN(industry_l1_code) AS industry_l1_code,
                COUNT(*) FILTER (
                    WHERE total_return_open IS NOT NULL
                      AND total_return_close IS NOT NULL
                      AND isfinite(total_return_open)
                      AND isfinite(total_return_close)
                      AND total_return_open > 0
                      AND total_return_close > 0
                )::INTEGER AS entry_intraday_peer_count,
                AVG(
                    CASE
                        WHEN total_return_open IS NOT NULL
                         AND total_return_close IS NOT NULL
                         AND isfinite(total_return_open)
                         AND isfinite(total_return_close)
                         AND total_return_open > 0
                         AND total_return_close > 0
                        THEN LN(total_return_close / total_return_open)
                        ELSE NULL
                    END
                ) AS entry_intraday_peer_log_return
            FROM peer_rows
            GROUP BY event_id
            ORDER BY event_id
            """
        ).fetch_df()
        industry_daily_rows = connection.execute(
            f"""
            SELECT
                c.event_id,
                CAST(p.date AS DATE) AS date,
                p.industry_peer_count,
                p.industry_peer_log_return
            FROM event_clock AS c
            INNER JOIN read_parquet('{sql_path(peer_path)}') AS p
              ON p.con_code = c.ts_code
             AND CAST(p.date AS DATE) > CAST(c.entry_date AS DATE)
             AND CAST(p.date AS DATE) <= CAST(c.cash_availability_date AS DATE)
            ORDER BY c.event_id, p.date
            """
        ).fetch_df()
        market_quality = connection.execute(
            f"""
            SELECT
                (SELECT COUNT(*) FROM read_parquet('{sql_path(market_path)}')) AS market_rows,
                (SELECT COUNT(*) FROM read_parquet('{sql_path(peer_path)}')) AS peer_rows,
                (SELECT MIN(CAST(date AS DATE)) FROM read_parquet('{sql_path(market_path)}')) AS market_first_date,
                (SELECT MAX(CAST(date AS DATE)) FROM read_parquet('{sql_path(market_path)}')) AS market_last_date,
                (SELECT MIN(CAST(date AS DATE)) FROM read_parquet('{sql_path(peer_path)}')) AS peer_first_date,
                (SELECT MAX(CAST(date AS DATE)) FROM read_parquet('{sql_path(peer_path)}')) AS peer_last_date
            """
        ).fetchone()
    finally:
        connection.close()
    receipt = {
        "status": "PASS_FROZEN_MARKET_SUBSETS_EXTRACTED",
        "event_market_row_count": len(event_market_rows),
        "industry_entry_stat_count": len(industry_entry_stats),
        "industry_daily_row_count": len(industry_daily_rows),
        "market_rows": int(market_quality[0]),
        "peer_rows": int(market_quality[1]),
        "market_first_date": str(market_quality[2]),
        "market_last_date": str(market_quality[3]),
        "peer_first_date": str(market_quality[4]),
        "peer_last_date": str(market_quality[5]),
    }
    return event_market_rows, industry_entry_stats, industry_daily_rows, receipt


def render_report(
    metrics: dict[str, Any], gates: dict[str, Any], receipt: dict[str, Any]
) -> str:
    primary = metrics["primary"]
    secondary = metrics["secondary_h00300"]
    lines = [
        "# 全面现金要约 8% 价差 V1.0.1 盲评结果",
        "",
        f"- 历史状态：`{gates['historical_status']}`",
        f"- 证据截止日：`{receipt['evidence_cutoff']}`",
        f"- 冻结事件：{metrics['source_event_count']} 个",
        f"- 观察价差通过：{metrics['observed_spread_pass_count']} 个",
        f"- 次日开盘成交：{metrics['entry_fill_count']} 个",
        f"- 主、次标签均解决：{metrics['primary_target_event_count']} 个",
        "",
        "## 压力情景核心指标",
        "",
        f"- 胜率：{primary['win_rate']}",
        f"- Wilson 95% 下界：{primary['wilson_95_lower']}",
        f"- 平均行业超额：{primary['mean']}",
        f"- 中位行业超额：{primary['median']}",
        f"- 盈亏比：{primary['payoff_ratio']}",
        f"- 利润因子：{primary['profit_factor']}",
        f"- H00300 超额胜率：{secondary['win_rate']}",
        "",
        "## 门槛",
        "",
        f"- 证据门：{gates['evidence_passed_count']}/{gates['evidence_required_count']}",
        f"- 经济门：{gates['economic_passed_count']}/{gates['economic_required_count']}",
        f"- 失败证据门：{', '.join(gates['failed_evidence_checks']) or '无'}",
        f"- 失败经济门：{', '.join(gates['failed_economic_checks']) or '无'}",
        "",
        "## 治理结论",
        "",
        "本结果只属于历史发现。即使全部通过，也必须先进入独立 Shadow；本研究未生成仓位、订单或实盘授权。",
        "",
    ]
    return "\n".join(lines)


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    protocol = verify_protocol(config)
    output_keys = (
        "event_clock",
        "event_ledger",
        "event_ledger_csv",
        "acceptance_metrics",
        "gate_results",
        "run_receipt",
        "research_report",
    )
    outputs = {key: project_path(config, "artifacts", key) for key in output_keys}
    existing = [relative(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(f"盲评产物已存在，禁止覆盖：{existing}")
    small = load_small_inputs(config)
    event_clock = build_event_clock(
        small["final_event_outcomes"],
        small["final_corporate_actions"],
        small["lifecycle_review_archive"],
        small["trading_calendar"],
        small["security_master"],
        small["security_status_intervals"],
        config,
    )
    h00300, h00300_validation = validate_h00300_completion(config)
    market_rows, entry_stats, daily_rows, market_receipt = extract_market_inputs(
        config, event_clock
    )
    ledger = evaluate_event_ledger(
        event_clock,
        market_rows,
        entry_stats,
        daily_rows,
        h00300,
        small["final_corporate_actions"],
        config,
    )
    metrics = json_safe(calculate_acceptance_metrics(ledger, config))
    gates = json_safe(evaluate_acceptance_gates(metrics, config))
    atomic_write_parquet(outputs["event_clock"], event_clock)
    atomic_write_parquet(outputs["event_ledger"], ledger)
    atomic_write_csv(outputs["event_ledger_csv"], ledger)
    atomic_write_text(
        outputs["acceptance_metrics"],
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        outputs["gate_results"],
        json.dumps(gates, ensure_ascii=False, indent=2) + "\n",
    )
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_TENDER_EVALUATION_RUN_COMPLETE",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "historical_status": gates["historical_status"],
        "protocol_verification": protocol,
        "h00300_validation": h00300_validation,
        "market_input_receipt": market_receipt,
        "event_count": len(ledger),
        "observed_spread_pass_count": metrics["observed_spread_pass_count"],
        "entry_fill_count": metrics["entry_fill_count"],
        "evaluated_event_count": metrics["primary_target_event_count"],
        "event_clock": relative(outputs["event_clock"]),
        "event_clock_sha256": sha256_file(outputs["event_clock"]),
        "event_ledger": relative(outputs["event_ledger"]),
        "event_ledger_sha256": sha256_file(outputs["event_ledger"]),
        "event_ledger_csv": relative(outputs["event_ledger_csv"]),
        "event_ledger_csv_sha256": sha256_file(outputs["event_ledger_csv"]),
        "acceptance_metrics": relative(outputs["acceptance_metrics"]),
        "acceptance_metrics_sha256": sha256_file(outputs["acceptance_metrics"]),
        "gate_results": relative(outputs["gate_results"]),
        "gate_results_sha256": sha256_file(outputs["gate_results"]),
        "event_set_changed": False,
        "thresholds_changed": False,
        "costs_changed": False,
        "liquidity_filter_applied": False,
        "market_cap_filter_applied": False,
        "outlier_deleted": False,
        "winsorization_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_write_text(
        outputs["research_report"], render_report(metrics, gates, receipt)
    )
    receipt["research_report"] = relative(outputs["research_report"])
    receipt["research_report_sha256"] = sha256_file(outputs["research_report"])
    atomic_write_text(
        outputs["run_receipt"],
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
    )
    return receipt


def verify_results(config: dict[str, Any]) -> dict[str, Any]:
    protocol = verify_protocol(config)
    paths = {
        key: project_path(config, "artifacts", key)
        for key in (
            "event_clock",
            "event_ledger",
            "event_ledger_csv",
            "acceptance_metrics",
            "gate_results",
            "run_receipt",
            "research_report",
        )
    }
    if not paths["run_receipt"].exists():
        raise RuntimeError("盲评运行回执不存在")
    receipt = json.loads(paths["run_receipt"].read_text(encoding="utf-8"))
    failures: list[str] = []
    if receipt.get("status") != "PASS_TENDER_EVALUATION_RUN_COMPLETE":
        failures.append(f"STATUS:{receipt.get('status')}")
    for key in (
        "event_clock",
        "event_ledger",
        "event_ledger_csv",
        "acceptance_metrics",
        "gate_results",
        "research_report",
    ):
        actual = sha256_file(paths[key]) if paths[key].exists() else None
        if actual != receipt.get(f"{key}_sha256"):
            failures.append(f"HASH:{key}:{actual}!={receipt.get(f'{key}_sha256')}")
    ledger = pd.read_parquet(paths["event_ledger"])
    recalculated_metrics = json_safe(calculate_acceptance_metrics(ledger, config))
    recalculated_gates = json_safe(
        evaluate_acceptance_gates(recalculated_metrics, config)
    )
    stored_metrics = json.loads(paths["acceptance_metrics"].read_text(encoding="utf-8"))
    stored_gates = json.loads(paths["gate_results"].read_text(encoding="utf-8"))
    if canonical_json(recalculated_metrics) != canonical_json(stored_metrics):
        failures.append("METRICS_RECALCULATION_MISMATCH")
    if canonical_json(recalculated_gates) != canonical_json(stored_gates):
        failures.append("GATES_RECALCULATION_MISMATCH")
    if receipt.get("historical_status") != stored_gates.get("historical_status"):
        failures.append("RECEIPT_STATUS_MISMATCH")
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "PASS_TENDER_EVALUATION_RESULTS_VERIFIED"
            if not failures
            else "FAILED_TENDER_EVALUATION_RESULTS_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "historical_status": stored_gates.get("historical_status"),
        "event_count": len(ledger),
        "evaluated_event_count": recalculated_metrics["primary_target_event_count"],
        "protocol_verification": protocol,
        "position_mapping": False,
        "order_generation": False,
        "live_authorized": False,
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def freeze_results(config: dict[str, Any]) -> dict[str, Any]:
    verification = verify_results(config)
    manifest_path = project_path(config, "artifacts", "results_archive_manifest")
    if manifest_path.exists():
        raise RuntimeError(f"盲评结果已冻结，禁止覆盖：{relative(manifest_path)}")
    tracked = [
        project_path(config, "artifacts", key)
        for key in (
            "protocol_manifest",
            "event_clock",
            "event_ledger",
            "event_ledger_csv",
            "acceptance_metrics",
            "gate_results",
            "run_receipt",
            "research_report",
        )
    ]
    files = {relative(path): sha256_file(path) for path in tracked}
    manifest = {
        "protocol_id": f"{PROTOCOL_ID}_RESULTS_FREEZE",
        "status": RESULTS_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "historical_status": verification["historical_status"],
        "files": files,
        "content_sha256": hashlib.sha256(
            canonical_json(files).encode("utf-8")
        ).hexdigest(),
        "verification": verification,
        "event_set_changed": False,
        "thresholds_changed": False,
        "costs_changed": False,
        "liquidity_filter_applied": False,
        "parameter_rescue_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def verify_archive(config: dict[str, Any]) -> dict[str, Any]:
    results = verify_results(config)
    manifest_path = project_path(config, "artifacts", "results_archive_manifest")
    if not manifest_path.exists():
        raise RuntimeError("盲评结果冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str | None]] = []
    if manifest.get("status") != RESULTS_STATUS:
        failures.append(
            {
                "path": relative(manifest_path),
                "expected": RESULTS_STATUS,
                "actual": str(manifest.get("status")),
            }
        )
    for item_path, expected_hash in (manifest.get("files") or {}).items():
        item = ROOT / item_path
        actual = sha256_file(item) if item.exists() else None
        if actual != expected_hash:
            failures.append(
                {"path": item_path, "expected": expected_hash, "actual": actual}
            )
    if failures:
        raise RuntimeError(
            "盲评结果冻结哈希失败：" + json.dumps(failures, ensure_ascii=False)
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_TENDER_EVALUATION_RESULTS_ARCHIVE_VERIFIED",
        "historical_status": manifest["historical_status"],
        "checked_file_count": len(manifest["files"]),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": manifest["content_sha256"],
        "results_verification": results,
        "position_mapping": False,
        "order_generation": False,
        "live_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结并执行全面现金要约8%价差盲评")
    parser.add_argument(
        "--mode",
        choices=(
            "validate-plan",
            "freeze-protocol",
            "verify-protocol",
            "run",
            "verify-results",
            "freeze-results",
            "verify-archive",
        ),
        default="validate-plan",
    )
    args = parser.parse_args()
    config = load_config()
    if args.mode == "validate-plan":
        result = validate_plan(config)
    elif args.mode == "freeze-protocol":
        result = freeze_protocol(config)
    elif args.mode == "verify-protocol":
        result = verify_protocol(config)
    elif args.mode == "run":
        result = run_evaluation(config)
    elif args.mode == "verify-results":
        result = verify_results(config)
    elif args.mode == "freeze-results":
        result = freeze_results(config)
    else:
        result = verify_archive(config)
    print(json.dumps(json_safe(result), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
