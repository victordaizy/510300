"""冻结并运行预期—消息—价格响应链的 PIT 入场审计。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.expectations_news_price_response_chain_pit_preflight_v1 import (  # noqa: E402
    evaluate_pit_chain_gates,
    preflight_adjudication,
)


PROGRAM_ID = "510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_CHAIN_PIT_PREFLIGHT_V1"
CONFIG_PATH = ROOT / "config/510300_expectations_news_price_response_chain_pit_preflight_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/expectations_news_price_response_chain_pit_preflight_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("PIT 链预检配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    _atomic_bytes_new(path, (payload + "\n").encode("utf-8"))


def atomic_text_new(path: Path, value: str) -> None:
    _atomic_bytes_new(path, value.encode("utf-8"))


def atomic_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _registered_files(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    parent = config["parent_contract"]
    source = config["source_contract"]
    return {
        parent["measurement_manifest"]: ("", "FROZEN_OWN_OBJECT_MEASUREMENT_PROTOCOL"),
        parent["measurement_status"]: ("", "FROZEN_OWN_OBJECT_MEASUREMENT_STATUS"),
        source["periodic_report_events"]["path"]: (
            source["periodic_report_events"]["expected_sha256"],
            "OFFICIAL_PERIODIC_EVENT_METADATA",
        ),
        source["periodic_report_receipt"]["path"]: (
            source["periodic_report_receipt"]["expected_sha256"],
            "OFFICIAL_PERIODIC_EVENT_ARCHIVE_RECEIPT",
        ),
        source["preliminary_official_metadata"]["path"]: (
            source["preliminary_official_metadata"]["expected_sha256"],
            "OFFICIAL_PRELIMINARY_EVENT_METADATA",
        ),
        source["preliminary_core_facts_report"]["path"]: (
            source["preliminary_core_facts_report"]["expected_sha256"],
            "OFFICIAL_CORE_FACT_EXTRACTION_STATUS",
        ),
        source["preliminary_core_facts"]["path"]: (
            source["preliminary_core_facts"]["expected_sha256"],
            "OFFICIAL_CORE_FACT_EXTRACTION_OUTPUT",
        ),
        source["component_state"]["path"]: (
            source["component_state"]["expected_sha256"],
            "MEMBERSHIP_WEIGHT_AND_EXPECTATION_CLOCK_AUDIT_SOURCE",
        ),
    }


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_contract"]
    manifest = json.loads(
        project_path(parent["measurement_manifest"]).read_text(encoding="utf-8")
    )
    status = json.loads(
        project_path(parent["measurement_status"]).read_text(encoding="utf-8")
    )
    if manifest.get("manifest_payload_sha256") != parent[
        "expected_manifest_payload_sha256"
    ]:
        raise ValueError("自有对象测量协议哈希不一致")
    if status.get("status_payload_sha256") != parent[
        "expected_status_payload_sha256"
    ]:
        raise ValueError("自有对象测量状态哈希不一致")
    if status.get("status") != parent["expected_status"]:
        raise ValueError("自有对象测量状态不允许进入 PIT 链预检")
    for relative, (expected, label) in _registered_files(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"{label} 缺失：{path}")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"{label} 哈希不一致")
    source = config["source_contract"]
    periodic_receipt = json.loads(
        project_path(source["periodic_report_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if periodic_receipt.get("status") != source["periodic_report_receipt"][
        "expected_status"
    ]:
        raise ValueError("定期报告事件档案回执状态不一致")
    facts_report = json.loads(
        project_path(source["preliminary_core_facts_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if facts_report.get("status") != source["preliminary_core_facts_report"][
        "expected_status"
    ]:
        raise ValueError("官方核心事实提取状态不一致")
    return status


def _output_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(value)
        for key, value in config["artifacts"].items()
        if key != "protocol_manifest"
    ]


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"PIT 链预检冻结清单已存在：{manifest_path}")
    existing = [str(path) for path in _output_paths(config) if path.exists()]
    if existing:
        raise RuntimeError(f"冻结前发现同名 PIT 链预检输出：{existing}")
    parent_status = validate_inputs(config)
    registered: dict[str, Any] = {}
    for relative, (_, role) in _registered_files(config).items():
        path = project_path(relative)
        registered[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_PIT_CHAIN_INPUT_VALUE_READ",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered,
        "parent_status_payload_sha256": parent_status["status_payload_sha256"],
        "pit_hard_gates_frozen": True,
        "market_price_value_read": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json_new(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("PIT 链预检协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_hash(payload) != expected_hash:
        raise ValueError("PIT 链预检清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("PIT 链预检配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("PIT 链预检配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"PIT 链预检实现发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"PIT 链预检输入发生漂移或缺失：{relative}")
    validate_inputs(config)
    return manifest


def _build_metrics(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_contract"]
    component = pd.read_parquet(
        project_path(source["component_state"]["path"]),
        columns=[
            "origin",
            "stock_code",
            "diagnostic_index_weight",
            "diagnostic_index_weight_status",
            "income_available_at",
        ],
    )
    official_weight_rows = component["diagnostic_index_weight_status"].astype(
        str
    ).str.startswith("PASS") & component["diagnostic_index_weight"].notna()
    periodic = pd.read_parquet(
        project_path(source["periodic_report_events"]["path"]),
        columns=[
            "ts_code",
            "report_period",
            "official_timestamp_at",
            "official_internal_date_equal",
        ],
    )
    periodic_time = pd.to_datetime(
        periodic["official_timestamp_at"], errors="coerce"
    )
    non_midnight = (
        periodic_time.notna()
        & (
            periodic_time.dt.hour.ne(0)
            | periodic_time.dt.minute.ne(0)
            | periodic_time.dt.second.ne(0)
        )
    )
    periodic_receipt = json.loads(
        project_path(source["periodic_report_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    preliminary = pd.read_parquet(
        project_path(source["preliminary_official_metadata"]["path"]),
        columns=["ts_code", "announcement_timestamp_at", "status"],
    )
    preliminary_time = pd.to_datetime(
        preliminary["announcement_timestamp_at"], errors="coerce"
    )
    preliminary_non_midnight = (
        preliminary_time.notna()
        & (
            preliminary_time.dt.hour.ne(0)
            | preliminary_time.dt.minute.ne(0)
            | preliminary_time.dt.second.ne(0)
        )
    )
    facts_report = json.loads(
        project_path(source["preliminary_core_facts_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    a3_denominator = int(facts_report["a3_denominator_count"])
    first_public_pass = int(facts_report["first_public_pass_count"])
    exact_share = float(non_midnight.sum() / len(periodic)) if len(periodic) else 0.0
    fact_rate = float(facts_report["a3_success_rate"])
    first_public_coverage = (
        first_public_pass / a3_denominator if a3_denominator else 0.0
    )
    official_fact_contract_pass = facts_report.get("status") != (
        "BLOCKED_DATA_CONTRACT_FAILED"
    )
    complete_clustering = bool(
        official_fact_contract_pass
        and exact_share
        >= float(config["hard_gates"]["minimum_exact_publication_timestamp_share"])
    )
    event_price_clock = bool(
        exact_share
        >= float(config["hard_gates"]["minimum_exact_publication_timestamp_share"])
    )
    return {
        "component_state_rows": int(len(component)),
        "component_state_origins": int(component["origin"].nunique()),
        "component_state_symbols": int(component["stock_code"].nunique()),
        "official_pit_weight_rows": int(official_weight_rows.sum()),
        "official_pit_weight_row_share": float(official_weight_rows.mean()),
        "income_availability_date_nonnull_share": float(
            component["income_available_at"].notna().mean()
        ),
        "periodic_event_rows": int(len(periodic)),
        "periodic_event_symbols": int(periodic["ts_code"].nunique()),
        "periodic_event_archive_pass": periodic_receipt.get("status") == source[
            "periodic_report_receipt"
        ]["expected_status"],
        "periodic_exact_timestamp_rows": int(non_midnight.sum()),
        "periodic_exact_timestamp_share": exact_share,
        "periodic_unique_symbol_period_keys": int(
            periodic.drop_duplicates(["ts_code", "report_period"]).shape[0]
        ),
        "preliminary_metadata_rows": int(len(preliminary)),
        "preliminary_metadata_symbols": int(preliminary["ts_code"].nunique()),
        "preliminary_exact_timestamp_rows": int(preliminary_non_midnight.sum()),
        "preliminary_exact_timestamp_share": float(
            preliminary_non_midnight.mean()
        ),
        "official_fact_extraction_success_count": int(
            facts_report["a3_success_count"]
        ),
        "official_fact_extraction_denominator": a3_denominator,
        "official_fact_extraction_success_rate": fact_rate,
        "first_public_fact_pass_count": first_public_pass,
        "first_public_fact_coverage": first_public_coverage,
        "official_fact_contract_pass": official_fact_contract_pass,
        "complete_event_clustering_universe": complete_clustering,
        "event_price_clock_admitted": event_price_clock,
        "market_price_value_read": False,
    }


def _build_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    gates: pd.DataFrame,
    adjudication: dict[str, Any],
) -> str:
    lines = [
        "# 510300_EXPECTATIONS_NEWS_PRICE_RESPONSE_CHAIN_PIT_PREFLIGHT_V1",
        "",
        "## 裁决",
        "",
        f"`{adjudication['chain_status']}`。在全部输入硬门槛通过前，未读取任何事件后市场价格值。",
        "",
        f"- 协议哈希：`{manifest['manifest_payload_sha256']}`",
        f"- 硬门槛：`{adjudication['passed_gate_count']}/{adjudication['total_gate_count']}` 通过。",
        "- `MARKET_PRICE_VALUE_READ=false`",
        "- `RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`。",
        "",
        "## 门槛结果",
        "",
        "| 门槛 | 实际值 | 要求 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for row in gates.itertuples(index=False):
        lines.append(
            f"| {row.gate_id} | {row.actual_value:.4f} | {row.required_value:.4f} | {row.gate_status} |"
        )
    lines.extend(
        [
            "",
            "## 可用基础与实际缺口",
            "",
            f"- 官方定期报告事件档案已有 `{metrics['periodic_event_rows']:,}` 行、`{metrics['periodic_event_symbols']:,}` 个证券，档案时序回执通过。",
            f"- 但只有 `{metrics['periodic_exact_timestamp_rows']:,}` 行（`{metrics['periodic_exact_timestamp_share']:.4%}`）具有非午夜时间；绝大多数只证明发布日期，不能区分盘前、盘中、盘后。",
            f"- 初步业绩官方元数据有 `{metrics['preliminary_metadata_rows']:,}` 行，其中非午夜时间覆盖 `{metrics['preliminary_exact_timestamp_share']:.2%}`；这一部分时钟较好，但核心事实提取仅 `{metrics['official_fact_extraction_success_count']}/{metrics['official_fact_extraction_denominator']}`（`{metrics['official_fact_extraction_success_rate']:.2%}`）。",
            f"- first-public 核心事实为 `{metrics['first_public_fact_pass_count']}/{metrics['official_fact_extraction_denominator']}`（`{metrics['first_public_fact_coverage']:.2%}`）。",
            f"- 42,000 行月度成分状态中，正式 PIT 历史指数权重通过行数为 `{metrics['official_pit_weight_rows']}`；现有权重是市值代理或未版本化诊断权重。",
            "",
            "## 主动修复顺序",
            "",
            "1. 取得并版本化历史成分/权重公告，使每个事件时点能证明成员资格与权重。",
            "2. 为定期报告补充真实发布时间或冻结统一的次日开盘归属规则；午夜日期不能被当成精确时钟。",
            "3. 把官方 first-public 数值事实提取覆盖从 55.42% 提升至冻结的 90%，并完成修订链去重。",
            "4. 在完整公告宇宙上按证券和时间聚类；只有随后才允许读取对应价格响应路径。",
            "",
            "当前没有事件级消息吸收结论、总回报预测、仓位或交易授权；保持 `ABSTAIN / POSITION_UNSET`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(config)
    metrics = _build_metrics(config)
    gates = evaluate_pit_chain_gates(metrics, config["hard_gates"])
    adjudication = preflight_adjudication(gates)
    artifacts = config["artifacts"]
    paths = {name: project_path(relative) for name, relative in artifacts.items()}
    pending = [path for name, path in paths.items() if name != "protocol_manifest" and path.exists()]
    if pending:
        raise RuntimeError(f"PIT 链预检输出已存在，禁止覆盖：{pending}")
    atomic_parquet_new(paths["gate_table"], gates)
    payload = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": adjudication["chain_status"],
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "metrics": metrics,
        "gates": dataframe_records(gates),
        "adjudication": adjudication,
        "market_price_value_read": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    payload["payload_sha256"] = canonical_hash(payload)
    atomic_json_new(paths["preflight_json"], payload)
    report = _build_report(manifest, metrics, gates, adjudication)
    atomic_text_new(paths["final_report"], report)
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": adjudication["chain_status"],
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "gate_table": {
            "path": paths["gate_table"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["gate_table"]),
            "rows": len(gates),
            "passed_gate_count": adjudication["passed_gate_count"],
            "failed_gate_count": int(
                adjudication["total_gate_count"]
                - adjudication["passed_gate_count"]
            ),
        },
        "metrics": metrics,
        "failed_gates": adjudication["failed_gates"],
        "next_research_action": "REMEDIATE_PIT_MEMBERSHIP_TIMESTAMP_FACT_AND_CLUSTERING_CONTRACTS",
        "market_price_value_read": False,
        "price_response_measurement_allowed": False,
        "current_forecast_specification": "REJECTED_FROZEN",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "current_investable_strategy": "NONE",
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json_new(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行预期—消息—价格响应 PIT 入场审计")
    parser.add_argument("--phase", choices=["freeze", "audit", "all", "verify"], default="all")
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"audit", "all"}:
        run_preflight(config, manifest)
    if args.phase == "verify":
        print("PIT 链预检协议与全部冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
