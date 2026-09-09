from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402
from scripts import collect_csi300_pit_fundamental_underreaction_official_facts_v1 as collector  # noqa: E402
from scripts import merge_csi300_pit_official_facts_v1_7_direct_pdf_statistics as prior  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_V1_7_RESIDUAL_DIRECT_PDF_MERGE_V1"
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_7_RESIDUAL_DIRECT_PDF_MERGED_V1_0_0"
)
ADMISSION_PHASE = "RESIDUAL_DIRECT_PDF_STATISTICS"
MERGE_RECEIPT_FIELD = "residual_direct_pdf_statistics_merge_receipt"
TIMEZONE = ZoneInfo("Asia/Shanghai")

CONFIG_PATH = prior.BASE_CONFIG_PATH
BASE_RECEIPT_PATH = prior.OUTPUT_RECEIPT_PATH
RESIDUAL_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_7_residual_direct_pdf_statistics_final/receipt.json"
)

AUDIT_ROOT = prior.AUDIT_ROOT
RAW_ROOT = prior.RAW_ROOT
CHECKPOINT_ROOT = RAW_ROOT / "checkpoints_v1_7_residual_direct_pdf_merged"
OUTPUT_REQUIREMENT_PATH = (
    AUDIT_ROOT / "official_fact_requirement_ledger_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_QUEUE_PATH = (
    AUDIT_ROOT / "official_fact_document_queue_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_PATCH_MANIFEST_PATH = (
    AUDIT_ROOT
    / "official_fact_checkpoint_patch_manifest_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_RESIDUAL_PRIORITY_PATH = (
    AUDIT_ROOT / "official_fact_residual_priority_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_FACTS_PATH = (
    RAW_ROOT / "official_financial_facts_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_DEPENDENCY_PATH = (
    RAW_ROOT / "event_fact_dependency_ledger_v1_7_residual_direct_pdf.parquet"
)
OUTPUT_RECEIPT_PATH = RAW_ROOT / "receipt_v1_7_residual_direct_pdf.json"
OUTPUT_REPORT_PATH = (
    ROOT
    / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
    "OFFICIAL_FACTS_V1_7_RESIDUAL_DIRECT_PDF.md"
)


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(metric["metric_id"]): metric
        for metric in result.get("metrics") or []
        if metric.get("metric_id")
    }


def _metric_source(
    metric: dict[str, Any],
    residual_row: dict[str, Any],
) -> dict[str, Any]:
    try:
        locator = json.loads(str(metric.get("source_locator") or "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"残缺直读指标定位不是JSON：{residual_row['announcement_id']}|"
            f"{metric['metric_id']}"
        ) from error
    context = locator.get("source_context") or {}
    expected_hash = str(context.get("official_pdf_sha256") or "")
    expected_announcement_id = str(context.get("announcement_id") or "")
    matching_sources = [
        source
        for source in residual_row.get("source_results") or []
        if not source.get("error")
        and (
            expected_hash
            and str(source.get("observed_pdf_sha256") or "") == expected_hash
            or expected_announcement_id
            and str(source.get("announcement_id") or "")
            == expected_announcement_id
        )
    ]
    if len(matching_sources) != 1:
        raise RuntimeError(
            f"残缺直读指标来源不唯一：{residual_row['announcement_id']}|"
            f"{metric['metric_id']}|{len(matching_sources)}"
        )
    source = matching_sources[0]
    return {
        "source_kind": str(source.get("source_kind") or "OFFICIAL_PDF"),
        "official_pdf_url": str(source["official_pdf_url"]),
        "official_pdf_announcement_id": str(source["announcement_id"]),
        "official_pdf_sha256": str(source["observed_pdf_sha256"]),
        "official_pdf_size_bytes": int(source["observed_pdf_size_bytes"]),
        "pdf_page_count": int(source["pdf_page_count"]),
        "path": str(source["path"]),
        "admission_phase": ADMISSION_PHASE,
    }


def _checkpoint_output_path(result: dict[str, Any]) -> Path:
    year = str(result["report_period"])[:4]
    return CHECKPOINT_ROOT / year / f"{result['announcement_id']}.json.gz"


def _apply_residual_result(
    residual_row: dict[str, Any],
    *,
    residual_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = ROOT / str(residual_row["merged_checkpoint_path"])
    observed_source_sha256 = prior._sha256_file(source_path)
    expected_source_sha256 = str(residual_row["merged_checkpoint_sha256"])
    if observed_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            f"残缺直读来源检查点哈希漂移：{residual_row['announcement_id']}"
        )
    result = prior._load_json_gzip(source_path)
    before_missing = list(result.get("missing_metrics") or [])
    if before_missing != list(residual_row.get("before_missing_metrics") or []):
        raise RuntimeError(
            f"残缺直读来源缺失状态不一致：{residual_row['announcement_id']}"
        )

    before_metrics = _metric_map(result)
    for metric_id in residual_row.get("removed_metric_ids") or []:
        parser._v1_4._drop_metrics(result, (str(metric_id),))

    metric_sources = dict(result.get("metric_source_documents") or {})
    changed_metrics = list(residual_row.get("changed_metrics") or [])
    for metric in changed_metrics:
        metric_id = str(metric["metric_id"])
        parser._v1_4._put_metric(
            result,
            parser._base.MetricEvidence(**metric),
        )
        metric_sources[metric_id] = _metric_source(metric, residual_row)
    parser._v1_4._refresh_result(result)

    after_missing = list(result.get("missing_metrics") or [])
    expected_after_missing = list(residual_row.get("after_missing_metrics") or [])
    if after_missing != expected_after_missing:
        raise RuntimeError(
            f"残缺直读合并复算不一致：{residual_row['announcement_id']}"
        )
    metric_ids = list(_metric_map(result))
    if len(metric_ids) != len(set(metric_ids)):
        raise RuntimeError(f"公告-指标键重复：{residual_row['announcement_id']}")
    if len(metric_ids) + len(after_missing) != len(parser.REQUIRED_METRICS):
        raise RuntimeError(f"九指标守恒失败：{residual_row['announcement_id']}")

    source_parser_version = str(result.get("parser_version") or "")
    source_checkpoint_status = str(result.get("checkpoint_status") or "")
    document_complete = not after_missing
    result["pre_residual_merge_parser_version"] = source_parser_version
    result["pre_residual_merge_checkpoint_status"] = source_checkpoint_status
    result["parser_version"] = MERGED_PARSER_VERSION
    result["checkpoint_status"] = (
        "PARSED_COMPLETE" if document_complete else "PARSED_INCOMPLETE"
    )
    result["document_terminal"] = True
    result["document_complete"] = document_complete
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if document_complete
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )
    result["missing_metrics"] = after_missing
    result["metric_source_documents"] = metric_sources
    result[MERGE_RECEIPT_FIELD] = {
        "protocol_id": PROTOCOL_ID,
        "source_checkpoint_path": prior._relative(source_path),
        "source_checkpoint_sha256": observed_source_sha256,
        "residual_receipt_sha256": residual_receipt_sha256,
        "newly_admitted_metric_ids": list(
            residual_row.get("newly_admitted_metric_ids") or []
        ),
        "corrected_metric_ids": list(residual_row.get("corrected_metric_ids") or []),
        "removed_metric_ids": list(residual_row.get("removed_metric_ids") or []),
        "final_missing_metrics": after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    result["market_price_read"] = False
    result["future_return_read"] = False
    result["future_label_created"] = False
    result["portfolio_return_calculated"] = False
    result["return_evaluation"] = "NOT_ALLOWED"

    output_path = _checkpoint_output_path(result)
    prior._atomic_write_gzip_json(output_path, result)
    output_sha256 = prior._sha256_file(output_path)
    after_metrics = _metric_map(result)
    manifest_row = {
        "announcement_id": str(result["announcement_id"]),
        "ts_code": str(result["ts_code"]),
        "report_period": str(result["report_period"]),
        "period_type": str(result["period_type"]),
        "source_checkpoint_path": prior._relative(source_path),
        "source_checkpoint_sha256": observed_source_sha256,
        "merged_checkpoint_path": prior._relative(output_path),
        "merged_checkpoint_sha256": output_sha256,
        "metric_count_before": len(before_metrics),
        "metric_count_after": len(after_metrics),
        "newly_admitted_metric_count": len(
            residual_row.get("newly_admitted_metric_ids") or []
        ),
        "corrected_metric_count": len(
            residual_row.get("corrected_metric_ids") or []
        ),
        "removed_metric_count": len(residual_row.get("removed_metric_ids") or []),
        "final_missing_metric_count": len(after_missing),
        "final_missing_metrics_json": json.dumps(
            after_missing,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "document_complete": document_complete,
        "checkpoint_status": result["checkpoint_status"],
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    return result, manifest_row


def _render_report(receipt: dict[str, Any], residual: pd.DataFrame) -> str:
    report = prior._render_report(receipt, residual)
    return report.replace(
        "# 沪深300基本面反应不足：官方PDF逐份统计与覆盖复算",
        "# 沪深300基本面反应不足：残缺官方PDF直读补齐与覆盖复算",
        1,
    )


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_receipt = prior._load_json(BASE_RECEIPT_PATH)
    residual_receipt = prior._load_json(RESIDUAL_RECEIPT_PATH)
    if (
        base_receipt.get("market_price_read") is not False
        or base_receipt.get("future_return_read") is not False
        or residual_receipt.get("market_price_read") is not False
        or residual_receipt.get("future_return_read") is not False
    ):
        raise RuntimeError("输入收据不满足无收益读取门槛")
    if (
        residual_receipt.get("target_exception_count") != 0
        or residual_receipt.get("source_exception_count") != 0
        or residual_receipt.get("removed_unreplaced_metric_count") != 0
    ):
        raise RuntimeError("残缺直读最终收据存在异常或未替换撤回")

    base_requirement_path = prior._verify_base_artifact(
        base_receipt,
        "requirement_ledger",
    )
    base_queue_path = prior._verify_base_artifact(base_receipt, "document_queue")
    base_facts_path = prior._verify_base_artifact(base_receipt, "facts")
    prior._verify_base_artifact(base_receipt, "dependency_ledger")
    base_requirement_runtime = pd.read_parquet(base_requirement_path)
    base_queue = pd.read_parquet(base_queue_path)
    base_facts = pd.read_parquet(base_facts_path)

    residual_rows = list(residual_receipt.get("results") or [])
    residual_map = {
        str(row["announcement_id"]): row for row in residual_rows
    }
    if len(residual_map) != len(residual_rows):
        raise RuntimeError("残缺直读公告键重复")
    queue_ids = set(base_queue["announcement_id"].astype(str))
    if not set(residual_map).issubset(queue_ids):
        raise RuntimeError("残缺直读目标超出当前文档队列")

    residual_receipt_sha256 = prior._sha256_file(RESIDUAL_RECEIPT_PATH)
    merged_checkpoints: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for index, announcement_id in enumerate(sorted(residual_map), start=1):
        checkpoint, manifest_row = _apply_residual_result(
            residual_map[announcement_id],
            residual_receipt_sha256=residual_receipt_sha256,
        )
        merged_checkpoints[announcement_id] = checkpoint
        manifest_rows.append(manifest_row)
        if index % 50 == 0 or index == len(residual_map):
            print(f"合并残缺直读检查点 {index}/{len(residual_map)}", flush=True)

    patch_manifest = pd.DataFrame(manifest_rows).sort_values(
        ["report_period", "ts_code", "announcement_id"],
        kind="stable",
    )
    prior._atomic_write_parquet(OUTPUT_PATCH_MANIFEST_PATH, patch_manifest)

    queue_state = base_queue.copy()
    if "residual_statistics_merge_protocol_id" not in queue_state.columns:
        queue_state["residual_statistics_merge_protocol_id"] = None
    if "residual_statistics_changed_metric_count" not in queue_state.columns:
        queue_state["residual_statistics_changed_metric_count"] = 0
    if "residual_statistics_source_checkpoint_sha256" not in queue_state.columns:
        queue_state["residual_statistics_source_checkpoint_sha256"] = None
    manifest_map = {
        str(row["announcement_id"]): row
        for row in patch_manifest.to_dict("records")
    }
    for announcement_id, manifest_row in manifest_map.items():
        mask = queue_state["announcement_id"].astype(str).eq(announcement_id)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"文档队列公告键不唯一：{announcement_id}")
        checkpoint = merged_checkpoints[announcement_id]
        queue_state.loc[mask, "checkpoint_path"] = manifest_row[
            "merged_checkpoint_path"
        ]
        queue_state.loc[mask, "checkpoint_sha256"] = manifest_row[
            "merged_checkpoint_sha256"
        ]
        queue_state.loc[mask, "checkpoint_status"] = checkpoint["checkpoint_status"]
        queue_state.loc[mask, "document_terminal"] = True
        queue_state.loc[mask, "document_complete"] = bool(
            checkpoint["document_complete"]
        )
        queue_state.loc[mask, "document_status"] = checkpoint["document_status"]
        queue_state.loc[mask, "missing_metric_count"] = len(
            checkpoint["missing_metrics"]
        )
        queue_state.loc[mask, "missing_metrics_json"] = json.dumps(
            checkpoint["missing_metrics"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        queue_state.loc[mask, "residual_statistics_merge_protocol_id"] = PROTOCOL_ID
        queue_state.loc[mask, "residual_statistics_changed_metric_count"] = int(
            manifest_row["newly_admitted_metric_count"]
            + manifest_row["corrected_metric_count"]
        )
        queue_state.loc[
            mask,
            "residual_statistics_source_checkpoint_sha256",
        ] = manifest_row["source_checkpoint_sha256"]
    prior._atomic_write_parquet(OUTPUT_QUEUE_PATH, queue_state)

    patch_ids = set(merged_checkpoints)
    queue_map = {
        str(row["announcement_id"]): row
        for row in queue_state.to_dict("records")
    }
    rebuilt_fact_rows: list[dict[str, Any]] = []
    for announcement_id in sorted(patch_ids):
        rebuilt_fact_rows.extend(
            prior._fact_rows_from_checkpoint(
                merged_checkpoints[announcement_id],
                queue_map[announcement_id],
            )
        )
    rebuilt_facts = pd.DataFrame(
        rebuilt_fact_rows,
        columns=collector.FACT_COLUMNS,
    )
    facts = pd.concat(
        [
            base_facts.loc[
                ~base_facts["announcement_id"].astype(str).isin(patch_ids)
            ],
            rebuilt_facts,
        ],
        ignore_index=True,
    )
    if facts.duplicated(["announcement_id", "metric_id"]).any():
        raise RuntimeError("合并事实公告-指标键不唯一")
    if not facts["metric_value_cny"].map(math.isfinite).all():
        raise RuntimeError("合并事实存在非有限数值")
    if facts["market_price_read"].any() or facts["future_return_read"].any():
        raise RuntimeError("合并事实读取状态漂移")
    facts = facts.sort_values(
        ["report_period", "ts_code", "announcement_id", "metric_id"],
        kind="stable",
    ).reset_index(drop=True)
    observed_metric_counts = facts.groupby("announcement_id")["metric_id"].size()
    expected_metric_counts = queue_state.set_index("announcement_id")[
        "missing_metric_count"
    ].map(lambda value: len(parser.REQUIRED_METRICS) - int(value))
    if not observed_metric_counts.reindex(expected_metric_counts.index).fillna(0).astype(
        int
    ).eq(expected_metric_counts.astype(int)).all():
        raise RuntimeError("合并事实数量与文档缺失计数不守恒")
    prior._atomic_write_parquet(OUTPUT_FACTS_PATH, facts)

    state_columns = [
        "checkpoint_status",
        "document_terminal",
        "document_complete",
        "missing_metric_count",
        "missing_metrics_json",
        "requirement_status",
    ]
    base_requirement = base_requirement_runtime.drop(
        columns=[
            column
            for column in state_columns
            if column in base_requirement_runtime.columns
        ]
    )
    runtime_requirement, dependency = collector.build_event_dependency_ledger(
        base_requirement,
        queue_state,
    )
    coverage = collector.coverage_metrics(dependency, config)
    prior._atomic_write_parquet(OUTPUT_REQUIREMENT_PATH, runtime_requirement)
    prior._atomic_write_parquet(OUTPUT_DEPENDENCY_PATH, dependency)

    residual_priority = prior._residual_priority(runtime_requirement, queue_state)
    prior._atomic_write_parquet(OUTPUT_RESIDUAL_PRIORITY_PATH, residual_priority)

    counterfactual_queue = queue_state.copy()
    incomplete_mask = ~counterfactual_queue["document_complete"]
    counterfactual_queue.loc[incomplete_mask, "checkpoint_status"] = "PARSED_COMPLETE"
    counterfactual_queue.loc[incomplete_mask, "document_complete"] = True
    counterfactual_queue.loc[incomplete_mask, "document_terminal"] = True
    counterfactual_queue.loc[incomplete_mask, "missing_metric_count"] = 0
    counterfactual_queue.loc[incomplete_mask, "missing_metrics_json"] = "[]"
    _, counterfactual_dependency = collector.build_event_dependency_ledger(
        base_requirement,
        counterfactual_queue,
    )
    counterfactual_coverage = collector.coverage_metrics(
        counterfactual_dependency,
        config,
    )

    base_coverage = base_receipt["coverage"]
    admission_checks = prior._admission_checks(coverage, config)
    ready_count = int(dependency["target_event_ready"].sum())
    base_ready_count = int(base_receipt["counts"]["ready_target_event_count"])
    complete_count = int(queue_state["document_complete"].sum())
    terminal_count = int(queue_state["document_terminal"].sum())
    status = (
        str(config["admission"]["pass_status"])
        if coverage["coverage_gate_passed"]
        else str(config["admission"]["coverage_fail_status"])
    )
    counts = {
        "target_event_count": int(len(dependency)),
        "requirement_count": int(len(runtime_requirement)),
        "temporally_available_requirement_count": int(
            runtime_requirement["queued_for_official_pdf"].sum()
        ),
        "queued_document_count": int(len(queue_state)),
        "terminal_document_count": terminal_count,
        "complete_document_count": complete_count,
        "incomplete_terminal_document_count": terminal_count - complete_count,
        "nonterminal_document_count": int(len(queue_state) - terminal_count),
        "download_failed_document_count": int(
            queue_state["checkpoint_status"].eq("DOWNLOAD_FAILED").sum()
        ),
        "fact_row_count": int(len(facts)),
        "base_fact_row_count": int(len(base_facts)),
        "new_fact_row_count": int(len(facts) - len(base_facts)),
        "ready_target_event_count": ready_count,
        "base_ready_target_event_count": base_ready_count,
        "newly_ready_target_event_count": ready_count - base_ready_count,
    }
    residual_missing_metric_counts: Counter[str] = Counter()
    for value in queue_state.loc[
        ~queue_state["document_complete"],
        "missing_metrics_json",
    ]:
        residual_missing_metric_counts.update(json.loads(str(value)))

    receipt: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": "1.0.0",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "status": status,
        "parser_version": MERGED_PARSER_VERSION,
        "counts": counts,
        "checkpoint_status_counts": queue_state["checkpoint_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "requirement_status_counts": runtime_requirement["requirement_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "dependency_status_counts": dependency["dependency_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "coverage": coverage,
        "admission_checks": admission_checks,
        "coverage_change_from_previous_merge": {
            "overall_complete_event_ratio_before": float(
                base_coverage["overall_complete_event_ratio"]
            ),
            "overall_complete_event_ratio_after": float(
                coverage["overall_complete_event_ratio"]
            ),
            "overall_complete_event_ratio_delta": float(
                coverage["overall_complete_event_ratio"]
                - base_coverage["overall_complete_event_ratio"]
            ),
            "publication_year_rows": prior._coverage_delta(
                base_coverage["publication_year_coverage"],
                coverage["publication_year_coverage"],
                ["publication_year"],
            ),
            "industry_rows": prior._coverage_delta(
                base_coverage["industry_coverage"],
                coverage["industry_coverage"],
                ["industry_l1_code", "industry_l1"],
            ),
        },
        "counterfactual_all_remaining_documents_complete": {
            "remaining_document_count": int((~queue_state["document_complete"]).sum()),
            "ready_target_event_count": int(
                counterfactual_dependency["target_event_ready"].sum()
            ),
            "coverage": counterfactual_coverage,
        },
        "residual": {
            "document_count": int(len(residual_priority)),
            "missing_metric_counts": dict(
                sorted(
                    residual_missing_metric_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "single_document_unlockable_event_count": int(
                residual_priority["single_document_unlock_event_count"].sum()
            ),
            "document_with_positive_single_unlock_count": int(
                residual_priority["single_document_unlock_event_count"].gt(0).sum()
            ),
        },
        "provenance": {
            "config_path": prior._relative(CONFIG_PATH),
            "config_sha256": prior._sha256_file(CONFIG_PATH),
            "base_receipt_path": prior._relative(BASE_RECEIPT_PATH),
            "base_receipt_sha256": prior._sha256_file(BASE_RECEIPT_PATH),
            "residual_receipt_path": prior._relative(RESIDUAL_RECEIPT_PATH),
            "residual_receipt_sha256": residual_receipt_sha256,
            "base_artifact_sha256_verified": True,
            "checkpoint_patch_count": int(len(patch_manifest)),
        },
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "signal_score_calculated": False,
        "portfolio_return_calculated": False,
        "positions_generated": False,
        "orders_generated": False,
        "broker_connection_performed": False,
        "trading_authorization": "NO_TRADE",
        "return_evaluation": "NOT_ALLOWED",
        "next_allowed_step": (
            "FREEZE_PHASE_1_ANALYSIS_PROTOCOL_BEFORE_ANY_RETURN_READ"
            if coverage["coverage_gate_passed"]
            else "CONTINUE_OFFICIAL_PDF_FACT_COMPLETION_WITH_RESIDUAL_PRIORITY"
        ),
    }
    report = _render_report(receipt, residual_priority)
    prior._atomic_write_text(OUTPUT_REPORT_PATH, report)
    receipt["artifacts"] = {
        "requirement_ledger_path": prior._relative(OUTPUT_REQUIREMENT_PATH),
        "requirement_ledger_sha256": prior._sha256_file(OUTPUT_REQUIREMENT_PATH),
        "document_queue_path": prior._relative(OUTPUT_QUEUE_PATH),
        "document_queue_sha256": prior._sha256_file(OUTPUT_QUEUE_PATH),
        "checkpoint_patch_manifest_path": prior._relative(
            OUTPUT_PATCH_MANIFEST_PATH
        ),
        "checkpoint_patch_manifest_sha256": prior._sha256_file(
            OUTPUT_PATCH_MANIFEST_PATH
        ),
        "residual_priority_path": prior._relative(OUTPUT_RESIDUAL_PRIORITY_PATH),
        "residual_priority_sha256": prior._sha256_file(
            OUTPUT_RESIDUAL_PRIORITY_PATH
        ),
        "facts_path": prior._relative(OUTPUT_FACTS_PATH),
        "facts_sha256": prior._sha256_file(OUTPUT_FACTS_PATH),
        "dependency_ledger_path": prior._relative(OUTPUT_DEPENDENCY_PATH),
        "dependency_ledger_sha256": prior._sha256_file(OUTPUT_DEPENDENCY_PATH),
        "report_markdown_path": prior._relative(OUTPUT_REPORT_PATH),
        "report_markdown_sha256": prior._sha256_file(OUTPUT_REPORT_PATH),
    }
    prior._atomic_write_json(OUTPUT_RECEIPT_PATH, receipt)
    print(
        "残缺官方PDF直读合并完成："
        f"完整文档={complete_count}/{len(queue_state)}；"
        f"事实={len(facts)}；"
        f"可用事件={ready_count}/{len(dependency)}；"
        f"总体覆盖={coverage['overall_complete_event_ratio']:.6%}；"
        f"覆盖门={coverage['coverage_gate_passed']}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
