from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import a_share_hs_cninfo_periodic_report_metadata_v1 as base  # noqa: E402
from research.a_share_hs_cninfo_periodic_report_event_archive_v2_1 import (  # noqa: E402
    normalize_records_with_chronology,
)
from scripts import (  # noqa: E402
    audit_a_share_hs_cninfo_periodic_report_metadata_v2_0_1_outputs as audit,
)


CONFIG_PATH = ROOT / "config/a_share_hs_cninfo_periodic_report_event_archive_v2_1.yaml"


def project_path(relative: str) -> Path:
    return ROOT / relative


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = project_path(config["artifacts"]["protocol_manifest"])
    if not path.exists():
        raise FileNotFoundError(f"缺少V2.1冻结清单：{path}")
    manifest = load_json(path)
    if manifest.get("status") != "FROZEN_EVENT_ARCHIVE_V2_1_BEFORE_MARKET_OUTCOMES":
        raise RuntimeError("V2.1冻结清单状态错误")
    checks = []
    for item in [*manifest["files"], *manifest["references"]]:
        actual = base.sha256_file(project_path(item["path"]))
        checks.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matched": actual == item["sha256"],
            }
        )
    if not all(item["matched"] for item in checks):
        raise RuntimeError("V2.1核心文件或引用证据哈希不一致")
    return {
        "manifest": path.relative_to(ROOT).as_posix(),
        "manifest_sha256": base.sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "checks": checks,
        "all_matched": True,
    }


def reconstruct_raw_records(
    config: dict[str, Any],
    *,
    source_receipt: dict[str, Any],
    inventory: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_by_stream, convergence = audit.recompute_streams(source_receipt, inventory)
    base_receipt = load_json(project_path(config["references"]["base_receipt"]))
    first_config = yaml.safe_load(
        project_path(config["references"]["first_correction_config"]).read_text(
            encoding="utf-8"
        )
    )
    source_config = yaml.safe_load(
        project_path(config["references"]["v2_0_1_config"]).read_text(
            encoding="utf-8"
        )
    )
    prior_root = project_path(
        first_config["artifacts"]["correction_checkpoint_root"]
    )
    categories = list(first_config["sharding"]["categories"])
    corrected_days = set(source_config["partition_proof"]["corrected_days"])
    records_by_id: dict[str, dict[str, Any]] = {}
    duplicate_across_partitions = 0
    identity_conflicts = []
    base_complete_months = 0
    day_partition_months = 0
    loaded_day_count = 0

    def add_partition(records: list[dict[str, Any]], label: str) -> None:
        nonlocal duplicate_across_partitions
        partition_ids, partition_records = audit.ids_and_records(records)
        overlap = partition_ids & set(records_by_id)
        duplicate_across_partitions += len(overlap)
        for identifier in overlap:
            previous = records_by_id[identifier]
            current = partition_records[identifier]
            keys = [
                "secCode",
                "announcementTitle",
                "announcementTime",
                "adjunctUrl",
            ]
            if any(previous.get(key) != current.get(key) for key in keys):
                identity_conflicts.append(f"{label}:{identifier}")
        for identifier, record in partition_records.items():
            records_by_id.setdefault(identifier, record)

    for interval in base_receipt["interval_receipts"]:
        if interval["status"] == "COMPLETE":
            payload = audit.load_gzip_json(project_path(interval["checkpoint_file"]))
            if payload.get("status") != "COMPLETE":
                raise ValueError(f"基础月检查点状态错误：{interval['label']}")
            records = payload["records"]
            if len(records) != int(payload["declared_total"]):
                raise ValueError(f"基础月检查点行数错误：{interval['label']}")
            add_partition(records, interval["label"])
            base_complete_months += 1
            continue
        day_partition_months += 1
        cursor = date.fromisoformat(interval["start_date"])
        end_date = date.fromisoformat(interval["end_date"])
        while cursor <= end_date:
            day_key = cursor.isoformat()
            if day_key in corrected_days:
                category_records = [
                    records_by_stream[(day_key, category)] for category in categories
                ]
                category_counts = [len(records) for records in category_records]
                merged = [record for records in category_records for record in records]
                merged_ids, merged_by_id = audit.ids_and_records(merged)
                if len(merged_ids) != sum(category_counts):
                    raise ValueError(f"修正日跨类别公告ID重复：{day_key}")
                records = [merged_by_id[key] for key in sorted(merged_by_id)]
            else:
                records = audit.load_prior_day_records(
                    cursor,
                    checkpoint_root=prior_root,
                    categories=categories,
                )
            add_partition(records, day_key)
            loaded_day_count += 1
            cursor = date.fromordinal(cursor.toordinal() + 1)

    evidence = {
        "convergence_stream_count": convergence["stream_count"],
        "convergence_passed_stream_count": convergence["passed_stream_count"],
        "checkpoint_hash_check_count": convergence["checkpoint_hash_check_count"],
        "all_checkpoint_hashes_match": convergence["all_checkpoint_hashes_match"],
        "base_complete_month_count": base_complete_months,
        "day_partition_month_count": day_partition_months,
        "loaded_day_count": loaded_day_count,
        "raw_record_count": len(records_by_id),
        "duplicate_across_partitions": duplicate_across_partitions,
        "identity_conflicts": identity_conflicts,
    }
    return [records_by_id[key] for key in sorted(records_by_id)], evidence


def replacement_evidence(
    invalid: pd.DataFrame, events: pd.DataFrame
) -> list[dict[str, Any]]:
    results = []
    for row in invalid.itertuples(index=False):
        replacements = events.loc[
            events["ts_code"].eq(row.ts_code)
            & events["report_period"].eq(row.report_period)
        ]
        results.append(
            {
                "invalid_announcement_id": str(row.announcement_id),
                "ts_code": str(row.ts_code),
                "report_period": pd.Timestamp(row.report_period).date().isoformat(),
                "invalid_event_publication_date": pd.Timestamp(
                    row.official_url_date
                ).date().isoformat(),
                "replacement_found": len(replacements) == 1,
                "replacement_announcement_id": (
                    str(replacements.iloc[0]["announcement_id"])
                    if len(replacements) == 1
                    else None
                ),
                "replacement_event_publication_date": (
                    pd.Timestamp(
                        replacements.iloc[0]["event_publication_date"]
                    ).date().isoformat()
                    if len(replacements) == 1
                    else None
                ),
            }
        )
    return results


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    timezone = ZoneInfo(config["protocol"]["timezone"])
    started_at = datetime.now(timezone)
    frozen = verify_manifest(config)
    source_config = yaml.safe_load(
        project_path(config["references"]["v2_0_1_config"]).read_text(
            encoding="utf-8"
        )
    )
    base_config = yaml.safe_load(
        project_path(config["references"]["base_config"]).read_text(encoding="utf-8")
    )
    source_receipt = load_json(project_path(config["references"]["v2_0_1_receipt"]))
    source_report = load_json(project_path(config["references"]["v2_0_1_report"]))
    failed_audit = load_json(
        project_path(config["references"]["failed_independent_audit"])
    )
    inventory = pd.read_csv(
        project_path(config["references"]["v2_0_1_inventory"]),
        dtype=str,
        keep_default_na=False,
    )
    raw_records, reconstruction = reconstruct_raw_records(
        config,
        source_receipt=source_receipt,
        inventory=inventory,
    )
    all_normalized, events, invalid = normalize_records_with_chronology(
        raw_records,
        config=base_config,
        retrieved_at=datetime.now(timezone),
    )
    replacements = replacement_evidence(invalid, events)

    exclusion_path = project_path(config["artifacts"]["chronology_exclusions"])
    base.atomic_parquet(invalid, exclusion_path)
    event_path = project_path(config["artifacts"]["normalized_event_archive"])
    known_anomaly_ids = sorted(str(value) for value in config["event_contract"]["anomaly_ids"])
    actual_anomaly_ids = sorted(invalid["announcement_id"].astype(str).tolist())
    failed_audit_gates = [
        item["gate_id"] for item in failed_audit["gates"] if not item["passed"]
    ]
    source_evidence_valid = (
        source_receipt["status"]
        == "PASS_OFFICIAL_PERIODIC_REPORT_METADATA_ARCHIVE_V2_0_1_COMPLETE"
        and source_report["passed_gate_count"] == source_report["total_gate_count"] == 8
        and failed_audit["status"]
        == "FAILED_INDEPENDENT_OFFICIAL_PERIODIC_REPORT_METADATA_V2_0_1_AUDIT"
        and failed_audit["passed_gate_count"] == 10
        and failed_audit["total_gate_count"] == 11
        and failed_audit_gates == ["A7_EVENT_PARQUET_CONTRACT"]
        and int(
            failed_audit["event_validation"]["publication_before_report_period_count"]
        )
        == int(config["event_contract"]["expected_known_anomaly_count"])
    )
    raw_valid = (
        reconstruction["convergence_stream_count"] == 56
        and reconstruction["convergence_passed_stream_count"] == 56
        and reconstruction["all_checkpoint_hashes_match"]
        and reconstruction["base_complete_month_count"]
        == int(config["raw_reconstruction"]["base_complete_month_count"])
        and reconstruction["day_partition_month_count"]
        == int(config["raw_reconstruction"]["day_partition_month_count"])
        and reconstruction["raw_record_count"]
        == int(config["raw_reconstruction"]["expected_raw_record_count"])
        and reconstruction["duplicate_across_partitions"] == 0
        and not reconstruction["identity_conflicts"]
    )
    anomaly_valid = (
        len(invalid) == int(config["event_contract"]["expected_known_anomaly_count"])
        and actual_anomaly_ids == known_anomaly_ids
        and invalid["chronology_status"]
        .eq("REJECTED_EVENT_PUBLICATION_BEFORE_REPORT_PERIOD")
        .all()
        and all(item["replacement_found"] for item in replacements)
    )
    cutoff = pd.Timestamp(config["protocol"]["evidence_cutoff"])
    event_valid = (
        not events.empty
        and events.isna().sum().sum() == 0
        and events["announcement_id"].is_unique
        and not events.duplicated(["ts_code", "report_period"]).any()
        and set(events["period_type"]) == {"FY", "H1", "Q1", "Q3"}
        and events["event_publication_date"].ge(events["report_period"]).all()
        and events["event_publication_date"].le(cutoff).all()
        and events["official_internal_date_equal"].all()
        and events["official_pdf_url"]
        .str.startswith("https://static.cninfo.com.cn/")
        .all()
        and set(events["announcement_id"].astype(str)).issubset(
            {str(item["announcementId"]) for item in raw_records}
        )
    )

    pdf_checks = []
    pdf_error = None
    if event_valid:
        sample = base.deterministic_pdf_sample(
            events, int(config["validation"]["pdf_sample_size"])
        )
        session = None
        try:
            session = base.build_session(base_config["source"]["search_page_url"])
            for url in sample["official_pdf_url"].tolist():
                pdf_checks.append(
                    base.verify_pdf_prefix(
                        session,
                        str(url),
                        accepted_host=config["validation"]["official_pdf_host_required"],
                        timeout_seconds=90,
                    )
                )
        except Exception as error:  # noqa: BLE001
            pdf_error = str(error)
        finally:
            if session is not None:
                session.close()
    pdf_verified = sum(
        bool(item["official_host"])
        and item["status_code"] == 200
        and bool(item["pdf_header"])
        for item in pdf_checks
    )
    pdf_valid = (
        pdf_error is None
        and len(pdf_checks) == int(config["validation"]["pdf_sample_size"])
        and pdf_verified == len(pdf_checks)
    )
    gates = [
        {
            "gate_id": "G1_FROZEN_REFERENCES_VERIFIED",
            "description": "V2.1核心文件、V2.0.1证据、失败审计及两份PDF诊断哈希全部匹配",
            "passed": bool(frozen["all_matched"]),
        },
        {
            "gate_id": "G2_SOURCE_FAILURE_ISOLATED_TO_CHRONOLOGY",
            "description": "V2.0.1采集8/8通过且独立审计唯一失败为两条事件日早于报告期",
            "passed": source_evidence_valid,
        },
        {
            "gate_id": "G3_RAW_ARCHIVE_RECONSTRUCTED",
            "description": "307097个原始公告ID由56个收敛流、97个月和31个月日分区无重复重建",
            "passed": raw_valid,
        },
        {
            "gate_id": "G4_CHRONOLOGY_ANOMALIES_IDENTIFIED_AND_REPLACED",
            "description": "通用时序规则识别已知两条异常且每个证券报告期找到后续有效事件",
            "passed": anomaly_valid,
        },
        {
            "gate_id": "G5_FINAL_EVENT_CONTRACT_VALID",
            "description": "事件键唯一无空值，日期不早于报告期且不晚于截止日，官方日期和主机有效",
            "passed": event_valid,
        },
        {
            "gate_id": "G6_FIXED_PDF_SAMPLE_VERIFIED",
            "description": "修复后事件档案固定32份PDF抽检全部通过",
            "passed": pdf_valid,
        },
        {
            "gate_id": "G7_DISCOVERY_DATA_ONLY_GOVERNANCE",
            "description": "未读取市场价格或未来收益，未计算信号组合仓位订单",
            "passed": True,
        },
    ]
    passed_count = sum(item["passed"] for item in gates)
    status = (
        "PASS_OFFICIAL_PERIODIC_REPORT_EVENT_ARCHIVE_V2_1_CHRONOLOGY_VALID"
        if passed_count == len(gates)
        else "NO_VIEW_OFFICIAL_PERIODIC_REPORT_EVENT_ARCHIVE_V2_1_INVALID"
    )
    event_hash = None
    if status.startswith("PASS_"):
        base.atomic_parquet(events, event_path)
        event_hash = base.sha256_file(event_path)
    completed_at = datetime.now(timezone)
    receipt = {
        "protocol_id": config["protocol"]["protocol_id"],
        "version": config["protocol"]["version"],
        "status": status,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "frozen_verification": frozen,
        "source_status": source_receipt["status"],
        "reconstruction": reconstruction,
        "raw_record_count": len(raw_records),
        "all_normalized_record_count": len(all_normalized),
        "chronology_exclusion_count": len(invalid),
        "chronology_exclusion_ids": actual_anomaly_ids,
        "replacement_evidence": replacements,
        "event_count": len(events),
        "event_symbol_count": int(events["ts_code"].nunique()),
        "pdf_checks": pdf_checks,
        "pdf_error": pdf_error,
        "governance": {
            "market_price_read": False,
            "future_return_read": False,
            "signal_calculated": False,
            "ic_calculated": False,
            "portfolio_return_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
        },
        "outputs": {
            "normalized_event_archive": {
                "path": event_path.relative_to(ROOT).as_posix(),
                "sha256": event_hash,
            },
            "chronology_exclusions": {
                "path": exclusion_path.relative_to(ROOT).as_posix(),
                "sha256": base.sha256_file(exclusion_path),
            },
        },
    }
    receipt["receipt_content_sha256"] = base.canonical_hash(receipt)
    receipt_path = project_path(config["artifacts"]["run_receipt"])
    base.atomic_json(receipt_path, receipt)
    report = {
        "protocol_id": config["protocol"]["protocol_id"],
        "status": status,
        "evidence_cutoff": config["protocol"]["evidence_cutoff"],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "metrics": {
            "raw_record_count": len(raw_records),
            "chronology_exclusion_count": len(invalid),
            "event_count": len(events),
            "event_symbol_count": int(events["ts_code"].nunique()),
            "pdf_sample_count": len(pdf_checks),
            "pdf_verified_count": pdf_verified,
        },
        "gates": gates,
        "passed_gate_count": passed_count,
        "total_gate_count": len(gates),
        "replacement_evidence": replacements,
        "artifacts": {
            "normalized_event_archive": config["artifacts"]["normalized_event_archive"],
            "chronology_exclusions": config["artifacts"]["chronology_exclusions"],
            "run_receipt": config["artifacts"]["run_receipt"],
        },
        "governance": receipt["governance"],
    }
    base.atomic_json(project_path(config["artifacts"]["report_json"]), report)
    lines = [
        "# A股沪深巨潮定期报告事件档案 V2.1",
        "",
        f"- 状态：`{status}`",
        f"- 通过门数：{passed_count}/{len(gates)}",
        f"- 原始公告：{len(raw_records)}",
        f"- 时序异常排除：{len(invalid)}",
        f"- 最终事件：{len(events)}，覆盖证券：{events['ts_code'].nunique()}",
        f"- PDF抽检：{pdf_verified}/{len(pdf_checks)}",
        "",
        "## 门禁",
        "",
    ]
    lines.extend(
        f"- {'PASS' if item['passed'] else 'FAIL'} `{item['gate_id']}`：{item['description']}"
        for item in gates
    )
    lines.extend(
        [
            "",
            "## 替代事件",
            "",
        ]
    )
    lines.extend(
        f"- `{item['invalid_announcement_id']}` → "
        f"`{item['replacement_announcement_id']}`，"
        f"事件日 `{item['replacement_event_publication_date']}`"
        for item in replacements
    )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本产物只证明事件日期档案通过逻辑时序和官方来源门禁。",
            "- 未读取市场价格、未来收益、成本、持仓或订单。",
            "",
        ]
    )
    markdown_path = project_path(config["artifacts"]["report_markdown"])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "状态": status,
                "通过门数": f"{passed_count}/{len(gates)}",
                "原始公告": len(raw_records),
                "时序排除": len(invalid),
                "最终事件": len(events),
                "替代事件": replacements,
                "报告": config["artifacts"]["report_markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
