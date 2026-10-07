from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/a_share_hs_cninfo_periodic_report_metadata_v2_0_1.yaml"
AUDIT_JSON = ROOT / "reports/audit/A_SHARE_HS_CNINFO_PERIODIC_REPORT_METADATA_V2_0_1_OUTPUT_AUDIT.json"
AUDIT_MARKDOWN = ROOT / "reports/audit/A_SHARE_HS_CNINFO_PERIODIC_REPORT_METADATA_V2_0_1_OUTPUT_AUDIT.md"
CATEGORY_LABELS = {
    "category_ndbg_szsh": "ndbg",
    "category_bndbg_szsh": "bndbg",
    "category_yjdbg_szsh": "yjdbg",
    "category_sjdbg_szsh": "sjdbg",
}


def project_path(relative: str) -> Path:
    return ROOT / relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = project_path(config["artifacts"]["protocol_manifest"])
    manifest = load_json(path)
    checks = []
    for item in [*manifest["files"], *manifest["references"]]:
        actual = sha256_file(project_path(item["path"]))
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "actual": actual,
                "matched": actual == item["sha256"],
            }
        )
    return {
        "status": manifest.get("status"),
        "file_count": len(manifest["files"]),
        "reference_count": len(manifest["references"]),
        "all_hashes_match": all(item["matched"] for item in checks),
        "checks": checks,
    }


def ids_and_records(records: list[dict[str, Any]]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get("announcementId") or "")
        if not identifier:
            raise ValueError("检查点存在空公告ID")
        if identifier in by_id:
            existing = by_id[identifier]
            stable_fields = [
                "secCode",
                "secName",
                "orgId",
                "announcementTitle",
                "announcementTime",
                "adjunctUrl",
            ]
            if any(existing.get(key) != record.get(key) for key in stable_fields):
                raise ValueError(f"同一公告ID身份字段冲突：{identifier}")
            continue
        by_id[identifier] = record
    return set(by_id), by_id


def recompute_streams(
    receipt: dict[str, Any], inventory: pd.DataFrame
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    records_by_stream: dict[tuple[str, str], list[dict[str, Any]]] = {}
    stream_checks = []
    checkpoint_hash_checks = []
    for stream in receipt["stream_receipts"]:
        query_date = stream["query_date"]
        category = stream["category"]
        selected = inventory.loc[
            inventory["query_date"].eq(query_date)
            & inventory["category"].eq(category)
        ].copy()
        direction_unions: dict[str, set[str]] = {}
        record_by_id: dict[str, dict[str, Any]] = {}
        declared_values: list[int] = []
        pass_integrity = True
        for direction in ["asc", "desc"]:
            direction_rows = selected.loc[selected["direction"].eq(direction)].sort_values(
                "round_number"
            )
            ids: set[str] = set()
            for row in direction_rows.itertuples(index=False):
                path = project_path(str(row.checkpoint_file))
                actual_hash = sha256_file(path)
                checkpoint_hash_checks.append(actual_hash == str(row.checkpoint_sha256))
                payload = load_gzip_json(path)
                pass_integrity = pass_integrity and payload.get("status") == "PASS_CAPTURED"
                pass_integrity = pass_integrity and bool(payload.get("reached_last_page"))
                pass_integrity = pass_integrity and payload.get("error") is None
                declared_values.extend(int(value) for value in payload["declared_totals"])
                pass_ids, pass_records = ids_and_records(payload["records"])
                ids.update(pass_ids)
                for identifier, record in pass_records.items():
                    record_by_id.setdefault(identifier, record)
            direction_unions[direction] = ids
        maximum_declared = max(declared_values, default=None)
        unions_equal = direction_unions["asc"] == direction_unions["desc"]
        union_ids = direction_unions["asc"] if unions_equal else set()
        count_match = maximum_declared is not None and len(union_ids) == maximum_declared
        recorded = stream["evaluation"]
        recorded_hash_match = (
            canonical_hash(sorted(direction_unions["asc"]))
            == recorded["ascending_union_sha256"]
            and canonical_hash(sorted(direction_unions["desc"]))
            == recorded["descending_union_sha256"]
        )
        passed = (
            stream["status"] == "CONVERGED"
            and pass_integrity
            and unions_equal
            and count_match
            and recorded_hash_match
            and len(union_ids) == int(recorded["union_unique_id_count"])
        )
        stream_checks.append(
            {
                "query_date": query_date,
                "category": category,
                "passed": passed,
                "union_count": len(union_ids),
                "maximum_declared_total": maximum_declared,
                "legal_empty_stream": bool(recorded.get("legal_empty_stream")),
            }
        )
        records_by_stream[(query_date, category)] = [
            record_by_id[identifier] for identifier in sorted(union_ids)
        ]
    return records_by_stream, {
        "stream_count": len(stream_checks),
        "passed_stream_count": sum(item["passed"] for item in stream_checks),
        "legal_empty_stream_count": sum(
            item["legal_empty_stream"] for item in stream_checks
        ),
        "all_checkpoint_hashes_match": all(checkpoint_hash_checks),
        "checkpoint_hash_check_count": len(checkpoint_hash_checks),
        "stream_checks": stream_checks,
    }


def load_prior_day_records(
    query_date: date,
    *,
    checkpoint_root: Path,
    categories: list[str],
) -> list[dict[str, Any]]:
    combined = checkpoint_root / f"{query_date.isoformat()}__combined.json.gz"
    if combined.exists():
        payload = load_gzip_json(combined)
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"前序合并日检查点状态错误：{combined}")
        records = payload["records"]
        if len(records) != int(payload["declared_total"]):
            raise ValueError(f"前序合并日检查点行数错误：{combined}")
        return records
    merged: list[dict[str, Any]] = []
    for category in categories:
        path = checkpoint_root / (
            f"{query_date.isoformat()}__{CATEGORY_LABELS[category]}.json.gz"
        )
        payload = load_gzip_json(path)
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"前序类别日检查点状态错误：{path}")
        records = payload["records"]
        if len(records) != int(payload["declared_total"]):
            raise ValueError(f"前序类别日检查点行数错误：{path}")
        merged.extend(records)
    _, by_id = ids_and_records(merged)
    return [by_id[key] for key in sorted(by_id)]


def reconstruct_raw_archive_ids(
    config: dict[str, Any],
    records_by_stream: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    base_receipt = load_json(project_path(config["references"]["base_receipt"]))
    first_config = yaml.safe_load(
        project_path(config["references"]["first_correction_config"]).read_text(
            encoding="utf-8"
        )
    )
    prior_root = project_path(
        first_config["artifacts"]["correction_checkpoint_root"]
    )
    categories = list(first_config["sharding"]["categories"])
    corrected_days = set(config["partition_proof"]["corrected_days"])
    all_ids: set[str] = set()
    duplicate_across_partitions = 0
    complete_months = 0
    day_partition_months = 0
    loaded_day_count = 0

    def add_partition(records: list[dict[str, Any]]) -> None:
        nonlocal duplicate_across_partitions
        partition_ids, _ = ids_and_records(records)
        duplicate_across_partitions += len(partition_ids & all_ids)
        all_ids.update(partition_ids)

    for interval in base_receipt["interval_receipts"]:
        if interval["status"] == "COMPLETE":
            payload = load_gzip_json(project_path(interval["checkpoint_file"]))
            if payload.get("status") != "COMPLETE":
                raise ValueError(f"基础月检查点状态错误：{interval['label']}")
            records = payload["records"]
            if len(records) != int(payload["declared_total"]):
                raise ValueError(f"基础月检查点行数错误：{interval['label']}")
            add_partition(records)
            complete_months += 1
            continue
        day_partition_months += 1
        cursor = date.fromisoformat(interval["start_date"])
        end_date = date.fromisoformat(interval["end_date"])
        while cursor <= end_date:
            day_key = cursor.isoformat()
            if day_key in corrected_days:
                merged = [
                    record
                    for category in categories
                    for record in records_by_stream[(day_key, category)]
                ]
                _, by_id = ids_and_records(merged)
                records = [by_id[key] for key in sorted(by_id)]
            else:
                records = load_prior_day_records(
                    cursor,
                    checkpoint_root=prior_root,
                    categories=categories,
                )
            add_partition(records)
            loaded_day_count += 1
            cursor = date.fromordinal(cursor.toordinal() + 1)
    return {
        "unique_raw_id_count": len(all_ids),
        "duplicate_ids_across_partitions": duplicate_across_partitions,
        "base_complete_month_count": complete_months,
        "day_partition_month_count": day_partition_months,
        "loaded_day_count": loaded_day_count,
        "all_ids": all_ids,
    }


def validate_event_archive(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    receipt: dict[str, Any],
    raw_ids: set[str],
) -> dict[str, Any]:
    required_columns = {
        "announcement_id",
        "ts_code",
        "period_type",
        "report_period",
        "event_publication_date",
        "official_timestamp_date",
        "official_internal_date_equal",
        "official_pdf_url",
        "query_interval",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    expected_period_end = {
        "FY": (12, 31),
        "H1": (6, 30),
        "Q1": (3, 31),
        "Q3": (9, 30),
    }
    period_end_valid = pd.Series(True, index=frame.index)
    for period_type, (month, day_value) in expected_period_end.items():
        mask = frame["period_type"].eq(period_type)
        period_end_valid.loc[mask] = (
            frame.loc[mask, "report_period"].dt.month.eq(month)
            & frame.loc[mask, "report_period"].dt.day.eq(day_value)
        )
    publication_month = frame["event_publication_date"].dt.strftime("%Y-%m")
    cutoff = pd.Timestamp(config["protocol"]["evidence_cutoff"])
    event_ids = set(frame["announcement_id"].astype(str))
    return {
        "row_count": len(frame),
        "symbol_count": int(frame["ts_code"].nunique()),
        "missing_columns": missing_columns,
        "null_cell_count": int(frame.isna().sum().sum()),
        "duplicate_announcement_id_count": int(
            frame.duplicated(["announcement_id"]).sum()
        ),
        "duplicate_ts_code_report_period_count": int(
            frame.duplicated(["ts_code", "report_period"]).sum()
        ),
        "period_types": sorted(frame["period_type"].unique().tolist()),
        "period_end_invalid_count": int((~period_end_valid).sum()),
        "publication_before_report_period_count": int(
            (frame["event_publication_date"] < frame["report_period"]).sum()
        ),
        "publication_after_cutoff_count": int(
            (frame["event_publication_date"] > cutoff).sum()
        ),
        "query_interval_mismatch_count": int(
            frame["query_interval"].ne(publication_month).sum()
        ),
        "internal_date_mismatch_count": int(
            (~frame["official_internal_date_equal"]).sum()
        ),
        "official_pdf_host_mismatch_count": int(
            (~frame["official_pdf_url"].str.startswith("https://static.cninfo.com.cn/")).sum()
        ),
        "event_id_missing_from_reconstructed_raw_count": len(event_ids - raw_ids),
        "receipt_event_count_match": len(frame) == int(receipt["event_count"]),
        "receipt_symbol_count_match": int(frame["ts_code"].nunique())
        == int(receipt["event_symbol_count"]),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    receipt_path = project_path(config["artifacts"]["acquisition_receipt"])
    report_path = project_path(config["artifacts"]["report_json"])
    event_path = project_path(config["artifacts"]["normalized_event_archive"])
    inventory_path = project_path(config["artifacts"]["convergence_inventory"])
    receipt = load_json(receipt_path)
    report = load_json(report_path)
    manifest = verify_manifest(config)
    inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    records_by_stream, convergence = recompute_streams(receipt, inventory)
    raw_reconstruction = reconstruct_raw_archive_ids(config, records_by_stream)
    raw_ids = raw_reconstruction.pop("all_ids")
    events = pd.read_parquet(event_path)
    event_validation = validate_event_archive(
        events,
        config=config,
        receipt=receipt,
        raw_ids=raw_ids,
    )

    receipt_copy = copy_for_hash = json.loads(json.dumps(receipt, ensure_ascii=False))
    recorded_receipt_hash = copy_for_hash.pop("receipt_content_sha256")
    actual_receipt_hash = canonical_hash(receipt_copy)
    day_reconciliation_pass = (
        len(receipt["day_reconciliation"]) == 14
        and all(item["passed"] for item in receipt["day_reconciliation"])
        and all(
            len(item["probes"])
            == int(config["convergence"]["combined_total_probe_repetitions_per_day"])
            and item["all_probes_complete"]
            for item in receipt["combined_total_probes"]
        )
    )
    partition_pass = (
        raw_reconstruction["base_complete_month_count"] == 97
        and raw_reconstruction["day_partition_month_count"] == 31
        and raw_reconstruction["duplicate_ids_across_partitions"] == 0
        and raw_reconstruction["unique_raw_id_count"]
        == int(receipt["global_unique_record_count"])
        == 307097
    )
    event_contract_pass = (
        event_validation["row_count"] == 173449
        and event_validation["symbol_count"] == 5435
        and not event_validation["missing_columns"]
        and event_validation["null_cell_count"] == 0
        and event_validation["duplicate_announcement_id_count"] == 0
        and event_validation["duplicate_ts_code_report_period_count"] == 0
        and event_validation["period_types"] == ["FY", "H1", "Q1", "Q3"]
        and event_validation["period_end_invalid_count"] == 0
        and event_validation["publication_before_report_period_count"] == 0
        and event_validation["publication_after_cutoff_count"] == 0
        and event_validation["query_interval_mismatch_count"] == 0
        and event_validation["event_id_missing_from_reconstructed_raw_count"] == 0
        and event_validation["receipt_event_count_match"]
        and event_validation["receipt_symbol_count_match"]
    )
    date_and_pdf_path_pass = (
        event_validation["internal_date_mismatch_count"] == 0
        and event_validation["official_pdf_host_mismatch_count"] == 0
    )
    pdf_checks = receipt["pdf_checks"]
    pdf_pass = (
        len(pdf_checks) == 32
        and all(
            item["official_host"]
            and item["status_code"] == 200
            and item["pdf_header"]
            for item in pdf_checks
        )
    )
    governance = receipt["governance"]
    governance_pass = all(value is False for value in governance.values())
    gates = [
        {
            "gate_id": "A1_FROZEN_HASHES_MATCH",
            "passed": manifest["all_hashes_match"],
            "detail": f"{manifest['file_count']}个核心文件、{manifest['reference_count']}份引用",
        },
        {
            "gate_id": "A2_RECEIPT_AND_REPORT_SELF_CONSISTENT",
            "passed": recorded_receipt_hash == actual_receipt_hash
            and receipt["status"]
            == "PASS_OFFICIAL_PERIODIC_REPORT_METADATA_ARCHIVE_V2_0_1_COMPLETE"
            and report["status"] == receipt["status"]
            and report["passed_gate_count"] == report["total_gate_count"] == 8,
            "detail": "收据内容哈希、状态和8/8原验收门",
        },
        {
            "gate_id": "A3_OUTPUT_FILE_HASHES_MATCH",
            "passed": sha256_file(event_path)
            == receipt["outputs"]["normalized_event_archive"]["sha256"]
            and sha256_file(inventory_path)
            == receipt["outputs"]["convergence_inventory"]["sha256"],
            "detail": "事件Parquet与收敛清单",
        },
        {
            "gate_id": "A4_CONVERGENCE_CHECKPOINTS_RECOMPUTED",
            "passed": convergence["stream_count"] == 56
            and convergence["passed_stream_count"] == 56
            and convergence["legal_empty_stream_count"] == 23
            and convergence["all_checkpoint_hashes_match"],
            "detail": (
                f"{convergence['passed_stream_count']}/56流；"
                f"{convergence['legal_empty_stream_count']}个合法空集；"
                f"{convergence['checkpoint_hash_check_count']}个检查点哈希"
            ),
        },
        {
            "gate_id": "A5_ALL_14_DAYS_RECONCILED",
            "passed": day_reconciliation_pass,
            "detail": "14日类别并集及原V2八次总数探针",
        },
        {
            "gate_id": "A6_RAW_ARCHIVE_RECONSTRUCTED",
            "passed": partition_pass,
            "detail": (
                f"97完整月+31日分区月；"
                f"{raw_reconstruction['unique_raw_id_count']}个全局唯一原始ID"
            ),
        },
        {
            "gate_id": "A7_EVENT_PARQUET_CONTRACT",
            "passed": event_contract_pass,
            "detail": "173449事件、5435证券、公告ID与证券报告期键唯一",
        },
        {
            "gate_id": "A8_EVENT_IDS_SUBSET_OF_RAW_ARCHIVE",
            "passed": event_validation["event_id_missing_from_reconstructed_raw_count"] == 0,
            "detail": "所有归一化事件公告ID均可回溯到重建原始分区",
        },
        {
            "gate_id": "A9_OFFICIAL_DATE_AND_HOST_CONTRACT",
            "passed": date_and_pdf_path_pass,
            "detail": "内部日期0不一致，官方PDF主机0不一致",
        },
        {
            "gate_id": "A10_FIXED_PDF_SAMPLE",
            "passed": pdf_pass,
            "detail": "32/32官方PDF状态、主机和文件头",
        },
        {
            "gate_id": "A11_DISCOVERY_ONLY_GOVERNANCE",
            "passed": governance_pass,
            "detail": "价格、未来收益、信号、组合、仓位、订单和券商连接均为false",
        },
    ]
    passed_count = sum(item["passed"] for item in gates)
    status = (
        "PASS_INDEPENDENT_OFFICIAL_PERIODIC_REPORT_METADATA_V2_0_1_AUDIT"
        if passed_count == len(gates)
        else "FAILED_INDEPENDENT_OFFICIAL_PERIODIC_REPORT_METADATA_V2_0_1_AUDIT"
    )
    audit = {
        "audit_id": "A_SHARE_HS_CNINFO_PERIODIC_REPORT_METADATA_V2_0_1_OUTPUT_AUDIT",
        "status": status,
        "source_status": receipt["status"],
        "passed_gate_count": passed_count,
        "total_gate_count": len(gates),
        "gates": gates,
        "manifest_verification": manifest,
        "convergence_recomputation": convergence,
        "raw_reconstruction": raw_reconstruction,
        "event_validation": event_validation,
        "receipt_hash": {
            "recorded": recorded_receipt_hash,
            "actual": actual_receipt_hash,
            "matched": recorded_receipt_hash == actual_receipt_hash,
        },
        "audited_artifacts": {
            "event_archive": config["artifacts"]["normalized_event_archive"],
            "receipt": config["artifacts"]["acquisition_receipt"],
            "report": config["artifacts"]["report_json"],
            "inventory": config["artifacts"]["convergence_inventory"],
        },
    }
    atomic_json(AUDIT_JSON, audit)
    lines = [
        "# A股沪深巨潮官方定期报告元数据 V2.0.1 独立输出审计",
        "",
        f"- 状态：`{status}`",
        f"- 通过门数：{passed_count}/{len(gates)}",
        "- 审计方法：独立读取冻结清单、gzip检查点、收敛清单、收据和Parquet；未调用采集器主流程。",
        "",
        "## 门禁",
        "",
    ]
    lines.extend(
        f"- {'PASS' if item['passed'] else 'FAIL'} `{item['gate_id']}`：{item['detail']}"
        for item in gates
    )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 通过只证明官方历史事件日期档案完整且可回溯。",
            "- 未检验ORJ、未来收益、交易成本、组合或实盘可行性。",
            "",
        ]
    )
    AUDIT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "状态": status,
                "通过门数": f"{passed_count}/{len(gates)}",
                "重建原始ID": raw_reconstruction["unique_raw_id_count"],
                "事件行": event_validation["row_count"],
                "报告": AUDIT_MARKDOWN.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
