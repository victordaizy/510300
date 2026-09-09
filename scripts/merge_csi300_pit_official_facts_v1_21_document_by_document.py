from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import merge_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as merge  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_V1_21_DOCUMENT_BY_DOCUMENT_MERGE_V1"
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_21_DOCUMENT_BY_DOCUMENT_MERGED_V1_0_0"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")
BASE_RECEIPT_PATH = (
    ROOT
    / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "receipt_v1_19_direct_text_only.json"
)
COMBINED_ROOT = ROOT / "tmp/pit_v1_21_document_by_document_merge"
COMBINED_RECEIPT_PATH = COMBINED_ROOT / "combined_receipt.json"
ADAPTED_RECEIPT_PATH = COMBINED_ROOT / "merge_input_receipt.json"


def _relative(path: Path) -> str:
    return merge.prior._relative(path)


def _receipt_paths() -> list[Path]:
    paths = {
        path.resolve()
        for pattern in ("pit_v1_20*/receipt.json", "pit_v1_21*/receipt.json")
        for path in (ROOT / "tmp").glob(pattern)
        if path.is_file()
    }
    paths.discard(COMBINED_RECEIPT_PATH.resolve())
    paths.discard(ADAPTED_RECEIPT_PATH.resolve())
    return sorted(paths, key=lambda path: _relative(path))


def _load_candidate_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("target_exception_count") != 0
        or receipt.get("source_exception_count") != 0
        or receipt.get("market_price_read") is not False
        or receipt.get("future_return_read") is not False
        or receipt.get("return_evaluation") != "NOT_ALLOWED"
    ):
        raise RuntimeError(f"候选回执异常或触碰收益读取边界：{_relative(path)}")
    return receipt


def _metric_map(row: dict[str, Any], *, receipt_path: Path) -> dict[str, dict[str, Any]]:
    metrics = list(row.get("new_metrics") or [])
    result = {
        str(metric["metric_id"]): metric
        for metric in metrics
        if metric.get("metric_id")
    }
    if len(result) != len(metrics):
        raise RuntimeError(
            f"新增指标键为空或重复：{row.get('announcement_id')}|{_relative(receipt_path)}"
        )
    declared_ids = [str(value) for value in row.get("newly_admitted_metric_ids") or []]
    if set(declared_ids) != set(result):
        raise RuntimeError(
            f"新增指标清单与事实不一致：{row.get('announcement_id')}|"
            f"{_relative(receipt_path)}"
        )
    for metric_id, metric in result.items():
        value = float(metric["metric_value_cny"])
        if not math.isfinite(value):
            raise RuntimeError(
                f"新增指标不是有限数值：{row.get('announcement_id')}|{metric_id}"
            )
    return result


def _source_context(metric: dict[str, Any]) -> tuple[str, str]:
    try:
        locator = json.loads(str(metric.get("source_locator") or "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"指标来源定位不是合法JSON：{metric.get('metric_id')}"
        ) from error
    context = locator.get("source_context") or {}
    return (
        str(context.get("announcement_id") or ""),
        str(context.get("official_pdf_sha256") or ""),
    )


def _matching_source(
    metric: dict[str, Any],
    row: dict[str, Any],
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    announcement_id, pdf_sha256 = _source_context(metric)
    if not announcement_id and not pdf_sha256:
        raise RuntimeError(
            f"指标缺少官方PDF来源身份：{row['announcement_id']}|{metric['metric_id']}"
        )
    matches = [
        source
        for source in row.get("source_results") or []
        if not source.get("error")
        and (
            pdf_sha256
            and str(source.get("observed_pdf_sha256") or "") == pdf_sha256
            or announcement_id
            and str(source.get("announcement_id") or "") == announcement_id
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"指标官方PDF来源不唯一：{row['announcement_id']}|{metric['metric_id']}|"
            f"{len(matches)}|{_relative(receipt_path)}"
        )
    source = dict(matches[0])
    source_path = ROOT / str(source["path"])
    if not source_path.is_file():
        raise RuntimeError(f"官方PDF文件不存在：{source_path}")
    observed_sha256 = merge.prior._sha256_file(source_path)
    if observed_sha256 != str(source["observed_pdf_sha256"]):
        raise RuntimeError(
            f"官方PDF哈希漂移：{row['announcement_id']}|{metric['metric_id']}"
        )
    return source


def _source_key(source: dict[str, Any]) -> tuple[str, str]:
    return (
        str(source.get("announcement_id") or ""),
        str(source.get("observed_pdf_sha256") or ""),
    )


def _row_rank(candidate: dict[str, Any]) -> tuple[int, int, str]:
    row = candidate["row"]
    return (
        len(row.get("after_missing_metrics") or []),
        -len(candidate["metrics"]),
        str(candidate["receipt_path"]),
    )


def _same_metric_value(
    announcement_id: str,
    metric_id: str,
    candidates: list[dict[str, Any]],
) -> None:
    reference = candidates[0]["metrics"][metric_id]
    reference_value = float(reference["metric_value_cny"])
    reference_scope = str(reference.get("statement_scope") or "")
    reference_period_scope = str(reference.get("value_period_scope") or "")
    for candidate in candidates[1:]:
        metric = candidate["metrics"][metric_id]
        if (
            not math.isclose(
                float(metric["metric_value_cny"]),
                reference_value,
                rel_tol=0.0,
                abs_tol=0.005,
            )
            or str(metric.get("statement_scope") or "") != reference_scope
            or str(metric.get("value_period_scope") or "")
            != reference_period_scope
        ):
            paths = ",".join(
                _relative(Path(item["receipt_path"])) for item in candidates
            )
            raise RuntimeError(
                f"同一公告指标存在冲突：{announcement_id}|{metric_id}|{paths}"
            )


def _validate_candidate_row(
    row: dict[str, Any],
    *,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if (
        row.get("market_price_read") is not False
        or row.get("future_return_read") is not False
        or row.get("return_evaluation") != "NOT_ALLOWED"
    ):
        raise RuntimeError(
            f"候选结果触碰收益读取边界：{row.get('announcement_id')}|"
            f"{_relative(receipt_path)}"
        )
    metrics = _metric_map(row, receipt_path=receipt_path)
    checkpoint_path = ROOT / str(row["checkpoint_path"])
    observed_checkpoint_sha256 = merge.prior._sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != str(row["checkpoint_sha256"]):
        raise RuntimeError(
            f"候选结果基线检查点哈希漂移：{row['announcement_id']}|"
            f"{_relative(receipt_path)}"
        )
    checkpoint = merge.prior._load_json_gzip(checkpoint_path)
    identity_fields = ("announcement_id", "ts_code", "report_period", "period_type")
    if any(str(checkpoint[field]) != str(row[field]) for field in identity_fields):
        raise RuntimeError(
            f"候选结果与基线检查点身份不一致：{row['announcement_id']}|"
            f"{_relative(receipt_path)}"
        )
    before_missing = [str(value) for value in checkpoint.get("missing_metrics") or []]
    if before_missing != [str(value) for value in row.get("before_missing_metrics") or []]:
        raise RuntimeError(
            f"候选结果基线缺失状态不一致：{row['announcement_id']}|"
            f"{_relative(receipt_path)}"
        )
    if not set(metrics).issubset(before_missing):
        raise RuntimeError(
            f"候选结果新增了非缺失指标：{row['announcement_id']}|"
            f"{_relative(receipt_path)}"
        )
    expected_after = [value for value in before_missing if value not in metrics]
    if expected_after != [str(value) for value in row.get("after_missing_metrics") or []]:
        raise RuntimeError(
            f"候选结果补齐前后不守恒：{row['announcement_id']}|"
            f"{_relative(receipt_path)}"
        )
    return checkpoint, metrics


def _collect_candidates() -> tuple[list[Path], dict[str, list[dict[str, Any]]]]:
    receipt_paths = _receipt_paths()
    candidates: dict[str, list[dict[str, Any]]] = {}
    for receipt_path in receipt_paths:
        receipt = _load_candidate_receipt(receipt_path)
        receipt_sha256 = merge.prior._sha256_file(receipt_path)
        for row in receipt.get("results") or []:
            if not row.get("new_metrics"):
                continue
            checkpoint, metrics = _validate_candidate_row(
                row,
                receipt_path=receipt_path,
            )
            announcement_id = str(row["announcement_id"])
            candidates.setdefault(announcement_id, []).append(
                {
                    "row": row,
                    "checkpoint": checkpoint,
                    "metrics": metrics,
                    "receipt_path": receipt_path,
                    "receipt_sha256": receipt_sha256,
                }
            )
    if not candidates:
        raise RuntimeError("没有发现可合并的逐份官方PDF新增指标")
    return receipt_paths, candidates


def _combine_announcement(
    announcement_id: str,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[Path]]:
    checkpoint_paths = {
        str(candidate["row"]["checkpoint_path"]) for candidate in candidates
    }
    checkpoint_hashes = {
        str(candidate["row"]["checkpoint_sha256"]) for candidate in candidates
    }
    if len(checkpoint_paths) != 1 or len(checkpoint_hashes) != 1:
        raise RuntimeError(f"同一公告候选结果基线不唯一：{announcement_id}")

    best_candidate = min(candidates, key=_row_rank)
    checkpoint = best_candidate["checkpoint"]
    before_missing = [str(value) for value in checkpoint.get("missing_metrics") or []]
    selected_metrics: dict[str, dict[str, Any]] = {}
    selected_sources: dict[tuple[str, str], dict[str, Any]] = {}
    used_receipts: set[Path] = set()

    for metric_id in before_missing:
        metric_candidates = [
            candidate for candidate in candidates if metric_id in candidate["metrics"]
        ]
        if not metric_candidates:
            continue
        metric_candidates.sort(key=_row_rank)
        _same_metric_value(announcement_id, metric_id, metric_candidates)
        selected = metric_candidates[0]
        metric = dict(selected["metrics"][metric_id])
        source = _matching_source(
            metric,
            selected["row"],
            receipt_path=selected["receipt_path"],
        )
        key = _source_key(source)
        existing = selected_sources.get(key)
        if existing is None:
            selected_sources[key] = source
        else:
            stable_fields = (
                "announcement_id",
                "official_pdf_url",
                "observed_pdf_sha256",
                "observed_pdf_size_bytes",
                "pdf_page_count",
            )
            if any(str(existing.get(field)) != str(source.get(field)) for field in stable_fields):
                raise RuntimeError(
                    f"同一官方PDF来源元数据冲突：{announcement_id}|{metric_id}"
                )
            if len(json.dumps(source, ensure_ascii=False, sort_keys=True)) > len(
                json.dumps(existing, ensure_ascii=False, sort_keys=True)
            ):
                selected_sources[key] = source
        selected_metrics[metric_id] = metric
        used_receipts.add(Path(selected["receipt_path"]))

    if not selected_metrics:
        raise RuntimeError(f"公告没有可合并指标：{announcement_id}")
    admitted_ids = [value for value in before_missing if value in selected_metrics]
    after_missing = [value for value in before_missing if value not in selected_metrics]
    new_metrics = [selected_metrics[value] for value in admitted_ids]
    row = {
        "announcement_id": announcement_id,
        "ts_code": str(checkpoint["ts_code"]),
        "report_period": str(checkpoint["report_period"]),
        "period_type": str(checkpoint["period_type"]),
        "checkpoint_path": next(iter(checkpoint_paths)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "merged_checkpoint_path": next(iter(checkpoint_paths)),
        "merged_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "before_missing_metrics": before_missing,
        "after_missing_metrics": after_missing,
        "newly_admitted_metric_ids": admitted_ids,
        "changed_metric_ids": admitted_ids,
        "new_metrics": new_metrics,
        "changed_metrics": new_metrics,
        "corrected_metric_ids": [],
        "removed_metric_ids": [],
        "source_results": sorted(
            selected_sources.values(),
            key=lambda source: _source_key(source),
        ),
        "source_receipt_paths": sorted(_relative(path) for path in used_receipts),
        "document_complete_before_targeted_ocr": False,
        "document_complete_after_targeted_ocr": not after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    return row, used_receipts


def _build_merge_input_receipt() -> dict[str, Any]:
    discovered_paths, candidate_map = _collect_candidates()
    results: list[dict[str, Any]] = []
    used_receipts: set[Path] = set()
    for announcement_id in sorted(candidate_map):
        row, row_receipts = _combine_announcement(
            announcement_id,
            candidate_map[announcement_id],
        )
        results.append(row)
        used_receipts.update(row_receipts)

    source_receipts = [
        {
            "path": _relative(path),
            "sha256": merge.prior._sha256_file(path),
        }
        for path in sorted(used_receipts, key=_relative)
    ]
    combined = {
        "protocol_id": (
            "CSI300_PIT_OFFICIAL_FACTS_V1_21_DOCUMENT_BY_DOCUMENT_"
            "COMBINED_RECEIPT_V1"
        ),
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "base_receipt_path": _relative(BASE_RECEIPT_PATH),
        "base_receipt_sha256": merge.prior._sha256_file(BASE_RECEIPT_PATH),
        "discovered_receipt_count": len(discovered_paths),
        "contributing_receipt_count": len(source_receipts),
        "source_receipts": source_receipts,
        "target_count": len(results),
        "newly_completed_document_count": sum(
            not row["after_missing_metrics"] for row in results
        ),
        "newly_admitted_metric_count": sum(
            len(row["newly_admitted_metric_ids"]) for row in results
        ),
        "target_exception_count": 0,
        "source_exception_count": 0,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    merge.prior._atomic_write_json(COMBINED_RECEIPT_PATH, combined)
    adapted = {
        **combined,
        "source_combined_receipt_path": _relative(COMBINED_RECEIPT_PATH),
        "source_combined_receipt_sha256": merge.prior._sha256_file(
            COMBINED_RECEIPT_PATH
        ),
        "removed_unreplaced_metric_count": 0,
    }
    merge.prior._atomic_write_json(ADAPTED_RECEIPT_PATH, adapted)
    return adapted


def main() -> int:
    receipt = _build_merge_input_receipt()
    print(
        "逐份回执汇总完成："
        f"公告={receipt['target_count']}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"新增完整文档={receipt['newly_completed_document_count']}；"
        f"采用回执={receipt['contributing_receipt_count']}/"
        f"{receipt['discovered_receipt_count']}",
        flush=True,
    )

    audit_root = merge.AUDIT_ROOT
    raw_root = merge.RAW_ROOT
    merge.PROTOCOL_ID = PROTOCOL_ID
    merge.MERGED_PARSER_VERSION = MERGED_PARSER_VERSION
    merge.ADMISSION_PHASE = "DOCUMENT_BY_DOCUMENT_OFFICIAL_PDF"
    merge.MERGE_RECEIPT_FIELD = "document_by_document_merge_receipt"
    merge.BASE_RECEIPT_PATH = BASE_RECEIPT_PATH
    merge.RESIDUAL_RECEIPT_PATH = ADAPTED_RECEIPT_PATH
    merge.CHECKPOINT_ROOT = raw_root / "checkpoints_v1_21_document_by_document"
    merge.OUTPUT_REQUIREMENT_PATH = (
        audit_root
        / "official_fact_requirement_ledger_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_QUEUE_PATH = (
        audit_root / "official_fact_document_queue_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_PATCH_MANIFEST_PATH = (
        audit_root
        / "official_fact_checkpoint_patch_manifest_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_RESIDUAL_PRIORITY_PATH = (
        audit_root
        / "official_fact_residual_priority_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_FACTS_PATH = (
        raw_root / "official_financial_facts_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_DEPENDENCY_PATH = (
        raw_root / "event_fact_dependency_ledger_v1_21_document_by_document.parquet"
    )
    merge.OUTPUT_RECEIPT_PATH = raw_root / "receipt_v1_21_document_by_document.json"
    merge.OUTPUT_REPORT_PATH = (
        ROOT
        / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
        "OFFICIAL_FACTS_V1_21_DOCUMENT_BY_DOCUMENT.md"
    )
    return merge.main()


if __name__ == "__main__":
    raise SystemExit(main())
