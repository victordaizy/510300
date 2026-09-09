from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1_3 as collector,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
STATUS_BLOCKED = "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE"
STATUS_REACHABLE = "OPTIMISTIC_OFFICIAL_PDF_FACT_COVERAGE_GATE_STILL_REACHABLE"
RETURN_EVALUATION = "NOT_ALLOWED"
OUTPUT_JSON = (
    ROOT
    / "reports/data_quality/"
    "csi300_pit_fundamental_underreaction_official_facts_v1_3_optimistic_bound.json"
)
OUTPUT_MARKDOWN = (
    ROOT
    / "reports/data_quality/"
    "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_3_OPTIMISTIC_BOUND.md"
)
PREFLIGHT_JSON = (
    ROOT
    / "reports/data_quality/"
    "csi300_pit_fundamental_underreaction_enhancement_v1_preflight.json"
)

REQUIREMENT_COLUMNS = [
    "dependency_announcement_id",
    "target_announcement_id",
    "target_ts_code",
    "target_report_period",
    "target_period_type",
    "target_publication_date",
    "target_publication_year",
    "target_industry_l1",
    "target_industry_l1_code",
    "queued_for_official_pdf",
    "dependency_temporal_status",
]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_checkpoints(
    checkpoint_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    incomplete_rows: list[dict[str, Any]] = []
    for path in sorted(checkpoint_root.rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        announcement_id = str(payload.get("announcement_id") or "")
        if not announcement_id:
            raise ValueError(f"检查点缺少公告ID：{path}")
        if announcement_id in checkpoints:
            raise ValueError(f"检查点公告ID重复：{announcement_id}")
        parser_version = str(payload.get("parser_version") or "")
        if parser_version != collector.PARSER_VERSION:
            raise ValueError(
                "V1.3检查点解析器版本不一致："
                f"{announcement_id}|{parser_version or 'MISSING'}"
            )
        checkpoints[announcement_id] = payload
        if payload.get("checkpoint_status") == "PARSED_INCOMPLETE":
            missing_metrics = [str(value) for value in payload.get("missing_metrics") or []]
            if not missing_metrics:
                raise ValueError(f"不完整检查点未声明缺失指标：{announcement_id}")
            incomplete_rows.append(
                {
                    "announcement_id": announcement_id,
                    "official_pdf_announcement_id": str(
                        payload.get("official_pdf_announcement_id") or announcement_id
                    ),
                    "missing_metrics": missing_metrics,
                    "missing_metric_count": len(missing_metrics),
                    "checkpoint_path": _relative(path),
                }
            )
    return checkpoints, sorted(incomplete_rows, key=lambda row: row["announcement_id"])


def _optimistic_queue_state(
    queue_ids: pd.DataFrame,
    incomplete_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    state = queue_ids.copy()
    state["announcement_id"] = state["announcement_id"].astype(str)
    state["checkpoint_status"] = "PARSED_COMPLETE"
    state["document_terminal"] = True
    state["document_complete"] = True
    state["missing_metric_count"] = 0
    state["missing_metrics_json"] = "[]"
    incomplete_by_id = {
        row["announcement_id"]: row["missing_metrics"] for row in incomplete_rows
    }
    missing_queue_ids = sorted(set(incomplete_by_id) - set(state["announcement_id"]))
    if missing_queue_ids:
        raise ValueError(f"不完整检查点不在冻结PDF队列：{missing_queue_ids}")
    for announcement_id, missing_metrics in incomplete_by_id.items():
        mask = state["announcement_id"].eq(announcement_id)
        state.loc[mask, "checkpoint_status"] = "PARSED_INCOMPLETE"
        state.loc[mask, "document_complete"] = False
        state.loc[mask, "missing_metric_count"] = len(missing_metrics)
        state.loc[mask, "missing_metrics_json"] = json.dumps(
            missing_metrics,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return state


def _coverage_snapshot(
    requirement: pd.DataFrame,
    queue_state: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runtime, dependency = collector._base.build_event_dependency_ledger(  # noqa: SLF001
        requirement,
        queue_state,
    )
    coverage = collector._base.coverage_metrics(dependency, config)  # noqa: SLF001
    return runtime, dependency, coverage


def _render_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    baseline = result["metadata_only_ceiling"]
    optimistic = result["optimistic_bound"]
    gates = result["gate_components"]
    lines = [
        "# CSI300 点时基本面反应不足：V1.3 官方事实覆盖乐观上界",
        "",
        f"- 结论状态：`{result['status']}`",
        f"- 解析器：`{result['parser_version']}`",
        f"- 检查点：{counts['checkpoint_count']:,}/{counts['queued_document_count']:,}",
        f"- 已完整：{counts['parsed_complete_document_count']:,}",
        f"- 已确认不完整：{counts['parsed_incomplete_document_count']:,}",
        f"- 尚未处理、在上界中假定全部完美：{counts['optimistically_assumed_complete_document_count']:,}",
        f"- 收益评价：`{result['return_evaluation']}`",
        "- 市场价格读取：否",
        "- 未来收益读取：否",
        (
            f"- 主预检：`{result['main_preflight']['status']}`，"
            f"{result['main_preflight']['passed_gate_count']}/"
            f"{result['main_preflight']['total_gate_count']}门通过"
        ),
        (
            "- 主预检唯一阻断门："
            f"`{', '.join(result['main_preflight']['blocking_gate_ids'])}`"
        ),
        "",
        "## 上界定义",
        "",
        (
            "本报告不是当前完成率。它把所有尚未处理或可重试的官方PDF一律假定为九项事实完整，"
            "只保留已经由冻结V1.3解析器确认的 `PARSED_INCOMPLETE`，并保留公告元数据缺失和点时"
            "不可用约束。因此，这是同一V1.3版本继续批量采集所能达到的数学乐观上界。"
        ),
        "",
        "## 冻结门槛判定",
        "",
        "| 门槛 | 要求 | 乐观上界 | 结果 |",
        "| --- | ---: | ---: | --- |",
        (
            f"| 总体目标事件完整率 | {gates['overall']['required_ratio']:.2%} "
            f"({gates['overall']['required_count']:,}个) | "
            f"{gates['overall']['optimistic_ratio']:.4%} "
            f"({gates['overall']['optimistic_count']:,}个) | "
            f"{'通过' if gates['overall']['passed'] else '失败'} |"
        ),
        (
            f"| 各公告年度最低完整率 | {gates['publication_year']['required_ratio']:.2%} | "
            f"{gates['publication_year']['optimistic_ratio']:.4%} | "
            f"{'通过' if gates['publication_year']['passed'] else '失败'} |"
        ),
        (
            f"| 合格行业最低完整率 | {gates['industry']['required_ratio']:.2%} | "
            f"{gates['industry']['optimistic_ratio']:.4%} | "
            f"{'通过' if gates['industry']['passed'] else '失败'} |"
        ),
        "",
        f"- 仅受元数据/点时可用性约束的理论天花板：{baseline['ready_target_event_count']:,}/"
        f"{baseline['target_event_count']:,}（{baseline['overall_complete_event_ratio']:.4%}）",
        f"- 加入现有不完整PDF后的V1.3乐观上界：{optimistic['ready_target_event_count']:,}/"
        f"{optimistic['target_event_count']:,}（{optimistic['overall_complete_event_ratio']:.4%}）",
        f"- 已确认不完整PDF额外阻断目标事件：{optimistic['target_events_lost_to_known_incomplete_documents']:,}",
        f"- 距总体门槛仍缺：{gates['overall']['shortfall_count']:,}个完整目标事件",
        "",
        "## 缺失指标分布",
        "",
        "| 指标 | 涉及不完整PDF数 |",
        "| --- | ---: |",
    ]
    for metric, count in result["incomplete_metric_counts"].items():
        lines.append(f"| `{metric}` | {count:,} |")
    lines.extend(
        [
            "",
            "## 停止边界",
            "",
        ]
    )
    if result["status"] == STATUS_BLOCKED:
        lines.extend(
            [
                (
                    "总体冻结门槛在同一V1.3解析器下已经不可达。继续下载其余PDF不能提高本报告中的"
                    "乐观上界，因此停止V1.3批量采集，不启动60/120日相对510300收益评价。"
                ),
                "",
                (
                    "主预检的通用下一步 `BUILD_OR_COMPLETE_OFFICIAL_ORIGINAL_PDF_VERIFIED_FACT_ARCHIVE_"
                    "THEN_RERUN_PREFLIGHT` 只表达G6尚未通过；本报告是其后的可达性裁决，已把V1.3"
                    "分支收窄为停止，不再继续本版本批量采集。"
                ),
                "",
                (
                    "只有另行冻结、先在现有不完整样本上通过固定重放门的版本化解析器修订，才可能"
                    "改变该上界；不得通过放宽覆盖率、删除样本、修改目标期、替换指标或读取收益来救援。"
                ),
            ]
        )
    else:
        lines.append(
            "同一V1.3解析器的乐观上界尚未跌破门槛，但这不等于当前数据门已通过；仍不得读取收益。"
        )
    lines.extend(
        [
            "",
            "## 已确认不完整公告",
            "",
            "| 队列公告ID | 实际PDF公告ID | 缺失指标 |",
            "| --- | --- | --- |",
        ]
    )
    for row in result["incomplete_documents"]:
        lines.append(
            f"| `{row['announcement_id']}` | `{row['official_pdf_announcement_id']}` | "
            f"{', '.join(f'`{metric}`' for metric in row['missing_metrics'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    collector._patch_base_collector()  # noqa: SLF001
    config, provenance = collector._base.load_and_verify_config()  # noqa: SLF001
    artifacts = config["artifacts"]
    requirement_path = collector._base.project_path(  # noqa: SLF001
        artifacts["requirement_ledger"]
    )
    queue_path = collector._base.project_path(artifacts["document_queue"])  # noqa: SLF001
    checkpoint_root = collector._base.project_path(  # noqa: SLF001
        artifacts["checkpoint_root"]
    )
    receipt_path = collector._base.project_path(artifacts["receipt"])  # noqa: SLF001
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("parser_version") != collector.PARSER_VERSION:
        raise ValueError("当前采集收据与V1.3解析器版本不一致")
    if receipt.get("return_evaluation") != RETURN_EVALUATION:
        raise ValueError("当前采集收据未保持收益评价禁止状态")
    if receipt.get("market_price_read") or receipt.get("future_return_read"):
        raise ValueError("当前采集收据显示曾读取市场价格或未来收益")
    preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    if preflight.get("return_evaluation") != RETURN_EVALUATION:
        raise ValueError("主预检未保持收益评价禁止状态")
    blocking_gate_ids = [str(value) for value in preflight.get("blocking_gate_ids") or []]
    if blocking_gate_ids != ["G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS"]:
        raise ValueError(f"主预检阻断门不符合预期：{blocking_gate_ids}")
    preflight_gates = {
        str(gate["gate_id"]): gate for gate in preflight.get("gates") or []
    }
    preflight_g6 = preflight_gates["G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS"]
    preflight_g7 = preflight_gates[
        "G7_NO_MARKET_PRICE_OR_FUTURE_RETURN_READ_BEFORE_G1_TO_G6_PASS"
    ]
    current_receipt_sha256 = collector._base.sha256_file(receipt_path)  # noqa: SLF001
    if preflight_g6.get("details", {}).get("receipt_sha256") != current_receipt_sha256:
        raise ValueError("主预检未引用当前采集收据")
    if preflight_g6.get("observed") != receipt.get("status"):
        raise ValueError("主预检G6与当前采集状态不一致")
    if not preflight_g7.get("passed"):
        raise ValueError("主预检G7未确认禁止读取市场价格和未来收益")

    requirement = pd.read_parquet(requirement_path, columns=REQUIREMENT_COLUMNS)
    queue_ids = pd.read_parquet(queue_path, columns=["announcement_id"])
    queue_ids["announcement_id"] = queue_ids["announcement_id"].astype(str)
    if queue_ids["announcement_id"].duplicated().any():
        raise ValueError("冻结PDF队列公告ID不唯一")

    checkpoints, incomplete_rows = _read_checkpoints(checkpoint_root)
    status_counts = Counter(
        str(payload.get("checkpoint_status") or "MISSING")
        for payload in checkpoints.values()
    )
    receipt_counts = receipt["counts"]
    expected_receipt_counts = {
        "queued_document_count": len(queue_ids),
        "terminal_document_count": len(checkpoints),
        "complete_document_count": status_counts["PARSED_COMPLETE"],
        "incomplete_terminal_document_count": status_counts["PARSED_INCOMPLETE"],
    }
    mismatched_receipt_counts = {
        key: {
            "receipt": int(receipt_counts.get(key, -1)),
            "recomputed": int(value),
        }
        for key, value in expected_receipt_counts.items()
        if int(receipt_counts.get(key, -1)) != int(value)
    }
    if mismatched_receipt_counts:
        raise ValueError(f"当前采集收据计数不一致：{mismatched_receipt_counts}")
    optimistic_state = _optimistic_queue_state(queue_ids, incomplete_rows)
    perfect_state = _optimistic_queue_state(queue_ids, [])
    _, perfect_dependency, perfect_coverage = _coverage_snapshot(
        requirement,
        perfect_state,
        config,
    )
    optimistic_runtime, optimistic_dependency, optimistic_coverage = _coverage_snapshot(
        requirement,
        optimistic_state,
        config,
    )

    perfect_ready = perfect_dependency.set_index("announcement_id")["target_event_ready"]
    optimistic_ready = optimistic_dependency.set_index("announcement_id")[
        "target_event_ready"
    ]
    lost_target_ids = sorted(
        perfect_ready.index[perfect_ready & ~optimistic_ready].astype(str).tolist()
    )
    incomplete_ids = {row["announcement_id"] for row in incomplete_rows}
    lost_blockers = (
        optimistic_runtime.loc[
            optimistic_runtime["target_announcement_id"].astype(str).isin(lost_target_ids)
            & optimistic_runtime["dependency_announcement_id"].astype(str).isin(incomplete_ids),
            ["target_announcement_id", "dependency_announcement_id"],
        ]
        .astype(str)
        .drop_duplicates()
        .groupby("target_announcement_id")["dependency_announcement_id"]
        .apply(lambda values: sorted(values.tolist()))
        .to_dict()
    )

    target_count = int(len(optimistic_dependency))
    optimistic_ready_count = int(optimistic_dependency["target_event_ready"].sum())
    perfect_ready_count = int(perfect_dependency["target_event_ready"].sum())
    expected_target_count = int(
        config["inputs"]["target_inventory"]["expected_target_event_count"]
    )
    if target_count != expected_target_count:
        raise ValueError(
            f"目标事件数不一致：{target_count:,}!={expected_target_count:,}"
        )
    if perfect_ready_count - optimistic_ready_count != len(lost_target_ids):
        raise ValueError("已确认不完整PDF阻断的目标事件无法勾稽")
    admission = config["admission"]
    overall_required_ratio = float(admission["required_complete_event_ratio"])
    year_required_ratio = float(
        admission["required_complete_event_ratio_each_publication_year"]
    )
    industry_required_ratio = float(
        admission["required_complete_event_ratio_each_industry_with_at_least_50_targets"]
    )
    overall_required_count = math.ceil(target_count * overall_required_ratio)
    overall_passed = bool(
        optimistic_coverage["overall_complete_event_ratio"] >= overall_required_ratio
    )
    year_passed = bool(
        optimistic_coverage["minimum_publication_year_complete_event_ratio"]
        >= year_required_ratio
    )
    industry_passed = bool(
        optimistic_coverage["minimum_eligible_industry_complete_event_ratio"]
        >= industry_required_ratio
    )
    status = (
        STATUS_REACHABLE
        if overall_passed and year_passed and industry_passed
        else STATUS_BLOCKED
    )
    metric_counts = Counter(
        metric for row in incomplete_rows for metric in row["missing_metrics"]
    )
    incomplete_metric_counts = {
        metric: int(metric_counts[metric])
        for metric in collector.REQUIRED_METRICS
        if metric_counts[metric]
    }

    result = {
        "protocol_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_3_OPTIMISTIC_BOUND",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "status": status,
        "definition": (
            "ALL_UNPROCESSED_OR_RETRYABLE_OFFICIAL_PDFS_ASSUMED_PARSED_COMPLETE;"
            "EXISTING_V1_3_PARSED_INCOMPLETE_CHECKPOINTS_REMAIN_INCOMPLETE;"
            "METADATA_AND_POINT_IN_TIME_UNAVAILABILITY_REMAINS"
        ),
        "parser_version": collector.PARSER_VERSION,
        "current_acquisition_receipt_status": receipt["status"],
        "provenance": {
            **provenance,
            "requirement_ledger_path": _relative(requirement_path),
            "requirement_ledger_sha256": collector._base.sha256_file(  # noqa: SLF001
                requirement_path
            ),
            "document_queue_path": _relative(queue_path),
            "document_queue_sha256": collector._base.sha256_file(queue_path),  # noqa: SLF001
            "current_receipt_path": _relative(receipt_path),
            "current_receipt_sha256": current_receipt_sha256,
            "main_preflight_path": _relative(PREFLIGHT_JSON),
            "main_preflight_sha256": collector._base.sha256_file(  # noqa: SLF001
                PREFLIGHT_JSON
            ),
            "checkpoint_root": _relative(checkpoint_root),
        },
        "main_preflight": {
            "status": preflight["status"],
            "passed_gate_count": int(preflight["passed_gate_count"]),
            "total_gate_count": int(preflight["total_gate_count"]),
            "blocking_gate_ids": blocking_gate_ids,
            "g6_observed": preflight_g6["observed"],
            "g7_observed": preflight_g7["observed"],
            "generated_next_allowed_step": preflight["next_allowed_step"],
            "terminal_interpretation": (
                "THE_GENERIC_G6_COMPLETION_BRANCH_IS_NARROWED_TO_STOP_FOR_V1_3_"
                "BECAUSE_THE_OPTIMISTIC_COVERAGE_BOUND_IS_BELOW_THE_FROZEN_GATE"
            ),
        },
        "counts": {
            "target_event_count": target_count,
            "queued_document_count": int(len(queue_ids)),
            "checkpoint_count": int(len(checkpoints)),
            "parsed_complete_document_count": int(status_counts["PARSED_COMPLETE"]),
            "parsed_incomplete_document_count": int(status_counts["PARSED_INCOMPLETE"]),
            "other_checkpoint_count": int(
                len(checkpoints)
                - status_counts["PARSED_COMPLETE"]
                - status_counts["PARSED_INCOMPLETE"]
            ),
            "optimistically_assumed_complete_document_count": int(
                len(queue_ids) - len(checkpoints)
            ),
        },
        "checkpoint_status_counts": dict(sorted(status_counts.items())),
        "metadata_only_ceiling": {
            "target_event_count": target_count,
            "ready_target_event_count": perfect_ready_count,
            "overall_complete_event_ratio": float(
                perfect_coverage["overall_complete_event_ratio"]
            ),
            "minimum_publication_year_complete_event_ratio": float(
                perfect_coverage["minimum_publication_year_complete_event_ratio"]
            ),
            "minimum_eligible_industry_complete_event_ratio": float(
                perfect_coverage["minimum_eligible_industry_complete_event_ratio"]
            ),
        },
        "optimistic_bound": {
            "target_event_count": target_count,
            "ready_target_event_count": optimistic_ready_count,
            "overall_complete_event_ratio": float(
                optimistic_coverage["overall_complete_event_ratio"]
            ),
            "minimum_publication_year_complete_event_ratio": float(
                optimistic_coverage["minimum_publication_year_complete_event_ratio"]
            ),
            "minimum_eligible_industry_complete_event_ratio": float(
                optimistic_coverage["minimum_eligible_industry_complete_event_ratio"]
            ),
            "target_events_lost_to_known_incomplete_documents": int(
                perfect_ready_count - optimistic_ready_count
            ),
            "coverage_gate_passed": bool(
                optimistic_coverage["coverage_gate_passed"]
            ),
        },
        "gate_components": {
            "overall": {
                "required_ratio": overall_required_ratio,
                "required_count": overall_required_count,
                "optimistic_ratio": float(
                    optimistic_coverage["overall_complete_event_ratio"]
                ),
                "optimistic_count": optimistic_ready_count,
                "shortfall_count": max(
                    0,
                    overall_required_count - optimistic_ready_count,
                ),
                "passed": overall_passed,
            },
            "publication_year": {
                "required_ratio": year_required_ratio,
                "optimistic_ratio": float(
                    optimistic_coverage[
                        "minimum_publication_year_complete_event_ratio"
                    ]
                ),
                "passed": year_passed,
            },
            "industry": {
                "required_ratio": industry_required_ratio,
                "optimistic_ratio": float(
                    optimistic_coverage[
                        "minimum_eligible_industry_complete_event_ratio"
                    ]
                ),
                "passed": industry_passed,
            },
        },
        "incomplete_metric_counts": incomplete_metric_counts,
        "incomplete_documents": incomplete_rows,
        "target_events_lost_to_known_incomplete_documents": [
            {
                "target_announcement_id": target_id,
                "blocking_incomplete_dependency_announcement_ids": lost_blockers.get(
                    target_id,
                    [],
                ),
            }
            for target_id in lost_target_ids
        ],
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "signal_score_calculated": False,
        "portfolio_return_calculated": False,
        "positions_generated": False,
        "orders_generated": False,
        "broker_connection_performed": False,
        "trading_authorization": False,
        "return_evaluation": RETURN_EVALUATION,
        "next_allowed_step": (
            "STOP_V1_3_BULK_AND_KEEP_RETURN_EVALUATION_NOT_ALLOWED"
            if status == STATUS_BLOCKED
            else "CONTINUE_FACT_ACQUISITION_ONLY_RETURN_EVALUATION_REMAINS_NOT_ALLOWED"
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    collector._base.atomic_write_json(OUTPUT_JSON, result)  # noqa: SLF001
    collector._base.atomic_write_text(  # noqa: SLF001
        OUTPUT_MARKDOWN,
        _render_markdown(result),
    )
    print(f"状态：{status}")
    print(
        "V1.3乐观上界："
        f"{optimistic_ready_count:,}/{target_count:,}="
        f"{optimistic_coverage['overall_complete_event_ratio']:.6%}"
    )
    print(
        f"总体门槛：{overall_required_count:,}/{target_count:,}；"
        f"短缺：{max(0, overall_required_count - optimistic_ready_count):,}个目标事件"
    )
    print("本程序未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
