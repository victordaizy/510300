"""运行 510300 PIT 盈余信息扩散候选的 outcome-blind 数据准入。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pit_earnings_information_diffusion_10d_risk_v1 import (
    active_member_counts_on_dates,
    build_member_event_audit,
    choose_status,
    gate,
    read_json,
    sha256_file,
    validate_manifest,
    validate_membership_intervals,
    validate_weights,
)


CONFIG_PATH = ROOT / "config/510300_pit_earnings_information_diffusion_10d_risk_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def project_path(value: str) -> Path:
    return ROOT / value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def facts_audit(config: dict[str, Any], member_events: pd.DataFrame) -> dict[str, Any]:
    """读取事实阶段的状态；只有正式汇总存在时才读取安全状态列。"""

    inputs = config["inputs"]
    progress_path = project_path(inputs["fact_checkpoint_progress_receipt"])
    progress = read_json(progress_path) if progress_path.is_file() else {}
    automation_path = project_path(inputs["fact_automation_report"])
    facts_path = project_path(inputs["first_public_facts"])
    automation = read_json(automation_path) if automation_path.is_file() else None
    result: dict[str, Any] = {
        "progress_receipt_path": inputs["fact_checkpoint_progress_receipt"],
        "progress_receipt_exists": progress_path.is_file(),
        "progress_status": str(progress.get("status") or "MISSING"),
        "progress_updated_at": progress.get("updated_at"),
        "target_count": progress.get("target_count"),
        "reusable_checkpoint_count_before_run": progress.get(
            "reusable_checkpoint_count_before_run"
        ),
        "selected_task_count": progress.get("selected_task_count"),
        "completed_task_count": progress.get("completed_task_count"),
        "new_checkpoint_count": progress.get("new_checkpoint_count"),
        "explicit_failure_count": progress.get("explicit_failure_count"),
        "market_price_read": bool(progress.get("market_price_read", False)),
        "future_return_read": bool(progress.get("future_return_read", False)),
        "automation_report_path": inputs["fact_automation_report"],
        "automation_report_exists": automation_path.is_file(),
        "automation_status": automation.get("status") if automation else None,
        "facts_path": inputs["first_public_facts"],
        "facts_exists": facts_path.is_file(),
        "human_review_completed": False,
        "member_first_public_weighted_fact_count": None,
        "years_with_at_least_20_member_first_public_weighted_facts": None,
    }
    if automation:
        result.update(
            {
                "pdf_coverage": automation.get("pdf_coverage"),
                "A2_status": automation.get("A2_status"),
                "a3_success_rate": automation.get("a3_success_rate"),
                "A3_status": automation.get("A3_status"),
                "first_public_pass_count_all_a_share": automation.get(
                    "first_public_pass_count"
                ),
                "actual_completed_human_review_pairs": int(
                    automation.get("actual_completed_human_review_pairs", 0)
                ),
            }
        )
        result["human_review_completed"] = bool(
            result["actual_completed_human_review_pairs"]
            >= int(automation.get("formal_dual_review", {}).get("sample_count", 500))
        )
    if automation and facts_path.is_file():
        safe_columns = [
            "announcement_id",
            "is_first_numeric_core_event",
            "target_core_fact_complete",
            "market_price_read",
            "future_return_read",
        ]
        facts = pd.read_parquet(facts_path, columns=safe_columns)
        for audit_column in ("market_price_read", "future_return_read"):
            if facts[audit_column].fillna(False).astype(bool).any():
                raise ValueError(f"正式事实表 {audit_column} 出现 true")
        facts["announcement_id"] = facts["announcement_id"].astype(str)
        eligible = facts.loc[
            facts["is_first_numeric_core_event"].fillna(False).astype(bool)
            & facts["target_core_fact_complete"].fillna(False).astype(bool),
            ["announcement_id"],
        ]
        member = member_events.loc[
            member_events["valid_strict_prior_weight"].fillna(False).astype(bool),
            ["announcement_id", "notice_date"],
        ].copy()
        member["announcement_id"] = member["announcement_id"].astype(str)
        joined = member.merge(eligible, on="announcement_id", how="inner", validate="one_to_one")
        annual = joined["notice_date"].dt.year.value_counts()
        result["member_first_public_weighted_fact_count"] = int(len(joined))
        result["member_first_public_weighted_fact_count_by_year"] = (
            annual.sort_index().to_dict()
        )
        result["years_with_at_least_20_member_first_public_weighted_facts"] = int(
            annual.ge(20).sum()
        )
        result["facts_market_price_read_true"] = int(
            facts["market_price_read"].fillna(False).sum()
        )
        result["facts_future_return_read_true"] = int(
            facts["future_return_read"].fillna(False).sum()
        )
    return result


def render_markdown(result: dict[str, Any]) -> str:
    g = result["gates"]
    e = result["event_coverage"]
    f = result["facts"]
    failed = [name for name, value in g.items() if value["passed"] is False]
    pending = [name for name, value in g.items() if value["passed"] is None]
    return "\n".join(
        [
            "# 510300 PIT 盈余信息扩散 10 日风险 V1：数据可行性",
            "",
            f"- 权威状态：`{result['status']}`",
            "- 研究阶段：`DATA_FEASIBILITY_ONLY`",
            "- 模型动作：`ABSTAIN`",
            "- 模型仓位目标：`UNSET`",
            "- 市场价格读取：`0`",
            "- 未来数据读取：`0`",
            "- 预测模型/未来10日标签/组合回测：均未创建",
            "",
            "## 顺序启动",
            "",
            f"期权链前序状态为 `{result['activation']['observed_predecessor_status']}`；"
            f"顺序启动门为 `{result['activation']['status']}`。",
            "",
            "## 公告与成员覆盖",
            "",
            f"- 全 A 股目标公告：{e['all_target_announcement_count']:,}",
            f"- 历史沪深300成员公告：{e['member_target_announcement_count']:,}",
            f"- 历史成员发行人：{e['member_unique_issuer_count']:,}",
            f"- 不同公告日期：{e['member_distinct_notice_date_count']:,}",
            f"- 日期级保守时钟占比：{e['date_only_conservative_ratio']:.2%}",
            f"- 首个拥有严格前序合格权重的成员公告日："
            f"`{e['first_valid_strict_prior_weight_event_date']}`",
            f"- 该日起严格前序权重覆盖率："
            f"{e['weight_coverage_after_first_eligible_date']:.2%}",
            "",
            "现有权重从 2016-08 才开始，因此不能制造 2015 年加权历史。"
            "同日权重不用于同日公告，所有公告最早在公告日之后首个交易日使用。",
            "",
            "## 事实抽取状态",
            "",
            f"- 检查点状态：`{f['progress_status']}`",
            f"- 本轮完成：{f.get('completed_task_count')}/{f.get('selected_task_count')}",
            f"- 显式失败：{f.get('explicit_failure_count')}",
            f"- 自动汇总状态：`{f.get('automation_status') or 'NOT_YET_AVAILABLE'}`",
            f"- 双人独立人工复核完成：`{str(f.get('human_review_completed')).lower()}`",
            "",
            "## 门槛结论",
            "",
            f"- 已失败硬门：{failed if failed else '无'}",
            f"- 尚不可评价硬门：{pending if pending else '无'}",
            "",
            "当前状态若为 `RUNNING` 或 `PENDING`，均不等于数据通过；"
            "在正式 `PASS_*_DATA_FEASIBILITY_ONLY` 前不得冻结预测模型、读取未来10日标签或运行组合回测。",
            "",
            "## 来源与限制",
            "",
            "- 盈利公告元数据和 PDF 来自巨潮资讯官方源；首次事实仍需冻结抽取和人工复核。",
            "- 历史成分是第三方调样公告重建，不得称作官方历史成分库。",
            "- 月度权重来自 Tushare 历史快照，供应商历史修订版本不可证明。",
            "- 不使用分析师一致预期，不使用供应商财务历史，不以 IF、宏观或价格指标补缺。",
            "",
        ]
    )


def main() -> int:
    created_at = datetime.now(TIMEZONE).isoformat()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = config["paths"]
    manifest_path = project_path(paths["manifest"])
    manifest = validate_manifest(ROOT, manifest_path)

    predecessor = read_json(project_path(config["sequential_activation"]["predecessor_result"]))
    observed_predecessor_status = str(predecessor.get("status") or "MISSING")
    required_predecessor_status = config["sequential_activation"][
        "required_predecessor_status"
    ]
    activation_passed = observed_predecessor_status == required_predecessor_status

    receipt = read_json(project_path(config["inputs"]["official_metadata_recovery_receipt"]))
    metadata_columns = [
        "announcement_id",
        "ts_code",
        "announcement_type",
        "target_title_type",
        "announcement_timestamp_at",
        "status",
        "market_price_read",
        "future_return_read",
    ]
    metadata = pd.read_parquet(
        project_path(config["inputs"]["official_metadata"]), columns=metadata_columns
    )
    inventory = pd.read_parquet(
        project_path(config["inputs"]["official_pdf_inventory"]),
        columns=[
            "announcement_id",
            "pdf_status",
            "pdf_sha256",
            "market_price_read",
            "future_return_read",
        ],
    )
    intervals = pd.read_parquet(
        project_path(config["inputs"]["historical_membership_intervals"])
    )
    weights = pd.read_parquet(project_path(config["inputs"]["historical_weights"]))
    membership_status = read_json(
        project_path(config["inputs"]["historical_membership_status"])
    )
    weight_status = read_json(project_path(config["inputs"]["historical_weights_status"]))
    official_rebalance = read_json(
        project_path(config["inputs"]["official_rebalance_source_admission"])
    )
    trading_calendar = pd.read_parquet(
        project_path(config["inputs"]["trading_calendar"]),
        columns=["exchange", "date", "is_open", "available_at", "source"],
    )

    membership_summary = validate_membership_intervals(intervals)
    weights, weight_summary = validate_weights(weights)
    events, event_summary = build_member_event_audit(
        metadata,
        inventory,
        intervals,
        weights,
        trading_calendar,
        int(config["point_in_time_contract"]["maximum_weight_snapshot_age_calendar_days"]),
        int(
            config["point_in_time_contract"][
                "maximum_notice_to_availability_lag_calendar_days"
            ]
        ),
    )
    active_counts = active_member_counts_on_dates(
        intervals, pd.to_datetime(events["availability_date"])
    )
    membership_summary["active_member_count_on_availability_dates_min"] = int(
        active_counts.min()
    )
    membership_summary["active_member_count_on_availability_dates_max"] = int(
        active_counts.max()
    )
    membership_summary["availability_date_count_checked"] = int(len(active_counts))
    anomalous_counts = active_counts.loc[
        active_counts.ne(int(config["hard_gates"]["required_active_members_per_notice_date"]))
    ]
    membership_summary["anomalous_availability_date_count"] = int(
        len(anomalous_counts)
    )
    membership_summary["anomalous_member_count_distribution"] = (
        anomalous_counts.value_counts().sort_index().to_dict()
    )
    membership_summary["anomalous_availability_date_count_by_year"] = (
        anomalous_counts.groupby(anomalous_counts.index.year).size().to_dict()
    )
    membership_summary["first_anomalous_availability_date"] = (
        str(anomalous_counts.index.min()) if len(anomalous_counts) else None
    )
    membership_summary["last_anomalous_availability_date"] = (
        str(anomalous_counts.index.max()) if len(anomalous_counts) else None
    )
    membership_summary["status_receipt"] = membership_status
    membership_summary["official_rebalance_cycle_source_gate"] = {
        "status": official_rebalance.get("gates", {}).get("D1", {}).get("status"),
        "observed_cycle_count": official_rebalance.get("gates", {})
        .get("D1", {})
        .get("observed_cycle_count"),
        "complete_cycle_count": official_rebalance.get("gates", {})
        .get("D3", {})
        .get("complete_cycle_count"),
        "return_evaluation": official_rebalance.get("return_evaluation"),
    }
    weight_summary["status_receipt"] = weight_status

    pdf_pass = inventory["pdf_status"].isin(
        {"PASS_DOWNLOADED_VERIFIED_PDF", "PASS_REUSED_VERIFIED_PDF"}
    )
    pdf_coverage = float(pdf_pass.mean())
    hard = config["hard_gates"]
    metadata_passed = (
        receipt.get("status") == hard["official_metadata_required_status"]
        and int(event_summary["duplicate_target_announcement_ids"])
        <= int(hard["maximum_duplicate_announcement_ids"])
        and pdf_coverage >= float(hard["minimum_official_pdf_coverage"])
        and not bool(metadata["market_price_read"].fillna(False).any())
        and not bool(metadata["future_return_read"].fillna(False).any())
    )
    membership_weights_passed = (
        membership_summary["duplicate_symbol_opt_in_rows"]
        <= int(hard["maximum_duplicate_membership_intervals"])
        and membership_summary["invalid_interval_order_rows"] == 0
        and membership_summary["overlapping_interval_rows"] == 0
        and membership_summary["active_member_count_on_availability_dates_min"]
        == int(hard["required_active_members_per_notice_date"])
        and membership_summary["active_member_count_on_availability_dates_max"]
        == int(hard["required_active_members_per_notice_date"])
        and weight_summary["snapshot_count"] >= int(hard["minimum_weight_snapshot_count"])
        and weight_summary["member_count_min"]
        == int(hard["required_members_per_weight_snapshot"])
        and weight_summary["member_count_max"]
        == int(hard["required_members_per_weight_snapshot"])
        and weight_summary["weight_sum_min_percent"]
        >= float(hard["minimum_weight_sum_percent"])
        and weight_summary["weight_sum_max_percent"]
        <= float(hard["maximum_weight_sum_percent"])
        and weight_summary["duplicate_date_security_rows"]
        <= int(hard["maximum_duplicate_weight_rows"])
        and event_summary["weight_coverage_after_first_eligible_date"]
        >= float(hard["minimum_member_event_weight_coverage_after_first_eligible_date"])
        and event_summary["member_distinct_notice_date_count"]
        >= int(hard["minimum_distinct_member_notice_dates"])
    )

    facts = facts_audit(config, events)
    automation_status = facts.get("automation_status")
    automation_passed: bool | None = None
    if automation_status is not None:
        automation_passed = bool(
            automation_status == hard["facts_automation_required_status"]
            and facts.get("A2_status") == "PASS"
            and facts.get("A3_status") == "PASS"
            and facts.get("a3_success_rate") is not None
            and float(facts["a3_success_rate"])
            >= float(hard["minimum_fact_extraction_success_rate"])
        )
    facts_prevalence_passed: bool | None = None
    if facts.get("member_first_public_weighted_fact_count") is not None:
        facts_prevalence_passed = bool(
            int(facts["member_first_public_weighted_fact_count"])
            >= int(hard["minimum_first_public_weighted_fact_count"])
            and int(facts["years_with_at_least_20_member_first_public_weighted_facts"])
            >= int(hard["minimum_years_with_20_first_public_weighted_facts"])
        )

    status = choose_status(
        activation_passed=activation_passed,
        metadata_passed=metadata_passed,
        membership_weights_passed=membership_weights_passed,
        facts_progress_status=str(facts["progress_status"]),
        facts_automation_status=automation_status,
        facts_automation_passed=automation_passed,
        human_review_completed=bool(facts["human_review_completed"]),
        facts_prevalence_passed=facts_prevalence_passed,
        statuses=config["status_machine"],
    )

    gates: dict[str, dict[str, Any]] = {
        "sequential_activation": gate(
            activation_passed,
            observed_predecessor_status,
            required_predecessor_status,
            "期权分支必须先正式阻断",
        ),
        "official_metadata_and_pdf": gate(
            metadata_passed,
            {"receipt_status": receipt.get("status"), "pdf_coverage": pdf_coverage},
            {
                "receipt_status": hard["official_metadata_required_status"],
                "minimum_pdf_coverage": hard["minimum_official_pdf_coverage"],
            },
            "公告 ID 唯一且价格/未来收益读取标记均为假",
        ),
        "historical_membership_and_weights": gate(
            membership_weights_passed,
            {
                "active_member_count_min": membership_summary[
                    "active_member_count_on_availability_dates_min"
                ],
                "active_member_count_max": membership_summary[
                    "active_member_count_on_availability_dates_max"
                ],
                "anomalous_availability_date_count": membership_summary[
                    "anomalous_availability_date_count"
                ],
                "weight_snapshot_count": weight_summary["snapshot_count"],
                "weight_coverage_after_start": event_summary[
                    "weight_coverage_after_first_eligible_date"
                ],
            },
            {
                "active_member_count": hard["required_active_members_per_notice_date"],
                "minimum_weight_snapshot_count": hard["minimum_weight_snapshot_count"],
                "minimum_weight_coverage_after_start": hard[
                    "minimum_member_event_weight_coverage_after_first_eligible_date"
                ],
            },
            "成分区间与严格前序月度权重仅用于数据准入",
        ),
        "fact_checkpoint_execution": gate(
            facts["progress_status"]
            == "PASS_MULTIPROCESS_FACT_TASKS_ATTEMPTED_WITH_EXPLICIT_RESULTS",
            facts["progress_status"],
            "PASS_MULTIPROCESS_FACT_TASKS_ATTEMPTED_WITH_EXPLICIT_RESULTS",
            "RUNNING 状态保留为运行中，不改写为失败",
        ),
        "fact_automation": {
            "passed": automation_passed,
            "actual": automation_status,
            "required": hard["facts_automation_required_status"],
            "note": "正式冻结运行器尚未汇总时保持不可评价",
        },
        "first_public_weighted_fact_prevalence": {
            "passed": facts_prevalence_passed,
            "actual": {
                "count": facts.get("member_first_public_weighted_fact_count"),
                "years_with_at_least_20": facts.get(
                    "years_with_at_least_20_member_first_public_weighted_facts"
                ),
            },
            "required": {
                "minimum_count": hard["minimum_first_public_weighted_fact_count"],
                "minimum_years_with_at_least_20": hard[
                    "minimum_years_with_20_first_public_weighted_facts"
                ],
            },
            "note": "仅在正式首次公开事实表生成后评价",
        },
        "formal_dual_human_review": {
            "passed": (
                bool(facts["human_review_completed"])
                if automation_passed is True
                else None
            ),
            "actual": facts.get("actual_completed_human_review_pairs", 0),
            "required": "FROZEN_SAMPLE_DUAL_INDEPENDENT_HUMAN_REVIEW_COMPLETE",
            "note": "自动事实门通过前不可评价；AI 不得代填评审人身份或独立性声明",
        },
        "outcome_blindness": gate(
            True,
            {"market_price_reads": 0, "future_data_reads": 0},
            {"market_price_reads": 0, "future_data_reads": 0},
            "未构造标签、模型、收益、仓位或订单",
        ),
    }
    if facts["progress_status"] == "RUNNING_MULTIPROCESS_FACT_CHECKPOINT_EXECUTION":
        gates["fact_checkpoint_execution"]["passed"] = None
    result = {
        "candidate_id": config["protocol"]["candidate_id"],
        "version": config["protocol"]["version"],
        "status": status,
        "created_at": created_at,
        "research_stage": "DATA_FEASIBILITY_ONLY",
        "activation": {
            "status": "PASS" if activation_passed else "FAIL",
            "predecessor_result": config["sequential_activation"]["predecessor_result"],
            "observed_predecessor_status": observed_predecessor_status,
            "required_predecessor_status": required_predecessor_status,
        },
        "manifest": {
            "path": paths["manifest"],
            "sha256": sha256_file(manifest_path),
            "status": manifest["status"],
        },
        "official_metadata": {
            "recovery_status": receipt.get("status"),
            "row_count": int(len(metadata)),
            "target_title_row_count": int(metadata["target_title_type"].fillna(False).sum()),
            "pdf_inventory_count": int(len(inventory)),
            "pdf_pass_count": int(pdf_pass.sum()),
            "pdf_coverage": pdf_coverage,
            "market_price_read_true": int(
                metadata["market_price_read"].fillna(False).sum()
            ),
            "future_return_read_true": int(
                metadata["future_return_read"].fillna(False).sum()
            ),
        },
        "membership": membership_summary,
        "weights": weight_summary,
        "event_coverage": event_summary,
        "facts": facts,
        "gates": gates,
        "limitations": [
            "HISTORICAL_MEMBERSHIP_IS_THIRD_PARTY_RECONSTRUCTION_NOT_OFFICIAL_ARCHIVE",
            "WEIGHT_HISTORY_VENDOR_REVISION_VINTAGE_NOT_PROVEN",
            "WEIGHT_HISTORY_STARTS_2016_08_NOT_2015",
            "NOTICE_DATE_AVAILABILITY_IS_CONSERVATIVELY_LAGGED_TO_NEXT_TRADING_DAY",
            "FACT_AUTOMATION_AND_DUAL_HUMAN_REVIEW_NOT_COMPLETE_UNLESS_EXPLICITLY_PASSED",
        ],
        "future_data_reads": 0,
        "market_price_reads": 0,
        "future_10d_label_created": False,
        "prediction_model_created": False,
        "portfolio_evaluation_performed": False,
        "model_action": "ABSTAIN",
        "model_position_target": "UNSET",
        "existing_holdings_override_allowed": False,
        "paper_signal_allowed": False,
        "shadow_signal_allowed": False,
        "live_trading_authorized": False,
    }
    result_path = project_path(paths["result_json"])
    markdown_path = project_path(paths["result_markdown"])
    events_path = project_path(paths["member_event_audit"])
    atomic_write_parquet(events_path, events)
    result["artifacts"] = {
        "member_event_audit": {
            "path": paths["member_event_audit"],
            "sha256": sha256_file(events_path),
            "rows": int(len(events)),
        }
    }
    atomic_write_json(result_path, result)
    atomic_write_text(markdown_path, render_markdown(result))
    print(json.dumps({
        "status": status,
        "result_json": paths["result_json"],
        "result_markdown": paths["result_markdown"],
        "member_event_rows": len(events),
        "facts_progress": facts["progress_status"],
        "future_data_reads": 0,
        "market_price_reads": 0,
    }, ensure_ascii=False, indent=2))
    return 0 if status not in {
        config["status_machine"]["blocked_metadata"],
        config["status_machine"]["blocked_membership_or_weights"],
        config["status_machine"]["blocked_facts"],
    } else 3


if __name__ == "__main__":
    raise SystemExit(main())
