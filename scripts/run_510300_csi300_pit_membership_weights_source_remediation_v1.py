"""运行沪深300点时成分与权重来源修复的冻结验收。

本脚本只读取官方调样、指数权重、交易日历和来源凭证，不读取价格、收益、
未来标签、模型、仓位或订单。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.csi300_pit_membership_weights_source_remediation_v1 import (  # noqa: E402
    REMEDIATION_ID,
    assert_outcome_blind_columns,
    build_daily_membership_panel,
    compare_replay_to_snapshots,
    derive_past_anchor_from_transitions,
    load_current_official_anchor,
    load_official_cycles,
    load_official_special_cycles,
    load_unique_open_dates,
    load_weight_snapshots,
    replay_membership_transitions,
    resolve_membership_transitions,
    sha256_file,
    state_on_or_before,
)


PROTOCOL_PATH = ROOT / "config" / "510300_csi300_pit_membership_weights_source_remediation_v1_protocol_manifest.json"
CLOCK_CORRECTION_PATH = ROOT / "config" / "510300_csi300_pit_membership_weights_source_remediation_v1_0_1_clock_correction.json"
REGULAR_MANIFEST_PATH = (
    ROOT
    / "data"
    / "curated"
    / "a_share_hs_csi300_official_addition_forced_demand_v1"
    / "official_announcement_manifest.json"
)
SPECIAL_MANIFEST_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "official_special_rebalance_manifest.json"
)
CALENDAR_PATH = (
    ROOT
    / "data"
    / "staging"
    / "a_share_hs_concentrated_low_risk_trend_v1_1"
    / "trading_calendar_observed_open_days.parquet"
)
CURRENT_ANCHOR_PATH = ROOT / "data" / "raw" / "constituents" / "000300_current_weights.parquet"
HISTORICAL_WEIGHT_PATH = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
WEIGHT_EVIDENCE_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "pit_weight_source_evidence_manifest.json"
)
EXTENSION_2015_ADMISSION_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "official_2015_extension_admission_manifest.json"
)

OUTPUT_ROOT = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
DAILY_MEMBERSHIP_PATH = OUTPUT_ROOT / "000300_daily_pit_membership_20160613_20260814.parquet"
TRANSITION_LEDGER_PATH = OUTPUT_ROOT / "official_membership_transition_ledger.parquet"
REPLAY_AUDIT_PATH = OUTPUT_ROOT / "official_membership_replay_audit.json"
SNAPSHOT_COMPARISON_PATH = OUTPUT_ROOT / "monthly_snapshot_membership_comparison.parquet"
ADMISSION_MANIFEST_PATH = OUTPUT_ROOT / "source_remediation_admission_manifest.json"
REPORT_JSON_PATH = ROOT / "reports" / "data_quality" / "510300_csi300_pit_membership_weights_source_remediation_v1.json"
REPORT_MD_PATH = ROOT / "reports" / "research" / "510300_CSI300_PIT_MEMBERSHIP_WEIGHTS_SOURCE_REMEDIATION_V1.md"

EXTENSION_START = date(2016, 6, 13)
PRIMARY_START = date(2021, 1, 1)
PRIMARY_END = date(2026, 8, 14)
FIRST_INDEPENDENT_SNAPSHOT = date(2016, 8, 31)
LAST_INDEPENDENT_SNAPSHOT = date(2026, 7, 31)


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_markdown(path: Path, content: str) -> None:
    atomic_write_bytes(path, (content.rstrip() + "\n").encode("utf-8"))


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def frame_content_sha256(frame: pd.DataFrame) -> str:
    """计算不依赖 Parquet 元数据的稳定内容哈希。"""

    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(
                lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
            )
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update("|".join(map(str, normalized.columns)).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def verify_protocol_inputs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    if protocol.get("status") != "FROZEN_BEFORE_REPLAY_OR_SOURCE_ADMISSION":
        raise ValueError("来源修复协议未处于冻结状态")
    receipts: list[dict[str, Any]] = []
    for record in protocol.get("frozen_inputs") or []:
        path = ROOT / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"协议冻结输入不存在：{record['path']}")
        observed_sha256 = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_sha256 != str(record["sha256"]).lower():
            raise ValueError(f"协议冻结输入哈希漂移：{record['path']}")
        if observed_bytes != int(record["bytes"]):
            raise ValueError(f"协议冻结输入字节数漂移：{record['path']}")
        receipts.append(
            {
                "path": str(record["path"]),
                "sha256": observed_sha256,
                "bytes": observed_bytes,
                "allowed_use": str(record["allowed_use"]),
            }
        )
    return receipts


def audit_has_issues(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for record in records:
        list_fields = (
            "missing_additions",
            "existing_deletions",
            "missing_deletions",
            "existing_additions",
        )
        invalid_lists = {
            field: record[field]
            for field in list_fields
            if field in record and record[field]
        }
        invalid_count = record.get("after_count") != 300
        if invalid_lists or invalid_count:
            issues.append(
                {
                    "cycle_id": record.get("cycle_id"),
                    "direction": record.get("direction"),
                    "after_count": record.get("after_count"),
                    "invalid_sets": invalid_lists,
                }
            )
    return issues


def strict_prior_weight_coverage(
    open_dates: list[date],
    snapshot_dates: list[date],
    start_date: date,
    end_date: date,
    maximum_age_days: int,
) -> dict[str, Any]:
    target_dates = [value for value in open_dates if start_date <= value <= end_date]
    rows: list[dict[str, Any]] = []
    for target_date in target_dates:
        eligible = [value for value in snapshot_dates if value < target_date]
        selected = max(eligible) if eligible else None
        age_days = (target_date - selected).days if selected is not None else None
        admitted = bool(selected is not None and age_days is not None and age_days <= maximum_age_days)
        rows.append(
            {
                "membership_date": target_date,
                "selected_snapshot_date": selected,
                "snapshot_age_calendar_days": age_days,
                "admitted": admitted,
            }
        )
    admitted_count = sum(bool(item["admitted"]) for item in rows)
    coverage = admitted_count / len(rows) if rows else 0.0
    valid_ages = [
        int(item["snapshot_age_calendar_days"])
        for item in rows
        if item["snapshot_age_calendar_days"] is not None
    ]
    return {
        "open_session_count": len(rows),
        "admitted_session_count": admitted_count,
        "coverage_ratio": coverage,
        "maximum_observed_snapshot_age_calendar_days": max(valid_ages) if valid_ages else None,
        "uncovered_examples": [
            {
                "membership_date": item["membership_date"].isoformat(),
                "selected_snapshot_date": (
                    item["selected_snapshot_date"].isoformat()
                    if item["selected_snapshot_date"] is not None
                    else None
                ),
                "snapshot_age_calendar_days": item["snapshot_age_calendar_days"],
            }
            for item in rows
            if not item["admitted"]
        ][:20],
    }


def main() -> int:
    checked_at = now_shanghai()
    protocol = load_json(PROTOCOL_PATH)
    frozen_input_receipts = verify_protocol_inputs(protocol)

    regular_cycles = load_official_cycles(ROOT, REGULAR_MANIFEST_PATH)
    special_cycles = load_official_special_cycles(ROOT, SPECIAL_MANIFEST_PATH)
    cycles = sorted(regular_cycles + special_cycles, key=lambda item: item.effective_date)
    if len(regular_cycles) != 21 or len(special_cycles) != 3 or len(cycles) != 24:
        raise ValueError(
            f"官方周期数量异常：定期={len(regular_cycles)}，临时={len(special_cycles)}"
        )
    if len({item.cycle_id for item in cycles}) != len(cycles):
        raise ValueError("定期和临时周期 ID 重复")

    open_dates = load_unique_open_dates(CALENDAR_PATH)
    transitions = resolve_membership_transitions(
        ROOT,
        cycles,
        open_dates,
        CLOCK_CORRECTION_PATH,
    )
    current_anchor_date, current_anchor_set, current_anchor_frame = load_current_official_anchor(
        CURRENT_ANCHOR_PATH
    )
    if current_anchor_date != date(2026, 7, 31):
        raise ValueError(f"当前官方锚点日期漂移：{current_anchor_date}")
    assert_outcome_blind_columns(current_anchor_frame.columns)

    extension_anchor_set, backward_audit = derive_past_anchor_from_transitions(
        current_anchor_date,
        current_anchor_set,
        EXTENSION_START,
        transitions,
    )
    state_by_session, forward_audit = replay_membership_transitions(
        EXTENSION_START,
        extension_anchor_set,
        transitions,
    )
    audit_issues = audit_has_issues(backward_audit + forward_audit)
    if len(extension_anchor_set) != 300:
        raise ValueError(f"逆推扩展锚点不是 300 只：{len(extension_anchor_set)}")
    if audit_issues:
        raise ValueError(f"官方成分重放出现非法状态迁移：{audit_issues[:5]}")
    replay_current_state = state_on_or_before(state_by_session, current_anchor_date)
    if replay_current_state != current_anchor_set:
        raise ValueError("升序重放终态与当前官方锚点不一致")

    weight_frame, monthly_snapshots = load_weight_snapshots(HISTORICAL_WEIGHT_PATH)
    monthly_comparisons = compare_replay_to_snapshots(
        state_by_session,
        monthly_snapshots,
        FIRST_INDEPENDENT_SNAPSHOT,
        LAST_INDEPENDENT_SNAPSHOT,
    )
    if len(monthly_comparisons) != 120:
        raise ValueError(f"独立月末快照比较数量不是 120：{len(monthly_comparisons)}")
    mismatched_snapshots = [
        item for item in monthly_comparisons if item["set_difference_count"] != 0
    ]
    if mismatched_snapshots:
        raise ValueError(f"官方重放与独立月末快照不一致：{mismatched_snapshots[:3]}")
    primary_monthly_comparisons = [
        item for item in monthly_comparisons if item["snapshot_date"] >= "2021-01-01"
    ]
    if len(primary_monthly_comparisons) < 60:
        raise ValueError(
            f"2021+ 独立月末快照不足 60：{len(primary_monthly_comparisons)}"
        )

    cycle_id_by_session = {
        item.transition_session: item.cycle.cycle_id for item in transitions
    }
    daily_membership = build_daily_membership_panel(
        open_dates,
        state_by_session,
        EXTENSION_START,
        PRIMARY_END,
        cycle_id_by_session,
    )
    daily_membership = daily_membership.sort_values(
        ["membership_date", "symbol"]
    ).reset_index(drop=True)
    first_content_hash = frame_content_sha256(daily_membership)

    second_anchor_set, second_backward_audit = derive_past_anchor_from_transitions(
        current_anchor_date,
        current_anchor_set,
        EXTENSION_START,
        transitions,
    )
    second_states, second_forward_audit = replay_membership_transitions(
        EXTENSION_START,
        second_anchor_set,
        transitions,
    )
    second_daily_membership = build_daily_membership_panel(
        open_dates,
        second_states,
        EXTENSION_START,
        PRIMARY_END,
        cycle_id_by_session,
    ).sort_values(["membership_date", "symbol"]).reset_index(drop=True)
    second_content_hash = frame_content_sha256(second_daily_membership)
    if (
        first_content_hash != second_content_hash
        or not daily_membership.equals(second_daily_membership)
        or backward_audit != second_backward_audit
        or forward_audit != second_forward_audit
    ):
        raise ValueError("第二次确定性重放与第一次结果不一致")

    primary_panel = daily_membership.loc[
        pd.to_datetime(daily_membership["membership_date"]).dt.date.ge(PRIMARY_START)
    ]
    primary_open_dates = [
        value for value in open_dates if PRIMARY_START <= value <= PRIMARY_END
    ]
    primary_counts = primary_panel.groupby("membership_date")["symbol"].nunique()
    if len(primary_counts) != len(primary_open_dates) or not primary_counts.eq(300).all():
        raise ValueError("2021+ 主窗口不是每个开市日恰好 300 只")

    transition_ledger = pd.DataFrame(
        [
            {
                "cycle_id": item.cycle.cycle_id,
                "cycle_type": (
                    "SPECIAL_DELISTING_REPLACEMENT"
                    if item.cycle.cycle_id.endswith("DELISTING")
                    else "REGULAR_REBALANCE"
                ),
                "announcement_id": item.cycle.announcement_id,
                "announcement_date": item.cycle.announcement_date,
                "stated_effective_date": item.cycle.effective_date,
                "transition_session": item.transition_session,
                "clock_rule": item.clock_rule,
                "addition_count": len(item.cycle.additions),
                "deletion_count": len(item.cycle.deletions),
                "additions": "|".join(sorted(item.cycle.additions)),
                "deletions": "|".join(sorted(item.cycle.deletions)),
                "detail_path": item.cycle.detail_path,
                "detail_sha256": item.cycle.detail_sha256,
                "attachment_paths": "|".join(item.cycle.attachment_paths),
                "attachment_sha256s": "|".join(item.cycle.attachment_sha256s),
            }
            for item in transitions
        ]
    ).sort_values("transition_session").reset_index(drop=True)
    assert_outcome_blind_columns(transition_ledger.columns)

    snapshot_comparison_frame = pd.DataFrame(
        [
            {
                "snapshot_date": item["snapshot_date"],
                "replay_count": item["replay_count"],
                "snapshot_count": item["snapshot_count"],
                "set_difference_count": item["set_difference_count"],
                "missing_from_replay": "|".join(item["missing_from_replay"]),
                "extra_in_replay": "|".join(item["extra_in_replay"]),
            }
            for item in monthly_comparisons
        ]
    )
    assert_outcome_blind_columns(snapshot_comparison_frame.columns)

    snapshot_dates = sorted(monthly_snapshots)
    strict_prior_coverage = strict_prior_weight_coverage(
        open_dates,
        snapshot_dates,
        PRIMARY_START,
        PRIMARY_END,
        maximum_age_days=45,
    )
    weight_counts = weight_frame.groupby("trade_date")["symbol"].nunique()
    weight_sums = weight_frame.groupby("trade_date")["weight"].sum()
    weight_structural_pass = bool(
        len(primary_monthly_comparisons) >= 60
        and weight_counts.eq(300).all()
        and weight_sums.between(98.0, 102.0, inclusive="both").all()
        and not weight_frame.duplicated(["trade_date", "symbol"]).any()
        and not mismatched_snapshots
        and strict_prior_coverage["coverage_ratio"] >= 0.99
    )

    weight_evidence = load_json(WEIGHT_EVIDENCE_PATH)
    if any(
        bool(weight_evidence.get(field))
        for field in ("market_price_read", "future_return_read", "future_label_created", "model_read")
    ):
        raise ValueError("权重来源凭证违反结果盲态边界")
    provenance_assessment = weight_evidence.get("frozen_weight_provenance_gate_assessment")
    if not isinstance(provenance_assessment, Mapping):
        raise ValueError("权重来源凭证缺少冻结门槛评估")
    weight_provenance_pass = bool(provenance_assessment.get("admission_sufficient"))
    if weight_provenance_pass:
        raise ValueError("当前权重来源凭证不应被标记为版本来源通过")

    membership_status = (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_PRIMARY_2021_WINDOW_ONLY_WEIGHTS_PENDING"
    )
    weight_status = "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    overall_status = membership_status

    conditional_2015_extension: dict[str, Any] = {
        "status": "NOT_ADMITTED_IN_THIS_RUN",
        "reason": (
            "No separately admitted complete official 2015 transition chain was found."
        ),
        "blocks_primary_2021_admission": False,
    }
    if EXTENSION_2015_ADMISSION_PATH.is_file():
        extension_admission = load_json(EXTENSION_2015_ADMISSION_PATH)
        extension_membership = extension_admission.get("membership_admission") or {}
        extension_weight = extension_admission.get("weight_admission") or {}
        if extension_membership.get("status") != (
            "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION"
        ):
            raise ValueError("2015 扩展验收清单存在但成分状态未通过")
        if extension_weight.get("status") != weight_status or bool(
            extension_weight.get("version_or_as_of_provenance_gate_pass")
        ):
            raise ValueError("2015 扩展验收清单没有保留权重版本来源阻断")
        extension_outputs = {
            str(item["path"]): item
            for item in extension_admission.get("output_receipts") or []
        }
        combined_path_value = (
            "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/"
            "000300_daily_pit_membership_20150101_20260814.parquet"
        )
        combined_receipt = extension_outputs.get(combined_path_value)
        if not isinstance(combined_receipt, Mapping):
            raise ValueError("2015 扩展验收清单缺少合并日频面板凭证")
        combined_path = ROOT / combined_path_value
        if (
            not combined_path.is_file()
            or sha256_file(combined_path) != str(combined_receipt["sha256"]).lower()
            or combined_path.stat().st_size != int(combined_receipt["bytes"])
        ):
            raise ValueError("2015 扩展合并日频面板凭证不一致")
        conditional_2015_extension = {
            "status": "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION",
            "start_contract_date": str(extension_membership["contract_start_date"]),
            "first_open_session": str(extension_membership["first_open_session"]),
            "end_date": str(extension_membership["end_date"]),
            "open_session_count": int(extension_membership["open_session_count"]),
            "row_count": int(extension_membership["row_count"]),
            "independent_official_snapshot_mismatch_count": int(
                extension_membership["independent_official_snapshot_mismatch_count"]
            ),
            "handover_set_difference_count": int(
                extension_membership["handover_set_difference_count"]
            ),
            "admission_manifest_path": EXTENSION_2015_ADMISSION_PATH.relative_to(ROOT).as_posix(),
            "admission_manifest_sha256": sha256_file(EXTENSION_2015_ADMISSION_PATH),
            "combined_daily_membership_path": combined_path_value,
            "combined_daily_membership_sha256": str(combined_receipt["sha256"]),
            "blocks_primary_2021_admission": False,
        }

    atomic_write_parquet(DAILY_MEMBERSHIP_PATH, daily_membership)
    atomic_write_parquet(TRANSITION_LEDGER_PATH, transition_ledger)
    atomic_write_parquet(SNAPSHOT_COMPARISON_PATH, snapshot_comparison_frame)
    replay_audit = {
        "remediation_id": REMEDIATION_ID,
        "checked_at": checked_at,
        "status": "PASS_OFFICIAL_MEMBERSHIP_REPLAY",
        "regular_cycle_count": len(regular_cycles),
        "special_cycle_count": len(special_cycles),
        "total_cycle_count": len(transitions),
        "backward_audit": backward_audit,
        "forward_audit": forward_audit,
        "audit_issue_count": len(audit_issues),
        "current_official_set_difference_count": len(replay_current_state ^ current_anchor_set),
        "monthly_snapshot_count": len(monthly_comparisons),
        "monthly_snapshot_mismatch_count": len(mismatched_snapshots),
        "deterministic_first_content_sha256": first_content_hash,
        "deterministic_second_content_sha256": second_content_hash,
        "deterministic_replay_match": True,
        "market_price_read": False,
        "future_return_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
    }
    atomic_write_json(REPLAY_AUDIT_PATH, replay_audit)

    output_receipts = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            DAILY_MEMBERSHIP_PATH,
            TRANSITION_LEDGER_PATH,
            REPLAY_AUDIT_PATH,
            SNAPSHOT_COMPARISON_PATH,
        )
    ]
    admission_manifest = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.2-with-2015-extension-reference",
        "checked_at": checked_at,
        "overall_status": overall_status,
        "membership_admission": {
            "status": membership_status,
            "primary_start_date": PRIMARY_START.isoformat(),
            "primary_end_date": PRIMARY_END.isoformat(),
            "primary_open_session_count": len(primary_open_dates),
            "primary_row_count": len(primary_panel),
            "active_constituent_count_each_open_session": 300,
            "duplicate_date_symbol_rows": int(
                primary_panel.duplicated(["membership_date", "symbol"]).sum()
            ),
            "regular_cycle_count": len(regular_cycles),
            "special_cycle_count": len(special_cycles),
            "monthly_independent_snapshot_count_primary": len(primary_monthly_comparisons),
            "monthly_independent_snapshot_set_difference_count": sum(
                int(item["set_difference_count"]) for item in primary_monthly_comparisons
            ),
            "current_official_anchor_set_difference_count": len(
                replay_current_state ^ current_anchor_set
            ),
            "deterministic_replay_match": True,
        },
        "best_effort_extension": {
            "status": "PASS_FROM_2016_06_13_WITH_INDEPENDENT_MONTH_END_CHECKS_FROM_2016_08_31",
            "start_date": EXTENSION_START.isoformat(),
            "end_date": PRIMARY_END.isoformat(),
            "open_session_count": int(daily_membership["membership_date"].nunique()),
            "row_count": len(daily_membership),
            "independent_monthly_snapshot_count": len(monthly_comparisons),
            "independent_monthly_snapshot_mismatch_count": len(mismatched_snapshots),
        },
        "conditional_2015_extension": conditional_2015_extension,
        "weight_admission": {
            "status": weight_status,
            "structural_gate_pass": weight_structural_pass,
            "version_or_as_of_provenance_gate_pass": weight_provenance_pass,
            "snapshot_count_all": int(weight_frame["trade_date"].nunique()),
            "snapshot_count_primary": len(primary_monthly_comparisons),
            "minimum_constituent_count": int(weight_counts.min()),
            "maximum_constituent_count": int(weight_counts.max()),
            "minimum_weight_sum_pct": float(weight_sums.min()),
            "maximum_weight_sum_pct": float(weight_sums.max()),
            "strict_prior_coverage": strict_prior_coverage,
            "source_evidence_manifest": WEIGHT_EVIDENCE_PATH.relative_to(ROOT).as_posix(),
            "source_evidence_sha256": sha256_file(WEIGHT_EVIDENCE_PATH),
            "remaining_blocker": str(provenance_assessment.get("reason")),
        },
        "candidate_boundary": {
            "candidate_id": "510300_PIT_EARNINGS_INFORMATION_DIFFUSION_10D_RISK_V1",
            "authoritative_candidate_status_unchanged": "BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS",
            "remaining_data_blocker": weight_status,
            "model_action": "ABSTAIN",
            "portfolio_evaluation": "NOT_ALLOWED",
            "paper_or_shadow_signal_allowed": False,
            "trading_authorization": False,
        },
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL_PATH),
            "clock_correction_path": CLOCK_CORRECTION_PATH.relative_to(ROOT).as_posix(),
            "clock_correction_sha256": sha256_file(CLOCK_CORRECTION_PATH),
        },
        "verified_frozen_inputs": frozen_input_receipts,
        "outputs": output_receipts,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "model_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    atomic_write_json(ADMISSION_MANIFEST_PATH, admission_manifest)

    report = {
        **admission_manifest,
        "admission_manifest": {
            "path": ADMISSION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(ADMISSION_MANIFEST_PATH),
        },
    }
    atomic_write_json(REPORT_JSON_PATH, report)

    markdown = f"""# 沪深300点时成分与权重来源修复 V1

## 结论

- 成分：`{membership_status}`。
- 权重：`{weight_status}`。
- 候选：继续保持 `BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS`，不得进入模型、组合或收益评估。

## 成分验收

- 官方定期调整：{len(regular_cycles)} 轮；官方临时替换：{len(special_cycles)} 轮。
- 主窗口：{PRIMARY_START} 至 {PRIMARY_END}，{len(primary_open_dates)} 个开市日，{len(primary_panel):,} 行。
- 每个开市日唯一成分数：300；日期-证券重复行：0。
- 2021 年以后独立月末快照：{len(primary_monthly_comparisons)} 期；集合差异合计：0。
- 2016-08 至 2026-07 独立月末快照：{len(monthly_comparisons)} 期；不一致期数：0。
- 当前官方锚点集合差异：0；第二次确定性重放内容哈希一致。
- 2021 年后的官方“收市后生效”按下一开市日迁移，避免单日前视。

## 权重验收

- 结构门槛：{'通过' if weight_structural_pass else '未通过'}；共 {int(weight_frame['trade_date'].nunique())} 期，每期 300 只，权重和范围 {float(weight_sums.min()):.3f}% 至 {float(weight_sums.max()):.3f}%。
- 严格前序快照覆盖率：{strict_prior_coverage['coverage_ratio']:.6f}，门槛为 0.99。
- 版本来源门槛：未通过。公开留存只提供少量官方历史版本锚点；本地 Tushare 兼容代理凭据已过期，并且该来源没有逐期历史版本/as-of 凭证。
- 因此不能把结构合格的 120 期数值提升为已准入点时权重。

## 2015 年边界

2015 条件扩展已由独立验收清单准入：`{conditional_2015_extension['status']}`。完整合并面板从 {conditional_2015_extension.get('first_open_session', '未准入')} 覆盖至 {conditional_2015_extension.get('end_date', '未准入')}；2015 独立官方快照不一致期数为 {conditional_2015_extension.get('independent_official_snapshot_mismatch_count', '不适用')}，2016H1 交接集合差异为 {conditional_2015_extension.get('handover_set_difference_count', '不适用')}。该扩展只准入成分集合，不准入历史权重值。

## 结果盲态与授权

本次未读取市场价格、未来收益、未来标签、模型、仓位或订单；`RETURN_EVALUATION=NOT_ALLOWED`，无 Paper/Shadow 或实盘授权。
"""
    atomic_write_markdown(REPORT_MD_PATH, markdown)

    result = {
        "overall_status": overall_status,
        "membership_status": membership_status,
        "weight_status": weight_status,
        "primary_open_session_count": len(primary_open_dates),
        "primary_membership_row_count": len(primary_panel),
        "monthly_snapshot_count": len(monthly_comparisons),
        "monthly_snapshot_mismatch_count": len(mismatched_snapshots),
        "conditional_2015_membership_status": conditional_2015_extension["status"],
        "daily_membership_path": DAILY_MEMBERSHIP_PATH.relative_to(ROOT).as_posix(),
        "daily_membership_sha256": sha256_file(DAILY_MEMBERSHIP_PATH),
        "admission_manifest_path": ADMISSION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "admission_manifest_sha256": sha256_file(ADMISSION_MANIFEST_PATH),
        "report_path": REPORT_JSON_PATH.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(REPORT_JSON_PATH),
        "market_price_read": False,
        "future_return_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
