"""重放、验收并交付 2015-01-01 起的沪深300点时成分扩展。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.csi300_pit_membership_weights_source_remediation_v1 import (  # noqa: E402
    REMEDIATION_ID,
    MembershipTransition,
    assert_outcome_blind_columns,
    build_daily_membership_panel,
    derive_past_anchor_from_transitions,
    load_official_2015_extension_cycles,
    load_official_cycles,
    load_official_special_cycles,
    load_unique_open_dates,
    replay_membership_transitions,
    resolve_membership_transitions,
    sha256_file,
    state_on_or_before,
)
from scripts.acquire_510300_csi300_pit_weights_evidence_v1 import (  # noqa: E402
    parse_official_closeweight,
)


PROTOCOL_PATH = ROOT / "config" / "510300_csi300_pit_membership_weights_source_remediation_v1_protocol_manifest.json"
SOURCE_MANIFEST_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "official_2015_extension_manifest.json"
)
CLOCK_ADDENDUM_PATH = (
    ROOT
    / "config"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_0_2_2015_extension_clock_addendum.json"
)
ORIGINAL_CLOCK_PATH = (
    ROOT
    / "config"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_0_1_clock_correction.json"
)
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
MAIN_DAILY_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "000300_daily_pit_membership_20160613_20260814.parquet"
)
MAIN_ADMISSION_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "source_remediation_admission_manifest.json"
)
WEIGHT_EVIDENCE_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "pit_weight_source_evidence_manifest.json"
)
OUTPUT_ROOT = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
EXTENSION_DAILY_PATH = OUTPUT_ROOT / "000300_daily_pit_membership_20150101_20160612.parquet"
COMBINED_DAILY_PATH = OUTPUT_ROOT / "000300_daily_pit_membership_20150101_20260814.parquet"
TRANSITION_LEDGER_PATH = OUTPUT_ROOT / "official_2015_extension_transition_ledger.parquet"
SNAPSHOT_COMPARISON_PATH = OUTPUT_ROOT / "official_2015_extension_snapshot_comparison.parquet"
REPLAY_AUDIT_PATH = OUTPUT_ROOT / "official_2015_extension_replay_audit.json"
ADMISSION_PATH = OUTPUT_ROOT / "official_2015_extension_admission_manifest.json"
REPORT_JSON_PATH = ROOT / "reports" / "data_quality" / "510300_csi300_pit_membership_2015_extension_v1.json"
REPORT_MD_PATH = ROOT / "reports" / "research" / "510300_CSI300_PIT_MEMBERSHIP_2015_EXTENSION_V1.md"

EXTENSION_START = date(2015, 1, 1)
EXTENSION_END = date(2016, 6, 12)
HANDOVER_SESSION = date(2016, 6, 13)
FULL_END = date(2026, 8, 14)
ANCHOR_DATE = date(2015, 3, 2)
INDEPENDENT_ANCHOR_DATES = (date(2015, 7, 1), date(2015, 9, 1))


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
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def atomic_write_markdown(path: Path, content: str) -> None:
    atomic_write_bytes(path, (content.rstrip() + "\n").encode("utf-8"))


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def frame_content_sha256(frame: pd.DataFrame) -> str:
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


def verify_frozen_inputs(manifest: Mapping[str, Any], expected_status: str) -> list[dict[str, Any]]:
    if manifest.get("status") != expected_status:
        raise ValueError(f"冻结清单状态不符：{manifest.get('status')}")
    receipts: list[dict[str, Any]] = []
    for record in manifest.get("frozen_inputs") or []:
        path = ROOT / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入不存在：{record['path']}")
        observed = sha256_file(path)
        if observed != str(record["sha256"]).lower():
            raise ValueError(f"冻结输入哈希漂移：{record['path']}")
        if int(path.stat().st_size) != int(record["bytes"]):
            raise ValueError(f"冻结输入字节数漂移：{record['path']}")
        receipts.append(
            {
                "path": str(record["path"]),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
    return receipts


def load_anchor_sets(source_manifest: Mapping[str, Any]) -> dict[date, frozenset[str]]:
    result: dict[date, frozenset[str]] = {}
    for record in source_manifest.get("official_membership_anchors") or []:
        path = ROOT / str(record["archive_path"])
        if not path.is_file() or sha256_file(path) != str(record["sha256"]).lower():
            raise ValueError(f"官方 2015 锚点不存在或哈希漂移：{record.get('snapshot_date')}")
        parsed = parse_official_closeweight(path.read_bytes())
        snapshot_date = parsed.snapshot_date
        symbols = frozenset(parsed.frame["symbol"])
        if len(symbols) != 300:
            raise ValueError(f"官方 2015 锚点不是 300 只：{snapshot_date}")
        set_digest = hashlib.sha256("\n".join(sorted(symbols)).encode("utf-8")).hexdigest()
        if set_digest != str(record["constituent_set_sha256"]):
            raise ValueError(f"官方 2015 锚点集合哈希漂移：{snapshot_date}")
        result[snapshot_date] = symbols
    required = {ANCHOR_DATE, *INDEPENDENT_ANCHOR_DATES}
    if set(result) != required:
        raise ValueError(f"官方 2015 锚点日期不完整：{sorted(result)}")
    return result


def replay_once(
    transitions: Sequence[MembershipTransition],
    open_dates: Sequence[date],
    anchor_set: frozenset[str],
) -> tuple[pd.DataFrame, dict[date, frozenset[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    start_set, backward_audit = derive_past_anchor_from_transitions(
        ANCHOR_DATE,
        anchor_set,
        EXTENSION_START,
        transitions,
    )
    if len(start_set) != 300:
        raise ValueError(f"2015-01-01 逆推锚点不是 300 只：{len(start_set)}")
    states, forward_audit = replay_membership_transitions(
        EXTENSION_START,
        start_set,
        transitions,
    )
    issues = [
        item
        for item in backward_audit + forward_audit
        if item.get("missing_additions")
        or item.get("existing_deletions")
        or item.get("missing_deletions")
        or item.get("existing_additions")
        or int(item.get("after_count", 300)) != 300
    ]
    if issues:
        raise ValueError(f"2015 官方成分重放出现非法迁移：{issues[:3]}")
    cycle_by_session = {
        item.transition_session: item.cycle.cycle_id for item in transitions
    }
    panel = build_daily_membership_panel(
        open_dates,
        states,
        EXTENSION_START,
        EXTENSION_END,
        cycle_by_session,
        source_contract="CSI_OFFICIAL_REBALANCE_REPLAY_V1_0_2_2015_EXTENSION",
        anchor_cycle_id="REVERSED_2015_03_02_OFFICIAL_ANCHOR",
    ).sort_values(["membership_date", "symbol"]).reset_index(drop=True)
    return panel, states, backward_audit, forward_audit


def main() -> int:
    checked_at = now_shanghai()
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_REPLAY_OR_SOURCE_ADMISSION":
        raise ValueError("来源修复协议未处于冻结状态")
    clock_addendum = load_json(CLOCK_ADDENDUM_PATH)
    frozen_input_receipts = verify_frozen_inputs(
        clock_addendum,
        "FROZEN_BEFORE_DAILY_MEMBERSHIP_OUTPUT",
    )
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    if any(
        bool(source_manifest.get(field))
        for field in ("market_price_read", "future_return_read", "future_label_created", "model_read")
    ):
        raise ValueError("2015 扩展来源清单违反结果盲态边界")

    open_dates = load_unique_open_dates(CALENDAR_PATH)
    extension_cycles = load_official_2015_extension_cycles(ROOT, SOURCE_MANIFEST_PATH)
    if len(extension_cycles) != 5:
        raise ValueError(f"2015 官方扩展周期数量不是 5：{len(extension_cycles)}")
    extension_transitions = resolve_membership_transitions(
        ROOT,
        extension_cycles,
        open_dates,
        CLOCK_ADDENDUM_PATH,
    )
    anchor_sets = load_anchor_sets(source_manifest)

    extension_panel, states, backward_audit, forward_audit = replay_once(
        extension_transitions,
        open_dates,
        anchor_sets[ANCHOR_DATE],
    )
    first_extension_content_hash = frame_content_sha256(extension_panel)

    comparisons: list[dict[str, Any]] = []
    for snapshot_date in (ANCHOR_DATE, *INDEPENDENT_ANCHOR_DATES):
        replay_set = state_on_or_before(states, snapshot_date)
        official_set = anchor_sets[snapshot_date]
        comparisons.append(
            {
                "snapshot_date": snapshot_date,
                "role": "REPLAY_ANCHOR" if snapshot_date == ANCHOR_DATE else "INDEPENDENT_OFFICIAL_CHECK",
                "replay_count": len(replay_set),
                "official_count": len(official_set),
                "set_difference_count": len(replay_set ^ official_set),
                "missing_from_replay": "|".join(sorted(official_set - replay_set)),
                "extra_in_replay": "|".join(sorted(replay_set - official_set)),
            }
        )
    if any(item["set_difference_count"] for item in comparisons):
        raise ValueError(f"2015 官方锚点比较不一致：{comparisons}")

    regular_cycles = load_official_cycles(ROOT, REGULAR_MANIFEST_PATH)
    special_cycles = load_official_special_cycles(ROOT, SPECIAL_MANIFEST_PATH)
    original_cycles = sorted(regular_cycles + special_cycles, key=lambda item: item.effective_date)
    original_transitions = resolve_membership_transitions(
        ROOT,
        original_cycles,
        open_dates,
        ORIGINAL_CLOCK_PATH,
    )
    handover_matches = [
        item
        for item in original_transitions
        if item.transition_session == HANDOVER_SESSION and item.cycle.announcement_id == 3855
    ]
    if len(handover_matches) != 1:
        raise ValueError("2016H1 官方主链交接迁移未唯一定位")
    handover = handover_matches[0]
    pre_handover_set = state_on_or_before(states, EXTENSION_END)
    missing_deletions = sorted(handover.cycle.deletions - pre_handover_set)
    existing_additions = sorted(handover.cycle.additions & pre_handover_set)
    if missing_deletions or existing_additions:
        raise ValueError(
            "2015 扩展与 2016H1 官方主链交接非法："
            f"缺失调出={missing_deletions}，已存在调入={existing_additions}"
        )
    post_handover_set = frozenset(
        (set(pre_handover_set) - set(handover.cycle.deletions))
        | set(handover.cycle.additions)
    )
    if len(post_handover_set) != 300:
        raise ValueError("2016H1 交接后的成分数不是 300")

    main_daily = pd.read_parquet(MAIN_DAILY_PATH)
    assert_outcome_blind_columns(main_daily.columns)
    main_daily = main_daily.copy()
    main_daily["membership_date"] = pd.to_datetime(
        main_daily["membership_date"], errors="raise"
    ).dt.date
    first_main_date = min(main_daily["membership_date"])
    if first_main_date != HANDOVER_SESSION:
        raise ValueError(f"主链首日不是 2016-06-13：{first_main_date}")
    first_main_set = frozenset(
        main_daily.loc[main_daily["membership_date"].eq(HANDOVER_SESSION), "symbol"]
    )
    handover_set_difference_count = len(post_handover_set ^ first_main_set)
    if handover_set_difference_count:
        raise ValueError("2015 扩展与 2016-06-13 已验收主链集合不一致")

    combined = pd.concat([extension_panel, main_daily], ignore_index=True)
    combined = combined.sort_values(["membership_date", "symbol"]).reset_index(drop=True)
    assert_outcome_blind_columns(combined.columns)
    if combined.duplicated(["membership_date", "symbol"]).any():
        raise ValueError("2015-2026 合并成分面板存在日期证券重复行")
    combined_counts = combined.groupby("membership_date")["symbol"].nunique()
    expected_open_dates = [
        value for value in open_dates if EXTENSION_START <= value <= FULL_END
    ]
    if list(combined_counts.index) != expected_open_dates or not combined_counts.eq(300).all():
        raise ValueError("2015-2026 合并面板未覆盖每个开市日恰好 300 只")
    first_combined_date = min(combined["membership_date"])
    last_combined_date = max(combined["membership_date"])
    if first_combined_date != expected_open_dates[0] or last_combined_date != FULL_END:
        raise ValueError("2015-2026 合并面板首尾交易日不符合契约")

    second_panel, second_states, second_backward, second_forward = replay_once(
        extension_transitions,
        open_dates,
        anchor_sets[ANCHOR_DATE],
    )
    second_extension_content_hash = frame_content_sha256(second_panel)
    if (
        not extension_panel.equals(second_panel)
        or first_extension_content_hash != second_extension_content_hash
        or states != second_states
        or backward_audit != second_backward
        or forward_audit != second_forward
    ):
        raise ValueError("2015 扩展第二次确定性重放与第一次不一致")

    transition_ledger = pd.DataFrame(
        [
            {
                "cycle_id": item.cycle.cycle_id,
                "cycle_type": (
                    "SPECIAL_REPLACEMENT"
                    if "SPECIAL" in item.cycle.cycle_id
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
            for item in extension_transitions
        ]
    ).sort_values("transition_session").reset_index(drop=True)
    snapshot_comparison = pd.DataFrame(comparisons).sort_values("snapshot_date").reset_index(drop=True)
    assert_outcome_blind_columns(transition_ledger.columns)
    assert_outcome_blind_columns(snapshot_comparison.columns)

    atomic_write_parquet(EXTENSION_DAILY_PATH, extension_panel)
    atomic_write_parquet(COMBINED_DAILY_PATH, combined)
    atomic_write_parquet(TRANSITION_LEDGER_PATH, transition_ledger)
    atomic_write_parquet(SNAPSHOT_COMPARISON_PATH, snapshot_comparison)

    replay_audit = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.2-2015-extension",
        "checked_at": checked_at,
        "status": "PASS_OFFICIAL_2015_MEMBERSHIP_REPLAY_AND_HANDOVER",
        "cycle_count": len(extension_transitions),
        "change_count_total": sum(len(item.cycle.additions) for item in extension_transitions),
        "backward_audit": backward_audit,
        "forward_audit": forward_audit,
        "anchor_set_difference_count": comparisons[0]["set_difference_count"],
        "independent_official_snapshot_count": 2,
        "independent_official_snapshot_mismatch_count": sum(
            int(item["set_difference_count"] != 0)
            for item in comparisons
            if item["role"] == "INDEPENDENT_OFFICIAL_CHECK"
        ),
        "handover_2016_h1_announcement_id": handover.cycle.announcement_id,
        "handover_set_difference_count": handover_set_difference_count,
        "deterministic_first_content_sha256": first_extension_content_hash,
        "deterministic_second_content_sha256": second_extension_content_hash,
        "deterministic_replay_match": True,
        "market_price_read": False,
        "future_return_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
    }
    atomic_write_json(REPLAY_AUDIT_PATH, replay_audit)

    main_admission = load_json(MAIN_ADMISSION_PATH)
    weight_evidence = load_json(WEIGHT_EVIDENCE_PATH)
    weight_assessment = weight_evidence.get("frozen_weight_provenance_gate_assessment")
    if not isinstance(weight_assessment, Mapping) or bool(weight_assessment.get("admission_sufficient")):
        raise ValueError("权重版本来源门槛状态异常")
    weight_status = str(weight_assessment.get("status"))
    if weight_status != "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS":
        raise ValueError(f"权重阻断状态漂移：{weight_status}")
    if main_admission.get("overall_status") != (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_PRIMARY_2021_WINDOW_ONLY_WEIGHTS_PENDING"
    ):
        raise ValueError("2021+ 主链验收状态漂移")

    output_receipts = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            EXTENSION_DAILY_PATH,
            COMBINED_DAILY_PATH,
            TRANSITION_LEDGER_PATH,
            SNAPSHOT_COMPARISON_PATH,
            REPLAY_AUDIT_PATH,
        )
    ]
    admission = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.2-2015-extension",
        "checked_at": checked_at,
        "overall_status": "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION_WEIGHTS_STILL_BLOCKED",
        "membership_admission": {
            "status": "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION",
            "contract_start_date": EXTENSION_START.isoformat(),
            "first_open_session": first_combined_date.isoformat(),
            "end_date": FULL_END.isoformat(),
            "open_session_count": int(combined["membership_date"].nunique()),
            "row_count": len(combined),
            "active_constituent_count_each_open_session": 300,
            "duplicate_date_symbol_rows": int(
                combined.duplicated(["membership_date", "symbol"]).sum()
            ),
            "official_2015_cycle_count": len(extension_transitions),
            "official_2015_change_count": sum(
                len(item.cycle.additions) for item in extension_transitions
            ),
            "official_anchor_set_difference_count": comparisons[0]["set_difference_count"],
            "independent_official_snapshot_count": 2,
            "independent_official_snapshot_mismatch_count": 0,
            "handover_set_difference_count": handover_set_difference_count,
            "deterministic_replay_match": True,
        },
        "weight_admission": {
            "status": weight_status,
            "version_or_as_of_provenance_gate_pass": False,
            "membership_anchor_use_does_not_admit_historical_weight_values": True,
        },
        "candidate_boundary": {
            "authoritative_candidate_status_unchanged": "BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS",
            "remaining_data_blocker": weight_status,
            "return_evaluation": "NOT_ALLOWED",
            "model_action": "ABSTAIN",
            "paper_or_shadow_signal_allowed": False,
            "trading_authorization": False,
        },
        "frozen_input_receipts": frozen_input_receipts,
        "output_receipts": output_receipts,
        "content_hashes": {
            "extension_daily_membership": first_extension_content_hash,
            "combined_daily_membership": frame_content_sha256(combined),
        },
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "model_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    atomic_write_json(ADMISSION_PATH, admission)
    atomic_write_json(REPORT_JSON_PATH, admission)

    report = f"""# 沪深300点时成分 2015 扩展验收报告

## 结论

- 成分状态：`{admission['membership_admission']['status']}`。
- 合并面板：{first_combined_date.isoformat()} 至 {FULL_END.isoformat()}，共 {admission['membership_admission']['open_session_count']} 个开市日、{admission['membership_admission']['row_count']} 行，每日恰好 300 只。
- 2015 官方调样：5 轮、共 {admission['membership_admission']['official_2015_change_count']} 进 {admission['membership_admission']['official_2015_change_count']} 出。
- 2015-03-02 官方锚点集合差异为 0；2015-07-01、2015-09-01 两个独立官方快照差异均为 0。
- 与 2016H1 官方主链在 2016-06-13 的集合差异为 0；第二次确定性重放哈希一致。
- 权重状态仍为 `{weight_status}`。2015 权重文件在这里只作为成分集合锚点，不能据此把历史权重值判为合格。

## 研究边界

本轮没有读取市场价格、未来收益或标签，没有运行预测模型、组合回测、Paper/Shadow 或实盘交易。被冻结候选仍保持 `BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS`；当前剩余数据障碍已缩窄为历史权重的逐期版本或 as-of 来源凭证。

## 主要交付物

- 完整日频成分面板：`{COMBINED_DAILY_PATH.relative_to(ROOT).as_posix()}`
- 2015 扩展来源清单：`{SOURCE_MANIFEST_PATH.relative_to(ROOT).as_posix()}`
- 2015 扩展验收清单：`{ADMISSION_PATH.relative_to(ROOT).as_posix()}`
- 重放审计：`{REPLAY_AUDIT_PATH.relative_to(ROOT).as_posix()}`
- 官方快照比较：`{SNAPSHOT_COMPARISON_PATH.relative_to(ROOT).as_posix()}`
"""
    atomic_write_markdown(REPORT_MD_PATH, report)

    final = {
        "overall_status": admission["overall_status"],
        "membership_status": admission["membership_admission"]["status"],
        "weight_status": weight_status,
        "first_open_session": first_combined_date.isoformat(),
        "last_open_session": last_combined_date.isoformat(),
        "open_session_count": admission["membership_admission"]["open_session_count"],
        "row_count": admission["membership_admission"]["row_count"],
        "official_2015_cycle_count": len(extension_transitions),
        "official_2015_change_count": admission["membership_admission"]["official_2015_change_count"],
        "independent_official_snapshot_mismatch_count": 0,
        "handover_set_difference_count": handover_set_difference_count,
        "deterministic_replay_match": True,
        "combined_daily_path": COMBINED_DAILY_PATH.relative_to(ROOT).as_posix(),
        "combined_daily_sha256": sha256_file(COMBINED_DAILY_PATH),
        "admission_path": ADMISSION_PATH.relative_to(ROOT).as_posix(),
        "admission_sha256": sha256_file(ADMISSION_PATH),
        "report_path": REPORT_MD_PATH.relative_to(ROOT).as_posix(),
        "return_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
