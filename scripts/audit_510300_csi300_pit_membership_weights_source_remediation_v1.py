"""独立复核沪深300点时成分交付，并确认权重阻断未被越过。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
SOURCE_MANIFEST_PATH = CURATED_ROOT / "official_2015_extension_manifest.json"
MAIN_ADMISSION_PATH = CURATED_ROOT / "source_remediation_admission_manifest.json"
EXTENSION_ADMISSION_PATH = CURATED_ROOT / "official_2015_extension_admission_manifest.json"
WEIGHT_EVIDENCE_PATH = CURATED_ROOT / "pit_weight_source_evidence_manifest.json"
MAIN_DAILY_PATH = CURATED_ROOT / "000300_daily_pit_membership_20160613_20260814.parquet"
EXTENSION_DAILY_PATH = CURATED_ROOT / "000300_daily_pit_membership_20150101_20160612.parquet"
COMBINED_DAILY_PATH = CURATED_ROOT / "000300_daily_pit_membership_20150101_20260814.parquet"
MAIN_TRANSITION_PATH = CURATED_ROOT / "official_membership_transition_ledger.parquet"
EXTENSION_TRANSITION_PATH = CURATED_ROOT / "official_2015_extension_transition_ledger.parquet"
MAIN_SNAPSHOT_PATH = CURATED_ROOT / "monthly_snapshot_membership_comparison.parquet"
EXTENSION_SNAPSHOT_PATH = CURATED_ROOT / "official_2015_extension_snapshot_comparison.parquet"
CALENDAR_PATH = (
    ROOT
    / "data"
    / "staging"
    / "a_share_hs_concentrated_low_risk_trend_v1_1"
    / "trading_calendar_observed_open_days.parquet"
)
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_independent_audit.json"
)

FORBIDDEN_WORDS = {
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "future",
    "label",
    "target",
    "position",
    "order",
    "pnl",
    "sharpe",
}
ALLOWED_CALENDAR_FIELDS = {"is_open", "open_date", "first_open_date", "next_open_date"}
EXPECTED_2015_TRANSITIONS = {
    "2015_SPECIAL_HYSEC_MERGER": (date(2015, 1, 26), 1),
    "2015_SPECIAL_CNR_OPG_DELISTING": (date(2015, 5, 20), 2),
    "2015_H1_REGULAR": (date(2015, 6, 15), 18),
    "2015_H2_REGULAR": (date(2015, 12, 14), 20),
    "2015_SPECIAL_CHINA_MERCHANTS_PROPERTY_MERGER": (date(2015, 12, 30), 1),
}


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_columns(columns: Iterable[object], label: str) -> None:
    violations: list[str] = []
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in ALLOWED_CALENDAR_FIELDS:
            continue
        words = {
            word
            for word in __import__("re").split(r"[^a-z0-9]+", normalized)
            if word
        }
        if words & FORBIDDEN_WORDS:
            violations.append(str(column))
    if violations:
        raise ValueError(f"{label} 出现禁止的结果或交易字段：{violations}")


def verify_receipts(records: Iterable[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for record in records:
        path = ROOT / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{label} 文件不存在：{record['path']}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != str(record["sha256"]).lower():
            raise ValueError(f"{label} 文件哈希不一致：{record['path']}")
        if int(path.stat().st_size) != int(record["bytes"]):
            raise ValueError(f"{label} 文件字节数不一致：{record['path']}")
        receipts.append(
            {
                "path": str(record["path"]),
                "sha256": observed_sha256,
                "bytes": path.stat().st_size,
            }
        )
    return receipts


def normalize_panel(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "membership_date",
        "index_code",
        "symbol",
        "state_transition_session",
        "state_cycle_id",
        "source_contract",
    }
    if set(frame.columns) != required:
        raise ValueError(f"{label} 字段不等于冻结集合：{sorted(frame.columns)}")
    check_columns(frame.columns, label)
    frame = frame.copy()
    frame["membership_date"] = pd.to_datetime(frame["membership_date"], errors="raise").dt.date
    frame["state_transition_session"] = pd.to_datetime(
        frame["state_transition_session"], errors="raise"
    ).dt.date
    if set(frame["index_code"].astype(str)) != {"000300"}:
        raise ValueError(f"{label} 包含非 000300 指数记录")
    if frame.duplicated(["membership_date", "symbol"]).any():
        raise ValueError(f"{label} 存在日期证券重复行")
    counts = frame.groupby("membership_date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise ValueError(f"{label} 存在非 300 只日期")
    return frame.sort_values(["membership_date", "symbol"]).reset_index(drop=True)


def main() -> int:
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    main_admission = load_json(MAIN_ADMISSION_PATH)
    extension_admission = load_json(EXTENSION_ADMISSION_PATH)
    weight_evidence = load_json(WEIGHT_EVIDENCE_PATH)

    if source_manifest.get("status") != "PASS_OFFICIAL_2015_MEMBERSHIP_EXTENSION_SOURCES_ACQUIRED":
        raise ValueError("2015 扩展来源状态未通过")
    if extension_admission.get("overall_status") != (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION_WEIGHTS_STILL_BLOCKED"
    ):
        raise ValueError("2015 扩展验收状态不正确")
    if main_admission.get("overall_status") != (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_PRIMARY_2021_WINDOW_ONLY_WEIGHTS_PENDING"
    ):
        raise ValueError("2021+ 主窗口验收状态不正确")
    assessment = weight_evidence.get("frozen_weight_provenance_gate_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("权重来源评估缺失")
    if bool(assessment.get("admission_sufficient")) or assessment.get("status") != (
        "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    ):
        raise ValueError("权重版本来源阻断没有被原样保留")

    source_hash_receipts: list[dict[str, Any]] = []
    for source in source_manifest.get("source_objects") or []:
        path = ROOT / str(source["archive_path"])
        if not path.is_file() or sha256_file(path) != str(source["sha256"]).lower():
            raise ValueError(f"2015 官方来源对象缺失或哈希漂移：{source['archive_path']}")
        source_hash_receipts.append(
            {
                "path": str(source["archive_path"]),
                "sha256": str(source["sha256"]),
                "bytes": path.stat().st_size,
            }
        )
    classifications = source_manifest.get("announcement_search_classification") or []
    classification_counts = {
        "included": sum(str(item["classification"]).startswith("INCLUDED") for item in classifications),
        "excluded_smart": sum(str(item["classification"]).startswith("EXCLUDED") for item in classifications),
        "handover": sum(str(item["classification"]).startswith("HANDOVER") for item in classifications),
    }
    if classification_counts != {"included": 5, "excluded_smart": 8, "handover": 1}:
        raise ValueError(f"官方定向公告分类计数不一致：{classification_counts}")

    receipt_checks = verify_receipts(
        extension_admission.get("output_receipts") or [],
        "2015 扩展输出",
    )
    extension = normalize_panel(EXTENSION_DAILY_PATH, "2015 扩展面板")
    main = normalize_panel(MAIN_DAILY_PATH, "2016H1 后主面板")
    combined = normalize_panel(COMBINED_DAILY_PATH, "2015-2026 合并面板")

    reconstructed = pd.concat([extension, main], ignore_index=True).sort_values(
        ["membership_date", "symbol"]
    ).reset_index(drop=True)
    if not reconstructed.equals(combined):
        raise ValueError("合并面板不等于 2015 扩展与已验收主面板的逐行拼接")
    if set(extension["membership_date"]) & set(main["membership_date"]):
        raise ValueError("2015 扩展面板与主面板日期重叠")
    if min(main["membership_date"]) != date(2016, 6, 13):
        raise ValueError("主面板首日不是 2016-06-13")

    calendar = pd.read_parquet(CALENDAR_PATH)
    check_columns(calendar.columns, "交易日历")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.date
    calendar_open_dates = sorted(
        set(
            calendar.loc[
                calendar["is_open"].astype(bool)
                & calendar["date"].between(date(2015, 1, 1), date(2026, 8, 14)),
                "date",
            ]
        )
    )
    combined_dates = sorted(combined["membership_date"].unique())
    if combined_dates != calendar_open_dates:
        raise ValueError("合并面板日期集合不等于冻结交易日历的开放会话")
    if len(combined_dates) != 2823 or len(combined) != 846900:
        raise ValueError(
            f"合并面板规模异常：开市日={len(combined_dates)}，行={len(combined)}"
        )

    primary = combined.loc[
        combined["membership_date"].between(date(2021, 1, 1), date(2026, 8, 14))
    ]
    if primary["membership_date"].nunique() != 1361 or len(primary) != 408300:
        raise ValueError("2021+ 主窗口规模与既有验收不一致")

    extension_transitions = pd.read_parquet(EXTENSION_TRANSITION_PATH)
    main_transitions = pd.read_parquet(MAIN_TRANSITION_PATH)
    check_columns(extension_transitions.columns, "2015 迁移账本")
    check_columns(main_transitions.columns, "主迁移账本")
    extension_transitions["transition_session"] = pd.to_datetime(
        extension_transitions["transition_session"], errors="raise"
    ).dt.date
    observed_transition_contract = {
        str(row.cycle_id): (row.transition_session, int(row.addition_count))
        for row in extension_transitions.itertuples(index=False)
    }
    if observed_transition_contract != EXPECTED_2015_TRANSITIONS:
        raise ValueError(f"2015 迁移账本不符合冻结契约：{observed_transition_contract}")
    if not extension_transitions["addition_count"].equals(extension_transitions["deletion_count"]):
        raise ValueError("2015 迁移账本存在调入调出数量不相等")
    if len(main_transitions) != 24 or len(extension_transitions) != 5:
        raise ValueError("主链或 2015 扩展迁移数量异常")

    extension_snapshots = pd.read_parquet(EXTENSION_SNAPSHOT_PATH)
    main_snapshots = pd.read_parquet(MAIN_SNAPSHOT_PATH)
    check_columns(extension_snapshots.columns, "2015 官方快照比较")
    check_columns(main_snapshots.columns, "月末快照比较")
    extension_snapshots["snapshot_date"] = pd.to_datetime(
        extension_snapshots["snapshot_date"], errors="raise"
    ).dt.date
    if set(extension_snapshots["snapshot_date"]) != {
        date(2015, 3, 2),
        date(2015, 7, 1),
        date(2015, 9, 1),
    }:
        raise ValueError("2015 官方快照比较日期不完整")
    if not extension_snapshots["set_difference_count"].eq(0).all():
        raise ValueError("2015 官方快照存在集合差异")
    if len(main_snapshots) != 120 or not main_snapshots["set_difference_count"].eq(0).all():
        raise ValueError("2016-2026 月末快照比较未保持 120 期全零差异")

    first_extension_set = frozenset(
        extension.loc[
            extension["membership_date"].eq(min(extension["membership_date"])),
            "symbol",
        ]
    )
    last_extension_set = frozenset(
        extension.loc[
            extension["membership_date"].eq(max(extension["membership_date"])),
            "symbol",
        ]
    )
    first_main_set = frozenset(
        main.loc[main["membership_date"].eq(min(main["membership_date"])), "symbol"]
    )
    if len(first_extension_set) != 300 or len(last_extension_set) != 300 or len(first_main_set) != 300:
        raise ValueError("交接边界集合不是 300 只")
    handover_additions = set(
        str(main_transitions.loc[main_transitions["announcement_id"].eq(3855), "additions"].iloc[0]).split("|")
    )
    handover_deletions = set(
        str(main_transitions.loc[main_transitions["announcement_id"].eq(3855), "deletions"].iloc[0]).split("|")
    )
    reconstructed_first_main = frozenset(
        (set(last_extension_set) - handover_deletions) | handover_additions
    )
    if reconstructed_first_main != first_main_set:
        raise ValueError("独立重算的 2016H1 交接集合与主面板首日不一致")

    if any(
        bool(extension_admission.get(field))
        for field in ("market_price_read", "future_return_read", "future_label_created", "model_read")
    ):
        raise ValueError("扩展验收清单违反结果盲态边界")
    candidate = extension_admission.get("candidate_boundary") or {}
    if candidate.get("return_evaluation") != "NOT_ALLOWED":
        raise ValueError("候选收益评估边界未保持 NOT_ALLOWED")
    if bool(candidate.get("paper_or_shadow_signal_allowed")) or bool(
        candidate.get("trading_authorization")
    ):
        raise ValueError("候选信号或交易授权被意外开启")

    result = {
        "audit_id": "510300_CSI300_PIT_MEMBERSHIP_WEIGHTS_SOURCE_REMEDIATION_V1_INDEPENDENT_AUDIT",
        "checked_at": now_shanghai(),
        "status": "PASS_MEMBERSHIP_ADMISSION_WEIGHT_BLOCK_PRESERVED",
        "membership": {
            "status": "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION",
            "first_open_session": min(combined["membership_date"]).isoformat(),
            "last_open_session": max(combined["membership_date"]).isoformat(),
            "open_session_count": len(combined_dates),
            "row_count": len(combined),
            "daily_constituent_count": 300,
            "duplicate_date_symbol_rows": 0,
            "official_2015_cycle_count": len(extension_transitions),
            "official_2015_change_count": int(extension_transitions["addition_count"].sum()),
            "official_2015_snapshot_count": len(extension_snapshots),
            "official_2015_snapshot_set_difference_count": int(
                extension_snapshots["set_difference_count"].sum()
            ),
            "main_monthly_snapshot_count": len(main_snapshots),
            "main_monthly_snapshot_set_difference_count": int(
                main_snapshots["set_difference_count"].sum()
            ),
            "handover_set_difference_count": len(reconstructed_first_main ^ first_main_set),
        },
        "weight": {
            "status": "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS",
            "version_or_as_of_provenance_gate_pass": False,
            "membership_anchor_use_does_not_admit_weight_values": True,
        },
        "announcement_classification_counts": classification_counts,
        "verified_source_object_count": len(source_hash_receipts),
        "verified_output_receipts": receipt_checks,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    atomic_write_json(OUTPUT_PATH, result)
    print(
        json.dumps(
            {
                **result,
                "output_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "output_sha256": sha256_file(OUTPUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
