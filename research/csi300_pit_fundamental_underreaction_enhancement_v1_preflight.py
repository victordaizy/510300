from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 预检只允许读取这些非收益输入。价格文件即使存在也不在此集合中。
PRE_RETURN_READ_PATHS = frozenset(
    {
        "data/raw/cninfo/a_share_hs_periodic_report_events_v2_1.parquet",
        "reports/data_quality/A_SHARE_HS_CNINFO_PERIODIC_REPORT_EVENT_ARCHIVE_V2_1.json",
        "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet",
        "reports/data_quality/510300_csi300_pit_membership_2015_extension_v1.json",
        "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/industry_l1_intervals.parquet",
        "reports/data_quality/a_share_hs_concentrated_low_risk_trend_v1_3_combined_data_audit.json",
        "reports/data_quality/csi300_financial_vintage_audit_20260819.json",
        "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/official_financial_facts.parquet",
        "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/event_fact_dependency_ledger.parquet",
        "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/receipt.json",
        "reports/research/510300_stress_transmission_and_exhaustion_atlas_v1.json",
        "reports/research/510300_regime_transition_router_v1.json",
        "reports/research/510300_liquidity_stress_long_cycle_replication_v1.json",
    }
)

FORBIDDEN_PRE_RETURN_MARKERS = (
    "training_panel.parquet",
    "holdout_panel.parquet",
    "training_H00300.parquet",
    "holdout_H00300.parquet",
    "510300_daily.parquet",
    "forward_return",
    "stock_outcomes",
)


def project_path(relative_path: str | Path) -> Path:
    """把协议中的 POSIX 相对路径解析为本项目下的 Windows 兼容路径。"""

    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def status_gate(
    gate_id: str,
    description: str,
    passed: bool,
    observed: Any,
    required: Any,
    **details: Any,
) -> dict[str, Any]:
    gate = {
        "gate_id": gate_id,
        "description": description,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }
    if details:
        gate["details"] = details
    return gate


def first_open_strictly_after(
    event_dates: pd.Series,
    open_dates: pd.Series | pd.Index | np.ndarray,
) -> pd.Series:
    """返回每个公告日之后的第一个共同开市日；没有后续日期时返回 NaT。"""

    normalized_events = pd.to_datetime(event_dates, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    normalized_open = np.sort(
        pd.to_datetime(pd.Index(open_dates), errors="coerce")
        .dropna()
        .unique()
        .to_numpy(dtype="datetime64[ns]")
    )
    output = np.full(
        len(normalized_events), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    valid = ~pd.isna(normalized_events)
    if normalized_open.size and valid.any():
        positions = np.searchsorted(normalized_open, normalized_events[valid], side="right")
        within = positions < normalized_open.size
        valid_indices = np.flatnonzero(valid)
        output[valid_indices[within]] = normalized_open[positions[within]]
    return pd.Series(pd.to_datetime(output), index=event_dates.index)


def attach_point_in_time_industry(
    target_events: pd.DataFrame,
    industry_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """按成份判定日连接当时有效且已经可用的唯一行业区间。"""

    required = {
        "ts_code",
        "industry_l1",
        "industry_l1_code",
        "valid_from",
        "valid_to",
        "available_at",
    }
    missing = sorted(required.difference(industry_intervals.columns))
    if missing:
        raise ValueError(f"行业区间缺少字段：{missing}")

    intervals = industry_intervals.loc[:, sorted(required)].copy()
    for column in ("valid_from", "valid_to", "available_at"):
        intervals[column] = pd.to_datetime(intervals[column], errors="coerce")
    candidates = target_events.merge(intervals, on="ts_code", how="left")
    event_day = pd.to_datetime(candidates["membership_date"], errors="coerce")
    valid_to = candidates["valid_to"]
    available_day = pd.to_datetime(candidates["available_at"], errors="coerce").dt.normalize()
    valid = (
        (candidates["valid_from"] <= event_day)
        & (valid_to.isna() | (valid_to >= event_day))
        & (available_day <= event_day)
    )
    matched = candidates.loc[valid].copy()
    matched = matched.sort_values(
        ["announcement_id", "valid_from", "available_at"],
        ascending=[True, False, False],
    ).drop_duplicates("announcement_id", keep="first")

    industry_columns = [
        "announcement_id",
        "industry_l1",
        "industry_l1_code",
        "valid_from",
        "valid_to",
        "available_at",
    ]
    return target_events.merge(matched[industry_columns], on="announcement_id", how="left")


def build_target_event_inventory(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    *,
    event_start: str,
    event_end: str,
    allowed_period_types: Iterable[str],
    excluded_industry_codes: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_columns = {
        "announcement_id",
        "ts_code",
        "report_period",
        "period_type",
        "event_publication_date",
        "official_timestamp_at",
        "official_pdf_url",
    }
    membership_columns = {"membership_date", "symbol"}
    missing_events = sorted(event_columns.difference(events.columns))
    missing_membership = sorted(membership_columns.difference(membership.columns))
    if missing_events:
        raise ValueError(f"官方公告事件档案缺少字段：{missing_events}")
    if missing_membership:
        raise ValueError(f"沪深300成份档案缺少字段：{missing_membership}")

    selected = events.loc[:, sorted(event_columns)].copy()
    selected["announcement_id"] = selected["announcement_id"].astype("string")
    selected["event_publication_date"] = pd.to_datetime(
        selected["event_publication_date"], errors="coerce"
    ).dt.normalize()
    selected["report_period"] = pd.to_datetime(selected["report_period"], errors="coerce")
    selected = selected.loc[
        selected["event_publication_date"].between(
            pd.Timestamp(event_start), pd.Timestamp(event_end), inclusive="both"
        )
        & selected["period_type"].astype(str).isin(set(map(str, allowed_period_types)))
    ].copy()

    members = membership.loc[:, ["membership_date", "symbol"]].copy()
    members["membership_date"] = pd.to_datetime(
        members["membership_date"], errors="coerce"
    ).dt.normalize()
    members["symbol"] = members["symbol"].astype("string")
    selected["membership_date"] = first_open_strictly_after(
        selected["event_publication_date"], members["membership_date"]
    )
    target = selected.merge(
        members,
        left_on=["membership_date", "ts_code"],
        right_on=["membership_date", "symbol"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="symbol")
    target = attach_point_in_time_industry(target, industry_intervals)

    excluded = set(map(str, excluded_industry_codes))
    target["industry_interval_matched"] = target["industry_l1_code"].notna()
    target["pre_registered_industry_excluded"] = (
        target["industry_l1_code"].astype("string").isin(excluded)
    )
    target["included_in_phase1_fact_target"] = (
        target["industry_interval_matched"] & ~target["pre_registered_industry_excluded"]
    )
    target["publication_year"] = target["event_publication_date"].dt.year.astype("Int64")
    target["inventory_status"] = np.select(
        [
            ~target["industry_interval_matched"],
            target["pre_registered_industry_excluded"],
        ],
        ["NO_VIEW_PIT_INDUSTRY_MISSING", "PRE_REGISTERED_FINANCIAL_INDUSTRY_EXCLUSION"],
        default="TARGET_OFFICIAL_FACT_ACQUISITION",
    )
    target = target.sort_values(
        ["event_publication_date", "ts_code", "report_period", "announcement_id"]
    ).reset_index(drop=True)

    industry_coverage = (
        float(target["industry_interval_matched"].mean()) if len(target) else 0.0
    )
    metrics = {
        "candidate_event_count_before_membership": int(len(selected)),
        "pit_csi300_member_event_count": int(len(target)),
        "pit_industry_matched_event_count": int(target["industry_interval_matched"].sum()),
        "pit_industry_coverage_ratio": industry_coverage,
        "pre_registered_financial_industry_exclusion_count": int(
            target["pre_registered_industry_excluded"].sum()
        ),
        "official_fact_target_event_count": int(
            target["included_in_phase1_fact_target"].sum()
        ),
        "target_symbol_count": int(
            target.loc[target["included_in_phase1_fact_target"], "ts_code"].nunique()
        ),
        "target_event_count_by_publication_year": {
            str(int(year)): int(count)
            for year, count in target.loc[
                target["included_in_phase1_fact_target"]
            ]["publication_year"]
            .value_counts()
            .sort_index()
            .items()
        },
        "target_event_count_by_period_type": {
            str(period): int(count)
            for period, count in target.loc[
                target["included_in_phase1_fact_target"]
            ]["period_type"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return target, metrics


def evaluate_legacy_gate(components: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    observed_components: list[dict[str, Any]] = []
    all_match = True
    for component in components:
        evidence_path = project_path(str(component["evidence"]))
        exists = evidence_path.exists()
        payload = load_json(evidence_path) if exists else {}
        observed_status = payload.get("status")
        required_status = component["required_status"]
        matches = bool(exists and observed_status == required_status)
        all_match &= matches
        observed_components.append(
            {
                "study_id": component["study_id"],
                "path": str(component["evidence"]),
                "exists": exists,
                "observed_status": observed_status,
                "required_status": required_status,
                "matches": matches,
                "sha256": sha256_file(evidence_path) if exists else None,
            }
        )
    return status_gate(
        "G1_LEGACY_STRESS_PROGRAM_TERMINAL_STATES_MATCH",
        "三项旧压力研究保持冻结终局，历史择时开发关闭",
        all_match,
        [item["observed_status"] for item in observed_components],
        [item["required_status"] for item in observed_components],
        components=observed_components,
    )


def evaluate_fact_gate(
    config: Mapping[str, Any],
    target_inventory: pd.DataFrame,
) -> dict[str, Any]:
    source = nested_get(config, ("sources", "official_fact_archive"), {})
    fact_contract = config["official_fact_contract"]
    admission = config["data_admission"]
    paths = {
        "facts": project_path(source["facts"]),
        "dependency_ledger": project_path(source["dependency_ledger"]),
        "receipt": project_path(source["receipt"]),
    }
    existence = {name: path.exists() for name, path in paths.items()}
    if not all(existence.values()):
        return status_gate(
            "G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS",
            "每个入模财务数值均须在首份官方原始完整PDF中逐项定位",
            False,
            "MISSING_OFFICIAL_FACT_ARTIFACTS",
            source["required_receipt_status"],
            artifact_exists=existence,
            missing_paths=[
                str(paths[name].relative_to(PROJECT_ROOT)).replace("\\", "/")
                for name, exists in existence.items()
                if not exists
            ],
        )

    receipt = load_json(paths["receipt"])
    facts = pd.read_parquet(paths["facts"])
    dependency = pd.read_parquet(paths["dependency_ledger"])
    required_fact_columns = set(fact_contract["required_identifier_columns"])
    required_dependency_columns = {
        "announcement_id",
        "target_event_ready",
        "required_document_count",
        "complete_document_count",
        "missing_metric_count",
        "missing_document_count",
        "dependency_status",
    }
    missing_fact_columns = sorted(required_fact_columns.difference(facts.columns))
    missing_dependency_columns = sorted(
        required_dependency_columns.difference(dependency.columns)
    )
    if missing_fact_columns or missing_dependency_columns:
        return status_gate(
            "G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS",
            "官方事实与依赖账本结构完整",
            False,
            "SCHEMA_MISMATCH",
            "REQUIRED_COLUMNS_PRESENT",
            missing_fact_columns=missing_fact_columns,
            missing_dependency_columns=missing_dependency_columns,
        )

    facts = facts.copy()
    facts["announcement_id"] = facts["announcement_id"].astype("string")
    dependency = dependency.copy()
    dependency["announcement_id"] = dependency["announcement_id"].astype("string")
    target = target_inventory.loc[
        target_inventory["included_in_phase1_fact_target"]
    ].copy()
    target["announcement_id"] = target["announcement_id"].astype("string")
    joined = target.merge(
        dependency[
            [
                "announcement_id",
                "target_event_ready",
                "missing_metric_count",
                "missing_document_count",
            ]
        ],
        on="announcement_id",
        how="left",
        validate="one_to_one",
    )
    joined["target_event_ready"] = joined["target_event_ready"].fillna(False).astype(bool)

    overall_ratio = float(joined["target_event_ready"].mean()) if len(joined) else 0.0
    by_year = joined.groupby("publication_year", dropna=False)["target_event_ready"].mean()
    year_min = float(by_year.min()) if len(by_year) else 0.0
    industry_groups = joined.groupby("industry_l1_code", dropna=False)["target_event_ready"].agg(
        ["size", "mean"]
    )
    eligible_industry_groups = industry_groups.loc[
        industry_groups["size"] >= 50, "mean"
    ]
    industry_min = (
        float(eligible_industry_groups.min()) if len(eligible_industry_groups) else 0.0
    )

    required_metrics = set(map(str, fact_contract["required_metrics"]))
    observed_metrics = set(facts["metric_id"].dropna().astype(str).unique())
    missing_metrics = sorted(required_metrics.difference(observed_metrics))
    key_duplicates = int(facts.duplicated(["announcement_id", "metric_id"]).sum())
    admitted_status = fact_contract["admitted_verification_status"]
    bad_verification_count = int(
        (facts["verification_status"].astype(str) != admitted_status).sum()
    )
    invalid_host_count = int(
        facts["official_pdf_url"]
        .astype(str)
        .map(lambda value: urlparse(value).hostname != "static.cninfo.com.cn")
        .sum()
    )
    missing_evidence_count = int(
        facts[["official_pdf_sha256", "source_page", "source_locator", "source_label"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    receipt_status_matches = receipt.get("status") == source["required_receipt_status"]
    receipt_governance_clean = all(
        receipt.get(key) is False
        for key in (
            "market_price_read",
            "future_return_read",
            "future_label_created",
            "portfolio_return_calculated",
        )
    )
    passed = all(
        [
            receipt_status_matches,
            receipt_governance_clean,
            not missing_metrics,
            key_duplicates == 0,
            bad_verification_count == 0,
            invalid_host_count == 0,
            missing_evidence_count == 0,
            overall_ratio >= admission["minimum_official_fact_complete_event_ratio"],
            year_min >= admission["minimum_complete_event_ratio_each_publication_year"],
            industry_min
            >= admission[
                "minimum_complete_event_ratio_each_industry_with_at_least_50_targets"
            ],
        ]
    )
    return status_gate(
        "G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS",
        "官方PDF逐项验证、证据定位和事件依赖覆盖率全部通过",
        passed,
        receipt.get("status"),
        source["required_receipt_status"],
        receipt_status_matches=receipt_status_matches,
        receipt_governance_clean=receipt_governance_clean,
        fact_row_count=int(len(facts)),
        dependency_row_count=int(len(dependency)),
        key_duplicate_count=key_duplicates,
        missing_required_metrics=missing_metrics,
        bad_verification_count=bad_verification_count,
        invalid_official_pdf_host_count=invalid_host_count,
        missing_source_evidence_count=missing_evidence_count,
        overall_complete_event_ratio=overall_ratio,
        minimum_publication_year_complete_event_ratio=year_min,
        minimum_eligible_industry_complete_event_ratio=industry_min,
        facts_sha256=sha256_file(paths["facts"]),
        dependency_ledger_sha256=sha256_file(paths["dependency_ledger"]),
        receipt_sha256=sha256_file(paths["receipt"]),
    )


def run_preflight(config: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    sources = config["sources"]
    legacy = config["legacy_program_closure"]
    admission = config["data_admission"]
    universe = config["universe"]
    periods = config["periods"]

    events_path = project_path(sources["official_periodic_events"]["path"])
    event_receipt_path = project_path(sources["official_periodic_events"]["receipt"])
    membership_path = project_path(sources["csi300_membership"]["path"])
    membership_receipt_path = project_path(sources["csi300_membership"]["receipt"])
    industry_path = project_path(sources["industry_intervals"]["path"])
    industry_receipt_path = project_path(sources["industry_intervals"]["receipt"])
    vintage_audit_path = project_path(sources["vendor_financial_snapshot"]["vintage_audit"])

    required_base_paths = [
        events_path,
        event_receipt_path,
        membership_path,
        membership_receipt_path,
        industry_path,
        industry_receipt_path,
        vintage_audit_path,
    ]
    missing_base_paths = [str(path) for path in required_base_paths if not path.exists()]
    if missing_base_paths:
        raise FileNotFoundError(f"预检基础输入缺失：{missing_base_paths}")

    event_receipt = load_json(event_receipt_path)
    membership_receipt = load_json(membership_receipt_path)
    industry_receipt = load_json(industry_receipt_path)
    vintage_audit = load_json(vintage_audit_path)

    events = pd.read_parquet(events_path)
    membership = pd.read_parquet(membership_path)
    industry = pd.read_parquet(industry_path)
    inventory, inventory_metrics = build_target_event_inventory(
        events,
        membership,
        industry,
        event_start=periods["feature_event_start"],
        event_end=periods["feature_event_end"],
        allowed_period_types=universe["period_types"],
        excluded_industry_codes=universe["excluded_industry_l1_codes"],
    )

    gates: list[dict[str, Any]] = [evaluate_legacy_gate(legacy["components"])]

    event_status = event_receipt.get("status")
    event_governance_clean = all(
        nested_get(event_receipt, ("governance", key)) is False
        for key in ("market_price_read", "future_return_read", "portfolio_return_calculated")
    )
    gates.append(
        status_gate(
            "G2_OFFICIAL_EVENT_CHRONOLOGY_PASS",
            "官方定期报告事件时序档案通过且没有读取收益",
            event_status == sources["official_periodic_events"]["required_status"]
            and event_governance_clean,
            event_status,
            sources["official_periodic_events"]["required_status"],
            governance_clean=event_governance_clean,
            event_count=nested_get(event_receipt, ("metrics", "event_count")),
            event_symbol_count=nested_get(event_receipt, ("metrics", "event_symbol_count")),
            input_sha256=sha256_file(events_path),
        )
    )

    membership_status = nested_get(
        membership_receipt, ("membership_admission", "status")
    )
    membership_structure_clean = (
        nested_get(
            membership_receipt,
            ("membership_admission", "active_constituent_count_each_open_session"),
        )
        == 300
        and nested_get(
            membership_receipt, ("membership_admission", "duplicate_date_symbol_rows")
        )
        == 0
    )
    gates.append(
        status_gate(
            "G3_OFFICIAL_PIT_CSI300_MEMBERSHIP_PASS",
            "官方调样回放在研究窗口逐日保持300只且无重复",
            membership_status == sources["csi300_membership"]["required_status"]
            and membership_structure_clean,
            membership_status,
            sources["csi300_membership"]["required_status"],
            structural_checks_pass=membership_structure_clean,
            historical_index_weights_required=False,
            input_sha256=sha256_file(membership_path),
        )
    )

    industry_status = industry_receipt.get("status")
    industry_expected_hash = nested_get(
        industry_receipt, ("checks", "industry_intervals", "sha256")
    )
    industry_actual_hash = sha256_file(industry_path)
    industry_hash_matches = bool(
        industry_expected_hash and industry_expected_hash == industry_actual_hash
    )
    industry_coverage = inventory_metrics["pit_industry_coverage_ratio"]
    gates.append(
        status_gate(
            "G4_PIT_INDUSTRY_INTERVAL_COVERAGE_PASS",
            "公告时点行业区间哈希匹配且目标事件覆盖率达到固定门槛",
            industry_status == sources["industry_intervals"]["required_receipt_status"]
            and industry_hash_matches
            and industry_coverage
            >= admission["minimum_industry_interval_coverage_ratio"],
            {
                "receipt_status": industry_status,
                "coverage_ratio": industry_coverage,
                "hash_matches": industry_hash_matches,
            },
            {
                "receipt_status": sources["industry_intervals"]["required_receipt_status"],
                "minimum_coverage_ratio": admission[
                    "minimum_industry_interval_coverage_ratio"
                ],
                "hash_matches": True,
            },
            input_sha256=industry_actual_hash,
        )
    )

    vintage_status = vintage_audit.get("status")
    vendor_quarantined = (
        vintage_status == sources["vendor_financial_snapshot"]["required_audit_status"]
        and nested_get(vintage_audit, ("governance", "may_call_strict_point_in_time"))
        is False
        and nested_get(vintage_audit, ("governance", "return_or_model_calculation_performed"))
        is False
    )
    gates.append(
        status_gate(
            "G5_VENDOR_REVISION_RISK_IS_QUARANTINED",
            "2026统一回取的供应商财务值只作PDF定位候选，不冒充历史事实",
            vendor_quarantined,
            vintage_status,
            sources["vendor_financial_snapshot"]["required_audit_status"],
            may_call_strict_point_in_time=nested_get(
                vintage_audit, ("governance", "may_call_strict_point_in_time")
            ),
            vintage_verified_count=nested_get(
                vintage_audit, ("audit_results", "vintage_verified_count")
            ),
            revision_possible_count=nested_get(
                vintage_audit, ("audit_results", "revision_possible_count")
            ),
            audit_sha256=sha256_file(vintage_audit_path),
        )
    )

    gates.append(evaluate_fact_gate(config, inventory))

    pre_return_paths_clean = not any(
        marker.lower() in path.lower()
        for path in PRE_RETURN_READ_PATHS
        for marker in FORBIDDEN_PRE_RETURN_MARKERS
    )
    gates.append(
        status_gate(
            "G7_NO_MARKET_PRICE_OR_FUTURE_RETURN_READ_BEFORE_G1_TO_G6_PASS",
            "本预检只读取公告、成份、行业、事实准入与终局状态证据",
            pre_return_paths_clean,
            "NO_MARKET_PRICE_OR_FUTURE_RETURN_READ",
            "NO_MARKET_PRICE_OR_FUTURE_RETURN_READ",
            allowed_read_paths=sorted(PRE_RETURN_READ_PATHS),
        )
    )

    first_six_pass = all(gate["passed"] for gate in gates[:6])
    all_pass = first_six_pass and gates[6]["passed"]
    status = admission["pass_status"] if all_pass else admission["fail_status"]
    report = {
        "study_id": config["study"]["study_id"],
        "protocol_version": config["study"]["version"],
        "status": status,
        "phase": "DATA_ADMISSION_PREFLIGHT",
        "inventory_metrics": inventory_metrics,
        "gates": gates,
        "passed_gate_count": int(sum(gate["passed"] for gate in gates)),
        "total_gate_count": len(gates),
        "blocking_gate_ids": [gate["gate_id"] for gate in gates if not gate["passed"]],
        "return_evaluation": "ALLOWED_FOR_FROZEN_PHASE_1_ONLY"
        if all_pass
        else admission["fail_return_evaluation"],
        "next_allowed_step": (
            "RUN_FROZEN_PHASE_1_60D_120D_RELATIVE_RETURN_MECHANISM_EVALUATION"
            if all_pass
            else "BUILD_OR_COMPLETE_OFFICIAL_ORIGINAL_PDF_VERIFIED_FACT_ARCHIVE_THEN_RERUN_PREFLIGHT"
        ),
        "governance": {
            "market_price_read": False,
            "future_return_read": False,
            "future_label_created": False,
            "signal_score_calculated": False,
            "portfolio_return_calculated": False,
            "strategy_sharpe_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
            "trading_authorization": False,
        },
    }
    return report, inventory
