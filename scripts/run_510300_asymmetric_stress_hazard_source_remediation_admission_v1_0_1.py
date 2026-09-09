"""核验并裁决 510300 非对称压力风险 V1.0.1 来源修复。

本程序只核验版本化来源与逐日可得性边界，不读取标签、未来收益、模型、
组合或账户状态；父级 V1.0.0 准入结果保持不可变。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import (  # noqa: E402
    OBSERVATION_CUTOFF,
    OBSERVATION_START,
    REMEDIATION_ID,
    sha256_file,
    validate_dr007_daily,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_PATH = (
    ROOT / "config" / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1.yaml"
)
PARENT_STATUS_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "510300_asymmetric_stress_hazard_v1_source_admission_v1.json"
)
PARENT_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_asymmetric_stress_hazard_v1_source_admission_v1_receipt.json"
)
CURATED_ROOT = (
    ROOT
    / "data"
    / "curated"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
)
MEMBERSHIP_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "000300_daily_pit_membership_20150101_20260814.parquet"
)
STATUS_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1.json"
)
REPORT_PATH = (
    ROOT
    / "reports"
    / "research"
    / "510300_ASYMMETRIC_STRESS_HAZARD_V1_SOURCE_REMEDIATION_V1_0_1.md"
)
RECEIPT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1_receipt.json"
)


def now_shanghai() -> str:
    return datetime.now(TIMEZONE).isoformat()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点不是对象：{relative(path)}")
    return value


def expected_path(logical_path: str) -> Path:
    path = ROOT / logical_path
    if not path.is_file():
        raise FileNotFoundError(f"冻结文件不存在：{logical_path}")
    return path


def require_hash(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"冻结文件哈希漂移：{relative(path)}；期望={expected_sha256}，实际={actual}"
        )


def verify_manifest_artifacts(manifest: dict[str, Any]) -> None:
    for artifact_name, record in manifest.get("artifacts", {}).items():
        path = expected_path(str(record["path"]))
        require_hash(path, str(record["sha256"]))
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"来源产物字节数漂移：{artifact_name}")


def verify_parent(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["immutable_parent"]
    require_hash(PARENT_STATUS_PATH, parent["source_admission_status_sha256"])
    require_hash(PARENT_RECEIPT_PATH, parent["source_admission_receipt_sha256"])
    status = load_json(PARENT_STATUS_PATH)
    receipt = load_json(PARENT_RECEIPT_PATH)
    if status.get("status") != parent["required_parent_status"]:
        raise ValueError("父级来源准入状态与 V1.0.1 合同不一致")
    if receipt.get("receipt_payload_sha256") != parent["parent_receipt_payload_sha256"]:
        raise ValueError("父级来源准入回执 payload 哈希与 V1.0.1 合同不一致")
    if status.get("feature_values_constructed") is not False:
        raise ValueError("父级来源准入错误地包含特征值")
    return status


def verify_sw(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = config["remediated_sources"]["sw_pit_industry"]
    manifest_path = expected_path(contract["acquisition_manifest"])
    require_hash(manifest_path, contract["acquisition_manifest_sha256"])
    manifest = load_json(manifest_path)
    if manifest.get("status") != contract["status"]:
        raise ValueError("申万来源 manifest 状态不一致")
    verify_manifest_artifacts(manifest)
    member_day_path = expected_path(contract["member_day_artifact"])
    require_hash(member_day_path, contract["member_day_artifact_sha256"])
    frame = pd.read_parquet(
        member_day_path,
        columns=["membership_date", "symbol", "mapping_status"],
    )
    frame["membership_date"] = pd.to_datetime(frame["membership_date"])
    if len(frame) != 846_900 or frame["membership_date"].nunique() != 2_823:
        raise ValueError("申万 member-day 产物行数或交易日数不一致")
    counts = frame.groupby("membership_date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise ValueError("申万 member-day 产物不是每日严格 300 只")
    allowed_statuses = {
        "PIT_AVAILABLE_BY_MARKET_CLOSE",
        "NO_VIEW_NO_PROVABLE_RECORD_BY_MARKET_CLOSE",
    }
    if not set(frame["mapping_status"].unique()).issubset(allowed_statuses):
        raise ValueError("申万 member-day 产物含未授权 mapping_status")
    available = frame["mapping_status"].eq("PIT_AVAILABLE_BY_MARKET_CLOSE")
    daily_available = frame.assign(_available=available).groupby("membership_date")[
        "_available"
    ].sum()
    metrics = {
        "member_day_count": int(len(frame)),
        "session_count": int(frame["membership_date"].nunique()),
        "pit_available_member_days": int(available.sum()),
        "pit_no_view_member_days": int((~available).sum()),
        "pit_coverage_ratio": float(available.mean()),
        "full_300_member_session_count": int(daily_available.eq(300).sum()),
        "no_view_session_count": int(daily_available.lt(300).sum()),
        "first_full_300_member_session": (
            daily_available.loc[daily_available.eq(300)].index.min().date().isoformat()
            if daily_available.eq(300).any()
            else None
        ),
    }
    for key, value in metrics.items():
        if manifest["coverage_metrics"].get(key) != value:
            raise ValueError(f"申万来源覆盖指标与 manifest 不一致：{key}")
    result = {
        "source_id": "sw_pit_industry",
        "status": "PASS_OFFICIAL_SW_PIT_WITH_DATE_LEVEL_NO_VIEW",
        "admitted": True,
        "whole_source_blocked": False,
        "date_level_no_view_required": True,
        "availability_clock": contract["availability_rule"],
        "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
        "metrics": {
            **metrics,
            "structural_effective_date_missing_member_days": manifest[
                "coverage_metrics"
            ]["structural_effective_date_missing_member_days"],
            "history_raw_sha256": manifest["source_metrics"]["history_raw_sha256"],
            "codebook_raw_sha256": manifest["source_metrics"]["codebook_raw_sha256"],
        },
        "limitations": [
            "ANY_SESSION_WITH_FEWER_THAN_300_PIT_AVAILABLE_INDUSTRIES_IS_NO_VIEW",
            "RECORD_UPDATED_AT_USED_AS_CONSERVATIVE_AVAILABILITY_CLOCK",
            "NO_CURRENT_CLASSIFICATION_BACKFILL",
        ],
        "feature_values_constructed": False,
    }
    return result, manifest


def verify_pboc(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = config["remediated_sources"]["pboc_7d_reverse_repo_policy_rate"]
    manifest_path = expected_path(contract["acquisition_manifest"])
    require_hash(manifest_path, contract["acquisition_manifest_sha256"])
    manifest = load_json(manifest_path)
    if manifest.get("status") != contract["status"]:
        raise ValueError("央行 7 天逆回购来源 manifest 状态不一致")
    verify_manifest_artifacts(manifest)
    anchor = manifest["pre_window_anchor"]
    if anchor["notice_date"] != str(contract["pre_window_anchor_date"]):
        raise ValueError("央行 7 天逆回购期初锚点日期不一致")
    if abs(float(anchor["rate_percent"]) - float(contract["pre_window_anchor_rate_percent"])) > 1e-12:
        raise ValueError("央行 7 天逆回购期初锚点利率不一致")
    summary = manifest["summary"]
    if summary["notice_last_date"] != OBSERVATION_CUTOFF.isoformat():
        raise ValueError("央行公告账本没有覆盖冻结观察截止日")
    if summary["published_7d_rate_date_count"] <= 0 or summary["rate_change_count"] <= 0:
        raise ValueError("央行 7 天逆回购账本没有实际披露利率或变更")
    if summary["interpolation_performed"] or summary["inferred_change_dates_used"]:
        raise ValueError("央行 7 天逆回购账本使用了禁止的插值或推断变更日")
    result = {
        "source_id": "pboc_7d_reverse_repo_policy_rate",
        "status": "PASS_OFFICIAL_PBOC_7D_RATE_WITH_PRE_WINDOW_ANCHOR",
        "admitted": True,
        "whole_source_blocked": False,
        "date_level_no_view_required": True,
        "availability_clock": contract["availability_rule"],
        "state_carry_rule": contract["state_carry_rule"],
        "missing_value_rule": "NO_VIEW_WHEN_RATE_NOT_ACTUALLY_PUBLISHED",
        "metrics": {
            **summary,
            "pre_window_anchor_date": anchor["notice_date"],
            "pre_window_anchor_rate_percent": float(anchor["rate_percent"]),
            "list_page_count_archived": manifest["list_page_count_archived"],
            "article_count_archived": manifest["article_count_archived"],
        },
        "limitations": [
            "ONLY_ACTUALLY_PUBLISHED_7D_OPERATION_RATES_ADMITTED",
            "ZERO_OPERATION_WITHOUT_PUBLISHED_RATE_REMAINS_NO_VIEW",
            "NO_INTERPOLATION_OR_INFERRED_CHANGE_DATES",
        ],
        "feature_values_constructed": False,
    }
    return result, manifest


def verify_dr007(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    contract = config["remediated_sources"]["dr007_daily"]
    probe_path = expected_path(contract["public_retention_probe_manifest"])
    require_hash(probe_path, contract["public_retention_probe_manifest_sha256"])
    probe = load_json(probe_path)
    if probe.get("status") != "BLOCKED_PUBLIC_ENDPOINT_RETENTION_DOES_NOT_COVER_FROZEN_WINDOW":
        raise ValueError("DR007 公共保留窗口证据状态漂移")
    manifest_path = CURATED_ROOT / "dr007_source_acquisition_manifest.json"
    if not manifest_path.is_file():
        return (
            {
                "source_id": "dr007_daily",
                "status": "BLOCKED_LICENSED_DR007_HISTORY_REQUIRED",
                "admitted": False,
                "whole_source_blocked": True,
                "date_level_no_view_required": False,
                "availability_clock": None,
                "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
                "metrics": {
                    "frozen_input_present": False,
                    "required_coverage_start": str(contract["required_coverage_start"]),
                    "required_coverage_cutoff": str(contract["required_coverage_cutoff"]),
                    "public_retention_observed_start": str(
                        contract["public_retention_observed_start"]
                    ),
                },
                "limitations": [
                    "PUBLIC_CHINAMONEY_ENDPOINT_RETENTION_STARTS_AFTER_FROZEN_WINDOW",
                    "LOCAL_AUTHORIZED_DR007_EXPORT_ABSENT",
                    "FORBIDDEN_SUBSTITUTES=FDR007,R007,FR007,EXCHANGE_REPO_R_007",
                ],
                "feature_values_constructed": False,
            },
            None,
        )
    manifest = load_json(manifest_path)
    if not str(manifest.get("status", "")).startswith("PASS_DR007_SOURCE_ADMITTED_"):
        raise ValueError("DR007 导入 manifest 未通过")
    artifact = manifest["curated_artifact"]
    path = expected_path(artifact["path"])
    require_hash(path, artifact["sha256"])
    frame = pd.read_parquet(path, columns=["date", "dr007", "series_code", "availability_rule"])
    if not frame["series_code"].eq("DR007").all():
        raise ValueError("DR007 导入产物混入其他序列")
    if not frame["availability_rule"].eq(
        contract["required_availability_rule"]
    ).all():
        raise ValueError("DR007 导入产物可得时钟不符合合同")
    membership = pd.read_parquet(MEMBERSHIP_PATH, columns=["membership_date"])
    metrics = validate_dr007_daily(
        frame[["date", "dr007"]],
        market_dates=membership["membership_date"].drop_duplicates(),
        source_identity=manifest["metrics"]["source_identity"],
    )
    return (
        {
            "source_id": "dr007_daily",
            "status": manifest["status"],
            "admitted": True,
            "whole_source_blocked": False,
            "date_level_no_view_required": metrics["missing_market_session_count"] > 0,
            "availability_clock": contract["required_availability_rule"],
            "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
            "metrics": metrics,
            "limitations": (
                ["MISSING_MARKET_SESSIONS_REMAIN_DATE_LEVEL_NO_VIEW"]
                if metrics["missing_market_session_count"] > 0
                else []
            ),
            "feature_values_constructed": False,
        },
        manifest,
    )


def updated_channels(
    parent: dict[str, Any], dr007_admitted: bool
) -> dict[str, dict[str, Any]]:
    channels = copy.deepcopy(parent["channel_results"])
    for channel_id in ("F3", "T2", "T3"):
        channel = channels[channel_id]
        channel["admitted"] = True
        channel["status"] = "PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW"
        channel["blockers"] = []
        channel["date_level_no_view_required"] = True
        channel["feature_values_constructed"] = False
        channel["required_sources"] = [
            (
                "sw_pit_industry"
                if source == "pit_industry_intervals"
                else source
            )
            for source in channel["required_sources"]
        ]
    m2 = channels["M2"]
    m2["required_sources"] = [
        (
            "pboc_7d_reverse_repo_policy_rate"
            if source == "reverse_repo_policy_rate_7d"
            else source
        )
        for source in m2["required_sources"]
    ]
    m2["feature_values_constructed"] = False
    m2["blockers"] = (
        []
        if dr007_admitted
        else [
            {
                "source_id": "dr007_daily",
                "status": "BLOCKED_LICENSED_DR007_HISTORY_REQUIRED",
            }
        ]
    )
    m2["admitted"] = dr007_admitted
    m2["status"] = (
        "PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW"
        if dr007_admitted
        else "BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED"
    )
    m2["date_level_no_view_required"] = dr007_admitted
    return channels


def build_report(status: dict[str, Any]) -> str:
    sw = status["source_results"]["sw_pit_industry"]
    pboc = status["source_results"]["pboc_7d_reverse_repo_policy_rate"]
    dr007 = status["source_results"]["dr007_daily"]
    lines = [
        "# 510300_ASYMMETRIC_STRESS_HAZARD_V1 来源修复 V1.0.1",
        "",
        "## 裁决",
        "",
        f"- 状态：`{status['status']}`。",
        f"- 当前整源阻断通道：`{', '.join(status['blocked_channels']) if status['blocked_channels'] else '无'}`。",
        "- 父级 V1.0.0 结果保持不可变；没有重跑 BAD10，没有修改标签、窗口或阈值。",
        "- 未构造 M/F/T 特征，未读取未来收益或组合指标，未训练模型，未生成仓位、Paper/Shadow、订单或券商动作。",
        "",
        "## 已补齐并准入",
        "",
        (
            "- 申万官方 PIT 行业："
            f"{sw['metrics']['member_day_count']} 个 member-day，其中 "
            f"{sw['metrics']['pit_available_member_days']} 个在当日收盘前可证明可得，"
            f"{sw['metrics']['pit_no_view_member_days']} 个保持 `NO_VIEW`；"
            f"完整 300 只行业映射的交易日为 {sw['metrics']['full_300_member_session_count']} 个。"
        ),
        (
            "- 央行 7 天逆回购：归档 "
            f"{pboc['metrics']['article_count_archived']} 篇必要候选公告，正式账本含 "
            f"{pboc['metrics']['notice_count']} 条（含一个观察期前锚点）；"
            f"实际披露利率日 {pboc['metrics']['published_7d_rate_date_count']} 个，"
            f"实际变化点 {pboc['metrics']['rate_change_count']} 个。"
        ),
        (
            "- 零操作公告未披露利率的数量为 "
            f"{pboc['metrics']['zero_operation_without_published_rate_count']}；"
            "均未猜值、未插值、未倒推变更日。"
        ),
        "- `F3`、`T2`、`T3` 的整源阻断解除；受行业可得性或成分回报缺口影响的日期仍逐日 `NO_VIEW`。",
        "",
        "## 唯一剩余缺口",
        "",
        f"- DR007：`{dr007['status']}`。",
        (
            "- ChinaMoney 公共端点本次证明的起始日为 "
            f"{dr007['metrics'].get('public_retention_observed_start', '不适用')}，"
            f"不能覆盖冻结区间 {OBSERVATION_START.isoformat()} 至 {OBSERVATION_CUTOFF.isoformat()}。"
        ),
        "- 项目内只有 FDR007；它与 R007、FR007、交易所 R-007 一样不得替代 DR007。",
        "- 需要一份获授权的 DR007 日度加权平均利率导出，以及对应来源证明 JSON。导入器只接受 `DR007`、百分数口径，并统一采用“利率日后的下一交易日开盘可得”这一保守时钟。",
        "",
        "## 允许的后续动作",
        "",
        f"`{status['next_allowed_action']}`",
        "",
        "在 DR007 通过前，来源仍未全部准入，禁止特征构造、G2、收益评估和任何交易动作。",
        "",
        "## 权限边界",
        "",
        "- `RESEARCH_STATE=DISCOVERY_ONLY`",
        "- `MODEL_POSITION_TARGET=UNSET`",
        "- `ORDER_AUTHORIZATION=NOT_AUTHORIZED`",
        "- `POSITION_IMPACT=0`",
        "- `RETURN_EVALUATION=NOT_ALLOWED`",
    ]
    return "\n".join(lines) + "\n"


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def payload_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    completed_at = now_shanghai()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["remediation_id"] != REMEDIATION_ID:
        raise ValueError("V1.0.1 来源合同 remediation_id 不一致")
    parent = verify_parent(config)
    sw_result, sw_manifest = verify_sw(config)
    pboc_result, pboc_manifest = verify_pboc(config)
    dr007_result, dr007_manifest = verify_dr007(config)
    dr007_admitted = bool(dr007_result["admitted"])
    channels = updated_channels(parent, dr007_admitted)
    blocked_channels = sorted(
        channel_id
        for channel_id, result in channels.items()
        if not result["admitted"]
    )
    all_admitted = len(blocked_channels) == 0
    status = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "remediation_id": REMEDIATION_ID,
        "stage_id": "POINT_IN_TIME_M_F_T_SOURCE_REMEDIATION_V1_0_1",
        "version": "1.0.1",
        "completed_at": completed_at,
        "status": (
            "PASS_SOURCE_REMEDIATION_ALL_REQUIRED_SOURCES_ADMITTED"
            if all_admitted
            else "NO_VIEW_SOURCE_REMEDIATION_PARTIAL_DR007_BLOCKED"
        ),
        "research_disposition": "SOURCE_READY" if all_admitted else "NO_VIEW",
        "all_required_sources_admitted": all_admitted,
        "blocked_channels": blocked_channels,
        "source_results": {
            "sw_pit_industry": sw_result,
            "pboc_7d_reverse_repo_policy_rate": pboc_result,
            "dr007_daily": dr007_result,
        },
        "channel_results": channels,
        "parent_source_admission": {
            "status": parent["status"],
            "completed_at": parent["completed_at"],
            "receipt_payload_sha256": config["immutable_parent"][
                "parent_receipt_payload_sha256"
            ],
            "mutated": False,
        },
        "feature_construction_allowed": all_admitted,
        "g2_allowed": all_admitted,
        "feature_values_constructed": False,
        "label_artifacts_read": False,
        "bad10_census_rerun": False,
        "label_window_threshold_changed": False,
        "model_trained": False,
        "probability_threshold_selected": False,
        "return_evaluation": "NOT_PERFORMED" if all_admitted else "NOT_ALLOWED",
        "portfolio_metrics_read": False,
        "position_generated": False,
        "paper_shadow_generated": False,
        "broker_action_performed": False,
        "order_generated": False,
        "live_trading_authorized": False,
        "position_impact": 0,
        "rescue_allowed": False,
        "substitute_or_proxy_used": False,
        "next_allowed_action": (
            "CONSTRUCT_FROZEN_M_F_T_FEATURES_UNDER_G2"
            if all_admitted
            else "IMPORT_LICENSED_DR007_WEIGHTED_AVERAGE_DAILY_EXPORT_WITH_PROVENANCE"
        ),
    }
    atomic_write_json(STATUS_PATH, status)
    atomic_write_bytes(REPORT_PATH, build_report(status).encode("utf-8"))

    receipt_without_hash: dict[str, Any] = {
        "program_id": status["program_id"],
        "remediation_id": REMEDIATION_ID,
        "stage_id": status["stage_id"],
        "version": status["version"],
        "status": "IMMUTABLE_SOURCE_REMEDIATION_RECEIPT_COMPLETE",
        "source_remediation_status": status["status"],
        "completed_at": completed_at,
        "git_head": git_head(),
        "parent_receipt_payload_sha256": config["immutable_parent"][
            "parent_receipt_payload_sha256"
        ],
        "contract": {
            "path": relative(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
            "bytes": CONFIG_PATH.stat().st_size,
        },
        "source_manifests": {
            "sw": {
                "path": config["remediated_sources"]["sw_pit_industry"][
                    "acquisition_manifest"
                ],
                "sha256": config["remediated_sources"]["sw_pit_industry"][
                    "acquisition_manifest_sha256"
                ],
            },
            "pboc": {
                "path": config["remediated_sources"][
                    "pboc_7d_reverse_repo_policy_rate"
                ]["acquisition_manifest"],
                "sha256": config["remediated_sources"][
                    "pboc_7d_reverse_repo_policy_rate"
                ]["acquisition_manifest_sha256"],
            },
            "dr007": (
                {
                    "path": relative(
                        CURATED_ROOT / "dr007_source_acquisition_manifest.json"
                    ),
                    "sha256": sha256_file(
                        CURATED_ROOT / "dr007_source_acquisition_manifest.json"
                    ),
                }
                if dr007_manifest is not None
                else {
                    "status": "BLOCKED_LICENSED_DR007_HISTORY_REQUIRED",
                    "public_probe_path": config["remediated_sources"]["dr007_daily"][
                        "public_retention_probe_manifest"
                    ],
                    "public_probe_sha256": config["remediated_sources"][
                        "dr007_daily"
                    ]["public_retention_probe_manifest_sha256"],
                }
            ),
        },
        "outputs": {
            "status": {
                "path": relative(STATUS_PATH),
                "sha256": sha256_file(STATUS_PATH),
                "bytes": STATUS_PATH.stat().st_size,
            },
            "human_report": {
                "path": relative(REPORT_PATH),
                "sha256": sha256_file(REPORT_PATH),
                "bytes": REPORT_PATH.stat().st_size,
            },
        },
        "all_required_sources_admitted": all_admitted,
        "blocked_channels": blocked_channels,
        "feature_construction_allowed": all_admitted,
        "g2_allowed": all_admitted,
        "feature_values_constructed": False,
        "label_artifacts_read": False,
        "bad10_census_rerun": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "next_allowed_action": status["next_allowed_action"],
    }
    receipt = {
        **receipt_without_hash,
        "receipt_payload_sha256": payload_sha256(receipt_without_hash),
    }
    atomic_write_json(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "status": status["status"],
                "blocked_channels": blocked_channels,
                "all_required_sources_admitted": all_admitted,
                "next_allowed_action": status["next_allowed_action"],
                "receipt_payload_sha256": receipt["receipt_payload_sha256"],
                "outputs": {
                    "status": relative(STATUS_PATH),
                    "report": relative(REPORT_PATH),
                    "receipt": relative(RECEIPT_PATH),
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
