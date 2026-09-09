"""在冻结后对四态来源做一次有界字段与权限探测。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    file_evidence,
    normalize_project_relative_path,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _normalize_probe_frame(
    frame: pd.DataFrame | None,
    required_fields: list[str],
    source_name: str,
    *,
    require_nonempty: bool = True,
) -> pd.DataFrame:
    if frame is None:
        raise EvidenceContractError(f"{source_name} 返回 None")
    if require_nonempty and frame.empty:
        raise EvidenceContractError(f"{source_name} 返回空表")
    missing = sorted(set(required_fields).difference(frame.columns))
    if missing:
        raise EvidenceContractError(f"{source_name} 缺少冻结字段：{missing}")
    return frame.loc[:, required_fields].copy()


def validate_endpoint_frames(
    config: Mapping[str, Any], frames: Mapping[str, pd.DataFrame | None]
) -> dict[str, pd.DataFrame]:
    provider = config["provider_contract"]
    normalized: dict[str, pd.DataFrame] = {}
    for source_name in ("daily", "suspend_d", "dividend", "adj_factor"):
        fields = [str(value) for value in provider[source_name]["required_fields"]]
        normalized[source_name] = _normalize_probe_frame(
            frames.get(source_name), fields, source_name
        )
    daily = normalized["daily"]
    numeric = daily[["open", "high", "low", "close", "pre_close", "pct_chg"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric[["open", "high", "low", "close", "pre_close"]] <= 0).any().any():
        raise EvidenceContractError("daily 探测价格或涨跌幅字段非法")
    expected_pct = (numeric["close"] / numeric["pre_close"] - 1.0) * 100.0
    tolerance = float(
        config["corporate_action_reconciliation"]["comparison"][
            "pct_chg_identity_absolute_tolerance_percentage_points"
        ]
    )
    if (expected_pct - numeric["pct_chg"]).abs().gt(tolerance).any():
        raise EvidenceContractError("daily 的 close/pre_close 与 pct_chg 恒等式失败")
    suspension_values = set(normalized["suspend_d"]["suspend_type"].dropna().astype(str))
    if not suspension_values.issubset({"S", "R"}):
        raise EvidenceContractError(f"suspend_d 返回未知类型：{sorted(suspension_values)}")
    adj_values = pd.to_numeric(normalized["adj_factor"]["adj_factor"], errors="coerce")
    if adj_values.isna().any() or adj_values.le(0.0).any():
        raise EvidenceContractError("adj_factor 探测值必须为有限正数")
    return normalized


def _endpoint_candidates(preferred: str) -> list[str]:
    candidates = [preferred]
    for candidate in (
        "https://api.tushare.pro",
        "https://fast.xiaodefa.cn",
        "https://tt.xiaodefa.cn",
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _query_frames(api: object, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    probe = config["probe_contract"]
    daily_fields = ",".join(config["provider_contract"]["daily"]["required_fields"])
    suspension_fields = ",".join(
        config["provider_contract"]["suspend_d"]["required_fields"]
    )
    dividend_fields = ",".join(
        config["provider_contract"]["dividend"]["required_fields"]
    )
    adj_fields = ",".join(config["provider_contract"]["adj_factor"]["required_fields"])
    return {
        "daily": api.daily(
            ts_code=str(probe["daily_symbol"]),
            start_date=str(probe["daily_start"]).replace("-", ""),
            end_date=str(probe["daily_end"]).replace("-", ""),
            fields=daily_fields,
        ),
        "suspend_d": api.suspend_d(
            trade_date=str(probe["suspension_trade_date"]).replace("-", ""),
            suspend_type=str(probe["suspension_type"]),
            fields=suspension_fields,
        ),
        "dividend": api.dividend(
            ts_code=str(probe["dividend_symbol"]),
            fields=dividend_fields,
        ),
        "adj_factor": api.adj_factor(
            ts_code=str(probe["adj_factor_symbol"]),
            start_date=str(probe["adj_factor_start"]).replace("-", ""),
            end_date=str(probe["adj_factor_end"]).replace("-", ""),
            fields=adj_fields,
        ),
    }


def _write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise EvidenceContractError(f"探测快照已存在，禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise EvidenceContractError(f"探测临时文件已存在：{temporary}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_probe(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    probe = config["probe_contract"]
    receipt_path = ROOT / Path(normalize_project_relative_path(str(probe["receipt_path"])))
    status_path = ROOT / Path(normalize_project_relative_path(str(probe["status_path"])))
    if receipt_path.exists() or status_path.exists():
        raise EvidenceContractError("有界探测输出已存在，禁止重复探测或覆盖")

    secret, preferred = credentials()
    failures: list[dict[str, str]] = []
    selected_endpoint: str | None = None
    normalized: dict[str, pd.DataFrame] | None = None
    for endpoint in _endpoint_candidates(preferred):
        api = ts.pro_api(secret)
        api._DataApi__http_url = endpoint
        try:
            normalized = validate_endpoint_frames(config, _query_frames(api, config))
            selected_endpoint = endpoint
            break
        except Exception as exc:
            message = str(exc).replace(secret, "<EPHEMERAL_CREDENTIAL_REMOVED>")
            failures.append(
                {
                    "endpoint": endpoint,
                    "error_type": type(exc).__name__,
                    "message": message[:500],
                }
            )
    if selected_endpoint is None or normalized is None:
        raise EvidenceContractError(f"所有允许节点的小样本探测均失败：{failures}")

    snapshot_dir = ROOT / Path(
        normalize_project_relative_path(str(probe["snapshot_directory"]))
    )
    snapshot_evidence: dict[str, dict[str, Any]] = {}
    for source_name, frame in normalized.items():
        target = snapshot_dir / f"{source_name}_probe.parquet"
        _write_parquet_new(frame, target)
        snapshot_evidence[source_name] = file_evidence(target, project_root=ROOT)
        snapshot_evidence[source_name]["row_count"] = int(len(frame))
        snapshot_evidence[source_name]["columns"] = frame.columns.astype(str).tolist()

    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCE_PROBE_V1",
        "created_at": created_at,
        "status": "PASS_BOUNDED_ENDPOINT_SCHEMA_AND_IDENTITY_PROBE",
        "addendum_id": config["program"]["addendum_id"],
        "selected_endpoint": selected_endpoint,
        "snapshots": snapshot_evidence,
        "failed_endpoint_attempts": failures,
        "semantic_boundaries": {
            "suspend_d_official_exchange_evidence_eligible": False,
            "suspend_d_zero_return_authorization": False,
            "adj_factor_direct_return_construction_allowed": False,
            "bulk_collection_authorized_by_this_receipt": True,
            "historical_four_state_coverage_claimed": False,
        },
        "credential_persisted": False,
        "performance_values_read": False,
        "security_audit_performed": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    status: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "addendum_id": config["program"]["addendum_id"],
        "updated_at": created_at,
        "source_probe": "PASS",
        "next_allowed_step": "FREEZE_BOUNDED_HISTORICAL_ACQUISITION_AND_FOUR_STATE_LEDGER_BUILD",
        "official_exchange_suspension_source": "NOT_ADMITTED",
        "historical_four_state_coverage": "NOT_RUN",
        "model_state": "NO_VIEW_PENDING_FOUR_STATE_LEDGER",
        "g0_status": "NOT_PASSED",
        "g1_through_g7_status": "NOT_RUN",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
        "receipt": file_evidence(receipt_path, project_root=ROOT),
    }
    atomic_write_json_new(status_path, status)
    print(
        "四态来源有界探测通过；只解锁下一步历史准入构建，"
        "未解锁模型、绩效、仓位或交易。"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        run_probe(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"四态来源有界探测失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
