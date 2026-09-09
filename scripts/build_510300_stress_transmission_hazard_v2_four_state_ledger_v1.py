"""从冻结采集输入构建 V2 成分收益四态与日度覆盖账本。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    verify_manifest_payload,
)
from research.stress_transmission_hazard_v2 import ConstituentReturnState
from research.stress_transmission_hazard_v2_four_state_ledger_v1 import (
    build_four_state_ledgers,
    normalize_membership,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _verify_file_evidence(evidence: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(evidence["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != str(
        evidence["sha256"]
    ):
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return path


def verify_acquisition(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _project_path(str(config["acquisition"]["outputs"]["acquisition_manifest"]))
    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise EvidenceContractError("采集 manifest 必须是对象")
    verify_manifest_payload(payload)
    if payload.get("status") != "PASS_BOUNDED_DAILY_AND_ACTION_CANDIDATE_ACQUISITION":
        raise EvidenceContractError("采集 manifest 不是通过状态")
    if payload.get("performance_values_read") is not False:
        raise EvidenceContractError("采集 manifest 的绩效读取状态非法")
    _verify_file_evidence(payload["daily"]["panel"], "新未复权日线面板")
    _verify_file_evidence(payload["dividend"]["panel"], "候选公司行动面板")
    _verify_file_evidence(payload["dividend"]["candidate_symbols"], "行动候选证券清单")
    return payload


def _write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise EvidenceContractError(f"账本输出已存在，禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise EvidenceContractError(f"账本临时输出已存在：{temporary}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_text_new(text: str, path: Path) -> None:
    if path.exists():
        raise EvidenceContractError(f"报告输出已存在，禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise EvidenceContractError(f"报告临时输出已存在：{temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _state_counts_on_member_days(
    membership: pd.DataFrame, classified: pd.DataFrame
) -> pd.DataFrame:
    members = normalize_membership(membership)
    joined = members.merge(
        classified[["date", "symbol", "constituent_return_state"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if joined["constituent_return_state"].isna().any():
        raise EvidenceContractError("显式四态账本仍缺少点时成员日")
    counts = (
        joined.groupby(["date", "constituent_return_state"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for state in ConstituentReturnState:
        if state.value not in counts:
            counts[state.value] = 0
    state_columns = [state.value for state in ConstituentReturnState]
    if not counts[state_columns].sum(axis=1).eq(300).all():
        raise EvidenceContractError("日度四态计数合计不是 300")
    return counts[["date", *state_columns]]


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    acquisition = verify_acquisition(config)
    output_contract = config["ledger_outputs"]
    outputs = {key: _project_path(str(value)) for key, value in output_contract.items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise EvidenceContractError(f"四态账本最终输出已存在，禁止覆盖：{existing}")

    legacy_contract = config["inputs"]["legacy_unadjusted_daily_seed"]
    legacy = pd.read_parquet(
        _project_path(str(legacy_contract["path"])),
        columns=legacy_contract["allowed_columns"],
    )
    fresh = pd.read_parquet(_verify_file_evidence(acquisition["daily"]["panel"], "新日线"))
    dividends = pd.read_parquet(
        _verify_file_evidence(acquisition["dividend"]["panel"], "公司行动")
    )
    membership = pd.read_parquet(
        _project_path(str(config["inputs"]["point_in_time_membership"]["path"]))
    )
    validation = config["daily_validation"]
    result = build_four_state_ledgers(
        legacy_daily=legacy,
        fresh_daily=fresh,
        dividends=dividends,
        membership=membership,
        price_tolerance=float(
            validation["cross_source_overlap"]["price_absolute_tolerance_cny"]
        ),
        pct_tolerance=float(validation["pct_chg_absolute_tolerance_percentage_points"]),
        member_coverage_minimum=float(
            config["four_state_execution"]["daily_member_coverage_minimum"]
        ),
    )
    state_counts = _state_counts_on_member_days(membership, result.classified_returns)

    _write_parquet_new(result.classified_returns, outputs["classified_returns"])
    _write_parquet_new(result.corporate_action_ledger, outputs["corporate_actions"])
    _write_parquet_new(result.daily_coverage, outputs["daily_coverage"])
    _write_parquet_new(state_counts, outputs["daily_state_counts"])

    state_totals = {
        state.value: int(state_counts[state.value].sum()) for state in ConstituentReturnState
    }
    coverage = result.daily_coverage
    no_view = coverage["aggregation_state"].eq("NO_VIEW")
    action_counts = result.corporate_action_ledger[
        "corporate_action_status"
    ].value_counts(dropna=False).to_dict()
    report = "\n".join(
        [
            "# 510300 压力传导危险率 V2：成分收益四态与覆盖账本 V1",
            "",
            "## 执行结果",
            "",
            f"- 点时成员交易日：{len(coverage):,}",
            f"- 显式点时成员日：{int(state_counts[[state.value for state in ConstituentReturnState]].to_numpy().sum()):,}",
            f"- `TRADED_VALID`：{state_totals[ConstituentReturnState.TRADED_VALID.value]:,}",
            f"- `OFFICIAL_SUSPENSION`：{state_totals[ConstituentReturnState.OFFICIAL_SUSPENSION.value]:,}（当前没有准入交易所官方源）",
            f"- `CORPORATE_ACTION_UNRESOLVED`：{state_totals[ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value]:,}",
            f"- `SUPPLIER_MISSING_OR_CONFLICT`：{state_totals[ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value]:,}",
            f"- 覆盖通过日：{int((~no_view).sum()):,}",
            f"- `NO_VIEW` 日：{int(no_view.sum()):,}",
            f"- 最低可用成员比例：{float(coverage['usable_member_ratio'].min()):.6f}",
            f"- 公司行动候选裁决：{action_counts}",
            "",
            "## 边界",
            "",
            "供应商停牌候选没有被冒充为交易所官方证明；缺行未填 0。旧链 `total_return_*`、链接复权因子和停牌列没有进入本账本。公司行动仅在实施公告严格早于除权日且理论除权参考价通过时解析。",
            "",
            "本报告只包含数据状态与覆盖，不包含 AUC、未来收益、夏普、回撤、净值、仓位或订单。",
            "",
        ]
    )
    _write_text_new(report, outputs["report"])

    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    evidence = {
        key: file_evidence(outputs[key], project_root=ROOT)
        for key in (
            "classified_returns",
            "corporate_actions",
            "daily_coverage",
            "daily_state_counts",
            "report",
        )
    }
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_LEDGER_V1",
        "created_at": now,
        "status": "PASS_FOUR_STATE_LEDGER_WITH_EXPLICIT_NO_VIEW_CHAIN",
        "outputs": evidence,
        "metrics": {
            "market_session_count": int(len(coverage)),
            "point_in_time_member_day_count": int(
                state_counts[[state.value for state in ConstituentReturnState]].to_numpy().sum()
            ),
            "state_totals": state_totals,
            "view_allowed_day_count": int((~no_view).sum()),
            "no_view_day_count": int(no_view.sum()),
            "minimum_usable_member_ratio": float(coverage["usable_member_ratio"].min()),
            "action_status_counts": {str(k): int(v) for k, v in action_counts.items()},
        },
        "invariants": {
            "official_suspension_count_is_zero_without_exchange_source": True,
            "unusable_states_have_null_return": True,
            "missing_return_zero_filled": False,
            "legacy_adjusted_return_columns_read": False,
            "performance_values_read": False,
        },
        "return_evaluation": "NOT_ALLOWED",
        "security_audit_performed": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["receipt"], receipt)
    status: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
        "updated_at": now,
        "input_acquisition": "PASS",
        "four_state_ledger": "PASS_WITH_NO_VIEW_DATES_RETAINED",
        "next_allowed_step": "FREEZE_POINT_IN_TIME_M_F_T_FEATURE_BUILD",
        "model_state": "NO_VIEW_ON_FAILED_COVERAGE_DATES_OTHERWISE_FEATURE_BUILD_ALLOWED",
        "g0_status": "NOT_PASSED_PENDING_FEATURE_NO_FUTURE_TESTS_AND_CLEAN_REPLAY",
        "g1_through_g7_status": "NOT_RUN",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
        "receipt": file_evidence(outputs["receipt"], project_root=ROOT),
    }
    atomic_write_json_new(outputs["status"], status)
    print("成分收益四态与日度覆盖账本已构建；绩效与交易权限继续关闭。")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        build(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"四态账本构建失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
