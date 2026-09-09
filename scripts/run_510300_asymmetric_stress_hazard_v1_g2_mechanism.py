"""按原始旧文件路线运行 510300 非对称压力风险 V1 的实际 G2。"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_g2_mechanism_v1 import (
    G2MechanismError,
    assemble_feature_panel,
    assign_effective_industry,
    build_constituent_return_index,
    build_etf_total_return_features,
    build_independent_units,
    build_internal_features,
    build_macro_features,
    evaluate_g2,
    parse_subperiods,
    prepare_membership,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    atomic_write_text_new,
    canonical_sha256,
    file_evidence,
    git_head,
    strict_json_text,
)
from scripts.freeze_510300_asymmetric_stress_hazard_v1_g2_mechanism import (
    DEFAULT_CONFIG,
    verify_frozen_contract,
)


def _path(relative: str) -> Path:
    return ROOT / Path(relative)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{field} 必须是对象")
    return value


def _assert_outputs_absent(outputs: Mapping[str, Any]) -> None:
    existing = [
        str(relative)
        for name, relative in outputs.items()
        if name != "manifest" and _path(str(relative)).exists()
    ]
    if existing:
        raise EvidenceContractError(f"G2 输出已存在，禁止覆盖：{existing}")


def _write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        frame.to_parquet(temporary, index=False)
        if path.exists():
            raise EvidenceContractError(f"并发创建冲突：{path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _human_report(result: Mapping[str, Any]) -> str:
    gate = _mapping(result["g2_gate"], field="g2_gate")
    macro = _mapping(gate["macro_lead_gate"], field="macro_lead_gate")
    internal = _mapping(gate["internal_bad10_gate"], field="internal_bad10_gate")
    period_rows = [
        (
            f"| `{item['subperiod_id']}` | {item['scoreable_daily_observations']} | "
            f"{item['spearman_M_to_internal_change_t_plus_5']} | "
            f"{'是' if item['strictly_positive'] else '否'} |"
        )
        for item in macro["subperiods"]
    ]
    decision = (
        "G2 两项条件同时通过，下一步只允许冻结并运行 G3 固定概率模型。"
        if gate["passed"]
        else "G2 至少一项条件未通过；按旧文件整条 V1 在此冻结，不加特征、不改窗口。"
    )
    return "\n".join(
        [
            "# 510300_ASYMMETRIC_STRESS_HAZARD_V1 G2 机制链结果",
            "",
            "## 裁决",
            "",
            f"- 状态：`{gate['status']}`",
            f"- G2 通过：`{str(gate['passed']).lower()}`",
            f"- 宏观领先门：`{str(macro['passed']).lower()}`",
            f"- 内部结构对 BAD10 门：`{str(internal['passed']).lower()}`",
            "",
            decision,
            "",
            "## M 领先内部恶化",
            "",
            "| 固定子期 | 有效日 | Spearman(M, 内部分数未来5日变化) | 正方向 |",
            "|---|---:|---:|---:|",
            *period_rows,
            "",
            (
                f"正方向子期 `{macro['positive_subperiod_count']}` / "
                f"要求 `{macro['required_positive_subperiods']}`。"
            ),
            "",
            "## F/T 对 BAD10",
            "",
            f"- 可评分独立事件：`{internal['scoreable_event_count']}` / 要求 `{internal['minimum_event_count']}`。",
            f"- 可评分独立非事件块：`{internal['scoreable_non_event_count']}` / 要求 `{internal['minimum_non_event_count']}`。",
            f"- 内部分数 ROC AUC：`{internal['internal_score_roc_auc']}`。",
            f"- 价格基准 ROC AUC：`{internal['price_baseline_roc_auc']}`。",
            f"- AUC 差：`{internal['auc_difference_internal_minus_price']}`。",
            "",
            "## 本阶段边界",
            "",
            "- 已按历史生效日/失效日区间构造点时申万一级行业，没有用当前行业回填。",
            "- 已构造固定 M/F/T、内部 `sqrt(F*T)` 与价格风险基准。",
            "- 未训练概率模型，未选择阈值，未读取组合夏普、回撤或仓位。",
            "- `RESEARCH_STATE=DISCOVERY_ONLY`，`POSITION_IMPACT=0`。",
            "",
        ]
    )


def run() -> dict[str, Any]:
    config, manifest = verify_frozen_contract(DEFAULT_CONFIG)
    outputs = _mapping(config["outputs"], field="outputs")
    _assert_outputs_absent(outputs)
    inputs = _mapping(config["inputs"], field="inputs")
    started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    def load_parquet(name: str) -> pd.DataFrame:
        return pd.read_parquet(_path(str(_mapping(inputs[name], field=name)["path"])))

    etf_prices = load_parquet("etf_price")
    etf_dividends = pd.read_csv(
        _path(str(_mapping(inputs["etf_dividends"], field="etf_dividends")["path"])),
        encoding="utf-8",
    )
    start = str(config["program"]["observation_start"])
    cutoff = str(config["program"]["observation_cutoff"])
    etf = build_etf_total_return_features(
        etf_prices,
        etf_dividends,
        start=start,
        cutoff=cutoff,
    )
    market_dates = pd.DatetimeIndex(etf["date"])

    membership = prepare_membership(
        load_parquet("pit_membership"),
        start=start,
        cutoff=cutoff,
    )
    membership_with_industry = assign_effective_industry(
        membership,
        load_parquet("sw_industry_history"),
    )
    constituent_index = build_constituent_return_index(
        load_parquet("constituent_history"),
        load_parquet("constituent_current"),
        market_dates=market_dates,
        cutover_date=str(config["feature_contract"]["constituent_price_cutover_date"]),
    )
    internal, sector = build_internal_features(
        membership_with_industry=membership_with_industry,
        constituent_index=constituent_index,
        etf_features=etf,
        lookback_days=int(
            config["feature_contract"]["constituent_return_lookback_market_days"]
        ),
        change_days=int(
            config["feature_contract"]["transmission_change_market_days"]
        ),
    )
    macro = build_macro_features(
        market_dates=market_dates,
        pe=load_parquet("csi300_pe"),
        bond=load_parquet("china_10y"),
        dr007=load_parquet("dr007"),
        policy=load_parquet("reverse_repo_7d"),
        credit=load_parquet("tsf_first_release"),
    )
    features = assemble_feature_panel(etf=etf, internal=internal, macro=macro)
    events = pd.read_csv(
        _path(str(_mapping(inputs["independent_events"], field="events")["path"])),
        encoding="utf-8",
    )
    non_events = pd.read_csv(
        _path(
            str(
                _mapping(inputs["independent_non_events"], field="non_events")[
                    "path"
                ]
            )
        ),
        encoding="utf-8",
    )
    units = build_independent_units(
        events=events,
        non_events=non_events,
        features=features,
    )
    gate_config = _mapping(config["g2_gate"], field="g2_gate")
    gate = evaluate_g2(
        features=features,
        units=units,
        subperiods=parse_subperiods(config["fixed_subperiods"]),
        lead_days=int(config["feature_contract"]["macro_lead_market_days"]),
        minimum_events=int(gate_config["minimum_scoreable_independent_events"]),
        minimum_non_events=int(
            gate_config["minimum_scoreable_independent_non_event_blocks"]
        ),
        required_positive_subperiods=int(
            gate_config["macro_direction_positive_subperiods_required"]
        ),
    )

    feature_columns = [
        "M",
        "F",
        "T",
        "internal_score",
        "full_mechanism_score",
        "price_baseline",
    ]
    result: dict[str, Any] = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_MECHANISM_CHAIN",
        "version": str(config["program"]["version"]),
        "continuation_basis": "USER_DIRECTED_ORIGINAL_FILE_ROUTE_20260903",
        "started_at": started_at,
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "git_head": git_head(ROOT),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "status": gate["status"],
        "passed": gate["passed"],
        "g2_gate": gate,
        "feature_coverage": {
            "market_session_count": int(len(features)),
            **{
                f"{column}_valid_count": int(features[column].notna().sum())
                for column in feature_columns
            },
            "first_internal_score_date": (
                features.loc[features["internal_score"].notna(), "date"]
                .min()
                .date()
                .isoformat()
            ),
            "first_full_mechanism_score_date": (
                features.loc[features["full_mechanism_score"].notna(), "date"]
                .min()
                .date()
                .isoformat()
            ),
            "industry_mapped_member_days": int(
                membership_with_industry["industry_l1_code"].notna().sum()
            ),
            "industry_unmapped_member_days": int(
                membership_with_industry["industry_l1_code"].isna().sum()
            ),
        },
        "feature_values_constructed": True,
        "independent_bad10_units_read": True,
        "probability_model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "order_generated": False,
        "broker_action_performed": False,
        "position_impact": 0,
        "authority": {
            "research_state": "DISCOVERY_ONLY",
            "model_position_target": "UNSET",
            "order_authorization": "NOT_AUTHORIZED",
            "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
        },
    }

    feature_path = _path(str(outputs["feature_panel"]))
    sector_path = _path(str(outputs["sector_return_panel"]))
    units_path = _path(str(outputs["independent_unit_panel"]))
    result_path = _path(str(outputs["result"]))
    report_path = _path(str(outputs["human_report"]))
    receipt_path = _path(str(outputs["receipt"]))
    _write_parquet_new(feature_path, features)
    _write_parquet_new(sector_path, sector)
    _write_parquet_new(units_path, units)
    atomic_write_json_new(result_path, result)
    atomic_write_text_new(report_path, _human_report(result))

    receipt: dict[str, Any] = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_MECHANISM_CHAIN",
        "version": str(config["program"]["version"]),
        "completed_at": result["completed_at"],
        "git_head": result["git_head"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "status": "IMMUTABLE_G2_MECHANISM_RECEIPT_COMPLETE",
        "research_status": gate["status"],
        "gate_passed": gate["passed"],
        "g3_allowed": gate["g3_allowed"],
        "outputs": {
            "feature_panel": file_evidence(feature_path, project_root=ROOT),
            "sector_return_panel": file_evidence(sector_path, project_root=ROOT),
            "independent_unit_panel": file_evidence(units_path, project_root=ROOT),
            "result": file_evidence(result_path, project_root=ROOT),
            "human_report": file_evidence(report_path, project_root=ROOT),
        },
        "probability_model_trained": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_path, receipt)
    return result


def main() -> int:
    try:
        result = run()
    except (EvidenceContractError, G2MechanismError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL_G2_MECHANISM_RUN: {exc}", file=sys.stderr)
        return 1
    print(str(result["status"]))
    gate = result["g2_gate"]
    print(f"passed={gate['passed']}")
    print(
        "macro_positive_subperiods="
        f"{gate['macro_lead_gate']['positive_subperiod_count']}"
    )
    print(
        "internal_auc="
        f"{gate['internal_bad10_gate']['internal_score_roc_auc']}"
    )
    print(
        "price_auc="
        f"{gate['internal_bad10_gate']['price_baseline_roc_auc']}"
    )
    print(f"next_allowed_action={gate['next_allowed_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
