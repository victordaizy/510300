"""冻结或验证 510300 非对称压力风险 V1 的实际 G2 机制协议。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_g2_mechanism_v1 import (
    G2_FAIL_STATUS,
    G2_PASS_STATUS,
    parse_subperiods,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    ensure_paths_committed_at_head,
    git_head,
    read_json_strict,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "config/510300_asymmetric_stress_hazard_v1_g2_mechanism.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("G2 配置必须是对象")
    return payload


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{field} 必须是对象")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    program = _mapping(config.get("program"), field="program")
    expected = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_MECHANISM_CHAIN",
        "version": "1.0.3",
        "program_correction": "NORMALIZE_TIMEZONE_AWARE_DATETIME_PRECISION_BEFORE_FEATURE_RESULT",
        "continuation_basis": "USER_DIRECTED_ORIGINAL_FILE_ROUTE_20260903",
        "research_state": "DISCOVERY_ONLY",
        "position_impact": 0,
        "observation_start": "2015-01-05",
        "observation_cutoff": "2026-08-14",
        "probability_model_training_allowed": False,
        "threshold_selection_allowed": False,
        "portfolio_evaluation_allowed": False,
        "extra_temporal_preflight_is_active_gate": False,
    }
    for field, required in expected.items():
        actual = program.get(field)
        normalized = (
            str(actual)
            if field in {"observation_start", "observation_cutoff"}
            else actual
        )
        if normalized != required:
            raise EvidenceContractError(
                f"G2 程序字段漂移：{field}，expected={required!r}，actual={actual!r}"
            )
    if program.get("executable_assets") != ["510300.SH", "CASH_CNY"]:
        raise EvidenceContractError("可执行资产边界漂移")
    periods = parse_subperiods(config.get("fixed_subperiods", []))
    expected_periods = [
        ("P1_2015_2017", "2015-01-05", "2017-12-29"),
        ("P2_2018_2020", "2018-01-02", "2020-12-31"),
        ("P3_2021_2023", "2021-01-04", "2023-12-29"),
        ("P4_2024_CUTOFF", "2024-01-02", "2026-08-14"),
    ]
    actual_periods = [
        (item.period_id, item.start.date().isoformat(), item.end.date().isoformat())
        for item in periods
    ]
    if actual_periods != expected_periods:
        raise EvidenceContractError("固定四子期漂移")

    feature = _mapping(config.get("feature_contract"), field="feature_contract")
    fixed = {
        "constituent_return_lookback_market_days": 20,
        "transmission_change_market_days": 5,
        "macro_lead_market_days": 5,
        "constituent_price_cutover_date": "2020-02-03",
        "industry_interval_rule": "EFFECTIVE_DATE_INCLUSIVE_OUT_DATE_EXCLUSIVE",
        "internal_score": "SQRT_F_TIMES_T",
        "full_score": "CUBE_ROOT_M_TIMES_F_TIMES_T",
        "all_fixed_channels_required": True,
    }
    for field, required in fixed.items():
        actual = feature.get(field)
        if field == "constituent_price_cutover_date":
            actual = str(actual)
        if actual != required:
            raise EvidenceContractError(f"G2 特征口径漂移：{field}")

    gate = _mapping(config.get("g2_gate"), field="g2_gate")
    expected_gate = {
        "macro_direction_positive_subperiods_required": 3,
        "macro_direction_total_subperiods": 4,
        "internal_score_must_strictly_beat_price_baseline_roc_auc": True,
        "minimum_scoreable_independent_events": 30,
        "minimum_scoreable_independent_non_event_blocks": 120,
        "event_unit_score_date": "EVENT_ORIGIN_START",
        "non_event_unit_score_date": "BLOCK_ORIGIN_DATE",
        "pass_status": G2_PASS_STATUS,
        "failure_status": G2_FAIL_STATUS,
        "feature_or_window_rescue_after_failure_allowed": False,
    }
    for field, required in expected_gate.items():
        if gate.get(field) != required:
            raise EvidenceContractError(f"G2 门控字段漂移：{field}")


def _input_evidence(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    inputs = _mapping(config.get("inputs"), field="inputs")
    for name, raw in sorted(inputs.items()):
        contract = _mapping(raw, field=f"inputs.{name}")
        relative = str(contract.get("path", ""))
        path = ROOT / Path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"G2 输入不存在：{relative}")
        actual = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual["bytes"] != int(contract.get("bytes", -1)):
            raise EvidenceContractError(f"G2 输入字节漂移：{relative}")
        if actual["sha256"] != str(contract.get("sha256", "")):
            raise EvidenceContractError(f"G2 输入哈希漂移：{relative}")
        evidence[str(name)] = actual
    return evidence


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    outputs = _mapping(config.get("outputs"), field="outputs")
    manifest_path = ROOT / Path(str(outputs["manifest"]))
    if manifest_path.exists():
        raise EvidenceContractError("G2 manifest 已存在，禁止覆盖")
    pre_result_files = list(
        _mapping(config.get("freeze_contract"), field="freeze_contract").get(
            "pre_result_files", []
        )
    )
    committed = ensure_paths_committed_at_head(ROOT, pre_result_files)
    payload: dict[str, Any] = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_MECHANISM_CHAIN",
        "version": "1.0.3",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "git_head_before_manifest": git_head(ROOT),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "pre_result_files": committed,
        "inputs": _input_evidence(config),
        "result_values_read": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    atomic_write_json_new(manifest_path, payload)
    return payload


def verify_frozen_contract(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    validate_config(config)
    manifest_path = ROOT / Path(str(config["outputs"]["manifest"]))
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("G2 manifest 必须是对象")
    expected_payload_hash = str(manifest.get("manifest_payload_sha256", ""))
    without_hash = dict(manifest)
    without_hash.pop("manifest_payload_sha256", None)
    if canonical_sha256(without_hash) != expected_payload_hash:
        raise EvidenceContractError("G2 manifest payload 哈希不一致")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise EvidenceContractError("G2 配置在冻结后漂移")
    pre_result_files = list(config["freeze_contract"]["pre_result_files"])
    committed = ensure_paths_committed_at_head(
        ROOT,
        [*pre_result_files, manifest_path.relative_to(ROOT).as_posix()],
    )
    for relative, expected in manifest["pre_result_files"].items():
        if committed[relative]["working_sha256"] != expected["working_sha256"]:
            raise EvidenceContractError(f"G2 冻结文件漂移：{relative}")
    if _input_evidence(config) != manifest.get("inputs"):
        raise EvidenceContractError("G2 输入证据漂移")
    return config, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            _, manifest = verify_frozen_contract()
            status = "PASS_G2_MECHANISM_FROZEN_CONTRACT_VERIFIED"
        else:
            manifest = freeze()
            status = "PASS_G2_MECHANISM_CONTRACT_FROZEN"
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL_G2_MECHANISM_FREEZE: {exc}", file=sys.stderr)
        return 1
    print(status)
    print(f"manifest_payload_sha256={manifest['manifest_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
