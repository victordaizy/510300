"""按 V1.0.1 点时覆盖选择器采集四态账本输入。"""

from __future__ import annotations

import argparse
import copy
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
)
from research.stress_transmission_hazard_v2_four_state_ledger_v1 import (
    DAILY_PROVIDER_COLUMNS,
    DIVIDEND_COLUMNS,
    combine_daily_sources,
    identify_action_candidate_symbols,
)
from scripts.acquire_510300_stress_transmission_hazard_v2_four_state_inputs_v1 import (
    _checkpoint_evidence,
    _parallel_collect,
    _write_csv_new,
    _write_parquet_new,
    acquire_daily_symbol,
    acquire_dividend_symbol,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1 import (
    DEFAULT_CONFIG as V1_CONFIG,
    load_config as load_v1_config,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def effective_config(correction: Mapping[str, Any]) -> dict[str, Any]:
    """把唯一选择器修正和新输出命名空间叠加到冻结 V1。"""

    base = copy.deepcopy(load_v1_config(V1_CONFIG))
    selector = correction["selector_correction"]
    outputs = correction["corrected_outputs"]
    base["program"]["execution_id"] = correction["program"]["correction_id"]
    base["program"]["version"] = correction["program"]["version"]
    base["acquisition"]["fresh_daily"]["symbol_universe_rule"] = selector[
        "corrected_symbol_universe_rule"
    ]
    base["acquisition"]["fresh_daily"]["expected_symbol_count"] = selector[
        "corrected_expected_symbol_count"
    ]
    base["acquisition"]["fresh_daily"]["checkpoint_directory"] = outputs[
        "daily_checkpoint_directory"
    ]
    base["acquisition"]["fresh_daily"]["normalized_panel_output"] = outputs[
        "fresh_daily_panel"
    ]
    base["acquisition"]["dividend"]["checkpoint_directory"] = outputs[
        "dividend_checkpoint_directory"
    ]
    base["acquisition"]["dividend"]["normalized_panel_output"] = outputs[
        "dividend_panel"
    ]
    acquisition_outputs = base["acquisition"]["outputs"]
    acquisition_outputs["action_candidate_symbols"] = outputs["action_candidate_symbols"]
    acquisition_outputs["acquisition_manifest"] = outputs["acquisition_manifest"]
    acquisition_outputs["acquisition_receipt"] = outputs["acquisition_receipt"]
    acquisition_outputs["acquisition_status"] = outputs["acquisition_status"]
    ledger_outputs = base["ledger_outputs"]
    ledger_outputs["classified_returns"] = outputs["classified_returns"]
    ledger_outputs["corporate_actions"] = outputs["corporate_actions"]
    ledger_outputs["daily_coverage"] = outputs["daily_coverage"]
    ledger_outputs["daily_state_counts"] = outputs["daily_state_counts"]
    ledger_outputs["report"] = outputs["ledger_report"]
    ledger_outputs["receipt"] = outputs["ledger_receipt"]
    ledger_outputs["status"] = outputs["ledger_status"]
    return base


def select_fresh_daily_symbols(
    membership: pd.DataFrame,
    *,
    legacy_seed_last_date: str,
) -> list[str]:
    """只选择旧种子截止日后仍需点时覆盖的历史成员。"""

    required = {"membership_date", "index_code", "symbol"}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise EvidenceContractError(f"点时成员缺少字段：{missing}")
    dates = pd.to_datetime(membership["membership_date"], errors="coerce")
    valid = (
        dates.gt(pd.Timestamp(legacy_seed_last_date))
        & membership["index_code"].astype(str).eq("000300")
    )
    return sorted(membership.loc[valid, "symbol"].astype(str).str.upper().unique())


def run_acquisition(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    correction = load_config(config_path)
    config = effective_config(correction)
    acquisition = config["acquisition"]
    outputs = acquisition["outputs"]
    output_paths = {key: _project_path(str(value)) for key, value in outputs.items()}
    fresh_panel_path = _project_path(
        str(acquisition["fresh_daily"]["normalized_panel_output"])
    )
    dividend_panel_path = _project_path(
        str(acquisition["dividend"]["normalized_panel_output"])
    )
    final_outputs = [fresh_panel_path, dividend_panel_path, *output_paths.values()]
    existing = [str(path) for path in final_outputs if path.exists()]
    if existing:
        raise EvidenceContractError(f"V1.0.1 最终采集输出已存在，禁止覆盖：{existing}")

    membership_contract = config["inputs"]["point_in_time_membership"]
    membership = pd.read_parquet(_project_path(str(membership_contract["path"])))
    daily_symbols = select_fresh_daily_symbols(
        membership,
        legacy_seed_last_date=str(
            correction["selector_correction"]["legacy_seed_last_date"]
        ),
    )
    expected_count = int(acquisition["fresh_daily"]["expected_symbol_count"])
    if len(daily_symbols) != expected_count:
        raise EvidenceContractError(
            f"修正后新日线证券数漂移：expected={expected_count}，actual={len(daily_symbols)}"
        )
    excluded = set(correction["selector_correction"]["excluded_overlap_only_symbols"])
    if excluded.intersection(daily_symbols):
        raise EvidenceContractError("修正后选择器仍包含仅重叠期证券")

    secret, _ = credentials()
    endpoint = str(config["source_admission_dependencies"]["selected_endpoint"])
    control = acquisition["request_control"]
    workers = int(control["maximum_workers"])
    retries = int(control["retries"])
    delay = float(control["minimum_delay_seconds_per_request"])
    pct_tolerance = float(
        config["daily_validation"]["pct_chg_absolute_tolerance_percentage_points"]
    )
    daily_dir = _project_path(str(acquisition["fresh_daily"]["checkpoint_directory"]))
    daily_fields = [str(value) for value in acquisition["fresh_daily"]["fields"]]
    if daily_fields != DAILY_PROVIDER_COLUMNS:
        raise EvidenceContractError("日线字段顺序与冻结实现不一致")
    daily_results = _parallel_collect(
        daily_symbols,
        lambda symbol: acquire_daily_symbol(
            symbol=symbol,
            secret=secret,
            endpoint=endpoint,
            start_date=str(acquisition["fresh_daily"]["start_date"]),
            end_date=str(acquisition["fresh_daily"]["end_date"]),
            fields=daily_fields,
            checkpoint_dir=daily_dir,
            retries=retries,
            delay_seconds=delay,
            pct_tolerance=pct_tolerance,
        ),
        workers=workers,
        label="未复权日线 V1.0.1",
    )
    fresh_panel = (
        pd.concat([daily_results[symbol] for symbol in daily_symbols], ignore_index=True)
        .sort_values(["ts_code", "trade_date"], kind="stable")
        .reset_index(drop=True)
    )

    legacy_contract = config["inputs"]["legacy_unadjusted_daily_seed"]
    legacy = pd.read_parquet(
        _project_path(str(legacy_contract["path"])),
        columns=legacy_contract["allowed_columns"],
    )
    combined = combine_daily_sources(
        legacy,
        fresh_panel,
        price_tolerance=float(
            config["daily_validation"]["cross_source_overlap"][
                "price_absolute_tolerance_cny"
            ]
        ),
        pct_tolerance=pct_tolerance,
    )
    action_symbols = identify_action_candidate_symbols(
        combined,
        absolute_tolerance_cny=float(
            config["corporate_action_reconciliation"][
                "expected_pre_close_absolute_tolerance_cny"
            ]
        ),
    )
    action_symbol_frame = pd.DataFrame(
        {"symbol": action_symbols, "reason": "DAILY_PRE_CLOSE_ACTION_CANDIDATE"}
    )

    dividend_dir = _project_path(str(acquisition["dividend"]["checkpoint_directory"]))
    dividend_fields = [str(value) for value in acquisition["dividend"]["fields"]]
    if dividend_fields != DIVIDEND_COLUMNS:
        raise EvidenceContractError("分红送转字段顺序与冻结实现不一致")
    dividend_results = _parallel_collect(
        action_symbols,
        lambda symbol: acquire_dividend_symbol(
            symbol=symbol,
            secret=secret,
            endpoint=endpoint,
            fields=dividend_fields,
            checkpoint_dir=dividend_dir,
            retries=retries,
            delay_seconds=delay,
        ),
        workers=workers,
        label="公司行动候选 V1.0.1",
    )
    nonempty = [
        dividend_results[symbol]
        for symbol in action_symbols
        if not dividend_results[symbol].empty
    ]
    dividend_panel = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=[*dividend_fields, "retrieved_at"])
    )
    dividend_panel = dividend_panel.sort_values(
        ["ts_code", "ex_date", "imp_ann_date"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    _write_parquet_new(fresh_panel, fresh_panel_path)
    _write_parquet_new(dividend_panel, dividend_panel_path)
    _write_csv_new(action_symbol_frame, output_paths["action_candidate_symbols"])
    daily_checkpoints = _checkpoint_evidence(daily_dir, daily_symbols)
    dividend_checkpoints = _checkpoint_evidence(dividend_dir, action_symbols)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "acquisition_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_INPUT_ACQUISITION_V1_0_1",
        "created_at": now,
        "status": "PASS_CORRECTED_PIT_COVERAGE_DAILY_AND_ACTION_CANDIDATE_ACQUISITION",
        "selector": correction["selector_correction"],
        "endpoint": endpoint,
        "daily": {
            "symbol_count": len(daily_symbols),
            "row_count": int(len(fresh_panel)),
            "panel": file_evidence(fresh_panel_path, project_root=ROOT),
            "checkpoints": daily_checkpoints,
        },
        "dividend": {
            "candidate_symbol_count": len(action_symbols),
            "row_count": int(len(dividend_panel)),
            "panel": file_evidence(dividend_panel_path, project_root=ROOT),
            "candidate_symbols": file_evidence(
                output_paths["action_candidate_symbols"], project_root=ROOT
            ),
            "checkpoints": dividend_checkpoints,
        },
        "excluded_bulk_sources": acquisition["excluded_bulk_sources"],
        "credential_persisted": False,
        "adjusted_or_legacy_total_return_read": False,
        "performance_values_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(output_paths["acquisition_manifest"], manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_INPUT_ACQUISITION_V1_0_1",
        "created_at": now,
        "status": manifest["status"],
        "acquisition_manifest": file_evidence(
            output_paths["acquisition_manifest"], project_root=ROOT
        ),
        "daily_symbol_count": len(daily_symbols),
        "action_candidate_symbol_count": len(action_symbols),
        "next_allowed_step": "BUILD_FOUR_STATE_RETURN_AND_COVERAGE_LEDGERS_V1_0_1",
        "performance_values_read": False,
        "credential_persisted": False,
        "position_impact": 0,
    }
    atomic_write_json_new(output_paths["acquisition_receipt"], receipt)
    status = {
        "program_id": config["program"]["program_id"],
        "execution_id": correction["program"]["correction_id"],
        "updated_at": now,
        "input_acquisition": "PASS",
        "four_state_ledger": "NOT_RUN",
        "model_state": "NO_VIEW_PENDING_FOUR_STATE_LEDGER",
        "g0_status": "NOT_PASSED",
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "receipt": file_evidence(output_paths["acquisition_receipt"], project_root=ROOT),
    }
    atomic_write_json_new(output_paths["acquisition_status"], status)
    print("V1.0.1 四态输入采集通过；只允许构建四态与覆盖账本。")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        run_acquisition(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"V1.0.1 四态输入采集失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
