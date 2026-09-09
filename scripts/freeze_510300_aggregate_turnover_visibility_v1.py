"""在首次读取510300候选收益前冻结全市场换手率可见度V1。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from aggregate_turnover_visibility_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_aggregate_turnover_visibility_v1.yaml",
    "research/aggregate_turnover_visibility_v1.py",
    "scripts/build_510300_aggregate_turnover_inputs_v1.py",
    "scripts/freeze_510300_aggregate_turnover_visibility_v1.py",
    "scripts/run_510300_aggregate_turnover_visibility_v1.py",
    "tests/test_510300_aggregate_turnover_visibility_v1.py",
]


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze() -> dict[str, Any]:
    config = load_config()
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"冻结清单已经存在，禁止覆盖：{MANIFEST_PATH}")
    result_keys = [
        "result_json",
        "result_markdown",
        "signal_table",
        "mechanism_table",
        "base_ledger",
        "base_trades",
        "stress_ledger",
        "stress_trades",
        "buy_hold_ledger",
        "buy_hold_trades",
    ]
    preexisting = [
        config["paths"][key]
        for key in result_keys
        if _project_path(config["paths"][key]).exists()
    ]
    if preexisting:
        raise RuntimeError(f"冻结前已经存在候选结果：{preexisting}")

    tracked: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结文件：{relative}")
        tracked[relative] = sha256_file(path)

    input_keys = [
        "etf_daily",
        "etf_input_audit",
        "dividends",
        "benchmark_total_return",
        "official_source_archive",
        "official_monthly_components",
        "monthly_factor",
        "input_audit",
        "backtest_engine",
    ]
    inputs: dict[str, str] = {}
    for key in input_keys:
        relative = config["inputs"][key]["path"]
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        inputs[relative] = sha256_file(path)

    audit_path = _project_path(config["inputs"]["input_audit"]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("输入审计不是PASS，禁止冻结")
    if audit.get("candidate_outcomes_computed") is not False:
        raise RuntimeError("输入审计曾计算候选结果，禁止冻结")
    if audit.get("portfolio_returns_computed") is not False:
        raise RuntimeError("输入审计曾计算组合收益，禁止冻结")
    if audit.get("calendar", {}).get("only_date_column_read") is not True:
        raise RuntimeError("输入审计未证明冻结前只读取510300日期列")
    if audit.get("calendar", {}).get("price_columns_read") is not False:
        raise RuntimeError("输入审计在冻结前读取了510300价格列")

    prior_manifests = sorted(
        path
        for path in (ROOT / "config").glob("510300_*_manifest.json")
        if path.resolve() != MANIFEST_PATH.resolve()
    )
    prior_manifest_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifests
    }
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "state": "FROZEN_BEFORE_FIRST_RESULT",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "candidate_outcomes_read_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
        "only_510300_date_column_read_by_input_builder_before_freeze": True,
        "full_510300_price_file_loaded_for_coverage_inspection_before_freeze": True,
        "candidate_factor_joined_to_prices_before_freeze": False,
        "future_return_series_computed_before_freeze": False,
        "sharpe_computed_before_freeze": False,
        "official_factor_values_read_before_freeze": True,
        "pre_2015_510300_returns_read": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "paper_sample_end": config["literature"]["primary_paper"]["sample_end"],
        "evaluation_start": config["dates"]["evaluation_start"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
        "input_audit_sha256": sha256_file(audit_path),
        "official_source_archive_sha256": inputs[
            config["inputs"]["official_source_archive"]["path"]
        ],
        "historical_availability_status": config["data_contract"][
            "historical_availability_status"
        ],
        "szse_proxy_scope": "STOCK_TOTAL_INCLUDING_B_AND_CDR",
        "szse_proxy_scope_frozen": True,
        "selection_bias_control": {
            "rule": config["selection_bias"]["trial_count_rule"],
            "prior_510300_manifest_count": len(prior_manifests),
            "total_trial_count_including_current": len(prior_manifests) + 1,
            "prior_manifest_hashes": prior_manifest_hashes,
            "count_is_conservative_upper_bound_not_independence_claim": True,
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)
    receipt_path = _project_path(config["paths"]["freeze_receipt"])
    receipt = {
        "project_id": manifest["project_id"],
        "status": "FROZEN_BEFORE_FIRST_RESULT",
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": manifest["config_sha256"],
        "tracked_file_count": len(tracked),
        "input_file_count": len(inputs),
        "prior_510300_manifest_count": len(prior_manifests),
        "total_trial_count_including_current": len(prior_manifests) + 1,
        "result_preexisted_at_freeze": False,
        "candidate_outcomes_read_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
        "only_510300_date_column_read_by_input_builder_before_freeze": True,
        "full_510300_price_file_loaded_for_coverage_inspection_before_freeze": True,
        "candidate_factor_joined_to_prices_before_freeze": False,
        "future_return_series_computed_before_freeze": False,
        "sharpe_computed_before_freeze": False,
        "official_source_archive_sha256": manifest["official_source_archive_sha256"],
        "historical_archive_reconstruction_not_timestamp_proof": True,
        "szse_proxy_scope_frozen": True,
        "live_trading_authorized": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    receipt = freeze()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
