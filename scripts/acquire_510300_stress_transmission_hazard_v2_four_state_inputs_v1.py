"""按冻结范围采集 V2 四态账本所需未复权日线与候选公司行动。"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
    sha256_file,
)
from research.stress_transmission_hazard_v2_four_state_ledger_v1 import (
    DAILY_PROVIDER_COLUMNS,
    DIVIDEND_COLUMNS,
    combine_daily_sources,
    identify_action_candidate_symbols,
    normalize_provider_daily,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise EvidenceContractError(f"临时输出已存在：{temporary}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_csv_new(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise EvidenceContractError(f"临时输出已存在：{temporary}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _read_checkpoint(
    path: Path,
    required_columns: list[str],
    symbol: str,
    label: str,
    *,
    allow_empty: bool,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise EvidenceContractError(f"{label}断点缺少字段：{path.name} -> {missing}")
    if frame.empty and not allow_empty:
        raise EvidenceContractError(f"{label}断点为空：{path.name}")
    if not frame.empty and not frame["ts_code"].astype(str).eq(symbol).all():
        raise EvidenceContractError(f"{label}断点混入其他证券：{path.name}")
    return frame


def _api(secret: str, endpoint: str) -> object:
    api = ts.pro_api(secret)
    api._DataApi__http_url = endpoint
    return api


def _retry_query(
    query: Callable[[], pd.DataFrame | None],
    *,
    retries: int,
    delay_seconds: float,
    label: str,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        time.sleep(delay_seconds)
        try:
            result = query()
            if result is None:
                raise EvidenceContractError(f"{label} 返回 None")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds * attempt)
    raise EvidenceContractError(f"{label} 连续 {retries} 次失败：{type(last_error).__name__}")


def acquire_daily_symbol(
    *,
    symbol: str,
    secret: str,
    endpoint: str,
    start_date: str,
    end_date: str,
    fields: list[str],
    checkpoint_dir: Path,
    retries: int,
    delay_seconds: float,
    pct_tolerance: float,
) -> pd.DataFrame:
    checkpoint = checkpoint_dir / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists():
        frame = _read_checkpoint(
            checkpoint,
            [*fields, "retrieved_at"],
            symbol,
            "日线",
            allow_empty=False,
        )
        normalize_provider_daily(frame, pct_tolerance=pct_tolerance)
        return frame
    api = _api(secret, endpoint)
    query_fields = ",".join(fields)
    frame = _retry_query(
        lambda: api.daily(
            ts_code=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            fields=query_fields,
        ),
        retries=retries,
        delay_seconds=delay_seconds,
        label=f"daily({symbol})",
    )
    if frame.empty:
        raise EvidenceContractError(f"daily({symbol}) 返回空表")
    frame = frame.loc[:, fields].copy()
    frame["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    normalize_provider_daily(frame, pct_tolerance=pct_tolerance)
    _write_parquet_new(frame, checkpoint)
    return frame


def acquire_dividend_symbol(
    *,
    symbol: str,
    secret: str,
    endpoint: str,
    fields: list[str],
    checkpoint_dir: Path,
    retries: int,
    delay_seconds: float,
) -> pd.DataFrame:
    checkpoint = checkpoint_dir / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists():
        return _read_checkpoint(
            checkpoint,
            [*fields, "retrieved_at"],
            symbol,
            "分红送转",
            allow_empty=True,
        )
    api = _api(secret, endpoint)
    query_fields = ",".join(fields)
    frame = _retry_query(
        lambda: api.dividend(ts_code=symbol, fields=query_fields),
        retries=retries,
        delay_seconds=delay_seconds,
        label=f"dividend({symbol})",
    )
    if frame.empty:
        frame = pd.DataFrame(columns=fields)
    else:
        missing = sorted(set(fields).difference(frame.columns))
        if missing:
            raise EvidenceContractError(f"dividend({symbol}) 缺少字段：{missing}")
        frame = frame.loc[:, fields].copy()
        if not frame["ts_code"].astype(str).eq(symbol).all():
            raise EvidenceContractError(f"dividend({symbol}) 混入其他证券")
    frame["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    _write_parquet_new(frame, checkpoint)
    return frame


def _parallel_collect(
    symbols: list[str],
    worker: Callable[[str], pd.DataFrame],
    *,
    workers: int,
    label: str,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, symbol): symbol for symbol in symbols}
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            if completed % 25 == 0 or completed == len(symbols):
                print(
                    f"{label}进度 {completed}/{len(symbols)}，成功 {len(results)}，失败 {len(failures)}",
                    flush=True,
                )
    if failures:
        raise EvidenceContractError(
            f"{label}仍有 {len(failures)} 个证券失败，样例={dict(list(failures.items())[:10])}"
        )
    return results


def _checkpoint_evidence(directory: Path, symbols: list[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        path = directory / f"{symbol.replace('.', '_')}.parquet"
        if not path.is_file():
            raise EvidenceContractError(f"断点文件缺失：{path}")
        evidence[symbol] = file_evidence(path, project_root=ROOT)
    return evidence


def run_acquisition(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    acquisition = config["acquisition"]
    outputs = acquisition["outputs"]
    output_paths = {
        key: _project_path(str(value)) for key, value in outputs.items()
    }
    fresh_panel_path = _project_path(
        str(acquisition["fresh_daily"]["normalized_panel_output"])
    )
    dividend_panel_path = _project_path(
        str(acquisition["dividend"]["normalized_panel_output"])
    )
    final_outputs = [fresh_panel_path, dividend_panel_path, *output_paths.values()]
    existing = [str(path) for path in final_outputs if path.exists()]
    if existing:
        raise EvidenceContractError(f"最终采集输出已存在，禁止覆盖：{existing}")

    membership_contract = config["inputs"]["point_in_time_membership"]
    membership = pd.read_parquet(_project_path(str(membership_contract["path"])))
    membership_dates = pd.to_datetime(membership["membership_date"], errors="coerce")
    fresh_start = pd.Timestamp(acquisition["fresh_daily"]["start_date"])
    daily_symbols = sorted(
        membership.loc[membership_dates.ge(fresh_start), "symbol"]
        .astype(str)
        .str.upper()
        .unique()
    )
    if len(daily_symbols) != int(acquisition["fresh_daily"]["expected_symbol_count"]):
        raise EvidenceContractError(
            f"新日线证券数漂移：expected={acquisition['fresh_daily']['expected_symbol_count']}，actual={len(daily_symbols)}"
        )

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
        label="未复权日线",
    )
    fresh_panel = (
        pd.concat([daily_results[symbol] for symbol in daily_symbols], ignore_index=True)
        .sort_values(["ts_code", "trade_date"], kind="stable")
        .reset_index(drop=True)
    )

    legacy = pd.read_parquet(
        _project_path(str(config["inputs"]["legacy_unadjusted_daily_seed"]["path"])),
        columns=config["inputs"]["legacy_unadjusted_daily_seed"]["allowed_columns"],
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
        label="公司行动候选",
    )
    nonempty_dividends = [
        dividend_results[symbol] for symbol in action_symbols if not dividend_results[symbol].empty
    ]
    dividend_panel = (
        pd.concat(nonempty_dividends, ignore_index=True)
        if nonempty_dividends
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
        "acquisition_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_INPUT_ACQUISITION_V1",
        "created_at": now,
        "status": "PASS_BOUNDED_DAILY_AND_ACTION_CANDIDATE_ACQUISITION",
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
        "security_audit_performed": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(output_paths["acquisition_manifest"], manifest)
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_INPUT_ACQUISITION_V1",
        "created_at": now,
        "status": manifest["status"],
        "acquisition_manifest": file_evidence(
            output_paths["acquisition_manifest"], project_root=ROOT
        ),
        "daily_symbol_count": len(daily_symbols),
        "action_candidate_symbol_count": len(action_symbols),
        "next_allowed_step": "BUILD_FOUR_STATE_RETURN_AND_COVERAGE_LEDGERS",
        "performance_values_read": False,
        "credential_persisted": False,
        "position_impact": 0,
    }
    atomic_write_json_new(output_paths["acquisition_receipt"], receipt)
    status: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
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
    print("四态账本输入采集通过；未复权日线与候选公司行动已形成不可覆盖收据。")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        run_acquisition(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"四态输入采集失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
