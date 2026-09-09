"""按冻结规则重放510300微观结构双影子V1的留出与前瞻快照。

本程序只接受带逐文件SHA-256的只追加快照；先复算冻结历史前缀，再把
冻结日后的完整、连续信号日追加到同一状态机。规则、阈值、成本、基准
和起点扰动均从冻结配置读取，不允许命令行覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import validate_etf_microstructure_dual_shadow_v1_frozen as frozen
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
POLICY = "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_DUAL_SHADOW"
TIMEZONE = ZoneInfo("Asia/Shanghai")
VALIDATOR_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_v1_validation_implementation_receipt.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "runs"
)
STATUS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1_status.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _normalize(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError(f"{name}缺少date字段")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result["date"].duplicated().any():
        raise ValueError(f"{name}存在重复日期")
    return result


def _validate_frozen_validator() -> dict[str, Any]:
    receipt = json.loads(VALIDATOR_RECEIPT.read_text(encoding="utf-8"))
    validator_path = PROJECT_ROOT / receipt["validator"]
    actual = _sha256(validator_path)
    expected = receipt["validator_sha256"]
    if actual != expected:
        raise ValueError(
            f"冻结独立验证器哈希漂移：expected={expected},actual={actual}"
        )
    return {
        "validator": str(validator_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "expected_sha256": expected,
        "actual_sha256": actual,
    }


def _load_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"快照清单不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ValueError("快照项目编号不匹配")
    frames: dict[str, pd.DataFrame] = {}
    for name in ["etf", "benchmark", "nav", "fund_share", "margin"]:
        record = manifest["datasets"][name]
        path = PROJECT_ROOT / record["file"]
        if not path.resolve().is_relative_to(snapshot_dir.resolve()):
            raise ValueError(f"快照文件越出快照目录：{path}")
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"快照文件哈希漂移：{name},expected={record['sha256']},actual={actual}"
            )
        frames[name] = _normalize(pd.read_parquet(path), name=f"snapshot_{name}")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    manifest["manifest_file"] = str(manifest_path.relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    )
    return manifest, frames


def _read_frozen_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    paths = {
        name: PROJECT_ROOT / item["path"] for name, item in config["inputs"].items()
    }
    frames = {
        "nav": _normalize(pd.read_parquet(paths["nav_daily"]), name="frozen_nav"),
        "fund_share": _normalize(
            pd.read_parquet(paths["fund_share_daily"]), name="frozen_fund_share"
        ),
        "margin": _normalize(
            pd.read_parquet(paths["margin_detail_daily"]), name="frozen_margin"
        ),
        "etf": _normalize(pd.read_parquet(paths["etf_execution"]), name="frozen_etf"),
        "benchmark": _normalize(
            pd.read_parquet(paths["benchmark_total_return"]), name="frozen_benchmark"
        ),
    }
    dividends = pd.read_csv(paths["cash_distributions"])
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)
    return frames, dividends


def _combine_at_cutoff(
    historical: pd.DataFrame,
    snapshot: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    name: str,
) -> pd.DataFrame:
    left = historical.loc[historical["date"].le(cutoff)].copy()
    right = snapshot.loc[snapshot["date"].gt(cutoff)].copy()
    combined = pd.concat([left, right], ignore_index=True, sort=False)
    combined.sort_values("date", inplace=True)
    if combined["date"].duplicated().any():
        raise ValueError(f"{name}在冻结切点拼接后存在重复日期")
    combined.reset_index(drop=True, inplace=True)
    return combined


def _contiguous_signal_calendar(
    etf: pd.DataFrame,
    nav: pd.DataFrame,
    shares: pd.DataFrame,
    margin: pd.DataFrame,
    signal_cutoff: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    expected = pd.DatetimeIndex(
        etf.loc[etf["date"].gt(signal_cutoff), "date"].sort_values().unique()
    )
    valid_sets: dict[str, pd.DatetimeIndex] = {}
    for name, frame, columns in [
        ("nav", nav, ["close_premium_to_nav"]),
        ("fund_share", shares, ["fund_shares"]),
        ("margin", margin, ["rzye", "rqye", "rqyl", "rzrqye"]),
    ]:
        missing_columns = set(columns).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{name}缺少字段：{sorted(missing_columns)}")
        valid = frame.loc[frame[columns].notna().all(axis=1), "date"]
        valid_sets[name] = pd.DatetimeIndex(valid.unique())
    complete = expected
    for dates in valid_sets.values():
        complete = complete.intersection(dates)
    prefix: list[pd.Timestamp] = []
    first_incomplete: pd.Timestamp | None = None
    for date in expected:
        if date in complete:
            prefix.append(date)
        else:
            first_incomplete = date
            break
    missing = {
        name: [date.date().isoformat() for date in expected.difference(dates)]
        for name, dates in valid_sets.items()
    }
    audit = {
        "expected_post_cutoff_etf_dates": [date.date().isoformat() for date in expected],
        "contiguous_complete_signal_dates": [date.date().isoformat() for date in prefix],
        "first_incomplete_signal_date": (
            first_incomplete.date().isoformat() if first_incomplete is not None else None
        ),
        "missing_by_source": missing,
        "later_complete_rows_after_first_gap_ignored": int(
            max(len(complete) - len(prefix), 0)
        ),
    }
    return pd.DatetimeIndex(prefix), audit


def _build_signal(
    config: dict[str, Any],
    combined: dict[str, pd.DataFrame],
    post_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    start = pd.Timestamp(config["dates"]["signal_history_start"])
    cutoff = pd.Timestamp(config["dates"]["frozen_signal_cutoff"])
    end = post_dates.max() if len(post_dates) else cutoff
    calendar = combined["etf"].loc[
        combined["etf"]["date"].between(start, end), ["date"]
    ]
    signal = (
        calendar.merge(
            combined["nav"][["date", "close_premium_to_nav"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            combined["fund_share"][["date", "fund_shares"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            combined["margin"][["date", "rzye", "rqye", "rqyl", "rzrqye"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    )
    required = [
        "close_premium_to_nav",
        "fund_shares",
        "rzye",
        "rqye",
        "rqyl",
        "rzrqye",
    ]
    if signal[required].isna().any(axis=None):
        bad = signal.loc[signal[required].isna().any(axis=1), "date"]
        raise ValueError(
            "连续信号日历仍存在缺失："
            + ",".join(date.date().isoformat() for date in bad[:20])
        )
    return signal.reset_index(drop=True)


def _contiguous_market(
    etf: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_dates = set(benchmark["date"])
    expected_post = etf.loc[etf["date"].gt(market_cutoff), "date"].tolist()
    prefix: list[pd.Timestamp] = []
    first_gap: pd.Timestamp | None = None
    for date in expected_post:
        if date in benchmark_dates:
            prefix.append(date)
        else:
            first_gap = date
            break
    allowed_end = prefix[-1] if prefix else market_cutoff
    market = etf.loc[etf["date"].le(allowed_end), ["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        benchmark.loc[benchmark["date"].le(allowed_end), ["date", "close"]].rename(
            columns={"close": "benchmark_close"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("连续执行行情前缀存在缺失")
    audit = {
        "contiguous_post_cutoff_market_dates": [date.date().isoformat() for date in prefix],
        "first_market_alignment_gap": first_gap.date().isoformat() if first_gap is not None else None,
        "market_last_date": allowed_end.date().isoformat(),
    }
    return market.reset_index(drop=True), audit


def _evaluate_period(
    execution: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    *,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    perturbations: list[int],
) -> pd.DataFrame:
    selected = np.flatnonzero(execution["date"].between(start, end).to_numpy())
    if len(selected) == 0:
        return pd.DataFrame()
    last = int(selected[-1])
    initial_capital = float(config["account"]["initial_capital_cny"])
    base_costs, stress_costs, objective = frozen._cost_models(config)
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        if perturbation >= len(selected):
            continue
        first = int(selected[perturbation])
        if first == 0:
            continue
        sample = execution.iloc[first - 1 : last + 1][
            ["date", "etf_open", "etf_close", "benchmark_close"]
        ].reset_index(drop=True)
        states = np.ones(len(sample), dtype=np.int8)
        states[0] = 0
        states[1:] = np.where(
            execution.loc[first:last, "cash_execution"].to_numpy(bool), 0, 1
        )
        benchmark_ledger = build_benchmark_ledger(sample, initial_capital)
        base_ledger, base_trades = simulate_binary_path(
            sample,
            dividends,
            states,
            costs=base_costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        stress_ledger, stress_trades = simulate_binary_path(
            sample,
            dividends,
            states,
            costs=stress_costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        base_summary = summarize_path(
            base_ledger, base_trades, benchmark_ledger, objective=objective
        )
        stress_summary = summarize_path(
            stress_ledger, stress_trades, benchmark_ledger, objective=objective
        )
        row: dict[str, Any] = {
            "policy": POLICY,
            "period": period,
            "start_perturbation": perturbation,
            "first_execution_date": execution.loc[first, "date"],
            "last_execution_date": execution.loc[last, "date"],
            "execution_day_count": int(last - first + 1),
            "cash_day_count": int((states[1:] == 0).sum()),
            "cash_day_share": float((states[1:] == 0).mean()),
        }
        for prefix, summary in [("base", base_summary), ("stress", stress_summary)]:
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if metrics.empty:
        return output
    for period, group in metrics.groupby("period", sort=True):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(
            group["stress_rolling_242d_excess_median"], errors="coerce"
        )
        both = group["stress_both_20pct_gates"].fillna(False).astype(bool)
        output.append(
            {
                "period": period,
                "start_count": int(len(group)),
                "stress_annualized_excess_minimum": _safe_float(annual.min()),
                "stress_annualized_excess_median": _safe_float(annual.median()),
                "stress_rolling_242d_excess_minimum": _safe_float(
                    rolling.dropna().min()
                ),
                "every_available_start_both_20pct_gates": bool(both.all()),
                "start_pass_ratio": float(both.mean()),
            }
        )
    return output


def _daily_evidence(features: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict[str, Any]]:
    columns = [
        "date",
        "close_premium_to_nav",
        "micro_risk_vote_count",
        "cash_consensus3",
        "risk_short_inventory_drought",
        "risk_short_inventory_crowding",
        "cash_raw",
        "one_day_open_total_return",
        "raw_transition",
        "raw_shadow_net_contribution",
        "shadow242_net_sum",
        "shadow60_net_sum",
        "shadow_gate_active",
        "cash_final",
        "state_final",
    ]
    result: list[dict[str, Any]] = []
    for _, row in features.loc[features["date"].gt(cutoff), columns].iterrows():
        result.append(
            {
                column: (
                    row[column].date().isoformat()
                    if column == "date"
                    else _safe_float(row[column])
                    if isinstance(row[column], (float, np.floating))
                    else bool(row[column])
                    if isinstance(row[column], (bool, np.bool_))
                    else int(row[column])
                    if isinstance(row[column], (int, np.integer))
                    else None
                    if pd.isna(row[column])
                    else row[column]
                )
                for column in columns
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="运行双影子V1固定规则留出/前瞻重放")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    evaluated_at = datetime.now(TIMEZONE)
    run_id = arguments.run_id or evaluated_at.strftime("%Y%m%dT%H%M%S%z")
    output_dir = OUTPUT_ROOT / run_id
    if output_dir.exists():
        raise FileExistsError(f"重放目录已存在，拒绝覆盖：{output_dir}")

    validator_audit = _validate_frozen_validator()
    config, candidate_manifest, freeze_hash_mismatches = frozen._load_freeze()
    snapshot_dir = Path(arguments.snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    snapshot_manifest, snapshot = _load_snapshot(snapshot_dir.resolve())
    historical, dividends = _read_frozen_inputs(config)
    signal_cutoff = pd.Timestamp(config["dates"]["frozen_signal_cutoff"])
    market_cutoff = pd.Timestamp(config["dates"]["frozen_execution_market_cutoff"])

    combined = {
        "nav": _combine_at_cutoff(
            historical["nav"], snapshot["nav"], signal_cutoff, name="nav"
        ),
        "fund_share": _combine_at_cutoff(
            historical["fund_share"],
            snapshot["fund_share"],
            signal_cutoff,
            name="fund_share",
        ),
        "margin": _combine_at_cutoff(
            historical["margin"], snapshot["margin"], signal_cutoff, name="margin"
        ),
        "etf": _combine_at_cutoff(
            historical["etf"], snapshot["etf"], market_cutoff, name="etf"
        ),
        "benchmark": _combine_at_cutoff(
            historical["benchmark"],
            snapshot["benchmark"],
            market_cutoff,
            name="benchmark",
        ),
    }
    post_signal_dates, signal_audit = _contiguous_signal_calendar(
        combined["etf"],
        combined["nav"],
        combined["fund_share"],
        combined["margin"],
        signal_cutoff,
    )
    signal = _build_signal(config, combined, post_signal_dates)
    market, market_audit = _contiguous_market(
        combined["etf"], combined["benchmark"], market_cutoff
    )
    features = frozen._rebuild_features(signal, market, dividends, config)
    historical_prefix = features.loc[features["date"].le(signal_cutoff)].reset_index(
        drop=True
    )
    historical_replication = frozen._compare_features(historical_prefix)

    execution = market.merge(
        features[["date", "cash_final"]], on="date", how="left", validate="one_to_one"
    )
    execution["cash_execution"] = execution["cash_final"].shift(1).fillna(False)
    available_signal_end = features["date"].max()
    eligible_execution_dates = execution.loc[
        execution["date"].gt(available_signal_end), "date"
    ]
    if not eligible_execution_dates.empty:
        evaluation_end = eligible_execution_dates.iloc[0]
    else:
        evaluation_end = min(available_signal_end, execution["date"].max())
    execution = execution.loc[execution["date"].le(evaluation_end)].reset_index(drop=True)

    perturbations = [
        int(value) for value in config["objective"]["start_perturbations_trading_days"]
    ]
    metrics_parts = [
        _evaluate_period(
            execution,
            dividends,
            config,
            period="COMBINED_HISTORY_EXTENDED_FIXED_RULE",
            start=pd.Timestamp(config["dates"]["development_start"]),
            end=evaluation_end,
            perturbations=perturbations,
        ),
        _evaluate_period(
            execution,
            dividends,
            config,
            period="POST_CUTOFF_FIXED_RULE_HOLDOUT",
            start=signal_cutoff + pd.Timedelta(days=1),
            end=evaluation_end,
            perturbations=perturbations,
        ),
    ]
    metrics = pd.concat(
        [part for part in metrics_parts if not part.empty], ignore_index=True, sort=False
    )
    aggregates = _aggregate(metrics)
    combined_aggregate = next(
        (
            item
            for item in aggregates
            if item["period"] == "COMBINED_HISTORY_EXTENDED_FIXED_RULE"
        ),
        None,
    )
    holdout_aggregate = next(
        (
            item
            for item in aggregates
            if item["period"] == "POST_CUTOFF_FIXED_RULE_HOLDOUT"
        ),
        None,
    )
    new_features = features.loc[features["date"].gt(signal_cutoff)].copy()
    daily = _daily_evidence(features, signal_cutoff)
    prospective_start = pd.Timestamp(candidate_manifest["frozen_at_asia_shanghai"]).tz_localize(
        None
    ).normalize()
    prospective_rows = new_features.loc[new_features["date"].gt(prospective_start)]

    historical_ok = historical_replication["total_mismatches"] == 0
    new_rows_available = not new_features.empty
    enough_holdout = bool(
        holdout_aggregate
        and holdout_aggregate.get("stress_rolling_242d_excess_minimum") is not None
    )
    combined_pass = bool(
        combined_aggregate
        and combined_aggregate["start_count"] == len(perturbations)
        and combined_aggregate["every_available_start_both_20pct_gates"]
    )
    if not historical_ok:
        status = "FAILED_HISTORICAL_PREFIX_DRIFT"
    elif not new_rows_available:
        status = "PARTIAL_SUCCESS_NO_COMPLETE_POST_CUTOFF_SIGNAL"
    elif enough_holdout and not holdout_aggregate["every_available_start_both_20pct_gates"]:
        status = "FIXED_RULE_HOLDOUT_FAILED_NO_PARAMETER_RESCUE"
    elif enough_holdout:
        status = "FIXED_RULE_HOLDOUT_OBJECTIVE_PASS_FORWARD_CONTINUES"
    elif combined_pass:
        status = "FIXED_RULE_HOLDOUT_PARTIAL_SUCCESS_242D_PENDING"
    else:
        status = "FIXED_RULE_EXTENDED_HISTORY_BELOW_OBJECTIVE_FORWARD_CONTINUES"

    output_dir.mkdir(parents=True, exist_ok=False)
    features_path = output_dir / "daily_features_and_states.parquet"
    metrics_path = output_dir / "period_start_metrics.parquet"
    _atomic_parquet(features, features_path)
    _atomic_parquet(metrics, metrics_path)
    latest = daily[-1] if daily else None
    payload = {
        "project_id": PROJECT_ID,
        "candidate_policy": POLICY,
        "status": status,
        "evaluated_at_asia_shanghai": evaluated_at.isoformat(),
        "run_id": run_id,
        "snapshot": {
            "manifest": snapshot_manifest["manifest_file"],
            "manifest_sha256": snapshot_manifest["manifest_sha256"],
            "snapshot_status": snapshot_manifest["status"],
            "retrieved_at_asia_shanghai": snapshot_manifest[
                "retrieved_at_asia_shanghai"
            ],
        },
        "freeze": {
            "candidate_manifest_sha256": frozen._sha256(frozen.MANIFEST_FILE),
            "candidate_config_sha256": frozen._sha256(frozen.CONFIG_FILE),
            "freeze_hash_mismatches": freeze_hash_mismatches,
            "validator": validator_audit,
            "historical_prefix_replication": historical_replication,
        },
        "coverage": {
            "signal": signal_audit,
            "market": market_audit,
            "post_cutoff_signal_row_count": int(len(new_features)),
            "post_cutoff_first_signal_date": (
                new_features["date"].min().date().isoformat()
                if not new_features.empty
                else None
            ),
            "post_cutoff_last_signal_date": (
                new_features["date"].max().date().isoformat()
                if not new_features.empty
                else None
            ),
            "evaluation_last_execution_date": evaluation_end.date().isoformat(),
            "prospective_after_freeze_signal_row_count": int(len(prospective_rows)),
        },
        "fixed_rule_evidence": {
            "daily_post_cutoff": daily,
            "latest_research_state": latest,
            "post_cutoff_raw_cash_days": int(new_features["cash_raw"].sum()),
            "post_cutoff_final_cash_days": int(new_features["cash_final"].sum()),
            "post_cutoff_mature_target_rows": int(
                new_features["one_day_open_total_return"].notna().sum()
            ),
            "aggregates": aggregates,
            "combined_extended_all_five_starts_pass": combined_pass,
            "holdout_has_242d_rolling_window": enough_holdout,
        },
        "interpretation": {
            "post_cutoff_class": "冻结后首次读取的固定规则短留出；不等同于冻结时刻之后逐日形成的真正前瞻样本",
            "prospective_class": "仅日期严格晚于2026-08-28的合格信号行计入真正前瞻计数",
            "objective_status": "不足242个留出交易日时，不把短期年化外推当作20个百分点目标已获验证",
            "no_parameter_rescue": True,
        },
        "evidence": {
            "features": str(features_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "boundaries": {
            "research_candidate": True,
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    run_report = output_dir / "run_report.json"
    _atomic_json(payload, run_report)
    _atomic_json(payload, STATUS_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 2 if status.startswith("FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
