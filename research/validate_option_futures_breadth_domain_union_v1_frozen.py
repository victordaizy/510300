"""独立实现并验证已冻结的510300期权-期货-全A四域二元策略。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_loader
from binary_state_feasibility_v1 import (
    CostModel,
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "510300_option_futures_breadth_domain_union_v1.yaml"
VALIDATION_DATA_CUTOFF = pd.Timestamp("2025-12-31")
OPTION_STATE_START = pd.Timestamp("2020-07-01")
DEVELOPMENT_START = pd.Timestamp("2022-03-09")
DEVELOPMENT_END = pd.Timestamp("2023-12-29")
VALIDATION_START = pd.Timestamp("2024-01-02")
VALIDATION_END = pd.Timestamp("2024-12-31")
REPLICATION_START = pd.Timestamp("2025-01-02")
REPLICATION_END = pd.Timestamp("2025-12-31")
COMBINED_START = VALIDATION_START
COMBINED_END = REPLICATION_END
START_PERTURBATIONS = list(range(5))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _load_and_verify_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(CONFIG_FILE)
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / config["paths"]["manifest"]
    receipt_path = PROJECT_ROOT / config["paths"]["freeze_receipt"]
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("冻结清单或冻结回执缺失")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if _sha256(CONFIG_FILE) != manifest["config_sha256"]:
        raise RuntimeError("冻结配置哈希漂移")
    if _sha256(manifest_path) != receipt["manifest_sha256"]:
        raise RuntimeError("冻结清单哈希漂移")
    if receipt.get("status") != "FROZEN_SUCCESS":
        raise RuntimeError("冻结回执状态不是成功")
    evidence_paths = {
        "discovery_script": PROJECT_ROOT / config["development_evidence"]["discovery_script"],
        "discovery_report": PROJECT_ROOT / config["development_evidence"]["discovery_report"],
        "discovery_features": PROJECT_ROOT / config["development_evidence"]["discovery_features"],
        "discovery_metrics": PROJECT_ROOT / config["development_evidence"]["discovery_metrics"],
    }
    for name, path in evidence_paths.items():
        if not path.is_file() or _sha256(path) != manifest["development_evidence_sha256"][name]:
            raise RuntimeError(f"冻结开发证据哈希漂移：{name}")
    return config, manifest, receipt


def _read_filtered(path: Path, date_column: str, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[(date_column, "<=", VALIDATION_DATA_CUTOFF)],
    )
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
    if frame.empty or frame[date_column].max() > VALIDATION_DATA_CUTOFF:
        raise ValueError(f"验证输入为空或越过截止日：{path}")
    return frame


def _rolling_percentile(values: pd.Series, window: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def rank_last(array: np.ndarray) -> float:
        valid = array[np.isfinite(array)]
        if len(valid) < minimum:
            return float("nan")
        return float(np.count_nonzero(valid <= valid[-1]) / len(valid))

    return numeric.rolling(window, min_periods=minimum).apply(rank_last, raw=True)


def _build_option_gamma_rank(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    eod_path = PROJECT_ROOT / config["inputs"]["option_eod"]["path"]
    risk_path = PROJECT_ROOT / config["inputs"]["option_risk"]["path"]
    stats_path = PROJECT_ROOT / config["inputs"]["option_statistics"]["path"]
    eod_columns = [
        "trade_date",
        "contract_code",
        "option_type",
        "expiry_date",
        "contract_unit",
        "is_adjusted",
        "volume",
        "open_interest",
        "underlying_close",
    ]
    risk_columns = [
        "trade_date",
        "contract_code",
        "option_type",
        "delta",
        "gamma",
        "implied_volatility",
    ]
    eod = _read_filtered(eod_path, "trade_date", eod_columns)
    risk = _read_filtered(risk_path, "trade_date", risk_columns)
    statistics = _read_filtered(stats_path, "trade_date", ["trade_date"])
    eod["expiry_date"] = pd.to_datetime(eod["expiry_date"], errors="coerce").dt.normalize()
    eod["dte"] = (eod["expiry_date"] - eod["trade_date"]).dt.days
    panel = eod.merge(
        risk,
        on=["trade_date", "contract_code", "option_type"],
        how="inner",
        validate="one_to_one",
    )
    numeric_columns = [
        "contract_unit",
        "volume",
        "open_interest",
        "underlying_close",
        "delta",
        "gamma",
        "implied_volatility",
    ]
    panel[numeric_columns] = panel[numeric_columns].apply(pd.to_numeric, errors="coerce")
    rule = config["option_gamma_domain"]["contract_filter"]
    eligible = panel.loc[
        ~panel["is_adjusted"].astype(bool)
        & panel["dte"].between(
            int(rule["minimum_calendar_days_to_expiry"]),
            int(rule["maximum_calendar_days_to_expiry"]),
        )
        & panel["contract_unit"].gt(0)
        & panel["volume"].ge(0)
        & panel["open_interest"].ge(0)
        & panel[numeric_columns].notna().all(axis=1)
    ].copy()
    option_sign = np.where(eligible["option_type"].eq("C"), 1.0, -1.0)
    eligible["gamma_weight"] = (
        eligible["gamma"].abs()
        * eligible["open_interest"]
        * eligible["contract_unit"]
        * eligible["underlying_close"].pow(2)
    )
    eligible["signed_gamma_weight"] = option_sign * eligible["gamma_weight"]
    daily = eligible.groupby("trade_date", sort=True).agg(
        gamma_weight=("gamma_weight", "sum"),
        signed_gamma_weight=("signed_gamma_weight", "sum"),
        eligible_contract_count=("contract_code", "size"),
    )
    daily["gamma_balance"] = daily["signed_gamma_weight"] / daily["gamma_weight"].replace(
        0.0, np.nan
    )
    calendar = pd.DataFrame(
        {"date": sorted(set(statistics["trade_date"]).union(set(daily.index)))}
    )
    daily = daily.reset_index().rename(columns={"trade_date": "date"})
    result = calendar.merge(daily, on="date", how="left", validate="one_to_one")
    result["rank252_gamma_balance"] = _rolling_percentile(result["gamma_balance"])
    audit = {
        "option_eod_rows": int(len(eod)),
        "option_risk_rows": int(len(risk)),
        "eligible_contract_rows": int(len(eligible)),
        "option_calendar_rows": int(len(result)),
        "option_first_date": result["date"].min().date().isoformat(),
        "option_last_date": result["date"].max().date().isoformat(),
    }
    return result[["date", "rank252_gamma_balance"]], audit


def _build_common_domains(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    futures_path = PROJECT_ROOT / config["inputs"]["if0_futures"]["path"]
    spot_path = PROJECT_ROOT / config["inputs"]["csi300_spot"]["path"]
    stats_path = PROJECT_ROOT / config["inputs"]["option_statistics"]["path"]
    all_a_path = PROJECT_ROOT / config["inputs"]["all_a_features"]["path"]
    futures = _read_filtered(
        futures_path,
        "date",
        ["date", "close", "volume", "open_interest"],
    )
    spot = _read_filtered(spot_path, "date", ["date", "close"])
    statistics = _read_filtered(
        stats_path,
        "trade_date",
        ["trade_date", "call_volume", "put_volume"],
    )
    all_a = _read_filtered(all_a_path, "date", ["date", "top10pct_amount_share"])
    common = futures.rename(
        columns={
            "close": "futures_close",
            "volume": "futures_volume",
            "open_interest": "futures_open_interest",
        }
    )
    common = common.merge(
        spot.rename(columns={"close": "spot_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    common = common.merge(
        statistics.rename(columns={"trade_date": "date"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    common.sort_values("date", inplace=True)
    common.reset_index(drop=True, inplace=True)
    common["basis"] = np.log(common["futures_close"] / common["spot_close"])
    common["basis_z60"] = (
        common["basis"] - common["basis"].rolling(60, min_periods=60).mean()
    ) / common["basis"].rolling(60, min_periods=60).std(ddof=1)
    common["basis_change5"] = common["basis"] - common["basis"].shift(5)
    common["futures_open_interest_log_change5"] = np.log(
        common["futures_open_interest"] / common["futures_open_interest"].shift(5)
    )
    common["option_log_put_call_volume"] = np.log(
        (common["put_volume"] + 1.0) / (common["call_volume"] + 1.0)
    )
    common = common.merge(all_a, on="date", how="inner", validate="one_to_one")
    common.sort_values("date", inplace=True)
    common.reset_index(drop=True, inplace=True)
    rank_sources = [
        "top10pct_amount_share",
        "basis_z60",
        "basis_change5",
        "futures_open_interest_log_change5",
        "option_log_put_call_volume",
    ]
    for column in rank_sources:
        common[f"rank252_{column}"] = _rolling_percentile(common[column])

    common["domain_breadth"] = common["rank252_top10pct_amount_share"].le(0.20)
    common["domain_basis"] = common["rank252_basis_z60"].le(0.10) | common[
        "rank252_basis_change5"
    ].le(0.05)
    common["domain_futures_oi"] = common[
        "rank252_futures_open_interest_log_change5"
    ].le(0.15)
    common["domain_option_put_call_volume"] = common[
        "rank252_option_log_put_call_volume"
    ].le(0.10)
    domain_columns = [
        "domain_breadth",
        "domain_basis",
        "domain_futures_oi",
        "domain_option_put_call_volume",
    ]
    common["orthogonal_domain_vote_count"] = common[domain_columns].sum(axis=1)
    audit = {
        "futures_rows": int(len(futures)),
        "spot_rows": int(len(spot)),
        "statistics_rows": int(len(statistics)),
        "all_a_rows": int(len(all_a)),
        "common_rows": int(len(common)),
        "common_first_date": common["date"].min().date().isoformat(),
        "common_last_date": common["date"].max().date().isoformat(),
    }
    return common[["date", *domain_columns, "orthogonal_domain_vote_count"]], audit


def _cost_models(config: dict[str, Any]) -> tuple[CostModel, CostModel, dict[str, Any]]:
    account = config["account"]
    costs = config["costs"]
    common = {
        "commission_rate": float(costs["commission_rate_per_leg"]),
        "minimum_commission": float(costs["minimum_commission_cny_per_leg"]),
        "cash_annual_rate": float(account["cash_annual_rate"]),
        "trading_days_per_year": int(account["trading_days_per_year"]),
        "lot_size": int(account["lot_size_shares"]),
    }
    base = CostModel(slippage_bps=float(costs["base_slippage_bps_per_leg"]), **common)
    stress = CostModel(slippage_bps=float(costs["stress_slippage_bps_per_leg"]), **common)
    objective = {
        "annualization_trading_days": int(account["trading_days_per_year"]),
        "rolling_window_trading_days": int(config["objective"]["rolling_window_trading_days"]),
        "minimum_annualized_excess": float(config["objective"]["minimum_annualized_net_excess"]),
        "minimum_rolling_excess_median": float(
            config["objective"]["minimum_rolling_242d_excess_median"]
        ),
    }
    return base, stress, objective


def _build_state_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    option, option_audit = _build_option_gamma_rank(config)
    domains, domain_audit = _build_common_domains(config)
    market_loader.DEVELOPMENT_CUTOFF = VALIDATION_DATA_CUTOFF
    dividends = market_loader._load_dividends()
    market = market_loader._load_market(dividends).copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[market["date"].le(VALIDATION_DATA_CUTOFF)].sort_values("date")
    frame = market.merge(option, on="date", how="left", validate="one_to_one")
    frame = frame.merge(domains, on="date", how="left", validate="one_to_one").reset_index(drop=True)

    frame["option_cash_state"] = False
    option_indices = np.flatnonzero(frame["date"].ge(OPTION_STATE_START).to_numpy())
    previous_state = 1
    for execution_index in option_indices:
        if execution_index <= 0:
            raise ValueError("期权状态机缺少前一信号日")
        signal_index = execution_index - 1
        raw_score = frame.loc[signal_index, "rank252_gamma_balance"]
        finite = bool(pd.notna(raw_score) and np.isfinite(float(raw_score)))
        score = float(raw_score) if finite else float("nan")
        if previous_state == 1:
            state = 0 if finite and score >= 0.90 else 1
        else:
            state = 1 if finite and score < 0.50 else 0
        frame.loc[execution_index, "option_cash_state"] = state == 0
        previous_state = state

    frame["final_state"] = 1
    frame["signal_domain_vote_count"] = 0
    for execution_index in range(1, len(frame)):
        signal_index = execution_index - 1
        raw_votes = frame.loc[signal_index, "orthogonal_domain_vote_count"]
        votes = 0 if pd.isna(raw_votes) else int(raw_votes)
        frame.loc[execution_index, "signal_domain_vote_count"] = votes
        cash = bool(frame.loc[execution_index, "option_cash_state"] or votes >= 2)
        frame.loc[execution_index, "final_state"] = 0 if cash else 1
    frame["final_state"] = frame["final_state"].astype(np.int8)
    frame["signal_domain_vote_count"] = frame["signal_domain_vote_count"].astype(np.int8)
    audit = {
        "market_rows": int(len(frame)),
        "market_first_date": frame["date"].min().date().isoformat(),
        "market_last_date": frame["date"].max().date().isoformat(),
        "option": option_audit,
        "domains": domain_audit,
        "post_2025_rows_read": 0,
    }
    return frame, dividends, audit


def _compare_development_states(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    frozen_path = PROJECT_ROOT / config["development_evidence"]["discovery_features"]
    frozen = pd.read_parquet(frozen_path)
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    current = frame.loc[
        frame["date"].between(DEVELOPMENT_START, DEVELOPMENT_END),
        [
            "date",
            "option_cash_state",
            "domain_breadth",
            "domain_basis",
            "domain_futures_oi",
            "domain_option_put_call_volume",
            "orthogonal_domain_vote_count",
        ],
    ].copy()
    frozen = frozen.loc[
        frozen["date"].between(DEVELOPMENT_START, DEVELOPMENT_END),
        [
            "date",
            "option_hysteresis_cash_state",
            "domain_breadth",
            "domain_basis",
            "domain_futures_oi",
            "domain_option_put_call_volume",
            "orthogonal_domain_vote_count",
        ],
    ].rename(columns={"option_hysteresis_cash_state": "option_cash_state"})
    merged = current.merge(frozen, on="date", suffixes=("_current", "_frozen"), validate="one_to_one")
    mismatch: dict[str, int] = {}
    for column in [
        "option_cash_state",
        "domain_breadth",
        "domain_basis",
        "domain_futures_oi",
        "domain_option_put_call_volume",
        "orthogonal_domain_vote_count",
    ]:
        mismatch[column] = int(
            merged[f"{column}_current"].fillna(False).ne(
                merged[f"{column}_frozen"].fillna(False)
            ).sum()
        )
    status = "PASS" if len(current) == len(frozen) == len(merged) and sum(mismatch.values()) == 0 else "FAIL"
    return {
        "status": status,
        "current_rows": int(len(current)),
        "frozen_rows": int(len(frozen)),
        "matched_rows": int(len(merged)),
        "mismatch_by_field": mismatch,
    }


def _evaluate_period(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    base_costs, stress_costs, objective = _cost_models(config)
    initial_capital = float(config["account"]["initial_capital_cny"])
    selection_indices = np.flatnonzero(frame["date"].between(start, end).to_numpy())
    if len(selection_indices) < 200:
        raise ValueError(f"验证区间交易日不足：{period_name}")
    rows: list[dict[str, Any]] = []
    for perturbation in START_PERTURBATIONS:
        first_execution_index = int(selection_indices[perturbation])
        last_execution_index = int(selection_indices[-1])
        anchor_index = first_execution_index - 1
        sample = frame.iloc[anchor_index : last_execution_index + 1][
            ["date", "etf_open", "etf_close", "benchmark_close"]
        ].reset_index(drop=True)
        states = np.zeros(len(sample), dtype=np.int8)
        states[1:] = frame.loc[
            first_execution_index:last_execution_index,
            "final_state",
        ].to_numpy(dtype=np.int8)
        benchmark = build_benchmark_ledger(sample, initial_capital)
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
        base = summarize_path(base_ledger, base_trades, benchmark, objective=objective)
        stress = summarize_path(stress_ledger, stress_trades, benchmark, objective=objective)
        execution_states = states[1:]
        row: dict[str, Any] = {
            "period": period_name,
            "start_perturbation": perturbation,
            "first_execution_date": frame.loc[first_execution_index, "date"],
            "last_execution_date": frame.loc[last_execution_index, "date"],
            "execution_day_count": int(len(execution_states)),
            "cash_day_count": int((execution_states == 0).sum()),
            "cash_day_share": float((execution_states == 0).mean()),
        }
        for prefix, summary in [("base", base), ("stress", stress)]:
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_period(metrics: pd.DataFrame) -> dict[str, Any]:
    annual = pd.to_numeric(metrics["stress_annualized_excess"], errors="coerce")
    rolling = pd.to_numeric(metrics["stress_rolling_242d_excess_median"], errors="coerce")
    base_annual = pd.to_numeric(metrics["base_annualized_excess"], errors="coerce")
    base_rolling = pd.to_numeric(metrics["base_rolling_242d_excess_median"], errors="coerce")
    every_pass = bool(
        len(metrics) == len(START_PERTURBATIONS)
        and metrics["stress_both_20pct_gates"].astype(bool).all()
    )
    return {
        "period": str(metrics.iloc[0]["period"]),
        "perturbation_count": int(len(metrics)),
        "base_annualized_excess_minimum": _safe_float(base_annual.min()),
        "base_annualized_excess_median": _safe_float(base_annual.median()),
        "base_rolling_242d_excess_minimum": _safe_float(base_rolling.min()),
        "stress_annualized_excess_minimum": _safe_float(annual.min()),
        "stress_annualized_excess_median": _safe_float(annual.median()),
        "stress_annualized_excess_maximum": _safe_float(annual.max()),
        "stress_rolling_242d_excess_minimum": _safe_float(rolling.min()),
        "stress_rolling_242d_excess_median": _safe_float(rolling.median()),
        "cash_day_share_median": float(metrics["cash_day_share"].median()),
        "stress_trade_leg_count_median": _safe_float(metrics["stress_trade_leg_count"].median()),
        "stress_perturbation_pass_ratio": float(metrics["stress_both_20pct_gates"].mean()),
        "every_start_both_gates": every_pass,
        "rolling_metric_available_count": int(rolling.notna().sum()),
    }


def _development_reproduction_check(
    aggregate: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    frozen = receipt["candidate"]
    fields = [
        "stress_annualized_excess_minimum",
        "stress_annualized_excess_median",
        "stress_annualized_excess_maximum",
        "stress_rolling_242d_excess_minimum",
        "stress_rolling_242d_excess_median",
        "cash_day_share_median",
        "stress_trade_leg_count_median",
        "stress_perturbation_pass_ratio",
    ]
    differences = {
        field: abs(float(aggregate[field]) - float(frozen[field])) for field in fields
    }
    maximum = max(differences.values())
    return {
        "status": "PASS" if maximum <= 1e-12 else "FAIL",
        "maximum_absolute_difference": maximum,
        "absolute_difference_by_field": differences,
    }


def main() -> int:
    config, manifest, receipt = _load_and_verify_protocol()
    frame, dividends, input_audit = _build_state_frame(config)
    state_drift = _compare_development_states(frame, config)
    if state_drift["status"] != "PASS":
        raise RuntimeError(f"独立实现与冻结开发状态不一致：{state_drift}")

    development_metrics = _evaluate_period(
        frame,
        dividends,
        config,
        "DEVELOPMENT_REPRODUCTION_2022_2023",
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    development_aggregate = _aggregate_period(development_metrics)
    reproduction = _development_reproduction_check(development_aggregate, receipt)
    if reproduction["status"] != "PASS":
        raise RuntimeError(f"独立实现未能重现冻结开发结果：{reproduction}")

    validation_2024 = _evaluate_period(
        frame,
        dividends,
        config,
        "FIXED_RULE_2024",
        VALIDATION_START,
        VALIDATION_END,
    )
    replication_2025 = _evaluate_period(
        frame,
        dividends,
        config,
        "FIXED_RULE_2025",
        REPLICATION_START,
        REPLICATION_END,
    )
    combined = _evaluate_period(
        frame,
        dividends,
        config,
        "FIXED_RULE_COMBINED_2024_2025",
        COMBINED_START,
        COMBINED_END,
    )
    validation_2024_aggregate = _aggregate_period(validation_2024)
    replication_2025_aggregate = _aggregate_period(replication_2025)
    combined_aggregate = _aggregate_period(combined)
    passed = bool(combined_aggregate["every_start_both_gates"])
    status = (
        "FIXED_RULE_VALIDATION_PASSED_FORWARD_SHADOW_REQUIRED"
        if passed
        else "FIXED_RULE_VALIDATION_REJECTED_NO_PARAMETER_RESCUE"
    )

    output_metrics = pd.concat(
        [development_metrics, validation_2024, replication_2025, combined],
        ignore_index=True,
    )
    metrics_path = PROJECT_ROOT / config["paths"]["validation_metrics"]
    states_path = PROJECT_ROOT / config["paths"]["validation_states"]
    report_path = PROJECT_ROOT / config["paths"]["validation_report"]
    state_columns = [
        "date",
        "rank252_gamma_balance",
        "option_cash_state",
        "domain_breadth",
        "domain_basis",
        "domain_futures_oi",
        "domain_option_put_call_volume",
        "orthogonal_domain_vote_count",
        "signal_domain_vote_count",
        "final_state",
    ]
    _atomic_parquet(output_metrics, metrics_path)
    _atomic_parquet(frame[state_columns], states_path)

    input_paths = {
        name: PROJECT_ROOT / item["path"]
        for name, item in config["inputs"].items()
        if "path" in item
    }
    payload = {
        "status": status,
        "project_id": config["protocol"]["project_id"],
        "candidate_name": config["protocol"]["candidate_name"],
        "frozen_protocol": {
            "config_sha256": manifest["config_sha256"],
            "manifest_sha256": receipt["manifest_sha256"],
            "freeze_timestamp": receipt["frozen_at"],
            "validation_implementation_sha256": _sha256(Path(__file__)),
        },
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "information_only_assets": ["510300期权", "IF0股指期货", "全A横截面"],
        },
        "independent_implementation_checks": {
            "development_state_drift": state_drift,
            "development_metric_reproduction": reproduction,
        },
        "development_reproduction": development_aggregate,
        "fixed_rule_2024": validation_2024_aggregate,
        "fixed_rule_2025": replication_2025_aggregate,
        "primary_fixed_rule_combined_2024_2025": combined_aggregate,
        "decision": {
            "goal_gate_passed": passed,
            "next_action": (
                "冻结为Paper/Shadow候选并开始真正前向验证"
                if passed
                else "永久淘汰该固定候选，禁止用2024-2025结果调整参数"
            ),
            "historical_evidence_limit": config["protocol"]["historical_evidence_limit"],
        },
        "input_audit": input_audit,
        "input_sha256": {
            name: _sha256(path) for name, path in input_paths.items() if path.is_file()
        },
        "cost_models": {
            "base": asdict(_cost_models(config)[0]),
            "stress": asdict(_cost_models(config)[1]),
        },
        "artifacts": {
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "states": str(states_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "boundaries": {
            "data_ceiling": VALIDATION_DATA_CUTOFF.date().isoformat(),
            "post_2025_rows_read": 0,
            "formula_or_threshold_changes_after_freeze": 0,
            "position_mapping": "DISABLED",
            "orders": "DISABLED",
            "broker": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    _atomic_json(payload, report_path)
    print(
        json.dumps(
            {
                "status": status,
                "development_checks": payload["independent_implementation_checks"],
                "fixed_rule_2024": validation_2024_aggregate,
                "fixed_rule_2025": replication_2025_aggregate,
                "primary_fixed_rule_combined_2024_2025": combined_aggregate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
