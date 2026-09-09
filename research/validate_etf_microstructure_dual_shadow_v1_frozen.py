"""独立复算已冻结的510300 ETF微观结构双影子门候选。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_contract
from binary_state_feasibility_v1 import (
    CostModel,
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
POLICY = "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_DUAL_SHADOW"
CONFIG_FILE = PROJECT_ROOT / "config" / "510300_etf_microstructure_dual_shadow_v1.yaml"
MANIFEST_FILE = (
    PROJECT_ROOT / "config" / "510300_etf_microstructure_dual_shadow_v1_manifest.json"
)
FROZEN_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "510300_etf_microstructure_shadow_gate_discovery_v5.parquet"
)
FROZEN_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_etf_microstructure_shadow_gate_discovery_v5"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "validation" / "510300_etf_microstructure_dual_shadow_v1"
)
OUTPUT_FEATURES = OUTPUT_DIR / "replicated_daily_features.parquet"
OUTPUT_METRICS = OUTPUT_DIR / "replicated_start_perturbation_metrics.parquet"
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "510300_etf_microstructure_dual_shadow_v1_replication.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if not CONFIG_FILE.exists() or not MANIFEST_FILE.exists():
        raise FileNotFoundError("冻结配置或清单不存在")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("冻结配置项目编号不匹配")
    if manifest.get("project_id") != PROJECT_ID or not manifest.get(
        "implementation_frozen", False
    ):
        raise ValueError("冻结清单无效")
    mismatches: dict[str, str] = {}
    for group in ["tracked_files", "historical_input_files_at_freeze"]:
        for relative, expected in manifest[group].items():
            path = PROJECT_ROOT / relative
            actual = _sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = f"expected={expected},actual={actual}"
    if mismatches:
        raise ValueError(f"冻结文件哈希漂移：{mismatches}")
    return config, manifest, mismatches


def _normalize(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ValueError(f"{column}存在重复日期")
    return result


def _load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    signal_cutoff = pd.Timestamp(config["dates"]["frozen_signal_cutoff"])
    market_cutoff = pd.Timestamp(config["dates"]["frozen_execution_market_cutoff"])
    paths = {
        name: PROJECT_ROOT / item["path"]
        for name, item in config["inputs"].items()
    }
    nav = _normalize(pd.read_parquet(paths["nav_daily"]))
    shares = _normalize(pd.read_parquet(paths["fund_share_daily"]))
    margin = _normalize(pd.read_parquet(paths["margin_detail_daily"]))
    etf = _normalize(pd.read_parquet(paths["etf_execution"]))
    benchmark = _normalize(pd.read_parquet(paths["benchmark_total_return"]))
    dividends = pd.read_csv(paths["cash_distributions"])
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    start = pd.Timestamp(config["dates"]["signal_history_start"])
    calendar = etf.loc[etf["date"].between(start, signal_cutoff), ["date"]].copy()
    signal = (
        calendar.merge(
            nav[["date", "close_premium_to_nav"]], on="date", how="left", validate="one_to_one"
        )
        .merge(shares[["date", "fund_shares"]], on="date", how="left", validate="one_to_one")
        .merge(
            margin[["date", "rzye", "rqye", "rqyl", "rzrqye"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    )
    value_columns = [
        "close_premium_to_nav",
        "fund_shares",
        "rzye",
        "rqye",
        "rqyl",
        "rzrqye",
    ]
    if signal[value_columns].isna().any(axis=None):
        raise ValueError("冻结交易日信号输入存在缺失")
    market = etf.loc[etf["date"].le(market_cutoff), ["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        benchmark.loc[benchmark["date"].le(market_cutoff), ["date", "close"]].rename(
            columns={"close": "benchmark_close"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("冻结执行行情与基准对齐失败")
    audit = {
        "signal_rows": int(len(signal)),
        "signal_first_date": signal["date"].min().date().isoformat(),
        "signal_last_date": signal["date"].max().date().isoformat(),
        "market_rows": int(len(market)),
        "market_last_date": market["date"].max().date().isoformat(),
        "post_signal_cutoff_rows_read": 0,
        "post_market_cutoff_rows_read": 0,
    }
    return signal, market, dividends, audit


def _rolling_percentile(
    series: pd.Series, window: int, minimum: int
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def rank(values: np.ndarray) -> float:
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        finite = values[np.isfinite(values)]
        if len(finite) < minimum:
            return np.nan
        return float(np.count_nonzero(finite <= current) / len(finite))

    return numeric.rolling(window, min_periods=minimum).apply(rank, raw=True)


def _state_machine(
    rank: pd.Series,
    *,
    low_entry: float | None = None,
    low_recovery: float | None = None,
    high_entry: float | None = None,
    high_recovery: float | None = None,
) -> pd.Series:
    low_mode = low_entry is not None
    if low_mode == (high_entry is not None):
        raise ValueError("状态机必须且只能指定一个方向")
    state = False
    output: list[bool] = []
    for value in pd.to_numeric(rank, errors="coerce"):
        if pd.notna(value):
            number = float(value)
            if not state:
                state = number <= float(low_entry) if low_mode else number >= float(high_entry)
            else:
                recovered = (
                    number >= float(low_recovery)
                    if low_mode
                    else number <= float(high_recovery)
                )
                if recovered:
                    state = False
        output.append(state)
    return pd.Series(output, index=rank.index, dtype=bool)


def _open_target(
    signal: pd.DataFrame, market: pd.DataFrame, dividends: pd.DataFrame
) -> pd.Series:
    price = market[["date", "etf_open"]].copy()
    dividend_map = dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    next_open = price["etf_open"].shift(-1)
    following_open = price["etf_open"].shift(-2)
    following_date = price["date"].shift(-2)
    following_dividend = following_date.map(
        lambda date: float(dividend_map.get(date, 0.0)) if pd.notna(date) else 0.0
    )
    price["target"] = (following_open + following_dividend) / next_open - 1.0
    return signal[["date"]].merge(
        price[["date", "target"]], on="date", how="left", validate="one_to_one"
    )["target"]


def _rebuild_features(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = signal.copy()
    percentile = config["rolling_percentile"]
    window = int(percentile["window_trading_rows"])
    minimum = int(percentile["minimum_valid_rows"])
    premium = pd.to_numeric(frame["close_premium_to_nav"], errors="raise")
    frame["premium_std20"] = premium.rolling(20, min_periods=20).std(ddof=0)
    frame["premium_range20"] = premium.rolling(20, min_periods=20).max() - premium.rolling(
        20, min_periods=20
    ).min()
    frame["premium_std10"] = premium.rolling(10, min_periods=10).std(ddof=0)
    for source, horizon, output in [
        ("fund_shares", 20, "fund_shares_log_change20"),
        ("rzye", 40, "rzye_log_change40"),
        ("rzrqye", 3, "rzrqye_log_change3"),
        ("rqye", 20, "rqye_log_change20"),
    ]:
        logged = np.log(pd.to_numeric(frame[source], errors="raise"))
        frame[output] = logged - logged.shift(horizon)
    frame["short_inventory_scale20"] = (
        pd.to_numeric(frame["rqyl"], errors="raise").rolling(20, min_periods=20).sum()
        / pd.to_numeric(frame["rzrqye"], errors="raise").replace(0.0, np.nan)
    )
    sources = [
        "premium_std20",
        "premium_range20",
        "premium_std10",
        "fund_shares_log_change20",
        "rzye_log_change40",
        "rzrqye_log_change3",
        "rqye_log_change20",
        "short_inventory_scale20",
    ]
    for source in sources:
        frame[f"rank252_{source}"] = _rolling_percentile(frame[source], window, minimum)

    frame["risk_premium_std20"] = _state_machine(
        frame["rank252_premium_std20"], low_entry=0.10, low_recovery=0.30
    )
    frame["risk_premium_range20"] = _state_machine(
        frame["rank252_premium_range20"], low_entry=0.15, low_recovery=0.30
    )
    frame["risk_premium_std10"] = _state_machine(
        frame["rank252_premium_std10"], low_entry=0.20, low_recovery=0.60
    )
    frame["risk_fund_share_contraction"] = _state_machine(
        frame["rank252_fund_shares_log_change20"], low_entry=0.15, low_recovery=0.70
    )
    frame["risk_financing_expansion40"] = _state_machine(
        frame["rank252_rzye_log_change40"], high_entry=0.85, high_recovery=0.50
    )
    frame["risk_total_margin_shock3"] = _state_machine(
        frame["rank252_rzrqye_log_change3"], low_entry=0.02, low_recovery=0.95
    )
    frame["risk_short_balance_contraction20"] = frame[
        "rank252_rqye_log_change20"
    ].le(0.08)
    frame["risk_short_inventory_drought"] = frame[
        "rank252_short_inventory_scale20"
    ].le(0.01)
    frame["risk_short_inventory_crowding"] = _state_machine(
        frame["rank252_short_inventory_scale20"], high_entry=0.95, high_recovery=0.70
    )
    votes = [
        "risk_premium_std20",
        "risk_premium_range20",
        "risk_premium_std10",
        "risk_fund_share_contraction",
        "risk_financing_expansion40",
        "risk_total_margin_shock3",
        "risk_short_balance_contraction20",
    ]
    frame["micro_risk_vote_count"] = frame[votes].sum(axis=1).astype(np.int8)
    frame["cash_consensus3"] = frame["micro_risk_vote_count"].ge(
        int(config["micro_consensus"]["required_votes_for_cash"])
    )
    frame["cash_raw"] = (
        frame["cash_consensus3"]
        | frame["risk_short_inventory_drought"]
        | frame["risk_short_inventory_crowding"]
    )
    frame["one_day_open_total_return"] = _open_target(frame, market, dividends)
    frame["raw_transition"] = frame["cash_raw"].ne(
        frame["cash_raw"].shift(1).fillna(False)
    )
    gate = config["dual_shadow_gate"]
    cash_daily = float(config["account"]["cash_annual_rate"]) / int(
        config["account"]["trading_days_per_year"]
    )
    frame["raw_shadow_net_contribution"] = np.where(
        frame["cash_raw"] & frame["one_day_open_total_return"].notna(),
        cash_daily - frame["one_day_open_total_return"],
        0.0,
    ) - frame["raw_transition"].astype(float) * float(
        gate["stress_cost_per_raw_transition"]
    )
    matured = frame["raw_shadow_net_contribution"].shift(
        int(gate["maturity_lag_signal_rows"])
    )
    long_spec = gate["long_window"]
    short_spec = gate["short_window"]
    frame["shadow242_net_sum"] = matured.rolling(
        int(long_spec["matured_signal_rows"]),
        min_periods=int(long_spec["minimum_matured_rows"]),
    ).sum()
    frame["shadow60_net_sum"] = matured.rolling(
        int(short_spec["matured_signal_rows"]),
        min_periods=int(short_spec["minimum_matured_rows"]),
    ).sum()
    annual_gate = frame["shadow242_net_sum"].gt(
        float(long_spec["active_when_net_sum_above"])
    ) | frame["shadow242_net_sum"].isna()
    short_gate = frame["shadow60_net_sum"].gt(
        float(short_spec["active_when_net_sum_above"])
    ) | frame["shadow60_net_sum"].isna()
    frame["shadow_gate_active"] = annual_gate & short_gate
    frame["cash_final"] = frame["cash_raw"] & frame["shadow_gate_active"]
    frame["state_final"] = np.where(frame["cash_final"], 0, 1).astype(np.int8)
    return frame


def _compare_features(rebuilt: pd.DataFrame) -> dict[str, Any]:
    frozen = pd.read_parquet(FROZEN_FEATURES)
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    if list(frozen.columns) != list(rebuilt[frozen.columns].columns):
        raise ValueError("冻结特征字段顺序无法复算")
    current = rebuilt[frozen.columns].copy()
    if not frozen["date"].equals(current["date"]):
        raise ValueError("冻结特征日期无法复算")
    mismatches: dict[str, int] = {}
    maximum_numeric_difference = 0.0
    for column in frozen.columns:
        if column == "date":
            continue
        left = frozen[column]
        right = current[column]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_integer_dtype(left):
            count = int((left.fillna(False) != right.fillna(False)).sum())
        else:
            left_number = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_number = pd.to_numeric(right, errors="coerce").to_numpy(float)
            equal = np.isclose(left_number, right_number, rtol=1e-12, atol=1e-12, equal_nan=True)
            count = int((~equal).sum())
            finite = np.isfinite(left_number) & np.isfinite(right_number)
            if finite.any():
                maximum_numeric_difference = max(
                    maximum_numeric_difference,
                    float(np.max(np.abs(left_number[finite] - right_number[finite]))),
                )
        mismatches[column] = count
    return {
        "rows": int(len(frozen)),
        "columns": int(len(frozen.columns)),
        "mismatch_by_column": mismatches,
        "total_mismatches": int(sum(mismatches.values())),
        "maximum_numeric_difference": maximum_numeric_difference,
    }


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


def _evaluate(
    features: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    execution = market.merge(
        features[["date", "cash_final"]], on="date", how="left", validate="one_to_one"
    )
    execution["cash_execution"] = execution["cash_final"].shift(1).fillna(False)
    base_costs, stress_costs, objective = _cost_models(config)
    initial_capital = float(config["account"]["initial_capital_cny"])
    periods = {
        "DEVELOPMENT_2022_2025": (
            pd.Timestamp(config["dates"]["development_start"]),
            pd.Timestamp(config["dates"]["development_end"]),
        ),
        "TEMPORAL_REPLICATION_2026_YTD": (
            pd.Timestamp(config["dates"]["temporal_replication_start"]),
            pd.Timestamp(config["dates"]["temporal_replication_end"]),
        ),
        "COMBINED_HISTORY_TO_2026_08_14": (
            pd.Timestamp(config["dates"]["development_start"]),
            pd.Timestamp(config["dates"]["temporal_replication_end"]),
        ),
    }
    perturbations = [int(value) for value in config["objective"]["start_perturbations_trading_days"]]
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        selected = np.flatnonzero(execution["date"].between(start, end).to_numpy())
        last = int(selected[-1])
        for perturbation in perturbations:
            first = int(selected[perturbation])
            sample = execution.iloc[first - 1 : last + 1][
                ["date", "etf_open", "etf_close", "benchmark_close"]
            ].reset_index(drop=True)
            states = np.ones(len(sample), dtype=np.int8)
            states[0] = 0
            states[1:] = np.where(
                execution.loc[first:last, "cash_execution"].to_numpy(bool), 0, 1
            )
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
            for prefix, summary in [("base", base), ("stress", stress)]:
                for key, value in summary.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _compare_metrics(rebuilt: pd.DataFrame) -> dict[str, Any]:
    frozen = pd.read_parquet(FROZEN_METRICS)
    frozen = frozen.loc[frozen["policy"].eq(POLICY)].sort_values(
        ["period", "start_perturbation"]
    ).reset_index(drop=True)
    current = rebuilt.sort_values(["period", "start_perturbation"]).reset_index(drop=True)
    common = [column for column in current.columns if column in frozen.columns]
    mismatches: dict[str, int] = {}
    maximum_numeric_difference = 0.0
    for column in common:
        left = frozen[column]
        right = current[column]
        if pd.api.types.is_datetime64_any_dtype(left):
            count = int((pd.to_datetime(left) != pd.to_datetime(right)).sum())
        elif pd.api.types.is_numeric_dtype(left):
            left_number = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_number = pd.to_numeric(right, errors="coerce").to_numpy(float)
            equal = np.isclose(left_number, right_number, rtol=1e-12, atol=1e-12, equal_nan=True)
            count = int((~equal).sum())
            finite = np.isfinite(left_number) & np.isfinite(right_number)
            if finite.any():
                maximum_numeric_difference = max(
                    maximum_numeric_difference,
                    float(np.max(np.abs(left_number[finite] - right_number[finite]))),
                )
        else:
            count = int((left.astype(str) != right.astype(str)).sum())
        mismatches[column] = count
    return {
        "rows": int(len(current)),
        "columns_compared": int(len(common)),
        "mismatch_by_column": mismatches,
        "total_mismatches": int(sum(mismatches.values())),
        "maximum_numeric_difference": maximum_numeric_difference,
    }


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, group in metrics.groupby("period", sort=True):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        rows.append(
            {
                "period": period,
                "stress_annualized_excess_minimum": _safe_float(annual.min()),
                "stress_annualized_excess_median": _safe_float(annual.median()),
                "stress_rolling_242d_excess_minimum": _safe_float(rolling.dropna().min()),
                "stress_start_pass_ratio": float(
                    group["stress_both_20pct_gates"].fillna(False).astype(bool).mean()
                ),
                "every_start_both_20pct_gates": bool(
                    group["stress_both_20pct_gates"].fillna(False).astype(bool).all()
                ),
            }
        )
    return rows


def main() -> int:
    config, manifest, hash_mismatches = _load_freeze()
    signal, market, dividends, input_audit = _load_inputs(config)
    features = _rebuild_features(signal, market, dividends, config)
    feature_comparison = _compare_features(features)
    metrics = _evaluate(features, market, dividends, config)
    metric_comparison = _compare_metrics(metrics)
    aggregates = _aggregate(metrics)
    combined = next(
        row for row in aggregates if row["period"] == "COMBINED_HISTORY_TO_2026_08_14"
    )
    passed = bool(
        not hash_mismatches
        and feature_comparison["total_mismatches"] == 0
        and metric_comparison["total_mismatches"] == 0
        and combined["every_start_both_20pct_gates"]
    )
    status = (
        "INDEPENDENT_REPLICATION_PASS_HISTORICAL_ONLY_FORWARD_REQUIRED"
        if passed
        else "INDEPENDENT_REPLICATION_FAILED"
    )
    output_columns = list(pd.read_parquet(FROZEN_FEATURES).columns)
    _atomic_parquet(features[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "passed": passed,
        "candidate_policy": POLICY,
        "manifest_sha256": _sha256(MANIFEST_FILE),
        "config_sha256": _sha256(CONFIG_FILE),
        "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
        "input_audit": input_audit,
        "feature_replication": feature_comparison,
        "metric_replication": metric_comparison,
        "aggregates": aggregates,
        "boundaries": {
            "signal_data_ceiling": config["dates"]["frozen_signal_cutoff"],
            "execution_market_ceiling": config["dates"]["frozen_execution_market_cutoff"],
            "post_cutoff_rows_read": 0,
            "historical_rows_were_seen_during_discovery": True,
            "unseen_forward_day_count": 0,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
        "evidence": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
