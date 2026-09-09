"""用期权市场信息决定510300下一交易日满仓或空仓的开发期检验。

期权仅作为信息源；唯一可交易资产仍是510300，组合状态严格为0或1。
所有信号在T日收盘后形成，并在T+1交易日开盘执行。脚本只读取
2023-12-31及以前的数据，不访问保留的2024年及以后样本。
"""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_module
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)
from multi_domain_severity_rank_5d_discovery_v0 import _rolling_last_percentile
from research.option_ten_factor_direction_v1 import build_iv_features


PROJECT_ID = "510300_OPTION_INFORMATION_BINARY_DISCOVERY_V1"
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")
SELECTION_START = pd.Timestamp("2020-07-01")
SELECTION_END = pd.Timestamp("2023-12-29")
START_PERTURBATIONS = list(range(5))

OPTION_EOD_FILE = (
    PROJECT_ROOT / "data" / "raw" / "return_tail" / "options" / "510300_tushare_eod.parquet"
)
OPTION_RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "return_tail"
    / "options"
    / "510300_sse_risk_indicators.parquet"
)
OPTION_STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "return_tail"
    / "options"
    / "510300_sse_daily_statistics.parquet"
)
IV_CONFIG_FILE = PROJECT_ROOT / "config" / "510300_option_ten_factor_direction_v1.yaml"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_option_information_binary_discovery_v1.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_information_binary_discovery_v1"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_option_information_binary_discovery_v1.json"
)


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


def _normalize_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    output = frame.copy()
    output[column] = pd.to_datetime(output[column], errors="raise").dt.normalize()
    return output


def _read_development_parquet(path: Path, date_column: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[(date_column, "<=", DEVELOPMENT_CUTOFF)],
    )
    frame = _normalize_date_column(frame, date_column)
    if frame.empty or frame[date_column].max() > DEVELOPMENT_CUTOFF:
        raise ValueError(f"开发期输入为空或越过截止日：{path}")
    return frame


def _build_option_features() -> tuple[pd.DataFrame, dict[str, Any]]:
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
    stats_columns = [
        "trade_date",
        "call_volume",
        "put_volume",
        "total_volume",
        "open_interest",
        "call_open_interest",
        "put_open_interest",
    ]
    eod = _read_development_parquet(OPTION_EOD_FILE, "trade_date", eod_columns)
    risk = _read_development_parquet(OPTION_RISK_FILE, "trade_date", risk_columns)
    stats = _read_development_parquet(OPTION_STATS_FILE, "trade_date", stats_columns)
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
    eligible = panel.loc[
        ~panel["is_adjusted"].astype(bool)
        & panel["dte"].between(5, 90)
        & panel["contract_unit"].gt(0)
        & panel["open_interest"].ge(0)
        & panel["volume"].ge(0)
        & panel[numeric_columns].notna().all(axis=1)
    ].copy()
    if eligible.empty:
        raise ValueError("开发期没有合格的期权风险指标记录")

    option_sign = np.where(eligible["option_type"].eq("C"), 1.0, -1.0)
    gamma_weight = (
        eligible["gamma"].abs()
        * eligible["open_interest"]
        * eligible["contract_unit"]
        * eligible["underlying_close"].pow(2)
    )
    delta_weight = (
        eligible["delta"].abs()
        * eligible["open_interest"]
        * eligible["contract_unit"]
        * eligible["underlying_close"]
    )
    eligible["gamma_abs_oi"] = gamma_weight
    eligible["gamma_option_signed_oi"] = option_sign * gamma_weight
    eligible["gamma_delta_signed_oi"] = np.sign(eligible["delta"]) * gamma_weight
    eligible["delta_abs_oi"] = delta_weight
    eligible["delta_signed_oi"] = np.sign(eligible["delta"]) * delta_weight

    aggregate = eligible.groupby("trade_date", sort=True).agg(
        gamma_abs_oi=("gamma_abs_oi", "sum"),
        gamma_option_signed_oi=("gamma_option_signed_oi", "sum"),
        gamma_delta_signed_oi=("gamma_delta_signed_oi", "sum"),
        delta_abs_oi=("delta_abs_oi", "sum"),
        delta_signed_oi=("delta_signed_oi", "sum"),
        eligible_contract_count=("contract_code", "size"),
    )
    aggregate["gamma_balance"] = (
        aggregate["gamma_option_signed_oi"] / aggregate["gamma_abs_oi"].replace(0.0, np.nan)
    )
    aggregate["gamma_signed"] = (
        aggregate["gamma_delta_signed_oi"] / aggregate["gamma_abs_oi"].replace(0.0, np.nan)
    )
    aggregate["delta_balance"] = (
        aggregate["delta_signed_oi"] / aggregate["delta_abs_oi"].replace(0.0, np.nan)
    )
    aggregate = aggregate.reset_index().rename(columns={"trade_date": "date"})

    stats = stats.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    for column in stats_columns[1:]:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    stats["put_call_volume_ratio"] = (stats["put_volume"] + 1.0) / (stats["call_volume"] + 1.0)
    stats["put_call_open_interest_ratio"] = (
        (stats["put_open_interest"] + 1.0) / (stats["call_open_interest"] + 1.0)
    )
    for column in ["total_volume", "open_interest", "put_call_volume_ratio", "put_call_open_interest_ratio"]:
        stats[f"{column}_chg1"] = np.log(stats[column].clip(lower=0.0) + 1.0).diff(1)
        stats[f"{column}_chg5"] = np.log(stats[column].clip(lower=0.0) + 1.0).diff(5)
    stats.rename(columns={"trade_date": "date"}, inplace=True)

    iv_config = yaml.safe_load(IV_CONFIG_FILE.read_text(encoding="utf-8"))
    iv_panel = panel.copy()
    iv_features = build_iv_features(iv_panel, iv_config)
    iv_features["date"] = pd.to_datetime(iv_features["date"], errors="raise").dt.normalize()
    iv_features.sort_values("date", inplace=True)
    iv_features["atm_iv_30d_chg5"] = iv_features["atm_iv_30d"].diff(5)
    iv_features["atm_put_call_iv_skew_chg5"] = iv_features["iv_skew_25d"].diff(5)

    features = aggregate.merge(stats, on="date", how="outer", validate="one_to_one")
    features = features.merge(iv_features, on="date", how="outer", validate="one_to_one")
    features.sort_values("date", inplace=True)
    features.drop_duplicates("date", keep="last", inplace=True)
    features.reset_index(drop=True, inplace=True)

    rank_sources = [
        "gamma_balance",
        "gamma_signed",
        "delta_balance",
        "open_interest",
        "total_volume_chg5",
        "atm_put_call_iv_skew_chg5",
    ]
    for column in rank_sources:
        features[f"rank252_{column}"] = _rolling_last_percentile(features[column])
    audit = {
        "eod_rows_read": int(len(eod)),
        "risk_rows_read": int(len(risk)),
        "eligible_contract_rows": int(len(eligible)),
        "feature_rows": int(len(features)),
        "first_feature_date": features["date"].min().date().isoformat(),
        "last_feature_date": features["date"].max().date().isoformat(),
        "maximum_source_date": max(
            eod["trade_date"].max(), risk["trade_date"].max(), stats["date"].max()
        ).date().isoformat(),
        "post_2023_rows_read": 0,
        "rank_window": 252,
        "rank_minimum_history": 126,
    }
    return features, audit


def _build_policy_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    gb = frame["rank252_gamma_balance"]
    gs = frame["rank252_gamma_signed"]
    db = frame["rank252_delta_balance"]
    oi = frame["rank252_open_interest"]
    vol = frame["rank252_total_volume_chg5"]
    skew = frame["rank252_atm_put_call_iv_skew_chg5"]

    primitive = {
        "GB10": gb.ge(0.90),
        "GB15": gb.ge(0.85),
        "GB20": gb.ge(0.80),
        "GS15": gs.ge(0.85),
        "GS20": gs.ge(0.80),
        "DB15": db.ge(0.85),
        "DB20": db.ge(0.80),
        "OI20": oi.ge(0.80),
        "VOL_LOW05": vol.le(0.05),
        "VOL_LOW08": vol.le(0.08),
        "SKEW_HIGH10": skew.ge(0.90),
        "SKEW_HIGH15": skew.ge(0.85),
    }
    policies = {f"PULSE_{name}": signal.fillna(False) for name, signal in primitive.items()}

    core_narrow = pd.concat([primitive["GB10"], primitive["GS15"], primitive["DB15"]], axis=1).sum(axis=1)
    core_broad = pd.concat([primitive["GB20"], primitive["GS20"], primitive["DB20"]], axis=1).sum(axis=1)
    auxiliary_narrow = pd.concat(
        [primitive["GB10"], primitive["VOL_LOW05"], primitive["SKEW_HIGH10"]], axis=1
    ).sum(axis=1)
    auxiliary_broad = pd.concat(
        [primitive["GB20"], primitive["OI20"], primitive["VOL_LOW08"], primitive["SKEW_HIGH15"]],
        axis=1,
    ).sum(axis=1)
    for threshold in [1, 2, 3]:
        policies[f"PULSE_CORE_NARROW_VOTE{threshold}"] = core_narrow.ge(threshold)
        policies[f"PULSE_CORE_BROAD_VOTE{threshold}"] = core_broad.ge(threshold)
        policies[f"PULSE_AUX_NARROW_VOTE{threshold}"] = auxiliary_narrow.ge(threshold)
    for threshold in [1, 2, 3, 4]:
        policies[f"PULSE_AUX_BROAD_VOTE{threshold}"] = auxiliary_broad.ge(threshold)
    return policies


def _evaluate_policy(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy_name: str,
    signal: pd.Series,
    start_perturbation: int,
) -> dict[str, Any]:
    selection_indices = np.flatnonzero(
        frame["date"].between(SELECTION_START, SELECTION_END).to_numpy()
    )
    first_execution_index = int(selection_indices[start_perturbation])
    last_execution_index = int(selection_indices[-1])
    anchor_index = first_execution_index - 1
    if anchor_index < 0:
        raise ValueError("缺少首个执行日的前一交易日信号锚点")

    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states = np.zeros(len(sample), dtype=np.int8)
    signal_values = signal.fillna(False).to_numpy(dtype=bool)
    for execution_index in range(first_execution_index, last_execution_index + 1):
        sample_index = execution_index - anchor_index
        states[sample_index] = 0 if signal_values[execution_index - 1] else 1

    benchmark = build_benchmark_ledger(sample, market_module.INITIAL_CAPITAL_CNY)
    base_ledger, base_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=market_module.BASE_COSTS,
        initial_capital=market_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    stress_ledger, stress_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=market_module.STRESS_COSTS,
        initial_capital=market_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    base = summarize_path(base_ledger, base_trades, benchmark, objective=market_module.OBJECTIVE)
    stress = summarize_path(stress_ledger, stress_trades, benchmark, objective=market_module.OBJECTIVE)
    execution_states = states[1:]
    row: dict[str, Any] = {
        "policy": policy_name,
        "start_perturbation": start_perturbation,
        "first_execution_date": sample.loc[1, "date"],
        "last_execution_date": sample.loc[len(sample) - 1, "date"],
        "execution_day_count": int(len(execution_states)),
        "cash_day_count": int((execution_states == 0).sum()),
        "cash_day_share": float((execution_states == 0).mean()),
    }
    for prefix, summary in [("base", base), ("stress", stress)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    return row


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in metrics.groupby("policy", sort=True):
        stress_annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        stress_rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        base_rolling = pd.to_numeric(group["base_rolling_242d_excess_median"], errors="coerce")
        every_pass = bool(
            len(group) == len(START_PERTURBATIONS)
            and group["stress_both_20pct_gates"].astype(bool).all()
        )
        rows.append(
            {
                "policy": policy,
                "perturbation_count": int(len(group)),
                "base_annualized_excess_minimum": _safe_float(base_annual.min()),
                "base_annualized_excess_median": _safe_float(base_annual.median()),
                "base_rolling_242d_excess_minimum": _safe_float(base_rolling.min()),
                "stress_annualized_excess_minimum": _safe_float(stress_annual.min()),
                "stress_annualized_excess_median": _safe_float(stress_annual.median()),
                "stress_annualized_excess_maximum": _safe_float(stress_annual.max()),
                "stress_rolling_242d_excess_minimum": _safe_float(stress_rolling.min()),
                "stress_rolling_242d_excess_median": _safe_float(stress_rolling.median()),
                "stress_every_perturbation_both_20pct_gates": every_pass,
                "stress_perturbation_pass_ratio": float(group["stress_both_20pct_gates"].mean()),
                "cash_day_share_median": float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": _safe_float(group["stress_trade_leg_count"].median()),
                "development_gate": every_pass,
            }
        )
    rows.sort(
        key=lambda item: item["stress_annualized_excess_median"]
        if item["stress_annualized_excess_median"] is not None
        else -float("inf"),
        reverse=True,
    )
    return rows


def main() -> int:
    required = [OPTION_EOD_FILE, OPTION_RISK_FILE, OPTION_STATS_FILE, IV_CONFIG_FILE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    features, feature_audit = _build_option_features()
    market_module.DEVELOPMENT_CUTOFF = DEVELOPMENT_CUTOFF
    dividends = market_module._load_dividends()
    market = market_module._load_market(dividends).copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[market["date"].le(DEVELOPMENT_CUTOFF)].sort_values("date")
    frame = market.merge(features, on="date", how="left", validate="one_to_one").reset_index(drop=True)
    selection_rows = int(frame["date"].between(SELECTION_START, SELECTION_END).sum())
    if selection_rows < 800:
        raise ValueError("期权信息开发选择段覆盖不足")

    policies = _build_policy_signals(frame)
    metric_rows: list[dict[str, Any]] = []
    for policy_name, signal in policies.items():
        for perturbation in START_PERTURBATIONS:
            metric_rows.append(
                _evaluate_policy(frame, dividends, policy_name, signal, perturbation)
            )
    metrics = pd.DataFrame(metric_rows)
    aggregates = _aggregate(metrics)
    passed = [row for row in aggregates if row["development_gate"]]
    status = (
        "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
        if passed
        else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
    )

    signal_frame = frame[["date"]].copy()
    for policy_name, signal in policies.items():
        signal_frame[f"signal__{policy_name}"] = signal.astype(np.int8)
    output_features = features.merge(signal_frame, on="date", how="left", validate="one_to_one")
    _atomic_parquet(output_features, OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    payload = {
        "status": status,
        "project_id": PROJECT_ID,
        "scope": {
            "execution_asset": "510300.SH",
            "information_source": "510300期权市场日频风险指标与成交持仓统计",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "option_execution_allowed": False,
        },
        "development_contract": {
            "source_data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "selection_period": [SELECTION_START.date().isoformat(), SELECTION_END.date().isoformat()],
            "start_perturbations": START_PERTURBATIONS,
            "initial_capital_cny": market_module.INITIAL_CAPITAL_CNY,
            "commission_rate_per_leg": market_module.STRESS_COSTS.commission_rate,
            "base_slippage_bps_per_leg": market_module.BASE_COSTS.slippage_bps,
            "stress_slippage_bps_per_leg": market_module.STRESS_COSTS.slippage_bps,
            "cash_annual_rate": market_module.STRESS_COSTS.cash_annual_rate,
            "lot_size": market_module.STRESS_COSTS.lot_size,
            "minimum_annualized_net_excess": market_module.OBJECTIVE["minimum_annualized_excess"],
            "minimum_rolling_242d_excess_median": market_module.OBJECTIVE[
                "minimum_rolling_excess_median"
            ],
            "all_five_starts_must_pass": True,
        },
        "candidate_count": len(policies),
        "passed_candidate_count": len(passed),
        "best_candidate": aggregates[0],
        "passed_candidates": passed,
        "frontier": aggregates,
        "feature_audit": feature_audit,
        "selection_rows": selection_rows,
        "input_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in required},
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
        },
        "boundaries": {
            "validation_or_holdout_rows_read": 0,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(policies),
                "passed_candidate_count": len(passed),
                "best_candidate": aggregates[0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
