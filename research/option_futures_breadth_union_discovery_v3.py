"""期权制度、IF期货桥梁与全A成交结构的510300二元正交并集开发检验。

信息源可以来自期权、股指期货与全A横截面，但唯一执行资产是510300，状态只能
是满仓或空仓。所有输入截至2023-12-31，信号于T日收盘后形成，T+1开盘执行。
"""

from __future__ import annotations

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
import option_information_binary_discovery_v1 as option_v1
import option_information_state_machine_discovery_v2 as option_v2
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)
from futures_option_bridge_v1 import build_features as build_futures_features
from multi_domain_severity_rank_5d_discovery_v0 import _rolling_last_percentile


PROJECT_ID = "510300_OPTION_FUTURES_BREADTH_UNION_DISCOVERY_V3"
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")
SELECTION_START = pd.Timestamp("2022-03-09")
SELECTION_END = pd.Timestamp("2023-12-29")
START_PERTURBATIONS = list(range(5))

FUTURES_CONFIG_FILE = PROJECT_ROOT / "config" / "510300_futures_option_bridge_v1.yaml"
FUTURES_FILE = PROJECT_ROOT / "data" / "raw" / "futures" / "IF0_daily_raw.parquet"
SPOT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_raw.parquet"
OPTION_STATS_FILE = option_v1.OPTION_STATS_FILE
BENCHMARK_RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
)
ALL_A_FILE = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "510300_daily_consensus_catchup_v1"
    / "all_a_features_2016_2025.parquet"
)
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_option_futures_breadth_union_discovery_v3.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_futures_breadth_union_discovery_v3"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_futures_breadth_union_discovery_v3"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_option_futures_breadth_union_discovery_v3.json"
)


def _read_filtered(path: Path, date_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[(date_column, "<=", DEVELOPMENT_CUTOFF)])
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
    if frame.empty or frame[date_column].max() > DEVELOPMENT_CUTOFF:
        raise ValueError(f"开发输入为空或越过截止日：{path}")
    return frame


def _build_common_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    config = yaml.safe_load(FUTURES_CONFIG_FILE.read_text(encoding="utf-8"))
    futures = _read_filtered(FUTURES_FILE, "date")
    spot = _read_filtered(SPOT_FILE, "date")
    statistics = _read_filtered(OPTION_STATS_FILE, "trade_date")
    benchmark = _read_filtered(BENCHMARK_RAW_FILE, "date")
    futures_features = build_futures_features(futures, spot, statistics, benchmark, config)
    futures_features["date"] = pd.to_datetime(futures_features["date"], errors="raise").dt.normalize()

    all_a = pd.read_parquet(
        ALL_A_FILE,
        columns=["date", "top10pct_amount_share"],
        filters=[("date", "<=", DEVELOPMENT_CUTOFF)],
    )
    all_a["date"] = pd.to_datetime(all_a["date"], errors="raise").dt.normalize()
    if all_a.empty or all_a["date"].max() > DEVELOPMENT_CUTOFF:
        raise ValueError("全A开发输入为空或越过截止日")

    source_columns = [
        "date",
        "basis_z60",
        "basis_change5",
        "futures_open_interest_log_change5",
        "option_log_put_call_volume",
    ]
    common = futures_features[source_columns].merge(
        all_a,
        on="date",
        how="inner",
        validate="one_to_one",
    )
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
        common[f"rank252_{column}"] = _rolling_last_percentile(common[column])

    common["trigger_breadth_concentration_low20"] = common[
        "rank252_top10pct_amount_share"
    ].le(0.20)
    common["trigger_basis_level_low10"] = common["rank252_basis_z60"].le(0.10)
    common["trigger_basis_change_low05"] = common["rank252_basis_change5"].le(0.05)
    common["trigger_futures_oi_change_low15"] = common[
        "rank252_futures_open_interest_log_change5"
    ].le(0.15)
    common["trigger_put_call_volume_low10"] = common[
        "rank252_option_log_put_call_volume"
    ].le(0.10)
    trigger_columns = [column for column in common.columns if column.startswith("trigger_")]
    common["orthogonal_vote_count"] = common[trigger_columns].sum(axis=1)
    audit = {
        "futures_rows_read": int(len(futures)),
        "common_feature_rows": int(len(common)),
        "common_first_date": common["date"].min().date().isoformat(),
        "common_last_date": common["date"].max().date().isoformat(),
        "rank_window": 252,
        "rank_minimum_history": 126,
        "post_2023_rows_read": 0,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): option_v1._sha256(path)
            for path in [
                FUTURES_CONFIG_FILE,
                FUTURES_FILE,
                SPOT_FILE,
                OPTION_STATS_FILE,
                BENCHMARK_RAW_FILE,
                ALL_A_FILE,
            ]
        },
    }
    return common, audit


def _build_market_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common, common_audit = _build_common_features()
    option_features = pd.read_parquet(option_v1.OUTPUT_FEATURES)
    option_features["date"] = pd.to_datetime(option_features["date"], errors="raise").dt.normalize()
    if option_features.empty or option_features["date"].max() > DEVELOPMENT_CUTOFF:
        raise ValueError("期权V1特征为空或包含2024年及以后数据")

    market_module.DEVELOPMENT_CUTOFF = DEVELOPMENT_CUTOFF
    dividends = market_module._load_dividends()
    market = market_module._load_market(dividends).copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[market["date"].le(DEVELOPMENT_CUTOFF)].sort_values("date")
    option_columns = [
        "date",
        "rank252_gamma_balance",
        "rank252_gamma_signed",
        "rank252_delta_balance",
    ]
    frame = market.merge(option_features[option_columns], on="date", how="left", validate="one_to_one")
    frame = frame.merge(common, on="date", how="left", validate="one_to_one").reset_index(drop=True)

    option_spec = option_v2.StatePolicy(
        "HYSTERESIS",
        "rank252_gamma_balance",
        0.90,
        recovery=0.50,
    )
    option_start_indices = np.flatnonzero(
        frame["date"].between(option_v1.SELECTION_START, option_v1.SELECTION_END).to_numpy()
    )
    option_states, option_decisions = option_v2._states_for_range(
        frame,
        int(option_start_indices[0]),
        int(option_start_indices[-1]),
        option_spec,
    )
    frame["option_hysteresis_cash_state"] = False
    option_anchor = int(option_start_indices[0]) - 1
    frame.loc[
        option_anchor : int(option_start_indices[-1]),
        "option_hysteresis_cash_state",
    ] = option_states == 0
    frame["option_hysteresis_cash_state"] = frame["option_hysteresis_cash_state"].astype(bool)
    common_audit["option_state_decision_rows"] = int(len(option_decisions))
    common_audit["selection_rows"] = int(frame["date"].between(SELECTION_START, SELECTION_END).sum())
    return frame, dividends, common_audit


def _state_vector(
    frame: pd.DataFrame,
    first_execution_index: int,
    last_execution_index: int,
    minimum_orthogonal_votes: int | None,
    include_option_state: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    anchor_index = first_execution_index - 1
    states = np.zeros(last_execution_index - anchor_index + 1, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for execution_index in range(first_execution_index, last_execution_index + 1):
        signal_index = execution_index - 1
        option_cash = bool(frame.loc[execution_index, "option_hysteresis_cash_state"])
        votes_value = frame.loc[signal_index, "orthogonal_vote_count"]
        votes = 0 if pd.isna(votes_value) else int(votes_value)
        vote_cash = bool(
            minimum_orthogonal_votes is not None and votes >= minimum_orthogonal_votes
        )
        cash = bool((include_option_state and option_cash) or vote_cash)
        state = 0 if cash else 1
        states[execution_index - anchor_index] = state
        rows.append(
            {
                "signal_date": pd.Timestamp(frame.loc[signal_index, "date"]),
                "execution_date": pd.Timestamp(frame.loc[execution_index, "date"]),
                "option_cash": option_cash,
                "orthogonal_vote_count": votes,
                "vote_cash": vote_cash,
                "state": state,
            }
        )
    return states, pd.DataFrame(rows)


def _evaluate(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy_name: str,
    minimum_votes: int | None,
    include_option_state: bool,
    start_perturbation: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_indices = np.flatnonzero(
        frame["date"].between(SELECTION_START, SELECTION_END).to_numpy()
    )
    first_execution_index = int(selection_indices[start_perturbation])
    last_execution_index = int(selection_indices[-1])
    anchor_index = first_execution_index - 1
    if anchor_index < 0:
        raise ValueError("缺少首个执行日的前一信号日")
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states, decisions = _state_vector(
        frame,
        first_execution_index,
        last_execution_index,
        minimum_votes,
        include_option_state,
    )
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
    row: dict[str, Any] = {
        "policy": policy_name,
        "minimum_orthogonal_votes": minimum_votes,
        "include_option_state": include_option_state,
        "start_perturbation": start_perturbation,
        "cash_day_count": int(decisions["state"].eq(0).sum()),
        "cash_day_share": float(decisions["state"].eq(0).mean()),
        "execution_day_count": int(len(decisions)),
    }
    for prefix, summary in [("base", base), ("stress", stress)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    decisions.insert(0, "policy", policy_name)
    decisions.insert(1, "start_perturbation", start_perturbation)
    return row, decisions


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in metrics.groupby("policy", sort=True):
        first = group.iloc[0]
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        base_rolling = pd.to_numeric(group["base_rolling_242d_excess_median"], errors="coerce")
        passed = bool(
            len(group) == len(START_PERTURBATIONS)
            and group["stress_both_20pct_gates"].astype(bool).all()
        )
        rows.append(
            {
                "policy": policy,
                "minimum_orthogonal_votes": None
                if pd.isna(first["minimum_orthogonal_votes"])
                else int(first["minimum_orthogonal_votes"]),
                "include_option_state": bool(first["include_option_state"]),
                "base_annualized_excess_minimum": option_v1._safe_float(base_annual.min()),
                "base_annualized_excess_median": option_v1._safe_float(base_annual.median()),
                "base_rolling_242d_excess_minimum": option_v1._safe_float(base_rolling.min()),
                "stress_annualized_excess_minimum": option_v1._safe_float(annual.min()),
                "stress_annualized_excess_median": option_v1._safe_float(annual.median()),
                "stress_annualized_excess_maximum": option_v1._safe_float(annual.max()),
                "stress_rolling_242d_excess_minimum": option_v1._safe_float(rolling.min()),
                "stress_rolling_242d_excess_median": option_v1._safe_float(rolling.median()),
                "cash_day_share_median": float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": option_v1._safe_float(
                    group["stress_trade_leg_count"].median()
                ),
                "stress_perturbation_pass_ratio": float(
                    group["stress_both_20pct_gates"].mean()
                ),
                "development_gate": passed,
            }
        )
    rows.sort(
        key=lambda row: row["stress_annualized_excess_median"]
        if row["stress_annualized_excess_median"] is not None
        else -float("inf"),
        reverse=True,
    )
    return rows


def main() -> int:
    required = [
        FUTURES_CONFIG_FILE,
        FUTURES_FILE,
        SPOT_FILE,
        OPTION_STATS_FILE,
        BENCHMARK_RAW_FILE,
        ALL_A_FILE,
        option_v1.OUTPUT_FEATURES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        option_v1._atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    frame, dividends, frame_audit = _build_market_frame()
    policies: dict[str, tuple[int | None, bool]] = {
        "OPTION_HYSTERESIS_ONLY": (None, True),
        "ORTHOGONAL_VOTE2_ONLY": (2, False),
        "OPTION_OR_ORTHOGONAL_VOTE1": (1, True),
        "OPTION_OR_ORTHOGONAL_VOTE2": (2, True),
        "OPTION_OR_ORTHOGONAL_VOTE3": (3, True),
    }
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for policy_name, (minimum_votes, include_option_state) in policies.items():
        for perturbation in START_PERTURBATIONS:
            metrics, decisions = _evaluate(
                frame,
                dividends,
                policy_name,
                minimum_votes,
                include_option_state,
                perturbation,
            )
            metric_rows.append(metrics)
            decision_frames.append(decisions)
    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    frontier = _aggregate(metrics)
    passed = [row for row in frontier if row["development_gate"]]
    status = (
        "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
        if passed
        else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
    )

    feature_columns = [
        "date",
        "option_hysteresis_cash_state",
        "orthogonal_vote_count",
        "trigger_breadth_concentration_low20",
        "trigger_basis_level_low10",
        "trigger_basis_change_low05",
        "trigger_futures_oi_change_low15",
        "trigger_put_call_volume_low10",
        "rank252_gamma_balance",
        "rank252_top10pct_amount_share",
        "rank252_basis_z60",
        "rank252_basis_change5",
        "rank252_futures_open_interest_log_change5",
        "rank252_option_log_put_call_volume",
    ]
    option_v1._atomic_parquet(frame[feature_columns], OUTPUT_FEATURES)
    option_v1._atomic_parquet(metrics, OUTPUT_METRICS)
    option_v1._atomic_parquet(decisions, OUTPUT_DECISIONS)
    payload = {
        "status": status,
        "project_id": PROJECT_ID,
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "information_only_assets": ["510300期权", "IF0股指期货", "全A横截面"],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "selection_period": [SELECTION_START.date().isoformat(), SELECTION_END.date().isoformat()],
            "start_perturbations": START_PERTURBATIONS,
            "candidate_count": len(policies),
            "option_state": "Gamma余额历史分位达到90%进入空仓，跌回50%以下回到满仓",
            "orthogonal_vote_members": {
                "all_a_top10pct_amount_share": "共同历史滚动252日低20%",
                "if_basis_z60": "共同历史滚动252日低10%",
                "if_basis_change5": "共同历史滚动252日低5%",
                "if_open_interest_change5": "共同历史滚动252日低15%",
                "option_put_call_volume": "共同历史滚动252日低10%",
            },
            "primary_union": "期权制度为空仓，或五项正交信息至少两票触发",
            "stress_costs": {
                "commission_rate_per_leg": market_module.STRESS_COSTS.commission_rate,
                "slippage_bps_per_leg": market_module.STRESS_COSTS.slippage_bps,
                "cash_annual_rate": market_module.STRESS_COSTS.cash_annual_rate,
                "lot_size": market_module.STRESS_COSTS.lot_size,
            },
            "gate": "五个起点在压力成本下年化净超额和滚动242日超额中位数均不低于20%",
        },
        "passed_candidate_count": len(passed),
        "best_candidate": frontier[0],
        "passed_candidates": passed,
        "frontier": frontier,
        "frame_audit": frame_audit,
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
            "decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
        },
        "evidence_limit": "本分支复用已经看过的历史研究输入，不再宣称存在未见历史验证集；通过后必须先冻结，再进入前向Shadow验证。",
        "boundaries": {
            "post_2023_rows_read": 0,
            "position_mapping": "DISABLED",
            "orders": "DISABLED",
            "broker": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    option_v1._atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(policies),
                "passed_candidate_count": len(passed),
                "best_candidate": frontier[0],
                "passed_candidates": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
