"""固定专家双影子并集V1的预声明发现运行。

本脚本只组合三个已经存在的固定原始空仓专家，不搜索阈值。所有专家使用与
ETF微观结构V1完全相同的反事实双影子门，正式执行资产仍只有510300与现金。
输出是历史已污染的发现证据，不能直接解释为前瞻通过或实盘授权。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

from binary_state_feasibility_v1 import (
    CostModel,
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_FIXED_EXPERT_DUAL_SHADOW_UNION_DISCOVERY_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_FILE = (
    PROJECT_ROOT / "config" / "510300_fixed_expert_dual_shadow_union_discovery_v1.yaml"
)
MANIFEST_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1_manifest.json"
)
FREEZE_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1_freeze_receipt.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1"
)
OUTPUT_STATES = OUTPUT_ROOT / "daily_states.parquet"
OUTPUT_METRICS = OUTPUT_ROOT / "start_perturbation_metrics.parquet"
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1.json"
)

POLICIES = [
    "MICRO_DUAL_REFERENCE",
    "OPTION_DOMAIN_DUAL",
    "CONSENSUS_DUAL",
    "INDIVIDUAL_GATED_UNION",
    "INDIVIDUAL_GATED_VOTE2",
    "RAW_UNION_SHARED_DUAL",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在：{temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在：{temporary}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _load_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    receipt = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("发现配置项目编号不匹配")
    if manifest.get("project_id") != PROJECT_ID or receipt.get("project_id") != PROJECT_ID:
        raise ValueError("发现清单或冻结收据项目编号不匹配")
    if receipt.get("status") != "DISCOVERY_IMPLEMENTATION_AND_INPUTS_FROZEN_BEFORE_FIRST_RUN":
        raise ValueError("发现冻结收据状态无效")
    if receipt.get("manifest_sha256") != _sha256(MANIFEST_FILE):
        raise ValueError("发现清单哈希与冻结收据不匹配")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest["tracked_files"].items():
        path = PROJECT_ROOT / relative
        actual = _sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"发现协议或输入哈希漂移：{mismatches}")
    return config, manifest, receipt


def _coerce_bool(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{label}存在空值")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="raise")
    values = set(numeric.astype(int).unique().tolist())
    if not values.issubset({0, 1}) or not np.allclose(
        numeric.to_numpy(float), numeric.astype(int).to_numpy(float)
    ):
        raise ValueError(f"{label}不是严格0/1状态")
    return numeric.astype(int).astype(bool)


def _load_market(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    end = pd.Timestamp(config["periods"]["common_discovery_end"])
    inputs = config["inputs"]
    etf_path = PROJECT_ROOT / inputs["etf_execution"]["path"]
    benchmark_path = PROJECT_ROOT / inputs["benchmark_total_return"]["path"]
    etf = pd.read_parquet(
        etf_path,
        columns=["date", "open", "close"],
        filters=[("date", "<=", end.to_pydatetime())],
    )
    benchmark = pd.read_parquet(
        benchmark_path,
        columns=["date", "close"],
        filters=[("date", "<=", end.to_pydatetime())],
    )
    for frame, label in [(etf, "510300"), (benchmark, "H00300")]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ValueError(f"{label}行情存在重复日期")
        if frame["date"].max() > end:
            raise ValueError(f"{label}加载了2025年之后的值")
    market = etf.rename(columns={"open": "etf_open", "close": "etf_close"}).merge(
        benchmark.rename(columns={"close": "benchmark_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    for column in ["etf_open", "etf_close", "benchmark_close"]:
        market[column] = pd.to_numeric(market[column], errors="raise")
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("执行行情存在空值")
    dividend_path = PROJECT_ROOT / inputs["cash_distributions"]["path"]
    dividends = pd.read_csv(dividend_path)
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends = dividends.loc[dividends["ex_date"].le(end)].copy()
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)
    audit = {
        "market_rows": int(len(market)),
        "market_first_date": market["date"].min().date().isoformat(),
        "market_last_date": market["date"].max().date().isoformat(),
        "post_2025_market_values_loaded": int(market["date"].gt(end).sum()),
        "dividend_event_count_through_2025": int(len(dividends)),
        "last_dividend_ex_date_loaded": dividends["ex_date"].max().date().isoformat(),
    }
    return market, dividends, audit


def _load_expert_panel(config: dict[str, Any], market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = config["inputs"]
    warmup = pd.Timestamp(config["periods"]["shadow_warmup_start"])
    end = pd.Timestamp(config["periods"]["common_discovery_end"])

    micro_item = inputs["microstructure_states"]
    micro = pd.read_parquet(
        PROJECT_ROOT / micro_item["path"],
        columns=["date", "cash_raw", "cash_final"],
        filters=[("date", "<=", end.to_pydatetime())],
    )
    micro["date"] = pd.to_datetime(micro["date"], errors="raise").dt.normalize()
    micro = micro.loc[micro["date"].ge(warmup)].copy()
    micro.rename(
        columns={"cash_raw": "raw_micro", "cash_final": "upstream_micro_final"},
        inplace=True,
    )
    micro["raw_micro"] = _coerce_bool(micro["raw_micro"], "micro.raw_cash")
    micro["upstream_micro_final"] = _coerce_bool(
        micro["upstream_micro_final"], "micro.cash_final"
    )

    option_item = inputs["option_futures_breadth_states"]
    option = pd.read_parquet(
        PROJECT_ROOT / option_item["path"],
        columns=["date", "final_state"],
        filters=[("date", "<=", end.to_pydatetime())],
    )
    option["date"] = pd.to_datetime(option["date"], errors="raise").dt.normalize()
    option = option.loc[option["date"].ge(warmup)].copy()
    option_state = pd.to_numeric(option.pop("final_state"), errors="raise")
    if not set(option_state.astype(int).unique()).issubset({0, 1}):
        raise ValueError("option_domain.final_state不是0/1")
    option["raw_option_domain"] = option_state.astype(int).eq(0)

    consensus_item = inputs["daily_consensus_states"]
    consensus = pd.read_parquet(
        PROJECT_ROOT / consensus_item["path"],
        columns=["start_perturbation", "signal_date", "cash_signal"],
        filters=[
            ("start_perturbation", "==", 0),
            ("signal_date", "<=", end.to_pydatetime()),
        ],
    )
    if not pd.to_numeric(consensus["start_perturbation"], errors="raise").eq(0).all():
        raise ValueError("consensus读取到了非0起点行")
    consensus["date"] = pd.to_datetime(
        consensus["signal_date"], errors="raise"
    ).dt.normalize()
    consensus = consensus.loc[consensus["date"].ge(warmup), ["date", "cash_signal"]].copy()
    consensus.rename(columns={"cash_signal": "raw_consensus"}, inplace=True)
    consensus["raw_consensus"] = _coerce_bool(
        consensus["raw_consensus"], "consensus.cash_signal"
    )

    for frame, label in [
        (micro, "micro"),
        (option, "option_domain"),
        (consensus, "consensus"),
    ]:
        if frame["date"].duplicated().any():
            raise ValueError(f"{label}专家存在重复信号日")
        if frame["date"].max() > end:
            raise ValueError(f"{label}专家加载了2025年之后的值")

    panel = market.loc[market["date"].between(warmup, end), ["date"]].copy()
    panel = panel.merge(micro, on="date", how="left", validate="one_to_one")
    panel = panel.merge(option, on="date", how="left", validate="one_to_one")
    panel = panel.merge(consensus, on="date", how="left", validate="one_to_one")
    required = [
        "raw_micro",
        "upstream_micro_final",
        "raw_option_domain",
        "raw_consensus",
    ]
    missing = {column: int(panel[column].isna().sum()) for column in required}
    if any(missing.values()):
        raise ValueError(f"共同区间专家状态缺失，禁止填充：{missing}")
    for column in required:
        panel[column] = panel[column].astype(bool)
    audit = {
        "rows": int(len(panel)),
        "first_date": panel["date"].min().date().isoformat(),
        "last_date": panel["date"].max().date().isoformat(),
        "missing_by_required_column": missing,
        "post_2025_expert_values_loaded": int(panel["date"].gt(end).sum()),
        "raw_cash_day_counts": {
            "micro": int(panel["raw_micro"].sum()),
            "option_domain": int(panel["raw_option_domain"].sum()),
            "consensus": int(panel["raw_consensus"].sum()),
        },
    }
    return panel, audit


def _open_to_open_target(
    dates: pd.Series, market: pd.DataFrame, dividends: pd.DataFrame
) -> pd.Series:
    execution = market[["date", "etf_open"]].copy()
    dividend_by_ex_date = (
        dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    next_open = execution["etf_open"].shift(-1)
    following_open = execution["etf_open"].shift(-2)
    following_date = execution["date"].shift(-2)
    following_dividend = following_date.map(
        lambda value: float(dividend_by_ex_date.get(value, 0.0))
        if pd.notna(value)
        else 0.0
    )
    execution["one_day_open_total_return"] = (
        following_open + following_dividend
    ) / next_open - 1.0
    mapped = pd.DataFrame({"date": dates}).merge(
        execution[["date", "one_day_open_total_return"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    return mapped["one_day_open_total_return"]


def _apply_dual_gate(
    frame: pd.DataFrame, raw_column: str, prefix: str, config: dict[str, Any]
) -> None:
    contract = config["dual_shadow_gate"]
    raw = frame[raw_column].astype(bool)
    transition = raw.ne(raw.shift(1).fillna(False))
    cash_daily = float(config["account"]["cash_annual_rate"]) / int(
        config["account"]["trading_days_per_year"]
    )
    contribution = pd.Series(
        np.where(
            raw & frame["one_day_open_total_return"].notna(),
            cash_daily - frame["one_day_open_total_return"],
            0.0,
        ),
        index=frame.index,
        dtype=float,
    ) - transition.astype(float) * float(contract["stress_cost_per_raw_transition"])
    matured = contribution.shift(int(contract["maturity_lag_signal_rows"]))
    long_spec = contract["long_window"]
    short_spec = contract["short_window"]
    long_sum = matured.rolling(
        int(long_spec["matured_signal_rows"]),
        min_periods=int(long_spec["minimum_matured_rows"]),
    ).sum()
    short_sum = matured.rolling(
        int(short_spec["matured_signal_rows"]),
        min_periods=int(short_spec["minimum_matured_rows"]),
    ).sum()
    gate = (
        long_sum.gt(float(long_spec["active_when_net_sum_above"]))
        | long_sum.isna()
    ) & (
        short_sum.gt(float(short_spec["active_when_net_sum_above"]))
        | short_sum.isna()
    )
    frame[f"{prefix}_transition"] = transition.astype(bool)
    frame[f"{prefix}_shadow_net_contribution"] = contribution
    frame[f"{prefix}_shadow242_net_sum"] = long_sum
    frame[f"{prefix}_shadow60_net_sum"] = short_sum
    frame[f"{prefix}_gate_active"] = gate.astype(bool)
    frame[f"{prefix}_gated_cash"] = (raw & gate).astype(bool)


def _build_policy_states(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = panel.copy()
    frame["one_day_open_total_return"] = _open_to_open_target(
        frame["date"], market, dividends
    )
    _apply_dual_gate(frame, "raw_micro", "micro", config)
    _apply_dual_gate(frame, "raw_option_domain", "option_domain", config)
    _apply_dual_gate(frame, "raw_consensus", "consensus", config)
    frame["raw_union"] = frame[
        ["raw_micro", "raw_option_domain", "raw_consensus"]
    ].any(axis=1)
    _apply_dual_gate(frame, "raw_union", "raw_union", config)

    individually_gated = frame[
        ["micro_gated_cash", "option_domain_gated_cash", "consensus_gated_cash"]
    ]
    frame["policy_MICRO_DUAL_REFERENCE"] = frame["micro_gated_cash"]
    frame["policy_OPTION_DOMAIN_DUAL"] = frame["option_domain_gated_cash"]
    frame["policy_CONSENSUS_DUAL"] = frame["consensus_gated_cash"]
    frame["policy_INDIVIDUAL_GATED_UNION"] = individually_gated.any(axis=1)
    frame["policy_INDIVIDUAL_GATED_VOTE2"] = individually_gated.sum(axis=1).ge(2)
    frame["policy_RAW_UNION_SHARED_DUAL"] = frame["raw_union_gated_cash"]
    mismatch = frame["micro_gated_cash"].ne(frame["upstream_micro_final"])
    replication = {
        "micro_final_state_rows_compared": int(len(frame)),
        "micro_final_state_mismatches": int(mismatch.sum()),
        "status": "PASS" if not mismatch.any() else "FAIL",
    }
    if mismatch.any():
        examples = frame.loc[mismatch, ["date", "micro_gated_cash", "upstream_micro_final"]]
        raise ValueError(
            "独立双影子实现未复现微观结构冻结状态："
            f"{examples.head(10).to_dict('records')}"
        )
    return frame, replication


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
    stress = CostModel(
        slippage_bps=float(costs["stress_slippage_bps_per_leg"]), **common
    )
    objective = {
        "annualization_trading_days": int(account["trading_days_per_year"]),
        "rolling_window_trading_days": int(
            config["objective"]["rolling_window_trading_days"]
        ),
        "minimum_annualized_excess": float(
            config["objective"]["minimum_annualized_net_excess"]
        ),
        "minimum_rolling_excess_median": float(
            config["objective"]["minimum_rolling_242d_excess_median"]
        ),
    }
    return base, stress, objective


def _evaluate(
    states: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    policy_columns = [f"policy_{name}" for name in POLICIES]
    execution = market.merge(
        states[["date", *policy_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    start = pd.Timestamp(config["periods"]["common_discovery_start"])
    end = pd.Timestamp(config["periods"]["common_discovery_end"])
    selected = np.flatnonzero(execution["date"].between(start, end).to_numpy())
    if len(selected) < int(config["objective"]["minimum_common_signal_rows"]):
        raise ValueError("共同发现区间交易日不足")
    first_required_signal_position = int(selected[0]) - 1
    if execution.loc[first_required_signal_position:int(selected[-1]) - 1, policy_columns].isna().any(axis=None):
        raise ValueError("发现区间的前一日信号存在缺失")
    for column in policy_columns:
        execution[f"cash_execution_{column}"] = execution[column].shift(1)
    base_costs, stress_costs, objective = _cost_models(config)
    initial_capital = float(config["account"]["initial_capital_cny"])
    perturbations = [
        int(value)
        for value in config["objective"]["start_perturbations_trading_days"]
    ]
    last = int(selected[-1])
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        cash_column = f"cash_execution_policy_{policy}"
        for perturbation in perturbations:
            first = int(selected[perturbation])
            sample = execution.iloc[first - 1 : last + 1][
                ["date", "etf_open", "etf_close", "benchmark_close"]
            ].reset_index(drop=True)
            states_array = np.ones(len(sample), dtype=np.int8)
            states_array[0] = 0
            states_array[1:] = np.where(
                execution.loc[first:last, cash_column].to_numpy(bool), 0, 1
            )
            benchmark = build_benchmark_ledger(sample, initial_capital)
            base_ledger, base_trades = simulate_binary_path(
                sample,
                dividends,
                states_array,
                costs=base_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=True,
            )
            stress_ledger, stress_trades = simulate_binary_path(
                sample,
                dividends,
                states_array,
                costs=stress_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=True,
            )
            base = summarize_path(base_ledger, base_trades, benchmark, objective=objective)
            stress = summarize_path(
                stress_ledger, stress_trades, benchmark, objective=objective
            )
            row: dict[str, Any] = {
                "policy": policy,
                "period": "COMMON_DISCOVERY_2022_2025",
                "start_perturbation": perturbation,
                "first_execution_date": execution.loc[first, "date"],
                "last_execution_date": execution.loc[last, "date"],
                "execution_day_count": int(last - first + 1),
                "cash_day_count": int((states_array[1:] == 0).sum()),
                "cash_day_share": float((states_array[1:] == 0).mean()),
            }
            for prefix, summary in [("base", base), ("stress", stress)]:
                for key, value in summary.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        group = metrics.loc[metrics["policy"].eq(policy)].copy()
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="raise")
        rolling = pd.to_numeric(
            group["stress_rolling_242d_excess_median"], errors="raise"
        )
        both = group["stress_both_20pct_gates"].astype(bool)
        rows.append(
            {
                "policy": policy,
                "start_count": int(len(group)),
                "stress_annualized_excess_minimum": float(annual.min()),
                "stress_annualized_excess_median": float(annual.median()),
                "stress_annualized_excess_maximum": float(annual.max()),
                "stress_rolling_242d_excess_median_minimum": float(rolling.min()),
                "stress_rolling_242d_excess_median_across_starts": float(
                    rolling.median()
                ),
                "cash_day_share_median": float(
                    pd.to_numeric(group["cash_day_share"], errors="raise").median()
                ),
                "stress_trade_leg_count_median": float(
                    pd.to_numeric(group["stress_trade_leg_count"], errors="raise").median()
                ),
                "all_five_starts_pass_both_gates": bool(
                    len(group) == 5 and both.all()
                ),
                "start_pass_ratio": float(both.mean()),
            }
        )
    return rows


def _state_diagnostics(states: pd.DataFrame) -> list[dict[str, Any]]:
    start = pd.Timestamp("2022-03-09")
    end = pd.Timestamp("2025-12-31")
    selected = states.loc[states["date"].between(start, end)].copy()
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        cash = selected[f"policy_{policy}"].astype(bool)
        rows.append(
            {
                "policy": policy,
                "signal_rows": int(len(selected)),
                "cash_signal_days": int(cash.sum()),
                "cash_signal_share": float(cash.mean()),
                "signal_transitions": int(cash.ne(cash.shift(1).fillna(False)).sum()),
            }
        )
    return rows


def main() -> int:
    if OUTPUT_REPORT.exists() or OUTPUT_STATES.exists() or OUTPUT_METRICS.exists():
        raise FileExistsError("发现V1输出已存在，拒绝覆盖或重跑")
    config, manifest, receipt = _load_contract()
    market, dividends, market_audit = _load_market(config)
    panel, expert_audit = _load_expert_panel(config, market)
    states, replication = _build_policy_states(panel, market, dividends, config)
    metrics = _evaluate(states, market, dividends, config)
    aggregates = _aggregate(metrics)
    passing = [
        item["policy"] for item in aggregates if item["all_five_starts_pass_both_gates"]
    ]
    status = (
        "DISCOVERY_POLICY_FOUND_REQUIRES_INDEPENDENT_REPLICATION"
        if passing
        else "NO_FIXED_EXPERT_UNION_POLICY_MET_20PCT"
    )

    _atomic_parquet(states, OUTPUT_STATES)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "generated_at_asia_shanghai": datetime.now(TIMEZONE).isoformat(),
        "evidence_label": config["protocol"]["evidence_label"],
        "freeze": {
            "config_sha256": _sha256(CONFIG_FILE),
            "manifest_sha256": _sha256(MANIFEST_FILE),
            "receipt_sha256": _sha256(FREEZE_RECEIPT),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "tracked_file_count": int(len(manifest["tracked_files"])),
            "frozen_at_asia_shanghai": receipt["frozen_at_asia_shanghai"],
        },
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "information_experts": ["micro", "option_domain", "consensus"],
            "post_2025_values_read": 0,
        },
        "input_audit": {
            "market": market_audit,
            "experts": expert_audit,
        },
        "independent_micro_gate_replication": replication,
        "predeclared_policy_count": len(POLICIES),
        "predeclared_policies": POLICIES,
        "aggregates": aggregates,
        "state_diagnostics": _state_diagnostics(states),
        "passing_discovery_policies": passing,
        "decision": {
            "goal_achieved": False,
            "strategy_candidate_created": bool(passing),
            "candidate_status_if_any": (
                "DISCOVERY_ONLY_REQUIRES_SEPARATE_FREEZE_AND_FORWARD"
                if passing
                else None
            ),
            "parameter_rescue_after_result": "FORBIDDEN",
            "next_action": (
                "对全部通过者做独立复算、预声明稳健性和2026固定规则补数诊断"
                if passing
                else "保留否定结果，转向尚未使用的正交信息源"
            ),
        },
        "artifacts": {
            "daily_states": {
                "file": _relative(OUTPUT_STATES),
                "sha256": _sha256(OUTPUT_STATES),
            },
            "start_metrics": {
                "file": _relative(OUTPUT_METRICS),
                "sha256": _sha256(OUTPUT_METRICS),
            },
        },
        "boundaries": {
            "historically_contaminated_discovery_only": True,
            "current_state_signal_created": False,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
