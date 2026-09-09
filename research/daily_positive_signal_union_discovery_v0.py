"""组合彼此独立且开发期单日现金优势为正的逐日风险信号。"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import pandas as pd

from daily_tail_state_machine_discovery_v0 import (
    PolicySpec,
    START_PERTURBATIONS,
    _aggregate,
    _evaluate_policy_start,
)
from direct_one_day_cash_advantage_rank_discovery_v0 import _add_one_day_diagnostics
from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    PROJECT_ROOT,
    SELECTION_START,
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)


EXACT_FRONTIER_LIMIT = 40

DIRECT_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_one_day_cash_advantage_rank_discovery_v0.parquet"
)
GLOBAL_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_global_contagion_state_machine_discovery_v0.parquet"
)
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_daily_positive_signal_union_discovery_v0.parquet"
OUTPUT_FRONTIER = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_positive_signal_union_discovery_v0"
    / "combination_gross_frontier.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_positive_signal_union_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_positive_signal_union_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_daily_positive_signal_union_discovery_v0.json"

DIRECT_COLUMNS = [
    "date",
    "target1_end_date",
    "one_day_open_total_return",
    "signed_cash_advantage1",
    "avoidable_loss1",
    "avoidable_loss5",
    "risk_rolling252_global_selloff1_baseline",
    "risk_rolling252_rf_signed_all",
    "risk_rolling252_hgb_signed_all",
    "risk_rolling252_gbr_q90_loss_all",
    "risk_rolling252_elastic_signed_all",
]
GLOBAL_COLUMNS = [
    "date",
    "risk_rolling252_vix_level",
    "risk_rolling252_contagion_joint",
]


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    direct = _normalize_dates(pd.read_parquet(DIRECT_FILE, columns=DIRECT_COLUMNS))
    global_scores = _normalize_dates(pd.read_parquet(GLOBAL_FILE, columns=GLOBAL_COLUMNS))
    frame = market.merge(direct, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_scores, on="date", how="left", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(selection_mask.sum()) < 450:
        raise ValueError("正边际信号组合的选择段覆盖不足")
    audit = {
        "rows": int(len(frame)),
        "selection_rows": int(selection_mask.sum()),
        "selection_first_date": frame.loc[selection_mask, "date"].min().date().isoformat(),
        "selection_last_date": frame.loc[selection_mask, "date"].max().date().isoformat(),
    }
    return frame, audit


def _base_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    signals = {
        "GLOBAL_SELLOFF_TOP05": frame["risk_rolling252_global_selloff1_baseline"] >= 0.95,
        "RF_SIGNED_TOP03": frame["risk_rolling252_rf_signed_all"] >= 0.97,
        "HGB_SIGNED_TOP04": frame["risk_rolling252_hgb_signed_all"] >= 0.96,
        "GBR_Q90_LOSS_TOP04": frame["risk_rolling252_gbr_q90_loss_all"] >= 0.96,
        "ELASTIC_SIGNED_TOP01": frame["risk_rolling252_elastic_signed_all"] >= 0.99,
        "VIX_LEVEL_TOP05": frame["risk_rolling252_vix_level"] >= 0.95,
        "CONTAGION_JOINT_TOP05": frame["risk_rolling252_contagion_joint"] >= 0.95,
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in signals.items()}


def _build_rule_frontier(
    frame: pd.DataFrame,
    signals: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    rules: dict[str, dict[str, Any]] = {}

    def register(family: str, members: tuple[str, ...], minimum_votes: int) -> None:
        short = "__".join(members)
        name = f"{family}_V{minimum_votes}_{short}"
        votes = pd.concat([signals[member] for member in members], axis=1).sum(axis=1)
        decision = votes >= minimum_votes
        rules[name] = {
            "family": family,
            "members": list(members),
            "minimum_votes": minimum_votes,
            "decision": decision.astype(bool),
        }

    for name in signals:
        register("INDIVIDUAL", (name,), 1)
    names = tuple(signals)
    for size in [2, 3, 4, 5]:
        for members in itertools.combinations(names, size):
            register("UNION", members, 1)
    for size in range(3, len(names) + 1):
        for members in itertools.combinations(names, size):
            register("CONSENSUS2", members, 2)
    for size in range(4, len(names) + 1):
        for members in itertools.combinations(names, size):
            register("CONSENSUS3", members, 3)

    total_avoidable = float(frame.loc[selection_mask, "avoidable_loss1"].sum())
    rows: list[dict[str, Any]] = []
    for name, rule in rules.items():
        selected = rule["decision"] & selection_mask
        advantage = frame.loc[selected, "signed_cash_advantage1"]
        selected_loss = frame.loc[selected, "avoidable_loss1"]
        state = selected.loc[selection_mask].astype(np.int8)
        transitions = int(state.diff().abs().fillna(state.iloc[0]).sum())
        rows.append(
            {
                "rule": name,
                "family": rule["family"],
                "members": json.dumps(rule["members"], ensure_ascii=False),
                "minimum_votes": int(rule["minimum_votes"]),
                "selected_day_count": int(selected.sum()),
                "gross_signed_cash_advantage1_sum": float(advantage.sum()),
                "gross_signed_cash_advantage1_mean": _safe_float(advantage.mean()),
                "gross_positive_day_rate": _safe_float((advantage > 0.0).mean()),
                "avoidable_loss1_capture": _safe_float(
                    float(selected_loss.sum()) / total_avoidable if total_avoidable > 0 else np.nan
                ),
                "signal_transition_count": transitions,
            }
        )
    frontier = pd.DataFrame(rows).sort_values(
        ["gross_signed_cash_advantage1_sum", "signal_transition_count"],
        ascending=[False, True],
    )
    frontier.reset_index(drop=True, inplace=True)
    frontier["gross_rank"] = np.arange(1, len(frontier) + 1)
    mandatory = frontier["family"].eq("INDIVIDUAL")
    frontier["selected_for_exact_account"] = (
        (frontier["gross_rank"] <= EXACT_FRONTIER_LIMIT) | mandatory
    )
    return frontier, rules


def main() -> int:
    required = [DIRECT_FILE, GLOBAL_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, frame_audit = _load_frame()
    signals = _base_signals(frame)
    frontier, rules = _build_rule_frontier(frame, signals)
    selected_rules = frontier[frontier["selected_for_exact_account"]].copy()

    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    signal_output_columns: list[str] = []
    for _, frontier_row in selected_rules.iterrows():
        rule_name = str(frontier_row["rule"])
        signal_column = f"signal_{int(frontier_row['gross_rank']):03d}"
        frame[signal_column] = rules[rule_name]["decision"].astype(float)
        signal_output_columns.append(signal_column)
        policy_name = f"POSITIVE_UNION_{int(frontier_row['gross_rank']):03d}_{frontier_row['family']}"
        spec = PolicySpec("PULSE", signal_column, 0.5)
        for perturbation in START_PERTURBATIONS:
            metrics, decisions = _evaluate_policy_start(
                frame,
                dividends,
                policy_name,
                spec,
                perturbation,
            )
            metrics.update(
                {
                    "combination_rule": rule_name,
                    "combination_family": str(frontier_row["family"]),
                    "members": str(frontier_row["members"]),
                    "minimum_votes": int(frontier_row["minimum_votes"]),
                    "gross_rank": int(frontier_row["gross_rank"]),
                }
            )
            metrics, decisions = _add_one_day_diagnostics(metrics, decisions, frame)
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    metadata = metrics.groupby("policy", sort=False).first()[
        ["combination_rule", "combination_family", "members", "minimum_votes", "gross_rank"]
    ]
    one_day = metrics.groupby("policy", sort=False)[
        ["bad1_recall", "good1_false_exit", "avoidable_loss1_capture", "selected_signed_cash_advantage1_sum"]
    ].median()
    for aggregate in aggregates:
        meta = metadata.loc[aggregate["policy"]]
        aggregate.update(
            {
                "combination_rule": str(meta["combination_rule"]),
                "combination_family": str(meta["combination_family"]),
                "members": json.loads(str(meta["members"])),
                "minimum_votes": int(meta["minimum_votes"]),
                "gross_rank": int(meta["gross_rank"]),
                "bad1_recall_median": _safe_float(one_day.loc[aggregate["policy"], "bad1_recall"]),
                "good1_false_exit_median": _safe_float(
                    one_day.loc[aggregate["policy"], "good1_false_exit"]
                ),
                "avoidable_loss1_capture_median": _safe_float(
                    one_day.loc[aggregate["policy"], "avoidable_loss1_capture"]
                ),
                "selected_signed_cash_advantage1_sum_median": _safe_float(
                    one_day.loc[aggregate["policy"], "selected_signed_cash_advantage1_sum"]
                ),
            }
        )
    best = aggregates[0]
    passed = bool(best["development_gate"])

    output_columns = [
        "date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "one_day_open_total_return",
        "signed_cash_advantage1",
        "avoidable_loss1",
        "avoidable_loss5",
        *signal_output_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(frontier, OUTPUT_FRONTIER)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    _atomic_parquet(decisions, OUTPUT_DECISIONS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DAILY_POSITIVE_SIGNAL_UNION_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "decision_frequency": "每交易日",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "base_signal_count": int(len(signals)),
            "combination_families": ["INDIVIDUAL", "UNION", "CONSENSUS2", "CONSENSUS3"],
            "gross_frontier_rule_count": int(len(frontier)),
            "exact_account_frontier_limit_plus_individuals": EXACT_FRONTIER_LIMIT,
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "base_signals": list(signals),
        "best_gross_combination": frontier.iloc[0].to_dict(),
        "best_development_policy": best,
        "all_exact_policy_aggregates": aggregates,
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "combination_gross_frontier": str(OUTPUT_FRONTIER.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "正边际信号组合通过开发门槛；冻结成员、投票规则、输入哈希与成本后再读取独立验证。"
            if passed
            else "正边际信号的并集与共识仍未达到20%开发门槛；保留组合前沿并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "combination_rule_count": len(frontier),
                "exact_rule_count": len(aggregates),
                "best_gross_combination": payload["best_gross_combination"],
                "best_development_policy": best,
                "validation_2021_plus_loaded": False,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
