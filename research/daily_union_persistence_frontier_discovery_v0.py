"""检验高质量逐日组合信号在固定持续期下的净超额。"""

from __future__ import annotations

import json
from typing import Any

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
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)


SOURCE_RANKS = [1, 5, 6, 8, 27, 31]
HOLD_DAYS = [2, 3, 5, 10]

SOURCE_FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "510300_daily_positive_signal_union_discovery_v0.parquet"
SOURCE_REPORT_FILE = PROJECT_ROOT / "reports" / "discovery" / "510300_daily_positive_signal_union_discovery_v0.json"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_union_persistence_frontier_discovery_v0.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_union_persistence_frontier_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_union_persistence_frontier_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_daily_union_persistence_frontier_discovery_v0.json"
)


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> pd.DataFrame:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    signal_columns = [f"signal_{rank:03d}" for rank in SOURCE_RANKS]
    source = pd.read_parquet(
        SOURCE_FEATURE_FILE,
        columns=[
            "date",
            "one_day_open_total_return",
            "signed_cash_advantage1",
            "avoidable_loss1",
            "avoidable_loss5",
            *signal_columns,
        ],
    )
    source = _normalize_dates(source)
    frame = market.merge(source, on="date", how="inner", validate="one_to_one")
    return frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)


def _source_metadata() -> dict[int, dict[str, Any]]:
    payload = json.loads(SOURCE_REPORT_FILE.read_text(encoding="utf-8"))
    metadata: dict[int, dict[str, Any]] = {}
    for item in payload["all_exact_policy_aggregates"]:
        rank = int(item["gross_rank"])
        if rank in SOURCE_RANKS:
            metadata[rank] = {
                "combination_rule": item["combination_rule"],
                "combination_family": item["combination_family"],
                "members": item["members"],
                "minimum_votes": item["minimum_votes"],
            }
    missing = sorted(set(SOURCE_RANKS) - set(metadata))
    if missing:
        raise ValueError(f"来源报告缺少组合排名：{missing}")
    return metadata


def _policies() -> dict[str, tuple[int, PolicySpec, str]]:
    policies: dict[str, tuple[int, PolicySpec, str]] = {}
    for rank in SOURCE_RANKS:
        column = f"signal_{rank:03d}"
        policies[f"PERSIST_R{rank:03d}_PULSE"] = (rank, PolicySpec("PULSE", column, 0.5), "PULSE")
        for days in HOLD_DAYS:
            policies[f"PERSIST_R{rank:03d}_HOLD{days:02d}"] = (
                rank,
                PolicySpec("FIXED_HOLD", column, 0.5, hold_days=days),
                f"HOLD{days:02d}",
            )
        for days in [3, 5]:
            policies[f"PERSIST_R{rank:03d}_CROSS_HOLD{days:02d}"] = (
                rank,
                PolicySpec("FIXED_HOLD", column, 0.5, hold_days=days, crossing_only=True),
                f"CROSS_HOLD{days:02d}",
            )
    return policies


def main() -> int:
    required = [SOURCE_FEATURE_FILE, SOURCE_REPORT_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame = _load_frame()
    metadata = _source_metadata()
    policies = _policies()
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for policy_name, (rank, spec, persistence) in policies.items():
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
                    "source_gross_rank": rank,
                    "persistence": persistence,
                    **metadata[rank],
                }
            )
            metrics, decisions = _add_one_day_diagnostics(metrics, decisions, frame)
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    aggregate_meta = metrics.groupby("policy", sort=False).first()[
        [
            "source_gross_rank",
            "persistence",
            "combination_rule",
            "combination_family",
            "members",
            "minimum_votes",
        ]
    ]
    one_day = metrics.groupby("policy", sort=False)[
        ["bad1_recall", "good1_false_exit", "avoidable_loss1_capture", "selected_signed_cash_advantage1_sum"]
    ].median()
    for aggregate in aggregates:
        meta = aggregate_meta.loc[aggregate["policy"]]
        aggregate.update(
            {
                "source_gross_rank": int(meta["source_gross_rank"]),
                "persistence": str(meta["persistence"]),
                "combination_rule": str(meta["combination_rule"]),
                "combination_family": str(meta["combination_family"]),
                "members": meta["members"],
                "minimum_votes": int(meta["minimum_votes"]),
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

    _atomic_parquet(frame, OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    _atomic_parquet(decisions, OUTPUT_DECISIONS)
    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DAILY_UNION_PERSISTENCE_FRONTIER_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "source_gross_ranks": SOURCE_RANKS,
            "persistence_families": ["PULSE", "HOLD02", "HOLD03", "HOLD05", "HOLD10", "CROSS_HOLD03", "CROSS_HOLD05"],
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "policy_count": int(len(policies)),
        "best_development_policy": best,
        "all_policy_aggregates": aggregates,
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "组合信号持续期通过开发门槛；冻结组合、持续期、输入哈希与成本后再读取独立验证。"
            if passed
            else "组合信号延长空仓仍未达到20%开发门槛；保留持续期前沿并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "policy_count": len(policies),
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
