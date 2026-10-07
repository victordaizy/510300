"""只读筛查既有账户，保留原裁决，保存可在独立目录重算的证据快照。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_saved_candidates_sharpe13_v1"
OUT = ROOT / "reports" / "research" / STUDY
CONTRACT = ROOT / "config" / f"{STUDY}.json"
PAYLOAD: dict[str, bytes] = {}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(path: Path) -> bytes:
    key = path.relative_to(ROOT).as_posix()
    if key not in PAYLOAD:
        PAYLOAD[key] = path.read_bytes()
    return PAYLOAD[key]


def json_source(path: Path) -> dict:
    return json.loads(snapshot(path).decode("utf-8-sig")) if path.exists() else {}


def rows_source(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(io.StringIO(snapshot(path).decode("utf-8-sig"))))


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def point_pass(row: dict, contract: dict) -> bool:
    sr, dd = number(row.get("net_sharpe")), number(row.get("max_drawdown"))
    return sr is not None and dd is not None and sr >= contract["net_sharpe_minimum"] and -contract["maximum_drawdown_magnitude"] <= dd <= 0


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (OUT / "result.json").exists():
        raise FileExistsError("本轮筛查已完成，不覆盖。")
    OUT.mkdir(parents=True, exist_ok=True)
    contract = json_source(CONTRACT)
    mandate = json_source(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    if contract["net_sharpe_minimum"] != mandate["target_net_sharpe"] or contract["initial_capital_cny"] != mandate["capital_cny"]:
        raise ValueError("本轮合同与当前用户目标不一致。")
    snapshot(Path(__file__).resolve())
    records: list[dict] = []
    sources: list[dict] = []
    for path in sorted((ROOT / "reports" / "research").glob("510300_*/metrics.csv")):
        if path.parent == OUT:
            continue
        rows = rows_source(path)
        config = json_source(ROOT / "config" / f"{path.parent.name}.json")
        result = json_source(path.parent / "result.json")
        acceptance = json_source(path.parent / "acceptance_outcome.json")
        candidates = config.get("candidate_models", result.get("candidate_models", []))
        sources.append({
            "source": path.relative_to(ROOT).as_posix(), "rows": len(rows),
            "bytes": len(snapshot(path)), "sha256": sha256(snapshot(path)),
            "study_status": acceptance.get("status", result.get("status", "UNKNOWN")),
            "independent_validation": result.get("independent_validation", "NOT_ESTABLISHED_IN_THIS_SOURCE"),
        })
        for index, row in enumerate(rows, 2):
            records.append({
                "study": path.parent.name, "source": path.relative_to(ROOT).as_posix(),
                "source_csv_line": index, "model": row.get("model", ""),
                "cost": row.get("cost", ""),
                **{k: row.get(k, "") for k in ("net_sharpe", "max_drawdown", "annualized_return", "trading_days", "mean_exposure", "trade_count")},
                "initial_capital_cny": config.get("initial_capital", "UNKNOWN"),
                "annual_days": config.get("annual_days", "UNKNOWN"),
                "declared_candidate_here": row.get("model") in candidates,
                "numerical_threshold_pass_only": point_pass(row, contract),
                "original_disposition": acceptance.get("status", result.get("status", "UNKNOWN")),
                "independent_validation": result.get("independent_validation", "NOT_ESTABLISHED_IN_THIS_SOURCE"),
            })

    hits = [r for r in records if r["numerical_threshold_pass_only"]]
    unique_keys = {(r["model"], r["cost"], r["net_sharpe"], r["max_drawdown"], r["annualized_return"], r["trading_days"]) for r in hits}
    models = sorted({r["model"] for r in hits})
    comparisons: list[dict] = []
    cycle_rows: list[dict] = []
    for model in models:
        appearances = [r for r in records if r["model"] == model]
        own = sorted({r["study"] for r in appearances if r["declared_candidate_here"]})
        if not own:
            raise ValueError(f"无法定位模型的原始候选研究：{model}")
        if len(own) > 1:
            raise ValueError(f"模型有多个原始候选研究，需明确版本：{model}，{own}")
        directory = ROOT / "reports" / "research" / own[0]
        primary = [r for r in appearances if r["study"] == own[0]]
        by_cost = {r["cost"]: r for r in primary}
        if len(primary) != len(by_cost):
            raise ValueError(f"同一模型成本场景重复：{model}")
        early = [r for r in rows_source(directory / "earlier_diagnostics.csv") if r.get("model") == model]
        earlier_by_cost = {r["cost"]: r for r in early}
        config = json_source(ROOT / "config" / f"{own[0]}.json")
        if config["initial_capital"] != contract["initial_capital_cny"] or config["annual_days"] != contract["trading_days_per_year"]:
            raise ValueError(f"来源资本或年化口径不同：{model}")
        result = json_source(directory / "result.json")
        original_decision = json_source(directory / "acceptance_outcome.json")
        coverage = rows_source(directory / "account_coverage.csv")
        raw_cycles = rows_source(directory / "saved_actual_cycles.csv")
        for optional in ("candidate_outcomes.json", "saved_verification_receipt.json", "era_metrics.csv", "yearly_metrics.csv"):
            p = directory / optional
            if p.exists():
                snapshot(p)
        rules = config.get("rules")
        if isinstance(rules, str) and (ROOT / rules).exists():
            snapshot(ROOT / rules)
        cycles = [r for r in raw_cycles if r.get("model", model) == model and r.get("period") == "evaluation" and r.get("cost") == "STRESS"]
        for row in cycles:
            cycle_rows.append({"model": model, "source_study": own[0], **row})
        annual_days = int(config["annual_days"])
        days = int(by_cost["STRESS"]["trading_days"])
        held = [r for r in coverage if r.get("model", model) == model and r.get("period") == "evaluation" and r.get("cost") == "STRESS"]
        held_closes = int(held[0]["holding_closes"]) if len(held) == 1 else None
        stress_profit = sum(float(r["net_profit"]) for r in cycles)
        best_profit = max((float(r["net_profit"]) for r in cycles), default=0)
        original_stress = next(r for r in rows_source(directory / "metrics.csv") if r["model"] == model and r["cost"] == "STRESS")
        expected_profit = float(original_stress["cumulative_return"]) * float(config["initial_capital"])
        if abs(stress_profit - expected_profit) > 1e-5:
            raise ValueError(f"保存周期利润与账户累计利润不一致：{model}")
        comparisons.append({
            "model": model, "source_study": own[0],
            "capital_cny": config["initial_capital"],
            "main_start": config["evaluation_start"], "main_end": config["data_cutoff"],
            "main_base_sharpe": by_cost["BASE"]["net_sharpe"],
            "main_stress_sharpe": by_cost["STRESS"]["net_sharpe"],
            "main_stress_cagr": by_cost["STRESS"]["annualized_return"],
            "main_stress_drawdown": by_cost["STRESS"]["max_drawdown"],
            "main_base_drawdown": by_cost["BASE"]["max_drawdown"],
            "main_both_costs_numerical_pass": all(point_pass(by_cost[c], contract) for c in ("BASE", "STRESS")),
            "earlier_start": config["earlier_start"], "earlier_end": config["earlier_terminal"],
            "earlier_base_sharpe": earlier_by_cost.get("BASE", {}).get("net_sharpe"),
            "earlier_stress_sharpe": earlier_by_cost.get("STRESS", {}).get("net_sharpe"),
            "earlier_base_drawdown": earlier_by_cost.get("BASE", {}).get("max_drawdown"),
            "earlier_stress_drawdown": earlier_by_cost.get("STRESS", {}).get("max_drawdown"),
            "earlier_both_costs_numerical_pass": all(point_pass(earlier_by_cost.get(c, {}), contract) for c in ("BASE", "STRESS")),
            "stress_cycles": len(cycles), "stress_cycles_per_242_days": len(cycles) * annual_days / days,
            "stress_terminal_exit_cycles": sum(str(r.get("terminal_exit", "")).lower() == "true" for r in cycles),
            "saved_cycle_profit_identity_error_cny": stress_profit - expected_profit,
            "stress_held_closes": held_closes,
            "stress_cash_close_fraction": 1-held_closes/days if held_closes is not None else None,
            "largest_cycle_share_of_total_net_profit": best_profit/stress_profit if stress_profit > 0 else None,
            "cycle_profit_sum": stress_profit,
            "independent_validation": result.get("independent_validation", "NOT_ESTABLISHED"),
            "original_disposition": original_decision.get("status", "UNKNOWN"),
            "original_decision": original_decision.get("decision", ""),
            "promotion": "NOT_PROMOTED_USED_HISTORY_AND_ORIGINAL_CLOSURE_PRESERVED",
        })

    paired = [r for r in comparisons if r["main_both_costs_numerical_pass"]]
    summary = {
        "study_id": contract["study_id"], "completed_at": datetime.now(timezone.utc).isoformat(),
        "stage_status": "COMPLETED_SAVED_RESULT_SCREEN_NO_STRATEGY_PROMOTION",
        "current_target_net_sharpe": contract["net_sharpe_minimum"],
        "current_max_drawdown_magnitude": contract["maximum_drawdown_magnitude"],
        "previous_goal_turn_classification": "PROGRESS_USER_TARGET_CHANGED_AND_RECENT_ACCOUNTS_REASSESSED",
        "scan_scope": "reports/research/510300_*/metrics.csv，一级研究目录保存指标；不是全仓库所有研究的穷尽扫描",
        "metric_files": len(sources), "metric_rows_including_repeated_controls": len(records),
        "numerical_pass_rows_including_repeated_controls": len(hits),
        "deduplicated_passing_model_cost_metric_records": len(unique_keys),
        "models_with_at_least_one_passing_cost": len(models),
        "models_passing_both_main_cost_scenarios": len(paired),
        "paired_models_also_passing_both_earlier_cost_scenarios": sum(r["earlier_both_costs_numerical_pass"] for r in paired),
        "independently_validated_candidates_found_in_screen": 0,
        "new_fits": 0, "new_backtest_accounts": 0, "new_forward_events": 0,
        "goal_achieved": False, "goal_status": "ACTIVE", "position_impact": 0,
        "candidate_results": comparisons,
        "verification_scope": "本轮核对保存指标、去重、原始裁决和保存周期汇总；未重跑原始行情到账户引擎",
    }
    write_csv(OUT / "all_saved_metric_rows.csv", records, list(records[0]))
    write_csv(OUT / "metric_source_inventory.csv", sources, list(sources[0]))
    write_csv(OUT / "deduplicated_candidate_comparison.csv", comparisons, list(comparisons[0]))
    if cycle_rows:
        write_csv(OUT / "passing_model_saved_cycles.csv", cycle_rows, list(dict.fromkeys(k for r in cycle_rows for k in r)))
    write_json(OUT / "result.json", summary)
    snapshot_dir = OUT / "source_snapshot"
    for rel, data in PAYLOAD.items():
        destination = snapshot_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    write_json(OUT / "source_snapshot_manifest.json", [
        {"path": k, "bytes": len(v), "sha256": sha256(v)} for k, v in sorted(PAYLOAD.items())
    ])
    print(json.dumps({k:v for k,v in summary.items() if k != "candidate_results"}, ensure_ascii=False, indent=2))
    print("新目标下主要区间双成本点值过线的候选：")
    for row in paired:
        print(json.dumps({k:row[k] for k in ("model", "main_stress_sharpe", "earlier_stress_sharpe", "stress_cycles", "stress_cycles_per_242_days", "stress_cash_close_fraction", "original_disposition")},ensure_ascii=False))


if __name__ == "__main__":
    main()
