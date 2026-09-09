"""只读已保存候选，合并重复引用并定位四情景薄弱项。"""
import json
import math
from collections import defaultdict
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_saved_candidate_frontier_20260909"
PROTOCOL = ROOT / "docs/510300_SAVED_CANDIDATE_FRONTIER_DIAGNOSTIC_20260909.md"
FIELDS = ["net_sharpe", "annualized_return", "cumulative_return", "max_drawdown", "trading_days", "trade_count", "commission", "slippage_cost", "annualized_return_excess_vs_buy_hold"]
NORMAL = {"", "OPEN_ONLY", "LEGACY_OPEN", "CLOSE_NEXT_OPEN", "BUY_HOLD"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def signature(rows):
    normalized = [{k: round(float(r[k]), 6 if k in {"commission", "slippage_cost"} else 12) if r.get(k) is not None and math.isfinite(float(r[k])) else None for k in FIELDS} for r in rows]
    return json.dumps(normalized, sort_keys=True)


def table(folder, result, filename, key, fallback):
    path = folder / filename
    if path.is_file():
        return pd.read_csv(path), path, "FULL_SAVED_CSV"
    rows = result.get(key, [])
    if rows:
        return pd.DataFrame(rows), folder / "result.json", "FULL_SAVED_RESULT_LIST"
    if fallback and isinstance(result.get("primary"), list):
        return pd.DataFrame(result["primary"]), folder / "result.json", "PRIMARY_SUMMARY_ONLY"
    return pd.DataFrame(), folder / "result.json", "NO_METRIC_ROWS"


def main():
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 130, "诊断只覆盖已完成第130轮的固定索引")
    OUT.mkdir()
    write_json(OUT / "READ_STARTED.json", {"started_at": now(), "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "index_sha256": digest(index_path), "new_models_or_accounts": 0}, exclusive=True)
    sources = [(r["round"], r["title"], ROOT / r["result"]) for r in index["completed_rounds"]]
    parent22 = next(p for n, _, p in sources if n == 22)
    for s in json.loads(parent22.read_text(encoding="utf-8"))["completed_stages"]:
        p = ROOT / s["result"]["path"]
        if p.name == "result.json":
            sources.append((22, s["stage"], p))
    versions, coverage, files, observations = defaultdict(dict), [], {}, 0
    for number, title, path in sources:
        result = json.loads(path.read_text(encoding="utf-8"))
        main_frame, main_path, main_scope = table(path.parent, result, "metrics.csv", "all_metrics", True)
        early_frame, early_path, early_scope = table(path.parent, result, "earlier_diagnostics.csv", "earlier_diagnostics", False)
        for source in {path, main_path, early_path}:
            files[str(source.relative_to(ROOT))] = {"path": str(source.relative_to(ROOT)), "sha256": digest(source)}
        coverage.append({"round": number, "title": title, "source": str(path.relative_to(ROOT)), "main_scope": main_scope,
            "main_rows": len(main_frame), "earlier_scope": early_scope, "earlier_rows": len(early_frame)})
        if main_frame.empty:
            continue
        for frame in [main_frame, early_frame]:
            if frame.empty:
                continue
            frame["original_assumption"] = frame["assumption"].fillna("") if "assumption" in frame else ""
            frame["comparison_assumption"] = frame.original_assumption.map(lambda s: "NEXT_OPEN" if s in NORMAL else str(s))
        for (model, assumption), group in main_frame.groupby(["model", "comparison_assumption"]):
            pair = [clean(group[group.cost.eq(c)].iloc[0].to_dict()) for c in ["BASE", "STRESS"] if len(group[group.cost.eq(c)]) == 1]
            if len(pair) != 2:
                coverage.append({"round": number, "title": title, "source": str(path.relative_to(ROOT)), "model": model, "main_scope": "COST_PAIR_NOT_UNIQUE_OR_INCOMPLETE"})
                continue
            early = early_frame[early_frame.model.eq(model) & early_frame.comparison_assumption.eq(assumption)] if not early_frame.empty else early_frame
            earlier_pair = [clean(early[early.cost.eq(c)].iloc[0].to_dict()) for c in ["BASE", "STRESS"] if not early.empty and len(early[early.cost.eq(c)]) == 1]
            if len(earlier_pair) != 2:
                earlier_pair = []
            observations += 1
            key = (model, assumption, signature(pair))
            early_key = signature(earlier_pair) if earlier_pair else "UNKNOWN"
            reference = {"round": number, "title": title, "folder": str(path.parent.relative_to(ROOT)), "main_source": str(main_path.relative_to(ROOT)),
                "earlier_source": str(early_path.relative_to(ROOT)) if earlier_pair else None, "original_assumption": pair[0].get("original_assumption", "")}
            if early_key not in versions[key]:
                versions[key][early_key] = {"model": model, "name": pair[0].get("name") or model, "assumption": assumption,
                    "main": pair, "earlier": earlier_pair, "sources": []}
            versions[key][early_key]["sources"].append(reference)
    candidates = []
    for alternatives in versions.values():
        known = [k for k in alternatives if k != "UNKNOWN"]
        if "UNKNOWN" in alternatives and len(known) == 1:
            alternatives[known[0]]["sources"].extend(alternatives.pop("UNKNOWN")["sources"])
        candidates.extend(alternatives.values())
    rows = []
    for i, c in enumerate(candidates):
        c["candidate_id"] = f"SAVED_{i+1:04d}"
        row = {"candidate_id": c["candidate_id"], "model": c["model"], "name": c["name"], "assumption": c["assumption"],
            "source_rounds": ",".join(map(str, sorted({s["round"] for s in c["sources"]}))), "source_references": len(c["sources"])}
        measurements = c["main"]+c["earlier"]
        complete = len(measurements) == 4
        same_days = [m.get("trading_days") for m in measurements] == [1604, 1604, 1219, 1219]
        defined = complete and all(m.get("net_sharpe") is not None for m in measurements)
        row["status"] = "COMPLETE_COMPARABLE" if complete and same_days and defined else "EARLIER_NOT_AVAILABLE" if not complete else "DIFFERENT_CALENDAR_LENGTH" if not same_days else "UNDEFINED_SHARPE"
        for tag, measurement in zip(["main_base", "main_stress", "earlier_base", "earlier_stress"], measurements):
            row.update({f"{tag}_{k}": measurement.get(k) for k in FIELDS})
        if row["status"] == "COMPLETE_COMPARABLE":
            row["minimum_sharpe"] = min(m["net_sharpe"] for m in measurements)
            excess = [m.get("annualized_return_excess_vs_buy_hold") for m in measurements]
            row["minimum_annualized_excess"] = min(excess) if all(x is not None for x in excess) else None
            row["all_four_positive_excess"] = all(x is not None and x > 0 for x in excess)
            row["all_four_sharpe_at_least_1_2"] = row["minimum_sharpe"] >= 1.2
        rows.append(row)
    frame = pd.DataFrame(rows)
    ranked = frame[frame.status.eq("COMPLETE_COMPARABLE") & frame.assumption.eq("NEXT_OPEN")].sort_values(["minimum_sharpe", "candidate_id"], ascending=[False, True])
    primary = frame[frame.assumption.eq("NEXT_OPEN") & frame.main_base_trading_days.eq(1604)].sort_values("main_base_net_sharpe", ascending=False)
    for name, data in [("all_saved_candidates.csv", frame), ("standard_four_scenario_ranking.csv", ranked), ("main_base_only_ranking.csv", primary), ("source_coverage.csv", pd.DataFrame(coverage))]:
        data.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write_json(OUT / "candidate_sources.json", clean(candidates), exclusive=True)
    write_json(OUT / "source_files.json", list(files.values()), exclusive=True)
    summary = {"completed_at": now(), "status": "SAVED_METRIC_FRONTIER_COMPUTED_TOP_LEDGER_CHECK_PENDING", "through_round": 130,
        "source_result_documents": len(sources), "named_cost_pair_observations": observations, "distinct_saved_named_performance_versions": len(frame),
        "standard_four_scenario_comparisons": len(ranked), "statuses": frame.status.value_counts().to_dict(),
        "all_four_sharpe_at_least_1_2": int(ranked.all_four_sharpe_at_least_1_2.sum()),
        "all_four_positive_excess": int(ranked.all_four_positive_excess.sum()), "new_models_or_accounts": 0,
        "not_new_strategy_round": True, "goal_achieved": False, "security_audit_performed": False}
    write_json(OUT / "result.json", summary, exclusive=True)
    columns = ["candidate_id", "model", "source_rounds", "main_base_net_sharpe", "main_stress_net_sharpe", "earlier_base_net_sharpe", "earlier_stress_net_sharpe", "minimum_sharpe", "minimum_annualized_excess", "all_four_positive_excess"]
    print(json.dumps({"覆盖": summary, "四情景前十二": clean(ranked.head(12)[columns].to_dict("records")),
        "主基础前五": clean(primary.head(5)[columns].to_dict("records"))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
