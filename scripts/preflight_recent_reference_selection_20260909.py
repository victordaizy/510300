"""仅核对月度选择所需的保存收益、目标与终点时钟，不计算排名。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/510300_equal_reference_pair_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    old_hashes = {str(Path(v["path"])): v["sha256"] for v in cfg["frozen_files"]}
    old_preflight = root / "reports/research/510300_equal_reference_pair_preflight_20260909/result.json"
    source_check = json.loads(old_preflight.read_text(encoding="utf-8"))
    paths = [cfg_path, old_preflight, root / "docs/510300_RECENT_REFERENCE_SELECTION_NEXT_20260909.md",
        root / "research/past_sharpe_parent_selector_inputs_v1.py", root / "config/510300_past_sharpe_parent_selector_v1.json",
        root / "docs/510300_PAST_SHARPE_PARENT_SELECTOR_V1.md", root / "reports/research/510300_equal_reference_pair_v1/saved_verification_receipt.json"]
    for item in source_check["sources"]:
        require(digest(root / item["path"]) == item["sha256"], "下一月度选择来源相对各半预检发生改变")
        paths.append(root / item["path"])
    data = pd.read_parquet(root / cfg["features"])
    models = {"DOWNSIDE_REFERENCE_RISK": root / "reports/research/510300_downside_reference_risk_v1",
        "CONTINUOUS_REFERENCE_MIN_VARIANCE": root / "reports/research/510300_continuous_reference_min_variance_v1"}
    returns_coverage, targets_coverage = [], []
    for period, start, end in [("evaluation", cfg["evaluation_start"], cfg["data_cutoff"]),
        ("earlier_diagnostic", cfg["earlier_start"], cfg["earlier_terminal"])]:
        frame = data[data.date.le(end)]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for model, folder in models.items():
            path = folder / period / "BASE" / f"{model}_ledger.parquet"
            require(digest(path) == old_hashes[str(path.relative_to(root))], "月度评分收益不再等于134已绑定的来源")
            paths.append(path)
            ledger = pd.read_parquet(path)
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "月度评分收益日期不完整")
            require(np.isfinite(ledger.net_return).all() and ledger.net_return.gt(-1).all(), "月度评分收益缺失或非法")
            require(ledger.mark_clock.iloc[:-1].eq("CLOSE").all() and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL", "月度评分收益时钟与末日开盘不匹配")
            returns_coverage.append({"period": period, "model": model, "cost": "BASE", "ledger_rows": len(ledger),
                "known_close_rows": len(ledger)-1, "terminal_open_rows": 1, "first_date": str(ledger.date.iloc[0].date()),
                "last_close_date": str(ledger.date.iloc[-2].date()), "terminal_date": str(ledger.date.iloc[-1].date())})
            for cost in cfg["costs"]:
                target_path = folder / period / cost / f"{model}_decisions.parquet"
                target = pd.read_parquet(target_path)
                require(np.array_equal(target.origin_index, indices), "月度选择所读父目标索引不同")
                require(pd.DatetimeIndex(target.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "月度选择父目标收盘不同")
                require(pd.DatetimeIndex(target.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "月度选择父目标不是下一开盘")
                targets_coverage.append({"period": period, "model": model, "cost": cost, "known_target_origins": len(target)})
    out = root / "reports/research/510300_recent_reference_selection_preflight_20260909/result.json"
    result = {"checked_at": now(), "status": "BASE_SCORE_RETURNS_MATCHED_TARGETS_AND_CLOSE_TERMINAL_CLOCKS_AVAILABLE",
        "proposed_round": 135, "registered": False, "return_sources": returns_coverage, "target_sources": targets_coverage,
        "new_scores_computed": 0, "new_selections_computed": 0, "new_targets_computed": 0, "new_models_fit": 0, "new_accounts_generated": 0,
        "sources": [{"path": str(p.relative_to(root)), "sha256": digest(p)} for p in sorted(set(paths))], "reviewer_source_sha256": digest(Path(__file__))}
    write_json(out, result, exclusive=True)
    index_path = root / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 134, "下一选择预检不能覆盖更晚轮次")
    index["next_work"].update(status="RECENT_REFERENCE_SELECTION_INPUT_CLOCK_PREFLIGHT_COMPLETE", preflight=str(out.relative_to(root)), registered=False)
    write_json(index_path, index)
    print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "target_sources"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
