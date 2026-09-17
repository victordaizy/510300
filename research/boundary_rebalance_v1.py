"""冻结已有持仓的边界再平衡规则，计算四条独立实际资金路径。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.boundary_rebalance_inputs_v1 import boundary_rebalance_frames, PRIMARY, CANDIDATES, MODELS
from research.boundary_rebalance_account_v1 import simulate_boundary_account
from research.saved_target_custom_account_runner_v1 import run_saved_target_custom_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_boundary_rebalance_v1"
CONFIG = ROOT / "config/510300_boundary_rebalance_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PARENTS = {MODELS[0]: P143}
CONTROLS = {MODELS[0]: (P143, "第143轮原中心再平衡"), "VINTAGE_REFERENCE_RISK": (P131, "第131轮原进入模型与普通波动预算"),
            "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第154轮边界再平衡已经冻结")
    old_path = ROOT / "config/510300_vortex_risk_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "边界再平衡六项必要测试尚未通过")
    cfg.update(study_id="510300_BOUNDARY_REBALANCE_V1", round=154, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), decision_clock="15:05:00", boundary_equality_tolerance=1e-12,
        registered_at=now(), new_model_fits=0, new_reference_accounts=0, parent_models=MODELS,
        rebalance_destination="NEAREST_BOUNDARY_FOR_EXISTING_POSITIVE_HOLDINGS", empty_positive_entry="ORIGINAL_PARENT_TARGET_CENTER",
        explicit_zero="FULL_EXIT_OVERRIDES_INTERVAL", interval_boundary="INCLUSIVE_WITH_FIXED_NUMERIC_EQUALITY_TOLERANCE",
        missing_target="NO_NEW_REQUEST_KEEP_ACTUAL_SHARES", second_weight_band_filter=False,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_BOUNDARY_REBALANCE_V1.md", source_budget_cny=0, independent_validation="NOT_ESTABLISHED",
        goal_achieved=False, position_impact=0, previous_goal_turn_classification="PROGRESS_ROUND153_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    names = ["config/510300_heikin_price_state_v1.json", "config/510300_trend_noise_reference_blend_v1.json",
        "research/boundary_rebalance_inputs_v1.py", "research/boundary_rebalance_account_v1.py",
        "research/saved_target_custom_account_runner_v1.py", "research/saved_target_batch_runner_v1.py",
        "research/saved_parent_target_alignment_v1.py", "research/event_clock_account_v1.py",
        "research/adaptive_allocation_v1.py", "research/intraday_overnight_increment_v1.py",
        "tests/test_boundary_rebalance_v1.py", "tests/test_trend_reference_router_v1.py", "scripts/prepare_boundary_account_20260909.py",
        "docs/510300_BOUNDARY_REBALANCE_NEXT_20260909.md", "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md", "config/510300_research_authority_v6.json"]
    paths = [Path(__file__), old_path, *(ROOT / name for name in names), OUT / "tests_receipt.json",
        *(ROOT / cfg[key] for key in ["features", "dividends", "rules"]), P143 / "saved_verification_receipt.json", P131 / "saved_verification_receipt.json"]
    known = {str(Path(row["path"])): row["sha256"] for row in old["frozen_files"]}
    parent_binding = json.loads((ROOT / "config/510300_heikin_price_state_v1.json").read_text(encoding="utf-8"))
    known.update({str(Path(row["path"])): row["sha256"] for row in parent_binding["frozen_files"]})
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known[str(Path(cfg[key]))], "边界研究行情或分红版本改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            path = P143 / period / cost / f"{MODELS[0]}_decisions.parquet"
            require(digest(path) == known[str(path.relative_to(ROOT))], "边界研究143父目标版本不同")
            source = pd.read_parquet(path, columns=["origin_index", "origin", "execution_date"])
            require(np.array_equal(source.origin_index, indices) and pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "边界研究143父目标时钟不同")
            paths.append(path)
            for model, (folder, _) in CONTROLS.items():
                path = folder / period / cost / f"{model}_ledger.parquet"
                require(digest(path) == known[str(path.relative_to(ROOT))], "边界研究保存对照版本不同")
                paths.append(path)
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第154轮一套最近边界再平衡规则已冻结，尚未生成新账户。", flush=True)


def run():
    return run_saved_target_custom_account(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, boundary_rebalance_frames, simulate_boundary_account)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
