"""第151轮共同冻结并批量运行两个平均K线候选。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.heikin_price_state_inputs_v1 import heikin_candles, heikin_price_frames, PRIMARY, CANDIDATES, MODELS
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_heikin_price_state_v1"
CONFIG = ROOT / "config/510300_heikin_price_state_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P150 = ROOT / "reports/research/510300_monotone_episode_budget_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PARENTS = {MODELS[0]: P143}
CONTROLS = {MODELS[0]: (P143, "第143轮趋势波动连续预算"),
    "EPISODE_BUDGET_NONINCREASING": (P150, "第150轮预算只减不增"),
    "EPISODE_BUDGET_NONDECREASING": (P150, "第150轮预算只增不减"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "平均K线批次已经冻结")
    old_path = ROOT / "config/510300_monotone_episode_budget_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "平均K线六项必要测试未通过")
    cfg.update(study_id="510300_HEIKIN_PRICE_STATE_V1", round=151, primary=PRIMARY, candidate_configurations=2,
        candidate_models=list(CANDIDATES), numeric_tolerance=1e-12, decision_clock="15:05:00", combination="HEIKIN_STANDALONE_AND_REFERENCE_CONFIRMATION",
        registered_at=now(), new_model_fits=0, new_reference_accounts=0, reference_cost_matching="SAME_PERIOD_AND_SAME_COST",
        parent_models=MODELS, outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_HEIKIN_PRICE_STATE_V1.md", source_budget_cny=0, independent_validation="NOT_ESTABLISHED",
        goal_achieved=False, position_impact=0, previous_goal_turn_classification="PROGRESS_ROUND150_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), old_path, ROOT / "config/510300_trend_noise_reference_blend_v1.json",
        ROOT / "research/heikin_price_state_inputs_v1.py", ROOT / "research/saved_target_batch_runner_v1.py",
        ROOT / "research/saved_parent_target_alignment_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_heikin_price_state_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "docs/510300_HEIKIN_PRICE_STATE_NEXT_20260909.md", ROOT / "config/510300_research_authority_v6.json",
        P143 / "saved_verification_receipt.json", P150 / "saved_verification_receipt.json"]
    known_hashes = {str(Path(row["path"])): row["sha256"] for row in old["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known_hashes[str(Path(cfg[key]))], "批次已有行情或分红改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            for model, folder in PARENTS.items():
                path = folder / period / cost / f"{model}_decisions.parquet"
                require(digest(path) == known_hashes[str(path.relative_to(ROOT))], "平均K线父目标版本改变")
                source = pd.read_parquet(path, columns=["origin_index", "origin", "execution_date"])
                require(np.array_equal(source.origin_index, indices) and pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                    pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "平均K线父目标时钟不同")
                paths.append(path)
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    checked = heikin_candles(data, cfg["numeric_tolerance"])
    cfg["existing_input_checks"] = {"full_calendar_rows": len(checked), "known_candle_directions": int(checked.heikin_direction.notna().sum()),
        "economic_close_matches_saved_wealth": True, "synthetic_prices_used_for_execution": False}
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第151轮两种平均K线使用方式已共同冻结，尚未生成新目标或账户。", flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, heikin_price_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
