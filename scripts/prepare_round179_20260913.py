"""冻结第179轮顺序状态选择策略并运行四个完整账户。"""
import json
from pathlib import Path

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.sequential_regime_strategy_selection_inputs_v1 import CANDIDATES, MODELS, PRIMARY, sequential_regime_strategy_selection_frames


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sequential_regime_strategy_selection_v1"
CONFIG = ROOT / "config/510300_sequential_regime_strategy_selection_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P168 = ROOT / "reports/research/510300_runs_opportunity_union_v1"
P174 = ROOT / "reports/research/510300_return_confirmation_auxiliary_batch_v1"
P165 = ROOT / "reports/research/510300_return_runs_state_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PARENTS = {MODELS[0]: P143, MODELS[1]: P168}
CONTROLS = {
    "EITHER_CONFIRMED_RUNS_AUXILIARY": (P174, "第174轮任一方向确认"),
    MODELS[0]: (P143, "第143轮核心"),
    MODELS[1]: (P168, "第168轮相加封顶"),
    "RETURN_RUNS_STATE": (P165, "第165轮辅助"),
    "BUY_HOLD": (P32, "买入持有"),
}


def freeze():
    require(not CONFIG.exists(), "第179轮已经冻结")
    previous = json.loads((ROOT / "config/510300_early_selected_regime_mapping_v1.json").read_text(encoding="utf-8"))
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 3, "第179轮必要测试未通过")
    keys = [
        "evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start",
        "earlier_terminal", "weight_band",
    ]
    cfg = {key: previous[key] for key in keys}
    cfg.update(
        study_id="510300_SEQUENTIAL_REGIME_STRATEGY_SELECTION_V1", round=179, primary=PRIMARY,
        candidate_configurations=1, candidate_models=list(CANDIDATES), registered_at=now(), decision_clock="15:05:00",
        parent_models=MODELS, selection_lookback=504, minimum_state_samples=60, minimum_candidate_sharpe=1.2,
        minimum_candidate_annual_return=0.10, annual_return_target=0.10, new_model_fits=0,
        new_reference_accounts=0, source_budget_cny=0, goal_achieved=False, position_impact=0,
        rules="docs/510300_SEQUENTIAL_REGIME_STRATEGY_SELECTION_RULES_20260913.md",
        independent_validation="NOT_ESTABLISHED", outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE",
        reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
    )
    paths = [
        Path(__file__), ROOT / "research/sequential_regime_strategy_selection_inputs_v1.py",
        ROOT / "research/saved_target_batch_runner_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/saved_target_account_checks_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_sequential_regime_strategy_selection_v1.py", OUT / "tests_receipt.json", ROOT / cfg["features"],
        ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "config/510300_research_authority_v6.json",
    ]
    for model, folder in PARENTS.items():
        parent_config = ROOT / "config" / f"{folder.name}.json"
        paths += [parent_config, ROOT / json.loads(parent_config.read_text(encoding="utf-8"))["rules"], folder / "saved_verification_receipt.json"]
        paths += [folder / period / cost / f"{model}_decisions.parquet" for period in ["evaluation", "earlier_diagnostic"] for cost in cfg["costs"]]
        paths += [folder / period / cost / f"{model}_ledger.parquet" for period in ["evaluation", "earlier_diagnostic"] for cost in cfg["costs"]]
    for model, (folder, _) in CONTROLS.items():
        paths += [folder / period / cost / f"{model}_ledger.parquet" for period in ["evaluation", "earlier_diagnostic"] for cost in cfg["costs"]]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第179轮已冻结。")


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, sequential_regime_strategy_selection_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
