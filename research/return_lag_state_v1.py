"""冻结单一相邻收益相关与累计方向规则，共读来源后运行四个完整账户。"""
import json
from pathlib import Path
from research.return_lag_state_inputs_v1 import PRIMARY, CANDIDATES, return_lag_state_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_return_lag_state_v1"
CONFIG = ROOT / "config/510300_return_lag_state_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P165 = ROOT / "reports/research/510300_return_runs_state_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"TREND_NOISE_REFERENCE_BLEND": (P143, "第143轮趋势波动连续预算"),
            "RETURN_RUNS_STATE": (P165, "第165轮收益强弱连续段"),
            "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第171轮相邻收益相关与累计方向规则已经冻结")
    prior_path = ROOT / "config/510300_runs_closed_cycle_budget_v1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "相邻收益相关与累计方向六项必要测试尚未通过")
    require((ROOT / "reports/research/510300_runs_closed_cycle_budget_v1/acceptance_outcome.json").is_file(), "第170轮尚未完成关闭")
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id="510300_RETURN_LAG_STATE_V1", round=171, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), registered_at=now(), decision_clock="15:05:00", parent_models=[],
        lag_window=60, momentum_window=60, volatility_window=20, risk_target=.1,
        entry_threshold=0., exit_threshold=0., confirmation_closes=2,
        lag_method="PEARSON_CORRELATION_OF_59_ADJACENT_PAIRS_WITHIN_60_COMPLETE_RETURNS",
        constant_pair_method="UNDEFINED_CORRELATION_RESET_STATE_NO_ZERO_FILL",
        undefined_statistic="NO_VIEW_RESET_DIRECTION_AND_COUNTERS",
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_ALL_THREE_COUNTERS",
        direction="TWO_POSITIVE_CORRELATION_AND_MOMENTUM_OR_SEPARATE_NONPOSITIVE_CORRELATION_MOMENTUM_EXIT",
        initial_direction=0, signal_parameters_fitted=False,
        new_model_fits=0, model_fit_count_scope="ZERO_SUPERVISED_MODELS_DESCRIPTIVE_WINDOW_STATISTICS_ONLY",
        new_reference_accounts=0, source_budget_cny=0, goal_achieved=False, position_impact=0,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_RETURN_LAG_STATE_V1.md", independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND170_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), prior_path, ROOT / "research/return_lag_state_inputs_v1.py",
        ROOT / "research/saved_target_batch_runner_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_return_lag_state_v1.py",
        ROOT / "tests/test_heikin_price_state_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "docs/510300_RETURN_LAG_STATE_NEXT_20260910.md", ROOT / "config/510300_research_authority_v6.json",
        P143 / "saved_verification_receipt.json", P165 / "saved_verification_receipt.json",
        ROOT / "reports/research/510300_runs_closed_cycle_budget_v1/acceptance_outcome.json"]
    known = {str(Path(item["path"])): item["sha256"] for item in prior["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known[str(Path(cfg[key]))], "共用行情或分红的已绑定版本改变")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model, (folder, _) in CONTROLS.items():
                path = folder / period / cost / f"{model}_ledger.parquet"
                require(digest(path) == known[str(path.relative_to(ROOT))], "已有对照账户版本改变")
                paths.append(path)
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第171轮一个固定相邻收益相关与累计方向规则已冻结，尚未生成新历史目标或账户。", flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, {}, CONTROLS, return_lag_state_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
