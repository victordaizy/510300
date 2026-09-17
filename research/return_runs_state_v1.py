"""冻结单一收益强弱连续段与上涨动量规则，共读来源后运行四个完整账户。"""
import json
from pathlib import Path
from research.return_runs_state_inputs_v1 import PRIMARY, CANDIDATES, return_runs_state_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_return_runs_state_v1"
CONFIG = ROOT / "config/510300_return_runs_state_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"TREND_NOISE_REFERENCE_BLEND": (P143, "第143轮趋势波动连续预算"),
            "VINTAGE_REFERENCE_RISK": (P131, "第131轮固定入场模型与普通波动预算"),
            "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第165轮收益强弱连续段与上涨动量规则已经冻结")
    prior_path = ROOT / "config/510300_ordinal_entropy_v1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "收益强弱连续段与上涨动量六项必要测试尚未通过")
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id="510300_RETURN_RUNS_STATE_V1", round=165, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), registered_at=now(), decision_clock="15:05:00", parent_models=[],
        runs_window=60, momentum_window=60, volatility_window=20, risk_target=.1,
        entry_threshold=-1., exit_threshold=0., confirmation_closes=2,
        runs_method="FULL_WINDOW_MEDIAN_STRONG_WEAK_RUN_COUNT_NORMALIZED_WITHOUT_CONTINUITY_CORRECTION",
        exact_tie_method="EXCLUDE_EXACT_WINDOW_MEDIAN_KEEP_TIME_ORDER_NO_ROUNDING",
        undefined_statistic="NO_VIEW_RESET_DIRECTION_AND_COUNTERS",
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_ALL_THREE_COUNTERS",
        direction="TWO_LOW_RUNS_POSITIVE_MOMENTUM_OR_SEPARATE_NONNEGATIVE_RUNS_NONPOSITIVE_MOMENTUM_EXIT",
        initial_direction=0, signal_parameters_fitted=False,
        new_model_fits=0, model_fit_count_scope="ZERO_SUPERVISED_MODELS_DESCRIPTIVE_WINDOW_STATISTICS_ONLY",
        new_reference_accounts=0, source_budget_cny=0, goal_achieved=False, position_impact=0,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_RETURN_RUNS_STATE_V1.md", independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND164_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), prior_path, ROOT / "research/return_runs_state_inputs_v1.py",
        ROOT / "research/saved_target_batch_runner_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_return_runs_state_v1.py",
        ROOT / "tests/test_heikin_price_state_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "docs/510300_RETURN_RUNS_STATE_NEXT_20260910.md", ROOT / "config/510300_research_authority_v6.json",
        P143 / "saved_verification_receipt.json", P131 / "saved_verification_receipt.json"]
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
    print("第165轮一个固定收益强弱连续段与上涨动量规则已冻结，尚未生成新历史目标或账户。", flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, {}, CONTROLS, return_runs_state_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
