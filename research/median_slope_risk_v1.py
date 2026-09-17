"""冻结单一中位数斜率规则，共读来源后运行四个完整账户。"""
import json
from pathlib import Path
from research.median_slope_risk_inputs_v1 import PRIMARY, CANDIDATES, median_slope_risk_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_median_slope_risk_v1"
CONFIG = ROOT / "config/510300_median_slope_risk_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"TREND_NOISE_REFERENCE_BLEND": (P143, "第143轮趋势波动连续预算"),
            "VINTAGE_REFERENCE_RISK": (P131, "第131轮固定入场模型与普通波动预算"),
            "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第155轮中位数斜率规则已经冻结")
    prior_path = ROOT / "config/510300_boundary_rebalance_v1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "中位数斜率六项必要测试尚未通过")
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id="510300_MEDIAN_SLOPE_RISK_V1", round=155, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), registered_at=now(), decision_clock="15:05:00", parent_models=[],
        slope_window=60, slope_pair_count=1770, slope_estimator="MEDIAN_OF_ALL_LOG_WEALTH_PAIR_SLOPES", volatility_window=20,
        risk_target=.1, direction="POSITIVE_SLOPE_LONG_KNOWN_NONPOSITIVE_ZERO", equal_slope="KNOWN_ZERO_TARGET",
        positive_direction_invalid_volatility="UNKNOWN_TARGET", missing_window="FULL_CONTIGUOUS_WINDOW_REQUIRED",
        new_model_fits=0, model_fit_count_scope="SUPERVISED_RETURN_MODELS_ONLY_NOT_ROLLING_DESCRIPTIVE_SLOPE_ESTIMATES",
        new_reference_accounts=0, source_budget_cny=0, goal_achieved=False, position_impact=0,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_MEDIAN_SLOPE_RISK_V1.md", independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND154_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), prior_path, ROOT / "research/median_slope_risk_inputs_v1.py",
        ROOT / "research/saved_target_batch_runner_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_median_slope_risk_v1.py",
        ROOT / "tests/test_heikin_price_state_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "docs/510300_MEDIAN_SLOPE_RISK_NEXT_20260909.md", ROOT / "config/510300_research_authority_v6.json",
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
    print("第155轮一个固定成对斜率规则已冻结，尚未生成新历史目标或账户。", flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, {}, CONTROLS, median_slope_risk_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
