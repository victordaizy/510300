"""冻结一套相邻高低价方向策略，复用批运行器计算四账户。"""
import json
from pathlib import Path
import pandas as pd
from research.vortex_risk_inputs_v1 import economic_ohlc, vortex_risk_frames, PRIMARY, CANDIDATES
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_vortex_risk_v1"
CONFIG = ROOT / "config/510300_vortex_risk_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"TREND_NOISE_REFERENCE_BLEND": (P143, "第143轮趋势波动连续预算"),
            "VINTAGE_REFERENCE_RISK": (P131, "第131轮原进入模型与普通波动预算"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第153轮相邻高低价规则已经冻结")
    old_path = ROOT / "config/510300_heikin_price_state_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "相邻高低价六项必要测试尚未通过")
    cfg.update(study_id="510300_VORTEX_RISK_V1", round=153, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), decision_clock="15:05:00", vortex_window=14, volatility_window=20, risk_target=.1,
        registered_at=now(), new_model_fits=0, new_reference_accounts=0, parent_models=[],
        price_basis="RAW_OHLC_PLUS_CURRENT_DIVIDEND_TIMES_PREVIOUS_TOTAL_WEALTH_OVER_PREVIOUS_RAW_CLOSE",
        equal_direction="KNOWN_ZERO_TARGET", zero_or_missing_range="UNKNOWN_DIRECTION",
        positive_direction_invalid_volatility="UNKNOWN_TARGET", missing_window="PRESERVE_ROWS_FULL_CONTIGUOUS_WINDOW_REQUIRED",
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_VORTEX_RISK_V1.md", source_budget_cny=0, independent_validation="NOT_ESTABLISHED",
        goal_achieved=False, position_impact=0, previous_goal_turn_classification="PROGRESS_ROUND152_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), old_path, ROOT / "config/510300_cycle_serial_error_exit_v1.json",
        ROOT / "config/510300_trend_noise_reference_blend_v1.json", ROOT / "config/510300_vintage_reference_risk_v1.json",
        ROOT / "research/vortex_risk_inputs_v1.py", ROOT / "research/saved_target_batch_runner_v1.py",
        ROOT / "research/saved_parent_target_alignment_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_vortex_risk_v1.py", ROOT / "tests/test_heikin_price_state_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
        ROOT / "docs/510300_VORTEX_RISK_NEXT_20260909.md", ROOT / "config/510300_research_authority_v6.json",
        P143 / "saved_verification_receipt.json", P131 / "saved_verification_receipt.json"]
    known = {}
    for name in ["510300_heikin_price_state_v1", "510300_cycle_serial_error_exit_v1", "510300_trend_noise_reference_blend_v1"]:
        previous = json.loads((ROOT / "config" / f"{name}.json").read_text(encoding="utf-8"))
        known.update({str(Path(row["path"])): row["sha256"] for row in previous["frozen_files"]})
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known[str(Path(cfg[key]))], "已有行情或分红版本改变")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model, (folder, _) in CONTROLS.items():
                path = folder / period / cost / f"{model}_ledger.parquet"
                require(str(path.relative_to(ROOT)) in known and digest(path) == known[str(path.relative_to(ROOT))], "保存比较账户的已绑定版本不同")
                paths.append(path)
    data = pd.read_parquet(ROOT / cfg["features"])
    checked = economic_ohlc(data)
    cfg["existing_input_checks"] = {"full_calendar_rows": len(checked), "known_economic_price_rows": int(checked.economic_close.notna().sum()),
        "economic_close_matches_saved_wealth": True, "economic_prices_used_for_execution": False}
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第153轮一套相邻高低价方向与普通波动预算已冻结，尚未生成新目标或账户。", flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, {}, CONTROLS, vortex_risk_frames)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
