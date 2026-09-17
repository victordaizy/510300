"""静态生成稀疏研究入口，沿用已验证的完整账户循环。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/sparse_vintage_exit_v1.py"
    if destination.exists():
        raise ValueError("稀疏研究入口已存在，禁止覆盖")
    old = (ROOT / "research/cycle_serial_error_exit_v1.py").read_text(encoding="utf-8")
    run = old[old.index("def run():"):]
    run = run.replace('    expected = pd.read_csv(PREFLIGHT / "saved_monthly_residual_correlation.csv")\n', "")
    run = run.replace("build_monthly_models(samples, originals, cfg, expected)", "build_monthly_models(samples, originals, cfg)")
    run = run.replace("SerialErrorExitController", "SparseVintageExitController")
    run = run.replace("相关误差", "稀疏退出").replace("周期相邻误差修正后固定版本退出", "原八因素稀疏学习后固定版本退出")
    run = run.replace("CYCLE_SERIAL_ERROR_EXIT_ACCOUNTS_COMPLETE", "SPARSE_VINTAGE_EXIT_ACCOUNTS_COMPLETE")
    marker = '    models, monthly, memberships, counts = build_monthly_models(samples, originals, cfg)'
    run = run.replace(marker, '    require(all(pd.Timestamp(r["fit_time"]) == data.date.iloc[r["fit_index"]] + pd.Timedelta(hours=15, minutes=5) for r in originals), "月度训练时钟不对应完整交易日历")\n' + marker)
    if "PREFLIGHT" in run or "SerialError" in run or "expected)" in run:
        raise ValueError("新入口仍包含旧相关误差依赖")
    prefix = '''"""拟合有限的稀疏退出模型，保留月度时钟并计算四个真实资金账户。"""
import json
import time
from pathlib import Path
import pandas as pd
import sklearn
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.sparse_vintage_exit_inputs_v1 import build_monthly_models, SparseVintageExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_vintage_exit_v1"
CONFIG = ROOT / "config/510300_sparse_vintage_exit_v1.json"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "SPARSE_VINTAGE_EXIT"
CONTROLS = {
    "ENTRY_VINTAGE_EXIT": (ROOT / "reports/research/510300_entry_vintage_exit_v1", "第128轮进入时固定原退出模型"),
    "VINTAGE_REFERENCE_RISK": (ROOT / "reports/research/510300_vintage_reference_risk_v1", "第131轮固定版本与普通波动预算"),
    "TREND_NOISE_REFERENCE_BLEND": (ROOT / "reports/research/510300_trend_noise_reference_blend_v1", "第143轮趋势波动连续预算"),
    "BUY_HOLD": (ROOT / "reports/research/510300_rearmed_session_exit_v1", "买入持有")}


def freeze():
    require(not CONFIG.exists(), "稀疏退出策略已经登记")
    parent_path = ROOT / "config/510300_within_cycle_exit_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
            "confirmation_days", "specification", "feature_columns", "feature_names", "feature_clip", "recent_cycles", "minimum_cycles", "minimum_rows"]
    cfg = {key: parent[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "七项稀疏退出必要测试尚未通过")
    cfg.update(study_id="510300_SPARSE_VINTAGE_EXIT_V1", round=156, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, source_models=str((P114 / "saved_models.json").relative_to(ROOT)),
        samples=str((P114 / "extended_reference_samples.parquet").relative_to(ROOT)),
        lasso_alpha=.0014, lasso_max_iter=10000, lasso_tolerance=1e-10, kkt_tolerance=1e-8,
        model_loss="CYCLE_EQUAL_AVERAGED_WITHIN_SQUARED_ERROR_OVER_TWO_PLUS_FIXED_L1_PENALTY",
        zero_initial_solution="ACCEPT_EXACT_KKT_SATISFIED_ZERO_VECTOR_BEFORE_ITERATION",
        solver="CYCLIC_COORDINATE_DESCENT_FROM_ZERO", solver_fallback=False, sklearn_version=sklearn.__version__,
        model_selection_clock="FIRST_CLOSE_OF_ACTUAL_FILLED_ENTRY", model_reselection="ONLY_ON_NEW_ACTUAL_CYCLE",
        cache="ACTUAL_ORDERED_MEMBERS_FEATURES_TARGET_WEIGHTS_AND_SETTINGS_PAST_SUCCESS_OR_FAILURE",
        planned_distinct_fits=25, planned_monthly_records=141, planned_eligible_months=114,
        rules="docs/510300_SPARSE_VINTAGE_EXIT_V1.md", new_reference_accounts=0, source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND155_COMPLETED_CLOSED_AND_DELIVERED")
    require(abs(cfg["lasso_alpha"] - 2 * (cfg["costs"]["BASE"]["commission"] + cfg["costs"]["BASE"]["slippage"])) < 1e-15, "固定惩罚尺度与基础成本口径不同")
    names = ["research/sparse_vintage_exit_inputs_v1.py", "research/learned_cycle_exit_v1.py", "research/within_cycle_exit_inputs_v1.py",
        "research/rearmed_cycle_exit_account_v1.py", "research/simple_intraday_protection_v1.py", "research/simple_session_divergence_v1.py",
        "research/simple_price_entry_exit_v1.py", "research/adaptive_allocation_v1.py", "research/intraday_overnight_increment_v1.py",
        "tests/test_sparse_vintage_exit_v1.py", "tests/test_median_continuation_v1.py", "config/510300_research_authority_v6.json",
        "docs/510300_SPARSE_VINTAGE_EXIT_NEXT_20260909.md"]
    paths = [Path(__file__), parent_path, OUT / "tests_receipt.json", *(ROOT / p for p in names),
             *(ROOT / cfg[key] for key in ["source_models", "samples", "rules", "features", "dividends"])]
    known = {}
    for name in ["510300_cycle_serial_error_exit_v1", "510300_median_slope_risk_v1"]:
        path = ROOT / "config" / (name + ".json")
        previous = json.loads(path.read_text(encoding="utf-8"))
        known.update({str(Path(item["path"])): item["sha256"] for item in previous["frozen_files"]})
        paths.append(path)
    for key in ["source_models", "samples", "features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known[str(Path(cfg[key]))], "既有训练、价格或分红来源版本改变")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model, (folder, _) in CONTROLS.items():
                path = folder / period / cost / f"{model}_ledger.parquet"
                require(digest(path) == known[str(path.relative_to(ROOT))], "保存对照已绑定版本改变")
                paths.append(path)
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第156轮一套稀疏退出规则已冻结，尚未拟合新模型或计算新账户。", flush=True)


'''
    destination.write_text(prefix + run, encoding="utf-8")
    print("稀疏研究入口已静态生成；旧相关误差入口保持原样。", flush=True)


if __name__ == "__main__":
    main()
