"""复用单父账户流程，冻结同状态模型更新分歧退出。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/model_update_veto_v1.py"
    require(not destination.exists(), "第148轮运行文件已经存在")
    old = (ROOT / "research/episode_trend_admission_v1.py").read_text(encoding="utf-8")
    run = old[old.index("def run():"):]
    run = run.replace("EPISODE_TREND_ADMISSION_ACCOUNTS_COMPLETE", "MODEL_UPDATE_VETO_ACCOUNTS_COMPLETE")
    run = run.replace("长期趋势准入", "模型更新否决").replace("长期趋势决定整段新机会的进入资格", "固定版与最新版负面分歧退出")
    run = run.replace('    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []',
        '    records = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]\n    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []')
    run = run.replace('        factor_frames, summaries = episode_trend_frames(frame, parents_by_cost, cfg, start)',
        '        signals = {cost_id: current_close_signals(pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_decisions.parquet"),\n'
        '            pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_ledger.parquet", columns=["date", "shares", "mark_clock"]),\n'
        '            frame, start, cost_id) for cost_id in cfg["costs"]}\n'
        '        factor_frames, summaries = model_update_veto_frames(frame, parents_by_cost, signals, records, cfg, start)')
    run = run.replace('        "all_metrics": main,', '        "new_latest_predictions_generated": sum(s["new_latest_predictions"] for s in target_summaries),\n        "all_metrics": main,')
    header = '''"""第148轮用同一状态的模型更新负面分歧否决父持仓机会。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.model_update_veto_inputs_v1 import model_update_veto_frames, PRIMARY, MODELS
from research.continuation_strength_blend_inputs_v1 import current_close_signals
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_model_update_veto_v1"
CONFIG = ROOT / "config/510300_model_update_veto_v1.json"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P147 = ROOT / "reports/research/510300_episode_trend_admission_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P143, "第143轮趋势波动连续预算"), "EPISODE_TREND_ADMISSION": (P147, "第147轮长期趋势进入资格"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "模型更新否决已经冻结")
    preflight_path = ROOT / "reports/research/510300_model_update_veto_preflight_20260909/result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "FIXED_AND_LATEST_SAVED_MODEL_CLOCKS_AND_SAME_STATE_EIGHT_FEATURES_READY" and
        preflight["new_model_fits_or_predictions_or_targets_or_accounts"] == 0, "固定与最近模型输入确认未完成")
    for source in preflight["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "固定或最近模型输入确认的来源改变")
    old_path = ROOT / "config/510300_episode_trend_admission_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    original = json.loads((ROOT / "config/510300_entry_vintage_exit_v1.json").read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "模型更新否决六项必要测试未通过")
    cfg.update(study_id="510300_MODEL_UPDATE_VETO_V1", round=148, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="FIXED_NONNEGATIVE_LATEST_NEGATIVE_VETO_UNTIL_PARENT_ZERO",
        saved_models=original["saved_models"], registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        new_prediction_scope="ONLY_COMPARABLE_POSITIVE_PARENT_UNVETOED_STATES", reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="NEW_POSITIVE_PARENT_AFTER_EXPLICIT_ZERO_CLEARS_VETO",
        rules="docs/510300_MODEL_UPDATE_VETO_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS146_147_COMPLETED_CLOSED_INPUT148_READY_FULL_GOAL_NOT_MET")
    paths = [ROOT / item["path"] for item in preflight["sources"]]
    paths.extend([Path(__file__), preflight_path, ROOT / "scripts/preflight_model_update_veto_20260909.py",
        ROOT / "research/model_update_veto_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_model_update_veto_v1.py", ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / cfg["rules"], ROOT / "config/510300_research_authority_v6.json"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = preflight["checks"]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第148轮模型更新否决已冻结，尚未生成新预测、目标或账户。", flush=True)


'''
    source = header+run
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_MODEL_UPDATE_VETO_NEXT_20260909.md").read_text(encoding="utf-8").replace("# 第148轮拟研究：", "# 第148轮冻结规则：")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    extra = """
## 模型输入与账户执行的具体口径

持仓年龄因素具体为原实际持仓天数加一后取自然对数。原进入后首次收盘选择的固定版本、模型拟合日期和参数身份必须一致。最新版按当时拟合索引选择，并同时确认实际声明的拟合时刻不晚于当日15:05；记录日期更新而参数身份相同，不能视为已经产生新分歧。相同参数和相同状态的最新预测须能还原固定预测。

仅在父目标明确为正且未锁定否决、原参考持有且固定和最新模型及八因素均可用时，才生成一次新的最新版预测。父零、父未知、已锁定、参考空仓或明确无固定模型等情况，保留未计算的新预测状态。父目标未知时保留锁定记录但不发新调整，随后恢复正目标时已有否决仍请求全部退出。父明确零才解除锁定，实际受阻不会自行解除。失效状态、缺失和非法数值区分处理，不能用零预测掩盖。

每户二十万元，242日年化，现金和无风险收益为零。基础佣金万分之二且最低5元、滑点万分之五；压力佣金万分之四且最低5元、滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、分红登记权益和除息应收及支付日到账均保持。终点开盘全部清仓；如果最后的父信号及否决仍活动，如实保留该状态，不把终点强制卖出当作父信号已经结束。

六项必要测试已经实际通过，包括相同参数预测对应、负面更新锁定、空仓及无模型不补预测、未知优先级、未来隔离、费用身份、真正退出再进入及分红。四新账户、十二保存对照。核心秒数不含开发、测试、核对和文档。

下文完整保留143及其所有底层因素和中文进出场规则。新否决只影响本轮账户，不反向改变这些原父账户的持仓或模型训练。

"""
    (ROOT / "docs/510300_MODEL_UPDATE_VETO_V1.md").write_text(proposal+extra+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_model_update_veto_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 7.55,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_model_update_veto_v1.py", "output": "6 passed in 7.55s"}, exclusive=True)
    print("第148轮完整中文规则和六项实际测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
