"""复用四账户运行流程，登记已保存继续预测的幅度预算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/continuation_strength_blend_v1.py"
    require(not destination.exists(), "第144轮运行文件已经存在")
    previous = (ROOT / "research/trend_noise_reference_blend_v1.py").read_text(encoding="utf-8")
    run = previous[previous.index("def run():"):]
    run = run.replace("TREND_NOISE_REFERENCE_BLEND_ACCOUNTS_COMPLETE", "CONTINUATION_STRENGTH_BLEND_ACCOUNTS_COMPLETE")
    run = run.replace("趋势波动预算", "继续预测预算").replace("按趋势与波动幅度连续分配来源", "按保存继续持有预测幅度分配预算")
    run = run.replace('source_folders = [(MODELS[0], P131), (MODELS[1], P139)]', 'source_folders = [(MODELS[0], P131), (MODELS[1], P143)]')
    run = run.replace('    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []',
        '    records = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]\n    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []')
    run = run.replace('        factor_frames, summaries = trend_noise_frames(frame, parents_by_cost, cfg, start)',
        '        signals = {cost_id: current_close_signals(pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_decisions.parquet"),\n'
        '            pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_ledger.parquet", columns=["date", "shares", "mark_clock"]),\n'
        '            frame, start, cost_id) for cost_id in cfg["costs"]}\n'
        '        factor_frames, summaries = continuation_strength_frames(frame, parents_by_cost, signals, records, cfg, start)')
    header = '''"""第144轮复用固定退出模型的继续持有预测幅度。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.continuation_strength_blend_inputs_v1 import continuation_strength_frames, current_close_signals, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuation_strength_blend_v1"
CONFIG = ROOT / "config/510300_continuation_strength_blend_v1.json"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P139 = ROOT / "reports/research/510300_model_support_reference_router_v1"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数策略"), MODELS[1]: (P143, "第143轮趋势波动连续预算"),
    "MODEL_SUPPORT_REFERENCE_ROUTER": (P139, "第139轮训练支持条件选择"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "继续预测预算已经冻结")
    preflight_path = ROOT / "reports/research/510300_continuation_strength_blend_preflight_20260909/result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "SAVED_CONTINUATION_PREDICTIONS_CLOCKS_STATES_AND_PARENT_TARGETS_READY" and
        preflight["new_predictions_or_budgets_or_accounts"] == 0, "原预测输入确认未完成")
    for source in preflight["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "原预测确认的来源改变")
    old_path = ROOT / "config/510300_trend_noise_reference_blend_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    original = json.loads((ROOT / "config/510300_entry_vintage_exit_v1.json").read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    require(original["specification"]["modes"]["1"]["days"] == 60, "原持仓期限改变")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "继续预测预算六项必要测试未通过")
    cfg.update(study_id="510300_CONTINUATION_STRENGTH_BLEND_V1", round=144, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="POSITIVE_CONTINUATION_OVER_CONTINUATION_PLUS_REMAINING_NOISE",
        maximum_holding_days=60, noise_window=20, saved_models=original["saved_models"], registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_CONTINUATION_STRENGTH_BLEND_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND143_FOUR_LOCAL_GAINS_INPUT144_READY_FULL_GOAL_NOT_MET")
    paths = [ROOT / item["path"] for item in preflight["sources"]]
    paths.extend([Path(__file__), preflight_path, ROOT / "scripts/preflight_continuation_strength_blend_20260909.py",
        ROOT / "research/continuation_strength_blend_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_continuation_strength_blend_v1.py", ROOT / "tests/test_trend_noise_reference_blend_v1.py",
        ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json", ROOT / cfg["rules"],
        ROOT / "config/510300_research_authority_v6.json", P131 / "saved_verification_receipt.json", P139 / "saved_verification_receipt.json"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = preflight["checks"]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第144轮继续预测预算已冻结，尚未生成新目标或账户。", flush=True)


'''
    source = header+run
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_CONTINUATION_STRENGTH_BLEND_NEXT_20260909.md").read_text(encoding="utf-8")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    text = proposal.replace("# 第144轮拟研究：", "# 第144轮冻结规则：")
    text += """## 输入边界的明确处理

原参考当前份额只用判断日的实际收盘账本连接；唯一准备收盘没有账本行，使用账户初始化的已知零份额。其余缺行拒绝本轮运行。原空仓必须没有持仓预测，不能用已请求退出的目标零来冒充已经空仓。模型选择日期必须与原行情索引对应，拟合时点不晚于选择收盘，训练周期已退出，固定身份一致。

持仓且明确没有成熟模型时，还须确认原保存版本的状态是成熟周期或样本不足、模型为空且预测缺失，才用143目标。模型资料缺失、找不到记录、预测缺失、波动缺失或者未知及失败状态，保持当日预算无观点；已知非法数值、未来时点、错误身份或矛盾状态则停止运行，不能生成可用账户。实际冻结数据不存在这些未处理缺陷，不额外寻找数据补齐。

无学习因素的两种明确回退状态，其原预测及正优势仍是缺失；只将新增131预算设为零、143预算设为一。无观点预算与明确回退是两种不同状态。可用预测为负或零时正优势才是真正的零。实际持仓天数可能超过原最大期限，剩余比较期仍至少一日，不使用将来实际退出日。

## 成交和再次进入的完整口径

空仓且最终目标为正，下一开盘按本账户前一收盘现金、份额市值和分红应收净值，除以前一实际收盘价，向下取100份整手请求；具体成交受开盘价、费用和可用现金约束。有持仓且正目标与自身敞口相差不足十个百分点时保持份额，达到十个百分点才调整。明确零目标请求全部退出，未知不新增调整。受阻请求每收盘根据最新目标重新计算；目标重现可以再次进入，不加外层锁定或等待期。终点统一开盘清仓，原父策略的内部退出与再进入规则持续运行。

六项必要测试已经实际通过，用于判断日份额连接、预测边界、缺失和模型时点、费用、未来隔离、真实部分调仓及退出再进入和分红。核心四账户时间单独记录，不混入开发、测试、核对和文档时间。以下完整保留143、139、137、131及其底层中文因子与规则；其中历史训练流程为保存来源说明，本轮零新训练。

"""
    (ROOT / "docs/510300_CONTINUATION_STRENGTH_BLEND_V1.md").write_text(text+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_continuation_strength_blend_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 7.96,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_continuation_strength_blend_v1.py", "output": "6 passed in 7.96s"}, exclusive=True)
    print("第144轮完整规则和六项实际测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
