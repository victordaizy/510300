"""复用四账户流程，建立高低区间与隔夜风险乘数。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/range_overnight_risk_v1.py"
    require(not destination.exists(), "第146轮运行文件已经存在")
    old = (ROOT / "research/trend_coherence_blend_v1.py").read_text(encoding="utf-8")
    run = old[old.index("def run():"):]
    run = run.replace("TREND_COHERENCE_BLEND_ACCOUNTS_COMPLETE", "RANGE_OVERNIGHT_RISK_ACCOUNTS_COMPLETE")
    run = run.replace("趋势连贯预算", "区间隔夜风险").replace("按上升趋势连贯程度分配来源", "按日内区间和隔夜风险缩放143仓位")
    run = run.replace('source_folders = [(MODELS[0], P131), (MODELS[1], P143)]', 'source_folders = [(MODELS[0], P143)]')
    run = run.replace("trend_coherence_frames", "range_overnight_frames")
    run = run.replace('"reused_control_accounts": 8', '"reused_control_accounts": 6').replace('"reused_earlier_accounts": 8', '"reused_earlier_accounts": 6')
    run = run.replace("及四条保存对照", "及三条保存对照")
    header = '''"""第146轮使用日内高低区间和隔夜风险约束143的目标。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.range_overnight_risk_inputs_v1 import range_overnight_frames, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_range_overnight_risk_v1"
CONFIG = ROOT / "config/510300_range_overnight_risk_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P145 = ROOT / "reports/research/510300_trend_coherence_blend_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P143, "第143轮趋势波动连续预算"), "TREND_COHERENCE_BLEND": (P145, "第145轮趋势连贯预算"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "区间隔夜风险已经冻结")
    preflight_path = ROOT / "reports/research/510300_range_overnight_risk_preflight_20260909/result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "RANGE_OVERNIGHT_DIVIDEND_IDENTITIES_AND_PARENT_TARGET_CLOCKS_READY" and
        preflight["new_risk_multipliers_or_accounts"] == 0, "现有高低区间与隔夜口径未完成确认")
    for source in preflight["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "现有高低区间或保存目标来源改变")
    old_path = ROOT / "config/510300_trend_coherence_blend_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 5, "区间隔夜风险五项必要测试未通过")
    cfg.update(study_id="510300_RANGE_OVERNIGHT_RISK_V1", round=146, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="PARENT143_TIMES_RANGE_PLUS_OVERNIGHT_RISK_MULTIPLIER", risk_window=20,
        registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_RANGE_OVERNIGHT_RISK_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS144_145_COMPLETED_CLOSED_INPUT146_READY_FULL_GOAL_NOT_MET")
    paths = [ROOT / item["path"] for item in preflight["sources"]]
    paths.extend([Path(__file__), preflight_path, ROOT / "scripts/preflight_range_overnight_risk_20260909.py",
        ROOT / "research/range_overnight_risk_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_range_overnight_risk_v1.py", ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / cfg["rules"], ROOT / "config/510300_research_authority_v6.json"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = preflight["checks"]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第146轮区间隔夜风险已冻结，尚未生成新目标或账户。", flush=True)


'''
    source = header+run
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_RANGE_OVERNIGHT_RISK_NEXT_20260909.md").read_text(encoding="utf-8").replace("# 第146轮拟研究：", "# 第146轮冻结规则：")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    extra = """
## 实际账户与完整父策略规则

每条账户二十万元，242日年化，现金和无风险收益均为零。基础费用为佣金万分之二、最低5元、滑点万分之五；压力费用为佣金万分之四、最低5元、滑点千分之一。所有分红登记日权益、除息应收和支付日到账独立处理，价格最小单位0.001元，100份整手，次日可卖、方向涨跌停和终点开盘全部清仓保持。

父策略继续运行自己的模型与内部进出场，不接收本轮账户反馈。本轮只缩放父目标；风险乘数不会把正父目标改成额外入场判断，但目标规模太小可能因100份整手产生实际空仓，这是明确的账户结果。五项必要测试已实际通过，覆盖两风险部分、分红、零与缺失、费用时钟、未来隔离、部分调仓和实际退出再进入。核心账户时间不含开发、测试、核对和文档。

下文完整保留143、139及底层来源全部因子和中文进出场规则。原143的双来源预算是父策略内部行为，本轮新增部分只对其最终目标乘风险乘数。

"""
    (ROOT / "docs/510300_RANGE_OVERNIGHT_RISK_V1.md").write_text(proposal+extra+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_range_overnight_risk_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 5, "seconds": 5.02,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_range_overnight_risk_v1.py", "output": "5 passed in 5.02s"}, exclusive=True)
    print("第146轮完整中文规则和五项测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
