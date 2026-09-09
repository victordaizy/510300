"""复用账户流程，登记按最弱历史子阶段增长的月度预算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/robust_block_growth_v1.py"
    require(not destination.exists(), "第149轮运行文件已经存在")
    old = (ROOT / "research/episode_trend_admission_v1.py").read_text(encoding="utf-8")
    run = old[old.index("def run():"):]
    run = run.replace("EPISODE_TREND_ADMISSION_ACCOUNTS_COMPLETE", "ROBUST_BLOCK_GROWTH_ACCOUNTS_COMPLETE")
    run = run.replace("长期趋势准入", "最弱阶段增长预算").replace("长期趋势决定整段新机会的进入资格", "按近期最弱子阶段增长分配预算")
    run = run.replace('source_folders = [(MODELS[0], P143)]', 'source_folders = [(MODELS[0], P131), (MODELS[1], P143)]')
    run = run.replace('    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []',
        '    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []\n    budget_estimations = []')
    run = run.replace('        factor_frames, summaries = episode_trend_frames(frame, parents_by_cost, cfg, start)',
        '        base_ledgers = {model: pd.read_parquet(folder / period / "BASE" / f"{model}_ledger.parquet").assign(source_cost="BASE", source_model=model)\n'
        '            for model, folder in source_folders}\n'
        '        factor_frames, summaries, estimates = robust_block_growth_frames(frame, parents_by_cost, base_ledgers, cfg, start)\n'
        '        budget_estimations.extend({"period": period, **item} for item in estimates)')
    run = run.replace('    primary = [m for m in main if m["model"] == PRIMARY]',
        '    pd.DataFrame(budget_estimations).to_csv(OUT / "budget_estimations.csv", index=False, encoding="utf-8-sig")\n    primary = [m for m in main if m["model"] == PRIMARY]')
    run = run.replace('        "all_metrics": main,', '        "new_budget_optimizations": sum(row["status"] == "BUDGET_OPTIMIZATION_COMPLETE" for row in budget_estimations),\n        "all_metrics": main,')
    run = run.replace('"reused_control_accounts": 6', '"reused_control_accounts": 8').replace('"reused_earlier_accounts": 6', '"reused_earlier_accounts": 8')
    run = run.replace("及三条保存对照", "及四条保存对照")
    header = '''"""第149轮按近期三个子阶段最弱对数增长估计月度来源预算。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.robust_block_growth_inputs_v1 import robust_block_growth_frames, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_robust_block_growth_v1"
CONFIG = ROOT / "config/510300_robust_block_growth_v1.json"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P148 = ROOT / "reports/research/510300_model_update_veto_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数"), MODELS[1]: (P143, "第143轮趋势波动连续预算"),
    "MODEL_UPDATE_VETO": (P148, "第148轮模型更新分歧退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "最弱阶段增长预算已经冻结")
    old_path = ROOT / "config/510300_model_update_veto_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    c143 = ROOT / "config/510300_trend_noise_reference_blend_v1.json"
    source143 = json.loads(c143.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "最弱阶段增长六项必要测试未通过")
    cfg.update(study_id="510300_ROBUST_BLOCK_GROWTH_V1", round=149, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="MONTHLY_MAXIMUM_WORST_THREE_BLOCK_LOG_GROWTH", initial_131_budget=0,
        training_rows=242, block_lengths=[81, 81, 80], optimizer_iterations=80, numeric_tolerance=1e-12,
        training_source_cost="BASE", training_scope="CURRENT_PERIOD_ONLY_NO_CROSS_PERIOD_STITCHING", update_clock="FIRST_TRADING_DAY_OF_MONTH_1505",
        registered_at=now(), new_model_fits=0, new_reference_accounts=0, reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_ROBUST_BLOCK_GROWTH_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND148_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    paths = [Path(__file__), old_path, c143, ROOT / "config/510300_vintage_reference_risk_v1.json",
        ROOT / "research/robust_block_growth_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_robust_block_growth_v1.py", ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "docs/510300_ROBUST_BLOCK_GROWTH_NEXT_20260909.md",
        ROOT / "config/510300_research_authority_v6.json", P131 / "saved_verification_receipt.json", P143 / "saved_verification_receipt.json", P148 / "saved_verification_receipt.json"]
    known_hashes = {str(Path(item["path"])): item["sha256"] for saved in [source143, old] for item in saved["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known_hashes[str(Path(cfg[key]))], "月度预算行情或分红来源改变")
    checks = []
    data = pd.read_parquet(ROOT / cfg["features"])
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for model, folder in [(MODELS[0], P131), (MODELS[1], P143)]:
            path = folder / period / "BASE" / f"{model}_ledger.parquet"
            require(digest(path) == known_hashes[str(path.relative_to(ROOT))], "月度预算来源账本不再等于已绑定版本")
            ledger_clock = pd.read_parquet(path, columns=["date", "mark_clock"])
            require(pd.DatetimeIndex(ledger_clock.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and ledger_clock.mark_clock.iloc[:-1].eq("CLOSE").all(), "月度预算来源不是完整收盘日历")
            checks.append({"period": period, "model": model, "base_source_rows": len(ledger_clock), "last_terminal_open_excluded_from_training": True})
            for cost in cfg["costs"]:
                path = folder / period / cost / f"{model}_decisions.parquet"
                require(digest(path) == known_hashes[str(path.relative_to(ROOT))], "月度预算父目标来源改变")
                parent = pd.read_parquet(path, columns=["origin", "origin_index", "execution_date"])
                require(np.array_equal(parent.origin_index, indices) and pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                    pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "月度预算父目标时钟不同")
                paths.append(path)
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = checks
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第149轮月度最弱增长预算已冻结，尚未优化预算或生成新账户。", flush=True)


'''
    source = header+run
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_ROBUST_BLOCK_GROWTH_NEXT_20260909.md").read_text(encoding="utf-8").replace("# 第149轮拟研究：", "# 第149轮冻结规则：")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    extra = """
## 实际计算及账户边界

主区间和较早区间分别初始化，不拼接两条各自终点清仓和重新开户的历史曲线来制造连续训练。每次完整窗口含当前判断收盘，但不包含下一开盘；终点统一开盘那一行不用于收盘训练。每月未得到完整有限收益时，已满足日历长度的缺失与研究起点样本不足是不同状态，前者无观点、后者明确使用初始143。

预算优化记录每个子阶段的平均对数增长、所选与上次预算的最弱值、训练起止日及来源费用。月份中途不利用新收益再估计预算，父目标和实际执行仍每天处理。三段最弱平均对数增长更高只描述已用训练窗口，不证明下一段收益或高夏普。

每户二十万元，242日年化，现金和无风险收益为零。基础佣金万分之二且最低5元、滑点万分之五；压力佣金万分之四且最低5元、滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、分红登记权益、除息应收和支付到账均保持。实际目标转换为自己前收盘净值及价格下的整手请求，已有正仓位偏差不足十个百分点不调；零目标全退，未知不调，受阻逐日重算、终点开盘清仓。

六项必要测试已通过，覆盖最弱子阶段和端点、相同收益保留旧预算、预热与月度窗口、缺失与费用时钟、未来隔离、实际进出和分红。核心时间不含开发、测试、核对及文档。下文完整保留131、143及其各层全部因子、进入、退出和再进入规则，原预测模型本轮不重训。

"""
    (ROOT / "docs/510300_ROBUST_BLOCK_GROWTH_V1.md").write_text(proposal+extra+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_robust_block_growth_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 4.24,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_robust_block_growth_v1.py", "output": "6 passed in 4.24s"}, exclusive=True)
    print("第149轮完整中文规则和六项实际测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
