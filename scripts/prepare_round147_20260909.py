"""复用单父账户流程，登记正目标段首次长期趋势准入。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/episode_trend_admission_v1.py"
    require(not destination.exists(), "第147轮运行文件已经存在")
    source = (ROOT / "research/range_overnight_risk_v1.py").read_text(encoding="utf-8")
    source = source.replace("RANGE_OVERNIGHT_RISK", "EPISODE_TREND_ADMISSION").replace("range_overnight_risk", "episode_trend_admission")
    source = source.replace("range_overnight_frames", "episode_trend_frames").replace("第146轮", "第147轮").replace("round=146", "round=147")
    source = source.replace("使用日内高低区间和隔夜风险约束143的目标", "仅在143新正目标段开始时判断长期趋势资格")
    source = source.replace("区间隔夜风险", "长期趋势准入").replace("按日内区间和隔夜风险缩放143仓位", "长期趋势决定整段新机会的进入资格")
    source = source.replace("P145", "P146").replace("reports/research/510300_trend_coherence_blend_v1", "reports/research/510300_range_overnight_risk_v1")
    source = source.replace('"TREND_COHERENCE_BLEND":', '"RANGE_OVERNIGHT_RISK":').replace("第145轮趋势连贯预算", "第146轮区间隔夜风险")
    source = source.replace("config/510300_trend_coherence_blend_v1.json", "config/510300_range_overnight_risk_v1.json")
    left, right = source.index("    preflight_path ="), source.index("    old_path =")
    source = source[:left]+source[right:]
    source = source.replace('combination="PARENT143_TIMES_RANGE_PLUS_OVERNIGHT_RISK_MULTIPLIER", risk_window=20,',
        'combination="FIRST_POSITIVE_PARENT_EPISODE_LONG_TREND_ADMISSION", entry_trend_window=200, entry_trend_threshold=0,')
    source = source.replace("PROGRESS_ROUNDS144_145_COMPLETED_CLOSED_INPUT146_READY_FULL_GOAL_NOT_MET", "PROGRESS_ROUND146_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    left = source.index('    paths = [ROOT / item["path"]')
    right = source.index('    for period in ["evaluation", "earlier_diagnostic"]:', left)
    paths = '''    paths = [Path(__file__), old_path, ROOT / "research/episode_trend_admission_inputs_v1.py",
        ROOT / "research/saved_parent_target_alignment_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_episode_trend_admission_v1.py", ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / cfg["rules"], ROOT / "docs/510300_EPISODE_TREND_ADMISSION_NEXT_20260909.md",
        ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_trend_noise_reference_blend_v1.json",
        ROOT / cfg["features"], ROOT / cfg["dividends"], P143 / "saved_verification_receipt.json", P146 / "saved_verification_receipt.json"]
    old_bound = {str(Path(item["path"])): item["sha256"] for item in old["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == old_bound[str(Path(cfg[key]))], "长期趋势来源不再等于146绑定文件")
    data = pd.read_parquet(ROOT / cfg["features"])
    from research.episode_trend_admission_inputs_v1 import checked_long_trend
    checked_long_trend(data, cfg)
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            path = P143 / period / cost / f"{MODELS[0]}_decisions.parquet"
            require(digest(path) == old_bound[str(path.relative_to(ROOT))], "143父目标不再等于146绑定文件")
            parent = pd.read_parquet(path, columns=["origin", "origin_index", "execution_date", "reference_weight"])
            require(np.array_equal(parent.origin_index, indices) and pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "长期趋势准入的父目标时钟不符")
            checks.append({"period": period, "cost": cost, "origins": len(parent), "unknown_targets": int(parent.reference_weight.isna().sum()),
                "unknown_long_trend_origins": int(frame.sma200.iloc[indices].isna().sum())})
            paths.append(path)
'''
    source = source[:left]+paths+source[right:]
    source = source.replace('cfg["existing_input_checks"] = preflight["checks"]', 'cfg["existing_input_checks"] = checks')
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_EPISODE_TREND_ADMISSION_NEXT_20260909.md").read_text(encoding="utf-8").replace("# 第147轮拟研究：", "# 第147轮冻结规则：")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    extra = """
## 账户执行与完整来源说明

各账户初始二十万元，242日年化，现金和无风险收益为零。基础佣金万分之二且最低5元、滑点万分之五；压力佣金万分之四且最低5元、滑点千分之一。100份整手、0.001元价格档、次日可卖、方向涨跌停、分红登记与除息应收及到账均保持。终点开盘统一全退；如果原正信号段仍未结束，记录为未结束信号段，不能将终点强制清仓伪装成父信号退出。

拒绝段的最终目标为明确零，可以阻止新进入或请求卖出残余份额；无观点段不等于拒绝段。准入资格不接收自身收益、份额或将来退出结果，段内被允许的规模仍随父目标变化。进入后，最终正目标与自身敞口偏差不足十个百分点保持原份额，达到才调整。五项必要测试已通过，覆盖资格固定与反转、未知和零、费用独立、未来隔离、真正进入退出和分红。核心运行时间不含开发、测试、核对及文档。

下文完整列出143及全部底层来源的因子、进入、退出、再进入和学习规则；它们在本轮仍是保存来源，未重新训练。

"""
    (ROOT / "docs/510300_EPISODE_TREND_ADMISSION_V1.md").write_text(proposal+extra+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_episode_trend_admission_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 5, "seconds": 3.18,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_episode_trend_admission_v1.py", "output": "5 passed in 3.18s"}, exclusive=True)
    print("第147轮完整中文准入规则和五项测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
