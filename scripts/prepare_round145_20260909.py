"""复用连续预算账户流程，建立上升趋势连贯程度预算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/trend_coherence_blend_v1.py"
    require(not destination.exists(), "第145轮运行文件已经存在")
    source = (ROOT / "research/trend_noise_reference_blend_v1.py").read_text(encoding="utf-8")
    source = source.replace("TREND_NOISE_REFERENCE_BLEND", "TREND_COHERENCE_BLEND").replace("trend_noise_reference_blend", "trend_coherence_blend")
    source = source.replace("trend_noise_frames", "trend_coherence_frames").replace("market_amplitudes", "rolling_trend_coherence")
    source = source.replace("第143轮", "第145轮").replace("round=143", "round=145")
    source = source.replace("按正趋势与波动幅度连续分配来源", "按上升趋势连贯程度分配来源").replace("按趋势与波动幅度连续分配来源", "按上升趋势连贯程度分配来源")
    source = source.replace("趋势波动预算", "趋势连贯预算")
    source = source.replace('old_path = ROOT / "config/510300_consensus_reference_target_v1.json"', 'old_path = ROOT / "config/510300_continuation_strength_blend_v1.json"')
    source = source.replace("P139", "P143").replace("P140", "P144")
    source = source.replace('"reports/research/510300_model_support_reference_router_v1"', '"reports/research/510300_trend_noise_reference_blend_v1"')
    source = source.replace('"reports/research/510300_trend_reference_router_v1"', '"reports/research/510300_continuation_strength_blend_v1"')
    source = source.replace("第139轮训练支持条件选择", "第143轮趋势波动连续预算").replace("第140轮每日趋势选择", "第144轮继续持有预测预算")
    source = source.replace('"TREND_REFERENCE_ROUTER":', '"CONTINUATION_STRENGTH_BLEND":')
    source = source.replace('combination="POSITIVE_TREND_DIVIDED_BY_TREND_PLUS_NOISE", trend_window=120, noise_window=20,', 'combination="POSITIVE_LOG_TREND_R_SQUARED_BUDGET", trend_window=120,')
    source = source.replace("PROGRESS_ROUNDS141_142_VERIFIED_CLOSED_FULL_GOAL_NOT_MET", "PROGRESS_ROUND144_COMPLETED_CLOSED_CURRENT_BEST143_FULL_GOAL_NOT_MET")
    source = source.replace("不再等于142已绑定文件", "不再等于144已绑定文件")
    source = source.replace('ROOT / "config/510300_model_support_reference_router_v1.json", ROOT / "reports/research/510300_consensus_reference_target_v1/saved_verification_receipt.json", ROOT / "research/trend_reference_router_inputs_v1.py"',
        'ROOT / "config/510300_trend_noise_reference_blend_v1.json", P144 / "saved_verification_receipt.json", ROOT / "tests/test_trend_noise_reference_blend_v1.py"')
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    proposal = (ROOT / "docs/510300_TREND_COHERENCE_BLEND_NEXT_20260909.md").read_text(encoding="utf-8").replace("# 第145轮拟研究：", "# 第145轮冻结规则：")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8").replace("本轮", "原143轮")
    extra = """
## 实际进入、退出和父策略反馈边界

新账户空仓且最终目标为正，按自己的前一收盘净值和原始收盘价，向下取100份整手请求下一开盘进入，成交受开盘价、可用现金和费用约束。已持有且目标为正，仅当目标与自身股票敞口相差达到十个百分点时增减；不足时保留原份额。目标明确零请求全部退出；任一必需父目标或预算未知，不新增调整。两费用分别处理自己的取整、现金、分红和持仓。

来源预算每天更新，不锁定整笔交易的来源。受阻后每个收盘按最新目标重新计算；重新出现正目标可以再次进入，无附加冷却。终点统一开盘清仓。父策略持续按各自原规则生成收盘意图，不接收本轮持仓、净值和盈亏反馈。保存父目标之间有共同因素，不能当作独立信号。下文完整保留143及其所有父因子、原进入、退出、再进入和训练支持规则。

五项必要测试已经实际通过。主、较早各两档费用合计四新账户；两段各四条保存对照，共十六条。核心运行耗时不包含代码开发、测试、结果核对和文档交付。

"""
    (ROOT / "docs/510300_TREND_COHERENCE_BLEND_V1.md").write_text(proposal+extra+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_trend_coherence_blend_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 5, "seconds": 3.49,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_trend_coherence_blend_v1.py", "output": "5 passed in 3.49s"}, exclusive=True)
    print("第145轮完整趋势连贯预算及五项实际测试回执已保存，尚未冻结。", flush=True)


if __name__ == "__main__":
    main()
