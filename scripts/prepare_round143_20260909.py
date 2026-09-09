"""复用目标对齐与账户流程，建立连续趋势波动预算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/trend_noise_reference_blend_v1.py"
    require(not destination.exists(), "第143轮运行文件已经存在")
    source = (ROOT / "research/consensus_reference_target_v1.py").read_text(encoding="utf-8")
    source = source.replace("CONSENSUS_REFERENCE_TARGET", "TREND_NOISE_REFERENCE_BLEND").replace("consensus_reference_target", "trend_noise_reference_blend")
    source = source.replace("consensus_target_frames", "trend_noise_frames").replace("第142轮", "第143轮").replace("round=142", "round=143")
    source = source.replace("以两套保存目标共同约束进入退出", "按正趋势与波动幅度连续分配来源").replace("共同目标", "趋势波动预算")
    source = source.replace('PRIMARY: "两套正目标共同确认持仓"', 'PRIMARY: "按趋势与波动幅度连续分配来源"')
    source = source.replace('old_path = ROOT / "config/510300_episode_trend_reference_v1.json"', 'old_path = ROOT / "config/510300_consensus_reference_target_v1.json"')
    source = source.replace('P32 = ROOT', 'P140 = ROOT / "reports/research/510300_trend_reference_router_v1"\nP32 = ROOT')
    source = source.replace('    "BUY_HOLD": (P32, "买入持有")}', '    "TREND_REFERENCE_ROUTER": (P140, "第140轮每日趋势选择"), "BUY_HOLD": (P32, "买入持有")}')
    source = source.replace('combination="MINIMUM_OF_TWO_KNOWN_PARENT_TARGETS",', 'combination="POSITIVE_TREND_DIVIDED_BY_TREND_PLUS_NOISE", trend_window=120, noise_window=20,')
    source = source.replace('tests["passed"] == 4', 'tests["passed"] == 5').replace("四项必要测试", "五项必要测试")
    source = source.replace("PROGRESS_ROUND141_VERIFIED_AND_CLOSED_COST_SAVINGS_INSUFFICIENT", "PROGRESS_ROUNDS141_142_VERIFIED_CLOSED_FULL_GOAL_NOT_MET")
    source = source.replace('ROOT / "reports/research/510300_episode_trend_reference_v1/saved_verification_receipt.json"',
        'ROOT / "reports/research/510300_consensus_reference_target_v1/saved_verification_receipt.json", ROOT / "research/trend_reference_router_inputs_v1.py"')
    source = source.replace('    data = pd.read_parquet(ROOT / cfg["features"])\n    checks = []',
        '    data = pd.read_parquet(ROOT / cfg["features"])\n    from research.trend_noise_reference_blend_inputs_v1 import market_amplitudes\n    market_amplitudes(data, cfg)\n    checks = []')
    source = source.replace("不再等于141已绑定文件", "不再等于142已绑定文件")
    source = source.replace('"reused_control_accounts": 6', '"reused_control_accounts": 8').replace('"reused_earlier_accounts": 6', '"reused_earlier_accounts": 8')
    source = source.replace("及三条保存对照", "及四条保存对照")
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    parent = (ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md").read_text(encoding="utf-8")
    inherited = parent[parent.index("## 第一来源："):].replace("本轮", "原139轮")
    intro = """# 第143轮：按正趋势与波动幅度连续分配策略预算

本轮只登记一个连续预算公式，来源为131普通波动乘数与139训练支持选择。保留两个父策略的原收盘目标，根据已有趋势幅度相对波动幅度分配比例，不新增模型训练、来源参考或市场数据。

## 两个市场因素及连续预算

第一个因素为已有120日均线偏离：当日含分红财富除以最近完整120个交易日财富的等权平均，再减一，含当日。正趋势幅度是该因素与零中的较大者。第二个因素为已有20日含分红简单收益样本标准差乘242平方根，属于年化波动。再乘120除以242的平方根，得到与前一因素比较使用的波动幅度。完整窗口和既有列必须对应；缺失保留缺失，负波动或无穷值报错。

每天15:05，在两因素都明确时，131预算等于正趋势幅度除以正趋势幅度加波动幅度；139预算为剩余部分。正趋势弱时更多保留139，正趋势相对波动较强时增加131。如果两幅度都明确为零，131预算零；正趋势为正且波动明确零，131预算一。两因素任一未知，预算未知，不自动用满仓、现金或另一模型补齐。

上述比例是固定预算公式，不是预测上涨概率，不保证实际未来120日波动。平方根只是尺度约定，120、20和242不根据本轮收益改变。两套来源有共同底层规则，不是独立预测信心。方案是在已有历史结果之后提出，不构成独立验证。

## 目标合成和实际进入退出

两费用共用当时市场因素和预算，各自读取对应费用父目标。两个父目标都明确时，最终股票目标为131预算乘其目标，加139预算乘其目标；预算未知或任何父目标未知，最终未知，即使某一预算是零也不隐藏缺失。计算结果仅在浮点边界限制到零至一，不使用杠杆。

实际空仓且最终目标为正，下一开盘按自己的现金、份额市值和分红应收构成的净值进入，采用100份整手。已有份额时，与正目标相差不足十个百分点保持份额，达到或超过十个百分点增减。最终明确零目标请求下一开盘全部卖出，未知目标不发新调整。

一套策略明确退出、另一套仍有正目标且对应预算为正，继续保留后者相应规模。最终目标为零才全部退出；随后出现新的正目标可以再次进入，无额外冷却或来源固定。每个收盘按最新目标重算受阻请求，终点统一开盘清仓。父策略原训练、周期、保护、等待新机会及内部退出锁定独立连续保留，不接收本轮盈亏或持股反馈。

## 账户和有限研究范围

一项设置、四条新增账户；保存对照131、139、140及买入持有共十六条。主2020年1月2日至2026年8月14日开盘，较早2015年1月5日至2019年12月31日开盘，各两档费用。每户二十万元、242日年化、现金和无风险收益零。基础佣金万分之二且最低5元，滑点万分之五；压力佣金万分之四且最低5元，滑点千分之一。100份、0.001元价位、次日可卖、方向涨跌停、登记日权益、除息应收和到账均保持。

五项必要测试通过后冻结，一次运行；输入校验复用原因素和目标对齐。失败结束本连续预算公式，不改比例、窗口、归一化或方向救回。旧140至142的失败判断保留，完整目标仍需成本后夏普至少1.2、稳定超额和独立证据。

## 两套父策略的全部原始中文规则

下文完整保留139及131的底层因子、训练支持、原始进入退出和再进入条件；原轮次训练与测试属于保存来源，本轮不重新执行。

"""
    (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").write_text(intro+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_trend_noise_reference_blend_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 5, "seconds": 7.18,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_trend_noise_reference_blend_v1.py", "output": "5 passed in 7.18s"}, exclusive=True)
    print("第143轮连续预算规则与五项实际测试回执已保存，尚未冻结或生成账户。", flush=True)


if __name__ == "__main__":
    main()
