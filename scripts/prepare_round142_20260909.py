"""建立无新窗口和比例参数的共同目标组合，复用完整账户计算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/consensus_reference_target_v1.py"
    require(not destination.exists(), "第142轮运行文件已经存在")
    source = (ROOT / "research/trend_reference_router_v1.py").read_text(encoding="utf-8")
    source = source.replace("TREND_REFERENCE_ROUTER", "CONSENSUS_REFERENCE_TARGET").replace("trend_reference_router", "consensus_reference_target")
    source = source.replace("trend_routed_frames", "consensus_target_frames").replace("第140轮", "第142轮").replace("round=140", "round=142")
    source = source.replace("按已有120日趋势选择保存策略", "以两套保存目标共同约束进入退出").replace("趋势选择", "共同目标")
    source = source.replace('PRIMARY: "按退出模型的共同目标策略"', 'PRIMARY: "两套正目标共同确认持仓"')
    source = source.replace('old_path = ROOT / "config/510300_model_support_reference_router_v1.json"', 'old_path = ROOT / "config/510300_episode_trend_reference_v1.json"')
    source = source.replace('P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"\n', '')
    source = source.replace('"JOINT_DOWNSIDE_REFERENCE_PAIR": (P137, "第137轮合成下行预算"), ', '')
    source = source.replace('trend_window=120, trend_threshold=0,', 'combination="MINIMUM_OF_TWO_KNOWN_PARENT_TARGETS",')
    source = source.replace('tests["passed"] == 6', 'tests["passed"] == 4').replace("六项必要测试", "四项必要测试")
    source = source.replace("PROGRESS_ROUNDS138_139_DELIVERED_MINIMUM_FOUR_SHARPE_IMPROVED", "PROGRESS_ROUND141_VERIFIED_AND_CLOSED_COST_SAVINGS_INSUFFICIENT")
    source = source.replace('    from research.consensus_reference_target_inputs_v1 import checked_trend\n    checked_trend(data)\n', '')
    source = source.replace('    data = pd.read_parquet(ROOT / cfg["features"])\n    checks = []',
        '    paths.extend([ROOT / "research/saved_parent_target_alignment_v1.py", ROOT / "tests/test_trend_reference_router_v1.py",\n        ROOT / "config/510300_model_support_reference_router_v1.json", ROOT / "reports/research/510300_episode_trend_reference_v1/saved_verification_receipt.json"])\n    data = pd.read_parquet(ROOT / cfg["features"])\n    checks = []')
    source = source.replace('                parent = pd.read_parquet(path)',
        '                require(digest(path) == old_hashes[str(path.relative_to(ROOT))], "共同目标父来源不再等于141已绑定文件")\n                parent = pd.read_parquet(path)')
    source = source.replace('        base_factors = factor_frames["BASE"]\n        base_factors[base_factors.selection_changed].to_csv(OUT / f"{period}_selection_changes.csv", index=False, encoding="utf-8-sig")\n', '')
    source = source.replace('"reused_control_accounts": 8', '"reused_control_accounts": 6').replace('"reused_earlier_accounts": 8', '"reused_earlier_accounts": 6')
    source = source.replace("及四条保存对照", "及三条保存对照")
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    parent = (ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md").read_text(encoding="utf-8")
    inherited = parent[parent.index("## 第一来源："):].replace("本轮", "原139轮")
    intro = """# 第142轮：两套明确正目标共同确认持仓

本轮只登记一个最终目标交集，来源是131普通波动乘数与139训练支持选择的对应费用收盘目标。两者都为正才有正的股票目标；任何一套明确退出且另一套已知，最终全部退出。本轮没有新增模型、价格窗口、比例参数或来源历史拟合。

## 本轮因素和共同目标

两个因素就是两套策略在同一收盘、同一费用情景下已经确定的股票目标比重。每天15:05，两目标均已知时取较小值作为本轮最终目标。两个正数产生较小的正目标；一个为零产生零目标；两个都为零也为零。任一目标未知，最终未知，即使另一目标恰好为零也保留资料缺失，不用它填补。

这是两个规则共同限制持仓，不是独立预测证据。139在训练不足时使用131，成熟后的来源也有共同底层成分，不能把同意的次数解释为两份独立信心。趋势、模型系数、训练支持及原退出都是父策略已保存的规则，本轮不另加趋势筛选或周期固定。

两费用分别使用对应费用父目标，不读取另一费用未来或实际成交仓位。完整保留准备收盘到终点前一个收盘的判断日历；终点只有统一开盘退出，不用当天后来形成的收盘。原父参考状态和模型独立连续运行，不由本轮账户盈亏或持股改变。

## 实际进入、调整、退出与再进入

外层实际空仓且最终目标大于零，下一开盘按自身现金、份额市值和分红应收构成的净值买入100份整手。实际持仓与正目标相差不足十个百分点保留份额，达到或超过十个百分点调整。明确零目标请求下一开盘全部卖出，未知目标不发新调整。

若一套父策略退出而另一套仍要求持仓，只有两目标都明确时本轮才给出零目标。若一套资料未知，则没有新目标；这不表示现金观点。两个目标之后重新明确为正，实际账户可以再次进入，无额外等待或来源锁定。受阻请求每收盘按最新目标重算，保留原账户语义；终点开盘全部清仓。

## 账户与有限比较

一项设置、主与较早各基础和压力费用，共四条新账户；131、139及买入持有共十二条保存对照。零新模型拟合、零新参考、零下载。主2020年1月2日至2026年8月14日开盘，较早2015年1月5日至2019年12月31日开盘。每户二十万元、242日年化、现金及无风险收益零。

基础佣金万分之二且最低5元，滑点万分之五；压力佣金万分之四且最低5元，滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、登记日分红权益、除息应收和实际到账均保持。四项必要测试通过后冻结，一次运行。失败结束此交集设置，不换来源或比例救回。结果是已看历史后的研究，不构成独立验证。

## 两套父策略的全部原始中文规则

以下完整列出139的底层因素、进入、退出及131的区别。原139轮所述历史训练、测试和结果属于保存来源，本轮只读取其原收盘目标。141已关闭的周期固定机制不参与本轮。

"""
    (ROOT / "docs/510300_CONSENSUS_REFERENCE_TARGET_V1.md").write_text(intro+inherited, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_consensus_reference_target_v1/tests_receipt.json", {
        "recorded_at": now(), "exit_code": 0, "passed": 4, "seconds": 3.55,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_consensus_reference_target_v1.py", "output": "4 passed in 3.55s"}, exclusive=True)
    print("第142轮共同目标规则与四项实际测试回执已保存，尚未冻结和生成账户。", flush=True)


if __name__ == "__main__":
    main()
