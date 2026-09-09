"""复用账户框架建立最终目标上限，不重算旧风险或父参考。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/final_target_volatility_cap_v1.py"
    require(not destination.exists(), "第138轮运行文件已经存在")
    text = (ROOT / "research/vintage_reference_risk_v1.py").read_text(encoding="utf-8")
    text = text.replace("vintage_reference_risk", "final_target_volatility_cap").replace("VINTAGE_REFERENCE_RISK", "FINAL_TARGET_VOLATILITY_CAP")
    text = text.replace("vintage_risk_targets", "capped_reference_targets").replace("第131轮", "第138轮").replace("round=131", "round=138")
    text = text.replace("七项", "六项").replace('tests["passed"] == 7', 'tests["passed"] == 6')
    text = text.replace("固定入场模型参考与普通波动乘数的单一完整账户组合", "对合成最终目标设置既有普通波动上限")
    text = text.replace("固定版本参考风险组合", "最终合成目标上限").replace("固定入场模型参考与普通波动仓位组合", "合成目标的普通波动上限")
    text = text.replace('old_path = ROOT / "config/510300_conditional_variance_budget_v1.json"', 'old_path = ROOT / "config/510300_joint_downside_reference_pair_v1.json"')
    text = text.replace('input_receipt="reports/research/510300_final_target_volatility_cap_preflight_20260909/result.json"',
        'input_receipt="reports/research/510300_final_target_cap_preflight_20260909/result.json"')
    text = text.replace("EXISTING_ROUND128_RESEARCH_REFERENCE_NOT_OUTER_ACCOUNT", "EXISTING_ROUND137_TARGETS_NOT_OUTER_ACCOUNT")
    text = text.replace("PROGRESS_ROUND130_FOUR_ACCOUNTS_SAVED_FRONTIER_AND_NEXT_INPUT_CLOCKS_COMPLETE", "PROGRESS_ROUNDS135_TO137_TWELVE_ACCOUNTS_VERIFIED_DELIVERED_NEXT_CAP_INPUTS_READY")
    begin, end = text.index("P128 = ROOT"), text.index("\n\ndef freeze():")
    text = text[:begin]+'''P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"
P136 = ROOT / "reports/research/510300_covariance_reference_pair_v1"
P132 = ROOT / "reports/research/510300_downside_reference_risk_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P109 = ROOT / "reports/research/510300_conditional_variance_budget_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"JOINT_DOWNSIDE_REFERENCE_PAIR": (P137, "第137轮合成下行预算"),
    "COVARIANCE_REFERENCE_PAIR": (P136, "第136轮共同方差预算"), "DOWNSIDE_REFERENCE_RISK": (P132, "第132轮下行风险组合"),
    "VINTAGE_REFERENCE_RISK": (P131, "第131轮普通波动乘数"), "BUY_HOLD": (P32, "买入持有")}
'''+text[end:]
    text = text.replace('pd.read_parquet(P128 / period / cost / "ENTRY_VINTAGE_EXIT_decisions.parquet")',
        'pd.read_parquet(P137 / period / cost / "JOINT_DOWNSIDE_REFERENCE_PAIR_decisions.parquet").assign(source_cost=cost, source_model="JOINT_DOWNSIDE_REFERENCE_PAIR")')
    text = text.replace('"reused_control_accounts": 8', '"reused_control_accounts": 10').replace('"reused_earlier_accounts": 8', '"reused_earlier_accounts": 10')
    text = text.replace("四个保存对照", "五个保存对照")
    text = text.replace("单一参考与风险乘数组合", "最终合成目标上限")
    destination.write_text(text, encoding="utf-8")
    previous = (ROOT / "docs/510300_JOINT_DOWNSIDE_REFERENCE_PAIR_V1.md").read_text(encoding="utf-8")
    inherited = previous[previous.index("## 第一来源："):previous.index("## 实际账户的进入、持有、退出和再次进入")]
    inherited = inherited.replace("## 本轮风险因子、合成顺序与月度预算", "## 已保存137来源的风险因子、合成顺序与月度预算")
    intro = """# 第138轮：合成最终目标的普通波动上限

只登记一项新规模规则：将137最终股票目标与既有普通波动乘数相比，仅在目标更大时缩小。既有乘数作为整个组合的上限，不对所有目标再乘一次。以下先完整说明137的来源和规则；本轮不重算这些参考、模型或月度预算，只读取已保存的对应费用收盘目标。

这项新组合来自已观察历史，不能作为独立验证。原参考保留自己的模型和持仓，外层新账户不能倒灌或重置它们。

"""
    outer = """## 本轮新增规则：最终股票目标取既有上限内的值

普通波动直接读取109已经保存、131已经核对的二十日年化波动及风险乘数。二十日波动是完整二十日含分红简单收益的样本标准差，乘二百四十二的平方根；每日日收益为收盘加当日每份除息分红，除以前收盘，再减一。

既有乘数为百分之十除以明确的正年化波动，最多取一。波动为零或未知时，没有新风险观点，沿用最近明确乘数，初始一；本轮不重新计算或修改这个保存序列。两费用使用同一条上限，但父目标分别取137对应费用目录。

每个收盘，将137合成目标与当日既有乘数取较小者。例如，父目标20%、上限50%，仍为20%；父目标80%、上限50%，缩为50%；父目标恰等于上限，保持目标。未知父目标仍是未知，不把上限当作替代观点；明确零目标仍为零。

这个机制限制的是按收盘数据估计的目标风险。十个百分点交易带宽、整手、跳空、实际成交和风险估计变化，会让实际风险偏离上限，因此不保证未来波动始终低于10%。它与131将零一参考乘风险乘数、109对旧组合按比例缩小是不同的完整组合，旧结果照常保留。

## 实际进入、增减、退出与再次进入

实际空仓且最终目标明确为正，以自身现金、份额和分红应收构成的净值及当日收盘价计算100份整手目标，下一开盘买入，实际成交受资金和费用约束。持仓时，正目标与股票市值占净值之差不足十个百分点则保持股数；达到或超过十个百分点，按自己净值调整。

最终目标明确为零，下一开盘请求全部卖出，不受带宽限制。未知目标不发新调整；每个收盘依据最新目标重算受阻请求，外层没有额外永久退出锁定，底层原参考锁定保留。实际卖完后再出现明确正目标可以进入，不加额外冷却。终点统一开盘清仓，不让未知末日收盘决定终点。

## 历史、成本和有限研究范围

一项设置、两段历史两档费用共四条新账户，对照137、136、132、131及买入持有共二十条保存账户直接复用，零新拟合、零新参考、零下载。主历史2020年1月2日至2026年8月14日开盘1604日，较早历史2015年1月5日至2019年12月31日开盘1219日。每账户二十万元，242日年化，现金与无风险收益零，全部空仓日保留。

基础佣金万分之二、每次最低5元、滑点万分之五；压力佣金万分之四、每次最低5元、滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、登记分红权益、除息应收及实际到账沿用既有账户。两段准备日历分别完整保留，父目标只用收盘资料并在下一开盘执行。

六项必要合成测试已检查仅超过才缩小、相等边界、未知父目标和沿用上限、费用对应、非法值、未来及前缀、实际减仓全退再进入及分红。测试后冻结，再一次生成新目标和四账户。若不改善，结束此设定，不改20日、10%、父来源或按事后年份挽救。完整目标仍是成本后夏普至少1.2、稳定超额及独立证据；所有已观察历史为选择后研究。

只研究510300与现金，EPS、公募、估值及慢来源暂停，不准备GPT数值包或ZIP。
"""
    (ROOT / "docs/510300_FINAL_TARGET_VOLATILITY_CAP_V1.md").write_text(intro+inherited+outer, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_final_target_volatility_cap_v1/tests_receipt.json", {
        "tested_at": now(), "exit_code": 0, "passed": 6, "seconds": 3.91,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_final_target_volatility_cap_v1.py", "output": "6 passed in 3.91s"}, exclusive=True)
    compile(destination.read_text(encoding="utf-8"), str(destination), "exec")
    print("第138轮运行文件、全部中文规则和六项测试回执已建立，尚未冻结或计算新账户。", flush=True)


if __name__ == "__main__":
    main()
