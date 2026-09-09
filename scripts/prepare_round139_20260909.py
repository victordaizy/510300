"""复用实际账户流程，建立训练支持条件选择的完整规则和运行文件。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json, now

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/model_support_reference_router_v1.py"
    require(not destination.exists(), "第139轮运行文件已经存在")
    text = (ROOT / "research/recent_reference_selection_v1.py").read_text(encoding="utf-8")
    text = text.replace("recent_reference_selection", "model_support_reference_router").replace("RECENT_REFERENCE_SELECTION", "MODEL_SUPPORT_REFERENCE_ROUTER")
    text = text.replace("selected_reference_frames", "support_routed_frames")
    text = text.replace("第135轮", "第139轮").replace("round=135", "round=139")
    text = text.replace("按已发生的历史净夏普选择两套来源或现金", "按已形成的模型训练支持选择已有策略")
    text = text.replace("按历史净夏普选择两套策略或现金", "按退出模型的训练支持选择策略")
    text = text.replace("月度历史表现选择", "模型训练支持选择").replace("月度选择", "训练支持选择")
    text = text.replace('config/510300_equal_reference_pair_v1.json', 'config/510300_joint_downside_reference_pair_v1.json')
    text = text.replace('selection_window=242, scoring_cost="BASE", initial_parent=PARENT,',
        'decision_clock="15:05:00", minimum_training_cycles=10, minimum_training_rows=100, saved_models="reports/research/510300_within_cycle_exit_v1/saved_models.json",')
    text = text.replace('input_receipt="reports/research/510300_model_support_reference_router_preflight_20260909/result.json"',
        'input_receipt="reports/research/510300_model_support_router_preflight_20260909/result.json"')
    text = text.replace('PROGRESS_ROUNDS133_134_EIGHT_ACCOUNTS_DELIVERED_NEXT_RANKING_PREFLIGHT_COMPLETE', 'PROGRESS_ROUND138_FOUR_ACCOUNTS_VERIFIED_CLOSED_AND_MODEL_SUPPORT_CLOCKS_READY')
    begin, end = text.index("P132 = ROOT"), text.index("\n\ndef freeze():")
    text = text[:begin]+'''P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"
P132 = ROOT / "reports/research/510300_downside_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数策略"), MODELS[1]: (P137, "第137轮合成下行预算"),
    "DOWNSIDE_REFERENCE_RISK": (P132, "第132轮下行风险组合"), "BUY_HOLD": (P32, "买入持有")}
'''+text[end:]
    begin = text.index('        source_folders = [(PARENT, P132), (MODELS[1], P91)]')
    end = text.index('        parents_by_cost =', begin)
    text = text[:begin]+'''        source_folders = [(MODELS[0], P131), (MODELS[1], P137)]
        records = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
        require(len(records) == 141 and sum(r["status"] == "FIT_COMPLETE" for r in records) == 114, "原模型训练支持数量改变")
'''+text[end:]
    text = text.replace('support_routed_frames(frame, base_ledgers, parents_by_cost, cfg, start)', 'support_routed_frames(frame, records, parents_by_cost, cfg, start)')
    text = text.replace('base_factors.selection_update_scheduled', 'base_factors.selection_changed')
    text = text.replace('{period}_selection_updates.csv', '{period}_selection_changes.csv')
    text = text.replace("父目标或平方风险来源变化", "父目标或模型记录变化")
    destination.write_text(text, encoding="utf-8")
    previous = (ROOT / "docs/510300_JOINT_DOWNSIDE_REFERENCE_PAIR_V1.md").read_text(encoding="utf-8")
    inherited = previous[previous.index("## 第一来源："):previous.index("## 实际账户的进入、持有、退出和再次进入")]
    inherited = inherited.replace("## 本轮风险因子、合成顺序与月度预算", "## 第137轮来源的合成顺序与月度预算").replace("本轮", "原137轮")
    intro = """# 第139轮：按退出模型训练支持选择已有策略

只登记一个条件选择，使用当时已经形成的原114退出模型记录：支持不足选131普通波动乘数，成熟选137合成下行预算。只读取保存记录和两套对应费用收盘目标，不重新训练、不重新计算父参考。以下完整列出137底层因子，131的差异随后说明；原参考状态不能由新账户倒灌或重置。

该方案是在观察历史后提出，不能当作独立验证，也不能把原历史拟合时点当作当年真实运行证据。

"""
    outer = """## 支持不足时的第131轮来源

131与上述132共用128的入场固定版本参考、全部八项持仓因子、原进入、价格退出、期限退出、学习退出和再次进入规则。区别是规模用普通二十日波动乘数，而不是132的下行波动。普通波动为完整二十日含分红简单收益样本标准差乘二百四十二平方根，乘数为百分之十除以正波动、最多一；零或缺失波动时沿用上次明确乘数，初始一。

128明确进入或持有意向为一时，131目标就是乘数；明确退出意向为零时目标零；未知意向时目标未知。所用普通波动及乘数为109已经保存并核对的资料。131基础和压力费用分别保留对应费用128参考的状态、盈亏及退出。本轮读取131最终收盘目标，不再次叠加其风险乘数。

## 本轮选择因素与判断时钟

使用原114共141条月度训练记录，已有27条支持不足、114条拟合完成。每天收盘后15:05进行研究判断，选择拟合时点不晚于当前15:05的最近一条记录。记录的拟合日要与完整行情索引一致，最近训练周期的实际退出日及索引不得晚于拟合日。不能只按日期忽略分钟，也不能读取以后的成熟记录。

本轮的选择因素为：是否已有训练记录、该记录的拟合状态、训练已结束周期数、训练状态行数、缺失训练因素行数、拟合是否失败及保存模型是否存在。最近记录为拟合完成、模型非空、可拟合、至少十个已结束周期和一百行状态，且训练因素无缺失、没有失败记录，才选137。否则，没有记录或原记录明确为支持不足，选择131。无法识别或内部矛盾的记录停止处理，不自行认定成熟。

支持判断只用于选择两套已保存策略，没有把未知预测补为零或现金，也不改变原学习模型、系数或训练窗口。两档费用共用同一支持状态，所选父目标读取对应费用目录。选中来源的目标未知则外层目标未知，不能临时换另一来源。

原历史首次成熟记录为2017年3月1日15:05，之后记录均成熟。因此本样本主要检验一次从训练准备阶段进入成熟阶段的转换，不是反复识别市场环境的证据。该日期是原门槛和已结束数据的结果，不是新的硬编码切换年份；未来训练记录或训练支持出现变化时仍按相同条件处理。

## 实际进入、退出、调整与再次进入

收盘后确定所选目标，下一开盘依据自己现金、持股和分红应收构成的净值，按100份整手计算买卖。空仓且正目标可进入；持仓与正目标相差不足十个百分点保留份额，达到或超过十个百分点按自身净值增减。

明确零目标请求下一开盘全部卖出，不受带宽限制；未知目标不发新调整。每个收盘重新根据目标处理受阻请求，外层不增加永久退出锁定，原参考内部锁定仍保留。卖完后明确正目标可以再次进入，不增加额外冷却。终点统一开盘优先清仓，不使用当日以后才形成的收盘或训练资料。

## 账户、成本与有限研究范围

一项设置、主与较早两段、基础和压力两档费用，共四条新账户。131、137、132及买入持有十六条保存对照复用，零新模型、零新参考、零下载。主历史2020年1月2日至2026年8月14日开盘1604日；较早历史2015年1月5日至2019年12月31日开盘1219日。每账户二十万元，242日年化、现金和无风险收益零、完整保留空仓日。

基础佣金万分之二、每次最低5元、滑点万分之五；压力佣金万分之四、每次最低5元、滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、登记日分红权益、除息应收及到账均保持。主历史如选择一直为137，仍检查完整实际账户一致性，不能只凭目标推断结果。

六项必要测试已检查支持状态和最近记录、15:06不能提前用于15:05、未来退出与非法状态、费用对应及所选未知目标、未来与前缀、实际减仓清仓再进入及分红。测试通过后冻结，再一次运行。失败不更改原十周期、一百行、时点、来源或年份边界挽救。旧131、137、138及其他记录保持；目标仍是成本后夏普至少1.2、稳定超额及独立证据。

只研究510300和现金，EPS、公募、估值及慢来源暂停，不准备GPT数值包或ZIP。
"""
    (ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md").write_text(intro+inherited+outer, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_model_support_reference_router_v1/tests_receipt.json", {
        "tested_at": now(), "exit_code": 0, "passed": 6, "seconds": 5.48,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_model_support_reference_router_v1.py", "output": "6 passed in 5.48s"}, exclusive=True)
    compile(destination.read_text(encoding="utf-8"), str(destination), "exec")
    print("第139轮运行文件、全部中文规则和六项测试回执已建立，尚未冻结或计算新账户。", flush=True)


if __name__ == "__main__":
    main()
