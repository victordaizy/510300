"""复用账户运行框架和已保存底层规则，建立直接共同下行研究。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json, now

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/joint_downside_reference_pair_v1.py"
    require(not destination.exists(), "第137轮文件已经建立")
    text = (ROOT / "research/covariance_reference_pair_v1.py").read_text(encoding="utf-8")
    text = text.replace("covariance_reference_pair", "joint_downside_reference_pair").replace("COVARIANCE_REFERENCE_PAIR", "JOINT_DOWNSIDE_REFERENCE_PAIR")
    text = text.replace("covariance_reference_frames", "joint_downside_reference_frames")
    text = text.replace("第136轮", "第137轮").replace("round=136", "round=137")
    text = text.replace("两套已保存收益的共同波动", "两套已保存收益合成后的下行幅度")
    text = text.replace("两套参考组合的月度最小方差预算", "两套参考合成收益的月度下行预算")
    text = text.replace("共同风险预算", "共同下行预算")
    text = text.replace('config/510300_recent_reference_selection_v1.json', 'config/510300_covariance_reference_pair_v1.json')
    text = text.replace('risk_window=242, risk_cost="BASE",', 'risk_window=242, downside_benchmark=0., risk_cost="BASE",')
    text = text.replace('input_receipt="reports/research/510300_recent_reference_selection_v1/saved_verification_receipt.json"',
        'input_receipt="reports/research/510300_covariance_reference_pair_v1/saved_verification_receipt.json"')
    text = text.replace('previous_goal_turn_classification="PROGRESS_ROUND135_VERIFIED_DELIVERED_NEW_COVARIANCE_COMPARISON"',
        'previous_goal_turn_classification="PROGRESS_ROUND136_MAIN_BOTH_COSTS_POINT_TARGET_WITH_EXCESS_EARLY_NOT_TARGET"')
    location = text.index('\n\ndef freeze():')
    text = text[:location]+'''\nP136 = ROOT / "reports/research/510300_covariance_reference_pair_v1"
CONTROLS["COVARIANCE_REFERENCE_PAIR"] = (P136, "第136轮共同方差预算")
'''+text[location:]
    text = text.replace('"reused_control_accounts": 8', '"reused_control_accounts": 10').replace('"reused_earlier_accounts": 8', '"reused_earlier_accounts": 10')
    text = text.replace("四条保存对照", "五条保存对照")
    destination.write_text(text, encoding="utf-8")
    previous = (ROOT / "docs/510300_COVARIANCE_REFERENCE_PAIR_V1.md").read_text(encoding="utf-8")
    bottom_rules = previous[previous.index("## 第一来源："):previous.index("## 本轮新增的风险因子及预算规则")]
    inherited_account = previous[previous.index("## 实际账户的进入、持有、退出和再次进入"):previous.index("## 完整账户与有限研究范围")]
    intro = """# 第137轮：两条参考合成后直接最小化下行平方

只登记一个固定的新风险目标，使用132下行风险目标和91连续两策略预算目标。月首根据过去完整基础账户收益，直接使两条收益先合成后的负收益平方平均值最小；底层不新训练、不重新运行参考。新风险定义没有改写136的共同方差方法和已知结果。选择来源使用了已观察历史，不是独立验证。

各底层参考保留自己的模型、持仓、退出锁定和再次进入状态；外层资金账户不能倒灌或重置参考。两条仍都只交易510300。

"""
    new_rule = """## 本轮风险因子、合成顺序与月度预算

每段开始前的准备收盘，两个预算各半。每月首个完整交易日收盘，取该段两条基础费用账户截至当日的最近242个完整日净收益，包含空仓日；月中保持预算。开始前尚不存在的父账户收益保持缺失，终点开盘清仓收益不能当作收盘资料。

第一预算乘132的当日净收益，加剩余预算乘91的当日净收益，得到该日合成收益；其为负则平方，否则记零。将全部242天相加并除以242，便是本轮需要最小化的下行平方幅度。下行基准零，不减去平均收益，不仅按负收益天数作分母，也不先各自截去正收益。先各自截去正收益会错误地取消两来源之间实际存在的正负抵消。

本轮保存的风险因素包括：上述合成后的下行平方平均值；本次更新前旧预算在同一窗口的下行平方平均值；以及风险对第一预算的导数。导数的计算为每一天合成收益中的负值部分，乘以当天132收益减91收益，再对242天取平均后乘二。该导数随第一预算单调不减，因此可在零至一之间寻找凸目标最小点。

当旧预算的导数为零，旧预算已经最优，保持原预算。如果整个区间的风险都向同一方向增加或减少，取对应端点；其他情况在旧预算与应移动的端点之间最多二分64次，直到双精度区间无法继续区分。导数为零的一段表示多个同样优的预算，选择距离旧预算最近的点；不利用未来收益挑选并列答案。64次是数值求解精度设置，不是搜索64种策略。

完整窗口中的风险为零或多个预算平坦，是明确的最优值或并列解；窗口不足或任何日期缺失，则为无新风险观点并保留旧预算。两种状态分别保存，不补零资料，也不删除缺失日。[PyPortfolioOpt官方半方差说明](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html)说明了直接组合半方差的凸问题，以及先计算半协方差再当普通方差使用的近似局限。本轮直接求两来源组合目标，不安装该库，也不把下行风险指标当作夏普。

两档费用共用基础费用收益得到的预算，每日目标仍是预算乘原对应费用收盘目标的合计。任一父目标未知，合成目标未知，即使那条预算为零也保持完整性要求；两个明确零目标才能明确清仓。一条退出、另一条有正目标及正预算时，保留后者相应的规模。真实账户仍由自身资金和成交计算，不能把参考收益线性相加当作成交结果。

"""
    ending = """## 完整账户与有限研究范围

一项设置，两段两费用共四条新账户；对照132、91、134、136及买入持有直接复用，共二十条保存对照，零新模型、零新参考、零下载。主历史2020年1月2日至2026年8月14日开盘1604日，较早历史2015年1月5日至2019年12月31日开盘1219日。各账户二十万元，242日年化，现金和无风险收益为零，空仓日全部保留。

基础佣金万分之二、每次最低5元、滑点万分之五；压力佣金万分之四、每次最低5元、滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、登记日分红权益、除息应收及实际到账均保持。原136已核对相同来源，本轮直接复用，无需新的外部补齐。

六项必要测试已检查解析样例、独立数值求解、抵消、平坦及边界解、费用分工与缺失、未来及前缀、实际部分退出、全退、再进入及分红。通过后冻结，再一次生成新目标和四账户。若没有改善，关闭这个风险目标，不改变下行基准、窗口、预算范围或来源救回。旧132、133、136及其他研究结果保留。完整目标仍为成本后夏普至少1.2、稳定超额及独立证据，当前不能因主历史点值超过门槛就宣布全部完成。

研究只使用510300与现金，EPS、公募、估值和慢来源暂停；不准备GPT数值包或ZIP。
"""
    (ROOT / "docs/510300_JOINT_DOWNSIDE_REFERENCE_PAIR_V1.md").write_text(intro+bottom_rules+new_rule+inherited_account+ending, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_joint_downside_reference_pair_v1/tests_receipt.json", {
        "tested_at": now(), "exit_code": 0, "passed": 6, "seconds": 3.74,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_joint_downside_reference_pair_v1.py", "output": "6 passed in 3.74s"}, exclusive=True)
    compile(destination.read_text(encoding="utf-8"), str(destination), "exec")
    print("第137轮文件、全部中文规则及六项测试回执已建立，尚未冻结或生成新账户。", flush=True)


if __name__ == "__main__":
    main()
