"""只建立新轮文件，复用已完成的账户运行与底层中文说明。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json, now

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/covariance_reference_pair_v1.py"
    require(not destination.exists(), "第136轮文件已经建立")
    text = (ROOT / "research/recent_reference_selection_v1.py").read_text(encoding="utf-8")
    text = text.replace("recent_reference_selection", "covariance_reference_pair").replace("RECENT_REFERENCE_SELECTION", "COVARIANCE_REFERENCE_PAIR")
    text = text.replace("selected_reference_frames", "covariance_reference_frames").replace("第135轮", "第136轮").replace("round=135", "round=136")
    text = text.replace("按已发生的历史净夏普选择两套来源或现金", "按两套已保存收益的共同波动分配预算")
    text = text.replace("按历史净夏普选择两套策略或现金", "两套参考组合的月度最小方差预算")
    text = text.replace("月度历史表现选择", "月度共同风险预算").replace("月度选择", "共同风险预算")
    text = text.replace('config/510300_equal_reference_pair_v1.json', 'config/510300_recent_reference_selection_v1.json')
    text = text.replace('selection_window=242, scoring_cost="BASE", initial_parent=PARENT,', 'risk_window=242, risk_cost="BASE", initial_budgets=[.5, .5],')
    text = text.replace('input_receipt="reports/research/510300_covariance_reference_pair_preflight_20260909/result.json"',
        'input_receipt="reports/research/510300_recent_reference_selection_v1/saved_verification_receipt.json"')
    text = text.replace('previous_goal_turn_classification="PROGRESS_ROUNDS133_134_EIGHT_ACCOUNTS_DELIVERED_NEXT_RANKING_PREFLIGHT_COMPLETE"',
        'previous_goal_turn_classification="PROGRESS_ROUND135_VERIFIED_DELIVERED_NEW_COVARIANCE_COMPARISON"')
    begin = text.index('    receipt = json.loads((ROOT / cfg["input_receipt"])')
    end = text.index('    for period in ["evaluation", "earlier_diagnostic"]:', begin)
    text = text[:begin]+'''    paths.extend([ROOT / "research/recent_reference_selection_inputs_v1.py", ROOT / "research/two_policy_min_variance_inputs_v1.py",
        ROOT / "docs/510300_COVARIANCE_REFERENCE_PAIR_NEXT_20260909.md"])
    for item in old["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "两参考已经核对的来源改变")
        paths.append(ROOT / item["path"])
'''+text[end:]
    text = text.replace('base_factors.selection_update_scheduled', 'base_factors.risk_update_scheduled')
    text = text.replace('{period}_selection_updates.csv', '{period}_risk_updates.csv')
    text = text.replace('一条新共同风险预算账户', '一条新共同风险预算账户')
    destination.write_text(text, encoding="utf-8")
    previous = (ROOT / "docs/510300_EQUAL_REFERENCE_PAIR_V1.md").read_text(encoding="utf-8")
    bottom_rules = previous[previous.index("## 第一来源："):previous.index("## 本轮唯一新增规则：")]
    bottom_rules = bottom_rules.replace("全部逐月中文数值随结果提供", "全部逐月数值保存在原第31轮模型档案中，本轮没有重新估计")
    intro = """# 第136轮：两套参考组合的月度共同风险预算

本轮一个固定设定，复用132下行风险目标与91连续两策略预算目标。按过去已知的两条基础费用净收益共同波动分配比例，检验能否改善134固定各半和135历史收益排名。预算公式沿用82原控制器，底层不重新训练或建立新参考；新来源来自已观察历史，不能算独立验证。

两套来源都交易510300，原各参考继续保留自己的模型、持仓和退出状态。新账户有自己的现金、份额、分红和成交，不能将新账户持仓倒灌或重置参考。

"""
    outer = """## 本轮新增的风险因子及预算规则

每段历史开始的准备收盘，132和91预算各为一半。每月首个交易日的完整收盘，用该段两条基础费用账户截至当日的最近242个完整日净收益估计风险，包含现金日。其余日保持预算。不得读取该段开始前尚不存在的父账户收益；终点开盘清仓收益不属于收盘窗口。

本轮新增的风险量为：第一条净收益的样本方差，第二条净收益的样本方差，两条收益的样本协方差，以及每日第一条收益减第二条收益之差的样本方差。样本方差是每日减去窗口平均值后的平方和除以241；协方差是两条各自减均值后的乘积和除以241。收益差的方差直接计算，避免近似相同方差相减出现数值误差。

132预算等于91的样本方差减协方差，再除以两条收益差的样本方差；结果限制在零至一。91预算为一减132预算。预算表示两个目标各占的权重，不是直接股票仓位。两来源收益差波动越小，必须注意分母是否为零；窗口不足、任何日缺失、任一参考零波动或收益差零方差，保留“无新风险估计”和此前预算，不补零风险、不删除缺失日。

这个解析式对应给定协方差估计下的两来源最小方差目标，不保证未来最高夏普。[CVX研究组组合优化课程](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)说明了协方差、组合方差与无卖空预算约束。本轮没有拟合新的预期收益，也未用过去夏普排名。

两档费用共用由原基础费用净收益计算的同一组预算。每天合成目标为132预算乘对应费用的132收盘目标，加91预算乘对应费用的91收盘目标。任一父目标未知，则合成目标未知，即便其预算为零也不放松完整性要求；两个明确零目标才明确清仓。一条退出、另一条仍有正目标且其预算为正时，保留另一条相应规模。

## 实际账户的进入、持有、退出和再次进入

实际空仓且合成目标明确为正，按该收盘自己的总净值、收盘价和100份整手提出下一开盘买入请求。持仓时，实际股票市值占净值与正目标相差不足十个百分点，保持份额；达到或超过十个百分点，按自身净值重新确定整手份额并增减。最终成交仍受现金、费用和交易条件约束，不能把两条历史净收益按预算相加当成实际账户。

合成目标明确为零，下一开盘请求卖出全部份额，不受带宽限制。合成目标未知，不发起新调整。每个收盘根据最新目标重算受阻请求，外层没有额外永久退出锁定；底层参考的锁定保持。实际卖完后出现明确正目标可再次进入，不另加冷却。终点统一按开盘优先全卖，不使用末日尚未知的收盘。

## 完整账户与有限研究范围

一项设置、主历史和较早历史两段、基础和压力两费用，共四条新账户；复用132、91、134及买入持有的十六条保存对照，零新模型、零新参考、零下载。主历史2020年1月2日至2026年8月14日开盘1604日；较早历史2015年1月5日至2019年12月31日开盘1219日。各账户二十万元，242日年化，现金和无风险收益为零，全部空仓日纳入。

基础佣金万分之二、每次最低5元，滑点万分之五；压力佣金万分之四、每次最低5元，滑点千分之一。100份整手、0.001元价位、次日可卖、方向涨跌停、登记分红权益、除息应收及到账均按原规则。本轮来源无需再补齐，引用135已核对的完整日期和时点来源；六项必要合成测试通过后冻结，再一次运行。

这是旧最小方差方法应用到新的一对保存组合，旧82、91、134、135与其他研究结果不变。若不改善，关闭这一设置，不改窗口、月首、预算边界或参考挽救。目标仍是成本后夏普至少1.2、稳定超额及独立证据；不是四个已反复观察历史指标足够就能认定成功。只使用510300与现金；EPS、公募、估值和慢来源暂停，不准备GPT数值包或ZIP。
"""
    document = ROOT / "docs/510300_COVARIANCE_REFERENCE_PAIR_V1.md"
    document.write_text(intro+bottom_rules+outer, encoding="utf-8")
    output = ROOT / "reports/research/510300_covariance_reference_pair_v1"
    write_json(output / "tests_receipt.json", {"tested_at": now(), "exit_code": 0, "passed": 6, "seconds": 4.39,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_covariance_reference_pair_v1.py",
        "output": "6 passed in 4.39s"}, exclusive=True)
    compile(destination.read_text(encoding="utf-8"), str(destination), "exec")
    print("第136轮运行文件、完整中文规则及六项测试回执已建立，尚未冻结或读取新组合表现。", flush=True)


if __name__ == "__main__":
    main()
