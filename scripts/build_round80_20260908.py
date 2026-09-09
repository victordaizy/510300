"""复用上一轮账户骨架，经济目标单独登记，旧来源不改动。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "research/economic_regret_v1.py"
    protocol = ROOT / "docs/510300_ECONOMIC_REGRET_EXIT_V1.md"
    if target.exists() or protocol.exists():
        raise ValueError("第80轮已有来源或协议，不覆盖")
    source = (ROOT / "research/directional_continuation_v1.py").read_text(encoding="utf-8")
    source = source.replace("directional_continuation", "economic_regret").replace("DIRECTIONAL_CONTINUATION", "ECONOMIC_REGRET")
    source = source.replace("directional_chinese_formula", "economic_chinese_formula").replace("DirectionalExitController", "EconomicExitController")
    source = source.replace("import FEATURES, CN, fit_economic_regret,", "import FEATURES, CN, economic_weights, economic_support, fit_economic_regret,")
    source = source.replace("第79轮", "第80轮").replace("round=79", "round=80").replace("方向概率", "经济损失")
    source = source.replace('P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"', 'P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"\nP79 = ROOT / "reports/research/510300_directional_continuation_v1"')
    source = source.replace('"REARM_NONE": (P32, "原价格时间退出，无学习")', '"DIRECTIONAL_CONTINUATION_EXIT": (P79, "第79轮纯方向概率退出")')
    source = source.replace('model_loss="CYCLE_EQUAL_WEIGHTED_BINARY_LOG_LOSS_L2"', 'model_loss="NORMALIZED_ABSOLUTE_CONTINUATION_ADVANTAGE_WEIGHTED_LOG_LOSS_L2"')
    source = source.replace('exit_probability=.5,', 'exit_score_threshold=.5, standardization_basis="ORIGINAL_CYCLE_EQUAL_WEIGHTS", score_is_calibrated_probability=False,')
    source = source.replace('PROGRESS_ROUND78_COMPLETED_AND_DELIVERED', 'PROGRESS_ROUND78_AND_79_COMPLETED_AND_DELIVERED')
    source = source.replace('        eligible = len(ids)', '        if len(rows):\n            rows["economic_weight"] = economic_weights(rows.target, rows.sample_weight)\n        eligible = len(ids)', 1)
    source = source.replace('                stored = fit_economic_regret(rows, cfg)\n                status = "FIT_COMPLETE" if stored is not None else "NO_VIEW_SINGLE_CLASS"',
        '                status = economic_support(rows)\n                if status == "ECONOMIC_FIT_SUPPORT_AVAILABLE":\n                    stored = fit_economic_regret(rows, cfg)\n                    require(stored is not None, "双类经济支持下没有模型")\n                    status = "FIT_COMPLETE"')
    source = source.replace('"sample_weight": float(r.sample_weight), "fit_status": status', '"sample_weight": float(r.sample_weight), "economic_weight": float(r.economic_weight), "fit_status": status')
    source = source.replace('"single_class_no_view": sum(r["status"] == "NO_VIEW_SINGLE_CLASS" for r in receipts),',
        '"single_class_no_view": sum(r["status"] == "NO_VIEW_SINGLE_ECONOMIC_CLASS" for r in receipts),\n        "zero_economic_no_view": sum(r["status"] == "NO_VIEW_NO_ECONOMIC_DIFFERENCE" for r in receipts),')
    source = source.replace("连续两个收盘预测继续占优概率低于一半，学习条件请求退出", "连续两个收盘经济持有评分低于一半，学习条件请求退出")
    source = source.replace("继续占优概率退出", "经济错判损失退出")
    target.write_text(source, encoding="utf-8")
    old = (ROOT / "docs/510300_DIRECTIONAL_CONTINUATION_EXIT_V1.md").read_text(encoding="utf-8")
    factor_table = old[old.index("## 因子和训练目标"):old.index("每个原参考持仓状态")]
    entry = old[old.index("## 进入、退出及再进入"):old.index("## 一次性评价")]
    entry = entry.replace("计算概率", "计算经济持有评分").replace("继续占优概率", "经济持有评分").replace("概率不可用", "评分不可用").replace("概率字段单独保存，不冒充收益预测数值", "评分字段单独保存，不冒充实际胜率或收益预测").replace("概率恢复", "评分恢复")
    text = """# 第80轮：按经济错判损失学习退出

唯一新假设是让错误决定的经济代价进入持仓退出训练。旧第79轮只学继续占优的方向，主0.487、较早0.620，没有改善；本轮不改变正类、阈值、确认天数、输入、时间窗口或成交规则，只改变训练损失的权重。前一目标轮完成78和79账户、核对及交付，属于实际进展。

有界查重确认：旧第五轮直接策略效用是在市场日收益上学习连续仓位和换仓成本；旧一日/五日可避免损失研究含严重性加权回归和两阶段模型。它们不同于本轮原实际持仓继续价值样本的经济代价分类，旧失败仍保留，不声称历史从未使用过损失权重。成本敏感学习的一般依据见[Elkan的原始论文](https://cseweb.ucsd.edu/~elkan/rescale.pdf)，本轮具体权重是本研究的选择，不能从理论推断510300会有效。

""" + factor_table + """原始目标仍是“继续持有至原参考周期自然退出”相对“下一开盘卖出”的净收益差，已计原滑点、佣金差和新增分红权益，以原下一开盘持仓市值归一。正差额代表继续较好，负差额代表退出较好；沿用原标签对提前退出受阻成交的近似，不把标签近似当成已实现利润。

每月首个交易日收盘，只纳入已自然退出的最近最多20个原参考周期，至少10周期100状态。原141个检查时点、预计114个成熟月份和27个早期无模型保持。基础权重仍是每周期总量1、周期内等权；先按成熟时间筛选，再计算任何权重。

每条状态的经济损失权重，等于其基础权重乘原净收益差的绝对值。再将全部经济权重除其合计、乘原基础权重合计，保持训练总权重规模。这样正负两种错判都按会丢掉的收益比例重视，而不是按实际历史资金大小重视；最终各周期总权重可以不同，不能再称拟合损失按周期等权。正类仍为原差额严格大于零。

零差额权重为零，不进入分类损失；零权重状态仍保存在成熟样本和原标准化统计中。整个训练窗口经济差额全零时没有经济判断；只有一类拥有正经济权重时没有双类模型，都保留无观点，不填0或1。归一只使用当次已成熟标签，不让未来大涨大跌改变过去权重。

为单独比较损失权重，八项因子的均值和总体标准差仍按原周期基础权重计算。标准差不高于0.000000000001的项尺度置1；每项减均值除尺度，再限于负5至5。模型最小化上述经济权重的二元对数损失加系数平方惩罚；带截距、正则倒数1、纯平方惩罚、有限内存拟牛顿求解、容差一亿分之一、最多1000次迭代，不按类别另平衡，不热启动。未收敛或非有限系数没有模型，不换求解器。

标准化输入乘各月系数后加截距，再取评分相反数的自然指数、加1，以1除以上述数，得到零至一的经济持有评分。它是受错误代价影响的决策分数，不是校准胜率，也不是预期收益金额。所有实际月度系数、基础和经济权重另存；数值仍可按中文规则复算。

""" + entry + """## 一次性评价

主评价2020年1月2日至2026年8月14日开盘终点，共1604日；较早2015年1月5日至2019年12月31日开盘，共1219日。各20万元开始，242日年化，现金与无风险收益为零，完整空仓日保留。基础每边佣金万分之二、最低5元、滑点万分之五；压力佣金万分之四、最低5元、滑点千分之一。最小100份、价格最小变化0.001元；T+1、方向涨跌停、可用现金和全部分红权利不变。

只登记这一项经济损失目标。6项必要测试通过后冻结，直接跑两段两费用4个新账户；原平均收益退出、第79轮方向概率、原各半组合、买入持有复用16个保存对照。失败不扫权重指数、归一、阈值、正则或确认天数。评价历史已经多次观察，点估计达到1.2仍不能代替独立验证；未达到就保留失败并继续不同机制。EPS、公募、其他慢数据继续暂停，不做GPT数值包或额外安全审计。
"""
    protocol.write_text(text, encoding="utf-8")
    print("第80轮经济退出来源与短中文协议已生成，尚未冻结或运行。")


if __name__ == "__main__":
    main()
