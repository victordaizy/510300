"""交付第121轮简洁中文结果及已准备的下一项，不重复回测。"""
import json
from research.joint_entry_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round


def main():
    next_path = ROOT / "docs/510300_CYCLE_ANALOGUE_NEXT_20260909.md"
    status = "COMPLETED_JOINT_POLICY_WEAKER_THAN_MYOPIC_BOTH_NOT_TARGET"
    decision = ("本轮两个固定方法均未达到净夏普1.2。联合进出场主基础／压力净夏普0.090／0.042，年化0.21%／负0.53%，最大回撤37.14%／39.03%；"
        "只看单日奖励的对照为0.460／0.453，年化6.78%／6.65%，最大回撤35.72%／35.89%。较早历史联合方案约0.000／负0.044，对照0.131／0.115。"
        "两者均不能替代第114轮局部候选，更不能宣布完整目标成功。主基础联合方案70次成交，对照12次；实际成交路径的价格加分红利润在扣费前分别为13476.20和111199.50元，"
        "费用分别10672.30和2211.13元。这是保存成交路径分解，未另跑零费用账户。后续价值的引入没有改善本次真实账户结果，关闭固定两方法，不调参数挽救。")
    verification = ("九项最终必要测试4.14秒；两个方法、八个新独立账户以及训练情景和模型核心计算36.30秒，均不含设计、文档和保存核对耗时。"
        "16590个名义单步情景来自3318个不同训练日期，不能当作独立交易业绩；141个月经验模型全部可用，514次十二状态策略方程求解，保存282份策略表，零新全程参考账户。"
        "已核3456个市场状态、16590个单步经济、141组策略方程、11292次实际收盘决策、172个完整周期和48条分年增量；没有重跑账户。"
        "1692个同状态策略格中301个动作不同。名义训练存在1次方向受阻，八个实际账户均无未成交或未知模型日期。"
        "下一项已完成现成八因子及不同成熟周期支持检查：1461行、34个周期，114月支持、27月原不足保留；未读目标列、未训练收益模型、未跑新账户。"
        "历史区间已反复观察，不是独立验证。目标继续，EPS及慢源暂停，无GPT数值包或额外安全审计。")
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300联合进入持有退出_第121轮_20260909/联合进入持有退出_结果及全部中文规则.md",
        "联合进入、持有和退出", status, decision, verification, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "CYCLE_ANALOGUE_INPUT_SUPPORT_COMPLETE_DESIGN_PENDING", "复用成熟自然周期的八状态，比较不同历史周期的相似状态继续价值；不补慢源")
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
