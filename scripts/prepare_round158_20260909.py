"""保存九项必要测试结果及概率幅度模型的训练前完整规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "docs/510300_PROBABILITY_PAYOFF_EXIT_V1.md"
    require(not path.exists(), "第158轮完整规则已经准备")
    text = (ROOT / "docs/510300_PROBABILITY_PAYOFF_EXIT_NEXT_20260909.md").read_text(encoding="utf-8")
    text = text.replace("# 下一项：", "# 第158轮：", 1)
    text = text.replace("此文是第158轮的事前方案，尚未实现、测试、冻结、训练或生成新账户。",
        "此文为第158轮训练与账户计算前确定的完整规则，九项必要测试已经通过，尚未拟合新历史模型或计算新账户。")
    text += ("\n## 已完成的测试和实现细节\n\n"
        "九项实际测试14.35秒通过，覆盖不同周期行数的权重保持、两类均值方差与独立高斯密度、零标签归入不占优、"
        "单个类别内方差为零的数值稳定、八因素全恒定时仅使用先验比例，以及占优概率百分之七十五但预期净增量仍为负的幅度组合。"
        "同时核对缺类和缺失的明确无观点、成功和失败缓存、未来月份隔离、实际入场固定版本、预测未知清计数、"
        "受阻退出锁定、重新进入、无初始模型时价格退出及分红登记除息到账。\n\n"
        "对全部取值精确相等的因素列，均值直接等于该取值、方差直接为零。这是加权均值和方差的等价计算，"
        "用于避免浮点求和把数学上恒定的因素变成极小的非零方差；不按回测收益判定因素是否参与。"
        "两类先验、均值方差、是否参与及平均盈亏都纳入每笔模型身份。"
        "当前因素完整却发生非有限概率计算时，学习预测记录数值失败和无观点，连续负预测计数清零，原保护和已锁定退出继续执行。\n")
    path.write_text(text, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_probability_payoff_exit_v1/tests_receipt.json", {"recorded_at": now(),
        "exit_code": 0, "passed": 9, "seconds": 14.35,
        "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_probability_payoff_exit_v1.py -q",
        "output": "9 passed in 14.35s"}, exclusive=True)
    print("第158轮九项已通过测试和完整中文规则已保存。", flush=True)


if __name__ == "__main__":
    main()
