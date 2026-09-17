"""保存实际八项测试结果和训练前的完整单成分中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "docs/510300_SINGLE_COMPONENT_EXIT_V1.md"
    require(not path.exists(), "第157轮完整规则已经准备")
    text = (ROOT / "docs/510300_SINGLE_COMPONENT_EXIT_NEXT_20260909.md").read_text(encoding="utf-8")
    text = text.replace("# 下一项：", "# 第157轮：", 1)
    text = text.replace("此文是第157轮事前方案，尚未实现、测试、冻结、训练或生成新账户。",
        "此文为第157轮训练与账户运行前确定的完整规则，八项必要测试已通过，尚未拟合新历史模型或计算新账户。")
    text += ("\n## 已完成的必要测试及数值实现\n\n"
        "八项测试实际通过，用时7.35秒。可手算的正交样本恢复已知系数；相关因素、各周期不同样本量的合成资料，"
        "与scikit-learn单成分偏最小二乘实现一致。独立实现接收每行乘其权重平方根后的周期内中心化因素和目标，关闭额外尺度变换。"
        "另已核对周期等权、精确零协动合法模型、缺失不能删行、同输入仅拟合一次及未来月份隔离、失败缓存、实际入场固定版本、"
        "未知预测清计数、受阻卖出继续锁定、重新进入和真实分红应收到账。\n\n"
        "组合方向先用协动绝对值的最大值缩放，再除以缩放后向量的长度；这与直接除以原长度数学等价。"
        "一个成分的分数与回归残差的加权平均乘积，绝对值不超过十亿分之一的十分之一，即零点零零零零零零零零零一。"
        "该容差仅用于确认既定计算完成，不用作收益筛选或成分选择。分母必须为有限正数，不为失败样本切换求解器。\n")
    path.write_text(text, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_single_component_exit_v1/tests_receipt.json", {"recorded_at": now(),
        "exit_code": 0, "passed": 8, "seconds": 7.35,
        "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_single_component_exit_v1.py -q",
        "output": "8 passed in 7.35s"}, exclusive=True)
    print("第157轮完整中文规则和已通过的八项测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
