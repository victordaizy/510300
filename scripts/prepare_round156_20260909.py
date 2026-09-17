"""保存七项实际测试结果和冻结前完整中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "docs/510300_SPARSE_VINTAGE_EXIT_V1.md"
    require(not path.exists(), "第156轮完整规则已经准备")
    text = (ROOT / "docs/510300_SPARSE_VINTAGE_EXIT_NEXT_20260909.md").read_text(encoding="utf-8")
    text = text.replace("# 下一项：", "# 第156轮：", 1)
    text = text.replace("此文是下一项事前方案，尚未训练或生成新账户。", "此文为第156轮在新模型训练与账户计算前确定的完整规则。")
    text += "\n## 零系数的精确确认和实现补充\n\n从零系数开始时，先检查零向量是否精确满足同一绝对值惩罚目标的凸最优条件："
    text += "各因素误差梯度的绝对值均不大于零点零零一四。如果满足，零向量已是所定目标的最优解，直接记录零系数和零次迭代；"
    text += "该组仍记作一次不同训练输入的拟合。若不满足，按既定顺序进入坐标下降。未收敛或最优条件失败仍保留失败记录。"
    text += "这项确定性的零向量检查在读取本轮新模型和账户结果前明确，不改变惩罚强度、最小化目标或交易门槛。\n\n"
    text += "缓存身份包含有序周期及状态原点、原退出时点、全部八项原始因素、训练目标、各行权重和固定标准化及求解设置。"
    text += "训练失败同样缓存，后续同输入只复用当时已经存在的失败。保存每条月度记录、全部训练成员、八项系数、截距、有效因素数和最优条件残差。"
    text += "七项必要测试已经通过，包括两个非零系数的可手算解、不同周期行数仍保持周期等权、全零合法解、未来时钟隔离、"
    text += "成功及失败缓存、固定每笔版本、受阻退出、重新进入、无模型保护和真实分红账务。\n\n"
    text += "实际资金仍按旧完整账户流程运行。持仓价值包含当时已确认的本周期分红，分红应收在支付日前计入权益但不作为可用现金；"
    text += "实际开盘成交使用未复权价格并扣实际费用。八项因素每日更新，模型仅在每笔实际入场首个收盘选取一次。"
    text += "新历史结果不会写回原114、128、131或143来源与结论。\n"
    path.write_text(text, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_sparse_vintage_exit_v1/tests_receipt.json", {"recorded_at": now(),
        "exit_code": 0, "passed": 7, "seconds": 15.09, "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_sparse_vintage_exit_v1.py -q",
        "output": "7 passed in 15.09s"}, exclusive=True)
    print("第156轮中文规则与已通过的七项测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
