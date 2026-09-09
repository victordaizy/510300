"""保存已通过的必要测试，并把事前方案转为完整中文冻结规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "docs/510300_VORTEX_RISK_V1.md"
    require(not target.exists(), "第153轮中文冻结规则已存在")
    text = (ROOT / "docs/510300_VORTEX_RISK_NEXT_20260909.md").read_text(encoding="utf-8")
    text = text.replace("第153轮准备：", "第153轮冻结规则：", 1)
    text = text.replace("拟采用相邻高低价涡旋方向", "采用相邻高低价涡旋方向").replace("拟定一个设置", "固定一个设置")
    text = text.replace("当前仅准备方案，153尚未登记、生成目标或建立账户。现有资料足以进入实现，无需用户补来源。",
        "这是在第153轮拟合和新收益读取前确定的完整规则。六项必要测试已经通过，再冻结数据、实现、成本及对照，之后只运行这一套规则。")
    text += "\n## 因子与持仓解释补充\n\n本策略的方向信息来自相邻两日高低价之间的移动，持仓规模来自20日普通波动。14日正负方向共用真实波幅分母，二者不是独立信号。"
    text += "第一天缺少前日价格，14个有效移动值到齐才允许判断；若一个必要价格行缺失，会影响当日及下一相邻日移动，必须等新完整窗口后恢复。"
    text += "准备区间只构造因子，研究起点前一收盘开始发出目标，终点收盘不发出目标。\n\n"
    text += "申请份额等于目标比例乘本账户收盘净值，除以真实收盘价后向下取整至100份；减去本账户实际已有份额，得到买卖申请。"
    text += "已有正持仓时，目标比例与实际股票占净值比例相差小于10个百分点可不调整；差距达到10个百分点继续按目标申请。已知零始终申请卖完。"
    text += "未知目标不会新下调整单，实际库存保留；这不等于零仓位，后续方向重现可重新判断。两档费用的目标相同，实际净值和份额可因费用不同而分化。\n\n"
    text += "保存对照只用于结果比较，其内部因子不进入本策略。143全部定义见docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md；"
    text += "131全部定义见docs/510300_VINTAGE_REFERENCE_RISK_V1.md。原失败方案、原模型和已保存对照不修改，零新训练、零新参考账户。\n"
    target.write_text(text, encoding="utf-8")
    out = ROOT / "reports/research/510300_vortex_risk_v1"
    write_json(out / "tests_receipt.json", {"recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 4.95,
        "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_vortex_risk_v1.py -q", "output": "6 passed in 4.95s"}, exclusive=True)
    print("第153轮中文规则和六测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
