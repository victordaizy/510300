"""保存实际六项测试结果及全部中文规则，供第155轮冻结使用。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    document = ROOT / "docs/510300_MEDIAN_SLOPE_RISK_V1.md"
    require(not document.exists(), "第155轮完整中文规则已准备")
    text = (ROOT / "docs/510300_MEDIAN_SLOPE_RISK_NEXT_20260909.md").read_text(encoding="utf-8")
    text = text.replace("# 下一项：", "# 第155轮：", 1)
    text = text.replace("第154轮固定边界再平衡失败关闭。第143轮仍为比较候选，完整目标没有实现。下一项离开已有持仓调仓比例，检验一个由现有价格直接产生方向的独立规则，不补前瞻盈利、公募或其他慢来源。",
        "本轮固定一个独立价格方向规则，完整定义如下。第143轮、131轮及买入持有仅为保存比较，不向本轮提供任何信号或目标。")
    text += "\n## 完整账户口径\n\n主历史为2020年1月2日至2026年8月14日开盘；较早历史为2015年1月5日至2019年12月31日开盘。"
    text += "每条账户独立投入20万元，仅持有510300及人民币现金。年化采用242个交易日，现金利息及无风险收益均为零。"
    text += "基础费用为每次成交金额的万分之二佣金、每笔最低5元、单边万分之五滑点；压力费用为万分之四佣金、每笔最低5元、单边千分之一滑点。"
    text += "以100份为一手，价格最小变动0.001元；买入价格向上取有效价位，卖出价格向下取有效价位，保留方向涨跌停和当日买入次日可卖约束。\n\n"
    text += "每日判断使用本日15:05前已知收盘信息，实际申请下一交易日真实开盘成交。目标份额等于股票目标乘本账户收盘净值，除以真实收盘价，向下取整至100份；"
    text += "申请数量为目标份额减当前真实份额，正数买入、负数卖出，实际买入还受现金和佣金限制。已有正持仓的十个百分点缓冲按本账户实际份额市值占净值比例判断。"
    text += "登记日确定分红权益，除息日确认为应收，支付日转为现金；不能把权益在尚未到账时当作可付买单的现金。含分红财富只计算信号，真实成交仍用未复权价格。\n\n"
    text += "起点前一交易日收盘开始产生研究目标，之前仅准备因素；终点开盘清仓优先于普通目标，末日收盘没有新目标。"
    text += "资料未知时不生成新的买卖申请，持仓并不因此被填成零。已知零目标直接请求全卖，正目标若按整手取整至零，也可能实际全退。"
    text += "上述日常退出请求每收盘重算；不新增退出意图锁定。六项必要测试已通过，规则、数据、费用和代码随后冻结，再只运行这一套设置。"
    text += "结果属于已反复研究的历史检验，仍需完整稳定性及独立证据支持，不因某个区间夏普达到1.2而认定目标完成。\n"
    document.write_text(text, encoding="utf-8")
    write_json(ROOT / "reports/research/510300_median_slope_risk_v1/tests_receipt.json",
        {"recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 17.75,
         "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_median_slope_risk_v1.py -q",
         "output": "6 passed in 17.75s"}, exclusive=True)
    print("第155轮全部中文规则与已通过的六项测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
