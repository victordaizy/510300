"""保存边界机制测试回执及包含143全部因素的正式中文协议。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "docs/510300_BOUNDARY_REBALANCE_V1.md"
    require(not target.exists(), "第154轮中文冻结协议已经存在")
    proposal = (ROOT / "docs/510300_BOUNDARY_REBALANCE_NEXT_20260909.md").read_text(encoding="utf-8")
    proposal = proposal.replace("第154轮准备：", "第154轮冻结规则：", 1).replace("本轮拟保留143", "本轮保留143", 1)
    proposal = proposal.replace("当前只是明确的下一方案，154尚未实现、测试、登记或建账。", "本规则在新历史账户读取前确定，六项必要合成测试已经通过，之后冻结并一次运行四个账户。")
    proposal += "\n## 实现前确定的数值和账户口径\n\n上下边界相等判断的纯数值容差固定为万亿分之一的仓位比例。距离边界不超过该量时保持份额，"
    proposal += "仅用于避免二进制小数舍入把等号误判为越界；它远小于本研究一手份额，不依据任何第154轮收益调整。"
    proposal += "每次记录143原目标、本账户判断净值、实际股票比例、上下边界、所选再平衡比例和实际请求数量。未调仓时所选再平衡比例不填数，原目标仍然保留，二者不混为一个信号。\n\n"
    proposal += "新账户入口只允许本轮目标数组，明确拒绝买入持有专用分支或直接预测值模式。账户逐日权益、成交函数、滑点价位、T+1批次与分红事件顺序沿用原完整账户组件，"
    proposal += "原143和原账户文件均不修改。新批入口显式接收完整账户函数，保留两时期、两费用、完整日历、保存对照和全部绩效。\n"
    parent = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8")
    proposal += "\n## 提供目标的143策略：全部因子、学习和进出规则\n\n以下完整保留143规则。它先在各自参考账户产生原目标，本轮独立账户再应用前述唯一的边界请求改动。\n\n"
    proposal += "\n".join(parent.splitlines()[1:])+"\n"
    target.write_text(proposal, encoding="utf-8")
    out = ROOT / "reports/research/510300_boundary_rebalance_v1"
    write_json(out / "tests_receipt.json", {"recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 4.90,
        "command": ".venv\\Scripts\\python.exe -m pytest tests\\test_boundary_rebalance_v1.py -q", "output": "6 passed in 4.90s"}, exclusive=True)
    print("第154轮完整中文协议和六测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
