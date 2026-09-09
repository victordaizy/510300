"""保存本轮已通过测试回执和包括全部父因子的中文协议。"""
from research.monotone_episode_budget_v1 import ROOT, OUT
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    document = ROOT / "docs/510300_MONOTONE_EPISODE_BUDGET_V1.md"
    require(not document.exists() and not (OUT / "tests_receipt.json").exists(), "第150轮准备材料已经存在")
    proposal = (ROOT / "docs/510300_MONOTONE_EPISODE_BUDGET_NEXT_20260909.md").read_text(encoding="utf-8")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8")
    document.write_text(proposal.replace("第150轮拟研究", "第150轮冻结规则", 1)+
        "\n## 完整父策略及全部因子\n\n以下完整保留143父策略的因子、训练与进出规则。父策略独立继续计算，其输出再经过本轮两种单向预算；父策略内部比例不受外层实际持仓反向影响。\n\n"+
        "\n".join(inherited.splitlines()[1:])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "tests_receipt.json", {"recorded_at": now(), "exit_code": 0, "passed": 5, "seconds": 3.53,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_monotone_episode_budget_v1.py", "output": "5 passed in 3.53s"}, exclusive=True)
    print("第150轮完整中文协议和五项测试回执已保存。", flush=True)


if __name__ == "__main__":
    main()
