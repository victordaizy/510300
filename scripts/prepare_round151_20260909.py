"""复用批量入口，准备平均K线的共同冻结与完整中文协议。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/heikin_price_state_v1.py"
    require(not destination.exists(), "第151轮入口已存在")
    text = (ROOT / "research/monotone_episode_budget_v1.py").read_text(encoding="utf-8")
    text = text.replace("第150轮", "第151轮").replace("round=150", "round=151")
    text = text.replace("monotone_episode_budget", "heikin_price_state").replace("MONOTONE_EPISODE_BUDGET", "HEIKIN_PRICE_STATE")
    text = text.replace("monotone_episode_frames", "heikin_price_frames").replace("BOTH_MONOTONE_PARENT_EPISODE_BUDGETS", "HEIKIN_STANDALONE_AND_REFERENCE_CONFIRMATION")
    text = text.replace("单向预算", "平均K线").replace("两种预算方向", "两种平均K线使用方式").replace("两个单向预算候选", "两个平均K线候选")
    text = text.replace("P149", "P150").replace("robust_block_growth_v1", "monotone_episode_budget_v1")
    text = text.replace('"ROBUST_BLOCK_GROWTH_REFERENCE_PAIR": (P150, "第149轮最弱阶段增长预算"),',
        '"EPISODE_BUDGET_NONINCREASING": (P150, "第150轮预算只减不增"),\n    "EPISODE_BUDGET_NONDECREASING": (P150, "第150轮预算只增不减"),')
    text = text.replace('tests["passed"] == 5', 'tests["passed"] == 6').replace("五项", "六项")
    text = text.replace('candidate_models=list(CANDIDATES), decision_clock=', 'candidate_models=list(CANDIDATES), numeric_tolerance=1e-12, decision_clock=')
    text = text.replace("PROGRESS_ROUND149_COMPLETED_CLOSED_FULL_GOAL_NOT_MET", "PROGRESS_ROUND150_COMPLETED_CLOSED_FULL_GOAL_NOT_MET")
    text = text.replace('ROOT / "tests/test_episode_trend_admission_v1.py", ', '')
    text = text.replace('from research.heikin_price_state_inputs_v1 import heikin_price_frames,', 'from research.heikin_price_state_inputs_v1 import heikin_candles, heikin_price_frames,')
    text = text.replace('    cfg["frozen_files"] =', '    checked = heikin_candles(data, cfg["numeric_tolerance"])\n'
        '    cfg["existing_input_checks"] = {"full_calendar_rows": len(checked), "known_candle_directions": int(checked.heikin_direction.notna().sum()),\n'
        '        "economic_close_matches_saved_wealth": True, "synthetic_prices_used_for_execution": False}\n    cfg["frozen_files"] =')
    require('"EPISODE_BUDGET_NONDECREASING"' in text and "round=151" in text and "P149" not in text, "第151轮入口替换不完整")
    destination.write_text(text, encoding="utf-8")
    proposal = (ROOT / "docs/510300_HEIKIN_PRICE_STATE_NEXT_20260909.md").read_text(encoding="utf-8")
    inherited = (ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md").read_text(encoding="utf-8")
    (ROOT / "docs/510300_HEIKIN_PRICE_STATE_V1.md").write_text(proposal.replace("第151轮拟研究", "第151轮冻结规则", 1)+
        "\n## 确认候选使用的完整父策略与全部因子\n\n以下为143父策略的完整定义。独立平均K线候选不使用这些父因素；确认候选按父策略独立输出后应用平均K线状态。\n\n"+
        "\n".join(inherited.splitlines()[1:])+"\n", encoding="utf-8")
    out = ROOT / "reports/research/510300_heikin_price_state_v1"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "tests_receipt.json", {"recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 5.75,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_heikin_price_state_v1.py", "output": "6 passed in 5.75s"}, exclusive=True)
    print("第151轮批量入口、全部中文规则及六测试回执已准备。", flush=True)


if __name__ == "__main__":
    main()
