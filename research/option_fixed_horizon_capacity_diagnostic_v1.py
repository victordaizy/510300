"""固定十日、不做持有中减仓的期权机会集容量诊断；明确使用未来信息。"""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "reports/research/510300_option_credit_spread_router_daily_v1"
PUT = ROOT / "reports/research/510300_protected_put_spread_daily_v1"
OUT = ROOT / "reports/research/510300_option_fixed_horizon_capacity_diagnostic_v1"


def dynamic_bound(n, events):
    outgoing = {}
    for number, event in enumerate(events):
        outgoing.setdefault(event["start_i"], []).append((number, event))
    value = np.full(n + 1, -np.inf)
    value[0] = 0.
    parent = [None] * (n + 1)
    for i in range(n):
        if value[i] > value[i + 1]:
            value[i + 1] = value[i]
            parent[i + 1] = (i, None)
        for number, event in outgoing.get(i, []):
            j = event["end_i"]
            candidate = value[i] + math.log1p(event["optimistic_account_return"])
            if candidate > value[j]:
                value[j] = candidate
                parent[j] = (i, number)
    path, j = [], n
    while j > 0:
        i, number = parent[j]
        if number is not None:
            path.append(events[number])
        j = i
    return math.exp(value[-1]), list(reversed(path))


def main():
    spec = importlib.util.spec_from_file_location("frozen_router_for_capacity", ROUTER / "code/option_credit_spread_router_daily_v1.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    h, _ = r.modules(ROUTER)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("容量诊断已有固定记录，禁止覆盖。")
    files = [PUT / "results/candidates.parquet", PUT / "results/payoff_labels.parquet",
             ROUTER / "results/call_candidates.parquet", ROUTER / "results/payoff_labels.parquet",
             ROUTER / "inputs/option_eod.parquet", ROUTER / "inputs/daily_terms.parquet"]
    protocol = {"at": h.now(), "scope": "固定已选两腿、开盘入场和十日后开盘退出，不做持有中减仓的机会集",
                "uses_future_information": True, "is_strategy": False, "compute_sharpe": False,
                "relaxations": ["事后完美选择方向、入场与不参与", "允许非整数张数", "每次最坏到期损失含固定手续费最多权益5%",
                                "放松五日ES、回撤余量、流动性数量和保证金限制", "同一资金不重叠占用，允许退出开盘立即开始下一笔"],
                "cost": "仍保留父研究压力每腿价格和每张每侧5元；最大损失分母忽略额外退出滑点储备以保持乐观",
                "censored": "末端未满十日或未知退出的候选按保护价差价值可降至零的最大净信用利润上界计，不当作实际盈利",
                "limits": "只约束不减仓的固定十日形式；不是所有动态减仓、其他持有期或结构的上限，也不证明可预测性",
                "files": {p.relative_to(ROOT).as_posix(): h.digest(p) for p in files}, "script_sha256": h.digest(Path(__file__))}
    h.save(OUT / "protocol.json", protocol, exclusive=True)
    multiple, _ = dynamic_bound(3, [{"start_i": 0, "end_i": 2, "optimistic_account_return": .1},
                                    {"start_i": 1, "end_i": 3, "optimistic_account_return": .2}])
    np.testing.assert_allclose(multiple, 1.2, atol=1e-12)
    panel, quotes, market, calendar, events_unused = r.sources(ROUTER, h)
    days = calendar[(calendar >= h.START) & (calendar <= h.END)]
    all_events, exclusions = [], []
    for structure, candidate_path, label_path in [
        ("PUT", PUT / "results/candidates.parquet", PUT / "results/payoff_labels.parquet"),
        ("CALL", ROUTER / "results/call_candidates.parquet", ROUTER / "results/payoff_labels.parquet"),
    ]:
        c = pd.read_parquet(candidate_path)
        labels = pd.read_parquet(label_path).set_index("idx")
        for candidate in c[c.selection_status.eq("READY") & c.date.between(h.START, h.END)].to_dict("records"):
            terms = h.terms_at(candidate, candidate["date"], quotes)
            if not h.valid_quote(terms, "open"):
                exclusions.append({"date": candidate["date"], "structure": structure, "reason": "NO_ACTUAL_ENTRY_QUOTE"})
                continue
            actual_credit = h.credit(terms, "open", "STRESS")
            width = h.width_cash(terms)
            maximum_loss = width - actual_credit + 4 * h.FEE
            if actual_credit >= width or maximum_loss <= 0:
                exclusions.append({"date": candidate["date"], "structure": structure, "reason": "INVALID_ENTRY_CREDIT"})
                continue
            label = labels.loc[candidate["idx"]]
            known = label.status10 == "READY"
            profit = float(label.pnl10) if known else actual_credit - 4 * h.FEE
            start_i = int(days.get_loc(candidate["date"]))
            end_i = int(days.searchsorted(candidate["exit10_date"]))
            assert end_i > start_i
            all_events.append({"date": candidate["date"], "exit10_date": candidate["exit10_date"], "structure": structure,
                               "start_i": start_i, "end_i": end_i, "actual_label_known": known,
                               "unit_profit_or_optimistic_bound": profit, "unit_terminal_maximum_loss": maximum_loss,
                               "optimistic_account_return": max(0., .05 * profit / maximum_loss)})
    summaries = []
    for name, sides in [("PUT_ONLY", {"PUT"}), ("CALL_ONLY", {"CALL"}), ("BOTH", {"PUT", "CALL"})]:
        eligible = [e for e in all_events if e["structure"] in sides]
        multiple, path = dynamic_bound(len(days), eligible)
        pd.DataFrame(path).to_parquet(OUT / f"future_selected_path_{name}.parquet", index=False)
        summaries.append({"opportunity_set": name, "equity_multiple_upper": multiple, "ending_equity_upper_cny": h.INITIAL * multiple,
                          "annualized_return_upper": multiple ** (252 / len(days)) - 1, "selected_completed_or_bounded_events": len(path),
                          "selected_unknown_events_using_maximum_payoff_bound": sum(not e["actual_label_known"] for e in path),
                          "sharpe": "NOT_COMPUTED_ORACLE_NOT_STRATEGY"})
    result = {"at": h.now(), "status": "FUTURE_INFORMATION_CAPACITY_DIAGNOSTIC_ONLY", "uses_future_information": True,
              "event_count": len(all_events), "known_payoff_events": sum(e["actual_label_known"] for e in all_events),
              "unknown_payoff_bound_events": sum(not e["actual_label_known"] for e in all_events), "exclusions": exclusions,
              "comparison": summaries, "goal_achieved": False, "new_strategy_accounts": 0, "dynamic_all_strategy_ceiling_claimed": False}
    h.save(OUT / "result.json", result)
    text = ["本诊断明确使用未来信息，没有生成可交易策略，也没有计算策略夏普。其范围是已经固定的两腿合约、十日后退出、持有期间不减仓的形式。",
            "", "允许事后完美选择方向、入场和空仓，非整数张数；忽略ES、回撤余量、数量和保证金限制，只保留单次理论最大损失含固定费用不超过权益5%。保留压力交易价格和固定费用，以动态规划保证同一资金不重叠占用。允许退出开盘立即再入场，亦比真实09:00决策更乐观。", ""]
    for row in summaries:
        text.append(f"{row['opportunity_set']}的乐观年化容量为{row['annualized_return_upper']:.2%}，20万元对应末端权益上界{row['ending_equity_upper_cny']:,.2f}元；路径选择{row['selected_completed_or_bounded_events']}个事件，其中{row['selected_unknown_events_using_maximum_payoff_bound']}个末端未成熟事件按最大可得净权利金估计。")
        text.append("")
    text.append("该结果只能回答这个固定形式的收益容量。它不能证明事前知道哪一侧更好，也不是包含任意提前减仓、换期限或换合约结构的总体上限。若容量高于目标，说明这个宽松上限未排除目标；不能据此把失败的预测模型晋升。")
    (OUT / "容量诊断结论.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    for rel, value in protocol["files"].items():
        assert h.digest(ROOT / rel) == value
    print(json.dumps(h.clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
