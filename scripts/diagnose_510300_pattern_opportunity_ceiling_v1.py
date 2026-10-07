"""固定五日形态机会的事后收益上限；使用未来信息，绝不是可交易模型。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
OUT = WORKSPACE / "reports/research/510300_pattern_opportunity_ceiling_diagnostic_v1"


def solve(d, intervals, start, end, excluded):
    wealth = np.full(len(d), -np.inf)
    choices = [[] for _ in range(len(d))]
    wealth[start] = 200000.
    by_start = {}
    for row in intervals:
        if row["signal_id"] not in excluded:
            by_start.setdefault(row["signal_idx"], []).append(row)
    for i in range(start, end + 1):
        if i > start and wealth[i - 1] > wealth[i]:
            wealth[i], choices[i] = wealth[i - 1], choices[i - 1].copy()
        for row in by_start.get(i, []):
            k = row["exit_idx"]
            candidate = wealth[i] * row["continuous_net_multiplier"]
            if candidate > wealth[k]:
                wealth[k] = candidate
                choices[k] = choices[i] + [row["signal_id"]]
    return {"initial_cny": 200000, "optimistic_ending_wealth_cny": float(wealth[end]),
            "optimistic_net_cagr_ceiling": float((wealth[end] / 200000) ** (252 / (end - start + 1)) - 1),
            "chosen_signals_with_future_knowledge": choices[end], "chosen_count": len(choices[end])}


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("诊断已有结果，禁止覆盖。")
    spec = importlib.util.spec_from_file_location("ceiling_parent", SOURCE / "code/parent_engine.py")
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(SOURCE / "inputs/features.parquet")
    s = pd.read_parquet(SOURCE / "inputs/signals.parquet")
    start = int(d.index[d.date.ge("2021-01-04")][0])
    end = int(d.index[d.date.le("2026-08-14")][-1])
    s = s.loc[s.signal_idx.between(start, end - 1)].copy()
    original = pd.read_parquet(SOURCE / "results/accounts/PRIMARY/STRESS/A/trades.parquet")
    biggest = str(original.loc[original.net_pnl.idxmax(), "signal_id"])
    p.save_json(OUT / "diagnostic_definition.json", {"created_at": p.now(), "source": SOURCE.relative_to(WORKSPACE).as_posix(),
                "question": "在固定形态、固定失效和五日退出下，只靠选择入场能否支撑年化10%。",
                "future_information_used": True, "tradable_strategy": False, "new_model_fits": 0, "new_tradable_accounts": 0,
                "bound": "允许事后知道每笔输赢、自由剔除亏损，连续份额、无最低佣金、全部应得分红立即可再投资；同成本滑点和比例佣金。",
                "capital_constraint": "若选中旧交易，必须在其退出当日收盘或以后才接受下一形态确认；同日信号允许事后选择任意一个，放松原优先序。",
                "preserved": "形态定义、原失效规则、五日退出、T+1和原不可成交状态；不延长持有寻找最高价。",
                "sensitivity": "排除原压力母账户中贡献最大的一个完整周期，重新求其他全部机会的乐观上限。仅说明集中程度，不生成策略。",
                "biggest_original_signal": biggest, "script_sha256": p.digest(Path(__file__))})
    results, all_rows, unfilled = [], [], []
    for cost in p.COSTS:
        rows, incomplete = [], []
        for signal in s.to_dict("records"):
            r = p.trade_outcome(d, signal, cost)
            if r["label_status"] == "UNFILLED_ENTRY":
                unfilled.append({"cost": cost, "signal_id": signal["signal_id"]})
                continue
            if r["label_status"] != "MATURE" or int(r["exit_idx"]) > end:
                incomplete.append({"signal_id": signal["signal_id"], "status": r["label_status"]})
                continue
            c = p.COSTS[cost][0]
            multiplier = (r["exit_price"] * (1 - c) + r["dividend_per_share"]) / (r["entry_price"] * (1 + c))
            row = {"cost": cost, "signal_id": signal["signal_id"], "signal_idx": int(signal["signal_idx"]),
                   "entry_idx": int(r["entry_idx"]), "exit_idx": int(r["exit_idx"]), "entry_date": r["entry_date"],
                   "exit_date": r["exit_date"], "continuous_net_multiplier": multiplier}
            rows.append(row)
        # 本主区间全部信号均已成熟，才能宣称是完整固定机会集上限。
        assert not incomplete, incomplete
        all_rows.extend(rows)
        results.append({"cost": cost, "case": "ALL_FIXED_OPPORTUNITIES", **solve(d, rows, start, end, set())})
        results.append({"cost": cost, "case": "WITHOUT_LARGEST_ORIGINAL_CYCLE", **solve(d, rows, start, end, {biggest})})
    pd.DataFrame(all_rows).to_parquet(OUT / "fixed_opportunity_multipliers.parquet", index=False)
    days = end - start + 1
    p.save_json(OUT / "result.json", {"status": "DESCRIPTIVE_ORACLE_UPPER_BOUND_NOT_A_STRATEGY", "future_information_used": True,
                "period": [d.date.iloc[start], d.date.iloc[end]], "trading_days": days, "signal_count": len(s),
                "target_cagr": .1, "required_ending_wealth_cny": 200000 * 1.1 ** (days / 252),
                "excluded_in_sensitivity": biggest, "unfilled_original_entries": unfilled, "results": results, "goal_achieved": False})
    print(json.dumps(p.clean({"固定机会数": len(s), "目标所需期末权益": 200000 * 1.1 ** (days / 252),
          "上限": [{"成本": r["cost"], "情景": r["case"], "事后年化上限": r["optimistic_net_cagr_ceiling"], "笔数": r["chosen_count"]} for r in results]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
