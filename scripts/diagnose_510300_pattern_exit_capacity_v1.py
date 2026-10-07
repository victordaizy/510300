"""固定形态内允许提前退出后的容量；完全事后上限，不是交易信号。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
OLD = WORKSPACE / "reports/research/510300_pattern_opportunity_ceiling_diagnostic_v1"
OUT = WORKSPACE / "reports/research/510300_pattern_exit_capacity_diagnostic_v1"


def optimize(opportunities, start, end, entry_selection, early_exit, excluded=()):
    """状态是当天收盘空仓；退出当日开盘释放本金，当天收盘才可接受新确认。"""
    value = np.ones(end + 1)
    paths = [[] for _ in range(end + 1)]
    excluded = set(excluded)
    groups = {}
    for op in opportunities:
        if op["signal_id"] not in excluded:
            groups.setdefault(op["signal_idx"], []).append(op)
    for i in range(end - 1, start - 1, -1):
        group = groups.get(i, [])
        value[i], paths[i] = value[i + 1], paths[i + 1].copy()
        if not group:
            continue
        if not entry_selection:
            group = sorted(group, key=lambda r: r["priority"])[:1]
            # 原优先信号不能成交时，其他同日形态不会自动补位。
            if not group[0]["exits"]:
                continue
            value[i] = -np.inf
        for op in group:
            exits = op["exits"] if early_exit else op["exits"][-1:]
            for action in exits:
                j = action["exit_idx"]
                score = action["multiplier"] * value[j]
                if score > value[i]:
                    value[i] = score
                    paths[i] = [{"signal_id": op["signal_id"], **action}] + paths[j]
    return {"wealth_multiplier": float(value[start]), "ending_wealth_cny": 200000 * float(value[start]),
            "cagr_ceiling": float(value[start] ** (252 / (end - start + 1)) - 1), "chosen_cycles": paths[start]}


def mechanism_checks():
    fixture = [{"signal_id": "先出现的机会", "signal_idx": 0, "priority": 0,
                "exits": [{"exit_idx": 2, "multiplier": 1.1}, {"exit_idx": 3, "multiplier": 1.15}]},
               {"signal_id": "后出现的机会", "signal_idx": 2, "priority": 0,
                "exits": [{"exit_idx": 4, "multiplier": 1.3}]}]
    early = optimize(fixture, 0, 4, False, True)
    natural = optimize(fixture, 0, 4, False, False)
    assert np.isclose(early["wealth_multiplier"], 1.43)
    assert np.isclose(natural["wealth_multiplier"], 1.15)
    bad = [{"signal_id": "必须接受的亏损机会", "signal_idx": 0, "priority": 0,
            "exits": [{"exit_idx": 2, "multiplier": .9}]}]
    assert np.isclose(optimize(bad, 0, 3, False, True)["wealth_multiplier"], .9)
    assert np.isclose(optimize(bad, 0, 3, True, True)["wealth_multiplier"], 1.)
    return ["提前卖出可释放下一确认的本金", "保留原退出时不能穿越重叠交易", "强制接受入场与允许筛选不同"]


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("容量诊断已完成，禁止覆盖。")
    spec = importlib.util.spec_from_file_location("exit_capacity_parent", SOURCE / "code/parent_engine.py")
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    d = pd.read_parquet(SOURCE / "inputs/features.parquet")
    signals = pd.read_parquet(SOURCE / "inputs/signals.parquet")
    start, end = int(d.index[d.date.ge("2021-01-04")][0]), int(d.index[d.date.le("2026-08-14")][-1])
    signals = signals.loc[signals.signal_idx.between(start, end - 1)].copy()
    old = json.loads((OLD / "result.json").read_text(encoding="utf-8"))
    biggest = old["excluded_in_sensitivity"]
    OUT.mkdir(parents=True, exist_ok=True)
    checks = mechanism_checks()
    p.save_json(OUT / "definition.json", {"created_at": p.now(), "future_information_used": True, "tradable_strategy": False,
                "cases": ["保留全部原入场、允许事后最优提前退出", "允许同时事后筛选入场和提前退出", "剔除原最大盈利周期后的联合上限"],
                "exit_domain": "买入次日到原失效或五日退出之间，真实可卖开盘；不延长自然边界。",
                "entry_clock": "当天收盘确认，次开盘；退出当天早晨释放本金，当天收盘可接受新确认。",
                "cost": "与原上限相同，保留双边比例佣金及报价滑点取整；放松整手及股息可再投资日期。",
                "method": "按日期倒序求最大财富乘数，显式区分可选入场和必须接受原优先信号。",
                "new_prediction_fits": 0, "new_tradable_accounts": 0, "script_sha256": p.digest(Path(__file__))})
    rows, result = [], []
    priorities = {fam: i for i, fam in enumerate(p.FAMILIES)}
    for cost in p.COSTS:
        ops = []
        c = p.COSTS[cost][0]
        for signal in signals.to_dict("records"):
            label = p.trade_outcome(d, signal, cost)
            op = {"signal_id": signal["signal_id"], "signal_idx": int(signal["signal_idx"]), "priority": priorities[signal["family"]], "exits": []}
            if label["label_status"] == "UNFILLED_ENTRY":
                ops.append(op)
                continue
            assert label["label_status"] == "MATURE" and int(label["exit_idx"]) <= end
            entry = int(label["entry_idx"])
            for j in range(entry + 1, int(label["exit_idx"]) + 1):
                if not p.tradable(d, j, "SELL"):
                    continue
                dividend = float(d.dividend.iloc[entry + 1:j + 1].sum())
                px = p.fill_price(float(d.open.iloc[j]), "SELL", cost)
                mult = (px * (1 - c) + dividend) / (label["entry_price"] * (1 + c))
                action = {"exit_idx": j, "exit_date": d.date.iloc[j], "multiplier": mult,
                          "entry_idx": entry, "entry_date": d.date.iloc[entry]}
                op["exits"].append(action)
                rows.append({"cost": cost, "signal_id": op["signal_id"], **action})
            ops.append(op)
        reference = optimize(ops, start, end, True, False)
        prior_bound = next(r for r in old["results"] if r["cost"] == cost and r["case"] == "ALL_FIXED_OPPORTUNITIES")
        assert np.isclose(reference["ending_wealth_cny"], prior_bound["optimistic_ending_wealth_cny"], atol=1e-7)
        checks.append(f"{cost}反向递推与原前向入场上限完全一致")
        for name, select, excluded in [("ALL_ORIGINAL_ENTRIES_EARLY_EXIT_ORACLE", False, set()),
                                       ("ENTRY_AND_EXIT_ORACLE", True, set()),
                                       ("ENTRY_AND_EXIT_WITHOUT_LARGEST_CYCLE", True, {biggest})]:
            r = optimize(ops, start, end, select, True, excluded)
            result.append({"cost": cost, "case": name, **r})
    pd.DataFrame(rows).to_parquet(OUT / "eligible_exit_intervals.parquet", index=False)
    p.save_json(OUT / "result.json", {"status": "ORACLE_CAPACITY_ONLY_NOT_A_STRATEGY", "future_information_used": True,
                "goal_achieved": False, "period": [d.date.iloc[start], d.date.iloc[end]], "results": result,
                "checks": checks, "new_prediction_fits": 0, "new_tradable_accounts": 0})
    print(json.dumps(p.clean([{"成本": r["cost"], "情景": r["case"], "事后年化上限": r["cagr_ceiling"], "周期数": len(r["chosen_cycles"])} for r in result]), ensure_ascii=False))


if __name__ == "__main__":
    main()
