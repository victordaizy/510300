"""冻结并运行2021—2026样本门槛放宽V2，不选择最好区间或再改门槛。"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sequential_patterns_2021_2026_v2"
sys.path.insert(0, str(OUT / "code"))
sys.path.insert(0, str(ROOT / "research"))
from sequential_patterns_regime_v1 import COSTS, account, digest, metrics, now, save_csv, save_json
from sequential_patterns_sample_relaxation_v2 import relax_saved_decisions


def main():
    if (OUT / "relaxed/comparison.csv").exists():
        raise RuntimeError("V2已有固定结果，禁止覆盖。")
    parent = ROOT / "reports/research/510300_sequential_patterns_regime_v1"
    shutil.copy2(parent / "results/training_events.parquet", OUT / "inputs/training_events.parquet")
    shutil.copy2(ROOT / "research/sequential_patterns_sample_relaxation_v2.py", OUT / "code/sequential_patterns_sample_relaxation_v2.py")
    shutil.copy2(Path(__file__), OUT / "code/run_sequential_patterns_relaxed_v2.py")
    protocol = {"study_id": "510300_SEQUENTIAL_PATTERNS_2021_2026_V2", "frozen_at": now(),
                "user_request": "如果继续放宽，21年到26年", "clarification": "扩大到2021—2026，并适度降低启停样本门槛",
                "only_parameter_changes": {"min_mature_events": {"old": 6, "new": 3}, "min_same_state_events": {"old": 4, "new": 2}},
                "start": "2021-01-04", "end": "2026-09-16", "initial_capital_cny": 200000,
                "target_net_cagr": .10, "target_net_sharpe": 1.2, "target_max_drawdown": .10,
                "annual_trade_minimum": None, "lookback": 504, "last_events": 12,
                "state_mapping_unchanged": True, "normal90_lower_requirement_unchanged": True,
                "mean_excess_requirement_unchanged": True, "costs_and_execution_unchanged": True,
                "new_shapes_or_models": 0, "new_market_collection": False,
                "evidence_class": "AUTHORIZED_FOLLOWUP_ON_ALREADY_OBSERVED_HISTORY_NOT_INDEPENDENT_VALIDATION",
                "variant_selection": "只运行这一档数量门槛变化，同时列近期启停与完整启停，不按收益挑选。",
                "forward_observations": 0, "orders_authorized": False}
    save_json(OUT / "protocol.json", protocol)
    (OUT / "protocol.md").write_text("""# 2021—2026样本门槛放宽V2

用户在V1结果后明确要求“扩大到2021—2026，并适度降低启停样本门槛”。原V1单独封存。V2唯一变化是最低近期成熟案例6→3、同当前状态案例4→2。价格形态、504日回看、最多12个案例、普通时点对照、压力成本收益下界>0、平均超额>0、状态映射和执行约束全部保持。

2021-01-04以20万元重新开始完整账户，至现有资料末日2026-09-16。启停资格可使用2021年以前已经成熟的数据，这是历史假设初始化，不能利用2021年之后的标签决定2021年初动作。2026年不是完整年份。此前2024—2026窗口与2020起历史已被看到，新的时间段不是未见验证。

分别列出旧规则与V2的近期启停、完整启停，再列不变的仅形态、固定市场状态、买入持有和现金基准。基础/压力成本独立账户共16条（12条原规则、4条V2），不把两个时期或各形态最佳月份拼接。V2采用原始已保存前序统计量，只重新判定两个样本数量条件，不拟合或改变收益置信要求。

主要验收仍是整体两档成本净年化10%、夏普1.2及回撤风险目标10%，年度次数不设下限。样本数变少只是探索性放宽，不建立高置信度。若V2仍不达标，封存本轮结果，不自动进一步降低成本、改变方向/窗口或移除状态条件。逐年、触发日准入差异、完整成交和成本全部保存。

当前资料只到9月16日，不能据此形成9月24日当前交易观点。前向观察为0，行情采集暂停保持。
""", encoding="utf-8")
    frozen_files = list((OUT / "inputs").glob("*")) + list((OUT / "code").glob("*.py")) + [OUT / "protocol.json", OUT / "protocol.md", OUT / "baseline_freeze.json"]
    save_json(OUT / "relaxation_freeze.json", {"frozen_at": now(), "before_first_v2_decision_and_account": True,
               "hashes": {p.relative_to(OUT).as_posix(): digest(p) for p in frozen_files}})
    base = pd.read_parquet(OUT / "inputs/v1_decisions.parquet")
    train = pd.read_parquet(OUT / "inputs/training_events.parquet")
    dec = relax_saved_decisions(base, train)
    (OUT / "relaxed").mkdir(exist_ok=True)
    dec.to_parquet(OUT / "relaxed/daily_decisions.parquet", index=False)
    save_csv(OUT / "relaxed/daily_decisions.csv", dec)
    diff = dec[(dec.FULL != base.FULL) | (dec.RECENT_ONLY != base.RECENT_ONLY)].copy()
    diff["old_FULL"] = base.loc[diff.index, "FULL"]
    diff["old_RECENT_ONLY"] = base.loc[diff.index, "RECENT_ONLY"]
    save_csv(OUT / "relaxed/eligibility_changes.csv", diff)
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    signals = pd.read_parquet(OUT / "inputs/signals.parquet")
    start, end = int(d.index[d.date >= "2021-01-01"][0]), len(d) - 1
    joined = signals.merge(dec, left_on=["signal_idx", "family"], right_on=["idx", "family"], suffixes=("", "_decision"))
    joined = joined[joined.signal_idx >= start]
    save_csv(OUT / "relaxed/trigger_decisions.csv", joined)
    rows = []
    for cost in COSTS:
        for policy in ("RECENT_ONLY", "FULL"):
            ledger, trades, rejected, terminal = account(d, div, signals, dec, start, end, policy, cost)
            folder = OUT / "relaxed/accounts" / cost / policy
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "daily.parquet", index=False)
            save_csv(folder / "daily.csv", ledger)
            save_csv(folder / "trades.csv", trades if len(trades) else pd.DataFrame(columns=["signal_id", "entry_idx", "exit_idx", "entry_date", "exit_date", "net_pnl", "holding_sessions"]))
            save_csv(folder / "rejected.csv", rejected if len(rejected) else pd.DataFrame(columns=["date", "signal_id", "reason"]))
            m = metrics(ledger, trades, terminal)
            save_json(folder / "metrics.json", m)
            rows.append(dict(variant="V2_SAMPLE_RELAXED", cost=cost, policy=policy, **m))
    result = pd.DataFrame(rows)
    save_csv(OUT / "relaxed/comparison.csv", result)
    all_results = pd.concat([pd.read_csv(OUT / "baseline/comparison.csv"), result], ignore_index=True)
    save_csv(OUT / "comparison.csv", all_results)
    summary = {"completed_at": now(), "new_accounts": 16, "new_fits": 0, "new_market_downloads": 0,
               "new_strategy_variants": 1, "relaxed_parameters": 2,
               "period": [d.date.iloc[start], d.date.iloc[end]], "sessions": end - start + 1,
               "v2_results": result.to_dict("records"),
               "extra_eligible_family_days_all_history": len(diff),
               "extra_full_eligible_family_days_2021_onwards": int(((dec.idx >= start) & dec.FULL & ~base.FULL).sum()),
               "confirmed_events_2021_onwards": len(joined),
               "v2_full_qualified_signals": int(joined.FULL.sum()),
               "numerical_target_pass": bool(result[result.policy == "FULL"].numerical_target_pass.all()),
               "validated_goal_achieved": False, "current_view": "NO_VIEW", "forward_observations": 0,
               "status": "FROZEN_V2_FAILURE" if not result[result.policy == "FULL"].numerical_target_pass.all() else "NUMERICAL_PASS_AWAIT_INDEPENDENT_VALIDATION"}
    save_json(OUT / "summary.json", summary)
    print(result[["cost", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles"]].to_string(index=False))
    print(json.dumps({"新增完整机制具备资格的形态日": summary["extra_full_eligible_family_days_2021_onwards"], "获准确认信号": summary["v2_full_qualified_signals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
