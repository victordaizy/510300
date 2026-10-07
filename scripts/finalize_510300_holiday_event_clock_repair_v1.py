"""固定同一可用时点内的来源排序，保留原失败记录，不改变任何研究规则。"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
REPAIR = ROOT / "clock_repair_v1"


def deterministic_join(left, right, time_column):
    left, right = left.copy(), right.copy()
    left["decision_time"] = pd.to_datetime(left.decision_time).astype("datetime64[ns]")
    right[time_column] = pd.to_datetime(right[time_column]).astype("datetime64[ns]")
    source_columns = [c for c in right.columns if c.endswith("source_date")]
    right = right.sort_values([time_column, *source_columns], kind="stable").drop_duplicates(time_column, keep="last")
    result = pd.merge_asof(left.sort_values("decision_time"), right, left_on="decision_time", right_on=time_column, direction="backward")
    known = result[time_column].notna()
    assert (result.loc[known, time_column] <= result.loc[known, "decision_time"]).all()
    return result


def module():
    spec = importlib.util.spec_from_file_location("frozen_holiday_risk_v1", ROOT / "code/holiday_event_capital_risk_v1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = module()
    p = m.parent(ROOT)
    m.check_freeze(ROOT, p)
    if (REPAIR / "STARTED.json").exists():
        raise RuntimeError("修复记录已经存在，禁止覆盖。")
    REPAIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), REPAIR / Path(__file__).name)
    p.save_json(REPAIR / "STARTED.json", {"started_at": p.now(), "repair": "同可用时点按原观察日期稳定排序，保留最新来源。",
                "original_freeze_sha256": p.digest(ROOT / "freeze.json"), "script_sha256": p.digest(Path(__file__)),
                "threshold_or_direction_changes": False, "initial_results_retained": "../results",
                "initial_failure": "截断历史后同一DR007可用时点的并列记录取值不确定；原运行在验证阶段停止。"})
    m.join_clock = deterministic_join
    d, signals, dividends = m.load(ROOT)
    views, closures = m.build_views(ROOT, d)
    checks = []
    for cutoff in ("2022-06-30", "2024-09-30", "2025-12-31"):
        partial, _ = m.build_views(ROOT, d.loc[d.date.le(cutoff)].copy())
        pd.testing.assert_frame_equal(views.iloc[:len(partial)].reset_index(drop=True), partial.reset_index(drop=True))
        checks.append(f"{cutoff}全部风险字段的截断不变性")
    old_views = pd.read_parquet(ROOT / "results/preopen_views.parquet")
    changes = {k: int(views[k].ne(old_views[k]).sum()) for k in ["common_known", *m.CANDIDATES]}
    need_accounts = any(changes.values())
    out = REPAIR / "results"
    out.mkdir(exist_ok=True)
    views.to_parquet(out / "preopen_views.parquet", index=False)
    closures.to_parquet(out / "holiday_episodes.parquet", index=False)
    fixed_decisions = pd.DataFrame([{"idx": i, "family": fam, "PATTERN_ONLY": True} for i in range(len(d)) for fam in p.FAMILIES])
    accounts, metrics = {}, []
    for period, (lo, hi) in m.PERIODS.items():
        start, end = int(d.index[d.date.ge(lo)][0]), int(d.index[d.date.le(hi)][-1])
        for cost in p.COSTS:
            for policy in m.POLICIES:
                source = ROOT / "results/accounts" / period / cost / policy
                if need_accounts:
                    if policy in ("BUY_HOLD", "CASH"):
                        result = p.account(d, dividends, signals, fixed_decisions, start, end, policy, cost)
                    else:
                        result = m.simulate(p, d, dividends, signals, views, start, end, policy, cost)
                    folder = out / "accounts" / period / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    for name, frame in zip(("ledger", "trades", "decisions"), result[:3]):
                        frame.to_parquet(folder / f"{name}.parquet", index=False)
                    p.save_json(folder / "terminal.json", result[3])
                else:
                    result = (*[pd.read_parquet(source / f"{k}.parquet") for k in ("ledger", "trades", "decisions")], json.loads((source / "terminal.json").read_text(encoding="utf-8")))
                accounts[(period, cost, policy)] = result
                ledger, trades, logs, terminal = result
                metric = {"period": period, "cost": cost, "policy": policy, **p.metrics(ledger, trades, terminal)}
                if len(trades):
                    metric["largest_cycle_cny"] = float(trades.net_pnl.max())
                    metric["profit_without_largest_cycle_cny"] = metric["net_profit_cny"] - metric["largest_cycle_cny"]
                    assert (trades.exit_idx > trades.entry_idx).all()
                if not terminal["open_position"]:
                    assert np.isclose(trades.net_pnl.sum() if len(trades) else 0., metric["net_profit_cny"], atol=1e-6)
                assert np.allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7)
                if not need_accounts:
                    saved = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
                    for key in ("net_sharpe", "net_cagr", "net_profit_cny", "max_drawdown"):
                        assert metric[key] is None and saved[key] is None or np.isclose(metric[key], saved[key], atol=1e-10)
                else:
                    p.save_json(folder / "metrics.json", metric)
                metrics.append(metric)
                if policy == "A" and need_accounts:
                    old = p.account(d, dividends, signals, fixed_decisions, start, end, "PATTERN_ONLY", cost)
                    pd.testing.assert_frame_equal(old[0], ledger[old[0].columns])
            print(f"已复核{period}、{cost}账户。", flush=True)
    p.save_json(out / "account_metrics.json", metrics)
    paired = m.compare(REPAIR, p, accounts)
    m.mechanism_diagnostics(REPAIR, p, d, signals, views, accounts)
    source_diff = ~(views.dr_source_date.eq(old_views.dr_source_date) | (views.dr_source_date.isna() & old_views.dr_source_date.isna()))
    p.save_json(REPAIR / "repair_result.json", {"status": "PASS", "source_date_rows_changed": int(source_diff.sum()),
                "risk_or_coverage_rows_changed": changes, "account_replays": len(metrics) if need_accounts else 0,
                "accounts_path": "results/accounts" if need_accounts else "../results/accounts", "checks": checks,
                "source_semantics": "周末银行间观测可以同一证券开盘变为可用；按源日期保留最近者，不使用未来记录。"})
    common = views.loc[views.date.between(*m.PERIODS["PRIMARY"])]
    passes = [k for k in m.CANDIDATES if all(next(x for x in metrics if x["period"] == "PRIMARY" and x["cost"] == c and x["policy"] == k)["numerical_target_pass"] for c in p.COSTS)]
    supported = [x["policy"] for x in paired if x["supports_positive_increment"]]
    p.save_json(ROOT / "verification.json", {"status": "PASS_AFTER_SOURCE_TIE_ORDER_REPAIR", "checked_at": p.now(),
                "account_count": len(metrics), "account_rows": sum(len(x[0]) for x in accounts.values()),
                "checks": checks, "original_A_equals_parent": True, "repair_result": "clock_repair_v1/repair_result.json",
                "scientific_independent_validation": False})
    p.save_json(ROOT / "result.json", {"status": "FROZEN_HISTORICAL_CANDIDATE_ONLY" if passes and supported else "FROZEN_NO_RELIABLE_CAPITAL_RISK_INCREMENT",
                "completed_at": p.now(), "canonical_results": "clock_repair_v1/results", "primary_dual_cost_numeric_pass": passes,
                "positive_increment_supported": supported, "common_days": int(common.common_known.sum()), "primary_days": len(common),
                "risk_days": {k: int(common[k].sum()) for k in m.CANDIDATES}, "strategy_goal_achieved": False,
                "calendar_source_first_versions_authenticated": False, "strict_forward_observations": 0,
                "scheduled_major_news_anticipation": "NOT_TESTED_IN_THIS_FROZEN_ROUND", "new_collection": False,
                "review_package_created": False, "terminated_strategy_revived": False})
    print(json.dumps(p.clean({"时钟修复": "完成", "交易判断变化": changes, "账户是否需重算": need_accounts, "目标通过候选": passes}), ensure_ascii=False))


if __name__ == "__main__":
    main()
