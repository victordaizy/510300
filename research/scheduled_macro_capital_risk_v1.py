"""提前公布的国民经济发布日程：事件发生前资金避险的独立固定实验。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_scheduled_macro_capital_risk_v1"
PRIOR = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
CANDIDATES = ("S_SCHEDULE", "S_FUNDS")
PERIODS = {"PRIMARY": ("2021-01-04", "2026-08-14"), "RECENT_DIAGNOSTIC": ("2024-08-20", "2026-08-14")}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def engines(root):
    m = load_module("scheduled_macro_frozen_account", root / "code/engine.py")
    m.CANDIDATES = CANDIDATES
    p = load_module("scheduled_macro_frozen_parent", root / "code/parent_engine.py")
    return m, p


def build_calendar_views(root):
    v = pd.read_parquet(root / "inputs/capital_views.parquet")
    events = pd.read_csv(root / "inputs/schedule.csv")
    historical = pd.read_csv(root / "inputs/calendar_history.csv")
    yearly = pd.read_csv(root / "inputs/calendar_2026.csv")
    hd = pd.to_datetime(historical.trade_date)
    opens = pd.DatetimeIndex(sorted(set(hd[hd < "2026-01-01"]) | set(pd.to_datetime(yearly.trade_date)))) + pd.Timedelta(hours=9, minutes=30)
    v["S_SCHEDULE"] = False
    v["S_FUNDS"] = False
    v["scheduled_event_id"] = ""
    # 本轮是原形态上的可选风险否决：缺乏日程或资金证据不改变原规则；不将无观点变为空仓。
    v["capital_known"] = v.funding_known & v.flow_known
    v["common_known"] = True
    records = []
    for event in events.to_dict("records"):
        scheduled = pd.Timestamp(event["scheduled_at"])
        available = pd.Timestamp(event["schedule_publication_date"]).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        k = int(opens.searchsorted(scheduled, side="right")) - 1
        status = "NO_PRIOR_OPEN" if k < 0 else "READY"
        risk_open = opens[k] if k >= 0 else pd.NaT
        decision = risk_open.normalize() + pd.Timedelta(hours=9) if k >= 0 else pd.NaT
        if k >= 0 and available > decision:
            status = "NO_VIEW_SCHEDULE_NOT_PUBLIC_AT_DECISION"
        date = risk_open.strftime("%Y-%m-%d") if k >= 0 else ""
        row = {"event_id": event["event_id"], "scheduled_at": scheduled, "schedule_available_upper": available,
               "risk_open": risk_open, "decision_time": decision, "status": status,
               "snapshot_state": event["schedule_snapshot_state"], "source_url": event["schedule_source_url"],
               "source_sha256": event["schedule_document_sha256"]}
        hit = v.date.eq(date)
        if status == "READY" and hit.any():
            v.loc[hit, "S_SCHEDULE"] = True
            v.loc[hit, "scheduled_event_id"] = str(event["event_id"])
            pressure = v.loc[hit, "capital_known"] & v.loc[hit, "funding_tightening"] & v.loc[hit, "micro_withdrawal"]
            v.loc[hit, "S_FUNDS"] = pressure
            row["capital_known"] = bool(v.loc[hit, "capital_known"].iloc[0])
            row["funds_risk"] = bool(pressure.iloc[0])
        records.append(row)
    return v, pd.DataFrame(records)


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("已经冻结，不能覆盖。")
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    for name in ("features.parquet", "signals.parquet", "dividends.csv", "calendar_history.csv", "calendar_2026.csv"):
        shutil.copy2(PRIOR / "inputs" / name, root / "inputs" / name)
    shutil.copy2(PRIOR / "clock_repair_v1/results/preopen_views.parquet", root / "inputs/capital_views.parquet")
    shutil.copy2(WORKSPACE / "data/curated/510300_nbs_1000_negative_information_drift_v2/event_ledger_pre_return.csv", root / "inputs/schedule.csv")
    shutil.copy2(PRIOR / "code/holiday_event_capital_risk_v1.py", root / "code/engine.py")
    shutil.copy2(PRIOR / "code/parent_engine.py", root / "code/parent_engine.py")
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    m, p = engines(root)
    v, event_records = build_calendar_views(root)
    # 原发布日程内容必须与既有事件账本一致，但不把当前抓取当成当时收到。
    sources = []
    for source in (WORKSPACE / "data/raw/official/nbs/national_economy_release_schedule_v1").glob("retrieved_at=*/year=202[1-6].html"):
        year = int(source.stem.split("=")[1])
        ledger = pd.read_csv(root / "inputs/schedule.csv")
        expected = ledger.loc[ledger.schedule_year.eq(year), "schedule_document_sha256"].unique()
        assert p.digest(source) in expected
        sources.append({"year": year, "path": source.relative_to(WORKSPACE).as_posix(), "sha256": p.digest(source)})
    assert len(sources) == 6
    assert (event_records.loc[event_records.status.eq("READY"), "schedule_available_upper"] <= event_records.loc[event_records.status.eq("READY"), "decision_time"]).all()
    protocol = {
        "study_id": "510300_SCHEDULED_MACRO_CAPITAL_RISK_V1", "frozen_at": p.now(), "periods": PERIODS,
        "question": "已知重大宏观数据日程是否提供事前避险价值；不继承旧NBS公布后五分钟漂移模型。",
        "parent": "固定三形态原账户A，原失效和五日退出。",
        "event_population": "既有国民经济运行情况完整110条计划记录；仅用日程信息，不按分钟行情质量、结果方向或旧模型入选条件筛选。",
        "known_at": "官方日程出版日期取23:59:59保守上界；年度修订只能在该版出版后使用。2022和2024早期不回填后来的修订日程。",
        "risk_open": "不晚于计划发布时间的最后一个证券开盘；在该开盘前09:00决定。计划10点公布时为同日09:30；休市日公布则为此前最后开盘。",
        "S_SCHEDULE": "所有已知计划窗口否决形态新买入，并卖出可卖旧份额。",
        "S_FUNDS": "只在同一计划窗口且已知宏观资金收紧、微观成交分类承压同时发生时否决或退出；资金定义、滞后和阈值沿用上一轮已固定定义。",
        "NO_VIEW": "对这层风险否决不形成新增请求，原形态策略继续；不把未知事件或未知资金自动当成空仓。",
        "after_event": "不补买曾被否决的旧形态、不机械重建已经退出的旧交易；后续新形态照原规则。",
        "sampling": "事件发生前未公开的日程留为NO_VIEW；不拿actual发布日期替换未知计划。",
        "costs": p.COSTS, "capital_cny": 200000, "cash_yield": 0, "annual_days": 252,
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "candidates": CANDIDATES, "fits": 0, "parameter_grids": 0,
        "statistics": "主压力两个候选相对A，20日联合循环区块2000次，单侧.05/2；全局重复研究的选择偏差仍存在。",
        "independence": False, "current_view": "NO_VIEW", "original_NBS_drift_rejection_preserved": True,
        "new_collection": False, "orders": False, "review_package": False, "source_references": sources,
    }
    p.save_json(root / "protocol.json", protocol)
    paths = [root / "protocol.json", *[x for x in (root / "inputs").iterdir() if x.is_file()], *[x for x in (root / "code").iterdir() if x.is_file()]]
    p.save_json(root / "freeze.json", {"frozen_at": p.now(), "files": [{"path": x.relative_to(root).as_posix(), "sha256": p.digest(x)} for x in paths]})
    print("已冻结两个计划公布前的资金避险候选，未读取其账户结果。")


def run(root):
    m, p = engines(root)
    record = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert p.digest(root / item["path"]) == item["sha256"]
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("已有运行记录，不能重跑覆盖。")
    p.save_json(root / "RUN_STARTED.json", {"started_at": p.now()})
    out = root / "results"
    out.mkdir(exist_ok=True)
    d, signals, dividends = m.load(root)
    views, records = build_calendar_views(root)
    views.to_parquet(out / "preopen_views.parquet", index=False)
    records.to_parquet(out / "scheduled_events.parquet", index=False)
    accounts, metrics = {}, []
    for period, (lo, hi) in PERIODS.items():
        start, end = int(d.index[d.date.ge(lo)][0]), int(d.index[d.date.le(hi)][-1])
        for cost in p.COSTS:
            for policy in ("A", *CANDIDATES):
                result = m.simulate(p, d, dividends, signals, views, start, end, policy, cost)
                ledger, trades, decisions, terminal = result
                accounts[(period, cost, policy)] = result
                folder = out / "accounts" / period / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                for name, frame in zip(("ledger", "trades", "decisions"), result[:3]):
                    frame.to_parquet(folder / f"{name}.parquet", index=False)
                metric = {"period": period, "cost": cost, "policy": policy, **p.metrics(ledger, trades, terminal)}
                metric["risk_entry_rejections"] = int(decisions.reason.eq("RISK_ENTRY_REJECTED").sum())
                metric["risk_exit_requests"] = int(decisions.reason.eq("RISK_REDUCTION").sum())
                if len(trades):
                    metric["largest_cycle_cny"] = float(trades.net_pnl.max())
                    metric["profit_without_largest_cycle_cny"] = metric["net_profit_cny"] - metric["largest_cycle_cny"]
                    assert (trades.exit_idx > trades.entry_idx).all()
                assert np.allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7)
                if not terminal["open_position"]:
                    assert np.isclose(trades.net_pnl.sum() if len(trades) else 0., metric["net_profit_cny"], atol=1e-6)
                if policy == "A":
                    original = pd.read_parquet(PRIOR / f"results/accounts/{period}/{cost}/A/ledger.parquet")
                    pd.testing.assert_frame_equal(original.drop(columns=["common_known"]), ledger.drop(columns=["common_known"]))
                p.save_json(folder / "metrics.json", metric)
                p.save_json(folder / "terminal.json", terminal)
                metrics.append(metric)
            print(f"已完成{period}、{cost}的计划公布前避险。", flush=True)
    p.save_json(out / "account_metrics.json", metrics)
    base_result = accounts[("PRIMARY", "STRESS", "A")]
    base = p.reserved_returns(base_result[0], base_result[3])
    n = len(base)
    rng = np.random.default_rng(202609242)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.save(out / "bootstrap_indices.npy", indices)
    paired = []
    for policy in CANDIDATES:
        account = accounts[("PRIMARY", "STRESS", policy)]
        delta = p.reserved_returns(account[0], account[3]) - base
        boot = delta[indices].mean(axis=1) * 252
        lower = float(np.quantile(boot, .025))
        paired.append({"policy": policy, "baseline": "A", "annual_mean_increment": float(delta.mean() * 252),
                       "ci95": np.quantile(boot, [.025, .975]).tolist(), "familywise_one_sided_lower": lower, "supports_positive_increment": lower > 0})
    p.save_json(out / "paired_increment.json", paired)
    passes = [k for k in CANDIDATES if all(next(x for x in metrics if x["period"] == "PRIMARY" and x["cost"] == c and x["policy"] == k)["numerical_target_pass"] for c in p.COSTS)]
    relevant = records.risk_open.between(pd.Timestamp(PERIODS["PRIMARY"][0]), pd.Timestamp(PERIODS["PRIMARY"][1]) + pd.Timedelta(days=1))
    p.save_json(root / "result.json", {"status": "FROZEN_HISTORICAL_CANDIDATE_ONLY" if passes else "FROZEN_NO_HIGH_SHARPE_SCHEDULED_EVENT_RULE",
                "completed_at": p.now(), "primary_dual_cost_numeric_pass": passes,
                "positive_increment_supported": [x["policy"] for x in paired if x["supports_positive_increment"]],
                "events_in_primary": int(relevant.sum()), "known_before_execution": int((relevant & records.status.eq("READY")).sum()),
                "unavailable_schedule_events": int((relevant & records.status.ne("READY")).sum()),
                "pressure_event_days": int(views.loc[views.date.between(*PERIODS["PRIMARY"]), "S_FUNDS"].sum()),
                "strategy_goal_achieved": False, "strict_forward_observations": 0, "old_NBS_rejection_preserved": True})
    p.save_json(root / "verification.json", {"status": "PASS", "accounts": len(metrics), "account_rows": sum(len(a[0]) for a in accounts.values()),
                "A_same_as_previous_frozen_engine": True, "sources_not_later_than_decision": True,
                "schedule_selection_excludes_no_events_based_on_outcomes_or_minute_data": True,
                "independent_validation": False})
    print("计划公布前避险已完成；未把数据公布后的结果回填为事前预期。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.root.resolve())


if __name__ == "__main__":
    main()
