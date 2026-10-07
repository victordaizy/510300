"""用已有回购与配股资料做低成本可行性检验，结果只评价观察到的事件代理。"""
import argparse
from collections import defaultdict
from datetime import datetime
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research.factor96_rapid_feasibility_v1 import (
    ROOT, PRICE_BASE, ENGINE_PATH, START, END, read, core, simulate, period_metrics,
)
from scripts.record_factor96_remaining_changes_v1 import update

OUT = ROOT / "reports/research/510300_factor96_rapid_capital_proxy_v1"
PRIOR = ROOT / "reports/research/510300_factor96_rapid_feasibility_v1"
BATCHES = ROOT / "reports/research/510300_factor96_repurchase_9_plan_context_v1/publication_batches_with_prior_context.json"
FACTS = ROOT / "reports/research/510300_factor96_rights_mixed_titles_v1/combined_source_facts.json"
MEMBERS = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/membership.parquet"
COVERAGE = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/daily_coverage.parquet"
POLICIES = ["REPURCHASE_HIGH", "REPURCHASE_LOW", "SUPPLY_RELIEF", "SUPPLY_INCREASE"]


def save(path, value):
    core.save(path, value, exclusive=True)


def freeze():
    save(OUT / "protocol.json", {
        "at": core.now(), "study_id": "510300_FACTOR96_RAPID_CAPITAL_PROXY_V1",
        "classification": "EXPLORATORY_INCOMPLETE_COVERAGE_EVENT_PROXIES",
        "user_priority": "先简单检验可行性，再展开有成效的方向；无效则换方向。",
        "latest_asset_choice": "仍只交易510300，重新寻找不同的收益机制；现金保留。",
        "repurchase": "仅用已存946个披露批次中正的新增实施回购，按当时成分纳入；每发行人同披露日去重。过去20交易日观察到的正回购批次数。",
        "repurchase_direction": "高于或等于此前两日历年70%分位、低于或等于30%分位分别比较；至少120个有效观测且两分位不同；均要求510300当日总收益为正。",
        "supply": "只用已保存A股上市公告原始字段、按目录日末可得代理，在当时CSI300成分中数未来20交易日内已公告的配股上市事件。不是可卖股数。",
        "supply_direction": "该已知事件数比5日前下降且510300当日总收益为正为RELIEF，增加且当日上涨为INCREASE对照；没有全市场供给量含义。",
        "clock": "每条披露在首个收盘不早于其目录日末的交易日进入观察；公告过去发生的回购只视为当前新披露，不当成未来买单。",
        "scope_boundary": "股份、自由流通分母、完整供给和用途约束仍未知；零表示所收集集合里没观察到，不证明市场没事件。不得升级为严格M01/M06。",
        "purpose": "判断当前代理有无继续投入理由，不将简单代理通过与正式候选通过混为一谈。",
        "account_rule": "原快速初筛的10间隔持有、50%仓位上限、成熟ES、回撤储备、T+1、整手、分红和两级成本完全复用。",
        "capital_CNY": [200000, 20000], "policies": POLICIES, "account_scenarios": 16,
        "evaluation_start": START, "evaluation_end": END, "annual_days": 242,
        "followup_filter": "20万元STRESS夏普>=0.8、CAGR>=5%、回撤<=10%、至少20次入场且两子期收益为正。",
        "network_requests": 0, "new_source_detail_work": False, "orders_authorized": False,
        "delivery_package_required": False, "goal_achieved": False})
    sources = [OUT / "protocol.json", Path(__file__), Path(core.__file__),
               ROOT / "research/factor96_rapid_feasibility_v1.py", ENGINE_PATH,
               PRICE_BASE / "market.parquet", PRICE_BASE / "price_features.parquet", PRICE_BASE / "dividends.csv",
               BATCHES, FACTS, MEMBERS, COVERAGE]
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in sources]})
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    mandate.update(latest_research_direction_instruction="无效方向结束，重新讨论收益机制；用户确认仍只交易510300。",
                   research_priority="RAPID_FEASIBILITY_FIRST", source_detail_work_deferred_by_user=True)
    update(mandate_path, mandate)
    print("已固定回购、供给两个简化机制及各一项反向对照，共16个账户，无新增采集。", flush=True)


def build(m):
    dates = pd.DatetimeIndex(m.date)
    member_frame = pd.read_parquet(MEMBERS)
    member_sets = {pd.Timestamp(day): set(group.symbol) for day, group in member_frame.groupby("membership_date")}
    coverage = pd.read_parquet(COVERAGE).set_index("date")
    allowed = coverage.aggregation_state.reindex(dates).eq("VIEW_ALLOWED").to_numpy()
    occurrence = defaultdict(set); admitted = []; outside = 0
    for row in read(BATCHES):
        if not row["positive_disclosed_increment"]:
            continue
        day = pd.Timestamp(row["known_at"]).tz_localize(None).normalize()
        i = int(dates.searchsorted(day))
        if i >= len(m):
            continue
        if row["symbol"] in member_sets.get(dates[i], set()) and allowed[i]:
            occurrence[i].add(row["symbol"])
            admitted.append({"batch_id": row["batch_id"], "symbol": row["symbol"], "known_at": row["known_at"],
                             "observation_idx": i, "observation_date": dates[i]})
        else:
            outside += 1
    day_counts = np.array([len(occurrence[i]) for i in range(len(m))], dtype=float)
    repurchase = pd.Series(day_counts).rolling(20).sum().where(allowed)
    listing_facts = [r for r in read(FACTS) if r.get("event_id") and r.get("updates", {}).get("announced_listing_date")]
    assert len({r["event_id"] for r in listing_facts}) == 36
    by_day = defaultdict(list)
    for row in listing_facts:
        known = pd.Timestamp(row["known_at"]).tz_localize(None).normalize()
        i = int(dates.searchsorted(known))
        by_day[i].append(row)
    active = {}; supply = np.full(len(m), np.nan); used = []
    for i, day in enumerate(dates):
        for row in sorted(by_day.get(i, []), key=lambda r: (r["known_at"], r["document_id"])):
            active[row["event_id"]] = row
        if not allowed[i]:
            continue
        matched = []
        for event_id, row in active.items():
            listing = pd.Timestamp(row["updates"]["announced_listing_date"])
            listing_index = int(dates.searchsorted(listing))
            if i < listing_index <= i + 20 and row["symbol"] in member_sets.get(day, set()):
                matched.append(event_id)
                used.append({"observation_date": day, "event_id": event_id, "source_id": row["document_id"],
                             "source_known_at": row["known_at"], "announced_listing_date": str(listing.date())})
        supply[i] = len(matched)
    f = pd.DataFrame({"date": m.date, "observed_repurchase_batches": day_counts,
                      "observed_repurchase_count20": repurchase, "known_listing_count20": supply})
    up = ((m.close + m.dividend) / m.close.shift() - 1).gt(0).to_numpy()
    low, high = np.full(len(m), np.nan), np.full(len(m), np.nan)
    for i, day in enumerate(dates):
        a = dates.searchsorted(day - pd.DateOffset(years=2))
        values = repurchase.iloc[a:i].dropna().to_numpy()
        if len(values) >= 120:
            low[i], high[i] = np.quantile(values, [.3, .7])
    distinct = np.isfinite(low) & np.isfinite(high) & (high > low)
    f["REPURCHASE_HIGH"] = distinct & (repurchase >= high) & up
    f["REPURCHASE_LOW"] = distinct & (repurchase <= low) & up
    change = pd.Series(supply).diff(5)
    f["SUPPLY_RELIEF"] = change.lt(0) & up
    f["SUPPLY_INCREASE"] = change.gt(0) & up
    f["repurchase_q30"], f["repurchase_q70"], f["supply_change5"] = low, high, change
    for row in used:
        assert pd.Timestamp(row["source_known_at"]).date() <= row["observation_date"].date()
        assert row["observation_date"].date() < pd.Timestamp(row["announced_listing_date"]).date()
    save(OUT / "observed_repurchase_batches.json", admitted)
    save(OUT / "observed_listing_windows.json", used)
    save(OUT / "coverage_diagnostic.json", {"positive_batches_in_current_members": len(admitted),
        "positive_batches_excluded_not_current_member_or_coverage": outside,
        "unique_issuer_observation_day_events": sum(len(v) for v in occurrence.values()),
        "distinct_listing_events_with_future_window_in_current_members": len({r["event_id"] for r in used}),
        "yearly_observed_repurchase_issuer_days": f.assign(year=f.date.dt.year).groupby("year").observed_repurchase_batches.sum().to_dict(),
        "not_complete_market_coverage": True})
    return f


def run():
    for f in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(f["path"])) == f["sha256"]
    save(OUT / "run_started.json", {"at": core.now(), "expected_accounts": 16})
    m = pd.read_parquet(PRICE_BASE / "market.parquet"); m.date = pd.to_datetime(m.date)
    m = m[m.date.le(END)].reset_index(drop=True)
    risk = pd.read_parquet(PRICE_BASE / "price_features.parquet")
    assert m.date.equals(risk.date)
    f = build(m); f["es95"] = risk.es95.to_numpy()
    f.to_parquet(OUT / "proxy_features.parquet", index=False)
    dividends = core.normalize_dividends(pd.read_csv(PRICE_BASE / "dividends.csv"))
    spec = importlib.util.spec_from_file_location("rapid_proxy_account_engine", ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec); sys.modules[spec.name] = engine; spec.loader.exec_module(engine)
    records = []
    for capital in (200000, 20000):
        for cost in ("BASE", "STRESS"):
            for policy in POLICIES:
                ledger, orders = simulate(m, f, dividends, policy, capital, cost, engine)
                directory = OUT / "accounts" / f"{capital}_{cost}_{policy}"; directory.mkdir(parents=True)
                ledger.to_parquet(directory / "ledger.parquet", index=False)
                orders.to_parquet(directory / "orders.parquet", index=False)
                metric = core.metrics(ledger, capital)
                metric.update(policy=policy, capital=capital, cost=cost,
                    entries=int(((ledger.filled_quantity > 0) & (ledger.shares_before == 0)).sum()),
                    ledger_path=str((directory / "ledger.parquet").relative_to(ROOT)))
                for label, a, b in (("early", START, "2020-12-31"), ("late", "2021-01-01", END)):
                    metric.update({label + "_" + key: value for key, value in period_metrics(ledger, a, b).items()})
                records.append(metric)
            print(f"事件代理：{capital}元、{cost}的4个账户已完成。", flush=True)
    table = pd.DataFrame(records)
    table["worth_followup"] = (table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1) & table.entries.ge(20)
                               & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0))
    table["joint_historical_target"] = table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1)
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table[(table.capital == 200000) & (table.cost == "STRESS")].sort_values("net_sharpe", ascending=False)
    result = {"at": core.now(), "study_id": "510300_FACTOR96_RAPID_CAPITAL_PROXY_V1",
        "classification": "EXPLORATORY_INCOMPLETE_COVERAGE_EVENT_PROXIES", "account_scenarios": 16,
        "primary_results": primary.to_dict("records"), "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(),
        "joint_historical_target_count": int(primary.joint_historical_target.sum()), "strict_T12_T13_validated": False,
        "new_network_requests": 0, "qualified_candidates": 0, "independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False, "delivery_package_created": False}
    save(OUT / "result.json", result)
    print(primary[["policy", "net_sharpe", "cagr", "max_drawdown", "entries", "worth_followup"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args(); freeze() if args.stage == "freeze" else run()
