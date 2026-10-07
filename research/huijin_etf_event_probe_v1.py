"""按已固定的汇金披露事件规则运行四个当前风险合同下的完整账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure
from research.adaptive_allocation_v1 import normalize_dividends
from research.selected_mix_daily_two_year_v1 import five_day_labels, tail_at, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles
from research.selected_mix_migration_factorial_v1 import account_checks
from research.huijin_etf_event_core_v1 import public_states, simulate, PRIMARY, CONTROL

OUT = ROOT / "reports/research/510300_huijin_etf_event_probe_v1"
SOURCE = ROOT / "reports/research/510300_huijin_etf_disclosure_completion_v1"
PRICE = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925"
OLD_TAIL = ROOT / "reports/research/510300_nfci_tail_forecast_increment_v1/forecasts.json"
CORE = ROOT / "research/huijin_etf_event_core_v1.py"


def prepare():
    source = read(SOURCE / "admitted_source_result.json")
    assert source["status"] == "CURRENT_OFFICIAL_CATALOG_COMPLETE_AND_ETF_FACTS_CLASSIFIED"
    assert digest(SOURCE / "classified_etf_facts.json") == source["facts_sha256"]
    data = pd.read_parquet(PRICE / "candidate_features.parquet")
    assert str(data.date.iloc[-1].date()) == "2026-09-24"
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    config = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    facts = read(SOURCE / "classified_etf_facts.json")
    states, episodes, receipts = public_states(data, facts)
    return data, dividends, config, facts, states, episodes, receipts


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本次四账户试验已冻结，不改变参数")
    protocol = read(OUT / "protocol.json")
    assert protocol["planned_new_accounts"] == 4
    data, dividends, config, facts, states, episodes, receipts = prepare()
    # 这里只验证公开状态的信息边界，不计算事件后收益。
    cutoff = int(data.date.searchsorted("2024-12-31", side="right")) - 1
    truncated, _, _ = public_states(data.iloc[:cutoff+1], facts)
    pd.testing.assert_frame_equal(truncated, states.iloc[:cutoff+1].reset_index(drop=True), check_exact=True)
    changed = data.copy()
    changed.loc[cutoff+1:, "wealth"] = 999999.
    mutated, _, _ = public_states(changed, facts)
    pd.testing.assert_frame_equal(mutated.iloc[:cutoff+1], truncated, check_exact=True)
    assert sum(row["episode_action"] == "REPEATED_BUY_DISCLOSURE_NO_EXTENSION" for row in receipts) == 1
    save(OUT / "implementation_clarifications.json", {
        "at": now(), "before_new_accounts": True, "same_day_sell_priority": True,
        "actual_fill_tracking": "等实际成交才计已入场；已成交事件因价格、时限、公开出售或风险全部退出后，均不在同一事件重入。",
        "pending_orders": "涨跌停等导致未成交时，每个收盘重新检查。退出未成交则继续请求退出。事件期限从公告评估日起算，不从后来的成交日延长。",
        "optional_rebalance": "只沿用10个百分点调仓带减少可选小调整，风险超限和明确退出始终优先。",
        "checks": {"future_news_and_price_do_not_change_public_prefix": True,
                   "repeated_disclosure_does_not_extend_window": True},
        "verification_replay": "四个完整账户之外，仅重放主方案压力账户至2024-12-31，逐值检查未来资料不改变已保存前缀；不作为第五个候选。",
        "account_start": "2015-01-05", "account_end": "2026-09-24", "next_official_trading_day": "2026-09-28",
        "statistics": "全日配对收益差，20交易日循环区块、2000次、种子20260925；事件稀疏及历史选择限制保留。"
    }, True)
    (OUT / "code").mkdir()
    for path in [Path(__file__), CORE]:
        shutil.copy2(path, OUT / "code" / path.name)
    sources = local_import_closure({Path(__file__), CORE})
    sources.update([SOURCE / "admitted_source_result.json", SOURCE / "admitted_documents.json", SOURCE / "classified_etf_facts.json",
                    PRICE / "candidate_features.parquet", PRICE / "protocol.json", PRICE / "result.json", OLD_TAIL,
                    ROOT / "config/510300_incremental_selected_intent_mix_v1.json", ROOT / "data/reference/510300_dividends.csv"])
    sources.update(ROOT / row["raw_path"] for row in facts)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "implementation_sha256": digest(OUT / "implementation_clarifications.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print(f"四账户源码及输入已固定；买入披露合并为{len(episodes)}个事件，其中2015年以来{sum(r['review_date'].year>=2015 for r in episodes)}个。", flush=True)


def verification(ledger, decisions):
    account_checks(ledger, decisions)
    assert ledger.shares.ge(0).all() and (ledger.shares % 100).eq(0).all()
    assert (decisions.execution_date > decisions.origin).all()
    assert decisions.loc[decisions.risk_required_reduction, "requested_quantity"].lt(0).all()
    inactive = decisions.public_episode_id.eq(-1)
    assert decisions.loc[inactive, "requested_quantity"].le(0).all()
    completed = decisions.episode_completed | decisions.risk_stopped
    assert decisions.loc[completed, "requested_quantity"].le(0).all()
    return {"cash_and_total_return_accounting": True, "risk_contract_and_t_plus_one_engine": True,
            "no_new_risk_outside_public_episode": True, "no_same_episode_reentry_after_exit": True}


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    assert digest(OUT / "implementation_clarifications.json") == frozen["implementation_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "冻结输入改变：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    data, dividends, config, facts, states, episodes, receipts = prepare()
    states.to_parquet(OUT / "public_event_states.parquet", index=False)
    save(OUT / "episodes.json", episodes, True)
    save(OUT / "disclosure_clock_receipts.json", receipts, True)
    labels = five_day_labels(data, dividends)
    first = int(data.index[data.date.ge("2015-01-05")][0])
    old = {r["origin_index"]: r for r in read(OLD_TAIL) if r["policy"] == "BASELINE"}
    tails, reproduced = [], 0
    for t in range(first-1, len(data)):
        prediction = tail_at(data, labels, t)
        assert prediction["available"] and prediction["latest_label_exit_index"] < t
        if t in old:
            for key in ["label_indices", "training_rows", "q05", "es95", "latest_label_exit_index"]:
                assert prediction[key] == old[t][key]
            reproduced += 1
        tails.append(prediction)
    save(OUT / "daily_two_year_tail_forecasts.json", tails, True)
    print(f"两年日更风险预测已完成{len(tails)}日，复现{reproduced}份旧预测；开始四个连续账户。", flush=True)
    accounts, measurements, yearly, windows, checks = {}, [], [], [], []
    for name in [CONTROL, PRIMARY]:
        for cost in ["BASE", "STRESS"]:
            ledger, decisions, checkpoint = simulate(data, dividends, config, states, episodes, tails, name, cost)
            check = verification(ledger, decisions)
            closed, pending = cycles(ledger)
            folder = OUT / "accounts" / cost / name
            folder.mkdir(parents=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "checkpoint.json", checkpoint, True)
            save(folder / "pending_cycle.json", pending, True)
            accounts[name, cost] = ledger
            measurement = {"policy": name, "cost": cost, **metrics(ledger), "completed_cycles": len(closed),
                           "stopped": checkpoint["risk_governor"]["stopped"], "terminal_shares": int(ledger.shares.iloc[-1])}
            measurement["point_target_pass"] = measurement["sharpe"] is not None and measurement["sharpe"] >= 1.2 and measurement["annual_return"] >= .1 and measurement["max_drawdown"] >= -.1
            measurements.append(measurement)
            checks.append({"policy": name, "cost": cost, **check})
            for year, part in ledger.groupby(ledger.date.dt.year, sort=True):
                index = int(part.index[0])
                capital = 200000. if index == 0 else float(ledger.equity.iloc[index-1])
                yearly.append({"policy": name, "cost": cost, "year": int(year), **metrics(part, capital)})
            for end in range(len(ledger)):
                boundary = ledger.date.iloc[end] - pd.DateOffset(years=2)
                if boundary < ledger.date.iloc[0]:
                    continue
                start = int(ledger.date.searchsorted(boundary, side="left"))
                capital = 200000. if start == 0 else float(ledger.equity.iloc[start-1])
                m = metrics(ledger.iloc[start:end+1], capital)
                windows.append({"policy": name, "cost": cost, "start": ledger.date.iloc[start], "end": ledger.date.iloc[end], **m,
                                "joint_point_pass": m["sharpe"] is not None and m["sharpe"] >= 1.2 and m["annual_return"] >= .1 and m["max_drawdown"] >= -.1})
            sharpe = "未定义" if measurement["sharpe"] is None else f"{measurement['sharpe']:.6f}"
            print(f"{name} {cost}：夏普{sharpe}、年化{measurement['annual_return']:.2%}、回撤{measurement['max_drawdown']:.2%}、完整周期{len(closed)}。", flush=True)
    cutoff = int(data.date.searchsorted("2024-12-31", side="right"))-1
    prefix_data = data.iloc[:cutoff+1].copy()
    prefix_states, prefix_episodes, _ = public_states(prefix_data, facts)
    prefix_tails = [r for r in tails if r["origin_index"] <= cutoff]
    prefix, prefix_decisions, _ = simulate(prefix_data, dividends, config, prefix_states, prefix_episodes,
                                         prefix_tails, PRIMARY, "STRESS", next_execution_date=data.date.iloc[cutoff+1])
    saved = accounts[PRIMARY, "STRESS"]
    pd.testing.assert_frame_equal(prefix, saved[saved.date.le(data.date.iloc[cutoff])].reset_index(drop=True), check_exact=True)
    original_decisions = pd.read_parquet(OUT / "accounts/STRESS" / PRIMARY / "decisions.parquet")
    pd.testing.assert_frame_equal(prefix_decisions.reset_index(drop=True),
                                  original_decisions[original_decisions.origin_index.le(cutoff)].reset_index(drop=True), check_exact=True)
    rng = np.random.default_rng(20260925)
    comparisons = [{"cost": cost, "left": PRIMARY, "right": CONTROL,
                    **paired_interval(accounts[PRIMARY, cost], accounts[CONTROL, cost], rng)} for cost in ["BASE", "STRESS"]]
    save(OUT / "yearly_metrics.json", yearly, True)
    save(OUT / "rolling_two_year_metrics.json", windows, True)
    primary = next(r for r in measurements if r["policy"] == PRIMARY and r["cost"] == "STRESS")
    result = {"at": now(), "study_id": read(OUT / "protocol.json")["study_id"],
              "status": "HISTORICAL_POINT_PASS_DEVELOPMENT_ONLY" if primary["point_target_pass"] else "FROZEN_NO_QUALIFIED_HUIJIN_DISCLOSURE_STRATEGY",
              "primary": primary, "all_accounts": measurements, "comparisons": comparisons,
              "source_documents": 114, "etf_disclosures": 8, "account_period_buy_episodes": sum(r["review_date"].year >= 2015 for r in episodes),
              "source_first_vintage_verified": False, "daily_risk_forecasts": len(tails),
              "saved_risk_forecasts_reproduced": reproduced, "new_full_accounts": 4, "new_return_model_fits": 0,
              "new_parameter_searches": 0, "prefix_replay_days": len(prefix), "prefix_replay_same_account_exact": True,
              "checks": checks, "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in windows if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("汇金披露与价格确认四账户检验完成，源目录缺口已补齐，整体高夏普目标仍须独立证据。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇金ETF披露的固定规则完整账户。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
