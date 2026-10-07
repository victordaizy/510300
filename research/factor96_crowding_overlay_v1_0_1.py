"""T06：固定价格基准上的融资与内部走弱削减层，保留各单条件对照。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as core
from research import factor96_exact_price_baseline_v1 as exact_base

ROOT = core.ROOT
OUT = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1"
FIRST = core.OUT / "completed_run"
INTERNAL = ROOT / "reports/research/510300_factor96_internal_reclaim_v1_0_1"
STUDY = "510300_FACTOR96_CROWDING_OVERLAY_V1_0_1"
POLICIES = ["PRICE_ALL", "PRICE_COMMON", "FINANCE_ONLY", "INTERNAL_ONLY", "FULL"]
save, now, digest = core.save, core.now, core.digest


def same_member_medians(returns, members, allowed, minimum=294, expected_members=300):
    """比较两日同一批成员的20日总收益中位数，缺失不填零。"""
    assert returns.index.equals(members.index) and returns.columns.equals(members.columns)
    log_return = np.log1p(returns)
    cumulative = np.expm1(log_return.rolling(20, min_periods=20).sum())
    complete = returns.notna().rolling(20, min_periods=20).sum().eq(20)
    common = members & members.shift(5, fill_value=False) & complete & complete.shift(5, fill_value=False)
    records = []
    for t in range(len(returns)):
        ids = np.flatnonzero(common.iloc[t].to_numpy())
        count = len(ids)
        med_now = float(np.median(cumulative.iloc[t, ids])) if count else np.nan
        med_prior = float(np.median(cumulative.iloc[t-5, ids])) if count and t >= 5 else np.nan
        known = bool(t >= 5 and count >= minimum and members.iloc[t].sum() == expected_members
                     and members.iloc[t-5].sum() == expected_members and allowed.iloc[t] and allowed.iloc[t-5])
        records.append({"date": returns.index[t], "common_members": count, "median20_current": med_now,
                        "median20_prior5_same_members": med_prior, "median20_change5": med_now-med_prior,
                        "internal_known": known, "internal_weak": bool(known and med_now < med_prior)})
    return pd.DataFrame(records), cumulative, common


def internal_features(market, classified, membership, daily_coverage):
    dates = pd.DatetimeIndex(market.date)
    universe = membership.rename(columns={"membership_date": "date"})
    assert universe.groupby("date").symbol.nunique().eq(300).all()
    symbols = pd.Index(sorted(universe.symbol.unique()))
    known = (classified.return_is_usable & classified.constituent_return_state.isin(["TRADED_VALID", "OFFICIAL_SUSPENSION"])
             & np.isfinite(classified.daily_total_shareholder_return) & classified.daily_total_shareholder_return.gt(-1))
    returns = classified.assign(value=classified.daily_total_shareholder_return.where(known)).pivot(index="date", columns="symbol", values="value")
    returns = returns.reindex(index=dates, columns=symbols)
    members = universe.assign(member=True).pivot(index="date", columns="symbol", values="member").reindex(index=dates, columns=symbols).eq(True)
    allowed = daily_coverage.set_index("date").aggregation_state.reindex(dates).eq("VIEW_ALLOWED")
    features, cumulative, common = same_member_medians(returns, members, allowed)
    return features, returns, members, cumulative, common


def financing_features(market, margin, price):
    aligned = market[["date"]].merge(margin[["date", "market_rzye"]], on="date", how="left", validate="one_to_one")
    aligned["F5"] = (aligned.market_rzye/aligned.market_rzye.shift(5)-1).where(aligned.market_rzye.rolling(6, min_periods=6).count().eq(6))
    aligned["F5_q80"] = aligned.F5.shift().rolling(252, min_periods=120).quantile(.8)
    aligned["r5"] = price.wealth/price.wealth.shift(5)-1
    aligned["finance_known"] = np.isfinite(aligned[["F5", "F5_q80", "r5"]]).all(axis=1)
    aligned["finance_crowded"] = aligned.finance_known & aligned.F5.gt(aligned.F5_q80) & aligned.r5.le(0)
    return aligned


def lag_features(market, price, internal, financing, lag):
    """经济统计日整体滞后一/二个交易日，价格基准和成熟风险仍用判断当日。"""
    assert lag in (1, 2)
    f = price[["date", "wealth", "es95", *core.STATE]].copy()
    f["ma20"] = f.wealth.rolling(20, min_periods=20).mean()
    f["ma20_exact_relation"] = exact_base.exact_wealth_ma_relation(market)
    f["price_long"] = f.ma20_exact_relation.eq(1).fillna(False).astype(bool)
    external = financing.drop(columns="date").join(internal.drop(columns="date"))
    external["stat_date"] = market.date
    external["stat_idx"] = np.arange(len(market))
    external = external.shift(lag)
    for col in ["finance_known", "finance_crowded", "internal_known", "internal_weak"]:
        external[col] = external[col].fillna(False).astype(bool)
    external["stat_idx"] = external.stat_idx.fillna(-1).astype(int)
    f = pd.concat([f, external], axis=1)
    f["common_known"] = f.finance_known & f.internal_known
    return f


def overlay_transition(active, clear_streak, known, financing, internal, policy, base_live):
    """共同条件触发；两个条件都明确消失两日才恢复，未知不冒充消失。"""
    if not base_live or policy in ["PRICE_ALL", "PRICE_COMMON"]:
        return False, 0
    if not known:
        return active, 0
    if policy == "FULL":
        trigger, clear = financing and internal, not financing and not internal
    elif policy == "FINANCE_ONLY":
        trigger, clear = financing, not financing
    elif policy == "INTERNAL_ONLY":
        trigger, clear = internal, not internal
    else:
        raise ValueError(f"未登记削减层：{policy}")
    if not active:
        return bool(trigger), 0
    clear_streak = clear_streak+1 if clear else 0
    return (False, 0) if clear_streak >= 2 else (True, clear_streak)


def reduced_quantity(baseline_quantity, active):
    """在风险限制后的基准份额上减半，100份基准可降到零但不清除削减状态。"""
    assert baseline_quantity >= 0 and baseline_quantity % 100 == 0
    return (baseline_quantity//200)*100 if active else baseline_quantity


def load_engine():
    spec = importlib.util.spec_from_file_location("factor96_crowding_engine", OUT/"code/account_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simulate(d, x, dividends, policy, capital, cost_name, start, end, engine):
    ids = np.flatnonzero(d.date.between(start, end))
    first, last = int(ids[0]), int(ids[-1])
    account = engine.Account(float(capital))
    cost, cfg = core.COSTS[cost_name], {"lot": 100, "tick": .001, "limit_fraction": .1}
    previous_equity, previous_mark, previous_reserve, peak = capital, float(d.close.iloc[first-1]), 0., capital
    stopped, exit_pending, base_live, active = False, False, False, False
    base_ceiling, clear_streak, cycle_no, base_entry_idx = 0, 0, 0, -1
    events, records, decisions = dividends.to_dict("records"), [], []
    for i in ids:
        row, f = d.iloc[i], x.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        old_base_live, old_active, old_streak, old_ceiling = base_live, active, clear_streak, base_ceiling
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0)*event["cash_dividend_per_share"]
                account.receivables[k], recognized = value, recognized+value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash, paid = account.cash+value, paid+value
        ref = float(d.close.iloc[i-1]-row.dividend)
        risk_target = core.target_quantity(engine, account, ref, peak, float(f.es95))
        can_enter = bool(f.price_long and (policy == "PRICE_ALL" or f.common_known))
        q, reason, initial_entry = 0, "无新的可进入价格基准", False
        if base_live:
            exit_pending |= bool(stopped or not f.price_long or i >= base_entry_idx+5)
            if exit_pending:
                q, reason = -old, "价格基准转弱、五日到期或风险停机，继续退出直到完成"
            else:
                base_ceiling = min(base_ceiling, risk_target)
                active, clear_streak = overlay_transition(active, clear_streak, bool(f.common_known),
                    bool(f.finance_crowded), bool(f.internal_weak), policy, True)
                target = reduced_quantity(base_ceiling, active)
                q = target-old
                reason = "覆盖层减至风险限制后原份额的一半" if active else "基准持有或两个条件消失两日后恢复"
        elif not stopped and can_enter and i < last:
            q, reason, initial_entry = risk_target, "价格基准下一开盘首次进入，削减层只管理已有周期", True
        requested_before_open = q
        if q > 0:
            open_risk = core.target_quantity(engine, account, op, peak, float(f.es95))
            if base_live:
                base_ceiling = min(base_ceiling, open_risk)
                q = max(0, min(q, reduced_quantity(base_ceiling, active)-old))
            else:
                q = min(q, open_risk)
        sellable = account.sellable(i)
        trade = engine.execute_order(account, q, op, float(row.previous_close), float(row.dividend), int(i), cost, cfg)
        if initial_entry and trade["filled_quantity"] > 0:
            base_live, base_ceiling, active, clear_streak = True, account.shares, False, 0
            base_entry_idx = int(i)
            cycle_no += 1
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash, paid = account.cash+value, paid+value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = engine.fill_price(close, -1, core.COSTS["STRESS"], .001)
            reserve = account.shares*(close-px)+engine.commission(account.shares, px, core.COSTS["STRESS"])
        equity = account.value(close)-reserve
        price_pnl = old*(op-previous_mark)+account.shares*(close-op)
        error = equity-previous_equity-price_pnl-recognized+trade["commission"]+trade["slippage_cost"]+reserve-previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1-equity/peak
        stopped |= drawdown >= .1
        records.append({"date": day, "idx": i, "policy": policy, "open": op, "mark": close, "cash": account.cash,
            "shares": account.shares, "dividend_receivable": account.receivable(), "terminal_exit_reserve": reserve,
            "equity": equity, "net_return": equity/previous_equity-1, "price_pnl": price_pnl, "dividend_recognized": recognized,
            "dividend_paid": paid, "exposure": account.shares*close/equity, "accounting_error": error, "drawdown": drawdown,
            "risk_stopped": stopped, "sellable_before": sellable, "terminal_unliquidated": bool(i == last and account.shares), **trade})
        decisions.append({"date": day, "origin": d.date.iloc[i-1], "origin_idx": i-1, "policy": policy,
            **{col: f[col] for col in ["stat_date", "stat_idx", "price_long", "common_known", "finance_known", "internal_known",
                "finance_crowded", "internal_weak", "F5", "F5_q80", "r5", "median20_change5", "common_members"]},
            "can_enter": can_enter, "es95_5d": f.es95, "base_live_before": old_base_live, "base_live": base_live,
            "base_ceiling_before": old_ceiling, "base_ceiling": base_ceiling, "overlay_before": old_active,
            "overlay_active": active, "clear_streak_before": old_streak, "clear_streak": clear_streak,
            "base_exit_pending": exit_pending, "cycle_no": cycle_no, "base_entry_idx": base_entry_idx, "initial_entry": initial_entry,
            "risk_target": risk_target, "reason": reason, "pre_open_request": requested_before_open,
            "requested_quantity": q, "filled_quantity": trade["filled_quantity"]})
        # 半仓因整手取整变成零，不等于价格基准已经结束，削减状态必须继续保存。
        if exit_pending and account.shares == 0:
            base_live, exit_pending, active, clear_streak, base_ceiling = False, False, False, 0, 0
            base_entry_idx = -1
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def prepare():
    assert not (OUT/"source_receipt.json").exists()
    for name in ["inputs", "code", "source_evidence", "program_before"]:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    paths = {
        "inputs/market.parquet": FIRST/"inputs/market.parquet", "inputs/dividends.csv": FIRST/"inputs/dividends.csv",
        "inputs/margin.parquet": FIRST/"inputs/margin.parquet", "inputs/price_features.parquet": FIRST/"daily_features_lag1.parquet",
        "inputs/risk_training_records.json": FIRST/"risk_training_records.json", "inputs/mature_risk_labels.parquet": FIRST/"mature_risk_labels.parquet",
        "inputs/classified.parquet": INTERNAL/"inputs/classified.parquet", "inputs/membership.parquet": INTERNAL/"inputs/membership.parquet",
        "inputs/daily_coverage.parquet": INTERNAL/"inputs/daily_coverage.parquet",
        "source_evidence/margin_metadata.json": FIRST/"inputs/margin_metadata.json",
        "source_evidence/margin_repair_receipt.json": core.OUT/"data_repair/source_receipt.json",
        "source_evidence/constituent_source_receipt.json": INTERNAL/"source_receipt.json",
        "source_evidence/return_build_receipt.json": INTERNAL/"source_evidence/return_build_receipt.json",
        "source_evidence/membership_admission.json": INTERNAL/"source_evidence/membership_admission.json",
        "source_evidence/membership_2015_admission.json": INTERNAL/"source_evidence/membership_2015_admission.json",
        "source_evidence/original_T06_numeric_limitation.json": ROOT/"reports/research/510300_factor96_crowding_overlay_v1/numeric_boundary_limitation.json",
        "source_evidence/original_T06_protocol.json": ROOT/"reports/research/510300_factor96_crowding_overlay_v1/protocol.json",
        "source_evidence/current_mandate.json": ROOT/"config/510300_existing_data_training_mandate_v1.json",
        "source_evidence/old_price_baseline.md": ROOT/"docs/510300_BREADTH_MAJORITY_TREND_V1.md",
        "source_evidence/old_leverage_discovery.py": ROOT/"research/market_leverage_cascade_5d_discovery_v0.py",
        "source_evidence/old_leverage_result.json": ROOT/"reports/discovery/510300_market_leverage_cascade_5d_discovery_v0.json",
        "source_evidence/old_cross_etf_contract.yaml": ROOT/"config/510300_cross_etf_forced_flow_binary_screen_v1_candidates.yaml",
        "source_evidence/old_cross_etf_result.json": ROOT/"reports/research/510300_cross_etf_forced_flow_binary_screen_v1.json",
        "source_evidence/share_publication_closure.json": ROOT/"reports/research/510300_fund_share_publication_receipts_closure_v1/result.json",
        "source_evidence/weekly_share_sample.json": ROOT/"data/raw/cross_etf_forced_flow_v1/sse_weekly_snapshots/2021-01-08.json",
        "source_evidence/weekly_share_receipt.json": ROOT/"reports/data_quality/510300_cross_etf_forced_flow_sse_weekly_v1.json",
    }
    receipts = []
    for name, path in paths.items():
        shutil.copy2(path, OUT/name)
        receipts.append({"original": path.relative_to(ROOT).as_posix(), "snapshot": name, "sha256": digest(path)})
    for p in (ROOT/"reports/research/510300_factor96_program_v1").iterdir():
        if p.is_file():
            shutil.copy2(p, OUT/"program_before"/p.name)
    source = json.loads((OUT/"source_evidence/constituent_source_receipt.json").read_text(encoding="utf-8"))
    for key in ["classified", "membership", "daily_coverage"]:
        expected = next(r["sha256"] for r in source["sources"] if r["snapshot"] == f"inputs/{key}.parquet")
        assert digest(OUT/f"inputs/{key}.parquet") == expected
    repair = json.loads((OUT/"source_evidence/margin_repair_receipt.json").read_text(encoding="utf-8"))
    assert digest(OUT/"inputs/margin.parquet") == repair["new_file_sha256"]
    save(OUT/"source_evidence/overlap_and_source_decisions.json", {"at": now(),
        "T07": {"status": "NOT_RUN_DAILY_SYSTEM_FLOW_SOURCE_GATE", "existing_data": "固定10只上海ETF周度份额，沪深300子池3只，2021年起；没有当时动态全体系、逐日NAV及份额发布时点。",
                "old_terminal_rule": "既有五条周度二元申赎规则固定失败，禁止改ETF池、阈值、方向、时钟或组合救援。",
                "new_definition_missing": ["历史同指数上市产品池及退出", "日频份额与拆分调整", "同口径前日NAV", "与数值对应的历史发布时钟"],
                "pcf_boundary": "公开申赎清单的开市前规则与前日NAV，不自动证明规模查询表TOT_VOL的历史首次公开时点；不重扫已关闭的2220条同一份额响应。"},
        "T06": {"status": "ADMITTED_OVERLAY_QUESTION_DISCOVERY_ONLY", "difference": "旧融资级联是大量全A特征与分类器的二元收益发现；本次是已固定价格基准下，融资增长高分位但价格停滞与同成员中位收益变弱共同触发的减半/恢复层，保留两项单独削减及无削减对照。",
                "baseline_boundary": "沿用旧20日均线价格条件作为观察基准，采用T06表列5日持有上限及当前共同风险合同。不是重跑旧PRICE20_ONLY完整规则，也不登记或晋升为新价格策略。",
                "five_day_field_resolution": "T06表中max_hold_days=5与条件恢复同时出现。本轮把5日明确作为各臂共同的基准最大持有期限，而非削减状态届时强制恢复；价格或到期退出优先。",
                "F05_boundary": "F5严格按过去252日80分位定义；余额/自由流通市值未额外使用，不声称已测试全部F05。",
                "E05_boundary": "同成员20日中位收益的5日变化是主触发；ETF与中位收益差另存，不把缺权重补成权重信号。"}}, True)
    save(OUT/"source_receipt.json", {"at": now(), "sources": receipts, "hashes_match_previous_frozen_sources": True,
        "new_network_data_requests": 0, "new_strategy_returns_evaluated": False,
        "historical_first_publication_receipts": "NOT_ESTABLISHED", "member_weights_used": False}, True)
    print("T06来源准备完成，复用已修正融资、四态成分收益和点时成员；T07来源门单独保留。", flush=True)


def freeze():
    assert (OUT/"source_receipt.json").exists() and not (OUT/"freeze.json").exists()
    assert json.loads((OUT/"prefreeze_test_receipt.json").read_text(encoding="utf-8"))["exit_code"] == 0
    protocol = {"study_id": STUDY, "at": now(), "primary": "FULL", "new_primary_overlays": 1, "new_entry_strategies": 0,
        "periods": core.PERIODS, "capital": [200000, 20000], "costs": core.COSTS, "annual_days": 242,
        "policies": POLICIES, "lag1_primary": True, "lag2_sensitivity_policies": ["PRICE_COMMON", "FULL"], "expected_accounts": 56,
        "numerical_boundary": "原始0.001元报价与分红的有理数财富比较，严格相等为非多头；仅修复原T06数值实现，不改经济条件。",
        "base": "财富收盘>包含当日20个财富收盘均值时，现金账户下一开盘进入；<=均值、入场后5个开盘间隔或共同风险停机，下一合法开盘退出。每周期原始实际入场份额为上限，只能随风险预算削减。",
        "baseline_changes_disclosed": "只借用旧20日价格状态，采用T06表列5日上限，不借用旧6%止损/8%追踪/60日/等待/重入完整合同，当前共同ES/回撤合同对各臂一致；基准不当新策略晋升。",
        "five_day_field_resolution": "候选表max_hold_days=5固定解释为基准最大持有5个开盘间隔，覆盖层状态仍按两个条件消失两日恢复；到期优先退出，不在第5日先恢复再退出。",
        "entry_coverage": "PRICE_COMMON、两个单条件及FULL初始进入共同要求融资与同成员源均已知；PRICE_ALL不要求。持仓期缺失不自动退出，也不视为触发消失。",
        "F5": "融资余额[s]/余额[s-5]-1；必须连续6个A股日有值；P80使用严格此前252日、至少120条有效F5。",
        "financing_condition": "F5>P80且同经济日510300含分红5日收益<=0；不追加余额/流通市值或绝对融资阈值。",
        "internal_condition": "s和s-5均为点时成员且两边20日总收益完整的同一批股票，至少294/300，两日四态日门允许；其s日20日总收益中位数严格低于s-5。",
        "membership_and_missing": "按当时成员，不用当前池；每个20日必须全部有效收益，官方已确认停牌收益0可用，其他缺失不填零。",
        "clock": "主版本统计日s的信息在下一A股日收盘判断，随后开盘执行。融资、价格停滞和成分状态用同一个s；源首版未认证，时钟是保守历史研究假设。延迟两日仅敏感性。",
        "cut": "已进入基准周期后，FULL两个条件同时为真才削减；先按当日ES/回撤预算限制原份额，再将允许份额减半并向下取整100份。不是把50%名义目标改25%后忽略已收缩风险预算。",
        "restore": "FULL要求融资条件及内部条件都明确为假，连续2个判断日后恢复风险允许的原份额；未知重置消失计数但保留削减。单条件对照同样等待该条件为假两日。",
        "priority": "价格基准退出/停机优先且请求不可撤回；削减到0仍保持基准周期与覆盖层状态，不在下一日误当新周期全仓进入。",
        "risk": "冻结两年成熟五日ES95每天更新；最高50%目标、2.5%ES预算、10%跳空下5%预算、回撤剩余空间折半、回撤10%后清仓且不恢复。",
        "account": "100份、0.001元、T+1、方向涨跌停、分红权益/应收/付款、最低5元费用、期末压力退出成本准备；完整现金日保留。只允许原计划削减后恢复，不摊平加码。",
        "acceptance": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1, "both_main_capitals_stress_required": True},
        "bootstrap": {"comparisons": ["FULL_MINUS_PRICE_COMMON", "FULL_MINUS_FINANCE_ONLY", "FULL_MINUS_INTERNAL_ONLY"],
                      "period": "MAIN", "capital": 200000, "cost": "STRESS", "block": 20, "draws": 4000, "seed": 20261001},
        "selection": "本库第6个固定主问题，其中5个入场候选、1个覆盖层；旧失败保留，本轮不同于新增独立入场。",
        "stop": "固定联合覆盖层不达标就结束；不改80分位、5日、20日、两日恢复、基准或数据门，不晋升单条件对照。",
        "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "orders_authorized": False,
        "goal_achieved": False, "goal_status": "active"}
    for name, path in {"factor96_crowding_overlay_v1_0_1.py": Path(__file__), "factor96_margin_repair_v1.py": Path(core.__file__),
        "factor96_exact_price_baseline_v1.py": Path(exact_base.__file__),
        "test_factor96_exact_price_baseline_v1.py": ROOT/"tests/test_factor96_exact_price_baseline_v1.py",
        "account_engine.py": ROOT/"research/intraday_overnight_increment_v1.py",
        "test_factor96_crowding_overlay_v1_0_1.py": ROOT/"tests/test_factor96_crowding_overlay_v1_0_1.py"}.items():
        shutil.copy2(path, OUT/"code"/name)
    save(OUT/"protocol.json", protocol, True)
    files = [p for folder in ["inputs", "code", "source_evidence", "program_before"] for p in (OUT/folder).rglob("*") if p.is_file()]
    files += [OUT/"source_receipt.json", OUT/"protocol.json", OUT/"prefreeze_test_receipt.json"]
    save(OUT/"freeze.json", {"at": now(), "before_new_features_and_strategy_returns": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in files]}, True)
    print("T06联合削减层及单条件对照已冻结，共56个账户情景。", flush=True)


def run():
    frozen = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    for r in frozen["files"]:
        assert digest(OUT/r["path"]) == r["sha256"], r["path"]
    assert digest(Path(__file__)) == digest(OUT/"code/factor96_crowding_overlay_v1_0_1.py")
    assert digest(Path(core.__file__)) == digest(OUT/"code/factor96_margin_repair_v1.py")
    assert digest(Path(exact_base.__file__)) == digest(OUT/"code/factor96_exact_price_baseline_v1.py")
    save(OUT/"run_started.json", {"at": now(), "freeze_sha256": digest(OUT/"freeze.json")}, True)
    market = pd.read_parquet(OUT/"inputs/market.parquet")
    market = market[market.date.le("2025-12-31")].reset_index(drop=True)
    price = pd.read_parquet(OUT/"inputs/price_features.parquet")
    assert market.date.equals(price.date)
    internal, returns, members, cumulative, common = internal_features(market, pd.read_parquet(OUT/"inputs/classified.parquet"),
        pd.read_parquet(OUT/"inputs/membership.parquet"), pd.read_parquet(OUT/"inputs/daily_coverage.parquet"))
    internal["ETF20_minus_median"] = price.wealth/price.wealth.shift(20)-1-internal.median20_current
    for name, frame in [("constituent_returns", returns), ("membership_mask", members), ("constituent_return20", cumulative), ("common_member_mask", common)]:
        frame.to_parquet(OUT/(name+".parquet"))
    internal.to_parquet(OUT/"internal_features.parquet", index=False)
    financing = financing_features(market, pd.read_parquet(OUT/"inputs/margin.parquet"), price)
    financing.to_parquet(OUT/"financing_features.parquet", index=False)
    div, engine = core.normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv")), load_engine()
    metrics, annual, stages, accounts, overlays = [], [], [], {}, []
    for lag in [1, 2]:
        x = lag_features(market, price, internal, financing, lag)
        x.to_parquet(OUT/f"daily_features_lag{lag}.parquet", index=False)
        for period, (start, end) in core.PERIODS.items():
            v = x[x.date.between(start, end)]
            stages.append({"period": period, "lag": lag, "days": len(v), "common_known": int(v.common_known.sum()),
                "finance_condition": int((v.common_known & v.finance_crowded).sum()),
                "internal_condition": int((v.common_known & v.internal_weak).sum()),
                "joint_condition": int((v.common_known & v.finance_crowded & v.internal_weak).sum()),
                "price_long_joint": int((v.price_long & v.common_known & v.finance_crowded & v.internal_weak).sum())})
            for capital in [200000, 20000]:
                for cost in core.COSTS:
                    for policy in (POLICIES if lag == 1 else ["PRICE_COMMON", "FULL"]):
                        ledger, decisions = simulate(market, x, div, policy, capital, cost, start, end, engine)
                        folder = OUT/f"accounts/{period}/{capital}/{cost}/LAG{lag}/{policy}"
                        folder.mkdir(parents=True, exist_ok=True)
                        ledger.to_parquet(folder/"ledger.parquet", index=False)
                        decisions.to_parquet(folder/"decisions.parquet", index=False)
                        cycles = core.cycle_records(ledger, capital)
                        cycles.to_csv(folder/"cycles.csv", index=False, encoding="utf-8-sig")
                        closed = cycles[cycles.closed.astype(bool)]
                        key = {"period": period, "capital": capital, "cost": cost, "lag": lag, "policy": policy}
                        m = core.metrics(ledger, capital)
                        metrics.append({**key, **m, "closed_cycles": len(closed), "win_rate": closed.profit.gt(0).mean() if len(closed) else None,
                            "point_pass": m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["cagr"] >= .1 and m["max_drawdown"] <= .1})
                        cut = decisions.overlay_active & ~decisions.overlay_before
                        restored = ~decisions.overlay_active & decisions.overlay_before & ~decisions.base_exit_pending
                        overlays.append({**key, "cut_episodes": int(cut.sum()), "restore_episodes": int(restored.sum()),
                            "reduced_decision_days": int(decisions.overlay_active.sum()), "initial_base_cycles": int(decisions.cycle_no.max()),
                            "reduced_zero_share_days": int((decisions.overlay_active & ledger.shares.eq(0)).sum())})
                        prior = capital
                        for year, part in ledger.groupby(ledger.date.dt.year):
                            annual.append({**key, "year": int(year), **core.metrics(part, prior)})
                            prior = float(part.equity.iloc[-1])
                        accounts[period, capital, cost, lag, policy] = ledger
                        if period == "MAIN" and capital == 200000 and cost == "STRESS":
                            print(f"T06 {policy}/延迟{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，触发削减{int(cut.sum())}次。", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT/"metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlays).to_csv(OUT/"overlay_counts.csv", index=False, encoding="utf-8-sig")
    save(OUT/"signal_stage_counts.json", stages)
    selected = frame[(frame.period == "MAIN") & (frame.cost == "STRESS") & (frame.lag == 1) & (frame.policy == "FULL")]
    a = accounts["MAIN", 200000, "STRESS", 1, "FULL"]
    n = len(a)
    starts = np.random.default_rng(20261001).integers(0, n, size=(4000, int(np.ceil(n/20))))
    indices = ((starts[:, :, None]+np.arange(20)) % n).reshape(4000, -1)[:, :n]
    np.savez_compressed(OUT/"bootstrap_indices.npz", indices=indices)
    increments = []
    for policy in ["PRICE_COMMON", "FINANCE_ONLY", "INTERNAL_ONLY"]:
        diff = a.net_return.to_numpy()-accounts["MAIN", 200000, "STRESS", 1, policy].net_return.to_numpy()
        boot = diff[indices].mean(axis=1)*242
        increments.append({"comparison": "FULL_MINUS_"+policy, "annual_arithmetic_increment": diff.mean()*242,
                           "ci95_low": np.quantile(boot, .025), "ci95_high": np.quantile(boot, .975), "global_selection_adjusted": False})
    save(OUT/"paired_increment.json", increments)
    save(OUT/"result.json", {"study_id": STUDY, "at": now(), "new_accounts": len(frame), "new_primary_overlays": 1,
        "new_entry_strategies": 0, "primary_rows": selected.to_dict("records"), "historical_joint_point_pass": bool(selected.point_pass.all()),
        "increments": increments, "goal_achieved": False, "goal_status": "active", "independent_forward_observations": 0,
        "external_review": "NOT_PERFORMED", "orders_authorized": False})
    print(f"T06固定削减层完成{len(frame)}账户，共同目标通过：{bool(selected.point_pass.all())}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run}[args.action]()
