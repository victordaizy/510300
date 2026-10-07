"""把固定月度高分机会接入只交易510300和现金的连续账户。"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import multidim_money_surprise_score_v1 as study


ROOT, OUT = study.ROOT, study.OUT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import factor96_margin_repair_v1 as core

ENGINE_PATH = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/code/account_engine.py"
RISK_PATH = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/price_features.parquet"
DIVIDENDS = ROOT / "data/reference/510300_dividends.csv"
COST = {"commission": .0004, "minimum": 5., "slippage": .001}


def engine_module():
    spec = importlib.util.spec_from_file_location("money_joint_fixed_order_account", ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return engine


def prepare() -> None:
    if (OUT / "account_protocol.json").exists():
        raise RuntimeError("账户合同已固定，不覆盖。")
    result = study.read(OUT / "result.json")
    primary = next(row for row in result["opportunity_summary"] if row["model"] == "joint" and row["era"] == "2024—2025")
    assert primary["n"] > 0 and primary["mean_net5"] > 0
    mandate = study.read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    cases = [{"id": f"{model}_{capital}", "model": model, "capital": capital} for model in ["state", "joint"] for capital in [200000, 20000]]
    cases += [{"id": "cash_200000", "model": "cash", "capital": 200000}, {"id": "buy_hold50_200000", "model": "buy_hold50", "capital": 200000}]
    study.save("account_protocol.json", {
        "registered_at": study.common.now(), "study_id": study.STUDY,
        "reason": "固定主要期高分事件收益为正，按原研究步骤接入完整账户。M2变量在主要期未被树使用的事实保留。",
        "prior_seen": "已看到5次高分机会均值+1.90%及其中2024年9月贡献集中；本次不改变任何信号、持有期、阈值或成本。",
        "calendar": ["2021-01-04", "2025-12-31"], "primary_subperiod": ["2024-01-01", "2025-12-31"],
        "cases": cases, "capital_rule": "20万元主账户与2万元成本比较账户分别计算。",
        "candidate_signals": "只使用评分阶段保存的score>=80且预测净收益>0事件；已有仓位不加仓；只对预定下一开盘有效；退出计划为入场后第5个交易日开盘。",
        "order_quantity_clock": "开盘前已决定整数份额。以已知前收盘除息参考价及当日10%上限价规划最差买价，按上限价检验现金与风险后提交固定份额，实际开盘只决定限价内成交与否，不重新计算较有利的数量。",
        "liquidity_approximation": "日线开盘压力价为成交代理；方向涨跌停或压力价越限均不成交；无逐笔队列保证。买单当日失效，必要卖单顺延下一可卖开盘。",
        "risk_contract": mandate["account_tail_risk_contract"],
        "risk_estimate": "复用既有每日成熟五日收益ES：过去两年、三项价格状态、至少252成熟原点、126近邻的最差7项均值；只读前一交易日ES。",
        "sizing": "在已知账户参考净值下，以预先可计算的价格上限估计持仓市值，并把调整和退出压力费用一并纳入50%仓位、2.5%ES、5%缺口与半回撤剩余空间限制。持仓仅减不加。",
        "drawdown": "收盘回撤>=10%后下一可卖开盘退出并永久停止该账户；不保证实际最大回撤始终小于10%。",
        "dividends": "登记日收盘确定权益，除息日记应收，支付日收盘转现金；应收不用于支付买单。",
        "terminal": "期末按收盘估值并扣未平仓压力退出准备，不强平制造完整周期。",
        "benchmarks": "全现金零利息；期初50%预算买入持有为描述性参考，不遵循动态ES和止损，不能当作合格候选。",
        "cost": COST, "lot": 100, "tick": .001, "annual_days": 242, "cash_rate": 0., "risk_free_rate": 0.,
        "evaluation": "全部现金日纳入夏普和年化；列原连续账户完整期、较早期、主要期，不重置资金择优分段。逐年自然完成周期保留零次年份。",
        "account_targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1, "minimum_cycles_each_full_year": 5},
        "source_receipts": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": study.sha(p)} for p in [study.DAILY, RISK_PATH, DIVIDENDS, ENGINE_PATH, OUT/"逐事件联合评分.csv", OUT/"全部固定高分机会.csv"]],
        "parameter_search": False, "new_forecast": False, "orders_authorized": False, "goal_achieved": False,
    })
    print("已固定六个连续账户及开盘前固定份额合同，不修改评分信号。", flush=True)


def planned_target(engine, account, reference, limit_price, peak, es, benchmark=False) -> tuple[int, dict]:
    nav = account.value(reference)
    remaining = max(0., nav-.9*peak)
    gap_budget = min(.05*nav, .5*remaining)
    planning = {"nav_before_open_at_reference": nav, "reference_price": reference, "limit_price": limit_price,
                "prior_peak": peak, "prior_es95": es, "gap_budget": gap_budget, "es_budget": .025*nav}
    if not benchmark and not np.isfinite(es):
        return 0, planning
    quantity = math.floor(.5*nav/limit_price/100)*100
    if account.shares:
        quantity = min(quantity, account.shares)
    while quantity > 0:
        delta = quantity-account.shares
        execution_px = engine.fill_price(limit_price, 1 if delta > 0 else -1, COST, .001)
        friction = abs(delta)*abs(execution_px-limit_price)+engine.commission(delta, execution_px, COST)
        exit_px = engine.fill_price(limit_price, -1, COST, .001)
        exit_cost = quantity*(limit_price-exit_px)+engine.commission(quantity, exit_px, COST)
        notional = quantity*limit_price
        cash_ok = delta <= 0 or delta*limit_price+engine.commission(delta, limit_price, COST) <= account.cash+1e-8
        risk_ok = benchmark or (notional*es+friction+exit_cost <= .025*nav+1e-8 and notional*.1+friction+exit_cost <= gap_budget+1e-8)
        if cash_ok and risk_ok and notional <= .5*(nav-friction)+1e-8:
            planning.update(planned_target=quantity, planned_notional=notional, planned_friction=friction,
                            planned_exit_cost=exit_cost, planned_ES_loss=notional*es+friction+exit_cost if np.isfinite(es) else None,
                            planned_gap_loss=notional*.1+friction+exit_cost)
            return quantity, planning
        quantity -= 100
    planning.update(planned_target=0, planned_notional=0., planned_friction=0., planned_exit_cost=0., planned_ES_loss=0., planned_gap_loss=0.)
    return 0, planning


def simulate(market, risk, dividends, selected, case, engine):
    indices = np.flatnonzero(market.date.between("2021-01-04", "2025-12-31"))
    first, last = int(indices[0]), int(indices[-1])
    account = engine.Account(float(case["capital"]))
    selected = {pd.Timestamp(r.entry_date): r for r in selected.itertuples(index=False)}
    events = dividends.to_dict("records")
    previous_nav = peak = float(case["capital"])
    previous_mark, reserve_before = float(market.close.iloc[first-1]), 0.
    entry_idx, stopped, pending_exit = None, False, False
    rows = []
    for i in indices:
        row = market.iloc[i]
        day, old = row.date, account.shares
        recognized, paid = 0., 0.
        for key, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(key, 0)*event["cash_dividend_per_share"]
                if amount:
                    account.receivables[key] = amount
                    recognized += amount
            if event["payment_date"] < day and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount
        reference = float(market.close.iloc[i-1]-row.dividend)
        upper = math.floor(reference*1.1/.001+.5+1e-9)*.001
        es = float(risk.es95.iloc[i-1])
        quantity, reason, planning = 0, "现金或持有", {}
        if old and case["model"] in ["state", "joint"]:
            pending_exit |= stopped or i >= entry_idx+5
            if pending_exit:
                quantity, reason = -old, "回撤停止退出" if stopped else "固定五日到期退出"
            else:
                target, planning = planned_target(engine, account, reference, upper, peak, es)
                quantity, reason = target-old, "按已知预算减仓"
        elif not old and not stopped and day in selected and case["model"] in ["state", "joint"]:
            quantity, planning = planned_target(engine, account, reference, upper, peak, es)
            reason = "已公布事件的固定高分入场"
        elif case["model"] == "buy_hold50" and i == first:
            quantity, planning = planned_target(engine, account, reference, upper, peak, es, benchmark=True)
            reason = "期初50%预算持有参考"
        # 数量已由已知参考价与限价确定；以下才读取实际开盘成交代理。
        execution = engine.execute_order(account, quantity, float(row.open), float(row.previous_close), float(row.dividend), int(i), COST, {"lot":100,"tick":.001,"limit_fraction":.1})
        if quantity > 0 and execution["filled_quantity"]:
            assert execution["filled_quantity"] == quantity
            assert execution["fill_price"] <= upper+1e-9
            entry_idx, pending_exit = int(i), False
        for key, event in enumerate(events):
            if event["payment_date"] == day and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount
            if event["record_date"] == day:
                account.entitlements[key] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = engine.fill_price(float(row.close), -1, COST, .001)
            reserve = account.shares*(float(row.close)-px)+engine.commission(account.shares, px, COST)
        equity = account.value(float(row.close))-reserve
        price_pnl = old*(float(row.open)-previous_mark)+account.shares*(float(row.close)-float(row.open))
        identity_error = equity-previous_nav-price_pnl-recognized+execution["commission"]+execution["slippage_cost"]+reserve-reserve_before
        assert abs(identity_error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1-equity/peak
        if case["model"] in ["state", "joint"]:
            stopped |= drawdown >= .1
        rows.append({"date":day,"idx":int(i),"equity":equity,"net_return":equity/previous_nav-1,
                     "cash":account.cash,"shares":account.shares,"mark":float(row.close),"dividend_receivable":account.receivable(),
                     "dividend_recognized":recognized,"dividend_paid":paid,"terminal_exit_reserve":reserve,
                     "exposure":account.shares*float(row.close)/equity,"drawdown":drawdown,"risk_stopped":stopped,
                     "reason":reason,"accounting_error":identity_error,"prior_reference_price":reference,"precommitted_upper_price":upper,
                     "risk_observation_date":market.date.iloc[i-1],**execution,**planning})
        if account.shares == 0:
            entry_idx, pending_exit = None, False
        previous_nav, previous_mark, reserve_before = equity, float(row.close), reserve
    return pd.DataFrame(rows)


def period_metrics(ledger, capital, start, end):
    positions = np.flatnonzero(ledger.date.between(start,end))
    first = int(positions[0])
    initial = capital if first == 0 else float(ledger.equity.iloc[first-1])
    block = ledger.iloc[positions].copy()
    result = core.metrics(block, initial)
    result.pop("sharpe_252_diagnostic")
    result.pop("cagr_252_diagnostic")
    result["start_equity"] = initial
    return result


def run() -> None:
    if (OUT / "account_result.json").exists():
        raise RuntimeError("完整账户已运行，不改变信号重跑。")
    protocol = study.read(OUT / "account_protocol.json")
    for source in protocol["source_receipts"]:
        assert study.sha(ROOT/source["path"]) == source["sha256"]
    engine = engine_module()
    market = pd.read_parquet(study.DAILY)
    risk_source = pd.read_parquet(RISK_PATH)
    risk = market[["date"]].merge(risk_source[["date","es95"]],on="date",how="left",validate="one_to_one")
    dividends = engine.normalize_dividends(pd.read_csv(DIVIDENDS))
    opportunity = pd.read_csv(OUT / "全部固定高分机会.csv")
    summaries, all_cycles, checks = [], [], []
    for case in protocol["cases"]:
        selected = opportunity[opportunity.model.eq(case["model"])]
        ledger = simulate(market, risk, dividends, selected, case, engine)
        path = OUT / "accounts" / (case["id"]+".parquet")
        path.parent.mkdir(exist_ok=True)
        ledger.to_parquet(path,index=False)
        saved = pd.read_parquet(path)
        np.testing.assert_allclose(saved.equity, saved.cash+saved.shares*saved.mark+saved.dividend_receivable-saved.terminal_exit_reserve,rtol=0,atol=1e-7)
        assert saved.cash.ge(-1e-7).all() and saved.shares.ge(0).all() and saved.shares.mod(100).eq(0).all()
        buys = saved[saved.filled_quantity.gt(0)]
        if case["model"] in ["state","joint"]:
            assert set(buys.date).issubset(set(pd.to_datetime(selected.entry_date)))
            assert (buys.planned_ES_loss <= buys.es_budget+1e-8).all()
            assert (buys.planned_gap_loss <= buys.gap_budget+1e-8).all()
            assert saved.exposure.max() <= .5+1e-8
        assert buys.requested_quantity.eq(buys.filled_quantity).all()
        cycles = core.cycle_records(saved,case["capital"])
        for cycle in cycles.to_dict("records"):
            block = saved[saved.date.ge(cycle["entry"]) & saved.date.le(cycle["exit"] if cycle["closed"] else saved.date.iloc[-1])]
            cycle.update(case_id=case["id"],capital=case["capital"],model=case["model"],
                         paid_friction=float((block.commission+block.slippage_cost).sum()),
                         gross_same_holdings_profit=float(cycle["profit"]+block.commission.sum()+block.slippage_cost.sum()),
                         initial_quantity=int(block.filled_quantity.iloc[0]))
            all_cycles.append(cycle)
        if case["model"] in ["state","joint"]:
            assert cycles.closed.all() and int(saved.shares.iloc[-1]) == 0
            assert abs(float(cycles.profit.sum())-(float(saved.equity.iloc[-1])-case["capital"])) < 1e-6
        for period,start,end in [("完整期2021—2025","2021-01-01","2025-12-31"),("较早期2021—2023","2021-01-01","2023-12-31"),("主要期2024—2025","2024-01-01","2025-12-31")]:
            stats = period_metrics(saved,case["capital"],start,end)
            in_period = cycles[pd.to_datetime(cycles.entry).between(start,end) & cycles.closed]
            positive,negative = in_period[in_period.profit.gt(0)],in_period[in_period.profit.lt(0)]
            annual = {str(year):int((pd.to_datetime(in_period.entry).dt.year==year).sum()) for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1)}
            stats.update(case_id=case["id"],model=case["model"],capital=case["capital"],period=period,
                         completed_cycles=len(in_period),annual_complete_cycles=annual,
                         cycle_win_rate=len(positive)/len(in_period) if len(in_period) else None,
                         realized_cash_payoff=float(positive.profit.mean()/-negative.profit.mean()) if len(positive) and len(negative) else None,
                         largest_cycle_share_of_net_profit=float(in_period.profit.max()/stats["net_profit"]) if len(in_period) and stats["net_profit"]>0 else None,
                         all_years_at_least_five=all(n>=5 for n in annual.values()),
                         point_financial_targets=bool(stats["net_sharpe"] is not None and stats["net_sharpe"]>=1.2 and stats["cagr"]>=.1 and stats["max_drawdown"]<=.1),
                         ledger_path=str(path.relative_to(ROOT)).replace("\\","/"))
            summaries.append(stats)
        checks.append({"case_id":case["id"],"daily_rows":len(saved),"max_accounting_error":float(saved.accounting_error.abs().max()),
                       "max_close_exposure":float(saved.exposure.max()),"no_open_based_quantity_resizing":True,
                       "buy_fills":len(buys),"unfilled_orders":int((saved.requested_quantity.ne(0)&saved.filled_quantity.eq(0)).sum()),
                       "risk_reductions":int((saved.reason.eq("按已知预算减仓")&saved.filled_quantity.lt(0)).sum()),
                       "max_drawdown_stop_triggered":bool(saved.risk_stopped.any())})
        full = summaries[-3]
        print(f"账户{case['id']}：完整期净夏普{full['net_sharpe']}，年化{full['cagr']:.4%}，净利润{full['net_profit']:.2f}元。",flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "完整账户指标.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_cycles).to_csv(OUT / "全部账户自然交易周期.csv",index=False,encoding="utf-8-sig")
    study.save("account_result.json",{"completed_at":study.common.now(),"status":"COMPLETED_FIXED_RISK_ACCOUNTS",
                                      "accounts":summaries,"cycles":all_cycles,"checks":checks,
                                      "new_accounts":len(protocol["cases"]),"orders_authorized":False,"goal_achieved":False})
    print(pd.DataFrame(summaries)[["case_id","period","net_profit","net_sharpe","cagr","max_drawdown","completed_cycles"]].to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="固定月度高分机会的受约束账户")
    parser.add_argument("stage",choices=["prepare","run"])
    stage=parser.parse_args().stage
    {"prepare":prepare,"run":run}[stage]()
