"""按用户要求先做现有数据可行性初筛，统一规则比较十张因子卡。"""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as core
from scripts.record_factor96_remaining_changes_v1 import update

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_rapid_feasibility_v1"
PRICE_BASE = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs"
MARGIN_PATH = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/margin.parquet"
ENGINE_PATH = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/code/account_engine.py"
IDS = ["A03", "A06", "B05", "C03", "C05", "D01", "D02", "D03", "F03", "P01"]
START, END, HOLD = "2017-01-03", "2025-12-31", 10
REQUEST = "请尽快啊，抓大放小 先简单检验可行性，再对有成效的具体展开分析"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    core.save(path, value, exclusive=True)


def freeze():
    program = ROOT / "reports/research/510300_factor96_program_v1"
    status = read(program / "status.json")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    save(OUT / "authority_change.json", {"at": core.now(), "user_request": REQUEST,
        "previous_round": status["latest_round"], "previous_turn_classification": "PROGRESS_15_MIXED_TITLES_REVIEWED",
        "new_priority": "EXISTING_DATA_RAPID_FEASIBILITY_BEFORE_SOURCE_DETAIL",
        "deferred_work": "浪潮及其他公告的逐条约束补证；当前没有运行中的来源采集进程。",
        "original_goal_preserved": True, "orders_authorized": False, "delivery_package_required": False})
    cards = [r for r in read(program / "factor_progress.json") if r["id"] in IDS]
    save(OUT / "original_factor_cards.json", cards)
    save(OUT / "protocol.json", {"at": core.now(), "study_id": "510300_FACTOR96_RAPID_FEASIBILITY_V1",
        "classification": "EXPLORATORY_SCREEN_NOT_ACCEPTANCE", "user_request": REQUEST,
        "factor_cards": IDS, "operationalization": {
            "A03": "20日新高频率，另存5日相对前15日的频率加速。",
            "A06": "5日速度为单变量筛查；60日均价偏离另列，不把二者压成一分，不冒充原卡完整联合检验。",
            "B05": "最近3个负收益日价格冲击均值/此前3个负收益日均值；负日限定最近20交易日，另存回撤。",
            "C03": "10日正、负收益成交效率之差，乘同窗平均成交额归一。",
            "C05": "此前252日OLS解释对数成交额：绝对收益、20日波动和星期；当天残差。",
            "D01": "20日日内与隔夜对数收益差的标准化和，含现金分红并保持财富分解恒等式。",
            "D02": "负隔夜缺口由当日正收益吸收的比率；正隔夜日未知，另存缺口大小。",
            "D03": "5日平均收盘位置CLV，另存正CLV天数。",
            "F03": "融资5日变化对指数5日收益及20日波动的252个先前合格日OLS残差，整体后移1交易日。",
            "P01": "20日下行平方收益占比，另存总波动。"},
        "signals": "每项只做HIGH/LOW两方向，共20条；当前值分别>=此前两日历年70%分位或<=30%分位；至少120个历史有效值，两个分位相同时都不入场。",
        "threshold_clock": "分位数只含当前日前的数据；每日更新，无收益拟合、无参数网格。",
        "execution": "信号收盘形成、次日开盘买入；一次持有10个开盘间隔，持仓中忽略新信号；中途只按既有账户风险预算减仓。",
        "account": "使用既有原始价、现金分红、100份整手、T+1、涨跌停、佣金和滑点引擎；上限50%，既有ES和回撤储备约束不变；现金利息0，Sharpe相对0现金基准。",
        "risk_stop": "收盘账户回撤达到10%后下一个合法开盘退出，本次连续账户不恢复。",
        "costs": core.COSTS, "capital_CNY": [200000, 20000], "annual_days": 242,
        "benchmark": ["BUY_HOLD_50", "CASH"], "planned_account_scenarios": 88,
        "evaluation_start": START, "evaluation_end": END,
        "sample_reason": "统一使用已有风险和融资数据均覆盖的2017至2025年完整区间；2026不纳入本次初筛，未宣称它是独立未见验证集。",
        "period_diagnostics": ["2017-2020", "2021-2025"],
        "next_stage_filter": "20万元STRESS完整账户Sharpe>=0.8、CAGR>=5%、最大回撤<=10%、至少20次入场，两个固定子期净收益均为正。只决定是否值得细查，不是最终夏普1.2/CAGR10%达标。",
        "no_rescue": "所有20条及全部成本账户同表；不在初筛结果后换方向、窗口、阈值、持有期或拼接子期。",
        "source_work_rule": "先得到初筛结果，再对有成效的候选做针对性证据核验；保留全部旧失败。",
        "new_network_requests": 0, "orders_authorized": False, "delivery_package_required": False})
    paths = [OUT / "protocol.json", OUT / "original_factor_cards.json", Path(__file__), ENGINE_PATH,
             Path(core.__file__), PRICE_BASE / "market.parquet", PRICE_BASE / "price_features.parquet",
             PRICE_BASE / "dividends.csv", MARGIN_PATH]
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in paths]})
    mandate.update(latest_research_priority_instruction=REQUEST, research_priority="RAPID_FEASIBILITY_FIRST",
                   source_detail_work_deferred_by_user=True, latest_priority_receipt=str((OUT / "authority_change.json").relative_to(ROOT)),
                   goal_status="active", goal_achieved=False)
    update(mandate_path, mandate)
    status.update(research_priority="RAPID_FEASIBILITY_FIRST", source_detail_work_deferred_by_user=True,
                  next_independent_source_action="RUN_EXISTING_DATA_RAPID_FEASIBILITY_SCREEN")
    update(program / "status.json", status)
    print("已登记用户新优先级：10张因子卡、20个方向、88个统一账户，先检验可行性。", flush=True)


def prior_residual(x, y, dates):
    output = np.full(len(y), np.nan)
    for i in range(len(y)):
        if not np.isfinite(x[i]).all() or not np.isfinite(y[i]):
            continue
        lower = dates[i] - pd.DateOffset(years=2)
        eligible = np.flatnonzero((np.arange(i) >= dates.searchsorted(lower)) &
                                 np.isfinite(x[:i]).all(axis=1) & np.isfinite(y[:i]))[-252:]
        if len(eligible) < 126:
            continue
        beta = np.linalg.lstsq(x[eligible], y[eligible], rcond=None)[0]
        output[i] = y[i] - x[i] @ beta
    return output


def features(m, margin):
    dates = pd.DatetimeIndex(m.date)
    r = np.log((m.close + m.dividend) / m.close.shift())
    w = np.exp(r.fillna(0).cumsum()); rv = r.rolling(20).std(ddof=1)
    f = pd.DataFrame({"date": m.date, "wealth": w, "rv20": rv, "drawdown20": w / w.rolling(20).max() - 1})
    hit = w.gt(w.shift().rolling(20).max()).where(w.shift().rolling(20).count().eq(20))
    f["A03"] = hit.rolling(20).mean()
    f["A03_acceleration"] = hit.rolling(5).mean() - hit.shift(5).rolling(15).mean()
    f["A06"] = r.rolling(5).sum() / (rv * np.sqrt(5))
    f["A06_distance60"] = np.log(w / w.rolling(60).mean()) / (rv * np.sqrt(60))
    relative_amount = m.amount / m.amount.shift().rolling(20).median()
    impact = r.abs() / relative_amount.where(relative_amount.gt(0))
    decline = np.full(len(m), np.nan)
    for i in range(20, len(m)):
        idx = np.flatnonzero((r.iloc[max(0, i-19):i+1] < 0).to_numpy()) + max(0, i-19)
        if len(idx) >= 6:
            values = impact.iloc[idx[-6:]].to_numpy(float)
            if np.isfinite(values).all() and values[:3].mean() > 0:
                decline[i] = values[3:].mean() / values[:3].mean()
    f["B05"] = decline
    positive, negative = r.gt(0), r.lt(0)
    up = r.where(positive, 0).rolling(10).sum() / m.amount.where(positive, 0).rolling(10).sum().replace(0, np.nan)
    down = (-r).where(negative, 0).rolling(10).sum() / m.amount.where(negative, 0).rolling(10).sum().replace(0, np.nan)
    f["C03"] = (up - down) * m.amount.rolling(10).mean()
    weekdays = np.column_stack([(dates.dayofweek == k).astype(float) for k in range(1, 5)])
    x = np.column_stack([np.ones(len(m)), r.abs(), rv, weekdays])
    f["C05"] = prior_residual(x, np.log(m.amount.where(m.amount.gt(0))).to_numpy(), dates)
    intra = np.log((m.close + m.dividend) / (m.open + m.dividend))
    overnight = np.log((m.open + m.dividend) / m.close.shift())
    assert np.nanmax(np.abs(intra + overnight - r)) < 1e-12
    spread = intra - overnight
    f["D01"] = spread.rolling(20).sum() / (spread.rolling(20).std(ddof=1) * np.sqrt(20))
    f["D02"] = (intra.clip(lower=0) / overnight.abs()).where(overnight.lt(-1e-12))
    f["D02_gap"] = overnight
    clv = (2*m.close - m.high - m.low) / (m.high - m.low).replace(0, np.nan)
    f["D03"] = clv.rolling(5).mean(); f["D03_positive_days"] = clv.gt(0).astype(float).where(clv.notna()).rolling(5).sum()
    f["P01"] = r.clip(upper=0).pow(2).rolling(20).sum() / r.pow(2).rolling(20).sum().replace(0, np.nan)
    bal = m[["date"]].merge(margin[["date", "market_rzye"]], on="date", how="left", validate="one_to_one").market_rzye
    change = (bal / bal.shift(5) - 1).where(bal.rolling(6).count().eq(6))
    financing_x = np.column_stack([np.ones(len(m)), r.rolling(5).sum(), rv])
    f["F03"] = pd.Series(prior_residual(financing_x, change.to_numpy(), dates)).shift(1)
    for key in IDS:
        raw = f[key].to_numpy(float); low = np.full(len(m), np.nan); high = low.copy(); count = np.zeros(len(m), dtype=int)
        for i, day in enumerate(dates):
            a = dates.searchsorted(day - pd.DateOffset(years=2))
            history = raw[a:i]; history = history[np.isfinite(history)]; count[i] = len(history)
            if len(history) >= 120:
                low[i], high[i] = np.quantile(history, [.3, .7])
        known = np.isfinite(raw) & np.isfinite(low) & np.isfinite(high) & (high > low)
        f[key + "_LOW"] = known & (raw <= low); f[key + "_HIGH"] = known & (raw >= high)
        f[key + "_q30"] = low; f[key + "_q70"] = high; f[key + "_history_count"] = count
    return f


def simulate(m, f, dividends, policy, capital, cost_name, engine):
    account = engine.Account(float(capital)); cost = core.COSTS[cost_name]
    config = {"lot": 100, "tick": .001, "limit_fraction": .1}
    indices = np.flatnonzero(m.date.between(START, END)); first, last = int(indices[0]), int(indices[-1])
    previous_equity, previous_mark, reserve_before, peak = float(capital), float(m.close.iloc[first-1]), 0., float(capital)
    entry, ever_bought, stopped, pending_exit = None, False, False, False
    ledger = []; trades = []; events = dividends.to_dict("records")
    rows = list(m.itertuples(index=False)); es = f.es95.to_numpy(float)
    signals = np.zeros(len(m), dtype=bool) if policy in ("BUY_HOLD_50", "CASH") else f[policy].to_numpy(bool)
    for i in indices:
        row = rows[i]; day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[k] = amount; recognized += amount
            if event["payment_date"] < day and k in account.receivables:
                paid_now = account.receivables.pop(k); account.cash += paid_now; paid += paid_now
        quantity, reason = 0, "现金或持有"
        reference = float(rows[i-1].close - row.dividend)
        if policy == "BUY_HOLD_50":
            if not ever_bought:
                quantity = math.floor(.5 * account.value(reference) / reference / 100) * 100
                reason = "半仓买入持有基准"
        elif policy != "CASH":
            if old:
                pending_exit |= stopped or i >= entry + HOLD
                if pending_exit:
                    quantity, reason = -old, "回撤退出" if stopped else "十个开盘间隔到期"
                else:
                    target = min(old, core.target_quantity(engine, account, reference, peak, es[i-1]))
                    quantity, reason = target - old, "风险预算减仓"
            elif not stopped and signals[i-1]:
                quantity = core.target_quantity(engine, account, reference, peak, es[i-1])
                quantity = min(quantity, core.target_quantity(engine, account, op, peak, es[i-1]))
                reason = "前收盘信号入场"
        execution = engine.execute_order(account, quantity, op, float(row.previous_close), float(row.dividend), int(i), cost, config)
        if execution["filled_quantity"] > 0 and old == 0:
            entry, ever_bought, pending_exit = int(i), True, False
        if quantity:
            trades.append({"date": day, "idx": int(i), "signal_idx": int(i)-1, "reason": reason, **execution})
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                paid_now = account.receivables.pop(k); account.cash += paid_now; paid += paid_now
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            exit_price = engine.fill_price(close, -1, core.COSTS["STRESS"], .001)
            reserve = account.shares * (close-exit_price) + engine.commission(account.shares, exit_price, core.COSTS["STRESS"])
        equity = account.value(close) - reserve
        price_pnl = old*(op-previous_mark) + account.shares*(close-op)
        error = equity-previous_equity-price_pnl-recognized + execution["commission"] + execution["slippage_cost"] + reserve-reserve_before
        assert abs(error) < 1e-6, (policy, day, error)
        account.assert_valid(); peak = max(peak, equity); dd = 1-equity/peak
        if policy not in ("BUY_HOLD_50", "CASH"):
            stopped |= dd >= .1
        ledger.append({"date": day, "idx": int(i), "equity": equity, "net_return": equity/previous_equity-1,
            "shares": account.shares, "cash": account.cash, "dividend_receivable": account.receivable(),
            "dividend_recognized": recognized, "dividend_paid": paid, "mark": close,
            "exposure": account.shares*close/equity, "drawdown": dd, "risk_stopped": stopped,
            "terminal_exit_reserve": reserve, "accounting_error": error, "reason": reason, **execution})
        if not account.shares:
            entry, pending_exit = None, False
        previous_equity, previous_mark, reserve_before = equity, close, reserve
    return pd.DataFrame(ledger), pd.DataFrame(trades)


def period_metrics(ledger, start, end):
    part = ledger.loc[ledger.date.between(start, end)]
    if part.empty:
        return {"net_sharpe": None, "cagr": None, "net_profit_fraction": None}
    r = part.net_return.to_numpy(float); total = float(np.prod(1+r))
    return {"net_sharpe": float(np.sqrt(242)*r.mean()/r.std(ddof=1)) if r.std(ddof=1) > 1e-15 else None,
            "cagr": total**(242/len(r))-1, "net_profit_fraction": total-1}


def run():
    for f in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(f["path"])) == f["sha256"]
    save(OUT / "run_started.json", {"at": core.now(), "expected_account_scenarios": 88})
    m = pd.read_parquet(PRICE_BASE / "market.parquet")
    m.date = pd.to_datetime(m.date); m = m.loc[m.date.le(END)].reset_index(drop=True)
    risk = pd.read_parquet(PRICE_BASE / "price_features.parquet")
    risk.date = pd.to_datetime(risk.date)
    assert m.date.equals(risk.date) and len(m) == 3307
    assert set(m.symbol) == {"510300.SH"} and m.date.is_unique and m.date.is_monotonic_increasing
    dividends = core.normalize_dividends(pd.read_csv(PRICE_BASE / "dividends.csv"))
    margin = pd.read_parquet(MARGIN_PATH); margin.date = pd.to_datetime(margin.date)
    f = features(m, margin); f["es95"] = risk.es95.to_numpy()
    # 切掉未来数据重新计算一个历史截点，核对原始因子与信号不依赖未来。
    cut = int(np.flatnonzero(m.date.ge("2021-01-04"))[0])
    prefix = features(m.iloc[:cut+1].copy(), margin.loc[margin.date.le(m.date.iloc[cut])].copy())
    pd.testing.assert_frame_equal(f.drop(columns="es95").iloc[:cut+1], prefix, check_exact=False, rtol=1e-10, atol=1e-12)
    f.to_parquet(OUT / "screen_features.parquet", index=False)
    spec = importlib.util.spec_from_file_location("rapid_frozen_account_engine", ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec); sys.modules[spec.name] = engine; spec.loader.exec_module(engine)
    policies = [key + "_" + side for key in IDS for side in ("HIGH", "LOW")] + ["BUY_HOLD_50", "CASH"]
    records = []; yearly = []
    for capital in (200000, 20000):
        for cost in ("BASE", "STRESS"):
            for policy in policies:
                ledger, orders = simulate(m, f, dividends, policy, capital, cost, engine)
                directory = OUT / "accounts" / f"{capital}_{cost}_{policy}"; directory.mkdir(parents=True)
                ledger.to_parquet(directory / "ledger.parquet", index=False)
                orders.to_parquet(directory / "orders.parquet", index=False)
                metrics = core.metrics(ledger, capital)
                cycles = core.cycle_records(ledger, capital)
                metrics.update(policy=policy, capital=capital, cost=cost,
                    entries=int(((ledger.filled_quantity > 0) & (ledger.shares_before == 0)).sum()),
                    closed_cycles=int(cycles.closed.sum()),
                    positive_cycles=int((cycles.closed & cycles.profit.gt(0)).sum()),
                    stopped_at=ledger.loc[ledger.risk_stopped, "date"].min().isoformat() if ledger.risk_stopped.any() else None,
                    ledger_path=str((directory / "ledger.parquet").relative_to(ROOT)))
                for label, a, b in (("early", START, "2020-12-31"), ("late", "2021-01-01", END)):
                    metrics.update({label + "_" + k: v for k, v in period_metrics(ledger, a, b).items()})
                records.append(metrics)
                for year in range(2017, 2026):
                    yearly.append({"policy": policy, "capital": capital, "cost": cost, "year": year,
                                   **period_metrics(ledger, f"{year}-01-01", f"{year}-12-31")})
            print(f"已完成{capital}元、{cost}成本的22个账户。", flush=True)
    table = pd.DataFrame(records)
    table["joint_historical_target"] = (table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1))
    table["worth_followup"] = (table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1) & table.entries.ge(20)
                               & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0))
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table[(table.capital == 200000) & (table.cost == "STRESS") & ~table.policy.isin(["BUY_HOLD_50", "CASH"])].sort_values("net_sharpe", ascending=False)
    primary.to_csv(OUT / "初筛排序.csv", index=False, encoding="utf-8-sig")
    result = {"at": core.now(), "study_id": "510300_FACTOR96_RAPID_FEASIBILITY_V1", "classification": "EXPLORATORY_SCREEN_NOT_ACCEPTANCE",
        "factor_cards": 10, "direction_candidates": 20, "account_scenarios": len(table), "new_network_requests": 0,
        "sample": [START, END], "primary_top5": primary.head(5).to_dict("records"),
        "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(),
        "joint_historical_target_count": int(primary.joint_historical_target.sum()),
        "qualified_candidates": 0, "independent_forward_observations": 0, "goal_achieved": False,
        "causal_prefix_check": "PASS", "accounting_identity_max_error": float(table.max_identity_error.max()),
        "orders_authorized": False, "delivery_package_created": False,
        "selection_warning": "20方向多重探索；此前历史已被多轮研究，分位规则因果计算不等于独立前向验证。"}
    save(OUT / "result.json", result)
    print(json.dumps(core.clean({"账户数": len(table), "最佳压力成本候选": result["primary_top5"][0],
                                  "值得展开": result["worth_followup"], "联合达标数": result["joint_historical_target_count"]}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args(); freeze() if args.stage == "freeze" else run()
