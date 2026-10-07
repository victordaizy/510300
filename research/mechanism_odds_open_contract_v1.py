"""以收盘财富为锚的事前开盘价格合同；固定研究版本，无交易连接。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_mechanism_odds_open_contract_v1"
PARENT = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1"
COSTS = {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005},
         "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}
POLICIES = ("LIMIT_VALUE", "OPEN_VALUE", "LIMIT_FIXED", "BUY_HOLD_50", "CASH")
STATE = ["pressure5", "trend20", "log_rv5_rv60"]
START, END = "2017-01-03", "2025-12-31"
K, MIN_TRAIN, H = 126, 252, 5
TICK, LOT = .001, 100


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat() if pd.notna(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def engine(path=None):
    path = path or OUT / "code/account_engine.py"
    spec = importlib.util.spec_from_file_location("odds_frozen_account_engine", path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def check_freeze():
    for row in read(OUT / "freeze.json")["files"]:
        if digest(OUT / row["path"]) != row["sha256"]:
            raise ValueError("冻结文件发生变化：" + row["path"])


def freeze():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("本版本已冻结，不覆盖或重新登记。")
    registration = read(OUT / "design_registration.json")
    tests = read(OUT / "prefreeze_tests.json")
    if tests["exit_code"] != 0:
        raise ValueError("必要实现测试尚未通过。")
    files = {
        "inputs/market.parquet": PARENT / "inputs/market.parquet",
        "inputs/dividends.csv": PARENT / "inputs/dividends.csv",
        "inputs/dividend_coverage.json": PARENT / "inputs/dividend_coverage.json",
        "inputs/authority_snapshot.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "code/account_engine.py": PARENT / "code/account_engine.py",
        "code/mechanism_odds_open_contract_v1.py": Path(__file__),
        "code/test_mechanism_odds_open_contract_v1.py": ROOT / "tests/test_mechanism_odds_open_contract_v1.py",
        "references/用户框架.md": Path(r"C:\Users\戴周阳\Downloads\510300_机制与赔率交易框架_V1_20260928.md"),
        "references/用户框架.html": Path(r"C:\Users\戴周阳\Downloads\510300_机制与赔率交易框架_V1_20260928 (1).html"),
        "references/用户粘贴框架.txt": Path(r"E:\CodexData\.codex\attachments\b4169768-6acd-4684-af9a-525edaed831b\pasted-text-1.txt"),
        "references/旧瓶颈结论.md": ROOT / "reports/research/510300_factor96_bottleneck_diagnostic_v1/瓶颈诊断结论.md",
        "references/旧日更结论.md": ROOT / "reports/research/510300_factor96_daily_state_shrink_v1/研究结论.md",
    }
    source_records = []
    for relative, source in files.items():
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_records.append({"source": str(source), "copy": relative, "sha256": digest(target)})
    authority = read(OUT / "inputs/authority_snapshot.json")
    assert authority["orders_authorized"] is False
    assert authority["executable_assets"] == ["510300.SH", "CASH_CNY"]
    protocol = {**registration, "frozen_at": now(), "account_scenarios": 20,
        "implementation": {
            "entry_quantity": "在收盘计划中按压力费用、参考价与限价较高者和冻结风险约束决定整手股数；开盘不重算模型或风险股数。",
            "limit_boundary": "先对126个未来清算场景逐个扣压力卖出滑点、价位及佣金，再反解买入佣金后的最高价格，向下取整后再保留一档。",
            "value_exit": "剩余期限场景的期望清算现金小于或等于下一开盘立即清算的前收盘参考现金时退出；共同的次日分红从两边扣去。",
            "tail_estimate": "126个已成熟五日场景最差ceil(5%*126)个的平均负收益，最低0；重叠标签不是独立126次机会。",
            "buy_gap": "事前限价不能在看到低开后按新的失效解释撤销，符合限价且方向未封板的低开接受模拟成交。",
            "auction_proxy": "仅模拟下一次开盘集合竞价，未成交当日到期；不利用全日高低或成交量。无盘口、排队及开盘容量证据，属于日线成交近似。",
            "unknown": "缺少分布不新买；已有持仓仅按风险和原期限管理，不把缺失当作负收益预测。",
            "terminal": "末日按市值估值，同时扣本账户对应成本的清算准备；不是实际末日成交。",
            "forward_status": "NO_VIEW_HISTORICAL_EXPERIMENT_ONLY",
            "cost_scenarios": "所有计划均用STRESS估价；BASE/STRESS只改变实际模拟成交成本，不选择有利信号。",
            "source_boundary": "历史回取数据；分红公告日期从已保存官方来源链接提取并检查早于登记日。不声称首见快照或真实成交。",
            "uncertainty": "固定种子20260928，20日循环区块2000次；给定策略历史收益的不确定性，不校正项目全部选择偏差。"},
        "prior_trials": "本项目已做多轮价格、近邻分布、五日持有与剩余价值研究。本轮非全新信息，不重置试验次数。",
        "source_copies": source_records}
    save(OUT / "protocol.json", protocol)
    frozen = list(files) + ["design_registration.json", "protocol.json", "prefreeze_tests.json"]
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": x, "sha256": digest(OUT / x)} for x in frozen]})
    print("一个主方案、两个必要对照及20个账户情景已冻结，未读取新结果。", flush=True)


def features(market):
    r = np.log((market.close + market.dividend) / market.close.shift())
    rv20 = r.rolling(20).std(ddof=1)
    return pd.DataFrame({"date": market.date,
        "pressure5": r.rolling(5).sum() / (rv20 * np.sqrt(5)),
        "trend20": r.rolling(20).sum() / (rv20 * np.sqrt(20)),
        "log_rv5_rv60": np.log(r.rolling(5).std(ddof=1) / r.rolling(60).std(ddof=1))})


def labels(market):
    """收盘持有份额至未来开盘的财富；只有退出时点过去后才可用于训练。"""
    n = len(market)
    out = np.full((n, H), np.nan)
    coupon = np.r_[0., np.cumsum(market.dividend.to_numpy(float))]
    op, close = market.open.to_numpy(float), market.close.to_numpy(float)
    for h in range(1, H + 1):
        origin = np.arange(n - h - 1)
        end = origin + h + 1
        earned = coupon[end + 1] - coupon[origin + 1]
        out[origin, h - 1] = (op[end] + earned) / close[origin] - 1
    return out


def forecast_one(market, x, y, i):
    dates = pd.DatetimeIndex(market.date)
    first = int(dates.searchsorted(dates[i] - pd.DateOffset(years=2)))
    # 最长标签退出为原点+6；当前日开盘已经发生，可以进入盘后训练。
    ids = np.arange(first, max(first, i - H))
    known = np.isfinite(x[STATE].to_numpy(float)[ids]).all(axis=1)
    known &= np.isfinite(y[ids]).all(axis=1)
    ids = ids[known]
    if len(ids) < MIN_TRAIN or not np.isfinite(x.loc[i, STATE].to_numpy(float)).all():
        return None
    z = x.loc[ids, STATE].to_numpy(float)
    center, scale = z.mean(axis=0), z.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1.
    z = np.clip((z - center) / scale, -5, 5)
    query = np.clip((x.loc[i, STATE].to_numpy(float) - center) / scale, -5, 5)
    distance = np.sum((z - query) ** 2, axis=1)
    chosen = ids[np.lexsort((ids, distance))[:K]]
    return chosen, {"idx": i, "date": dates[i], "training_start_idx": first,
        "training_n": len(ids), "latest_maturity_idx": int(ids.max() + H + 1),
        "neighbor_latest_maturity_idx": int(chosen.max() + H + 1),
        "mean": center.tolist(), "scale": scale.tolist(), "neighbors": len(chosen)}


def forecasts(market, x, y):
    indices = np.full((len(market), K), -1, dtype=np.int32)
    records = []
    for i in range(len(market)):
        if str(market.date.iloc[i].date()) < START:
            continue
        result = forecast_one(market, x, y, i)
        if result is None:
            records.append({"idx": i, "date": market.date.iloc[i], "status": "NO_VIEW_TRAINING"})
            continue
        ids, record = result
        indices[i] = ids
        records.append({**record, "status": "AVAILABLE", "mean_h5": float(y[ids, 4].mean()),
                        "probability_h5_positive": float((y[ids, 4] > 0).mean())})
    return indices, records


def liquidation_values(e, values, quantity):
    """含分红总财富统一折价是保守代理；真实会计中分红单独入账。"""
    c = COSTS["STRESS"]
    price = np.floor((np.maximum(values, TICK) * (1-c["slippage"])) / TICK + 1e-10) * TICK
    return price - np.maximum(quantity * price * c["commission"], c["minimum"]) / quantity


def risk_quantity(e, nav, cash, peak, reference, es, current=0):
    if not np.isfinite([nav, cash, peak, reference, es]).all() or reference <= 0:
        return 0
    remaining = max(0., nav - .9 * peak)
    cap = min(.5, .025 / max(es, 1e-12), .5 * remaining / (.1 * nav))
    q = math.floor(max(0, cap) * nav / reference / LOT) * LOT
    c = COSTS["STRESS"]
    while q > 0:
        change = q - current
        px = e.fill_price(reference, 1 if change > 0 else -1, c, TICK)
        friction = abs(change) * abs(px-reference) + e.commission(change, px, c)
        sell = e.fill_price(reference, -1, c, TICK)
        reserve = q * (reference-sell) + e.commission(q, sell, c)
        enough_cash = change <= 0 or change * px + e.commission(change, px, c) <= cash + 1e-8
        if (enough_cash and q * reference * es + friction + reserve <= .025 * nav + 1e-8
                and q * reference * .1 + friction + reserve <= min(.05*nav, .5*remaining) + 1e-8
                and q * reference <= .5*(nav-friction) + 1e-8):
            return q
        q -= LOT
    return 0


def buy_plan(e, nav, cash, peak, close, next_dividend, scenarios, es, policy):
    reference = close - next_dividend
    terminal = close * (1 + scenarios) - next_dividend
    q = risk_quantity(e, nav, cash, peak, reference, es)
    c = COSTS["STRESS"]
    while q > 0:
        terminal_net = float(liquidation_values(e, terminal, q).mean())
        limit = min(terminal_net / (1+c["commission"]), terminal_net-c["minimum"]/q)
        limit = math.floor(limit/TICK + 1e-10)*TICK - TICK
        if limit <= 0 or (policy == "OPEN_VALUE" and e.fill_price(reference, 1, c, TICK) > limit):
            return {"quantity": 0, "reason": "参考价格不满足净优势", "limit_price": None}
        # 同一股数先同时满足前收盘与最高可能成交报价的风险预算。
        risk_ref = max(reference, limit) if policy != "OPEN_VALUE" else reference
        allowed = risk_quantity(e, nav, cash, peak, risk_ref, es)
        if q <= allowed:
            net_scenarios = liquidation_values(e, terminal, q) - limit - max(limit*c["commission"], c["minimum"]/q)
            return {"quantity": q, "reason": "事前限价买入" if policy != "OPEN_VALUE" else "事前承诺开盘买入",
                "limit_price": limit if policy != "OPEN_VALUE" else None,
                "estimated_max_buy": limit, "expected_terminal_net_per_share": terminal_net,
                "expected_net_at_limit": float(net_scenarios.mean()),
                "win_probability_at_limit": float((net_scenarios > 0).mean()),
                "reference_price": reference, "es95": es}
        q = min(q-LOT, allowed)
    return {"quantity": 0, "reason": "整手或风险预算不足", "limit_price": None}


def execute(e, account, plan, row, idx, cost):
    cfg = {"lot": LOT, "tick": TICK, "limit_fraction": .1}
    q = int(plan["quantity"])
    limit = plan.get("limit_price")
    if q > 0 and limit is not None and e.fill_price(row.open, 1, cost, TICK) > limit + 1e-10:
        event = e.execute_order(account, 0, row.open, row.previous_close, row.dividend, idx, cost, cfg)
        event.update(requested_quantity=q, status="UNFILLED_PRECOMMITTED_LIMIT")
        return event
    return e.execute_order(account, q, row.open, row.previous_close, row.dividend, idx, cost, cfg)


def simulate(market, dividends, y, neighbors, policy, capital, cost_name, e):
    account = e.Account(float(capital))
    cost = COSTS[cost_name]
    ids = np.flatnonzero(market.date.between(START, END))
    first, last = int(ids[0]), int(ids[-1])
    events = dividends.to_dict("records")
    rows = list(market.itertuples(index=False))
    peak, previous_equity = float(capital), float(capital)
    previous_mark, previous_reserve = float(rows[first-1].close), 0.
    expiry, stopped, exit_pending, entered = None, False, False, False
    plan = {"quantity": 0, "limit_price": None, "reason": "首日尚无本账户前收盘计划", "decision_idx": first-1}
    ledger, orders, decisions = [], [], []
    for i in ids:
        row, day = rows[i], rows[i].date
        old = account.shares
        sellable_before = account.sellable(int(i))
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[k] = amount
                    recognized += amount
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        assert plan["decision_idx"] < i
        fill = execute(e, account, plan, row, int(i), cost)
        assert fill["filled_quantity"] >= -sellable_before
        if fill["filled_quantity"] > 0 and old == 0:
            expiry, entered = int(i + H), True
        if plan["quantity"]:
            orders.append({"date": day, "idx": int(i), **plan, **fill})
        if account.shares == 0:
            expiry, exit_pending = None, False
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = e.fill_price(row.close, -1, cost, TICK)
            reserve = account.shares*(row.close-px) + e.commission(account.shares, px, cost)
        equity = account.value(row.close)-reserve
        price_pnl = old*(row.open-previous_mark)+account.shares*(row.close-row.open)
        error = equity-previous_equity-price_pnl-recognized+fill["commission"]+fill["slippage_cost"]+reserve-previous_reserve
        assert abs(error) < 1e-6, (day, error)
        account.assert_valid()
        peak = max(peak, equity)
        dd = 1-equity/peak
        if policy not in ("BUY_HOLD_50", "CASH"):
            stopped |= dd >= .1
        ledger.append({"date": day, "idx": int(i), "equity": equity, "net_return": equity/previous_equity-1,
            "shares": account.shares, "cash": account.cash, "dividend_receivable": account.receivable(),
            "dividend_recognized": recognized, "dividend_paid": paid, "mark": row.close,
            "exposure": account.shares*row.close/equity, "drawdown": dd, "risk_stopped": stopped,
            "terminal_exit_reserve": reserve, "accounting_error": error, "decision_idx": plan["decision_idx"],
            "sellable_before": sellable_before, "locked_after": account.shares-account.sellable(int(i)), **fill})
        previous_equity, previous_mark, previous_reserve = equity, row.close, reserve
        if i == last:
            continue
        next_div = float(rows[i+1].dividend)
        reference = float(row.close-next_div)
        nids = neighbors[i]
        available = bool(nids[0] >= 0)
        plan = {"quantity": 0, "limit_price": None, "reason": "现金或持有", "decision_idx": int(i),
                "decision_date": day, "expiry_idx": expiry, "forecast_available": available}
        if policy == "BUY_HOLD_50":
            if not entered:
                plan.update(quantity=math.floor(.5*equity/reference/LOT)*LOT, reason="半仓买入持有对照")
        elif policy != "CASH":
            if available:
                samples = y[nids, 4]
                es = max(0., -float(np.sort(samples)[:math.ceil(.05*K)].mean()))
            else:
                es = np.nan
            if account.shares:
                exit_pending |= stopped or (i+1 >= expiry)
                if not exit_pending and policy != "LIMIT_FIXED" and available:
                    remaining = expiry-(i+1)
                    terminal = row.close*(1+y[nids, remaining-1])-next_div
                    future_net = float(liquidation_values(e, terminal, account.shares).mean())
                    immediate = e.fill_price(reference, -1, COSTS["STRESS"], TICK)
                    immediate -= e.commission(account.shares, immediate, COSTS["STRESS"])/account.shares
                    plan.update(remaining_horizon=remaining, expected_hold_per_share=future_net,
                                immediate_sell_reference_per_share=immediate)
                    exit_pending = future_net <= immediate
                if exit_pending:
                    plan.update(quantity=-account.shares, reason="回撤或期限或剩余价值退出")
                elif available:
                    target = min(account.shares, risk_quantity(e, equity, account.cash, peak, reference, es, account.shares))
                    plan.update(quantity=target-account.shares, reason="事前风险预算减仓" if target < account.shares else "按剩余价值继续持有")
            elif not stopped and available:
                plan.update(buy_plan(e, equity, account.cash, peak, row.close, next_div, y[nids, 4], es, policy))
            else:
                plan["reason"] = "回撤已停止" if stopped else "NO_VIEW：成熟样本不足"
        decisions.append(deepcopy(plan))
    return pd.DataFrame(ledger), pd.DataFrame(orders), pd.DataFrame(decisions)


def metrics(ledger, capital):
    r = ledger.net_return.to_numpy(float)
    equity = np.r_[float(capital), ledger.equity.to_numpy(float)]
    np.testing.assert_allclose(r, equity[1:]/equity[:-1]-1, rtol=0, atol=1e-12)
    std = r.std(ddof=1)
    total = equity[-1]/capital
    days = (pd.Timestamp(ledger.date.iloc[-1])-pd.Timestamp(ledger.date.iloc[0])).days+1
    centered = r-r.mean()
    gamma0 = float(np.mean(centered**2))
    lrv = gamma0+2*sum((1-k/21)*float(np.mean(centered[k:]*centered[:-k])) for k in range(1, 21))
    return {"days": len(r), "end_equity": equity[-1], "net_profit": equity[-1]-capital,
        "net_sharpe": np.sqrt(242)*r.mean()/std if std > 1e-15 else None,
        "sharpe_252_diagnostic": np.sqrt(252)*r.mean()/std if std > 1e-15 else None,
        "hac20_sharpe_diagnostic": np.sqrt(242)*r.mean()/np.sqrt(lrv) if lrv > 1e-20 else None,
        "calendar_cagr": total**(365.2425/days)-1, "legacy_cagr_242": total**(242/len(r))-1,
        "max_drawdown": float(1-np.min(equity/np.maximum.accumulate(equity))),
        "mean_exposure": ledger.exposure.mean(), "maximum_close_exposure": ledger.exposure.max(),
        "entries": int(((ledger.filled_quantity > 0)&(ledger.shares_before == 0)).sum()),
        "fills": int(ledger.filled_quantity.ne(0).sum()), "commission": ledger.commission.sum(),
        "slippage": ledger.slippage_cost.sum(), "worst_day": r.min(),
        "turnover_initial_capital": ledger.notional.sum()/capital,
        "terminal_exit_reserve": ledger.terminal_exit_reserve.iloc[-1],
        "maximum_identity_error": ledger.accounting_error.abs().max(),
        "risk_stopped": bool(ledger.risk_stopped.any())}


def cycles(ledger):
    records, active = [], None
    for row in ledger.itertuples():
        if row.filled_quantity > 0 and row.shares_before == 0:
            active = {"entry": row.date, "pnl": 0., "fees": 0.}
        if active is not None:
            active["pnl"] -= row.filled_quantity * (row.fill_price if pd.notna(row.fill_price) else 0.) + row.commission
            active["pnl"] += row.dividend_recognized
            active["fees"] += row.commission+row.slippage_cost
            if row.shares == 0:
                records.append({**active, "exit": row.date, "closed": True})
                active = None
    if active is not None:
        last = ledger.iloc[-1]
        active["pnl"] += last.shares*last.mark-last.terminal_exit_reserve
        records.append({**active, "exit": None, "closed": False})
    return pd.DataFrame(records, columns=["entry", "exit", "pnl", "fees", "closed"])


def load_data():
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le(END)].reset_index(drop=True)
    required = ["open", "high", "low", "close", "previous_close", "dividend"]
    assert market.date.is_unique and market.date.is_monotonic_increasing
    assert set(market.symbol) == {"510300.SH"}
    assert np.isfinite(market[[c for c in required if c != "previous_close"]].to_numpy()).all()
    assert np.isfinite(market.previous_close.iloc[1:]).all()
    np.testing.assert_allclose(market.previous_close.iloc[1:], market.close.iloc[:-1], rtol=0, atol=1e-12)
    assert (market[["open", "high", "low", "close"]] > 0).all().all()
    e = engine()
    dividends = e.normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    announcements = []
    for event in dividends.to_dict("records"):
        if event["ex_date"] > pd.Timestamp(END):
            continue
        match = re.search(r"/(\d{4}-\d{2}-\d{2})/", event["source"])
        if match is None:
            match = re.search(r"/c/c_(\d{8})_", event["source"])
        if not match or pd.Timestamp(match.group(1)) > event["record_date"]:
            raise ValueError("分红公告来源时点不足，不能预先调整订单。")
        announcements.append({"ex_date": event["ex_date"], "announcement_date_from_source_url": match.group(1), "source": event["source"]})
    expected = market.date.map(dividends.set_index("ex_date").cash_dividend_per_share).fillna(0.)
    np.testing.assert_allclose(market.dividend, expected, rtol=0, atol=1e-12)
    return market, dividends, announcements


def uncertainty(ledgers):
    rng = np.random.default_rng(20260928)
    n = len(ledgers["LIMIT_VALUE"])
    starts = rng.integers(0, n, size=(2000, math.ceil(n/20)))
    idx = ((starts[:, :, None]+np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=idx.astype(np.int32))
    r = ledgers["LIMIT_VALUE"].net_return.to_numpy(float)
    draws = r[idx]
    sd = draws.std(axis=1, ddof=1)
    sharpe = np.divide(np.sqrt(242)*draws.mean(axis=1), sd, out=np.full(2000, np.nan), where=sd > 1e-15)
    result = {"conditional_historical_sharpe_95_interval": np.nanquantile(sharpe, [.025, .975]).tolist()}
    for name in ("OPEN_VALUE", "LIMIT_FIXED", "BUY_HOLD_50"):
        excess = r-ledgers[name].net_return.to_numpy(float)
        distribution = excess[idx].mean(axis=1)*242
        result[name] = {"annualized_arithmetic_return_difference": excess.mean()*242,
                       "paired_block_95_interval": np.quantile(distribution, [.025, .975]).tolist()}
    result["boundary"] = "未校正全部历史试验选择偏差，不是未来成功概率。"
    return result


def render_report(table, interval, prediction, source_count):
    primary = table[(table.capital == 200000)&(table.cost == "STRESS")]
    p = primary[primary.policy == "LIMIT_VALUE"].iloc[0]
    names = {"LIMIT_VALUE": "主方案：事前限价＋剩余价值", "OPEN_VALUE": "对照：事前承诺开盘＋剩余价值",
             "LIMIT_FIXED": "对照：事前限价＋固定期限", "BUY_HOLD_50": "半仓买入持有", "CASH": "现金"}
    rows = ["|方案|净夏普|日历年化|最大回撤|入场次数|", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        s = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        rows.append(f"|{names[row.policy]}|{s}|{row.calendar_cagr:.2%}|{row.max_drawdown:.2%}|{row.entries}|")
    small = table[(table.capital == 20000)&(table.cost == "STRESS")&(table.policy == "LIMIT_VALUE")].iloc[0]
    decision = "历史点值通过，但尚无独立验证，目标仍未完成" if p.joint_historical_target else "未达到目标；冻结版本停止调参，目标尚未实现"
    return f"""本轮{decision}。20万元主方案在2017—2025年压力成本下，净夏普{p.net_sharpe:.3f}，日历复合年化{p.calendar_cagr:.2%}，最大回撤{p.max_drawdown:.2%}。完成的是一项价格与持有合同实验，不能把框架落地或实现测试通过称为已有稳定收益能力。

{chr(10).join(rows)}

2万元主方案压力净夏普{small.net_sharpe:.3f}、日历年化{small.calendar_cagr:.2%}、最大回撤{small.max_drawdown:.2%}。两个本金、两档费用、三个实验方案和两个基准共20个完整账户情景全部保留，没有选择对照替代主方案，也没有拼接有利年份。

本轮落实了三件事：以已知收盘财富为锚估计未来清算场景；开盘之前确定最高买价与整手份额；已有持仓按原合同剩余期限比较继续持有和立即退出，不重复扣买入费用。开盘发生后，只应用事前限价、现金和T+1成交约束，不重算股数、择时取消低开买单或利用当日最高最低价。

五日标准化压力、二十日趋势及短长波动比沿用既有价格信息，固定126近邻、最近两日历年成熟训练、每天更新。最长标签已到期才训练，最短剩余期和最长原期限使用同一成熟样本池。它们不识别真实被迫卖出或政策传导；R/L只是候选解释，V因缺乏足够估值及催化证据未启用。与旧分布及剩余价值研究的差别在protocol.json中公开，本轮不重置历史试验量。

主方案夏普的20日区块95%区间为[{interval['conditional_historical_sharpe_95_interval'][0]:.3f}, {interval['conditional_historical_sharpe_95_interval'][1]:.3f}]。相对开盘对照的年化日均收益差为{interval['OPEN_VALUE']['annualized_arithmetic_return_difference']:.2%}，配对区间为[{interval['OPEN_VALUE']['paired_block_95_interval'][0]:.2%}, {interval['OPEN_VALUE']['paired_block_95_interval'][1]:.2%}]。区块区间仍不消除长期反复研究的选择偏差。

主夏普保持242交易日，不接受附件的252日建议作为达标捷径；252换算和20阶HAC敏感性另列。CAGR用真实日历跨度，同时保存旧242日口径。现金收益与无风险基准均为0。原始价格成交，100份整手，0.001价位，T+1，方向封板保守不成交；分红资格、应收、到账和末日清算准备分别入账。BASE单边佣金万二、滑点万五；STRESS单边佣金万四、滑点千一，最低佣金均5元。

主方案产生{int(p.entries)}次入场，成交与未成交在orders.parquet中；压力费用佣金{p.commission:.2f}元，滑点{p.slippage:.2f}元，平均收盘仓位{p.mean_exposure:.2%}。风险预算是在已知价格上事前约束，跳空和延迟卖出可能突破事后比例，最大回撤不是保证。

原始分红表覆盖核对{source_count}次公告来源。预测评价包含{prediction['mature_evaluation_rows']}个成熟日原点，五日期末正收益概率Brier分数{prediction['brier_score']:.4f}；这不是126个独立案例，也不是实际退出盈利概率。完整账户收益负责最后裁决。

![主账户净值与回撤](主账户净值与回撤.png)

后续按结果执行：本版本不改近邻数、窗口、方向、限价缓冲或退出期限。若未通过，保留失败，停止从同一历史上救援本版本；若历史通过，也必须另行积累冻结后新数据。当前独立前向样本0，外部审阅未进行，当前市场观点NO_VIEW，无交易授权，没有恢复计划采集，也未制作交付ZIP。

本轮只核对直接影响结论的来源、时间顺序、成交和账户恒等式。日线缺少集合竞价容量、排队和逐笔成交，限价触及仅是模拟近似。自由叙事、因果机制、当前市场机会和真实成交均未被验证。

方法参考：[夏普统计误差与序列相关](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios)；交易规则参考：[上交所股票ETF的T+1说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)、[交易单位与价格档位](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734754.shtml)。本轮在线读取官方说明，不据此宣称已复核2026年全部规则附件。
"""


def plot_accounts(ledgers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    names = {"LIMIT_VALUE": "事前限价＋剩余价值（主方案）", "OPEN_VALUE": "开盘承诺＋剩余价值", "LIMIT_FIXED": "事前限价＋固定期限", "BUY_HOLD_50": "半仓持有"}
    for name, label in names.items():
        d = ledgers[name]
        axes[0].plot(d.date, d.equity/200000, label=label, linewidth=1.7 if name == "LIMIT_VALUE" else 1.1)
        axes[1].plot(d.date, -100*d.drawdown, linewidth=1.5 if name == "LIMIT_VALUE" else 1.)
    axes[0].axhline(1, color="grey", linewidth=.6)
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[0].set_ylabel("完整账户净值")
    axes[1].set_ylabel("回撤（%）")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("510300 事前价格与持有合同固定实验｜20万元｜压力成本\n2017—2025年已研究历史；含现金日、分红与费用，无独立前向验证", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "主账户净值与回撤.png", dpi=160)
    plt.close(fig)


def run():
    check_freeze()
    save(OUT / "run_started.json", {"at": now(), "expected_account_scenarios": 20})
    market, dividends, announcements = load_data()
    x, y = features(market), labels(market)
    neighbors, fits = forecasts(market, x, y)
    x.to_parquet(OUT / "features.parquet", index=False)
    np.savez_compressed(OUT / "forecasts.npz", labels=y, neighbors=neighbors)
    save(OUT / "fit_records.json", fits)
    save(OUT / "dividend_source_clock.json", announcements)
    records, yearly, periods = [], [], []
    e, primary_ledgers = engine(), {}
    for capital in (200000, 20000):
        for cost in COSTS:
            for policy in POLICIES:
                ledger, orders, decisions = simulate(market, dividends, y, neighbors, policy, capital, cost, e)
                path = OUT / "accounts" / f"{capital}_{cost}_{policy}"
                path.mkdir(parents=True)
                ledger.to_parquet(path / "ledger.parquet", index=False)
                orders.to_parquet(path / "orders.parquet", index=False)
                decisions.to_parquet(path / "decisions.parquet", index=False)
                trades = cycles(ledger)
                trades.to_csv(path / "cycles.csv", index=False, encoding="utf-8-sig")
                result = metrics(ledger, capital)
                if len(trades):
                    np.testing.assert_allclose(trades.pnl.sum(), result["net_profit"], rtol=0, atol=1e-6)
                result.update(capital=capital, cost=cost, policy=policy,
                    closed_cycles=int(trades.closed.sum()),
                    largest_winner=float(trades.pnl.max()) if len(trades) else None,
                    largest_cycle_share_of_total_profit=float(trades.pnl.max()/result["net_profit"]) if len(trades) and result["net_profit"] > 0 else None,
                    limit_unfilled=int(orders.status.eq("UNFILLED_PRECOMMITTED_LIMIT").sum()) if len(orders) else 0)
                records.append(result)
                for name, begin, end in [(str(z), f"{z}-01-01", f"{z}-12-31") for z in range(2017, 2026)] + [("2017-2020", START, "2020-12-31"), ("2021-2025", "2021-01-04", END)]:
                    subset = ledger[ledger.date.between(begin, end)]
                    initial = float(subset.equity.iloc[0]/(1+subset.net_return.iloc[0]))
                    row = {"capital": capital, "cost": cost, "policy": policy, "period": name, **metrics(subset, initial)}
                    (periods if "-" in name else yearly).append(row)
                if capital == 200000 and cost == "STRESS":
                    primary_ledgers[policy] = ledger
            print(f"已完成{capital}元、{cost}成本的五个账户。", flush=True)
    table = pd.DataFrame(records)
    table["joint_historical_target"] = table.net_sharpe.ge(1.2)&table.calendar_cagr.ge(.1)&table.max_drawdown.le(.1)
    table.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(OUT / "period_metrics.csv", index=False, encoding="utf-8-sig")
    interval = uncertainty(primary_ledgers)
    save(OUT / "uncertainty.json", interval)
    evaluation = []
    for row in fits:
        i = row["idx"]
        if row["status"] == "AVAILABLE" and np.isfinite(y[i, 4]):
            evaluation.append({"idx": i, "date": market.date.iloc[i], "predicted_probability": row["probability_h5_positive"],
                "actual_positive": bool(y[i, 4] > 0), "predicted_mean": row["mean_h5"], "actual_return": y[i, 4]})
    evaluation = pd.DataFrame(evaluation)
    evaluation.to_parquet(OUT / "prediction_evaluation.parquet", index=False)
    prediction = {"mature_evaluation_rows": len(evaluation), "brier_score": float(np.mean((evaluation.predicted_probability-evaluation.actual_positive.astype(float))**2)),
        "mse": float(np.mean((evaluation.predicted_mean-evaluation.actual_return)**2)), "probability_is_trade_win_probability": False}
    save(OUT / "prediction_summary.json", prediction)
    p = table[(table.policy == "LIMIT_VALUE")&(table.capital == 200000)&(table.cost == "STRESS")].iloc[0]
    result = {"at": now(), "study_id": "510300_MECHANISM_ODDS_OPEN_CONTRACT_V1", "account_scenarios": len(table),
        "status": "HISTORICAL_POINT_PASS_REQUIRES_INDEPENDENT_EVIDENCE" if p.joint_historical_target else "REJECTED_FROZEN_NO_PARAMETER_RESCUE",
        "primary": p.to_dict(), "all_historical_target_passes": int(table.joint_historical_target.sum()),
        "independent_forward_observations": 0, "qualified_strategies": 0, "goal_achieved": False,
        "current_market_view": "NO_VIEW", "external_review": "NOT_PERFORMED", "orders_authorized": False,
        "new_market_downloads": 0, "delivery_package_created": False}
    save(OUT / "result.json", result)
    (OUT / "研究结论.md").write_text(render_report(table, interval, prediction, len(announcements)), encoding="utf-8")
    plot_accounts(primary_ledgers)
    save(OUT / "run_completed.json", {"at": now(), "status": result["status"]})
    print(json.dumps(clean({"主方案": result["primary"], "结论": result["status"]}), ensure_ascii=False), flush=True)


def verify():
    check_freeze()
    market, dividends, announcements = load_data()
    saved = np.load(OUT / "forecasts.npz")
    x = pd.read_parquet(OUT / "features.parquet")
    y, neighbors = saved["labels"], saved["neighbors"]
    np.testing.assert_allclose(labels(market), y, rtol=0, atol=0, equal_nan=True)
    pd.testing.assert_frame_equal(features(market), x)
    schedule = read(OUT / "fit_records.json")
    valid = [r for r in schedule if r["status"] == "AVAILABLE"]
    for record in valid:
        i = record["idx"]
        assert record["latest_maturity_idx"] <= i
        assert (neighbors[i]+H+1 <= i).all()
        assert (neighbors[i] >= record["training_start_idx"]).all()
        assert len(np.unique(neighbors[i])) == K
    # 三个相隔数年的截点，删去以后全部数据仍须得到同一特征及近邻。
    for offset in (0, len(valid)//2, len(valid)-1):
        i = valid[offset]["idx"]
        prefix = market.iloc[:i+1].copy()
        selected, _ = forecast_one(prefix, features(prefix), labels(prefix), i)
        np.testing.assert_array_equal(selected, neighbors[i])
    metrics_table = pd.read_csv(OUT / "metrics.csv")
    rows, maxerror = 0, 0.
    for row in metrics_table.itertuples():
        folder = OUT / "accounts" / f"{row.capital}_{row.cost}_{row.policy}"
        d = pd.read_parquet(folder / "ledger.parquet")
        plans = pd.read_parquet(folder / "decisions.parquet")
        recomputed = metrics(d, row.capital)
        for key, value in recomputed.items():
            stored = getattr(row, key)
            if value is None:
                assert pd.isna(stored)
            elif isinstance(value, (bool, np.bool_)):
                assert bool(stored) == bool(value)
            else:
                np.testing.assert_allclose(stored, value, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(d.equity, d.cash+d.shares*d.mark+d.dividend_receivable-d.terminal_exit_reserve, rtol=0, atol=1e-7)
        assert (d.filled_quantity >= -d.sellable_before).all()
        assert (d.cash >= -1e-7).all()
        assert (d.decision_idx < d.idx).all()
        assert (plans.decision_idx < d.idx.max()).all()
        by_idx = plans.set_index("decision_idx")
        for rec in d.iloc[1:].itertuples():
            plan = by_idx.loc[rec.idx-1]
            assert rec.requested_quantity == plan.quantity
            if rec.filled_quantity > 0 and pd.notna(plan.limit_price):
                assert rec.fill_price <= plan.limit_price+1e-10
        rows += len(d)
        maxerror = max(maxerror, recomputed["maximum_identity_error"])
    receipt = {"at": now(), "status": "PASS_SAVED_INPUT_CLOCK_ORDER_ACCOUNT_METRICS", "accounts": len(metrics_table),
        "ledger_rows": rows, "forecast_origins": len(valid), "maximum_account_identity_error": maxerror,
        "future_prefix_checks": 3, "new_models_fitted": 0, "new_accounts": 0,
        "boundary": "只读复算及必要时序检查，不是外部审阅或独立策略验证。"}
    save(OUT / "saved_verification_receipt.json", receipt)
    print(json.dumps(clean(receipt), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机制赔率开盘合同固定实验")
    parser.add_argument("stage", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.stage]()
