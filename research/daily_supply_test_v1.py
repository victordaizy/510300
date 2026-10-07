"""日线收复后的第二次供给测试：固定三项比较及完整账户。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_daily_supply_test_v1"
PARENT = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1"
POLICIES = ["DIRECT_RECLAIM", "PRICE_TEST", "SUPPLY_TEST"]
NAMES = {"DIRECT_RECLAIM": "首次收复直接入场", "PRICE_TEST": "等待价格回测再转强", "SUPPLY_TEST": "回测同时缩量缩幅再转强"}
COSTS = {"BASE": (.0002, .0005), "STRESS": (.0004, .001)}
CAPITAL = 200000.
START = pd.Timestamp("2015-01-01")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def helper(frozen=True):
    path = OUT / "code/weekly_daily_entry_locations_v1.py" if frozen else ROOT / "research/weekly_daily_entry_locations_v1.py"
    spec = importlib.util.spec_from_file_location("daily_supply_price_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze():
    if OUT.exists():
        raise RuntimeError("本研究已登记，不覆盖原始记录。")
    OUT.mkdir(parents=True)
    files = {}
    for source, dest in [
        (PARENT / "inputs/prices.parquet", "inputs/prices.parquet"),
        (PARENT / "inputs/dividends.csv", "inputs/dividends.csv"),
        (PARENT / "inputs/dividend_coverage.json", "inputs/dividend_coverage.json"),
        (ROOT / "config/510300_existing_data_training_mandate_v1.json", "inputs/authority.json"),
        (ROOT / "research/weekly_daily_entry_locations_v1.py", "code/weekly_daily_entry_locations_v1.py"),
        (Path(__file__).resolve(), "code/daily_supply_test_v1.py"),
        (ROOT / "tests/test_daily_supply_test_v1.py", "code/test_daily_supply_test_v1.py"),
    ]:
        destination = OUT / dest
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[dest] = {"source": source.relative_to(ROOT).as_posix(), "sha256": digest(destination)}
    protocol = {
        "study": "510300_DAILY_SUPPLY_TEST_V1", "at": now(), "primary": "SUPPLY_TEST",
        "user_request": "好的请继续，直到实现；一年至少五次交易，要么胜率高，要么盈亏比大；只用日线和周线。",
        "hypothesis": "跌破已知日线支撑后收复，随后较高低点的回测若缩量且缩幅，再转强是否比直接收复或仅价格回测具有更好净收益分布。",
        "source": "https://www.wyckoffanalytics.com/wyckoff-method/",
        "source_scope": "实际阅读作者原文的Spring/Test及LPS定义；没有声称观看完整视频。数值规则为本研究操作化，不是作者原版收益规则。",
        "source_limits": "成交量是成交双方共同形成的总量；缩量缩幅只是供给减弱假说的可观测代理，不是真实主动卖单或机构吸筹证据。",
        "duplicate_review": ["第26轮已有同日假跌破及量价修复。本轮增量是先收复、再回测、再转强的先后信息。",
            "第55轮已检验突破后回踩。该分支保持结束；本轮不新增突破回踩版本。",
            "上一轮周枢轴每锚点首触发仍保留失败；本轮使用日线已知20日低点，不改旧协议。"],
        "price_clock": "原始OHLC加截至当日已除息累计现金分红；真实成交用原始价格。当前日只使用上一完整自然周背景。",
        "parent": "空闲时，日最低低于此前20日最低，且收盘重新高于该已知低点，创建一次收复事件。必须周指标预热完毕。事件未结束时不重建。",
        "stop_target": "事件发生时固定：失效线=事件日低点-0.5个前日ATR20；目标=事件日收盘+2*(事件日收盘-失效线)。后续不追移、不要求实际入场计划比达到2。",
        "test": "从事件后第1日至第10日，找首次低点高于事件低点、低点<=原支撑+0.5个原ATR、收盘>原支撑且低于事件收盘的回测；只取第一次。",
        "supply": "这次回测的成交量小于收复日成交量，且日高低价差小于收复日日高低价差。两项都由回测当日收盘确定。",
        "confirmation": "回测后第1至第3个交易日，首次收盘超过回测日最高，发出价格回测信号；原回测同时满足缩量缩幅则也是主方案信号。",
        "failure": "等待中收盘<=固定失效线、>=原固定目标或等待到期均结束；结束当日不新建事件；失败及未完成过程全部保留。",
        "policies": NAMES,
        "execution": "信号后一个交易日开盘代理；开盘必须在原止损与目标间，涨停或无量不能买。信号只对下一开盘有效。收盘达到目标/失效/持有20交易日则下一可卖开盘卖出，跌停顺延。无日内触价成交假设。",
        "account": {"initial_capital": CAPITAL, "assets": ["510300.SH", "CASH_CNY"], "start": "2015-01-01", "annual_days": 252,
            "cash_return": 0, "single_position": True, "lot": 100, "tick": .001, "t_plus_one": True,
            "risk": "沿用最高50%仓位、五日ES预算2.5%、单次10%冲击损失预算5%、回撤剩余空间只消耗一半。买卖压力成本一并计入预算。",
            "risk_updates": "持仓收盘后按已知波动重新计算最大合法份额；只能减持，次日开盘执行。分批减持不增加完整周期次数。回撤>=10%触发退出并永久停止该账户。",
            "risk_es": "使用此前2日历年已成熟五交易日开盘至开盘含分红回报，至少252条；按决策日RV20的对数距离选择126条最近波动状态，最差ceil(126*5%)条均值的负值为ES。训练标签退出日<=当前日，匹配不读取标签收益。",
            "sizing": "下一开盘前以昨收减当日已知除息确定份额与预算；实际开盘仅缩减份额，不能因跌价扩大原数量。",
            "dividends": "登记日收盘确定权益，除息日记应收，支付日收盘转现金；不提前使用应收现金。",
            "terminal": "期末按收盘净值计价并预留实际卖出成本；未完成持仓不算已完成周期，不做已知终点强平。"},
        "costs": COSTS, "minimum_commission": 5,
        "frequency": "每完整自然年至少5个完整周期，按入场年；2015—2025逐年检查，2026不完整单列。节点研究也保留不足5年的结果，不拼接凑次数。",
        "acceptance": {"win_rate_or_payoff": [0.55, 2.0], "these_thresholds_are_working_definitions": True,
            "positive_mean_net": True, "net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1,
            "increment": "主方案相对价格测试的全日历账户净收益差，20日循环区块2000次95%下界>0才证明本轮增量；全部相同抽样。"},
        "diagnostics": "分别列2015—2019、2020—2023、2024—2026；MACD日/周方向、波动高低、确认日相对量只分组解释，不由结果形成新过滤。",
        "fixed_search_budget": "三项同源比较，两档成本，共6条完整连续账户；没有参数网格、收益后救援、旧失败复活或分钟线。",
        "independent_validation": "NOT_ESTABLISHED_HISTORY_PREVIOUSLY_USED", "orders_authorized": False,
    }
    save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"source": "本轮固定定义", "sha256": digest(OUT / "protocol.json")}
    save_json(OUT / "freeze.json", {"at": now(), "before_new_signals_and_returns": True, "files": files})
    print("日线二次供给测试已固定：三个比较、两档成本，开始历史检验。", flush=True)


def load():
    p = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for col in ["record_date", "ex_date", "payment_date"]:
        div[col] = pd.to_datetime(div[col])
    d, w = helper().features(p, div)
    d["support20"] = d.al.shift().rolling(20).min()
    return d, w, div


def detect(d):
    episodes, steps, signals = [], [], []
    active = None

    def signal(ep, i, policy):
        r = d.iloc[i]
        return {"event_id": ep["event_id"], "policy": policy, "parent_idx": ep["parent_idx"],
                "parent_date": ep["parent_date"], "signal_idx": i, "signal_date": r.date,
                "test_idx": ep.get("test_idx", -1), "test_date": ep.get("test_date"),
                "stop_index": ep["stop_index"], "target_index": ep["target_index"],
                "signal_cash_shift": float(r.cash_shift), "signal_close": float(r.close),
                "max_entry_raw_at_signal": np.nan, "supply_pass": bool(ep.get("supply_pass", False)),
                "weekly_macd_rising": bool(r.weekly_macd_rising), "daily_macd_rising": bool(r.daily_macd_rising),
                "low_volatility": bool(r.low_volatility), "relative_volume_high": bool(r.relative_volume_high),
                "relative_volume": float(r.relative_volume), "prior_rv_ratio": float(r.prior_rv_ratio)}

    for i in range(21, len(d)):
        r = d.iloc[i]
        if r.date < START or not r.context_available:
            continue
        if active is None:
            if not (r.al < r.support20 < r.ac):
                continue
            atr = float(d.atr20.iloc[i - 1])
            stop = float(r.al) - .5 * atr
            active = {"event_id": "TEST_" + r.date.strftime("%Y%m%d"), "parent_idx": i,
                      "parent_date": r.date, "support": float(r.support20), "parent_low": float(r.al),
                      "parent_close": float(r.ac), "parent_volume": float(r.volume),
                      "parent_spread": float(r.ah - r.al), "atr": atr, "stop_index": stop,
                      "target_index": float(r.ac + 2 * (r.ac - stop)), "stage": "WAIT_TEST"}
            signals.append(signal(active, i, "DIRECT_RECLAIM"))
            steps.append({"event_id": active["event_id"], "idx": i, "date": r.date, "action": "RECLAIM_CREATED"})
            continue
        age = i - active["parent_idx"]
        action, terminal = "WAIT", False
        if r.ac <= active["stop_index"]:
            action, terminal = "FAILED_STRUCTURE", True
        elif r.ac >= active["target_index"]:
            action, terminal = "TARGET_PASSED_BEFORE_TEST_ENTRY", True
        elif active["stage"] == "WAIT_TEST":
            if age > 10:
                action, terminal = "EXPIRED_NO_RETEST", True
            elif (r.al > active["parent_low"] and r.al <= active["support"] + .5 * active["atr"]
                  and active["support"] < r.ac < active["parent_close"]):
                active.update(test_idx=i, test_date=r.date, test_high=float(r.ah), test_volume=float(r.volume),
                              test_spread=float(r.ah - r.al), stage="WAIT_CONFIRM",
                              supply_pass=bool(r.volume < active["parent_volume"] and r.ah - r.al < active["parent_spread"]))
                action = "FIRST_RETEST_OBSERVED"
        else:
            test_age = i - active["test_idx"]
            if test_age > 3:
                action, terminal = "EXPIRED_NO_CONFIRM", True
            elif r.ac > active["test_high"]:
                signals.append(signal(active, i, "PRICE_TEST"))
                if active["supply_pass"]:
                    signals.append(signal(active, i, "SUPPLY_TEST"))
                action, terminal = "CONFIRMED", True
        steps.append({"event_id": active["event_id"], "idx": i, "date": r.date, "action": action})
        if terminal:
            episodes.append({**active, "state": action, "end_idx": i, "end_date": r.date})
            active = None
    if active is not None:
        episodes.append({**active, "state": "RIGHT_CENSORED", "end_idx": len(d) - 1, "end_date": d.date.iloc[-1]})
    return pd.DataFrame(episodes), pd.DataFrame(steps), pd.DataFrame(signals)


def risk_estimates(d):
    """选择仅按已知RV20距离，收益只用于已成熟训练集的尾部统计。"""
    rows, membership = [], []
    gross = ((d.ao.shift(-6) - d.ao.shift(-1)) / d.open.shift(-1)).to_numpy(float)
    vol = d.rv20.to_numpy(float)
    dates = d.date
    for i in np.flatnonzero(dates.ge(START)):
        left = int(dates.searchsorted(dates.iloc[i] - pd.DateOffset(years=2)))
        indices = np.arange(max(20, left), max(20, i - 5))
        indices = indices[np.isfinite(gross[indices]) & np.isfinite(vol[indices]) & (vol[indices] > 0)]
        if len(indices) < 252 or not (vol[i] > 0):
            rows.append({"idx": i, "date": dates.iloc[i], "n_train": len(indices), "es95": np.nan, "status": "NO_ESTIMATE"})
            continue
        distance = np.abs(np.log(vol[indices]) - np.log(vol[i]))
        chosen = indices[np.lexsort((indices, distance))[:126]]
        worst = np.sort(gross[chosen])[:7]
        rows.append({"idx": i, "date": dates.iloc[i], "n_train": len(indices), "es95": max(0., -float(worst.mean())),
                     "latest_label_exit_idx": int(chosen.max() + 6), "status": "KNOWN_AT_CLOSE"})
        membership.extend({"decision_idx": i, "origin_idx": int(k), "label_exit_idx": int(k + 6),
                           "gross_return5": float(gross[k])} for k in chosen)
    return pd.DataFrame(rows), pd.DataFrame(membership)


def fill(raw, side, cost):
    value = raw * (1 + side * COSTS[cost][1]) / .001
    return (math.ceil(value - 1e-9) if side > 0 else math.floor(value + 1e-9)) * .001


def fee(notional, cost):
    return max(5., notional * COSTS[cost][0]) if notional > 0 else 0.


def limits(nav, peak, es):
    return {"nav": nav, "position_budget": .5 * nav, "es_budget": .025 * nav,
            "gap_budget": min(.05 * nav, .5 * max(0., nav - .9 * peak)), "es95": es}


def risk_ok(q, raw, budgets, buying=False, existing=0):
    if not q:
        return True
    buy_cost = q * (fill(raw, 1, "STRESS") - raw) + fee(q * fill(raw, 1, "STRESS"), "STRESS") if buying else 0.
    if existing > q:
        reduced = existing - q
        buy_cost += reduced * (raw - fill(raw, -1, "STRESS")) + fee(reduced * fill(raw, -1, "STRESS"), "STRESS")
    exit_cost = q * (raw - fill(raw, -1, "STRESS")) + fee(q * fill(raw, -1, "STRESS"), "STRESS")
    notional = q * raw
    return (notional <= budgets["position_budget"] - .5 * buy_cost + 1e-8
            and notional * budgets["es95"] + exit_cost + buy_cost <= budgets["es_budget"] + 1e-8
            and .1 * notional + exit_cost + buy_cost <= budgets["gap_budget"] + 1e-8)


def cap_quantity(raw, budgets, maximum, cash=None, existing=0):
    if not np.isfinite(budgets["es95"]):
        return 0
    q = int(min(maximum, budgets["position_budget"] / raw) // 100) * 100
    while q:
        if risk_ok(q, raw, budgets, buying=cash is not None, existing=existing):
            debit = q * fill(raw, 1, "STRESS") + fee(q * fill(raw, 1, "STRESS"), "STRESS")
            if cash is None or debit <= cash + 1e-8:
                return q
        q -= 100
    return 0


def at_limit(d, i, side):
    reference = d.close.iloc[i - 1] - d.dividend.iloc[i]
    threshold = round(float(reference) * (1 + side * .1), 3)
    return bool(d.volume.iloc[i] <= 0 or (d.open.iloc[i] >= threshold - .0005 if side > 0 else d.open.iloc[i] <= threshold + .0005))


def account(d, dividends, signals, risks, policy, cost, start=START):
    first = int(d.index[d.date.ge(start)][0])
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    candidates = {int(r.signal_idx): r._asdict() for r in signals.loc[signals.policy.eq(policy)].itertuples(index=False)}
    cash, q, receivable = CAPITAL, 0, 0.
    previous_nav, peak = CAPITAL, CAPITAL
    active, pending_sell, pending_reason = None, None, None
    stopped = False
    entitlements, outstanding = {}, {}
    events = dividends.to_dict("records")
    ledger, trades, orders, rejections = [], [], [], []
    for i in range(first, len(d)):
        r = d.iloc[i]
        old_q = q
        commission_today = slippage_today = accrued = paid = 0.
        sold_today = False
        for k, event in enumerate(events):
            if event["ex_date"] == r.date:
                dq = entitlements.get(k, 0)
                amount = dq * event["cash_dividend_per_share"]
                if amount:
                    outstanding[k] = amount
                    receivable += amount
                    accrued += amount
                    assert active is not None, "登记持仓在除息前异常丢失。"
                    active["dividend_cny"] += amount
        if q and pending_sell is not None:
            if i > active["entry_idx"] and not at_limit(d, i, -1):
                sold = max(0, q - pending_sell)
                if sold:
                    px = fill(float(r.open), -1, cost)
                    charge = fee(sold * px, cost)
                    cash += sold * px - charge
                    q -= sold
                    active["sell_net_cny"] += sold * px - charge
                    active["sell_fees"] += charge
                    commission_today += charge
                    slip = sold * (float(r.open) - px)
                    slippage_today += slip
                    sold_today = True
                    orders.append({"date": r.date, "idx": i, "event_id": active["event_id"], "side": "SELL", "quantity": sold,
                                   "raw_open": r.open, "fill_price": px, "commission": charge, "slippage": slip, "reason": pending_reason})
                    if q == 0:
                        pnl = active["sell_net_cny"] + active["dividend_cny"] - active["buy_debit"]
                        trades.append({**active, "exit_idx": i, "exit_date": r.date, "exit_reason": pending_reason,
                                       "net_pnl": pnl, "net_return": pnl / active["buy_debit"], "holding_sessions": i - active["entry_idx"]})
                        active = None
                pending_sell, pending_reason = None, None
            else:
                rejections.append({"date": r.date, "idx": i, "event_id": active["event_id"], "reason": "SELL_DEFERRED_T1_OR_LIMIT"})
        sig = candidates.get(i - 1)
        if sig is not None:
            why = None
            es = risk.get(i - 1, np.nan)
            if stopped:
                why = "ACCOUNT_STOPPED"
            elif q or sold_today:
                why = "EXISTING_POSITION_OR_SAME_DAY_EXIT"
            elif not np.isfinite(es):
                why = "MISSING_KNOWN_RISK_ESTIMATE"
            elif at_limit(d, i, 1):
                why = "UP_LIMIT_OR_NO_VOLUME"
            elif not (sig["stop_index"] < r.ao < sig["target_index"]):
                why = "OPEN_OUTSIDE_STRUCTURE"
            else:
                reference = float(d.close.iloc[i - 1] - r.dividend)
                pre_nav = cash + receivable
                budget = limits(pre_nav, peak, es)
                planned = cap_quantity(reference, budget, 100000000, cash)
                filled = cap_quantity(float(r.open), budget, planned, cash)
                if filled:
                    px = fill(float(r.open), 1, cost)
                    charge = fee(filled * px, cost)
                    debit = filled * px + charge
                    assert debit <= cash + 1e-8 and risk_ok(filled, float(r.open), budget, True)
                    cash -= debit
                    q = filled
                    commission_today += charge
                    slip = q * (px - float(r.open))
                    slippage_today += slip
                    active = {**sig, "entry_idx": i, "entry_date": r.date, "entry_price": px, "entry_raw": r.open,
                              "entry_equity": pre_nav, "entry_quantity": q, "buy_debit": debit, "entry_fee": charge,
                              "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.}
                    orders.append({"date": r.date, "idx": i, "event_id": sig["event_id"], "side": "BUY", "quantity": q,
                                   "raw_open": r.open, "fill_price": px, "commission": charge, "slippage": slip,
                                   "reason": "NEXT_OPEN", "pre_open_quantity": planned, "known_es95": es,
                                   "reference_price": reference, **budget})
                else:
                    why = "CASH_OR_RISK_BUDGET_BELOW_ONE_LOT"
            if why:
                rejections.append({"date": r.date, "idx": i, "event_id": sig["event_id"], "reason": why})
        for k, event in enumerate(events):
            if event["record_date"] == r.date:
                entitlements[k] = q
            if event["payment_date"] <= r.date and k in outstanding:
                amount = outstanding.pop(k)
                cash += amount
                receivable -= amount
                paid += amount
        nav = cash + q * float(r.close) + receivable
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if dd >= .1:
            stopped = True
        if q:
            why = "ACCOUNT_DRAWDOWN_STOP" if stopped else "CLOSE_STRUCTURE_FAILED" if r.ac <= active["stop_index"] else (
                  "CLOSE_TARGET_REACHED" if r.ac >= active["target_index"] else "TIME_20_SESSIONS" if i - active["entry_idx"] + 1 >= 20 else None)
            if why:
                pending_sell, pending_reason = 0, why
            elif pending_sell is None:
                es = risk.get(i, np.nan)
                keep = cap_quantity(float(r.close), limits(nav, peak, es), q, existing=q)
                if keep < q:
                    pending_sell, pending_reason = keep, "KNOWN_RISK_REDUCTION"
        previous_close = float(d.close.iloc[i - 1])
        price_pnl = old_q * (float(r.open) - previous_close) + q * (float(r.close) - float(r.open))
        error = nav - previous_nav - price_pnl - accrued + commission_today + slippage_today
        assert abs(error) < 1e-6 and cash >= -1e-6 and receivable >= -1e-6 and q % 100 == 0
        ledger.append({"date": r.date, "idx": i, "cash": cash, "shares": q, "close": r.close, "receivable": receivable,
                       "dividend_accrual": accrued, "dividend_paid": paid, "commission": commission_today,
                       "slippage": slippage_today, "equity": nav, "net_return": nav / previous_nav - 1,
                       "drawdown": dd, "risk_stopped": stopped, "accounting_error": error,
                       "event_id": active["event_id"] if active else "", "pending_sell_to": pending_sell})
        previous_nav = nav
    reserve = q * (float(d.close.iloc[-1]) - fill(float(d.close.iloc[-1]), -1, cost)) + fee(q * fill(float(d.close.iloc[-1]), -1, cost), cost) if q else 0.
    terminal = {"open_shares": q, "reserved_exit_cost": reserve, "active_trade": active,
                "known_pending_exit": pending_reason, "stopped": stopped}
    return pd.DataFrame(ledger), pd.DataFrame(trades), pd.DataFrame(orders), pd.DataFrame(rejections), terminal


def trade_stats(t):
    if t.empty:
        return {"cycles": 0, "win_rate": np.nan, "payoff": np.nan, "mean_net": np.nan, "net_pnl": 0., "working_or_edge": False}
    r = t.net_return.to_numpy(float)
    positive, negative = r[r > 0], r[r < 0]
    win = float((r > 0).mean())
    payoff = float(positive.mean() / -negative.mean()) if len(positive) and len(negative) else np.nan
    return {"cycles": len(t), "win_rate": win, "payoff": payoff, "mean_net": float(r.mean()),
            "net_pnl": float(t.net_pnl.sum()),
            "working_or_edge": bool(r.mean() > 0 and t.net_pnl.sum() > 0 and (win >= .55 or payoff >= 2))}


def account_stats(ledger, trades, terminal):
    r = ledger.net_return.to_numpy(float).copy()
    end = float(ledger.equity.iloc[-1]) - terminal["reserved_exit_cost"]
    previous = float(ledger.equity.iloc[-2]) if len(ledger) > 1 else CAPITAL
    r[-1] = end / previous - 1
    curve = CAPITAL * np.cumprod(1 + r)
    dd = float((1 - curve / np.maximum.accumulate(np.r_[CAPITAL, curve])[1:]).max())
    sd = r.std(ddof=1)
    sh = float(r.mean() / sd * np.sqrt(252)) if sd > 1e-14 else np.nan
    cg = float((end / CAPITAL) ** (252 / len(ledger)) - 1)
    annual = []
    for year in range(2015, 2027):
        count = int(pd.to_datetime(trades.entry_date).dt.year.eq(year).sum()) if len(trades) else 0
        annual.append({"year": year, "cycles": count, "full_year": year < 2026, "pass_five": count >= 5 if year < 2026 else None})
    freq = all(row["cycles"] >= 5 for row in annual if row["full_year"])
    trade = trade_stats(trades)
    return {"net_sharpe": sh, "net_cagr": cg, "max_drawdown": dd, "ending_equity": end,
            "full_year_frequency_pass": freq, "minimum_cycles_full_year": min(x["cycles"] for x in annual if x["full_year"]),
            "all_numeric_targets": bool(sh >= 1.2 and cg >= .1 and dd <= .1 and freq and trade["working_or_edge"]),
            "risk_stopped": terminal["stopped"], "terminal_open_shares": terminal["open_shares"],
            "total_friction": float((ledger.commission + ledger.slippage).sum()), **trade}, annual


def check_frozen():
    for name, item in read_json(OUT / "freeze.json")["files"].items():
        assert digest(OUT / name) == item["sha256"], name
    assert digest(Path(__file__).resolve()) == digest(OUT / "code/daily_supply_test_v1.py")


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("已有研究结果，不改变参数重跑。")
    check_frozen()
    d, w, div = load()
    result_dir = OUT / "results"
    result_dir.mkdir(exist_ok=True)
    episodes, steps, signals = detect(d)
    risks, members = risk_estimates(d)
    for name, data in [("daily_features", d), ("weekly_features", w), ("episodes", episodes), ("steps", steps),
                       ("signals", signals), ("risk_estimates", risks), ("risk_training_members", members)]:
        data.to_parquet(result_dir / (name + ".parquet"), index=False)
        if name not in ["daily_features", "risk_training_members"]:
            data.to_csv(result_dir / (name + ".csv"), index=False, encoding="utf-8-sig")
    print(f"已识别{len(episodes)}个收复过程，三项表达共有{len(signals)}条信号；开始固定账户。", flush=True)
    points, point_status = [], []
    h = helper()
    for sig in signals.to_dict("records"):
        status, labels = h.label_one(sig, d, div)
        point_status.append(status)
        points.extend(labels)
    pd.DataFrame(point_status).to_parquet(result_dir / "point_status.parquet", index=False)
    pd.DataFrame(points).to_parquet(result_dir / "event_labels.parquet", index=False)
    metrics, years, contextual, periods, stress_returns = [], [], [], [], {}
    for policy in POLICIES:
        for cost in COSTS:
            ledger, trades, orders, rejections, terminal = account(d, div, signals, risks, policy, cost)
            folder = result_dir / "accounts" / policy / cost
            folder.mkdir(parents=True, exist_ok=True)
            for name, data in [("daily", ledger), ("trades", trades), ("orders", orders), ("rejections", rejections)]:
                data.to_parquet(folder / (name + ".parquet"), index=False)
                data.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")
            save_json(folder / "terminal.json", terminal)
            summary, annual = account_stats(ledger, trades, terminal)
            metrics.append({"policy": policy, "cost": cost, **summary})
            years.extend({"policy": policy, "cost": cost, **row} for row in annual)
            if cost == "STRESS":
                r = ledger.net_return.to_numpy(float).copy()
                r[-1] = summary["ending_equity"] / float(ledger.equity.iloc[-2]) - 1
                stress_returns[policy] = r
                for a, b, tag in [("2015-01-01", "2019-12-31", "2015-2019"), ("2020-01-01", "2023-12-31", "2020-2023"), ("2024-01-01", "2026-12-31", "2024-2026")]:
                    t = trades.loc[pd.to_datetime(trades.entry_date).between(a, b)] if len(trades) else trades
                    periods.append({"policy": policy, "period": tag, **trade_stats(t)})
                for field in ["daily_macd_rising", "weekly_macd_rising", "low_volatility", "relative_volume_high"]:
                    for value in [False, True]:
                        t = trades.loc[trades[field].eq(value)] if len(trades) else trades
                        contextual.append({"policy": policy, "context": field, "value": value, **trade_stats(t)})
            print(f"{NAMES[policy]} / {cost}：完整周期{summary['cycles']}，净夏普{summary['net_sharpe']:.4f}，年化{summary['net_cagr']:.2%}。", flush=True)
    rng = np.random.default_rng(202610012)
    n = len(stress_returns[POLICIES[0]])
    draws = rng.integers(0, n, size=(2000, int(np.ceil(n / 20))))
    idx = ((draws[:, :, None] + np.arange(20)[None, None, :]) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(result_dir / "bootstrap_indices.npz", indices=idx)
    comparisons = []
    for a, b in [("SUPPLY_TEST", "PRICE_TEST"), ("PRICE_TEST", "DIRECT_RECLAIM")]:
        difference = stress_returns[a] - stress_returns[b]
        means = difference[idx].mean(axis=1) * 252
        comparisons.append({"policy": a, "reference": b, "annual_arithmetic_increment": float(difference.mean() * 252),
                            "lower95": float(np.quantile(means, .025)), "upper95": float(np.quantile(means, .975))})
    for name, data in [("账户结果", metrics), ("逐年次数", years), ("固定时期", periods), ("背景逐项观察", contextual), ("固定配对增量", comparisons)]:
        pd.DataFrame(data).to_csv(result_dir / (name + ".csv"), index=False, encoding="utf-8-sig")
    primary = next(x for x in metrics if x["policy"] == "SUPPLY_TEST" and x["cost"] == "STRESS")
    outcome = {"study": "510300_DAILY_SUPPLY_TEST_V1", "at": now(), "parent_episodes": len(episodes),
               "signals_by_policy": signals.groupby("policy").size().to_dict(), "daily_rows": len(d),
               "data_end": d.date.iloc[-1], "new_accounts": 6, "new_alpha_fits": 0, "minute_reads": 0,
               "primary": primary, "paired_increment": comparisons[0],
               "status": "HISTORICAL_CANDIDATE_ONLY" if primary["all_numeric_targets"] and comparisons[0]["lower95"] > 0 else "REJECTED_FROZEN_NO_PARAMETER_RESCUE",
               "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "orders_authorized": False}
    save_json(OUT / "result.json", outcome)
    return outcome


def verify():
    check_frozen()
    folder = OUT / "results"
    d, w, div = load()
    saved_signals = pd.read_parquet(folder / "signals.parquet")
    checks = []
    # 历史前缀复算只判断信号和周背景，不重跑收益；检查未来是否能改变先前触发。
    for end in [800, 1200, 1700, 2300, 2900, len(d) - 1]:
        _, _, s = detect(d.iloc[:end + 1].copy())
        expected = saved_signals.loc[saved_signals.signal_idx.le(end)].reset_index(drop=True)
        pd.testing.assert_frame_equal(s.reset_index(drop=True), expected, check_dtype=False)
        checks.append({"prefix_last_idx": end, "signals": len(s), "pass": True})
    assert (saved_signals.loc[saved_signals.policy.ne("DIRECT_RECLAIM"), "signal_idx"] >
            saved_signals.loc[saved_signals.policy.ne("DIRECT_RECLAIM"), "test_idx"]).all()
    tests = saved_signals.loc[saved_signals.policy.eq("SUPPLY_TEST")]
    assert tests.supply_pass.all()
    members = pd.read_parquet(folder / "risk_training_members.parquet")
    assert (members.label_exit_idx <= members.decision_idx).all()
    raw = ((d.ao.shift(-6) - d.ao.shift(-1)) / d.open.shift(-1)).to_numpy(float)
    assert np.max(np.abs(raw[members.origin_idx.to_numpy(int)] - members.gross_return5)) < 1e-12
    metrics = pd.read_csv(folder / "账户结果.csv")
    total_days, total_cycles, maximum_error = 0, 0, 0.
    for m in metrics.to_dict("records"):
        path = folder / "accounts" / m["policy"] / m["cost"]
        ledger, trades, orders = [pd.read_parquet(path / (name + ".parquet")) for name in ["daily", "trades", "orders"]]
        terminal = read_json(path / "terminal.json")
        measured, annual = account_stats(ledger, trades, terminal)
        for key in ["net_sharpe", "net_cagr", "max_drawdown", "ending_equity", "cycles", "win_rate", "payoff"]:
            assert (pd.isna(measured[key]) and pd.isna(m[key])) or abs(measured[key] - m[key]) < 1e-8, key
        assert not len(trades) or (trades.exit_idx > trades.entry_idx).all()
        q = 0
        for o in orders.itertuples():
            q += o.quantity if o.side == "BUY" else -o.quantity
            assert q >= 0 and q % 100 == 0
            assert abs(fill(o.raw_open, 1 if o.side == "BUY" else -1, m["cost"]) - o.fill_price) < 1e-10
        assert q == terminal["open_shares"]
        accounted = float(trades.net_pnl.sum()) if len(trades) else 0.
        active = terminal["active_trade"]
        if active is not None:
            accounted += active["sell_net_cny"] + active["dividend_cny"] + q * float(d.close.iloc[-1]) - active["buy_debit"]
        assert abs(accounted - (float(ledger.equity.iloc[-1]) - CAPITAL)) < 1e-6
        total_days += len(ledger)
        total_cycles += len(trades)
        maximum_error = max(maximum_error, float(ledger.accounting_error.abs().max()))
    receipt = {"status": "PASS_SIGNAL_PREFIX_MATURE_RISK_AND_SAVED_ACCOUNT_RECOMPUTATION",
               "signal_prefix_checks": checks, "recomputed_account_days": total_days, "recomputed_cycles": total_cycles,
               "mature_risk_memberships": len(members), "maximum_accounting_error": maximum_error,
               "independent_validation": False}
    save_json(OUT / "verification.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="日线二次供给测试固定研究。")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "verify": verify}[args.command]()
    if result is not None:
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
