"""候选库首批：融资压力缓和及偿还活动的固定机制检验。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from research.factor96_library_intake_v1 import OUT, ROOT, digest

STUDY = "510300_FACTOR96_MARGIN_REPAIR_V1"
PRICE = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925/candidate_features.parquet"
MARGIN = ROOT / "data/raw/market_margin_validation_v2/market_margin_sh_sz_daily_2015_2025.parquet"
DIVIDENDS = ROOT / "data/reference/510300_dividends.csv"
COSTS = {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005},
         "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}
PERIODS = {"EARLY": ("2017-01-03", "2020-12-31"), "MAIN": ("2021-01-04", "2025-12-31")}
POLICIES = ["T03", "T03_PRICE", "T05", "T05_PRICE", "BUY_HOLD_50", "CASH"]
STATE = ["pressure5", "trend20", "log_rv5_rv60"]


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).isoformat() if pd.notna(x) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def save(path, data, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def engine():
    path = OUT / "code/account_engine.py"
    spec = importlib.util.spec_from_file_location("factor96_frozen_account", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_dividends(frame):
    frame = frame.copy()
    for col in ["record_date", "ex_date", "payment_date"]:
        frame[col] = pd.to_datetime(frame[col]).dt.normalize()
    assert frame.record_date.lt(frame.ex_date).all()
    assert frame.ex_date.le(frame.payment_date).all()
    return frame


def price_features(market):
    """全部在当日收盘可算；财富高低价只用于信号，原始价用于成交。"""
    d = market.copy().reset_index(drop=True)
    r = np.log((d.close + d.dividend) / d.close.shift())
    w = np.exp(r.fillna(0).cumsum())
    hi, lo = w * (d.high + d.dividend) / (d.close + d.dividend), w * (d.low + d.dividend) / (d.close + d.dividend)
    rv20 = r.rolling(20).std(ddof=1)
    tr = pd.concat([hi - lo, (hi - w.shift()).abs(), (lo - w.shift()).abs()], axis=1).max(axis=1)
    x = pd.DataFrame({"date": d.date, "wealth": w, "high_w": hi, "low_w": lo,
                      "r": r, "rv20": rv20, "atr20": tr.rolling(20).mean(), "ma20": w.rolling(20).mean()})
    x["pressure5"] = r.rolling(5).sum() / (rv20 * np.sqrt(5))
    x["trend20"] = r.rolling(20).sum() / (rv20 * np.sqrt(20))
    x["log_rv5_rv60"] = np.log(r.rolling(5).std(ddof=1) / r.rolling(60).std(ddof=1))
    impact = r.abs() / d.amount.where(d.amount.gt(0))
    def signed_median(values, sign):
        ids = np.flatnonzero((values[:, 0] * sign) > 0)
        return np.median(values[ids, 1]) if len(ids) >= 5 else np.nan
    c01 = np.full(len(d), np.nan)
    for i in range(19, len(d)):
        block = np.column_stack([r.iloc[i-19:i+1], impact.iloc[i-19:i+1]])
        down, up = signed_median(block, -1), signed_median(block, 1)
        if np.isfinite([down, up]).all() and up > 0:
            c01[i] = down / up
    x["C01"] = c01
    speed = r.rolling(3).mean() - r.shift(3).rolling(3).mean()
    b06 = speed.gt(0) & w.gt(hi.shift())
    x["B06_first"] = b06 & ~b06.shift(fill_value=False)
    x["T05_price_signal"] = x.B06_first & x.C01.le(x.C01.shift(5))
    x["T05_stop"] = lo.rolling(6).min()
    x["T05_take"] = w.shift().rolling(20).mean()
    x["shock"] = r.lt(-2 * rv20.shift())
    x["T03_price_signal"] = False
    x["shock_origin"] = -1
    x["T03_stop"], x["T03_take"] = np.nan, np.nan
    last_event, active = -1000, None
    for i in range(len(x)):
        if active is not None:
            e = active
            if i - e > 3:
                active = None
            elif i > e and lo.iloc[i] >= lo.iloc[e] and r.iloc[i] > 0:
                x.loc[i, ["T03_price_signal", "shock_origin", "T03_stop", "T03_take"]] = [True, e,
                    lo.iloc[e] - .5 * x.atr20.iloc[e-1], x.ma20.iloc[e-1]]
                active = None
        if bool(x.shock.iloc[i]) and i - last_event >= 10:
            last_event, active = i, i
    return x


def margin_features(market, margin, lag):
    """先对交易日历对齐再位移；有缺口则不将跨缺口变化伪装成单日数据。"""
    m = market[["date"]].merge(margin[["date", "market_rzye", "market_rzmre"]], on="date", how="left", validate="one_to_one")
    bal, buy = m.market_rzye, m.market_rzmre
    repay = buy - bal.diff()
    m["F01"] = bal / bal.shift(5) - 1
    m.loc[bal.rolling(6).count().lt(6), "F01"] = np.nan
    m["F02"] = m.F01 - m.F01.shift(5)
    m["F06_implied_repay5"] = repay.rolling(5).sum() / bal.shift(5)
    m["F06_q80"] = m.F06_implied_repay5.shift().rolling(252, min_periods=120).quantile(.8)
    m["margin_stat_date"] = m.date.where(bal.notna())
    result = m[["F01", "F02", "F06_implied_repay5", "F06_q80", "margin_stat_date"]].shift(lag)
    result["margin_known"] = np.isfinite(result[["F01", "F02", "F06_implied_repay5", "F06_q80"]]).all(axis=1)
    assert (result.loc[result.margin_known, "margin_stat_date"] < market.loc[result.margin_known, "date"]).all()
    return result


def mature_risk(market, features, dividends):
    """每日只从此前两年且五日结果已成熟的原点中取126个相似状态。"""
    d, x = market, features
    labels = np.full(len(d), np.nan)
    for j in range(len(d)-6):
        a, b = j+1, j+6
        div = dividends.loc[dividends.record_date.ge(d.date.iloc[a]) & dividends.record_date.lt(d.date.iloc[b]), "cash_dividend_per_share"].sum()
        labels[j] = (d.open.iloc[b] + div) / d.open.iloc[a] - 1
    raw = x[STATE].to_numpy(float)
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(labels)
    es = np.full(len(d), np.nan)
    saved = []
    for t in range(len(d)):
        if not np.isfinite(raw[t]).all():
            continue
        lower = d.date.iloc[t] - pd.DateOffset(years=2)
        ids = np.flatnonzero(valid & (np.arange(len(d))+6 <= t) & d.date.ge(lower).to_numpy())
        if len(ids) < 252:
            continue
        mean, sd = raw[ids].mean(axis=0), raw[ids].std(axis=0, ddof=1)
        sd[sd < 1e-12] = 1.
        z, current = np.clip((raw[ids]-mean)/sd, -5, 5), np.clip((raw[t]-mean)/sd, -5, 5)
        chosen = ids[np.lexsort((ids, ((z-current)**2).sum(axis=1)))[:126]]
        tail_count = math.ceil(len(chosen)*.05)
        es[t] = max(0., -float(np.sort(labels[chosen])[:tail_count].mean()))
        saved.append({"decision_idx": t, "date": d.date.iloc[t], "training_count": len(ids),
            "latest_training_exit_idx": int(ids.max()+6), "selected_indices": chosen.tolist(),
            "training_mean": mean.tolist(), "training_std": sd.tolist(), "es95_5d": es[t]})
    return es, pd.DataFrame({"origin_idx": np.arange(len(d)), "exit_idx": np.arange(len(d))+6, "gross_return5": labels}), saved


def target_quantity(e, account, reference, peak, es, desired=.5):
    """按假设压力成本计入建仓/减仓及最终卖出成本；不保证实际尾部损失。"""
    nav = account.value(reference)
    if not np.isfinite(es) or nav <= 0:
        return 0
    remaining = max(0., nav - .9*peak)
    cap = min(desired, .5, .025/max(es, 1e-12), .5*remaining/(.1*nav))
    q = math.floor(max(0., cap)*nav/reference/100)*100
    cost = COSTS["STRESS"]
    while q > 0:
        change = q - account.shares
        px = e.fill_price(reference, 1 if change > 0 else -1, cost, .001)
        friction = abs(change)*abs(px-reference) + e.commission(change, px, cost)
        exit_px = e.fill_price(reference, -1, cost, .001)
        reserve = q*(reference-exit_px) + e.commission(q, exit_px, cost)
        if (q*reference*es + friction + reserve <= .025*nav + 1e-8 and
            q*reference*.1 + friction + reserve <= min(.05*nav, .5*remaining) + 1e-8 and
            q*reference <= .5*(nav-friction) + 1e-8):
            return q
        q -= 100
    return 0


def simulate(d, x, dividends, policy, capital, cost_name, start, end, e):
    ids = np.flatnonzero(d.date.between(start, end))
    first, last = int(ids[0]), int(ids[-1])
    account = e.Account(float(capital))
    cost, cfg = COSTS[cost_name], {"lot": 100, "tick": .001, "limit_fraction": .1}
    previous_equity, previous_mark, previous_reserve, peak = capital, float(d.close.iloc[first-1]), 0., capital
    entry_idx, stop_level, take_level = None, np.nan, np.nan
    stopped, exit_pending, benchmark_entered = False, False, False
    records, decisions = [], []
    events = dividends.to_dict("records")
    for i in ids:
        row, f = d.iloc[i], x.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0)*event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        reference = float(d.close.iloc[i-1] - row.dividend)
        q, reason, active_signal = 0, "无新买入条件", False
        if policy == "BUY_HOLD_50":
            if not benchmark_entered and i < last:
                target = math.floor(.5*account.value(reference)/reference/100)*100
                q, reason = target, "期初固定半仓买入持有基准"
        elif policy != "CASH":
            family = policy[:3]
            active_signal = bool(f[f"{family}_price_signal"])
            if policy == "T03":
                active_signal &= bool(f.margin_known and f.F01 < 0 and f.F02 > 0)
            elif policy == "T05":
                active_signal &= bool(f.margin_known and f.F06_implied_repay5 > f.F06_q80)
            if old:
                exit_pending |= (float(f.wealth) < stop_level or float(f.wealth) >= take_level or i >= entry_idx+5)
                if stopped or exit_pending:
                    q, reason = -old, "回撤停机" if stopped else "结构退出或五日到期"
                else:
                    target = min(old, target_quantity(e, account, reference, peak, float(f.es95)))
                    q, reason = target-old, "既有持仓只按风险预算减仓"
            elif not stopped and active_signal and i < last:
                q = target_quantity(e, account, reference, peak, float(f.es95))
                reason = "信号确认后的下一开盘申请"
        before_gap = q
        if q > 0 and policy != "BUY_HOLD_50":
            q = min(q, target_quantity(e, account, op, peak, float(f.es95)))
        sellable = account.sellable(i)
        trade = e.execute_order(account, q, op, float(row.previous_close), float(row.dividend), int(i), cost, cfg)
        if trade["filled_quantity"] > 0 and old == 0:
            entry_idx, benchmark_entered = int(i), True
            if policy not in ["BUY_HOLD_50", "CASH"]:
                stop_level, take_level = float(f[f"{policy[:3]}_stop"]), float(f[f"{policy[:3]}_take"])
                exit_pending = False
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = e.fill_price(close, -1, COSTS["STRESS"], .001)
            reserve = account.shares*(close-px) + e.commission(account.shares, px, COSTS["STRESS"])
        equity = account.value(close)-reserve
        price_pnl = old*(op-previous_mark) + account.shares*(close-op)
        error = equity-previous_equity-price_pnl-recognized+trade["commission"]+trade["slippage_cost"]+reserve-previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1-equity/peak
        stopped |= drawdown >= .1
        records.append({"date": day, "idx": i, "policy": policy, "open": op, "mark": close,
            "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
            "terminal_exit_reserve": reserve, "equity": equity, "net_return": equity/previous_equity-1,
            "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
            "exposure": account.shares*close/equity, "accounting_error": error, "drawdown": drawdown,
            "risk_stopped": stopped if policy not in ["BUY_HOLD_50", "CASH"] else False,
            "sellable_before": sellable, "terminal_unliquidated": bool(i == last and account.shares), **trade})
        decisions.append({"date": day, "origin": d.date.iloc[i-1], "origin_idx": i-1,
            "active_signal": active_signal, "margin_stat_date": f.margin_stat_date, "margin_known": f.margin_known,
            "F01": f.F01, "F02": f.F02, "F06_implied_repay5": f.F06_implied_repay5,
            "F06_q80": f.F06_q80, "es95_5d": f.es95, "reason": reason,
            "pre_open_request": before_gap, "requested_quantity": q, "filled_quantity": trade["filled_quantity"],
            "entry_idx": entry_idx, "frozen_stop": stop_level, "frozen_take": take_level})
        if account.shares == 0:
            entry_idx, exit_pending = None, False
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def metrics(ledger, capital):
    r = ledger.net_return.to_numpy(float)
    nav = np.r_[capital, ledger.equity.to_numpy(float)]
    np.testing.assert_allclose(r, nav[1:]/nav[:-1]-1, rtol=0, atol=1e-12)
    sd = r.std(ddof=1)
    return {"days": len(r), "end_equity": nav[-1], "net_profit": nav[-1]-capital,
        "net_sharpe": np.sqrt(242)*r.mean()/sd if sd > 1e-15 else None,
        "sharpe_252_diagnostic": np.sqrt(252)*r.mean()/sd if sd > 1e-15 else None,
        "cagr": (nav[-1]/capital)**(242/len(r))-1,
        "cagr_252_diagnostic": (nav[-1]/capital)**(252/len(r))-1,
        "max_drawdown": float(1-np.min(nav/np.maximum.accumulate(nav))),
        "mean_exposure": ledger.exposure.mean(), "fills": int(ledger.filled_quantity.ne(0).sum()),
        "commission": ledger.commission.sum(), "slippage": ledger.slippage_cost.sum(),
        "terminal_exit_reserve": ledger.terminal_exit_reserve.iloc[-1],
        "max_identity_error": ledger.accounting_error.abs().max()}


def cycle_records(ledger, capital):
    rows, active, prev = [], None, capital
    for i, r in enumerate(ledger.itertuples()):
        if r.filled_quantity > 0 and r.shares_before == 0:
            active = {"entry": r.date, "entry_row": i, "start_equity": prev}
        if active is not None and r.shares == 0:
            rows.append({**active, "exit": r.date, "profit": r.equity-active["start_equity"], "closed": True})
            active = None
        prev = r.equity
    if active is not None:
        rows.append({**active, "exit": None, "profit": prev-active["start_equity"], "closed": False})
    return pd.DataFrame(rows, columns=["entry", "entry_row", "start_equity", "exit", "profit", "closed"])


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮已经冻结；不覆盖冻结文件。")
    inputs, code = OUT / "inputs", OUT / "code"
    inputs.mkdir(exist_ok=True)
    code.mkdir(exist_ok=True)
    paths = {"market.parquet": PRICE, "margin.parquet": MARGIN, "dividends.csv": DIVIDENDS,
        "margin_metadata.json": MARGIN.parent / "metadata.json",
        "prior_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "old_margin_rejection.json": ROOT / "reports/discovery/510300_market_leverage_cascade_5d_discovery_v0.json",
        "dividend_coverage.json": PRICE.parent / "candidate_dividend_coverage.json"}
    for name, path in paths.items():
        shutil.copy2(path, inputs/name)
    shutil.copy2(ROOT/"research/intraday_overnight_increment_v1.py", code/"account_engine.py")
    shutil.copy2(Path(__file__), code/"factor96_margin_repair_v1.py")
    shutil.copy2(ROOT/"research/factor96_library_intake_v1.py", code/"factor96_library_intake_v1.py")
    shutil.copy2(ROOT/"tests/test_factor96_margin_repair_v1.py", code/"test_factor96_margin_repair_v1.py")
    protocol = {"study_id": STUDY, "registered_at": now(), "primary_candidates": ["T03", "T05"],
        "candidate_count": 2, "parameter_search": False, "periods": PERIODS, "capital_cny": [200000, 20000],
        "annual_days": 242, "annual_252": "只披露转换，不据此改变判定", "cash_rate": 0, "rf": 0,
        "costs": COSTS, "primary_margin_lag_sessions": 1, "extra_delay_sensitivity": 2,
        "execution": "收盘确认后次开盘；100份，0.001价位，T+1，方向涨跌停保守不成交；开盘只缩减已提交份额。",
        "candidate_T03": "B04日对数总收益低于前20日标准差负2倍，事件10日去重；第1至3日首个低价不低于冲击日且总收益为正的确认；已公开F5<0且F5减前非重叠5日F5>0。",
        "candidate_T05": "B06三日平均收益较前三日提高且收盘超前日高点的首次确认；C01不高于5日前；此前已公开5日隐含偿还/期初余额高于前252日80分位，最少120值。",
        "exit_T03": "收盘到事件前MA20，或低于事件日低点减0.5事件前ATR20，或买入开盘后5个开盘间隔；次开盘卖。",
        "exit_T05": "收盘到确认前MA20，或跌破确认日及此前5日最低点，或买入开盘后5个间隔；次开盘卖。",
        "interpretation_choices": ["T03不创新低比较当日低价与冻结冲击日低价；不增加中间各日全都守住的新条件。",
            "T05首个确认是B06由假变真；确认区明确固定最近6个完整日；分位当前值排除。",
            "T03草案提到波动回落，但进入种子无rv3阈值，本轮不另造过滤条件。",
            "两个价格对照仅去掉融资条件，风险预算、期限和成本不变。"],
        "financing_measurement": "由官方余额恒等式推导净融资=余额变化、偿还=买入-余额变化；称隐含偿还，含权益调整，非纯现金还款或强平。",
        "source_limit": "历史源为2026年回取；沪市全段交叉核对，深市部分交叉核对和官方补缺；真实首次发布/修订版本未逐日认证。",
        "novelty": "旧研究检验融资杠杆和广泛卖压的5日风险分类；本轮是固定价格修复事件内融资收缩减速/高隐含偿还的增量，不再训练旧分类器或改旧规则。",
        "risk": {"maximum_target": .5, "conditional_five_day_es95_budget": .025, "gap_stress_return": -.1,
            "gap_stress_budget": .05, "drawdown_stop": .1, "remaining_drawdown_half_budget": True,
            "model": "每日此前2日历年，成熟5日开盘总回报，3个价格状态标准化截断5，取126最近邻，至少252训练；只用于风险仓位。",
            "training_maturity": "原点j的开盘j+1至j+6，退出开盘索引不晚于当前收盘；同日风险输出前已可知。",
            "missing": "缺风险估计不新增；已有仓位退出；所有现金日保留。"},
        "benchmark": "期初半仓买入持有，剩余现金；不实施策略止损；CASH零利息，夏普未定义。",
        "terminal": "末日持仓保持原状并扣压力退出成本准备，不伪造成交；应收分红持续计值。",
        "statistics": {"paired_increment": "主期20万元压力相对匹配价格对照日收益差", "block_length": 20,
            "bootstrap_draws": 4000, "seed": 20260927, "family_holm_candidates": 2,
            "global_96_candidate_selection_adjusted": False, "warning": "家族校正不能清除全库和旧研究选择偏差。"},
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1,
            "historical_point_pass": "主期压力20万元且2万元均达到三个线；提前列出两项，不挑最好再回填。",
            "independent_validation": "NOT_ESTABLISHED；两历史时期都已被研究，不称独立样本外。"},
        "stop_condition": "固定规则不达标即保留失败；不改窗口、退出、方向补救。本轮结果不能授权交易。",
        "goal_status": "active", "orders_authorized": False}
    save(OUT/"protocol.json", protocol, True)
    files = list(inputs.iterdir()) + list(code.iterdir()) + [OUT/"protocol.json", OUT/"factor_registry.json", OUT/"strategy_registry.json"]
    save(OUT/"freeze.json", {"frozen_at": now(), "before_new_strategy_returns": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in files]}, True)
    print("已冻结2项候选、价格对照、两种本金、两个历史时期与额外披露延迟。", flush=True)


def run():
    frozen = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        assert digest(OUT/row["path"]) == row["sha256"], row["path"]
    assert digest(Path(__file__)) == digest(OUT/"code/factor96_margin_repair_v1.py")
    save(OUT/"run_started.json", {"started_at": now(), "freeze_sha256": digest(OUT/"freeze.json")}, True)
    d = pd.read_parquet(OUT/"inputs/market.parquet")
    d = d[d.date.le("2025-12-31")].reset_index(drop=True)
    div = normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv"))
    margin = pd.read_parquet(OUT/"inputs/margin.parquet")
    expected = d.loc[d.date.ge("2015-01-05"), "date"].tolist()
    assert margin.date.tolist() == expected, "融资日历不完整或排序错误"
    np.testing.assert_allclose(d.dividend, d.date.map(div.set_index("ex_date").cash_dividend_per_share).fillna(0), rtol=0, atol=1e-12)
    x = price_features(d)
    es, labels, saved = mature_risk(d, x, div)
    x["es95"] = es
    labels.to_parquet(OUT/"mature_risk_labels.parquet", index=False)
    save(OUT/"risk_training_records.json", saved)
    save(OUT/"data_admission.json", {"market_start": d.date.min(), "market_end": d.date.max(),
        "margin_rows": len(margin), "margin_start": margin.date.min(), "margin_end": margin.date.max(),
        "margin_calendar_complete": True, "historic_first_publication_receipts": "NOT_ESTABLISHED",
        "dividend_event_count_to_cutoff": int(div.ex_date.le(d.date.max()).sum()),
        "official_balance_identity_source": "https://www.sse.com.cn/market/othersdata/margin/sum/",
        "repayment_negative_observations": int((margin.market_rzmre-margin.market_rzye.diff()).lt(0).sum())})
    e, rows, annual, accounts, cycle_summary = engine(), [], [], {}, []
    for lag in [1, 2]:
        f = pd.concat([x, margin_features(d, margin, lag)], axis=1)
        f.to_parquet(OUT/f"daily_features_lag{lag}.parquet", index=False)
        policies = POLICIES if lag == 1 else ["T03", "T05"]
        for period, (start, end) in PERIODS.items():
            for capital in [200000, 20000]:
                for cost in COSTS:
                    for policy in policies:
                        ledger, decisions = simulate(d, f, div, policy, capital, cost, start, end, e)
                        folder = OUT/f"accounts/{period}/{capital}/{cost}/LAG{lag}/{policy}"
                        folder.mkdir(parents=True, exist_ok=True)
                        ledger.to_parquet(folder/"ledger.parquet", index=False)
                        decisions.to_parquet(folder/"decisions.parquet", index=False)
                        cc = cycle_records(ledger, capital)
                        cc.to_csv(folder/"cycles.csv", index=False, encoding="utf-8-sig")
                        keys = {"period": period, "capital": capital, "cost": cost, "lag": lag, "policy": policy}
                        m = metrics(ledger, capital)
                        closed = cc[cc.closed.astype(bool)]
                        rows.append({**keys, **m, "closed_cycles": len(closed), "win_rate": float(closed.profit.gt(0).mean()) if len(closed) else None,
                            "point_pass": m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["cagr"] >= .1 and m["max_drawdown"] <= .1})
                        cycle_summary.append({**keys, "closed_cycles": len(closed), "pending_cycles": len(cc)-len(closed),
                            "positive_cycle_profit": closed.loc[closed.profit.gt(0), "profit"].sum(),
                            "negative_cycle_profit": closed.loc[closed.profit.lt(0), "profit"].sum(),
                            "largest_cycle_profit": float(closed.profit.max()) if len(closed) else None})
                        previous = capital
                        for year, group in ledger.groupby(ledger.date.dt.year):
                            annual.append({**keys, "year": int(year), **metrics(group, previous)})
                            previous = float(group.equity.iloc[-1])
                        accounts[period, capital, cost, lag, policy] = ledger
                        if period == "MAIN" and capital == 200000 and cost == "STRESS":
                            print(f"{policy}/延迟{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，回撤{m['max_drawdown']:.3%}。", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cycle_summary).to_csv(OUT/"cycle_summary.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(20260927)
    baseline = accounts["MAIN", 200000, "STRESS", 1, "T03"]
    n, block, draws = len(baseline), 20, 4000
    starts = rng.integers(0, n, size=(draws, math.ceil(n/block)))
    index = ((starts[:,:,None]+np.arange(block)) % n).reshape(draws, -1)[:,:n]
    np.savez_compressed(OUT/"bootstrap_indices.npz", indices=index)
    comparisons = []
    for policy in ["T03", "T05"]:
        a = accounts["MAIN", 200000, "STRESS", 1, policy]
        b = accounts["MAIN", 200000, "STRESS", 1, policy+"_PRICE"]
        diff = a.net_return.to_numpy()-b.net_return.to_numpy()
        boot = diff[index].mean(axis=1)*242
        mean = diff.mean()*242
        p = (1+np.sum(boot-mean >= mean))/(draws+1)
        comparisons.append({"policy": policy, "annual_arithmetic_increment": mean,
            "ci95_low": np.quantile(boot,.025), "ci95_high": np.quantile(boot,.975),
            "one_sided_p": p, "bootstrap_draws": draws,
            "positive_increment_within_family": bool(mean > 0 and np.quantile(boot,.025) > 0)})
    order = np.argsort([r["one_sided_p"] for r in comparisons])
    adjusted = 0.
    for rank, j in enumerate(order):
        adjusted = max(adjusted, min(1., (2-rank)*comparisons[j]["one_sided_p"]))
        comparisons[j]["holm_p"] = adjusted
    save(OUT/"paired_increment.json", comparisons)
    frame = pd.DataFrame(rows)
    candidates = frame[frame.period.eq("MAIN") & frame.cost.eq("STRESS") & frame.lag.eq(1) & frame.policy.isin(["T03", "T05"])]
    pass_ids = [p for p,g in candidates.groupby("policy") if bool(g.point_pass.all())]
    save(OUT/"result.json", {"study_id": STUDY, "completed_at": now(), "new_candidate_definitions": 2,
        "account_count": len(frame), "main_stress_candidate_rows": candidates.to_dict("records"),
        "historical_joint_point_pass_ids": pass_ids, "family_increment": comparisons,
        "status": "COMPLETE_FIXED_RESEARCH_NO_INDEPENDENT_QUALIFICATION", "goal_achieved": False,
        "goal_status": "active", "independent_forward_observations": 0, "external_review": "NOT_PERFORMED",
        "orders_authorized": False, "current_validated_strategy": "NONE", "parameter_rescue": False})
    print(f"固定首批完成：{len(frame)}个完整账户，候选共同点值通过{len(pass_ids)}项；目标未完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
