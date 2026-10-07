"""510300节前与公开冲击后的资金避险：固定规则、完整账户和增量比较。"""
from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
POLICIES = ("A", "A_MATCH", "H_CALENDAR", "H_PRESSURE", "EVENT_REACTION", "COMBINED", "BUY_HOLD", "CASH")
CANDIDATES = ("H_CALENDAR", "H_PRESSURE", "EVENT_REACTION", "COMBINED")
PERIODS = {"PRIMARY": ("2021-01-04", "2026-08-14"), "RECENT_DIAGNOSTIC": ("2024-08-20", "2026-08-14")}
SOURCES = {
    "features.parquet": "reports/research/510300_sequential_patterns_regime_v1/results/features.parquet",
    "signals.parquet": "reports/research/510300_sequential_patterns_regime_v1/results/signals.parquet",
    "dividends.csv": "reports/research/510300_sequential_patterns_regime_v1/inputs/dividends.csv",
    "calendar_history.csv": "data/reference/a_share_hs_trading_calendar_2010_2026_v1.csv",
    "calendar_2026.csv": "data/reference/sse_trade_calendar_2026.csv",
    "calendar_2026_metadata.json": "data/reference/sse_trade_calendar_2026.metadata.json",
    "dr007.parquet": "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet",
    "policy_rate.parquet": "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet",
    "moneyflow.parquet": "data/features/000300_official_weighted_moneyflow_daily.parquet",
    "GSPC.parquet": "data/raw/cross_market_chart_ml_v1/GSPC_daily.parquet",
    "VIX.parquet": "data/raw/us_china_overnight_v1/VIX_daily.parquet",
}


def parent(root=ROOT):
    spec = importlib.util.spec_from_file_location("holiday_risk_frozen_parent", root / "code/parent_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(root=ROOT):
    data = pd.read_parquet(root / "inputs/features.parquet").reset_index(drop=True)
    data["date"] = pd.to_datetime(data.date).dt.strftime("%Y-%m-%d")
    signals = pd.read_parquet(root / "inputs/signals.parquet")
    dividends = pd.read_csv(root / "inputs/dividends.csv")
    return data, signals, dividends


def join_clock(left, right, time_column):
    """右侧保留缺值记录；不向前填补未知观测，也不跳过缺值寻找较好旧值。"""
    left, right = left.copy(), right.copy()
    left["decision_time"] = pd.to_datetime(left.decision_time).astype("datetime64[ns]")
    right[time_column] = pd.to_datetime(right[time_column]).astype("datetime64[ns]")
    right = right.sort_values(time_column).drop_duplicates(time_column, keep="last")
    result = pd.merge_asof(left.sort_values("decision_time"), right, left_on="decision_time", right_on=time_column, direction="backward")
    known = result[time_column].notna()
    assert (result.loc[known, time_column] <= result.loc[known, "decision_time"]).all()
    return result


def next_session_available(source_dates, market_dates):
    ix = market_dates.searchsorted(pd.DatetimeIndex(source_dates), side="right")
    available = pd.Series(pd.NaT, index=np.arange(len(ix)), dtype="datetime64[ns]")
    valid = ix < len(market_dates)
    available.loc[valid] = market_dates[ix[valid]].to_numpy() + pd.Timedelta(hours=9, minutes=30)
    return available.to_numpy()


def build_views(root, d):
    dates = pd.DatetimeIndex(pd.to_datetime(d.date))
    f = pd.DataFrame({"idx": np.arange(len(d)), "date": d.date, "decision_time": dates + pd.Timedelta(hours=9)})

    dr = pd.read_parquet(root / "inputs/dr007.parquet")[["date", "dr007"]].copy()
    dr["dr_available_at"] = next_session_available(pd.to_datetime(dr.date), dates)
    dr = dr.rename(columns={"date": "dr_source_date"}).dropna(subset=["dr_available_at"])
    f = join_clock(f, dr, "dr_available_at")
    policy = pd.read_parquet(root / "inputs/policy_rate.parquet")
    policy = policy.loc[policy.seven_day_rate_percent.notna(), ["published_at", "seven_day_rate_percent"]].rename(columns={"published_at": "policy_available_at"})
    f = join_clock(f, policy, "policy_available_at")
    f["funding_gap_pp"] = f.dr007 - f.seven_day_rate_percent
    f["funding_change5_pp"] = f.dr007.diff(5)
    f["funding_known"] = f[["funding_gap_pp", "funding_change5_pp"]].notna().all(axis=1) & ((f.decision_time - f.dr_source_date).dt.total_seconds() <= 14 * 86400)
    f["funding_tightening"] = f.funding_known & f.funding_gap_pp.gt(0) & f.funding_change5_pp.gt(0)

    money = pd.read_parquet(root / "inputs/moneyflow.parquet").copy()
    money["date"] = pd.to_datetime(money.date)
    money = money.set_index("date").reindex(dates)
    valid = money.large_flow_weight_coverage.ge(.98) & (pd.to_datetime(money.weight_snapshot_date) < money.index)
    # 原聚合允许同日月末快照，本实验剔除这类日子并重新计算连续五交易日指标。
    flow5 = money.official_weighted_large_extra_large_intensity_1d.where(valid).rolling(5, min_periods=5).mean()
    breadth5 = money.official_weighted_positive_large_flow_share_1d.where(valid).rolling(5, min_periods=5).mean()
    m = pd.DataFrame({"flow_source_date": dates, "flow5": flow5.to_numpy(), "flow_breadth5": breadth5.to_numpy(),
                      "flow_available_at": next_session_available(dates, dates)})
    f = join_clock(f, m.dropna(subset=["flow_available_at"]), "flow_available_at")
    f["flow_known"] = f[["flow5", "flow_breadth5"]].notna().all(axis=1) & ((f.decision_time - f.flow_source_date).dt.total_seconds() <= 14 * 86400)
    f["micro_withdrawal"] = f.flow_known & f.flow5.lt(0) & f.flow_breadth5.lt(.5)

    for symbol in ("GSPC", "VIX"):
        raw = pd.read_parquet(root / f"inputs/{symbol}.parquet").sort_values("date").reset_index(drop=True)
        source = pd.to_datetime(raw.date).dt.normalize()
        daily = np.log(raw.close).diff()
        prior_vol = daily.shift(1).rolling(20, min_periods=20).std(ddof=1)
        available = (source + pd.Timedelta(hours=17)).dt.tz_localize("America/New_York").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        g = pd.DataFrame({f"{symbol}_source_date": source, f"{symbol}_available_at": available,
                          f"{symbol}_return": daily, f"{symbol}_z": daily / prior_vol.replace(0, np.nan)})
        f = join_clock(f, g, f"{symbol}_available_at")
        f[f"{symbol}_known"] = f[f"{symbol}_z"].notna() & ((f.decision_time - f[f"{symbol}_available_at"]).dt.total_seconds() <= 7 * 86400)
    f["global_known"] = f.GSPC_known & f.VIX_known
    f["global_shock"] = f.global_known & (f.GSPC_z.le(-2) | f.VIX_z.ge(2))

    history = pd.read_csv(root / "inputs/calendar_history.csv")
    calendar2026 = pd.read_csv(root / "inputs/calendar_2026.csv")
    historical_dates = pd.to_datetime(history.trade_date)
    opens = pd.DatetimeIndex(sorted(set(historical_dates[historical_dates < "2026-01-01"]) | set(pd.to_datetime(calendar2026.trade_date))))
    ix = opens.get_indexer(dates)
    in_calendar = ix >= 0
    pre = np.zeros(len(d), dtype=bool)
    gap_days = np.zeros(len(d), dtype=int)
    last_dates = np.full(len(d), "", dtype=object)
    reopens = np.full(len(d), "", dtype=object)
    closure_rows = []
    for k in range(len(opens) - 1):
        gap = int((opens[k + 1] - opens[k]).days)
        if gap < 4:
            continue
        # 固定节前三个开盘时点；周末补班不会创造证券交易日。
        affected = in_calendar & (ix >= k - 2) & (ix <= k)
        pre[affected], gap_days[affected] = True, gap
        last_dates[affected], reopens[affected] = opens[k].strftime("%Y-%m-%d"), opens[k + 1].strftime("%Y-%m-%d")
        closure_rows.append({"last_open_date": opens[k].strftime("%Y-%m-%d"), "reopen_date": opens[k + 1].strftime("%Y-%m-%d"), "date_gap_days": gap,
                             "calendar_status": "OFFICIAL_ANNUAL_NOTICE_CONTENT" if opens[k].year == 2026 else "RECONSTRUCTED_UNAUTHENTICATED_SCHEDULE"})
    f["calendar_known"] = in_calendar
    f["preholiday3"] = pre
    f["holiday_gap_days"] = gap_days
    f["holiday_last_open"] = last_dates
    f["holiday_reopen"] = reopens
    f["calendar_authenticated_year"] = dates.year == 2026
    f["common_known"] = f.funding_known & f.flow_known & f.global_known & f.calendar_known
    f["H_CALENDAR"] = f.common_known & f.preholiday3
    f["H_PRESSURE"] = f.common_known & f.preholiday3 & f.funding_tightening & f.micro_withdrawal
    f["EVENT_REACTION"] = f.common_known & f.global_shock & (f.funding_tightening | f.micro_withdrawal)
    f["COMBINED"] = f.H_PRESSURE | f.EVENT_REACTION
    for col in ("A", "A_MATCH"):
        f[col] = False
    return f, pd.DataFrame(closure_rows)


def simulate(p, d, dividends, signals, views, start, end, policy, cost):
    """在原形态策略基础上增加九点风险否决；缺信息不卖、风险消失不追买旧信号。"""
    cash, shares, receivable = 200000.0, 0, 0.0
    active, exit_due, pending = None, None, None
    peak = previous = 200000.0
    ledger, trades, decisions, book = [], [], [], []
    by_ex = dict(tuple(dividends.groupby("ex_date")))
    order = {fam: k for k, fam in enumerate(p.FAMILIES)}
    by_signal = {int(k): g.sort_values("family", key=lambda x: x.map(order)).to_dict("records") for k, g in signals.groupby("signal_idx")}
    cycle = 0
    for i in range(start, end + 1):
        r, v = d.iloc[i], views.iloc[i]
        accrual = paid = fees = notional = 0.0
        if shares and r.date in by_ex:
            for item in by_ex[r.date].itertuples():
                amount = shares * float(item.cash_dividend_per_share)
                book.append({"payment_date": item.payment_date, "amount": amount})
                receivable += amount
                accrual += amount
                active["dividend_cny"] += amount
        unpaid = []
        for item in book:
            if item["payment_date"] <= r.date:
                cash += item["amount"]
                receivable -= item["amount"]
                paid += item["amount"]
            else:
                unpaid.append(item)
        book = unpaid
        risk = bool(v[policy]) if policy in CANDIDATES else False
        if shares and risk and exit_due is None:
            exit_due = max(i, active["entry_idx"] + 1)
            active["exit_reason"] = "RISK_REDUCTION"
            active["exit_requested_idx"] = i
            decisions.append({"date": r.date, "idx": i, "signal_id": active["signal_id"], "reason": "RISK_REDUCTION", "risk": True})
        if shares and exit_due is not None and i >= exit_due and i > active["entry_idx"]:
            if p.tradable(d, i, "SELL"):
                px = p.fill_price(float(r.open), "SELL", cost)
                fee = p.commission(shares * px, cost)
                cash += shares * px - fee
                fees += fee
                notional += shares * px
                active.update(exit_idx=i, exit_date=r.date, exit_price=px, exit_fee=fee, quantity=shares,
                              net_pnl=shares * (px - active["entry_price"]) + active["dividend_cny"] - active["entry_fee"] - fee,
                              holding_sessions=i - active["entry_idx"])
                trades.append(active.copy())
                shares, active, exit_due = 0, None, None
            else:
                decisions.append({"date": r.date, "idx": i, "signal_id": active["signal_id"], "reason": "SELL_DEFERRED_LIMIT_OR_NO_VOLUME"})
        if not shares and pending is not None:
            sig = pending
            missing = policy != "A" and not bool(v.common_known)
            invalid_open = r.ao <= sig["stop_index"]
            if missing or risk:
                decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "reason": "NO_VIEW_ENTRY" if missing else "RISK_ENTRY_REJECTED", "risk": risk})
            elif p.tradable(d, i, "BUY") and not invalid_open:
                px = p.fill_price(float(r.open), "BUY", cost)
                shares = p.buy_quantity(cash, px, cost)
                if shares:
                    fee = p.commission(shares * px, cost)
                    cash -= shares * px + fee
                    fees += fee
                    notional += shares * px
                    cycle += 1
                    active = dict(sig, cycle=cycle, entry_idx=i, entry_date=r.date, entry_price=px, entry_fee=fee,
                                  dividend_cny=0.0, entry_equity=previous)
                    decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "reason": "BUY_FILLED"})
                else:
                    decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "reason": "INSUFFICIENT_CASH"})
            else:
                decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "reason": "ENTRY_GAP_INVALIDATED" if invalid_open else "BUY_UNFILLED_LIMIT_OR_NO_VOLUME"})
            pending = None
        if shares and exit_due is None:
            invalid = r.ac < active["stop_index"]
            timed = i - active["entry_idx"] + 1 >= 5
            if invalid or timed:
                exit_due = i + 1
                active["exit_reason"] = "INVALIDATED" if invalid else "TIME_EXIT"
                active["exit_requested_idx"] = i
        equity = cash + shares * float(r.close) + receivable
        peak = max(peak, equity)
        ledger.append({"idx": i, "date": r.date, "cash_cny": cash, "shares": shares, "close": r.close,
                       "receivable_cny": receivable, "dividend_accrual_cny": accrual, "dividend_paid_cny": paid,
                       "fees_cny": fees, "notional_cny": notional, "equity_cny": equity,
                       "daily_return": equity / previous - 1, "drawdown": equity / peak - 1,
                       "exposure": shares * float(r.close) / equity, "active_signal_id": active["signal_id"] if active else "",
                       "exit_pending": exit_due is not None, "preopen_risk": risk, "common_known": bool(v.common_known)})
        assert cash >= -1e-6 and shares % 100 == 0 and receivable >= -1e-6
        previous = equity
        if i < end and i in by_signal:
            for sig in by_signal[i]:
                reason = "CAPITAL_OCCUPIED_OR_PRIORITY" if shares or pending is not None else "PATTERN_PENDING_PREOPEN"
                if reason == "PATTERN_PENDING_PREOPEN":
                    pending = sig
                decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "reason": reason})
    terminal = {"open_position": bool(shares), "shares": shares, "last_signal_id": active["signal_id"] if active else None,
                "pending_sell": exit_due is not None, "terminal_haircut_cny": 0.0,
                "valuation_policy": "终点收盘估值另留卖出成本，不伪造已知终点清仓。"}
    if shares:
        px = p.fill_price(float(d.close.iloc[end]), "SELL", cost)
        terminal["terminal_haircut_cny"] = shares * (float(d.close.iloc[end]) - px) + p.commission(shares * px, cost)
    return pd.DataFrame(ledger), pd.DataFrame(trades), pd.DataFrame(decisions), terminal


def mechanism_checks(p):
    dates = pd.bdate_range("2020-01-01", periods=12).strftime("%Y-%m-%d")
    close = np.array([4, 4.02, 4.04, 4.01, 4.06, 4.1, 4.12, 4.09, 4.13, 4.15, 4.17, 4.2])
    d = pd.DataFrame({"date": dates, "open": close - .01, "high": close + .02, "low": close - .03, "close": close,
                      "ao": close - .01, "ac": close, "dividend": 0., "volume": 100000.})
    d.loc[4, "dividend"] = .03
    dv = pd.DataFrame([{"ex_date": dates[4], "payment_date": dates[6], "cash_dividend_per_share": .03}])
    sig = pd.DataFrame([{"signal_id": "人工路径", "family": "BREAKOUT", "signal_idx": 1, "signal_date": dates[1], "stop_index": 3.}])
    v = pd.DataFrame({"common_known": [True] * 12, **{k: [False] * 12 for k in CANDIDATES}})
    decision = pd.DataFrame([{"idx": i, "family": "BREAKOUT", "PATTERN_ONLY": True} for i in range(12)])
    checks = []
    for cost in p.COSTS:
        old = p.account(d, dv, sig, decision, 0, 11, "PATTERN_ONLY", cost)
        new = simulate(p, d, dv, sig, v, 0, 11, "A", cost)
        pd.testing.assert_frame_equal(old[0], new[0][old[0].columns])
        assert np.isclose(old[1].net_pnl.sum(), new[1].net_pnl.sum())
        checks.append(f"{cost}母版账户同一及股息应收到账")
    risky = v.copy()
    risky.loc[2, "H_CALENDAR"] = True
    no_entry = simulate(p, d, dv, sig, risky, 0, 11, "H_CALENDAR", "STRESS")
    assert not len(no_entry[1]) and no_entry[0].shares.eq(0).all()
    checks.append("开盘前风险否决入场且以后不补买旧信号")
    risky.loc[2, "H_CALENDAR"] = False
    risky.loc[3, "H_CALENDAR"] = True
    early = simulate(p, d, dv, sig, risky, 0, 11, "H_CALENDAR", "STRESS")
    assert int(early[1].entry_idx.iloc[0]) == 2 and int(early[1].exit_idx.iloc[0]) == 3
    checks.append("公开后首个可执行开盘卖出仍承担买入日隔夜")
    missing = v.copy()
    missing.loc[3:, "common_known"] = False
    hold = simulate(p, d, dv, sig, missing, 0, 11, "H_CALENDAR", "STRESS")
    assert int(hold[1].exit_idx.iloc[0]) == 7
    checks.append("持有期间缺信息不制造避险卖出")
    missing.loc[2, "common_known"] = False
    assert not len(simulate(p, d, dv, sig, missing, 0, 11, "A_MATCH", "STRESS")[1])
    checks.append("同覆盖基准使用相同缺信息入场限制")
    return checks


def freeze(root=ROOT):
    if (root / "freeze.json").exists():
        raise RuntimeError("本轮已经冻结，禁止覆盖。")
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    sources = []
    for name, path in SOURCES.items():
        source = WORKSPACE / path
        shutil.copy2(source, root / "inputs" / name)
        sources.append({"snapshot": f"inputs/{name}", "original_path": path})
    shutil.copy2(WORKSPACE / "reports/research/510300_all_research_abcd_increment_v1/code/parent_engine.py", root / "code/parent_engine.py")
    shutil.copy2(Path(__file__), root / "code/holiday_event_capital_risk_v1.py")
    p = parent(root)
    checks = mechanism_checks(p)
    preview_data, _, _ = load(root)
    preview_views, _ = build_views(root, preview_data)
    assert len(preview_views) == len(preview_data)
    checks.append("风险特征与来源时钟可完整构造，未运行历史账户")
    protocol = {
        "study_id": "510300_HOLIDAY_EVENT_CAPITAL_RISK_V1", "frozen_at": p.now(),
        "user_direction": "综合宏观、微观资金与技术形态追求510300高夏普，新增节日及重大事件资金避险；不用审核包和汇总表格。",
        "periods": PERIODS, "policies": POLICIES, "primary_candidates": CANDIDATES,
        "clock": "形态和原退出仍为收盘判断次开盘；新增风险在执行日09:00判断，同日09:30尝试。",
        "macro": "DR007相对7天逆回购政策利率>0且已知DR007相对五交易日前上升。DR007次交易日09:30才视作已知，09:00判断因此多保留一个时段；政策利率用原published_at。",
        "micro": "官方月度权重聚合的大单超大单成交分类强度，最近连续五日均值<0且正强度权重占比五日均值<50%。覆盖>=98%，权重快照严格早于观察日，同日快照置缺并重算。微观数据同样次交易日09:30视作已知。不是资金所有人身份或真实净流出。",
        "global": "标普500日对数收益/此前20日波动<=-2，或VIX日对数变化/此前20日波动>=2。每条美东17:00才视作可用，转换北京时间并与09:00对齐；最大信息年龄7自然日。",
        "calendar": "相邻开市日日期间隔>=4自然日，休市前三个交易日的09:00为固定节前窗口；不依据收益选择节日。2021—2025为既有历史开市日重建，尚无逐年首发通知认证；2026沿用本地官方年度通知。",
        "H_CALENDAR": "同覆盖资料已知时，所有节前三日否决新入场并退出既有可卖份额。",
        "H_PRESSURE": "节前三日且宏观资金收紧且微观成交分类承压时避险。",
        "EVENT_REACTION": "海外冲击已经公开且宏观收紧或微观承压，执行风险退出/入场否决；不宣称预知战争、会议、政策新闻。",
        "COMBINED": "H_PRESSURE或EVENT_REACTION；唯一预先固定联合，不追认事后赢家。",
        "missing": "四候选及A_MATCH同样要求入场时全部资料可用；持仓后缺值不强行卖出，原形态止损和五日退出仍有效。最长DR与微观年龄14自然日。",
        "account": {"capital_cny": 200000, "assets": ["510300.SH", "CASH_CNY"], "costs": p.COSTS, "cash_yield": 0, "annual_days": 252,
                    "lot": 100, "minimum_commission": 5, "T_plus_1": True, "reenter_old_signal": False},
        "inference": "四候选与A_MATCH主压力账户作2000次20日联合循环区块重抽；单侧alpha=.05/4。同期数字门及增量均通过也仅为本轮历史探索。",
        "target": {"net_sharpe": 1.2, "net_cagr": .10, "max_drawdown": .10},
        "search": {"fits": 0, "threshold_grid": 0, "horizon_grid": 0, "direction_changes_allowed": False},
        "historical_source_first_versions_authenticated": False,
        "independent_validation": False, "new_market_data_collection": False, "orders": False, "make_review_package": False,
        "old_rejections_preserved": ["510300_CALENDAR_LIQUIDITY_TIMING_V1", "510300_THURSDAY_WEEKLY_REDUCTION_V1", "510300_ALL_RESEARCH_ABCD_INCREMENT_V1"],
        "terminated_strategy_revived": False, "sources": sources,
    }
    p.save_json(root / "protocol.json", protocol)
    artifacts = [root / "protocol.json", *sorted(x for x in (root / "inputs").iterdir() if x.is_file()), *sorted(x for x in (root / "code").iterdir() if x.is_file())]
    p.save_json(root / "freeze.json", {"frozen_at": p.now(), "mechanism_checks": checks,
                "files": [{"path": x.relative_to(root).as_posix(), "sha256": p.digest(x)} for x in artifacts]})
    print("已冻结：四项固定资金避险规则、同覆盖基准及两档成本；没有参数搜索。")


def check_freeze(root, p):
    import json
    record = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        assert p.digest(root / item["path"]) == item["sha256"], item["path"]


def compare(root, p, accounts):
    rng = np.random.default_rng(202609241)
    baseline = accounts[("PRIMARY", "STRESS", "A_MATCH")]
    n = len(baseline[0])
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.save(root / "results/bootstrap_indices.npy", indices)
    base = p.reserved_returns(baseline[0], baseline[3])
    rows = []
    for policy in CANDIDATES:
        current = accounts[("PRIMARY", "STRESS", policy)]
        r = p.reserved_returns(current[0], current[3])
        delta = r - base
        samples = delta[indices].mean(axis=1) * 252
        point = float(delta.mean() * 252)
        lo, hi = np.quantile(samples, [.025, .975])
        adjusted_low = float(np.quantile(samples, .05 / len(CANDIDATES)))
        rows.append({"policy": policy, "baseline": "A_MATCH", "annual_mean_increment": point, "ci95": [lo, hi],
                     "familywise_one_sided_lower": adjusted_low, "supports_positive_increment": adjusted_low > 0})
    p.save_json(root / "results/paired_increment.json", rows)
    return rows


def mechanism_diagnostics(root, p, d, signals, v, accounts):
    rows = []
    by_signal = signals.set_index("signal_id")
    for policy in CANDIDATES:
        ledger, trades, decisions, terminal = accounts[("PRIMARY", "STRESS", policy)]
        skipped = decisions.loc[decisions.reason.eq("RISK_ENTRY_REJECTED")]
        for item in skipped.to_dict("records"):
            sig = by_signal.loc[item["signal_id"]].to_dict()
            sig["signal_id"] = item["signal_id"]
            outcome = p.trade_outcome(d, sig, "STRESS")
            if outcome.get("exit_date", "9999") > PERIODS["PRIMARY"][1]:
                outcome["label_status"] = "PENDING_AT_RESEARCH_END"
                outcome["net_return"] = None
            rows.append({"policy": policy, "decision_date": item["date"], "action": "ENTRY_REJECTED", **outcome})
    pd.DataFrame(rows).to_parquet(root / "results/rejected_entry_counterfactuals.parquet", index=False)
    # 仅作为机制描述：当天开盘至下一开盘可执行持有期间，而非前夜已经发生的跌幅。
    lead = (d.open.shift(-1) + d.dividend.shift(-1)) / d.open - 1
    population = v.date.between(*PERIODS["PRIMARY"]) & v.common_known & v.date.shift(-1).le(PERIODS["PRIMARY"][1])
    summaries = []
    for name, mask in [("COMMON_DAYS", population), *[(k, population & v[k]) for k in CANDIDATES]]:
        returns = lead.loc[mask].dropna()
        summaries.append({"condition": name, "n": len(returns), "mean_next_open_return": returns.mean(),
                          "loss_rate": returns.lt(0).mean(), "tail_below_minus_2pct": returns.le(-.02).mean(),
                          "interpretation": "全部日期的描述性机制对照，不是可独立执行策略或因果结论。"})
    p.save_json(root / "results/risk_window_diagnostics.json", summaries)


def run(root=ROOT):
    p = parent(root)
    check_freeze(root, p)
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("本轮已有运行记录，不能以改参数重跑覆盖。")
    p.save_json(root / "RUN_STARTED.json", {"started_at": p.now()})
    out = root / "results"
    out.mkdir(exist_ok=True)
    d, signals, dividends = load(root)
    views, closures = build_views(root, d)
    views.to_parquet(out / "preopen_views.parquet", index=False)
    closures.to_parquet(out / "holiday_episodes.parquet", index=False)
    decisions = pd.DataFrame([{"idx": i, "family": fam, "PATTERN_ONLY": True} for i in range(len(d)) for fam in p.FAMILIES])
    accounts, metrics, verification = {}, [], []
    for period, (lo, hi) in PERIODS.items():
        start = int(d.index[d.date.ge(lo)][0])
        end = int(d.index[d.date.le(hi)][-1])
        for cost in p.COSTS:
            for policy in POLICIES:
                if policy in ("BUY_HOLD", "CASH"):
                    result = p.account(d, dividends, signals, decisions, start, end, policy, cost)
                else:
                    result = simulate(p, d, dividends, signals, views, start, end, policy, cost)
                accounts[(period, cost, policy)] = result
                ledger, trades, logs, terminal = result
                folder = out / "accounts" / period / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                for name, frame in [("ledger", ledger), ("trades", trades), ("decisions", logs)]:
                    frame.to_parquet(folder / f"{name}.parquet", index=False)
                m = {"period": period, "cost": cost, "policy": policy, **p.metrics(ledger, trades, terminal)}
                if len(trades):
                    m["largest_cycle_cny"] = float(trades.net_pnl.max())
                    m["profit_without_largest_cycle_cny"] = m["net_profit_cny"] - m["largest_cycle_cny"]
                p.save_json(folder / "metrics.json", m)
                p.save_json(folder / "terminal.json", terminal)
                metrics.append(m)
                assert np.allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7)
                if len(trades):
                    assert (trades.exit_idx > trades.entry_idx).all()
                if not terminal["open_position"]:
                    assert np.isclose(trades.net_pnl.sum() if len(trades) else 0., m["net_profit_cny"], atol=1e-6)
                if policy == "A":
                    original = p.account(d, dividends, signals, decisions, start, end, "PATTERN_ONLY", cost)
                    pd.testing.assert_frame_equal(original[0], ledger[original[0].columns])
                    verification.append(f"{period}_{cost}_原形态逐日等价")
            print(f"已完成{period}、{cost}八个完整账户。", flush=True)
    p.save_json(out / "account_metrics.json", metrics)
    paired = compare(root, p, accounts)
    mechanism_diagnostics(root, p, d, signals, views, accounts)
    # 截断未来市场输入只重算已知信息。日历仍是预先声明的日历输入，不假装是行情。
    for cutoff in ("2022-06-30", "2024-09-30", "2025-12-31"):
        subset = d.loc[d.date.le(cutoff)].copy()
        partial, _ = build_views(root, subset)
        pd.testing.assert_frame_equal(views.iloc[:len(partial)].reset_index(drop=True), partial.reset_index(drop=True))
        verification.append(f"{cutoff}_未来行情截断后风险判断不变")
    p.save_json(root / "verification.json", {"status": "PASS", "checked_at": p.now(), "account_count": len(metrics),
                "account_rows": sum(len(x[0]) for x in accounts.values()), "checks": verification,
                "clock_tests": "全部来源可用时点<=09:00；资金及微观保留下一开盘发布约束。", "independent_scientific_validation": False})
    passes = [k for k in CANDIDATES if all(next(x for x in metrics if x["period"] == "PRIMARY" and x["cost"] == c and x["policy"] == k)["numerical_target_pass"] for c in p.COSTS)]
    supported = [x["policy"] for x in paired if x["supports_positive_increment"]]
    common = views.loc[views.date.between(*PERIODS["PRIMARY"])]
    p.save_json(root / "result.json", {"status": "FROZEN_HISTORICAL_CANDIDATE_ONLY" if passes and supported else "FROZEN_NO_RELIABLE_CAPITAL_RISK_INCREMENT",
                "completed_at": p.now(), "primary_dual_cost_numeric_pass": passes, "positive_increment_supported": supported,
                "common_days": int(common.common_known.sum()), "primary_days": len(common),
                "risk_days": {k: int(common[k].sum()) for k in CANDIDATES}, "strategy_goal_achieved": False,
                "calendar_source_first_versions_authenticated": False, "strict_forward_observations": 0,
                "scheduled_major_news_anticipation": "NOT_TESTED_NO_COMPLETE_PRIOR_ANNOUNCEMENT_LEDGER",
                "risk_reaction_scope": "海外已发生价格波动与国内资金代理的联动，不覆盖所有重大事件。",
                "collection_started": False, "review_package_created": False, "terminated_strategy_revived": False})
    print("本轮资金避险已完成；结果与相同数据覆盖对照一并保留，没有选择近期最好窗口作为结论。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.root.resolve())


if __name__ == "__main__":
    main()
