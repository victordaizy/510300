"""T06只读复核：分类源算术、同成员中位数、时钟、削减状态及保存账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tol=1e-10):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float),
                               atol=tol, rtol=0, equal_nan=True)


def exact_baseline_relation(market):
    """独立从当前财富向前回推，比较20与历史相对财富之和。"""
    price = market.close.to_numpy(dtype=float)
    div = market.dividend.to_numpy(dtype=float)
    assert np.allclose(price*1000, np.rint(price*1000), rtol=0, atol=1e-7)
    quotes = [Fraction(int(round(v*1000)), 1000) for v in price]
    cash = [Fraction(str(float(v))) for v in div]
    out = [pd.NA]*len(price)
    for t in range(19, len(price)):
        relative, total = Fraction(1), Fraction(1)
        for j in range(t, t-19, -1):
            relative *= quotes[j-1]/(quotes[j]+cash[j])
            total += relative
        difference = Fraction(20)-total
        out[t] = 1 if difference > 0 else (-1 if difference < 0 else 0)
    return pd.Series(out, dtype="Int8", name="ma20_exact_relation")


def check_features(root, market):
    """每个窗口直接相乘，与研究代码滚动对数和交叉核对。"""
    classified = pd.read_parquet(root/"inputs/classified.parquet")
    traded = classified.return_is_usable & classified.constituent_return_state.eq("TRADED_VALID")
    suspended = classified.return_is_usable & classified.constituent_return_state.eq("OFFICIAL_SUSPENSION")
    q = classified.loc[traded]
    derived = (q.unadjusted_close*q.post_to_pre_share_ratio+q.cash_distribution_per_pre_event_share
               -q.subscription_cash_outflow_per_pre_event_share)/q.previous_unadjusted_close-1
    close(q.daily_total_shareholder_return, derived, 1e-12)
    assert classified.loc[suspended, "corporate_action_status"].eq("NONE_CONFIRMED").all()
    assert classified.loc[suspended, "previous_unadjusted_close"].gt(0).all()
    assert classified.loc[suspended, "official_suspension"].all()
    close(classified.loc[suspended, "daily_total_shareholder_return"], 0., 0.)
    values = pd.Series(np.nan, index=classified.index)
    values.loc[traded] = derived
    values.loc[suspended] = 0.
    allowed_return = np.isfinite(values) & values.gt(-1)
    membership = pd.read_parquet(root/"inputs/membership.parquet").rename(columns={"membership_date": "date"})
    assert not membership.duplicated(["date", "symbol"]).any()
    assert membership.groupby("date").symbol.nunique().eq(300).all()
    dates, symbols = pd.DatetimeIndex(market.date), sorted(membership.symbol.unique())
    returns = classified.assign(value=values.where(allowed_return)).pivot(index="date", columns="symbol", values="value")
    returns = returns.reindex(index=dates, columns=symbols)
    saved_returns = pd.read_parquet(root/"constituent_returns.parquet")
    assert returns.index.equals(saved_returns.index) and returns.columns.equals(saved_returns.columns)
    close(returns, saved_returns, 1e-12)
    members = membership.assign(value=True).pivot(index="date", columns="symbol", values="value")
    members = members.reindex(index=dates, columns=symbols).eq(True)
    saved_members = pd.read_parquet(root/"membership_mask.parquet")
    assert members.index.equals(saved_members.index) and members.columns.equals(saved_members.columns)
    assert np.array_equal(members.to_numpy(), saved_members.to_numpy())
    r = returns.to_numpy()
    r20 = np.full(r.shape, np.nan)
    valid = np.zeros(r.shape, bool)
    for t in range(19, len(r)):
        valid[t] = np.isfinite(r[t-19:t+1]).all(axis=0)
        r20[t, valid[t]] = np.prod(1+r[t-19:t+1, valid[t]], axis=0)-1
    close(pd.read_parquet(root/"constituent_return20.parquet"), r20, 1e-12)
    mask = np.zeros(r.shape, bool)
    mask[5:] = members.to_numpy()[5:] & members.to_numpy()[:-5] & valid[5:] & valid[:-5]
    assert np.array_equal(mask, pd.read_parquet(root/"common_member_mask.parquet").to_numpy())
    coverage = pd.read_parquet(root/"inputs/daily_coverage.parquet").set_index("date")
    allowed = coverage.aggregation_state.reindex(dates).eq("VIEW_ALLOWED").to_numpy()
    internal = pd.read_parquet(root/"internal_features.parquet")
    assert internal.date.equals(market.date)
    for t, row in enumerate(internal.itertuples()):
        n = int(mask[t].sum())
        now_median = float(np.median(r20[t, mask[t]])) if n else np.nan
        past_median = float(np.median(r20[t-5, mask[t]])) if n else np.nan
        known = bool(t >= 5 and n >= 294 and members.iloc[t].sum() == 300
                     and members.iloc[t-5].sum() == 300 and allowed[t] and allowed[t-5])
        assert row.common_members == n and row.internal_known == known
        close([row.median20_current, row.median20_prior5_same_members, row.median20_change5],
              [now_median, past_median, now_median-past_median], 1e-12)
        assert row.internal_weak == bool(known and now_median < past_median)
    log_return = np.log((market.close+market.dividend)/market.close.shift())
    wealth = np.exp(log_return.fillna(0).cumsum())
    close(internal.ETF20_minus_median, wealth/wealth.shift(20)-1-internal.median20_current)
    margin = pd.read_parquet(root/"inputs/margin.parquet").set_index("date").market_rzye.reindex(dates).to_numpy()
    finance = pd.read_parquet(root/"financing_features.parquet")
    f5, quantile = np.full(len(market), np.nan), np.full(len(market), np.nan)
    for t in range(len(market)):
        if t >= 5 and np.isfinite(margin[t-5:t+1]).all():
            f5[t] = margin[t]/margin[t-5]-1
        sample = f5[max(0, t-252):t]
        sample = sample[np.isfinite(sample)]
        if len(sample) >= 120:
            quantile[t] = np.quantile(sample, .8)
    close(finance.F5, f5)
    close(finance.F5_q80, quantile)
    r5 = wealth/wealth.shift(5)-1
    close(finance.r5, r5)
    known = np.isfinite(f5) & np.isfinite(quantile) & np.isfinite(r5)
    crowded = known & (f5 > quantile) & (r5 <= 0)
    assert np.array_equal(finance.finance_known, known)
    assert np.array_equal(finance.finance_crowded, crowded)
    features = {}
    previous_price = pd.read_parquet(root/"inputs/price_features.parquet")
    ma20 = wealth.rolling(20, min_periods=20).mean()
    for lag in [1, 2]:
        x = pd.read_parquet(root/f"daily_features_lag{lag}.parquet")
        assert x.date.equals(market.date)
        close(x.wealth, wealth)
        close(x.ma20, ma20)
        exact_relation = exact_baseline_relation(market)
        assert x.ma20_exact_relation.equals(exact_relation)
        assert x.price_long.equals(exact_relation.eq(1).fillna(False).astype(bool).rename("price_long"))
        close(x.es95, previous_price.es95)
        assert x.stat_date.equals(market.date.shift(lag).rename("stat_date"))
        assert x.stat_idx.tolist() == [-1]*lag+list(range(len(x)-lag))
        for original in [internal, finance]:
            for col in original.columns.drop("date"):
                expected = original[col].shift(lag)
                if original[col].dtype == bool:
                    expected = expected.fillna(False).astype(bool)
                    assert x[col].equals(expected), col
                else:
                    close(x[col], expected)
        assert x.common_known.equals((x.finance_known & x.internal_known).rename("common_known"))
        features[lag] = x
    return features, {"classified_traded_rows": int(traded.sum()), "official_suspension_rows": int(suspended.sum()),
                      "constituent_return_cells": int(returns.size), "internal_feature_days": len(internal)}


def friction_price(reference, side, slip=.001):
    value = reference*(1+side*slip)/.001
    return (math.ceil(value-1e-10) if side > 0 else math.floor(value+1e-10))*.001


def fee(quantity, px, rate=.0004):
    return max(abs(quantity)*px*rate, 5.) if quantity else 0.


def risk_quantity(cash, shares, receivable, reference, peak, es):
    """从预算不等式寻找最大整手，不调用研究目标函数。"""
    nav = cash+shares*reference+receivable
    if not np.isfinite(es) or nav <= 0:
        return 0
    remaining = max(0., nav-.9*peak)
    cap = min(.5, .025/max(es, 1e-12), .5*remaining/(.1*nav))
    maximum = math.floor(max(0., cap)*nav/reference/100)*100
    for target in range(maximum, 0, -100):
        change = target-shares
        px = friction_price(reference, 1 if change > 0 else -1)
        friction = abs(change)*abs(px-reference)+fee(change, px)
        sell = friction_price(reference, -1)
        reserve = target*(reference-sell)+fee(target, sell)
        if (target*reference*es+friction+reserve <= .025*nav+1e-8
            and target*reference*.1+friction+reserve <= min(.05*nav, .5*remaining)+1e-8
            and target*reference <= .5*(nav-friction)+1e-8):
            return target
    return 0


def independent_metrics(z, capital):
    nav = np.r_[capital, z.equity.to_numpy()]
    r = nav[1:]/nav[:-1]-1
    sd = r.std(ddof=1)
    return {"days": len(z), "end_equity": nav[-1], "net_profit": nav[-1]-capital,
            "net_sharpe": np.sqrt(242)*r.mean()/sd if sd > 1e-15 else np.nan,
            "sharpe_252_diagnostic": np.sqrt(252)*r.mean()/sd if sd > 1e-15 else np.nan,
            "cagr": (nav[-1]/capital)**(242/len(r))-1,
            "cagr_252_diagnostic": (nav[-1]/capital)**(252/len(r))-1,
            "max_drawdown": 1-np.min(nav/np.maximum.accumulate(nav)), "mean_exposure": z.exposure.mean(),
            "fills": z.filled_quantity.ne(0).sum(), "commission": z.commission.sum(), "slippage": z.slippage_cost.sum(),
            "terminal_exit_reserve": z.terminal_exit_reserve.iloc[-1], "max_identity_error": z.accounting_error.abs().max()}


def check_state_and_orders(market, z, d, source, capital, cost, policy):
    base, active, pending, stopped = False, False, False, False
    ceiling, streak, cycle, entry = 0, 0, 0, -1
    peak = float(capital)
    lots = []
    rate, slip = (.0002, .0005) if cost == "BASE" else (.0004, .001)
    for j, row in enumerate(z.itertuples()):
        decision, f = d.iloc[j], source.iloc[j]
        idx, old = int(row.idx), int(row.shares_before)
        assert bool(decision.base_live_before) == base and bool(decision.overlay_before) == active
        assert int(decision.base_ceiling_before) == ceiling and int(decision.clear_streak_before) == streak
        # cash_before 已计入当日盘前到期分红，尚未计入当日收盘付款。
        cash_before = float(row.cash_before)
        prior_cash = float(z.cash.iloc[j-1]) if j else float(capital)
        prior_receivable = float(z.dividend_receivable.iloc[j-1]) if j else 0.
        late_payment_before_open = cash_before-prior_cash
        old_receivable = prior_receivable+row.dividend_recognized-late_payment_before_open
        ref = float(market.close.iloc[idx-1]-market.dividend.iloc[idx])
        risk = risk_quantity(cash_before, old, old_receivable, ref, peak, float(f.es95))
        assert int(decision.risk_target) == risk, (j, "风险份额")
        can_enter = bool(f.price_long and (policy == "PRICE_ALL" or f.common_known))
        assert bool(decision.can_enter) == can_enter
        requested, initial = 0, False
        if base:
            pending |= bool(stopped or not f.price_long or idx >= entry+5)
            if pending:
                requested = -old
            else:
                ceiling = min(ceiling, risk)
                if policy in ["PRICE_ALL", "PRICE_COMMON"]:
                    active, streak = False, 0
                elif not f.common_known:
                    streak = 0
                else:
                    conditions = [bool(f.finance_crowded), bool(f.internal_weak)]
                    relevant = conditions if policy == "FULL" else [conditions[0 if policy == "FINANCE_ONLY" else 1]]
                    if active:
                        streak = streak+1 if not any(relevant) else 0
                        if streak >= 2:
                            active, streak = False, 0
                    else:
                        active, streak = all(relevant), 0
                requested = (ceiling//200*100 if active else ceiling)-old
        elif not stopped and can_enter and j < len(z)-1:
            requested, initial = risk, True
        assert requested == decision.pre_open_request and initial == bool(decision.initial_entry)
        if requested > 0:
            cap_at_open = risk_quantity(cash_before, old, old_receivable, row.open, peak, float(f.es95))
            if base:
                ceiling = min(ceiling, cap_at_open)
                requested = max(0, min(requested, (ceiling//200*100 if active else ceiling)-old))
            else:
                requested = min(requested, cap_at_open)
        assert requested == decision.requested_quantity == row.requested_quantity
        sellable = sum(q for bought, q in lots if bought < idx)
        assert row.sellable_before == sellable
        actual, status = 0, "NO_TRADE"
        if requested:
            side = 1 if requested > 0 else -1
            px = friction_price(row.open, side, slip)
            basis = float(market.previous_close.iloc[idx]-market.dividend.iloc[idx])
            lower = math.floor(basis*.9/.001+.5+1e-9)*.001
            upper = math.floor(basis*1.1/.001+.5+1e-9)*.001
            blocked = (side > 0 and (row.open >= upper-1e-9 or px > upper+1e-9)) or (side < 0 and (row.open <= lower+1e-9 or px < lower-1e-9))
            if blocked:
                status = "UNFILLED_DIRECTIONAL_LIMIT"
            else:
                if requested > 0:
                    affordable = int(max(0, math.floor((cash_before+1e-9)/(px*100))))*100
                    while affordable and affordable*px+fee(affordable, px, rate) > cash_before+1e-8:
                        affordable -= 100
                    actual = min(requested, affordable)
                else:
                    actual = -min(-requested, sellable)
                status = "UNFILLED_CASH_OR_T_PLUS_ONE" if actual == 0 else ("FILLED" if actual == requested else "PARTIALLY_FILLED_CASH_OR_T_PLUS_ONE")
        assert row.filled_quantity == actual and row.status == status, (j, "模拟成交")
        if actual > 0:
            lots.append((idx, actual))
        elif actual < 0:
            left, updated = -actual, []
            for bought, quantity in lots:
                sold = min(left, quantity) if bought < idx else 0
                left -= sold
                if quantity > sold:
                    updated.append((bought, quantity-sold))
            assert left == 0
            lots = updated
        assert sum(q for _, q in lots) == row.shares
        if initial and actual > 0:
            base, ceiling, active, streak, entry = True, int(row.shares), False, 0, idx
            cycle += 1
        assert bool(decision.base_live) == base and bool(decision.overlay_active) == active
        assert decision.base_ceiling == ceiling and decision.clear_streak == streak
        assert bool(decision.base_exit_pending) == pending and decision.cycle_no == cycle and decision.base_entry_idx == entry
        peak = max(peak, row.equity)
        stopped |= row.equity/peak <= .9
        assert bool(row.risk_stopped) == stopped
        if pending and row.shares == 0:
            base, pending, active, streak, ceiling, entry = False, False, False, 0, 0, -1


def check_accounts(root, market, features):
    table, annual = pd.read_csv(root/"metrics.csv"), pd.read_csv(root/"annual_metrics.csv")
    overlays = pd.read_csv(root/"overlay_counts.csv")
    assert len(table) == 56 and not table.duplicated(["period", "capital", "cost", "lag", "policy"]).any()
    dividends = pd.read_csv(root/"inputs/dividends.csv", parse_dates=["record_date", "ex_date", "payment_date"])
    returns, rows = {}, 0
    for ordinal, item in enumerate(table.itertuples(), 1):
        folder = root/f"accounts/{item.period}/{item.capital}/{item.cost}/LAG{item.lag}/{item.policy}"
        z, d = pd.read_parquet(folder/"ledger.parquet"), pd.read_parquet(folder/"decisions.parquet")
        source = features[item.lag].iloc[d.origin_idx].reset_index(drop=True)
        assert z.date.equals(d.date) and source.date.equals(d.origin.rename("date"))
        assert (d.origin < d.date).all() and (d.stat_date.dropna() < d.origin[d.stat_date.notna()]).all()
        assert d.origin_idx.tolist() == (z.idx-1).tolist()
        assert d.stat_idx.equals(source.stat_idx)
        for col in ["price_long", "common_known", "finance_known", "internal_known", "finance_crowded", "internal_weak"]:
            assert d[col].equals(source[col]), col
        for col in ["F5", "F5_q80", "r5", "median20_change5", "common_members"]:
            close(d[col], source[col])
        close(d.es95_5d, source.es95)
        close(z.filled_quantity, d.filled_quantity, 0.)
        nav = np.r_[item.capital, z.equity]
        close(z.net_return, nav[1:]/nav[:-1]-1, 1e-12)
        close(z.equity, z.cash+z.shares*z.mark+z.dividend_receivable-z.terminal_exit_reserve, 1e-7)
        close(z.shares, z.filled_quantity.cumsum(), 0.)
        close(z.shares_before, z.shares.shift(fill_value=0), 0.)
        close(z.cash, item.capital+(-z.filled_quantity*z.fill_price.fillna(0)-z.commission+z.dividend_paid).cumsum(), 1e-7)
        close(z.exposure, z.shares*z.mark/z.equity)
        close(z.drawdown, 1-nav[1:]/np.maximum.accumulate(nav)[1:], 1e-12)
        prior_mark = np.r_[market.close.iloc[int(z.idx.iloc[0])-1], z.mark.to_numpy()[:-1]]
        pnl = z.shares_before*(z.open-prior_mark)+z.shares*(z.mark-z.open)
        close(z.price_pnl, pnl, 1e-7)
        reserve_change = z.terminal_exit_reserve.diff().fillna(z.terminal_exit_reserve.iloc[0])
        close(np.diff(nav), pnl+z.dividend_recognized-z.commission-z.slippage_cost-reserve_change, 1e-7)
        assert z.cash.min() >= -1e-7 and z.shares.min() >= 0 and z.shares.mod(100).eq(0).all()
        traded = z.filled_quantity.ne(0)
        rate, slip = (.0002, .0005) if item.cost == "BASE" else (.0004, .001)
        close(z.commission, np.where(traded, np.maximum(abs(z.filled_quantity)*z.fill_price.fillna(0)*rate, 5), 0), 1e-9)
        for row in z.loc[traded].itertuples():
            close(row.fill_price, friction_price(row.open, 1 if row.filled_quantity > 0 else -1, slip))
        close(z.slippage_cost, abs(z.filled_quantity)*abs(z.fill_price.fillna(0)-z.open), 1e-8)
        assert z.terminal_exit_reserve.iloc[:-1].eq(0).all()
        last = z.iloc[-1]
        sell = friction_price(last.mark, -1)
        close(last.terminal_exit_reserve, last.shares*(last.mark-sell)+fee(last.shares, sell), 1e-9)
        recognized, paid = np.zeros(len(z)), np.zeros(len(z))
        for event in dividends.itertuples():
            held = z.loc[z.date.eq(event.record_date), "shares"]
            amount = float(held.iloc[0])*event.cash_dividend_per_share if len(held) else 0.
            recognized[z.date.eq(event.ex_date)] += amount
            payment = np.flatnonzero(z.date.ge(event.payment_date).to_numpy())
            if len(payment):
                paid[payment[0]] += amount
        close(z.dividend_recognized, recognized, 1e-7)
        close(z.dividend_paid, paid, 1e-7)
        close(z.dividend_receivable, np.cumsum(recognized-paid), 1e-7)
        check_state_and_orders(market, z, d, source, item.capital, item.cost, item.policy)
        for col, value in independent_metrics(z, item.capital).items():
            close(getattr(item, col), value)
        match = (annual.period == item.period) & (annual.capital == item.capital) & (annual.cost == item.cost) & (annual.lag == item.lag) & (annual.policy == item.policy)
        prior = item.capital
        for year, part in z.groupby(z.date.dt.year):
            saved = annual[match & annual.year.eq(year)]
            assert len(saved) == 1
            for col, value in independent_metrics(part, prior).items():
                close(saved.iloc[0][col], value)
            prior = part.equity.iloc[-1]
        matched = overlays[(overlays.period == item.period) & (overlays.capital == item.capital) & (overlays.cost == item.cost) & (overlays.lag == item.lag) & (overlays.policy == item.policy)]
        assert len(matched) == 1
        saved = matched.iloc[0]
        assert saved.cut_episodes == (d.overlay_active & ~d.overlay_before).sum()
        assert saved.restore_episodes == (~d.overlay_active & d.overlay_before & ~d.base_exit_pending).sum()
        assert saved.reduced_decision_days == d.overlay_active.sum()
        assert saved.initial_base_cycles == d.cycle_no.max()
        assert saved.reduced_zero_share_days == (d.overlay_active & z.shares.eq(0)).sum()
        closed = ((z.shares_before > 0) & z.shares.eq(0)).sum()
        assert closed == item.closed_cycles
        returns[item.period, item.capital, item.cost, item.lag, item.policy] = z.net_return.to_numpy()
        rows += len(z)
        if ordinal % 14 == 0:
            print(f"已复核{ordinal}份T06账户：同日来源、减半恢复、费用与退出均一致。", flush=True)
    return table, returns, rows, dividends


def check_risk(root, market, features, dividends):
    records = read(root/"inputs/risk_training_records.json")
    labels = pd.read_parquet(root/"inputs/mature_risk_labels.parquet")
    for row in labels.dropna().itertuples():
        a, b = int(row.origin_idx)+1, int(row.exit_idx)
        cash = dividends.loc[dividends.record_date.ge(market.date.iloc[a]) & dividends.record_date.lt(market.date.iloc[b]), "cash_dividend_per_share"].sum()
        close(row.gross_return5, (market.open.iloc[b]+cash)/market.open.iloc[a]-1, 1e-12)
    states = features[1][["pressure5", "trend20", "log_rv5_rv60"]].to_numpy()
    valid = np.isfinite(states).all(axis=1) & np.isfinite(labels.gross_return5)
    lookup = {r["decision_idx"]: r for r in records}
    assert len(lookup) == len(records)
    for t in range(len(market)):
        ids = np.flatnonzero(valid & (np.arange(len(market))+6 <= t) & market.date.ge(market.date.iloc[t]-pd.DateOffset(years=2)).to_numpy())
        if len(ids) < 252 or not np.isfinite(states[t]).all():
            assert t not in lookup and pd.isna(features[1].es95.iloc[t])
            continue
        record = lookup[t]
        mean, sd = states[ids].mean(axis=0), states[ids].std(axis=0, ddof=1)
        sd[sd < 1e-12] = 1.
        current = np.clip((states[t]-mean)/sd, -5, 5)
        normalized = np.clip((states[ids]-mean)/sd, -5, 5)
        distance = ((normalized-current)**2).sum(axis=1)
        selected = sorted(zip(distance.tolist(), ids.tolist()))[:126]
        indices = [i for _, i in selected]
        assert record["selected_indices"] == indices and record["training_count"] == len(ids)
        assert record["latest_training_exit_idx"] == int(ids.max()+6) <= t
        close(record["training_mean"], mean)
        close(record["training_std"], sd)
        es = max(0., -np.sort(labels.gross_return5.iloc[indices])[:math.ceil(.05*len(indices))].mean())
        close([record["es95_5d"], features[1].es95.iloc[t], features[2].es95.iloc[t]], [es, es, es], 1e-12)
    return len(records)


def verify(root):
    frozen, started = read(root/"freeze.json"), read(root/"run_started.json")
    for item in frozen["files"]:
        assert digest(root/item["path"]) == item["sha256"], item["path"]
    assert started["freeze_sha256"] == digest(root/"freeze.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(started["at"])
    market = pd.read_parquet(root/"inputs/market.parquet")
    market = market[market.date.le("2025-12-31")].reset_index(drop=True)
    features, source_counts = check_features(root, market)
    print("分类收益、同成员中位数、融资分位和延迟时钟均已复算。", flush=True)
    table, returns, rows, dividends = check_accounts(root, market, features)
    risk_n = check_risk(root, market, features, dividends)
    indices = np.load(root/"bootstrap_indices.npz")["indices"]
    full = returns["MAIN", 200000, "STRESS", 1, "FULL"]
    assert indices.shape == (4000, len(full)) and indices.min() >= 0 and indices.max() < len(full)
    for other, result in zip(["PRICE_COMMON", "FINANCE_ONLY", "INTERNAL_ONLY"], read(root/"paired_increment.json"), strict=True):
        assert result["comparison"] == "FULL_MINUS_"+other
        difference = full-returns["MAIN", 200000, "STRESS", 1, other]
        draws = difference[indices].mean(axis=1)*242
        close([result["annual_arithmetic_increment"], result["ci95_low"], result["ci95_high"]],
              [difference.mean()*242, *np.quantile(draws, [.025, .975])], 1e-12)
    primary = table[(table.period == "MAIN") & (table.cost == "STRESS") & (table.lag == 1) & (table.policy == "FULL")]
    assert len(primary) == 2
    passed = bool(((primary.net_sharpe >= 1.2) & (primary.cagr >= .1) & (primary.max_drawdown <= .1)).all())
    result = read(root/"result.json")
    assert passed == result["historical_joint_point_pass"] and result["new_accounts"] == len(table)
    assert not result["goal_achieved"] and result["new_entry_strategies"] == 0 and result["new_primary_overlays"] == 1
    return {"status": "PASS_SAVED_CROWDING_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION", "ledgers": len(table), "ledger_rows": rows,
            **source_counts, "mature_risk_records": risk_n, "new_accounts": 0, "new_random_draws": 0,
            "network_requests": 0, "external_review": "NOT_PERFORMED", "independent_forward_validation": False,
            "scope": "从固定分类源交叉复算收益、同成员中位数和所有保存账户；不认证历史首次发布版本，不构成策略有效或外部审阅。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
