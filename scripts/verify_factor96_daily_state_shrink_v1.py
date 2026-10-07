"""从冻结输入复核校准时钟和保存账本，不生成新研究账户或访问网络。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def near(a, b, atol=1e-11):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-11, equal_nan=True)


def coefficients(raw, target, ids, row):
    x, y = raw[ids], target[ids]
    mean = x.sum(axis=0)/len(ids)
    scale = np.sqrt(((x-mean)**2).sum(axis=0)/len(ids))
    scale[scale <= 1e-12] = 1.
    z = (x-mean)/scale
    intercept = y.sum()/len(ids)
    slope = (z*(y-intercept)[:, None]).sum(axis=0)/(np.sum(z*z, axis=0)+len(ids))
    variance = np.sum((y-intercept)**2)/(len(ids)-1)
    for actual, expected in [(row["mean"], mean), (row["scale"], scale), (row["slope"], slope),
                             (row["intercept"], intercept), (row["variance"], variance)]:
        near(actual, expected)
    return mean, scale, slope, intercept, variance


def check_models(root, d, price, cumulative, members, coverage, margin, labels, lag):
    f = pd.read_parquet(root/f"daily_forecasts_lag{lag}.parquet")
    dates = pd.DatetimeIndex(d.date)
    assert f.date.equals(d.date)
    valid_members = members & cumulative.notna()
    count = valid_members.sum(axis=1)
    allowed = count.ge(294) & members.sum(axis=1).eq(300) & coverage.set_index("date").aggregation_state.reindex(dates).eq("VIEW_ALLOWED")
    breadth = ((valid_members & cumulative.gt(0)).sum(axis=1)/count.where(count.gt(0))).where(allowed)
    bal = d[["date"]].merge(margin[["date", "market_rzye"]], on="date", how="left").market_rzye
    financing = (bal/bal.shift(5)-1).where(bal.rolling(6).count().eq(6))
    er = price.wealth.diff(20)/price.wealth.diff().abs().rolling(20).sum().replace(0, np.nan)
    expected = np.column_stack([er, breadth.shift(lag), financing.shift(lag)])
    raw = f[["A01", "E01", "F01"]].to_numpy(float)
    near(raw, expected)
    assert f.features_known.eq(np.isfinite(raw).all(axis=1)).all()
    near(f.es95, price.es95)
    assert f.external_stat_idx.eq(pd.Series(np.arange(len(f))).shift(lag).fillna(-1).astype(int)).all()
    for date, stat in zip(f.date, f.external_stat_date):
        assert pd.isna(stat) or stat < date
    y = labels.gross_return5.to_numpy(float)
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(y)
    index = np.arange(len(f))
    inner = read(root/f"inner_models_lag{lag}.json")
    outer = read(root/f"outer_models_lag{lag}.json")
    groups = defaultdict(list)
    for row in inner:
        t, s = row["decision_idx"], row["validation_idx"]
        lower = dates[t]-pd.DateOffset(years=2)
        assert pd.Timestamp(row["lower"]) == lower
        assert s % 5 == 0 and s+6 < t and row["validation_exit_idx"] == s+6
        assert dates[s] >= dates[t]-pd.DateOffset(years=1)
        ids = np.flatnonzero(valid & (dates >= lower) & (index+6 < s))
        assert len(ids) == row["training_count"] >= 60
        assert [ids.min(), ids.max()] == [row["training_first_idx"], row["training_last_idx"]]
        assert hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() == row["training_ids_sha256"]
        mean, scale, slope, intercept, variance = coefficients(raw, y, ids, row)
        prediction = intercept+(raw[s]-mean)/scale*slope
        near(row["predictions"], prediction)
        near(row["target"], y[s])
        near(row["residuals"], y[s]-prediction)
        near(row["standardized_error"], (prediction.mean()-y[s])/np.sqrt(variance))
        groups[t].append(row)
    outer_ids = []
    for row in outer:
        t = row["decision_idx"]
        outer_ids.append(t)
        lower = dates[t]-pd.DateOffset(years=2)
        ids = np.flatnonzero(valid & (dates >= lower) & (index+6 < t))
        assert len(ids) >= 252 and row["training_indices"] == ids.tolist()
        assert row["latest_training_exit_idx"] == ids.max()+6 < t
        assert pd.Timestamp(row["lower"]) == lower and pd.Timestamp(row["date"]) == dates[t]
        mean, scale, slope, intercept, variance = coefficients(raw, y, ids, row)
        prediction = intercept+(raw[t]-mean)/scale*slope
        group = groups[t]
        expected_s = []
        for s in ids[(ids % 5 == 0) & (dates[ids] >= dates[t]-pd.DateOffset(years=1))]:
            training = np.flatnonzero(valid & (dates >= lower) & (index+6 < s))
            if len(training) >= 60 and np.var(y[training], ddof=1) > 1e-12:
                expected_s.append(int(s))
        assert [r["validation_idx"] for r in group] == row["inner_validation_indices"] == expected_s
        assert len(group) >= 20
        correction = np.mean([r["residuals"] for r in group], axis=0)
        shrink = len(group)/(len(group)+60.)
        calibrated = prediction+correction*shrink
        near(row["raw_predictions"], prediction)
        near(row["corrections"], correction)
        near(row["shrink"], shrink)
        near(row["calibrated_predictions"], calibrated)
        g, paused, clear, stops, restores = 0., False, 0, 0, 0
        for r in group[-20:]:
            g = max(0., g+r["standardized_error"]-.25)
            if not paused and g >= 5:
                paused, clear, stops = True, 0, stops+1
            elif paused:
                clear = clear+1 if g <= 2.5 else 0
                if clear >= 5:
                    paused, clear, restores = False, 0, restores+1
        near(row["cusum"], g)
        assert [row[k] for k in ["monitor_paused", "monitor_clear", "monitor_stop_count", "monitor_restore_count"]] == [paused, clear, stops, restores]
        spread = np.std(calibrated)/np.sqrt(variance)
        agreement = np.mean(calibrated > .0028)
        net = float(calibrated.mean()-.0028)
        scaled = net*agreement/(1+spread) if net > 0 else net
        mus = {"NO_STATE": float(intercept-.0028), "STATIC_EQUAL": float(prediction.mean()-.0028),
            "CALIBRATED": net, "DISAGREEMENT": scaled, "MONITORED": 0. if paused else net, "FULL": 0. if paused else scaled}
        for name, mu in mus.items():
            near(row["policy_net_expectations"][name], mu)
            units = int((Decimal(str(row["policy_net_expectations"][name]))/(Decimal(4)*Decimal(str(row["variance"]))) * 8).to_integral_value(rounding=ROUND_FLOOR))
            fraction = min(4, max(0, units))/8
            assert fraction == row["fractions"][name] == f.loc[t, "fraction_"+name]
        near(f.loc[t, ["raw_forecast", "calibrated_forecast", "no_state_forecast", "variance", "disagreement", "positive_agreement", "cusum"]].to_numpy(float),
             [mus["STATIC_EQUAL"], net, mus["NO_STATE"], variance, spread, agreement, g])
    assert np.flatnonzero(f.model_known).tolist() == outer_ids
    assert f.loc[~f.model_known, ["fraction_"+p for p in ["NO_STATE", "STATIC_EQUAL", "CALIBRATED", "DISAGREEMENT", "MONITORED", "FULL"]]].eq(0).all().all()
    assert f.loc[~f.model_known, "model_status"].eq("NO_VIEW").all()
    assert f.attrs["fit_attempt_counts"] == {"outer": len(outer), "inner": len(inner)}
    return f, len(outer), len(inner)


def fill_price(price, side, costs):
    value = price*(1+side*costs["slippage"])/.001
    return (math.ceil(value-1e-10) if side > 0 else math.floor(value+1e-10))*.001


def fee(q, price, costs):
    return max(abs(q)*price*costs["commission"], 5.) if q else 0.


def risk_quantity(cash, shares, receivable, reference, peak, es, desired, stress):
    nav = cash+shares*reference+receivable
    if not np.isfinite(es) or nav <= 0:
        return 0
    remaining = max(0., nav-.9*peak)
    cap = max(0., min(desired, .5, .025/max(es, 1e-12), .5*remaining/(.1*nav)))
    q = math.floor(cap*nav/reference/100)*100
    while q > 0:
        change = q-shares
        px = fill_price(reference, 1 if change > 0 else -1, stress)
        friction = abs(change)*abs(px-reference)+fee(change, px, stress)
        sale = fill_price(reference, -1, stress)
        reserve = q*(reference-sale)+fee(q, sale, stress)
        if q*reference*es+friction+reserve <= .025*nav+1e-8 and q*reference*.1+friction+reserve <= min(.05*nav, .5*remaining)+1e-8 and q*reference <= .5*(nav-friction)+1e-8:
            return q
        q -= 100
    return 0


def metric_values(ledger, capital):
    nav = np.r_[capital, ledger.equity.to_numpy(float)]
    r = nav[1:]/nav[:-1]-1
    near(ledger.net_return, r)
    sd = np.std(r, ddof=1)
    return {"end_equity": nav[-1], "net_profit": nav[-1]-capital,
        "net_sharpe": np.sqrt(242)*np.mean(r)/sd if sd > 1e-15 else np.nan,
        "sharpe_252_diagnostic": np.sqrt(252)*np.mean(r)/sd if sd > 1e-15 else np.nan,
        "cagr": (nav[-1]/capital)**(242/len(r))-1, "cagr_252_diagnostic": (nav[-1]/capital)**(252/len(r))-1,
        "max_drawdown": 1-np.min(nav/np.maximum.accumulate(nav)), "fills": int(ledger.filled_quantity.ne(0).sum()),
        "commission": ledger.commission.sum(), "slippage": ledger.slippage_cost.sum(), "mean_exposure": ledger.exposure.mean(),
        "terminal_exit_reserve": ledger.terminal_exit_reserve.iloc[-1], "max_identity_error": ledger.accounting_error.abs().max()}


def verify(root):
    frozen, started = read(root/"freeze.json"), read(root/"run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(started["at"])
    assert digest(root/"freeze.json") == started["freeze_sha256"]
    for item in frozen["files"]:
        assert digest(root/item["path"]) == item["sha256"], item
    for item in read(root/"source_receipt.json")["sources"]:
        assert digest(root/item["snapshot"]) == item["sha256"]
    d = pd.read_parquet(root/"inputs/market.parquet")
    d = d[d.date <= "2025-12-31"].reset_index(drop=True)
    price = pd.read_parquet(root/"inputs/price_features.parquet")
    labels = pd.read_parquet(root/"inputs/labels.parquet")
    div = pd.read_csv(root/"inputs/dividends.csv")
    for c in ["record_date", "ex_date", "payment_date"]:
        div[c] = pd.to_datetime(div[c])
    expected_labels = np.full(len(d), np.nan)
    for j in range(len(d)-6):
        a, b = j+1, j+6
        entitled = div.record_date.ge(d.date.iloc[a]) & div.record_date.lt(d.date.iloc[b]) & div.ex_date.le(d.date.iloc[b])
        expected_labels[j] = (d.open.iloc[b]+div.loc[entitled, "cash_dividend_per_share"].sum())/d.open.iloc[a]-1
    near(labels.gross_return5, expected_labels)
    cumulative, members, returns = [pd.read_parquet(root/"inputs"/name) for name in ["constituent_return20.parquet", "membership_mask.parquet", "constituent_returns.parquet"]]
    near(cumulative.to_numpy(), np.expm1(np.log1p(returns).rolling(20, min_periods=20).sum()).to_numpy())
    coverage, margin = [pd.read_parquet(root/"inputs"/name) for name in ["daily_coverage.parquet", "margin.parquet"]]
    frames, counts = {}, {}
    for lag in [1, 2]:
        print(f"开始核对时钟{lag}的特征、内层与外层系数。", flush=True)
        frames[lag], outer_count, inner_count = check_models(root, d, price, cumulative, members, coverage, margin, labels, lag)
        counts[lag] = {"outer": outer_count, "inner": inner_count}
        print(f"时钟{lag}通过：{outer_count}个判断日，{inner_count}条内层记录。", flush=True)
    protocol, result = read(root/"protocol.json"), read(root/"result.json")
    summary = pd.read_csv(root/"metrics.csv")
    annual = pd.read_csv(root/"annual_metrics.csv")
    assert len(summary) == summary.account_id.nunique() == protocol["planned_accounts"] == result["new_accounts"] == 112
    events = div.to_dict("records")
    for account_number, metric in enumerate(summary.to_dict("records"), 1):
        folder = root/"accounts"/metric["account_id"]
        ledger, decisions = pd.read_parquet(folder/"ledger.parquet"), pd.read_parquet(folder/"decisions.parquet")
        start, end = protocol["periods"][metric["period"]]
        assert ledger.date.reset_index(drop=True).equals(d.loc[d.date.between(start, end), "date"].reset_index(drop=True))
        assert len(ledger) == len(decisions) and ledger.date.equals(decisions.date)
        capital, cost = float(metric["capital"]), protocol["costs"][metric["cost"]]
        cash, shares, prev, peak = capital, 0, capital, capital
        receivables, entitlements, lots = {}, {}, []
        entry_idx, entry_wealth, stopped, pending = -1, np.nan, False, False
        for k, row in enumerate(ledger.itertuples()):
            idx, date = int(row.idx), row.date
            source, decision = d.iloc[idx], decisions.iloc[k]
            recognized, paid = 0., 0.
            for e, event in enumerate(events):
                if event["ex_date"] == date:
                    value = entitlements.get(e, 0)*event["cash_dividend_per_share"]
                    receivables[e], recognized = value, recognized+value
                if event["payment_date"] < date and e in receivables:
                    value = receivables.pop(e)
                    cash, paid = cash+value, paid+value
            near(row.cash_before, cash, 1e-6)
            assert row.shares_before == shares
            q, filled = int(row.requested_quantity), int(row.filled_quantity)
            assert q == decision.requested_quantity and filled == decision.filled_quantity and q % 100 == 0
            if pd.notna(metric["lag"]):
                f = frames[int(metric["lag"])].iloc[idx-1]
                desired = float(f["fraction_"+metric["policy"]]) if f.model_known else 0.
                stress = protocol["costs"]["STRESS"]
                reference = d.close.iloc[idx-1]-source.dividend
                target = risk_quantity(cash, shares, sum(receivables.values()), reference, peak, float(f.es95), desired, stress)
                if shares:
                    pending |= bool(stopped or not f.model_known or desired == 0 or idx >= entry_idx+20)
                if pending or stopped:
                    expected_request = -shares
                elif shares:
                    expected_request = target-shares
                    if f.wealth < entry_wealth:
                        expected_request = min(expected_request, 0)
                else:
                    expected_request = target if k < len(ledger)-1 else 0
                assert decision.pre_open_request == expected_request
                if expected_request > 0:
                    at_open = risk_quantity(cash, shares, sum(receivables.values()), source.open, peak, float(f.es95), desired, stress)
                    expected_request = max(0, min(expected_request, at_open-shares))
                    if shares and f.wealth*(source.open+source.dividend)/d.close.iloc[idx-1] < entry_wealth:
                        expected_request = 0
                assert q == expected_request
                assert decision.entry_idx_before == entry_idx and decision.exit_pending == pending
                near(decision.entry_wealth_before, entry_wealth)
            sellable = sum(number for day, number in lots if day < idx)
            assert sellable == row.sellable_before
            expected_fill = 0
            if q:
                side = 1 if q > 0 else -1
                px = fill_price(source.open, side, cost)
                basis = source.previous_close-source.dividend
                lower = math.floor(basis*.9/.001+.5+1e-9)*.001
                upper = math.floor(basis*1.1/.001+.5+1e-9)*.001
                blocked = (side > 0 and (source.open >= upper-1e-9 or px > upper+1e-9)) or (side < 0 and (source.open <= lower+1e-9 or px < lower-1e-9))
                if not blocked:
                    if q > 0:
                        affordable = math.floor(max(0, cash+1e-9)/px/100)*100
                        while affordable > 0 and affordable*px+fee(affordable, px, cost) > cash+1e-8:
                            affordable -= 100
                        expected_fill = min(q, affordable)
                    else:
                        expected_fill = -min(-q, sellable)
            assert filled == expected_fill
            if filled:
                near(row.fill_price, px)
                near(row.commission, fee(filled, px, cost))
                near(row.slippage_cost, abs(filled)*abs(px-source.open))
                cash -= filled*px+row.commission
                if not shares and filled > 0 and pd.notna(metric["lag"]):
                    entry_idx = idx
                    entry_wealth = float(f.wealth)*(source.open+source.dividend)/d.close.iloc[idx-1]
                shares += filled
                if filled > 0:
                    lots.append((idx, filled))
                else:
                    remaining = -filled
                    new_lots = []
                    for day, number in lots:
                        sold = min(remaining, number) if day < idx else 0
                        remaining -= sold
                        if number > sold:
                            new_lots.append((day, number-sold))
                    assert remaining == 0
                    lots = new_lots
            else:
                assert row.commission == row.slippage_cost == 0
            for e, event in enumerate(events):
                if event["payment_date"] == date and e in receivables:
                    value = receivables.pop(e)
                    cash, paid = cash+value, paid+value
                if event["record_date"] == date:
                    entitlements[e] = shares
            reserve = 0.
            if k == len(ledger)-1 and shares:
                sale = fill_price(source.close, -1, protocol["costs"]["STRESS"])
                reserve = shares*(source.close-sale)+fee(shares, sale, protocol["costs"]["STRESS"])
            equity = cash+shares*source.close+sum(receivables.values())-reserve
            peak = max(peak, equity)
            near([row.cash, row.equity, row.dividend_receivable, row.terminal_exit_reserve, row.dividend_recognized, row.dividend_paid,
                  row.net_return, row.drawdown], [cash, equity, sum(receivables.values()), reserve, recognized, paid, equity/prev-1, 1-equity/peak], 1e-6)
            assert shares == row.shares and shares >= 0 and cash >= -1e-7
            if pd.notna(metric["lag"]):
                f = frames[int(metric["lag"])].iloc[idx-1]
                desired = float(f["fraction_"+metric["policy"]]) if f.model_known else 0.
                assert decision.origin_idx == idx-1 and decision.origin == f.date
                assert decision.desired_fraction == desired
                assert decision.model_known == bool(f.model_known) and decision.model_status == f.model_status
                if decision.entry_idx_before >= 0 and q > 0:
                    assert f.wealth >= decision.entry_wealth_before
                    assert float(f.wealth)*(source.open+source.dividend)/d.close.iloc[idx-1] >= decision.entry_wealth_before
                if decision.exit_pending or (k and ledger.risk_stopped.iloc[k-1]):
                    assert q <= 0
                if k and ledger.risk_stopped.iloc[k-1]:
                    assert row.risk_stopped
                stopped |= 1-equity/peak >= .1
                assert bool(row.risk_stopped) == stopped
                if not shares:
                    entry_idx, entry_wealth, pending = -1, np.nan, False
            prev = equity
        for name, value in metric_values(ledger, capital).items():
            near(metric[name], value, 1e-6)
        for year, block in ledger.groupby(ledger.date.dt.year):
            saved = annual[(annual.account_id == metric["account_id"]) & (annual.year == year)].iloc[0]
            initial = block.equity.iloc[0]/(1+block.net_return.iloc[0])
            for name, value in metric_values(block, initial).items():
                near(saved[name], value, 1e-6)
        if account_number % 14 == 0:
            print(f"保存账本及仓位请求通过{account_number}/112。", flush=True)
    primary = summary[(summary.period == "MAIN") & (summary.cost == "STRESS") & summary.lag.eq(1) & summary.policy.eq("FULL")]
    passed = bool(len(primary) == 2 and ((primary.net_sharpe >= 1.2) & (primary.cagr >= .1) & (primary.max_drawdown <= .1)).all())
    assert passed == result["primary_historical_point_pass"]
    indices = np.load(root/"bootstrap_indices.npz")["indices"]
    full = pd.read_parquet(root/"accounts/MAIN_200000_STRESS_lag1_FULL/ledger.parquet")
    assert indices.shape == (4000, len(full)) and indices.min() >= 0 and indices.max() < len(full)
    for pair in read(root/"paired_increment.json"):
        other = pd.read_parquet(root/f"accounts/MAIN_200000_STRESS_lag1_{pair['control']}/ledger.parquet")
        diff = full.net_return.to_numpy()-other.net_return.to_numpy()
        near(pair["annual_mean_return_difference"], diff.mean()*242)
        near(pair["interval95"], np.quantile(diff[indices].mean(axis=1)*242, [.025, .975]))
    assert not result["goal_achieved"] and not result["orders_authorized"]
    return {"status": "PASS_SAVED_DAILY_SHRINK_CLOCK_COEFFICIENTS_AND_ACCOUNTS", "account_ledgers_replayed": 112,
        "verifier_sha256": digest(Path(__file__)),
        "model_counts": counts, "primary_historical_point_pass": passed, "new_accounts": 0,
        "new_experimental_models": 0, "network_requests": 0, "external_review": "NOT_PERFORMED",
        "scope": "保存系数用训练充分统计量复核，112保存账本重新核对资金份额分红与指标；不是新实验或独立策略验证。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
