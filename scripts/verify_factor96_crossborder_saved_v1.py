"""T10只读复算：原始跨境数据、逐日OLS、成熟风险和48份保存账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=0, atol=tolerance, equal_nan=True)


def check_sources(root, market):
    """逐个截止时点搜索原始记录，独立累乘收益，不调用研究实现。"""
    raw = load(root / "raw/ASHR_chart_2013_2025.json")["chart"]["result"][0]
    assert not raw.get("events", {}).get("splits") and not raw.get("events", {}).get("capitalGains")
    us_date = pd.to_datetime(raw["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    us_close = np.array(raw["indicators"]["quote"][0]["close"], float)
    cash = np.zeros(len(us_close))
    for item in raw.get("events", {}).get("dividends", {}).values():
        day = pd.to_datetime(item["date"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        found = np.flatnonzero(us_date == day)
        assert len(found) == 1
        cash[found[0]] += item["amount"]
    us_gross = np.r_[np.nan, (us_close[1:]+cash[1:])/us_close[:-1]]
    us_available = [pd.Timestamp(d.date().isoformat()+" 17:00", tz="America/New_York").tz_convert("Asia/Shanghai") for d in us_date]
    saved_us = pd.read_parquet(root / "inputs/ashr.parquet")
    close(saved_us.close, us_close)
    close(saved_us.cash_distribution, cash)
    close(saved_us.log_return, np.log(us_gross))
    assert saved_us.available_at.tolist() == us_available
    raw_fx = {}
    for p in sorted((root / "raw/fx").glob("*.json")):
        payload = load(p)
        assert str(payload["head"]["rep_code"]) == "200"
        usd = payload["data"]["head"].index("USD/CNY")
        for item in payload["records"]:
            value = float(item["values"][usd])
            assert item["date"] not in raw_fx or value == raw_fx[item["date"]]
            raw_fx[item["date"]] = value
    fx = pd.read_parquet(root / "inputs/fx.parquet")
    assert fx.date.is_unique and len(raw_fx) == len(fx)
    for row in fx.itertuples():
        close(row.first_release_value, raw_fx[row.date.date().isoformat()], 0.)
        assert pd.Timestamp(row.available_at) == pd.Timestamp(row.date.date().isoformat()+" 09:15", tz="Asia/Shanghai")
        assert digest(root / "raw/fx" / Path(row.raw_path).name) == row.source_hash
    aligned = pd.read_parquet(root / "aligned_external.parquet")
    assert aligned.date.tolist() == market.date.tolist()
    fx_times = pd.to_datetime(fx.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    prior_fix, prior_fix_day, prior_clock = np.nan, None, None
    for t, row in enumerate(aligned.itertuples()):
        decision = pd.Timestamp(row.date.date().isoformat()+" 09:20", tz="Asia/Shanghai")
        assert row.forecast_at == decision
        latest_ids = [i for i, stamp in enumerate(us_available) if stamp <= decision]
        baseline_ids = [] if prior_clock is None else [i for i, stamp in enumerate(us_available) if stamp <= prior_clock]
        latest = latest_ids[-1] if latest_ids else -1
        baseline = baseline_ids[-1] if baseline_ids else -1
        n = max(0, latest-baseline) if baseline >= 0 else 0
        assert row.us_session_count == n
        if latest >= 0:
            assert row.latest_us_date == us_date[latest]
            assert row.latest_us_available_at == us_available[latest] <= decision
            assert row.latest_us_date < row.date
        if baseline >= 0:
            assert row.baseline_us_date == us_date[baseline]
        expected_us = np.nan
        if n > 0 and (row.date-us_date[latest]).days <= 7 and (market.date.iloc[t-1]-us_date[baseline]).days <= 7:
            expected_us = np.log(np.prod(us_gross[baseline+1:latest+1]))
        close(row.ashr_log_return, expected_us, 1e-12)
        available = np.flatnonzero((fx_times <= decision).to_numpy())
        current_fix, current_fix_day = np.nan, None
        if len(available):
            k = available[-1]
            current_fix, current_fix_day = fx.first_release_value.iloc[k], fx.date.iloc[k]
            assert row.fx_date == current_fix_day and row.fx_available_at == fx_times.iloc[k]
            close(row.fx_rate, current_fix)
        expected_fx = np.nan
        if t and current_fix_day == row.date and prior_fix_day == market.date.iloc[t-1]:
            expected_fx = np.log(current_fix/prior_fix)
        close(row.fx_log_return, expected_fx, 1e-12)
        prior_fix, prior_fix_day, prior_clock = current_fix, current_fix_day, decision
    response = np.log((market.close+market.dividend)/market.close.shift())
    gap = np.log((market.open+market.dividend)/market.close.shift())
    close(aligned.a_log_return, response)
    close(aligned.a_gap_log_return, gap)
    close(aligned.prior_a_log_return, response.shift())
    expected_known = np.isfinite(aligned[["ashr_log_return", "fx_log_return", "prior_a_log_return"]]).all(axis=1)
    assert aligned.external_known.equals(expected_known.rename("external_known"))
    return aligned, len(us_close), len(fx)


def check_models(root, market, aligned):
    """用正规方程复算训练系数，与研究代码的SVD解法交叉核对。"""
    model = pd.read_parquet(root / "response_models.parquet")
    records = load(root / "response_training_records.json")
    x = np.column_stack([np.ones(len(model)), aligned[["ashr_log_return", "fx_log_return", "prior_a_log_return"]]])
    y, gap = aligned.a_log_return.to_numpy(), aligned.a_gap_log_return.to_numpy()
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & np.isfinite(gap)
    by_day = {int(r["idx"]): r for r in records}
    assert len(by_day) == len(records) == model.model_known.sum()
    for t in range(len(model)):
        ids = np.flatnonzero(valid & (np.arange(len(model)) < t) & aligned.date.ge(aligned.date.iloc[t]-pd.DateOffset(years=2)).to_numpy())
        eligible = np.isfinite(x[t]).all() and len(ids) >= 252
        if eligible:
            eligible = np.linalg.matrix_rank(x[ids]) == 4
        if not eligible:
            assert t not in by_day and not model.model_known.iloc[t]
            continue
        record = by_day[t]
        assert record["training_indices"] == ids.tolist()
        assert record["forecast_at"] == model.forecast_at.iloc[t].isoformat()
        assert model.training_n.iloc[t] == len(ids) and model.training_max_idx.iloc[t] == ids.max() < t
        a = x[ids]
        beta = np.linalg.solve(a.T @ a, a.T @ y[ids])
        beta_gap = np.linalg.solve(a.T @ a, a.T @ gap[ids])
        sigma = np.sqrt(np.sum((y[ids]-a @ beta)**2)/(len(ids)-4))
        sigma_gap = np.sqrt(np.sum((gap[ids]-a @ beta_gap)**2)/(len(ids)-4))
        close(record["beta"], beta, 1e-9)
        close(record["beta_gap"], beta_gap, 1e-9)
        close([record["sigma"], record["sigma_gap"]], [sigma, sigma_gap], 1e-12)
        close(model.loc[t, ["predicted_response", "response_sigma", "predicted_gap", "gap_sigma"]],
              [x[t] @ beta, sigma, x[t] @ beta_gap, sigma_gap], 1e-12)
    close(model.response_residual, model.a_log_return-model.predicted_response)
    close(model.response_z, model.response_residual/model.response_sigma)
    expected_flags = {"price_signal": model.model_known & model.a_log_return.gt(0), "external_positive": model.ashr_log_return.gt(0),
                      "underreaction": model.response_z.lt(-1), "overreaction": model.response_z.gt(1)}
    for col, value in expected_flags.items():
        assert model[col].equals(value.rename(col)), col
    wealth = np.exp(aligned.a_log_return.fillna(0.).cumsum())
    low = wealth*(market.low+market.dividend)/(market.close+market.dividend)
    features = {}
    for lag in [0, 1]:
        saved = pd.read_parquet(root / f"daily_features_lag{lag}.parquet")
        close(saved.wealth, wealth)
        close(saved.low_w, low)
        close(saved.frozen_stop, low.shift(lag))
        source_idx = pd.Series(np.arange(len(model))).shift(lag).fillna(-1).astype(int)
        assert saved.source_idx.tolist() == source_idx.tolist()
        assert saved.source_day.equals(model.date.shift(lag).rename("source_day"))
        for col in ["predicted_response", "response_sigma", "response_z", "ashr_log_return", "fx_log_return", "training_max_idx"]:
            close(saved[col], model[col].shift(lag))
        for col, values in expected_flags.items():
            expected = values.shift(lag, fill_value=False)
            if col == "price_signal":
                expected &= wealth.ge(low.shift(lag))
            assert saved[col].equals(expected.rename(col)), (lag, col)
        features[lag] = saved
    close(features[0].es95, features[1].es95)
    return features, len(records)


def metrics(z, capital):
    nav = np.r_[float(capital), z.equity.to_numpy()]
    returns = nav[1:]/nav[:-1]-1
    sd = returns.std(ddof=1)
    return {"net_sharpe": np.sqrt(242)*returns.mean()/sd if sd > 1e-15 else np.nan,
            "cagr": (nav[-1]/capital)**(242/len(returns))-1, "max_drawdown": 1-np.min(nav/np.maximum.accumulate(nav)),
            "mean_exposure": z.exposure.mean()}


def check_accounts(root, market, features):
    dividends = pd.read_csv(root / "inputs/dividends.csv", parse_dates=["record_date", "ex_date", "payment_date"])
    table = pd.read_csv(root / "metrics.csv")
    annual = pd.read_csv(root / "annual_metrics.csv")
    assert len(table) == 48 and not table.duplicated(["period", "capital", "cost", "lag", "policy"]).any()
    returns, total_rows = {}, 0
    for ordinal, row in enumerate(table.itertuples(), 1):
        folder = root / f"accounts/{row.period}/{row.capital}/{row.cost}/LAG{row.lag}/{row.policy}"
        z = pd.read_parquet(folder / "ledger.parquet")
        d = pd.read_parquet(folder / "decisions.parquet")
        nav = np.r_[float(row.capital), z.equity]
        close(z.net_return, nav[1:]/nav[:-1]-1, 1e-12)
        close(z.equity, z.cash+z.shares*z.mark+z.dividend_receivable-z.terminal_exit_reserve, 1e-7)
        close(z.shares, z.filled_quantity.cumsum(), 0.)
        fill = z.fill_price.astype(float).fillna(0.)
        close(z.cash, row.capital+(-z.filled_quantity*fill-z.commission+z.dividend_paid).cumsum(), 1e-7)
        reserve_change = z.terminal_exit_reserve.diff().fillna(z.terminal_exit_reserve.iloc[0])
        close(np.diff(nav), z.price_pnl+z.dividend_recognized-z.commission-z.slippage_cost-reserve_change, 1e-7)
        close(z.drawdown, 1-nav[1:]/np.maximum.accumulate(nav)[1:], 1e-12)
        assert z.cash.min() >= -1e-7 and z.shares.min() >= 0 and z.shares.mod(100).eq(0).all()
        assert (d.origin < d.date).all() and z.date.equals(d.date) and z.filled_quantity.equals(d.filled_quantity)
        assert (z.loc[z.filled_quantity.lt(0), "filled_quantity"].abs() <= z.loc[z.filled_quantity.lt(0), "sellable_before"]).all()
        cost_rate, slip = (.0002, .0005) if row.cost == "BASE" else (.0004, .001)
        traded, sign = z.filled_quantity.ne(0), np.sign(z.filled_quantity)
        close(z.commission, np.where(traded, np.maximum(abs(z.filled_quantity)*fill*cost_rate, 5.), 0.), 1e-9)
        expected_fill = np.where(sign > 0, np.ceil((z.open*(1+slip)-1e-12)/.001)*.001, np.floor((z.open*(1-slip)+1e-12)/.001)*.001)
        close(fill[traded], expected_fill[traded], 1e-10)
        close(z.slippage_cost, abs(z.filled_quantity)*abs(fill-z.open), 1e-8)
        source = features[row.lag].iloc[d.origin_idx].reset_index(drop=True)
        assert source.date.equals(d.origin.rename("date"))
        close(d.es95_5d, source.es95)
        for col in ["price_signal", "model_known", "external_positive", "underreaction", "overreaction", "source_idx"]:
            assert d[col].equals(source[col]), col
        for col in ["ashr_log_return", "fx_log_return", "predicted_response", "response_sigma", "response_z", "training_max_idx"]:
            close(d[col], source[col])
        expected = source.price_signal.copy()
        if row.policy != "PRICE_COMMON":
            expected &= source.external_positive
        if row.policy == "FULL":
            expected &= source.underreaction
        if row.policy == "OVERREACTION":
            expected &= source.overreaction
        assert d.active_signal.equals(expected.rename("active_signal"))
        buys = d.filled_quantity.gt(0)
        assert d.loc[buys, "active_signal"].all()
        assert (d.loc[buys, "source_day"] <= d.loc[buys, "origin"]).all()
        assert (d.loc[buys, "training_max_idx"] < d.loc[buys, "source_idx"]).all()
        assert (d.loc[buys, "latest_us_date"] < d.loc[buys, "source_day"]).all()
        assert (d.loc[buys, "fx_date"] <= d.loc[buys, "source_day"]).all()
        close(d.loc[buys, "frozen_stop"], source.loc[buys, "frozen_stop"])
        # 单独追踪冻结止损、持有期限及停机后不重新进入。
        entry_idx, stop, pending, stopped = None, np.nan, False, False
        for j in range(len(z)):
            old = int(z.shares_before.iloc[j])
            if old:
                trigger = stopped or source.wealth.iloc[j] < stop or int(z.idx.iloc[j]) >= entry_idx+5
                pending |= trigger
                if pending:
                    assert d.pre_open_request.iloc[j] == -old
                else:
                    assert d.pre_open_request.iloc[j] <= 0
                close(d.frozen_stop.iloc[j], stop)
            elif z.filled_quantity.iloc[j] > 0:
                assert not stopped
                entry_idx, stop = int(z.idx.iloc[j]), float(source.frozen_stop.iloc[j])
                pending = False
            stopped |= z.drawdown.iloc[j] >= .1
            assert bool(z.risk_stopped.iloc[j]) == stopped
            if z.shares.iloc[j] == 0:
                entry_idx, pending = None, False
        for name, value in metrics(z, row.capital).items():
            close(getattr(row, name), value)
        group = annual[(annual.period == row.period) & (annual.capital == row.capital) & (annual.cost == row.cost) & (annual.lag == row.lag) & (annual.policy == row.policy)]
        prior = row.capital
        for year, part in z.groupby(z.date.dt.year):
            saved = group[group.year.eq(year)]
            assert len(saved) == 1
            for name, value in metrics(part, prior).items():
                close(saved.iloc[0][name], value)
            prior = part.equity.iloc[-1]
        recognized, paid = np.zeros(len(z)), np.zeros(len(z))
        for event in dividends.itertuples():
            shares = z.loc[z.date.eq(event.record_date), "shares"]
            amount = float(shares.iloc[0])*event.cash_dividend_per_share if len(shares) else 0.
            recognized[z.date.eq(event.ex_date)] += amount
            payment = np.flatnonzero(z.date.ge(event.payment_date).to_numpy())
            if len(payment):
                paid[payment[0]] += amount
        close(z.dividend_recognized, recognized, 1e-7)
        close(z.dividend_paid, paid, 1e-7)
        close(z.dividend_receivable, np.cumsum(recognized-paid), 1e-7)
        assert int(((z.shares.shift(fill_value=0) > 0) & z.shares.eq(0)).sum()) == row.closed_cycles
        returns[row.period, row.capital, row.cost, row.lag, row.policy] = z.net_return.to_numpy()
        total_rows += len(z)
        if ordinal % 16 == 0:
            print(f"已复核{ordinal}份T10账户，时序、费用、分红和冻结退出一致。", flush=True)
    return table, returns, total_rows, dividends


def check_risk(root, market, features, dividends):
    risk = load(root / "inputs/risk_training_records.json")
    labels = pd.read_parquet(root / "inputs/mature_risk_labels.parquet")
    for record in risk:
        selected = np.array(record["selected_indices"], int)
        t = record["decision_idx"]
        assert selected.max()+6 <= t
        assert market.date.iloc[selected.min()] >= market.date.iloc[t]-pd.DateOffset(years=2)
        es = max(0., -np.sort(labels.gross_return5.iloc[selected])[:int(np.ceil(.05*len(selected)))].mean())
        close([record["es95_5d"], features[0].es95.iloc[t]], [es, es], 1e-12)
    for row in labels.dropna().itertuples():
        entry, exit_ = int(row.origin_idx)+1, int(row.exit_idx)
        cash = dividends.loc[dividends.record_date.ge(market.date.iloc[entry]) & dividends.record_date.lt(market.date.iloc[exit_]), "cash_dividend_per_share"].sum()
        close(row.gross_return5, (market.open.iloc[exit_]+cash)/market.open.iloc[entry]-1, 1e-12)
    return len(risk)


def verify(root):
    frozen, started = load(root / "freeze.json"), load(root / "run_started.json")
    for row in frozen["files"]:
        assert digest(root / row["path"]) == row["sha256"], row["path"]
    assert started["freeze_sha256"] == digest(root / "freeze.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(started["at"])
    market = pd.read_parquet(root / "inputs/market.parquet")
    market = market[market.date.le("2025-12-31")].reset_index(drop=True)
    aligned, us_n, fx_n = check_sources(root, market)
    features, model_n = check_models(root, market, aligned)
    table, returns, total_rows, dividends = check_accounts(root, market, features)
    risk_n = check_risk(root, market, features, dividends)
    indices = np.load(root / "bootstrap_indices.npz")["indices"]
    a = returns["MAIN", 200000, "STRESS", 0, "FULL"]
    assert indices.shape == (4000, len(a))
    for other, increment in zip(["POSITIVE", "PRICE_COMMON"], load(root / "paired_increment.json"), strict=True):
        assert increment["comparison"] == "FULL_MINUS_" + other
        diff = a - returns["MAIN", 200000, "STRESS", 0, other]
        draws = diff[indices].mean(axis=1)*242
        close([increment["annual_arithmetic_increment"], increment["ci95_low"], increment["ci95_high"]],
              [diff.mean()*242, *np.quantile(draws, [.025, .975])], 1e-12)
    selected = table[(table.period == "MAIN") & (table.cost == "STRESS") & (table.lag == 0) & (table.policy == "FULL")]
    assert len(selected) == 2
    passed = bool(((selected.net_sharpe >= 1.2) & (selected.cagr >= .1) & (selected.max_drawdown <= .1)).all())
    assert passed == load(root / "result.json")["historical_joint_point_pass"]
    return {"status": "PASS_SAVED_CROSSBORDER_SOURCE_CLOCK_OLS_ACCOUNT_RECOMPUTATION", "ledgers": len(table), "ledger_rows": total_rows,
            "ashr_raw_rows": us_n, "fx_raw_rows": fx_n, "response_model_records": model_n, "mature_risk_records": risk_n,
            "new_accounts": 0, "new_random_draws": 0, "network_requests": 0, "external_review": "NOT_PERFORMED",
            "independent_forward_validation": False, "scope": "原始来源、可得时钟、逐日模型和保存账本复核；不认证历史首版或策略有效性。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
