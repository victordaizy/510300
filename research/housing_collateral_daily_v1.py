"""住房价格下跌范围的固定5日/20日增量；两年日更、五日风险预算。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.housing_collateral_monthly_source_v1 as source
import research.loan_maturity_composition_daily_v1 as loan
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

prior, shared, maturity = loan.prior, loan.shared, loan.maturity
parent, engine, distribution = prior.parent, prior.engine, prior.distribution
OUT = ROOT / "reports/research/510300_housing_collateral_daily_v1"
STUDY = "510300_HOUSING_COLLATERAL_DAILY_V1"
START, END = parent.START, parent.END
FEATURE = "second_hand_down_share"
CONTROL, SIGNAL = "PRICE_AND_FUNDING", "PRICE_FUNDING_AND_HOUSING_BREADTH"
MODELS = {"HISTORY": [], CONTROL: [*parent.BASE, "funding_innovation"], SIGNAL: [*parent.BASE, "funding_innovation", FEATURE]}
SCOPES = ["PUBLISHED_MEASUREMENT", "SAME_BASE_ONLY"]
HORIZONS = [5, 20]
PRIMARY = SIGNAL + "__PUBLISHED_MEASUREMENT__FIXED20_NO_ADD"


def policy_name(model, scope, horizon):
    return f"{model}__{scope}__FIXED{horizon}_NO_ADD"


def sharpe_text(value):
    return "未定义（零波动）" if value is None else f"{value:.6f}"


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    q = (pd.read_parquet(source.PANEL) if releases is None else releases).copy()
    q["known_at"] = pd.to_datetime(q.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    q = q[["known_at", "stat_month", "index_base_year", FEATURE]].rename(columns={"known_at": "housing_known_at", "stat_month": "housing_stat_month"})
    assert not q.housing_known_at.duplicated().any()
    x = pd.merge_asof(x.sort_values("decision_time"), q.sort_values("housing_known_at"), left_on="decision_time", right_on="housing_known_at", direction="backward")
    x["housing_age_days"] = (x.decision_time - x.housing_known_at).dt.total_seconds() / 86400
    x["bank_information_known"] = x.common_known
    x["common_known"] &= x.housing_age_days.between(0, 45) & x[FEATURE].notna()
    assert x.loc[x.common_known, "housing_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert x.idx.tolist() == list(range(len(x)))
    return x


def labels_for_horizons(market, dividends):
    rows = []
    for i in range(len(market) - 20):
        row = {"idx": i, "date": market.date.iloc[i], "exit_idx": i + 20, "exit_date": market.date.iloc[i + 20]}
        for h in HORIZONS:
            end = market.date.iloc[i + h]
            amount = dividends.loc[dividends.record_date.ge(row["date"]) & dividends.record_date.lt(end), "cash_dividend_per_share"].sum()
            row[f"gross_return{h}"] = (float(market.open.iloc[i + h]) + amount) / float(market.open.iloc[i]) - 1
            row[f"dividend{h}"] = float(amount)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_day(i, x, nested, count, labels, scope):
    day, lower = x.date.iloc[i], x.date.iloc[i] - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "scope": scope, "lower_bound": lower, "status": "NO_VIEW_SOURCE", "n_train": 0}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    innovations = prior.innovation_series(i, x, nested, count)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    ids = pool.idx.to_numpy(int)
    allowed = x.common_known.iloc[ids].to_numpy(bool) & np.isfinite(innovations[ids])
    allowed &= x.housing_known_at.iloc[ids].ge(lower.tz_localize("Asia/Shanghai")).to_numpy(bool)
    if scope == "SAME_BASE_ONLY":
        allowed &= x.index_base_year.iloc[ids].eq(x.index_base_year.iloc[i]).to_numpy(bool)
    pool = pool.loc[allowed]
    ids = pool.idx.to_numpy(int)
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None,
        release_months=int(x.housing_stat_month.iloc[ids].nunique()), training_base_years=sorted(x.index_base_year.iloc[ids].unique().astype(int).tolist()))
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        receipt["status"] = "NO_VIEW_TRAINING"
        return [], [], receipt
    receipt["status"] = "UPDATED"
    values = x.loc[ids, [*parent.BASE, FEATURE]].copy()
    values["funding_innovation"] = innovations[ids]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, saved = [], []
    for model, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            assert np.isfinite(raw).all()
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            z = np.clip((raw - mean) / scale, -5, 5)
            live = np.clip((np.array([current[c] for c in columns]) - mean) / scale, -5, 5)
            chosen = pool.iloc[np.lexsort((ids, np.sum((z - live) ** 2, axis=1)))[:126]]
        else:
            chosen = pool
        record = {**receipt, "model": model, "n_selected": len(chosen)}
        for h in HORIZONS:
            stats = distribution.empirical_statistics(chosen[f"gross_return{h}"])
            record.update({f"mu_{h}": stats["mu5"], f"variance_{h}": stats["variance5"],
                f"q05_{h}": stats["q05"], f"es95_{h}": stats["es95"]})
        predictions.append(record)
        saved.append({"idx": i, "scope": scope, "model": model, "features": columns,
            "training_indices": ids.tolist(), "selected_indices": chosen.idx.tolist(),
            "mean": mean.tolist(), "scale": scale.tolist(), "current_values": {c: float(current[c]) for c in MODELS[SIGNAL]}})
    return predictions, saved, receipt


def forecast_proxy(row, horizon):
    return SimpleNamespace(mu5=float(getattr(row, f"mu_{horizon}")), variance5=float(getattr(row, f"variance_{horizon}")), es95=float(row.es95_5))


def simulate(market, x, dividends, predictions, model, scope, horizon, cost_name, start=START):
    assert horizon in HORIZONS
    first = int(np.flatnonzero(market.date.ge(start))[0])
    selected = predictions[predictions.model.eq(model) & predictions.scope.eq(scope)]
    lookup = {int(r.idx): r for r in selected.itertuples()}
    account, cfg = engine.Account(200000.), {"lot": 100, "tick": .001, "limit_fraction": .1}
    cost, policy = distribution.COSTS[cost_name], policy_name(model, scope, horizon)
    previous_equity, previous_mark, previous_reserve, peak = 200000., float(market.close.iloc[first - 1]), 0., 200000.
    stopped, last_view, entry_idx = False, first - 6, None
    records, decisions, events = [], [], dividends.to_dict("records")
    for i in range(first, len(market)):
        row = market.iloc[i]
        day, op, close = row.date, float(row.open), float(row.close)
        old_shares, recognized, paid = account.shares, 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        ref, forecast = float(row.previous_close - row.dividend), lookup.get(i)
        if forecast is not None:
            last_view = i
        deadline = entry_idx + horizon if entry_idx is not None else None
        plan = {"requested_quantity": 0, "target_shares": account.shares, "reason": "NO_VIEW_HOLD_OLD_INVENTORY"}
        if stopped:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="DRAWDOWN_STOP")
        elif account.shares and i >= deadline:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="FIXED_HORIZON_MATURITY")
        elif forecast is not None:
            proxy = forecast_proxy(forecast, horizon)
            if account.shares:
                plan = {**maturity.risk_only_plan(account, ref, peak, proxy), "reason": "FIXED_MATURITY_DAILY_FIVE_DAY_RISK"}
            else:
                plan = {**parent.plan(engine, account, ref, peak, proxy, float(x.pressure5.iloc[i])), "reason": "HORIZON_MATCHED_ENTRY"}
            plan["forecast_mean"] = plan.pop("mu5")
            plan["forecast_variance"] = plan.pop("variance5")
            plan["risk_forecast_horizon"] = 5
        elif i - last_view >= 5:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="NO_VIEW_INVENTORY_EXPIRED")
        requested = int(plan["requested_quantity"])
        before_gap = requested
        if requested > 0:
            while requested > 0 and not parent.risk_valid(engine, account.shares + requested, op, plan, distribution.COSTS["STRESS"]):
                requested -= 100
        sellable = account.sellable(i)
        execution = engine.execute_order(account, requested, op, float(row.previous_close), float(row.dividend), i, cost, cfg)
        if execution["filled_quantity"] > 0 and old_shares == 0:
            entry_idx = i
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == len(market) - 1 and account.shares:
            exit_price = engine.fill_price(close, -1, distribution.COSTS["STRESS"], .001)
            reserve = account.shares * (close - exit_price) + engine.commission(account.shares, exit_price, distribution.COSTS["STRESS"])
        equity = account.value(close) - reserve
        price_pnl = old_shares * (op - previous_mark) + account.shares * (close - op)
        error = equity - previous_equity - price_pnl - recognized + execution["commission"] + execution["slippage_cost"] + reserve - previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        stopped |= drawdown >= .1
        records.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "open": op, "mark": close,
            "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
            "terminal_exit_reserve": reserve, "equity": equity, "net_return": equity / previous_equity - 1,
            "pnl": equity - previous_equity, "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
            "exposure": account.shares * close / equity, "accounting_error": error, "drawdown": drawdown,
            "risk_stopped": stopped, "sellable_before": sellable, "terminal_unliquidated": bool(i == len(market) - 1 and account.shares), **execution})
        decisions.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "prediction_available": forecast is not None,
            "pressure5": float(x.pressure5.iloc[i]), "pre_open_requested_quantity": before_gap,
            "gap_checked_request": requested, "filled_quantity": execution["filled_quantity"], "actual_open": op,
            "cycle_entry_idx": entry_idx, "deadline_idx": entry_idx + horizon if entry_idx is not None else None,
            "forecast_horizon": horizon, **plan})
        if account.shares == 0:
            entry_idx = None
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def implementation_checks():
    dates = pd.bdate_range("2020-01-01", periods=60)
    market = pd.DataFrame({"date": dates, "open": 4., "close": 4., "previous_close": 4., "dividend": 0.})
    x = pd.DataFrame({"pressure5": np.ones(60)})
    predictions = pd.DataFrame({"idx": np.arange(1, 60), "model": CONTROL, "scope": SCOPES[0],
        "mu_5": .03, "mu_20": .03, "variance_5": .001, "variance_20": .001, "es95_5": .025})
    div = pd.DataFrame(columns=["ex_date", "record_date", "payment_date", "cash_dividend_per_share"])
    for h in HORIZONS:
        ledger, decisions = simulate(market, x, div, predictions, CONTROL, SCOPES[0], h, "STRESS", "2020-01-02")
        parent.account_verification(ledger, decisions)
        assert not (ledger.shares.shift(1, fill_value=0).gt(0) & ledger.filled_quantity.gt(0)).any()
        assert decisions.loc[decisions.reason.eq("FIXED_HORIZON_MATURITY"), "idx"].tolist() == list(range(1 + h, 60, h + 1))
        if h == 5:
            old_pred = predictions.rename(columns={"mu_5": "mu5", "variance_5": "variance5", "es95_5": "es95"})
            old, _ = maturity.simulate(market, x, div, old_pred, CONTROL, "FIXED5_NO_ADD", "STRESS", "2020-01-02")
            np.testing.assert_allclose(ledger[["shares", "equity", "cash", "filled_quantity"]], old[["shares", "equity", "cash", "filled_quantity"]], atol=1e-9, rtol=0)
    sparse, _ = simulate(market, x, div, predictions.iloc[:1], CONTROL, SCOPES[0], 20, "STRESS", "2020-01-02")
    assert sparse.loc[sparse.idx.ge(6), "shares"].eq(0).all()
    opposing = predictions.copy()
    opposing["mu_5"] = -.03
    five, _ = simulate(market, x, div, opposing, CONTROL, SCOPES[0], 5, "STRESS", "2020-01-02")
    twenty, _ = simulate(market, x, div, opposing, CONTROL, SCOPES[0], 20, "STRESS", "2020-01-02")
    assert five.filled_quantity.eq(0).all() and twenty.filled_quantity.gt(0).any()
    opposing["es95_20"] = 999.
    changed_tail, _ = simulate(market, x, div, opposing, CONTROL, SCOPES[0], 20, "STRESS", "2020-01-02")
    np.testing.assert_allclose(twenty.equity, changed_tail.equity, atol=1e-12, rtol=0)
    test_div = pd.DataFrame({"record_date": [dates[1], dates[6], dates[21]], "cash_dividend_per_share": [.1, .2, .4]})
    lab = labels_for_horizons(market, test_div).set_index("idx")
    np.testing.assert_allclose(lab.loc[1, ["dividend5", "dividend20"]].to_numpy(float), [.1, .3], rtol=0, atol=1e-12)
    dates_long = pd.bdate_range("2021-01-04", periods=900)
    n, i = len(dates_long), len(dates_long) - 1
    train_x = pd.DataFrame({"idx": np.arange(n), "date": dates_long, "common_known": True,
        "source_idx": np.arange(n), "housing_known_at": dates_long.tz_localize("Asia/Shanghai") - pd.Timedelta(days=1),
        "housing_stat_month": dates_long.strftime("%Y-%m"), "index_base_year": np.where(np.arange(n) >= 700, 2025, 2020)})
    for j, field in enumerate([*parent.BASE, FEATURE]):
        train_x[field] = np.sin(np.arange(n) / (11 + j))
    training_labels = pd.DataFrame({"idx": np.arange(n - 20), "date": dates_long[:-20],
        "exit_idx": np.arange(n - 20) + 20, "gross_return5": np.sin(np.arange(n - 20)) / 100,
        "gross_return20": np.cos(np.arange(n - 20)) / 20})
    nested = pd.DataFrame({"decision_idx": i, "source_idx": np.arange(n), "actual": np.sin(np.arange(n) / 17), "CALENDAR": 0.})
    before, records, receipt = fit_day(i, train_x, nested, n, training_labels, SCOPES[0])
    assert receipt["status"] == "UPDATED" and len(before) == 3
    altered = training_labels.copy()
    invalid = altered.exit_idx.ge(i) | altered.date.lt(dates_long[-1] - pd.DateOffset(years=2))
    altered.loc[invalid, ["gross_return5", "gross_return20"]] = -9999.
    after, rebuilt, _ = fit_day(i, train_x, nested, n, altered, SCOPES[0])
    assert before == after and records == rebuilt
    assert all(r["training_indices"] == records[0]["training_indices"] for r in records)
    p_strict, _, strict_receipt = fit_day(i, train_x, nested, n, training_labels, SCOPES[1])
    assert not p_strict and strict_receipt["status"] == "NO_VIEW_TRAINING" and strict_receipt["n_train"] < 252
    return {"five_day_matches_existing_engine": True, "fixed_5_and_20_exit_dates": True,
        "no_add_cash_T1_risk": True, "missing_forecast_exits_after_five_days": True, "dividend_record_boundary": True,
        "entry_uses_matching_horizon_mean": True, "five_day_risk_not_twenty_day_risk": True,
        "common_pool_future_and_expired_labels_ignored": True, "insufficient_same_base_returns_no_view": True}


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("住房增量试验已经固定。")
    sr = read(source.OUT / "result.json")
    assert sr["status"] == "COMPLETE_HOUSING_BREADTH_SOURCES_READY_WITH_REBASE_BREAK" and sr["admitted_months"] == 68
    checks, x = implementation_checks(), information()
    q = pd.read_parquet(source.PANEL)
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed = q.copy()
    changed.loc[changed.known_at.ge(cutoff), FEATURE] += 700
    after = information(changed)
    columns = [FEATURE, "common_known", "index_base_year"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), columns], after.loc[after.decision_time.lt(cutoff), columns], check_exact=True)
    for name in ["inputs", "results", "code", "accounts"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "已公开二手房价格下跌覆盖范围，是否改善相同股票压力与银行资金意外后的修复判断？财富和抵押品作用可能慢于五日，所以主期限20交易日，五日为事前固定对照。",
        "feature": "唯一新增信息为70城已公布二手住宅环比指数<100的城市数/70；没有把它转为房产市值损失、沪深300抵押率、违约率或资金流。",
        "source": sr, "information_clock": "原公布日日末才可使用；45自然日有效，不回填统计月末。首版未认证。",
        "rebase": "主研究比较同一定义的已公布环比下跌广度，允许2020和2025基期混合，但不声称权重变化无影响。强制另做SAME_BASE_ONLY主20日期限及同池价格资金对照；2026年同基期不足252时保持NO_VIEW。两种处理都披露，不挑较好者。",
        "training": "每个交易日向前两日历年；来源公布也在两年内；5日和20日都必须已成熟，20日退出开盘严格早于当前日。两期限共同池至少252个原点；固定126近邻、训练内标准化clip5、同距原点编号优先。",
        "overlap": "252个日原点、126近邻和重复月度值均不是独立机会；两年最多约24个月报。报告实际月报数，完整公布间隔与20交易日区块均保留。",
        "models": MODELS, "scopes": SCOPES, "horizons": HORIZONS,
        "accounts": "主混合基期：2信息模型×2期限×2费用=8；严格同基期：2信息模型×20日期限×2费用=4；共12新账户。HISTORY仅预测对照。",
        "decision": "空仓且此前5日价格弱势时，枚举100份目标，最大化期限一致的w*均值-2*w^2*方差-调整成本-退出储备；参数不变。持仓不加仓，只按当日五日风险缩量，或固定第5/20个后续开盘到期。缺预测满5日退出。",
        "risk": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "risk_horizon": "风险约束始终使用五日ES；20日均值和方差只用于20日期限收益评价，不把20日ES冒充五日ES。长持有仍承担连续跳空和风险预测错误。",
        "period": [START, END], "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "costs": distribution.COSTS,
        "annual_days": 242, "cash_and_risk_free": 0, "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparison": "各期限同池信息增量；主20日减5日；同基期主20日信息增量。20日循环及月报完整间隔各2000次，seed2026092613。固定2020-2023/2024-末端，另外单列2026基期变更后。",
        "gate": "主压力账户共同三目标通过，主信息增量两种区块95%下界均正且两个固定时期为正；严格同基期敏感性信息增量两区块下界也正。仍需要独立前向。",
        "monitor": "固定五日相位，最近60个成熟五日q05突破率>15%记警报；另报告20日非重叠监控，样本不足不当稳定。监控不修改本轮账户。",
        "availability": "2021年初之前、训练不足与未知时段保留NO_VIEW及完整账户现金日，不能当已经历市场验证。",
        "implementation_checks": checks, "source_future_perturbation": True,
        "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False, "current_market_view": "NO_VIEW"}
    save(OUT / "protocol.json", protocol, True)
    paths = [Path(__file__), Path(source.__file__), source.PANEL, source.OUT / "result.json", source.OUT / "released_records.json",
        Path(prior.__file__), Path(parent.__file__), Path(maturity.__file__), Path(engine.__file__), Path(distribution.__file__),
        Path(loan.__file__), ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
        parent.DIVIDENDS, parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/mature_labels.parquet",
        prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
        prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"住房广度的12账户已固定，共同信息日{x.common_known.sum()}。", flush=True)


def verify_predictions(pred, saved, x, labels, groups, count):
    lookup = pred.set_index(["idx", "scope", "model"])
    sample, pools = labels.set_index("idx", drop=False), {}
    for record in saved:
        i, scope, model = record["idx"], record["scope"], record["model"]
        pool = sample.loc[record["training_indices"]]
        lower = x.date.iloc[i] - pd.DateOffset(years=2)
        assert pool.date.ge(lower).all() and pool.exit_idx.lt(i).all()
        assert x.loc[pool.idx, "housing_known_at"].ge(lower.tz_localize("Asia/Shanghai")).all()
        assert x.common_known.iloc[pool.idx.to_numpy(int)].all()
        assert pools.setdefault((i, scope), record["training_indices"]) == record["training_indices"]
        if scope == SCOPES[1]:
            assert x.index_base_year.iloc[pool.idx.to_numpy(int)].eq(x.index_base_year.iloc[i]).all()
        for h in HORIZONS:
            stats = distribution.empirical_statistics(sample.loc[record["selected_indices"], f"gross_return{h}"])
            for old, field in [("mu5", "mu"), ("variance5", "variance"), ("q05", "q05"), ("es95", "es95")]:
                np.testing.assert_allclose(stats[old], lookup.loc[(i, scope, model), f"{field}_{h}"], atol=1e-14, rtol=0)
    dates = sorted({key[0] for key in pools})
    checked = []
    for i in sorted({dates[0], dates[len(dates) // 2], dates[-1]}):
        altered = labels.copy()
        forbidden = altered.exit_idx.ge(i) | altered.date.lt(x.date.iloc[i] - pd.DateOffset(years=2))
        altered.loc[forbidden, ["gross_return5", "gross_return20"]] = -9999.
        for scope in SCOPES:
            a, _, _ = fit_day(i, x.iloc[:i + 1], groups[i], count, labels, scope)
            b, _, _ = fit_day(i, x.iloc[:i + 1], groups[i], count, altered, scope)
            assert a == b
        checked.append(x.date.iloc[i])
    return {"saved_distributions_recomputed": len(saved) * 2, "source_clock_inside_two_years": True,
        "both_horizons_strictly_mature": True, "future_and_expired_label_perturbation_dates": checked}


def forecast_diagnostics(pred, labels, x):
    scored = pred.merge(labels[["idx", "gross_return5", "gross_return20", "exit_idx"]], on="idx", how="left", validate="many_to_one")
    # 联合池最后20日标签不全；风险监控另接保存的完整五日成熟结果。
    full5 = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")[["idx", "gross_return5"]]
    scored = scored.drop(columns="gross_return5").merge(full5, on="idx", how="left", validate="many_to_one")
    scored = scored.merge(x[["idx", "pressure5", "index_base_year"]], on="idx", how="left", validate="many_to_one")
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    evaluation, monitors = [], []
    first = int(x.index[x.date.ge(START)][0])
    for (scope, model), rows in scored.groupby(["scope", "model"]):
        for h in HORIZONS:
            mature = rows[rows[f"gross_return{h}"].notna()].copy()
            error = mature[f"gross_return{h}"] - mature[f"q05_{h}"]
            evaluation.append({"scope": scope, "model": model, "horizon": h, "n": len(mature),
                "MSE": float(((mature[f"gross_return{h}"] - mature[f"mu_{h}"]) ** 2).mean()),
                "q05_breach_fraction": float(error.lt(0).mean()), "quantile_loss": float(np.maximum(.05 * error, -.95 * error).mean())})
            sample = mature[(mature.idx - first) % h == 0].copy()
            sample["breach"] = sample[f"gross_return{h}"].lt(sample[f"q05_{h}"])
            sample["breach_rate60"] = sample.breach.rolling(60, min_periods=60).mean()
            monitors.append({"scope": scope, "model": model, "horizon": h, "nonoverlap_rows": len(sample),
                "assessable_rows": int(sample.breach_rate60.notna().sum()), "alerts": int(sample.breach_rate60.gt(.15).sum())})
    return evaluation, monitors


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for p, sha in frozen["sources"].items():
        assert digest(ROOT / p) == sha, p
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    x = information()
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    labels = labels_for_horizons(market, dividends)
    old = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet").set_index("idx")
    np.testing.assert_allclose(labels.gross_return5, old.loc[labels.idx, "gross_return5"], atol=1e-14, rtol=0)
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    labels.to_parquet(OUT / "inputs/mature_labels.parquet", index=False)
    count = len(pd.read_parquet(prior.funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"]))
    nested = pd.read_parquet(prior.funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    rows, saved, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        for scope in SCOPES:
            p, m, r = fit_day(int(i), x, groups.get(int(i)), count, labels, scope)
            rows.extend(p)
            saved.extend(m)
            receipts.append(r)
        if len(receipts) % 600 == 0:
            print(f"住房两期限共同分布已算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(rows), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", saved, True)
    checks = verify_predictions(pred, saved, x, labels, groups, count)
    evaluation, monitors = forecast_diagnostics(pred, labels, x)
    accounts, account_checks, cycle_rows = {}, [], []
    for scope in SCOPES:
        for h in HORIZONS if scope == SCOPES[0] else [20]:
            for model in [CONTROL, SIGNAL]:
                policy = policy_name(model, scope, h)
                for cost in distribution.COSTS:
                    ledger, decisions = simulate(market, x, dividends, pred, model, scope, h, cost)
                    verified = parent.account_verification(ledger, decisions)
                    assert not (ledger.shares.shift(1, fill_value=0).gt(0) & ledger.filled_quantity.gt(0)).any()
                    due = decisions[decisions.reason.eq("FIXED_HORIZON_MATURITY")]
                    assert due.idx.ge(due.deadline_idx).all()
                    folder = OUT / "accounts" / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    ledger.to_parquet(folder / "ledger.parquet", index=False)
                    decisions.to_parquet(folder / "decisions.parquet", index=False)
                    finished, pending = cycles(ledger)
                    finished.to_parquet(folder / "cycles.parquet", index=False)
                    save(folder / "pending_cycle.json", pending, True)
                    accounts[policy, cost] = ledger
                    account_checks.append({"policy": policy, "cost": cost, **verified})
                    cycle_rows.append({"policy": policy, "cost": cost, "closed": len(finished), "open": pending is not None})
                    m = metrics(ledger)
                    print(f"{policy}/{cost}：夏普{sharpe_text(m['sharpe'])}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092613)
    rebase_first_decision = x.loc[x.index_base_year.eq(2025), "date"].min().strftime("%Y-%m-%d")
    for scope in SCOPES:
        for h in HORIZONS if scope == SCOPES[0] else [20]:
            for cost in distribution.COSTS:
                left = accounts[policy_name(SIGNAL, scope, h), cost]
                right = accounts[policy_name(CONTROL, scope, h), cost]
                periods = []
                for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END), (rebase_first_decision, END)]:
                    mask = left.date.between(lo, hi)
                    periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
                comparisons.append({"scope": scope, "horizon": h, "cost": cost, "purpose": "HOUSING_INFORMATION_INCREMENT",
                    "daily_blocks": paired_interval(left, right, rng), "release_blocks": loan.release_block_interval(left, right, x.assign(loan_stat_month=x.housing_stat_month), rng), "fixed_periods": periods})
    for cost in distribution.COSTS:
        comparisons.append({"scope": SCOPES[0], "horizon": 20, "cost": cost, "purpose": "TWENTY_MINUS_FIVE_HORIZON",
            "daily_blocks": paired_interval(accounts[PRIMARY, cost], accounts[policy_name(SIGNAL, SCOPES[0], 5), cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increments = [r for r in comparisons if r["purpose"] == "HOUSING_INFORMATION_INCREMENT" and r["horizon"] == 20 and r["cost"] == "STRESS"]
    gate = all(r["daily_blocks"]["lower_95"] > 0 and r["release_blocks"]["lower_95"] > 0 for r in increments)
    main_increment = next(r for r in increments if r["scope"] == SCOPES[0])
    gate &= all(p["annual_arithmetic_difference"] > 0 for p in main_increment["fixed_periods"][:2])
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, main["profit"], atol=1e-6, rtol=0)
    schedules = []
    for scope, group in schedule.groupby("scope"):
        ok = group[group.status.eq("UPDATED")]
        schedules.append({"scope": scope, "statuses": group.status.value_counts().to_dict(),
            "first_prediction": ok.date.min(), "last_prediction": ok.date.max(),
            "training_rows_min": int(ok.n_train.min()), "training_rows_max": int(ok.n_train.max()),
            "release_months_min": int(ok.release_months.min()), "release_months_max": int(ok.release_months.max()),
            "last_status": group.iloc[-1].to_dict()})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and gate else "FROZEN_NO_QUALIFIED_HOUSING_COLLATERAL_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons, "development_increment_gate": gate,
        "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "new_full_accounts": len(accounts), "prediction_records": len(pred), "conditional_distributions": len(pred) * 2,
        "unique_prediction_days": int(pred.idx.nunique()), "schedules": schedules, "forecast_evaluation": evaluation,
        "mature_monitors": monitors, "prediction_checks": checks, "account_checks": account_checks, "cycle_counts": cycle_rows,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": main["profit"]},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "source_months": 68, "new_parameter_searches": 0, "new_funding_regression_fits": 0,
        "new_independent_forward_observations": 0, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False, "orders_authorized": False}, True)
    print("住房下跌范围与两种期限的12个完整账户已计算完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="住房价格下跌覆盖范围的日更研究")
    parser.add_argument("command", choices=["check", "freeze", "run"])
    command = parser.parse_args().command
    if command == "check":
        print(implementation_checks())
    elif command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
