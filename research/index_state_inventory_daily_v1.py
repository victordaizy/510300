"""纯510300指数账户：最近两年每日状态分布、资金增量与库存风险预算。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
PARENT = ROOT / "reports/research/510300_daily_liquidity_insurance_tail_v1"
OUT = ROOT / "reports/research/510300_index_state_inventory_daily_v1"
STUDY = "510300_INDEX_STATE_INVENTORY_DAILY_V1"
PRICE = ["pressure5", "trend20", "log_rv5_rv60"]
MACRO = [*PRICE, "credit_acceleration3_pp", "funding_gap_pp"]
MODELS = {"HISTORY": [], "PRICE": PRICE, "MACRO": MACRO}
POLICIES = ["HISTORY", "PRICE_DOWN_ONLY", "PRICE", "MACRO", "BUY_HOLD"]


def modules(root):
    path = root / "code/distribution_primitives.py"
    if not path.exists():
        path = PARENT / "code/daily_liquidity_insurance_tail_v1.py"
    spec = importlib.util.spec_from_file_location("index_state_distribution", path)
    h = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = h
    spec.loader.exec_module(h)
    e = h.engine(root if (root / "code/account_engine.py").exists() else PARENT)
    return h, e


def freeze(root):
    h, _ = modules(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("指数状态研究已有冻结记录。")
    for folder in ["inputs", "code", "results", "accounts"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    paths = {"decision_information.parquet": PARENT / "results/decision_information.parquet",
             "market.parquet": PARENT / "results/market.parquet", "mature_labels.parquet": PARENT / "results/mature_labels.parquet",
             "dividends.csv": PARENT / "inputs/dividends.csv",
             "authority_update.json": ROOT / "reports/research/510300_index_only_resume_20260925/authority_update.json",
             "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
             "prior_result.json": PARENT / "result.json"}
    for name, path in paths.items():
        shutil.copy2(path, root / "inputs" / name)
    shutil.copy2(__file__, root / "code/index_state_inventory_daily_v1.py")
    shutil.copy2(PARENT / "code/daily_liquidity_insurance_tail_v1.py", root / "code/distribution_primitives.py")
    shutil.copy2(PARENT / "code/account_engine.py", root / "code/account_engine.py")
    protocol = {
        "study_id": STUDY, "at": h.now(), "user_instruction": "不做期权了，我只做指数",
        "assets": ["510300.SH", "CASH_CNY"], "primary": "MACRO", "capital": 200000,
        "period": [h.START, h.END], "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "question": "在纯指数的共同资料下，资金利差和信用变化能否通过少量条件交互改善收益及尾部分布，并改善完整库存账户。",
        "price_features": PRICE, "macro_features": MACRO[len(PRICE):],
        "definitions": "三个价格特征沿用已冻结的五日抛压、二十日趋势和5/60日实现波动比，均至昨收盘；社融同比三月变化及DR007-政策利率沿用逐项发布时间。不称NFCI或中性利率差。",
        "option_inputs_used": False, "pure_index_source_columns": "仅选日期、价格特征、信用与利率及各自可用时钟；既有宽面板中的期权字段不读入模型或共同覆盖条件。",
        "training": "每天仅使用当日向前两个日历年内、五日退出开盘严格早于当天的成熟标签；最少252共同样本。未来是否有结果不限制当前原点。",
        "target": "开盘至第五个后续交易日开盘的标的含已获分红权益收益；不含入场前跳空，不把重叠五日收益当独立样本。",
        "models": {"HISTORY": "全部共同成熟历史的经验分布", "PRICE": "价格三变量的平方误差树，max_depth2、min_samples_leaf126、random_state20260925",
                   "MACRO": "在相同树设置及共同训练池内增加两资金变量；当前叶内历史构成条件收益分布"},
        "tree_limits": "最多四个叶，实际两年样本通常至多三个；不扫描深度、叶数或变量；按训练均值和样本标准差标准化并截断[-5,5]，保存全部状态条件与样本。",
        "moments": "当前叶内收益均值、样本方差、5%分位和最差ceil(n*5%)个收益均值的损失；包含所有尾部观察，不只保留成功形态。",
        "decision": "逐日枚举合法100份目标，最大化w*mu5-2*w^2*variance5-调整成本/权益-退出储备/权益；允许上涨趋势或下跌修复中的正期望入场；缺预测最长保留五日。",
        "ablation": "PRICE_DOWN_ONLY复用同一PRICE预测与风险预算，只额外禁止五日抛压非正时增仓；其余账户不预设买入方向状态。",
        "risk": {"five_day_ES95_budget": .025, "gap_stress_return": -.1, "gap_loss_budget": .05,
                 "maximum_position_fraction": .5, "drawdown_headroom_fraction": .5, "drawdown_trigger": .1,
                 "after_trigger": "下一可卖开盘退出且本次不恢复；T+1与涨跌停可能使实际回撤超过触发值"},
        "execution": "09:00据昨收盘及已知除息额给固定份额；开盘仅能因预算或现金缩小买入，不扩张；T+1、100份、tick0.001、分红登记应收到账分账。",
        "terminal": "没有利用最后样本日预先卖出；照常决定并收盘计价，末端若有库存额外计压力清算费用储备，保留未平仓状态。",
        "costs": h.COSTS, "annual_days": 252, "cash_interest": 0,
        "monitor": "五日固定相位成熟原点，60个原点的5%预测分位越界率>15%记录预警；不看完结果再改启停门槛。宏观月度重复值不当新增独立公告。",
        "comparison": "MACRO对PRICE是主比较；PRICE对PRICE_DOWN_ONLY为方向状态限制诊断；固定20日区块2000次seed2026092510。另保留固定2021—2023/2024—末端及所有滚动两年。",
        "candidate_gate": "压力账户三目标同时达到，主增量95%区间下界>0且两个固定期间增量都正，才列开发候选；仍需独立验证。",
        "duplicate_boundary": ["旧直接效用研究为季度更新、扩展/756日、多特征连续策略优化；本次是用户要求的两年日更、最多四状态条件分布。",
                               "旧条件承接研究固定最近126邻居、加入期权定价信息且只允许五日下跌后增仓；本次不使用期权，资金交互由浅树形成状态，并独立比较买入方向限制。",
                               "宏观信息和价格历史已经反复研究；本次是有限开发比较，不宣称新独立数据或复活旧终止组合。"],
        "parameter_search": False, "new_market_downloads": 0, "orders_authorized": False,
    }
    h.save(root / "protocol.json", protocol, True)
    files = [root / "protocol.json", *sorted((root / "code").glob("*")), *sorted((root / "inputs").glob("*"))]
    h.save(root / "freeze.json", {"at": h.now(), "before_new_state_fits_and_accounts": True,
                                 "files": {p.relative_to(root).as_posix(): h.digest(p) for p in files}}, True)
    print("纯指数两年日更状态分布与账户规则已冻结，不含期权资产或特征。", flush=True)


def inputs(root, e):
    d = pd.read_parquet(root / "inputs/market.parquet")
    cols = ["idx", "date", "decision_time", *MACRO, "funding_known", "credit_known", "dr_available_at", "policy_available_at", "credit_available_at"]
    x = pd.read_parquet(root / "inputs/decision_information.parquet", columns=cols)
    x["common_known"] = x.funding_known.fillna(False).astype(bool) & x.credit_known.fillna(False).astype(bool)
    x["common_known"] &= np.isfinite(x[MACRO]).all(axis=1)
    for col in ["dr_available_at", "policy_available_at", "credit_available_at"]:
        assert pd.to_datetime(x.loc[x.common_known, col]).le(pd.to_datetime(x.loc[x.common_known, "decision_time"])).all()
    labels = pd.read_parquet(root / "inputs/mature_labels.parquet")
    frame = x.merge(labels.drop(columns="date"), on="idx", how="left", validate="one_to_one")
    dividends = e.normalize_dividends(pd.read_csv(root / "inputs/dividends.csv"))
    return d, x, frame, dividends


def tree_leaf(tree, x):
    node, path = 0, []
    while tree["left"][node] != -1:
        j, threshold = tree["feature"][node], tree["threshold"][node]
        lower = float(np.float32(x[j])) <= threshold
        path.append({"feature_index": j, "threshold_standardized": threshold, "less_equal": lower})
        node = tree["left"][node] if lower else tree["right"][node]
    return node, path


def fit_at(current, frame, h):
    day = current["date"]
    lower = day - pd.DateOffset(years=2)
    train = frame[frame.date.ge(lower) & frame.exit_date.lt(day) & frame.common_known & frame.gross_return5.notna()]
    receipt = {"idx": int(current["idx"]), "date": day, "lower_bound": lower,
               "training_count": len(train), "latest_training_exit": train.exit_date.max() if len(train) else None}
    if len(train) < 252 or not current["common_known"]:
        return [], [], {**receipt, "status": "NO_VIEW"}
    y = train.gross_return5.to_numpy(float)
    predictions, models = [], []
    for model, columns in MODELS.items():
        mean, scale, tree, path, leaf = [], [], None, [], None
        selected = np.ones(len(train), bool)
        if columns:
            raw = train[columns].to_numpy(float)
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1
            z = np.clip((raw - mean) / scale, -5, 5)
            x = np.clip((np.array([current[k] for k in columns]) - mean) / scale, -5, 5)
            fitted = DecisionTreeRegressor(max_depth=2, min_samples_leaf=126, random_state=20260925).fit(z, y)
            t = fitted.tree_
            tree = {"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
                    "threshold": t.threshold.tolist(), "samples": t.n_node_samples.tolist()}
            leaf, path = tree_leaf(tree, x)
            selected = fitted.apply(z) == leaf
            np.testing.assert_allclose(y[selected].mean(), fitted.predict(x.reshape(1, -1))[0], atol=1e-12)
        statistics = h.empirical_statistics(y[selected])
        predictions.append({**receipt, "model": model, "leaf": leaf, "selected_count": int(selected.sum()), **statistics})
        models.append({**receipt, "model": model, "features": columns, "mean": mean, "scale": scale,
                       "tree": tree, "leaf": leaf, "state_path": path, "training_indices": train.idx.to_list(),
                       "selected_indices": train.idx.to_numpy()[selected].tolist()})
    return predictions, models, {**receipt, "status": "UPDATED"}


def learn(root, frame, h):
    pred, models, receipts = [], [], []
    for current in frame[frame.date.ge(h.START)].to_dict("records"):
        p, m, receipt = fit_at(current, frame, h)
        pred.extend(p)
        models.extend(m)
        receipts.append(receipt)
        if len(receipts) % 300 == 0:
            print(f"纯指数每日条件分布已更新至{current['date'].date()}。", flush=True)
    prediction = pd.DataFrame(pred)
    prediction.to_parquet(root / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    h.save(root / "results/saved_models.json", models)
    return prediction, models


def simulate(h, e, d, x, dividends, predictions, policy, cost_name):
    first = int(np.flatnonzero(d.date.ge(h.START))[0])
    model = "PRICE" if policy == "PRICE_DOWN_ONLY" else policy
    pred = {int(r.idx): r for r in predictions[predictions.model.eq(model)].itertuples()}
    account = e.Account(200000.)
    cfg = {"lot": 100, "tick": .001, "limit_fraction": .1}
    cost = h.COSTS[cost_name]
    previous_equity, previous_mark, previous_reserve, peak = 200000., float(d.close.iloc[first - 1]), 0., 200000.
    stopped, last_view = False, first - 6
    records, decisions = [], []
    events = dividends.to_dict("records")
    for i in range(first, len(d)):
        row = d.iloc[i]
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
        ref = float(row.previous_close - row.dividend)
        prediction = pred.get(i)
        plan = {"requested_quantity": 0, "target_shares": account.shares, "reason": "NO_VIEW_HOLD_OLD_INVENTORY"}
        if policy == "BUY_HOLD":
            if i == first:
                quantity = e.affordable_quantity(account.cash, e.fill_price(ref, 1, cost, .001), cost, 100)
                plan.update(requested_quantity=quantity, target_shares=quantity, reason="BUY_HOLD_INITIAL")
        elif stopped:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="DRAWDOWN_STOP")
        elif prediction is not None:
            last_view = i
            pressure = float(x.pressure5.iloc[i]) if policy == "PRICE_DOWN_ONLY" else 1.
            plan = {**h.plan(e, account, ref, peak, prediction, pressure), "reason": "DAILY_STATE_DISTRIBUTION"}
        elif i - last_view >= 5:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="NO_VIEW_INVENTORY_EXPIRED")
        requested = int(plan["requested_quantity"])
        before_gap = requested
        if requested > 0 and policy != "BUY_HOLD":
            while requested > 0 and not h.risk_valid(e, account.shares + requested, op, plan, h.COSTS["STRESS"]):
                requested -= 100
        sellable = account.sellable(i)
        execution = e.execute_order(account, requested, op, float(row.previous_close), float(row.dividend), i, cost, cfg)
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == len(d) - 1 and account.shares:
            exit_price = e.fill_price(close, -1, h.COSTS["STRESS"], .001)
            reserve = account.shares * (close - exit_price) + e.commission(account.shares, exit_price, h.COSTS["STRESS"])
        equity = account.value(close) - reserve
        price_pnl = old_shares * (op - previous_mark) + account.shares * (close - op)
        error = equity - previous_equity - price_pnl - recognized + execution["commission"] + execution["slippage_cost"] + reserve - previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        if policy != "BUY_HOLD" and drawdown >= .1:
            stopped = True
        records.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "open": op, "mark": close,
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "terminal_exit_reserve": reserve, "equity": equity, "net_return": equity / previous_equity - 1,
                        "pnl": equity - previous_equity, "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * close / equity, "accounting_error": error, "drawdown": drawdown,
                        "risk_stopped": stopped, "sellable_before": sellable,
                        "terminal_unliquidated": bool(i == len(d) - 1 and account.shares), **execution})
        decisions.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "prediction_available": prediction is not None,
                          "pressure5": float(x.pressure5.iloc[i]), "pre_open_requested_quantity": before_gap,
                          "gap_checked_request": requested, "filled_quantity": execution["filled_quantity"], "actual_open": op, **plan})
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def checks(h):
    dates, day = pd.date_range("2022-01-05", periods=253), pd.Timestamp("2024-01-03")
    frame = pd.DataFrame({"idx": range(253), "date": dates, "exit_date": [pd.Timestamp("2023-01-01")] * 252 + [day],
                          "common_known": True, "gross_return5": .01})
    for j, col in enumerate(MACRO):
        frame[col] = np.sin(np.arange(253) / (j + 2))
    current = {"idx": 999, "date": day, "common_known": True, **{col: 0. for col in MACRO}}
    before, models, _ = fit_at(current, frame, h)
    assert all(252 not in m["training_indices"] for m in models)
    frame.loc[252, "gross_return5"] = -1e9
    after, _, _ = fit_at(current, frame, h)
    np.testing.assert_allclose([p["mu5"] for p in before], [p["mu5"] for p in after], atol=1e-12)
    return {"future_labels_excluded": True, "training_window_two_calendar_years": True, "no_option_features": not any("iv" in k or "option" in k for k in MACRO)}


def verify(root):
    h, e = modules(root)
    h.verify_freeze(root)
    d, x, frame, _ = inputs(root, e)
    frame = frame.set_index("idx", drop=False)
    models = json.loads((root / "results/saved_models.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(root / "results/predictions.parquet").set_index(["idx", "model"])
    for model in models:
        pool = frame.loc[model["training_indices"]]
        assert pool.date.ge(pd.Timestamp(model["lower_bound"])).all() and pool.exit_date.lt(pd.Timestamp(model["date"])).all()
        selected = frame.loc[model["selected_indices"]]
        if model["tree"]:
            value = frame.loc[model["idx"], model["features"]].to_numpy(float)
            z = np.clip((value - np.array(model["mean"])) / np.array(model["scale"]), -5, 5)
            leaf, _ = tree_leaf(model["tree"], z)
            assert leaf == model["leaf"] and len(selected) >= 126
            assert sum(child == -1 for child in model["tree"]["left"]) <= 4
        stats = h.empirical_statistics(selected.gross_return5)
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(stats[key], predictions.loc[(model["idx"], model["model"]), key], atol=1e-12)
    prefix = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        chosen = [m for m in models if m["model"] == "MACRO" and pd.Timestamp(m["date"]) <= cutoff][-1]
        current = frame.loc[chosen["idx"]].to_dict()
        subset = frame[frame.date.le(current["date"])].copy()
        subset.loc[subset.exit_date.ge(current["date"]), "gross_return5"] = -1e9
        p, _, _ = fit_at(current, subset, h)
        for row in p:
            np.testing.assert_allclose(row["mu5"], predictions.loc[(row["idx"], row["model"]), "mu5"], atol=1e-12)
        prefix.append(str(current["date"].date()))
    for cost in h.COSTS:
        for policy in POLICIES:
            path = root / "accounts" / cost
            ledger = pd.read_parquet(path / f"{policy}_ledger.parquet")
            decision = pd.read_parquet(path / f"{policy}_decisions.parquet")
            np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable - ledger.terminal_exit_reserve, atol=1e-6)
            assert ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all()
            assert (-ledger.filled_quantity.clip(upper=0)).le(ledger.sellable_before).all()
            assert ledger.accounting_error.abs().max() < 1e-6
            if policy != "BUY_HOLD":
                for row in decision[decision.filled_quantity.gt(0)].to_dict("records"):
                    actual = ledger.loc[ledger.date.eq(row["date"]), "shares"].iloc[0]
                    assert h.risk_valid(e, actual, row["actual_open"], row, h.COSTS["STRESS"])
    receipt = {"at": h.now(), "saved_distributions_recomputed": len(models), "historical_prefix_checks": prefix,
               "accounts_reconciled": 10, "cash_dividend_T1_verified": True, "entry_tail_budgets_verified": True, "mechanism_checks": checks(h)}
    h.save(root / "verification.json", receipt)
    return receipt


def run(root):
    h, e = modules(root)
    h.verify_freeze(root)
    assert h.digest(Path(__file__)) == h.digest(root / "code/index_state_inventory_daily_v1.py")
    h.save(root / "RUN_STARTED.json", {"at": h.now()}, True)
    h.save(root / "mechanism_checks.json", checks(h))
    d, x, frame, dividends = inputs(root, e)
    x.to_parquet(root / "results/decision_information.parquet", index=False)
    pred, models = learn(root, frame, h)
    scored = pred.merge(frame[["idx", "exit_date", "gross_return5"]], on="idx", validate="many_to_one")
    scored["squared_error"] = (scored.gross_return5 - scored.mu5).pow(2)
    scored["tail_breach"] = scored.gross_return5 < scored.q05
    scored.to_parquet(root / "results/scored_predictions.parquet", index=False)
    monitors = []
    anchor = int(pred.idx.min())
    for model, values in scored[scored.gross_return5.notna() & ((scored.idx - anchor) % 5).eq(0)].groupby("model"):
        values = values.sort_values("idx").copy()
        values["breach_rate60"] = values.tail_breach.rolling(60, min_periods=60).mean()
        values["observation_alert"] = values.breach_rate60.gt(.15)
        monitors.append(values)
    monitor = pd.concat(monitors, ignore_index=True)
    monitor.to_parquet(root / "results/nonoverlap_tail_monitor.parquet", index=False)
    metrics, ledgers, yearly, rolling = [], {}, [], []
    for cost in h.COSTS:
        folder = root / "accounts" / cost
        folder.mkdir(exist_ok=True)
        for policy in POLICIES:
            ledger, decisions = simulate(h, e, d, x, dividends, pred, policy, cost)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            m = {"cost": cost, "policy": policy, **h.summary(e, ledger)}
            m["max_drawdown_magnitude"] = abs(m["max_drawdown"])
            m["historical_point_targets_met"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["annualized_return"] >= .1 and m["max_drawdown_magnitude"] <= .1
            m["terminal_position"] = bool(ledger.terminal_unliquidated.iloc[-1])
            m["terminal_reserve_cny"] = float(ledger.terminal_exit_reserve.iloc[-1])
            metrics.append(m)
            ledgers[(policy, cost)] = ledger
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"policy": policy, "cost": cost, "year": int(year), **e.return_metrics(group.net_return, 252)})
            if cost == "STRESS":
                dates = pd.DatetimeIndex(ledger.date)
                for i, start in enumerate(dates):
                    end = int(dates.searchsorted(start + pd.DateOffset(years=2)))
                    if end >= len(dates):
                        break
                    measure = e.return_metrics(ledger.net_return.iloc[i:end + 1], 252)
                    rolling.append({"policy": policy, "start": start, "end": dates[end], **measure,
                                    "joint_targets": measure["net_sharpe"] is not None and measure["net_sharpe"] >= 1.2 and measure["annualized_return"] >= .1 and abs(measure["max_drawdown"]) <= .1})
            print(f"{cost}/{policy}完成：夏普{m['net_sharpe']}，年化{m['annualized_return']:.3%}，回撤{m['max_drawdown_magnitude']:.3%}。", flush=True)
    h.save(root / "results/account_metrics.json", metrics)
    h.save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    comparisons = []
    for left, right in [("MACRO", "PRICE"), ("PRICE", "PRICE_DOWN_ONLY")]:
        a, b = ledgers[(left, "STRESS")], ledgers[(right, "STRESS")]
        diff = a.net_return.to_numpy() - b.net_return.to_numpy()
        rng, draws = np.random.default_rng(2026092510), []
        for _ in range(2000):
            starts = rng.integers(0, len(diff), size=math.ceil(len(diff) / 20))
            indices = ((starts[:, None] + np.arange(20)) % len(diff)).ravel()[:len(diff)]
            draws.append(float(diff[indices].mean() * 252))
        periods = []
        for start, end in [(h.START, "2023-12-31"), ("2024-01-01", h.END)]:
            mask = a.date.between(start, end).to_numpy()
            periods.append({"start": start, "end": end, "annual_arithmetic_increment": float(diff[mask].mean() * 252)})
        comparisons.append({"left": left, "right": right, "annual_arithmetic_increment": float(diff.mean() * 252),
                            "ci95": np.quantile(draws, [.025, .975]), "fixed_periods": periods})
    primary = next(m for m in metrics if m["cost"] == "STRESS" and m["policy"] == "MACRO")
    increment_pass = comparisons[0]["ci95"][0] > 0 and all(v["annual_arithmetic_increment"] > 0 for v in comparisons[0]["fixed_periods"])
    result = {"study_id": STUDY, "at": h.now(), "primary": primary, "goal_achieved": False,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment_pass else "FROZEN_NO_QUALIFIED_INDEX_STATE_INVENTORY",
              "daily_distribution_updates": len(models), "new_tree_fits": sum(m["tree"] is not None for m in models),
              "unique_prediction_days": int(pred.idx.nunique()), "new_accounts": 10,
              "prediction_metrics": scored[scored.gross_return5.notna()].groupby("model").agg(n=("idx", "size"), MSE=("squared_error", "mean"), tail_breach=("tail_breach", "mean")).reset_index().to_dict("records"),
              "tail_monitor_alerts": monitor.groupby("model").observation_alert.sum().to_dict(),
              "account_increments": comparisons, "macro_increment_gate_met": increment_pass,
              "all_account_metrics": metrics, "historical_candidates": [m for m in metrics if m["historical_point_targets_met"] and m["policy"] != "BUY_HOLD"],
              "independent_validation": False, "options_used": False, "orders_authorized": False, "new_market_downloads": 0}
    h.save(root / "result.json", result, True)
    verification = verify(root)
    print(json.dumps(h.clean({"status": result["status"], "primary": primary, "verification": verification}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="纯指数两年每日更新状态库存实验")
    parser.add_argument("command", choices=["freeze", "run", "verify", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "run":
        run(args.out)
    elif args.command == "verify":
        h, _ = modules(args.out)
        print(json.dumps(h.clean(verify(args.out)), ensure_ascii=False, indent=2))
    else:
        h, _ = modules(args.out)
        print(json.dumps(checks(h), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
