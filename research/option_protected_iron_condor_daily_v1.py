"""同到期四腿铁鹰：固定结构、共同状态、每日两年训练与完整现金账户。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
ROUTER = ROOT / "reports/research/510300_option_credit_spread_router_daily_v1"
PAIR = ROOT / "reports/research/510300_option_paired_state_value_daily_v1"
BUYBACK = ROOT / "reports/research/510300_put_spread_buyback_cost_daily_v1"
OUT = ROOT / "reports/research/510300_option_protected_iron_condor_daily_v1"
STUDY = "510300_OPTION_PROTECTED_IRON_CONDOR_DAILY_V1"
BASE = ["trend_magnitude", "volatility_richness", "credit_to_maximum_loss"]
MACRO = ["funding_gap_pp", "credit_acceleration3_pp", "GSPC_z", "VIX_z"]
MODELS = {"EMPIRICAL": [], "LINEAR_PRICE": BASE, "TREE_PRICE": BASE, "TREE_MACRO": BASE + MACRO}
POLICIES = ["UNCONDITIONAL", *MODELS]


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condor_terms(candidate, day, quotes):
    terms = []
    for key, side, kind in [("put_long_code", 1, "P"), ("call_long_code", 1, "C"),
                            ("put_short_code", -1, "P"), ("call_short_code", -1, "C")]:
        code = candidate[key]
        q = quotes.get((day, code))
        if q is None or not np.isfinite(q.get("unit", np.nan)) or q["option_type"] != kind:
            return None
        terms.append({"code": code, "side": side, "unit": int(q["unit"]), "strike": float(q["effective_strike"]),
                      "option_type": kind, "quote": q})
    if len({t["unit"] for t in terms}) != 1 or len({t["quote"]["expiry_date"] for t in terms}) != 1:
        return None
    if not terms[0]["strike"] < terms[2]["strike"] <= terms[3]["strike"] < terms[1]["strike"]:
        return None
    return terms


def condor_width(terms):
    return max(terms[2]["strike"] - terms[0]["strike"], terms[1]["strike"] - terms[3]["strike"]) * terms[0]["unit"]


def condor_margin(terms, spot, field="settlement"):
    total = 0.
    for long_index, short_index in [(0, 2), (1, 3)]:
        short = terms[short_index]
        premium, strike = float(short["quote"][field]), short["strike"]
        if not np.isfinite(premium) or premium < 0:
            return math.inf
        if short["option_type"] == "P":
            naked = min(premium + max(.12 * spot - max(spot - strike, 0.), .07 * strike), strike) * short["unit"]
        else:
            naked = (premium + max(.12 * spot - max(strike - spot, 0.), .07 * spot)) * short["unit"]
        wing = abs(short["strike"] - terms[long_index]["strike"]) * short["unit"]
        total += max(wing, naked)
    return 1.2 * total


def modules(root):
    rpath = root / "code/router_source.py"
    router = imported("condor_sources", rpath if rpath.exists() else ROUTER / "code/option_credit_spread_router_daily_v1.py")
    hpath = root / "code/four_leg_cash_engine.py"
    h = imported("condor_cash_engine", hpath if hpath.exists() else ROUTER / "code/spread_engine.py")
    h.terms_at, h.width_cash, h.margin = condor_terms, condor_width, condor_margin
    return router, h


def four_leg_source():
    source = (ROUTER / "code/spread_engine.py").read_text(encoding="utf-8")
    transformations = [
        ("ordered = terms if opening else [terms[1], terms[0]]", "ordered = terms if opening else [t for t in terms if t['side'] < 0] + [t for t in terms if t['side'] > 0]", 1),
        ('short_buy = fill(terms[1]["quote"]["open"], 1, scenario) * terms[1]["unit"] * quantity + FEE * quantity',
         'short_buy = sum(fill(t["quote"]["open"], 1, scenario) * t["unit"] + FEE for t in terms if t["side"] < 0) * quantity', 1),
        ('prior_long_cost = fill(prior_terms[0]["quote"]["close"], 1, "STRESS") * prior_terms[0]["unit"] + FEE',
         'prior_long_cost = sum(fill(t["quote"]["close"], 1, "STRESS") * t["unit"] + FEE for t in prior_terms if t["side"] > 0)', 1),
        ('long_cost = fill(terms[0]["quote"]["open"], 1, "STRESS") * terms[0]["unit"] + FEE',
         'long_cost = sum(fill(t["quote"]["open"], 1, "STRESS") * t["unit"] + FEE for t in terms if t["side"] > 0)', 1),
    ]
    receipt = []
    for before, after, count in transformations:
        assert source.count(before) == count
        source = source.replace(before, after)
        receipt.append({"before": before, "after": after, "occurrences": count})
    for before, after in [("4 * FEE", "8 * FEE"), ("2 * FEE", "4 * FEE"), (".0002 * terms[0]", ".0004 * terms[0]")]:
        count = source.count(before)
        assert count > 0
        source = source.replace(before, after)
        receipt.append({"before": before, "after": after, "occurrences": count})
    return source, receipt


def freeze(root):
    _, h = modules(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("四腿规则已有冻结记录。")
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    paths = {"put_information.parquet": BUYBACK / "results/information_and_labels.parquet",
             "call_information.parquet": ROUTER / "results/call_information_and_labels.parquet",
             "previous_state_result.json": PAIR / "result.json",
             "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json"}
    for name in ["option_eod.parquet", "option_risk.parquet", "daily_terms.parquet", "market.parquet",
                 "calendar_prices.parquet", "dividends.csv", "authority_update.json"]:
        paths[name] = ROUTER / "inputs" / name
    for name, source in paths.items():
        shutil.copy2(source, root / "inputs" / name)
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    shutil.copy2(ROUTER / "code/option_credit_spread_router_daily_v1.py", root / "code/router_source.py")
    code, edits = four_leg_source()
    (root / "code/four_leg_cash_engine.py").write_text(code, encoding="utf-8")
    h.save(root / "code/engine_derivation.json", {"source": str(ROUTER / "code/spread_engine.py"), "changes": edits})
    protocol = {
        "study_id": STUDY, "at": h.now(), "primary": "TREE_MACRO", "policies": POLICIES,
        "economic_question": "同时提供上下两侧有保护保险，削弱选择单侧所混入的方向判断，剩余方差风险补偿能否覆盖四腿成本。",
        "structure": "复用原0.30/0.10Delta两侧合约；仅同到期、同单位、认沽短K不高于认购短K的组合；四腿各一张，两个保护先买再卖两侧保险",
        "selection_not_tuned": "不改0.30/0.10选腿、30—75日到期优先45日、成交量100和持仓500门、原观察日及十日持有；不为凑同到期重新选腿",
        "maximum_payoff_liability": "非交叠四腿到期最大负债为两侧翼宽金额较大值；含费最大损失=最大翼宽-压力净权利金+八次每张5元费用+预估退出滑点",
        "margin": "全程1.2倍两侧各max(翼宽,单腿义务仓保证金)之和，不假享组合减免；逐腿检查先买两保护、退出先买平两空头的现金",
        "new_target": "(观察日四腿压力净权利金-第10日四腿压力买回成本-40元费用)/事前最大损失",
        "features": {"price": BASE, "macro": MACRO, "trend_magnitude": "昨日已知20日趋势的绝对值",
                     "volatility_richness": "两侧log(隐波/实现波动率)均值", "credit_to_maximum_loss": "观察日压力净权利金/事前最大损失"},
        "models": {"UNCONDITIONAL": "相同资料可用日持续报价", "EMPIRICAL": "过去两年成熟净收益比均值",
                   "LINEAR_PRICE": "固定三价格特征岭回归lambda10", "TREE_PRICE": "三价格特征深度2且叶最少63样本的树",
                   "TREE_MACRO": "加四宏观，树设置相同，random_state20260925"},
        "training": "最近两个日历年，每日更新；最少252已成熟十日标签、全部模型共同历史池；train均值std并截断[-5,5]",
        "entry": "预测净收益比正才入场；信用下限=max(观察日信用-预测比*已知最大损失,40元费用+原退出滑点储备)，实际压力报价不足不成交",
        "tail": "另以实际5日四腿压力损益/事前损失，过去两年252成熟样本，原三个状态特征最近邻126，最差5%均值ES；不得把两侧ES直接相加",
        "risk": "20万元、每组初始最大亏损5%、入场五日ES95预算2.5%、回撤余量一半、逐日剩余最大亏损减张、DD10后停止新增；不是回撤保证",
        "costs": "保持BASE/STRESS，日开盘价格代理，每腿每侧5元，不利tick取整，四腿和双向均收费",
        "tail_monitor": "固定五日相位，60个成熟非交叠原点的q05越界率>15%记预警；只监控，不事后改变账户启停",
        "evaluation": "完整账户、固定前后期间、全部滚动两年，20日区块2000次seed2026092506；已见历史开发证据",
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "limitation": "组合降低方向倾斜但不保证Delta为零；不同腿日线不证明同步成交；没有增加独立未来样本",
        "new_market_downloads": 0, "orders_authorized": False,
    }
    h.save(root / "protocol.json", protocol, exclusive=True)
    files = [root / "protocol.json", *sorted((root / "code").glob("*")), *sorted((root / "inputs").glob("*"))]
    h.save(root / "freeze.json", {"at": h.now(), "before_new_condor_payoff_labels": True,
                                 "files": {p.relative_to(root).as_posix(): h.digest(p) for p in files}}, exclusive=True)
    print("同到期四腿铁鹰与全部费用、状态及尾部规则已冻结。", flush=True)


def information(root, h, quotes):
    put = pd.read_parquet(root / "inputs/put_information.parquet")
    call = pd.read_parquet(root / "inputs/call_information.parquet").set_index("idx", drop=False)
    rows = []
    for p in put.to_dict("records"):
        current = {k: p[k] for k in ["idx", "date", "observed_date", "exit5_date", "exit10_date", "expiry", "observed_spot", "trend20", "trend20_current", *MACRO]}
        current.update(selection_status="INCOMPATIBLE_KNOWN_STRUCTURE", features_known=False)
        c = call.loc[p["idx"]].to_dict() if p["idx"] in call.index else None
        if c is None or p["selection_status"] != "READY" or c["selection_status"] != "READY":
            rows.append(current)
            continue
        current.update(put_long_code=p["long_code"], put_short_code=p["short_code"],
                       call_long_code=c["long_code"], call_short_code=c["short_code"])
        terms = h.terms_at(current, p["observed_date"], quotes)
        if not h.valid_quote(terms, "close"):
            rows.append(current)
            continue
        prior_credit = h.credit(terms, "close", "STRESS")
        raw_credit = -sum(t["side"] * t["unit"] * t["quote"]["close"] for t in terms)
        exit_slip = h.close_cost(terms, "close", "STRESS") - raw_credit
        width = h.width_cash(terms)
        loss = width - prior_credit + 8 * h.FEE + exit_slip
        current.update(selection_status="READY", prior_stress_credit=prior_credit, raw_credit=raw_credit,
                       prior_exit_slip=exit_slip, width_cash=width, known_loss=loss,
                       capacity=min(p["capacity"], c["capacity"]), log_iv=(p["log_iv"] + c["log_iv"]) / 2,
                       log_iv_rv20=(p["log_iv_rv20"] + c["log_iv_rv20"]) / 2,
                       volatility_richness=(p["log_iv_rv20"] + c["log_iv_rv20"]) / 2,
                       trend_magnitude=abs(p["trend20_current"]), credit_to_maximum_loss=prior_credit / loss,
                       entry_net_delta=p["long_delta"] - p["short_delta"] + c["long_delta"] - c["short_delta"])
        current["features_known"] = bool(p["features_known"] and c["features_known"] and np.isfinite([current[k] for k in BASE + MACRO]).all() and loss > 0)
        rows.append(current)
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / "results/candidates.parquet", index=False)
    return frame


def fit_at(current, frame):
    day = current["date"]
    lower = day - pd.DateOffset(years=2)
    train = frame[frame.date.ge(lower) & frame.exit10_date.lt(day) & frame.status10.eq("READY") & frame.features_known]
    train = train[np.isfinite(train[BASE + MACRO + ["value"]]).all(axis=1)]
    receipt = {"idx": int(current["idx"]), "date": day, "lower_bound": lower, "training_count": len(train),
               "latest_training_exit": train.exit10_date.max() if len(train) else None}
    if len(train) < 252 or not current["features_known"]:
        return [], []
    y = train.value.to_numpy(float)
    predictions, models = [], []
    for model, features in MODELS.items():
        mean, scale, beta, tree, path = [], [], [], None, []
        intercept = float(y.mean())
        value = intercept
        if features:
            raw = train[features].to_numpy(float)
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1
            z = np.clip((raw - mean) / scale, -5, 5)
            x = np.clip((np.array([current[k] for k in features]) - mean) / scale, -5, 5)
            if model.startswith("TREE"):
                estimator = DecisionTreeRegressor(max_depth=2, min_samples_leaf=63, random_state=20260925).fit(z, y)
                t = estimator.tree_
                tree = {"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
                        "threshold": t.threshold.tolist(), "value": t.value.ravel().tolist(), "samples": t.n_node_samples.tolist()}
                value, path = tree_value(tree, x)
                np.testing.assert_allclose(value, estimator.predict(x.reshape(1, -1))[0], atol=1e-12)
            else:
                center = z.mean(axis=0)
                beta = np.linalg.solve((z - center).T @ (z - center) + 10 * np.eye(len(features)), (z - center).T @ (y - y.mean()))
                intercept = float(y.mean() - center @ beta)
                value = float(intercept + x @ beta)
        predictions.append({**receipt, "model": model, "predicted_value": value})
        models.append({**receipt, "model": model, "features": features, "training_indices": train.idx.to_list(),
                       "mean": mean, "scale": scale, "beta": beta, "intercept": intercept, "tree": tree, "state_path": path})
    return predictions, models


def tree_value(tree, x):
    node, path = 0, []
    while tree["left"][node] != -1:
        j, threshold = tree["feature"][node], tree["threshold"][node]
        lower = float(np.float32(x[j])) <= threshold
        path.append({"feature_index": j, "threshold_standardized": threshold, "less_equal": lower})
        node = tree["left"][node] if lower else tree["right"][node]
    return float(tree["value"][node]), path


def learn(root, h, frame):
    predictions, models, tails, signals = [], [], [], []
    for current in frame[frame.date.ge(h.START) & frame.selection_status.eq("READY")].to_dict("records"):
        pred, fitted = fit_at(current, frame)
        predictions.extend(pred)
        models.extend(fitted)
        tail, tail_model = h.tail_at(current["date"], current, frame)
        if tail_model:
            tails.append({**tail_model, "idx": current["idx"]})
        if not fitted or tail_model is None:
            continue
        row = {**current, **tail}
        row["minimum_credit_UNCONDITIONAL"] = 8 * h.FEE + row["prior_exit_slip"]
        row["gate_UNCONDITIONAL"] = row["prior_stress_credit"] > row["minimum_credit_UNCONDITIONAL"]
        for p in pred:
            model, value = p["model"], p["predicted_value"]
            minimum = max(row["prior_stress_credit"] - value * row["known_loss"], 8 * h.FEE + row["prior_exit_slip"])
            row[f"minimum_credit_{model}"] = minimum
            row[f"gate_{model}"] = value > 0 and row["prior_stress_credit"] > minimum
        signals.append(row)
        if len(signals) % 300 == 0:
            print(f"四腿净收益与尾部分布已更新至{current['date'].date()}。", flush=True)
    prediction = pd.DataFrame(predictions)
    signal = pd.DataFrame(signals)
    prediction.to_parquet(root / "results/predictions.parquet", index=False)
    signal.to_parquet(root / "results/signals.parquet", index=False)
    h.save(root / "results/saved_models.json", models)
    h.save(root / "results/tail_models.json", tails)
    return prediction, models, tails, signal


def mechanism_checks(h):
    terms = [{"side": side, "strike": strike, "unit": 10000} for side, strike in [(1, 2.8), (1, 4.3), (-1, 3.0), (-1, 4.0)]]
    payoff = []
    for spot in np.linspace(0, 8, 1601):
        payoff.append(sum(t["side"] * t["unit"] * max((t["strike"] - spot) if i in [0, 2] else (spot - t["strike"]), 0.) for i, t in enumerate(terms)))
    np.testing.assert_allclose(min(payoff), -condor_width(terms), atol=1e-7)
    np.testing.assert_allclose(max(payoff), 0., atol=1e-7)
    dates = pd.date_range("2022-01-05", periods=253)
    day = pd.Timestamp("2024-01-03")
    frame = pd.DataFrame({"idx": range(253), "date": dates, "exit10_date": [pd.Timestamp("2023-01-01")] * 252 + [day],
                          "features_known": True, "status10": "READY", "value": .1})
    for j, col in enumerate(BASE + MACRO):
        frame[col] = np.sin(np.arange(253) / (j + 2))
    current = {"idx": 999, "date": day, "features_known": True, **{col: 0. for col in BASE + MACRO}}
    before, models = fit_at(current, frame)
    assert all(252 not in model["training_indices"] for model in models)
    frame.loc[252, "value"] = -1e9
    after, _ = fit_at(current, frame)
    np.testing.assert_allclose([p["predicted_value"] for p in before], [p["predicted_value"] for p in after], atol=1e-12)
    return {"asymmetric_wings_maximum_payoff_verified": True, "no_positive_terminal_liability_value": True,
            "future_labels_do_not_affect_model": True, "four_legs_eight_commissions": 8 * h.FEE == 40}


def verify(root):
    _, h = modules(root)
    h.verify_freeze(root)
    frame = pd.read_parquet(root / "results/information_and_labels.parquet").set_index("idx", drop=False)
    models = json.loads((root / "results/saved_models.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(root / "results/predictions.parquet").set_index(["idx", "model"])
    for model in models:
        train = frame.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all() and train.exit10_date.lt(pd.Timestamp(model["date"])).all()
        value = model["intercept"]
        if model["features"]:
            current = frame.loc[model["idx"], model["features"]].to_numpy(float)
            x = np.clip((current - np.array(model["mean"])) / np.array(model["scale"]), -5, 5)
            value = tree_value(model["tree"], x)[0] if model["tree"] else value + x @ np.array(model["beta"])
        np.testing.assert_allclose(value, pred.loc[(model["idx"], model["model"]), "predicted_value"], atol=1e-12)
    tails = json.loads((root / "results/tail_models.json").read_text(encoding="utf-8"))
    for model in tails:
        current = frame.loc[model["idx"]]
        recomputed, stored = h.tail_at(pd.Timestamp(model["date"]), current, frame)
        np.testing.assert_allclose(recomputed["ES95_risk_unit"], model["ES95_risk_unit"], atol=1e-12)
        assert stored["neighbor_indices"] == model["neighbor_indices"]
    prefixes = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        day = frame.loc[frame.date.le(cutoff) & frame.features_known, "date"].max()
        current = frame.loc[frame.date.eq(day)].iloc[0].to_dict()
        subset = frame[frame.date.le(day)].copy()
        subset.loc[subset.exit10_date.ge(day), "value"] = 1e9
        prefix, _ = fit_at(current, subset)
        for p in prefix:
            np.testing.assert_allclose(p["predicted_value"], pred.loc[(p["idx"], p["model"]), "predicted_value"], atol=1e-12)
        prefixes.append(str(day.date()))
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            path = root / "accounts" / scenario / policy
            ledger, fills, decisions = [pd.read_parquet(path / f"{name}.parquet") for name in ["ledger", "fills", "decisions"]]
            np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.option_value - ledger.terminal_exit_reserve, atol=1e-6)
            if len(fills):
                flow = fills.groupby("date").cash_change.sum().reindex(ledger.date, fill_value=0.).to_numpy()
                np.testing.assert_allclose(ledger.cash, h.INITIAL + np.cumsum(flow), atol=1e-6)
                assert fills.cash_after_leg.ge(-1e-7).all()
                assert fills.groupby(["trade_id", "date", "opening"]).size().eq(4).all()
                for _, group in fills.groupby(["trade_id", "date", "opening"], sort=False):
                    assert group.action.to_list() == [1, 1, -1, -1]
            entered = decisions[decisions.action.eq("ENTRY_FILLED")]
            if len(entered):
                assert entered.actual_maximum_loss.le(entered.budget + 1e-7).all()
                assert entered.actual_ES95.le(.025 * entered.known_equity + 1e-7).all()
    result = {"at": h.now(), "saved_models_recomputed": len(models), "tail_updates_recomputed": len(tails),
              "prefix_and_future_poisoning": prefixes, "accounts_reconciled": 10,
              "four_leg_fills_and_cash_order_verified": True, "entry_risk_budgets_verified": True,
              "mechanism_checks": mechanism_checks(h)}
    h.save(root / "verification.json", result)
    return result


def run(root):
    r, h = modules(root)
    h.verify_freeze(root)
    assert h.digest(Path(__file__)) == h.digest(root / "code/option_protected_iron_condor_daily_v1.py")
    h.save(root / "RUN_STARTED.json", {"at": h.now()}, exclusive=True)
    (root / "results").mkdir(exist_ok=True)
    h.save(root / "mechanism_checks.json", mechanism_checks(h))
    _, quotes, market, calendar, events = r.sources(root, h)
    candidates = information(root, h, quotes)
    labels = h.payoff_labels(root, candidates, quotes, calendar)
    frame = candidates.merge(labels.drop(columns=["date", "exit5_date", "exit10_date"]), on="idx", how="left", validate="one_to_one")
    frame["future_buyback_cost"] = np.nan
    for idx, row in frame[frame.status10.eq("READY")].iterrows():
        frame.loc[idx, "future_buyback_cost"] = h.close_cost(h.terms_at(row, row.exit10_date, quotes), "open", "STRESS")
    frame["value"] = (frame.prior_stress_credit - frame.future_buyback_cost - 8 * h.FEE) / frame.known_loss
    frame.to_parquet(root / "results/information_and_labels.parquet", index=False)
    pred, models, tails, signals = learn(root, h, frame)
    scored = pred.merge(frame[["idx", "status10", "value"]], on="idx", validate="many_to_one")
    scored = scored[scored.status10.eq("READY")].copy()
    scored["MSE"] = (scored.predicted_value - scored.value).pow(2)
    scored.to_parquet(root / "results/scored_predictions.parquet", index=False)
    metrics, yearly, rolling, ledgers = [], [], [], {}
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            account = h.simulate(policy, scenario, signals, quotes, market, calendar, events)
            path = root / "accounts" / scenario / policy
            path.mkdir(parents=True, exist_ok=True)
            for name, value in account.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(path / f"{name}.parquet", index=False)
                else:
                    h.save(path / f"{name}.json", value)
            measure = h.account_metrics(policy, scenario, account)
            metrics.append(measure)
            y, w = h.yearly_and_rolling(policy, scenario, account["ledger"])
            yearly.extend(y)
            rolling.extend(w)
            ledgers[(policy, scenario)] = account["ledger"]
            print(f"{scenario}/{policy}完成：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.3%}，回撤{measure['max_drawdown']:.3%}。", flush=True)
    h.save(root / "results/account_metrics.json", metrics)
    h.save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    comparisons = []
    for left, right in [("TREE_MACRO", "TREE_PRICE"), ("TREE_PRICE", "LINEAR_PRICE")]:
        diff = ledgers[(left, "STRESS")]["return"].to_numpy() - ledgers[(right, "STRESS")]["return"].to_numpy()
        rng, draws = np.random.default_rng(2026092506), []
        for _ in range(2000):
            starts = rng.integers(0, len(diff), size=math.ceil(len(diff) / 20))
            idx = ((starts[:, None] + np.arange(20)) % len(diff)).ravel()[:len(diff)]
            draws.append(float(diff[idx].mean() * 252))
        comparisons.append({"left": left, "right": right, "annual_arithmetic_increment": float(diff.mean() * 252), "ci95": np.quantile(draws, [.025, .975])})
    tail_summary = h.monitor(root, signals, labels)
    primary = next(m for m in metrics if m["policy"] == "TREE_MACRO" and m["scenario"] == "STRESS")
    result = {"study_id": STUDY, "at": h.now(), "primary": primary, "goal_achieved": False,
              "status": "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["eligible_historical_candidate"] else "FROZEN_NO_QUALIFIED_IRON_CONDOR",
              "new_value_estimates": len(models), "new_tree_fits": sum(m["model"].startswith("TREE") for m in models),
              "new_ridge_fits": sum(m["model"] == "LINEAR_PRICE" for m in models), "new_tail_updates": len(tails), "accounts": 10,
              "candidate_status": candidates.selection_status.value_counts().to_dict(), "label_status5": labels.status5.value_counts().to_dict(),
              "label_status10": labels.status10.value_counts().to_dict(), "common_decision_days": len(signals),
              "prediction_metrics": scored.groupby("model").agg(n=("idx", "size"), MSE=("MSE", "mean")).reset_index().to_dict("records"),
              "account_increments": comparisons, "tail_monitor": tail_summary,
              "entry_delta_median_absolute": float(signals.entry_net_delta.abs().median()),
              "historical_candidates": [m for m in metrics if m["eligible_historical_candidate"]],
              "independent_validation": False, "orders_authorized": False}
    h.save(root / "result.json", result)
    verification = verify(root)
    print(json.dumps(h.clean({"status": result["status"], "primary": primary, "verification": verification}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="同到期四腿有保护保险完整账户研究")
    parser.add_argument("command", choices=["freeze", "run", "verify", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "run":
        run(args.out)
    elif args.command == "verify":
        _, h = modules(args.out)
        print(json.dumps(h.clean(verify(args.out)), ensure_ascii=False, indent=2))
    else:
        _, h = modules(args.out)
        four_leg_source()
        print(json.dumps(mechanism_checks(h), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
