"""有保护认沽/认购价差的事前状态选择；复用既有认沽模型，新增认购证据。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
PUT = ROOT / "reports/research/510300_protected_put_spread_daily_v1"
BUYBACK = ROOT / "reports/research/510300_put_spread_buyback_cost_daily_v1"
OUT = ROOT / "reports/research/510300_option_credit_spread_router_daily_v1"
STUDY = "510300_OPTION_CREDIT_SPREAD_ROUTER_DAILY_V1"
POLICIES = ["PUT_ONLY", "CALL_ONLY", "ROUTER_EMPIRICAL", "ROUTER_PRICE", "ROUTER_MACRO"]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def general_terms(candidate, day, quotes):
    result = []
    for code, side in [(candidate["long_code"], 1), (candidate["short_code"], -1)]:
        q = quotes.get((day, code))
        if q is None or q["unit"] is None or not np.isfinite(q["unit"]):
            return None
        result.append({"code": code, "side": side, "unit": int(q["unit"]), "strike": float(q["effective_strike"]),
                       "option_type": q["option_type"], "quote": q})
    if result[0]["unit"] != result[1]["unit"] or result[0]["option_type"] != result[1]["option_type"]:
        return None
    if result[0]["option_type"] == "P" and result[0]["strike"] >= result[1]["strike"]:
        return None
    if result[0]["option_type"] == "C" and result[0]["strike"] <= result[1]["strike"]:
        return None
    return result


def general_width(terms):
    return abs(terms[1]["strike"] - terms[0]["strike"]) * terms[0]["unit"]


def general_margin(terms, spot, field="settlement"):
    short = terms[1]
    premium = float(short["quote"][field])
    if not np.isfinite(premium) or premium < 0:
        return math.inf
    strike = short["strike"]
    if short["option_type"] == "C":
        naked = (premium + max(.12 * spot - max(strike - spot, 0.), .07 * spot)) * short["unit"]
    else:
        naked = min(premium + max(.12 * spot - max(spot - strike, 0.), .07 * strike), strike) * short["unit"]
    return 1.2 * max(general_width(terms), naked)


def modules(root):
    hp = root / "code/spread_engine.py"
    bp = root / "code/buyback_model.py"
    h = import_file("router_frozen_cash_engine", hp if hp.exists() else PUT / "code/protected_put_spread_daily_v1.py")
    b = import_file("router_frozen_buyback_model", bp if bp.exists() else BUYBACK / "code/put_spread_buyback_cost_daily_v1.py")
    # 只扩展保护翼方向和法定单腿保证金公式，不改父现金、成本、退出和风控函数。
    h.terms_at, h.width_cash, h.margin = general_terms, general_width, general_margin
    return h, b


def freeze(root):
    h, b = modules(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("本版已冻结。")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "option_eod.parquet": PUT / "inputs/option_eod.parquet",
        "option_risk.parquet": PUT / "inputs/option_risk.parquet",
        "daily_terms.parquet": PUT / "results/daily_terms.parquet",
        "market.parquet": PUT / "inputs/market.parquet",
        "calendar_prices.parquet": PUT / "inputs/calendar_prices.parquet",
        "dividends.csv": PUT / "inputs/dividends.csv",
        "macro_information.parquet": BUYBACK / "inputs/macro_information.parquet",
        "put_signals.parquet": BUYBACK / "results/signals.parquet",
        "put_result.json": BUYBACK / "result.json",
        "put_protocol.json": BUYBACK / "protocol.json",
        "parent_account_protocol.json": PUT / "protocol.json",
        "authority_update.json": PUT / "authority_update.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, source in paths.items():
        (root / "inputs").mkdir(exist_ok=True)
        shutil.copy2(source, root / "inputs" / name)
    (root / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    shutil.copy2(PUT / "code/protected_put_spread_daily_v1.py", root / "code/spread_engine.py")
    shutil.copy2(BUYBACK / "code/put_spread_buyback_cost_daily_v1.py", root / "code/buyback_model.py")
    shutil.copy2(PUT / "code/account_primitives_source.py", root / "code/account_primitives_source.py")
    protocol = {
        "study_id": STUDY, "at": h.now(), "primary": "ROUTER_MACRO", "policies": POLICIES,
        "economic_question": "在同一标的上，按事前净收益补偿选择承担上行或下行的有限保险风险，能否优于固定方向。",
        "research_iteration_after_parent_failure": True,
        "unchanged_put": "复用上一冻结研究的认沽定价和五日尾部预测，不重训或改动认沽预测",
        "new_call_structure": "卖出标准认购Delta最接近+0.30；买入同到期更高行权价、Delta最接近+0.10的认购",
        "selection": "与认沽相同：观察日前两个交易日，30—75日到期优先45日，各腿成交>=100、持仓>=500，按Delta距离/行权价/代码稳定排序，持有10交易日且留5日到期缓冲",
        "call_models": "逐日、过去两个日历年、最少252共同成熟十日买回成本标签；直接复用上一版历史均值/价格岭回归/宏观岭回归定义和lambda10",
        "price_features": b.PRICE_COLUMNS, "macro_features": b.MACRO_COLUMNS,
        "call_tail": "五日实际压力收益/事前最大损失，过去两年252成熟样本，三个原特征下最近邻126，和认沽使用同一规则",
        "common_coverage": "两方向的三种定价与五日尾部均可用的日期；固定方向和路由共享该覆盖，未来标签不做入场过滤",
        "route_score": "(观察日压力信用额-预测买回成本及全部费用下限)/事前最大损失金额；仅正值可选",
        "route": "每个模型各自比较认沽与认购，正值最高者入选，同分认沽优先；两者均不正则空仓。只用09:00已知值，禁止按随后赢家选择方向",
        "policies_detail": {"PUT_ONLY": "固定认沽，用价格买回成本模型", "CALL_ONLY": "固定认购，用价格买回成本模型",
                            "ROUTER_EMPIRICAL": "两侧各自历史平均成本定价后选择", "ROUTER_PRICE": "两侧价格条件成本后选择", "ROUTER_MACRO": "两侧加入宏观条件后选择"},
        "while_held": "一组在仓不叠加第二组；每天更新记录选择，但持仓遵循原十日退出和逐日风险减仓，不为了当日另一侧更好就隐含换仓",
        "risk": "20万元；初始理论最大损失含费用<=权益5%，入场ES95<=2.5%，回撤剩余空间减半预算；逐日余下最大损失检查，DD10后停止新增；实际不保证回撤上限",
        "margin": "保护性认购改用认购义务仓公式；全程保留1.2倍max(价差宽度,单腿保证金)，并检查逐腿过渡现金",
        "costs_and_fills": "父BASE/STRESS、每腿每侧5元与不利tick取整不变；日线多腿开盘价格代理，缺同步可成交报价",
        "primary_increment": "ROUTER_MACRO相对ROUTER_PRICE；同时保留两方向独立PRICE和经验路由",
        "bootstrap": {"block_sessions": 20, "draws": 2000, "seed": 2026092504},
        "evaluation": "完整账户、固定2021—2023/2024—末端、全部滚动两年；不拼接最佳方向与时期",
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "independent_validation": False, "new_market_downloads": 0, "orders_authorized": False,
    }
    h.save(root / "protocol.json", protocol, exclusive=True)
    files = [root / "protocol.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*"))]
    h.save(root / "freeze.json", {"at": h.now(), "before_new_call_payoff_labels": True,
                                 "files": {p.relative_to(root).as_posix(): h.digest(p) for p in files}}, exclusive=True)
    print("两侧保险的事前路由已冻结；认沽预测原样复用，尚未读新增认购收益。", flush=True)


def sources(root, h):
    market, calendar = h.load_market(root)
    eod = pd.read_parquet(root / "inputs/option_eod.parquet")
    risk = pd.read_parquet(root / "inputs/option_risk.parquet")
    for frame in [eod, risk]:
        frame["trade_date"] = pd.to_datetime(frame.trade_date).astype("datetime64[ns]")
    term = pd.read_parquet(root / "inputs/daily_terms.parquet").rename(columns={"date": "trade_date", "strike": "effective_strike"})
    panel = eod.merge(term, on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    panel = panel.merge(risk[["trade_date", "contract_code", "delta", "implied_volatility"]], on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    panel["expiry_date"] = pd.to_datetime(panel.expiry_date).astype("datetime64[ns]")
    quotes = {(pd.Timestamp(r.trade_date), str(r.contract_code)): r._asdict() for r in panel.itertuples(index=False)}
    divs = pd.read_csv(root / "inputs/dividends.csv")
    events = {pd.Timestamp(r.ex_date): float(r.cash_dividend_per_share) for r in divs.itertuples()}
    return panel, quotes, market, calendar, events


def call_candidates(root, h, panel, quotes, market, calendar):
    daily = {day: group for day, group in panel.groupby("trade_date", sort=True)}
    first = int(calendar.searchsorted(panel.trade_date.min())) + 2
    rows = []
    for i in range(first, int(calendar.searchsorted(h.END, side="right"))):
        day, observed = calendar[i], calendar[i - 2]
        row = {"idx": i, "date": day, "observed_date": observed, "exit5_date": calendar[i + 5], "exit10_date": calendar[i + 10],
               "selection_status": "NO_ELIGIBLE_CALL_PAIR", "structure": "CALL"}
        source = daily.get(observed)
        if source is None:
            rows.append(row)
            continue
        source = source.copy()
        source["dte"] = (source.expiry_date - observed).dt.days
        ok = source.standard.fillna(False) & source.option_type.eq("C") & source.dte.between(30, 75)
        ok &= source.volume.ge(100) & source.open_interest.ge(500) & source.close.gt(0)
        ok &= source.delta.between(0, 1) & source.implied_volatility.between(.01, 3)
        possible = []
        for expiry, group in source[ok].groupby("expiry_date", sort=True):
            if group.effective_strike.nunique() < 2:
                continue
            group = group.assign(distance=(group.delta - .30).abs())
            short = group.sort_values(["distance", "effective_strike", "contract_code"], kind="stable").iloc[0]
            upper = group[group.effective_strike.gt(short.effective_strike)].copy()
            if upper.empty:
                continue
            upper["distance"] = (upper.delta - .10).abs()
            long = upper.sort_values(["distance", "effective_strike", "contract_code"], kind="stable").iloc[0]
            possible.append((abs(int((expiry - observed).days) - 45), expiry, short, long))
        if not possible:
            rows.append(row)
            continue
        _, expiry, short, long = sorted(possible, key=lambda x: (x[0], x[1]))[0]
        row.update(short_code=str(short.contract_code), long_code=str(long.contract_code), expiry=expiry,
                   short_strike=float(short.effective_strike), long_strike=float(long.effective_strike),
                   short_delta=float(short.delta), long_delta=float(long.delta),
                   capacity=int(math.floor(.01 * min(short.volume, short.open_interest, long.volume, long.open_interest))))
        if row["exit10_date"] >= expiry - pd.Timedelta(days=5):
            rows.append({**row, "selection_status": "KNOWN_EXPIRY_TOO_CLOSE"})
            continue
        iv = float((short.implied_volatility + long.implied_volatility) / 2)
        rv = float(market.loc[observed, "vol20"])
        row.update(log_iv=math.log(iv), log_iv_rv20=math.log(iv / rv), trend20=float(market.loc[observed, "trend20_feature"]),
                   observed_spot=float(market.loc[observed, "close"]))
        legs = h.terms_at(row, observed, quotes)
        raw_credit = -sum(t["side"] * t["unit"] * t["quote"]["close"] for t in legs)
        stress_credit = h.credit(legs, "close", "STRESS")
        exit_slip = h.close_cost(legs, "close", "STRESS") - raw_credit
        known_loss = h.width_cash(legs) - stress_credit + 4 * h.FEE + exit_slip
        row.update(raw_credit=raw_credit, prior_stress_credit=stress_credit, prior_exit_slip=exit_slip,
                   known_loss=known_loss, width_cash=h.width_cash(legs), unit=legs[0]["unit"], selection_status="READY")
        if known_loss <= 0 or not np.isfinite([row["trend20"], known_loss, iv, rv]).all():
            row["selection_status"] = "INVALID_KNOWN_INPUT"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / "results/call_candidates.parquet", index=False)
    return frame


def call_information(root, h, b, c, labels, quotes, market):
    macro = pd.read_parquet(root / "inputs/macro_information.parquet")
    times = ["dr_available_at", "policy_available_at", "credit_available_at", "GSPC_available_at", "VIX_available_at"]
    macro = macro[["date", "decision_time", "trend20", *b.MACRO_COLUMNS, *times]].rename(columns={"trend20": "trend20_current"})
    frame = c.merge(macro, on="date", how="left", validate="one_to_one")
    frame["credit_fraction"] = frame.prior_stress_credit / frame.width_cash
    previous_return = np.log1p(market.total_simple).shift(1).reindex(pd.DatetimeIndex(frame.date)).to_numpy(float)
    frame["spot_change_since_quote"] = previous_return / (np.exp(frame.log_iv) / np.sqrt(252))
    frame["features_known"] = np.isfinite(frame[b.PRICE_COLUMNS + b.MACRO_COLUMNS]).all(axis=1)
    for column in times:
        frame["features_known"] &= frame[column].notna() & frame[column].le(frame.decision_time)
    frame = frame.merge(labels.drop(columns=["date", "exit5_date", "exit10_date"]), on="idx", how="left", validate="one_to_one")
    frame["future_buyback_cost"] = np.nan
    frame["target_cost_fraction"] = np.nan
    for row in frame[frame.status10.eq("READY")].to_dict("records"):
        legs = h.terms_at(row, row["exit10_date"], quotes)
        value = h.close_cost(legs, "open", "STRESS")
        loc = frame.index[frame.idx.eq(row["idx"])][0]
        frame.loc[loc, "future_buyback_cost"] = value
        frame.loc[loc, "target_cost_fraction"] = value / row["width_cash"]
    frame.to_parquet(root / "results/call_information_and_labels.parquet", index=False)
    return frame


def learn_calls(root, h, b, frame):
    predictions, models, tail_models, signals, receipts = [], [], [], [], []
    for current in frame[frame.date.ge(h.START) & frame.selection_status.eq("READY")].to_dict("records"):
        p, m, r = b.fit_at(current, frame)
        predictions.extend(p)
        models.extend(m)
        tail, tm = h.tail_at(current["date"], current, frame)
        receipts.append({**r, "tail_status": tail["tail_status"]})
        if tm is not None:
            tail_models.append(tm)
        if p and tm is not None:
            # 明确去掉未来标签与买回成本，只把事前字段交给账户。
            row = {k: v for k, v in current.items() if k not in {"future_buyback_cost", "target_cost_fraction", "status5", "status10", "pnl5", "pnl10", "risk_return5", "risk_return10"}}
            row.update(tail)
            for prediction in p:
                model = prediction["model"]
                minimum = max(prediction["predicted_cost_cny"] + 4 * h.FEE, 4 * h.FEE + row["prior_exit_slip"])
                row[f"minimum_credit_{model}"] = minimum
                row[f"gate_{model}"] = row["prior_stress_credit"] > minimum
                row[f"forecast_buyback_cost_{model}"] = prediction["predicted_cost_cny"]
            signals.append(row)
        if len(receipts) % 300 == 0:
            print(f"认购成本与尾部分布已更新至{current['date'].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(root / "results/call_predictions.parquet", index=False)
    pd.DataFrame(signals).to_parquet(root / "results/call_signals.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/call_update_receipts.parquet", index=False)
    h.save(root / "results/call_models.json", models)
    h.save(root / "results/call_tail_models.json", tail_models)
    return pred, models, tail_models, pd.DataFrame(signals)


def choose_side(put, call, policy):
    model = "EMPIRICAL" if policy == "ROUTER_EMPIRICAL" else "MACRO" if policy == "ROUTER_MACRO" else "PRICE"
    possibilities = [put] if policy == "PUT_ONLY" else [call] if policy == "CALL_ONLY" else [put, call]
    ranked = [(float((r["prior_stress_credit"] - r[f"minimum_credit_{model}"]) / r["known_loss"]), index, r)
              for index, r in enumerate(possibilities)]
    score, _, selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    selected = selected.copy()
    selected[f"minimum_credit_{policy}"] = selected[f"minimum_credit_{model}"]
    selected[f"gate_{policy}"] = score > 0
    selected["route_score"] = score
    selected["route_model"] = model
    return selected


def route(root, h, calls):
    puts = pd.read_parquet(root / "inputs/put_signals.parquet")
    puts["structure"] = "PUT"
    pm = {r["date"]: r for r in puts.to_dict("records")}
    cm = {r["date"]: r for r in calls.to_dict("records")}
    common = sorted(set(pm) & set(cm))
    by_policy, records = {}, []
    for policy in POLICIES:
        rows = []
        for day in common:
            choice = choose_side(pm[day], cm[day], policy)
            rows.append(choice)
            records.append({"date": day, "policy": policy, "selected_structure": choice["structure"],
                            "positive": choice[f"gate_{policy}"], "score": choice["route_score"],
                            "short_code": choice["short_code"], "long_code": choice["long_code"],
                            "minimum_credit": choice[f"minimum_credit_{policy}"]})
        frame = pd.DataFrame(rows)
        frame.to_parquet(root / "results" / f"signals_{policy}.parquet", index=False)
        by_policy[policy] = frame
    records = pd.DataFrame(records)
    records.to_parquet(root / "results/routing_choices.parquet", index=False)
    return by_policy, records


def checks(h, b):
    for kind, long_k, short_k in [("P", 3.8, 4.), ("C", 4.2, 4.)]:
        terms = [{"side": 1, "unit": 10000, "strike": long_k, "option_type": kind, "quote": {"settlement": .01}},
                 {"side": -1, "unit": 10000, "strike": short_k, "option_type": kind, "quote": {"settlement": .05}}]
        np.testing.assert_allclose(general_width(terms), 2000., atol=1e-8)
        for spot in [0., 2., 3.8, 4., 4.2, 8., 100.]:
            payoff = sum(t["side"] * t["unit"] * max(spot - t["strike"] if kind == "C" else t["strike"] - spot, 0.) for t in terms)
            assert -2000. - 1e-7 <= payoff <= 1e-7
        assert general_margin(terms, 4.) >= 2400.
    p = {"structure": "PUT", "prior_stress_credit": 400., "known_loss": 1600., "minimum_credit_PRICE": 300.}
    c = {"structure": "CALL", "prior_stress_credit": 300., "known_loss": 1700., "minimum_credit_PRICE": 400.}
    assert choose_side(p, c, "ROUTER_PRICE")["structure"] == "PUT"
    p["future_profit"] = -1e9
    c["future_profit"] = 1e9
    assert choose_side(p, c, "ROUTER_PRICE")["structure"] == "PUT"
    assert general_margin([{"side": 1, "unit": 10000, "strike": 4.2, "option_type": "C", "quote": {"settlement": .01}},
                           {"side": -1, "unit": 10000, "strike": 4., "option_type": "C", "quote": {"settlement": .05}}], 4.) == 6360.
    return {"both_directions_have_bounded_payoff": True, "call_margin_formula": True,
            "routing_ignores_future_winner": True, **b.check(h)}


def verify(root):
    h, b = modules(root)
    h.verify_freeze(root)
    frame = pd.read_parquet(root / "results/call_information_and_labels.parquet").set_index("idx", drop=False)
    pred = pd.read_parquet(root / "results/call_predictions.parquet").set_index(["idx", "model"])
    models = json.loads((root / "results/call_models.json").read_text(encoding="utf-8"))
    tails = json.loads((root / "results/call_tail_models.json").read_text(encoding="utf-8"))
    for model in models:
        train = frame.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all() and train.exit10_date.lt(pd.Timestamp(model["date"])).all()
        current = frame.loc[model["idx"]]
        value = model["intercept"]
        if model["features"]:
            z = np.clip((current[model["features"]].to_numpy(float) - np.array(model["mean"])) / np.array(model["scale"]), -5, 5)
            value += float(z @ np.array(model["beta"]))
        np.testing.assert_allclose(max(0., value), pred.loc[(model["idx"], model["model"]), "predicted_cost_fraction"], atol=1e-12)
    for model in tails:
        train = frame.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all() and train.exit5_date.lt(pd.Timestamp(model["date"])).all()
        vals = frame.loc[model["neighbor_indices"], "risk_return5"].to_numpy(float)
        expected = max(0., -float(np.sort(vals)[:int(math.ceil(.05 * len(vals)))].mean()))
        np.testing.assert_allclose(expected, model["ES95_risk_unit"], atol=1e-12)
    prefixes = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        subset = [m for m in models if m["model"] == "MACRO" and pd.Timestamp(m["date"]) <= cutoff]
        if not subset:
            continue
        model = subset[-1]
        current = frame.loc[model["idx"]].to_dict()
        before = frame[frame.date.le(current["date"])].copy()
        before.loc[before.exit10_date.ge(current["date"]), "target_cost_fraction"] = 1e9
        forecasts, _, _ = b.fit_at(current, before)
        for item in forecasts:
            np.testing.assert_allclose(item["predicted_cost_fraction"], pred.loc[(model["idx"], item["model"]), "predicted_cost_fraction"], atol=1e-12)
        prefixes.append(str(pd.Timestamp(model["date"]).date()))
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            folder = root / "accounts" / scenario / policy
            l = pd.read_parquet(folder / "ledger.parquet")
            f = pd.read_parquet(folder / "fills.parquet")
            d = pd.read_parquet(folder / "decisions.parquet")
            np.testing.assert_allclose(l.equity, l.cash + l.option_value - l.terminal_exit_reserve, atol=1e-6)
            if len(f):
                flow = f.groupby("date").cash_change.sum().reindex(l.date, fill_value=0.).to_numpy()
                np.testing.assert_allclose(h.INITIAL + np.cumsum(flow), l.cash, atol=1e-6)
                assert f.cash_after_leg.ge(-1e-7).all()
            entered = d[d.action.eq("ENTRY_FILLED")]
            if len(entered):
                assert entered.actual_maximum_loss.le(entered.budget + 1e-7).all()
                assert entered.actual_ES95.le(.025 * entered.known_equity + 1e-7).all()
    result = {"at": h.now(), "new_call_predictions_recomputed": len(models), "call_tail_updates_recomputed": len(tails),
              "two_year_and_maturity_checks": True, "prefix_and_future_poisoning": prefixes,
              "accounts_reconciled": 10, "entry_risk_budgets_verified": True, "mechanism_checks": checks(h, b)}
    h.save(root / "verification.json", result)
    return result


def run(root):
    h, b = modules(root)
    h.verify_freeze(root)
    assert h.digest(Path(__file__)) == h.digest(root / "code/option_credit_spread_router_daily_v1.py")
    h.save(root / "RUN_STARTED.json", {"at": h.now()}, exclusive=True)
    (root / "results").mkdir(exist_ok=True)
    h.save(root / "mechanism_checks.json", checks(h, b))
    panel, quotes, market, calendar, events = sources(root, h)
    c = call_candidates(root, h, panel, quotes, market, calendar)
    labels = h.payoff_labels(root, c, quotes, calendar)
    frame = call_information(root, h, b, c, labels, quotes, market)
    pred, models, tails, calls = learn_calls(root, h, b, frame)
    signals, choices = route(root, h, calls)
    pred_metrics = b.prediction_evaluation(root, h, pred, frame)
    metrics, yearly, rolling, ledgers = [], [], [], {}
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            account = h.simulate(policy, scenario, signals[policy], quotes, market, calendar, events)
            if len(account["cycles"]):
                mapping = choices[choices.policy.eq(policy)][["date", "selected_structure"]].rename(columns={"date": "entry_date"})
                account["cycles"] = account["cycles"].merge(mapping, on="entry_date", validate="many_to_one")
            folder = root / "accounts" / scenario / policy
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in account.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(folder / f"{name}.parquet", index=False)
                else:
                    h.save(folder / f"{name}.json", value)
            measure = h.account_metrics(policy, scenario, account)
            metrics.append(measure)
            y, r = h.yearly_and_rolling(policy, scenario, account["ledger"])
            yearly.extend(y)
            rolling.extend(r)
            ledgers[(policy, scenario)] = account["ledger"]
            print(f"{scenario}/{policy}完成：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.3%}，回撤{measure['max_drawdown']:.3%}。", flush=True)
    h.save(root / "results/account_metrics.json", metrics)
    h.save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    paired = ledgers[("ROUTER_MACRO", "STRESS")]["return"].to_numpy() - ledgers[("ROUTER_PRICE", "STRESS")]["return"].to_numpy()
    rng = np.random.default_rng(2026092504)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, len(paired), size=math.ceil(len(paired) / 20))
        idx = ((starts[:, None] + np.arange(20)) % len(paired)).ravel()[:len(paired)]
        draws.append(float(paired[idx].mean() * 252))
    periods = []
    for start, end in [(h.START, pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), h.END)]:
        for policy in POLICIES:
            l = ledgers[(policy, "STRESS")]
            periods.append({"policy": policy, "start": start, "end": end, **h.return_metrics(l.loc[l.date.between(start, end), "return"])})
    h.save(root / "results/fixed_periods.json", periods)
    main = next(m for m in metrics if m["policy"] == "ROUTER_MACRO" and m["scenario"] == "STRESS")
    result = {"study_id": STUDY, "at": h.now(), "primary": main, "goal_achieved": False,
              "status": "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["eligible_historical_candidate"] else "FROZEN_NO_QUALIFIED_CREDIT_SPREAD_ROUTER",
              "new_call_cost_estimates": len(models), "new_call_ridge_fits": sum(m["model"] != "EMPIRICAL" for m in models),
              "new_call_tail_updates": len(tails), "accounts": len(metrics), "put_models_retrained": 0,
              "all_call_candidate_dates": len(c), "ready_call_candidate_dates": int(c.selection_status.eq("READY").sum()),
              "call_label_status5": labels.status5.value_counts().to_dict(), "call_label_status10": labels.status10.value_counts().to_dict(),
              "common_decision_days": int(choices.date.nunique()),
              "positive_choice_counts": choices[choices.positive].groupby(["policy", "selected_structure"]).size().reset_index(name="count").to_dict("records"),
              "call_prediction_metrics": pred_metrics, "paired_macro_minus_price_annual_mean": float(paired.mean() * 252),
              "paired_increment_ci95": np.quantile(draws, [.025, .975]),
              "historical_candidates": [m for m in metrics if m["eligible_historical_candidate"]],
              "independent_validation": False, "orders_authorized": False}
    h.save(root / "result.json", result)
    verification = verify(root)
    print(json.dumps(h.clean({"status": result["status"], "primary": main, "verification": verification}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="有保护的双方向期权保险日更研究")
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
        h, b = modules(args.out)
        print(json.dumps(checks(h, b), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
