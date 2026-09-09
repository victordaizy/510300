"""用当时已经结束的参考持仓周期学习提前退出，保持完整账户约束。"""
from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import commission, digest, fill_price, now, require, write_json
from research.learned_cycle_exit_account_v1 import simulate_learned_exit
from research.simple_intraday_protection_v1 import NAMES, make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_learned_cycle_exit_v1"
CONFIG = ROOT / "config/510300_learned_cycle_exit_v1.json"
FEATURES = ["log_holding_days", "cycle_return", "cycle_drawdown", "entry_mode", "mom5", "mom20", "sma120", "vol20"]
CN = ["持仓时间对数", "含分红浮盈浮亏", "持仓回撤", "入场模式", "五日涨跌", "二十日涨跌", "一百二十日均线偏离", "二十日波动"]
KINDS = {"RIDGE": "线性退出", "TREE": "两层树退出"}


def state_values(data, t, cycle, current_value, peak_value):
    return np.array([np.log1p(t - cycle["entry_index"] + 1), current_value / cycle["entry_cost_cny"] - 1,
                     current_value / peak_value - 1, cycle["mode"], data.mom5.iloc[t], data.mom20.iloc[t],
                     data.sma120.iloc[t], data.vol20.iloc[t]], dtype=float)


def continuation_label(data, dividends, quantity, first_exit, later_exit, cost, tick):
    """标签按股权登记日比较新增权益，早已取得而尚未到账的权益在两边抵消。"""
    early = fill_price(float(data.open.iloc[first_exit]), -1, cost, tick)
    late = fill_price(float(data.open.iloc[later_exit]), -1, cost, tick)
    eligible = dividends[(dividends.record_date >= data.date.iloc[first_exit]) & (dividends.record_date < data.date.iloc[later_exit])]
    extra_dividend = quantity * float(eligible.cash_dividend_per_share.sum())
    difference = quantity * (late - early) - commission(quantity, late, cost) + commission(quantity, early, cost) + extra_dividend
    return difference / (quantity * float(data.open.iloc[first_exit])), extra_dividend


def training_rows(samples, fit_index, config):
    """先剔除当时尚未结束的周期，再选择最近完整周期。"""
    mature = samples[samples.exit_index <= fit_index].copy()
    cycles = mature[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"])
    ids = cycles.tail(config["recent_cycles"]).cycle_id.to_list()
    chosen = mature[mature.cycle_id.isin(ids)].copy().sort_values(["cycle_id", "origin_index"])
    if len(chosen):
        chosen["sample_weight"] = 1 / chosen.groupby("cycle_id").origin_index.transform("count")
    else:
        chosen["sample_weight"] = pd.Series(dtype=float)
    return chosen, ids


def fit_one(rows, kind, config):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "退出训练输入或标签不完整")
    if kind == "RIDGE":
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.)
        design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
        model = Ridge(alpha=config["ridge_alpha"], solver="svd", fit_intercept=True)
        model.fit(design, y, sample_weight=weights)
        return {"kind": kind, "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_), "feature_clip": config["feature_clip"]}
    model = DecisionTreeRegressor(max_depth=config["tree_max_depth"], min_samples_leaf=config["tree_min_samples_leaf"],
                                  min_weight_fraction_leaf=config["tree_min_weight_fraction_leaf"], random_state=config["random_seed"])
    model.fit(x, y, sample_weight=weights)
    tree = model.tree_
    return {"kind": kind, "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(), "value": tree.value[:, 0, 0].tolist()}


def predict(model, values):
    if model["kind"] == "RIDGE":
        x = np.clip((values - np.array(model["mean"])) / np.array(model["scale"]), -model["feature_clip"], model["feature_clip"])
        return float(model["intercept"] + x @ np.array(model["coefficients"]))
    x = values.astype(np.float32)
    node = 0
    while model["left"][node] >= 0:
        node = model["left"][node] if x[model["feature"][node]] <= model["threshold"][node] else model["right"][node]
    return float(model["value"][node])


def chinese_formula(model):
    if model["kind"] == "RIDGE":
        lines = [f"预测继续持有收益的截距为{model['intercept']:.10f}；将下列各项相加。每个因子先减均值、除标准差，并限制在负5到5。"]
        lines += [f"- {name}：均值{mean:.10f}，标准差{scale:.10f}，标准化后的系数{coefficient:.10f}。"
                  for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"])]
        return lines
    lines = []
    def visit(node, conditions):
        if model["left"][node] < 0:
            lines.append("- " + ("，且".join(conditions) if conditions else "全部状态") + f"：预测继续持有收益为{model['value'][node]:.8%}。")
            return
        name, threshold = CN[model["feature"][node]], model["threshold"][node]
        visit(model["left"][node], conditions + [f"{name}不高于{threshold:.10f}"])
        visit(model["right"][node], conditions + [f"{name}高于{threshold:.10f}"])
    visit(0, [])
    return lines


class ExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [m["fit_index"] for m in models]
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        estimate = None
        status = "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["fit_index"] <= t and stored["latest_exit_index"] <= stored["fit_index"], "退出模型使用了未来周期")
            estimate = predict(stored["model"], values)
            status = "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
                "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
                "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, values))}


def freeze():
    old = json.loads((ROOT / "config/510300_simple_intraday_protection_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "candidate_specs", "earlier_start", "earlier_terminal"]}
    cfg.update({"study_id": "510300_LEARNED_CYCLE_EXIT_V1", "round": 31, "registered_at": now(),
                "primary": "S1_TREND_REBOUND__RIDGE", "candidate_configurations": 6, "signal_names": NAMES, "model_kinds": KINDS,
                "reference_start": "2013-06-03", "recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100,
                "feature_columns": FEATURES, "feature_names": CN, "feature_clip": 5., "ridge_alpha": 1.,
                "tree_max_depth": 2, "tree_min_samples_leaf": 20, "tree_min_weight_fraction_leaf": .1,
                "random_seed": 20260907, "confirmation_days": 2, "model_refit": "MONTH_FIRST_CLOSE",
                "rules": "docs/510300_LEARNED_CYCLE_EXIT_V1.md", "position_impact": 0,
                "python_library_version": sklearn.__version__, "independent_validation": "NOT_ESTABLISHED"})
    paths = [Path(__file__), ROOT / "research/learned_cycle_exit_account_v1.py", ROOT / "research/simple_intraday_protection_v1.py",
             ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "tests/test_learned_cycle_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第31轮已登记六个学习退出设置，尚未训练本轮模型或读取新账户收益。", flush=True)


def build_samples(data, dividends, cfg):
    rules = make_rules(data)
    samples, reference_info = [], []
    for key in NAMES:
        def recorder(t, cycle, current_value, peak_value):
            return {"learning_cycle_id": cycle["cycle_id"], "learned_exit_requested": False,
                    **dict(zip(FEATURES, state_values(data, t, cycle, current_value, peak_value)))}
        ledger, decisions, cycles = simulate_learned_exit(data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"], rules[key], cfg["candidate_specs"][key], recorder)
        save_account(OUT / "reference", key, ledger, decisions)
        cycles.to_csv(OUT / "reference" / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
        count = 0
        for cycle in cycles.to_dict("records"):
            if pd.isna(cycle.get("exit_date")) or "研究终点" in cycle["exit_reasons"]:
                continue
            end = int(np.flatnonzero(data.date == pd.Timestamp(cycle["exit_date"]))[0])
            states = decisions[(decisions.learning_cycle_id == cycle["cycle_id"]) & (decisions.requested_quantity == 0)]
            for row in states.to_dict("records"):
                t = int(row["origin_index"])
                if t + 1 >= end or not np.isfinite([row[k] for k in FEATURES]).all():
                    continue
                target, extra = continuation_label(data, dividends, cycle["entry_quantity"], t + 1, end, cfg["costs"]["BASE"], cfg["tick"])
                samples.append({"signal": key, "cycle_id": int(cycle["cycle_id"]), "origin_index": t, "origin": data.date.iloc[t],
                                "early_exit_index": t + 1, "early_exit_date": data.date.iloc[t + 1], "exit_index": end,
                                "mature_date": data.date.iloc[end], "target": target, "extra_dividend_cny": extra,
                                "reference_quantity": int(cycle["entry_quantity"]), **{k: row[k] for k in FEATURES}})
                count += 1
        reference_info.append({"signal": key, "natural_sample_cycles": len({r["cycle_id"] for r in samples if r["signal"] == key}),
                               "sample_rows": count, "reference_metrics": summarize(ledger, cfg)})
        print(f"{NAMES[key]}：原参考账户和{count}条持仓状态记录已保存。", flush=True)
    result = pd.DataFrame(samples)
    result.to_parquet(OUT / "all_reference_samples.parquet", index=False)
    write_json(OUT / "reference_summary.json", {"reference_accounts": reference_info, "count": 3, "not_primary_evaluation": True})
    return result


def train_models(data, samples, cfg):
    first = int(np.flatnonzero(data.date >= pd.Timestamp(cfg["earlier_start"]))[0]) - 1
    first_month = data.date.dt.to_period("M") != data.date.shift(1).dt.to_period("M")
    schedule = sorted(set([first] + [int(t) for t in np.flatnonzero(first_month) if first <= t < len(data) - 1]))
    all_models, receipts, memberships = {}, [], []
    chinese = ["# 每月训练完成的退出模型规则", "", "模型只使用当时已经结束的周期。以下输入、系数和分支均按各次训练保存；没有模型时沿用原退出。", ""]
    for key in NAMES:
        local = samples[samples.signal == key]
        for kind in KINDS:
            series = []
            for t in schedule:
                rows, cycles = training_rows(local, t, cfg)
                usable = len(cycles) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
                record = {"signal": key, "kind": kind, "fit_index": t, "fit_origin": str(data.date.iloc[t].date()),
                          "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                          "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS",
                          "training_cycles": cycles, "training_cycle_count": len(cycles), "training_rows": len(rows),
                          "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                          "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
                          "model": fit_one(rows, kind, cfg) if usable else None}
                require(not len(rows) or (rows.exit_index <= t).all(), "训练样本周期尚未结束")
                series.append(record)
                receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles"]})
                if usable:
                    memberships += [{"signal": key, "kind": kind, "fit_index": t, "cycle_id": int(r.cycle_id),
                                     "origin_index": int(r.origin_index), "exit_index": int(r.exit_index), "sample_weight": float(r.sample_weight)}
                                    for r in rows.itertuples()]
                    chinese += [f"## {NAMES[key]}／{KINDS[kind]}／{record['fit_origin']}", "",
                                f"使用{len(cycles)}个已结束周期、{len(rows)}条状态；最后周期结束于{record['latest_exit_date']}。", ""]
                    chinese += chinese_formula(record["model"]) + [""]
            all_models[key + "__" + kind] = series
            print(f"{NAMES[key]}／{KINDS[kind]}：{sum(x['status']=='FIT_COMPLETE' for x in series)}次月度训练完成。", flush=True)
    write_json(OUT / "saved_models.json", {"models": all_models, "feature_names": dict(zip(FEATURES, CN))})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    return all_models, receipts


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "学习退出登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = build_samples(data, dividends, cfg)
    models, receipts = train_models(data, samples, cfg)
    metrics, earlier, yearly, eras, coverage = [], [], [], [], []
    parent = ROOT / "reports/research/510300_simple_intraday_protection_v1"
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], metrics),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rules = make_rules(frame)
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, {}
            for key, name in NAMES.items():
                for kind, kind_name in KINDS.items():
                    model_id = key + "__" + kind
                    controller = ExitController(frame, models[model_id], cfg["confirmation_days"])
                    ledger, decisions, cycles = simulate_learned_exit(frame, dividends, cfg, cost, start, rules[key], cfg["candidate_specs"][key], controller)
                    save_account(folder, model_id, ledger, decisions)
                    cycles.to_csv(folder / f"{model_id}_cycles.csv", index=False, encoding="utf-8-sig")
                    finished = cycles.dropna(subset=["exit_date"])
                    require((finished.holding_intervals >= 1).all(), "学习退出违反次日可卖约束")
                    require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "学习退出账户结算不完整")
                    accounts[model_id], names[model_id] = ledger, name + "＋" + kind_name
                    holding = decisions[decisions.learning_cycle_id.notna()]
                    coverage.append({"period": period, "cost": cost_id, "model": model_id, "holding_decision_rows": len(holding),
                                     "model_available_rows": int((holding.learning_status == "PREDICTION_AVAILABLE").sum()),
                                     "no_model_rows": int((holding.learning_status != "PREDICTION_AVAILABLE").sum()),
                                     "learned_exit_cycles": int(finished.exit_reasons.str.contains("学习条件", regex=False).sum()),
                                     "completed_round_trips": len(finished)})
                control = key + "__CLOSE_NEXT_OPEN"
                source = parent / period / cost_id
                ledger = pd.read_parquet(source / f"{control}_ledger.parquet")
                decisions = pd.read_parquet(source / f"{control}_decisions.parquet")
                save_account(folder, control, ledger, decisions)
                accounts[control], names[control] = ledger, name + "＋原退出"
            bh = pd.read_parquet(parent / period / cost_id / "BUY_HOLD_ledger.parquet")
            bh.to_parquet(folder / "BUY_HOLD_ledger.parquet", index=False)
            accounts["BUY_HOLD"], names["BUY_HOLD"] = bh, "买入持有"
            base = summarize(bh, cfg)
            for model_id, ledger in accounts.items():
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(bh.date)), "学习退出完整评价日期不一致")
                m = {"cost": cost_id, "model": model_id, "name": names[model_id], **summarize(ledger, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                if period == "evaluation":
                    for year, group in ledger.groupby(ledger.date.dt.year):
                        yearly.append({"cost": cost_id, "model": model_id, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = ledger[(ledger.date >= left) & (ledger.date <= right)]
                        eras.append({"cost": cost_id, "model": model_id, "era": label, **summarize(group, cfg)})
            pd.DataFrame({"date": bh.date, **{k: v.net_return.to_numpy() for k, v in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：六种学习退出和四个对照账户已完成。", flush=True)
    for filename, rows in [("metrics.csv", metrics), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in metrics if m["cost"] == "BASE" and m["model"] in models), key=lambda x: x["net_sharpe"] if x["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "LEARNED_CYCLE_EXIT_ACCOUNTS_COMPLETE",
              "candidate_configurations": 6, "evaluation_accounts": 20, "new_accounts_generated": 12, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": 20, "new_earlier_diagnostic_accounts": 12, "reused_earlier_accounts": 8,
              "reference_accounts": 3, "training_sample_rows": len(samples),
              "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts),
              "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts),
              "all_metrics": metrics, "earlier_diagnostics": earlier, "model_coverage": coverage,
              "primary": [m for m in metrics if m["model"] == cfg["primary"]], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in metrics if m["model"] in models),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "完成拟合": result["completed_fits"], "基础费用结果":
        [{k: m[k] for k in ["model", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in metrics if m["cost"] == "BASE"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
