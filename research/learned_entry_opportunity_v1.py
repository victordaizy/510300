"""只用已完成交易训练一个三因子入场模型，复用原退出模型。"""
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController, predict
from research.learned_entry_opportunity_account_v1 import simulate_entry_opportunity
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_learned_entry_opportunity_v1"
CONFIG = ROOT / "config/510300_learned_entry_opportunity_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "CYCLE_ENTRY_RIDGE"
FEATURES = ["mom20", "sma120", "vol20"]
CHINESE = ["二十日涨跌", "一百二十日均线偏离", "二十日波动"]


def completed_cycle_target(cycle, dividends, exit_net_proceeds):
    eligible = dividends[(dividends.record_date >= pd.Timestamp(cycle["entry_date"])) &
                         (dividends.record_date < pd.Timestamp(cycle["exit_date"]))]
    rights = int(cycle["entry_quantity"]) * float(eligible.cash_dividend_per_share.sum())
    profit = exit_net_proceeds + rights - cycle["entry_cost_cny"]
    return profit / cycle["entry_cost_cny"], rights, profit


def training_rows(samples, fit_index, config):
    mature = samples[samples.exit_index <= fit_index].sort_values(["exit_index", "cycle_id"])
    require(not mature.cycle_id.duplicated().any(), "入场训练的一笔完整交易被重复计数")
    return mature.tail(config["recent_cycles"]).copy()


def fit_entry(rows, config):
    x, y = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "入场模型训练数据不完整")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    model = Ridge(alpha=config["ridge_alpha"], solver="svd", fit_intercept=True).fit(design, y)
    return {"kind": "RIDGE", "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_), "feature_clip": config["feature_clip"]}


class EntryOpportunityGate:
    def __init__(self, data, models):
        self.data, self.models = data, models
        self.fit_indices = [int(m["fit_index"]) for m in models]
        require(self.fit_indices == sorted(set(self.fit_indices)), "入场模型训练时间未严格递增")

    def __call__(self, t, account, mode, quantity):
        values = self.data.loc[self.data.index[t], FEATURES].to_numpy(float)
        j = bisect_right(self.fit_indices, t) - 1
        record = self.models[j] if j >= 0 else None
        ready = record is not None and record["status"] == "FIT_COMPLETE" and np.isfinite(values).all()
        result = {"entry_allowed": True, "entry_prediction": None,
                  "entry_model_status": "NO_VIEW_NO_MATURE_MODEL" if record is None or record["status"] != "FIT_COMPLETE" else "NO_VIEW_ENTRY_FEATURE_MISSING",
                  "entry_fit_origin": None if record is None else record.get("fit_origin"),
                  **{"entry_opportunity_" + key: value for key, value in zip(FEATURES, values)}}
        if ready:
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "入场模型含有未成熟交易或未来拟合")
            value = predict(record["model"], values)
            require(np.isfinite(value), "入场收益预测不可计算")
            result.update(entry_allowed=value > 0., entry_prediction=value, entry_model_status="PREDICTION_AVAILABLE")
        return result


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_LEARNED_ENTRY_OPPORTUNITY_V1", round=36, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, feature_columns=FEATURES, feature_names=CHINESE,
               reference_start="2013-06-03", recent_cycles=20, minimum_cycles=10, ridge_alpha=1., feature_clip=5.,
               threshold=0., model_refit="MONTH_FIRST_CLOSE", reject_consumes_opportunity=True,
               rules="docs/510300_LEARNED_ENTRY_OPPORTUNITY_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/learned_entry_opportunity_account_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
             ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"],
             ROOT / "tests/test_learned_entry_opportunity_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第36轮单一完整交易入场模型已登记，三因子、训练和接受规则固定。", flush=True)


def build_reference(data, dividends, cfg, exits):
    ledger, decisions, cycles = simulate_rearmed_exit(data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"],
        make_rules(data)["D60_INTRA"], cfg["specification"], ExitController(data, exits, cfg["confirmation_days"]))
    save_account(OUT / "reference", "REARM_RIDGE", ledger, decisions)
    cycles.to_csv(OUT / "reference/REARM_RIDGE_cycles.csv", index=False, encoding="utf-8-sig")
    sample_rows = []
    for cycle in cycles.to_dict("records"):
        if pd.isna(cycle.get("exit_date")) or "研究终点" in cycle["exit_reasons"]:
            continue
        t = int(np.flatnonzero(data.date == pd.Timestamp(cycle["entry_origin"]))[0])
        end = int(np.flatnonzero(data.date == pd.Timestamp(cycle["exit_date"]))[0])
        values = data.iloc[t][FEATURES].to_numpy(float)
        if not np.isfinite(values).all():
            continue
        sale = ledger[(ledger.date == pd.Timestamp(cycle["exit_date"])) & (ledger.filled_quantity < 0)]
        require(len(sale) == 1, "参考周期实际卖出不可唯一核对")
        target, rights, profit = completed_cycle_target(cycle, dividends, float(sale.notional.iloc[0] - sale.commission.iloc[0]))
        sample_rows.append({"cycle_id": int(cycle["cycle_id"]), "origin_index": t, "origin": data.date.iloc[t],
                            "entry_date": cycle["entry_date"], "exit_index": end, "exit_date": cycle["exit_date"],
                            "entry_cost_cny": cycle["entry_cost_cny"], "dividend_rights_cny": rights,
                            "net_profit_cny": profit, "target": target, **dict(zip(FEATURES, values))})
    samples = pd.DataFrame(sample_rows)
    samples.to_parquet(OUT / "complete_trade_samples.parquet", index=False)
    write_json(OUT / "reference_summary.json", {"reference_accounts": 1, "completed_sample_cycles": len(samples),
               "reference_metrics": summarize(ledger, cfg), "not_primary_evaluation": True})
    print(f"连续历史参考账户完成，取得{len(samples)}笔自然结束的有效交易。", flush=True)
    return samples


def train_models(data, samples, cfg):
    first = int(np.flatnonzero(data.date >= pd.Timestamp(cfg["earlier_start"]))[0]) - 1
    changed = data.date.dt.to_period("M") != data.date.shift(1).dt.to_period("M")
    schedule = sorted(set([first] + [int(t) for t in np.flatnonzero(changed) if first <= t < len(data) - 1]))
    models, receipts, membership = [], [], []
    chinese = ["# 完整交易入场模型：每次实际训练的中文规则", "", "预测值为完整交易净收益率，严格大于零才接受一次机会。以下均值、尺度、系数和截距来自当时已结束交易。", ""]
    for t in schedule:
        rows = training_rows(samples, t, cfg)
        usable = len(rows) >= cfg["minimum_cycles"]
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                  "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES", "training_cycles": len(rows),
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                  "latest_exit_date": rows.exit_date.max() if len(rows) else None, "model": fit_entry(rows, cfg) if usable else None}
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k != "model"})
        membership += [{"fit_index": t, "cycle_id": int(r.cycle_id), "exit_index": int(r.exit_index), "used_in_fit": usable} for r in rows.itertuples()]
        chinese += [f"## {record['fit_origin']}", "", f"当时最近完整交易{len(rows)}笔，最后退出日{record['latest_exit_date']}。", ""]
        if not usable:
            chinese += ["不足十笔，未训练；预测保持缺失，沿用原进入条件。", ""]
            continue
        model = record["model"]
        chinese += [f"截距为{model['intercept']:.10f}。各因子先减下列均值、除以下列尺度，再限制在负五至正五，乘对应系数后与截距相加。", ""]
        chinese += [f"- {name}：均值{mean:.10f}，尺度{scale:.10f}，系数{coefficient:.10f}。"
                    for name, mean, scale, coefficient in zip(CHINESE, model["mean"], model["scale"], model["coefficients"])]
        chinese += [""]
    write_json(OUT / "saved_entry_models.json", {"models": models, "feature_names": dict(zip(FEATURES, CHINESE))})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(membership).to_csv(OUT / "training_memberships.csv", index=False, encoding="utf-8-sig")
    (OUT / "每月入场模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    print(f"入场模型完成{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次拟合，{sum(r['status']!='FIT_COMPLETE' for r in receipts)}个时点样本不足。", flush=True)
    return models, receipts


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "完整交易入场登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    exits = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    samples = build_reference(data, dividends, cfg, exits)
    models, receipts = train_models(data, samples, cfg)
    main, early, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_entry_opportunity(frame, dividends, cfg, cost, start,
                make_rules(frame)["D60_INTRA"], cfg["specification"], ExitController(frame, exits, cfg["confirmation_days"]), EntryOpportunityGate(frame, models))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "完整交易入场账户结算失败")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "完整交易入场违反买入次日可卖")
            checks = decisions[decisions.entry_model_status.notna()]
            coverage.append({"period": period, "cost": cost_id, "entry_checks": len(checks),
                             "available_predictions": int((checks.entry_model_status == "PREDICTION_AVAILABLE").sum()),
                             "no_view_checks": int((checks.entry_model_status != "PREDICTION_AVAILABLE").sum()),
                             "nonpositive_rejections": int((checks.entry_allowed == False).sum()),
                             "actual_buys": int((ledger.filled_quantity > 0).sum())})
            accounts = {PRIMARY: ledger}
            names = {PRIMARY: "完整交易学习入场"}
            for key, name in [("REARM_RIDGE", "原线性退出＋等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(PARENT / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "完整交易入场评价日期不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个完整交易入场账户和两个对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("entry_model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "LEARNED_ENTRY_OPPORTUNITY_COMPLETE",
              "candidate_configurations": 1, "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4,
              "earlier_diagnostic_accounts": 6, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 4,
              "reference_accounts": 1, "completed_sample_cycles": len(samples), "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts),
              "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts), "all_metrics": main, "earlier_diagnostics": early,
              "primary": [m for m in main if m["model"] == PRIMARY], "entry_model_coverage": coverage,
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新设置": result["primary"], "较早": [m for m in early if m["model"] == PRIMARY],
                      "入场检查": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
