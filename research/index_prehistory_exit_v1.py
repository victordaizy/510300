"""仅补充上市前指数成熟周期，直接检验原ETF进出场完整账户。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.index_prehistory_exit_inputs_v1 import price_features, natural_price_episodes, pool_samples, choose_mature_samples, monthly_schedule
from research.learned_cycle_exit_v1 import FEATURES, CN, ExitController, fit_one, chinese_formula
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_index_prehistory_exit_v1"
CONFIG = ROOT / "config/510300_index_prehistory_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
SOURCE = ROOT / "reports/research/510300_index_preinception_source_probe_20260908"
PRIMARY = "INDEX_PREHISTORY_REARM_RIDGE"
NAME = "上市前指数周期补充的线性退出"
CONTROLS = {"REARM_RIDGE": (P32, "原第32轮线性退出"), "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def original_config():
    return json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))


def prepare():
    require(not (OUT / "input_receipt.json").exists(), "新增样本已保存，不重复读取生成")
    cfg = original_config()
    admission = json.loads((SOURCE / "source_admission.json").read_text(encoding="utf-8"))
    require(admission["status"] == "PASS_HISTORICAL_PRICE_EXTENSION_PUBLISHED_PERIOD_ONLY", "新增指数价格来源未准入")
    source_path = ROOT / admission["source_file"]
    require(digest(source_path) == admission["source_sha256"], "已准入价格文件身份不符")
    etf = pd.read_parquet(ROOT / cfg["features"])
    prices = pd.read_parquet(source_path)
    prices = prices[prices.date.lt(etf.date.iloc[0])].copy()
    require(len(prices) == 1733 and prices.date.min() == pd.Timestamp("2005-04-08") and prices.date.max() == pd.Timestamp("2012-05-25"), "上市前已发布价格范围不符")
    frame = price_features(prices)
    episodes, index_samples = natural_price_episodes(frame, cfg["candidate_specs"]["D60_INTRA"])
    old = pd.read_parquet(P31 / "all_reference_samples.parquet")
    old = old[old.signal.eq("D60_INTRA")].copy()
    combined = pool_samples(index_samples, old, etf.date)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("index_price_features", frame), ("index_price_episodes", episodes), ("index_price_samples", index_samples), ("pooled_reference_samples", combined)]:
        data.to_parquet(OUT / f"{name}.parquet", index=False)
    episodes.to_csv(OUT / "指数自然周期及未完成状态.csv", index=False, encoding="utf-8-sig")
    counts = []
    for t in monthly_schedule(etf, cfg["earlier_start"]):
        rows, ids = choose_mature_samples(combined, etf.date.iloc[t], cfg["recent_cycles"])
        counts.append({"fit_index": t, "fit_origin": etf.date.iloc[t], "cycles": len(ids), "rows": len(rows),
            "index_cycles": rows.loc[rows.source.eq("INDEX_PRICE"), "cycle_id"].nunique(),
            "etf_cycles": rows.loc[rows.source.eq("ETF_REFERENCE"), "cycle_id"].nunique(),
            "eligible": len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]})
    pd.DataFrame(counts).to_csv(OUT / "训练覆盖预检.csv", index=False, encoding="utf-8-sig")
    receipt = {"prepared_at": now(), "status": "PRICE_ANALOG_SAMPLES_READY_NO_MODEL_OR_NEW_ACCOUNT", "source_sha256": digest(source_path),
        "index_rows": len(frame), "first_feature_valid_date": str(frame.loc[frame.feature_valid, "date"].min().date()),
        "index_episodes": len(episodes), "naturally_closed_index_episodes": int(episodes.status.eq("NATURALLY_CLOSED").sum()),
        "censored_index_episodes": int(episodes.status.ne("NATURALLY_CLOSED").sum()), "index_sample_cycles": int(index_samples.cycle_id.nunique()),
        "index_sample_rows": len(index_samples), "original_etf_cycles": int(old.cycle_id.nunique()), "original_etf_rows": len(old),
        "monthly_origins": len(counts), "eligible_months": sum(row["eligible"] for row in counts),
        "first_eligible_month": next((row for row in counts if row["eligible"]), None), "new_models": 0, "new_accounts": 0,
        "input_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [OUT / "index_price_features.parquet", OUT / "index_price_episodes.parquet", OUT / "index_price_samples.parquet", OUT / "pooled_reference_samples.parquet", Path(__file__), ROOT / "research/index_prehistory_exit_inputs_v1.py", ROOT / "docs/510300_INDEX_PREHISTORY_EXIT_V1.md"]]}
    write_json(OUT / "input_receipt.json", receipt, exclusive=True)
    print(json.dumps({k: v for k, v in receipt.items() if k != "input_files"}, ensure_ascii=False, default=str), flush=True)


def freeze():
    require(not CONFIG.exists(), "第93轮已登记，不能重复冻结")
    old = original_config()
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "recent_cycles", "minimum_cycles", "minimum_rows", "feature_clip", "ridge_alpha", "confirmation_days"]}
    cfg.update(study_id="510300_INDEX_PREHISTORY_EXIT_V1", round=93, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        specification=old["candidate_specs"]["D60_INTRA"], source_scope="PUBLISHED_PRICE_INDEX_PRE_ETF_ONLY_20050408_20120525",
        new_training_warmup_returns=252, sample_clock="NATURAL_EXIT_DATE_NO_TERMINAL_LABEL", sample_weight="EQUAL_CYCLE_BOTH_SOURCES",
        training_clock="ORIGINAL_20141231_AND_MONTH_FIRST_COMPLETE_CLOSE", new_reference_accounts=0,
        rules="docs/510300_INDEX_PREHISTORY_EXIT_V1.md", independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "本轮必要测试未通过")
    inputs = json.loads((OUT / "input_receipt.json").read_text(encoding="utf-8"))
    paths = [CONFIG.parent / "510300_learned_cycle_exit_v1.json", ROOT / cfg["features"], ROOT / cfg["dividends"], P31 / "all_reference_samples.parquet", SOURCE / "source_admission.json",
        ROOT / json.loads((SOURCE / "source_admission.json").read_text(encoding="utf-8"))["source_file"], OUT / "input_receipt.json", OUT / "tests_receipt.json",
        ROOT / "tests/test_index_prehistory_exit_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py"]
    for item in inputs["input_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "准备阶段输入或方案发生变化")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第93轮单项指数早期经验补充已冻结，尚未训练或读取新策略账户收益。", flush=True)


def train(data, samples, cfg):
    models, receipts, memberships = [], [], []
    chinese = ["# 第93轮每月八项因子模型", "", "以下是逐月实际训练保存的系数。指数样本仅为价格类比；所有周期必须先成熟。", ""]
    for t in monthly_schedule(data, cfg["earlier_start"]):
        rows, ids = choose_mature_samples(samples, data.date.iloc[t], cfg["recent_cycles"])
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
            "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "training_cycles": ids,
            "training_cycle_count": len(ids), "training_rows": len(rows), "index_cycles": int(rows.loc[rows.source.eq("INDEX_PRICE"), "cycle_id"].nunique()),
            "etf_cycles": int(rows.loc[rows.source.eq("ETF_REFERENCE"), "cycle_id"].nunique()),
            "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
            "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
            "model": fit_one(rows, "RIDGE", cfg) if usable else None}
        require(not len(rows) or (rows.mature_date.le(data.date.iloc[t]).all() and rows.exit_index.le(t).all()), "合并训练出现未来周期")
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["training_cycles", "model"]})
        if usable:
            memberships.extend({"fit_index": t, "row_id": r.row_id, "cycle_id": r.cycle_id, "source": r.source, "origin": r.origin, "mature_date": r.mature_date, "sample_weight": r.sample_weight} for r in rows.itertuples())
            chinese += [f"## {record['fit_origin']}", "", f"共{len(ids)}个成熟周期、{len(rows)}条状态，其中指数周期{record['index_cycles']}个、ETF周期{record['etf_cycles']}个，最后成熟日期{record['latest_exit_date']}。", "", *chinese_formula(record["model"]), ""]
        else:
            chinese += [f"## {record['fit_origin']}", "", f"只有{len(ids)}个成熟周期、{len(rows)}条状态，未达到原训练门槛，沿用原价格退出。", ""]
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": dict(zip(FEATURES, CN))}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月八项模型中文规则.md").write_text("\n".join(chinese), encoding="utf-8")
    print(f"本轮{sum(r['status']=='FIT_COMPLETE' for r in models)}次月度训练完成，随后直接计算四个完整账户。", flush=True)
    return models


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "指数早期经验冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(OUT / "pooled_reference_samples.parquet")
    models = train(data, samples, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            controller = ExitController(frame, models, cfg["confirmation_days"])
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "早期经验账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                "prediction_available_closes": int(decisions.learning_status.eq("PREDICTION_AVAILABLE").sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧完整账户日历不一致")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：早期经验完整账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "INDEX_PREHISTORY_EXIT_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": sum(m["status"] == "FIT_COMPLETE" for m in models), "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "account_coverage": coverage, "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"] == PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"prepare": prepare, "freeze": freeze, "run": run}[sys.argv[1]]()
