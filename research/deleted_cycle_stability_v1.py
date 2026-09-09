"""单一条件逐周期删除一致退出模型的完整历史账户检验。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import training_rows, chinese_formula, CN
from research.deleted_cycle_stability_inputs_v1 import fit_deletions, DeletedCycleExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_deleted_cycle_stability_v1"
CONFIG = ROOT / "config/510300_deleted_cycle_stability_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "DELETED_CYCLE_STABILITY"
CONTROLS = {"REARM_RIDGE": (P32, "原平均继续收益学习退出"), "REARM_NONE": (P32, "原价格时间退出，无学习"),
    "PANIC_LEARNED_HALF": (P46, "原急跌与学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "本轮已登记，不重复冻结")
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "confirmation_days", "specification", "saved_models"]}
    cfg.update({k: train[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_names", "feature_clip", "ridge_alpha"]})
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "逐周期删除一致与成熟成交测试尚未通过")
    for item in old["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原入场退出及模型来源身份改变")
    cfg.update(study_id="510300_DELETED_CYCLE_EXIT_STABILITY_V1", round=72, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        fit_intercept=True, solver="svd", model_loss="CYCLE_EQUAL_WEIGHTED_SQUARE_ERROR_WITH_ORIGINAL_RIDGE_PENALTY", planned_deletion_fits=2055, all_members_must_be_negative=True, deletion_backfill=False,
        fit_failure="NO_VIEW_MODEL_FIT_FAILED_KEEP_ORIGINAL_EXITS_NO_FALLBACK_SOLVER", rules="docs/510300_DELETED_CYCLE_EXIT_STABILITY_V1.md",
        samples=str((P31 / "all_reference_samples.parquet").relative_to(ROOT)), new_reference_accounts=0, source_budget_cny=0,
        previous_goal_turn_classification="PROGRESS_ROUND71_COMPLETED_AND_DELIVERED", position_impact=0, goal_achieved=False,
        independent_validation="NOT_ESTABLISHED", environment={"sklearn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__})
    paths = [Path(__file__), ROOT / "research/deleted_cycle_stability_inputs_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
        ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
        ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["samples"],
        ROOT / "config/510300_rearmed_session_exit_v1.json", ROOT / "config/510300_learned_cycle_exit_v1.json", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "tests/test_deleted_cycle_stability_v1.py", ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, name) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第72轮一个条件逐周期删除一致继续价值设置已冻结，尚未拟合或读取新账户收益。", flush=True)


def train_models(data, samples, originals, cfg):
    models, receipts, memberships, deletion_receipts, coefficient_rows = [], [], [], [], []
    chinese = ["# 每月逐周期删除退出模型：全部因子系数与中文规则", "", "原模型不重拟合；每个已成熟主样本的全部删除模型均保存。表中小数采用10位展示，实际计算使用保存完整精度。", ""]
    attempts = 0
    for position, original in enumerate(originals):
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "原模型与本轮成熟样本不同")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (original["status"] == "FIT_COMPLETE"), "成熟模型支持时点改变")
        stored, failure, deletions = None, None, []
        status = "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"
        if eligible:
            deletions = fit_deletions(rows, t, cfg)
            attempts += len(deletions)
            require(len(deletions) == len(ids), "没有尝试全部原周期的删除")
            if all(d["status"] == "FIT_COMPLETE" for d in deletions):
                stored = {"kind": "DELETED_CYCLE_COMMITTEE", "base_model": original["model"], "deletions": deletions}
                status = "FIT_COMPLETE"
            else:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", "至少一个删除模型数值求解失败，整组学习判断缺失"
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
            "status": status, "eligible_for_fit": eligible, "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
            "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None, "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
            "failure": failure, "deletion_fits_attempted": len(deletions), "deletion_fits_completed": sum(d["status"] == "FIT_COMPLETE" for d in deletions),
            "model": stored, "failed_month_deletion_details": deletions if failure else None}
        require(not len(rows) or (rows.exit_index <= t).all(), "参考周期在训练时点尚未完成")
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles", "failed_month_deletion_details"]})
        if eligible:
            memberships.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                "sample_weight": float(r.sample_weight), "fit_status": status} for r in rows.itertuples())
        chinese += [f"## {record['fit_origin']}", "", f"主样本已完成周期{len(ids)}个，状态{len(rows)}条，尝试删除模型{len(deletions)}个。", ""]
        formula_models = []
        if eligible:
            formula_models.append(("原全部周期模型，直接复用", original["model"]))
        for deleted in deletions:
            deletion_receipts.append({"fit_index": t, "fit_origin": record["fit_origin"], **{k: v for k, v in deleted.items() if k != "model"}})
            if deleted["model"] is not None:
                formula_models.append((f"删除原周期{deleted['deleted_cycle_id']}后的模型", deleted["model"]))
            else:
                chinese += [f"删除原周期{deleted['deleted_cycle_id']}后求解失败，本月整组学习判断不可用。", ""]
        if not formula_models:
            chinese += ["本月成熟样本不足，没有学习模型，继续原价格及时间退出。", ""]
        for label, model in formula_models:
            chinese += [f"### {label}", ""]
            formula = chinese_formula(model)
            chinese += [formula[0], "", *formula[1:], ""]
            coefficient_rows.extend({"模型训练日": record["fit_origin"], "模型说明": label, "因子": name, "训练均值": mean,
                "训练标准差": scale, "标准化系数": coef, "截距": model["intercept"], "截断绝对值": model["feature_clip"]}
                for name, mean, scale, coef in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
        if (position + 1) % 20 == 0:
            print(f"已处理{position + 1}个原月度时点，累计完成{attempts}次删除拟合尝试。", flush=True)
    require(attempts == cfg["planned_deletion_fits"], "实际删除拟合次数与登记不符")
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": cfg["feature_names"], "loss": cfg["model_loss"]}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(deletion_receipts).to_csv(OUT / "deletion_training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(OUT / "全部月度模型系数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月删除周期退出模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    print(f"原成熟时点不变：完整模型组{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次，共{attempts}个删除模型。", flush=True)
    return models, receipts


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "逐周期删除一致冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")].copy()
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    models, receipts = train_models(data, samples, originals, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], DeletedCycleExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "逐周期删除一致账户结算不符")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "逐周期删除一致退出违反次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()), "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()), "completed_round_trips": len(cycles),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "waiting_for_new_condition_rows": int(decisions.action.str.contains("旧入场条件", regex=False).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "逐周期删除一致退出"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "逐周期删除一致及对照完整日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：一个新逐周期删除一致账户和四个保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "DELETED_CYCLE_STABILITY_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_reference_accounts": 0,
        "new_model_fits": sum(r["deletion_fits_attempted"] for r in receipts), "completed_fits": sum(r["deletion_fits_completed"] for r in receipts),
        "complete_model_updates": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "failed_submodel_fits": sum(r["deletion_fits_attempted"] - r["deletion_fits_completed"] for r in receipts),
        "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts), "failed_fits": sum(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in receipts),
        "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "训练次数": result["new_model_fits"], "求解失败": result["failed_fits"], "持仓覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
