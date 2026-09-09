"""只分解已保存实际路径的经济差异，不生成反事实账户或新策略。"""
import json
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
P98 = ROOT / "reports/research/510300_tangency_reference_budget_v1"
OUT = ROOT / "reports/research/510300_saved_budget_state_differences_20260908"
PARENTS = {"ROUND91": ("510300_continuous_reference_min_variance_v1", "CONTINUOUS_REFERENCE_MIN_VARIANCE"), "ROUND97": ("510300_universal_reference_budget_v1", "UNIVERSAL_REFERENCE_BUDGET")}


def classify(row):
    cash = row.panic_budget+row.learned_budget == 0
    if row.budget_status.startswith("NO_VIEW"):
        return "无新估计，沿用此前现金选择" if cash else "无新估计，沿用此前非现金预算"
    if row.budget_status == "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN":
        return "本月明确现金选择"
    if row.budget_status == "TANGENCY_BUDGET_AVAILABLE":
        return "本月有效切点预算"
    return "初始预算或其余原状态"


def main():
    require(not (OUT / "result.json").exists(), "保存路径差异诊断已完成")
    OUT.mkdir(parents=True, exist_ok=True)
    daily, aggregate, overall, sources = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            lp = P98 / period / cost / "TANGENCY_REFERENCE_BUDGET_ledger.parquet"
            dp = P98 / period / cost / "TANGENCY_REFERENCE_BUDGET_decisions.parquet"
            current, decisions = pd.read_parquet(lp), pd.read_parquet(dp)
            require(pd.DatetimeIndex(current.origin).equals(pd.DatetimeIndex(decisions.origin)), "当日实际路径与前收盘预算状态错位")
            labels = [classify(row) for row in decisions.itertuples()]
            sources.extend([lp, dp])
            for parent, (folder, model) in PARENTS.items():
                path = ROOT / "reports/research" / folder / period / cost / f"{model}_ledger.parquet"
                saved = pd.read_parquet(path)
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(saved.date)), "实际路径日历不同")
                sources.append(path)
                d = pd.DataFrame({"period": period, "cost": cost, "comparison": "ROUND98_MINUS_"+parent, "date": current.date, "origin": current.origin, "budget_state": labels})
                for field in ["pnl", "price_pnl", "dividend_recognized", "commission", "slippage_cost"]:
                    d[field+"_difference"] = current[field].to_numpy()-saved[field].to_numpy()
                d["new_holding_close"] = current.shares.gt(0).to_numpy(int)
                d["parent_holding_close"] = saved.shares.gt(0).to_numpy(int)
                d["both_original_states_idle"] = decisions.panic_state.add(decisions.learned_state).eq(0).to_numpy(int)
                economics = d.price_pnl_difference+d.dividend_recognized_difference-d.commission_difference-d.slippage_cost_difference
                require((economics-d.pnl_difference).abs().max() < 1e-6, "日度经济差异未配平")
                require(abs(d.pnl_difference.sum()-(current.equity.iloc[-1]-saved.equity.iloc[-1])) < 1e-6, "差异没有回到完整终值")
                daily.extend(d.to_dict("records"))
                numeric = [c for c in d if c.endswith("_difference") or c.endswith("_close") or c == "both_original_states_idle"]
                for label, group in d.groupby("budget_state", sort=False):
                    aggregate.append({"period": period, "cost": cost, "comparison": "ROUND98_MINUS_"+parent, "budget_state": label, "days": len(group), **group[numeric].sum().to_dict()})
                overall.append({"period": period, "cost": cost, "comparison": "ROUND98_MINUS_"+parent, "days": len(d), **d[numeric].sum().to_dict()})
    pd.DataFrame(daily).to_parquet(OUT / "saved_daily_economic_differences.parquet", index=False)
    pd.DataFrame(aggregate).to_csv(OUT / "按当时预算状态的真实路径差异.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overall).to_csv(OUT / "完整账户经济差异.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "SAVED_ACTUAL_PATH_DIFFERENCES_READY_NO_CAUSAL_ATTRIBUTION_OR_NEW_POLICY", "daily_comparisons": len(daily), "state_groups": len(aggregate), "overall": overall,
        "interpretation": "按98前一收盘已知状态归集实际日盈亏差异，只说明两条既有路径在这些日期表现不同；含此前不同现金和持仓形成的路径依赖，不是该状态的独立因果效果或可拼接策略。",
        "new_strategy_configurations": 0, "new_model_fits": 0, "new_accounts": 0, "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(sources+[Path(__file__)]))]}
    write_json(OUT / "result.json", result, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 98, "诊断前序不是98轮")
    index["next_work"].update(status="SAVED_BUDGET_STATE_DIFFERENCES_READY_NEW_METHOD_NOT_REGISTERED", prepared_input=str((OUT / "result.json").relative_to(ROOT)))
    index.update(updated_at=now(), process_state_note="96至98完成4新设置16新完整账户；99已保存11292个日度经济比较，尚未登记新策略或账户。")
    write_json(index_path, index)
    print(json.dumps({"状态": result["status"], "比较日行": len(daily), "完整账户差异": overall, "基础费用状态分组": [r for r in aggregate if r["cost"] == "BASE"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
