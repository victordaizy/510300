"""仅从保存的模型与成交结果解释费用变化如何影响本轮退出判断。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_vintage_exit_v1"


def main():
    require(not (OUT / "saved_cost_feedback_receipt.json").exists(), "费用反馈说明已经完成")
    records = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    models = {r["fixed_prediction_identity"]: r["model"] for r in records if r["model"]}
    require(all(m["coefficients"][1] < 0 for m in models.values()), "含成本浮盈亏系数并非全部为负")
    checks, timing = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        frames, cycles = {}, {}
        for cost in ["BASE", "STRESS"]:
            folder = OUT / period / cost
            frames[cost] = pd.read_parquet(folder / "SPARSE_VINTAGE_EXIT_decisions.parquet")
            cycles[cost] = pd.read_csv(folder / "SPARSE_VINTAGE_EXIT_cycles.csv")
        left, right = frames["BASE"], frames["STRESS"]
        paired = left.merge(right, on=["origin_index", "model_selection_origin"], suffixes=("_base", "_stress"))
        paired = paired[paired.learning_status_base.eq("PREDICTION_AVAILABLE") & paired.learning_status_stress.eq("PREDICTION_AVAILABLE")]
        for row in paired.to_dict("records"):
            require(row["fixed_prediction_identity_base"] == row["fixed_prediction_identity_stress"], "同日同次进入使用了不同模型")
            model = models[row["fixed_prediction_identity_base"]]
            values = [np.array([row[f + suffix] for f in model["features"]], float) for suffix in ["_base", "_stress"]]
            z = [np.clip((v - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"]) for v in values]
            contribution = (z[1] - z[0]) * model["coefficients"]
            other = contribution.copy()
            other[1] = 0.
            require(np.max(abs(other)) < 1e-12, "两档费用预测差还含有其他有效因素变化")
            difference = row["continuation_prediction_stress"] - row["continuation_prediction_base"]
            require(abs(difference - contribution[1]) < 1e-12, "费用反馈预测差未完整对上")
            checks.append({"period": period, "origin": row["origin_base"], "entry_date": row["model_selection_origin"],
                "base_prediction": row["continuation_prediction_base"], "stress_prediction": row["continuation_prediction_stress"],
                "prediction_difference": difference, "own_profit_contribution": contribution[1],
                "different_prediction_sign": (row["continuation_prediction_base"] < 0) != (row["continuation_prediction_stress"] < 0),
                "base_exit_requested": row["learned_exit_requested_base"], "stress_exit_requested": row["learned_exit_requested_stress"]})
        merged = cycles["BASE"].merge(cycles["STRESS"], on="entry_date", how="outer", suffixes=("_base", "_stress"), indicator=True)
        changed = merged[merged._merge.ne("both") | merged.exit_date_base.ne(merged.exit_date_stress)].copy()
        changed.insert(0, "period", period)
        timing.extend(changed.to_dict("records"))
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / "saved_cost_feedback_prediction_checks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timing).to_csv(OUT / "saved_cost_feedback_changed_cycles.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_SAME_ENTRY_SAME_MODEL_COST_DEPENDENT_PREDICTIONS_EXPLAINED",
        "common_entry_prediction_pairs": len(checks), "different_sign_pairs": int(checks.different_prediction_sign.sum()),
        "changed_entry_or_exit_rows": len(timing), "all_distinct_cycle_return_coefficients_negative": True,
        "maximum_prediction_difference_error": float(abs(checks.prediction_difference - checks.own_profit_contribution).max()),
        "new_models": 0, "new_accounts": 0,
        "interpretation": "费用进入本账户浮盈亏因素，改变同一模型的预测、退出和后续重入路径；并非固定交易路径上增加费用能增加利润。"}
    write_json(OUT / "saved_cost_feedback_receipt.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    print(checks[checks.different_prediction_sign].to_string(index=False), flush=True)
    print(pd.DataFrame(timing)[["period", "entry_date", "exit_date_base", "exit_date_stress", "net_profit_cny_base", "net_profit_cny_stress"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
