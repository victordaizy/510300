"""仅核对既有入场信息和成熟成员，尚不拟合或计算候选收益。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/research/510300_entry_context_intercept_preflight_20260908"
    paths = {"features": root / "reports/research/510300_adaptive_allocation_v1/features.parquet",
             "cycles": root / "reports/research/510300_learned_cycle_exit_v1/reference/D60_INTRA_cycles.csv",
             "models": root / "reports/research/510300_within_cycle_exit_v1/saved_models.json",
             "entry_gate": root / "docs/510300_ENTRY_PAYOFF_GATE_V1.md",
             "parent_advantage": root / "docs/510300_CONTEXTUAL_PARENT_ADVANTAGE_V1.md"}
    data, cycles = pd.read_parquet(paths["features"]), pd.read_csv(paths["cycles"])
    models = json.loads(paths["models"].read_text(encoding="utf-8"))["models"]
    require(cycles.cycle_id.is_unique, "原参考周期编号不唯一")
    rows = []
    for cycle in cycles.itertuples():
        origin = int(cycle.entry_index)-1
        require(origin >= 0 and pd.Timestamp(cycle.entry_origin) == data.date.iloc[origin], "实际买入前收盘日期不符")
        require(pd.Timestamp(cycle.entry_date) == data.date.iloc[int(cycle.entry_index)], "原买入日与完整日历不同")
        values = data.loc[origin, ["mom20", "sma120", "vol20"]].to_numpy(float)
        rows.append({"cycle_id": int(cycle.cycle_id), "entry_origin_index": origin, "entry_origin": cycle.entry_origin,
                     "entry_date": cycle.entry_date, "exit_date": cycle.exit_date,
                     **dict(zip(["entry_mom20", "entry_sma120", "entry_vol20"], values)), "all_three_known": bool(np.isfinite(values).all())})
    table = pd.DataFrame(rows).set_index("cycle_id", drop=False)
    checks = []
    for record in models:
        if record["status"] != "FIT_COMPLETE":
            continue
        ids = record["training_cycles"]
        require({r["cycle_id"] for r in record["model"]["cycle_intercepts"]} == set(ids), "保存截距成员不同")
        selected = table.loc[ids]
        require(selected.all_three_known.all(), "成熟训练周期的进入三因子缺失")
        require((pd.to_datetime(selected.exit_date) <= pd.Timestamp(record["fit_origin"])).all(), "进入上下文包含未完成周期")
        checks.append({"fit_origin": record["fit_origin"], "mature_cycles": len(ids), "all_entry_information_known": True})
    require(len(checks) == 114, "可复用成熟模型次数改变")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "原参考周期买入前状态.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(out / "逐月成熟上下文覆盖.csv", index=False, encoding="utf-8-sig")
    result = {"checked_at": now(), "status": "SOURCE_AND_BOUNDED_METHOD_COMPARISON_COMPLETE_NOT_FITTED",
              "reference_cycles": len(table), "complete_entry_context_cycles": int(table.all_three_known.sum()),
              "mature_models_reusable": len(checks), "minimum_training_cycles": min(r["mature_cycles"] for r in checks),
              "new_models": 0, "new_accounts": 0, "new_references": 0,
              "method_difference": "原95预测整笔交易净收益决定是否进入；原90预测两组合未来二十日相对收益。本候选保留114已保存的周期内八项系数，另以已完成周期的进入三因子估计其剩余周期截距，只影响实际持有期间的继续价值。不是重开两项失败的交易规则。",
              "bounded_search": "research/docs/config/tests中检索周期截距、cycle_intercepts、cycle_fixed_effect和entry_context加intercept，仅发现114及其后续设想；不声称穷尽所有方法。",
              "limitations": "训练周期截距是估计量、每月仅十至二十周期，预测误差和同源选择风险仍在；只有完整实际账户能检验增量。",
              "sources": [{"name": name, "path": str(path.relative_to(root)), "sha256": digest(path)} for name, path in paths.items()]}
    write_json(out / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
