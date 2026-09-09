"""保存按持仓天数拆分的成熟周期支持，不拟合或回测。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import training_rows
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    p31 = root / "reports/research/510300_learned_cycle_exit_v1"
    out = root / "reports/research/510300_finite_horizon_age_support_20260908"
    paths = [p31 / "all_reference_samples.parquet", p31 / "reference/D60_INTRA_cycles.csv", p31 / "saved_models.json"]
    samples = pd.read_parquet(paths[0]); samples = samples[samples.signal.eq("D60_INTRA")].copy()
    cycles = pd.read_csv(paths[1]).set_index("cycle_id")
    samples["holding_age"] = samples.origin_index-samples.cycle_id.map(cycles.entry_index)+1
    np.testing.assert_allclose(np.log1p(samples.holding_age), samples.log_holding_days, atol=1e-12, rtol=0)
    require(not samples.duplicated(["cycle_id", "holding_age"]).any(), "同周期同持仓日有重复训练状态")
    models = json.loads(paths[2].read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    cells, months = [], []
    for original in models:
        if original["status"] != "FIT_COMPLETE":
            continue
        selected, ids = training_rows(samples, original["fit_index"], {"recent_cycles": 20})
        require(ids == original["training_cycles"] and len(selected) == original["training_rows"], "逐期支持改变原成熟成员")
        counts = selected.groupby("holding_age").cycle_id.nunique()
        usable = counts[counts.ge(10)].index
        months.append({"fit_origin": original["fit_origin"], "training_cycles": len(ids), "original_state_rows": len(selected),
                       "age_cells": len(counts), "age_cells_with_ten_cycles": len(usable),
                       "member_rows_in_supported_ages": int(selected.holding_age.isin(usable).sum()),
                       "first_supported_age": int(usable.min()) if len(usable) else None, "last_supported_age": int(usable.max()) if len(usable) else None})
        cells.extend({"fit_origin": original["fit_origin"], "fit_index": original["fit_index"], "holding_age": int(age),
                      "distinct_complete_cycles": int(count), "at_least_ten_cycles": bool(count >= 10),
                      "at_least_one_hundred_rows_for_this_age": bool(count >= 100)} for age, count in counts.items())
    frame = pd.DataFrame(cells)
    require(len(months) == 114 and frame.distinct_complete_cycles.max() <= 20, "按持仓日拆分超过原二十完整周期限制")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "每月每个持仓日的独立周期数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(months).to_csv(out / "逐月支持汇总.csv", index=False, encoding="utf-8-sig")
    result = {"checked_at": now(), "status": "FINITE_HORIZON_AGE_SUPPORT_RECORDED_NO_NEW_MODEL_OR_ACCOUNT", "mature_months": len(months),
              "age_range": [int(samples.holding_age.min()), int(samples.holding_age.max())], "month_age_cells": len(cells),
              "cells_with_ten_cycles": int(frame.at_least_ten_cycles.sum()), "cells_below_ten_cycles": int((~frame.at_least_ten_cycles).sum()),
              "cells_with_one_hundred_rows": int(frame.at_least_one_hundred_rows_for_this_age.sum()),
              "original_month_membership_rows_with_repeats": sum(m["original_state_rows"] for m in months),
              "supported_membership_rows_with_repeats": sum(m["member_rows_in_supported_ages"] for m in months),
              "first_mature_month": months[0], "last_mature_month": months[-1], "new_models": 0, "new_accounts": 0, "new_reference_accounts": 0,
              "interpretation": "按持仓日逐期回归具备一部分十周期支持，但每期不可能有一百行，因为最多二十完整周期且每周期每持仓日仅一行。若采用新方法，必须明确至少十独立周期/期，而原一百状态只作为全窗口条件；不能宣称逐期仍满足一百行。支持不足期限需保留未知和明确的原退出行为，不降低门槛。",
              "method_comparison": "52为所有年龄混合的同一模型固定六十次自举；96为空仓机会组买入等待迭代。本方向为按持仓年龄逐期回归已实现的后续策略现金流，尚未登记。",
              "source_paper": "https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf",
              "sources": [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in paths]}
    write_json(out / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
