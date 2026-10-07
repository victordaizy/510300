"""用保存结果复核同月比较及条件投影的金额、时序和区间。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_balance_source_comparability_v14"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT / "completion_receipt.json"
    if target.exists():
        raise RuntimeError("本轮交付已经复核，不覆盖。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for r in frozen["inputs"]:
        assert digest(OUT / "inputs" / r["name"]) == r["sha256"]
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    publication = json.loads((OUT / "publication_receipt.json").read_text(encoding="utf-8"))
    assert digest(OUT / result["report"]) == publication["report_sha256"]
    assert digest(OUT / result["figure"]) == publication["figure_sha256"]
    a = pd.read_csv(OUT / "results/104个月_完整输入与原结果.csv")
    raw = pd.read_csv(OUT / "inputs/outcomes.csv").set_index("stat_month")
    assert len(a) == 104 and a.stat_month.is_unique
    for field in ["E0_20_return", "E1_20_return", "E0_20_worst_path", "E1_20_worst_path"]:
        np.testing.assert_allclose(a[field].to_numpy(float), raw.loc[a.stat_month, field].to_numpy(float), equal_nan=True, atol=1e-13)
    yyyymm = pd.PeriodIndex(a.stat_month, freq="M")
    np.testing.assert_array_equal(a.month_id, yyyymm.asi8)
    coeff = pd.read_csv(OUT / "results/全部条件投影系数_不筛选.csv")
    zframe = a[a.training_regime.eq("M1_OLD_M2_MMF2018") & a.complete_design_row]
    names = ["intercept", "current_component", "base_component"] + [f"month_{m:02d}" for m in range(2, 13)] + protocol["continuous_design"]["context_controls"]
    x = np.column_stack([np.ones(len(zframe)), zframe.current_component, zframe.base_component]
                         + [zframe.calendar_month.eq(m).to_numpy(float) for m in range(2, 13)]
                         + [zframe[c].to_numpy(float) for c in protocol["continuous_design"]["context_controls"]])
    scale = np.r_[1., x[:, 1:].std(axis=0)]
    z = x/scale
    q, r = np.linalg.qr(z, mode="reduced")
    month = zframe.month_id.to_numpy(int)
    distance = np.abs(month[:, None]-month[None, :])
    kernel = np.maximum(1-distance/7, 0)
    checked_coefficients = 0
    for entry in ["E0", "E1"]:
        y = zframe[f"{entry}_20_return"].to_numpy(float)*100
        beta_scaled = np.linalg.solve(r, q.T@y)
        beta = beta_scaled/scale
        errors = y-z@beta_scaled
        scores = z*errors[:, None]
        inverse_r = np.linalg.inv(r)
        inverse_gram = inverse_r@inverse_r.T
        cov_scaled = inverse_gram@(scores.T@kernel@scores)@inverse_gram*len(z)/(len(z)-len(names))
        se = np.sqrt(np.diag(cov_scaled))/scale
        saved = coeff[coeff.entry.eq(entry) & coeff.stage.eq("C_CONTEXT")].set_index("feature").loc[names]
        np.testing.assert_allclose(beta, saved.coefficient_return_pp_per_feature_unit, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(se, saved.hac_se, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(beta-1.96*se, saved.low95_descriptive, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(beta+1.96*se, saved.high95_descriptive, rtol=1e-8, atol=1e-8)
        checked_coefficients += len(names)
    paired = pd.read_csv(OUT / "results/全部同月跨年配对及背景差.csv")
    assert len(paired) == 6
    assert paired.base_only_month.nunique() == 3
    assert set(paired.calendar_month) == {4, 8, 11}
    for entry in ["E0", "E1"]:
        recalculated = paired.groupby("calendar_month")[f"{entry}_return_difference_pp"].mean().mean()
        np.testing.assert_allclose(recalculated, result["old_group_difference_pp"][f"{entry}_same_month"], atol=1e-12)
    missing = a.loc[a.stat_month.eq("2026-08")].iloc[0]
    assert not missing.complete_design_row and pd.isna(missing.E0_20_return)
    count = pd.read_csv(OUT / "results/固定设计与样本数.csv")
    new = count[count.regime.eq("M1_NEW2025")].iloc[0]
    assert new.status == "NOT_ESTIMABLE_FULL_DESIGN" and new.n == 15 and new.residual_df == 0
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    checks = {"saved_original_labels_equal": True, "old_rows": len(zframe), "full_design_columns": len(names),
              "full_design_coefficients_and_hac_intervals_checked": checked_coefficients,
              "same_month_paired_difference_recomputed": True, "new_m1_not_estimated": True, "last_month_no_view_preserved": True}
    record = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_CALENDAR_COMPARISON_AND_QR_RECOMPUTATION",
              "continuation_classification": "PROGRESS", "goal_achieved": False, "independent_validation": False,
              "checks": checks, "visual_check": {"status": "PASS_VISUAL_INSPECTION", "figure_sha256": publication["figure_sha256"],
                                                 "notes": "图例已调整，中文、区间、脚注无重叠裁切，数值与保存结果一致。"},
              "report_sha256": publication["report_sha256"], "result_sha256": digest(OUT / "result.json"),
              "scope_limit": "数值复算不等于统计独立验证、因果识别或收益优势。"}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"完成状态": record["status"], "检查": checks, "目标完成": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
