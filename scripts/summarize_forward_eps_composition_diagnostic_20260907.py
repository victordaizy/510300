"""解释已保存的样本构成变化，不新增收益读取或策略检验。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity


def main():
    parent = ROOT / "reports/research/510300_forward_eps_coverage_representativeness_v1"
    output = ROOT / "reports/research/510300_forward_eps_composition_change_explanation_v1"
    output.mkdir(parents=True, exist_ok=False)
    source_path = parent / "monthly_company_composition_diagnostic.parquet"
    month_path = ROOT / "reports/research/510300_forward_eps_monthly_policy_v2_csi/monthly_forward_eps_features.parquet"
    sources = pd.read_parquet(source_path)
    months = pd.read_parquet(month_path)
    valid = set(months.loc[months.all_eps_features_valid, "origin"])
    selected = sources.loc[sources.origin.isin(valid) & sources.prior_origin.isin(valid) & ~sources.annual_target_roll].copy()
    names = {"eps_growth": "前瞻EPS增长", "profit_revision": "同年度利润预测修正", "reported_earnings_yield": "报告前瞻盈利收益率"}
    records = []
    for feature, group in selected.groupby("feature", sort=True):
        full = group.full_median_change.to_numpy(float)
        common = group.common_company_median_change.to_numpy(float)
        assert np.isfinite(full).all() and np.isfinite(common).all()
        changed = np.abs(full) > 1e-12
        common_changed = np.abs(common) > 1e-12
        records.append({"factor": feature, "因子": names[feature], "相邻同年度有效月份对数": len(group),
            "新增有效公司数中位数": float(group.new_company_count.median()),
            "失去有效公司数中位数": float(group.lost_company_count.median()),
            "全部公司中位数变化次数": int(changed.sum()),
            "共同公司中位数不变但全部公司变化次数": int((changed & ~common_changed).sum()),
            "双方中位数都变化但方向相反次数": int((changed & common_changed & (np.sign(full) != np.sign(common))).sum())})
    frame = pd.DataFrame(records)
    assert frame["相邻同年度有效月份对数"].eq(45).all()
    selected.to_parquet(output / "same_year_adjacent_valid_months.parquet", index=False)
    selected.to_csv(output / "同年度相邻有效月份构成变化.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output / "因子样本变化说明.csv", index=False, encoding="utf-8-sig")
    result = {"study_id": "510300_FORWARD_EPS_COMPOSITION_CHANGE_EXPLANATION_V1", "completed_at": now(),
        "status": "SAVED_SOURCE_COMPOSITION_DIAGNOSTIC_EXPLAINED_NO_NEW_RETURN_READ",
        "diagnostic_values_already_observed_before_this_summary": True,
        "same_calendar_year_only": True, "both_months_old_eps_source_valid": True,
        "source_receipt": [identity(p) for p in [source_path, month_path, Path(__file__)]],
        "rows": records, "not_causal_return_attribution": True, "not_variance_explained": True,
        "new_model_fits": 0, "new_account_evaluations": 0, "round_19_policy_changed": False, "goal_achieved": False}
    save(output / "result.json", result, exclusive=True)
    explanation = """

## 补充：因子变化还可能来自有资料的公司发生变化

对原国信数据检查45组相邻月份：前后两个月都满足原有EPS覆盖要求，而且不跨公历年，避免把新年预测目标年度切换混进来。对每项因子，同时看全部有效公司的中位数变化，以及前后两个月都有资料的共同公司的中位数变化。

| 因子 | 全部样本中位数变化次数 | 共同公司中位数没变、全部样本却变化的次数 | 两者都变化但方向相反的次数 |
| --- | --- | --- | --- |
| 前瞻EPS增长 | 41 | 16 | 2 |
| 同年度利润预测修正 | 11 | 6 | 0 |
| 报告前瞻盈利收益率 | 42 | 15 | 7 |

这说明样本公司的加入或退出会影响最终因子。前瞻EPS增长每对月份新增有效公司数的中位数为9家，失去有效公司数的中位数为10家；同年度利润修正对应13家和15家。因子整体上升不能直接解释为同一批公司的预期普遍上升。

共同公司的中位数不变，并不意味着其中每家公司的预测都没有调整；上述次数也不是收益贡献或解释比例。这是样本构成诊断。第十九轮三种模型和买卖规则保持原登记，待完整账户结束后，再同时区分来源覆盖、预测修正、样本更替与持仓变化的作用。

诊断记录：`reports/research/510300_forward_eps_composition_change_explanation_v1/result.json`，另存三因子汇总和135条月份因子明细。不读取新收益，不增加模型或账户，不制作GPT数值包。
"""
    delivery = ROOT / "deliverables/510300前瞻EPS多机构覆盖与完整规则_20260907/前瞻EPS多机构覆盖进度与三种策略完整规则.md"
    with delivery.open("a", encoding="utf-8") as handle:
        handle.write(explanation)
    continuation = ROOT / "docs/510300_FORWARD_EPS_TWO_INSTITUTION_CONTINUATION_20260907.md"
    with continuation.open("a", encoding="utf-8") as handle:
        handle.write(explanation)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = read(index_path)
    assert len(index["completed_rounds"]) == 18 and index["evaluation_accounts_in_this_resumption"] == 438
    index["updated_at"] = now()
    index["new_evidence"].append("原EPS45组同年度相邻有效月份中，共同公司中位数不变但全部公司中位数变化：EPS增长16次、利润修正6次、报告盈利收益率15次；分别2、0、7次方向相反。样本构成影响不等于盈利变化或收益贡献，第十九轮规则未改。")
    index["completed_source_rebuilds"].append({"study": result["study_id"],
        "result": output.relative_to(ROOT).as_posix() + "/result.json", **result})
    index["next_work"].append("第十九轮之后结合保存的共同公司诊断，识别中位数变化是否混有样本更替；不把诊断次数称收益解释比例，不追改已登记策略。")
    index_path.write_text(__import__("json").dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print("样本变化解释已保存，账户与策略计数未变。")


if __name__ == "__main__":
    main()
