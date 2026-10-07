"""核对货币余额来源的季节可比性及多层背景下的条件关联。"""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_balance_source_comparability_v14"
SOURCES = {
    "context.csv": ROOT / "reports/research/510300_macro_transmission_context_v4/results/104个月_当时可见多层证据_不含未来标签.csv",
    "funding.csv": ROOT / "reports/research/510300_funding_quantity_price_bridge_v12/results/104个月_货币数量与融资价格.csv",
    "outcomes.csv": ROOT / "reports/research/510300_macro_transmission_context_v4/results/104个月_多层证据与原后续路径.csv",
    "prior_groups.csv": ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2/results/主20日_当期余额支持与否.csv",
}
MONEY = ["current_component", "base_component"]
CONTROLS = ["pre_return20_pp", "pre_return60_pp", "log_rv20", "down_fraction", "corporate_total_yoy_percent",
            "corporate_long_yoy_percent", "household_long_yoy_percent", "orders_gap50", "orders_change1", "funding_gap_bp", "segmentation_gap_bp"]
ENTRIES = ["E0", "E1"]
GROUPS = ["改善_当期相对余额同向", "改善_当期相对余额未同向"]
OLD, NEW = "M1_OLD_M2_MMF2018", "M1_NEW2025"


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("研究范围已经固定，不覆盖。")
    for name in ["inputs", "results", "code", "figures"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": "510300_BALANCE_SOURCE_COMPARABILITY_V14",
        "previous_turn_classification": "PROGRESS：V13完成三家银行原公告、利润及每股收益传导复算。",
        "question": "旧口径余额来源分组的约2.09个百分点收益差，有多少同月份可比支持？在固定的价格、风险、信贷、订单和资金背景中，当期项与基数项的条件关联是否稳定？",
        "scope": "原104月，按旧新M1分别处理；行情及原20日E0/E1标签不变。所有原结果已见，是知情历史诊断，不是独立验证。",
        "calendar_support": "沿用V2两组，不改定义；分别按统计月份1至12列出所有观察。仅两个组均有样本的同月份构成共同支持；每个共同月份等权，组内同月年份等权。保存全部跨年份同月配对，但不当独立试验。",
        "continuous_design": {"main_outcome": "E0_20_return * 100", "delay_check": "E1_20_return * 100",
            "money_components": MONEY, "stages": {"A_RAW": "截距与两项货币分解", "B_CALENDAR": "A加统计月份固定效应", "C_CONTEXT": "B加以下11项事前背景"},
            "context_controls": CONTROLS, "credit_normalization": "同区间同比差/由当期值减同比差得到的去年同区间值*100；去年分母必须为正，沿用原口径断点与缺失。",
            "same_rows": "三层在同一完整行集合比较；不因为结果改变控制项或样本。",
            "admission": "完整设计须满列秩、残差自由度至少12；新口径不足时全套不估计，不减少控制项。",
            "interpretation": "OLS和残差相关只描述给定背景的样本内条件关联；价格、信贷等可能是传导中的变量，系数不是货币的总因果效应。",
            "uncertainty": "按实际统计月距离计算6个月Bartlett核HAC，乘n/(n-p)，给正态近似95%描述区间；只此带宽，不筛选显著项，不声称多重比较校正。",
            "stability": "固定完整设计逐一留出每个自然年，仅检查系数敏感性；不用于选模型，不冒称测试集预测。"},
        "limits": ["同月不等于春节日期或宏观冲击相同", "2018年同比来源缺失、2023年信贷口径断点继续缺失", "不填历史首版认证缺口", "没有新增预测模型、仓位模型或账户回测", "不恢复旧冻结失败分支"],
        "goal_achieved": False, "independent_validation": False, "orders_authorized": False}
    save("protocol.json", protocol)
    shutil.copy2(OUT / "protocol.json", ROOT / "config/510300_balance_source_comparability_v14.json")
    receipts = []
    for name, src in SOURCES.items():
        assert src.is_file()
        shutil.copy2(src, OUT / "inputs" / name)
        receipts.append({"name": name, "source": str(src), "sha256": digest(src)})
    save("freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": receipts})
    print(json.dumps({"已固定": protocol["study_id"], "输入文件": len(receipts), "状态": "FROZEN_BEFORE_NEW_CONDITIONAL_RESULTS"}, ensure_ascii=False))


def design(frame, stage):
    names = ["intercept"] + MONEY
    arrays = [np.ones(len(frame))] + [frame[x].to_numpy(float) for x in MONEY]
    if stage != "A_RAW":
        for month in range(2, 13):
            names.append(f"month_{month:02d}")
            arrays.append(frame.calendar_month.eq(month).to_numpy(float))
    if stage == "C_CONTEXT":
        names += CONTROLS
        arrays += [frame[x].to_numpy(float) for x in CONTROLS]
    return np.column_stack(arrays), names


def fit(frame, stage, entry):
    x, names = design(frame, stage)
    scale = np.ones(x.shape[1])
    scale[1:] = x[:, 1:].std(axis=0)
    if np.any(scale == 0):
        return {"status": "NOT_ESTIMABLE_CONSTANT_COLUMN"}, None
    z = x / scale
    rank = int(np.linalg.matrix_rank(z))
    n, p = x.shape
    if rank != p or n - rank < 12:
        return {"status": "NOT_ESTIMABLE_FULL_DESIGN", "n": n, "rank": rank, "columns": p, "residual_df": n-rank}, None
    y = frame[f"{entry}_20_return"].to_numpy(float) * 100
    beta_z = np.linalg.lstsq(z, y, rcond=None)[0]
    beta = beta_z / scale
    residual = y-z@beta_z
    scores = z * residual[:, None]
    meat = scores.T@scores
    month_ids = frame.month_id.to_numpy(int)
    for lag in range(1, 7):
        left, right = np.where((month_ids[:, None]-month_ids[None, :]) == lag)
        if len(left):
            s = scores[left].T@scores[right]
            meat += (1-lag/7) * (s+s.T)
    bread = np.linalg.inv(z.T@z)
    covariance_z = bread@meat@bread * n/(n-p)
    se = np.sqrt(np.maximum(np.diag(covariance_z), 0))/scale
    details = []
    for j, name in enumerate(names):
        other = np.delete(z, j, axis=1)
        xr = z[:, j] - other@np.linalg.lstsq(other, z[:, j], rcond=None)[0]
        yr = y - other@np.linalg.lstsq(other, y, rcond=None)[0]
        partial = float(np.corrcoef(xr, yr)[0, 1]) if j > 0 else np.nan
        if j > 0:
            np.testing.assert_allclose((xr@yr)/(xr@xr)/scale[j], beta[j], atol=1e-8)
        details.append({"feature": name, "coefficient_return_pp_per_feature_unit": beta[j], "hac_se": se[j],
            "low95_descriptive": beta[j]-1.96*se[j], "high95_descriptive": beta[j]+1.96*se[j],
            "partial_correlation": partial, "feature_sd": scale[j], "coefficient_per_sd_return_pp": beta[j]*scale[j]})
    rss = float(residual@residual)
    tss = float(((y-y.mean())**2).sum())
    meta = {"status": "ESTIMATED_DESCRIPTIVE_ONLY", "n": n, "rank": rank, "columns": p, "residual_df": n-p,
            "r2_in_sample": 1-rss/tss, "condition_number_scaled": float(np.linalg.cond(z)), "source_years": ";".join(map(str, sorted(frame.calendar_year.unique())))}
    return meta, pd.DataFrame(details)


def prepare():
    m = pd.read_csv(OUT / "inputs/context.csv")
    f = pd.read_csv(OUT / "inputs/funding.csv")
    funding_cols = ["stat_month", "snapshot_at", "funding_status", "fixing_known_at", "fdr_policy_gap_bp_recent20_mean", "fr_fdr_gap_bp_recent20_mean"]
    a = m.merge(f[funding_cols], on="stat_month", how="left", validate="one_to_one", suffixes=("", "_funding"))
    same = a.snapshot_at.fillna("").eq(a.snapshot_at_funding.fillna(""))
    assert same.all()
    a["calendar_month"] = pd.PeriodIndex(a.stat_month, freq="M").month
    a["calendar_year"] = pd.PeriodIndex(a.stat_month, freq="M").year
    a["month_id"] = pd.PeriodIndex(a.stat_month, freq="M").asi8
    mappings = {"current_component": "d3_relative_current_log_pp", "base_component": "d3_relative_base_revision_log_pp",
                "down_fraction": "v_down_fraction", "orders_change1": "orders_orders_change1_pp",
                "funding_gap_bp": "fdr_policy_gap_bp_recent20_mean", "segmentation_gap_bp": "fr_fdr_gap_bp_recent20_mean"}
    for dst, src in mappings.items():
        a[dst] = a[src]
    a["pre_return20_pp"] = a.past_return20 * 100
    a["pre_return60_pp"] = a.past_return60 * 100
    a["log_rv20"] = np.log(a.v_rv20.where(a.v_rv20 > 0))
    a["orders_gap50"] = a.orders_first_release_value - 50
    for stem in ["corporate_total", "corporate_long", "household_long"]:
        current, delta = a[f"loan_{stem}_ytd_yi"], a[f"loan_{stem}_ytd_yoy_change_yi"]
        prior = current - delta
        a[f"{stem}_yoy_percent"] = (100 * delta / prior).where(prior > 0)
    for name in ["loan_known_at", "orders_known_at", "fixing_known_at"]:
        known = pd.to_datetime(a[name], utc=True, errors="coerce")
        snap = pd.to_datetime(a.snapshot_at, utc=True, errors="coerce")
        assert not (known > snap).any(), name
    a.to_csv(OUT / "results/104个月_解释变量与缺失_不含未来标签.csv", index=False, encoding="utf-8-sig")
    outcomes = pd.read_csv(OUT / "inputs/outcomes.csv")
    fields = ["stat_month"] + [f"{e}_20_{s}" for e in ENTRIES for s in ["return", "status", "entry_date", "exit_date", "worst_path"]]
    return a.merge(outcomes[fields], on="stat_month", how="left", validate="one_to_one")


def calendar_support(a):
    old = a[a.training_regime.eq(OLD) & a.arithmetic_mechanism_group.isin(GROUPS)].copy()
    rows, memberships, pairs = [], [], []
    summary = []
    for regime in [OLD, NEW]:
        sub = a[a.training_regime.eq(regime) & a.arithmetic_mechanism_group.isin(GROUPS)].copy()
        common = []
        for month in range(1, 13):
            groups = [sub[sub.calendar_month.eq(month) & sub.arithmetic_mechanism_group.eq(g)] for g in GROUPS]
            supported = all(len(g) for g in groups)
            if supported:
                common.append(month)
            for gname, g in zip(GROUPS, groups):
                rows.append({"regime": regime, "calendar_month": month, "group": gname, "count": len(g), "has_two_groups": supported,
                             "stat_months": ";".join(g.stat_month), "E0_mean": g.E0_20_return.mean(), "E1_mean": g.E1_20_return.mean()})
            if supported:
                for left in groups[0].itertuples():
                    for right in groups[1].itertuples():
                        pair = {"regime": regime, "calendar_month": month, "current_supported_month": left.stat_month, "base_only_month": right.stat_month}
                        for entry in ENTRIES:
                            pair[f"{entry}_return_difference_pp"] = (getattr(left, f"{entry}_20_return")-getattr(right, f"{entry}_20_return"))*100
                        for field in ["pre_return20_pp", "pre_return60_pp", "corporate_total_yoy_percent", "corporate_long_yoy_percent", "household_long_yoy_percent", "orders_gap50", "funding_gap_bp"]:
                            pair[f"{field}_difference"] = getattr(left, field)-getattr(right, field)
                        pairs.append(pair)
        for row in sub.itertuples():
            memberships.append({"regime": regime, "stat_month": row.stat_month, "calendar_month": row.calendar_month, "group": row.arithmetic_mechanism_group, "common_support": row.calendar_month in common})
        for entry in ENTRIES:
            col = f"{entry}_20_return"
            for scope in ["ALL_ORIGINAL", "SAME_CALENDAR_MONTH_COMMON_SUPPORT"]:
                z = sub if scope == "ALL_ORIGINAL" else sub[sub.calendar_month.isin(common)]
                means = []
                for group in GROUPS:
                    g = z[z.arithmetic_mechanism_group.eq(group)]
                    weighted_mean = g[col].mean() if scope == "ALL_ORIGINAL" else g.groupby("calendar_month")[col].mean().mean()
                    means.append(weighted_mean)
                    summary.append({"regime": regime, "entry": entry, "scope": scope, "group": group, "count": len(g), "calendar_months": ";".join(map(str, common)) if scope != "ALL_ORIGINAL" else "ALL",
                        "return_mean_percent": weighted_mean*100, "return_median_percent": g[col].median()*100, "years": ";".join(map(str, sorted(g.calendar_year.unique()))),
                        "months": ";".join(g.stat_month), "status": "DESCRIPTIVE_DEPENDENT_OBSERVATIONS" if len(g) else "NO_COMMON_SUPPORT"})
                summary.append({"regime": regime, "entry": entry, "scope": scope, "group": "两组均值差", "count": len(z), "calendar_months": ";".join(map(str, common)),
                    "return_mean_percent": (means[0]-means[1])*100, "status": "DESCRIPTIVE_NOT_CAUSAL" if len(z) else "NO_COMMON_SUPPORT"})
    pd.DataFrame(rows).to_csv(OUT / "results/逐月份共同支持.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_csv(OUT / "results/全部分组月份_共同支持去留.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(OUT / "results/全部同月跨年配对及背景差.csv", index=False, encoding="utf-8-sig")
    summaries = pd.DataFrame(summary)
    summaries.to_csv(OUT / "results/原分组与同月可比结果.csv", index=False, encoding="utf-8-sig")
    prior = pd.read_csv(OUT / "inputs/prior_groups.csv")
    for r in summaries[summaries.scope.eq("ALL_ORIGINAL") & summaries.group.isin(GROUPS)].itertuples():
        previous = prior[prior.training_regime.eq(r.regime) & prior.entry.eq(r.entry) & prior.group.eq(r.group)].iloc[0]
        np.testing.assert_allclose(r.return_mean_percent, previous.return_mean*100, atol=1e-10)
        assert r.count == previous.mature_months
    assert len(old) == 31
    return summaries


def build():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮估计已保存，不重新拟合。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for row in frozen["inputs"]:
        assert digest(OUT / "inputs" / row["name"]) == row["sha256"]
    a = prepare()
    group_summary = calendar_support(a)
    required = MONEY + CONTROLS + [f"{e}_20_return" for e in ENTRIES]
    a["complete_design_row"] = a[required].notna().all(axis=1)
    a["missing_fields"] = a.apply(lambda r: ";".join(c for c in required if pd.isna(r[c])), axis=1)
    a.to_csv(OUT / "results/104个月_完整输入与原结果.csv", index=False, encoding="utf-8-sig")
    coefficients, summaries, leave_years, seasonality = [], [], [], []
    fits = 0
    for regime in [OLD, NEW]:
        subset = a[a.training_regime.eq(regime) & a.complete_design_row].copy()
        x, names = design(subset, "C_CONTEXT")
        rank = int(np.linalg.matrix_rank(x))
        if rank != x.shape[1] or len(subset)-rank < 12:
            summaries.append({"regime": regime, "status": "NOT_ESTIMABLE_FULL_DESIGN", "n": len(subset), "columns": len(names), "rank": rank, "residual_df": len(subset)-rank})
            continue
        for entry in ENTRIES:
            for stage in ["A_RAW", "B_CALENDAR", "C_CONTEXT"]:
                meta, detail = fit(subset, stage, entry)
                assert meta["status"] == "ESTIMATED_DESCRIPTIVE_ONLY"
                fits += 1
                summaries.append({"regime": regime, "entry": entry, "stage": stage, **meta})
                detail["regime"], detail["entry"], detail["stage"] = regime, entry, stage
                coefficients.append(detail)
            for year in sorted(subset.calendar_year.unique()):
                z = subset[subset.calendar_year.ne(year)]
                meta, detail = fit(z, "C_CONTEXT", entry)
                if detail is not None:
                    fits += 1
                    for row in detail[detail.feature.isin(MONEY)].to_dict("records"):
                        leave_years.append({"regime": regime, "entry": entry, "excluded_year": int(year), **meta, **row})
                else:
                    leave_years.append({"regime": regime, "entry": entry, "excluded_year": int(year), **meta})
        for scope, z in [("ALL_VALID_BALANCE", a[a.training_regime.eq(regime) & a[MONEY].notna().all(axis=1)]), ("SAME_COMPLETE_CONTEXT_ROWS", subset)]:
            for component in MONEY:
                value = z[component].to_numpy(float)
                within = z[component] - z.groupby("calendar_month")[component].transform("mean")
                r2 = 1-float(within@within)/float(((value-value.mean())**2).sum())
                seasonality.append({"regime": regime, "scope": scope, "component": component, "count": len(z), "calendar_month_r2_in_sample": r2,
                                    "interpretation": "统计月份均值的样本内解释比例；不是季节因果占比或预测成绩。"})
    c = pd.concat(coefficients, ignore_index=True)
    c.to_csv(OUT / "results/全部条件投影系数_不筛选.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "results/固定设计与样本数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leave_years).to_csv(OUT / "results/逐年留出_系数敏感性.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seasonality).to_csv(OUT / "results/货币分解项的月份结构.csv", index=False, encoding="utf-8-sig")
    summary = {"at": now(), "status": "COMPLETED_CALENDAR_SUPPORT_AND_CONDITIONAL_ASSOCIATION", "observations": len(a),
        "descriptive_fits": fits, "new_prediction_models": 0, "new_accounts": 0, "goal_achieved": False, "independent_validation": False,
        "full_context_rows_old": int((a.complete_design_row & a.training_regime.eq(OLD)).sum()),
        "full_context_rows_new": int((a.complete_design_row & a.training_regime.eq(NEW)).sum()),
        "checks": {"prior_group_counts_and_means_recomputed": True, "source_clocks_before_snapshots": True, "conditional_coefficients_verified_by_residual_identity": True,
                   "three_stages_use_same_observations": True, "new_m1_not_pooled": True}, "continuation_classification": "PROGRESS"}
    save("results/build_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False))
    print(group_summary[group_summary.group.eq("两组均值差")][["regime", "entry", "scope", "count", "return_mean_percent"]].to_string(index=False))
    print(c[c.feature.isin(MONEY)][["entry", "stage", "feature", "coefficient_return_pp_per_feature_unit", "low95_descriptive", "high95_descriptive", "partial_correlation"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "build"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else build()
