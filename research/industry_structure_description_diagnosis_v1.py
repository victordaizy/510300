"""完整保存描述结果与旧股票投票的对照，包含未知和信息过晚。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import industry_structure_description_study_v1 as study


def main():
    out, parent = study.OUT, study.parent
    if (out / "post_run_diagnosis.json").exists():
        raise RuntimeError("行业结构说明诊断已经保存，不覆盖。")
    daily = pd.read_parquet(out / "results/全部3488逐点行业结构_未知与源龄.parquet")
    cases = pd.read_parquet(out / "results/原四案例240行_行业结构与价量宏观.parquet")
    profiles = pd.read_parquet(out / "results/全部原事件0至20槽_固定行业传播.parquet")
    older_path = study.ROOT / "reports/research/510300_broker_fixed_cohort_propagation_v1/implementation_v1_0_1/results/全部事件0至20观察_固定组传播及未知.parquet"
    older = pd.read_parquet(older_path)
    older = older[["anchor_date", "date", "view_allowed", "descriptive_propagation_state", "leaders_positive_known_fraction", "followers_positive_known_fraction"]]
    older = older.rename(columns={"view_allowed": "stock_cohort_view_allowed"})
    for frame in (profiles, older):
        for name in ("date", "anchor_date"):
            frame[name] = pd.to_datetime(frame[name]).astype("datetime64[ns]")
    selected = profiles.loc[profiles.relative_session.isin([1, 5, 20])].copy()
    joined = selected.merge(older, on=["anchor_date", "date"], how="left", validate="one_to_one", indicator=True)
    if len(joined) != 429 or not joined._merge.eq("both").all():
        raise ValueError("全部143事件的三个固定时点与原股票观察没有完整对齐。")
    joined.drop(columns="_merge").to_parquet(out / "results/全部143事件三个固定时点_行业与原股票对照.parquet", index=False)
    joined.drop(columns="_merge").to_csv(out / "results/全部143事件三个固定时点_行业与原股票对照.csv", index=False, encoding="utf-8-sig")
    complete = joined.loc[joined.original_account_context.eq("COMPLETE")]
    contrasts = []
    for key, group in complete.groupby(["original_period", "relative_session", "descriptive_fixed_state", "descriptive_propagation_state"], dropna=False):
        contrasts.append({"period": key[0], "h": key[1], "industry_state": key[2], "stock_state": key[3],
            "original_cycles": len(group), "original_wins": int(group.original_cycle_net_pnl.gt(0).sum()),
            "original_losses": int(group.original_cycle_net_pnl.lt(0).sum()),
            "information_before_original_exit": int(group.information_before_original_exit_open.sum()),
            "industry_view_allowed": int(group.view_allowed.sum()), "stock_view_allowed": int(group.stock_cohort_view_allowed.sum()),
            "role": "全分组事后说明，禁止当作新策略胜率或择优窗口"})
    pd.DataFrame(contrasts).to_csv(out / "results/全部原周期三个固定时点_行业与股票全状态对照.csv", index=False, encoding="utf-8-sig")
    case_counts = []
    for case_id, group in cases.groupby("original_episode_id", sort=True):
        case_counts.append({"case_id": int(case_id), "rows": len(group), "daily_known": int(group.daily_view_allowed.sum()),
            "fixed_known": int(group.view_allowed.sum()), "daily_unknown": int((~group.daily_view_allowed).sum()),
            "fixed_unknown": int((~group.view_allowed).sum()), "date_min": group.date.min(), "date_max": group.date.max(),
            "fixed_unknown_with_formed_anchor": int((~group.view_allowed & group.anchor_status.eq("KNOWN_FIXED_INDUSTRY_ANCHOR")).sum()),
            "fixed_unknown_leader_group_incomplete": int((~group.view_allowed & group.leader_known_fraction.lt(.98)).sum()),
            "fixed_unknown_follower_group_incomplete": int((~group.view_allowed & group.follower_known_fraction.lt(.98)).sum())})
    years = []
    for year, group in daily.loc[daily.date.ge(pd.Timestamp("2015-01-05"))].groupby(daily.date.dt.year):
        known = group.loc[group.view_allowed]
        years.append({"year": int(year), "calendar_rows": len(group), "known": len(known), "unknown": len(group)-len(known),
            "known_fraction": len(known)/len(group), "known_source_age_over365": int(known.industry_source_age_days.gt(365).sum()),
            "retained_member_count_min": int(known.retained_member_count.min()) if len(known) else None,
            "retained_member_count_median": float(known.retained_member_count.median()) if len(known) else None,
            "retained_member_count_max": int(known.retained_member_count.max()) if len(known) else None})
    pd.DataFrame(years).to_csv(out / "results/全部年份结构可观察性_完整日历分母.csv", index=False, encoding="utf-8-sig")
    mismatch = complete.loc[complete.descriptive_fixed_state.eq("ETF_POSITIVE_FIXED_LEADERS_NONPOSITIVE")]
    mismatch.to_csv(out / "results/原周期ETF正但固定领先行业非正_全部1_5_20时点.csv", index=False, encoding="utf-8-sig")
    evaluation = daily.loc[daily.date.ge(pd.Timestamp("2015-01-05"))]
    known = evaluation.loc[evaluation.view_allowed]
    parent.write(out / "post_run_diagnosis.json", {"at": parent.original.now(), "purpose": "POST_RESULT_COMPLETE_DESCRIPTIVE_CONTEXT_NOT_FINANCIAL",
        "case_counts": case_counts, "yearly_structure_coverage": years, "old_stock_join_rows": len(joined),
        "all_context_contrasts": contrasts, "industry_price_discordant_original_cycle_rows": len(mismatch),
        "known_days_old_classification_over365": int(known.industry_source_age_days.gt(365).sum()),
        "known_retained_members_min": int(known.retained_member_count.min()),
        "known_retained_members_median": float(known.retained_member_count.median()),
        "known_retained_members_max": int(known.retained_member_count.max()),
        "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0, "financial_metrics": "NOT_COMPUTED",
        "raw_summary_sha256": parent.digest(out / "summary.json"), "old_stock_profile_sha256": parent.digest(older_path)})
    print("全部429固定时点已与原股票观察对齐；四案例双重可观察性、全部年份及全部状态保存，0新金融。", flush=True)


if __name__ == "__main__":
    main()
