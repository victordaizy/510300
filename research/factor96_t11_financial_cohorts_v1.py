"""T11披露日财务测量：过去两年同业参照，不读取市场价格。"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = {"announcement_id", "ts_code", "report_period", "available_date", "industry_l1_code",
            "industry_match_count", "is_financial", "member_at_available", "L02", "L04_change",
            "L02_known", "L04_known", "joint_known"}


def disclosure_frontier(measured):
    """同日只用最新报告；迟到的旧报告不能取代已公开的新期报告。"""
    assert REQUIRED.issubset(measured.columns)
    frame = measured.copy()
    for name in ["report_period", "available_date"]:
        frame[name] = pd.to_datetime(frame[name])
    assert frame.announcement_id.is_unique
    frame = frame.loc[frame.available_date.notna()].sort_values(
        ["ts_code", "available_date", "report_period", "announcement_id"], kind="stable")
    largest = frame.groupby("ts_code").report_period.cummax()
    return frame.loc[frame.report_period.eq(largest)].drop_duplicates(["ts_code", "available_date"], keep="last").reset_index(drop=True)


def measure_financial_cohorts(measured):
    """返回披露日篮子、公司测量和全部行业参考依赖，不产生入场或账户。"""
    front = disclosure_frontier(measured)
    front["report_age_days"] = (front.available_date - front.report_period).dt.days
    front["age_allowed"] = front.report_age_days.between(0, 200)
    front["quarter"] = front.report_period.dt.quarter
    member = front.loc[front.member_at_available.eq(True)].copy()
    # 行业按该历史公告当时的行业身份分组，不用今天的成员或行业回填。
    reference_ok = (member.industry_match_count.eq(1) & member.is_financial.eq(False)
                    & member.age_allowed & member.L04_known.eq(True) & np.isfinite(member.L04_change))
    pools = {key: table for key, table in member.loc[reference_ok].groupby(["industry_l1_code", "quarter"], sort=False)}
    company_rows, dependency_rows, daily_rows = [], [], []
    for day, cohort in member.groupby("available_date", sort=True):
        nonfinancial = cohort.loc[cohort.industry_match_count.eq(1) & cohort.is_financial.eq(False)]
        base = {"date": day, "all_member_reports": len(cohort), "nonfinancial_reports": len(nonfinancial),
                "industry_unknown_reports": int(cohort.industry_match_count.ne(1).sum()),
                "industry_count": int(nonfinancial.industry_l1_code.nunique()), "cohort_known": False,
                "L02_cohort": np.nan, "L04_industry_cohort": np.nan, "L04_raw_change_cohort": np.nan,
                "positive_financial_measurement": False,
                "announcement_ids": "|".join(sorted(nonfinancial.announcement_id.astype(str))),
                "measurement_only_no_entry_signal": True}
        if base["industry_unknown_reports"]:
            base["status"] = "NO_VIEW_INDUSTRY_IDENTITY"
        elif nonfinancial.empty:
            base["status"] = "NO_NONFINANCIAL_DISCLOSURE"
        else:
            base["status"] = "PENDING_COMPANY_MEASUREMENT"
        current_company = []
        for row in nonfinancial.itertuples(index=False):
            result = {"date": day, "announcement_id": str(row.announcement_id), "ts_code": row.ts_code,
                      "report_period": row.report_period, "industry_l1_code": row.industry_l1_code,
                      "L02": row.L02, "L04_change": row.L04_change, "industry_reference_count": 0,
                      "industry_reference_mean": np.nan, "industry_reference_sd": np.nan,
                      "L04_industry_z": np.nan, "known": False}
            if not (row.age_allowed and row.joint_known and np.isfinite([row.L02, row.L04_change]).all()):
                result["status"] = "NO_VIEW_LATEST_REPORT_FIELDS_OR_AGE"
            else:
                pool = pools.get((row.industry_l1_code, row.quarter))
                if pool is None:
                    refs = nonfinancial.iloc[0:0]
                else:
                    refs = pool.loc[pool.available_date.ge(day - pd.DateOffset(years=2))
                                    & pool.available_date.lt(day) & pool.report_period.lt(row.report_period)]
                values = refs.L04_change.to_numpy(float)
                count = len(values)
                average = float(values.mean()) if count else np.nan
                deviation = float(values.std(ddof=1)) if count >= 2 else np.nan
                result.update(industry_reference_count=count, industry_reference_mean=average, industry_reference_sd=deviation)
                for reference in refs.itertuples(index=False):
                    dependency_rows.append({"target_announcement_id": str(row.announcement_id), "target_date": day,
                        "target_report_period": row.report_period, "industry_l1_code": row.industry_l1_code,
                        "source_announcement_id": str(reference.announcement_id), "source_ts_code": reference.ts_code,
                        "source_available_date": reference.available_date, "source_report_period": reference.report_period,
                        "source_L04_change": reference.L04_change})
                if count < 2 or not np.isfinite(deviation) or deviation <= 0:
                    result["status"] = "NO_VIEW_INDUSTRY_REFERENCE_OR_ZERO_DISPERSION"
                else:
                    z = float((row.L04_change - average) / deviation)
                    result.update(L04_industry_z=z, known=bool(np.isfinite(z)),
                                  status="MEASURABLE" if np.isfinite(z) else "NO_VIEW_NONFINITE_INDUSTRY_SCORE")
            current_company.append(result)
            company_rows.append(result)
        if base["status"] == "PENDING_COMPANY_MEASUREMENT":
            # 任一新披露非金融公司缺项，就不把剩余公司冒充完整的新披露篮子。
            if not all(row["known"] for row in current_company):
                base["status"] = "NO_VIEW_INCOMPLETE_DISCLOSURE_COHORT"
            else:
                table = pd.DataFrame(current_company)
                industries = table.groupby("industry_l1_code")[["L02", "L04_industry_z", "L04_change"]].median()
                aggregates = industries.median()
                a, q, raw = (float(aggregates[name]) for name in ["L02", "L04_industry_z", "L04_change"])
                base.update(cohort_known=True, L02_cohort=a, L04_industry_cohort=q, L04_raw_change_cohort=raw,
                            positive_financial_measurement=bool(a > 1 and q >= 0 and raw >= 0), status="MEASURABLE_FINANCIAL_COHORT")
        daily_rows.append(base)
    daily = pd.DataFrame(daily_rows)
    companies = pd.DataFrame(company_rows)
    dependencies = pd.DataFrame(dependency_rows, columns=["target_announcement_id", "target_date", "target_report_period",
        "industry_l1_code", "source_announcement_id", "source_ts_code", "source_available_date", "source_report_period", "source_L04_change"])
    if not dependencies.empty:
        assert dependencies.source_available_date.lt(dependencies.target_date).all()
        assert dependencies.source_report_period.lt(dependencies.target_report_period).all()
    return daily, companies, dependencies
