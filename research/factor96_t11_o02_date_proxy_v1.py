"""T11名义日期代理的首轮反应测量，不计算未来持有收益或账户。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def measure_first_responses(cohorts: pd.DataFrame, market: pd.DataFrame):
    """财务表和价格分红表须先通过各自来源准入；未知不能按现金处理。"""
    required_financial = {"date", "cohort_known", "positive_financial_measurement",
                          "L02_cohort", "L04_industry_cohort", "L04_raw_change_cohort"}
    required_market = {"date", "raw_close", "cash_dividend", "price_source_admitted", "cash_dividend_known"}
    assert required_financial.issubset(cohorts.columns)
    assert required_market.issubset(market.columns)
    financial, prices = cohorts.copy(), market.copy()
    for frame in [financial, prices]:
        frame["date"] = pd.to_datetime(frame.date)
        assert frame.date.notna().all() and frame.date.is_unique
        assert frame.date.dt.normalize().equals(frame.date), "日期必须是单一交易日，不能混入盘中时刻"
    financial = financial.sort_values("date", kind="stable").reset_index(drop=True)
    prices = prices.sort_values("date", kind="stable").reset_index(drop=True)
    previous_close = prices.raw_close.shift(1)
    price_known = (prices.price_source_admitted.eq(True) & prices.price_source_admitted.shift(1).eq(True)
                   & prices.cash_dividend_known.eq(True) & np.isfinite(prices.raw_close)
                   & np.isfinite(previous_close) & np.isfinite(prices.cash_dividend)
                   & prices.raw_close.gt(0) & previous_close.gt(0) & prices.cash_dividend.ge(0))
    prices["first_response"] = ((prices.raw_close + prices.cash_dividend) / previous_close - 1).where(price_known)
    prices["earliest_entry_date"] = prices.date.shift(-1)
    prices["lag1_entry_date"] = prices.date.shift(-2)
    frame = financial.merge(prices[["date", "first_response", "earliest_entry_date", "lag1_entry_date"]],
                            on="date", how="left", validate="one_to_one")
    outputs, references = [], []
    for index, row in frame.iterrows():
        day = row["date"]
        values = np.array([row.L02_cohort, row.L04_industry_cohort, row.L04_raw_change_cohort], dtype=float)
        current_known = bool(row.cohort_known and np.isfinite(values).all())
        output = {"date": day, "cohort_known": current_known,
                  "positive_financial_measurement": bool(row.positive_financial_measurement) if current_known else None,
                  "L02_cohort": row.L02_cohort, "L04_industry_cohort": row.L04_industry_cohort,
                  "L04_raw_change_cohort": row.L04_raw_change_cohort,
                  "first_response": row.first_response, "negative_first_response": None,
                  "reference_count": 0, "reference_median": np.nan, "condition_known": False,
                  "underreaction_condition": None, "earliest_entry_date": row.earliest_entry_date,
                  "lag1_entry_date": row.lag1_entry_date, "measurement_only_no_entry_order": True,
                  "clock_basis": "NOMINAL_DATE_PROXY_NOT_HISTORICAL_FIRST_SEEN"}
        if not current_known:
            output["status"] = "NO_VIEW_CURRENT_FINANCIAL_COHORT"
        elif not np.isfinite(row.first_response):
            output["status"] = "NO_VIEW_FIRST_RESPONSE_OR_DIVIDEND"
        elif not row.positive_financial_measurement:
            output.update(status="FINANCIAL_CONDITION_NOT_MET", condition_known=True, underreaction_condition=False,
                          negative_first_response=bool(row.first_response < 0))
        else:
            past = frame.iloc[:index]
            reference = past.loc[past.date.ge(day - pd.DateOffset(years=2)) & past.date.lt(day)
                                 & past.cohort_known.eq(True) & past.positive_financial_measurement.eq(True)
                                 & np.isfinite(past.first_response)
                                 & np.isfinite(past.L02_cohort) & np.isfinite(past.L04_industry_cohort)
                                 & np.isfinite(past.L04_raw_change_cohort)]
            output["reference_count"] = len(reference)
            output["negative_first_response"] = bool(row.first_response < 0)
            for source in reference.itertuples(index=False):
                references.append({"target_date": day, "source_date": source.date,
                                   "source_first_response": source.first_response})
            if len(reference) < 2:
                output["status"] = "NO_VIEW_FEWER_THAN_TWO_PRIOR_ANALOGUES"
            else:
                median = float(reference.first_response.median())
                output.update(status="MEASURABLE_DATE_PROXY_CONDITION", reference_median=median,
                              condition_known=True, underreaction_condition=bool(row.first_response <= median))
        outputs.append(output)
    columns = ["date", "cohort_known", "positive_financial_measurement", "L02_cohort", "L04_industry_cohort",
               "L04_raw_change_cohort", "first_response", "negative_first_response", "reference_count",
               "reference_median", "condition_known", "underreaction_condition", "earliest_entry_date",
               "lag1_entry_date", "measurement_only_no_entry_order", "clock_basis", "status"]
    result = pd.DataFrame(outputs, columns=columns)
    for name in ["positive_financial_measurement", "negative_first_response", "underreaction_condition"]:
        result[name] = result[name].astype("boolean")
    dependency = pd.DataFrame(references, columns=["target_date", "source_date", "source_first_response"])
    if not dependency.empty:
        assert dependency.source_date.lt(dependency.target_date).all()
    return result, dependency
