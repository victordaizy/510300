"""CF、DR、RC 各自对象的历史测量有效性构造与汇总。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少必需列：{missing}")


def _latest_origin_on_or_before(
    origins: pd.DatetimeIndex, target: pd.Timestamp
) -> pd.Timestamp | pd.NaT:
    eligible = origins[origins <= target]
    return eligible[-1] if len(eligible) else pd.NaT


def _measurement_row(
    *,
    origin: pd.Timestamp,
    horizon: int,
    target_date: pd.Timestamp,
    module_id: str,
    object_id: str,
    value: float,
    unit: str,
    status: str,
    clock_status: str,
    target_snapshot_date: pd.Timestamp | pd.NaT = pd.NaT,
    weight_coverage: float = float("nan"),
    daily_observation_count: int | None = None,
    source_limitation: str,
) -> dict[str, Any]:
    return {
        "origin": origin,
        "horizon_market_days": int(horizon),
        "target_date": target_date,
        "module_id": module_id,
        "object_id": object_id,
        "measurement_value": value,
        "measurement_unit": unit,
        "measurement_status": status,
        "clock_status": clock_status,
        "target_snapshot_date": target_snapshot_date,
        "weight_coverage": weight_coverage,
        "daily_observation_count": daily_observation_count,
        "source_limitation": source_limitation,
        "direct_total_return_target": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }


def build_object_measurement_panel(
    *,
    ledger: pd.DataFrame,
    coverage: pd.DataFrame,
    cf_state: pd.DataFrame,
    present_value_state: pd.DataFrame,
    risk_capacity_daily: pd.DataFrame,
    total_return_index: pd.DataFrame,
    minimum_cf_weight_coverage: float,
    annualization_days: int,
) -> pd.DataFrame:
    """为八个预注册对象构造历史测量值，不拟合任何预测模型。"""

    _require_columns(
        ledger,
        [
            "origin",
            "horizon_market_days",
            "target_date",
            "scope",
            "earnings_growth_component_factor",
            "multiple_repricing_component_factor",
            "ledger_status",
            "financial_weight_coverage",
        ],
        "总回报成分账本",
    )
    _require_columns(
        coverage,
        ["origin", "horizon_market_days", "target_date"],
        "成分账本覆盖率",
    )
    _require_columns(
        cf_state,
        [
            "origin",
            "weighted_operating_cashflow_yoy",
            "operating_cashflow_weight_coverage",
            "cf_breadth",
            "operating_profit_weight_coverage",
            "cf_data_status",
        ],
        "CF 状态",
    )
    _require_columns(
        present_value_state,
        [
            "origin",
            "equity_risk_premium_state",
            "present_value_data_status",
        ],
        "DR 状态",
    )
    _require_columns(
        risk_capacity_daily,
        [
            "date",
            "member_average_pairwise_correlation_60d",
            "etf_amihud_20d",
        ],
        "RC 日状态",
    )
    _require_columns(total_return_index, ["date", "close"], "H00300 总回报指数")

    schedule = coverage[["origin", "horizon_market_days", "target_date"]].copy()
    for column in ("origin", "target_date"):
        schedule[column] = pd.to_datetime(schedule[column], errors="coerce").dt.normalize()
    financial = ledger.loc[
        ledger["scope"].eq("ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE")
    ].copy()
    for column in ("origin", "target_date"):
        financial[column] = pd.to_datetime(
            financial[column], errors="coerce"
        ).dt.normalize()
    financial = financial.set_index(["origin", "horizon_market_days"])

    cf = cf_state.copy()
    cf["origin"] = pd.to_datetime(cf["origin"], errors="coerce").dt.normalize()
    cf = cf.drop_duplicates("origin", keep="last").set_index("origin").sort_index()
    cf_origins = pd.DatetimeIndex(cf.index)
    pv = present_value_state.copy()
    pv["origin"] = pd.to_datetime(pv["origin"], errors="coerce").dt.normalize()
    pv = pv.drop_duplicates("origin", keep="last").set_index("origin").sort_index()
    pv_origins = pd.DatetimeIndex(pv.index)
    risk = risk_capacity_daily.copy()
    risk["date"] = pd.to_datetime(risk["date"], errors="coerce").dt.normalize()
    risk = risk.drop_duplicates("date", keep="last").set_index("date").sort_index()
    total = total_return_index[["date", "close"]].copy()
    total["date"] = pd.to_datetime(total["date"], errors="coerce").dt.normalize()
    total["close"] = pd.to_numeric(total["close"], errors="coerce")
    total = total.dropna().drop_duplicates("date", keep="last").set_index("date").sort_index()

    rows: list[dict[str, Any]] = []
    for schedule_row in schedule.itertuples(index=False):
        origin = pd.Timestamp(schedule_row.origin)
        target = pd.Timestamp(schedule_row.target_date)
        horizon = int(schedule_row.horizon_market_days)
        financial_row = financial.loc[(origin, horizon)]
        earnings_factor = float(financial_row["earnings_growth_component_factor"])
        multiple_factor = float(financial_row["multiple_repricing_component_factor"])
        financial_coverage = float(financial_row["financial_weight_coverage"])
        financial_valid = np.isfinite(earnings_factor) and np.isfinite(multiple_factor)
        financial_status = (
            "PARTIAL_PIT_VALUE_REVISION_AND_PROXY_WEIGHT_LIMITATION"
            if financial_valid
            else "NO_VIEW_FIXED_FINANCIAL_COHORT_MEASUREMENT"
        )
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="CF",
                object_id="EARNINGS_GROWTH",
                value=earnings_factor - 1.0 if financial_valid else float("nan"),
                unit="FACTOR_CHANGE",
                status=financial_status,
                clock_status="PIT_NOTICE_DATE_CLOCK_WITH_VALUE_REVISION_LIMITATION",
                target_snapshot_date=target,
                weight_coverage=financial_coverage,
                source_limitation="SECONDARY_FINANCIAL_VALUES_NOT_IMMUTABLE_FIRST_PUBLICATION_ARCHIVE",
            )
        )
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="DR",
                object_id="VALUATION_MULTIPLE_CHANGE",
                value=multiple_factor - 1.0 if financial_valid else float("nan"),
                unit="FACTOR_CHANGE",
                status=financial_status,
                clock_status="PIT_NOTICE_DATE_CLOCK_WITH_VALUE_REVISION_LIMITATION",
                target_snapshot_date=target,
                weight_coverage=financial_coverage,
                source_limitation="SECONDARY_FINANCIAL_VALUES_AND_PROXY_WEIGHTS_LIMIT_FULL_ADMISSION",
            )
        )

        cf_target_origin = _latest_origin_on_or_before(cf_origins, target)
        if pd.isna(cf_target_origin):
            cf_target = None
        else:
            cf_target = cf.loc[cf_target_origin]
        if cf_target is not None:
            ocf_value = float(cf_target["weighted_operating_cashflow_yoy"])
            ocf_coverage = float(cf_target["operating_cashflow_weight_coverage"])
            breadth_value = float(cf_target["cf_breadth"])
            breadth_coverage = float(
                min(
                    cf_target["operating_cashflow_weight_coverage"],
                    cf_target["operating_profit_weight_coverage"],
                )
            )
            cf_clock_pass = str(cf_target["cf_data_status"]).startswith("PASS")
        else:
            ocf_value = float("nan")
            ocf_coverage = float("nan")
            breadth_value = float("nan")
            breadth_coverage = float("nan")
            cf_clock_pass = False
        ocf_valid = (
            np.isfinite(ocf_value)
            and np.isfinite(ocf_coverage)
            and ocf_coverage >= minimum_cf_weight_coverage
            and cf_clock_pass
        )
        breadth_valid = (
            np.isfinite(breadth_value)
            and np.isfinite(breadth_coverage)
            and breadth_coverage >= minimum_cf_weight_coverage
            and cf_clock_pass
        )
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="CF",
                object_id="OPERATING_CASHFLOW_GROWTH",
                value=ocf_value if ocf_valid else float("nan"),
                unit="TTM_YEAR_OVER_YEAR_RATE",
                status="PARTIAL_PIT_VALUE_REVISION_AND_PROXY_WEIGHT_LIMITATION" if ocf_valid else "NO_VIEW_CF_TARGET_COVERAGE_OR_CLOCK",
                clock_status="LATEST_REGISTERED_ORIGIN_ON_OR_BEFORE_TARGET",
                target_snapshot_date=cf_target_origin,
                weight_coverage=ocf_coverage,
                source_limitation="SECONDARY_FINANCIAL_VALUES_NOT_IMMUTABLE_FIRST_PUBLICATION_ARCHIVE",
            )
        )
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="CF",
                object_id="EARNINGS_CASHFLOW_BREADTH",
                value=breadth_value if breadth_valid else float("nan"),
                unit="WEIGHT_SHARE",
                status="PARTIAL_PIT_VALUE_REVISION_AND_PROXY_WEIGHT_LIMITATION" if breadth_valid else "NO_VIEW_CF_BREADTH_COVERAGE_OR_CLOCK",
                clock_status="LATEST_REGISTERED_ORIGIN_ON_OR_BEFORE_TARGET",
                target_snapshot_date=cf_target_origin,
                weight_coverage=breadth_coverage,
                source_limitation="PROXY_WEIGHT_AND_SECONDARY_FINANCIAL_VALUE_LIMITATION",
            )
        )

        pv_target_origin = _latest_origin_on_or_before(pv_origins, target)
        pv_origin_valid = origin in pv.index
        pv_target_valid = pd.notna(pv_target_origin)
        if pv_origin_valid and pv_target_valid:
            pv_origin_row = pv.loc[origin]
            pv_target_row = pv.loc[pv_target_origin]
            erp_origin = float(pv_origin_row["equity_risk_premium_state"])
            erp_target = float(pv_target_row["equity_risk_premium_state"])
            pv_status_valid = str(pv_origin_row["present_value_data_status"]).startswith(
                "PASS"
            ) and str(pv_target_row["present_value_data_status"]).startswith("PASS")
        else:
            erp_origin = float("nan")
            erp_target = float("nan")
            pv_status_valid = False
        erp_valid = np.isfinite(erp_origin) and np.isfinite(erp_target) and pv_status_valid
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="DR",
                object_id="EQUITY_RISK_PREMIUM_GAP_CHANGE",
                value=erp_target - erp_origin if erp_valid else float("nan"),
                unit="ABSOLUTE_GAP_CHANGE",
                status="PARTIAL_PIT_VALUE_REVISION_AND_PROXY_WEIGHT_LIMITATION" if erp_valid else "NO_VIEW_ERP_GAP_STATE_OR_CLOCK",
                clock_status="LATEST_REGISTERED_ORIGIN_ON_OR_BEFORE_TARGET",
                target_snapshot_date=pv_target_origin,
                source_limitation="EARNINGS_YIELD_USES_SECONDARY_FINANCIAL_VALUES_AND_PROXY_WEIGHTS",
            )
        )

        total_window = total.loc[(total.index >= origin) & (total.index <= target), "close"]
        returns = total_window.pct_change(fill_method=None).dropna()
        exact_count = len(returns) == horizon
        realized_volatility = (
            float(returns.std(ddof=1) * np.sqrt(annualization_days))
            if exact_count and len(returns) >= 2
            else float("nan")
        )
        downside_semivariance = (
            float(np.mean(np.minimum(returns.to_numpy(dtype=float), 0.0) ** 2) * annualization_days)
            if exact_count and len(returns) >= 1
            else float("nan")
        )
        risk_status = (
            "PASS_EXACT_FORWARD_DAILY_MEASUREMENT"
            if exact_count
            else "NO_VIEW_FORWARD_DAILY_COUNT_NOT_EXACT"
        )
        for object_id, value, unit in (
            ("REALIZED_VOLATILITY", realized_volatility, "ANNUALIZED_STANDARD_DEVIATION"),
            ("DOWNSIDE_SEMIVARIANCE", downside_semivariance, "ANNUALIZED_SEMIVARIANCE"),
        ):
            rows.append(
                _measurement_row(
                    origin=origin,
                    horizon=horizon,
                    target_date=target,
                    module_id="RC",
                    object_id=object_id,
                    value=value,
                    unit=unit,
                    status=risk_status,
                    clock_status="EXACT_H00300_MARKET_DAY_WINDOW",
                    target_snapshot_date=target,
                    daily_observation_count=int(len(returns)),
                    source_limitation="REALIZED_OUTCOME_MEASUREMENT_NOT_A_FORECAST",
                )
            )

        correlation_value = float("nan")
        correlation_status = "NO_VIEW_TARGET_MEMBER_CORRELATION_MISSING"
        if target in risk.index:
            correlation_value = float(
                risk.loc[target, "member_average_pairwise_correlation_60d"]
            )
            if np.isfinite(correlation_value):
                correlation_status = "PASS_TARGET_MEMBER_CORRELATION_MEASUREMENT"
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="RC",
                object_id="MEMBER_CORRELATION",
                value=correlation_value,
                unit="AVERAGE_PAIRWISE_CORRELATION_60D",
                status=correlation_status,
                clock_status="EXACT_TARGET_MARKET_CLOSE",
                target_snapshot_date=target,
                source_limitation="POINT_IN_TIME_MEMBERSHIP_WITHOUT_OFFICIAL_WEIGHT_USE",
            )
        )

        liquidity_value = float("nan")
        liquidity_status = "NO_VIEW_ETF_AMIHUD_WINDOW_MISSING"
        risk_window = risk.loc[(risk.index >= origin) & (risk.index <= target)]
        if origin in risk.index and not risk_window.empty:
            origin_amihud = float(risk.loc[origin, "etf_amihud_20d"])
            forward_amihud = pd.to_numeric(
                risk_window["etf_amihud_20d"], errors="coerce"
            ).dropna()
            if origin_amihud > 0.0 and not forward_amihud.empty and (forward_amihud > 0.0).all():
                liquidity_value = float(np.log(forward_amihud / origin_amihud).max())
                liquidity_status = "PASS_FORWARD_ETF_AMIHUD_SHOCK_MEASUREMENT"
        rows.append(
            _measurement_row(
                origin=origin,
                horizon=horizon,
                target_date=target,
                module_id="RC",
                object_id="LIQUIDITY_SHOCK",
                value=liquidity_value,
                unit="MAX_LOG_AMIHUD_RATIO_VS_ORIGIN",
                status=liquidity_status,
                clock_status="DAILY_MARKET_CLOSE_FORWARD_WINDOW",
                target_snapshot_date=target,
                daily_observation_count=int(max(len(risk_window) - 1, 0)),
                source_limitation="ETF_AMIHUD_PRICE_IMPACT_PROXY_ONLY",
            )
        )

    result = pd.DataFrame(rows).sort_values(
        ["module_id", "object_id", "horizon_market_days", "origin"],
        kind="stable",
    ).reset_index(drop=True)
    expected_objects = {
        "CF": {"EARNINGS_GROWTH", "OPERATING_CASHFLOW_GROWTH", "EARNINGS_CASHFLOW_BREADTH"},
        "DR": {"VALUATION_MULTIPLE_CHANGE", "EQUITY_RISK_PREMIUM_GAP_CHANGE"},
        "RC": {"REALIZED_VOLATILITY", "DOWNSIDE_SEMIVARIANCE", "MEMBER_CORRELATION", "LIQUIDITY_SHOCK"},
    }
    for module_id, object_ids in expected_objects.items():
        actual = set(result.loc[result["module_id"].eq(module_id), "object_id"])
        if actual != object_ids:
            raise ValueError(f"{module_id} 对象集合不完整：{actual}")
    return result


def summarize_measurement_validity(
    panel: pd.DataFrame,
    *,
    minimum_observed_share: float,
    minimum_pass_share: float,
    minimum_distinct_years: int,
) -> pd.DataFrame:
    """按对象和 horizon 汇总 PASS/PARTIAL/NO_VIEW 覆盖。"""

    rows: list[dict[str, Any]] = []
    for (module_id, object_id, horizon), group in panel.groupby(
        ["module_id", "object_id", "horizon_market_days"], sort=True
    ):
        observed = group["measurement_value"].notna()
        passed = group["measurement_status"].str.startswith("PASS") & observed
        partial = group["measurement_status"].str.startswith("PARTIAL") & observed
        count = int(len(group))
        observed_count = int(observed.sum())
        pass_count = int(passed.sum())
        partial_count = int(partial.sum())
        no_view_count = count - observed_count
        observed_share = observed_count / count if count else 0.0
        pass_share = pass_count / count if count else 0.0
        usable_share = (pass_count + partial_count) / count if count else 0.0
        years = int(group.loc[observed, "origin"].dt.year.nunique())
        if observed_share < minimum_observed_share or years < minimum_distinct_years:
            status = "NO_VIEW_MEASUREMENT_COVERAGE_GATE"
        elif pass_share >= minimum_pass_share:
            status = "PASS_OWN_OBJECT_MEASUREMENT_VALIDITY"
        elif usable_share >= minimum_pass_share:
            status = "PARTIAL_OWN_OBJECT_MEASUREMENT_VALIDITY"
        else:
            status = "NO_VIEW_MEASUREMENT_RELIABILITY_GATE"
        rows.append(
            {
                "module_id": module_id,
                "object_id": object_id,
                "horizon_market_days": int(horizon),
                "origin_count": count,
                "observed_count": observed_count,
                "pass_count": pass_count,
                "partial_count": partial_count,
                "no_view_count": no_view_count,
                "observed_share": observed_share,
                "pass_share": pass_share,
                "usable_share": usable_share,
                "distinct_calendar_years": years,
                "first_observed_origin": group.loc[observed, "origin"].min(),
                "last_observed_origin": group.loc[observed, "origin"].max(),
                "measurement_validity_status": status,
                "return_prediction_allowed": False,
                "portfolio_evaluation_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["module_id", "object_id", "horizon_market_days"], kind="stable"
    ).reset_index(drop=True)
