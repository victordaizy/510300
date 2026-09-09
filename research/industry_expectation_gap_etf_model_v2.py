"""行业预期差—ETF模型V2：互斥经济引擎、重叠主题链和当前数据情境。

本模块不拟合模型、不读取未来收益、不生成仓位或订单。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class SectorTaxonomyError(ValueError):
    """板块分类或输入数据不满足冻结契约。"""


FUNDAMENTAL_SCORE = {
    "STRONG_IMPROVEMENT": 2.0,
    "IMPROVEMENT": 1.0,
    "MIXED": 0.0,
    "DETERIORATION": -1.0,
    "STRONG_DETERIORATION": -2.0,
}

EXPECTATION_SCORE = {
    "POSITIVE": 1.0,
    "BALANCED": 0.0,
    "NEGATIVE": -1.0,
}


def validate_taxonomy(
    industry_names: Sequence[str], taxonomy: Mapping[str, Any]
) -> dict[str, str]:
    """验证经济引擎恰好覆盖一次；主题只允许引用已知行业。"""

    expected = [str(value) for value in industry_names]
    if len(expected) != len(set(expected)):
        raise SectorTaxonomyError("行业底图本身存在重复行业")
    bucket_mapping: dict[str, str] = {}
    bucket_ids: list[str] = []
    for bucket in taxonomy["economic_buckets"]:
        bucket_id = str(bucket["id"])
        if bucket_id in bucket_ids:
            raise SectorTaxonomyError(f"经济引擎ID重复：{bucket_id}")
        bucket_ids.append(bucket_id)
        for industry in bucket["industries"]:
            industry = str(industry)
            if industry in bucket_mapping:
                raise SectorTaxonomyError(
                    f"行业{industry}同时进入{bucket_mapping[industry]}和{bucket_id}"
                )
            bucket_mapping[industry] = bucket_id
    missing = sorted(set(expected).difference(bucket_mapping))
    if missing:
        raise SectorTaxonomyError(f"经济引擎未覆盖输入行业：missing={missing}")

    theme_ids: list[str] = []
    for theme in taxonomy["themes"]:
        theme_id = str(theme["id"])
        if theme_id in theme_ids:
            raise SectorTaxonomyError(f"主题ID重复：{theme_id}")
        theme_ids.append(theme_id)
        core = [str(value) for value in theme.get("core", [])]
        adjacent = [str(value) for value in theme.get("adjacent", [])]
        repeated = sorted(set(core).intersection(adjacent))
        unknown = sorted((set(core) | set(adjacent)).difference(bucket_mapping))
        if repeated or unknown:
            raise SectorTaxonomyError(
                f"主题{theme_id}定义非法：core_adjacent_overlap={repeated}, unknown={unknown}"
            )
    return bucket_mapping


def validate_official_cics_snapshot(
    snapshot: pd.DataFrame, tolerance_percent: float
) -> pd.DataFrame:
    """验证中证CICS一级行业当前截面互斥且权重和为100%。"""

    required = {
        "date",
        "industry_level",
        "industry_name_cn",
        "industry_name_en",
        "weight_pct",
        "source",
    }
    missing = sorted(required.difference(snapshot.columns))
    if missing:
        raise SectorTaxonomyError(f"CICS快照缺少字段：{missing}")
    data = snapshot.loc[snapshot["industry_level"].eq(1)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["weight_pct"] = pd.to_numeric(data["weight_pct"], errors="coerce")
    if data.empty or data["date"].isna().any() or data["weight_pct"].isna().any():
        raise SectorTaxonomyError("CICS一级行业快照为空或存在非法值")
    if data["industry_name_en"].duplicated().any():
        raise SectorTaxonomyError("CICS一级行业名称重复")
    weight_sum = float(data["weight_pct"].sum())
    if abs(weight_sum - 100.0) > tolerance_percent:
        raise SectorTaxonomyError(f"CICS一级行业权重和异常：{weight_sum:.6f}%")
    return data.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def _compound_return(values: pd.Series, window: int) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < window:
        return None
    return float((1.0 + clean.iloc[-window:]).prod() - 1.0)


def _last_rolling_percentile(values: pd.Series, window: int) -> float | None:
    clean = pd.to_numeric(values, errors="coerce")
    rolling = (1.0 + clean).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
    history = rolling.dropna()
    if history.empty:
        return None
    return float(history.le(history.iloc[-1]).mean())


def build_economic_bucket_daily(
    industry_daily: pd.DataFrame,
    taxonomy: Mapping[str, Any],
) -> pd.DataFrame:
    """按每日行业权重把互斥行业聚合为经济引擎日收益。"""

    required = {
        "date",
        "industry_l1",
        "industry_weight",
        "return_coverage_weight",
        "industry_return_1d",
        "weighted_return_contribution_1d",
    }
    missing = sorted(required.difference(industry_daily.columns))
    if missing:
        raise SectorTaxonomyError(f"行业收益缺少字段：{missing}")
    names = sorted(industry_daily["industry_l1"].dropna().astype(str).unique())
    mapping = validate_taxonomy(names, taxonomy)
    bucket_names = {
        str(item["id"]): str(item["name_cn"])
        for item in taxonomy["economic_buckets"]
    }
    data = industry_daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["economic_bucket_id"] = data["industry_l1"].astype(str).map(mapping)
    data["economic_bucket_name_cn"] = data["economic_bucket_id"].map(bucket_names)
    if data[["date", "economic_bucket_id"]].isna().any().any():
        raise SectorTaxonomyError("经济引擎日收益存在空日期或未映射行业")
    rows: list[dict[str, Any]] = []
    for (date, bucket_id), group in data.groupby(
        ["date", "economic_bucket_id"], sort=True
    ):
        contribution = pd.to_numeric(
            group["weighted_return_contribution_1d"], errors="coerce"
        ).sum(min_count=1)
        coverage = float(
            pd.to_numeric(group["return_coverage_weight"], errors="coerce").sum()
        )
        weight = float(pd.to_numeric(group["industry_weight"], errors="coerce").sum())
        rows.append(
            {
                "date": pd.Timestamp(date),
                "economic_bucket_id": bucket_id,
                "economic_bucket_name_cn": bucket_names[bucket_id],
                "bucket_weight": weight,
                "return_coverage_weight": coverage,
                "bucket_return_1d": (
                    float(contribution / coverage)
                    if coverage > 0 and pd.notna(contribution)
                    else np.nan
                ),
                "weighted_return_contribution_1d": (
                    float(contribution) if pd.notna(contribution) else np.nan
                ),
                "industry_count": int(group["industry_l1"].nunique()),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["date", "economic_bucket_id"]
    ).reset_index(drop=True)
    if result[["date", "economic_bucket_id"]].duplicated().any():
        raise SectorTaxonomyError("经济引擎日收益存在重复日期大类")
    return result


def _weighted_label_summary(
    rows: pd.DataFrame, label_column: str, score_mapping: Mapping[str, float]
) -> dict[str, Any]:
    labels = rows[label_column].astype(str)
    scores = labels.map(score_mapping)
    observed = scores.notna()
    observed_weight = float(rows.loc[observed, "sector_weight"].sum())
    weighted_score = (
        float((rows.loc[observed, "sector_weight"] * scores.loc[observed]).sum())
        if observed_weight > 0
        else None
    )
    normalized_score = weighted_score / observed_weight if observed_weight > 0 else None
    return {
        "observed_weight": observed_weight,
        "weighted_score": weighted_score,
        "normalized_score": normalized_score,
        "label_weight": {
            label: float(rows.loc[labels.eq(label), "sector_weight"].sum())
            for label in sorted(labels.unique())
        },
    }


def build_industry_snapshot(
    forecast_rows: Sequence[Mapping[str, Any]],
    industry_daily: pd.DataFrame,
    windows: Sequence[int],
) -> pd.DataFrame:
    """把冻结的行业判断与更新后的行业价格位置合并。"""

    forecast = pd.DataFrame(list(forecast_rows)).copy()
    required = {
        "industry_l1",
        "sector_weight",
        "fundamental_60d",
        "fundamental_120d",
        "expectation_gap",
        "confidence",
    }
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise SectorTaxonomyError(f"冻结行业台账缺少字段：{missing}")
    daily = industry_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for item in forecast.to_dict(orient="records"):
        name = str(item["industry_l1"])
        history = daily.loc[daily["industry_l1"].astype(str).eq(name)].sort_values("date")
        if history.empty:
            raise SectorTaxonomyError(f"行业{name}没有历史收益")
        row = dict(item)
        row["price_data_as_of_date"] = history["date"].max().date().isoformat()
        for window in windows:
            row[f"return_{window}d"] = _compound_return(
                history["industry_return_1d"], int(window)
            )
        if 60 in windows:
            row["return_60d_history_percentile"] = _last_rolling_percentile(
                history["industry_return_1d"], 60
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sector_weight", ascending=False).reset_index(drop=True)


def build_bucket_snapshot(
    industry_snapshot: pd.DataFrame,
    bucket_daily: pd.DataFrame,
    taxonomy: Mapping[str, Any],
    windows: Sequence[int],
) -> list[dict[str, Any]]:
    """生成互斥经济引擎当前权重、价格、基本面和预期差摘要。"""

    mapping = validate_taxonomy(
        industry_snapshot["industry_l1"].astype(str).tolist(), taxonomy
    )
    data = industry_snapshot.copy()
    data["economic_bucket_id"] = data["industry_l1"].astype(str).map(mapping)
    definitions = {str(item["id"]): item for item in taxonomy["economic_buckets"]}
    rows: list[dict[str, Any]] = []
    for bucket_id, group in data.groupby("economic_bucket_id", sort=False):
        definition = definitions[bucket_id]
        history = bucket_daily.loc[
            bucket_daily["economic_bucket_id"].eq(bucket_id)
        ].sort_values("date")
        record: dict[str, Any] = {
            "bucket_id": bucket_id,
            "bucket_name_cn": definition["name_cn"],
            "industries": list(definition["industries"]),
            "index_weight": float(group["sector_weight"].sum()),
            "industry_count": int(group["industry_l1"].nunique()),
            "fundamental_60d": _weighted_label_summary(
                group, "fundamental_60d", FUNDAMENTAL_SCORE
            ),
            "fundamental_120d": _weighted_label_summary(
                group, "fundamental_120d", FUNDAMENTAL_SCORE
            ),
            "expectation_gap": _weighted_label_summary(
                group, "expectation_gap", EXPECTATION_SCORE
            ),
        }
        for window in windows:
            record[f"return_{window}d"] = _compound_return(
                history["bucket_return_1d"], int(window)
            )
        record["return_60d_history_percentile"] = _last_rolling_percentile(
            history["bucket_return_1d"], 60
        )
        rows.append(record)
    return sorted(rows, key=lambda item: item["index_weight"], reverse=True)


def build_theme_snapshot(
    industry_snapshot: pd.DataFrame,
    taxonomy: Mapping[str, Any],
    industry_flow: Mapping[str, Mapping[str, float | None]] | None = None,
) -> list[dict[str, Any]]:
    """生成允许重叠的主题覆盖；核心和邻接权重永不合成有效暴露。"""

    validate_taxonomy(industry_snapshot["industry_l1"].astype(str).tolist(), taxonomy)
    indexed = industry_snapshot.set_index("industry_l1", drop=False)
    flow = industry_flow or {}
    rows: list[dict[str, Any]] = []
    for theme in taxonomy["themes"]:
        core = [str(value) for value in theme.get("core", [])]
        adjacent = [str(value) for value in theme.get("adjacent", [])]
        core_rows = indexed.loc[core]
        adjacent_rows = indexed.loc[adjacent] if adjacent else indexed.iloc[0:0]

        def weighted_metric(frame: pd.DataFrame, column: str) -> float | None:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna()
            weights = frame.loc[valid, "sector_weight"].astype(float)
            return (
                float(np.average(values.loc[valid], weights=weights))
                if valid.any() and weights.sum() > 0
                else None
            )

        theme_flow: dict[str, float | None] = {}
        for window_name in ("flow_intensity_1d", "flow_intensity_5d"):
            numerator = 0.0
            denominator = 0.0
            for name in core:
                value = flow.get(name, {}).get(window_name)
                amount = flow.get(name, {}).get(window_name.replace("intensity", "amount_cny"))
                if value is not None and amount is not None and float(amount) > 0:
                    numerator += float(value) * float(amount)
                    denominator += float(amount)
            theme_flow[window_name] = numerator / denominator if denominator > 0 else None

        rows.append(
            {
                "theme_id": str(theme["id"]),
                "theme_name_cn": str(theme["name_cn"]),
                "core_industries": core,
                "adjacent_industries": adjacent,
                "core_index_weight": float(core_rows["sector_weight"].sum()),
                "adjacent_index_weight": float(adjacent_rows["sector_weight"].sum()),
                "core_return_20d_weighted_snapshot": weighted_metric(core_rows, "return_20d"),
                "core_return_60d_weighted_snapshot": weighted_metric(core_rows, "return_60d"),
                "core_expectation_gap": _weighted_label_summary(
                    core_rows, "expectation_gap", EXPECTATION_SCORE
                ),
                **theme_flow,
                "cross_theme_sum_allowed": False,
                "effective_weight": None,
            }
        )
    return rows


def classify_bucket_forward_priorities(
    buckets: list[dict[str, Any]], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """把基本面、预期差和价格阶段组合成定性研究优先级。

    该函数不使用未来收益，也不输出交易动作。
    """

    minimum_coverage = float(rules["minimum_expectation_coverage_of_bucket"])
    early_maximum = float(rules["early_recovery_maximum_60d_history_percentile"])
    extended_minimum = float(rules["extended_minimum_60d_history_percentile"])
    records: list[dict[str, Any]] = []
    for bucket in buckets:
        weight = float(bucket["index_weight"])
        gap = bucket["expectation_gap"]
        label_weight = gap["label_weight"]
        positive_weight = float(label_weight.get("POSITIVE", 0.0))
        negative_weight = float(label_weight.get("NEGATIVE", 0.0))
        observed_weight = float(gap["observed_weight"])
        coverage_ratio = observed_weight / weight if weight > 0 else 0.0
        fundamental_score = bucket["fundamental_60d"]["normalized_score"]
        percentile = bucket.get("return_60d_history_percentile")
        return_5d = bucket.get("return_5d")

        if percentile is None:
            price_phase = "PRICE_PHASE_UNOBSERVED"
        elif float(percentile) >= extended_minimum:
            price_phase = "LATE_OR_EXTENDED"
        elif float(percentile) <= early_maximum and return_5d is not None and float(return_5d) > 0:
            price_phase = "EARLY_RECOVERY_FROM_WEAK_BASE"
        elif float(percentile) <= early_maximum:
            price_phase = "WEAK_BASE_NOT_TURNED"
        elif bucket.get("return_20d") is not None and float(bucket["return_20d"]) > 0:
            price_phase = "MIDDLE_TREND"
        else:
            price_phase = "MIXED_OR_SOFT"

        if coverage_ratio < minimum_coverage:
            gap_state = "INSUFFICIENT_EXPECTATION_COVERAGE"
        elif positive_weight > negative_weight:
            gap_state = "NET_POSITIVE_GAP"
        elif negative_weight > positive_weight:
            gap_state = "NET_NEGATIVE_GAP"
        else:
            gap_state = "BALANCED_OR_NO_GAP"

        positive_fundamental = fundamental_score is not None and float(fundamental_score) > 0
        if gap_state == "NET_POSITIVE_GAP" and positive_fundamental:
            if price_phase == "EARLY_RECOVERY_FROM_WEAK_BASE":
                priority = "PRIORITY_FISH_MIDDLE_CANDIDATE"
            elif price_phase == "WEAK_BASE_NOT_TURNED":
                priority = "WATCH_FOR_TURN_NOT_ENTER"
            elif price_phase == "LATE_OR_EXTENDED":
                priority = "POSITIVE_BUT_FISH_TAIL_RISK"
            else:
                priority = "POSITIVE_GAP_MONITOR"
        elif gap_state == "NET_NEGATIVE_GAP":
            priority = "DETERIORATION_RISK_OR_COUNTERTREND"
        elif gap_state == "INSUFFICIENT_EXPECTATION_COVERAGE":
            priority = "NO_VIEW_EXPECTATION_DATA_GAP"
        elif positive_fundamental and price_phase == "EARLY_RECOVERY_FROM_WEAK_BASE":
            priority = "WATCH_FOR_EXPECTATION_UPGRADE"
        else:
            priority = "OBSERVE_NO_EDGE"
        records.append(
            {
                "bucket_id": bucket["bucket_id"],
                "bucket_name_cn": bucket["bucket_name_cn"],
                "research_priority": priority,
                "gap_state": gap_state,
                "price_phase": price_phase,
                "expectation_coverage_ratio": coverage_ratio,
                "positive_gap_weight": positive_weight,
                "negative_gap_weight": negative_weight,
                "fundamental_60d_normalized_score": fundamental_score,
                "not_a_trade_signal": True,
            }
        )
    rank = {
        "PRIORITY_FISH_MIDDLE_CANDIDATE": 0,
        "WATCH_FOR_TURN_NOT_ENTER": 1,
        "POSITIVE_GAP_MONITOR": 2,
        "WATCH_FOR_EXPECTATION_UPGRADE": 3,
        "POSITIVE_BUT_FISH_TAIL_RISK": 4,
        "DETERIORATION_RISK_OR_COUNTERTREND": 5,
        "NO_VIEW_EXPECTATION_DATA_GAP": 6,
        "OBSERVE_NO_EDGE": 7,
    }
    return sorted(records, key=lambda item: (rank[item["research_priority"]], -next(
        float(bucket["index_weight"]) for bucket in buckets if bucket["bucket_id"] == item["bucket_id"]
    )))


def build_industry_flow_context(
    moneyflow: pd.DataFrame,
    traded_amount: pd.DataFrame,
    industry_mapping: Mapping[str, str],
    as_of_date: pd.Timestamp,
    windows: Sequence[int],
) -> dict[str, dict[str, float | None]]:
    """按成交额归一化个股订单分类资金流，避免把行业规模误当强度。"""

    flow_required = {"date", "ts_code", "net_mf_amount"}
    amount_required = {"date", "con_code", "amount"}
    if missing := sorted(flow_required.difference(moneyflow.columns)):
        raise SectorTaxonomyError(f"资金流缺少字段：{missing}")
    if missing := sorted(amount_required.difference(traded_amount.columns)):
        raise SectorTaxonomyError(f"成交额缺少字段：{missing}")
    flow = moneyflow.copy()
    flow["date"] = pd.to_datetime(flow["date"], errors="coerce")
    flow["net_mf_amount"] = pd.to_numeric(flow["net_mf_amount"], errors="coerce")
    amount = traded_amount.copy()
    amount["date"] = pd.to_datetime(amount["date"], errors="coerce")
    amount["amount"] = pd.to_numeric(amount["amount"], errors="coerce")
    data = flow.merge(
        amount[["date", "con_code", "amount"]],
        left_on=["date", "ts_code"],
        right_on=["date", "con_code"],
        how="inner",
        validate="one_to_one",
    )
    data = data.loc[data["date"].le(pd.Timestamp(as_of_date))].copy()
    data["industry_l1"] = data["ts_code"].astype(str).map(industry_mapping)
    data = data.dropna(subset=["industry_l1", "amount", "net_mf_amount"])
    data["net_mf_cny"] = data["net_mf_amount"] * 10_000.0
    dates = sorted(data["date"].unique())
    result: dict[str, dict[str, float | None]] = {}
    for industry, group in data.groupby("industry_l1"):
        record: dict[str, float | None] = {}
        for window in windows:
            selected_dates = set(dates[-int(window) :])
            selected = group.loc[group["date"].isin(selected_dates)]
            net = float(selected["net_mf_cny"].sum())
            turnover = float(selected["amount"].sum())
            record[f"flow_net_cny_{window}d"] = net
            record[f"flow_amount_cny_{window}d"] = turnover
            record[f"flow_intensity_{window}d"] = net / turnover if turnover > 0 else None
        result[str(industry)] = record
    return result


def summarize_etf_flow(
    fund_share: pd.DataFrame,
    margin: pd.DataFrame,
    etf_daily: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    """汇总ETF份额、融资融券和成交额，保留各自数据日。"""

    cutoff = pd.Timestamp(as_of_date)
    shares = fund_share.copy()
    shares["date"] = pd.to_datetime(shares["date"], errors="coerce")
    shares["fund_shares"] = pd.to_numeric(shares["fund_shares"], errors="coerce")
    shares = shares.loc[shares["date"].le(cutoff)].dropna(
        subset=["date", "fund_shares"]
    ).sort_values("date").drop_duplicates("date", keep="last")
    margin_data = margin.copy()
    margin_data["date"] = pd.to_datetime(margin_data["date"], errors="coerce")
    margin_data = margin_data.loc[margin_data["date"].le(cutoff)].sort_values("date")
    market = etf_daily.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.loc[market["date"].le(cutoff)].sort_values("date")
    if shares.empty or margin_data.empty or market.empty:
        raise SectorTaxonomyError("ETF份额、融资融券或行情为空")
    latest_share = shares.iloc[-1]
    latest_margin = margin_data.iloc[-1]
    latest_market = market.iloc[-1]

    def share_change(period: int) -> float | None:
        if len(shares) <= period:
            return None
        base = float(shares.iloc[-period - 1]["fund_shares"])
        return float(latest_share["fund_shares"] / base - 1.0) if base > 0 else None

    return {
        "fund_share_as_of_date": latest_share["date"].date().isoformat(),
        "fund_shares": float(latest_share["fund_shares"]),
        "fund_share_change_1d": share_change(1),
        "fund_share_change_5d": share_change(5),
        "fund_share_semantics": "PRIMARY_MARKET_ACTIVITY_PROXY_NOT_EXACT_CASH_FLOW",
        "margin_as_of_date": latest_margin["date"].date().isoformat(),
        "margin_balance_cny": float(latest_margin["rzye"]),
        "securities_lending_balance_cny": float(latest_margin["rqye"]),
        "financing_net_buy_cny": float(latest_margin["financing_net_buy_cny"]),
        "market_as_of_date": latest_market["date"].date().isoformat(),
        "etf_close": float(latest_market["close"]),
        "etf_amount_cny": float(latest_market["amount"]),
    }


def build_citic_crosscheck(
    industry_snapshot: pd.DataFrame,
    citic_index_daily: pd.DataFrame,
    code_to_name: Mapping[str, str],
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    """用中信全行业指数单日收益交叉核对沪深300行业归因方向。"""

    current = citic_index_daily.copy()
    current["date"] = pd.to_datetime(current["trade_date"], errors="coerce")
    current = current.loc[current["date"].eq(pd.Timestamp(as_of_date))].copy()
    current["industry_l1"] = current["ts_code"].astype(str).map(code_to_name)
    current["citic_market_return_1d"] = (
        pd.to_numeric(current["pct_change"], errors="coerce") / 100.0
    )
    left = industry_snapshot[["industry_l1", "return_5d"]].copy()
    # 单日精确值由调用方在industry_snapshot附加，缺少时返回不可比。
    if "return_1d" in industry_snapshot.columns:
        left = industry_snapshot[["industry_l1", "return_1d"]].copy()
    else:
        left["return_1d"] = np.nan
    merged = left.merge(
        current[["industry_l1", "citic_market_return_1d"]],
        on="industry_l1",
        how="left",
        validate="one_to_one",
    )
    comparable = merged[["return_1d", "citic_market_return_1d"]].notna().all(axis=1)
    same_sign = np.sign(merged.loc[comparable, "return_1d"]).eq(
        np.sign(merged.loc[comparable, "citic_market_return_1d"])
    )
    return {
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "comparable_industry_count": int(comparable.sum()),
        "same_sign_count": int(same_sign.sum()),
        "same_sign_ratio": float(same_sign.mean()) if len(same_sign) else None,
        "semantics": "CSI300_CONSTITUENT_INDUSTRY_RETURN_VS_ALL_MARKET_CITIC_INDUSTRY_INDEX",
        "rows": merged.to_dict(orient="records"),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染V2当前板块情境报告。"""

    lines = [
        "# 行业预期差—ETF模型 V2：板块重构与数据补齐",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 状态：`{report['status']}`",
        f"- 原V1方向状态：`{report['original_prediction_state']}`（未改变）",
        "- 边界：`RESEARCH_ONLY / SHADOW_ONLY / NO_POSITION_CHANGE`",
        "",
        "## 一句话结论",
        "",
        report["headline"],
        "",
        "## 中证官方宽板块",
        "",
        "| CICS一级行业 | 权重 |",
        "|---|---:|",
    ]
    for row in report["official_cics_l1"]:
        lines.append(f"| {row['industry_name_cn']} ({row['industry_name_en']}) | {row['weight_pct']:.3f}% |")
    technology = report["official_technology_crosscheck"]
    lines.extend(
        [
            "",
            f"官方信息技术为 `{technology['information_technology_weight_pct']:.3f}%`，"
            f"通信服务为 `{technology['communication_services_weight_pct']:.3f}%`，"
            f"两者合计 `{technology['combined_weight_pct']:.3f}%`；合计只用于科技宽口径展示。",
            "",
            "## 互斥经济引擎",
            "",
            "| 经济引擎 | 权重 | 5日 | 20日 | 60日 | 60日历史分位 | 预期差净贡献 | 预期差覆盖 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["economic_buckets"]:
        gap = row["expectation_gap"]
        def pct(value: Any) -> str:
            return "未观察" if value is None else f"{float(value):.2%}"
        lines.append(
            f"| {row['bucket_name_cn']} | {row['index_weight']:.2%} | "
            f"{pct(row.get('return_5d'))} | {pct(row.get('return_20d'))} | "
            f"{pct(row.get('return_60d'))} | {pct(row.get('return_60d_history_percentile'))} | "
            f"{pct(gap.get('weighted_score'))} | {pct(gap.get('observed_weight'))} |"
        )
    lines.extend(
        [
            "",
            "## 未来研究排序（定性先行）",
            "",
            "| 经济引擎 | 研究优先级 | 预期差状态 | 价格阶段 | 预期差覆盖率 |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in report["qualitative_forward_priorities"]:
        lines.append(
            f"| {row['bucket_name_cn']} | {row['research_priority']} | "
            f"{row['gap_state']} | {row['price_phase']} | {row['expectation_coverage_ratio']:.1%} |"
        )
    lines.extend(
        [
            "",
            "该排序只回答下一步先研究谁：`PRIORITY_FISH_MIDDLE_CANDIDATE`仍不是买入信号；"
            "`WATCH_FOR_TURN_NOT_ENTER`明确表示基本面/预期差可能有利，但价格尚未确认。",
        ]
    )
    lines.extend(
        [
            "",
            "## 重叠主题链",
            "",
            "| 主题 | 核心覆盖 | 邻接覆盖 | 核心20日价格 | 核心60日价格 | 1日订单流强度 | 5日订单流强度 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["themes"]:
        def pct(value: Any) -> str:
            return "未观察" if value is None else f"{float(value):.2%}"
        lines.append(
            f"| {row['theme_name_cn']} | {row['core_index_weight']:.2%} | "
            f"{row['adjacent_index_weight']:.2%} | {pct(row['core_return_20d_weighted_snapshot'])} | "
            f"{pct(row['core_return_60d_weighted_snapshot'])} | {pct(row.get('flow_intensity_1d'))} | "
            f"{pct(row.get('flow_intensity_5d'))} |"
        )
    etf = report["etf_flow_and_liquidity"]
    lines.extend(
        [
            "",
            "主题之间存在重叠，禁止把核心覆盖或邻接覆盖跨主题相加。",
            "",
            "## ETF与市场流动性",
            "",
            f"- 510300份额截止：`{etf['fund_share_as_of_date']}`；1日变化 "
            f"`{etf['fund_share_change_1d']:.3%}`，5日变化 `{etf['fund_share_change_5d']:.3%}`。",
            f"- 融资融券截止：`{etf['margin_as_of_date']}`；融资净买入 "
            f"`{etf['financing_net_buy_cny']:,.0f}` 元。",
            f"- ETF行情截止：`{etf['market_as_of_date']}`；成交额 "
            f"`{etf['etf_amount_cny']:,.0f}` 元。",
            "- 份额变化是申赎活动代理，不是精确现金净流入；个股订单分类资金流不是ETF申赎。",
            "",
            "## 独立交叉核对",
            "",
            f"沪深300成分行业收益与全市场中信一级行业指数在 "
            f"`{report['citic_return_crosscheck']['comparable_industry_count']}` 个可比行业中，"
            f"同向比例为 `{report['citic_return_crosscheck']['same_sign_ratio']:.1%}`。"
            if report["citic_return_crosscheck"]["same_sign_ratio"] is not None
            else "中信行业指数交叉核对没有形成可比样本。",
            "",
            "## 仍未观察",
            "",
            *[f"- `{item}`" for item in report["fields_still_unobserved"]],
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "SectorTaxonomyError",
    "build_bucket_snapshot",
    "build_citic_crosscheck",
    "build_economic_bucket_daily",
    "build_industry_flow_context",
    "build_industry_snapshot",
    "classify_bucket_forward_priorities",
    "build_theme_snapshot",
    "render_markdown",
    "summarize_etf_flow",
    "validate_official_cics_snapshot",
    "validate_taxonomy",
]
