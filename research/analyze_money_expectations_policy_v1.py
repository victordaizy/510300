"""按固定口径核对货币预期与政策消息，生成日频图表数据及描述统计。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def event_returns(day, market: pd.DataFrame) -> dict:
    day = pd.Timestamp(day).normalize()
    dates = pd.DatetimeIndex(market.date)
    first = int(dates.searchsorted(day, side="right"))
    result = {"base_date": None, "return_1d": np.nan, "return_5d": np.nan, "return_20d": np.nan}
    if first < 1 or first >= len(market):
        return result
    base = first - 1
    if day < dates[0]:
        return result
    result["base_date"] = dates[base].date().isoformat()
    for h in (1, 5, 20):
        end = first + h - 1
        if end < len(market):
            result[f"return_{h}d"] = float(market.iloc[end].wealth / market.iloc[base].wealth - 1)
            result[f"end_{h}d"] = dates[end].date().isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="复算已保存的货币预期与政策描述研究")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    inputs = study / "inputs"
    out = args.output_dir.resolve() if args.output_dir else study / "results"
    out.mkdir(parents=True, exist_ok=True)
    market = pd.read_parquet(inputs / "market_daily.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.normalize()
    assert market.date.is_unique and (market.wealth > 0).all()
    official = pd.read_csv(inputs / "official_releases.csv")
    official["release_day"] = pd.to_datetime(official.published_at).dt.tz_localize(None).dt.normalize()
    assert len(official) == 104 and official.stat_month.is_unique
    official = official.sort_values("stat_month").reset_index(drop=True)
    assert np.allclose(official.m1_yoy_pp - official.m2_yoy_pp, official.spread_pp)

    calendar = pd.read_csv(inputs / "M2_calendar_raw.csv")
    calendar["release_date"] = pd.to_datetime(calendar.release_date).dt.normalize()
    reconciled = calendar.merge(official[["stat_month", "release_day", "m2_yoy_pp"]],
                                left_on="release_date", right_on="release_day", how="left")
    def calendar_reason(row):
        if pd.isna(row.actual) or pd.isna(row.forecast):
            return "MISSING_ACTUAL_OR_FORECAST"
        if row.release_date < official.release_day.min() or row.release_date > official.release_day.max():
            return "OUTSIDE_OFFICIAL_COVERAGE"
        if row.actual == 0:
            return "ZERO_ACTUAL_SUSPECT"
        if pd.isna(row.stat_month):
            return "NO_OFFICIAL_RELEASE_ON_DATE"
        if not np.isclose(row.actual, row.m2_yoy_pp):
            return "ACTUAL_MISMATCH"
        return "DATE_ACTUAL_MATCH_FORECAST_VINTAGE_UNPROVEN"
    reconciled["status"] = reconciled.apply(calendar_reason, axis=1)
    def nearby(row):
        candidates = official[(official.release_day - row.release_date).abs().dt.days <= 7]
        candidates = candidates[np.isclose(candidates.m2_yoy_pp, row.actual)]
        return ";".join(candidates.release_day.dt.strftime("%Y-%m-%d"))
    reconciled["same_actual_official_dates_within_7d_not_admitted"] = reconciled.apply(nearby, axis=1)
    save_csv(reconciled, out / "M2日历逐行核对.csv")

    forecasts = pd.read_csv(inputs / "wgc_all_extracted_forecasts.csv")
    forecasts["report_date"] = pd.to_datetime(forecasts.report_date)
    # 同一 PDF 可由多个文章链接指向；源内容身份去重，不能增加独立观察数。
    forecasts = forecasts.drop_duplicates(["report_date", "source_sha256", "series", "previous_pp", "forecast_pp"])
    pairs = []
    for (day, sha), frame in forecasts.groupby(["report_date", "source_sha256"], sort=True):
        fields = dict(report_date=day.date().isoformat(), source_sha256=sha,
                      source_url=frame.iloc[0].source_url, source_page=int(frame.iloc[0].page))
        for series in ("M1", "M2"):
            ss = frame[frame.series == series]
            fields[f"{series}_forecast_pp"] = float(ss.iloc[0].forecast_pp) if len(ss) == 1 else np.nan
            fields[f"{series}_previous_pp"] = float(ss.iloc[0].previous_pp) if len(ss) == 1 else np.nan
        next_release = official[official.release_day >= day]
        if next_release.empty:
            fields["status"] = "NO_FOLLOWING_OFFICIAL_RELEASE"
            pairs.append(fields)
            continue
        a = next_release.iloc[0]
        fields.update(stat_month=a.stat_month, release_date=a.release_day.date().isoformat(),
                      published_at=a.published_at, definition_version=a.definition_version,
                      M1_actual_pp=float(a.m1_yoy_pp), M2_actual_pp=float(a.m2_yoy_pp))
        prev = official[official.stat_month < a.stat_month].iloc[-1]
        have_pair = pd.notna(fields["M1_forecast_pp"]) and pd.notna(fields["M2_forecast_pp"])
        match_prev = all(np.isclose(fields[f"M{s}_previous_pp"], prev[f"m{s}_yoy_pp"])
                         for s in (1, 2) if pd.notna(fields[f"M{s}_previous_pp"]))
        if not have_pair:
            fields["status"] = "MISSING_PAIRED_M1_M2_FORECAST"
        elif day == a.release_day:
            fields["status"] = "SAME_DAY_PUBLICATION_TIME_UNPROVEN"
        elif (a.release_day - day).days > 20:
            fields["status"] = "REPORT_TOO_FAR_FROM_RELEASE"
        elif not match_prev:
            fields["status"] = "PREVIOUS_VALUES_DO_NOT_MATCH_PRIOR_OFFICIAL_PERIOD"
        elif prev.training_regime != a.training_regime:
            fields["status"] = "M1_DEFINITION_BOUNDARY"
        else:
            fields["status"] = "ADMITTED_RECONSTRUCTED_PRERELEASE_CONSENSUS_PROXY"
            fields["forecast_age_days"] = (a.release_day - day).days
        pairs.append(fields)
    pairs = pd.DataFrame(pairs).sort_values(["report_date", "source_sha256"])
    save_csv(pairs, out / "周报预期配对与准入.csv")
    selected = pairs[pairs.status == "ADMITTED_RECONSTRUCTED_PRERELEASE_CONSENSUS_PROXY"].copy()
    selected = selected.sort_values(["stat_month", "report_date"]).drop_duplicates("stat_month", keep="last")
    selected["actual_spread_pp"] = selected.M1_actual_pp - selected.M2_actual_pp
    selected["expected_spread_proxy_pp"] = selected.M1_forecast_pp - selected.M2_forecast_pp
    selected["m1_surprise_pp"] = selected.M1_actual_pp - selected.M1_forecast_pp
    selected["m2_surprise_pp"] = selected.M2_actual_pp - selected.M2_forecast_pp
    selected["spread_surprise_proxy_pp"] = selected.actual_spread_pp - selected.expected_spread_proxy_pp
    selected = pd.concat([selected.reset_index(drop=True), pd.DataFrame(
        [event_returns(day, market) for day in selected.release_date])], axis=1)
    save_csv(selected, out / "事前预期差与510300全部事件.csv")

    retrospective = pd.read_csv(inputs / "post_release_consensus_examples.csv")
    retrospective = retrospective.merge(official[["stat_month", "release_day", "m1_yoy_pp", "m2_yoy_pp"]], on="stat_month", validate="one_to_one")
    retrospective["spread_surprise_proxy_pp"] = (retrospective.m1_yoy_pp - retrospective.m2_yoy_pp) - (retrospective.M1_forecast_pp - retrospective.M2_forecast_pp)
    retrospective = pd.concat([retrospective, pd.DataFrame([event_returns(d, market) for d in retrospective.release_day])], axis=1)
    save_csv(retrospective, out / "公布后回述预期_不并入事前样本.csv")

    policy = pd.read_csv(inputs / "policy_shocks_author_446.csv")
    policy["AnnouncementDate"] = pd.to_datetime(policy.AnnouncementDate)
    assert len(policy) == 446 and policy.AnnouncementDate.is_unique
    policy = pd.concat([policy, pd.DataFrame([event_returns(d, market) for d in policy.AnnouncementDate])], axis=1)
    policy["measurement"] = "FULL_SAMPLE_PCA_NOT_PIT_SIGNAL"
    save_csv(policy, out / "政策冲击与510300全部事件.csv")
    correlations, groups = [], []
    for factor in ("Target", "Path"):
        for h in (1, 5, 20):
            sub = policy.dropna(subset=[factor, f"return_{h}d"])
            correlations.append({"factor": factor, "horizon": h, "n": len(sub),
                "pearson": sub[factor].corr(sub[f"return_{h}d"]),
                "spearman": float(spearmanr(sub[factor], sub[f"return_{h}d"]).statistic),
                "interpretation": "历史关联；窗口重叠；没有因果或样本外预测通过结论"})
            for sign, ss in (("负值_利率下行方向", sub[sub[factor] < 0]), ("正值_利率上行方向", sub[sub[factor] > 0])):
                vals = ss[f"return_{h}d"]
                groups.append({"factor": factor, "horizon": h, "sign": sign, "n": len(ss),
                    "mean_return": vals.mean(), "median_return": vals.median(), "positive_fraction": (vals > 0).mean()})
    save_csv(pd.DataFrame(correlations), out / "政策冲击各窗口相关性.csv")
    save_csv(pd.DataFrame(groups), out / "政策正负消息各窗口描述.csv")

    coverage = official[["stat_month", "release_day", "m1_yoy_pp", "m2_yoy_pp", "definition_version"]].copy()
    matched = reconciled[reconciled.status == "DATE_ACTUAL_MATCH_FORECAST_VINTAGE_UNPROVEN"].stat_month
    coverage["calendar_M2_actual_date_match_no_vintage"] = coverage.stat_month.isin(matched)
    coverage["prerelease_pair"] = coverage.stat_month.isin(selected.stat_month)
    coverage["postrelease_example_only"] = coverage.stat_month.isin(retrospective.stat_month) & ~coverage.prerelease_pair
    same_day = pairs[pairs.status == "SAME_DAY_PUBLICATION_TIME_UNPROVEN"].stat_month
    coverage["same_day_pair_time_unknown"] = coverage.stat_month.isin(same_day)
    coverage["market_response_available"] = coverage.release_day < market.date.max()
    coverage["strict_forward_immutable_snapshot"] = False
    coverage["paired_institution_spread_median"] = False
    save_csv(coverage, out / "104个月信息覆盖.csv")

    known = official.copy()
    known["available_at"] = pd.to_datetime(known.available_at_upper_bound).dt.tz_localize(None)
    prices = market[market.date >= "2018-01-01"].copy()
    prices["close_time"] = prices.date + pd.Timedelta(hours=15)
    prices = pd.merge_asof(prices.sort_values("close_time"), known[["available_at", "stat_month", "m1_yoy_pp", "m2_yoy_pp", "spread_pp", "training_regime"]].sort_values("available_at"),
                           left_on="close_time", right_on="available_at", direction="backward")
    assert (prices.available_at.dropna() <= prices.loc[prices.available_at.notna(), "close_time"]).all()
    save_csv(prices, out / "510300与当时已知货币数据_全部日线.csv")
    prices.to_parquet(out / "510300与当时已知货币数据_全部日线.parquet", index=False)
    summary = {
        "study_id": "510300_EXPECTATIONS_POLICY_EVIDENCE_V1", "market_start": str(market.date.min().date()),
        "market_cutoff": str(market.date.max().date()), "market_rows": len(market), "chart_daily_rows": len(prices),
        "official_months": len(official), "M2_calendar_total": len(calendar),
        "M2_calendar_both_values": int(calendar[["actual", "forecast"]].notna().all(axis=1).sum()),
        "calendar_status_counts": reconciled.status.value_counts().to_dict(),
        "prerelease_consensus_proxy_months": len(selected), "prerelease_months": selected.stat_month.tolist(),
        "positive_spread_surprises": int((selected.spread_surprise_proxy_pp > 0).sum()),
        "negative_spread_surprises": int((selected.spread_surprise_proxy_pp < 0).sum()),
        "zero_spread_surprises": int(np.isclose(selected.spread_surprise_proxy_pp, 0).sum()),
        "postrelease_example_months": len(retrospective),
        "policy_author_events": len(policy), "policy_price_matched_events": int(policy.return_1d.notna().sum()),
        "policy_first_price_matched_date": str(policy.loc[policy.return_1d.notna(), "AnnouncementDate"].min().date()),
        "policy_last_price_matched_date": str(policy.loc[policy.return_1d.notna(), "AnnouncementDate"].max().date()),
        "strict_forward_days": 0, "joint_predictive_test": "NOT_COMPUTED_INSUFFICIENT_PAIRED_CONSENSUS_AND_PIT_POLICY",
        "account_status": "NOT_RUN_INFORMATION_EVIDENCE_STUDY",
        "conclusion": "获得可核实的部分事前共识与历史政策消息数据；未建立超预期必涨或联合可交易增量证据。"
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected[["stat_month", "spread_surprise_proxy_pp", "return_1d", "return_5d", "return_20d"]].to_string(index=False))
    print(pd.DataFrame(correlations).drop(columns="interpretation").to_string(index=False))


if __name__ == "__main__":
    main()
