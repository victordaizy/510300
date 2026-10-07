"""复用已提取季度数，统一时间单位后完成可用性测量，不重读收益。"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.banker_survey_field_admission_v1 as fields
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_banker_survey_availability_completion_v1"
STUDY = "510300_BANKER_SURVEY_AVAILABILITY_COMPLETION_V1"
PANEL = fields.OUT / "released_banker_survey.parquet"


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("季度可用性已有完成记录。")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PANEL)
    assert len(frame) == 32 and frame.quarter.is_unique
    assert read(fields.OUT / "unresolved_fields.json") == []
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "completion_reason": "32份当季字段与原表已经全部保存。后续日期连接遇到pandas微秒与纳秒类型不同；只统一时间表示为纳秒，不改时间值、字段或来源。",
        "source_panel_sha256": digest(PANEL), "source_manifest_sha256": digest(fields.OUT / "source_identity_completion.json"),
        "extra_diagnostic": "除最近两年决策原点的季度状态数，另计原报告公布时间也在两年范围内的严格季度数。",
        "no_new_stock_returns": True, "new_accounts": 0, "new_market_downloads": 0}, True)
    states = frame.sort_values(["known_at", "quarter"]).drop_duplicates("known_at", keep="last")
    dates = pd.read_parquet(ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet", columns=["date"])
    dates["date"] = pd.to_datetime(dates.date).dt.as_unit("ns")
    dates["decision_time"] = (dates.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)).dt.as_unit("ns")
    states["known_at"] = pd.to_datetime(states.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    daily = pd.merge_asof(dates, states[["known_at", "quarter"]], left_on="decision_time", right_on="known_at", direction="backward")
    daily["age_days"] = (daily.decision_time - daily.known_at).dt.total_seconds() / 86400
    daily["available_within_120_days"] = daily.age_days.between(0, 120)
    ordinary, strict = [], []
    for i, row in daily.iterrows():
        prior = daily.iloc[:i]
        lower = row.date - pd.DateOffset(years=2)
        mask = prior.date.ge(lower) & prior.available_within_120_days
        ordinary.append(int(prior.loc[mask, "quarter"].nunique()))
        strict.append(int(prior.loc[mask & prior.known_at.ge(lower.tz_localize("Asia/Shanghai")), "quarter"].nunique()))
    daily["distinct_prior_two_year_quarter_states"] = ordinary
    daily["distinct_source_clock_inside_two_year_quarters"] = strict
    daily.to_parquet(OUT / "daily_availability_only.parquet", index=False)
    current = daily[daily.date.ge("2020-01-02")]
    repeated = frame[frame.known_at.duplicated(keep=False)][["quarter", "published_at", "known_at"]]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY, "status": "32_CURRENT_QUARTER_SOURCES_READY_STRATEGY_NOT_RUN",
        "source_quarters": len(frame), "unresolved_quarters": 0, "fields": list(fields.FIELDS),
        "source_panel": PANEL.relative_to(ROOT).as_posix(), "source_panel_sha256": digest(PANEL),
        "first_quarter": frame.quarter.min(), "last_quarter": frame.quarter.max(),
        "same_day_released_quarters": repeated.to_dict("records"), "unique_effective_release_days": len(states),
        "table_field_counts": frame.table_field_count.value_counts().to_dict(),
        "publication_delay_min_days": int(frame.publication_days_from_quarter_end.min()),
        "publication_delay_max_days": int(frame.publication_days_from_quarter_end.max()),
        "delays_over_90_days": frame.loc[frame.publication_days_from_quarter_end.gt(90), ["quarter", "published_at", "publication_days_from_quarter_end"]].to_dict("records"),
        "evaluated_market_days": len(current), "available_days_within_120": int(current.available_within_120_days.sum()),
        "prior_two_year_distinct_quarter_states_min": int(current.distinct_prior_two_year_quarter_states.min()),
        "prior_two_year_distinct_quarter_states_max": int(current.distinct_prior_two_year_quarter_states.max()),
        "strict_prior_two_year_distinct_quarters_min": int(current.distinct_source_clock_inside_two_year_quarters.min()),
        "strict_prior_two_year_distinct_quarters_max": int(current.distinct_source_clock_inside_two_year_quarters.max()),
        "latest_information_asof_last_market_day": current.iloc[-1].to_dict(),
        "historical_first_vintage_verified": False, "new_accounts": 0, "new_strategy_returns": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False}, True)
    (OUT / "completion_code.py").write_bytes(Path(__file__).read_bytes())
    print(f"32季当季来源完成，{len(states)}个有效公布日；严格两年内最多{max(strict)}个不同季度状态。", flush=True)


if __name__ == "__main__":
    run()
