"""国家统计局10时事件预收益准入的离线回归测试。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from research.nbs_1000_pre_return_admission_v1 import (
    _era_counts,
    _event_exclusion_reason,
    build_pre_return_event_ledger,
    discover_schedule_links,
    load_protocol,
    parse_annual_schedule,
    verify_protocol_manifest,
)
from research.stk_mins_source_admission_v1 import canonical_json_bytes, sha256_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEKDAY_MARKERS = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}


def _annual_html(
    year: int,
    publication: date,
    *,
    july_time: str = "10:00",
) -> tuple[bytes, list[pd.Timestamp]]:
    dates: list[pd.Timestamp] = []
    date_cells: list[str] = []
    time_cells: list[str] = []
    for month in range(1, 13):
        if month == 2:
            date_cells.append("……")
            time_cells.append("……")
            continue
        candidate = pd.Timestamp(date(year, month, 15))
        while candidate.weekday() >= 5:
            candidate += pd.Timedelta(days=1)
        dates.append(candidate)
        date_cells.append(f"{candidate.day}/{WEEKDAY_MARKERS[candidate.weekday()]}")
        time_cells.append(july_time if month == 7 else "10:00")
    header = "".join(f"<th>{month}月</th>" for month in range(1, 13))
    date_html = "".join(f"<td>{value}</td>" for value in date_cells)
    time_html = "".join(f"<td>{value}</td>" for value in time_cells)
    html = f"""<!doctype html>
<html><head><meta charset="UTF-8"><title>国家统计局信息公开</title></head>
<body>
<div>索 引 号 410A04-0401-{publication:%Y%m}-0001</div>
<div>成文日期 {publication.year}年{publication.month}月{publication.day}日</div>
<h2>{year}年国家统计局主要统计信息发布日程表</h2>
<table>
<tr><th>序号</th><th>内容</th>{header}</tr>
<tr><td>1</td><td>国民经济运行情况新闻发布会</td>{date_html}</tr>
<tr><td>1</td><td>国民经济运行情况新闻发布会</td>{time_html}</tr>
</table>
</body></html>"""
    return html.encode("utf-8"), dates


def test_strategy_protocol_v1_0_1_manifest_is_hash_valid() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    manifest = verify_protocol_manifest(PROJECT_ROOT)

    assert protocol["protocol"]["version"] == "1.0.1"
    assert protocol["nbs_event_contract"]["schedule_index"].endswith("/fbrcb/")
    assert manifest["protocol_remediation"]["research_definition_changed"] is False
    assert manifest["pre_freeze_attestations"]["nbs_event_return_read"] is False


def test_discover_schedule_links_requires_every_year() -> None:
    anchors = "".join(
        f'<a href="./{year}/schedule.html">{year}年国家统计局主要统计信息发布日程表</a>'
        for year in range(2017, 2027)
    )
    links = discover_schedule_links(
        f"<html><body>{anchors}</body></html>".encode("utf-8"),
        "https://www.stats.gov.cn/xxgk/sjfb/fbrcb/",
        ["www.stats.gov.cn", "stats.gov.cn"],
        2017,
        2026,
    )

    assert list(links) == list(range(2017, 2027))
    assert links[2017] == "https://www.stats.gov.cn/xxgk/sjfb/fbrcb/2017/schedule.html"


def test_parse_annual_schedule_preserves_non_1000_release() -> None:
    html, _ = _annual_html(2024, date(2023, 12, 29), july_time="15:00")

    rows = parse_annual_schedule(
        html,
        2024,
        "https://www.stats.gov.cn/example.html",
        sha256_bytes(html),
        ["国民经济运行情况", "国民经济运行情况新闻发布会"],
    )

    assert len(rows) == 11
    january = rows[0]
    july = next(row for row in rows if row["schedule_month"] == 7)
    assert january["scheduled_time"] == "10:00:00"
    assert january["schedule_publication_before_event"] is True
    assert january["schedule_snapshot_state"] == "ORIGINAL_PRE_YEAR_OFFICIAL_SCHEDULE"
    assert july["scheduled_time"] == "15:00:00"


def test_parse_accepts_official_metadata_when_visible_heading_is_empty() -> None:
    html, _ = _annual_html(2026, date(2025, 12, 25), july_time="15:00")
    html = html.replace(
        b"<h2>2026\xe5\xb9\xb4\xe5\x9b\xbd\xe5\xae\xb6\xe7\xbb\x9f\xe8\xae\xa1\xe5\xb1\x80\xe4\xb8\xbb\xe8\xa6\x81\xe7\xbb\x9f\xe8\xae\xa1\xe4\xbf\xa1\xe6\x81\xaf\xe5\x8f\x91\xe5\xb8\x83\xe6\x97\xa5\xe7\xa8\x8b\xe8\xa1\xa8</h2>",
        b"<h2></h2>",
    ).replace(
        b"<body>",
        "<body><div>信息类别 发布日程表</div><div>发布机构 国家统计局</div>".encode(
            "utf-8"
        ),
    )

    rows = parse_annual_schedule(
        html,
        2026,
        "https://www.stats.gov.cn/example-2026.html",
        sha256_bytes(html),
        ["国民经济运行情况", "国民经济运行情况新闻发布会"],
    )

    assert len(rows) == 11
    assert next(row for row in rows if row["schedule_month"] == 7)[
        "scheduled_time"
    ] == "15:00:00"


def test_event_exclusion_precedence_is_frozen() -> None:
    base = {
        "scheduled_date": pd.Timestamp("2024-07-15"),
        "scheduled_time": "10:00:00",
        "schedule_publication_before_event": True,
        "sse_open_day": True,
        "minute_data_state": "PASS_FOUR_WINDOWS_COMPLETE",
    }
    assert (
        _event_exclusion_reason(base, date(2017, 1, 1), date(2026, 9, 4))
        == "PASS_ELIGIBLE_PRE_RETURN"
    )
    non_time = dict(base, scheduled_time="15:00:00")
    assert (
        _event_exclusion_reason(non_time, date(2017, 1, 1), date(2026, 9, 4))
        == "EXCLUDED_NON_1000_RELEASE"
    )
    late = dict(base, schedule_publication_before_event=False)
    assert (
        _event_exclusion_reason(late, date(2017, 1, 1), date(2026, 9, 4))
        == "EXCLUDED_UNVERIFIABLE_SCHEDULE_REVISION"
    )
    weekend = dict(base, sse_open_day=False)
    assert (
        _event_exclusion_reason(weekend, date(2017, 1, 1), date(2026, 9, 4))
        == "EXCLUDED_NON_TRADING_DAY"
    )


def test_build_ledger_uses_only_schedule_and_minute_completeness(tmp_path: Path) -> None:
    protocol = load_protocol(PROJECT_ROOT)
    inventory_path = tmp_path / protocol["nbs_event_contract"]["source_inventory"]
    raw_dir = tmp_path / "data/raw/official/nbs/test"
    raw_dir.mkdir(parents=True)
    index_content = b"<html><body>index</body></html>"
    index_path = raw_dir / "schedule_index.html"
    index_path.write_bytes(index_content)
    records: list[dict[str, object]] = [
        {
            "kind": "INDEX",
            "year": None,
            "requested_url": protocol["nbs_event_contract"]["schedule_index"],
            "final_url": protocol["nbs_event_contract"]["schedule_index"],
            "retrieved_at": "2026-09-04T22:00:00+08:00",
            "raw_relative_path": index_path.relative_to(tmp_path).as_posix(),
            "bytes": len(index_content),
            "sha256": sha256_bytes(index_content),
        }
    ]
    all_event_dates: list[pd.Timestamp] = []
    for year in range(2017, 2027):
        html, dates = _annual_html(year, date(year - 1, 12, 1))
        all_event_dates.extend(dates)
        path = raw_dir / f"year={year}.html"
        path.write_bytes(html)
        records.append(
            {
                "kind": "ANNUAL",
                "year": year,
                "requested_url": f"https://www.stats.gov.cn/{year}.html",
                "final_url": f"https://www.stats.gov.cn/{year}.html",
                "retrieved_at": "2026-09-04T22:00:00+08:00",
                "raw_relative_path": path.relative_to(tmp_path).as_posix(),
                "bytes": len(html),
                "sha256": sha256_bytes(html),
            }
        )
    inventory = {
        "model_id": "510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1",
        "phase": "NBS_OFFICIAL_SCHEDULE_ACQUISITION",
        "state": "PASS_NBS_SCHEDULE_ACQUISITION",
        "generated_at": "2026-09-04T22:00:00+08:00",
        "schedule_index": protocol["nbs_event_contract"]["schedule_index"],
        "allowed_hosts": protocol["nbs_event_contract"]["allowed_hosts"],
        "years": list(range(2017, 2027)),
        "raw_retrieval_directory": raw_dir.relative_to(tmp_path).as_posix(),
        "records": records,
        "all_raw_hashes_present": True,
        "event_return_reads": 0,
        "model_training_run": False,
        "portfolio_evaluation_run": False,
        "position_impact": 0,
    }
    inventory["inventory_fingerprint"] = sha256_bytes(canonical_json_bytes(inventory))
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trading_dates = pd.date_range("2017-01-03", "2026-09-04", freq="B")
    quality = pd.DataFrame(
        {
            "trade_date": trading_dates,
            "exact_standard_241_labels": True,
            "pre_window_complete": True,
            "reaction_window_complete": True,
            "entry_window_complete": True,
            "exit_window_complete": True,
        }
    )
    quality_path = (
        tmp_path
        / "data/curated/510300_stk_mins_source_admission_v1/daily_quality_ledger.parquet"
    )
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_parquet(quality_path, index=False)

    ledger, manifest = build_pre_return_event_ledger(tmp_path, protocol)

    assert len(ledger) == 110
    assert manifest["event_return_reads"] == 0
    assert manifest["event_labels_created"] == 0
    assert "open" not in ledger.columns
    assert "close" not in ledger.columns
    assert "return" not in ledger.columns
    assert ledger["final_event_eligibility"].sum() >= 85
    assert ledger["model_pre_return_eligibility"].sum() >= 80


def test_era_counts_are_event_ordinal_based() -> None:
    ledger = pd.DataFrame(
        {
            "era": ["TRAINING_ORIGIN"] * 36
            + ["ERA_1"] * 20
            + ["ERA_2"] * 20
            + ["ERA_3"] * 15
            + ["NOT_MODEL_ELIGIBLE"] * 3
        }
    )

    assert _era_counts(ledger) == {
        "TRAINING_ORIGIN": 36,
        "ERA_1": 20,
        "ERA_2": 20,
        "ERA_3": 15,
    }
