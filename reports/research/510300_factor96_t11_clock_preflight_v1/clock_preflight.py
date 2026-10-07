"""保存T11公告日期代理的实际边界；不读取价格、财务数值或未来收益。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_t11_clock_preflight_v1"
BASE = "reports/research/510300_factor96_financial_parser_scope_v2/measurement_inputs/inputs/"
SOURCES = {
    "inputs/report_events.parquet": BASE + "report_events.parquet",
    "inputs/membership.parquet": BASE + "membership.parquet",
    "inputs/calendar.parquet": BASE + "calendar.parquet",
    "source_evidence/archive_receipt.json": "data/raw/cninfo/a_share_hs_periodic_report_event_archive_v2_1_receipt.json",
    "source_evidence/archive.py": "research/a_share_hs_cninfo_periodic_report_event_archive_v2_1.py",
    "source_evidence/archive_driver.py": "scripts/run_a_share_hs_cninfo_periodic_report_event_archive_v2_1.py",
    "source_evidence/normalizer.py": "research/a_share_hs_cninfo_periodic_report_metadata_v1.py",
    "source_evidence/original_config.yaml": "config/a_share_hs_cninfo_periodic_report_metadata_v1.yaml",
    "source_evidence/archive_config.yaml": "config/a_share_hs_cninfo_periodic_report_event_archive_v2_1.yaml",
    "source_evidence/measurement_protocol.json": "reports/research/510300_factor96_earnings_cashflow_measurement_v1/protocol.json",
}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def next_session(dates, sessions, offset):
    """沿用已冻结测量口径：严格在名义公告日期之后，超出日历则未知。"""
    positions = sessions.searchsorted(dates, side="right") + offset
    values = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = (positions < len(sessions)) & (dates.to_numpy() >= sessions[0].to_datetime64())
    values[valid] = sessions[positions[valid]].to_numpy(dtype="datetime64[ns]")
    return pd.Series(values, index=dates.index)


def main():
    assert not OUT.exists(), "时钟预检已存在，保留原结果，不覆盖运行。"
    OUT.mkdir(parents=True)
    save("protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_T11_CLOCK_PREFLIGHT_V1",
        "scope": "只检查公告元数据、历史成员及交易日历；不计算O02收益、行业聚合、入场或账户。",
        "prior_inspection": "执行前已观察全表时间分布和归档代码；本文件是方法与证据记录，不声称对元数据统计事前未知。",
        "inherited_clock": "名义公开日期之后的首个已覆盖交易日；沿用财报测量的额外延迟一日对照。",
        "window_mapping": "只列保守观察日及观察日之后的下一开盘。尚不选择开收盘或收收盘收益，不把映射当成真实首次反应证明。",
        "new_accounts": 0, "new_network_requests": 0, "returns_read": False,
        "goal_achieved": False, "orders_authorized": False,
    })
    copied = []
    for target, source in SOURCES.items():
        destination = OUT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
        copied.append({"path": target, "source_path": source,
                       "bytes": destination.stat().st_size, "sha256": digest(destination)})
    shutil.copy2(Path(__file__), OUT / "clock_preflight.py")
    save("input_identity.json", {"at": now(), "files": copied})

    events = pd.read_parquet(OUT / "inputs/report_events.parquet")
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    calendar = pd.read_parquet(OUT / "inputs/calendar.parquet")
    receipt = json.loads((OUT / "source_evidence/archive_receipt.json").read_text(encoding="utf-8-sig"))
    assert digest(OUT / "inputs/report_events.parquet") == receipt["outputs"]["normalized_event_archive"]["sha256"]
    assert events.announcement_id.is_unique
    assert not events.duplicated(["ts_code", "report_period"]).any()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(calendar.loc[calendar.is_open, "date"]).unique()))
    published = pd.to_datetime(events.event_publication_date)
    stamp = pd.to_datetime(events.official_timestamp_at, utc=True).dt.tz_convert("Asia/Shanghai")
    rebuilt = pd.to_datetime(events.retrieved_at, utc=True).dt.tz_convert("Asia/Shanghai")
    midnight = stamp.eq(stamp.dt.normalize())
    assert published.notna().all() and stamp.notna().all()
    assert stamp.dt.tz_localize(None).dt.normalize().eq(published).all()
    events = events.assign(
        metadata_midnight=midnight,
        historical_member_union=events.ts_code.isin(set(members.symbol)),
        nominal_observation_date=next_session(published, sessions, 0),
        delayed_observation_date=next_session(published, sessions, 1),
        earliest_open_after_nominal_observation=next_session(published, sessions, 1),
        earliest_open_after_delayed_observation=next_session(published, sessions, 2),
    )
    for name in ["nominal_observation_date", "delayed_observation_date"]:
        assert events.loc[events[name].notna(), name].gt(published[events[name].notna()]).all()
    for observation, entry in [
        ("nominal_observation_date", "earliest_open_after_nominal_observation"),
        ("delayed_observation_date", "earliest_open_after_delayed_observation"),
    ]:
        known = events[observation].notna() & events[entry].notna()
        assert events.loc[known, entry].gt(events.loc[known, observation]).all()

    subsets = {"ALL_CANONICAL_REPORTS": events, "HISTORICAL_MEMBER_UNION": events.loc[events.historical_member_union]}
    counts, annual = {}, []
    for name, table in subsets.items():
        counts[name] = {
            "rows": len(table), "midnight_metadata_rows": int(table.metadata_midnight.sum()),
            "nonmidnight_metadata_rows": int((~table.metadata_midnight).sum()),
            "known_nominal_observation_dates": int(table.nominal_observation_date.notna().sum()),
            "unknown_nominal_observation_dates": int(table.nominal_observation_date.isna().sum()),
            "known_open_after_nominal_observation": int(table.earliest_open_after_nominal_observation.notna().sum()),
        }
        yearly = table.assign(year=pd.to_datetime(table.event_publication_date).dt.year).groupby("year").agg(
            rows=("announcement_id", "size"), midnight_rows=("metadata_midnight", "sum"),
            known_nominal_observation_dates=("nominal_observation_date", "count"),
            known_open_after_nominal_observation=("earliest_open_after_nominal_observation", "count"),
        ).reset_index()
        annual.append(yearly.assign(scope=name))
    pd.concat(annual, ignore_index=True).to_csv(OUT / "yearly_clock_coverage.csv", index=False, encoding="utf-8-sig")
    member_events = subsets["HISTORICAL_MEMBER_UNION"].copy()
    member_events.to_parquet(OUT / "member_event_clock_map.parquet", index=False)
    events.loc[~events.metadata_midnight].to_parquet(OUT / "nonmidnight_metadata_rows.parquet", index=False)
    result = {
        "at": now(), "status": "COMPLETE_DATE_PROXY_MAP_STRICT_FIRST_PUBLICATION_NOT_ESTABLISHED",
        "counts": counts, "archive_rebuild_time_min": str(rebuilt.min()), "archive_rebuild_time_max": str(rebuilt.max()),
        "archive_rebuild_time_distinct": int(rebuilt.nunique()),
        "actual_clock_findings": [
            "零点记录只证明供应商元数据的日期，不能解释为当天开盘前已公开；非零时分秒也不自动等于首次公开证据。",
            "archive_driver.py在重建时将datetime.now(timezone)统一传给normalizer；retrieved_at不是历史逐份first_seen。",
            "原表按发行人和报告期保留档案中最早的合格原始全文；这不证明原文从未在同一URL被替换。",
            "标题规则排除了取消、更正、修订、更新后等文档；规范表不能独自提供T11所需的新增事实否定或更正退出链。",
        ],
        "next_stage_boundary": {
            "nominal_date_discovery": "须先单独冻结行业聚合、缺失覆盖规则、O02观察窗口、训练、入场及退出，才可作清楚标注的日期代理历史研究。",
            "strict_historical_first_seen": "NOT_ESTABLISHED",
            "correction_invalidation_chain": "NOT_ESTABLISHED_IN_CANONICAL_EVENT_TABLE",
            "O02": "NOT_COMPUTED", "T11": "NOT_RUN",
            "financial_source_batch": "独立运行中的V2解析不因本预检被重启或改动。",
        },
        "new_accounts": 0, "new_returns": 0, "new_network_requests": 0,
        "goal_achieved": False, "orders_authorized": False,
    }
    save("result.json", result)
    text = (
        "# T11公告时钟预检\n\n"
        "现有规范公告表可以映射名义日期之后的保守观察日，但尚不能证明历史真实首发时点，"
        "也没有包含更正或取消公告的完整退出链。O02尚未计算，T11账户尚未运行。\n\n"
        f"全表{len(events):,}条，零点元数据{int(midnight.sum()):,}条；历史成员并集{len(member_events):,}条。"
        f"统一的retrieved_at为{rebuilt.min()}，由重建程序写入，不是每份文档的历史first_seen。\n\n"
        "日期映射沿用既有冻结口径，观察日严格晚于名义公开日；潜在入场开盘严格晚于观察日。"
        "本预检没有读取市场价格，没有决定收益窗口、覆盖门槛或入场阈值。\n\n"
        "后续日期代理研究必须在首次收益计算之前固定具体算法，并单独披露首次发布版本、更正链与覆盖限制。"
        "单次历史回测不能成为目标达成或独立前向验证的证据。\n"
    )
    (OUT / "时钟边界说明.md").write_text(text, encoding="utf-8")
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save("result_freeze.json", {"at": now(), "files": files, "scope": "时钟预检结果；不包含财报V2运行目录的可变文件。"})
    print(json.dumps({"状态": result["status"], "统计": counts, "新账户": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
