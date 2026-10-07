"""只读重算已解压交付包的日期映射和更正候选；无需仓库或网络。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mapped(dates, sessions, offset):
    positions = sessions.searchsorted(dates, side="right") + offset
    result = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    inside = (positions < len(sessions)) & dates.ge(sessions[0]).to_numpy()
    result[inside] = sessions[positions[inside]].to_numpy(dtype="datetime64[ns]")
    return pd.Series(result, index=dates.index)


def infer_period(title):
    years = set(re.findall(r"(?<!\d)(20\d{2})年", title))
    ends = []
    for tokens, suffix in [(("第一季度报告", "一季度报告"), "03-31"),
                           (("半年度报告", "中期报告", "半年报"), "06-30"),
                           (("第三季度报告", "三季度报告"), "09-30")]:
        if any(token in title for token in tokens):
            ends.append(suffix)
    annual = re.sub("半年度报告|中期报告|半年报", "", title)
    if "年度报告" in annual or "年报" in annual:
        ends.append("12-31")
    return pd.Timestamp(next(iter(years)) + "-" + ends[0]) if len(years) == len(ends) == 1 else pd.NaT


def verify(root):
    index = pd.read_csv(root / "FILE_INDEX.csv", dtype={"path": str, "sha256": str})
    assert index.path.is_unique
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert observed == set(index.path) | {"FILE_INDEX.csv"}
    for record in index.to_dict("records"):
        path = root / record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], record["path"]
    clock = root / "evidence/clock"
    correction = root / "evidence/corrections"
    for base in [clock, correction]:
        for record in read(base / "result_freeze.json")["files"]:
            path = base / record["path"]
            assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]

    events = pd.read_parquet(clock / "inputs/report_events.parquet")
    members = pd.read_parquet(clock / "inputs/membership.parquet")
    member_codes = set(members.symbol)
    calendar = pd.read_parquet(clock / "inputs/calendar.parquet")
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(calendar.loc[calendar.is_open, "date"]).unique()))
    dates = pd.to_datetime(events.event_publication_date)
    stamps = pd.to_datetime(events.official_timestamp_at, utc=True).dt.tz_convert("Asia/Shanghai")
    assert stamps.dt.tz_localize(None).dt.normalize().eq(dates).all()
    recomputed = events.assign(metadata_midnight=stamps.eq(stamps.dt.normalize()),
        historical_member_union=events.ts_code.isin(member_codes), nominal_observation_date=mapped(dates, sessions, 0),
        delayed_observation_date=mapped(dates, sessions, 1),
        earliest_open_after_nominal_observation=mapped(dates, sessions, 1),
        earliest_open_after_delayed_observation=mapped(dates, sessions, 2))
    actual_members = pd.read_parquet(clock / "member_event_clock_map.parquet")
    expected_members = recomputed.loc[recomputed.historical_member_union]
    pd.testing.assert_frame_equal(actual_members.reset_index(drop=True), expected_members.reset_index(drop=True), check_dtype=False)
    for name, frame in [("ALL_CANONICAL_REPORTS", recomputed), ("HISTORICAL_MEMBER_UNION", expected_members)]:
        expected = {"rows": len(frame), "midnight_metadata_rows": int(frame.metadata_midnight.sum()),
                    "nonmidnight_metadata_rows": int((~frame.metadata_midnight).sum()),
                    "known_nominal_observation_dates": int(frame.nominal_observation_date.notna().sum()),
                    "unknown_nominal_observation_dates": int(frame.nominal_observation_date.isna().sum()),
                    "known_open_after_nominal_observation": int(frame.earliest_open_after_nominal_observation.notna().sum())}
        assert expected == read(clock / "result.json")["counts"][name]

    originals = {(r.ts_code, pd.Timestamp(r.report_period)): (str(r.announcement_id), pd.Timestamp(r.event_publication_date))
                 for r in events.itertuples(index=False)}
    rows, seen = [], set()
    with gzip.open(correction / "reconstructed_api_records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            identifier = str(raw.get("announcementId") or "")
            assert identifier not in seen
            seen.add(identifier)
            title = re.sub(r"[\s\u3000]+", "", html.unescape(re.sub(r"<[^>]+>", "", str(raw.get("announcementTitle") or "")))).replace("－", "-")
            if not any(token in title for token in ("更正", "取消", "修订", "更新", "补充", "差错", "追溯", "重述")):
                continue
            code = str(raw.get("secCode") or "").zfill(6)
            symbol = code + ".SH" if re.fullmatch(r"(600|601|603|605|688|689)\d{3}", code) else (
                code + ".SZ" if re.fullmatch(r"(000|001|002|003|300|301)\d{3}", code) else None)
            period = infer_period(title)
            match = re.search(r"finalpage/(\d{4}-\d{2}-\d{2})/", str(raw.get("adjunctUrl") or ""))
            published = pd.Timestamp(match.group(1)) if match else pd.NaT
            timestamp = pd.to_datetime(raw.get("announcementTime"), unit="ms", utc=True).tz_convert("Asia/Shanghai")
            equal_date = pd.notna(timestamp) and pd.notna(published) and timestamp.date() == published.date()
            original = originals.get((symbol, period))
            if pd.isna(period):
                status = "NO_VIEW_AMBIGUOUS_OR_MISSING_REPORT_PERIOD"
            elif not equal_date:
                status = "NO_VIEW_CLOCK_IDENTITY"
            elif original is not None and published < original[1]:
                status = "NO_VIEW_CANDIDATE_PRECEDES_CANONICAL_ORIGINAL"
            elif original is not None:
                status = "SAME_ISSUER_PERIOD_LATER_OR_SAME_DATE_CANDIDATE_ONLY"
            else:
                status = "NO_VIEW_NO_CANONICAL_ORIGINAL_FOR_EXPLICIT_PERIOD"
            rows.append({"announcement_id": identifier, "ts_code": symbol, "announcement_title": title,
                         "report_period": period, "event_publication_date": published,
                         "canonical_original_announcement_id": original[0] if original is not None else None,
                         "candidate_link_status": status, "in_historical_member_union": symbol in member_codes})
    rebuilt = pd.DataFrame(rows).sort_values("announcement_id").reset_index(drop=True)
    saved = pd.read_parquet(correction / "correction_candidates.parquet").sort_values("announcement_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, saved[rebuilt.columns], check_dtype=False)
    assert len(seen) == read(correction / "result.json")["reconstruction"]["raw_record_count"]
    for name, frame in [("all_candidates", rebuilt), ("historical_member_union_candidates", rebuilt.loc[rebuilt.in_historical_member_union])]:
        expected = {"rows": len(frame), "issuers": int(frame.ts_code.nunique()),
                    "candidate_link_status": dict(Counter(frame.candidate_link_status))}
        assert expected == read(correction / "result.json")["counts"][name]
    cancelled = rebuilt.loc[rebuilt.in_historical_member_union, "announcement_title"].str.contains("取消", regex=False).sum()
    assert int(cancelled) == read(correction / "title_status_clock_addendum.json")["historical_member_union_cancelled_title_rows"]
    return {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_METADATA_CLOCK_AND_CANDIDATE_RECOMPUTATION",
            "indexed_files_verified": len(index), "raw_records_recomputed": len(seen), "correction_candidates_recomputed": len(rebuilt),
            "historical_member_clock_rows_recomputed": len(expected_members), "new_accounts": 0, "new_network_requests": 0,
            "future_returns_read": False, "strict_historical_first_seen": "NOT_ESTABLISHED",
            "complete_information_invalidation_exit_chain": "NOT_ESTABLISHED", "goal_achieved": False,
            "external_review": "NOT_PERFORMED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
