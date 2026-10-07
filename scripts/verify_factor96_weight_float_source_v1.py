"""只读复核本轮原响应、历史文件和来源覆盖，不发起网络请求。"""

import argparse
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(root):
    result = read(root / "result.json")
    checked = 0
    for item in read(root / "freeze.json")["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["path"]
        checked += 1
    for item in read(root / "input_manifest.json")["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    initial = read(root / "source_manifest.json")
    registered = read(root / "protocol.json")
    for request, receipt in zip(registered["fixed_initial_requests"], initial, strict=True):
        assert request["id"] == receipt["id"] and request["params"] == receipt["params"]
        assert registered["frozen_at"] <= receipt["retrieved_at"]
        path = root / "sources" / Path(receipt["path"]).name
        assert path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"]
        assert receipt["status_code"] == 200
        if request["id"] != "cni_control":
            assert path.read_bytes().decode("utf-8") == "暂无数据"
        else:
            frame = pd.read_excel(io.BytesIO(path.read_bytes()))
            assert len(frame) == 500 and frame["日期"].nunique() == 1
            assert "总市值(亿元)" in frame and not any("自由流通" in str(c) for c in frame)
    extension = read(root / "archive_extension_manifest.json")
    for item in extension:
        path = root / "sources" / Path(item["path"]).name
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    assert read(root / "sources/wayback_csi_oss_alias.json") == []
    wayback = read(root / "sources/wayback_csi_official_alias.json")
    wayback = [dict(zip(wayback[0], row, strict=True)) for row in wayback[1:]]
    index_matches = [r for r in wayback if "000300closeweight.xls" in r["original"]]
    assert len(index_matches) == 9
    inputs = root / "inputs"
    old = "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1"
    prior = read(inputs / old / "pit_weight_source_evidence_manifest.json")
    historic = [r for r in prior["source_objects"] if r["kind"] == "CSI_OFFICIAL_CLOSEWEIGHT_VIA_INTERNET_ARCHIVE_CAPTURE"]
    assert {r["digest"] for r in index_matches} == {r["wayback_cdx_digest"] for r in historic}
    cc_protocol = read(root / "commoncrawl_protocol.json")
    requests = []
    for batch in cc_protocol["selected"]:
        receipt = read(root / "commoncrawl_receipts" / (batch["id"] + ".json"))
        assert receipt["network_attempts"] == 1 and receipt["status"] == "NO_CAPTURE_IN_FIXED_QUERY"
        assert receipt["params"] == cc_protocol["params"] and receipt["at"] >= cc_protocol["registered_at"]
        assert receipt["record_count"] == 0 and receipt["status_code"] == 404
        raw = root / receipt["source"]["path"]
        assert digest(raw) == receipt["source"]["sha256"]
        assert "No Captures found" in raw.read_text(encoding="utf-8")
        requests.append(receipt)
    assert len(requests) == 11
    assert len(initial) + len(extension) + len(requests) == result["new_source_http_requests"] == 17
    inventory = pd.read_csv(root / "official_snapshot_inventory.csv", keep_default_na=False)
    expected_weights = []
    sets = {}
    for row in inventory.to_dict("records"):
        path = root / row["archive_path"]
        assert digest(path) == row["sha256"]
        data = pd.read_excel(path, dtype=object)
        names = {str(c).lower().replace(" ", "").replace("\n", ""): c for c in data}
        def col(suffix):
            found = [v for k, v in names.items() if k.endswith(suffix)]
            assert len(found) == 1
            return found[0]
        assert set(data[col("indexcode")].astype(str).str.zfill(6)) == {"000300"}
        sample_dates = pd.to_datetime(data[col("date")].astype(str)).dt.strftime("%Y-%m-%d")
        assert set(sample_dates) == {row["snapshot_date"]}
        weights = [v for k, v in names.items() if "weight" in k]
        assert len(weights) == 1
        amounts = pd.to_numeric(data[weights[0]])
        codes = data[col("constituentcode")].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        symbols = codes + codes.map(lambda x: ".SH" if x[0] == "6" else ".SZ")
        assert len(data) == symbols.nunique() == 300 and abs(amounts.sum() - row["weight_sum_pct"]) < 1e-10
        frame = pd.DataFrame({"snapshot_date": sample_dates, "symbol": symbols, "weight_pct": amounts, "source_id": row["source_id"]})
        expected_weights.append(frame)
        sets[row["source_id"]] = set(symbols)
    expected = pd.concat(expected_weights).sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)
    actual = pd.read_parquet(root / "official_snapshot_weights.parquet").sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
    calendar = pd.read_parquet(inputs / "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/trading_calendar_observed_open_days.parquet", columns=["date", "is_open"])
    days = sorted({pd.Timestamp(d).date() for d in calendar.loc[calendar.is_open.eq(1), "date"]
                   if "2021-01-01" <= str(pd.Timestamp(d).date()) <= "2025-12-31"})
    member = pd.read_parquet(inputs / old / "000300_daily_pit_membership_20150101_20260814.parquet", columns=["membership_date", "symbol"])
    member["day"] = pd.to_datetime(member.membership_date).dt.date
    members = member.groupby("day").symbol.agg(set).to_dict()
    saved = pd.read_csv(root / "daily_weight_anchor_coverage.csv", keep_default_na=False)
    assert saved.date.to_list() == [str(d) for d in days]
    compatible_days, clock_days = 0, 0
    for row, day in zip(saved.to_dict("records"), days, strict=True):
        possible = [r for r in inventory.to_dict("records")
                    if datetime.fromisoformat(r["proven_available_at"]).date() < day
                    and datetime.fromisoformat(r["snapshot_date"]).date() < day
                    and day - datetime.fromisoformat(r["snapshot_date"]).date() <= timedelta(days=45)]
        assert bool(possible) == row["clock_age_covered"]
        if not possible:
            assert row["snapshot_date"] == row["source_id"] == ""
            assert row["weight_factor_view"] == "NO_VIEW" and not row["clock_age_membership_covered"]
            continue
        chosen = sorted(possible, key=lambda r: r["snapshot_date"])[-1]
        delta = len(members[day] ^ sets[chosen["source_id"]])
        assert row["source_id"] == chosen["source_id"] and int(float(row["membership_set_difference"])) == delta
        assert row["clock_age_membership_covered"] == (delta == 0)
        clock_days += 1
        compatible_days += int(delta == 0)
    assert len(days) == 1212 and clock_days == 45 and compatible_days == 20
    assert result["main_open_days"] == len(days) and result["clock_age_membership_anchor_covered_days"] == compatible_days
    vendor = pd.read_parquet(inputs / "data/raw/constituents/000300_historical_weights.parquet", columns=["trade_date", "con_code"])
    assert vendor.trade_date.nunique() == result["vendor_snapshot_count"] == 120 and len(vendor) == 36000
    schema = read(root / "local_share_schema_evidence.json")
    basic = pq.ParquetFile(root / schema["input_path"])
    assert basic.schema.names == schema["columns"] and basic.metadata.num_rows == schema["row_count"] == 573911
    assert "free_share" not in basic.schema.names and "float_share" in basic.schema.names
    assert result["new_accounts"] == result["new_admitted_weight_snapshots"] == result["new_admitted_free_float_rows"] == 0
    assert result["cumulative_admitted_account_scenarios"] + result["cumulative_invalid_implementation_accounts"] == result["cumulative_executed_account_scenarios"] == 584
    assert not result["goal_achieved"] and not result["orders_authorized"]
    return {"status": "PASS_SAVED_SOURCE_RESPONSES_AND_WEIGHT_COVERAGE", "frozen_files_checked": checked,
            "archived_direct_requests_checked": 17, "official_snapshots_checked": len(inventory),
            "daily_coverage_rows_checked": len(days), "clock_age_days": clock_days,
            "clock_age_membership_days": compatible_days, "new_accounts": 0, "network_requests": 0,
            "prior_accounts_rerun": 0, "external_review": "NOT_PERFORMED"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读检查保存的权重与自由流通来源证据")
    parser.add_argument("--root", type=Path, default=Path("reports/research/510300_factor96_weight_float_source_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False), flush=True)
