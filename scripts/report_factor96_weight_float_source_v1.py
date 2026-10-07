"""固定来源证据、计算覆盖缺口并写出本轮结果；不运行收益检验。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_weight_float_source_v1"
OLD = "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1"
MANIFEST = OLD + "/pit_weight_source_evidence_manifest.json"
MEMBERSHIP = OLD + "/000300_daily_pit_membership_20150101_20260814.parquet"
VENDOR = "data/raw/constituents/000300_historical_weights.parquet"
CALENDAR = "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/trading_calendar_observed_open_days.parquet"
BASIC = "data/raw/a_share_size_value_development_v1/daily_basic_signal_dates.parquet"
DAILY = "reports/research/510300_factor96_daily_state_shrink_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def copy_inputs():
    manifest = read(ROOT / MANIFEST)
    paths = [MANIFEST, MEMBERSHIP, VENDOR, CALENDAR, BASIC,
             "config/510300_csi300_pit_membership_weights_source_remediation_v1_protocol_manifest.json",
             "reports/data_quality/510300_csi300_pit_membership_weights_source_remediation_v1.json",
             "reports/research/510300_factor96_program_v1/status.json",
             "config/510300_existing_data_training_mandate_v1.json",
             DAILY + "/result.json", DAILY + "/metrics.csv", DAILY + "/delivery_receipt.json"]
    for item in manifest["source_objects"]:
        path = ROOT / item["archive_path"]
        assert identity(path)["sha256"] == item["sha256"], item["archive_path"]
        paths.append(item["archive_path"])
    records = []
    for name in paths:
        source, target = ROOT / name, OUT / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        assert identity(source) == identity(target)
        records.append({"original_path": name, "path": target.relative_to(OUT).as_posix(), **identity(target)})
    save("input_manifest.json", {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "files": records})


def parse_snapshot(path):
    data = pd.read_excel(path, dtype=object)
    def column(suffix):
        names = [c for c in data if str(c).strip().lower().endswith(suffix.lower())]
        assert len(names) == 1, (path.name, suffix, names)
        return names[0]
    date_col = column("Date")
    code_col = column("Constituent Code")
    weight_cols = [c for c in data if "weight" in str(c).lower()]
    assert len(weight_cols) == 1
    dates = pd.to_datetime(data[date_col].astype(str)).dt.normalize()
    codes = data[code_col].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    symbols = codes + codes.map(lambda c: ".SH" if c.startswith("6") else ".SZ")
    weights = pd.to_numeric(data[weight_cols[0]], errors="raise")
    assert dates.nunique() == 1 and len(data) == symbols.nunique() == 300
    assert 98 <= weights.sum() <= 102
    result = pd.DataFrame({"snapshot_date": dates.dt.strftime("%Y-%m-%d"), "symbol": symbols, "weight_pct": weights})
    return result.sort_values("symbol").reset_index(drop=True)


def analyse():
    inputs = OUT / "inputs"
    manifest = read(inputs / MANIFEST)
    snapshots, panel = [], []
    symbol_sets = {}
    for item in manifest["source_objects"]:
        if "CLOSEWEIGHT_" not in item["kind"]:
            continue
        frame = parse_snapshot(inputs / item["archive_path"])
        snapshot_date = frame.iloc[0]["snapshot_date"]
        assert snapshot_date == item["snapshot_date"]
        stamp = item.get("wayback_capture_timestamp_utc")
        capture = (pd.to_datetime(stamp, format="%Y%m%d%H%M%S", utc=True).tz_convert("Asia/Shanghai")
                   if stamp else pd.Timestamp(manifest["acquired_at"]))
        source_id = item["sha256"][:16]
        row = {"source_id": source_id, "snapshot_date": snapshot_date,
               "proven_available_at": capture.isoformat(),
               "availability_evidence": "ARCHIVE_CAPTURE" if stamp else "LOCAL_RETRIEVAL_RECEIPT",
               "historical_publication_time": "UNKNOWN",
               "constituents": len(frame), "weight_sum_pct": float(frame.weight_pct.sum()),
               "official_original_url": item.get("official_original_url", item["url"]),
               "archive_path": "inputs/" + item["archive_path"], "sha256": item["sha256"]}
        symbol_sets[source_id] = set(frame.symbol)
        snapshots.append(row)
        frame["source_id"] = source_id
        panel.append(frame)
    inventory = pd.DataFrame(snapshots).sort_values("snapshot_date")
    inventory.to_csv(OUT / "official_snapshot_inventory.csv", index=False, encoding="utf-8-sig")
    pd.concat(panel, ignore_index=True).to_parquet(OUT / "official_snapshot_weights.parquet", index=False)
    vendor = pd.read_parquet(inputs / VENDOR, columns=["trade_date", "con_code", "weight"])
    vendor["trade_date"] = pd.to_datetime(vendor.trade_date)
    monthly = []
    for month in pd.period_range("2019-01", "2025-12", freq="M"):
        official = inventory.loc[pd.to_datetime(inventory.snapshot_date).dt.to_period("M").eq(month)]
        monthly.append({"month": str(month),
                        "vendor_snapshot_count": int(vendor.loc[vendor.trade_date.dt.to_period("M").eq(month), "trade_date"].nunique()),
                        "archived_official_snapshot_count": len(official),
                        "official_snapshot_dates": "|".join(official.snapshot_date),
                        "proven_available_at": "|".join(official.proven_available_at),
                        "full_required_sequence_admitted": False})
    pd.DataFrame(monthly).to_csv(OUT / "monthly_weight_evidence_gap.csv", index=False, encoding="utf-8-sig")
    membership = pd.read_parquet(inputs / MEMBERSHIP, columns=["membership_date", "symbol"])
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    members = membership.groupby("membership_date").symbol.agg(set).to_dict()
    calendar = pd.read_parquet(inputs / CALENDAR, columns=["date", "is_open"])
    dates = sorted(pd.to_datetime(calendar.loc[calendar.is_open.eq(1), "date"]).drop_duplicates())
    dates = [t for t in dates if pd.Timestamp("2021-01-01") <= t <= pd.Timestamp("2025-12-31")]
    protocol = read(inputs / "config/510300_csi300_pit_membership_weights_source_remediation_v1_protocol_manifest.json")
    maximum_age = protocol["weight_clock"]["maximum_snapshot_age_calendar_days"]
    assert maximum_age == 45
    coverage = []
    for day in dates:
        eligible = []
        for row in snapshots:
            snap = pd.Timestamp(row["snapshot_date"])
            proven_day = pd.Timestamp(row["proven_available_at"]).tz_localize(None).normalize()
            if proven_day < day and snap < day and (day - snap).days <= maximum_age:
                eligible.append(row)
        latest = max(eligible, key=lambda x: x["snapshot_date"]) if eligible else None
        difference = len(symbol_sets[latest["source_id"]] ^ members[day]) if latest else None
        coverage.append({"date": day.strftime("%Y-%m-%d"), "snapshot_date": latest["snapshot_date"] if latest else None,
                         "source_id": latest["source_id"] if latest else None,
                         "clock_age_covered": latest is not None,
                         "membership_set_difference": difference,
                         "clock_age_membership_covered": latest is not None and difference == 0,
                         "weight_factor_view": "SOURCE_ANCHOR_ONLY" if latest and difference == 0 else "NO_VIEW"})
    covered = pd.DataFrame(coverage)
    covered.to_csv(OUT / "daily_weight_anchor_coverage.csv", index=False, encoding="utf-8-sig")
    basic = pq.ParquetFile(inputs / BASIC)
    metadata = pq.read_table(inputs / BASIC, columns=["trade_date", "ts_code"]).to_pandas()
    schema = {"input_path": "inputs/" + BASIC, "row_count": basic.metadata.num_rows,
              "columns": basic.schema.names, "has_free_share": "free_share" in basic.schema.names,
              "date_min": str(pd.to_datetime(metadata.trade_date).min().date()),
              "date_max": str(pd.to_datetime(metadata.trade_date).max().date()),
              "date_count": int(metadata.trade_date.nunique()), "security_count": int(metadata.ts_code.nunique()),
              "read_value_columns": ["trade_date", "ts_code"],
              "interpretation": "表有float_share和circ_mv，无free_share；普通流通口径不替代原因子所需自由流通口径。"}
    save("local_share_schema_evidence.json", schema)
    counts = vendor.groupby("trade_date").con_code.nunique()
    sums = vendor.groupby("trade_date").weight.sum()
    primary = inventory.loc[pd.to_datetime(inventory.snapshot_date).between("2021-01-01", "2025-12-31")]
    cc = read(OUT / "commoncrawl_result.json")
    assert cc["attempted_queries"] == 11 and cc["capture_records"] == 0
    result = {"study_id": "510300_FACTOR96_WEIGHT_FLOAT_SOURCE_V1",
              "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status": "SOURCE_CHECK_COMPLETED_NO_NEW_ADMITTED_WEIGHT_OR_FREE_FLOAT_SEQUENCE",
              "new_initial_cni_requests": 3, "new_wayback_queries": 2, "new_commoncrawl_catalogue_requests": 1,
              "new_commoncrawl_index_queries": cc["attempted_queries"], "new_source_http_requests": 17,
              "new_admitted_weight_snapshots": 0, "new_admitted_free_float_rows": 0,
              "vendor_snapshot_count": len(counts), "vendor_rows": len(vendor),
              "vendor_first_date": str(vendor.trade_date.min().date()), "vendor_last_date": str(vendor.trade_date.max().date()),
              "vendor_minimum_constituents": int(counts.min()), "vendor_maximum_constituents": int(counts.max()),
              "vendor_minimum_weight_sum_pct": float(sums.min()), "vendor_maximum_weight_sum_pct": float(sums.max()),
              "existing_official_archive_snapshots": 9, "existing_current_retrieval_snapshots": 1,
              "primary_2021_2025_official_archived_snapshots": len(primary),
              "primary_official_snapshot_dates": primary.snapshot_date.to_list(),
              "main_open_days": len(covered), "clock_age_anchor_covered_days": int(covered.clock_age_covered.sum()),
              "clock_age_membership_anchor_covered_days": int(covered.clock_age_membership_covered.sum()),
              "coverage_ratio": float(covered.clock_age_membership_covered.mean()),
              "coverage_definition": "仅证明日之后、快照严格在前、45日内且300成员完全相符的来源锚点；不是完整日度权重重建或因子准入。",
              "T04": "NOT_RUN_HISTORICAL_WEIGHTS_AND_INDUSTRY_REQUIRED",
              "T12": "NOT_RUN_FREE_FLOAT_AND_PURPOSE_VERSION_CHAIN_REQUIRED",
              "T13": "NOT_RUN_FREE_FLOAT_AND_COMPLETE_ISSUANCE_EVENT_CHAIN_REQUIRED",
              "source_exhaustiveness": "仅17个明确请求；未证明其他公开或付费来源不存在，不以HTTP200代表有数据。",
              "new_accounts": 0, "cumulative_admitted_account_scenarios": 432,
              "cumulative_invalid_implementation_accounts": 152, "cumulative_executed_account_scenarios": 584,
              "goal_achieved": False, "goal_status": "active", "current_market_view": "NO_VIEW",
              "independent_forward_accounts": 0, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save("result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    assert not (OUT / "result.json").exists()
    copy_inputs()
    analyse()
