"""构造已确认披露需求强度，不把未识别信息当作全市场实际零买入。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_plan_identity_v1 as identity
import research.corporate_repurchase_original_plan_completion_v1 as extension
import research.corporate_repurchase_execution_extractor_v1 as facts
import research.corporate_repurchase_index_documents_v1 as originals

OUT = ROOT / "reports/research/510300_corporate_repurchase_disclosed_demand_v1"
STUDY = "510300_CORPORATE_REPURCHASE_DISCLOSED_DEMAND_V1"
DAILY = ROOT / "data/raw/constituents/000300_constituent_daily.parquet"
WEIGHTS = ROOT / "data/raw/constituents/000300_historical_weights.parquet"
MARKET = ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet"
START, END = pd.Timestamp("2024-09-01"), pd.Timestamp("2026-09-24")


def has_direct_path(node, nodes, visited=None):
    visited = set() if visited is None else set(visited)
    if node["document_id"] in visited:
        return False
    visited.add(node["document_id"])
    if node["node_status"] == "ORIGINAL_PLAN_ROOT":
        return True
    if node["root_method"] != "EXPLICIT_PRIOR_PLAN_ANNOUNCEMENT":
        return False
    references = set(node["referenced_announcements"])
    return any(has_direct_path(prior, nodes, visited) for prior in nodes
        if prior["symbol"] == node["symbol"] and prior["root_id"] == node["root_id"]
        and prior["known_at"] <= node["known_at"] and prior["document_id"] != node["document_id"]
        and references.intersection(prior["document_announcements"]))


def witnesses(meta, resolved, nodes):
    if resolved.get("status") != "LINKED_ASOF_ORIGINAL_PLAN" or not resolved.get("strong_direct_reference"):
        return []
    available = [node for node in nodes if node["symbol"] == meta["symbol"] and node["root_id"] == resolved["root_id"] and node["known_at"] <= meta["known_at"]]
    if resolved["method"] == "EXPLICIT_FIRST_PUBLICATION_DATE":
        selected = [node for node in available if meta["explicit_first_publication"] in [node["catalogue_date"], node["signed_date"], node["explicit_first_publication"]]]
    else:
        selected = [node for node in available if set(meta["referenced_announcements"]).intersection(node["document_announcements"])]
    return [node["document_id"] for node in selected if has_direct_path(node, nodes)]


def prepare_links():
    documents = {row["document_id"]: row for row in read(originals.OUT / "documents.json")}
    for name in ["supplement_documents.json", "gateway_supplement_documents.json"]:
        for row in read(extension.OUT / "results" / name):
            if row["status"] == "PDF_TEXT_SAVED":
                documents[row["document_id"]] = row
    metas = [extension.metadata(row) for row in documents.values()]
    nodes, excluded = identity.build_nodes(metas)
    indexed = {row["document_id"]: row for row in metas}
    pilot = {row["document_id"]: row for row in read(ROOT / "reports/research/510300_corporate_repurchase_fact_ledger_v1/results/facts.json")}
    output = []
    for item in read(facts.OUT / "results/parsed_documents.json"):
        row = dict(item)
        meta = dict(indexed[row["document_id"]])
        if row["document_id"] in pilot:
            meta["known_at"] = max(meta["known_at"], pd.Timestamp(pilot[row["document_id"]]["known_at"]))
        row["known_at"] = meta["known_at"]
        resolved = identity.resolve(meta, nodes)
        direct = witnesses(meta, resolved, nodes)
        row.update(original_identity=resolved, direct_witness_ids=direct,
            confirmed_identity=bool(direct) and row["status"].startswith("EXTRACTED_"))
        if row["confirmed_identity"]:
            row["root_id"] = resolved["root_id"]
        output.append(row)
    return output, nodes, excluded


def disclosed_changes(rows):
    confirmed = [r for r in rows if r["confirmed_identity"]]
    confirmed.sort(key=lambda r: (r["known_at"], pd.Timestamp(r["economic_cutoff"]) if r.get("economic_cutoff") else pd.Timestamp.min, r["document_id"]))
    states, stopped, output = {}, set(), []
    for item in confirmed:
        row = dict(item)
        root = row["root_id"]
        old = states.get(root)
        row.update(reported_increment_cents=None, previous_confirmed_id=old["document_id"] if old else None,
            change_status="UNKNOWN_STARTING_BASELINE", usable_positive_disclosure=False)
        if root in stopped:
            row["change_status"] = "QUARANTINED_AFTER_IDENTITY_OR_REVISION_CONFLICT"
        elif old is None:
            if row["classification"] == "FIRST_EXECUTION" or row["cumulative_cents"] == 0:
                row["reported_increment_cents"] = row["cumulative_cents"]
                row["change_status"] = "EXPLICIT_FIRST_EXECUTION" if row["cumulative_cents"] else "EXPLICIT_ZERO_BASELINE"
        else:
            delta = row["cumulative_cents"] - old["cumulative_cents"]
            old_cutoff, new_cutoff = old.get("economic_cutoff"), row.get("economic_cutoff")
            reversed_cutoff = old_cutoff is not None and new_cutoff is not None and pd.Timestamp(new_cutoff) < pd.Timestamp(old_cutoff)
            identical_cutoff = old_cutoff is not None and new_cutoff is not None and pd.Timestamp(new_cutoff) == pd.Timestamp(old_cutoff)
            precision = max(row["reporting_resolution_cents"], old["reporting_resolution_cents"])
            if delta < -precision or reversed_cutoff or (old["classification"] == "COMPLETION" and abs(delta) > precision) or (identical_cutoff and abs(delta) > precision):
                row["change_status"] = "IDENTITY_OR_REVISION_CONFLICT"
                stopped.add(root)
            elif abs(delta) <= precision and delta != 0:
                row["change_status"] = "CHANGE_WITHIN_REPORTING_PRECISION"
            else:
                row["reported_increment_cents"] = delta
                row["change_status"] = "POSITIVE_CONFIRMED_REPORTED_CHANGE" if delta > 0 else "UNCHANGED_REPORTED_CUMULATIVE"
        if row["reported_increment_cents"] is not None:
            row["usable_positive_disclosure"] = row["reported_increment_cents"] > row["reporting_resolution_cents"]
        if root not in stopped:
            states[root] = row
        output.append(row)
    return output


def feature_series(changes, dates, amounts, weights, source_end):
    calendar = pd.DatetimeIndex(dates)
    wide = amounts.pivot(index="date", columns="con_code", values="amount").reindex(calendar)
    denominators = wide.rolling(20, min_periods=20).sum().shift(1)
    snapshots = {pd.Timestamp(day): group.set_index("con_code").weight.to_dict() for day, group in weights.groupby("trade_date")}
    snapshot_dates = sorted(snapshots)
    weight_index = np.searchsorted(np.asarray(snapshot_dates, dtype="datetime64[ns]"), calendar.to_numpy(dtype="datetime64[ns]"), side="left") - 1
    strengths, events = np.zeros(len(calendar)), []
    for item in changes:
        row = dict(item)
        day = pd.Timestamp(row["known_at"]).tz_localize(None).normalize()
        i = int(calendar.searchsorted(day, side="right"))
        row.update(decision_date=None, strength=None, normalization_status="NO_LATER_DECISION_DATE")
        if i >= len(calendar):
            events.append(row)
            continue
        row["decision_date"] = calendar[i]
        if not row["usable_positive_disclosure"]:
            row["normalization_status"] = "NO_CONFIRMED_POSITIVE_INCREMENT"
        elif weight_index[i] < 0:
            row["normalization_status"] = "NO_PRIOR_WEIGHT_SNAPSHOT"
        else:
            snapshot = snapshot_dates[weight_index[i]]
            weight = snapshots[snapshot].get(row["symbol"], 0.0)
            row.update(weight_snapshot=snapshot, weight_percent=weight)
            if (calendar[i] - snapshot).days > 45:
                row["normalization_status"] = "STALE_MONTHLY_WEIGHT_SNAPSHOT"
            elif weight <= 0:
                row["normalization_status"] = "OUTSIDE_PRIOR_INDEX_SNAPSHOT"
            else:
                denom = denominators.loc[calendar[i], row["symbol"]] if row["symbol"] in denominators.columns else np.nan
                row["trailing20_amount_cny"] = denom
                if not np.isfinite(denom) or denom <= 0:
                    row["normalization_status"] = "NO_COMPLETE_PRIOR_TURNOVER_WINDOW"
                elif i == 0 or calendar[i - 1] > source_end:
                    row["normalization_status"] = "TURNOVER_SOURCE_NOT_CURRENT"
                else:
                    row["strength"] = weight / 100.0 * (row["reported_increment_cents"] / 100.0) / denom
                    strengths[i] += row["strength"]
                    row["normalization_status"] = "CONFIRMED_REPORTED_DEMAND_NORMALIZED"
        events.append(row)
    covered = np.asarray((calendar >= START) & (calendar <= END))
    observed_days = pd.Series(covered.astype(int)).rolling(20, min_periods=20).sum().eq(20).to_numpy()
    previous_day = pd.Series(calendar).shift(1)
    current_denominator = previous_day.le(source_end).to_numpy()
    snapshot_known = np.asarray([j >= 0 and (calendar[i] - snapshot_dates[j]).days <= 45 for i, j in enumerate(weight_index)])
    known = observed_days & covered & current_denominator & snapshot_known
    intensity = pd.Series(strengths).rolling(20, min_periods=20).sum().where(known).to_numpy()
    frame = pd.DataFrame({"date": calendar, "confirmed_disclosure_strength_today": strengths,
        "confirmed_repurchase_intensity20": intensity, "repurchase_information_known": known,
        "actual_total_market_buying_known": False})
    return frame, events


def checks():
    base = {"confirmed_identity": True, "root_id": "A", "document_id": "1", "known_at": pd.Timestamp("2025-01-02", tz="Asia/Shanghai"),
        "economic_cutoff": "2025-01-01", "cumulative_cents": 10000, "reporting_resolution_cents": 1, "classification": "FIRST_EXECUTION"}
    repeat = {**base, "document_id": "2", "known_at": pd.Timestamp("2025-01-03", tz="Asia/Shanghai"), "classification": "PROGRESS"}
    later = {**repeat, "document_id": "3", "known_at": pd.Timestamp("2025-01-06", tz="Asia/Shanghai"), "economic_cutoff": "2025-01-05", "cumulative_cents": 15000}
    rows = disclosed_changes([base, repeat, later])
    assert [r["reported_increment_cents"] for r in rows] == [10000, 0, 5000]
    assert rows[:2] == disclosed_changes([base, repeat])
    invalid = {**later, "cumulative_cents": 5000}
    assert disclosed_changes([base, repeat, invalid])[-1]["change_status"] == "IDENTITY_OR_REVISION_CONFLICT"
    return {"repeat_not_new_flow": True, "future_disclosure_prefix_invariant": True,
        "negative_revision_not_cash_flow": True}


def run():
    tests = checks()
    rows, nodes, excluded = prepare_links()
    changes = disclosed_changes(rows)
    dates = pd.read_parquet(MARKET, columns=["date"]).date.astype("datetime64[ns]")
    amounts = pd.read_parquet(DAILY, columns=["date", "con_code", "amount"])
    amounts["date"] = pd.to_datetime(amounts.date).astype("datetime64[ns]")
    weights = pd.read_parquet(WEIGHTS)
    weights["trade_date"] = pd.to_datetime(weights.trade_date).astype("datetime64[ns]")
    series, events = feature_series(changes, dates, amounts, weights, amounts.date.max())
    # 后来的披露不能改变既有特征；这里不读取任何收益列。
    prefix_end = pd.Timestamp("2025-06-30")
    earlier = [r for r in changes if pd.Timestamp(r["known_at"]).tz_localize(None) < prefix_end + pd.Timedelta(days=1)]
    prefix, _ = feature_series(earlier, dates[dates <= prefix_end], amounts[amounts.date <= prefix_end], weights[weights.trade_date <= prefix_end], amounts.date.max())
    pd.testing.assert_frame_equal(series[series.date <= prefix_end].reset_index(drop=True), prefix)
    assert all(pd.Timestamp(r["known_at"]).tz_localize(None) < pd.Timestamp(r["decision_date"]) + pd.Timedelta(hours=9)
        for r in events if r["normalization_status"] == "CONFIRMED_REPORTED_DEMAND_NORMALIZED")
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(identity.__file__), Path(extension.__file__), Path(facts.__file__),
        originals.OUT / "documents.json", extension.OUT / "results/supplement_documents.json", extension.OUT / "results/gateway_supplement_documents.json",
        facts.OUT / "results/parsed_documents.json", DAILY, WEIGHTS, MARKET]
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "meaning": "过去20个决策日出现的、具有直接原方案证据且金额可辨的正向披露增量，经股票成交额和历史指数权重归一化。它是系统可确认的信息，不等于全部实际回购或资金净流入。",
        "identity_gate": "只接纳明确首次披露日或原方案公告编号的连接，而且引用路径不能依赖仅会议/预算相似的合并。",
        "increment": "同原方案累计报告额求差，明确首购从0开始；期初基数未知不生成增量，重复报告不新增，修订冲突隔离，不跨方案拼接。",
        "normalization": "每个有效披露的金额增量/该股前20交易日成交额之和，乘以前一期已保存指数权重，再按过去20决策日求和。只使用决策日前成交额，不用未来市值或成交量。",
        "zero_definition": "0只表示该固定识别流程没有新增可确认正向披露，不表示企业当天没有买入；未识别公告另存，不能称全市场真实买盘。",
        "coverage": "2024-09-01起至少20个已覆盖交易日，成交额仅至2026-08-19，月度权重最长45自然日；其余保留NO_VIEW。",
        "source_clock": "披露日末后最早下一交易日09:00可用，历史首发送达未认证；全部为历史重建。",
        "before_new_returns": True, "new_returns_loaded": False, "new_accounts": 0, "new_fits": 0,
        "implementation_checks": {**tests, "saved_series_future_prefix_invariant": True},
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}, "code_sha256": digest(Path(__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / "results/plan_nodes.json", nodes, True)
    save(OUT / "results/identity_and_amount_rows.json", rows, True)
    save(OUT / "results/confirmed_disclosure_events.json", events, True)
    series.to_parquet(OUT / "results/decision_information.parquet", index=False)
    qualified = series[series.repurchase_information_known]
    result = {"at": now(), "study_id": STUDY, "status": "CONFIRMED_DISCLOSED_DEMAND_SERIES_READY_FOR_FIXED_TEST",
        "all_source_records": len(rows), "directly_supported_execution_records": len(changes),
        "confirmed_original_schemes": len({r['root_id'] for r in changes}),
        "change_status_counts": pd.Series([r["change_status"] for r in changes]).value_counts().to_dict(),
        "normalization_status_counts": pd.Series([r["normalization_status"] for r in events]).value_counts().to_dict(),
        "feature_dates": len(qualified), "feature_start": qualified.date.min(), "feature_end": qualified.date.max(),
        "positive_feature_dates": int(qualified.confirmed_repurchase_intensity20.gt(0).sum()),
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print({"已确认披露需求序列完成": True, "直接证据记录": len(changes), "特征日期": len(qualified), "标准化状态": result['normalization_status_counts']}, flush=True)


if __name__ == "__main__":
    run()
