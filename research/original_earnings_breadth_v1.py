"""使用原始财报和当时成分股检验510300盈利改善广度，仅作研究。"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, eligible_training, model_for, save_account, summarize
from research.intraday_overnight_increment_v1 import (
    block_indices, holding_total_return, interval, normalize_dividends,
    now, require, return_metrics, write_json,
)
from research.policy_liquidity_quantity_v1 import simulate_policy

STUDY = "510300_ORIGINAL_EARNINGS_BREADTH_V1"
OUT = ROOT / "reports/research/510300_original_earnings_breadth_v1"
CONFIG = ROOT / "config/510300_original_earnings_breadth_v1.json"
MANIFEST = ROOT / "config/510300_original_earnings_breadth_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
BASE = "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1"
FACTS = ROOT / BASE / "official_financial_facts_v1_21_document_by_document.parquet"
FACT_RECEIPT = ROOT / BASE / "receipt_v1_21_document_by_document.json"
QUEUE = ROOT / "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1/official_fact_document_queue_v1_21_document_by_document.parquet"
EVENTS = ROOT / "data/raw/cninfo/a_share_hs_periodic_report_events_v2_1.parquet"
MEMBERS = ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet"
METRICS = {
    "PARENT_NET_PROFIT_YTD": "profit",
    "CORE_PARENT_NET_PROFIT_YTD": "core",
    "OPERATING_REVENUE_YTD": "revenue",
    "OPERATING_CASH_FLOW_YTD": "cashflow",
    "TOTAL_ASSETS_END": "assets",
}
GROWTH = ["profit", "core", "revenue", "cashflow"]
FCOLS = ([f"{k}_growth_median" for k in GROWTH]
         + [f"{k}_improve_share" for k in GROWTH]
         + ["positive_profit_share", "profit_cash_joint_improve_share", "profit_assets_annual_median",
            "cash_profit_assets_annual_median", "cash_profit_assets_change_median",
            "fundamental_coverage", "report_age_scaled", "recent_disclosure_share"])
GROUPS = ["PRICE", "EARNINGS", "FULL"]
ENSEMBLES = {
    "F1_PRIMARY_FULL_H60": ("FULL", 60),
    "F2_EARNINGS_H60": ("EARNINGS", 60),
    "F3_PRICE_H60": ("PRICE", 60),
    "F4_FULL_H20": ("FULL", 20),
}
NAMES = {
    "F1_PRIMARY_FULL_H60": "主方案：价格与原始财报合并，六十日两模型等权",
    "F2_EARNINGS_H60": "仅原始财报，六十日两模型等权",
    "F3_PRICE_H60": "仅价格对照，六十日两模型等权",
    "F4_FULL_H20": "价格与原始财报合并，二十日两模型等权",
    "BUY_HOLD": "买入持有",
}


def physical(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    return Path(r"E:\ResearchData\New project 8") / rel if rel.parts[0] == "data" else path


def sha(path: Path) -> str:
    return hashlib.sha256(physical(path).read_bytes()).hexdigest()


def ident(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": physical(path).stat().st_size,
            "sha256": sha(path)}


def original_amount(raw: str, multiplier: float) -> float:
    value = unicodedata.normalize("NFKC", str(raw)).replace(",", "").replace(" ", "").replace("−", "-")
    if value.startswith("["):
        return np.nan
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        return float(value) * float(multiplier)
    except ValueError:
        return np.nan


def symmetric_change(current, prior):
    current, prior = np.asarray(current, dtype=float), np.asarray(prior, dtype=float)
    denominator = np.abs(current) + np.abs(prior)
    return np.divide(2 * (current - prior), denominator, out=np.zeros_like(denominator), where=denominator != 0)


def next_session(dates, sessions: pd.DatetimeIndex) -> pd.Series:
    values = pd.to_datetime(dates)
    pos = sessions.searchsorted(values, side="right")
    result = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = pos < len(sessions)
    result[valid] = sessions[pos[valid]].to_numpy(dtype="datetime64[ns]")
    return pd.Series(result, index=dates.index)


def paired_events(events: pd.DataFrame, facts: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    e = events.copy()
    e["report_period"] = pd.to_datetime(e.report_period)
    e["event_publication_date"] = pd.to_datetime(e.event_publication_date)
    e["available_date"] = next_session(e.event_publication_date, sessions)
    wide = facts.pivot(index="announcement_id", columns="metric_id", values="verified_value").rename(columns=METRICS)
    for name in METRICS.values():
        if name not in wide:
            wide[name] = np.nan
    e = e.merge(wide[list(METRICS.values())], on="announcement_id", how="left", validate="one_to_one")
    prior = e[["ts_code", "report_period", "announcement_id", "available_date", *METRICS.values()]].copy()
    prior["report_period"] += pd.DateOffset(years=1)
    prior = prior.rename(columns={c: "prior_" + c for c in prior if c not in ("ts_code", "report_period")})
    e = e.merge(prior, on=["ts_code", "report_period"], how="left", validate="one_to_one")
    required = list(METRICS.values()) + ["prior_" + c for c in METRICS.values()]
    e["pair_valid"] = (np.isfinite(e[required]).all(axis=1) & (e.assets > 0) & (e.prior_assets > 0)
                       & e.prior_available_date.notna() & (e.prior_available_date <= e.available_date))
    for name in GROWTH:
        e[name + "_growth"] = symmetric_change(e[name], e["prior_" + name])
        e[name + "_improve"] = (e[name] > e["prior_" + name]).astype(float)
    factor = 12 / e.report_period.dt.month
    e["positive_profit"] = (e.profit > 0).astype(float)
    e["profit_cash_joint_improve"] = ((e.profit > e.prior_profit) & (e.cashflow > e.prior_cashflow)).astype(float)
    e["profit_assets_annual"] = e.profit / e.assets * factor
    e["cash_profit_assets_annual"] = (e.cashflow - e.profit) / e.assets * factor
    e["cash_profit_assets_change"] = e.cash_profit_assets_annual - (e.prior_cashflow - e.prior_profit) / e.prior_assets * factor
    e = e.loc[e.available_date.notna()].sort_values(["ts_code", "available_date", "report_period", "announcement_id"])
    largest = e.groupby("ts_code").report_period.cummax()
    e = e.loc[e.report_period == largest].drop_duplicates(["ts_code", "available_date"], keep="last")
    return e.reset_index(drop=True)


def aggregate_members(members: pd.DataFrame, events: pd.DataFrame, minimum: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = members[["membership_date", "symbol"]].rename(columns={"membership_date": "date", "symbol": "ts_code"}).copy()
    left["date"] = pd.to_datetime(left.date).astype("datetime64[ns]")
    left["ts_code"] = left.ts_code.astype(str)
    right = events.copy()
    right["available_date"] = pd.to_datetime(right.available_date).astype("datetime64[ns]")
    right["ts_code"] = right.ts_code.astype(str)
    panel = pd.merge_asof(left.sort_values(["date", "ts_code"]), right.sort_values(["available_date", "ts_code"]),
                          left_on="date", right_on="available_date", by="ts_code", direction="backward")
    panel["report_age_days"] = (panel.date - panel.report_period).dt.days
    panel["company_valid"] = panel.pair_valid.eq(True) & panel.report_age_days.between(0, 200)
    panel["recent_disclosure"] = ((panel.date - panel.available_date).dt.days < 20).astype(float)
    valid = panel.loc[panel.company_valid]
    rules = {**{f"{k}_growth_median": (f"{k}_growth", "median") for k in GROWTH},
             **{f"{k}_improve_share": (f"{k}_improve", "mean") for k in GROWTH},
             "positive_profit_share": ("positive_profit", "mean"),
             "profit_cash_joint_improve_share": ("profit_cash_joint_improve", "mean"),
             "profit_assets_annual_median": ("profit_assets_annual", "median"),
             "cash_profit_assets_annual_median": ("cash_profit_assets_annual", "median"),
             "cash_profit_assets_change_median": ("cash_profit_assets_change", "median"),
             "report_age_scaled": ("report_age_days", "median"),
             "recent_disclosure_share": ("recent_disclosure", "mean")}
    result = valid.groupby("date").agg(**rules)
    counts = panel.groupby("date").agg(member_count=("ts_code", "size"), valid_company_count=("company_valid", "sum"),
                                       latest_publication_date=("event_publication_date", "max"),
                                       latest_available_date=("available_date", "max"))
    result = counts.join(result)
    result["fundamental_coverage"] = result.valid_company_count / result.member_count
    result["report_age_scaled"] /= 200
    result["fundamental_valid"] = (result.member_count == 300) & (result.valid_company_count >= minimum) & np.isfinite(result[FCOLS]).all(axis=1)
    require((result.latest_available_date.dropna() <= result.latest_available_date.dropna().index).all(), "原始财报出现提前使用")
    return result.reset_index(), panel


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "source_receipt.json").exists(), "本轮来源已保存，禁止覆盖")
    f = pd.read_parquet(physical(FACTS))
    original_fact_count = len(f)
    e = pd.read_parquet(physical(EVENTS))
    m = pd.read_parquet(physical(MEMBERS))
    original_receipt = json.loads(physical(FACT_RECEIPT).read_text(encoding="utf-8"))
    source_status = original_receipt["status"]
    require(source_status == "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE", "原始档案状态与已知版本不符")
    require(sha(FACTS) == original_receipt["artifacts"]["facts_sha256"], "财务事实版本与来源回执不一致")
    require(sha(QUEUE) == original_receipt["artifacts"]["document_queue_sha256"], "原始文件选择清单与来源回执不一致")
    queue = pd.read_parquet(physical(QUEUE))
    require(not f.duplicated(["announcement_id", "metric_id"]).any(), "原始财务事实键重复")
    require(f.verification_status.eq("PASS_OFFICIAL_ORIGINAL_PDF_LABEL_VALUE_UNIT_VERIFIED").all(), "有未核实原始事实")
    require(not e.duplicated(["ts_code", "report_period"]).any(), "事件档案同证券同报告期重复")
    require(m.groupby("membership_date").symbol.nunique().eq(300).all(), "历史成分股数量不为300")
    f = f.loc[f.metric_id.isin(METRICS)].copy()
    for c in ("report_period", "event_publication_date"):
        f[c], e[c] = pd.to_datetime(f[c]), pd.to_datetime(e[c])
    keys = ["ts_code", "report_period", "event_publication_date"]
    joined = f[["announcement_id", *keys]].drop_duplicates().merge(e[["announcement_id", *keys]], on="announcement_id", how="left", suffixes=("_fact", "_event"), validate="one_to_one")
    for c in keys:
        require(joined[c + "_fact"].eq(joined[c + "_event"]).all(), "事实与原始公告档案不一致：" + c)
    f = f.merge(e[["announcement_id", "official_pdf_url"]].rename(columns={"official_pdf_url": "canonical_pdf_url"}), on="announcement_id", how="left", validate="many_to_one")
    f = f.merge(queue[["announcement_id", "resolved_official_pdf_url", "official_pdf_sha256", "source_route"]].rename(columns={"official_pdf_sha256": "resolved_pdf_sha256"}), on="announcement_id", how="left", validate="many_to_one")
    primary = f.official_pdf_url.eq(f.canonical_pdf_url) & f.announcement_id.eq(f.official_pdf_announcement_id)
    sibling = (f.official_pdf_url.eq(f.resolved_official_pdf_url) & f.official_pdf_sha256.eq(f.resolved_pdf_sha256)
               & f.source_route.eq("SAME_EVENT_OFFICIAL_FULL_PDF_SIBLING"))
    source_dates = pd.to_datetime(f.official_pdf_url.str.extract(r"finalpage/(\d{4}-\d{2}-\d{2})/")[0])
    f["source_identity_valid"] = (primary | sibling) & source_dates.eq(f.event_publication_date)
    f["verified_value"] = [original_amount(v, u) for v, u in zip(f.source_raw_value, f.source_unit_multiplier)]
    valid_amount = np.isfinite(f.verified_value)
    require(np.allclose(f.loc[valid_amount, "verified_value"], f.loc[valid_amount, "metric_value_cny"], rtol=1e-12, atol=.011), "原文金额与人民币单位换算不一致")
    f.loc[~f.source_identity_valid, "verified_value"] = np.nan
    require(f.statement_scope.eq("CONSOLIDATED_ONLY").all(), "不是合并口径")
    require(f.loc[f.metric_id != "TOTAL_ASSETS_END", "value_period_scope"].eq("YEAR_TO_DATE").all(), "利润现金流不是当期累计口径")
    require(f.loc[f.metric_id == "TOTAL_ASSETS_END", "value_period_scope"].eq("PERIOD_END").all(), "总资产不是期末口径")
    symbols = set(m.symbol)
    e = e.loc[e.ts_code.isin(symbols)].copy()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(m.membership_date).unique()))
    paired = paired_events(e, f, sessions)
    daily, panel = aggregate_members(m, paired)
    paired.to_parquet(OUT / "paired_original_events.parquet", index=False)
    panel.to_parquet(OUT / "daily_member_facts.parquet", index=False)
    daily.to_parquet(OUT / "daily_earnings_breadth.parquet", index=False)
    f.to_parquet(OUT / "selected_verified_facts.parquet", index=False)
    e.to_parquet(OUT / "original_event_subset.parquet", index=False)
    years = daily.groupby(daily.date.dt.year).agg(trading_days=("date", "size"), valid_days=("fundamental_valid", "sum"),
                                                 minimum_companies=("valid_company_count", "min"), median_companies=("valid_company_count", "median"),
                                                 maximum_companies=("valid_company_count", "max"))
    years.to_csv(OUT / "source_coverage_by_year.csv", encoding="utf-8-sig")
    receipt = {"study_id": STUDY, "prepared_at": now(), "status": "PASS_ORIGINAL_FACTS_CONTRACT_COVERAGE_ENFORCED_PER_DAY",
               "original_study_source_status_preserved": source_status, "source_rows_all_metrics": original_fact_count,
               "selected_fact_rows": len(f), "original_documents": int(f.announcement_id.nunique()),
               "amounts_independently_recomputed": int(valid_amount.sum()), "blank_or_non_numeric_rows_excluded": int((~valid_amount).sum()),
               "same_day_sibling_fact_rows_admitted": int(sibling.sum()),
               "unresolved_document_identity_rows_excluded": int((~f.source_identity_valid).sum()),
               "unresolved_document_identity_rows": f.loc[~f.source_identity_valid, ["announcement_id", "metric_id", "official_pdf_url"]].to_dict("records"),
               "valid_paired_reports": int(paired.pair_valid.sum()), "daily_membership_rows": len(panel), "yearly_coverage": years.reset_index().to_dict("records"),
               "own_candidate_returns_read": False, "all_original_pdfs_reparsed_this_round": False,
               "sources": [ident(p) for p in (FACTS, FACT_RECEIPT, QUEUE, EVENTS, MEMBERS)],
               "notes": ["原始文档页码、片段和哈希来自既有逐份提取档案，本轮未重新下载解析全部原始PDF。", "不沿用旧研究未通过的整体覆盖结论；本轮逐日240家门槛独立执行。", "等权盈利改善广度不等于指数EPS、历史市值权重或基金现金申赎。"]}
    write_json(OUT / "source_receipt.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False, default=str), flush=True)


def features() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_parquet(PARENT / "features.parquet")
    pc = json.loads((PARENT / "input_receipt.json").read_text(encoding="utf-8"))["features"]
    daily = pd.read_parquet(OUT / "daily_earnings_breadth.parquet")
    data = data.merge(daily, on="date", how="left", validate="one_to_one")
    data["feature_valid"] &= data.fundamental_valid.eq(True) & np.isfinite(data[pc + FCOLS]).all(axis=1)
    return data, pc


def columns(group: str, prices: list[str]) -> list[str]:
    return {"PRICE": prices, "EARNINGS": FCOLS, "FULL": prices + FCOLS}[group]


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "本轮已经冻结")
    config = json.loads((ROOT / "config/510300_total_reverse_repo_v2.json").read_text(encoding="utf-8"))
    for k in ("rule_names", "meta_names", "quantity_measure", "information_clock"):
        config.pop(k, None)
    config.update({"study_id": STUDY, "primary": "F1_PRIMARY_FULL_H60", "minimum_valid_companies": 240,
                   "maximum_report_age_calendar_days": 200, "information_clock": "NEXT_A_SHARE_SESSION_OPEN_AFTER_PUBLICATION_DATE",
                   "models": [{"id": f"{g}_{k}_H{h}", "group": g, "kind": k, "horizon": h}
                              for g in GROUPS for h in (20, 60) for k in ("RIDGE", "ET")],
                   "bootstrap_day_block": 60, "primary_horizon": 60, "missing_latest_report_fallback": False,
                   "financial_weighting": "EQUAL_WEIGHT_BREADTH_NOT_INDEX_EPS", "fund_flow_included": False,
                   "revised_vendor_financials_included": False})
    write_json(CONFIG, config, exclusive=True)
    paths = [CONFIG, Path(__file__), ROOT / "tests/test_original_earnings_breadth_v1.py",
             ROOT / "docs/510300_ORIGINAL_EARNINGS_BREADTH_V1_PROTOCOL.md", ROOT / "docs/510300_FACTOR_SCOPE_CORRECTION_20260906.md",
             ROOT / "config/510300_research_authority_v6.json", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/policy_liquidity_quantity_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             PARENT / "features.parquet", PARENT / "input_receipt.json", ROOT / config["inputs"]["dividends"],
             FACTS, FACT_RECEIPT, QUEUE, EVENTS, MEMBERS,
             ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_6.yaml",
             ROOT / "reports/data_quality/510300_csi300_pit_membership_2015_extension_v1.json",
             ROOT / "reports/data_quality/A_SHARE_HS_CNINFO_PERIODIC_REPORT_EVENT_ARCHIVE_V2_1.json"]
    paths += [OUT / x for x in ("source_receipt.json", "paired_original_events.parquet", "daily_earnings_breadth.parquet",
                                "daily_member_facts.parquet", "selected_verified_facts.parquet", "original_event_subset.parquet")]
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [ident(p) for p in paths],
                          "own_returns_read_before_freeze": False, "underlying_history_previously_observed": True}, exclusive=True)
    print(json.dumps({"状态": "第八轮已经冻结", "清单哈希": sha(MANIFEST)}, ensure_ascii=False), flush=True)


def train(data: pd.DataFrame, pc: list[str], dividends: pd.DataFrame, config: dict) -> dict:
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0]) - 1
    quarters = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [first] + [t for t in range(first + 1, len(data) - 1) if quarters[t] != quarters[t - 1]] + [len(data) - 1]
    labels = {}
    for h in (20, 60):
        y = np.full(len(data), np.nan)
        for t in range(len(data) - h - 1):
            y[t] = holding_total_return(data, dividends, t + 1, t + h + 1)[0]
        labels[h] = y
    predictions, receipts = {}, []
    (OUT / "final_models").mkdir(exist_ok=True)
    for item in config["models"]:
        h = item["horizon"]
        cols = columns(item["group"], pc)
        x, y = data[cols].to_numpy(float), labels[h]
        pred = np.full(len(data), np.nan)
        model = None
        for t, end in zip(cuts[:-1], cuts[1:]):
            ix = eligible_training(data, t, h, None)
            row = {"model": item["id"], "horizon": h, "fit_origin": data.date.iloc[t], "train_rows": len(ix), "feature_columns": cols}
            if len(ix) < config["minimum_train_samples"]:
                row["status"] = "NO_VIEW_INSUFFICIENT_MATURE_TRAINING_ROWS"
                receipts.append(row)
                continue
            model = model_for(item["kind"], config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d")))
            eligible = np.arange(t, end)[data.feature_valid.iloc[t:end].to_numpy()]
            with threadpool_limits(limits=1):
                model.fit(x[ix], y[ix])
                if len(eligible):
                    pred[eligible] = model.predict(x[eligible])
            row.update({"status": "FITTED", "first_train_origin": data.date.iloc[ix[0]], "last_train_origin": data.date.iloc[ix[-1]],
                        "last_label_exit": data.date.iloc[ix[-1] + h + 1],
                        "training_array_sha256": hashlib.sha256(x[ix].tobytes() + y[ix].tobytes()).hexdigest()})
            receipts.append(row)
        predictions[item["id"]] = pred
        if model is not None:
            joblib.dump({"model": model, "columns": cols, "specification": item, "last_fit_receipt": receipts[-1]},
                        OUT / "final_models" / (item["id"] + ".joblib"), compress=3)
        print(f"第八轮模型已完成：{item['id']}", flush=True)
    for key, (group, h) in ENSEMBLES.items():
        predictions[key] = np.mean(np.column_stack([predictions[f"{group}_{kind}_H{h}"] for kind in ("RIDGE", "ET")]), axis=1)
    pd.DataFrame(receipts).to_json(OUT / "training_receipts.json", orient="records", date_format="iso", force_ascii=False, indent=2)
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame({"date": data.date, **{f"Y{h}": y for h, y in labels.items()}}).to_parquet(OUT / "labels.parquet", index=False)
    return predictions


def run(expected: str) -> None:
    require(sha(MANIFEST) == expected, "第八轮清单哈希不符")
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], "冻结文件变化：" + item["path"])
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data, pc = features()
    data.to_parquet(OUT / "features.parquet", index=False)
    write_json(OUT / "feature_receipt.json", {"groups": {g: columns(g, pc) for g in GROUPS}, "valid_origins": int(data.feature_valid.sum()),
               "no_view_evaluation_origins": data.loc[(data.date >= "2019-12-31") & (data.date < config["data_cutoff"]) & ~data.feature_valid, "date"].tolist(),
               "source_clock": config["information_clock"], "equal_weight_breadth_is_index_eps": False})
    dividends = normalize_dividends(pd.read_csv(physical(ROOT / config["inputs"]["dividends"])))
    predictions = train(data, pc, dividends, config)
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for costid, cost in config["costs"].items():
        accounts = {}
        for key in [*predictions, "BUY_HOLD"]:
            h = 60 if key == "BUY_HOLD" else int(key.rsplit("H", 1)[-1])
            ledger, decisions = simulate_policy(data, dividends, config, cost, key, predictions.get(key), h)
            accounts[key] = ledger
            save_account(OUT / "evaluation" / costid, key, ledger, decisions)
        baseline = summarize(accounts["BUY_HOLD"], config)
        for key, ledger in accounts.items():
            result = {"cost": costid, "model": key, **summarize(ledger, config)}
            result["annualized_return_excess_vs_buy_hold"] = result["annualized_return"] - baseline["annualized_return"]
            result["meets_point_target"] = result["net_sharpe"] is not None and result["net_sharpe"] >= 1.2
            metrics.append(result)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": costid, "model": key, "year": int(year), **summarize(part, config)})
            for era, start, end in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", config["data_cutoff"])):
                eras.append({"cost": costid, "model": key, "era": era, **summarize(ledger.loc[(ledger.date >= start) & (ledger.date <= end)], config)})
        rr = pd.DataFrame({"date": accounts["BUY_HOLD"].date, **{k: v.net_return.to_numpy() for k, v in accounts.items()}})
        rr.to_parquet(OUT / f"{costid}_all_evaluation_returns.parquet", index=False)
        rng = np.random.default_rng(config["random_seed"])
        samples = {"primary_sharpe": [], "increment_vs_price": [], "increment_vs_earnings": []}
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(rr), config["bootstrap_day_block"])
            ret = rr[config["primary"]].to_numpy()[ix]
            samples["primary_sharpe"].append(return_metrics(ret, config["annual_days"])["net_sharpe"])
            for name, key in (("price", "F3_PRICE_H60"), ("earnings", "F2_EARNINGS_H60")):
                samples[f"increment_vs_{name}"].append(float((ret - rr[key].to_numpy()[ix]).mean() * config["annual_days"]))
        uncertainty[costid] = {k + "_95_interval": interval(v) for k, v in samples.items()}
        print(f"第八轮 {costid} 的17个完整账户已完成", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    primary = frame.loc[frame.model == config["primary"]]
    reached = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(), "number_of_candidates": 16,
              "evaluation_accounts": 34, "trained_model_count": 12, "uncertainty": uncertainty,
              "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "position_impact": 0,
              "fundamental_scope": "原始累计财报的等权改善广度，不等于指数EPS、公募现金申赎或修订后财报"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建原始财报改善广度，冻结并运行完整账户")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(sum((args.prepare, args.freeze, args.run)) == 1, "请选择来源准备、冻结、运行之一")
    if args.prepare:
        prepare()
    elif args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "缺少冻结清单哈希")
        run(args.expected_manifest_sha256)
