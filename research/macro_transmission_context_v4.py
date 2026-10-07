"""把资金来源、需求、定价和风险接成同一时点的证据链；不拟合策略。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_macro_transmission_context_v4"
R1 = "reports/research/510300_macro_volatility_observation_v2_run1"
R2 = "reports/research/510300_macro_volatility_mechanisms_v2_run2"
R3 = "reports/research/510300_volatility_money_decomposition_v3"
TZ = ZoneInfo("Asia/Shanghai")
INPUTS = {
    "monthly.csv": f"{R3}/results/波动分解_104个月_完整观察.csv",
    "weekly.csv": f"{R3}/results/波动分解_固定周度_完整观察.csv",
    "daily_vol.csv": f"{R3}/results/每日波动精确分解.csv",
    "market.csv": f"{R1}/inputs/market_daily.csv",
    "money_decomposition.csv": f"{R2}/results/104个月_余额基数分解_不含未来标签.csv",
    "money_causes.csv": f"{R1}/results/104个月_原因证据台账.csv",
    "loans.parquet": "reports/research/510300_rmb_loan_composition_completion_v1/released_loan_composition.parquet",
    "loan_originals.json": "reports/research/510300_rmb_loan_composition_completion_v1/extracted_originals.json",
    "deposits.parquet": "reports/research/510300_household_bank_balance_source_v1/released_household_bank_balance.parquet",
    "tsf_stock.parquet": "reports/research/510300_tsf_government_composition_source_v1/results/released_composition.parquet",
    "pmi_orders.parquet": "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet",
    "pmi_prices.parquet": "reports/research/510300_manufacturing_price_transmission_source_v1/released_manufacturing_price_transmission.parquet",
    "housing.parquet": "reports/research/510300_housing_collateral_monthly_source_v1/released_housing_breadth.parquet",
    "banker.parquet": "reports/research/510300_banker_survey_field_admission_v1/released_banker_survey.parquet",
    "entrepreneur.parquet": "reports/research/510300_entrepreneur_cash_field_admission_v1/released_entrepreneur_cash.parquet",
    "industrial_profit.json": "reports/research/510300_industrial_profit_response_v1/released_profit_records.json",
    "valuation.parquet": "data/curated/510300_stress_transmission_hazard_v2_mft_feature_execution_v1/csi300_earnings_yield_release_ledger.parquet",
    "bond_yield.parquet": "data/curated/510300_stress_transmission_hazard_v2_mft_feature_execution_v1/china_10y_yield_release_ledger.parquet",
    "policy_rate.parquet": "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet",
    "policy_facts.csv": "reports/research/510300_equity_support_policy_event_v1/inputs/policy_facts.csv",
}
NEW_SOURCES = {
    "pboc_20250714": "https://www.pbc.gov.cn/hanglingdao/128697/5580410/5580420/2025100917113619740/index.html",
    "tsf_202507": "https://app.www.gov.cn/govdata/gov/202508/13/534663/article.html",
    "tsf_202508": "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/523c260b344c4f1390664430295064a9/index.html",
}
LOAN_FIELDS = ["rmb_total", "household_total", "household_short", "household_long", "corporate_total", "corporate_short", "corporate_long", "bills", "nonbank_total"]


def now():
    return datetime.now(TZ).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def save(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame, name):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def load(name):
    p = OUT / "inputs" / name
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return json.loads(p.read_text(encoding="utf-8"))


def local_time(values):
    def one(value):
        t = pd.Timestamp(value)
        if pd.isna(t):
            return pd.NaT
        return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)
    return pd.Series([one(v) for v in values], index=values.index if isinstance(values, pd.Series) else None, dtype="datetime64[ns, Asia/Shanghai]")


def prepare():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == freeze["protocol_sha256"]
    receipts = []
    for name, source in INPUTS.items():
        p, dst = ROOT / source, OUT / "inputs" / name
        if dst.exists():
            assert digest(dst) == digest(p), "已有输入发生变化：" + name
        else:
            shutil.copy2(p, dst)
        receipts.append({"input": name, "original": source, "sha256": digest(dst), "bytes": dst.stat().st_size})
    save(OUT / "input_receipts.json", {"at": now(), "sources": receipts})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"已固定{len(receipts)}项本地输入，行情截至2026-09-11。")


def fetch_one(item):
    key, url = item
    path, receipt = OUT / "sources" / (key + ".html"), OUT / "sources" / (key + ".json")
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    proc = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10", "--max-time", "35", "--output", str(path), "--write-out", "%{http_code}", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=40)
    ok = proc.returncode == 0 and proc.stdout.strip() == "200"
    r = {"key": key, "url": url, "retrieved_at": now(), "http": proc.stdout.strip(), "status": "FETCHED" if ok else "NOT_FETCHED", "raw_path": str(path.relative_to(ROOT)), "sha256": digest(path) if ok else None, "error": proc.stderr[:400], "historical_first_vintage_verified": False}
    save(receipt, r)
    return r


def fetch():
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(fetch_one, NEW_SOURCES.items()))
    save(OUT / "sources" / "source_receipts.json", receipts)
    for r in receipts:
        print(r["key"], r["status"])


def credit_panel():
    x = load("loans.parquet").sort_values("stat_month").copy()
    originals = load("loan_originals.json")
    errors, prior_errors = {}, {}
    for r in originals:
        if r["stat_month"].endswith("-01"):
            prior_errors = {}
        current = {}
        for field in LOAN_FIELDS:
            atom = r["fields"][field]
            if atom is None:
                value = np.nan
            elif r["reported_interval"] == "YEAR_TO_DATE" or r["stat_month"].endswith("-01"):
                value = atom["display_rounding_half_yi"]
            else:
                value = atom["display_rounding_half_yi"] + prior_errors.get(field, np.nan)
            current[field] = value
            errors[(r["stat_month"], field)] = value
        prior_errors = current
    by_month = x.set_index("stat_month")
    for field in LOAN_FIELDS:
        x[field + "_ytd_rounding_half_yi"] = [errors[(m, field)] for m in x.stat_month]
        differences, bounds, statuses = [], [], []
        for r in x.to_dict("records"):
            last_year = str(pd.Period(r["stat_month"], freq="M") - 12)
            if last_year not in by_month.index:
                diff, bound, status = np.nan, np.nan, "NO_VIEW_NO_PRIOR_YEAR"
            elif r["statistical_regime"] != by_month.loc[last_year, "statistical_regime"]:
                diff, bound, status = np.nan, np.nan, "NO_VIEW_2023_SCOPE_BREAK"
            else:
                diff = r[field + "_ytd_yi"] - by_month.loc[last_year, field + "_ytd_yi"]
                bound = errors[(r["stat_month"], field)] + errors[(last_year, field)]
                status = "多增" if diff > bound else "少增" if diff < -bound else "舍入界内" if np.isfinite(diff + bound) else "NO_VIEW_MISSING_TERM"
            differences.append(diff)
            bounds.append(bound)
            statuses.append(status)
        x[field + "_ytd_yoy_change_yi"] = differences
        x[field + "_ytd_yoy_rounding_bound_yi"] = bounds
        x[field + "_ytd_yoy_direction"] = statuses
    # 同一金融统计原文与原M1/M2记录哈希一致，复用原已确认公布上界。
    money = load("monthly.csv").set_index("stat_month")
    assert all(x.source_sha256.to_numpy() == money.loc[x.stat_month, "source_sha256"].to_numpy())
    x["known_at"] = local_time(money.loc[x.stat_month, "available_at_upper_bound"].to_numpy()).to_numpy()
    csv(x, "104个月_信贷分项与同区间比较.csv")
    return x


def join_channel(base, channel, name, period, clock, fields, age_limit, age_on=None):
    q = channel.copy()
    q["_clock"] = local_time(q[clock])
    q = q.sort_values(["_clock", period]).drop_duplicates("_clock", keep="last")
    snap = local_time(base.snapshot_at)
    original_order = base.index
    values = []
    for i, t in snap.items():
        row = {f"{name}_status": "NO_VIEW_NO_PRIOR_RELEASE"}
        available = q[q._clock.le(t)] if pd.notna(t) else q.iloc[0:0]
        if len(available):
            z = available.iloc[-1]
            origin = local_time([z[age_on]])[0] if age_on else z._clock
            age = (t.normalize() - origin.normalize()).days
            valid = age <= age_limit
            row.update({f"{name}_period": str(z[period]), f"{name}_known_at": z._clock.isoformat(), f"{name}_age_days": age, f"{name}_status": "AVAILABLE_RECONSTRUCTED" if valid else "NO_VIEW_STALE_SOURCE"})
            for field in fields:
                row[f"{name}_{field}"] = z[field] if valid else np.nan
            for src in ["source_url", "url", "raw_path", "source_hash", "raw_sha256", "source_sha256", "historical_first_vintage_verified", "source_identity", "availability_rule"]:
                if src in q.columns:
                    row[f"{name}_{src}"] = z[src]
        values.append(row)
    return pd.concat([base, pd.DataFrame(values, index=original_order)], axis=1)


def orders_channel():
    q = load("pmi_orders.parquet").copy()
    q["known_at"] = local_time(q.available_at)
    p = load("pmi_prices.parquet")
    extras = p[~p.stat_month.isin(q.reference_period)]
    parsed = []
    for r in extras.to_dict("records"):
        raw = ROOT / r["raw_path"]
        assert digest(raw) == r["raw_sha256"]
        soup = BeautifulSoup(raw.read_bytes(), "html.parser")
        year, month = map(int, r["stat_month"].split("-"))
        target = f"{year}年{month}月"
        values = []
        for table in soup.find_all("table"):
            rows = [[re.sub(r"\s+", "", cell.get_text()) for cell in tr.find_all(["td", "th"], recursive=False)] for tr in table.find_all("tr")]
            if not any("PMI" in row for row in rows):
                continue
            headers = [row for row in rows if "新订单" in row and "生产" in row]
            if not headers:
                continue
            assert len(headers) == 1
            chosen = [row for row in rows if row and row[0] == target]
            assert len(chosen) == 1
            offset = len(chosen[0]) - len(headers[0])
            assert offset in [0, 2], "仅接受完整表头或日期和PMI跨行的两格偏移"
            values.append(float(chosen[0][offset + headers[0].index("新订单")]))
        unique = set(values)
        assert len(unique) == 1, "新订单当月值未唯一匹配：" + r["stat_month"]
        parsed.append({"reference_period": r["stat_month"], "published_at": r["published_at"], "known_at": r["known_at"], "first_release_value": unique.pop(), "source_url": r["url"], "raw_path": r["raw_path"], "source_hash": r["raw_sha256"]})
    if parsed:
        q = pd.concat([q, pd.DataFrame(parsed)], ignore_index=True)
    q = q.sort_values("reference_period").reset_index(drop=True)
    q["orders_change1_pp"] = q.first_release_value.diff()
    q["known_at"] = local_time(q.known_at)
    save(OUT / "results" / "新订单既有原文补全.json", parsed)
    return q


def with_prior_differences(frame, period, clock, fields, lags):
    q = frame.copy().sort_values(period).reset_index(drop=True)
    times = local_time(q[clock])
    for field in fields:
        for lag in lags:
            diff = q[field] - q[field].shift(lag)
            # 后来补发的旧季不能作为更早时点已知的比较基准。
            diff = diff.where(times.ge(times.shift(lag)))
            q[f"{field}_change{lag}_pp"] = diff
    return q


def price_context():
    m = load("market.csv")
    w = m.wealth
    p = pd.DataFrame({"observation_date": m.date})
    p["price_return252"] = w / w.shift(252) - 1
    high, low, ma = w.rolling(252).max(), w.rolling(252).min(), w.rolling(200).mean()
    p["price_position252"] = (w - low) / (high - low)
    p["price_drawdown252"] = w / high - 1
    p["price_ma200_gap"] = w / ma - 1
    p["price_ma200_slope20"] = ma / ma.shift(20) - 1
    p["price_path_description"] = np.select([
        (p.price_ma200_gap >= 0) & (p.price_ma200_slope20 >= 0),
        (p.price_ma200_gap < 0) & (p.price_ma200_slope20 < 0),
        (p.price_ma200_gap >= 0) & (p.price_ma200_slope20 < 0),
        (p.price_ma200_gap < 0) & (p.price_ma200_slope20 >= 0),
    ], ["价格在上行200日均线之上", "价格在下行200日均线之下", "价格已回到仍下行均线之上", "价格落在仍上行均线之下"], default="NO_VIEW")
    return p


def joint_state(r):
    if not np.isfinite(r.get("delta3_spread_pp", np.nan)):
        return "剪刀差变化缺失"
    if r["delta3_spread_pp"] <= 0:
        return "剪刀差未改善"
    co = r.get("loan_corporate_long_ytd_yoy_direction")
    hh = r.get("loan_household_long_ytd_yoy_direction")
    order = r.get("orders_first_release_value", np.nan)
    if co not in ["多增", "少增", "舍入界内"] or hh not in ["多增", "少增", "舍入界内"] or not np.isfinite(order):
        return "剪刀差改善_信贷或订单证据缺失"
    if co == hh == "多增" and order >= 50:
        return "剪刀差改善_信贷与订单共同支持"
    if co == hh == "少增" and order < 50:
        return "剪刀差改善_三项未支持"
    return "剪刀差改善_信贷订单分化或舍入边界"


def compose(origins, loans, order):
    labels = [c for c in origins.columns if c.startswith("E0_") or c.startswith("E1_") or c.startswith("delay_change_")]
    x = origins.drop(columns=labels).copy()
    source_index = x.index
    money = load("money_decomposition.csv")
    add = [c for c in money if c not in x or c == "stat_month"]
    x = x.merge(money[add], on="stat_month", how="left", validate="many_to_one")
    x = x.merge(price_context(), on="observation_date", how="left", validate="many_to_one")
    fields = ["rmb_stock_yoy_percent", "statistical_regime", "reported_interval", "cumulative_source_months"]
    fields += [c for c in loans if any(c.startswith(v + "_") for v in LOAN_FIELDS) and c not in fields]
    x = join_channel(x, loans, "loan", "stat_month", "known_at", fields, 62)
    d = load("deposits.parquet").copy()
    d["known_at"] = local_time(load("monthly.csv").set_index("stat_month").loc[d.stat_month, "available_at_upper_bound"].to_numpy()).to_numpy()
    x = join_channel(x, d, "deposit", "stat_month", "known_at", ["household_deposit_ytd_yi", "household_loan_ytd_yi", "household_bank_flow_gap_ytd_yi", "rmb_deposit_stock_yoy_percent"], 62)
    x = join_channel(x, load("tsf_stock.parquet"), "tsf", "reference_period", "conservative_known_at", ["share_percent", "government_share_yoy_change_pp", "total_tsf_yoy_percent"], 62)
    x = join_channel(x, order, "orders", "reference_period", "known_at", ["first_release_value", "orders_change1_pp"], 62)
    x = join_channel(x, load("pmi_prices.parquet"), "manufacturing", "stat_month", "known_at", ["input_price_diffusion", "output_price_diffusion", "output_minus_input_diffusion_pp", "method_group"], 62)
    x = join_channel(x, load("housing.parquet"), "housing", "stat_month", "known_at", ["second_hand_down_count", "second_hand_down_share"], 62)
    b = with_prior_differences(load("banker.parquet"), "quarter", "known_at", ["loan_demand_index", "loan_approval_index", "monetary_policy_perception_index"], [1, 4])
    x = join_channel(x, b, "banker", "quarter", "known_at", [c for c in b if c in ["loan_demand_index", "loan_approval_index", "monetary_policy_perception_index"] or "_change" in c], 120)
    e = with_prior_differences(load("entrepreneur.parquet"), "quarter", "known_at", ["sales_revenue_collection_index", "fund_turnover_index"], [1, 4])
    x = join_channel(x, e, "entrepreneur", "quarter", "known_at", [c for c in e if c in ["sales_revenue_collection_index", "fund_turnover_index"] or "_change" in c], 120)
    x = join_channel(x, pd.DataFrame(load("industrial_profit.json")), "profit", "stat_month", "known_at", ["profit_ytd_reported_yoy_pct", "receivable_collection_days", "receivable_days_published_yoy_change"], 62)
    x = join_channel(x, load("valuation.parquet"), "valuation", "observation_date", "available_at", ["original_pe_official", "csi300_earnings_yield"], 5, "observation_date")
    x = join_channel(x, load("bond_yield.parquet"), "bond", "observation_date", "available_at", ["china_10y_yield"], 5, "observation_date")
    x["earnings_yield_minus_bond_pp"] = x.valuation_csi300_earnings_yield - x.bond_china_10y_yield
    rates = load("policy_rate.parquet")
    x = join_channel(x, rates, "policy_rate", "notice_date", "published_at", ["seven_day_rate_percent"], 10000)
    outside = pd.to_datetime(x.observation_date).gt("2026-08-14")
    x.loc[outside, "policy_rate_status"] = "NO_VIEW_BEYOND_CATALOGUE_COVERAGE"
    x.loc[outside, "policy_rate_seven_day_rate_percent"] = np.nan
    facts = load("policy_facts.csv")
    facts["known_at"] = local_time(facts.source_available_upper)
    event_records = []
    for r in x.to_dict("records"):
        t = local_time([r["snapshot_at"]])[0]
        sub = facts[facts.known_at.le(t) & facts.known_at.gt(t - pd.Timedelta(days=31))] if pd.notna(t) else facts.iloc[0:0]
        event_records.append({"policy_recent_nodes": "|".join(sub.node_id), "policy_recent_titles": "；".join(sub.title), "policy_catalogue_status": "近31日有已覆盖事实" if len(sub) else "不完整目录内未匹配_不能认定无政策"})
    x = pd.concat([x, pd.DataFrame(event_records)], axis=1)
    x["joint_credit_orders_state"] = x.apply(joint_state, axis=1)
    x["economic_cause_unique"] = "NOT_IDENTIFIED_MULTIPLE_CHANNELS"
    x["remaining_upside"] = "NOT_IDENTIFIED_NO_EXPECTATION_AND_INDEX_EARNINGS_BRIDGE"
    # 截止日之后原点整体缺失，不能利用后来公告填成历史可见状态。
    out_of_snapshot = pd.to_datetime(x.observation_date).isna() | pd.to_datetime(x.observation_date).gt("2026-09-11")
    context = [c for c in x if c not in origins and c not in ["stat_month"]]
    for c in context:
        if c.endswith("_status"):
            x.loc[out_of_snapshot, c] = "NO_VIEW_AFTER_MARKET_CUTOFF"
        else:
            x[c] = x[c].where(~out_of_snapshot, pd.NA)
    x["context_snapshot_status"] = np.where(out_of_snapshot, "NO_VIEW_AFTER_MARKET_CUTOFF", "RECONSTRUCTED_MULTI_SOURCE_CONTEXT")
    x.index = source_index
    return x, labels


def source_coverage(x, kind):
    rows = []
    for name in ["loan", "deposit", "tsf", "orders", "manufacturing", "housing", "banker", "entrepreneur", "profit", "valuation", "bond", "policy_rate"]:
        for status, part in x.groupby(name + "_status", dropna=False):
            rows.append({"panel": kind, "channel": name, "status": status, "rows": len(part), "money_cycles": part.stat_month.nunique(), "source_periods": part[name + "_period"].nunique()})
    return rows


def summarize(x, kind):
    groups = []
    for (regime, joint), part in x.groupby(["training_regime", "joint_credit_orders_state"], dropna=False):
        periods = [("全期", part)] + list(part.groupby("analysis_period"))
        for period, p in periods:
            for variant in ["E0", "E1"]:
                prefix = variant + "_20"
                valid = p[p[prefix + "_return"].notna()]
                if not len(valid):
                    continue
                weights = 1 / valid.groupby("stat_month").stat_month.transform("size") if kind == "周度" else np.ones(len(valid))
                row = {"panel": kind, "regime": regime, "period": period, "joint_state": joint, "variant": variant, "rows": len(valid), "money_cycles": valid.stat_month.nunique(), "independent_validation": False}
                for col in [prefix + "_return", prefix + "_worst_path", prefix + "_future_rv", prefix + "_future_downside", "past_return20", "past_return60", "internal_breadth20", "price_position252", "valuation_original_pe_official"]:
                    mask = valid[col].notna()
                    row[col + "_mean"] = np.average(valid.loc[mask, col], weights=np.asarray(weights)[mask]) if mask.any() else np.nan
                row["positive_fraction"] = np.average(valid[prefix + "_return"].gt(0), weights=weights)
                row["worst_single_path"] = valid[prefix + "_worst_path"].min()
                groups.append(row)
    return groups


def flow_cases():
    specs = [
        ("2025-06", "pboc_20250714", "2025-07-14T18:28:37+08:00", 228300, 47400, 76600, 43200, 127400, 2796),
        ("2025-07", "tsf_202507", "2025-08-13T23:59:59+08:00", 239900, 51200, 89000, 48800, 123100, -694),
        ("2025-08", "tsf_202508", "2025-09-12T17:00:00+08:00", 265600, 46600, 102700, 46300, 129300, -4851),
    ]
    rows = []
    for month, key, clock, total, total_delta, gov, gov_delta, credit, credit_delta in specs:
        receipt = json.loads((OUT / "sources" / (key + ".json")).read_text(encoding="utf-8"))
        if receipt["status"] != "FETCHED":
            rows.append({"stat_month": month, "status": "NO_VIEW_SOURCE_NOT_SAVED", "source_url": NEW_SOURCES[key]})
            continue
        soup = BeautifulSoup((OUT / "sources" / (key + ".html")).read_bytes(), "html.parser")
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", soup.get_text(" ", strip=True)))
        checks = {
            "2025-06": ["22.83万亿元", "4.74万亿元", "7.66万亿元", "4.32万亿元", "12.74万亿元", "2796亿元"],
            "2025-07": ["23.99万亿元", "5.12万亿元", "8.9万亿元", "4.88万亿元", "12.31万亿元", "少增694亿元"],
            "2025-08": ["26.56万亿元", "4.66万亿元", "10.27万亿元", "4.63万亿元", "12.93万亿元", "少增4851亿元"],
        }[month]
        assert all(s in compact for s in checks), month + "社融原文与抄录未匹配"
        rows.append({"stat_month": month, "status": "VERIFIED_SAVED_OFFICIAL_CURRENT_PAGE", "known_at": clock, "source_url": NEW_SOURCES[key], "raw_sha256": receipt["sha256"], "total_tsf_ytd_yi": total, "total_tsf_ytd_yoy_increase_yi": total_delta, "government_bond_ytd_yi": gov, "government_bond_ytd_yoy_increase_yi": gov_delta, "rmb_to_real_economy_ytd_yi": credit, "rmb_to_real_economy_ytd_yoy_increase_yi": credit_delta, "nongovernment_tsf_yoy_increase_yi": total_delta - gov_delta, "government_share_of_tsf_yoy_increase": gov_delta / total_delta, "interpretation": "年初累计同比多增的算术构成；不是当月融资占比，不是私人部门占比或股票流入。", "historical_first_vintage_verified": False, "checks": "|".join(checks)})
    csv(pd.DataFrame(rows), "2025年6至8月_社融多增来源.csv")


def build():
    assert not (OUT / "results" / "build_receipt.json").exists(), "本轮已完成，不覆盖结果。"
    loans, orders = credit_panel(), orders_channel()
    coverage, summaries, receipts = [], [], []
    for kind, name in [("月度", "monthly.csv"), ("周度", "weekly.csv")]:
        parent = load(name)
        context, label_cols = compose(parent, loans, orders)
        dest = "104个月" if kind == "月度" else "445周"
        csv(context, dest + "_当时可见多层证据_不含未来标签.csv")
        before_hash = digest(OUT / "results" / (dest + "_当时可见多层证据_不含未来标签.csv"))
        full = pd.concat([context, parent[label_cols]], axis=1)
        csv(full, dest + "_多层证据与原后续路径.csv")
        coverage += source_coverage(full, kind)
        summaries += summarize(full, kind)
        receipts.append({"panel": kind, "rows": len(full), "context_columns": len(context.columns), "future_label_columns_reused": len(label_cols), "context_written_before_labels_sha256": before_hash})
        if kind == "月度":
            chosen = full[full.stat_month.between("2020-04", "2021-02") | full.stat_month.between("2024-04", "2025-08")]
            csv(chosen, "两个固定历史段_完整28个月.csv")
            cards = full[full.stat_month.isin(["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"])]
            csv(cards, "五个已知案例_完整证据.csv")
    flow_cases()
    csv(pd.DataFrame(coverage), "各层证据覆盖与缺失.csv")
    csv(pd.DataFrame(summaries), "信贷订单联合场景_全部历史分布.csv")
    save(OUT / "results" / "build_receipt.json", {"at": now(), "status": "MULTI_SOURCE_CONTEXT_BUILT_NOT_A_PREDICTION_MODEL", "panels": receipts, "new_models": 0, "new_accounts": 0, "historical_first_vintage_verified": False, "goal_status": "active", "goal_achieved": False})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("104个月与445周多层证据已对齐，未来路径沿用原标签；未拟合账户或收益模型。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "fetch", "build"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "build": build}[args.action]()
