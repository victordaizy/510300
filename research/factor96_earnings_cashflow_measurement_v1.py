"""从保存的公告事实复算单季盈利意外和滚动四季现金质量，不读取价格或收益。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
STUDY = "510300_FACTOR96_EARNINGS_CASHFLOW_MEASUREMENT_V1"
FACT_BASE = "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
SOURCES = {
    "inputs/original_facts.parquet": FACT_BASE + "official_financial_facts_v1_21_document_by_document.parquet",
    "inputs/original_receipt.json": FACT_BASE + "receipt_v1_21_document_by_document.json",
    "inputs/document_queue.parquet": "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1/official_fact_document_queue_v1_21_document_by_document.parquet",
    "inputs/report_events.parquet": "data/raw/cninfo/a_share_hs_periodic_report_events_v2_1.parquet",
    "inputs/membership.parquet": "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet",
    "inputs/industry_intervals.parquet": "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/industry_l1_intervals.parquet",
    "inputs/calendar.parquet": "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/trading_calendar_observed_open_days.parquet",
    "source_evidence/original_extraction_contract.yaml": "config/csi300_pit_fundamental_underreaction_official_facts_v1_6.yaml",
    "source_evidence/original_parent_contract.yaml": "config/csi300_pit_fundamental_underreaction_enhancement_v1.yaml",
    "source_evidence/old_earnings_config.json": "config/510300_original_earnings_breadth_v1.json",
    "source_evidence/old_earnings_source_receipt.json": "reports/research/510300_original_earnings_breadth_v1/source_receipt.json",
    "source_evidence/old_earnings_code.py": "research/original_earnings_breadth_v1.py",
    "source_evidence/secondary_aggregator_limitation.json": "reports/data_quality/510300_structural_equity_risk_premium_engine_v1_statement_collection.json",
    "source_evidence/mandate_before.json": "config/510300_existing_data_training_mandate_v1.json",
    "source_evidence/factor_registry.json": "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json",
    "source_evidence/strategy_registry.json": "reports/research/510300_factor96_mechanism_batch_v1/strategy_registry.json",
}
METRICS = ["PARENT_NET_PROFIT_YTD", "OPERATING_CASH_FLOW_YTD", "TOTAL_ASSETS_END"]
FINANCIAL = {"801780.SI", "801790.SI"}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while piece := stream.read(1024 * 1024):
            h.update(piece)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat() if pd.notna(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def amount(raw, multiplier):
    value = unicodedata.normalize("NFKC", str(raw)).replace(",", "").replace(" ", "").replace("−", "-")
    if value.startswith("["):
        return np.nan
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        return float(value) * float(multiplier)
    except (ValueError, TypeError):
        return np.nan


def next_open(dates, sessions, extra=0):
    dates = pd.to_datetime(dates)
    positions = sessions.searchsorted(dates, side="right") + extra
    result = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = (positions < len(sessions)) & (dates.to_numpy() >= sessions[0].to_datetime64())
    result[ok] = sessions[positions[ok]].to_numpy(dtype="datetime64[ns]")
    return pd.Series(result, index=dates.index)


def quarter_profit(current_ytd, preceding_ytd, quarter):
    return current_ytd if quarter == 1 else current_ytd - preceding_ytd


def trailing_flow(current_ytd, preceding_fy, previous_same_ytd, quarter):
    return current_ytd if quarter == 4 else current_ytd + preceding_fy - previous_same_ytd


def seasonal_surprise(current, previous, previous2):
    values = np.array([current, previous, previous2], dtype=float)
    if not np.isfinite(values).all():
        return np.nan, np.nan, np.nan
    mean, sd = float(np.mean(values[1:])), float(np.std(values[1:], ddof=1))
    return ((current - mean) / sd if sd > 0 else np.nan), mean, sd


def industry_at(points, intervals):
    """只在保存的行业有效区间和标注可用时刻内匹配；多重匹配保留未知。"""
    tables = {}
    for symbol, block in intervals.groupby("ts_code", sort=False):
        tables[str(symbol)] = block.to_dict("records")
    rows = []
    for r in points[["ts_code", "date"]].itertuples(index=False):
        moment = pd.Timestamp(r.date) + pd.Timedelta(hours=9, minutes=30)
        matches = [v for v in tables.get(str(r.ts_code), []) if v["valid_from"] <= r.date
                   and (pd.isna(v["valid_to"]) or r.date <= v["valid_to"])
                   and v["available_at"] <= moment]
        code = str(matches[0]["industry_l1_code"]) if len(matches) == 1 else None
        rows.append((code, len(matches), code in FINANCIAL if code is not None else False))
    return pd.DataFrame(rows, columns=["industry_l1_code", "industry_match_count", "is_financial"], index=points.index)


def prepare_and_freeze():
    assert not OUT.exists(), "本轮目录已存在，禁止覆盖冻结文件"
    OUT.mkdir(parents=True)
    paths = dict(SOURCES)
    for path in (ROOT / "reports/research/510300_factor96_program_v1").iterdir():
        if path.is_file():
            paths["program_before/" + path.name] = path.relative_to(ROOT).as_posix()
    paths["code/" + Path(__file__).name] = Path(__file__).relative_to(ROOT).as_posix()
    paths["code/test_factor96_earnings_cashflow_measurement_v1.py"] = "tests/test_factor96_earnings_cashflow_measurement_v1.py"
    copies = []
    for name, source in paths.items():
        source = ROOT / source
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copies.append({"original": source.relative_to(ROOT).as_posix(), "snapshot": name,
                       "bytes": target.stat().st_size, "sha256": digest(target)})
    protocol = {
        "at": now(), "study_id": STUDY, "candidate": "T11", "phase": "MEASUREMENT_ONLY_NO_RETURNS",
        "question": "现有原始财报事实能否在当时可用日期计算L02单季盈利意外、L04真正滚动四季现金质量及其同比变化？",
        "scope": "不读价格、标签、策略收益；不形成入场或账户；仅非金融L02/L04联合测量，银行非银另记排除，不声称金融分支已实现。",
        "selection_before_values": "已读来源模式、原提取覆盖回执和旧代码；未读本轮财务数值或任何本轮收益。来源按既有公告事实档案选定，不按策略表现选择。",
        "quarter_profit": "Q1=本期归母净利润累计值；Q2/Q3/Q4=本期累计值-同年上一季度累计值；除以上一季期末总资产，资产必须正。",
        "L02": "设u_t=单季归母利润/上一季资产。z_t=(u_t-mean(u_{t-4},u_{t-8}))/std(u_{t-4},u_{t-8},ddof=1)。只用前两年同季，零波动或任一缺项保留未知；不裁剪、不加epsilon、不回填。",
        "L02_limit": "仅两个同季历史观测，标准差可能不稳定；这是简单历史意外，不是分析师共识。最早依赖t-9季度资产，原始特征计算历史不等于模型两年训练窗。",
        "ttm_flow": "Q4=全年累计；其他季度=本年累计+上年全年累计-上年同期累计。CFO和归母利润分别计算。禁止YTD乘4/季数冒充TTM。",
        "L04": "q_t=(CFO_TTM_t-归母利润_TTM_t)/总资产_t；保存q_t-q_{t-4}。本轮输出公司原始同比变化，不输出行业标准化后的ETF聚合交易因子。",
        "scope_mismatch": "CFO为合并经营现金流，利润为归母净利润；少数股东口径未另行校正，不能称标准总利润应计质量指标。各期采用原公告当期列，相邻报表重述/合并范围变化可能影响相减。",
        "known_before": "公告名义公开日之后的首个已覆盖A股交易日开盘起保守可用；日历覆盖以前的公开日未知。每个历史依赖可用日必须不晚于目标可用日。另存再等待一个完整交易日的时钟，不将报告期当可用日。",
        "source_identity": "原提取回执的事实/队列哈希匹配；公告标识、发行人、报告期、发布日期与元数据一致；原链接或同日全文替代链接、PDF哈希、URL日期及原文单位金额通过；身份不符行排除，不推测修正。",
        "source_revision_limit": "原始PDF值优先于当前聚合表；档案仍不能证明当年首次发布版本或first_seen。本轮不重抓或重新解析全部PDF。不能认定严格历史首次可得已完成。",
        "universe": "测量历史成员并集的报告，标记可用日是否为成员；事件统计主表只计当日成员。公司成为成员前的已披露报告也允许供其随后成员日使用。逐日另按当日成员匹配当时最新报告，缺最新字段不退回老报告。",
        "industry": "现有申万有效区间valid_from<=day<=valid_to且available_at<=09:30；唯一匹配才已知。银行801780.SI和非银801790.SI排除。available_at来自供应商生效日期代理，不是历史抓取证明。",
        "daily_age": "最新已公开报告的报告期距测量日0至200日；最新报告缺项即未知。只列完整率，不新增整体准入门槛或降低旧240家门槛。",
        "daily_frontier": "同公司同可用日留报告期最大者；迟到旧报告不能覆盖更晚报告期。依赖可同日已知；之后发布的前期报告不能被目标事件反向使用。",
        "old_overlap": "旧ORIGINAL_EARNINGS_BREADTH使用YTD同比与12/月年化质量；本轮改为单季分拆、两年前同季意外和真正TTM，仅测量新定义。旧失败、原提取整体覆盖失败与旧240家每日门槛均不改。",
        "O02": "NOT_COMPUTED。本轮不选择首日反应阈值、行业聚合、入场映射或训练规则；须在后续首次收益计算前单独固定。",
        "no_substitution": "不以东财聚合表、当年年报后来比较栏、未来修订表填缺口；不改变零波动和迟报规则来提高可算率。",
        "new_accounts": 0, "new_network_requests": 0, "goal_achieved": False,
        "orders_authorized": False, "external_review": "NOT_PERFORMED", "independent_forward_observations": 0,
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "source_manifest.json", {"at": now(), "sources": copies})
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "freeze.json", {"at": now(), "files": files})
    print("已冻结单季利润与TTM现金质量测量协议、原始档案及代码；未读取收益。", flush=True)


def verify_inputs(out):
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        path = out / row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], row["path"]
    receipt = json.loads((out / "inputs/original_receipt.json").read_text(encoding="utf-8"))
    assert digest(out / "inputs/original_facts.parquet") == receipt["artifacts"]["facts_sha256"]
    assert digest(out / "inputs/document_queue.parquet") == receipt["artifacts"]["document_queue_sha256"]
    assert receipt["status"] == "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE"
    return len(frozen["files"])


def validated_sources(out):
    all_facts = pd.read_parquet(out / "inputs/original_facts.parquet")
    facts = all_facts.loc[all_facts.metric_id.isin(METRICS)].copy()
    events = pd.read_parquet(out / "inputs/report_events.parquet")
    queue = pd.read_parquet(out / "inputs/document_queue.parquet")
    members = pd.read_parquet(out / "inputs/membership.parquet")
    industry = pd.read_parquet(out / "inputs/industry_intervals.parquet")
    calendar = pd.read_parquet(out / "inputs/calendar.parquet")
    assert not facts.duplicated(["announcement_id", "metric_id"]).any()
    assert not events.duplicated(["ts_code", "report_period"]).any()
    assert facts.verification_status.eq("PASS_OFFICIAL_ORIGINAL_PDF_LABEL_VALUE_UNIT_VERIFIED").all()
    members["membership_date"] = pd.to_datetime(members.membership_date).astype("datetime64[ns]")
    assert not members.duplicated(["membership_date", "symbol"]).any()
    assert members.groupby("membership_date").symbol.nunique().eq(300).all()
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(calendar.loc[calendar.is_open, "date"]).unique()))
    assert set(members.membership_date).issubset(set(sessions))
    for col in ["report_period", "event_publication_date"]:
        facts[col] = pd.to_datetime(facts[col]).astype("datetime64[ns]")
        events[col] = pd.to_datetime(events[col]).astype("datetime64[ns]")
    for col in ["valid_from", "valid_to", "available_at"]:
        industry[col] = pd.to_datetime(industry[col])
    events["announcement_id"] = events.announcement_id.astype(str)
    for table in [facts, queue]:
        table["announcement_id"] = table.announcement_id.astype(str)
    facts["official_pdf_announcement_id"] = facts.official_pdf_announcement_id.astype(str)
    keys = ["ts_code", "report_period", "event_publication_date"]
    joined = facts[["announcement_id", *keys]].drop_duplicates().merge(
        events[["announcement_id", *keys]], on="announcement_id", how="left", suffixes=("_fact", "_event"), validate="one_to_one")
    for key in keys:
        assert joined[key + "_fact"].eq(joined[key + "_event"]).all(), key
    facts = facts.merge(events[["announcement_id", "official_pdf_url"]].rename(columns={"official_pdf_url": "canonical_pdf_url"}),
                        on="announcement_id", how="left", validate="many_to_one")
    facts = facts.merge(queue[["announcement_id", "resolved_official_pdf_url", "official_pdf_sha256", "source_route"]]
                        .rename(columns={"official_pdf_sha256": "resolved_pdf_sha256"}), on="announcement_id", how="left", validate="many_to_one")
    primary = facts.official_pdf_url.eq(facts.canonical_pdf_url) & facts.announcement_id.eq(facts.official_pdf_announcement_id)
    sibling = (facts.official_pdf_url.eq(facts.resolved_official_pdf_url) & facts.official_pdf_sha256.eq(facts.resolved_pdf_sha256)
               & facts.source_route.eq("SAME_EVENT_OFFICIAL_FULL_PDF_SIBLING"))
    source_dates = pd.to_datetime(facts.official_pdf_url.str.extract(r"finalpage/(\d{4}-\d{2}-\d{2})/")[0])
    facts["source_identity_valid"] = (primary | sibling) & source_dates.eq(facts.event_publication_date)
    facts["amount_recomputed"] = [amount(v, u) for v, u in zip(facts.source_raw_value, facts.source_unit_multiplier)]
    finite = np.isfinite(facts.amount_recomputed)
    assert np.allclose(facts.loc[finite, "amount_recomputed"], facts.loc[finite, "metric_value_cny"], rtol=1e-12, atol=.011)
    assert facts.statement_scope.eq("CONSOLIDATED_ONLY").all()
    assert facts.loc[facts.metric_id != "TOTAL_ASSETS_END", "value_period_scope"].eq("YEAR_TO_DATE").all()
    assert facts.loc[facts.metric_id == "TOTAL_ASSETS_END", "value_period_scope"].eq("PERIOD_END").all()
    facts["verified_value"] = facts.amount_recomputed.where(facts.source_identity_valid)
    facts["fact_status"] = np.where(~facts.source_identity_valid, "SOURCE_IDENTITY_REJECTED",
                                    np.where(finite, "VERIFIED_SAVED_ORIGINAL_FACT", "NON_NUMERIC_OR_BLANK"))
    events = events.loc[events.ts_code.isin(set(members.symbol))].copy()
    events["available_date"] = next_open(events.event_publication_date, sessions)
    events["delayed_available_date"] = next_open(events.event_publication_date, sessions, extra=1)
    events["quarter_ordinal"] = events.report_period.dt.to_period("Q-DEC").astype("int64")
    assert events.report_period.dt.is_quarter_end.all()
    return facts, events, members, industry, {
        "all_original_fact_rows": len(all_facts), "selected_fact_rows": len(facts),
        "recomputed_finite_amounts": int(finite.sum()), "blank_amounts": int((~finite).sum()),
        "identity_excluded_fact_rows": int((~facts.source_identity_valid).sum()),
        "same_day_full_report_sibling_rows": int(sibling.sum()), "historical_membership_rows": len(members),
        "canonical_reports_in_historical_member_union": len(events), "observed_sessions": len(sessions),
    }


class FinancialDependencies:
    def __init__(self, events, facts):
        self.events = {(r["ts_code"], r["quarter_ordinal"]): r for r in events.to_dict("records")}
        self.facts = {(r["announcement_id"], r["metric_id"]): r for r in facts.to_dict("records")}
        self.ledger = []
        self.target = None

    def get(self, quarter, metric, role):
        target = self.target
        source = self.events.get((target["ts_code"], quarter))
        fact, status, value = None, "MISSING_CANONICAL_REPORT", np.nan
        if source is not None:
            fact = self.facts.get((source["announcement_id"], metric))
            if pd.isna(source["available_date"]) or source["available_date"] > target["available_date"]:
                status = "DISCLOSED_AFTER_TARGET"
            elif fact is None:
                status = "MISSING_ORIGINAL_FACT"
            else:
                status, value = fact["fact_status"], fact["verified_value"]
                if metric == "TOTAL_ASSETS_END" and status == "VERIFIED_SAVED_ORIGINAL_FACT" and value <= 0:
                    status, value = "NONPOSITIVE_ASSET", np.nan
            if status != "VERIFIED_SAVED_ORIGINAL_FACT":
                value = np.nan
        self.ledger.append({"target_announcement_id": target["announcement_id"], "ts_code": target["ts_code"],
            "target_report_period": target["report_period"], "target_available_date": target["available_date"],
            "role": role, "source_quarter_ordinal": quarter, "source_report_period": pd.Period(ordinal=quarter, freq="Q-DEC").end_time.normalize(),
            "source_announcement_id": source["announcement_id"] if source is not None else None,
            "source_available_date": source["available_date"] if source is not None else pd.NaT,
            "metric_id": metric, "value": value, "status": status})
        return value

    def single_quarter(self, quarter, role):
        q = pd.Period(ordinal=quarter, freq="Q-DEC").quarter
        current = self.get(quarter, "PARENT_NET_PROFIT_YTD", role + "_CURRENT_YTD")
        previous = self.get(quarter - 1, "PARENT_NET_PROFIT_YTD", role + "_PREVIOUS_YTD") if q != 1 else np.nan
        assets = self.get(quarter - 1, "TOTAL_ASSETS_END", role + "_PREVIOUS_ASSETS")
        profit = quarter_profit(current, previous, q)
        return profit, assets, profit / assets if np.isfinite(assets) and assets > 0 else np.nan

    def ttm(self, quarter, metric, role):
        q = pd.Period(ordinal=quarter, freq="Q-DEC").quarter
        current = self.get(quarter, metric, role + "_CURRENT_YTD")
        fy = self.get(quarter - q, metric, role + "_PREVIOUS_FY") if q != 4 else np.nan
        prior = self.get(quarter - 4, metric, role + "_PREVIOUS_SAME_YTD") if q != 4 else np.nan
        return trailing_flow(current, fy, prior, q)


def measure_events(facts, events, members, industry):
    member_keys = members[["membership_date", "symbol"]].rename(columns={"membership_date": "available_date", "symbol": "ts_code"})
    targets = events.loc[events.available_date.notna()].merge(member_keys.assign(member_at_available=True),
                    on=["available_date", "ts_code"], how="left", validate="many_to_one")
    targets["member_at_available"] = targets.member_at_available.eq(True)
    points = targets[["ts_code", "available_date"]].rename(columns={"available_date": "date"})
    targets = pd.concat([targets, industry_at(points, industry)], axis=1)
    dependencies = FinancialDependencies(events, facts)
    records = []
    for target in targets.sort_values(["available_date", "ts_code", "report_period"]).to_dict("records"):
        result = {k: target[k] for k in ["announcement_id", "ts_code", "report_period", "quarter_ordinal", "event_publication_date",
                                        "available_date", "delayed_available_date", "industry_l1_code", "industry_match_count", "is_financial", "member_at_available"]}
        numerical = ["quarter_profit", "previous_quarter_assets", "quarter_roa", "prior_same_quarter_roa", "prior2_same_quarter_roa",
                     "seasonal_mean", "seasonal_sd", "L02", "ttm_profit", "ttm_cashflow", "assets", "cash_quality",
                     "prior_ttm_profit", "prior_ttm_cashflow", "prior_assets", "prior_cash_quality", "L04_change"]
        result.update({name: np.nan for name in numerical})
        result.update({"L02_known": False, "L04_known": False, "joint_known": False})
        if target["industry_match_count"] != 1:
            result["measurement_status"] = "NO_VIEW_INDUSTRY"
        elif target["is_financial"]:
            result["measurement_status"] = "EXCLUDED_FINANCIAL_CFO_NOT_COMPARABLE"
        else:
            dependencies.target = target
            q = target["quarter_ordinal"]
            profit, assets_before, roa = dependencies.single_quarter(q, "L02_CURRENT")
            _, _, prior_roa = dependencies.single_quarter(q - 4, "L02_PRIOR_YEAR")
            _, _, prior2_roa = dependencies.single_quarter(q - 8, "L02_PRIOR_TWO_YEARS")
            surprise, historical_mean, historical_sd = seasonal_surprise(roa, prior_roa, prior2_roa)
            ttm_p = dependencies.ttm(q, "PARENT_NET_PROFIT_YTD", "L04_CURRENT_PROFIT")
            ttm_c = dependencies.ttm(q, "OPERATING_CASH_FLOW_YTD", "L04_CURRENT_CFO")
            assets = dependencies.get(q, "TOTAL_ASSETS_END", "L04_CURRENT_ASSETS")
            prior_p = dependencies.ttm(q - 4, "PARENT_NET_PROFIT_YTD", "L04_PRIOR_PROFIT")
            prior_c = dependencies.ttm(q - 4, "OPERATING_CASH_FLOW_YTD", "L04_PRIOR_CFO")
            prior_assets = dependencies.get(q - 4, "TOTAL_ASSETS_END", "L04_PRIOR_ASSETS")
            quality = (ttm_c - ttm_p) / assets if np.isfinite(assets) and assets > 0 else np.nan
            prior_quality = (prior_c - prior_p) / prior_assets if np.isfinite(prior_assets) and prior_assets > 0 else np.nan
            delta = quality - prior_quality
            result.update(dict(zip(numerical, [profit, assets_before, roa, prior_roa, prior2_roa, historical_mean, historical_sd,
                surprise, ttm_p, ttm_c, assets, quality, prior_p, prior_c, prior_assets, prior_quality, delta])))
            result["L02_known"], result["L04_known"] = np.isfinite(surprise), np.isfinite(delta)
            result["joint_known"] = result["L02_known"] and result["L04_known"]
            result["measurement_status"] = ("MEASURABLE_L02_L04" if result["joint_known"] else
                "NO_VIEW_ZERO_SEASONAL_DISPERSION" if historical_sd == 0 else "NO_VIEW_FINANCIAL_DEPENDENCIES")
        records.append(result)
    return pd.DataFrame(records), pd.DataFrame(dependencies.ledger)


def daily_frontier(events):
    """后到的旧期报告不改变前沿；缺字段的新期报告仍替换旧期。"""
    frame = events.loc[events.available_date.notna()].sort_values(["ts_code", "available_date", "report_period", "announcement_id"]).copy()
    largest = frame.groupby("ts_code").report_period.cummax()
    return frame.loc[frame.report_period.eq(largest)].drop_duplicates(["ts_code", "available_date"], keep="last")


def daily_coverage(events, measured, members, industry):
    left = members[["membership_date", "symbol"]].rename(columns={"membership_date": "date", "symbol": "ts_code"}).copy()
    # 全部成员并非每天有新财报；行业身份按测量日重新匹配，不沿用发布日身份。
    left = pd.concat([left, industry_at(left, industry)], axis=1)
    fields = ["announcement_id", "L02", "L04_change", "L02_known", "L04_known", "joint_known"]
    # 报告在成为成员前公开时，也可供公司后来成为成员时使用。
    right = daily_frontier(events).merge(measured[fields], on="announcement_id", how="left", validate="one_to_one")
    keep = ["ts_code", "report_period", "available_date", *fields]
    panel = pd.merge_asof(left.sort_values(["date", "ts_code"]), right[keep].sort_values(["available_date", "ts_code"]),
                          left_on="date", right_on="available_date", by="ts_code", direction="backward")
    age = (panel.date - panel.report_period).dt.days
    panel["report_age_days"] = age
    panel["nonfinancial_known"] = panel.industry_match_count.eq(1) & ~panel.is_financial
    panel["latest_report_fresh"] = age.between(0, 200)
    for field in ["L02_known", "L04_known", "joint_known"]:
        panel[field] = panel[field].eq(True) & panel.nonfinancial_known & panel.latest_report_fresh
    valid = panel.loc[panel.joint_known]
    assert (valid.available_date <= valid.date).all()
    panel["joint_positive_measurement"] = panel.joint_known & (panel.L02 > 1) & (panel.L04_change >= 0)
    daily = panel.groupby("date").agg(member_count=("ts_code", "size"), industry_known_count=("industry_match_count", lambda v: int(v.eq(1).sum())),
        nonfinancial_count=("nonfinancial_known", "sum"), fresh_report_count=("latest_report_fresh", "sum"),
        L02_known_count=("L02_known", "sum"), L04_known_count=("L04_known", "sum"), joint_known_count=("joint_known", "sum"),
        joint_positive_count=("joint_positive_measurement", "sum"))
    daily["joint_coverage_nonfinancial"] = daily.joint_known_count / daily.nonfinancial_count.replace(0, np.nan)
    daily["measurement_only_no_entry_signal"] = True
    return daily.reset_index(), panel


def calculate(out, progress=False):
    facts, events, members, industry, source_counts = validated_sources(out)
    if progress:
        print(f"原文单位重算完成：{source_counts['selected_fact_rows']}条目标字段；正在建立季度依赖。", flush=True)
    measured, dependencies = measure_events(facts, events, members, industry)
    if progress:
        print(f"事件测量完成：{len(measured)}个历史成员并集公告、{len(dependencies)}条公式依赖；正在检查逐日最新报告覆盖。", flush=True)
    daily, panel = daily_coverage(events, measured, members, industry)
    targets = measured.loc[measured.member_at_available]
    yearly_events = targets.assign(year=targets.available_date.dt.year).groupby("year").agg(
        reports=("announcement_id", "size"), financial_excluded=("is_financial", "sum"),
        L02_known=("L02_known", "sum"), L04_known=("L04_known", "sum"), joint_known=("joint_known", "sum")).reset_index()
    yearly_daily = daily.assign(year=daily.date.dt.year).groupby("year").agg(trading_days=("date", "size"),
        nonfinancial_median=("nonfinancial_count", "median"), joint_min=("joint_known_count", "min"),
        joint_median=("joint_known_count", "median"), joint_max=("joint_known_count", "max"),
        coverage_median=("joint_coverage_nonfinancial", "median"), days_with_any_joint=("joint_known_count", lambda x: int((x > 0).sum()))).reset_index()
    dep_counts = dependencies.groupby(["status", "metric_id"]).size().rename("rows").reset_index()
    unique_missing = dependencies.loc[dependencies.status.ne("VERIFIED_SAVED_ORIGINAL_FACT")].groupby(
        ["ts_code", "source_report_period", "source_announcement_id", "metric_id", "status"], dropna=False).agg(
            affected_events=("target_announcement_id", "nunique"), role_occurrences=("role", "size"),
            first_target_available=("target_available_date", "min"), last_target_available=("target_available_date", "max")).reset_index()
    unique_missing = unique_missing.sort_values(["affected_events", "ts_code", "source_report_period"], ascending=[False, True, True]).reset_index(drop=True)
    used = dependencies.loc[dependencies.status.eq("VERIFIED_SAVED_ORIGINAL_FACT")]
    assert (used.source_available_date <= used.target_available_date).all()
    result = {"study_id": STUDY, "status": "COMPLETED_FROZEN_FINANCIAL_MEASUREMENT", **source_counts,
        "historical_union_report_events_measured": len(measured), "target_member_report_events": len(targets),
        "L02_measurable_events": int(targets.L02_known.sum()),
        "L04_measurable_events": int(targets.L04_known.sum()), "joint_measurable_events": int(targets.joint_known.sum()),
        "joint_measurable_issuers": int(targets.loc[targets.joint_known, "ts_code"].nunique()),
        "first_joint_available_date": targets.loc[targets.joint_known, "available_date"].min(),
        "event_status_counts": targets.measurement_status.value_counts().to_dict(),
        "formula_dependency_rows": len(dependencies), "verified_dependency_rows": len(used),
        "unavailable_dependency_rows": len(dependencies) - len(used), "unique_missing_field_keys_by_reason": len(unique_missing),
        "future_dependency_values_used": 0, "daily_member_panel_rows": len(panel), "daily_measurement_rows": len(daily),
        "yearly_events": yearly_events.to_dict("records"), "yearly_daily_coverage": yearly_daily.to_dict("records"),
        "source_status_preserved": "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE",
        "strict_first_historical_publication": "NOT_ESTABLISHED", "industry_clock": "VENDOR_EFFECTIVE_DATE_PROXY",
        "all_original_pdfs_reparsed": False, "O02": "NOT_COMPUTED", "T11": "NOT_RUN",
        "industry_standardization_and_etf_aggregation": "NOT_COMPUTED", "new_accounts": 0, "new_returns": 0,
        "new_network_requests": 0, "goal_achieved": False, "goal_status": "active", "current_market_view": "NO_VIEW",
        "external_review": "NOT_PERFORMED", "orders_authorized": False, "independent_forward_observations": 0}
    frames = {"verified_facts.parquet": facts, "member_report_measurements.parquet": measured,
        "formula_dependencies.parquet": dependencies, "daily_member_measurements.parquet": panel,
        "daily_coverage.parquet": daily, "missing_field_keys.parquet": unique_missing,
        "yearly_event_coverage.csv": yearly_events, "yearly_daily_coverage.csv": yearly_daily, "dependency_status_counts.csv": dep_counts}
    return frames, clean(result)


def run():
    assert not (OUT / "run_started.json").exists()
    count = verify_inputs(OUT)
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json"), "frozen_files": count})
    frames, result = calculate(OUT, progress=True)
    for name, frame in frames.items():
        if name.endswith(".parquet"):
            frame.to_parquet(OUT / name, index=False)
        else:
            frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    save(OUT / "result.json", result)
    print(json.dumps({k: result[k] for k in ["status", "target_member_report_events", "L02_measurable_events", "L04_measurable_events", "joint_measurable_events", "first_joint_available_date", "new_accounts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare-freeze", "run"])
    args = parser.parse_args()
    prepare_and_freeze() if args.action == "prepare-freeze" else run()
