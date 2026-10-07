"""仅核2026源字段和钟，保留原缓存前缀，不拟合、不运行账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import factor96_funding_relief_v1 as funding
from research import asymmetric_stress_hazard_source_remediation_v1_0_1 as policy_parser
from research import funding_availability_semantics_v1 as availability
from research import macro_technical_first_passage_inputs_v1 as model
from research import macro_technical_first_passage_study_v1 as study

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_macro_2026_source_coverage_intake_v1"
CARD = ROOT / "docs/510300_2026_MACRO_SOURCE_COVERAGE_INTAKE_V1.md"
PUBLISHED = funding.POLICY_ROOT / "pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet"
POLICY = funding.OUT / "inputs/policy_rate_ledger.parquet"
RAW_DR = ROOT / "data/raw/510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2/repo_daily/repo_daily_DR007_IB_20260101_20260814.json"
MANIFEST = funding.DR.parent / "dr007_tushare_source_acquisition_manifest.json"
PARENT = ROOT / "reports/research/510300_macro_funding_source_contract_v2"
ETF_FILES = [ROOT / "data/raw/flow" / n for n in
             ["510300_margin_detail_daily_tushare.parquet", "510300_margin_extension_20260818_v2.parquet"]]
MARKET_FILES = sorted([*(ROOT / "data/raw/market_margin_validation_v2").glob("*margin*.parquet"),
                       *(ROOT / "data/raw/market_margin_leverage_v0").glob("*margin*.parquet")])
BASE = availability.BASE / "results/原点全部技术特征.parquet"
PARENT_INPUT = PARENT / "results/全部3488当时已知技术与宏观_未知保留.parquet"


def rel(path):
    return path.absolute().relative_to(ROOT).as_posix()


def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def table(name, frame):
    p = OUT / "results" / name
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(p.with_suffix(".parquet"), index=False)


def freeze():
    if OUT.exists():
        raise FileExistsError("已有来源登记，不覆盖。")
    published = pd.read_parquet(PUBLISHED)
    published["notice_date"] = pd.to_datetime(published.notice_date)
    articles = [ROOT / p for p in published.loc[published.notice_date.dt.year.eq(2026), "raw_path"]]
    sources = [CARD, Path(__file__), Path(funding.__file__), Path(policy_parser.__file__), Path(availability.__file__), Path(model.__file__),
               BASE, PARENT_INPUT, PARENT / "summary.json", PARENT / "protocol.json", availability.FUND,
               availability.ORDERS, availability.MARGIN, funding.DR, RAW_DR, MANIFEST, POLICY, PUBLISHED,
               *MARKET_FILES, *ETF_FILES, *articles]
    sources = list(dict.fromkeys(sources))
    for p in sources:
        study.require(p.is_file(), "核对源不存在：" + rel(p))
    (OUT / "results").mkdir(parents=True)
    save_json(OUT / "protocol.json", {"at": study.now(), "study": "510300_2026_MACRO_SOURCE_COVERAGE_INTAKE_V1",
        "decision": "TECH.R199", "initial_discovery_before_registration": True,
        "purpose": rel(CARD), "calendar_rows": 3488, "fixed_market_source_files": [rel(p) for p in MARKET_FILES],
        "fixed_etf_only_files": [rel(p) for p in ETF_FILES], "all_2026_policy_articles": len(articles),
        "sources": [{"path": rel(p), "sha256": h(p)} for p in sources],
        "no_future_return_label_reads": True, "new_fits": 0, "new_accounts": 0, "new_network_requests": 0,
        "new_financial_candidate_registered": False, "original_rejected_model_preserved": True})
    print("TECH.R199登记完成：2026全部操作原件与固定融资资料，0金融用途。", flush=True)


def run():
    study.require(not (OUT / "RUN_STARTED.json").exists(), "本核对已启动，不重复。")
    protocol = study.read(OUT / "protocol.json")
    for item in protocol["sources"]:
        study.require(h(ROOT / item["path"]) == item["sha256"], "登记来源改变：" + item["path"])
    save_json(OUT / "RUN_STARTED.json", {"at": study.now(), "new_fits": 0, "new_accounts": 0})
    data = pd.read_parquet(BASE)
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    study.require(len(data) == 3488, "原观察日历改变。")
    dr = pd.read_parquet(funding.DR)
    dr["date"] = pd.to_datetime(dr.date).astype("datetime64[ns]")
    manifest = study.read(MANIFEST)
    dr_chunk = next(x for x in manifest["chunk_receipts"] if x["first_date"].startswith("2026"))
    study.require(h(RAW_DR) == dr_chunk["response_artifact"]["sha256"], "2026原响应与取得记录不同。")
    payload = study.read(RAW_DR)
    raw = pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    study.require(raw.ts_code.eq("DR007.IB").all() and raw.repo_maturity.eq("DR007").all(), "DR工具或期限不同。")
    raw["date"] = pd.to_datetime(raw.trade_date, format="%Y%m%d").astype("datetime64[ns]")
    r2026 = dr.loc[dr.date.dt.year.eq(2026)].sort_values("date")
    raw = raw.sort_values("date")
    study.require(raw.date.tolist() == r2026.date.tolist(), "2026原响应日历不同。")
    np.testing.assert_allclose(raw.weight.astype(float), r2026.dr007.astype(float), rtol=0, atol=1e-12)
    policy = pd.read_parquet(POLICY)
    policy["known_at"] = pd.to_datetime(policy.known_at).astype("datetime64[ns, Asia/Shanghai]")
    policy["effective_date"] = pd.to_datetime(policy.effective_date).astype("datetime64[ns]")
    extended, source = funding.funding_features(data, dr, policy, lag=1)
    old = pd.read_parquet(availability.FUND)
    columns = ["date", "fund_stat_date", "available_at", "policy_known_at", "effective_date", "dr007", "rate", "gap_pp", "source_age_days"]
    prefix = extended.loc[extended.date.le(old.date.max()), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[columns].reset_index(drop=True), prefix, check_exact=True, check_dtype=False)
    extended["legacy_quantile_signal_supported"] = extended.fund_known
    extended["fund_known"] = availability.source_mask(extended)
    table("完整3488资金源派生_仅扩已有日历不拟合", extended)
    table("DR统计源和原下一交易日可用钟", source)
    current = pd.read_parquet(PARENT_INPUT)
    derived = model.views(data, pd.read_parquet(availability.ORDERS), extended, pd.read_parquet(availability.MARGIN))
    current["date"] = pd.to_datetime(current.date).astype("datetime64[ns]")
    f = derived[["date", "orders_known", "funding_known", "margin_known", "macro_features_known", "joint_features_known",
                 "fund_stat_date", "funding_available_at", "funding_policy_known_at", "dr007", "rate", "funding_gap_pp",
                 "funding_gap_change5", "orders_reference_period", "orders_available_at", "margin_stat_date"]].copy()
    f["old_funding_known"] = current.funding_known
    f["old_joint_features_known"] = current.joint_features_known
    f["additional_funding_known"] = f.funding_known & ~f.old_funding_known
    f["additional_joint_origin"] = f.joint_features_known & ~f.old_joint_features_known
    table("全部3488源覆盖比较_联合未知保留", f)
    this_year = f.loc[f.date.dt.year.eq(2026)].copy()
    table("2026全部观察槽_订单资金两市融资支持分开", this_year)
    this_year["month"] = this_year.date.dt.to_period("M").astype(str)
    monthly = this_year.groupby("month").agg(calendar_slots=("date", "size"), orders_known=("orders_known", "sum"),
              funding_known=("funding_known", "sum"), funding_change5_known=("funding_gap_change5", "count"),
              market_margin_known=("margin_known", "sum"), joint_known=("joint_features_known", "sum")).reset_index()
    table("2026逐月源覆盖_不计算收益", monthly)
    coverage = []
    for item in [*MARKET_FILES, availability.MARGIN, *ETF_FILES]:
        z = pd.read_parquet(item)
        date_col = next((c for c in ["date", "trade_date"] if c in z), None)
        study.require(date_col is not None, "融资表未识别统计日。")
        dates = pd.to_datetime(z[date_col])
        coverage.append({"path": rel(item), "rows": len(z), "first": dates.min(), "last": dates.max(),
                         "rows_2026": int(dates.dt.year.eq(2026).sum()), "scope": "ETF_ONLY_510300" if item in ETF_FILES else "MARKET_OR_EXCHANGE_SUMMARY",
                         "has_market_rzye": "market_rzye" in z, "has_market_rzmre": "market_rzmre" in z,
                         "can_replace_market_summary": False if item in ETF_FILES else bool(dates.dt.year.eq(2026).any())})
    coverage = pd.DataFrame(coverage)
    table("已定位全部融资原件覆盖_不以ETF替代两市", coverage)
    published = pd.read_parquet(PUBLISHED)
    published["notice_date"] = pd.to_datetime(published.notice_date).astype("datetime64[ns]")
    rates = published.loc[published.notice_date.dt.year.eq(2026)].copy()
    article_checks = []
    for row in rates.itertuples(index=False):
        article = ROOT / row.raw_path
        parsed = policy_parser.parse_pboc_open_market_notice(article.read_bytes(), source_url=row.source_url)
        study.require(h(article) == row.raw_sha256, "2026公告原件哈希与记录不同。")
        study.require(pd.Timestamp(row.published_at).date() == pd.Timestamp(row.notice_date).date(), "操作日期和公布日期不同。")
        study.require(bool(row.has_seven_day_row) and np.isfinite(row.seven_day_rate_percent), "2026操作表无有限7天利率。")
        study.require(parsed.has_seven_day_row and parsed.notice_date == pd.Timestamp(row.notice_date).date(), "2026原7天操作行或日期不同。")
        study.require(pd.Timestamp(parsed.published_at) == pd.Timestamp(row.published_at), "2026原件公布钟不同。")
        study.require(parsed.announcement_number == row.announcement_number, "2026原件公告号不同。")
        study.require(parsed.seven_day_rate_percent == row.seven_day_rate_percent, "2026原件7天行利率不同。")
        article_checks.append({"notice_date": row.notice_date, "published_at": row.published_at,
                               "rate": row.seven_day_rate_percent, "raw_path": row.raw_path, "raw_sha256": row.raw_sha256,
                               "original_text_rate_present": True, "historical_first_vintage_certified": False})
    table("2026全部政策操作原件及钟核对", pd.DataFrame(article_checks))
    table("2026政策操作率公告原台账", rates)
    study.require(f.old_joint_features_known.equals(f.joint_features_known), "只扩资金后联合完整原点发生变化，须另外判定。")
    study.require(this_year.margin_known.sum() == 0 and this_year.joint_features_known.sum() == 0,
                  "2026两市融资或联合支持与缺口判断不同。")
    available = this_year.loc[this_year.funding_known]
    summary = {"at": study.now(), "study": protocol["study"], "registration_decision": "TECH.R199", "decision": "TECH.R200",
        "status": "COMPLETED_2026_SOURCE_COVERAGE_FUNDING_RECOVERABLE_MARKET_MARGIN_MISSING_NO_FINANCIAL_ADMISSION",
        "previous_goal_turn": "PROGRESS_ACTUAL_SOURCE_SEMANTICS_AND_EIGHT_FINANCIAL_ACCOUNTS_R198",
        "original_funding_prefix_rows_exact": len(prefix), "all_calendar_slots": len(f), "slots_2026": len(this_year),
        "dr_raw_2026_rows_exactly_reproduced": len(raw), "dr_source_last": dr.date.max().isoformat(),
        "funding_known_2026_under_inherited_contract": int(this_year.funding_known.sum()),
        "funding_change5_known_2026": int(this_year.funding_gap_change5.notna().sum()),
        "funding_known_2026_first": available.date.min().isoformat(), "funding_known_2026_last": available.date.max().isoformat(),
        "all_2026_policy_notices_and_originals": len(rates), "last_policy_notice": rates.notice_date.max().isoformat(),
        "policy_rate_levels_2026": sorted(rates.seven_day_rate_percent.unique().tolist()),
        "policy_missing_notice_slots_not_filled": True, "new_same_day_policy_coverage_not_assumed": True,
        "market_or_exchange_source_tables": len(MARKET_FILES) + 1,
        "all_located_market_summary_rows_2026": int(coverage.loc[coverage.scope.ne("ETF_ONLY_510300"), "rows_2026"].sum()),
        "etf_only_source_tables": 2, "etf_only_rows_2026": int(coverage.loc[coverage.scope.eq("ETF_ONLY_510300"), "rows_2026"].sum()),
        "etf_margin_substitution": False, "market_margin_known_2026": 0, "joint_known_2026": 0,
        "additional_joint_origins": int(f.additional_joint_origin.sum()), "original_2608_joint_mask_exactly_unchanged": True,
        "financial_admission": "NOT_ADMITTED_FUNDING_EXTENSION_ALONE_NO_JOINT_SUPPORT_OR_ACCOUNT_CHANGE",
        "source_value_reproduction_is_first_vintage_certification": False, "new_fits": 0, "new_accounts": 0,
        "new_return_label_reads": 0, "new_network_requests": 0, "goal_achieved": False,
        "overfitting_removed": False, "independent_validation": "NOT_ESTABLISHED", "new_net_sharpe": "NOT_COMPUTED",
        "next_source_need": "2026-01-05至2026-09-30沪深各交易所汇总融资余额与融资买入原件及可用钟；两个市场须齐全，不能以510300自身融资代替。"}
    save_json(OUT / "summary.json", summary)
    report = "# 2026宏观资料核对：资金可以恢复，联合模型仍缺两市融资\n\n"
    report += "本轮TECH.R199—R200只核本地来源和派生合同，0新模型、账户、股票收益标签和网络请求。原R198完整金融拒绝保持。\n\n"
    report += f"在原{len(this_year)}个2026观察槽中，DR原2026响应{len(raw)}行逐值复现。沿原统计日政策率、下一交易日09:30、16:00判断和源龄1—10日合同，只扩已有日历，可恢复{summary['funding_known_2026_under_inherited_contract']}个资金已知槽；五日变化{summary['funding_change5_known_2026']}槽可算，最后资金支持日{available.date.max():%Y-%m-%d}。3307原缓存的所有统计日、源钟、政策钟、实施日、资金值和价差精确保持。\n\n"
    report += f"2026政策操作公告{len(rates)}份，原HTML逐份核对，台账利率水平{summary['policy_rate_levels_2026']}%，最后公告{rates.notice_date.max():%Y-%m-%d}。同一操作率重复不是新政策冲击；公告缺日及8月后覆盖未知保留，历史回取不认证当年首版或传播。资金源的计算可用性继承旧合同，不另宣称所有日期都有同日公告证据。\n\n"
    report += monthly.to_markdown(index=False) + "\n\n"
    report += coverage.to_markdown(index=False) + "\n\n"
    report += "已定位的交易所/两市汇总表没有2026行。510300自身融资表虽覆盖2026，工具和经济含义不同，不能填入两市融资字段。2026原联合资料仍0完整原点；全部3488日的联合支持掩码与R198精确相同。资金扩展单独不能改变原联合模型的训练、预测或账户，故不重复金融检验，不调阈值或选更短样本。\n\n"
    report += "下一来源需求限定为2026-01-05至2026-09-30沪深两交易所每日汇总融资余额和融资买入额，须保留统计日、实际公布钟/保守上界和原件；须先核官方可获得性与既有准入限制，再另立有限来源申请。原模型失败不自动变成新候选，后续接入仍需完整用途登记；当前没有待跑金融。原13前瞻状态、策略与其他分支保持。\n\n"
    report += "[逐月覆盖](results/2026逐月源覆盖_不计算收益.csv)、[全部观察槽](results/2026全部观察槽_订单资金两市融资支持分开.csv)、[全部融资原件](results/已定位全部融资原件覆盖_不以ETF替代两市.csv)、[政策原件与钟](results/2026全部政策操作原件及钟核对.csv)、[完整实际结果](summary.json)。\n"
    (OUT / "2026资料可用性与下一实验准入.md").write_bytes(report.encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026宏观原统计源覆盖核对")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
