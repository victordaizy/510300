"""从510300原始年报核对旧持仓，并重算已有盈利事实的资产覆盖边界。"""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_eps_disclosed_holdings_diagnostic_v1.json"
OUT = ROOT / "reports/research/510300_eps_disclosed_holdings_diagnostic_v1"
ROW = re.compile(r"^(\d+)\s+(\d{6})\s+(.+?)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})$")
MONEY = re.compile(r"[\d,]+\.\d{2}")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cents(value: str) -> int:
    amount = Decimal(value.replace(",", "")) * 100
    if amount != amount.to_integral_value():
        raise ValueError("金额不是整数分")
    return int(amount)


def parse_report(document: dict, extracted: dict) -> tuple[pd.DataFrame, dict]:
    year = document["year"]
    pages = extracted["pages"]
    cover = re.sub(r"\s+", "", pages[0]["text"])
    expected = f"华泰柏瑞沪深300交易型开放式指数证券投资基金{year}年年度报告{year}年12月31日"
    if expected not in cover:
        raise ValueError(f"{year}年封面身份或期末日不一致")
    introduction = re.sub(r"\s+", "", "\n".join(x["text"] for x in pages[:8]))
    if "基金主代码510300" not in introduction:
        raise ValueError("基金主代码不符")
    sent = re.search(r"送出日期[：:]?(\d{4})年(\d{1,2})月(\d{1,2})日", cover)
    if sent is None:
        raise ValueError("送出日期缺失")
    sent_date = f"{int(sent[1]):04d}-{int(sent[2]):02d}-{int(sent[3]):02d}"
    announcement = document["metadata"]["SSEDATE"]
    conservative = max(sent_date, announcement)
    pdf_dates = {}
    for key in ["CreationDate", "ModDate"]:
        value = str(extracted["metadata"].get(key, ""))
        match = re.search(r"D:(\d{4})(\d{2})(\d{2})", value)
        if match:
            pdf_dates[key] = f"{match[1]}-{match[2]}-{match[3]}"
    later_version = any(x > conservative for x in pdf_dates.values())
    if not pdf_dates:
        raise ValueError("原PDF版本日期不可读")
    nav = []
    asset_stock = []
    active = False
    finished = False
    section = "ALL_STOCKS"
    holdings = []
    for page in pages:
        wrapped_nav = re.search(r"期末基金\s*\n\s*([\d,]+\.\d{2})[^\n]*\n\s*资产净值", page["text"])
        if wrapped_nav:
            nav.append((page["page"], cents(wrapped_nav[1])))
        for raw in page["text"].splitlines():
            line = raw.strip()
            compact = re.sub(r"\s+", "", line)
            if compact.startswith("期末基金资产净值"):
                matches = MONEY.findall(line)
                if matches:
                    nav.append((page["page"], cents(matches[0])))
            if compact.startswith("其中：股票") or compact.startswith("其中:股票"):
                matches = MONEY.findall(line)
                if matches:
                    asset_stock.append((page["page"], cents(matches[0])))
            if re.match(r"^8\.3(?:\s|期末)", line) and "股票投资明细" in compact and "..." not in compact:
                if active or finished:
                    raise ValueError("完整股票投资明细起点重复")
                active = True
                continue
            if active and compact.startswith("8.3.1"):
                section = "INDEX_STOCKS"
                continue
            if active and compact.startswith("8.3.2"):
                section = "ACTIVE_STOCKS"
                continue
            if active and re.match(r"^8\.4(?:\s|报告期)", line):
                active = False
                finished = True
            if not active:
                continue
            match = ROW.fullmatch(line)
            if match:
                rank, code, name, shares, value, percentage = match.groups()
                suffix = "SH" if code.startswith("6") else "SZ" if code.startswith(("0", "3")) else None
                if suffix is None:
                    raise ValueError("持仓证券市场代码未登记")
                holding = {"report_year": year, "report_end": f"{year}-12-31", "section": section,
                           "rank": int(rank), "ts_code": code + "." + suffix, "stock_name": re.sub(r"\s+", "", name),
                           "shares": float(shares.replace(",", "")), "fair_value_cents": cents(value),
                           "reported_nav_pct": float(percentage), "source_page": page["page"], "source_line": line}
                holdings.append(holding)
            elif re.match(r"^\d+\s+\d{6}\s", line):
                raise ValueError(f"股票行无法完整解析：{page['page']}页：{line}")
    if not finished or len(holdings) < 300:
        raise ValueError("全部股票表未完整关闭或股票数不足300")
    frame = pd.DataFrame(holdings)
    if frame.ts_code.duplicated().any():
        raise ValueError("完整股票表出现重复证券代码")
    for _, group in frame.groupby("section", sort=False):
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError("持仓表序号不连续")
    if len(nav) != 1:
        raise ValueError(f"期末基金资产净值不是唯一对象：{nav}")
    stock_total = int(frame.fair_value_cents.sum())
    matching_totals = [x for x in asset_stock if x[1] == stock_total]
    if not matching_totals:
        raise ValueError(f"股票总金额不相等：明细{stock_total}分；候选{asset_stock}")
    frame["stock_weight"] = frame.fair_value_cents / stock_total
    frame["nav_weight"] = frame.fair_value_cents / nav[0][1]
    errors = np.abs(100 * frame.nav_weight - frame.reported_nav_pct)
    if float(errors.max()) > 0.00500001:
        raise ValueError("逐行股票净值占比超出两位小数舍入界限")
    receipt = {"year": year, "status": "NO_VIEW_PDF_VERSION_LATER_THAN_PUBLICATION" if later_version
               else "PASS_ORIGINAL_DATED_DISCLOSURE_STOCK_TABLE_RECONCILED",
               "pdf": document["pdf"], "pdf_metadata": extracted["metadata"], "pdf_date_fields": pdf_dates,
               "report_end": f"{year}-12-31", "announcement_date": announcement, "sent_date": sent_date,
               "conservative_publication_date": conservative,
               "historical_http_first_publication_proven": False, "historical_intraday_clock_proven": False,
               "weight_is_current_official_index_weight": False, "fund_code_identity_pass": True,
               "full_table_rows": len(frame), "table_section_counts": {k: int(v) for k, v in frame.groupby("section").size().items()},
               "stock_total_cents": stock_total, "stock_total_source_page": matching_totals[-1][0],
               "fund_nav_cents": nav[0][1], "fund_nav_source_page": nav[0][0],
               "stock_table_vs_declared_difference_cents": 0,
               "maximum_reported_nav_percentage_point_rounding_error": float(errors.max()),
               "equity_fraction_of_fund_nav": stock_total / nav[0][1],
               "stock_table_pages": sorted(frame.source_page.unique().astype(int).tolist())}
    return frame, receipt


def diagnose(holdings: pd.DataFrame, receipt: dict, company: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    when = pd.Timestamp(receipt["conservative_publication_date"])
    origins = sorted(pd.to_datetime(company.origin).unique())
    origin = next((pd.Timestamp(x) for x in origins if x > when), None)
    if origin is None:
        raise ValueError("报告公开后没有已有月末原点")
    available = company.loc[company.origin.eq(origin)].copy()
    if len(available) != 300 or available.ts_code.duplicated().any():
        raise ValueError("已有公司月度表不是300家唯一公司")
    joined = holdings.merge(available.drop(columns="origin"), on="ts_code", how="left", validate="one_to_one", indicator=True)
    joined["origin"] = origin
    joined["historical_member_at_origin"] = joined["_merge"].eq("both")
    joined = joined.drop(columns="_merge")
    receipt.update({"diagnostic_origin": origin.strftime("%Y-%m-%d"),
                    "report_end_to_publication_days": (when - pd.Timestamp(receipt["report_end"])).days,
                    "report_end_to_origin_days": (origin - pd.Timestamp(receipt["report_end"])).days,
                    "held_stock_count_in_origin_member_set": int(joined.historical_member_at_origin.sum()),
                    "held_stock_weight_outside_origin_members": float(joined.loc[~joined.historical_member_at_origin, "stock_weight"].sum())})
    result = []
    for field in ["eps_growth", "profit_revision", "reported_earnings_yield"]:
        valid = np.isfinite(pd.to_numeric(joined[field], errors="coerce"))
        cover = float(joined.loc[valid, "stock_weight"].sum())
        unknown = float(joined.loc[~valid, "stock_weight"].sum())
        signed = float((np.sign(joined.loc[valid, field]) * joined.loc[valid, "stock_weight"]).sum())
        lower, upper = signed - unknown, signed + unknown
        applicable = field != "reported_earnings_yield"
        state = "NOT_APPLICABLE_REPORTED_PE_IS_NOT_A_DIRECTION_FORECAST"
        if applicable:
            state = "POSITIVE_EARNINGS_FACT_SIGN_ROBUST_TO_UNCOVERED_HOLDINGS" if lower > 0 else (
                "NEGATIVE_EARNINGS_FACT_SIGN_ROBUST_TO_UNCOVERED_HOLDINGS" if upper < 0 else "NO_VIEW_UNCOVERED_HOLDINGS_CAN_REVERSE_SIGN")
        top = joined.nlargest(10, "stock_weight")
        record = {"report_year": receipt["year"], "report_end": receipt["report_end"], "origin": receipt["diagnostic_origin"],
                  "field": field, "holdings_count": len(joined), "covered_holdings_count": int(valid.sum()),
                  "covered_count_fraction_of_held_stocks": float(valid.mean()), "covered_stock_weight": cover,
                  "covered_fund_nav_weight": float(joined.loc[valid, "nav_weight"].sum()), "uncovered_stock_weight": unknown,
                  "known_signed_weight": signed if applicable else None, "whole_stock_sign_lower_bound": lower if applicable else None,
                  "whole_stock_sign_upper_bound": upper if applicable else None, "earnings_fact_direction_status": state,
                  "etf_future_price_direction": "NOT_COMPUTED", "top_ten_covered_count": int(np.isfinite(top[field]).sum()),
                  "top_ten_missing_symbols": top.loc[~np.isfinite(top[field]), "ts_code"].tolist()}
        result.append(record)
    return joined, result


def computation() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = read(CONFIG)
    collection = read(OUT / "source_collection_result.json")
    if collection["status"] != "THREE_FIXED_ORIGINAL_REPORTS_ACQUIRED":
        raise ValueError("预定三个原件未全部取得")
    company = pd.read_parquet(ROOT / config["latest_company_information_source"])
    company["origin"] = pd.to_datetime(company.origin)
    rows, coverages, admissions = [], [], []
    for document in collection["documents"]:
        original = ROOT / document["pdf"]["path"]
        if identity(original) != document["pdf"]:
            raise ValueError("原PDF哈希变化")
        extracted = read(OUT / f"annual_{document['year']}_pages.json")
        if extracted["source"] != document["pdf"]:
            raise ValueError("页面提取来源哈希不符")
        holdings, receipt = parse_report(document, extracted)
        if receipt["status"].startswith("PASS"):
            joined, coverage = diagnose(holdings, receipt, company)
            rows.append(joined)
            coverages.extend(coverage)
        admissions.append(receipt)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    coverage = pd.DataFrame(coverages)
    result = {"study_id": config["study_id"], "status": "DISCLOSED_HOLDINGS_COVERAGE_DIAGNOSTIC_COMPLETE_NO_NEW_STRATEGY",
              "report_admissions": admissions, "coverage": coverages, "new_return_labels": 0, "new_model_fits": 0,
              "new_accounts_generated": 0, "new_historical_strategy_configurations": 0, "new_random_draws": 0,
              "current_official_index_weights_admitted": False, "paused_source_queues_resumed": False,
              "independent_prediction_validation": False, "goal_achieved": False}
    return detail, coverage, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "verify"], required=True)
    args = parser.parse_args()
    if args.mode == "run":
        config = read(CONFIG)
        inputs = [CONFIG, Path(__file__), OUT / "source_collection_result.json",
                  ROOT / config["latest_company_information_source"], ROOT / config["institution_evidence_source"]]
        inputs += [OUT / f"annual_{y}_pages.json" for y in config["report_years"]]
        save(OUT / "diagnostic_claim.json", {"started_at": now(), "new_strategy_outcomes_seen": False,
                                            "old_eps_strategy_results_already_seen": True, "inputs": [identity(p) for p in inputs]})
    else:
        for item in read(OUT / "diagnostic_claim.json")["inputs"]:
            if identity(ROOT / item["path"]) != item:
                raise ValueError("冻结诊断输入发生变化：" + item["path"])
    detail, coverage, result = computation()
    if args.mode == "run":
        detail.to_parquet(OUT / "holdings_and_eps_evidence.parquet", index=False)
        detail.to_csv(OUT / "逐只持仓与盈利事实覆盖.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(OUT / "三个原点资产覆盖与符号边界.csv", index=False, encoding="utf-8-sig")
        result["completed_at"] = now()
        save(OUT / "result.json", result)
    else:
        saved = read(OUT / "result.json")
        comparable = {k: v for k, v in saved.items() if k != "completed_at"}
        if result != comparable:
            raise ValueError("保存的诊断结果不能重算")
        restored = pd.read_parquet(OUT / "holdings_and_eps_evidence.parquet")
        for field in ["origin"]:
            detail[field] = detail[field].astype("datetime64[ns]")
            restored[field] = restored[field].astype("datetime64[ns]")
        pd.testing.assert_frame_equal(detail, restored, check_exact=True)
        print("保存结果重算通过：原件金额、披露时钟、逐只持仓、覆盖和符号区间均一致。", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
