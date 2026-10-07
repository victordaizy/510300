"""从已保存央行原文提取贷款构成，按当时披露统一为年初累计。"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_rmb_loan_composition_source_v1"
SOURCE = ROOT / "reports/research/510300_m1_m2_monthly_increment_v1/admitted_releases.csv"
STUDY = "510300_RMB_LOAN_COMPOSITION_SOURCE_V1"
NAMES = ["rmb_total", "household_total", "household_short", "household_long",
         "corporate_total", "corporate_short", "corporate_long", "bills", "nonbank_total"]
AMOUNT = r"(增加|减少)(\d+(?:\.\d+)?)(万亿元|亿元)"
PREFIX = r"(当月|\d{1,2}月份?|前[一二三四五六七八九十两\d]+个月|一季度|前三季度|上半年|全年)"


def amount(match):
    action, number, unit = match.groups()[-3:]
    factor = Decimal(10000) if unit == "万亿元" else Decimal(1)
    value = Decimal(number) * factor * (-1 if action == "减少" else 1)
    decimals = len(number.partition(".")[2])
    return {"value_yi": float(value), "literal": match.group(0),
            "display_rounding_half_yi": float(factor * Decimal(10) ** (-decimals) / 2)}


def extract_amount(text, prefix):
    found = list(re.finditer(prefix + AMOUNT, text))
    assert len(found) == 1, (prefix, len(found), text[:380])
    return amount(found[0])


def chinese_number(text):
    if text.isdigit():
        return int(text)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if text.startswith("十"):
        return 10 + digits[text[1:]]
    return digits[text]


def period_start(scope, month):
    m = pd.Period(month, freq="M")
    if scope == "当月":
        return month, "MONTH"
    numbered = re.fullmatch(r"(\d{1,2})月份?", scope)
    if numbered:
        assert int(numbered.group(1)) == m.month
        return month, "MONTH"
    expected = {"一季度": 3, "上半年": 6, "前三季度": 9, "全年": 12}
    if scope in expected:
        assert expected[scope] == m.month
    else:
        matched = re.fullmatch(r"前([一二三四五六七八九十两\d]+)个月", scope)
        assert matched and chinese_number(matched.group(1)) == m.month, (scope, month)
    return f"{m.year}-01", "YEAR_TO_DATE"


def parse(row):
    raw = ROOT / row["raw_path"]
    assert digest(raw) == row["source_sha256"]
    content = raw.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("gb18030")
    soup = BeautifulSoup(text, "html.parser")
    body = soup.find(id="zoom") or soup
    compact = re.sub(r"\s+", "", body.get_text(" ", strip=True)).replace("（", "(").replace("）", ")").replace("；", ";")
    assert compact.count("分部门看") == 1
    cut = compact.index("分部门看")
    totals = list(re.finditer(PREFIX + r"人民币贷款" + AMOUNT, compact[:cut]))
    assert totals, "找不到部门贷款之前的总量与统计区间。"
    total = totals[-1]
    begin, scope = period_start(total.group(1), row["stat_month"])
    department = compact[cut:cut + 900]
    household = re.search(r"住户(?:部门)?贷款[^;]+", department)
    corporate = re.search(r"(?:非金融企业及机关团体|企\(事\)业单位)贷款[^;]+", department)
    assert household and corporate
    fields = {"rmb_total": amount(total)}
    fields["household_total"] = extract_amount(household.group(0), r"住户(?:部门)?贷款")
    fields["household_short"] = extract_amount(household.group(0), r"短期贷款")
    fields["household_long"] = extract_amount(household.group(0), r"中长期贷款")
    fields["corporate_total"] = extract_amount(corporate.group(0), r"(?:非金融企业及机关团体|企\(事\)业单位)贷款")
    fields["corporate_short"] = extract_amount(corporate.group(0), r"短期贷款")
    fields["corporate_long"] = extract_amount(corporate.group(0), r"中长期贷款")
    fields["bills"] = extract_amount(corporate.group(0), r"票据融资")
    fields["nonbank_total"] = extract_amount(department, r"非银行业金融机构贷款")
    stocks = list(re.finditer(r"人民币贷款余额(\d+(?:\.\d+)?)万亿元,?同比(增长|下降)(\d+(?:\.\d+)?)%", compact[:cut].replace("，", ",")))
    assert len(stocks) == 1, "贷款余额与官方同比未唯一识别。"
    stock = stocks[0]
    known = pd.Timestamp(row["published_at"]).tz_convert("Asia/Shanghai").normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    assert pd.Period(row["stat_month"], freq="M").end_time.date() < known.date()
    notes = re.findall(r"注\d[：:].*?(?=注\d[：:]|$)", compact)
    return {"stat_month": row["stat_month"], "period_start": begin, "period_end": row["stat_month"],
            "reported_interval": scope, "period_literal": total.group(1), "published_at": row["published_at"],
            "conservative_known_at": known, "source_url": row["source_url"], "source_sha256": row["source_sha256"],
            "raw_path": row["raw_path"], "retrieved_at": row["retrieved_at"], "fields": fields,
            "rmb_stock_yi": float(Decimal(stock.group(1)) * 10000),
            "rmb_stock_yoy_percent": float(stock.group(3)) * (-1 if stock.group(2) == "下降" else 1),
            "total_evidence": total.group(0), "department_evidence": department[:department.find("月末，外币")] if "月末，外币" in department else department[:650],
            "methodology_notes": notes,
            "statistical_regime": "EXPANDED_2023" if row["stat_month"] >= "2023-01" else "PRE_2023",
            "historical_first_vintage_verified": False, "status": "SOURCE_COMPONENTS_EXTRACTED"}


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("来源提取规则已固定。")
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    assert len(source) == 104 and source.stat_month.is_unique
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "source_months": 104,
        "source_interval": [source.stat_month.min(), source.stat_month.max()], "source": SOURCE.relative_to(ROOT).as_posix(),
        "extraction": NAMES, "unit": "统一亿元，减少保留负号；同时保存每个字段原文字面量。",
        "interval": "以分部门看之前最近的人民币贷款增减句明确统计区间；当月直接，季度/半年/全年/前N个月累计，不根据数值大小猜测。",
        "canonical_ytd": "当月披露加此前同年已公布累计；若本次公布年初累计，直接采用本次值覆盖当前累计状态，但不回改此前发布的行。每年1月重新起算。",
        "monthly_differences": "累计披露减上次已知累计只作公布数之差，可能含修订和舍入，不认证为真实当月新增。",
        "components": "公开部门子项可能不是完整分类；子项和与部门总额的差保留，不强制凑平。",
        "precision": "显示位数的半单位仅为字面舍入幅度；末尾零是否省略未知，不把它当官方误差分布。",
        "known_at": "采用原文所载公布日结束，最早后续交易日使用；不回填至统计月末。历史不可变初版仍未认证。",
        "definition": "2018-2022与2023起扩展金融统计范围分开标记，不用后来回溯数字改前期。企事业贷款不是纯民企投资，居民中长贷不是全部真实住房需求。",
        "primary_measurement_candidate": "企业加居民中长期贷款年初累计新增 / 同口径人民币贷款年初累计新增；这是净增比率，不保证位于0至1，也不直接识别投资或股票买盘。",
        "new_market_downloads": 0, "new_accounts": 0, "new_strategy_returns": 0, "goal_achieved": False}, True)
    save(OUT / "freeze.json", {"at": now(), "source_table_sha256": digest(SOURCE),
        "raw_sources": {r.raw_path: r.source_sha256 for r in source.itertuples()}, "code_sha256": digest(Path(__file__))}, True)
    (OUT / "frozen_source_parser.py").write_bytes(Path(__file__).read_bytes())
    print("104个月贷款构成的原文、单位、区间和累计转换规则已固定，未读取股票收益。", flush=True)


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("来源提取已有终态。")
    frozen = read(OUT / "freeze.json")
    assert digest(SOURCE) == frozen["source_table_sha256"] and digest(Path(__file__)) == frozen["code_sha256"]
    releases, failures = [], []
    for row in pd.read_csv(SOURCE).sort_values("stat_month").to_dict("records"):
        try:
            releases.append(parse(row))
        except (AssertionError, ValueError, KeyError) as exc:
            failures.append({"stat_month": row["stat_month"], "raw_path": row["raw_path"], "error": str(exc)})
    save(OUT / "extracted_originals.json", releases, True)
    save(OUT / "extraction_failures.json", failures, True)
    transformed, current, current_month, dependencies = [], None, None, []
    for row in releases:
        month = pd.Period(row["stat_month"], freq="M")
        previous = current if current_month == str(month - 1) and month.month != 1 else None
        raw = {name: row["fields"][name]["value_yi"] for name in NAMES}
        old_dependencies = dependencies if previous is not None else []
        if row["reported_interval"] == "YEAR_TO_DATE" or month.month == 1:
            current = raw.copy()
            dependencies = [row["stat_month"]]
        elif previous is not None:
            current = {name: previous[name] + raw[name] for name in NAMES}
            dependencies = [*old_dependencies, row["stat_month"]]
        else:
            current = None
            failures.append({"stat_month": row["stat_month"], "error": "NO_PRIOR_CUMULATIVE_FOR_MONTHLY_RELEASE"})
            continue
        current_month = row["stat_month"]
        record = {k: v for k, v in row.items() if k not in ["fields", "department_evidence", "methodology_notes"]}
        record.update({name + "_ytd_yi": value for name, value in current.items()})
        record["cumulative_source_months"] = dependencies.copy()
        record["current_ytd_is_direct_report"] = row["reported_interval"] == "YEAR_TO_DATE" or month.month == 1
        for name in NAMES:
            record[name + "_reported_month_or_difference_yi"] = (current[name] - previous[name]) if previous is not None else (current[name] if month.month == 1 else np.nan)
        denominator = current["rmb_total"]
        record["ratio_available"] = denominator > 0
        for kind, numerator in [("long_term_net_loan_share_percent", current["household_long"] + current["corporate_long"]),
                                ("corporate_long_contribution_percent", current["corporate_long"]),
                                ("household_long_contribution_percent", current["household_long"]),
                                ("bill_contribution_percent", current["bills"])]:
            record[kind] = numerator / denominator * 100 if denominator > 0 else np.nan
        record["corporate_unlisted_net_component_ytd_yi"] = current["corporate_total"] - current["corporate_short"] - current["corporate_long"] - current["bills"]
        transformed.append(record)
    frame = pd.DataFrame(transformed)
    if len(frame):
        frame.to_parquet(OUT / "released_loan_composition.parquet", index=False)
        np.testing.assert_allclose(frame.long_term_net_loan_share_percent,
                                   frame.corporate_long_contribution_percent + frame.household_long_contribution_percent, equal_nan=True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "RELEASED_LOAN_COMPOSITION_READY" if len(frame) == 104 and not failures else "PARTIAL_SOURCE_PARSE_REQUIRES_EXPLICIT_COMPLETION",
        "source_months": 104, "extracted_months": len(releases), "canonical_ytd_months": len(frame),
        "reported_intervals": pd.Series([r["reported_interval"] for r in releases]).value_counts().to_dict(),
        "direct_cumulative_months": int(frame.current_ytd_is_direct_report.sum()) if len(frame) else 0,
        "regime_counts": frame.statistical_regime.value_counts().to_dict() if len(frame) else {},
        "failures": failures, "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "goal_achieved": False}, True)
    print(f"104个月中提取{len(releases)}个月、形成累计{len(frame)}个月；待处理{len(failures)}条。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人民币贷款部门和期限构成原文提取")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    globals()[args.command]()
