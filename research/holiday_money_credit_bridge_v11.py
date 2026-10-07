"""连接春节支付时点、存款持有人与信贷期限结构；不生成股票交易规则。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_holiday_money_credit_bridge_v11"
PARENT = ROOT / "reports/research/510300_macro_transmission_context_v4/results/104个月_当时可见多层证据_不含未来标签.csv"
MONTHS = ["2019-12", "2020-01", "2020-02", "2020-12", "2021-01", "2021-02", "2022-01"]
CALENDARS = [
    {"name": "2020年原定春节安排", "url": "https://app.www.gov.cn/govdata/gov/201911/21/451111/article.html",
     "available_at": "2019-11-21T23:59:59+08:00", "check": "1月24日至30日", "note": "原定安排只用于确认春节落在1月；不代表疫情后实际假期没有延长。"},
    {"name": "2021年春节安排", "url": "https://app.www.gov.cn/govdata/gov/202011/25/465322/article.html",
     "available_at": "2020-11-25T23:59:59+08:00", "check": "2月11日至17日", "note": "当时已知的春节时点。"},
]


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def text_of(path):
    return BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)


def freeze():
    for part in ["inputs", "sources", "results", "code", "figures"]:
        (OUT / part).mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        return
    protocol = {"at": now(), "study_id": "510300_HOLIDAY_MONEY_CREDIT_BRIDGE_V11", "previous_turn_classification": "PROGRESS：V10已完成成分风险与共同变动分解。",
        "question": "2021年1月M1高增与中长期贷款多增，分别来自余额扩张、基数、持有人和期限结构的哪些变化？",
        "scope": "延续既定2021年1月病例，比较2019年12月至2020年2月、2020年12月至2021年2月；另保留2022年1月央行的春节解释及2025口径差异。",
        "selection": "既有病例和结果已经看过；本轮只计算金额与时序，不选择新股票收益窗口。",
        "clock": "2021-02-09T21:00:00+08:00为当时观察点；2021年2月金额3月10日才公布，只能事后检验；2022与2025说明仅作机制解释。",
        "amounts": "从七份保存的央行原文解析公布值及精度；2020年2月同时保留单月和公告累计，2021年累计由两个单月相加。",
        "vintages": "跨公告的数值差不等于官方后来同口径同比；原报同比与本轮跨公告差分别保存；精度误差界不涵盖修订或口径差。",
        "seasonality": "春节跨月在当时已知，但1—2月比较不是统计季调，不把全部差异归给春节；疫情、经营与其他支付仍可能影响。",
        "identities": "M1非现金部分=M1−M0，不能等同非金融企业全部存款；不同持有人存款净变动不能证明同一笔资金发生迁移。",
        "new_definition": "2025个人活期纳入后，单位活期到个人活期在其他条件不变下成为M1内部划转；不据此声称春节影响消失。",
        "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False}
    save("protocol.json", protocol)
    (ROOT / "config/510300_holiday_money_credit_bridge_v11.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(PARENT, OUT / "inputs/monthly_context.csv")
    full = pd.read_csv(PARENT).set_index("stat_month")
    receipts = []
    for month in MONTHS:
        row = full.loc[month]
        path = ROOT / row.raw_path
        assert sha(path) == row.source_sha256
        dest = OUT / "sources" / (month + "_金融统计原文.html")
        shutil.copy2(path, dest)
        dest.with_suffix(".txt").write_text(text_of(dest), encoding="utf-8")
        receipts.append({"name": dest.name, "month": month, "url": row.source_url, "sha256": sha(dest), "source_path": row.raw_path,
                         "published_at": row.published_at, "historical_first_version_authenticated": False})
    for name in ["统计局_货币供应量编制方法.html", "央行_狭义货币与广义货币.html"]:
        source = ROOT / "reports/research/510300_money_balance_transmission_v8/sources" / name
        shutil.copy2(source, OUT / "sources" / name)
        receipts.append({"name": name, "sha256": sha(source), "source_path": source.relative_to(ROOT).as_posix(), "historical_first_version_authenticated": False, "role": "仅解释统计范围，不回填2021信息集。"})
    save("source_receipts.json", {"at": now(), "items": receipts})
    save("freeze.json", {"at": now(), "protocol_sha256": sha(OUT / "protocol.json"), "monthly_input_sha256": sha(OUT / "inputs/monthly_context.csv"), "sources": receipts})


def calendars():
    receipts = []
    for spec in CALENDARS:
        path = OUT / "sources" / (spec["name"] + ".html")
        rec_path = path.with_suffix(".receipt.json")
        if rec_path.exists():
            receipts.append(json.loads(rec_path.read_text(encoding="utf-8")))
            continue
        response = requests.get(spec["url"], timeout=(15, 40), headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        decoded = response.content.decode("utf-8")
        text = BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)
        assert spec["check"] in "".join(text.split()), spec["name"]
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(text, encoding="utf-8")
        rec = {**spec, "sha256": sha(path), "retrieved_at": now(), "historical_first_version_authenticated": False}
        rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        receipts.append(rec)
    save("calendar_receipts.json", receipts)


def amount(segment, label, balance=False):
    regex = label + (r".*?余额" if balance else "") + r"\s*(增加|减少)?\s*([0-9]+(?:\.[0-9]+)?)\s*(万亿元|亿元)"
    match = re.search(regex, segment)
    assert match, (label, segment[:200])
    direction, literal, unit = match.groups()
    multiplier = 10000 if unit == "万亿元" else 1
    value = float(literal) * multiplier * (-1 if direction == "减少" else 1)
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    return {"value_yi": value, "rounding_half_yi": .5 * 10 ** (-decimals) * multiplier, "literal": match.group(0)}


def flow_record(month, interval, loans, deposits):
    corporate_start = re.search(r"企（事）业单位贷款", loans).start()
    household_start = re.search(r"住户(?:部门)?贷款", loans).start()
    household, corporate = loans[household_start:corporate_start], loans[corporate_start:]
    fields = {
        "loan_total": amount(loans, "人民币贷款"),
        "loan_household": amount(household, r"住户(?:部门)?贷款"),
        "loan_household_short": amount(household, "短期贷款"),
        "loan_household_long": amount(household, "中长期贷款"),
        "loan_corporate": amount(corporate, "企（事）业单位贷款"),
        "loan_corporate_short": amount(corporate, "短期贷款"),
        "loan_corporate_long": amount(corporate, "中长期贷款"),
        "loan_bills": amount(corporate, "票据融资"),
        "loan_nonbank": amount(loans, "非银行业金融机构贷款"),
        "deposit_total": amount(deposits, "人民币存款"),
        "deposit_household": amount(deposits, "住户存款"),
        "deposit_corporate": amount(deposits, "非金融企业存款"),
        "deposit_fiscal": amount(deposits, "财政性存款"),
        "deposit_nonbank": amount(deposits, "非银行业金融机构存款"),
    }
    row = {"month": month, "interval": interval}
    for key, fact in fields.items():
        for name, value in fact.items():
            row[key + "_" + name] = value
    return row


def build():
    freeze()
    calendars()
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮金额已保存，不覆盖。")
    context = pd.read_csv(OUT / "inputs/monthly_context.csv").set_index("stat_month")
    money, flows = [], []
    for month in MONTHS:
        text = text_of(OUT / "sources" / (month + "_金融统计原文.html"))
        sections = [text.index(k) for k in ["一、", "二、", "三、", "四、"]]
        first = text[sections[0]:sections[1]]
        row = {"month": month, "published_at": context.loc[month].published_at, "m1_yoy_pp": context.loc[month].m1_yoy_pp,
               "m2_yoy_pp": context.loc[month].m2_yoy_pp, "spread_pp": context.loc[month].spread_pp}
        for code in ["M0", "M1", "M2"]:
            item = amount(first, r"[（(]" + code + r"[）)]", balance=True)
            for key, value in item.items(): row[code.lower() + "_" + key] = value
        row["noncash_m1_yi"] = row["m1_value_yi"] - row["m0_value_yi"]
        row["noncash_m1_rounding_half_yi"] = row["m1_rounding_half_yi"] + row["m0_rounding_half_yi"]
        row["available_at_20210209_21"] = pd.Timestamp(row["published_at"]) <= pd.Timestamp("2021-02-09T21:00:00+08:00")
        money.append(row)
        if month in ["2020-01", "2020-02", "2021-01", "2021-02"]:
            loan, deposit = text[sections[1]:sections[2]], text[sections[2]:sections[3]]
            if month == "2020-02":
                li, di = loan.index("2月当月人民币贷款"), deposit.index("2月当月人民币存款")
                flows.append(flow_record(month, "YEAR_TO_DATE_REPORTED", loan[:li], deposit[:di]))
                loan, deposit = loan[li:], deposit[di:]
            flows.append(flow_record(month, "MONTH_REPORTED", loan, deposit))
    monetary = pd.DataFrame(money).set_index("month")
    flow = pd.DataFrame(flows)
    monetary.reset_index().to_csv(OUT / "results/货币余额与发布时点.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(OUT / "results/原文贷款存款分项.csv", index=False, encoding="utf-8-sig")
    fields = [name[:-len("_value_yi")] for name in flow.columns if name.endswith("_value_yi")]
    compare = []
    for interval in ["JAN", "JAN_FEB"]:
        a = flow[(flow.month == ("2020-01" if interval == "JAN" else "2020-02")) & (flow.interval == ("MONTH_REPORTED" if interval == "JAN" else "YEAR_TO_DATE_REPORTED"))].iloc[0]
        selected = flow[(flow.month.isin(["2021-01"] if interval == "JAN" else ["2021-01", "2021-02"])) & (flow.interval == "MONTH_REPORTED")]
        for key in fields:
            old, current = float(a[key + "_value_yi"]), float(selected[key + "_value_yi"].sum())
            bound = float(a[key + "_rounding_half_yi"] + selected[key + "_rounding_half_yi"].sum())
            compare.append({"interval": interval, "field": key, "2020_value_yi": old, "2021_value_yi": current,
                "cross_release_difference_yi": current - old, "rounding_bound_only_yi": bound,
                "version_note": "跨公告对照，非后来官方同口径同比；精度界不包含修订差。",
                "available_at_20210209_21": interval == "JAN"})
    comparison = pd.DataFrame(compare)
    comparison.to_csv(OUT / "results/一月与两月累计_同区间结构对照.csv", index=False, encoding="utf-8-sig")
    movements = []
    for first, last in [("2019-12", "2020-01"), ("2020-01", "2020-02"), ("2019-12", "2020-02"), ("2020-12", "2021-01"), ("2021-01", "2021-02"), ("2020-12", "2021-02")]:
        left, right = monetary.loc[first], monetary.loc[last]
        row = {"start": first, "end": last}
        for field in ["m0", "m1", "m2"]:
            row[field + "_change_yi"] = float(right[field + "_value_yi"] - left[field + "_value_yi"])
            row[field + "_change_rounding_bound_yi"] = float(right[field + "_rounding_half_yi"] + left[field + "_rounding_half_yi"])
        row["noncash_m1_change_yi"] = float(right.noncash_m1_yi - left.noncash_m1_yi)
        row["spread_change_pp"] = float(right.spread_pp - left.spread_pp)
        movements.append(row)
    pd.DataFrame(movements).to_csv(OUT / "results/余额变动与剪刀差变动.csv", index=False, encoding="utf-8-sig")
    log_changes = []
    for endpoint, base_endpoint in [("2021-01", "2020-01"), ("2021-02", "2020-02")]:
        current, base = monetary.loc[endpoint], monetary.loc[base_endpoint]
        start, old_start = monetary.loc["2020-12"], monetary.loc["2019-12"]
        current_part = 100 * np.log((current.m1_value_yi / start.m1_value_yi) / (current.m2_value_yi / start.m2_value_yi))
        base_part = -100 * np.log((base.m1_value_yi / old_start.m1_value_yi) / (base.m2_value_yi / old_start.m2_value_yi))
        log_changes.append({"start": "2020-12", "end": endpoint, "current_relative_log_pp": current_part, "base_relative_log_pp": base_part,
            "total_relative_growth_log_pp": current_part + base_part, "ordinary_spread_change_pp": float(current.spread_pp - start.spread_pp),
            "note": "按公布余额跨公告复算；对数项不与普通百分点混加，不能据金额恒等式指定经济因果。"})
    pd.DataFrame(log_changes).to_csv(OUT / "results/一月与两月_当期基数对数分解.csv", index=False, encoding="utf-8-sig")
    # 保留跨公告差与官方后来同比的区别，不能把它们自动视作一致。
    january = comparison[(comparison.interval == "JAN") & comparison.field.isin(["loan_total", "deposit_total"])].copy()
    january["official_reported_yoy_yi"] = january.field.map({"loan_total": 2252., "deposit_total": 6245.})
    january["difference_from_official_yoy_yi"] = january.cross_release_difference_yi - january.official_reported_yoy_yi
    january.to_csv(OUT / "results/官方同比与跨公告差_保留版本残差.csv", index=False, encoding="utf-8-sig")
    source_2022 = text_of(OUT / "sources/2022-01_金融统计原文.html")
    assert "M1同比增长约2%" in "".join(source_2022.split()) and "单位活期存款会向个人存款转移" in source_2022
    summary = {"at": now(), "status": "HOLIDAY_BALANCE_AND_CREDIT_COMPOSITION_BUILT", "money_months": len(money), "reported_flow_rows": len(flow),
        "comparison_rows": len(comparison), "calendar_sources": 2, "official2022": {"reported_m1_yoy_pp": -1.9, "holiday_adjusted_about_pp": 2.0,
        "role": "独立年份官方机制解释，不能回填为2021事前信息。"}, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "independent_validation": False}
    save("results/build_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False))
    print(comparison[comparison.field.isin(["loan_corporate", "loan_corporate_long", "loan_household", "deposit_corporate", "deposit_household"])].to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    build()
