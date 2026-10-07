"""补齐多统计区间原文解析，保留2022年4月未公布的住户期限缺口。"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.rmb_loan_composition_source_v1 as prior
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_rmb_loan_composition_completion_v1"
STUDY = "510300_RMB_LOAN_COMPOSITION_COMPLETION_V1"
CORPORATE = r"(?:非金融企业及机关团体|企\(事\)业单位)贷款"


def optional_amount(text, prefix):
    found = list(re.finditer(prefix + r"(?:累计)?" + prior.AMOUNT, text))
    assert len(found) <= 1, (prefix, len(found))
    return prior.amount(found[0]) if found else None


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
    households = list(re.finditer(r"住户(?:部门)?贷款(?:累计)?(?:增加|减少)", compact))
    assert households
    blocks = []
    for household in households:
        start = household.start()
        corporate = re.search(CORPORATE, compact[start:])
        assert corporate
        corporate_start = start + corporate.start()
        nonbank = re.search(r"非银行业金融机构贷款(?:累计)?" + prior.AMOUNT, compact[corporate_start:])
        assert nonbank
        end = corporate_start + nonbank.end()
        totals = list(re.finditer(prior.PREFIX + r"[,，]?人民币贷款(?:累计)?" + prior.AMOUNT, compact[:start]))
        assert totals
        total = totals[-1]
        begin, scope = prior.period_start(total.group(1), row["stat_month"])
        hh_text, co_text = compact[start:corporate_start], compact[corporate_start:corporate_start + nonbank.start()]
        fields = {"rmb_total": prior.amount(total),
                  "household_total": optional_amount(hh_text, r"住户(?:部门)?贷款"),
                  "household_short": optional_amount(hh_text, r"短期贷款"),
                  "household_long": optional_amount(hh_text, r"中长期贷款"),
                  "corporate_total": optional_amount(co_text, CORPORATE),
                  "corporate_short": optional_amount(co_text, r"短期贷款"),
                  "corporate_long": optional_amount(co_text, r"中长期贷款"),
                  "bills": optional_amount(co_text, r"票据融资"), "nonbank_total": prior.amount(nonbank)}
        missing = [key for key, value in fields.items() if value is None]
        if missing:
            assert row["stat_month"] == "2022-04" and missing == ["household_short", "household_long"], (row["stat_month"], missing)
        blocks.append({"period_start": begin, "period_end": row["stat_month"], "reported_interval": scope,
                       "period_literal": total.group(1), "fields": fields, "total_evidence": total.group(0),
                       "department_evidence": compact[start:end], "missing_fields": missing})
    direct = [block for block in blocks if block["reported_interval"] == "YEAR_TO_DATE"]
    assert len(direct) <= 1
    chosen = direct[0] if direct else blocks[0]
    stocks = list(re.finditer(r"(?:月末|。)人民币贷款余额(\d+(?:\.\d+)?)万亿元,?同比(增长|下降)(\d+(?:\.\d+)?)%", compact.replace("，", ",")))
    assert len(stocks) == 1, "只取总贷款章节的月末人民币贷款余额，排除社融章节实体口径。"
    stock = stocks[0]
    known = pd.Timestamp(row["published_at"]).tz_convert("Asia/Shanghai").normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return {"stat_month": row["stat_month"], **chosen, "published_at": row["published_at"],
            "conservative_known_at": known, "source_url": row["source_url"], "source_sha256": row["source_sha256"],
            "raw_path": row["raw_path"], "retrieved_at": row["retrieved_at"],
            "rmb_stock_yi": float(stock.group(1)) * 10000,
            "rmb_stock_yoy_percent": float(stock.group(3)) * (-1 if stock.group(2) == "下降" else 1),
            "all_observed_blocks": blocks,
            "methodology_notes": re.findall(r"注\d[：:].*?(?=注\d[：:]|$)", compact),
            "statistical_regime": "EXPANDED_2023" if row["stat_month"] >= "2023-01" else "PRE_2023",
            "historical_first_vintage_verified": False, "status": "SOURCE_COMPONENTS_EXTRACTED"}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "result.json").exists():
        raise RuntimeError("补全已有终态。")
    if (OUT / "protocol.json").exists():
        assert (OUT / "initial_completion_failure.json").exists() and not (OUT / "extracted_originals.json").exists()
    else:
        save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "previous_status": read(prior.OUT / "result.json")["status"],
        "completion": "104个原文不变。2018/2019/2020年2月同文有当月与累计两组，保存两组并优先直接累计；2026年合并报告限定总贷款章节的月末余额，排除社融实体口径。",
        "genuine_missing": "2022-04住户披露按住房/其他消费/经营分类，没有短期/中长期，不能把住房贷款当中长期；住户期限累计4—5月保持未知，6月直接累计公布后恢复，不回补。",
        "comparison": "此前94个成功月份逐字段核对不变；本阶段仍不计算股票收益。",
        "source_table_sha256": digest(prior.SOURCE), "new_market_downloads": 0,
            "new_accounts": 0, "new_strategy_returns": 0}, True)
    releases = [parse(row) for row in pd.read_csv(prior.SOURCE).sort_values("stat_month").to_dict("records")]
    old = {r["stat_month"]: r for r in read(prior.OUT / "extracted_originals.json")}
    for row in releases:
        if row["stat_month"] in old:
            previous = old[row["stat_month"]]
            for name in prior.NAMES:
                assert row["fields"][name]["value_yi"] == previous["fields"][name]["value_yi"]
            assert row["reported_interval"] == previous["reported_interval"]
            np.testing.assert_allclose(row["rmb_stock_yi"], previous["rmb_stock_yi"], atol=1e-8)
            assert row["rmb_stock_yoy_percent"] == previous["rmb_stock_yoy_percent"]
    save(OUT / "extracted_originals.json", releases, True)
    current, last_month, dependencies, rows = {}, None, [], []
    last_known = pd.Timestamp("2017-01-01", tz="Asia/Shanghai")
    for r in releases:
        month = pd.Period(r["stat_month"], freq="M")
        assert pd.Timestamp(r["conservative_known_at"]) > last_known
        last_known = pd.Timestamp(r["conservative_known_at"])
        previous = current.copy() if month.month != 1 and last_month == str(month - 1) else {}
        direct = r["reported_interval"] == "YEAR_TO_DATE" or month.month == 1
        for name in prior.NAMES:
            value = r["fields"][name]["value_yi"] if r["fields"][name] is not None else None
            current[name] = value if direct else (previous[name] + value if previous.get(name) is not None and value is not None else None)
        dependencies = [r["stat_month"]] if direct else [*dependencies, r["stat_month"]]
        last_month = r["stat_month"]
        record = {k: v for k, v in r.items() if k not in ["fields", "all_observed_blocks", "department_evidence", "methodology_notes", "missing_fields"]}
        record.update({name + "_ytd_yi": value for name, value in current.items()})
        record["cumulative_source_months"] = dependencies.copy()
        record["current_ytd_is_direct_report"] = direct
        for name in prior.NAMES:
            value, before = current[name], previous.get(name)
            record[name + "_reported_month_or_difference_yi"] = (value - before) if value is not None and before is not None else (value if month.month == 1 else np.nan)
        denominator = current["rmb_total"]
        valid = denominator is not None and denominator > 0
        joint = valid and current["corporate_long"] is not None and current["household_long"] is not None
        record["ratio_available"] = joint
        record["long_term_net_loan_share_percent"] = (current["corporate_long"] + current["household_long"]) / denominator * 100 if joint else np.nan
        for field, name in [("corporate_long_contribution_percent", "corporate_long"),
                            ("household_long_contribution_percent", "household_long"), ("bill_contribution_percent", "bills")]:
            record[field] = current[name] / denominator * 100 if valid and current[name] is not None else np.nan
        record["corporate_unlisted_net_component_ytd_yi"] = current["corporate_total"] - current["corporate_short"] - current["corporate_long"] - current["bills"]
        rows.append(record)
    frame = pd.DataFrame(rows)
    assert len(frame) == 104 and frame.stat_month.is_unique
    assert frame.loc[~frame.ratio_available, "stat_month"].tolist() == ["2022-04", "2022-05"]
    np.testing.assert_allclose(frame.long_term_net_loan_share_percent,
                               frame.corporate_long_contribution_percent + frame.household_long_contribution_percent, equal_nan=True)
    frame.to_parquet(OUT / "released_loan_composition.parquet", index=False)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY, "status": "104_RELEASES_102_JOINT_COMPOSITION_MONTHS_READY",
        "source_months": 104, "extracted_months": len(releases), "canonical_ytd_months": len(frame),
        "joint_ratio_months": int(frame.ratio_available.sum()), "unknown_joint_ratio_months": ["2022-04", "2022-05"],
        "prior_successful_months_unchanged": len(old),
        "reported_intervals": frame.reported_interval.value_counts().to_dict(),
        "direct_cumulative_months": int(frame.current_ytd_is_direct_report.sum()),
        "regime_counts": frame.statistical_regime.value_counts().to_dict(),
        "new_market_downloads": 0, "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "goal_achieved": False}, True)
    (OUT / "source_completion_code.py").write_bytes(Path(__file__).read_bytes())
    print("104个月原文构成已解析；102个月组合比率可用，2022年4—5月缺口保留。此前94个月成功值不变。", flush=True)


if __name__ == "__main__":
    run()
