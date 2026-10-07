"""提取已公布的住户存款与贷款净增加差额，不把它称为净储蓄或股市流入。"""
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

import research.rmb_loan_composition_source_v1 as primitives
import research.rmb_loan_composition_completion_v1 as loans
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_household_bank_balance_source_v1"
STUDY = "510300_HOUSEHOLD_BANK_BALANCE_SOURCE_V1"
NAMES = ["rmb_deposit_total", "household_deposit"]
FEATURE = "household_deposit_loan_flow_gap_percent_of_deposits"


def selected(blocks):
    direct = [b for b in blocks if b["reported_interval"] == "YEAR_TO_DATE"]
    assert len(direct) <= 1 and blocks, "同一字段累计值不唯一，或缺少原文。"
    if direct:
        return direct[0]
    assert len(blocks) == 1, "当月值不唯一。"
    return blocks[0]


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
    compact = re.sub(r"\s+", "", body.get_text(" ", strip=True)).replace("，", ",")
    stocks = list(re.finditer(r"(?:月末|。)人民币存款余额(\d+(?:\.\d+)?)万亿元,?同比(增长|下降)(\d+(?:\.\d+)?)%", compact))
    assert len(stocks) == 1, "人民币存款余额与官方同比不唯一。"
    stock = stocks[0]
    section = compact[stock.start():]
    foreign = re.search(r"(?:\d{1,2}月末|月末),?(?:金融机构)?外币存款余额", section)
    assert foreign, "没有找到存款章节边界。"
    section = section[:foreign.start()]
    totals = list(re.finditer(primitives.PREFIX + r",?人民币存款(?:累计)?" + primitives.AMOUNT, section))
    assert totals, "找不到存款总量及统计区间。"
    total_blocks, household_blocks = [], []
    for match in totals:
        begin, interval = primitives.period_start(match.group(1), row["stat_month"])
        total_blocks.append({"period_start": begin, "reported_interval": interval,
                             "amount": primitives.amount(match), "evidence": match.group(0)})
    for match in re.finditer(r"住户(?:部门)?存款(?:累计)?" + primitives.AMOUNT, section):
        before = [m for m in totals if m.end() <= match.start()]
        assert before
        total = before[-1]
        begin, interval = primitives.period_start(total.group(1), row["stat_month"])
        household_blocks.append({"period_start": begin, "reported_interval": interval,
                                 "amount": primitives.amount(match), "evidence": section[total.start():match.end()]})
    # 某些原文另外并列住户与企业的直接累计；只取确实披露的住户字段。
    joint_pattern = (primitives.PREFIX + r"住户存款和非金融企业存款分别"
                     r"(增加|减少)(\d+(?:\.\d+)?)(万亿元|亿元)和(\d+(?:\.\d+)?)(万亿元|亿元)")
    for match in re.finditer(joint_pattern, section):
        begin, interval = primitives.period_start(match.group(1), row["stat_month"])
        literal = "".join(match.groups()[1:4])
        value = primitives.amount(re.fullmatch(primitives.AMOUNT, literal))
        household_blocks.append({"period_start": begin, "reported_interval": interval,
                                 "amount": value, "evidence": match.group(0), "partial_cumulative_disclosure": True})
    known = pd.Timestamp(row["published_at"]).tz_convert("Asia/Shanghai").normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    assert pd.Period(row["stat_month"], freq="M").end_time.date() < known.date()
    return {"stat_month": row["stat_month"], "published_at": row["published_at"], "conservative_known_at": known,
            "source_url": row["source_url"], "source_sha256": row["source_sha256"], "raw_path": row["raw_path"],
            "retrieved_at": row["retrieved_at"], "rmb_deposit_stock_yi": float(stock.group(1)) * 10000,
            "rmb_deposit_stock_yoy_percent": float(stock.group(3)) * (-1 if stock.group(2) == "下降" else 1),
            "stock_evidence": stock.group(0), "deposit_section_evidence": section,
            "fields": {"rmb_deposit_total": selected(total_blocks), "household_deposit": selected(household_blocks)},
            "all_observed_blocks": {"rmb_deposit_total": total_blocks, "household_deposit": household_blocks},
            "statistical_regime": "EXPANDED_2023" if row["stat_month"] >= "2023-01" else "PRE_2023",
            "historical_first_vintage_verified": False}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("住户存贷差额来源已有终态，不能覆盖。")
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(primitives.SOURCE).sort_values("stat_month")
    assert len(table) == 104 and table.stat_month.is_unique
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "source": "复用2018-01至2026-08的104份已保存央行金融统计原文，不新增市场下载。",
        "fields": NAMES + ["rmb_deposit_stock_yi", "rmb_deposit_stock_yoy_percent"],
        "interval": "每一字段独立按原文识别当月或累计。直接累计优先；当月仅加此前已公布同年累计，每年重置。2022-04另报住户与企业累计，住户字段采用直接累计，其他字段不强行共享其区间。",
        "feature": "100乘（住户存款年初累计净增－住户贷款年初累计净增）除以当期人民币存款余额；分母为正的规模归一化，不用存款净增作易过零分母。",
        "economic_limit": "只描述住户银行存贷净增加差额，不能等同国民账户净储蓄、可投资现金、净财富、预期或流入股票的资金。存款利息、房地产交易、理财赎回等均可能影响数值。",
        "loan_source": (loans.OUT / "released_loan_composition.parquet").relative_to(ROOT).as_posix(),
        "loan_scope": "仅复用已提取的住户贷款总额；2022-04缺少期限不代表住户总贷款缺失。",
        "known_at": "原文公布日结束后方可使用，不回填统计月；2023年扩机构范围独立标记。",
        "training_dependency": "按字段记录构成累计数的报告月份，主变量使用住户存款与贷款依赖的并集；所有依赖供后续两年窗口检验。",
        "historical_first_vintage_verified": False, "source_parser_pretested_without_stock_returns": True,
        "new_accounts": 0, "new_market_downloads": 0, "goal_achieved": False}, True)
    paths = [Path(__file__), Path(primitives.__file__), primitives.SOURCE,
             loans.OUT / "released_loan_composition.parquet", loans.OUT / "result.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "raw_sources": {r.raw_path: r.source_sha256 for r in table.itertuples()}, "before_new_strategy_returns": True}, True)
    originals = [parse(r) for r in table.to_dict("records")]
    loan_frame = pd.read_parquet(loans.OUT / "released_loan_composition.parquet").set_index("stat_month")
    current, dependencies, records, last_month = {}, {}, [], None
    for original in originals:
        month = pd.Period(original["stat_month"], freq="M")
        before = current.copy() if month.month != 1 and last_month == str(month - 1) else {}
        for name in NAMES:
            field = original["fields"][name]
            value = field["amount"]["value_yi"]
            direct = field["reported_interval"] == "YEAR_TO_DATE" or month.month == 1
            if direct:
                current[name], dependencies[name] = value, [str(month)]
            else:
                assert name in before, "不能越过累计缺口。"
                current[name] = before[name] + value
                dependencies[name] = [*dependencies[name], str(month)]
        last_month = str(month)
        loan = loan_frame.loc[str(month)]
        assert loan.source_sha256 == original["source_sha256"]
        assert pd.Timestamp(loan.conservative_known_at) == original["conservative_known_at"]
        assert pd.notna(loan.household_total_ytd_yi)
        record = {k: v for k, v in original.items() if k not in ["fields", "all_observed_blocks", "deposit_section_evidence"]}
        record.update({name + "_ytd_yi": value for name, value in current.items()})
        record["household_loan_ytd_yi"] = float(loan.household_total_ytd_yi)
        record["rmb_loan_stock_yoy_percent"] = float(loan.rmb_stock_yoy_percent)
        record["household_bank_flow_gap_ytd_yi"] = current["household_deposit"] - float(loan.household_total_ytd_yi)
        record[FEATURE] = 100 * record["household_bank_flow_gap_ytd_yi"] / record["rmb_deposit_stock_yi"]
        record["cumulative_source_months"] = sorted(set(dependencies["household_deposit"]) | set(loan.cumulative_source_months))
        record["household_deposit_cumulative_source_months"] = dependencies["household_deposit"].copy()
        record["rmb_deposit_total_cumulative_source_months"] = dependencies["rmb_deposit_total"].copy()
        record["household_deposit_direct_ytd"] = original["fields"]["household_deposit"]["reported_interval"] == "YEAR_TO_DATE" or month.month == 1
        record["ratio_available"] = True
        records.append(record)
    frame = pd.DataFrame(records)
    assert len(frame) == 104 and np.isfinite(frame[FEATURE]).all()
    np.testing.assert_allclose(frame[FEATURE] * frame.rmb_deposit_stock_yi / 100,
                               frame.household_deposit_ytd_yi - frame.household_loan_ytd_yi, atol=1e-8, rtol=0)
    assert frame.loc[frame.stat_month.eq("2022-04"), "household_deposit_ytd_yi"].iloc[0] == 71200
    frame.to_parquet(OUT / "released_household_bank_balance.parquet", index=False)
    save(OUT / "extracted_originals.json", originals, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "104_RELEASED_HOUSEHOLD_BANK_BALANCE_MONTHS_READY", "source_months": len(frame),
        "first_month": frame.stat_month.min(), "last_month": frame.stat_month.max(),
        "joint_ratio_months": int(frame.ratio_available.sum()), "feature": FEATURE,
        "feature_range_percent": [float(frame[FEATURE].min()), float(frame[FEATURE].max())],
        "negative_flow_gap_months": frame.loc[frame.household_bank_flow_gap_ytd_yi.lt(0), "stat_month"].tolist(),
        "household_deposit_direct_cumulative_months": int(frame.household_deposit_direct_ytd.sum()),
        "original_deposit_reporting_intervals": pd.Series([r["fields"]["household_deposit"]["reported_interval"] for r in originals]).value_counts().to_dict(),
        "partial_cumulative_202204_used": True, "loan_total_known_during_202204_202205": True,
        "new_market_downloads": 0, "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "goal_achieved": False}, True)
    (OUT / "source_code.py").write_bytes(Path(__file__).read_bytes())
    print("104个月住户存贷差额已提取，2022年4月单独公布累计正确保留，尚未计算策略收益。", flush=True)


if __name__ == "__main__":
    run()
