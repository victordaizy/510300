"""将同月八份融资公告按当时可执行条件分类，并核对有限后续事实。"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_funded_repurchase_transmission_v1"
CATALOGUE = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1/documents.json"


def main() -> None:
    source_index = {row["document_id"]: row for row in json.loads(CATALOGUE.read_text(encoding="utf-8"))}
    selected = json.loads((OUT / "next_existing_source_cohort.json").read_text(encoding="utf-8"))["rows"]
    members = pd.read_parquet(ROOT / "reports/research/510300_shareholder_disclosure_breadth_v1/inputs/membership.parquet")
    members["membership_date"] = pd.to_datetime(members.membership_date)
    reviewed = {
        "001965.SZ": {"name": "招商公路", "state": "待股东大会审议", "credit_headline_cny": 432_600_000, "note": "框架协议及承诺函，贷款以具体协议和条件为准"},
        "001979.SZ": {"name": "招商蛇口", "state": "待股东大会审议", "credit_headline_cny": 702_000_000, "note": "已有董事会方案，股东大会尚未通过"},
        "002714.SZ": {"name": "牧原股份", "state": "既有已批准计划增加融资", "credit_headline_cny": 2_400_000_000, "note": "原30—40亿元计划，未因贷款上调计划金额"},
        "300274.SZ": {"name": "阳光电源", "state": "既有已批准计划增加融资", "credit_headline_cny": 420_000_000, "note": "原5—10亿元计划；公告表述为提升资金使用效率"},
        "300498.SZ": {"name": "温氏股份", "state": "既有已批准计划增加融资", "credit_headline_cny": 2_000_000_000, "note": "一家银行不超过10亿元借款合同，另一家不超过10亿元承诺函；原计划上限18亿元，不能把20亿元直接当购买额"},
        "300751.SZ": {"name": "迈为股份", "state": "同次披露董事会方案和融资", "credit_headline_cny": None, "note": "此前已有提议；本次批准0.5—1亿元计划，借款不超过回购金额100%，不擅自换成独立固定授信"},
        "600028.SH": {"name": "中国石化", "state": "既有已批准计划增加融资", "credit_headline_cny": 900_000_000, "note": "发行人回购与集团增持分开；本行仅发行人回购融资"},
        "601872.SH": {"name": "招商轮船", "state": "待股东大会审议", "credit_headline_cny": 443_000_000, "note": "贷款承诺函存在，原回购方案仍需股东大会审议"},
    }
    cohort = []
    for row in selected:
        context = reviewed[row["symbol"]]
        text = "".join(json.loads((ROOT / row["text_path"]).read_text(encoding="utf-8")))
        if context["state"] == "待股东大会审议":
            compact = "".join(text.split())
            assert "尚需提交" in compact and "股东大会审议" in compact
        effective_day = members.loc[members.membership_date.ge(row["catalogue_date"]), "membership_date"].min()
        in_members = bool((members.membership_date.eq(effective_day) & members.symbol.eq(row["symbol"])).any())
        cohort.append({**row, **context, "membership_session": effective_day.strftime("%Y-%m-%d"), "is_member": in_members})
    assert len(cohort) == 8 and all(row["is_member"] for row in cohort)

    followups = [
        {"symbol": "002714.SZ", "source_id": "1221627004", "publication_date": "2024-11-05", "observation": "10月18日至10月31日已披露累计金额相同", "reported_change_cny": 0.0, "period_start": "2024-10-18", "period_end": "2024-10-31", "exact_zero_claim": False},
        {"symbol": "300274.SZ", "source_id": "1221625606", "publication_date": "2024-11-04", "observation": "10月20日融资公告至10月31日已披露累计金额相同", "reported_change_cny": 0.0, "period_start": "2024-10-20", "period_end": "2024-10-31", "exact_zero_claim": False},
        {"symbol": "300751.SZ", "source_id": "1221623700", "publication_date": "2024-11-04", "observation": "截至10月31日原文明示尚未实施本次回购", "reported_change_cny": 0.0, "period_start": "2024-10-18", "period_end": "2024-10-31", "exact_zero_claim": True},
        {"symbol": "300498.SZ", "source_id": "1221602724", "publication_date": "2024-11-01", "observation": "10月累计购买增加，统计区间跨贷款公告前后", "reported_change_cny": 299_972_874.13, "period_start": "2024-09-30", "period_end": "2024-10-31", "exact_zero_claim": False},
        {"symbol": "600028.SH", "source_id": "1221602418", "publication_date": "2024-11-02", "observation": "10月累计购买增加，统计区间跨贷款公告前后", "reported_change_cny": 188_823_082.00, "period_start": "2024-09-30", "period_end": "2024-10-31", "exact_zero_claim": False},
    ]
    for row in followups:
        src = source_index[row["source_id"]]
        row["source_url"] = src["source_url"]
        row["local_text_path"] = src["text_path"]
        row["causal_effect_of_loan_identified"] = False
    approvals = [
        {"symbol": "001979.SZ", "meeting_date": "2024-11-01", "source_id": "1221604801", "publication_date": "2024-11-02", "reviewed_pages": [1, 2]},
        {"symbol": "601872.SH", "meeting_date": "2024-11-01", "source_id": "1221640921", "publication_date": "2024-11-07", "reviewed_pages": [1, 2, 3], "earlier_resolution_date_referenced_but_not_loaded": "2024-11-02"},
    ]
    for row in approvals:
        row["source_url"] = source_index[row["source_id"]]["source_url"]
        row["meaning"] = "用于事后确认原执行条件确实需要等待，未将未来批准信息提前放入10月信号。"

    caps = {"002714.SZ": 58.60, "300274.SZ": 97.00, "300751.SZ": 120.00}
    prices = pd.read_parquet(ROOT / "data/raw/constituents/000300_constituent_daily.parquet", columns=["date", "con_code", "raw_low", "raw_high", "raw_close"], filters=[
        ("con_code", "in", list(caps)), ("date", ">=", pd.Timestamp("2024-10-21")), ("date", "<=", pd.Timestamp("2024-10-31")),
    ])
    cap_checks = []
    for symbol, cap in caps.items():
        rows = prices.loc[prices.con_code.eq(symbol)]
        assert len(rows) == 9 and rows.raw_low.notna().all()
        cap_checks.append({
            "symbol": symbol, "buyback_price_ceiling": cap, "observed_sessions": len(rows),
            "sessions_with_low_at_or_below_ceiling": int(rows.raw_low.le(cap).sum()),
            "interpretation": "仅排除整个阶段始终高于价格上限这一解释；不证明资金到账、无信息敏感期、提交过订单或有可获得成交。",
        })
    prices.to_csv(OUT / "未增加购买案例的价格上限核对.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cohort).to_csv(OUT / "八家公司公告时的执行条件.csv", index=False, encoding="utf-8-sig")
    parent_result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    summary = {
        "study_id": "510300_FUNDED_REPURCHASE_TRANSMISSION_V1", "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "EIGHT_NOTICE_COHORT_DISTINGUISHES_CAPACITY_APPROVAL_AND_EXECUTION",
        "continuation_classification": "PROGRESS_FINANCING_CONSTRAINTS_AND_EXECUTION_EVIDENCE",
        "selection": "既有回购目录2024年10月18日至31日标题明确贷款、融资支持或授信的全部8份文件；8家均为该日或随后首个交易日的历史沪深300成员。不是全市场完整融资清单。",
        "cohort": cohort, "classification_counts": {"待股东大会审议": 3, "既有已批准计划增加融资": 4, "同次披露董事会方案和融资": 1},
        "followups": followups, "subsequent_approval_evidence": approvals,
        "remaining_unread_approval_chain": ["001965.SZ"],
        "three_no_increase_or_unimplemented_cases": list(caps), "price_ceiling_checks": cap_checks,
        "parent_diagnostic_result": "reports/research/510300_funded_repurchase_transmission_v1/result.json",
        "price_context": parent_result["price_context"],
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_models": 0,
        "historical_reconstruction": True, "independent_validation": False,
        "net_sharpe": None, "goal_achieved": False, "current_market_forecast_made": False,
        "next_action": "实际购买时点还需要明确执行安排；先用已有原公告跟踪三个未增加购买案例的首次新执行及剩余期限，不从贷款标题直接产生510300买入规则。",
    }
    (OUT / "cohort_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table = "\n".join(f"| {row['name']} | {row['state']} | {row['note']} |" for row in cohort)
    report = f"""# 融资支持、执行条件与短期购买：八家公司对照

**融资渠道改善已是可核验事实，但它不能单独确定接下来几周的购买金额和股价方向。** 本轮从两家公司扩展到既有目录同月8份融资公告，逐份区分计划状态，并用少量后续公告反对过强的解释。全部公司均核对为当时沪深300成员；样本不按未来股价或购买量选择，但目录并非全市场融资事件全集。

## 公告当时，什么约束已改变

| 公司 | 公告时状态 | 对未来购买的实际含义 |
|---|---|---|
{table}

3家当时仍需股东大会批准，4家为已有已批准计划增加融资，1家同时披露正式董事会方案和贷款。待批准组中，招商蛇口和招商轮船后续报告确认股东大会在11月1日通过；这说明10月贷款公告与计划可执行时间并不同步。招商公路的后续批准链本轮未继续读取，保留未知。

来源示例：[招商蛇口融资公告](https://static.cninfo.com.cn/finalpage/2024-10-20/1221442994.PDF)、[招商轮船融资公告](https://static.cninfo.com.cn/finalpage/2024-10-21/1221443132.PDF)、[温氏融资公告](https://static.cninfo.com.cn/finalpage/2024-10-20/1221443239.PDF)。全部8份来源列在同目录CSV和结果文件中。

## 后续事实反对了哪种判断

- 牧原：到10月31日的累计回购仍与10月18日首次回购金额相同，约1亿元；24亿元贷款合同未对应月底前同额新购买。
- 阳光电源：到10月31日累计回购约2.712亿元，与10月20日融资公告披露的累计值相同。
- 迈为：11月4日公告明确，截至10月31日尚未实施本次回购。
- 温氏与中国石化：10月累计金额分别增加约3.00亿元、1.89亿元，但统计区间跨越贷款公告前后，不能全算作贷款引起的后续买入。

来源：[牧原10月末进展](https://static.cninfo.com.cn/finalpage/2024-11-05/1221627004.PDF)、[阳光电源10月末进展](https://static.cninfo.com.cn/finalpage/2024-11-04/1221625606.PDF)、[迈为10月末进展](https://static.cninfo.com.cn/finalpage/2024-11-04/1221623700.PDF)、[温氏10月末进展](https://static.cninfo.com.cn/finalpage/2024-11-01/1221602724.PDF)、[中国石化10月末进展](https://static.cninfo.com.cn/finalpage/2024-11-02/1221602418.PDF)。其中累计数字相同按披露精度解释，不扩展成未披露资金流的精确零值。

还核对了一项竞争解释：牧原、阳光电源、迈为在10月21日至31日的9个交易日，每天最低价均曾不高于公告回购上限。因而“整个阶段一直贵于回购上限，无法买入”不能解释这三个案例。日线触及仅提供必要价格条件；资金是否到账、信息敏感期、公司内部资金安排与实际委托仍未识别，不能再补写成“公司没有信心”。

## 怎样判断未来，而不是重复标题

这组材料形成了可操作的研究顺序：**是否已获执行批准 → 融资是否落实到可用资金 → 旧计划还差多少、何时结束 → 是否已有明确执行安排 → 规模对指数是否足够重要 → 价格还留多少空间。** 每一步对应不同的行为限制，不能把几项相关公告重复当成多份买盘。

中国石化较近期限的案例提供一条条件线索：10月末累计回购约2.36亿元，距原8亿元下限仍差约5.64亿元；11月22日完成时累计约8.16亿元，11月新增约5.80亿元。履行剩余计划可以解释这段购买，但缺乏对照，尚不能归因于期限或贷款。回购统计日之后、公告日之前可能已发生的购买，也不能再次算成未来需求。

在510300上，两次信息更新后的固定20个开盘间隔毛收益分别为+0.95%和−2.59%；观察起点相对9月23日收盘已经上涨约21.90%和26.01%。这只是相关历史窗口，含应得分红、未扣交易成本，既不是新策略账户，也不是政策因果收益。具体日期、价格和计算见《融资条件到实际购买的传导结论.md》及对应CSV。

因此本轮不把“融资获批”升级为直接买入规则。下一项需要解决的是实际执行时点及尚未兑现的需求，再判断其指数影响。已有单纯政策事件账户压力夏普约0.690，旧结果保持不变；新研究尚未证明成本后优势，净夏普1.2及年化10%的目标仍未实现。
"""
    (OUT / "八家公司融资与执行对照.md").write_text(report, encoding="utf-8")
    print("八份融资公告已分类：3份待股东大会审议、4份已有计划增加融资、1份同次批准方案和融资。")
    print("五家公司月末执行已核对；三个未增加或未实施案例均不能由全程高于价格上限解释。")
    print("仅形成行为约束与竞争解释证据，没有新增账户或宣称预测成功。")


if __name__ == "__main__":
    main()
