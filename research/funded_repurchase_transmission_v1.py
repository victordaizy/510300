"""核对贷款、计划与实际回购的传导，不重新运行已失败的账户。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_funded_repurchase_transmission_v1"
CATALOGUE = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1/documents.json"
MARKET = ROOT / "reports/research/510300_shareholder_disclosure_breadth_v1/inputs/market.parquet"
STOCKS = ROOT / "data/raw/constituents/000300_constituent_daily.parquet"


def write_json(name: str, value: dict | list) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    scope_path = OUT / "diagnostic_scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["price_context_rule"] = "10月21日和11月5日结束时所选公告均已公开；日期无精确时间时一律按23:59:59可知，下一交易日09:00完成审核，09:30开盘起至20个开盘间隔后的开盘作价格诊断。两窗相关，不组成新策略或独立样本。"
    scope["clock_clarification_before_price_context_computation"] = "把原文字中的16时更正为日期结束，保留原下一日开盘评价日期；该修正先于本脚本首次价格计算。"
    write_json("diagnostic_scope.json", scope)

    catalogue = {item["document_id"]: item for item in json.loads(CATALOGUE.read_text(encoding="utf-8"))}
    used_ids = ["1221307213", "1221432448", "1221443142", "1221627004", "1221906544",
                "1221354445", "1221443124", "1221602418", "1221817323"]
    source_rows = []
    text_by_id = {}
    for doc_id in used_ids:
        row = catalogue[doc_id]
        pages = json.loads((ROOT / row["text_path"]).read_text(encoding="utf-8"))
        text_by_id[doc_id] = "\n".join(pages)
        source_rows.append({key: row[key] for key in ["document_id", "symbol", "title", "catalogue_date", "source_url", "raw_path", "text_path"]})
    for doc_id, amount in [("1221443124", "47,315,369.00"), ("1221602418", "236,138,451.00"),
                           ("1221817323", "816,001,427.20"), ("1221627004", "10,000.11"),
                           ("1221906544", "49,997.08")]:
        assert amount in text_by_id[doc_id], f"原文金额未匹配：{doc_id}"
    source_rows.append({
        "document_id": "HKEX_2024102000005", "symbol": "600028.SH",
        "title": "中国石化集团获得增持贷款支持暨增加资金来源", "catalogue_date": "2024-10-20",
        "source_url": "https://www.hkexnews.hk/listedco/listconews/sehk/2024/1020/2024102000005_c.pdf",
        "raw_path": str((OUT / "sources/sinopec_parent_credit_20241020.pdf").relative_to(ROOT)),
        "text_path": str((OUT / "sources/sinopec_parent_credit_20241020_pages.json").relative_to(ROOT)),
    })
    write_json("issuer_sources.json", source_rows)

    plans = [
        {"symbol": "600028.SH", "actor": "上市公司", "kind": "A股回购",
         "board_date": "2024-08-23", "plan_minimum_cny": 800_000_000,
         "plan_maximum_cny": 1_500_000_000, "deadline": "2024-11-22",
         "deadline_basis": "8月23日起不超过3个月，三个月边界11月23日是周六，最后常规交易日为11月22日；不以实际完成日反推期限。",
         "purpose": "注销并减少注册资本", "loan_or_credit_max_cny": 900_000_000,
         "loan_terms_known_by_date": "2024-10-21", "loan_scope": "包含本轮但不限于本轮回购",
         "plan_amount_increased_in_loan_notice": False, "actual_drawdown_identified": False},
        {"symbol": "002714.SZ", "actor": "上市公司", "kind": "A股回购",
         "board_date": "2024-09-25", "plan_minimum_cny": 3_000_000_000,
         "plan_maximum_cny": 4_000_000_000, "deadline": "2025-09-24",
         "deadline_basis": "9月25日董事会通过起12个月内，按含首日区间计算。",
         "purpose": "员工持股或股权激励", "loan_or_credit_max_cny": 2_400_000_000,
         "loan_terms_known_by_date": "2024-10-20", "loan_scope": "已披露股份回购计划",
         "loan_rate": 0.0225, "loan_maturity": "2025-10-20",
         "plan_amount_increased_in_loan_notice": False, "actual_drawdown_identified": False},
    ]
    parent = {
        "symbol": "600028.SH", "actor": "控股股东及全资子公司", "kind": "股东增持",
        "old_plan_scope": "A股及H股合计", "old_plan_minimum_cny": 1_000_000_000,
        "old_plan_maximum_cny": 2_000_000_000, "old_plan_spent_approx_cny": 1_291_000_000,
        "a_share_spent_approx_cny": 549_000_000, "credit_max_cny": 700_000_000,
        "new_statement": "预计本轮及未来新一期计划后续A股增持金额不低于7亿元；具体实施仍取决于法规、市场和实际资金情况。",
        "hard_purchase_date_known": False, "new_total_plan_maximum_disclosed": False,
        "new_statement_is_unconditional_payment_obligation": False,
        "near_term_20_day_buy_amount_identified": False,
        "double_count_boundary": "不与发行人9亿元授信混为同一计划，不把A/H混合累计金额全算作A股需求。",
    }

    statements = [
        {"symbol": "600028.SH", "publication_date": "2024-10-10", "statistics_date": "2024-09-30", "spent_cny": 47_315_369.00, "source_id": "1221354445"},
        {"symbol": "600028.SH", "publication_date": "2024-11-02", "statistics_date": "2024-10-31", "spent_cny": 236_138_451.00, "source_id": "1221602418"},
        {"symbol": "600028.SH", "publication_date": "2024-11-23", "statistics_date": "2024-11-22", "spent_cny": 816_001_427.20, "source_id": "1221817323"},
        {"symbol": "002714.SZ", "publication_date": "2024-10-19", "statistics_date": "2024-10-18", "spent_cny": 100_001_100.00, "source_id": "1221432448"},
        {"symbol": "002714.SZ", "publication_date": "2024-11-05", "statistics_date": "2024-10-31", "spent_cny": 100_001_100.00, "source_id": "1221627004"},
        {"symbol": "002714.SZ", "publication_date": "2024-12-03", "statistics_date": "2024-11-30", "spent_cny": 499_970_800.00, "source_id": "1221906544"},
    ]
    changes = []
    for plan in plans:
        previous = None
        for row in [item for item in statements if item["symbol"] == plan["symbol"]]:
            if previous is not None:
                changes.append({
                    "symbol": row["symbol"], "statistics_start": previous["statistics_date"],
                    "statistics_end": row["statistics_date"], "known_on_date_end": row["publication_date"],
                    "reported_purchase_increment_cny": round(row["spent_cny"] - previous["spent_cny"], 2),
                    "source_id": row["source_id"], "increment_attributed_to_loan": False,
                })
            previous = row

    daily = pd.read_parquet(STOCKS, columns=["date", "con_code", "amount", "is_index_member"], filters=[
        ("con_code", "in", ["600028.SH", "002714.SZ"]),
        ("date", ">=", pd.Timestamp("2024-09-01")), ("date", "<=", pd.Timestamp("2024-11-30")),
    ])
    market = pd.read_parquet(MARKET)
    snapshots = []
    price_context = []
    reference = float(market.loc[market.date.eq("2024-09-23"), "close"].iloc[0])
    for cutoff in ["2024-10-21", "2024-11-05"]:
        for plan in plans:
            available = [row for row in statements if row["symbol"] == plan["symbol"] and row["publication_date"] <= cutoff]
            latest = max(available, key=lambda row: row["publication_date"])
            amount_rows = daily.loc[daily.con_code.eq(plan["symbol"]) & daily.date.lt(cutoff)].sort_values("date").tail(20)
            assert len(amount_rows) == 20
            denominator = float(amount_rows.amount.sum())
            gap = max(0.0, plan["plan_minimum_cny"] - latest["spent_cny"])
            day_row = daily.loc[daily.date.eq(cutoff) & daily.con_code.eq(plan["symbol"])]
            assert len(day_row) == 1
            snapshots.append({
                "cutoff_date_end": cutoff, "symbol": plan["symbol"],
                "latest_statistics_date": latest["statistics_date"], "latest_publication_date": latest["publication_date"],
                "reported_spent_cny": latest["spent_cny"], "disclosed_gap_to_minimum_cny": gap,
                "calendar_days_to_deadline": (pd.Timestamp(plan["deadline"]) - pd.Timestamp(cutoff)).days,
                "prior20_turnover_cny": denominator, "disclosed_gap_over_prior20_turnover": gap / denominator,
                "index_member_in_saved_daily": bool(day_row.is_index_member.iloc[0]),
                "actual_remaining_cash_known": False, "actual_future_daily_schedule_known": False,
            })
        entry_idx = int(market.index[market.date.gt(cutoff)][0])
        exit_idx = entry_idx + 20
        entry, exit_row = market.iloc[entry_idx], market.iloc[exit_idx]
        dividend = float(market.iloc[entry_idx + 1:exit_idx + 1].dividend.sum())
        price_context.append({
            "information_cutoff_date_end": cutoff, "decision_date": entry.date.strftime("%Y-%m-%d"),
            "entry_open": float(entry.open), "exit_date": exit_row.date.strftime("%Y-%m-%d"),
            "exit_open": float(exit_row.open), "dividend_entitlement_per_share": dividend,
            "gross_20_open_interval_return": float((exit_row.open + dividend) / entry.open - 1),
            "price_change_since_20240923_close_before_entry": float(entry.open / reference - 1),
            "kind": "历史解释窗口；并非账户交易、因果归因或独立样本",
        })

    write_json("plans_and_parent_statement.json", {"issuer_plans": plans, "parent_statement": parent})
    write_json("reported_execution_timeline.json", {"statements": statements, "changes": changes})
    pd.DataFrame(snapshots).to_csv(OUT / "当时可知的剩余计划与期限.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(OUT / "后续披露的购买变化.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(price_context).to_csv(OUT / "510300价格已反映部分与后续路径.csv", index=False, encoding="utf-8-sig")
    result = {
        "study_id": scope["study_id"], "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "LOAN_HEADLINE_NOT_INCREMENTAL_DEMAND_TIMING_REQUIRES_PLAN_CONTEXT",
        "continuation_classification": "PROGRESS_FINANCING_PLAN_AND_EXECUTION_DISTINGUISHED",
        "historical_reconstruction": True, "independent_prediction_validation": False,
        "cases": 2, "financing_channels_reviewed": 3,
        "reused_issuer_source_documents": len(used_ids), "new_issuer_source_documents": 1,
        "findings": [
            "两份公司回购贷款公告均未上调原回购金额；授信或合同提高融资可获得性，不能作为旧计划之外等额新增购买。",
            "中国石化集团另外表达后续A股购买预期，必须与发行人回购及旧A/H混合计划分开，短期安排未知。",
            "牧原贷款合同后至10月末未见累计金额增加，11月披露新增约4亿元；贷款规模不能直接预测几周内同额购买。",
            "中国石化在较近截止日前11月新增回购约5.80亿元；与到期履行解释相符，但两案例不足以识别期限因果效应。",
        ],
        "snapshots": snapshots, "reported_changes": changes, "price_context": price_context,
        "old_policy_account_rechecked": {"stress_net_sharpe": 0.689647, "stress_cagr": 0.046075, "source": "reports/research/510300_equity_support_policy_event_v1/results/账户指标.csv", "rerun": False},
        "new_accounts": 0, "new_strategy_return_tests": 0, "diagnostic_price_windows": 2,
        "net_sharpe": None, "remaining_edge_status": "NOT_ESTABLISHED",
        "goal_achieved": False, "current_market_forecast_made": False,
        "next_action": "判断接下来数周的购买应同时看资金可得、剩余承诺、期限及实际执行；后续需在更广的当时成分范围验证其指数影响，不重复单纯政策买入或过去回购强度规则。",
        "verification": "原公告金额与保存文本逐项对应；三条资金路径分开；每个快照只取截止日已公开统计；分红按持有权利计入价格诊断。",
    }
    write_json("result.json", result)

    context_table = "\n".join(
        f"| {row['information_cutoff_date_end']}结束 | {row['decision_date']} | "
        f"{row['price_change_since_20240923_close_before_entry']:.2%} | {row['exit_date']} | "
        f"{row['gross_20_open_interval_return']:.2%} |"
        for row in price_context
    )
    report = f"""# 融资支持怎样转成接下来几周的购买

本轮确认了一个影响判断的区别：**贷款额度、原回购计划和实际购买，是同一条资金传导链上不同阶段的数字。** 它们不能相加。两家公司的贷款公告都没有上调原回购计划金额；未来需求变化可能来自执行概率提高、执行提前或后续新计划，具体需要分别确认。

本轮是历史机制诊断，已经知道后续结果，不是当年的真实预测记录。选择牧原和中国石化用于区分期限、资金主体及用途，不代表首批公司的完整样本。未新增账户或拟合模型。

## 1. 政策首先改变了融资约束

2024年10月的政策允许符合条件的银行向公司及主要股东提供股票回购增持贷款，银行自主决定放贷并承担风险，合格贷款随后可申请央行再贷款。央行额度与企业贷款是同一传导链，不能当作两笔独立股票需求。[原政策通知](https://www.pbc.gov.cn/redianzhuanti/118742/5444129/5444142/481faf4556cb42a5b444677b9673ea5c/index.html)

因而应依次判断：原来约束是资金成本、资金缺口，还是企业没有购买意愿；贷款是否解决了这个约束；解决之后是否扩大计划、提高原计划完成的可能性、提前执行，或仅替换资金来源。这四种结果对未来几周的含义不同。

## 2. 同批贷款的三条路径

| 主体 | 贷款或授信 | 公告时原有安排 | 新信息及剩余问题 |
|---|---:|---|---|
| 中国石化上市公司 | 不超过9亿元 | 8—15亿元A股回购，原期限至2024年11月，资金原为自有 | 增加融资来源，公告未扩大原金额；授信还可覆盖其他回购，不能全部指定为本轮已提款 |
| 牧原股份上市公司 | 24亿元专项合同 | 30—40亿元回购，董事会2024年9月通过，期限12个月 | 为旧计划提供融资支持，合同并不等于已提款24亿元或立即买入24亿元 |
| 中国石化集团 | 7亿元授信 | 旧增持安排A/H合计10—20亿元，当时已约12.91亿元 | 另外表达本轮及未来一期后续A股购买预计不低于7亿元；具体日期、各期归属仍未确定 |

来源：[中国石化公司公告](https://static.cninfo.com.cn/finalpage/2024-10-21/1221443124.PDF)、[牧原专项合同公告](https://static.cninfo.com.cn/finalpage/2024-10-20/1221443142.PDF)、[中国石化集团增持公告](https://www.hkexnews.hk/listedco/listconews/sehk/2024/1020/2024102000005_c.pdf)。集团的措辞是有条件的购买预期，不能当作确定付款义务；A/H累计金额也不能全部变成A股需求。

## 3. 后续执行检验了什么

牧原10月18日首次回购约1亿元。11月5日公布的截至10月31日累计金额仍为约1亿元；12月3日公布的截至11月30日累计金额约5亿元。因此贷款合同之后到10月底，已披露累计金额未增加，11月则新增约4亿元。[10月末进度](https://static.cninfo.com.cn/finalpage/2024-11-05/1221627004.PDF)、[11月末进度](https://static.cninfo.com.cn/finalpage/2024-12-03/1221906544.PDF)

中国石化截至10月31日累计回购约2.36亿元，距8亿元下限的披露缺口约5.64亿元；原期限较近。11月22日完成时累计约8.16亿元，11月新增约5.80亿元，与完成原计划的解释相符。[10月末进度](https://static.cninfo.com.cn/finalpage/2024-11-02/1221602418.PDF)、[完成公告](https://static.cninfo.com.cn/finalpage/2024-11-23/1221817323.PDF)

这两例使“剩余承诺与期限”值得继续研究，但还不是期限导致买入的统计证据。部分累计区间跨越贷款公告前后，实际提款额和资金对应关系未确认，不能把区间内所有购买都归功于贷款。累计金额没有改变也只表示披露精度下没有新增，保留舍入和披露边界。

## 4. 在当时怎样推演未来

10月21日结束时，两个公司计划的披露缺口分别约7.53亿元、29.00亿元。中国石化按原安排剩32个自然日，牧原仍剩338日；这些是距计划结束的时间，不是每日匀速买入指令。到11月5日信息更新后，石化披露缺口约5.64亿元，距离期限17日，牧原仍约29亿元、距离期限323日。

合理的条件判断是：若石化继续履行且不修改方案，剩余期限内仍需完成相当规模的购买；若牧原维持长期方案，短期暂停购买与继续履约可以同时成立。资料中未公开的最新成交仍未知，因此“下限减已披露金额”只能叫披露缺口，不能保证全部是未来需求。

用途也影响后续：石化回购用于注销，牧原用于员工持股或股权激励。当前交易需求与以后股本、分配和激励效果应分开讨论。即使能判断公司将继续买入，股价仍受盈利、估值和其他投资者卖盘影响；这两家公司的买盘也不足以单独代表整个沪深300。

## 5. 价格是否已提前反映

以下沿用本地510300原始行情，按上述两个信息更新点计算固定20个开盘间隔；持有期间有权取得的分红计入毛收益。日期无准确时间的公告按当天结束后才可用，次日盘前完成判断。两段是相关的解释窗口，没有费用、仓位、排队或完整账户，不能计算成策略夏普。

| 信息齐全时点 | 后续观察起点开盘 | 相对9月23日收盘已变化 | 20间隔终点开盘 | 后续毛收益 |
|---|---|---:|---|---:|
{context_table}

前期涨幅只是已经发生的价格变化，不等于量化了市场预期；后续收益也不能全部归因于两家公司回购。预期差仍需当时的计划、市场定价与后续证据共同判断。

## 6. 对研究路径的实际改变

仓库已有的资本市场支持政策固定20日账户，主压力净夏普约0.690、年化约4.61%；过去回购强度叠加资金的账户也已失败。本轮只读核对，不重跑或改参数挽救。

接下来优先检验的对象是**有可核验资金来源、尚未完成且期限明确的购买安排，如何在接下来几周兑现，以及相对于当时成交规模和指数暴露是否有实际影响**。单纯贷款标题、计划上限和过去累计买入不再被当作同一项新增需求。先在有限可比阶段同时保留兑现与未兑现记录；只有出现指数层面的剩余交易优势，再进入完整账户。

净夏普1.2与年化10%的目标尚未实现。本轮没有当前市场方向判断，也没有将公司案例升级为交易指令。
"""
    (OUT / "融资条件到实际购买的传导结论.md").write_text(report, encoding="utf-8")
    print("贷款、原计划和实际购买已分开，原公告金额核对通过。")
    for row in snapshots:
        print(row["cutoff_date_end"], row["symbol"], "披露缺口亿元", round(row["disclosed_gap_to_minimum_cny"] / 1e8, 4), "距期限自然日", row["calendar_days_to_deadline"], "占过去20日成交额", round(row["disclosed_gap_over_prior20_turnover"], 6))
    for row in price_context:
        print("510300价格诊断", json.dumps(row, ensure_ascii=False))
    print("新增账户0，目标仍未实现。")


if __name__ == "__main__":
    main()
