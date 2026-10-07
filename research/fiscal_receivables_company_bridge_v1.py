"""将公司事前解释与现金流分项连接，保留尚未识别的付款和预期差。"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_fiscal_receivables_company_bridge_v1"
FACTS = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_2/repaired_verified_facts.parquet"
ANNUAL_PDF = "data/raw/cninfo/factor96_financial_parser_repair_v1/pdf/7148a2fddacf8b511e5b58fc1b67191d2ca07c5891c834faa73d251bcf79c185.pdf"
ANNUAL_URL = "https://static.cninfo.com.cn/finalpage/2025-03-28/1223403703.pdf"
Q3_URL = "https://static.cninfo.com.cn/finalpage/2024-10-31/1221570621.PDF"
Q3_CALL = "https://www.ccccltd.cn/tzzgx/tzzfw/tzzfw_wsly/202412/P020241211365171946496.pdf"
FY_CALL = "https://www.ccccltd.cn/tzzgx/tzzfw/tzzfw_wsly/202504/P020250408335347690441.pdf"


def save(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    scope = json.loads((OUT / "scope.json").read_text(encoding="utf-8"))
    symbols = scope["cohort"]
    names = {"601390.SH": "中国中铁", "601668.SH": "中国建筑", "601800.SH": "中国交建"}
    f = pd.read_parquet(FACTS, filters=[("ts_code", "in", symbols)])
    f["report_period"] = pd.to_datetime(f.report_period)
    f = f[f.metric_id.eq("OPERATING_CASH_FLOW_YTD") & f.report_period.isin(pd.to_datetime(["2024-09-30", "2024-12-31"]))]
    assert len(f) == 6 and f.verified_value.notna().all()
    assert f.fact_status.eq("VERIFIED_SAVED_ORIGINAL_FACT").all()
    members = pd.read_parquet(
        ROOT / "reports/research/510300_shareholder_disclosure_breadth_v1/inputs/membership.parquet",
        filters=[("membership_date", "==", pd.Timestamp("2024-11-08"))],
    )
    cohort = []
    for symbol in symbols:
        block = f[f.ts_code.eq(symbol)].sort_values("report_period")
        q3, annual = block.iloc[0], block.iloc[1]
        cohort.append({
            "symbol": symbol, "name": names[symbol],
            "member_on_20241108": bool(members.symbol.eq(symbol).any()),
            "reported_9m_operating_cashflow_cny": int(q3.verified_value),
            "reported_annual_operating_cashflow_cny": int(annual.verified_value),
            "difference_of_reported_cumulative_cashflow_cny": int(annual.verified_value - q3.verified_value),
            "q3_publication_date_proxy": str(pd.Timestamp(q3.event_publication_date).date()),
            "annual_publication_date_proxy": str(pd.Timestamp(annual.event_publication_date).date()),
            "q3_source_url": q3.official_pdf_url, "annual_source_url": annual.official_pdf_url,
            "interpretation": "仅为两次披露累计数的差额背景，不将其全部归因化债；未证明两次报告合并范围完全不变，未把年报未来数据前置。",
        })
    assert all(row["member_on_20241108"] for row in cohort)

    # 年度数值来自原PDF第172页合并现金流量表；2023使用该页经重述比较栏。
    cash = {
        "sales_cash": [746909097114, 717141389290],
        "tax_refunds": [2683062202, 2791569528],
        "other_operating_receipts": [39880068951, 24522628873],
        "operating_inflows": [789472228267, 744455587691],
        "supplier_payments": [657096564763, 611512779680],
        "employee_payments": [53657808184, 54096608934],
        "tax_payments": [29442132644, 24836590389],
        "other_operating_payments": [36769305898, 41948525506],
        "operating_outflows": [776965811489, 732394504509],
        "operating_net": [12506416778, 12061083182],
        "investing_inflows": [39175030030, 35403573705],
        "investing_outflows": [68793534077, 91272543830],
        "investing_net": [-29618504047, -55868970125],
        "fixed_and_intangible_asset_cash_payments": [27268101484, 38949725935],
    }
    for year_idx in [0, 1]:
        assert sum(cash[k][year_idx] for k in ["sales_cash", "tax_refunds", "other_operating_receipts"]) == cash["operating_inflows"][year_idx]
        assert sum(cash[k][year_idx] for k in ["supplier_payments", "employee_payments", "tax_payments", "other_operating_payments"]) == cash["operating_outflows"][year_idx]
        assert cash["operating_inflows"][year_idx] - cash["operating_outflows"][year_idx] == cash["operating_net"][year_idx]
        assert cash["investing_inflows"][year_idx] - cash["investing_outflows"][year_idx] == cash["investing_net"][year_idx]

    labels = {
        "sales_cash": "销售商品和提供劳务收款", "tax_refunds": "税费返还", "other_operating_receipts": "其他经营收款",
        "operating_inflows": "经营现金流入", "supplier_payments": "购买商品和接受劳务付款",
        "employee_payments": "职工付款", "tax_payments": "税费付款", "other_operating_payments": "其他经营付款",
        "operating_outflows": "经营现金流出", "operating_net": "经营现金净额",
        "investing_inflows": "投资现金流入", "investing_outflows": "投资现金流出", "investing_net": "投资现金净额",
        "fixed_and_intangible_asset_cash_payments": "购建固定无形和其他长期资产付款",
    }
    cash_rows = [{"metric": key, "项目": labels[key], "2024元": pair[0], "2023经重述元": pair[1], "变化元": pair[0] - pair[1], "2024亿元": pair[0] / 1e8, "2023经重述亿元": pair[1] / 1e8, "变化亿元": (pair[0] - pair[1]) / 1e8} for key, pair in cash.items()]

    # 同一年度附注第328页，下面三行是现金流量调节项，不是直接收回欠款金额。
    working_capital = [
        {"项目": "合同资产增加对应的现金调节", "2024元": -41507065110, "2023元": -13176782183},
        {"项目": "经营性应收及PPP合同资产增加对应的现金调节", "2024元": -98496452053, "2023元": -110341948504},
        {"项目": "经营性应付增加对应的现金调节", "2024元": 91027139888, "2023元": 73820007023},
    ]
    for row in working_capital:
        row["同比现金影响元"] = row["2024元"] - row["2023元"]
        row["同比现金影响亿元"] = row["同比现金影响元"] / 1e8
    profit = {"revenue_2024_cny": 771944258710, "revenue_2023_restated_cny": 758718749687, "parent_profit_2024_cny": 23384093178, "parent_profit_2023_restated_cny": 23816263046, "source_visual_page": 168}
    profit["revenue_growth"] = profit["revenue_2024_cny"] / profit["revenue_2023_restated_cny"] - 1
    profit["parent_profit_growth"] = profit["parent_profit_2024_cny"] / profit["parent_profit_2023_restated_cny"] - 1

    statements = [
        {"event_date": "2024-10-31", "source_url": Q3_CALL, "pages": [3, 4, 5, 6, 7], "record_kind": "本次对话已成功读取发行人PDF的网页提取文本；原PDF直连失败，后续保存文本请求也失败", "facts": ["公司因部分客户付款不理想而放缓相关订单执行", "公司争取全年经营现金流回正，同时承认难度", "公司讨论了四季度集中回款和付款节奏", "公司表示继续控制新增投资项目，并未承诺收到钱就大幅扩张投资"], "guaranteed_payment_amount_cny": None, "guaranteed_payment_date": None, "public_web_first_publication_verified": False, "clock_limit": "会议发生日不能自动当作所有投资者可获取全文的日期；12月PDF路径不单独证明12月11日公开。"},
        {"event_date": "2025-03-31", "source_url": FY_CALL, "pages": [4, 5], "record_kind": "本次对话已成功读取发行人PDF的网页提取文本；未取得原PDF字节归档", "facts": ["管理层将2024年现金流改善的重要作用归于化债相关清欠", "公司继续强调以收定支、投资项目条件和收益约束"], "claim_level": "管理层事后解释，未量化化债带来的独立现金额或因果贡献", "public_web_first_publication_verified": False},
    ]
    money_delta = {key: pair[0] - pair[1] for key, pair in cash.items()}
    assert money_delta["operating_inflows"] - money_delta["operating_outflows"] == money_delta["operating_net"]
    saved_old_2023 = pd.read_parquet(FACTS, filters=[("ts_code", "==", "601800.SH"), ("metric_id", "==", "OPERATING_CASH_FLOW_YTD")])
    old_2023 = saved_old_2023[pd.to_datetime(saved_old_2023.report_period).eq("2023-12-31")].iloc[0]
    vintage_note = {"original_2023_report_cashflow_cny": int(old_2023.verified_value), "2024_report_2023_restated_cashflow_cny": cash["operating_net"][1], "treatment": "本轮年度同比采用同一2024年报经重述比较数；旧原版事实不改写，不将版本差异解释为新的经济变化。"}

    pd.DataFrame(cohort).to_csv(OUT / "三家公司累计现金流背景.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cash_rows).to_csv(OUT / "中国交建现金流入流出分解.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(working_capital).to_csv(OUT / "中国交建营运资金调节项.csv", index=False, encoding="utf-8-sig")
    save("reviewed_facts.json", {"cohort": cohort, "cashflow": cash_rows, "working_capital": working_capital, "profit": profit, "management_statements": statements, "vintage_note": vintage_note, "annual_pdf_path": ANNUAL_PDF, "annual_source_url": ANNUAL_URL, "q3_source_url": Q3_URL, "reviewed_annual_pages": [168, 172, 328], "primary_cashflow_page_visual_reviewed": True})
    result = {
        "study_id": scope["study_id"], "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "CASH_RECOVERY_EXECUTION_CONSTRAINT_AND_SPENDING_DISTINGUISHED",
        "continuation_classification": "PROGRESS_COMPANY_CAUSE_AND_CASHFLOW_DECOMPOSITION",
        "reviewed_companies": 3, "detailed_cashflow_decompositions": 1,
        "historical_reconstruction": True, "independent_validation": False,
        "new_accounts": 0, "new_strategy_return_tests": 0, "net_sharpe": None,
        "new_primary_pdf_saved": 1, "reused_primary_pdf": 1,
        "issuer_call_documents_read_via_web": 2, "issuer_call_raw_pdfs_saved": 0,
        "operating_inflow_change_cny": money_delta["operating_inflows"],
        "operating_outflow_change_cny": money_delta["operating_outflows"],
        "operating_net_change_cny": money_delta["operating_net"],
        "investing_net_change_cny": money_delta["investing_net"],
        "parent_profit_growth": profit["parent_profit_growth"],
        "confirmed_future_payment_schedules": 0,
        "future_payment_schedule_claim_scope": "本轮已读资料中未见能连接付款主体、金额和日期的具体剩余付款安排，不代表全市场不存在。",
        "forecastable_mechanism": "客户付款约束会影响订单执行、收入转化与持续成本；现金改善还应分辨新增收款、付款节奏、应付融资及收缩投资。",
        "unidentified": ["化债独立带来的收款额", "管理层沟通全文的历史首次公开时间", "同口径市场预期和未反映空间", "指数层面的净收益优势"],
        "branch_disposition": "本组公司回款案例收束；不把年度现金转正或管理层信心升级为510300买入规则，不继续扩展同一内部回款取证。",
        "next_research_question": "转到最新可用时点的整体驱动判断：资金供给、国内需求与外部约束中，哪条正在改变未来1—4周的指数净影响，并写下可被新事实推翻的条件预测。",
        "goal_achieved": False, "current_market_forecast_made": False,
    }
    save("result.json", result)
    print(json.dumps({"经营流入变化亿元": money_delta["operating_inflows"] / 1e8, "经营流出变化亿元": money_delta["operating_outflows"] / 1e8, "经营净额变化亿元": money_delta["operating_net"] / 1e8, "投资净额改善亿元": money_delta["investing_net"] / 1e8, "归母利润同比": profit["parent_profit_growth"], "新账户": 0, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
