"""复用原公告，重建价格约束、后续购买与剩余计划，不生成交易信号。"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_repurchase_constraint_release_v1"
CATALOGUE = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1/documents.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def subtract(left: str, right: str) -> float:
    return float(Decimal(left) - Decimal(right))


def main() -> None:
    scope = read(OUT / "scope.json")
    new_sources = read(OUT / "source_receipts.json")
    old_sources = {row["document_id"]: row for row in read(CATALOGUE)}
    assert {r["document_id"] for r in new_sources} == {r["document_id"] for r in scope["rows"]}
    ids = ["1221627004", "1221625606", "1221906544", "1221803153", "1221765778", "1221918049", "1221956157"]
    reused = [{k: old_sources[i][k] for k in ["symbol", "document_id", "title", "catalogue_date", "source_url", "raw_path", "text_path"]} for i in ids]
    source_index = {row["document_id"]: row for row in new_sources + reused}
    texts = {i: "".join("".join(read(ROOT / row["text_path"])).split()) for i, row in source_index.items()}
    # 只检查本轮解释依赖的关键原文事实，避免把叙述与数字串错文件。
    assert all(token in texts["1221724180"] for token in ["120", "165", "尚未实施", "股价持续超过"])
    assert "52,067,952.44" in texts["1221765778"] and "126.63" in texts["1221765778"]
    assert "65,199,537.44" in texts["1221956157"] and "2024年11月27日" in texts["1221956157"]
    assert "已达回购方案中规定的回购资金总额下限" in texts["1221803153"]

    amendments = [
        {"symbol": "300014.SZ", "name": "亿纬锂能", "source_id": "1221508399", "publication_date": "2024-10-25", "old_ceiling": 58.0, "new_ceiling": 57.5, "mechanism": "现金分红后的机械调整", "corporate_action_date": "2024-05-21", "cash_dividend_equivalent_per_share": 0.4987788, "voluntary_price_constraint_relaxation": False, "note": "公告解释此前除息对应的调整；不得把10月披露日回填成5月已得到本公告，也不据下调判断公司悲观。"},
        {"symbol": "300498.SZ", "name": "温氏股份", "source_id": "1221684497", "publication_date": "2024-11-11", "old_ceiling": 27.01, "new_ceiling": 26.86, "mechanism": "现金分红后的机械调整", "corporate_action_date": "2024-11-18", "cash_dividend_equivalent_per_share": 0.1491942, "voluntary_price_constraint_relaxation": False, "note": "11月18日除息时生效，其余计划不变。"},
        {"symbol": "300751.SZ", "name": "迈为股份", "source_id": "1221724180", "publication_date": "2024-11-14", "old_ceiling": 120.0, "new_ceiling": 165.0, "mechanism": "股价超过旧限价后主动放宽执行价格", "corporate_action_date": None, "effective_date": "2024-11-15", "voluntary_price_constraint_relaxation": True, "note": "截至本公告尚未实施；原金额及期限不变。公司仍可因条件变化调整或终止计划，未承诺次日一定购买。"},
    ]
    for row in amendments:
        row["source_url"] = source_index[row["source_id"]]["source_url"]
        row["text_path"] = source_index[row["source_id"]]["text_path"]

    followups = [
        {"symbol": "002714.SZ", "name": "牧原股份", "prior_source_id": "1221627004", "source_id": "1221906544", "prior_statistical_date": "2024-10-31", "statistical_date": "2024-11-30", "publication_date": "2024-12-03", "prior_cumulative_cny": 100001100.0, "cumulative_cny": 499970800.0, "disclosed_increment_cny": subtract("499970800", "100001100"), "minimum_plan_cny": 3000000000.0, "reported_minimum_gap_cny": subtract("3000000000", "499970800"), "exact_first_new_purchase_date": None, "status_at_disclosure": "继续执行", "precision_note": "原文以万元保留两位；仅确认11月区间增量，不知道具体首次新购买日、贷款提款日或统计日后购买。"},
        {"symbol": "300274.SZ", "name": "阳光电源", "prior_source_id": "1221625606", "source_id": "1221803153", "prior_statistical_date": "2024-10-31", "statistical_date": None, "statistical_cutoff_description": "截至2024-11-21公告披露日", "publication_date": "2024-11-21", "prior_cumulative_cny": 271218738.05, "cumulative_cny": 501208542.21, "disclosed_increment_cny": subtract("501208542.21", "271218738.05"), "minimum_plan_cny": 500000000.0, "reported_minimum_gap_cny": 0.0, "unused_original_maximum_cny": subtract("1000000000", "501208542.21"), "exact_first_new_purchase_date": None, "status_at_disclosure": "实施完成", "precision_note": "没有披露11月首次新购买日或准确最后成交日；达到金额下限即宣布完成，未用上限不再是本计划未来购买安排。"},
        {"symbol": "300751.SZ", "name": "迈为股份", "prior_source_id": "1221765778", "source_id": "1221956157", "prior_statistical_date": "2024-11-15", "statistical_date": "2024-11-27", "publication_date": "2024-12-06", "prior_cumulative_cny": 52067952.44, "cumulative_cny": 65199537.44, "disclosed_increment_cny": subtract("65199537.44", "52067952.44"), "minimum_plan_cny": 50000000.0, "reported_minimum_gap_cny": 0.0, "unused_original_maximum_cny": subtract("100000000", "65199537.44"), "exact_first_new_purchase_date": "2024-11-15", "status_at_disclosure": "实施完成", "precision_note": "1313.1585万元是11月15日首笔之后至11月27日的增加，不能全部当作11月18日公告以后增加；12月6日完成披露不前置至11月27日。"},
    ]
    for row in followups:
        row["source_url"] = source_index[row["source_id"]]["source_url"]
        row["causal_effect_of_loan_identified"] = False
        row["reported_gap_is_guaranteed_future_demand"] = False

    state_updates = [
        {"known_date": "2024-11-14", "source_id": "1221724180", "fact_date": "2024-11-14", "reported_cumulative_cny": 0.0, "reported_minimum_gap_cny": 50000000.0, "unspent_maximum_cny": 100000000.0, "meaning": "原计划尚未执行，次日解除价格约束；支持可能开始执行，不能预测次日金额或股价。"},
        {"known_date": "2024-11-18", "source_id": "1221765778", "fact_date": "2024-11-15", "reported_cumulative_cny": 52067952.44, "reported_minimum_gap_cny": 0.0, "unspent_maximum_cny": subtract("100000000", "52067952.44"), "meaning": "首次购买已超过最低计划金额；尚未用完的最高额度是选择空间，并非必然新需求。"},
        {"known_date": "2024-12-03", "source_id": "1221918049", "fact_date": "2024-11-30", "reported_cumulative_cny": 65199537.44, "reported_minimum_gap_cny": 0.0, "unspent_maximum_cny": subtract("100000000", "65199537.44"), "meaning": "披露月末累计；本公告仍表示根据市场情况继续实施，不能提前知道12月6日的完成公告。"},
        {"known_date": "2024-12-06", "source_id": "1221956157", "fact_date": "2024-11-27", "reported_cumulative_cny": 65199537.44, "reported_minimum_gap_cny": 0.0, "unspent_maximum_cny": subtract("100000000", "65199537.44"), "meaning": "确认实际购买区间11月15日至27日，方案已完成，未用最高额度退出未来购买估计。"},
    ]

    stock = pd.read_parquet(
        ROOT / "data/raw/constituents/000300_constituent_daily.parquet",
        columns=["date", "con_code", "raw_open", "raw_high", "raw_low", "raw_close", "amount"],
        filters=[("con_code", "==", "300751.SZ"), ("date", ">=", pd.Timestamp("2024-10-01")), ("date", "<=", pd.Timestamp("2024-11-15"))],
    ).sort_values("date")
    before = stock[stock.date.le("2024-11-14")]
    october = stock[stock.date.between("2024-10-21", "2024-10-31")]
    last_three = before.tail(3)
    first_day = stock[stock.date.eq("2024-11-15")].iloc[0]
    close_oct18 = float(stock.loc[stock.date.eq("2024-10-18"), "raw_close"].iloc[0])
    close_nov14 = float(stock.loc[stock.date.eq("2024-11-14"), "raw_close"].iloc[0])
    assert len(october) == 9 and october.raw_low.le(120).all()
    assert len(last_three) == 3 and last_three.raw_low.gt(120).all()
    scale = {
        "october_sessions_with_low_le_old_cap": int(october.raw_low.le(120).sum()),
        "last_three_sessions_before_amendment": [d.strftime("%Y-%m-%d") for d in last_three.date],
        "last_three_lows": last_three.raw_low.tolist(),
        "close_20241018": close_oct18,
        "close_20241114": close_nov14,
        "raw_close_change_before_amendment": close_nov14 / close_oct18 - 1,
        "first_purchase_minimum_price": 126.63,
        "first_purchase_maximum_price": 135.45,
        "all_first_purchase_prices_exceeded_old_cap": 126.63 > 120,
        "first_purchase_cny": 52067952.44,
        "first_purchase_share_of_day_turnover": 52067952.44 / float(first_day.amount),
        "preceding_20_session_average_turnover_cny": float(before.tail(20).amount.mean()),
        "remaining_minimum_share_of_preceding_average_day_turnover": 50000000 / float(before.tail(20).amount.mean()),
        "saved_monthly_weight_proxy_percent": 0.081,
        "weight_snapshot_date": "2024-10-31",
        "weight_first_publication_verified": False,
        "hypothetical_10pct_stock_move_index_contribution_bp": (0.081 / 100) * 0.10 * 10000,
        "scale_interpretation": "成交占比不是净新增资金或价格冲击估计；月度权重只作量级说明，不作为已经验证披露时钟的交易特征。10%股价情景是算术示例而非预测。",
        "source_price_path": "data/raw/constituents/000300_constituent_daily.parquet",
    }
    pd.DataFrame(amendments).to_csv(OUT / "三份调价公告的原因.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(followups).to_csv(OUT / "三家公司后续购买.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(state_updates).to_csv(OUT / "迈为按披露日更新的剩余计划.csv", index=False, encoding="utf-8-sig")
    stock.to_csv(OUT / "迈为价格约束与成交额.csv", index=False, encoding="utf-8-sig")
    save("reviewed_facts.json", {"amendments": amendments, "followups": followups, "state_updates": state_updates, "scale": scale, "reused_sources": reused})
    result = {
        "study_id": scope["study_id"], "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "MECHANICAL_ADJUSTMENT_AND_PRICE_FEEDBACK_DISTINGUISHED",
        "continuation_classification": "PROGRESS_CONSTRAINT_CAUSE_AND_REMAINING_DEMAND",
        "historical_reconstruction": True, "independent_validation": False,
        "new_primary_pdf_documents": len(new_sources), "new_primary_pdf_pages": sum(r["pages"] for r in new_sources),
        "reviewed_amendments": len(amendments), "mechanical_adjustments": 2, "voluntary_constraint_relaxations": 1,
        "followed_companies": len(followups), "new_accounts": 0, "new_strategy_return_tests": 0,
        "net_sharpe": None, "goal_achieved": False, "current_market_forecast_made": False,
        "known_limits": ["本阶段主动解除价格约束仅1个案例", "没有识别贷款的因果购买贡献", "没有当时同口径市场预期，未量化超预期部分", "个股购买不能直接升级为510300交易规则"],
        "branch_disposition": "这组回购案例诊断到此收束，不继续追索不可观察的内部购买动机，也不展开参数搜索或完整账户。",
        "transferable_rule": "先判断变化来自机械口径、上游冲击还是市场反馈，再估计未来仍受约束的行为与尚未兑现的规模。",
        "scale": scale,
        "next_question": "把同一方法应用于指数层面的财政现金流：债务置换先缓解何种约束，何时才转化为企业回款、订单和盈利？",
    }
    save("result.json", result)
    print(json.dumps({"状态": result["status"], "原公告": len(new_sources), "主动调价案例": 1, "首次回购占当日成交额": scale["first_purchase_share_of_day_turnover"], "目标实现": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
