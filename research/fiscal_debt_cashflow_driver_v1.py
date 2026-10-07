"""保存一组有限财政原文，将政策额度、原因及后续兑现分开记录。"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_fiscal_debt_cashflow_driver_v1"
SOURCES = [
    {"id": "mof_20241012", "url": "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202410/t20241012_3945410.htm", "event_time": "2024-10-12T10:00:00+08:00", "publication_date": "2024-10-12", "role": "政策方向在前，具体金额未公布", "required": ["1.2万亿元", "较大规模债务限额", "具体资金数量"]},
    {"id": "mof_20241108", "url": "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202411/t20241109_3947230.htm", "event_time": "2024-11-08T16:00:00+08:00", "publication_date": "2024-11-09", "role": "批准额度、政策背景和用途", "required": ["每年2万亿元", "8000亿元", "税收收入不及预期", "土地出让收入大幅下降", "6000亿元"]},
    {"id": "mof_20250110", "url": "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202501/t20250110_3951525.htm", "event_time": "2025-01-10T15:00:00+08:00", "publication_date": "2025-01-10", "role": "后续发行进度；不前置成11月已知", "required": ["12月18日", "全部发行完毕", "2025年的2万亿元"]},
]


def save(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main() -> None:
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    scope_path = OUT / "scope.json"
    if not scope_path.exists():
        save("scope.json", {"at": now(), "study_id": "510300_FISCAL_DEBT_CASHFLOW_DRIVER_V1", "question": "债务置换为何增加政府融资，先改善什么约束，何时才可能影响企业回款和新需求？", "selection": "沿同一政策链读取方向预告、额度批准、首次选定后续实施说明；不按股市涨跌选择样本。", "historical_reconstruction": True, "sources": SOURCES, "current_market_forecast_made": False, "new_strategy_tests": 0})
    receipts = []
    for src in SOURCES:
        raw = OUT / "sources" / (src["id"] + ".html")
        receipt_path = OUT / "sources" / (src["id"] + "_receipt.json")
        if not raw.exists():
            response = requests.get(src["url"], timeout=(10, 25))
            response.raise_for_status()
            raw.write_bytes(response.content)
            receipt = {**src, "retrieved_at": now(), "http_status": response.status_code, "raw_path": raw.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(response.content).hexdigest()}
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        soup = BeautifulSoup(raw.read_bytes(), "html.parser")
        for node in soup(["script", "style"]):
            node.decompose()
        text = soup.get_text("\n", strip=True)
        compact = "".join(text.split())
        if not all(token in compact for token in src["required"]):
            raise ValueError("原文关键内容不完整：" + src["id"])
        text_path = raw.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        receipt["text_path"] = text_path.relative_to(ROOT).as_posix()
        receipts.append(receipt)
    save("source_receipts.json", receipts)

    facts = {
        "cause_statement": {"source": "mof_20241108", "reported_drivers": ["内需不足等因素带来经济压力", "税收收入不及预期", "土地出让收入下降", "地方原有债务利息及偿还压力挤占可用财力"], "claim_level": "财政部当时公开给出的政策背景，不是本研究对各原因的因果份额估计"},
        "policy_ledger_trillion_cny": [
            {"item": "新批准的债务置换限额", "total": 6.0, "years": [2024, 2025, 2026], "per_year": 2.0, "use": "置换存量隐性债务", "is_same_amount_of_new_final_demand": False},
            {"item": "从每年新增专项债中安排的化债资源", "total": 4.0, "years": [2024, 2025, 2026, 2027, 2028], "per_year": 0.8, "use": "补充政府性基金财力用于化债", "is_extra_on_top_of_all_special_bond_quotas": False},
            {"item": "2029年及以后到期棚改隐性债务按原合同偿还", "total": 2.0, "years": None, "per_year": None, "use": "移出此前截至2028年的集中化解压力", "is_new_cash_issuance": False},
        ],
        "non_additive_totals": {"2024_to_2026_swap_resources": 6.0 + 0.8 * 3, "2024_to_2028_swap_resources": 6.0 + 4.0, "pre_2029_pressure_after_arrangements": round(14.3 - 6.0 - 4.0 - 2.0, 1), "meaning": "8.4、10、12万亿元为不同覆盖和含义，不能再次相加；2万亿元按原合同偿还安排不是新发现金，也不是延长原合同到期日。"},
        "interest_saving": {"estimated_total_trillion_cny": 0.6, "years": 5, "kind": "当时官方测算，非已实现现金，也非可全部立即用于股市的资金"},
        "expectation_anchor": {"known_before_amount_announcement": "10月12日已公开较大规模化债的方向，具体数字待程序完成", "market_consensus_amount": None, "priced_in_fraction": None, "surprise_identified": False, "meaning": "已知政策方向可以作为预期锚，但不能代替市场一致预期；不能把全部政策额当作新信息。"},
        "followup": {"source": "mof_20250110", "fact_date": "2024-12-18", "known_from_this_source_date": "2025-01-10", "issued_2024_quota_trillion_cny": 2.0, "confirms": "当年置换额度已全部发行", "does_not_confirm": ["所有资金已进入上市公司现金账户", "全部形成新订单", "未来股市必然上涨"], "first_publication_on_fact_date_verified": False},
        "conditional_balance_sheet_examples": [
            {"kind": "旧银行贷款置换", "amount_units": 100, "assumptions": "简化为等额新政府债偿还已纳入社融的旧贷款，忽略费用及其他同时变化", "government_debt_change": 100, "old_loan_change": -100, "new_final_demand_mechanically_created": 0, "testable_next_observation": "对应旧贷款退出、期限和成本变化；仍要区分自然融资需求下降"},
            {"kind": "支付此前已确认收入的工程欠款", "amount_units": 100, "assumptions": "简化为该项旧应收按面值实际回收，忽略税费与其他同时变化", "company_cash_change": 100, "company_receivable_change": -100, "new_revenue_mechanically_created": 0, "testable_next_observation": "相关企业真实回款、债务和资金用途；经营现金流改善不必对应同期新订单"},
        ],
        "index_transmission_hypotheses": [
            {"channel": "银行", "positive": "相关借款人偿付风险缓解，部分资产质量或资本占用可能改善", "offset": "原高收益资产退出、再配置及负债成本影响净息差", "not_yet_identified": "各银行暴露、信用损失变化及净盈利影响"},
            {"channel": "工程建设与设备供应商", "positive": "若旧欠款实际清偿，可改善现金和偿债能力", "offset": "收回旧应收不自动新增利润；现金可能继续还债", "not_yet_identified": "收款主体、到账金额、时点与后续新订单"},
            {"channel": "指数整体", "positive": "尾部风险及资金链压力缓和，可能支持风险重新定价", "offset": "原收入压力仍可能存在，政策方向可能已被预期", "not_yet_identified": "行业净影响、市场原预期及剩余价格空间"},
        ],
    }
    assert abs(facts["non_additive_totals"]["2024_to_2026_swap_resources"] - 8.4) < 1e-9
    assert abs(facts["non_additive_totals"]["pre_2029_pressure_after_arrangements"] - 2.3) < 1e-9
    save("driver_facts.json", facts)
    result = {
        "study_id": "510300_FISCAL_DEBT_CASHFLOW_DRIVER_V1", "at": now(),
        "status": "POLICY_CAUSE_USE_EXPECTATION_AND_TRANSMISSION_DISTINGUISHED",
        "continuation_classification": "PROGRESS_UPSTREAM_CONSTRAINTS_AND_MACRO_CASHFLOW",
        "micro_diagnostic": "reports/research/510300_repurchase_constraint_release_v1/result.json",
        "new_primary_web_documents": len(receipts), "new_accounts": 0, "new_strategy_return_tests": 0,
        "net_sharpe": None, "goal_achieved": False, "independent_validation": False,
        "historical_reconstruction": True, "current_market_forecast_made": False,
        "preserved_failed_study": "reports/research/510300_tsf_composition_funding_daily_v1/result.json",
        "conclusion": "政府债融资增加的原因与用途决定传导方向；债务置换先缓和存量债务和现金约束，不能把全额度等同新增需求或利润。股价可提前反映，经济兑现慢不代表交易机会必然滞后。",
        "next_research_question": "对沪深300中有明确政府应收暴露的公司，用原公告区分可确认的旧欠款回收、新订单及原有市场预期，找尚未被反映的现金流变化；无具体到账或时间证据不扩写成需求复苏。",
        "remaining_unknowns": ["当时同口径市场一致预期", "具体上市公司收到的化债相关现金及用途", "对指数的盈利净效应", "完整账户成本后优势"],
    }
    save("result.json", result)
    print(json.dumps({"状态": result["status"], "官方资料": len(receipts), "新账户": 0, "目标实现": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
