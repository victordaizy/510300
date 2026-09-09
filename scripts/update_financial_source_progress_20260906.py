"""更新本次来源修复的真实进度，不增加策略轮次或回测数量。"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NOTE = ROOT / "docs/510300_FINANCIAL_SOURCE_CONTINUATION_20260906.md"


def main():
    data = json.loads(LATEST.read_text("utf-8"))
    assert len(data["completed_rounds"]) == 10
    assert data["registered_configurations_in_this_resumption"] == 161 and data["evaluation_accounts_in_this_resumption"] == 342
    receipt = json.loads((ROOT/"deliverables/510300夏普1.2持续研究_金融盈利口径重建_GPT审阅_20260906.delivery.json").read_text("utf-8"))
    assert receipt["new_strategy_evaluations"] == 0
    snapshot = ROOT / "reports/research/510300_pre_financial_source_progress_snapshot_20260906.json"
    with snapshot.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    data["updated_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    data["status"] = "TEN_ACCOUNT_ROUNDS_UNCHANGED_94_FINANCIAL_FACTS_REBUILT_TTM_ORIGINAL_EXTRACTION_CONTINUES"
    data["goal_achieved"] = False
    for study in data["running_studies"]:
        if study["study"] == "510300_ORIGINAL_EARNINGS_SOURCE_COMPLETION_V1":
            study.update({"status": "FIRST_BATCH_FINANCIAL_FACTS_REBUILT_FULL_HISTORY_COVERAGE_CONTINUES",
                          "new_adapter": "reports/research/510300_financial_original_facts_v1_1",
                          "documents_examined_including_replacements": 26, "documents_with_supported_layout": 24,
                          "new_adapter_fact_rows": 94, "explicit_cny_fact_rows": 90, "wrong_subsidiary_attachments_excluded": 2,
                          "legacy_adapter": "reports/research/510300_bank_financial_scope_adapter_v1",
                          "remaining_work": "首批24正确主体报告94数值已提取，90明确人民币；2份建行摘要缺营业利润，广发2022Q3四项币种未明确。补连续原始金融报告、普通股可享有利润、股数及历史成分权重。已补好下一批15份全年与上年同期原件，不再从仅2份8事实重新开始。"})
    data["running_studies"].append({"study": "510300_FINANCIAL_TTM_DEPENDENCIES_V1", "status": "FIFTEEN_ORIGINAL_DEPENDENCIES_ARCHIVED_ANNUAL_FACT_EXTRACTION_NEXT",
        "path": "reports/research/510300_financial_ttm_dependencies_v1", "anchor_companies": 12, "original_pdfs_archived": 15,
        "full_text_pages": 3863, "initial_front_subject_passed": 12, "additional_explicit_company_section_passed": 3,
        "annual_dependency_fact_rows_admitted": 0, "remaining_work": "提取新增年报及同期原始合并利润、其他权益工具分配与股数，区分准则改变和版本重述；不能机械加减披露EPS代替同股数滚动盈利。"})
    data["completed_source_rebuilds"].append({"study": "510300_FINANCIAL_ORIGINAL_FACTS_V1_1", "status": "FIRST_BATCH_NUMERICAL_REBUILD_COMPLETED_SCOPE_PARTIAL",
        "result": "reports/research/510300_financial_original_facts_v1_1/result.json", "fact_rows": 94, "explicit_cny_fact_rows": 90,
        "consolidated_accounting_identities": 46, "old_v1_fact_rows_unchanged": 82, "new_account_evaluations": 0})
    data["new_evidence"] += [
        "发现并排除2份中国平安名下的平安银行附件，原始目录中找回集团2023-10-28及2024-10-22报告；错误编号在第八轮有效事实表均为零行。",
        "金融版式修复得到24份正确主体报告94项累计事实，90项明确人民币；46项会计关系零差额，18项必要验证及41份PDF只读来源数值复核通过。",
        "平安银行2023/2024普通股每股收益若不扣优先股股利和永续债利息，以最新股数计算会分别高估5.40%和5.38%；最新股数不等同期间加权股数。",
        "中国平安2023季报同页调整前上年比较数会算出归母增长14.53%，调整后可比数为下降5.61%；三个值均属于2023披露版本，不能提前回填。",
        "新增15份全年和上年同期原始报告3863页已全部缓存，15份主体已通过封面或明确公司信息栏目确认，财务数值尚未提取。"]
    data["next_work"] = "继续从reports/research/510300_financial_ttm_dependencies_v1的15份完整原件提取全年与上年同期合并利润、普通股回报扣除、加权和最新股数。先解决相同主体准则与合并范围可比性，再建滚动盈利；不能机械加减不同分母EPS。首批24份94事实已完成，不重做；保留广发2022Q3币种缺口和建行摘要缺营业利润。扩展连续金融原始历史及历史指数权重，构建盈利估值、股东回报、公募需求的有限新候选与价格对照，固定规则后实际核算完整成本账户。全市场公募原始时钟与分类仍继续，单只ETF第十轮不重新调参。"
    data["latest_continuation_note"] = NOTE.relative_to(ROOT).as_posix()
    data["bank_eps_source_example"] = "reports/research/510300_financial_original_facts_v1_1/ordinary_share_eps_reconciliation.json"
    data["deliveries"].append({"rounds": [], "stage": "金融盈利原始口径修复与滚动盈利来源", "zip": Path(receipt["zip"]).relative_to(ROOT).as_posix(),
                              "md": "deliverables/510300金融盈利口径重建_20260906/盈利估值股东回报与公募需求_本轮口径修正.md",
                              "bytes": receipt["bytes"], "sha256": receipt["sha256"], "members": receipt["members"], "new_account_evaluations": 0})
    LATEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print("最新研究索引已更新：十轮161配置342账户不变，新增94项金融原始数值及15份后续原件。")


if __name__ == "__main__":
    main()
