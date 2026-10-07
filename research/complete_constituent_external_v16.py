"""在结果与图像检查完成后记录第十六轮进展，不宣告总研究目标达成。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import pandas as pd
from constituent_external_publish_v16 import OUT, load, report


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮完成回执已存在，不重复覆盖。")
    summary = load("两个病例_完整篮子与510300同口径比较.csv").set_index("stat_month")
    company = load("事前前五名_经营背景与后续收益.csv")
    checks = json.loads((OUT / "results/保存结果复算.json").read_text(encoding="utf-8"))
    assert checks["saved_stock_daily_rows_recomputed"] == 12000
    assert checks["maximum_account_and_risk_error"] < 1e-8
    assert (company.loc[company.stat_month == "2022-08", "net_profit_yoy_percent"] > 0).all()
    assert (company.loc[company.stat_month == "2022-08", "stock_return20_cc"] < 0).all()
    macro = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    for year, expected in [(2020, 2967), (2022, 2138)]:
        field = "loan_corporate_long_reported_month_or_difference_yi"
        assert macro.loc[f"{year}-08", field] - macro.loc[f"{year-1}-08", field] == expected
    # 文本勘误只重建报告，不改变已冻结的数值、资料或图表。
    report(summary, load("全部行业_收益与下行平方贡献.csv"), company, checks)
    viewed = [
        ("600036_2022H1.pdf", 36), ("000858_2022H1.pdf", 7),
        ("300750_2022H1.pdf", 13), ("600519_2022H1_巨潮.pdf", 7),
        ("601318_2022H1.pdf", 26), ("600109_20201013_复牌_巨潮.pdf", 1),
    ]
    write_json("results/图像核对回执.json", {"source_pages_viewed": [{"document": name, "pdf_page": page} for name, page in viewed],
               "source_page_renders_saved": 33, "chart_viewed": "figures/全部行业贡献与下行结构.png",
               "notes": "已检查主要原因段落、单位、表格列及脚注；行业全图中文、正负号、轴范围、数值和页脚清晰。不是外部人工审阅。"})
    at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    completion = {"at": at, "study_id": "510300_CONSTITUENT_EXTERNAL_BRIDGE_V16",
                  "status": "PASS_SAVED_CONSTITUENT_CASH_RISK_AND_SOURCE_FACT_RECOMPUTATION",
                  "continuation_classification": "PROGRESS", "goal_achieved": False, "independent_validation": False,
                  "checks": checks, "company_source_documents": 10, "suspension_source_documents": 1,
                  "first_vintage_authenticated": False, "frozen_parent_inputs_modified": False,
                  "report_sha256": digest(OUT / "第十六轮_成分股盈利与外部压力的实际传导.md"),
                  "chart_sha256": digest(OUT / "figures/全部行业贡献与下行结构.png"),
                  "company_table_sha256": digest(OUT / "results/事前前五名_经营背景与后续收益.csv"),
                  "source_page_locator_sha256": digest(OUT / "results/公司传导_14项原文定位与解释.csv"),
                  "scope_limit": "两病例的描述与公司传导证据；未识别盈利预期和折现冲击的因果份额，未证明预测优势，非ETF真实持仓归因或策略收益。"}
    write_json("completion_receipt.json", completion)
    write_json("goal_progress.json", {"at": at, "classification": "PROGRESS", "overall_goal_status": "active", "goal_achieved": False,
                "completed": ["两例各300只成分、12000个成分日、参考权重持股现金及风险贡献", "固定权重前五名十份中报、30项成对财务数值、37项专项数值、14项原因说明", "2022利润增长与息差回款成本新业务压力并存，以及广泛下跌的内部核对"],
                "next_bounded_question": "在扩大到剩余固定历史月份前，固定需求传导与资金迁移的相反可核验预期，并检查同向及反向病例；不把已选两例当独立验证。",
                "unresolved": ["市场当时未来盈利预期与其变化", "外部折现、风险溢价、公司和国内后续消息的各自作用", "更多独立病例下的分布增量"],
                "new_models": 0, "new_accounts": 0, "orders_authorized": False})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("constituent_external_publish_v16.py"), OUT / "code/constituent_external_publish_v16.py")
    print("第十六轮已完成：原字段与财务事实复算通过，主要原页及最终图已检查。")
    print("总目标保持进行中，尚未取得独立验证的确定性预测结论。")


if __name__ == "__main__":
    main()
