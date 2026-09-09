"""保存七份原PDF的目视读数与来源解释，不改写原预测或策略账户。"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

OUT = ROOT / "reports/research/510300_forward_eps_report_pe_identity_v1"
REVIEW = OUT / "original_page_review"
CASES = [
    {"report_id": "AP202310311607043071", "years": [2023, 2024, 2025], "eps": ["1.5", "1.6", "1.7"],
     "pe": ["15", "13", "13"], "quote": "11.07", "pages": ["AP202310311607043071_第一页.png"],
     "category": "原PDF侧栏证券身份不一致",
     "finding": "标题和正文为中信证券600030，右栏走势和相关研究为国联证券601456；右栏收盘价11.07不能据此确认为中信证券参考价。表内三年EPS及PE逐项与提取结果一致。",
     "handling": "保留原值和身份冲突证据，不用EPS乘PE反填报价。该报价未获对应证券确认。"},
    {"report_id": "AP202603301820851211", "years": [2026, 2027, 2028], "eps": ["0.73", "0.75", "0.78"],
     "pe": ["22.1", "20.0", "21.0"], "quote": "9.11", "pages": ["AP202603301820851211_第一页.png"],
     "category": "原PDF跨年度估值关系不一致",
     "finding": "东方证券报告原表的EPS乘PE分别为16.133、15.000、16.380，三年舍入区间没有共同价格，首页收盘价9.11。目视确认原表就是这些数字。",
     "handling": "无法确定哪项原值正确，不宣称其倒数为该参考价下真实盈利收益率。"},
    {"report_id": "AP202205191566530793", "years": [2022, 2023, 2024], "eps": ["1.43", "1.52", "1.59"],
     "pe": ["9.2", "8.6", "7.5"], "quote": "11.97", "pages": ["AP202205191566530793_第一页.png"],
     "category": "原PDF正文与表格PE冲突",
     "finding": "西部矿业正文在相同EPS下给出8.4、7.9、7.5倍，表格为9.2、8.6、7.5倍。提取器忠实保留原表。原件日期2022-05-01，平台信息日期较晚。",
     "handling": "同时记录正文冲突，不静默选择看起来合理的一组覆盖既有表格来源。"},
    {"report_id": "AP202207191576328247", "years": [2022, 2023, 2024], "eps": ["1.12", "1.41", "1.59"],
     "pe": ["24.0", "19.1", "16.5"], "quote": "26.22", "pages": ["AP202207191576328247_page1.png"],
     "category": "原PDF跨年度估值关系不一致",
     "finding": "中天科技表内EPS、PE及首页收盘价与提取结果相同；三年EPS乘PE不能在显示精度下共同对应一个价格。",
     "handling": "保留来源一致性缺口，不能凭目标年度单项较接近而称整表口径已统一。"},
    {"report_id": "AP202303131584213376", "years": [2022, 2023, 2024], "eps": ["1.0", "1.3", "1.6"],
     "pe": ["12.4", "8.0", "6.0"], "quote": "12.63", "pages": ["AP202303131584213376_page1.png"],
     "category": "原PDF跨年度估值关系不一致",
     "finding": "华泰证券表内三年EPS乘PE为12.4、10.4、9.6，无法同时还原首页12.63；目标合理价14.50至15.72另有明确标注，没有把目标价误读为收盘价。",
     "handling": "目标合理价、参考收盘价和预测表PE分开保留。"},
    {"report_id": "AP201708060777972154", "years": [2017, 2018, 2019], "eps": ["1.31", "1.64", "1.95"],
     "pe": ["11.29", "9.02", "7.56"], "quote": "20.99", "pages": ["AP201708060777972154_第一页.png"],
     "category": "表内价格一致但不同于首页报价，原因未确认",
     "finding": "隆基股份预测表和20.99收盘价均目视确认；三年EPS乘PE在约14.74至14.79存在共同区间。原件日期2017-08-01。没有证据确定是哪个价格时点或股份口径造成差异。",
     "handling": "将口径原因记为未知，不把约14.77改写为实际历史收盘价。"},
    {"report_id": "AP201704240527509760", "years": [2017, 2018, 2019], "eps": ["0.53", "0.66", "0.78"],
     "pe": ["18.98", "15.22", "12.92"], "quote": "11.94",
     "pages": ["AP201704240527509760_第一页.png", "AP201704240527509760_page3.png"],
     "category": "表内价格一致但不同于首页报价，原因未确认",
     "finding": "阳光电源首页收盘价11.94，第三页财务表EPS与PE乘积在约10.01至10.12存在共同区间。第三页原表还显示投入资本回报率的预测单元格为#VALUE!，说明原报告本身保留了表格错误；未据此猜测正确PE。",
     "handling": "记录两页原件；不从首页报价、净资产或其他年度反推替换原始预测。"},
]


def main():
    frame = pd.read_parquet(OUT / "annual_eps_pe_reference_identity.parquet")
    rows = []
    for case in CASES:
        selected = frame.loc[frame.report_id.eq(case["report_id"])].sort_values("target_fiscal_year")
        assert selected.target_fiscal_year.tolist() == case["years"]
        assert selected.eps_cell_as_reported.tolist() == case["eps"]
        assert selected.pe_exact.tolist() == case["pe"]
        assert selected.report_reference_price_exact.eq(case["quote"]).all()
        pdf = ROOT / selected.raw_pdf_path.iloc[0]
        assert identity(pdf)["sha256"] == selected.pdf_sha256.iloc[0]
        rows.append({**case, "ts_code": selected.ts_code.iloc[0], "sec_name": selected.sec_name.iloc[0],
                     "raw_pdf": identity(pdf), "viewed_page_images": [identity(REVIEW / p) for p in case["pages"]],
                     "manually_viewed": True, "saved_values_match_visible_original": True})
    save(OUT / "original_page_adjudication.json", {"reviewed_at": now(),
         "status": "SEVEN_ORIGINAL_REPORTS_VISUALLY_CONFIRMED_SOURCE_RELATION_DEFECTS",
         "selection": "已看关系诊断后，选择两机构最大差异样例及国信最早三份无共同价格样例；不是随机样本。",
         "reviewed_reports": 7, "reviewed_pages": 8, "annual_rows": 21, "cases": rows,
         "parser_error_found_in_these_visible_cells": False,
         "outside_relation_count_is_not_error_rate": True,
         "all_other_report_causes_adjudicated": False,
         "existing_facts_changed": False, "existing_accounts_changed": False,
         "round_19_original_reported_pe_baseline_preserved": True,
         "next_research": "固定同月份比较关系相容的报告PE、原报告PE以及不使用PE；另存来源及策略版本。",
         "new_models_fit": 0, "goal_achieved": False}, exclusive=True)
    print("七份原报告、八页图、二十一条年度EPS与PE逐项确认；已记录原件问题和未知原因。")


if __name__ == "__main__":
    main()
