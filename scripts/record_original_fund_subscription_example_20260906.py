"""核对510300原始季报的申购、赎回、期初与期末份额，不生成投资信号。"""
from pathlib import Path
from decimal import Decimal
import hashlib
import io
import json
import re
import sys

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, write_json

OUT = ROOT / "reports/research/510300_original_fund_subscription_reports_v1"
RAW = Path(r"E:\ResearchData\New project 8\data\raw\510300_original_fund_subscription_reports_v1")
SOURCES = {"sse": "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2021-10-27/510300_20211027_1_INbeqLrb.pdf",
           "cninfo": "https://static.cninfo.com.cn/finalpage/2021-10-27/1211391984.PDF"}


def main():
    require(not (OUT / "example_2021Q3_result.json").exists(), "该原始申赎实例已保存")
    OUT.mkdir(parents=True, exist_ok=True)
    sources, tables, facts = [], [], []
    for provider, url in SOURCES.items():
        source = RAW / ("510300_2021Q3_" + provider + ".pdf")
        content = source.read_bytes()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            cover = pdf.pages[0].extract_text()
            overview = pdf.pages[1].extract_text()
            compact = re.sub(r"\s+", "", cover)
            require("华泰柏瑞沪深300交易型开放式指数证券投资基金" in compact and "联接" not in compact, "基金身份不匹配")
            require(re.search(r"基金主代码\s+510300\b", overview), "基金主代码未证明为510300")
            require("报告送出日期:2021年10月27日" in compact.replace("：", ":"), "报告送出日期未匹配")
            candidates = [table for table in pdf.pages[11].extract_tables() if table and table[0][0] == "报告期期初基金份额总额"]
            require(len(candidates) == 1, "份额变动表缺失或歧义")
            table = candidates[0]
            require(len(table) == 5 and all(len(row) == 2 for row in table), "表格形状变化")
            require(table[3][1] == "-", "该实例的拆分格必须明确为横线，其他版式待单独处理")
            labels = ["报告期期初基金份额总额", "报告期期间基金总申购份额", "减：报告期期间基金总赎回份额", "报告期期末基金份额总额"]
            selected = [table[i] for i in (0, 1, 2, 4)]
            require([row[0] for row in selected] == labels, "份额行对应错误")
            values = [Decimal(row[1].replace(",", "")) for row in selected]
            require(values[0] + values[1] - values[2] == values[3], "期初加申购减赎回未对应期末")
            row = {"fund": "510300.SH", "period_start": "2021-07-01", "period_end": "2021-09-30",
                   "reported_send_date": "2021-10-27", "beginning_units": float(values[0]),
                   "gross_subscription_units": float(values[1]), "gross_redemption_units": float(values[2]),
                   "ending_units": float(values[3]), "net_subscription_units": float(values[1]-values[2]),
                   "net_units_divided_by_beginning_units": float((values[1]-values[2])/values[0]),
                   "split_raw_cell": "-", "split_cell_status": "REPORTED_DASH_NOT_MISSING_DOCUMENT",
                   "identity_residual_without_split": 0.0, "unit": "FUND_UNITS", "source_page": 12,
                   "exact_cash_flow_amount": None, "cash_flow_status": "NOT_COMPUTED_UNIT_FLOW_IS_NOT_CASH_FLOW"}
            tables.append(table)
            facts.append(row)
            sources.append({"provider": provider, "url": url, "path": source.relative_to(RAW.parents[2]).as_posix(),
                            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "pdf_metadata": pdf.metadata,
                            "reported_send_date_from_cover": "2021-10-27", "archive_url_date": "2021-10-27",
                            "raw_cover": cover, "raw_page_12": pdf.pages[11].extract_text(), "raw_table": table})
    require(sources[0]["sha256"] == sources[1]["sha256"] and tables[0] == tables[1] and facts[0] == facts[1], "两个官方档案内容未一致")
    dates = pd.read_parquet(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet").date
    next_session = str(dates.loc[dates > "2021-10-27"].min().date())
    result = {"recorded_at": now(), "status": "ONE_ORIGINAL_FUND_REPORT_SHARE_FLOW_EXAMPLE_VERIFIED_HISTORY_COLLECTION_CONTINUES",
              "reports": 1, "official_archives": 2, "sources": sources, "facts": facts[0],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "visually_checked_pages": [1,12], "exact_intraday_publication_time": None,
              "original_publication_catalog_record_verified": False,
              "provisional_next_session_after_reported_send_date": next_session,
              "provisional_clock_note": "依据报告送出日和两个官方存档路径日期，不把PDF创建时间称为公布时间；批量历史仍需原始公告目录与修订记录。",
              "source_admitted_for_portfolio": False, "new_portfolio_evaluation": False,
              "next_step": "恢复510300原始季度报告目录与各期份额申赎；核对首次公开、折算、修订、期初期末衔接。公募全市场仍须另补，不能用单只ETF代替。"}
    write_json(OUT / "example_2021Q3_result.json", result, exclusive=True)
    pd.DataFrame([facts[0]]).to_csv(OUT / "510300原始季报申购赎回份额_已核实实例.csv", index=False, encoding="utf-8-sig")
    text = """# 公募申购：510300原始季报份额变动实例

已找到并核对510300原始2021年三季报。上交所与巨潮两份文件逐字节相同，报告封面送出日和两个档案地址日期均为2021年10月27日；PDF元数据为此前一天生成。封面与第12页表格已查看。

| 项目 | 原始数量（份） |
|---|---:|
| 期初份额 | 9,186,787,690 |
| 本季申购 | 4,397,400,000 |
| 本季赎回 | 5,709,600,000 |
| 期末份额 | 7,874,587,690 |

期初加申购减赎回与期末精确相等。本季净申购为负1,312,200,000份，即净赎回13.122亿份。拆分格原文为横线，原始状态保留；未将缺失文件或未知记录填零。

这些是基金份额，不能称为同等人民币金额。即使乘季度末净值，也只是按一个时点估算的价值，不能冒充期间实际现金流。申购和赎回可能同时发生，二者总量与净额应分别保留。

该来源能补充直接针对510300的份额申赎；全市场公募申购仍需另行收集，不能用一只ETF代表全市场。该数据也不能回填到9月末，因为报告10月才送出。按送出日期保守计算的下一交易日为2021年10月28日；原始公告目录、首次发布与修订仍须在批量历史中核对。本例暂未获准进入新策略。

后续先恢复各期原始报告及日期，核对期初期末衔接、份额折算、基金身份和联接基金区别，再事先登记份额净申购强度与申赎结构等候选因子。它们是否预测未来回报，需要完整账户验证，不能从资料可取得直接推断有效。

原始材料：[上交所季报](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2021-10-27/510300_20211027_1_INbeqLrb.pdf)、[巨潮同版季报](https://static.cninfo.com.cn/finalpage/2021-10-27/1211391984.PDF)。本次为来源和口径核对，没有新回测。
"""
    (OUT / "公募申购_原始基金报告路线与已核实实例.md").write_text(text, encoding="utf-8")
    print(json.dumps({"状态": result["status"], "原始报告": 1, "官方同版档案": 2, "净申购份额": facts[0]["net_subscription_units"], "份额衔接误差": 0.0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
