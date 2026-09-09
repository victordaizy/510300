"""固定招商银行一份原始报告的表列与单位实例，不冒充通用财报解析器。"""
from pathlib import Path
import hashlib
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, write_json

OUT = ROOT / "reports/research/510300_original_earnings_source_completion_v1"
ID = "1211361859"


def main():
    path = OUT / "bank_eps_source_example_1211361859.json"
    require(not path.exists(), "银行每股盈利样本已保存")
    record = json.loads((OUT / "batch_01_records" / f"{ID}.json").read_text(encoding="utf-8"))
    raw = Path(r"E:\ResearchData\New project 8") / record["raw_path"]
    require(hashlib.sha256(raw.read_bytes()).hexdigest() == record["sha256"], "原始报告哈希不符")
    page = next(p["text"] for p in record["extracted_pages"] if p["page"] == 2)
    require(all(s in page for s in ["2021年1-9月", "2021年7-9月", "人民币百万元", "优先股股息和永续债利息", "客户存款净增加额"]), "原始页表头或权益注释不符")
    numeric = r"([-+]?\d[\d,]*(?:\.\d+)?)"
    rows = []

    def take(name, label, count, position, multiplier, unit, period):
        pattern = re.escape(label) + r"\s+" + r"\s+".join([numeric] * count)
        match = re.search(pattern, page)
        require(match is not None, "原始页目标行缺失：" + name)
        cells = match.groups()
        raw_value = cells[position]
        value = float(raw_value.replace(",", "")) * multiplier
        rows.append({"指标": name, "数值": value, "统一单位": unit, "报告范围": period, "原文数值": raw_value,
                     "原文全部数值列": "；".join(cells), "所用数值列序号": position + 1,
                     "来源页码": 2, "来源网址": record["official_pdf_url"], "原始文件哈希": record["sha256"]})

    take("归母净利润", "归属于本行股东的净利润", 4, 2, 1_000_000, "人民币元", "2021年1至9月累计")
    take("营业收入", "营业收入", 4, 2, 1_000_000, "人民币元", "2021年1至9月累计")
    take("普通股基本每股盈利", "归属于本行普通股股东的基本每股收益", 4, 2, 1, "人民币元每股", "2021年1至9月累计")
    take("普通股稀释每股盈利", "归属于本行普通股股东的稀释每股收益", 4, 2, 1, "人民币元每股", "2021年1至9月累计")
    take("期末总资产", "总资产", 3, 0, 1_000_000, "人民币元", "2021年9月30日余额")
    take("期末归母权益", "归属于本行股东权益", 3, 0, 1_000_000, "人民币元", "2021年9月30日余额")
    pd.DataFrame(rows).to_csv(OUT / "招商银行原始报告_普通股盈利口径实例.csv", index=False, encoding="utf-8-sig")
    result = {"verified_at": now(), "status": "ONE_ORIGINAL_BANK_REPORT_LABEL_UNIT_AND_COLUMN_VERIFIED",
              "announcement_id": ID, "ts_code": "600036.SH", "report_period": "2021-09-30",
              "event_publication_date": "2021-10-23", "conservative_first_available_session": "2021-10-25",
              "first_available_time": "2021-10-25T09:30:00+08:00", "source_url": record["official_pdf_url"],
              "source_pdf_sha256": record["sha256"], "source_page": 2, "page_image_visually_inspected": True,
              "fact_count": len(rows), "facts": rows, "generic_bank_parser_validated": False,
              "all_twenty_four_reports_individually_verified": False, "portfolio_return_calculated": False,
              "notes": ["表格同页同时列单季与年初累计及同比，当前样本的累计数在第三个数值列，不能默认取第一列。",
                        "普通股每股盈利口径要扣除归属于优先股和永续债等其他权益的回报，不等于未经调整的归母净利润直接除股数。",
                        "经营现金流受客户存款变化影响，不能直接套用工业企业现金利润差的经济解释。"]}
    write_json(path, result, exclusive=True)
    text = """# 银行每股盈利的原始报告实例

招商银行2021年第三季度报告第2页，已经逐列核对原始表头、金额单位和权益说明。下述内容仅证明这一份报告的具体字段，不代表24份新文件或全部银行历史已完成验证。

该表的归母净利润年初累计为936.15亿元，普通股基本每股盈利年初累计为3.62元。两项口径不能直接混用：报告说明计算普通股每股盈利、普通股净资产收益率等指标时，要扣除优先股股息及永续债利息；相应净资产也扣除其他权益工具余额。

同一行还列有当年第三季度单季每股盈利1.27元。单季数与1至9月累计数同时出现，是财报自动提取的实际风险。此页年初累计数位于第三个数值列，不能默认抓每行第一个数字。

报告还说明，经营现金流下降主要受客户存款净增加额变化影响。这一例子支持把银行与工业企业的现金流含义分开，不能看到银行现金流下降就直接断言利润质量恶化。

下一版盈利资料应分别保留普通股基本每股盈利、稀释每股盈利、普通股股数、其他权益回报扣除、累计期间和报告公布日。跨期汇总还要处理股本变化及追溯调整，不能直接把累计每股盈利当滚动十二个月盈利。股东回报中的分红、回购和其他权益工具收益要区分受益对象，避免重复计算。

来源为[招商银行2021年第三季度原始报告第2页](https://static.cninfo.com.cn/finalpage/2021-10-23/1211361859.PDF)。官方公告日期为2021年10月23日，本研究保守地从下一A股交易日2021年10月25日开盘起使用。原始PDF、页面图、逐列数值和哈希与本说明一同保存。尚未进行任何新账户评价。
"""
    (OUT / "金融公司每股盈利与股东回报_原始报告口径实例.md").write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "facts"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
