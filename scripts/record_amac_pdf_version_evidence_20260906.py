"""保存已归档公募月报PDF的版本元数据，不将其冒充最早公布时钟。"""
from pathlib import Path
import hashlib
import json
import sys

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, write_json

OUT = ROOT / "reports/research/510300_fundamental_and_fund_flow_rebuild_v1"
DEST = OUT / "amac_pdf_version_metadata_20260906.json"


def main():
    require(not DEST.exists(), "公募PDF版本记录已存在，禁止覆盖")
    records = json.loads((OUT / "amac_source_records.json").read_text(encoding="utf-8"))
    rows = []
    for record in records:
        path = Path(r"E:\ResearchData\New project 8") / record["raw_path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], "公募原始文件标识不符")
        with pdfplumber.open(path) as pdf:
            metadata = {str(k): str(v) for k, v in pdf.metadata.items()}
        created = metadata.get("CreationDate", "")
        rows.append({"report_month": record["report_month"], "catalogue_publication_date": record["publication_date"],
                     "source_url": record["source_url"], "raw_path": record["raw_path"], "sha256": record["sha256"],
                     "metadata": metadata, "creation_year_differs_from_report_year": created[2:6] != record["report_month"][:4],
                     "creation_date_used_as_first_release_clock": False})
    regenerated = [r for r in rows if r["report_month"] < "2023-11" and r["metadata"].get("CreationDate", "").startswith("D:20231121")]
    receipt = {"recorded_at": now(), "status": "HISTORICAL_FIRST_RELEASE_VERSION_NOT_PROVEN",
               "pdf_count": len(rows), "pre_2023_11_reports_created_2023_11_21_count": len(regenerated),
               "metadata_is_authenticated_publication_receipt": False, "historical_fund_flow_backtest_run": False,
               "conclusion": "多份旧月报的现存PDF在2023年11月21日重新生成。元数据可支持发现版本疑点，不能证明旧表数值曾在原历史日期公开，也不能替代最早发布回执。", "reports": rows}
    write_json(DEST, receipt, exclusive=True)
    lines = ["# 公募月报历史版本补充记录", "", "本轮对已归档77份官方月报保存PDF元数据，并与原始文件哈希对应。",
             f"其中，报告月份早于2023年11月、现存PDF却标记在2023年11月21日生成的有{len(regenerated)}份。",
             "PDF元数据可以被软件修改，本身不构成经认证的发布时间；现象说明还需要找到当时原始公告或逐字段可核对的同期公开材料，不能把现存表格直接回填到早期月份。", "",
             "已经完成的月度份额结构化结果继续保留。份额变化仍需区分申购赎回、新基金成立、份额拆分、清盘和分类调整；基金资产规模变化还包含净值涨跌，不能直接叫净申购。", "",
             "本次记录没有运行公募资金流策略，也没有把来源疑点改成收益为零。", "",
             "| 报告月份 | 当前目录日期 | 现存PDF生成元数据 |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['report_month']} | {row['catalogue_publication_date']} | {row['metadata'].get('CreationDate', '缺失')} |")
    (OUT / "公募月报PDF版本与原始公布时钟_补充说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "reports"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
