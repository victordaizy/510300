"""为预定三家公司核对原公告，复用已有PDF并保存提取页码。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import sys
from threading import Lock

import requests
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.operating_cashflow_sources_v7 import OUT, sha, save, now

PDF_TEXT_LOCK = Lock()


def get_one(spec):
    dest = OUT / "sources" / (spec["id"] + ".pdf")
    if not dest.exists():
        if spec.get("local_path"):
            src = ROOT / spec["local_path"]
            assert sha(src) == spec["expected_sha256"]
            shutil.copy2(src, dest)
        else:
            r = requests.get(spec["url"], timeout=(10,50))
            r.raise_for_status()
            assert r.content.startswith(b"%PDF")
            dest.write_bytes(r.content)
    text_path = dest.with_suffix(".txt")
    # PDFium在多线程中的调用串行，下载仍可并行。
    with PDF_TEXT_LOCK:
        document = pdfium.PdfDocument(str(dest))
        pages = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        document.close()
    text_path.write_text("\f".join(pages), encoding="utf-8")
    terms = ["合并现金流量表", "经营性应收项目", "经营活动产生的现金流量净额变动原因", "现金流量表补充资料", "经营活动现金流量", "经营活动产生的现金流量净额为", "现金流量净额增加", "现金流量净额减少"]
    matches = []
    for i, text in enumerate(pages):
        packed = "".join(text.split())
        hit = [word for word in terms if word in packed]
        if hit:
            matches.append({"pdf_page_one_based": i+1, "terms": hit})
    record = {**spec, "at": now(), "file": dest.name, "sha256": sha(dest), "bytes": dest.stat().st_size, "extracted_page_count": len(pages), "matched_pages": matches, "primary_scope": "上市公司原公告；与当前汇编数据的数值和修订口径另行核对。"}
    return record


def run():
    targets = json.loads((ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_1/effective_targets.json").read_text(encoding="utf-8"))
    specs = []
    for code, period, tag in [("601668.SH", "2025-06-30", "601668_2025H1"), ("601668.SH", "2024-12-31", "601668_2024FY"), ("601857.SH", "2025-06-30", "601857_2025H1")]:
        row = next(x for x in targets if x["ts_code"] == code and str(x["report_period"])[:10] == period)
        specs.append({"id": tag, "stock_code": code, "report_end": period, "url": row["url"], "local_path": row["pdf_relative_path"], "expected_sha256": row["sha256"]})
    specs.append({"id": "600048_2025H1", "stock_code": "600048.SH", "report_end": "2025-06-30", "url": "https://static.cninfo.com.cn/finalpage/2025-08-26/1224571945.PDF"})
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(get_one, spec) for spec in specs]
        for job in as_completed(jobs):
            record = job.result()
            records.append(record)
            print(record["id"], "原文页数", record["extracted_page_count"], "匹配页", record["matched_pages"], flush=True)
    save(OUT / "sources/primary_source_records.json", {"at": now(), "records": sorted(records, key=lambda x:x["id"]), "selection": "固定全体现金流TTM增量的前两名及末一名，非收益筛选；中国建筑另核前一年年报以区分期间与口径。"})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


if __name__ == "__main__":
    run()
