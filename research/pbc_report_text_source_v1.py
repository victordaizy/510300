"""取得固定季度的央行报告原文，构造两种不带方向的文本变化；只作研究。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_pbc_report_text_source_v1"
INDEX = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html"
CALENDAR = ROOT / "reports/research/510300_all_factor_macro_earnings_joint_v1/results/全部3488共同源视图_不足保留.parquet"
CN_QUARTERS = {"一": 1, "二": 2, "三": 3, "四": 4}
EXPECTED = ["2011Q4"] + [f"{year}Q{quarter}" for year in range(2012, 2027) for quarter in range(1, 5)
                          if (year, quarter) <= (2026, 2)]
POLICY_HEADINGS = ("下一阶段货币政策主要思路", "下一阶段主要政策思路", "下一阶段主要思路",
                   "下一阶段的货币政策主要思路", "下一阶段货币政策思路")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def compact(text):
    return re.sub(r"\s+", "", text)


def download(url, name):
    """每个地址只尝试一次，保存失败/原响应；不关闭TLS验证或重试。"""
    path = OUT / "sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"url": url, "request_started_at": now(), "attempts": 1, "status": "FAILED"}
    try:
        with requests.get(url, timeout=(10, 25), headers={"User-Agent": "Mozilla/5.0"}) as response:
            data = response.content
            path.write_bytes(data)
            result.update(http_status=response.status_code, final_url=response.url, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(), path=path.absolute().relative_to(ROOT).as_posix(),
                          content_type=response.headers.get("Content-Type", ""))
            result["status"] = "RECEIVED" if response.status_code == 200 else "HTTP_FAILED"
    except requests.RequestException as exc:
        result["error"] = str(exc)
    result["request_completed_at"] = now()
    save("receipts/" + name.replace(".", "_") + ".json", result)
    return result


def parse_index(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for anchor in soup.find_all("a", href=True):
        match = re.search(r"(20\d{2})年第([一二三四])季度.*货币政策执行报告", compact(anchor.get_text()))
        if not match:
            continue
        quarter = f"{match.group(1)}Q{CN_QUARTERS[match.group(2)]}"
        if quarter not in EXPECTED:
            continue
        dates = []
        for parent in anchor.parents:
            quarter_titles = re.findall(r"20\d{2}年第[一二三四]季度.*?货币政策执行报告", compact(parent.get_text()))
            if len(quarter_titles) > 1:
                break
            dates = re.findall(r"20\d{2}-\d{2}-\d{2}", parent.get_text())
            if dates:
                break
        rows.append({"quarter": quarter, "title": anchor.get_text(strip=True),
                     "page_url": urljoin(INDEX, anchor["href"]),
                     "index_publication_date": max(dates) if dates else None})
    frame = pd.DataFrame(rows, columns=["quarter", "title", "page_url", "index_publication_date"])
    frame = frame.drop_duplicates(["quarter", "page_url"])
    if frame.quarter.duplicated().any():
        raise ValueError("官方目录同季度有不同发布页，须保留冲突，不自动选有利版本。")
    return frame


def parse_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    dated = re.findall(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", text)
    attachments = []
    for anchor in soup.find_all("a", href=True):
        candidate = urljoin(url, anchor["href"])
        if urlparse(candidate).path.lower().endswith(".pdf") and not re.search(r"英文|English", anchor.get_text(), re.I):
            attachments.append(candidate)
    return {"page_publication_at": dated[0] + "+08:00" if dated else None,
            "pdf_urls": list(dict.fromkeys(attachments))}


def extract_pdf(path):
    document = pdfium.PdfDocument(str(path))
    pages = []
    try:
        for i in range(len(document)):
            page = document[i]
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
    finally:
        document.close()
    return pages


def split_sections(pages):
    """取正文最后一个章节标题，排除目录；未知章节不代用内容摘要。"""
    text = compact("\n".join(pages))
    economic_start = text.rfind("第四部分宏观经济分析")
    forecast_start = text.rfind("第五部分货币政策趋势")
    headings = [(text.rfind(name), name) for name in POLICY_HEADINGS]
    policy_start, used = max(headings)
    economic = text[economic_start:forecast_start] if 0 <= economic_start < forecast_start else None
    guidance = text[policy_start:] if forecast_start >= 0 and policy_start > forecast_start else None
    return {"economic": economic, "guidance": guidance, "policy_heading": used if guidance else None,
            "economic_characters": len(economic) if economic else 0,
            "guidance_characters": len(guidance) if guidance else 0}


def bigrams(text):
    counter = Counter()
    for segment in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        counter.update(segment[i:i + 2] for i in range(len(segment) - 1))
    return counter


def similarity(left, right):
    a, b = bigrams(left), bigrams(right)
    denominator = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    if denominator == 0:
        return None
    return sum(v * b.get(k, 0) for k, v in a.items()) / denominator


def first_known_origins(frame, dates):
    """把原文公布上界映射到实际15:05原点；周末/晚间不回填前收盘。"""
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("报告对应日历须唯一递增。")
    cuts = calendar.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    result = frame.copy()
    mapped = []
    for value in result.publication_upper:
        if not value:
            mapped.append(pd.NaT)
            continue
        pos = int(cuts.searchsorted(pd.Timestamp(value), side="left"))
        mapped.append(calendar[pos] if pos < len(calendar) else pd.NaT)
    result["first_known_origin"] = pd.to_datetime(mapped)
    return result


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    save("protocol.json", {"at": now(), "registration": "TECH.R263", "decision": "TECH.R264",
          "hypothesis": "政策指引文本变化与经济描述文本变化是不同信息，可能帮助区分上涨启动/延续背景；文本变化本身不定义涨跌方向。",
          "primary_mechanism_source": "https://www.jryj.org.cn/CN/abstract/abstract897.shtml",
          "paper_access": "期刊搜索摘要可读，完整原文尚未取得，不声称精确复现论文情绪字典或预测收益。",
          "opposing_evidence": "https://doi.org/10.1016/j.ememar.2019.05.002；研究即时价格反应且不持久，不能推出日周线延续收益。",
          "source_index": INDEX, "expected_quarters": EXPECTED, "expected_count": len(EXPECTED),
          "max_source_http_requests": 1 + 2 * len(EXPECTED), "one_attempt_per_url": True,
          "history_scope": "完整2012Q1—2026Q2加2011Q4基准；不选上涨报告或补历史无动作0。",
          "availability": "官方发布页完整时刻、目录公布日末和封面日期日末的最大上界；在原3488日15:05日历首次可知。实际下载时间另记，不冒充历史first_seen。",
          "sections": "正文第四部分宏观经济分析、第五部分中的下一阶段货币政策主要思路；最后标题定位排目录，缺失不代摘要。",
          "policy_heading_variants": POLICY_HEADINGS,
          "numeric_description": "中文字符二元频数余弦相似度，分别与立即上一季度比较；使用原文本，不拟合词表、不训练情绪字典、不读收益标签。",
          "direction": "NONE；相似度/1-相似度不是利好分、胜率、超预期或交易信号。",
          "missing": "发布页/原件/季度身份/章节/前一期任一不具备即未知；不跨缺失季度比较。",
          "minimum_section_characters": 100, "new_accounts": 0, "new_fits": 0, "new_labels": 0,
          "financial_admission": "NOT_ADMITTED_METHOD_NOT_REGISTERED", "independent_validation": "NOT_ESTABLISHED",
          "target_contract": "config/510300_high_return_sharpe_goal_v1.json",
          "scope": ["510300.SH", "CASH_CNY"], "orders_authorized": False,
          "previous_turn": "R262明确目标和完整差距，非模型改善；本轮系统实际恢复active，blocked审核重新计数",
          "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "goal_achieved": False})
    print("已冻结59季度官方原文范围及两章节变化定义，未读取新收益结果。", flush=True)


def report_page(row):
    receipt = download(row["page_url"], row["quarter"] + "_page.html")
    if receipt["status"] != "RECEIVED":
        return {**row, "page_status": receipt["status"], "pdf_urls": [], "page_publication_at": None}
    html = (ROOT / receipt["path"]).read_bytes().decode("utf-8", errors="replace")
    return {**row, "page_status": receipt["status"], **parse_page(html, row["page_url"])}


def run():
    protocol = read(OUT / "protocol.json")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol["code_sha256"]:
        raise RuntimeError("来源登记后源码变化。")
    save("run_started.json", {"at": now(), "planned_quarters": len(EXPECTED)})
    index = download(INDEX, "official_index.html")
    if index["status"] != "RECEIVED":
        save("summary.json", {"at": now(), "decision": "TECH.R264", "status": "OFFICIAL_INDEX_REQUEST_FAILED",
             "source_receipt": index, "new_accounts": 0, "new_fits": 0, "goal_achieved": False})
        save("run_completed.json", {"at": now(), "terminal": True, "status": "OFFICIAL_INDEX_REQUEST_FAILED"})
        print("官方目录请求失败，原失败保留；不重试或运行金融模型。", flush=True)
        return
    listing = parse_index((ROOT / index["path"]).read_bytes().decode("utf-8", errors="replace"))
    listing.to_parquet(OUT / "全部预定季度_官方目录.parquet", index=False)
    pages = {}
    if listing.page_url.duplicated().any():
        raise ValueError("不同季度指向同一发布页，须先核清季度身份，不重复请求。")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(report_page, row): row["quarter"] for row in listing.to_dict("records")}
        for future in as_completed(futures):
            row = future.result()
            pages[row["quarter"]] = row
            print(f"发布页{row['quarter']}：{row['page_status']}，中文PDF候选{len(row['pdf_urls'])}。", flush=True)
    pdf_receipts = {}
    pdf_groups = {}
    for quarter, row in pages.items():
        if len(row["pdf_urls"]) == 1:
            pdf_groups.setdefault(row["pdf_urls"][0], []).append(quarter)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download, url, min(quarters) + ".pdf"): quarters
                   for url, quarters in pdf_groups.items()}
        for future in as_completed(futures):
            quarters, receipt = futures[future], future.result()
            for quarter in quarters:
                pdf_receipts[quarter] = receipt
            print(f"原文{'/'.join(quarters)}：{receipt['status']}，{receipt.get('bytes', 0)}字节。", flush=True)
    results, texts = [], {}
    logging.getLogger("pypdfium2").setLevel(logging.ERROR)
    for quarter in EXPECTED:
        row = {"quarter": quarter, "status": "NO_VIEW_SOURCE", "historical_first_vintage": "NOT_CERTIFIED",
               "page_present": quarter in pages, "pdf_received": False, "sections_valid": False}
        page = pages.get(quarter, {})
        row.update({k: page.get(k) for k in ("page_url", "index_publication_date", "page_publication_at")})
        receipt = pdf_receipts.get(quarter, {})
        if receipt.get("status") == "RECEIVED":
            try:
                path = ROOT / receipt["path"]
                if not path.read_bytes().startswith(b"%PDF"):
                    raise ValueError("响应不是PDF原件")
                content = extract_pdf(path)
                cover = compact("\n".join(content[:2]))
                expected_year, expected_quarter = map(int, quarter.split("Q"))
                actual = re.search(r"(20\d{2})年第([一二三四1234])季度", cover)
                identity = bool(actual and int(actual.group(1)) == expected_year
                                and int(CN_QUARTERS.get(actual.group(2), actual.group(2))) == expected_quarter)
                date = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", cover)
                cover_date = f"{date.group(1)}-{int(date.group(2)):02}-{int(date.group(3)):02}" if date else None
                dates = [pd.Timestamp(row["page_publication_at"]) if row["page_publication_at"] else None]
                dates += [pd.Timestamp(d + "T23:59:59+08:00") if d else None for d in (row["index_publication_date"], cover_date)]
                upper = max((d for d in dates if d is not None), default=None)
                sections = split_sections(content)
                valid_sections = sections["economic_characters"] >= 100 and sections["guidance_characters"] >= 100
                row.update(pdf_received=True, pdf_url=receipt["url"], pdf_path=receipt["path"], pdf_sha256=receipt["sha256"],
                           pages=len(content), cover_identity_valid=identity, cover_date=cover_date,
                           publication_upper=upper.isoformat() if upper is not None else None,
                           publication_clock_conflict=bool(cover_date and row["index_publication_date"] and cover_date != row["index_publication_date"]),
                           sections_valid=valid_sections, **{k: v for k, v in sections.items() if k not in ("economic", "guidance")},
                           downloaded_at=receipt["request_completed_at"])
                row["status"] = "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS" if identity and valid_sections and upper is not None else "NO_VIEW_IDENTITY_CLOCK_OR_SECTION"
                (OUT / "texts").mkdir(exist_ok=True)
                (OUT / "texts" / (quarter + "_full.txt")).write_text("\n\f\n".join(content), encoding="utf-8")
                for part in ("economic", "guidance"):
                    if sections[part]:
                        (OUT / "texts" / (quarter + "_" + part + ".txt")).write_text(sections[part], encoding="utf-8")
                if row["status"] == "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS":
                    texts[quarter] = sections
            except Exception as exc:
                row.update(status="NO_VIEW_PDF_PARSE_FAILED", parse_error=str(exc))
        previous = EXPECTED[EXPECTED.index(quarter) - 1] if EXPECTED.index(quarter) > 0 else None
        row["previous_quarter"] = previous
        for part in ("economic", "guidance"):
            row[part + "_similarity"] = similarity(texts[quarter][part], texts[previous][part]) if quarter in texts and previous in texts else None
        results.append(row)
        print(f"提取{quarter}：{row['status']}。", flush=True)
    frame = pd.DataFrame(results)
    if "publication_upper" not in frame:
        frame["publication_upper"] = None
    frame = first_known_origins(frame, pd.read_parquet(CALENDAR, columns=["date"]).date)
    frame.to_parquet(OUT / "全部59季度_原件章节公布钟与未知.parquet", index=False)
    frame.to_csv(OUT / "全部59季度_原件章节公布钟与未知.csv", index=False, encoding="utf-8-sig")
    summary = {"at": now(), "decision": "TECH.R264", "status": "COMPLETED_FIXED_QUARTER_SOURCE_INTAKE",
               "expected_reports": len(EXPECTED), "listed_reports": len(listing), "pdf_received": int(frame.pdf_received.sum()),
               "two_sections_qualified": int(frame.status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS").sum()),
               "paired_two_channel_changes": int(frame.economic_similarity.notna().mul(frame.guidance_similarity.notna()).sum()),
               "all_quarters_complete": bool(frame.status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS").all()),
               "publication_date_conflicts": int(frame.get("publication_clock_conflict", pd.Series(False, index=frame.index)).fillna(False).sum()),
               "first_vintage_authenticated": False, "financial_admission": "NOT_ADMITTED_METHOD_NOT_REGISTERED",
               "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_trading_scores": 0,
               "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "orders_authorized": False}
    save("summary.json", summary)
    save("run_completed.json", {"at": now(), "terminal": True, "status": summary["status"]})
    print(f"固定59季度结束：取得{summary['pdf_received']}原件，双章节合格{summary['two_sections_qualified']}，双通道比较{summary['paired_two_channel_changes']}；尚无新策略收益。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="央行政策文本固定季度来源与两章节变化研究。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()


if __name__ == "__main__":
    main()
