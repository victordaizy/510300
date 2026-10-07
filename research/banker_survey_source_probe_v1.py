"""定向保存央行银行家问卷原始报告和公布时钟，尚不进行策略拟合。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_banker_survey_source_probe_v1"
STUDY = "510300_BANKER_SURVEY_SOURCE_PROBE_V1"
BASE = "https://www.pbc.gov.cn/diaochatongjisi/116219/116227/"
LISTS = [BASE + "index.html", *[BASE + f"11874-{i}.html" for i in range(2, 6)]]
QUARTERS = {"一": 1, "二": 2, "三": 3, "四": 4}


def fetch(url, name):
    receipt_path = OUT / "requests" / f"{name}.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("raw_path"):
            assert digest(ROOT / receipt["raw_path"]) == receipt["sha256"]
        return receipt
    if (OUT / "REQUESTS_STOPPED.json").exists():
        raise RuntimeError("来源已要求停止新增请求。")
    assert urlparse(url).hostname == "www.pbc.gov.cn"
    assert len(list((OUT / "requests").glob("*.json"))) < 80
    receipt = {"url": url, "requested_at": now(), "status": "REQUEST_STARTED"}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0", "Referer": BASE})
        path = OUT / "raw" / (name + ".bin")
        path.write_bytes(response.content)
        receipt.update(received_at=now(), http_status=response.status_code,
                       status="HTTP_OK" if response.ok else "HTTP_FAILURE", bytes=len(response.content),
                       raw_path=path.relative_to(ROOT).as_posix(), sha256=digest(path),
                       content_type=response.headers.get("Content-Type"), http_date=response.headers.get("Date"))
        if response.status_code in [403, 429]:
            save(OUT / "REQUESTS_STOPPED.json", {"at": now(), "http_status": response.status_code, "url": url}, True)
    except requests.RequestException as exc:
        receipt.update(received_at=now(), status="TRANSPORT_FAILURE", error=f"{type(exc).__name__}: {exc}")
    save(receipt_path, receipt, True)
    return receipt


def soup_for(receipt):
    assert receipt["status"] == "HTTP_OK", receipt
    raw = (ROOT / receipt["raw_path"]).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    return BeautifulSoup(text, "html.parser")


def catalogue():
    if (OUT / "catalogue.json").exists():
        raise RuntimeError("目录已经保存，直接进行原报告取得。")
    for folder in ["raw", "requests", "texts", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "purpose": "辨认贷款总体需求、银行贷款审批和货币政策感受的分别定义及已公布历史，避免用贷款数量的变化直接宣称需求变化。",
        "catalogues": LISTS, "quarter_lower": "2018Q1", "quarter_upper": "2026Q3",
        "request_limit": 80, "transport_timeout_seconds": 25,
        "requests": "仅指定央行公开目录与实际链接，保存原始响应。403/429停止，既有请求直接复用响应，不绕过。",
        "clock": "记录每份报告落地页所載公布时间。采用保守日末可用，不把报告季度末或PDF文件名时间当首发时钟。某些季度同时发布或延迟发布，均保留。",
        "vintage": "只允许以后提取每份当季原报告的当季数，附表早期回溯值只用于核对且不向过去回填。不可变首版尚未认证。",
        "semantics": "调查扩散指数不是贷款数量、实际批准率或中性利率；需核实选项和表头，不能按旧列号盲提取。",
        "sample_limit": "两年日更不会将约八个季度事实变成数百独立宏观样本，日频填充和延迟的样本损失需单独计量。",
        "new_strategy_returns": 0, "new_accounts": 0, "goal_achieved": False,
        "orders_authorized": False, "current_market_view": "NO_VIEW"}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "code_sha256": digest(Path(__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    found = {}
    for i, url in enumerate(LISTS, 1):
        receipt = fetch(url, f"catalogue_{i}")
        page = soup_for(receipt)
        for link in page.find_all("a", href=True):
            title = re.sub(r"\s+", "", link.get_text())
            match = re.fullmatch(r"(20\d{2})年第([一二三四])季度银行家问卷调查报告", title)
            if not match:
                continue
            quarter = f"{match.group(1)}Q{QUARTERS[match.group(2)]}"
            if not "2018Q1" <= quarter <= "2026Q3":
                continue
            source_url = urljoin(url, link["href"])
            row = {"quarter": quarter, "title": title, "source_url": source_url,
                   "catalogue_url": url, "catalogue_receipt": receipt["raw_path"]}
            if quarter in found:
                assert found[quarter]["source_url"] == source_url
            else:
                found[quarter] = row
        print(f"央行目录{i}已保存，累计定位{len(found)}份银行家原报告。", flush=True)
    save(OUT / "catalogue.json", sorted(found.values(), key=lambda r: r["quarter"]), True)


def acquire():
    assert digest(Path(__file__)) == read(OUT / "freeze.json")["code_sha256"]
    if (OUT / "result.json").exists():
        raise RuntimeError("原报告取得已完成，不能覆盖。")
    records, failures = [], []
    for item in read(OUT / "catalogue.json"):
        name = item["quarter"]
        record = {**item, "status": "STARTED"}
        try:
            html = fetch(item["source_url"], name + "_page")
            page = soup_for(html)
            content = page.get_text(" ", strip=True)
            clocks = re.findall(r"文章来源[：:]?\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", content)
            assert len(clocks) == 1, "原报告公布时钟未唯一识别。"
            links = {urljoin(item["source_url"], a["href"]) for a in page.find_all("a", href=True)
                     if re.search(r"\.pdf(?:$|[?#])", a["href"], flags=re.I)}
            assert len(links) == 1, "PDF原始附件未唯一识别。"
            pdf_url = next(iter(links))
            pdf = fetch(pdf_url, name + "_pdf")
            assert pdf["status"] == "HTTP_OK" and (ROOT / pdf["raw_path"]).read_bytes().startswith(b"%PDF"), "附件不是可读PDF。"
            with pdfplumber.open(ROOT / pdf["raw_path"]) as doc:
                texts = [p.extract_text() or "" for p in doc.pages]
            text_path = OUT / "texts" / (name + ".json")
            save(text_path, {"quarter": name, "pages": texts, "pdf_sha256": pdf["sha256"]}, True)
            merged = re.sub(r"\s+", "", "".join(texts))
            assert item["title"] in merged, "原报告标题与PDF季度不一致。"
            record.update(status="ORIGINAL_PDF_AND_CLOCK_SAVED_FIELDS_NOT_ADMITTED", published_at=clocks[0] + "+08:00",
                          conservative_known_at=clocks[0][:10] + "T23:59:59+08:00", page_raw_path=html["raw_path"],
                          page_sha256=html["sha256"], pdf_url=pdf_url, pdf_raw_path=pdf["raw_path"], pdf_sha256=pdf["sha256"],
                          page_count=len(texts), text_path=text_path.relative_to(ROOT).as_posix(),
                          historical_first_vintage_verified=False)
            records.append(record)
        except (AssertionError, ValueError, RuntimeError) as exc:
            record.update(status="SOURCE_OR_SCHEMA_UNRESOLVED", error=str(exc))
            failures.append(record)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
        if (len(records) + len(failures)) % 8 == 0:
            print(f"银行家原报告成功{len(records)}份，待处理{len(failures)}份。", flush=True)
    save(OUT / "saved_originals.json", records, True)
    save(OUT / "unresolved_sources.json", failures, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "ORIGINAL_SURVEYS_SAVED_FIELD_ADMISSION_PENDING" if not failures else "PARTIAL_ORIGINAL_SURVEYS_SAVED",
        "catalogued_quarters": len(read(OUT / "catalogue.json")), "saved_quarters": len(records),
        "unresolved_quarters": len(failures), "quarter_first": records[0]["quarter"] if records else None,
        "quarter_last": records[-1]["quarter"] if records else None,
        "latest_saved_publication": max(r["published_at"] for r in records) if records else None,
        "requests_saved": len(list((OUT / "requests").glob("*.json"))),
        "new_strategy_accounts": 0, "new_strategy_returns": 0, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False}, True)
    print(f"银行家季度原报告保存完成：{len(records)}份，未解决{len(failures)}份；尚未准入策略。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="央行银行家问卷历史原文探查")
    parser.add_argument("command", choices=["catalogue", "acquire"])
    globals()[parser.parse_args().command]()
