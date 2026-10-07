"""一次取得保存目录中未收到的28官方发布原件，复用原8份，不运行策略。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from threading import Lock
import time

import requests

from research import official_industry_publication_intake_v1 as previous

parent = previous.parent
ROOT = previous.ROOT
OUT = ROOT / "reports/research/510300_official_industry_full_sequence_v1"
INDEX = previous.OUT / "complete_catalog_publication_index.json"
REGISTRATION, RESULT = "TECH.R219", "TECH.R220"


def nodes():
    index = parent.read(INDEX)
    result = []
    for row in index["rows"]:
        item = dict(row)
        item["taxonomy"] = "CSRC_QUARTERLY" if row["catalog"].startswith("CSRC") else "CAPCO_2023"
        item["period_title"] = row["id"][:4] + ("年" + row["id"][-1] + "季度" if item["taxonomy"] == "CSRC_QUARTERLY" else
            "年" + ("上半年" if row["id"].endswith("H1") else "下半年"))
        result.append(item)
    if len(result) != 36 or len({item["id"] for item in result}) != 36 or sum(not item["already_received"] for item in result) != 28:
        raise ValueError("固定目录不是36唯一节点/28未收到。")
    return result


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("全发布来源用途已登记，不覆盖。")
    old = parent.read(previous.OUT / "summary.json")
    paths = [Path(__file__), Path(previous.__file__), ROOT / "docs/510300_OFFICIAL_INDUSTRY_FULL_SEQUENCE_V1.md",
        INDEX, previous.OUT / "summary.json", previous.OUT / "protocol.json"]
    paths.extend(ROOT / item["pdf_path"] for item in old["nodes"])
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": REGISTRATION, "decision": RESULT,
        "scope": "SAVED_CATALOG_ALL36_RELEASES_REUSE8_GET_REMAINING28_THEN_ROW_SOURCE_COVERAGE",
        "nodes": nodes(), "new_node_count": 28, "reuse_received_node_count": 8,
        "maximum_new_logical_gets": 84, "max_workers": 4,
        "request_policy": "每具体URL仅请求一次，无重试；附件为公开入口才允许一次唯一PDF解析；保存部分响应和失败。",
        "clock": "目录日期必须与页面日期字段一致，未知日内按公布日23:59；不是覆盖期末。",
        "source_role": "整表完整性与逐行来源资格两个字段，保持原3整表失败；明确行可用于新的逐行用途，未知行不补。",
        "row_source_eligible": "标题合格、正文证券集合=解析集合、无证券重复；证券本身门类/大类代码名称与行出处明确。",
        "latest_failed_source": "最新已公布节点结构失败/缺原件保持NO_VIEW，不回退旧节点；最新部分未知证券不借旧表倒补。",
        "first_vintage": "NOT_CERTIFIED", "missing_catalog_periods": ["2021Q4", "2022"],
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED_SOURCE_PURPOSE_ONLY",
        "files": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": parent.digest(path)} for path in paths]})
    print("R219全序列来源用途固定：36节点，复用8，新增28最多84GET；逐行用途另立，0金融。", flush=True)


class Collector:
    def __init__(self):
        self.lock = Lock()
        self.seen = set()
        self.receipts = []

    def get(self, key, url, suffix):
        path = OUT / "sources" / f"{key}.{suffix}"
        with self.lock:
            if url in self.seen or path.exists():
                record = {"key": key, "url": url, "status": "DUPLICATE_URL_NOT_REQUESTED",
                    "started_at": parent.original.now(), "finished_at": parent.original.now()}
                self.receipts.append(record)
                return None, record
            self.seen.add(url)
        record = {"key": key, "url": url, "started_at": parent.original.now(), "path": str(path.absolute().relative_to(ROOT))}
        body = None
        began = time.monotonic()
        try:
            with requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/pdf,*/*"},
                    timeout=(10, 25), stream=True) as response:
                record.update(http_status=response.status_code, final_url=response.url,
                    redirects=[{"status": item.status_code, "url": item.url} for item in response.history],
                    content_type=response.headers.get("Content-Type"))
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    for chunk in response.iter_content(chunk_size=65536):
                        if time.monotonic() - began > 120:
                            raise TimeoutError("单次来源请求超过120秒总时限，保留部分响应，不重试。")
                        stream.write(chunk)
                raw = path.read_bytes()
                record.update(bytes=len(raw), sha256=previous.sha256(raw),
                    status="HTTP_SUCCESS" if response.status_code == 200 else "HTTP_FAILURE")
                if response.status_code == 200:
                    body = raw
        except (requests.RequestException, TimeoutError) as error:
            record.update(status="REQUEST_FAILED", exception_type=type(error).__name__, error=str(error))
            if path.exists():
                record.update(partial_bytes=path.stat().st_size, partial_sha256=parent.digest(path))
        record["finished_at"] = parent.original.now()
        with self.lock:
            self.receipts.append(record)
        parent.write(OUT / "request_receipts" / f"{key}.json", record)
        print(f"{key}：{record['status']}。", flush=True)
        return body, record


def acquire(node, collector):
    item = dict(node)
    raw, receipt = collector.get(node["id"] + "_PAGE", node["url"], "html")
    item.update(page_receipt=receipt["key"], page_clock_verified=False, pdf_received=False,
        first_vintage="NOT_CERTIFIED", financial_admission="NOT_ESTABLISHED", source_reused=False)
    if raw is None:
        item["status"] = "PAGE_REQUEST_FAILED"
        return item
    soup, text = previous.html_text(raw)
    (OUT / "sources" / f"{node['id']}_PAGE.txt").write_text(text, encoding="utf-8")
    dates = set(re.findall(r"(?:发布时间|日期)\s*[:：]\s*(\d{4}-\d{2}-\d{2})", text))
    title = node["period_title"] in re.sub(r"\s+", "", text)
    if dates != {node["published"]} or not title:
        item.update(status="PAGE_TITLE_OR_PUBLICATION_CLOCK_NOT_VERIFIED", dates_found=sorted(dates), title_known=title)
        return item
    item.update(page_clock_verified=True, actual_publication_clock=node["published"] + "T23:59:00+08:00")
    link = previous.attachment(node, soup, receipt.get("final_url", node["url"]))
    if link is None:
        item["status"] = "PUBLIC_ATTACHMENT_NOT_UNIQUE_OR_ABSENT"
        return item
    item["attachment"] = link
    body, pdf_receipt = collector.get(node["id"] + "_ATTACHMENT", link["url"], "bin")
    if body is not None and not previous.pdf_magic(body):
        resolved = previous.linked_pdf(body, pdf_receipt.get("final_url", link["url"]))
        item["public_resolved_pdf"] = resolved
        if resolved is not None:
            body, pdf_receipt = collector.get(node["id"] + "_PDF", resolved, "bin")
    if previous.pdf_magic(body):
        path = OUT / "pdfs" / f"{node['id']}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(body)
        item.update(pdf_received=True, pdf_path=str(path.absolute().relative_to(ROOT)), pdf_sha256=previous.sha256(body),
            pdf_bytes=len(body), pdf_receipt=pdf_receipt["key"], status="OFFICIAL_PDF_AND_PAGE_RECEIVED_NOT_YET_PARSED")
    else:
        item["status"] = "ATTACHMENT_NOT_VALID_PDF_OR_REQUEST_FAILED"
    print(f"{node['id']}：{item['status']}。", flush=True)
    return item


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("全序列来源取得已经开始，不重复请求。")
    protocol = parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源代码或旧原件改变：" + item["path"])
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "maximum_new_logical_gets": 84, "new_accounts": 0})
    old = {item["id"]: item for item in parent.read(previous.OUT / "summary.json")["nodes"]}
    reused = []
    for node in protocol["nodes"]:
        if node["already_received"]:
            item = {**node, **old[node["id"]], "source_reused": True,
                "prior_intake_summary": str((previous.OUT / "summary.json").absolute().relative_to(ROOT))}
            reused.append(item)
    collector = Collector()
    selected = [node for node in protocol["nodes"] if not node["already_received"]]
    with ThreadPoolExecutor(max_workers=protocol["max_workers"]) as executor:
        added = list(executor.map(lambda node: acquire(node, collector), selected))
    records = sorted(reused + added, key=lambda item: (item["published"], item["id"]))
    receipts = sorted(collector.receipts, key=lambda item: item["started_at"])
    logical = sum(item["status"] != "DUPLICATE_URL_NOT_REQUESTED" for item in receipts)
    if logical > protocol["maximum_new_logical_gets"]:
        raise AssertionError("超过固定新增来源请求上限。")
    summary = {"at": parent.original.now(), "registration": REGISTRATION, "decision": RESULT,
        "status": "ALL_VISIBLE_PUBLICATION_NODES_TERMINAL_SOURCE_INTAKE_NOT_FINANCIAL",
        "nodes": records, "logical_gets": logical, "http_successes": sum(item["status"] == "HTTP_SUCCESS" for item in receipts),
        "request_failures": sum(item["status"] == "REQUEST_FAILED" for item in receipts),
        "http_failures": sum(item["status"] == "HTTP_FAILURE" for item in receipts),
        "duplicate_urls_not_requested": sum(item["status"] == "DUPLICATE_URL_NOT_REQUESTED" for item in receipts),
        "redirects": sum(len(item.get("redirects", [])) for item in receipts),
        "nodes_total": len(records), "reused_pdf_nodes": len(reused), "new_pdf_nodes_received": sum(item["pdf_received"] for item in added),
        "pdfs_received_total": sum(item["pdf_received"] for item in records),
        "page_clocks_verified": sum(item["page_clock_verified"] for item in records), "requests": receipts,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_price_rows": 0,
        "parser_status": "NOT_RUN", "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED",
        "missing_catalog_periods": ["2021Q4", "2022"], "financial_admission": "NOT_ESTABLISHED_SOURCE_PURPOSE_ONLY", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    print(f"全发布序列来源结束：新增{summary['new_pdf_nodes_received']}/28原件，总{summary['pdfs_received_total']}/36；0金融，失败保留。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全官方行业发布序列的有限来源取得。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
