"""有限获取原案例所需的官方行业快照、公布页面和版本方法；不生成策略。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
from threading import Lock
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from research import broker_stage_policy_study_v1 as parent

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_official_industry_publication_intake_v1"
REGISTRATION, RESULT = "TECH.R217", "TECH.R218"
CSRC = "https://www.csrc.gov.cn/csrc/c100103/"
CAPCO = "https://www.capco.org.cn/"
NODES = [
    {"id": "2014Q3", "period_title": "2014年3季度", "published": "2014-10-13", "taxonomy": "CSRC_QUARTERLY",
     "url": CSRC + "c1452017/content.shtml"},
    {"id": "2015Q1", "period_title": "2015年1季度", "published": "2015-05-04", "taxonomy": "CSRC_QUARTERLY",
     "url": CSRC + "c1452015/content.shtml"},
    {"id": "2018Q3", "period_title": "2018年3季度", "published": "2018-11-02", "taxonomy": "CSRC_QUARTERLY",
     "url": CSRC + "c1452001/content.shtml"},
    {"id": "2019Q1", "period_title": "2019年1季度", "published": "2019-04-19", "taxonomy": "CSRC_QUARTERLY",
     "url": CSRC + "c1451999/content.shtml"},
    {"id": "2019Q4", "period_title": "2019年4季度", "published": "2020-01-10", "taxonomy": "CSRC_QUARTERLY",
     "url": CSRC + "c1451996/content.shtml"},
    {"id": "2023H2", "period_title": "2023年下半年", "published": "2024-04-03", "taxonomy": "CAPCO_2023",
     "url": CAPCO + "xhgg/hyfl/hyfljg/202404/20240403/j_2024040315552200017121310425305369.html"},
    {"id": "2024H1", "period_title": "2024年上半年", "published": "2024-09-30", "taxonomy": "CAPCO_2023",
     "url": CAPCO + "xhgg/hyfl/hyfljg/202409/20240930/j_2024093015525600017276828969068447.html"},
    {"id": "2024H2", "period_title": "2024年下半年", "published": "2025-04-18", "taxonomy": "CAPCO_2023",
     "url": CAPCO + "xhgg/hyfl/hyfljg/202504/20250418/j_2025041815003000017449597508305299.html"},
]
CATALOGS = {"CSRC_OLD": CSRC + "common_list_2.shtml", "CSRC_RECENT": CSRC + "common_list.shtml",
    "CAPCO": CAPCO + "xhgg/hyfl/hyfljg/index.html"}
GUIDELINE = CAPCO + "xhdt/tzgg/202305/20230521/j_2023052117544500016846630061707656.html"
CASE_DATES = ["2015-01-05", "2015-06-29", "2019-01-08", "2019-06-19", "2020-04-01",
    "2024-09-24", "2024-09-30", "2024-10-08", "2025-07-03"]


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("该有限来源用途已经登记，不覆盖。")
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": REGISTRATION, "result": RESULT,
        "purpose": "OFFICIAL_PUBLICATION_CLOCK_AND_PDF_FEASIBILITY_NOT_FINANCIAL_RUN",
        "nodes": NODES, "catalogs": CATALOGS, "guideline": GUIDELINE, "case_dates": CASE_DATES,
        "maximum_logical_gets": 28, "logical_gets_definition": "12个固定HTML，8个附件各最多1个解析入口与1个由入口公开给出的PDF；HTTP重定向单列。",
        "network_policy": "每个具体URL只请求一次，无自动重试；失败、原响应和重定向保留。",
        "clock": "官方页面实际公布日期核对；未知日内时刻按北京时间当天23:59，不能按季度/半年末提前。",
        "versions": "CSRC季度与CAPCO2023半年度分开，不冒充原申万31行业修复或直接拼接。",
        "scope_limit": "8个按原四案例及退出反例选择的快照；不是全部2015至2026历史源，也不凭目录/页面成功宣称PDF或覆盖通过。",
        "case_role": "原案例和已知失败用于解释及来源诊断，不是独立样本。",
        "parsing": "公开原件与页文本可另以隔离解析层复核；有PDF还需检查标题、重复证券、表格归属和300成员覆盖，未知保留。",
        "new_accounts": 0, "new_fits": 0, "new_market_price_rows": 0, "new_labels": 0,
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED",
        "first_vintage": "当前取回官方原件不自动认证历史首版；网页公布日期与实际文件版本分开。",
        "files": [{"path": str(Path(__file__).absolute().relative_to(ROOT)), "sha256": parent.digest(Path(__file__))},
            {"path": "docs/510300_OFFICIAL_INDUSTRY_PUBLICATION_INTAKE_V1.md",
             "sha256": parent.digest(ROOT / "docs/510300_OFFICIAL_INDUSTRY_PUBLICATION_INTAKE_V1.md")}],
        "prior_financial": "TECH.R216_REJECTED_UNCHANGED", "goal_achieved": False})
    print("R217有限官方分类来源用途已固定：8快照、3目录、1方法页，最多28逻辑GET；0新账户。", flush=True)


class Collector:
    def __init__(self):
        self.lock = Lock()
        self.receipts = []

    def get(self, key, url, suffix):
        path = OUT / "sources" / f"{key}.{suffix}"
        if path.exists():
            raise RuntimeError("本来源原件已存在，不能重复请求或覆盖。")
        record = {"key": key, "url": url, "started_at": parent.original.now(), "path": str(path.absolute().relative_to(ROOT))}
        raw = None
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/pdf,*/*"},
                timeout=(10, 25))
            raw = response.content
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            record.update(http_status=response.status_code, final_url=response.url,
                redirects=[{"status": value.status_code, "url": value.url} for value in response.history],
                bytes=len(raw), sha256=sha256(raw), content_type=response.headers.get("Content-Type"),
                status="HTTP_SUCCESS" if response.status_code == 200 else "HTTP_FAILURE")
            if response.status_code != 200:
                raw = None
        except requests.RequestException as error:
            record.update(status="REQUEST_FAILED", exception_type=type(error).__name__, error=str(error))
        record["finished_at"] = parent.original.now()
        with self.lock:
            self.receipts.append(record)
        print(f"{key}：{record['status']}。", flush=True)
        return raw, record


def html_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    return soup, soup.get_text(" ", strip=True)


def attachment(node, soup, final_url):
    matches = []
    for link in soup.find_all("a", href=True):
        url = urljoin(final_url, link["href"])
        label = link.get_text(" ", strip=True)
        if not url.startswith(("https://", "http://")):
            continue
        if node["taxonomy"] == "CAPCO_2023":
            selected = "按股票代码排序" in label or "按股票代码排序" in unquote(url)
        else:
            selected = urlparse(url).path.lower().endswith(".pdf")
        if selected:
            matches.append({"url": url, "label": label})
    matches = list({row["url"]: row for row in matches}.values())
    return matches[0] if len(matches) == 1 else None


def linked_pdf(raw, final_url):
    soup, _ = html_text(raw)
    urls = []
    for tag in soup.find_all(["a", "iframe", "embed", "object"]):
        value = tag.get("href") or tag.get("src") or tag.get("data")
        if value:
            absolute = urljoin(final_url, value)
            urls.append(absolute)
            urls.extend(parse_qs(urlparse(absolute).query).get("file", []))
    text = raw.decode("utf-8", errors="replace")
    urls.extend(re.findall(r'''["'](https?://[^"'\s]+\.pdf(?:\?[^"'\s]*)?)["']''', text, flags=re.I))
    urls = list(dict.fromkeys(url for url in urls if url.startswith(("https://", "http://"))
        and urlparse(url).path.lower().endswith(".pdf")))
    return urls[0] if len(urls) == 1 else None


def pdf_magic(raw):
    return raw is not None and raw.lstrip().startswith(b"%PDF-")


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本有限来源取得已开始，不重复请求。")
    protocol = parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源取得代码或用途卡改变。")
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts": 0,
        "maximum_logical_gets": 28})
    collector = Collector()
    tasks = [(node["id"] + "_PAGE", node["url"], "html") for node in NODES]
    tasks.extend(("CATALOG_" + key, url, "html") for key, url in CATALOGS.items())
    tasks.append(("CAPCO_GUIDELINE_2023", GUIDELINE, "html"))
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {key: executor.submit(collector.get, key, url, suffix) for key, url, suffix in tasks}
        for key, future in futures.items():
            results[key] = future.result()
    nodes = []
    for node in NODES:
        item = dict(node)
        raw, page_receipt = results[node["id"] + "_PAGE"]
        item.update(page_receipt=page_receipt["key"], page_clock_verified=False, pdf_received=False,
            actual_publication_clock=None, first_vintage="NOT_CERTIFIED", financial_admission="NOT_ESTABLISHED")
        if raw is None:
            item["status"] = "PAGE_REQUEST_FAILED"
            nodes.append(item)
            continue
        soup, text = html_text(raw)
        (OUT / "sources" / f"{node['id']}_PAGE.txt").write_text(text, encoding="utf-8")
        dates = set(re.findall(r"(?:发布时间|日期)\s*[:：]\s*(\d{4}-\d{2}-\d{2})", text))
        title_known = node["period_title"] in re.sub(r"\s+", "", text)
        if dates != {node["published"]} or not title_known:
            item.update(status="PAGE_TITLE_OR_PUBLICATION_CLOCK_NOT_VERIFIED", dates_found=sorted(dates), title_known=title_known)
            nodes.append(item)
            continue
        item.update(page_clock_verified=True, actual_publication_clock=node["published"] + "T23:59:00+08:00")
        link = attachment(node, soup, page_receipt.get("final_url", node["url"]))
        if link is None:
            item["status"] = "PUBLIC_ATTACHMENT_NOT_UNIQUE_OR_ABSENT"
            nodes.append(item)
            continue
        item["attachment"] = link
        body, body_receipt = collector.get(node["id"] + "_ATTACHMENT", link["url"], "bin")
        if body is not None and not pdf_magic(body):
            resolved = linked_pdf(body, body_receipt.get("final_url", link["url"]))
            item["public_resolved_pdf"] = resolved
            if resolved is not None:
                body, body_receipt = collector.get(node["id"] + "_PDF", resolved, "bin")
        if pdf_magic(body):
            path = OUT / "pdfs" / f"{node['id']}.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(body)
            item.update(pdf_received=True, pdf_path=str(path.absolute().relative_to(ROOT)),
                pdf_sha256=sha256(body), pdf_bytes=len(body), pdf_receipt=body_receipt["key"],
                status="OFFICIAL_PDF_BYTES_AND_DATED_PAGE_RECEIVED_NOT_YET_PARSED")
        else:
            item["status"] = "ATTACHMENT_NOT_VALID_PDF_OR_REQUEST_FAILED"
        nodes.append(item)
        print(f"{node['id']}：{item['status']}。", flush=True)
    receipts = sorted(collector.receipts, key=lambda row: row["started_at"])
    if len(receipts) > protocol["maximum_logical_gets"]:
        raise AssertionError("超过冻结逻辑来源请求上限。")
    pdf_count = sum(row["pdf_received"] for row in nodes)
    summary = {"at": parent.original.now(), "decision": RESULT,
        "status": "ALL_EIGHT_OFFICIAL_PDFS_RECEIVED_NOT_FINANCIAL_ADMISSION" if pdf_count == 8
            else "PARTIAL_OFFICIAL_SOURCE_INTAKE_MISSING_PDFS_PRESERVED",
        "nodes": nodes, "logical_gets": len(receipts), "http_successes": sum(row["status"] == "HTTP_SUCCESS" for row in receipts),
        "request_failures": sum(row["status"] == "REQUEST_FAILED" for row in receipts),
        "http_failures": sum(row["status"] == "HTTP_FAILURE" for row in receipts),
        "successful_response_redirect_count": sum(len(row.get("redirects", [])) for row in receipts),
        "page_clocks_verified": sum(row["page_clock_verified"] for row in nodes), "pdfs_received": pdf_count,
        "requests": receipts, "parser_status": "NOT_RUN", "case_membership_coverage": "NOT_COMPUTED",
        "full_history_coverage": "NOT_ESTABLISHED_EIGHT_SELECTED_SNAPSHOTS_ONLY",
        "financial_metrics": "NOT_COMPUTED", "financial_admission": "NOT_ESTABLISHED",
        "new_accounts": 0, "new_fits": 0, "new_market_price_rows": 0, "new_labels": 0,
        "independent_validation": "NOT_ESTABLISHED", "first_vintage": "NOT_CERTIFIED", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    lines = ["# 官方行业分类公布日与原件取得", "",
        "TECH.R217—R218。来源取得用途，一次固定8个案例快照，3目录和1方法页；没有新账户、拟合或行情日线。", "",
        f"实际逻辑GET {len(receipts)}，成功HTTP {summary['http_successes']}，请求异常{summary['request_failures']}、HTTP失败{summary['http_failures']}。公布钟已核{summary['page_clocks_verified']}/8，PDF原件取得{pdf_count}/8；标题表格解析及成员覆盖尚未完成，不能据PDF成功宣称行业金融准入。", "",
        "| 快照 | 官方公布日 | 公布钟核对 | PDF字节取得 | 状态 |", "|---|---|---|---|---|"]
    for node in nodes:
        lines.append(f"| {node['id']} | {node['published']} | {node['page_clock_verified']} | {node['pdf_received']} | {node['status']} |")
    lines += ["", "未知日内时间保守用公布日23:59。2024H1于2024-09-30公布，不能用于09-24进入或09-30收盘。2024-09-24应核2023H2的可知版本，不能改用当前申万分类。", "",
        "旧申万行业区间的生效日代理、原T04权重门和R216失败保持。CSRC季度与CAPCO2023半年度属于不同版本，不能直接拼接；当前原件可能含后修订，公布页面钟不自动认证历史首版。", "",
        "下一步只解析已收到原件，逐页保留行和证券代码，核重复/行业归属及原300成员覆盖；缺失分类UNKNOWN。全部历史和原案例用于开发解释，8快照不是完整2015—2026覆盖；金融NOT_COMPUTED、独立未建立。", "",
        "[冻结协议](protocol.json) · [逐次请求与原件哈希](summary.json)"]
    (OUT / "官方分类_公布钟与原件取得.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"有限来源取得结束：{pdf_count}/8原件；0金融，所有失败保留。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="官方行业分类的有限公布钟与PDF可行性。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
