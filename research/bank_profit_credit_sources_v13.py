"""保存固定三家银行在观察点以前公布的业绩材料。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile
from xml.etree import ElementTree as ET

import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_bank_profit_credit_bridge_v13"
DOCS = [
    {"name": "招商银行_2020业绩快报.pdf", "url": "https://static.cninfo.com.cn/finalpage/2021-01-15/1209106000.PDF", "known_at": "2021-01-15T23:59:59+08:00", "scope": "2020集团初步核算，未经审计"},
    {"name": "兴业银行_2020业绩快报.pdf", "url": "https://static.cninfo.com.cn/finalpage/2021-01-15/1209105758.PDF", "known_at": "2021-01-15T23:59:59+08:00", "scope": "官网2021年1月14日列示；巨潮公告按次日晚保守纳入，初步核算"},
    {"name": "平安银行_2020年报.pdf", "url": "https://static.cninfo.com.cn/finalpage/2021-02-02/1209224370.PDF", "known_at": "2021-02-02T23:59:59+08:00", "scope": "2020年度完整报告"},
    {"name": "招商银行_2020三季报.pdf", "url": "https://static.cninfo.com.cn/finalpage/2020-10-31/1208662780.PDF", "known_at": "2020-10-31T23:59:59+08:00", "scope": "集团与本行口径分别保留，不混用"},
    {"name": "兴业银行_2020三季报.pdf", "url": "https://static.cninfo.com.cn/finalpage/2020-10-30/1208651184.PDF", "known_at": "2020-10-30T23:59:59+08:00", "scope": "2020年前三季度合并口径"},
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fetch(spec):
    path = OUT / "sources" / spec["name"]
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    r = requests.get(spec["url"], timeout=(12, 50), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if path.suffix == ".pdf":
        assert r.content.startswith(b"%PDF")
    else:
        assert r.content.startswith(b"PK")
    path.write_bytes(r.content)
    rec = {**spec, "retrieved_at": datetime.now().astimezone().isoformat(), "sha256": sha256(r.content).hexdigest(), "bytes": len(r.content), "first_vintage_authenticated": False}
    save(receipt_path, rec)
    print(json.dumps({"已保存": spec["name"], "字节": len(r.content)}, ensure_ascii=False), flush=True)
    return rec


def extract(spec):
    path = OUT / "sources" / spec["name"]
    destination = path.with_suffix(".pages.json")
    if destination.exists():
        return
    if path.suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
    else:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = ["".join(p.itertext()) for p in root.findall(".//w:p", ns)]
        pages = [{"page": None, "text": "\n".join(paragraphs), "format": "DOCX正文及表格，未冒充分页"}]
    save(destination, pages)
    print(json.dumps({"已提取": spec["name"], "页或文档块": len(pages)}, ensure_ascii=False), flush=True)


def main():
    if (OUT / "source_receipts.json").exists():
        raise RuntimeError("本轮原文已保存，禁止重复下载。")
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(fetch, DOCS))
    for spec in DOCS:
        extract(spec)
    url = "https://mobile.cib.com.cn/netbank/cn/aboutCIB/investor/announcements/"
    r = requests.get(url, timeout=(12, 40))
    r.raise_for_status()
    path = OUT / "sources/兴业银行_官方公告目录.html"
    path.write_bytes(r.content)
    receipts.append({"name": path.name, "url": url, "sha256": sha256(r.content).hexdigest(), "retrieved_at": datetime.now().astimezone().isoformat(), "role": "定位2020年度快报与官网列示日期，不作为历史首版认证"})
    save(OUT / "source_receipts.json", receipts)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("五份固定公司原件及发布目录已保存。", flush=True)


if __name__ == "__main__":
    main()
