"""检索微信公众号公开入口及其原始研报，记录可访问内容与资金字段。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup
import pypdfium2 as pdfium

from collect_bualuang_money_consensus_v1 import now, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_money_consensus_source_extension_v1/wechat"


def main() -> None:
    for folder in ("raw", "receipts", "extracts"):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    sources = [
        ("sogou_wechat_shares", "https://weixin.sogou.com/weixin?type=2&query=510300%20%E4%BB%BD%E9%A2%9D", "微信公众号检索入口"),
        ("etf_wanyi_20250612_syndication", "https://finance.sina.com.cn/roll/2025-06-12/doc-inezutkh1583924.shtml", "ETF万亿指数资金日报公开转载，待追原文"),
        ("htsc_etp_20201220", "https://crm.htsc.com.cn/doc/2020/10750401/b1a76cca-5e88-45c7-a48a-6f3592b829a4.pdf", "华泰证券原站量化周报，2020-12-20"),
        ("daily_etf_20190711", "https://pdf.dfcfw.com/pdf/H3_AP201907111338430984_1.pdf", "检索发现的2019-07-11原始ETF研究日报，待核对作者和字段"),
        ("daily_etf_20190306", "https://pdf.dfcfw.com/pdf/H3_AP201903061303140553_1.pdf", "检索发现的2019-03-06原始ETF研究日报，待核对作者和字段"),
        ("cs_20200319", "https://epaper.cs.com.cn/zgzqb/html/2020-03/19/nw.D110000zgzqb_20200319_3-A07.htm", "中国证券报2020-03-19原报资金表"),
    ]
    inventory = []
    for key, url, role in sources:
        receipt_path = OUT/"receipts"/(key+".json")
        if receipt_path.exists():
            inventory.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            continue
        path = OUT/"raw"/(key+".bin")
        rec = {"key":key, "url":url, "role":role, "retrieved_at":now()}
        process = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10", "--max-time", "35",
            "--output", str(path), "--write-out", "%{json}", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=40)
        meta = json.loads(process.stdout) if process.stdout.strip() else {}
        body = path.read_bytes() if path.exists() else b""
        rec.update(http_status=meta.get("http_code"), transport_exit=process.returncode, bytes=len(body),
                   sha256=hashlib.sha256(body).hexdigest(), resolved_url=meta.get("url_effective"), error=process.stderr[:300])
        if process.returncode == 0 and meta.get("http_code") == 200 and body.startswith(b"%PDF"):
            pdf_path = path.with_suffix(".pdf")
            path.rename(pdf_path)
            matches = []
            with pdfium.PdfDocument(pdf_path) as doc:
                rec["pages"] = len(doc)
                for page_id, page in enumerate(doc):
                    tp = page.get_textpage()
                    text = tp.get_text_range()
                    if page_id == 0:
                        rec["cover_excerpt_for_internal_qa"] = text[:1700]
                    for line in text.splitlines():
                        if "510300" in line or "T-1" in line or "份额" in line or "2019年" in line or "2019 年" in line:
                            matches.append({"page":page_id+1,"line":line})
                    tp.close()
                    page.close()
            save(OUT/"extracts"/(key+".json"), matches)
            rec["status"] = "ORIGINAL_REPORT_RETRIEVED_FIELDS_REQUIRE_ADMISSION"
        elif process.returncode == 0 and meta.get("http_code") == 200:
            soup = BeautifulSoup(body,"html.parser")
            rec["title"] = soup.title.get_text(" ",strip=True) if soup.title else ""
            rec["wechat_links"] = sorted({a.get("href") for a in soup.find_all("a",href=True) if "mp.weixin.qq.com" in a.get("href","")})
            rec["status"] = "HTML_RETRIEVED_NOT_YET_ADMITTED"
            rec["access_challenge_detected"] = any(t in soup.get_text() for t in ["访问过于频繁", "验证码", "请输入验证码", "用户您好，您的访问"])
        else:
            rec["status"] = "FETCH_INCOMPLETE_OR_HTTP_FAILURE"
        save(receipt_path,rec)
        inventory.append(rec)
        print(json.dumps({"来源":key,"状态":rec["status"],"HTTP":rec["http_status"],"字节":len(body)},ensure_ascii=False),flush=True)
    save(OUT/"inventory.json",inventory)


if __name__ == "__main__":
    main()
