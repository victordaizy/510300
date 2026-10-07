"""按已定位地址保存少量公开机制原文，不下载行情。"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2"
SOURCES = [
    {"id": "deposit_2023", "filename": "national_rmb_deposits_2023.htm", "url": "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111817163388840.htm", "author": "中国人民银行调查统计司", "document_date": "UNKNOWN_ORIGINAL_VINTAGE", "known_public_by": "TODAY_RETRIEVAL_ONLY", "date_evidence": "官方2023年度表当前附件，无逐月原始发布快照", "host_role": "官方年度表当前快照", "use": "提供2024年年初基数；仅回顾性分解"},
    {"id": "wealth_2024H1", "filename": "wealth_2024H1_original_media_copy.pdf", "url": "https://att.dahecube.com/f/240730/2c8d5fcda96e49f57648f5a66c4e951b", "author": "银行业理财登记托管中心", "document_date": "2024-07", "known_public_by": "2024-07-30", "date_evidence": "报告封面仅到月；2024-07-30中国金融信息网报道及媒体附件；未获得发行者站点秒级时间", "host_role": "媒体CDN保存的发行者原报告", "use": "2024年6月存量及半年募集总额，不能认定为存款净迁移；事后机制核对"},
    {"id": "pbc_2024Q3", "filename": "pbc_2024Q3_monetary_report.pdf", "url": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/5347949/afbfa5df25ee45889d916a2819b60a43/2024110815410752868.pdf", "author": "中国人民银行货币政策分析小组", "document_date": "2024-11-08", "known_public_by": "2024-11-08", "date_evidence": "报告封面及官方附件日期；保守按当日结束可用", "host_role": "官方原始文档", "use": "专栏4为11月发布的回顾，不能填入4月至9月当时输入"},
    {"id": "deposit_2024", "filename": "national_rmb_deposits_2024.htm", "url": "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111817274927346.htm", "author": "中国人民银行调查统计司", "document_date": "UNKNOWN_ORIGINAL_VINTAGE", "known_public_by": "TODAY_RETRIEVAL_ONLY", "date_evidence": "官方2024年度入口指向后迁移附件；没有逐月原发布快照", "host_role": "官方年度表当前快照", "use": "回顾性余额结构分解，不作历史决策输入"},
    {"id": "deposit_2025", "filename": "national_rmb_deposits_2025.htm", "url": "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2026/01/2026011516053141218.htm", "author": "中国人民银行调查统计司", "document_date": "UNKNOWN_ORIGINAL_VINTAGE", "known_public_by": "TODAY_RETRIEVAL_ONLY", "date_evidence": "官方2025年度入口当前附件；没有逐月原发布快照", "host_role": "官方年度表当前快照", "use": "回顾性余额结构分解，不作历史决策输入"},
    {"id": "wealth_release_date", "filename": "wealth_2024H1_release_date_news.html", "url": "https://www.cnfin.com/zs-lb/detail/20240730/4082559_1.html", "author": "中国金融信息网", "document_date": "2024-07-30", "known_public_by": "2024-07-30", "date_evidence": "报道页面日期", "host_role": "新闻报道", "use": "只辅助报告已公开日期，不用作定量数据来源"},
    {"id": "hand_interest_meeting", "filename": "pbc_hand_interest_meeting_official_repost.html", "url": "https://jrj.sh.gov.cn/ZXYW178/20240805/e7a428bfdf2a4c0bb3e05437a5998e3f.html", "author": "中国人民银行；上海市委金融办转载", "document_date": "2024-08-05", "known_public_by": "2024-08-05", "date_evidence": "转载页日期；报道8月1日会议，不将会议日当成此页发布时间", "host_role": "官方转载", "use": "规范手工补息已实施的同期政策证据，不量化解释各月M1"},
    {"id": "M1_revision", "filename": "pbc_m1_revision_official_repost.html", "url": "https://jrj.sh.gov.cn/SCDT197/20241203/e2d0d7d348ab4aaaa2fde9a976d3ae17.html", "author": "中国人民银行；上海市委金融办转载", "document_date": "2024-12-03", "known_public_by": "2024-12-03", "date_evidence": "官方转载页日期；新M1从2025年1月统计起", "host_role": "官方转载", "use": "M1构成变化，禁止把新M1-M0称为企业活期存款"},
]


def fetch(item: dict) -> dict:
    target = STUDY / "sources" / item["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    result = dict(item)
    if target.exists():
        result["status"] = "REUSED_SAVED"
    else:
        try:
            response = requests.get(item["url"], timeout=(30, 50), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if target.suffix.lower() == ".pdf" and not response.content.startswith(b"%PDF"):
                raise ValueError("响应不是PDF，未保存为有效源文档")
            target.write_bytes(response.content)
            result.update(status="FETCHED", retrieved_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), final_url=response.url, content_type=response.headers.get("Content-Type", ""))
        except Exception as exc:
            result.update(status="FAILED", error=str(exc))
            return result
    result.update(local_path=str(target.relative_to(STUDY)), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size)
    if target.suffix.lower() == ".pdf":
        with pdfplumber.open(target) as doc:
            pages = [{"pdf_page_1based": n + 1, "text": page.extract_text() or ""} for n, page in enumerate(doc.pages)]
        output = STUDY / "evidence" / f"{item['id']}_extracted_pages.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        result["pages"] = len(pages)
    return result


if __name__ == "__main__":
    receipt = STUDY / "sources/source_receipts.json"
    old = {row["id"]: row for row in json.loads(receipt.read_text(encoding="utf-8"))} if receipt.exists() else {}
    # 已两次超时的媒体CDN路线保持停止；在线原文另有阅读记录。
    active = [row for row in SOURCES if not (row["id"] in old and old[row["id"]].get("route_stopped_after_two_failures"))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(fetch, active))
    items += [old[row["id"]] for row in SOURCES if row not in active]
    for row in items:
        if row["status"] == "REUSED_SAVED" and row["id"] in old:
            row["retrieved_at"] = old[row["id"]].get("retrieved_at")
    receipt.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in items:
        print(row["id"], row["status"], row.get("bytes", ""), row.get("error", ""))
