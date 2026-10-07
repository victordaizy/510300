"""保存三例历史信息增量诊断的公开来源与49条报告的先行披露候选。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_disclosure_novelty_v1"

DOCUMENTS = [
    ("1219861327", "牧原2024一季度报告", "https://static.cninfo.com.cn/finalpage/2024-04-27/1219861327.PDF", "2024-04-27"),
    ("1220602782", "牧原2024半年业绩预告", "https://static.cninfo.com.cn/finalpage/2024-07-11/1220602782.PDF", "2024-07-11"),
    ("1220542009", "牧原2024六月经营简报", "https://static.cninfo.com.cn/finalpage/2024-07-06/1220542009.PDF", "2024-07-06"),
    ("1224501021", "小商品城2025半年报告", "https://static.cninfo.com.cn/finalpage/2025-08-18/1224501021.PDF", "2025-08-18"),
    ("1214442231", "上机数控2022半年报告", "https://static.cninfo.com.cn/finalpage/2022-08-30/1214442231.PDF", "2022-08-30"),
]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def title_period(title):
    text = re.sub(r"\s+", "", title)
    years = re.findall(r"(?:19|20)\d{2}", text)
    if len(set(years)) != 1:
        return None
    year = years[0]
    if any(term in text for term in ["前三季度", "前三季", "第三季度", "三季度", "1-9月", "1至9月", "1—9月"]):
        return f"{year}-09-30"
    if any(term in text for term in ["半年度", "半年", "上半年", "中期", "1-6月", "1至6月"]):
        return f"{year}-06-30"
    if any(term in text for term in ["第一季度", "一季度", "一季", "1-3月", "1至3月"]):
        return f"{year}-03-31"
    if "季度" in text or "月" in text:
        return None
    if "年" in text and any(term in text for term in ["业绩预", "业绩快报"]):
        return f"{year}-12-31"
    return None


def local_screen():
    sample = pd.read_parquet(ROOT / "reports/research/510300_historical_cash_quality_causes_v1/全部49条公司财务变化.parquet")
    metadata = pd.read_parquet(ROOT / "data/raw/cninfo/a_share_hs_first_preliminary_earnings_official_metadata_a_share_v1.parquet")
    metadata = metadata[metadata.ts_code.isin(sample.ts_code)].copy()
    metadata["candidate_report_period"] = metadata.announcement_title.map(title_period)
    metadata["official_pdf_date"] = pd.to_datetime(metadata.official_pdf_date)
    rows, candidates = [], []
    for s in sample.itertuples():
        prior = metadata[(metadata.ts_code == s.ts_code)
                         & metadata.candidate_report_period.eq(pd.Timestamp(s.report_period).strftime("%Y-%m-%d"))
                         & metadata.official_pdf_date.lt(pd.Timestamp(s.event_publication_date))]
        prior = prior.sort_values(["official_pdf_date", "announcement_id"])
        rows.append({"target_announcement_id": str(s.announcement_id), "ts_code": s.ts_code,
                     "report_period": pd.Timestamp(s.report_period).strftime("%Y-%m-%d"),
                     "target_nominal_date": pd.Timestamp(s.event_publication_date).strftime("%Y-%m-%d"),
                     "company_joint_positive": bool(s.company_joint_positive),
                     "prior_candidate_count": len(prior),
                     "prior_candidate_ids": "|".join(prior.announcement_id.astype(str)),
                     "status": "PRIOR_TITLE_CANDIDATES_FOUND_CONTENT_REVIEW_REQUIRED" if len(prior) else "NO_MATCH_IN_THIS_LOCAL_INDEX_NOT_PROOF_OF_ABSENCE"})
        for r in prior.itertuples():
            candidates.append({"target_announcement_id": str(s.announcement_id), "ts_code": s.ts_code,
                               "target_report_period": pd.Timestamp(s.report_period).strftime("%Y-%m-%d"),
                               "prior_announcement_id": str(r.announcement_id), "title": r.announcement_title,
                               "nominal_publication_date": r.official_pdf_date.strftime("%Y-%m-%d"),
                               "official_pdf_url": r.official_pdf_url,
                               "calendar_days_before_target": int((pd.Timestamp(s.event_publication_date) - r.official_pdf_date).days)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "49条报告的先行披露候选.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(OUT / "先行披露候选原文索引.csv", index=False, encoding="utf-8-sig")
    save_json(OUT / "metadata_screen_summary.json", {
        "reports": len(summary), "reports_with_prior_title_candidates": int(summary.prior_candidate_count.gt(0).sum()),
        "company_joint_positive_reports": int(summary.company_joint_positive.sum()),
        "company_joint_positive_with_prior_title_candidates": int((summary.company_joint_positive & summary.prior_candidate_count.gt(0)).sum()),
        "candidate_links": len(candidates), "content_review_complete": False,
        "title_screen_is_consensus_or_surprise_measure": False,
        "source": "data/raw/cninfo/a_share_hs_first_preliminary_earnings_official_metadata_a_share_v1.parquet"})
    print("先行披露标题候选筛选完成：", len(summary), "条报告，其中", int(summary.prior_candidate_count.gt(0).sum()), "条存在候选。")


def sources():
    destination = OUT / "sources"
    destination.mkdir(exist_ok=True)
    receipt_path = destination / "receipts.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else []
    receipts = {r["id"]: r for r in previous}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    for identity, name, url, publication in DOCUMENTS:
        path = destination / f"{identity}.pdf"
        if not path.exists():
            response = session.get(url, timeout=45)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"返回内容不是PDF：{identity}")
            path.write_bytes(response.content)
        text_path = destination / f"{identity}.pages.json"
        if not text_path.exists():
            document = pdfium.PdfDocument(str(path))
            pages = []
            for index in range(len(document)):
                page = document[index]
                text_page = page.get_textpage()
                pages.append({"page": index + 1, "text": text_page.get_text_range()})
                text_page.close()
                page.close()
            document.close()
            save_json(text_path, pages)
            (destination / f"{identity}.txt").write_text("\n\n".join(f"第{p['page']}页\n{p['text']}" for p in pages), encoding="utf-8")
        else:
            pages = json.loads(text_path.read_text(encoding="utf-8"))
        if identity not in receipts:
            receipts[identity] = {"id": identity, "name": name, "url": url, "nominal_publication_date": publication,
                                  "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                  "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": len(pages),
                                  "historical_first_version_proven": False}
        save_json(receipt_path, list(receipts.values()))
        print(name, "已保存", len(pages), "页。")


if __name__ == "__main__":
    local_screen()
    sources()
