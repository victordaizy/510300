"""针对财务测量暴露的七个具体字段，保存六份原公告证据；不改原冻结结果。"""
import argparse
from pathlib import Path
import json
import shutil

import pandas as pd
import pypdfium2 as pdfium
import requests

from research.factor96_earnings_cashflow_measurement_v1 import OUT, ROOT, clean, digest, now, save


FIELDS = [
    ("1214959878", "TOTAL_ASSETS_END"),
    ("1216653495", "TOTAL_ASSETS_END"),
    ("1219823220", "TOTAL_ASSETS_END"),
    ("1219906975", "PARENT_NET_PROFIT_YTD"),
    ("1223412328", "PARENT_NET_PROFIT_YTD"),
    ("1223412328", "OPERATING_CASH_FLOW_YTD"),
    ("1211423261", "OPERATING_CASH_FLOW_YTD"),
]
EXTRA = OUT / "anomaly_evidence"


def prepare():
    assert not EXTRA.exists()
    EXTRA.mkdir()
    facts = pd.read_parquet(OUT / "verified_facts.parquet")
    keys = pd.DataFrame(FIELDS, columns=["announcement_id", "metric_id"])
    selected = facts.merge(keys, on=["announcement_id", "metric_id"], how="inner", validate="one_to_one")
    assert len(selected) == 7 and selected.official_pdf_url.nunique() == 6
    selected.to_parquet(EXTRA / "flagged_fields_before_source_review.parquet", index=False)
    protocol = {
        "at": now(), "phase": "POST_MEASUREMENT_SOURCE_CONTRADICTION_CHECK", "selection_is_post_measurement": True,
        "trigger": "已冻结测量出现资产40元、百万元疑似漏换算、利润2元及CFO4元/0.7068元；极端L02/L04追溯到这些具体字段。未读价格或收益。",
        "targets": selected[["announcement_id", "ts_code", "metric_id", "verified_value", "source_raw_value", "source_unit", "source_page", "official_pdf_url", "official_pdf_sha256"]].to_dict("records"),
        "budget": "六个已明确官方PDF URL各一次请求，无搜索扩展、无失败自动重试；保留原响应、当前哈希与档案哈希比较及整份提取文本。",
        "purpose": "确认是否单位/页内行列/脚注误读；不按数值大小自动删除或修正，不修改先前冻结输入、代码和测量输出。",
        "effect": "确认错误后补充失效声明及依赖影响清单，旧的可计算数量只能称机械计算覆盖；本轮不基于有矛盾档案运行T11账户。",
        "boundary": "当前PDF哈希相同只能证明与保存提取档案同一文档，不证明历史首次发布或完全没有其余错误。",
        "new_accounts": 0, "strategy_returns_read": False, "orders_authorized": False,
    }
    save(EXTRA / "protocol_addendum.json", protocol)
    shutil.copy2(Path(__file__), EXTRA / Path(__file__).name)
    save(EXTRA / "request_freeze.json", {"at": now(), "measurement_freeze_sha256": digest(OUT / "freeze.json"),
        "measurement_result_sha256": digest(OUT / "result.json"),
        "files": [{"path": p.relative_to(EXTRA).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted(EXTRA.iterdir()) if p.is_file()]})
    print("补充核实计划已冻结：七个字段、六份原公告；原测量结果保留。", flush=True)


def collect():
    assert not (EXTRA / "collection_started.json").exists()
    freeze = json.loads((EXTRA / "request_freeze.json").read_text(encoding="utf-8"))
    for row in freeze["files"]:
        assert digest(EXTRA / row["path"]) == row["sha256"]
    assert digest(Path(__file__)) == digest(EXTRA / Path(__file__).name)
    save(EXTRA / "collection_started.json", {"at": now(), "request_freeze_sha256": digest(EXTRA / "request_freeze.json")})
    selected = pd.read_parquet(EXTRA / "flagged_fields_before_source_review.parquet")
    records = []
    for name in ["pdf", "text", "receipts", "visual"]:
        (EXTRA / name).mkdir()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/pdf"})
    for announcement, block in selected.groupby("announcement_id", sort=True):
        url = block.official_pdf_url.iloc[0]
        started = now()
        response = session.get(url, timeout=(15, 35))
        payload = response.content
        is_pdf = response.status_code == 200 and payload.startswith(b"%PDF")
        path = EXTRA / "pdf" / (announcement + (".pdf" if is_pdf else ".response.bin"))
        path.write_bytes(payload)
        receipt = {"started_at": started, "finished_at": now(), "requested_url": url, "final_url": response.url,
            "status_code": response.status_code, "headers": dict(response.headers), "bytes": len(payload), "sha256": digest(path),
            "redirect_history": [{"url": r.url, "status_code": r.status_code} for r in response.history],
            "pdf_valid_header": is_pdf, "archived_sha256": block.official_pdf_sha256.iloc[0],
            "matches_archived_extraction_pdf": digest(path) == block.official_pdf_sha256.iloc[0]}
        save(EXTRA / "receipts" / (announcement + ".json"), receipt)
        assert is_pdf, announcement
        document = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(len(document)):
            page = document[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            pages.append({"page": i + 1, "text": text})
            textpage.close()
            page.close()
        save(EXTRA / "text" / (announcement + ".json"), {"announcement_id": announcement, "pdf_sha256": digest(path), "pages": pages})
        document.close()
        records.append({"announcement_id": announcement, "url": url, "pdf": path.relative_to(OUT).as_posix(),
            "bytes": len(payload), "sha256": digest(path), "page_count": len(pages),
            "matches_archived_extraction_pdf": receipt["matches_archived_extraction_pdf"]})
        print(f"原公告{announcement}已保存，{len(pages)}页，与档案哈希一致={receipt['matches_archived_extraction_pdf']}。", flush=True)
    save(EXTRA / "documents.json", records)
    save(EXTRA / "collection_receipt.json", {"at": now(), "logical_requests": len(records),
        "http_responses_including_redirects": sum(1 + len(json.loads((EXTRA / "receipts" / (r["announcement_id"] + ".json")).read_text(encoding="utf-8"))["redirect_history"]) for r in records),
        "pdf_documents": len(records), "all_match_archived_extraction_pdf": all(r["matches_archived_extraction_pdf"] for r in records),
        "new_accounts": 0, "strategy_returns_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "collect"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else collect()
