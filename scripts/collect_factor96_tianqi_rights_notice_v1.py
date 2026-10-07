"""补取已存目录中唯一缺失的天齐2019事前发行公告，保留126文档基础结果。"""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pypdfium2 as pdfium
import requests
from research.factor96_rights_event_chains_v1 import OUT, SOURCE, read, save, digest, now


def main():
    target_id = "1207160457"
    directory = OUT / "tianqi_notice_supplement_v1"
    source_path = SOURCE / "本阶段未选中的配股标题.csv"
    candidates = list(csv.DictReader(source_path.open(encoding="utf-8-sig")))
    selected = [r for r in candidates if r["document_id"] == target_id]
    assert len(selected) == 1
    target = selected[0]
    assert target["title"] == "关于配股发行的公告" and target["symbol"] == "002466.SZ"
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    assert mandate["new_market_data_collection_enabled"] and not mandate["orders_authorized"]
    save(directory / "protocol_addendum.json", {"at": now(), "source_only": True,
         "reason": "基础126文档已识别36发行身份，天齐2019缺事前发行公告；在原有未选目录中定位到关于配股发行的公告，因标题含发行的而非发行公告未入旧角色筛选。",
         "fixed_target": target, "catalogue_sha256": digest(source_path),
         "transport": "仅此公开PDF一次请求，连接10秒、读取30秒、40MiB上限；失败保存状态，不自动改地址或扩大检索。",
         "preserve_base_result": "result.json及基础126文档事实不改写；补充证据和有效结果另存。",
         "new_accounts": 0, "new_returns": 0, "orders_authorized": False, "delivery_package_required": False})
    receipt = {"document_id": target_id, "requested_at": now(), "url": target["source_url"]}
    raw = directory / f"{target_id}.pdf"
    try:
        with requests.get(target["source_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 30), stream=True) as response:
            receipt.update(http_status=response.status_code, final_url=response.url,
                           content_type=response.headers.get("Content-Type"))
            total = 0
            with raw.open("xb") as handle:
                for chunk in response.iter_content(262144):
                    total += len(chunk)
                    if total > 40*1024*1024:
                        raise ValueError("来源文件超过固定上限")
                    handle.write(chunk)
            receipt["status"] = "HTTP_OK" if response.status_code == 200 else "HTTP_ERROR"
    except (requests.RequestException, ValueError) as error:
        receipt.update(status="REQUEST_FAILED", error_type=type(error).__name__)
    receipt["completed_at"] = now()
    if raw.exists():
        receipt.update(bytes=raw.stat().st_size, sha256=digest(raw), raw_path=str(raw))
    save(directory / "request_receipt.json", receipt)
    if receipt["status"] != "HTTP_OK" or raw.read_bytes()[:5] != b"%PDF-":
        print(json.dumps(receipt, ensure_ascii=False))
        return
    pages = []
    document = pdfium.PdfDocument(str(raw))
    try:
        for i in range(len(document)):
            page = document[i]
            text_page = page.get_textpage()
            try:
                pages.append({"page": i+1, "text": text_page.get_text_range()})
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    text_path = directory / f"{target_id}.json"
    save(text_path, {"document_id": target_id, "pages": pages})
    save(directory / "document.json", {**target, "status": "PDF_TEXT_SAVED", "review_role": "ISSUANCE_NOTICE",
         "raw_snapshot": str(raw), "text_snapshot": str(text_path), "raw_sha256": digest(raw),
         "text_sha256": digest(text_path), "pages": len(pages), "bytes": raw.stat().st_size})
    print("唯一目标原文已保存：", target_id, "页数", len(pages), "字节", raw.stat().st_size)


if __name__ == "__main__":
    main()
