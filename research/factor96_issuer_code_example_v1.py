"""补存目录代码差异的一份交易所原公告，只作身份线索附录。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_issuance_repurchase_sources_v1/issuer_code_example"
URL = "https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-02-07/e2d8b9d3-879b-4587-9e1e-235ebf4c8a44.PDF"


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def now():
    return datetime.now().astimezone().isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "protocol.json").exists()
    save("protocol.json", {"at": now(), "scope": "SOURCE_IDENTITY_EXAMPLE_AFTER_CATALOGUE_RESULTS",
        "selection": "修正目录159份其他证券代码文档中145份是查询302132而返回300114；针对这一明确代码差异检索交易所原公告。",
        "discovery": "已通过网页工具检索并阅读第三次提示性公告；这不是未知结果前的预测预注册。",
        "url": URL, "requests_limit": 1, "boundary": "该文告知计划启用日，不把它写成已核验实施日，不修改冻结目录或成员表。",
        "new_accounts": 0, "orders_authorized": False})
    (OUT / "collector_code.py").write_bytes(Path(__file__).read_bytes())
    receipt = {"requested_at": now(), "url": URL}
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 30))
    raw = response.content
    (OUT / "source.pdf").write_bytes(raw)
    receipt.update(completed_at=now(), http_status=response.status_code, bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(), final_url=response.url)
    save("receipt.json", receipt)
    assert response.status_code == 200 and raw.startswith(b"%PDF")
    pages = []
    document = pdfium.PdfDocument(OUT / "source.pdf")
    for number in range(len(document)):
        page = document[number]
        text = page.get_textpage()
        pages.append({"page": number + 1, "text": text.get_text_range()})
        text.close()
        if number in [0, 1]:
            bitmap = page.render(scale=1.4)
            bitmap.to_pil().save(OUT / f"page{number + 1}.png")
            bitmap.close()
        page.close()
    document.close()
    save("pages.json", pages)
    joined = re.sub(r"\s+", "", "".join(p["text"] for p in pages))
    assert "300114" in joined and "302132" in joined and "2025年2月17日" in joined
    assert "第三次提示性公告" in joined and "2025年2月7日" in joined
    current = pd.read_parquet(ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1/unique_documents.parquet")
    selected = current[(current.symbol == "302132.SZ") & (current.reported_security_codes == "300114")]
    assert len(selected) == 145
    selected.to_csv(OUT / "涉及历史代码的目录文档.csv", index=False, encoding="utf-8-sig")
    save("result.json", {"at": now(), "status": "ANNOUNCED_CODE_CHANGE_SCHEDULE_SOURCE_SAVED",
        "announcement_number": "2025-027", "pdf_signature_date": "2025-02-07",
        "former_code": "300114", "announced_new_code": "302132", "announced_activation_date": "2025-02-17",
        "reference_documents": len(selected), "implementation_confirmed_by_this_notice": False,
        "historical_first_publication_verified": False, "frozen_catalogue_modified": False,
        "membership_modified": False, "new_accounts": 0, "new_returns": 0, "network_requests": 1,
        "scope": "当前官方PDF支持当时公告安排；完整实施确认与其他代码有效期仍待后续原文。"})
    print("已保存代码变更第三次提示性公告；145份历史代码目录只附身份线索，不改写冻结数据。", flush=True)


if __name__ == "__main__":
    main()
