"""提取固定半年度报告的逐页文本，保存关键词所在页，供人工核对。"""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pdfplumber
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_external_bridge_v16"
TERMS = ["主要会计数据", "主要财务数据", "主要财务指标", "境外", "国外", "外销", "汇兑", "外币", "汇率风险", "利率风险", "敏感性分析", "净利息收益率", "净息差", "净投资收益率", "总投资收益率", "新业务价值", "终止筹划", "复牌"]


def main():
    receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))
    extracts, hits = [], []
    for r in receipts:
        if not r["status"].startswith("SAVED"):
            continue
        src = OUT / "sources" / r["name"]
        assert sha256(src.read_bytes()).hexdigest() == r["sha256"]
        dest = src.with_suffix(".pages.json")
        if dest.exists():
            pages = json.loads(dest.read_text(encoding="utf-8"))
        else:
            with pdfplumber.open(src) as reader:
                pages = []
                for i, page in enumerate(reader.pages):
                    pages.append({"page": i + 1, "text": page.extract_text() or ""})
                    page.close()
            dest.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
            src.with_suffix(".txt").write_text("\n\n".join(f"第{r['page']}页\n{r['text']}" for r in pages), encoding="utf-8")
        render = OUT / "figures" / (src.stem + "_第1页.png")
        if not render.exists():
            pdf = pdfium.PdfDocument(src)
            page = pdf[0]
            page.render(scale=1.2).to_pil().save(render)
            page.close()
            pdf.close()
        for page in pages:
            compact = "".join(page["text"].split())
            found = [term for term in TERMS if term in compact]
            if found:
                hits.append({"document": src.name, "page": page["page"], "terms": found})
        extracts.append({"document": src.name, "source_sha256": r["sha256"], "pages": len(pages),
                         "text_sha256": sha256(dest.read_bytes()).hexdigest(), "published_date_from_directory": r["published_date"]})
        print(f"已提取：{src.name}，共{len(pages)}页。", flush=True)
    for name, value in [("document_extract_receipt.json", extracts), ("results/公司报告_关键词页码.json", hits)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)


if __name__ == "__main__":
    main()
