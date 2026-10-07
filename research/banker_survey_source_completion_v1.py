"""补全一次传输失败和一份PDF重叠字形，不覆盖首次来源探查。"""
from pathlib import Path
import re
import sys
from urllib.parse import urljoin

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.banker_survey_source_probe_v1 as prior
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_banker_survey_source_completion_v1"
STUDY = "510300_BANKER_SURVEY_SOURCE_COMPLETION_V1"


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("季度来源补全已有终态。")
    (OUT / "texts").mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "previous_result": read(prior.OUT / "result.json"),
        "2020Q4": "原PDF包含同位置相同字形重复绘制；用pdfplumber在1点坐标容差内去重同字形后提取，禁止按字符相邻删除数字。原页已经渲染核对。",
        "2021Q3": "首次SSL传输EOF，允许一次相同公开URL普通重试，原失败保留；不关闭TLS验证、不绕过权限。",
        "new_strategy_returns": 0, "new_accounts": 0}, True)
    records = read(prior.OUT / "saved_originals.json")
    failures = []
    for item in read(prior.OUT / "unresolved_sources.json"):
        q = item["quarter"]
        try:
            if q == "2020Q4":
                html = read(prior.OUT / "requests/2020Q4_page.json")
                pdf = read(prior.OUT / "requests/2020Q4_pdf.json")
            elif q == "2021Q3":
                html = prior.fetch(item["source_url"], "2021Q3_page_transport_retry")
                page = prior.soup_for(html)
                links = {urljoin(item["source_url"], a["href"]) for a in page.find_all("a", href=True)
                         if re.search(r"\.pdf(?:$|[?#])", a["href"], flags=re.I)}
                assert len(links) == 1
                pdf = prior.fetch(next(iter(links)), "2021Q3_pdf")
            else:
                raise ValueError("未授权增加其他补全分支。")
            page = prior.soup_for(html)
            clocks = re.findall(r"文章来源[：:]?\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", page.get_text(" ", strip=True))
            assert len(clocks) == 1 and pdf["status"] == "HTTP_OK"
            with pdfplumber.open(ROOT / pdf["raw_path"]) as doc:
                texts = [p.dedupe_chars(tolerance=1, extra_attrs=()).extract_text() or "" for p in doc.pages]
            compact = re.sub(r"\s+", "", "".join(texts))
            assert item["title"] in compact
            path = OUT / "texts" / (q + ".json")
            save(path, {"quarter": q, "pages": texts, "pdf_sha256": pdf["sha256"], "dedupe_identical_glyph_tolerance_pt": 1}, True)
            record = {k: v for k, v in item.items() if k not in ["status", "error"]}
            record.update(status="ORIGINAL_PDF_AND_CLOCK_SAVED_FIELDS_NOT_ADMITTED", published_at=clocks[0] + "+08:00",
                          conservative_known_at=clocks[0][:10] + "T23:59:59+08:00", page_raw_path=html["raw_path"], page_sha256=html["sha256"],
                          pdf_url=pdf["url"], pdf_raw_path=pdf["raw_path"], pdf_sha256=pdf["sha256"], page_count=len(texts),
                          text_path=path.relative_to(ROOT).as_posix(), historical_first_vintage_verified=False)
            records.append(record)
        except (AssertionError, ValueError, RuntimeError) as exc:
            failures.append({"quarter": q, "error": str(exc)})
    records.sort(key=lambda r: r["quarter"])
    assert len({r["quarter"] for r in records}) == len(records)
    save(OUT / "saved_originals.json", records, True)
    save(OUT / "unresolved_sources.json", failures, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "32_ORIGINAL_QUARTER_REPORTS_READY" if len(records) == 32 and not failures else "PARTIAL_QUARTER_SOURCE_COMPLETION",
        "saved_quarters": len(records), "unresolved_quarters": len(failures), "first_quarter": records[0]["quarter"],
        "last_quarter": records[-1]["quarter"], "original_pdf_unchanged": True, "new_strategy_returns": 0,
        "new_accounts": 0, "historical_first_vintage_verified": False, "goal_achieved": False}, True)
    (OUT / "source_completion_code.py").write_bytes(Path(__file__).read_bytes())
    print(f"银行家季度来源补全后{len(records)}份，未解决{len(failures)}份。", flush=True)


if __name__ == "__main__":
    run()
