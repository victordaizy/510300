"""定位标题与作者栏之间的明确原文日期，不生成收益、仓位或新预测事实。"""
from pathlib import Path
from datetime import date
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, norm
from research.forward_eps_guosen_history_v1 import identity

SOURCE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
FACTS = ROOT / "reports/research/510300_forward_eps_soochow_facts_v4"
OUT = ROOT / "reports/research/510300_forward_eps_soochow_header_date_diagnostic_v1"


def observed_header_dates(text: str) -> list[dict]:
    text = norm(text)
    author = re.search(r"(?:首席|高级|资深)?证券分析师|(?m:^\s*分析师)", text)
    if author is None:
        return []
    prefix = text[:author.start()]
    if len(prefix) > 2000:
        return []
    matches = list(re.finditer(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", prefix))
    records = []
    for m in matches:
        tail = prefix[m.end():]
        if tail.strip():
            continue
        beginning = prefix.rfind("\n", 0, m.start()) + 1
        records.append({"original_line": prefix[beginning:].strip(), "date_text": m.group(),
                        "date": date(*map(int, m.groups())).isoformat(),
                        "characters_before_author": len(prefix), "date_starts_at": m.start(),
                        "title_precedes_date_on_same_line": bool(prefix[beginning:m.start()].strip())})
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / "protocol.json", {"registered_at": now(), "all_v4_date_failures_examined": True,
        "date_must_be_last_content_before_first_analyst_heading": True, "header_prefix_max_characters": 2000,
        "existing_original_date_only_no_added_values": True, "new_forecast_facts": 0,
        "files": [identity(Path(__file__)), identity(FACTS / "result.json"), identity(FACTS / "document_outcomes.json")]}, exclusive=True)
    failures = [r for r in read(FACTS / "document_outcomes.json")["rows"]
                if r.get("latest_share_field_reason") == "原件落款日与作者栏关系不唯一"]
    rows = []
    for failed in failures:
        rid = failed["report_id"]
        page_path = SOURCE / "page_texts" / (rid + ".json")
        record = read(SOURCE / "document_records" / (rid + ".json"))
        observed = observed_header_dates(read(page_path)["pages"][0])
        row = {"report_id": rid, "ts_code": failed["ts_code"], "matched_dates": observed,
               "unique_anchored_date": len(observed) == 1,
               "source_page_text": identity(page_path), "raw_pdf_path": record["source"]["raw_pdf_path"],
               "pdf_sha256": record["source"]["pdf_sha256"], "provider_metadata": record["provider_metadata"]}
        if len(observed) == 1:
            row.update(observed[0])
        rows.append(row)
    matched = [r for r in rows if r["unique_anchored_date"]]
    save(OUT / "header_date_evidence.json", {"rows": rows}, exclusive=True)
    pd.DataFrame([{k: v for k, v in row.items() if k not in ["matched_dates", "source_page_text", "provider_metadata"]}
                  for row in rows]).to_csv(OUT / "原件标题与作者栏日期定位.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "EXISTING_HEADER_DATE_LAYOUT_DIAGNOSTIC_COMPLETE",
              "examined_v4_date_failures": len(failures), "unique_date_immediately_before_first_analyst": len(matched),
              "title_and_date_same_extracted_line": sum(r["title_precedes_date_on_same_line"] for r in matched),
              "not_resolved_by_this_rule": len(rows) - len(matched), "new_forecast_facts": 0,
              "new_models_fit": 0, "new_account_evaluations": 0, "new_downloads": 0,
              "does_not_alone_admit_reports": True, "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    print(result, flush=True)


if __name__ == "__main__":
    main()
