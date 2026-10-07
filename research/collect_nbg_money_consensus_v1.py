"""固定日期补充2020年前后NBG原始周报的中国货币调查，不读取收益。"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pdfplumber
import pypdfium2 as pdfium

import collect_bualuang_money_consensus_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_money_consensus_source_extension_v1"
OUT = PARENT / "nbg"
HOST = "https://www.nbg.gr/-/jssmedia/Files/nbgportal/reports/migrated-data/files/greek/the-group/press-office/e-spot/reports/documents/"
MONTH_NAMES = "January February March April May June July August September October November December".split()


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError("NBG来源协议已存在，不覆盖。")
    for folder in ("inputs", "raw", "receipts", "parsed", "results", "figures", "code"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(PARENT / "inputs/official_releases.csv")
    chosen = official[official.stat_month.between("2019-12", "2020-12")]
    rows = []
    for r in chosen.itertuples():
        release = pd.Timestamp(r.published_at).tz_localize(None).normalize()
        for offset in range(1, 6):
            day = release - pd.Timedelta(days=offset)
            if day.weekday() != 1:
                continue
            for variant, fmt in (("short_year", "%d-%m-%y"), ("long_year", "%d-%m-%Y")):
                filename = "gmr_" + day.strftime(fmt) + ".pdf"
                rows.append({"stat_month": r.stat_month, "release_at": r.published_at, "offset_days": offset,
                    "report_date": str(day.date()), "variant": variant, "filename": filename, "url": HOST+filename})
    pd.DataFrame(rows).to_csv(OUT / "inputs/candidate_urls.csv", index=False, encoding="utf-8-sig")
    base.save(OUT / "protocol.json", {"study_id": "NBG_2020_PRERELEASE_CONSENSUS_COMPLETION_V1", "frozen_at": base.now(),
        "universe": "2019-12至2020-12全部13个所属月；没有公布前五日内周二报告的月份保留缺失",
        "selection": "实际公布前1至5日内的周二原站报告；短年份文件名优先，长年份仅兼容旧网站地址；仅补Bualuang及OCBC均缺少的月份",
        "admission": "封面日期正确且早于实际公布；中国栏M1/M2当月调查均存在、实际值栏空、前值与官方一致；S/A/P的来源定义必须在原页出现",
        "known_seed": "已看到gmr_10-03-20及gmr_12-05-20的表格线索。后者即使Actual仍空，日期晚于实际公布也不准入。尚未读取新增事件510300收益。",
        "return_clock_model_and_gates": "完整继承上一五日研究，不扩窗、不改模型和门槛；只作来源补齐",
        "market_return_reads": 0, "models_run": 0, "accounts_run": 0, "max_candidate_urls": len(rows)})
    base.save(OUT / "freeze_receipt.json", {"protocol_sha256": base.digest(OUT / "protocol.json"),
        "candidates_sha256": base.digest(OUT / "inputs/candidate_urls.csv"), "official_sha256": base.digest(PARENT / "inputs/official_releases.csv")})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(f"已冻结13个月、{len(rows)}个来源地址。", flush=True)


def parse(filename: str) -> dict:
    path = OUT / "raw" / filename
    output = OUT / "parsed" / (filename+".json")
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("parser_version") == 2:
            return cached
        shutil.copy2(output, output.with_suffix(".parser_v1.json"))
    result = {"filename": filename, "rows": [], "cover_dates": [], "calendar_pages": [], "parser_version": 2}
    if not path.exists():
        return dict(result, parse_status="NO_PDF")
    texts = []
    with pdfium.PdfDocument(path) as doc:
        for page in doc:
            tp = page.get_textpage()
            texts.append(tp.get_text_range())
            tp.close()
            page.close()
    result["page_count"] = len(texts)
    for month, day, year in re.findall(r"\b("+"|".join(MONTH_NAMES)+r")\s+(\d{1,2}),?\s+(20\d{2})\b", texts[0]):
        result["cover_dates"].append(f"{int(year):04}-{MONTH_NAMES.index(month)+1:02}-{int(day):02}")
    result["cover_dates"] = sorted(set(result["cover_dates"]))
    with pdfplumber.open(path) as doc:
        result["pdf_metadata"] = {str(k):str(v) for k,v in doc.metadata.items()}
        for i, original in enumerate(texts):
            if not re.search(r"Money\s+Supply\s+M[12]", original):
                continue
            page = doc.pages[i]
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if not all(s in text for s in ["CHINA", "Consensus", "Actual Outcome", "Previous Outcome"]):
                result["calendar_pages"].append({"page": i+1, "status": "COUNTRY_OR_SAP_DEFINITION_UNPROVEN"})
                continue
            result["calendar_pages"].append({"page": i+1, "status": "S_SURVEY_A_ACTUAL_P_PREVIOUS_HEADER"})
            # 三栏日历并未占满纸张；按Money字样的真实横坐标分栏，不能直接三等分纸宽。
            words = page.extract_words()
            starts = sorted({round(w["x0"], 0) for w in words if w["text"] == "Money"})
            for col, start in enumerate(starts):
                left, right = max(0, start-2), min(page.width, start+page.width/3)
                part = page.crop((left, 0, right, page.height)).extract_text(x_tolerance=2, y_tolerance=2) or ""
                if not any(w["text"] == "CHINA" and abs(w["x0"]-start) < 5 for w in words):
                    continue
                for line in part.splitlines():
                    match = re.search(r"Money\s+Supply\s+M([12])\s+\(YoY\)\s+(\w+)\s+(\.\.|--|-?\d+(?:[.,]\d+)?%?)\s+(\.\.|--|-?\d+(?:[.,]\d+)?%?)\s+(\.\.|--|-?\d+(?:[.,]\d+)?%?)", line)
                    if match:
                        series, period, survey, actual, prior = match.groups()
                        convert = lambda s: None if s in ("..", "--") else float(s.rstrip("%").replace(",", "."))
                        result["rows"].append({"series":"M"+series, "period":period[:3], "forecast_pp":convert(survey),
                            "actual_pp":convert(actual), "previous_pp":convert(prior), "page":i+1, "calendar_column":col+1,
                            "numeric_row":match.group(0)})
    result["parse_status"] = "PARSED" if result["rows"] else "NO_PARSEABLE_M1_M2_ROWS"
    base.save(output,result)
    return result


def collect() -> None:
    base.OUT = OUT
    official = pd.read_csv(PARENT / "inputs/official_releases.csv")
    choices = pd.read_csv(OUT / "inputs/candidate_urls.csv")
    admitted, audit = {}, []
    for variant in ("short_year", "long_year"):
        selected = choices[choices.variant.eq(variant) & ~choices.stat_month.isin(admitted)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for rec in pool.map(base.fetch, selected.to_dict("records")):
                parsed = parse(rec["filename"])
                answer = base.qualify(rec, parsed, official)
                answer["source_family"] = "NBG"
                audit.append(answer)
                if answer["status"] == "ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR":
                    admitted[rec["stat_month"]] = answer
                print(f"{rec['stat_month']} {rec['filename']}：{answer['status']}，{answer['reasons']}", flush=True)
    base.save(OUT / "results/source_admission_audit.json", audit)
    base.save(OUT / "results/admitted_pairs.json", list(admitted.values()))
    base.save(OUT / "result.json", {"status":"SOURCE_SEARCH_COMPLETED_NO_RETURNS_READ", "completed_at":base.now(),
        "universe_months":13, "admitted_pairs":len(admitted), "attempts":len(audit), "market_return_reads":0, "models_run":0, "accounts_run":0})
    print(f"来源核查完成，{len(admitted)}个月通过。", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NBG原始周报的固定日期补充")
    p.add_argument("action", choices=["freeze", "collect"])
    action = p.parse_args().action
    freeze() if action == "freeze" else collect()
