"""只扩展历史事前共识来源；固定日期检索，不读取510300收益。"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_money_consensus_source_extension_v1"
PREVIOUS = ROOT / "reports/research/510300_post_information_capital_adjustment_v1"
HOST = "https://research2.bualuang.co.th/upload/"
THAI_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
MONTHS = {v: k for k, v in enumerate(["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError("来源扩展目录已存在，禁止覆盖协议。")
    OUT.mkdir(parents=True)
    for folder in ("inputs", "raw", "receipts", "parsed", "results", "figures", "code"):
        (OUT / folder).mkdir()
    source = PREVIOUS / "inputs/official_releases.csv"
    shutil.copy2(source, OUT / "inputs/official_releases.csv")
    data = pd.read_csv(source)
    candidates = []
    for r in data.itertuples():
        release = pd.Timestamp(r.published_at).tz_localize(None).normalize()
        for offset in range(1, 6):
            day = release - pd.Timedelta(days=offset)
            filename = "Bls" + day.strftime("%y%m%d") + ".pdf"
            candidates.append({"stat_month": r.stat_month, "release_at": r.published_at, "offset_days": offset,
                               "report_date": day.strftime("%Y-%m-%d"), "filename": filename, "url": HOST + filename})
    pd.DataFrame(candidates).to_csv(OUT / "inputs/candidate_urls.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "study_id": "510300_MONEY_CONSENSUS_SOURCE_EXTENSION_V1", "frozen_at": now(),
        "parent": "510300_POST_INFORMATION_CAPITAL_ADJUSTMENT_V1", "role": "SOURCE_ADMISSION_ONLY_NO_RETURN_READ",
        "universe": "2018-01至2026-08原有104个月，旧/新M1口径严格分开",
        "source": "Bualuang Securities原站有日期日报；表中Survey字段，须同时含中国M1与M2",
        "source_selection": "按离央行公布日1至5个自然日前的固定日期次序；选择最近一个同时通过全部数字和日期核对的报告；不读取收益",
        "same_day_report": "EXCLUDED_WITHOUT_INTRADAY_PUBLICATION_PROOF",
        "prior_value_gate": "两项Prior必须与上一所属月的官方实际同比一致；缺少前月资料不放行，差异不擅自修正",
        "actual_column_gate": "两项Actual必须尚为空；不得将事后回述当事前预期",
        "report_date_gate": "封面泰文日期经佛历减543换算与文件名一致；报告日期早于央行实际发布日期",
        "period_gate": "表内所属月必须等于待配对央行月份；日历预计发布日期不替代央行实际发布时间",
        "history_limit": "今天回收的带日期原报告仍是历史重建，不建立不可变历史版本或独立前向证据",
        "sample_minimum_unchanged": "每一口径至少36训练加24评价；不因新增数值改变模型、收益窗口、费用或阈值",
        "funds_gate_unchanged": "本轮不以交易量或PCF篮子代替真实份额；没有资金历史公开证明，C/D仍不运行",
        "no_existing_result_overwrite": True, "market_returns_read": False, "models_run": 0, "accounts_run": 0,
        "pilot": "各年1月数据公布前一日，加已发现2024-06-12报告；按历年覆盖检查版式，不挑收益",
        "max_request_count": 521, "max_workers_http": 4, "pdf_processing": "主线程顺序解析，避免PDFium线程不安全",
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "freeze_receipt.json", {"frozen_at": now(), "official_sha256": digest(source),
         "protocol_sha256": digest(OUT / "protocol.json"), "candidates_sha256": digest(OUT / "inputs/candidate_urls.csv"),
         "known_seed": "Bls240612.pdf的搜索结果与原站原页已见，仅为来源线索；尚未计算其510300后续收益"})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("104个月、520个固定候选地址与准入规则已冻结。", flush=True)


def fetch(item: dict) -> dict:
    filename = item["filename"]
    receipt_file = OUT / "receipts" / (filename + ".json")
    if receipt_file.exists():
        old = json.loads(receipt_file.read_text(encoding="utf-8"))
        if old.get("fetch_status") != "NETWORK_ERROR" or old.get("transport") == "WINDOWS_CURL_SCHANNEL":
            return old
        shutil.copy2(receipt_file, receipt_file.with_suffix(".requests_failure.json"))
    rec = dict(item, retrieved_at=now())
    try:
        temporary = OUT / "raw" / (filename + ".download")
        command = ["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10",
                   "--max-time", "45", "--max-filesize", str(16*1024*1024),
                   "--output", str(temporary), "--write-out", "%{json}", item["url"]]
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=50)
        meta = json.loads(process.stdout) if process.stdout.strip() else {}
        body = temporary.read_bytes() if temporary.exists() else b""
        rec.update(transport="WINDOWS_CURL_SCHANNEL", status_code=meta.get("http_code"), content_type=meta.get("content_type"),
                   bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), resolved_url=meta.get("url_effective"),
                   transport_exit=process.returncode, transport_error=process.stderr[:300],
                   certificate_verification_disabled=False)
        if process.returncode == 0 and meta.get("http_code") == 200 and body.startswith(b"%PDF") and len(body) <= 16*1024*1024:
            (OUT / "raw" / filename).write_bytes(body)
            rec["fetch_status"] = "PDF_SAVED"
        else:
            rec["fetch_status"] = "NOT_ADMITTED_HTTP_OR_CONTENT"
        if temporary.exists():
            temporary.unlink()
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        rec.update(transport="WINDOWS_CURL_SCHANNEL", fetch_status="NETWORK_ERROR", error_type=type(exc).__name__, error=str(exc)[:300])
    save(receipt_file, rec)
    return rec


def parse(filename: str) -> dict:
    result_file = OUT / "parsed" / (filename + ".json")
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    result = {"filename": filename, "rows": [], "cover_dates": [], "calendar_pages": []}
    path = OUT / "raw" / filename
    if not path.exists():
        return dict(result, parse_status="NO_PDF")
    text_pages = []
    with pdfium.PdfDocument(path) as doc:
        for page in doc:
            text_page = page.get_textpage()
            text_pages.append(text_page.get_text_range())
            text_page.close()
            page.close()
    result["page_count"] = len(text_pages)
    compact_cover = re.sub(r"\s+", "", text_pages[0])
    for month, thai in enumerate(THAI_MONTHS, 1):
        for d, y in re.findall(r"(\d{1,2})" + thai + r"(25\d{2}|20\d{2})", compact_cover):
            year = int(y) - (543 if int(y) > 2400 else 0)
            if 1 <= int(d) <= 31:
                result["cover_dates"].append(f"{year:04}-{month:02}-{int(d):02}")
    result["cover_dates"] = sorted(set(result["cover_dates"]))
    page_ids = [i for i, t in enumerate(text_pages) if re.search(r"Money\s+Supply\s+M[12]", t, re.I)]
    with pdfplumber.open(path) as doc:
        result["pdf_metadata"] = {str(k): str(v) for k, v in doc.metadata.items()}
        for i in page_ids:
            page = doc.pages[i]
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            flat = re.sub(r"\s+", " ", text)
            if "China" not in text or not re.search(r"Survey\s+Actual\s+Prior", flat):
                result["calendar_pages"].append({"page": i + 1, "status": "HEADER_OR_COUNTRY_UNPROVEN"})
                continue
            result["calendar_pages"].append({"page": i + 1, "status": "SURVEY_ACTUAL_PRIOR_HEADER"})
            for line in text.splitlines():
                match = re.search(r"Money\s+Supply\s+M([12])\s+YoY\s+(\w{3})\s+(--|-?\d+(?:\.\d+)?%?)\s+(--|-?\d+(?:\.\d+)?%?)\s+(--|-?\d+(?:\.\d+)?%?)", line)
                if match:
                    series, period, survey, actual, prior = match.groups()
                    convert = lambda value: None if value == "--" else float(value.rstrip("%"))
                    result["rows"].append({"series": "M" + series, "period": period,
                        "forecast_pp": convert(survey), "actual_pp": convert(actual), "previous_pp": convert(prior),
                        "page": i+1, "numeric_row": match.group(0)})
    result["parse_status"] = "PARSED" if result["rows"] else "NO_PARSEABLE_M1_M2_ROWS"
    save(result_file, result)
    return result


def qualify(item: dict, parsed: dict, official: pd.DataFrame) -> dict:
    month = item["stat_month"]
    report_day = pd.Timestamp(item["report_date"])
    result = dict(item, status="NOT_ADMITTED", reasons=[])
    if parsed.get("parse_status") != "PARSED":
        result["reasons"].append(parsed.get("parse_status", "NO_PDF"))
    if item["report_date"] not in parsed["cover_dates"]:
        result["reasons"].append("COVER_DATE_NOT_VERIFIED")
    elif max(parsed["cover_dates"]) != item["report_date"]:
        result["reasons"].append("COVER_LATEST_DATE_CONFLICT")
    previous_month = str(pd.Period(month, freq="M") - 1)
    previous = official[official.stat_month == previous_month]
    if len(previous) != 1:
        result["reasons"].append("NO_PREVIOUS_OFFICIAL_MONTH")
    else:
        current = official[official.stat_month == month].iloc[0]
        if previous.iloc[0].training_regime != current.training_regime:
            result["reasons"].append("FORECAST_DEFINITION_AT_TRANSITION_UNPROVEN")
    pair = {}
    for row in parsed["rows"]:
        mm = MONTHS.get(row["period"])
        if mm is None:
            continue
        year = report_day.year - int(mm > report_day.month)
        if f"{year:04}-{mm:02}" != month:
            continue
        pair.setdefault(row["series"], []).append(row)
    for series, column in (("M1", "m1_yoy_pp"), ("M2", "m2_yoy_pp")):
        found = pair.get(series, [])
        if len(found) != 1:
            result["reasons"].append(series + "_MISSING_OR_AMBIGUOUS_PERIOD_ROW")
            continue
        row = found[0]
        result[series] = row
        if row["forecast_pp"] is None:
            result["reasons"].append(series + "_MISSING_SURVEY")
        if row["actual_pp"] is not None:
            result["reasons"].append(series + "_ACTUAL_ALREADY_REPORTED")
        if row["previous_pp"] is None or (len(previous) == 1 and not np.isclose(row["previous_pp"], previous.iloc[0][column], atol=1e-8)):
            result["reasons"].append(series + "_PRIOR_OFFICIAL_MISMATCH")
    if not result["reasons"]:
        result["status"] = "ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR"
        result["expected_spread_proxy_pp"] = result["M1"]["forecast_pp"] - result["M2"]["forecast_pp"]
        result["definition_version"] = official[official.stat_month == month].iloc[0].definition_version
    return result


def pilot() -> None:
    choices = pd.read_csv(OUT / "inputs/candidate_urls.csv")
    items = choices[(choices.stat_month.str.endswith("-01")) & choices.offset_days.eq(1)].to_dict("records")
    items.append({"stat_month": "2024-05", "release_at": "2024-06-14T17:01:00+08:00", "offset_days": 2,
                  "report_date": "2024-06-12", "filename": "Bls240612.pdf", "url": HOST + "Bls240612.pdf"})
    official = pd.read_csv(OUT / "inputs/official_releases.csv")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rec in pool.map(fetch, items):
            parsed = parse(rec["filename"])
            answer = qualify(rec, parsed, official)
            print(json.dumps({"月份": rec["stat_month"], "文件": rec["filename"], "下载": rec["fetch_status"],
                              "状态": answer["status"], "日期": parsed["cover_dates"], "数值": parsed["rows"], "原因": answer["reasons"]}, ensure_ascii=False), flush=True)


def collect() -> None:
    choices = pd.read_csv(OUT / "inputs/candidate_urls.csv")
    official = pd.read_csv(OUT / "inputs/official_releases.csv")
    admitted, audit = {}, []
    for offset in range(1, 6):
        selected = choices[choices.offset_days.eq(offset) & ~choices.stat_month.isin(admitted)]
        print(f"开始公布前第{offset}日来源检查：{len(selected)}个月。", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for count, rec in enumerate(pool.map(fetch, selected.to_dict("records")), 1):
                parsed = parse(rec["filename"])
                answer = qualify(rec, parsed, official)
                audit.append(answer)
                if answer["status"] == "ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR":
                    admitted[rec["stat_month"]] = answer
                if count % 10 == 0 or count == len(selected):
                    save(OUT / "progress.json", {"updated_at": now(), "offset": offset, "processed_this_offset": count,
                         "eligible_months": sorted(admitted), "markets_or_returns_read": False})
                    print(f"本层完成{count}/{len(selected)}，累计配对{len(admitted)}个月。", flush=True)
    save(OUT / "results/source_admission_audit.json", audit)
    save(OUT / "results/admitted_pairs.json", list(admitted.values()))
    simple = []
    for r in official.itertuples():
        a = admitted.get(r.stat_month)
        simple.append({"stat_month": r.stat_month, "definition_version": r.definition_version,
            "status": "ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR" if a else "NO_ADMITTED_PAIR_IN_FIXED_FIVE_DAY_SEARCH",
            "report_date": a["report_date"] if a else None, "source_url": a["url"] if a else None,
            "source_sha256": a["sha256"] if a else None,
            "M1_forecast_pp": a["M1"]["forecast_pp"] if a else None,
            "M2_forecast_pp": a["M2"]["forecast_pp"] if a else None,
            "M1_actual_pp": r.m1_yoy_pp, "M2_actual_pp": r.m2_yoy_pp,
            "published_at": r.published_at, "strict_forward_immutable": False})
    frame = pd.DataFrame(simple)
    frame.to_csv(OUT / "results/104个月新增来源准入.csv", index=False, encoding="utf-8-sig")
    save(OUT / "result.json", {"status": "SOURCE_SEARCH_COMPLETED_NO_RETURNS_READ", "completed_at": now(),
         "universe_months": len(frame), "admitted_pairs": len(admitted), "attempts": len(audit),
         "old_definition_pairs": sum(m < "2025-01" for m in admitted),
         "new_definition_pairs": sum(m >= "2025-01" for m in admitted),
         "remaining_missing_months": len(frame)-len(admitted), "market_return_reads": 0, "model_fits": 0, "accounts": 0})
    print((OUT / "result.json").read_text(encoding="utf-8")),


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="固定日期的Bualuang历史共识来源扩展")
    ap.add_argument("action", choices=["freeze", "pilot", "collect"])
    action = ap.parse_args().action
    {"freeze": freeze, "pilot": pilot, "collect": collect}[action]()
