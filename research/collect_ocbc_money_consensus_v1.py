"""补充2018—2020年原站日报共识；只按日期及口径准入，不读取市场收益。"""
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
OUT = PARENT / "ocbc"
MONTH_NAMES = "January February March April May June July August September October November December".split()


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError("OCBC来源协议已存在，不覆盖。")
    for folder in ("inputs", "raw", "receipts", "parsed", "results", "figures", "code"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(PARENT / "inputs/official_releases.csv")
    selected = official[official.stat_month.between("2018-01", "2020-12")]
    rows = []
    for r in selected.itertuples():
        release = pd.Timestamp(r.published_at).tz_localize(None).normalize()
        for offset in range(1, 6):
            day = release - pd.Timedelta(days=offset)
            for variant in ("lower", "title"):
                folder = "daily%20treasury%20outlook" if variant == "lower" else "Daily%20Treasury%20Outlook"
                prefix = "dto" if variant == "lower" else "DTO"
                rows.append({"stat_month": r.stat_month, "release_at": r.published_at,
                    "offset_days": offset, "report_date": day.strftime("%Y-%m-%d"), "variant": variant,
                    "filename": "ocbc_" + day.strftime("%Y%m%d") + "_" + variant + ".pdf",
                    "url": f"https://www.ocbc.com/assets/pdf/{folder}/{day.year}/{prefix}%20{day:%d%m%Y}.pdf"})
    pd.DataFrame(rows).to_csv(OUT / "inputs/candidate_urls.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "study_id": "OCBC_2018_2020_PRERELEASE_CONSENSUS_COMPLETION_V1", "frozen_at": base.now(),
        "role": "SOURCE_ADMISSION_ONLY_NO_RETURN_READ", "universe": "2018-01至2020-12全部36个月，不按收益选择月份",
        "source": "OCBC原站Daily Treasury Outlook的Bloomberg调查表；CH为中国",
        "date_search": "央行实际公布日前1至5自然日，由近至远；同日先小写地址，再标题大小写地址，仅是地址兼容；最近合格文件唯一入选",
        "admission": "沿用Bualuang数字门：封面日期匹配、M1/M2同所属月、Survey均有数字、Actual均空、Prior与前月央行原值一致、不得跨M1口径；另要求每条数字行的国家代码为CH",
        "source_priority_before_outcomes": "同月Bualuang合格值优先；OCBC仅补其缺月。保留重复月份两份来源的差异，不按误差或收益择优。此前WGC六例冻结不覆盖。",
        "history_boundary": "历史重建；当前回收原文件不等于不可变历史存档或严格前向。两项Survey之差只是剪刀差预期代理，不是直接调查的差值中位数。",
        "models_and_accounts": "不运行；下游仍沿用每口径36训练加至少24评价，不改变五日收益窗口、费用、模型和增量门",
        "known_seed": "已看过2018-07-10、2019-02-14原日报数字及2020目录线索；未读取新增月份510300收益",
        "max_candidate_urls": len(rows), "no_market_return_reads": True,
    }
    base.save(OUT / "protocol.json", protocol)
    base.save(OUT / "freeze_receipt.json", {"frozen_at": base.now(), "protocol_sha256": base.digest(OUT / "protocol.json"),
        "candidates_sha256": base.digest(OUT / "inputs/candidate_urls.csv"),
        "official_sha256": base.digest(PARENT / "inputs/official_releases.csv")})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(f"已冻结36个月、{len(rows)}个原站候选地址；不读取收益。", flush=True)


def parse(filename: str) -> dict:
    path = OUT / "raw" / filename
    output = OUT / "parsed" / (filename + ".json")
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
    # 原日报首页有明确报告日期；先保留全部完整日期，交给统一日期冲突门判断。
    for day, month, year in re.findall(r"\b(\d{1,2})\s+(" + "|".join(MONTH_NAMES) + r")\s+(20\d{2})\b", texts[0]):
        result["cover_dates"].append(f"{int(year):04}-{MONTH_NAMES.index(month)+1:02}-{int(day):02}")
    for month, day, year in re.findall(r"\b(" + "|".join(MONTH_NAMES) + r")\s+(\d{1,2}),?\s+(20\d{2})\b", texts[0]):
        result["cover_dates"].append(f"{int(year):04}-{MONTH_NAMES.index(month)+1:02}-{int(day):02}")
    result["cover_dates"] = sorted(set(result["cover_dates"]))
    with pdfplumber.open(path) as doc:
        result["pdf_metadata"] = {str(k): str(v) for k, v in doc.metadata.items()}
        for i, original in enumerate(texts):
            if not re.search(r"Money\s+Supply\s+M[12]", original):
                continue
            text = doc.pages[i].extract_text(x_tolerance=2, y_tolerance=3) or ""
            flat = re.sub(r"\s+", " ", text)
            if not re.search(r"Survey\s+Actual\s+Prior", flat):
                result["calendar_pages"].append({"page": i+1, "status": "HEADER_UNPROVEN"})
                continue
            result["calendar_pages"].append({"page": i+1, "status": "SURVEY_ACTUAL_PRIOR_HEADER"})
            for line in text.splitlines():
                match = re.search(r"\bCH\s+Money\s+Supply\s+M([12])\s+YoY\s+(\w{3})\s+(--|-?\d+(?:\.\d+)?%?)\s+(--|-?\d+(?:\.\d+)?%?)\s+(--|-?\d+(?:\.\d+)?%?)", line)
                if match:
                    series, period, forecast, actual, prior = match.groups()
                    convert = lambda s: None if s == "--" else float(s.rstrip("%"))
                    result["rows"].append({"series": "M"+series, "period": period, "forecast_pp": convert(forecast),
                        "actual_pp": convert(actual), "previous_pp": convert(prior), "page": i+1,
                        "country_code": "CH", "numeric_row": match.group(0)})
    result["parse_status"] = "PARSED" if result["rows"] else "NO_PARSEABLE_M1_M2_ROWS"
    base.save(output, result)
    return result


def collect(pilot: bool = False) -> None:
    base.OUT = OUT
    choices = pd.read_csv(OUT / "inputs/candidate_urls.csv")
    if pilot:
        choices = choices[(choices.report_date.isin(["2018-07-10", "2019-02-14", "2020-02-19"]))]
    official = pd.read_csv(PARENT / "inputs/official_releases.csv")
    admitted, audit = {}, []
    for offset in range(1, 6):
        for variant in ("lower", "title"):
            selected = choices[choices.offset_days.eq(offset) & choices.variant.eq(variant) & ~choices.stat_month.isin(admitted)]
            if selected.empty:
                continue
            print(f"检查公布前{offset}日、{variant}地址：{len(selected)}个月。", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for rec in pool.map(base.fetch, selected.to_dict("records")):
                    parsed = parse(rec["filename"])
                    answer = base.qualify(rec, parsed, official)
                    answer["source_family"] = "OCBC"
                    audit.append(answer)
                    if answer["status"] == "ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR":
                        admitted[rec["stat_month"]] = answer
                    if pilot:
                        print(json.dumps(answer, ensure_ascii=False), flush=True)
            base.save(OUT / "progress.json", {"updated_at": base.now(), "offset": offset, "variant": variant,
                "admitted_months": sorted(admitted), "no_market_return_reads": True})
            print(f"累计{len(admitted)}个月通过；已检查{len(audit)}个候选。", flush=True)
    suffix = "pilot_" if pilot else ""
    base.save(OUT / "results" / (suffix + "source_admission_audit.json"), audit)
    base.save(OUT / "results" / (suffix + "admitted_pairs.json"), list(admitted.values()))
    base.save(OUT / (suffix + "result.json"), {"status": "SOURCE_SEARCH_COMPLETED_NO_RETURNS_READ", "completed_at": base.now(),
        "admitted_pairs": len(admitted), "attempts": len(audit), "model_fits": 0, "accounts": 0, "market_return_reads": 0})
    print(f"来源检查完成，准入{len(admitted)}个月。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原站OCBC事前共识补齐")
    parser.add_argument("action", choices=["freeze", "pilot", "collect"])
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    else:
        collect(pilot=action == "pilot")
