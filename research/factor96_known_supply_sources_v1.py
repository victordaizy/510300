"""T13来源阶段：固定历史并集，取得解禁原文和更正，不读取收益标签。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reports/research/510300_factor96_known_supply_sources_v1"
STUDY = "510300_FACTOR96_KNOWN_SUPPLY_SOURCES_V1"
TYPES = ["ELIGIBLE_ORIGINAL_DISCLOSURE", "REVISION_TITLE_RETAINED_SEPARATELY"]
STOP = threading.Event()


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def select_targets(catalogue, membership):
    dates = pd.to_datetime(membership.membership_date)
    historical = membership[dates.between("2015-01-01", "2025-12-31")]
    symbols = set(historical.symbol)
    a = catalogue.copy()
    a["symbol"] = a.secCode.astype(str).str.zfill(6).map(lambda c: c+(".SH" if c.startswith("6") else ".SZ"))
    a["archive_date"] = pd.to_datetime(a.announcement_date)
    keep = a.symbol.isin(symbols) & a.archive_date.between("2015-01-01", "2025-12-31") & a.title_kind.isin(TYPES)
    selected = a.loc[keep].sort_values(["archive_date", "symbol", "announcementId"]).reset_index(drop=True)
    selected["document_id"] = selected.announcementId.astype(str)
    selected["source_url"] = "https://static.cninfo.com.cn/"+selected.adjunctUrl
    assert selected.document_id.is_unique
    assert selected.source_url.str.startswith("https://static.cninfo.com.cn/finalpage/").all()
    return selected


def freeze():
    assert not (OUT/"protocol.json").exists()
    for name in ["inputs", "source_evidence", "code", "raw", "text", "receipts", "documents", "program_before"]:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    sources = {
        "inputs/announcement_catalogue.parquet": "data/curated/510300_unlock_announcement_daily_source_v1/all_announcement_records.parquet",
        "inputs/membership.parquet": "reports/research/510300_factor96_crowding_overlay_v1/inputs/membership.parquet",
        "source_evidence/unlock_source_admission.json": "reports/research/510300_unlock_announcement_source_admission_v1/source_admission.json",
        "source_evidence/unlock_source_consolidation.json": "reports/research/510300_unlock_announcement_source_admission_v1/daily_source_consolidation.json",
        "source_evidence/old_four_pdf_cases.json": "reports/research/510300_unlock_announcement_source_admission_v1/primary_case_adjudication.json",
        "source_evidence/old_unlock_protocol.json": "config/510300_unlock_announcement_increment_v1.json",
        "source_evidence/old_unlock_result.json": "reports/research/510300_unlock_announcement_increment_v1/result.json",
        "source_evidence/old_unlock_report.md": "reports/research/510300_unlock_announcement_increment_v1/研究结论与下一步.md",
        "source_evidence/old_repurchase_feature_protocol.json": "reports/research/510300_corporate_repurchase_disclosed_demand_v1/protocol.json",
        "source_evidence/old_repurchase_feature_result.json": "reports/research/510300_corporate_repurchase_disclosed_demand_v1/result.json",
        "source_evidence/old_repurchase_model_protocol.json": "reports/research/510300_funding_repurchase_demand_daily_v1/protocol.json",
        "source_evidence/old_repurchase_model_result.json": "reports/research/510300_funding_repurchase_demand_daily_v1/result.json",
        "source_evidence/old_repurchase_maturity_protocol.json": "reports/research/510300_repurchase_fixed_maturity_account_v1/protocol.json",
        "source_evidence/old_repurchase_maturity_result.json": "reports/research/510300_repurchase_fixed_maturity_account_v1/result.json",
        "source_evidence/current_mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    }
    copied = []
    for target, original in sources.items():
        shutil.copy2(ROOT/original, OUT/target)
        copied.append({"original": original, "snapshot": target, "sha256": digest(OUT/target)})
    for path in (ROOT/"reports/research/510300_factor96_program_v1").iterdir():
        if path.is_file():
            shutil.copy2(path, OUT/"program_before"/path.name)
    a = pd.read_parquet(OUT/"inputs/announcement_catalogue.parquet")
    members = pd.read_parquet(OUT/"inputs/membership.parquet")
    selected = select_targets(a, members)
    selected.to_parquet(OUT/"targets.parquet", index=False)
    # 原目录响应逐项留存，原文下载阶段不重新搜索目录。
    provenance = []
    for item in selected[["raw_response_path", "raw_response_sha256"]].drop_duplicates().itertuples(index=False):
        original = ROOT/item.raw_response_path
        assert digest(original) == item.raw_response_sha256
        target = OUT/"catalogue_raw"/item.raw_response_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        provenance.append({"original": item.raw_response_path, "snapshot": target.relative_to(OUT).as_posix(), "sha256": item.raw_response_sha256})
    save(OUT/"source_evidence/catalogue_provenance.json", provenance, True)
    known = read(OUT/"source_evidence/old_four_pdf_cases.json")["facts"]
    reusable = []
    for item in known:
        assert digest(ROOT/item["source_pdf"]) == item["source_pdf_sha256"]
        target = OUT/"source_evidence/previous_pdf_cases"/(item["announcementId"]+".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/item["source_pdf"], target)
        if item["announcementId"] in set(selected.document_id):
            reusable.append(item["announcementId"])
    daily_path = ROOT/"data/raw/constituents/000300_constituent_daily.parquet"
    cols = ["date", "outstanding_share", "total_market_cap_cny", "market_cap_asof_date"]
    daily = pd.read_parquet(daily_path, columns=cols)
    save(OUT/"source_evidence/T12_normalization_gap.json", {"at": now(), "source": daily_path.relative_to(ROOT).as_posix(),
        "source_sha256": digest(daily_path), "rows": len(daily), "nonmissing": daily[cols[1:]].notna().sum().to_dict(),
        "date_start": str(pd.to_datetime(daily.date).min().date()), "date_end": str(pd.to_datetime(daily.date).max().date()),
        "T12_status": "NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE",
        "reason": "已有回购证据按成交额归一化，不能替代M01事件前自由流通市值；M02用途、终止及減额公告也未完整形成原定义台账。",
        "old_result_boundary": "旧回购近邻模型与固定五日期限账户均已失败，不改称T12成功或用新阈值救援。"}, True)
    protocol = {"at": now(), "study_id": STUDY, "phase": "SOURCE_AND_EVENT_FIELDS_ONLY_BEFORE_NEW_OUTCOMES",
        "question": "从旧公告密度推进到原文明示的上市流通日期、数量与更正版本，为T13已知供给窗口建立可复核事件源。",
        "selection": "2015至2025年每日点时沪深300成员的历史并集；原目录内全部合格原公告和全部更正候选，不按金额、日期后收益或公告后成败筛选。并集只决定下载上集，不代表各时点都能用这些股票。",
        "period": ["2015-01-01", "2025-12-31"], "target_documents": len(selected), "companies": selected.symbol.nunique(),
        "title_counts": selected.title_kind.value_counts().to_dict(), "reusable_targets": reusable,
        "transport": "3并发，每次请求后等待0.3秒；仅传输异常或5xx最多2次；403或429后停止新请求；失败和无文本不能填零。",
        "source": "复用已保存完整检索目录中的巨潮公开原PDF地址；保存原目录响应、原PDF、请求时钟和逐页文本。",
        "parse": "PDFium仅在主线程提取；日期和数量另设固定提取合同，保存每项上下文和原单位；未解析不意味着无解禁。",
        "versions": "更正、补充和取消标题保留为版本候选；不得提前用后来修订覆盖旧公告。辅助意见不直接计为发行人新供给。",
        "scope_boundary": "旧上市流通检索不能证明覆盖所有发行、缴款和增发配股日历，M06仍须另补；本轮不宣称完整T13数据已经准入。",
        "T12": "自由流通市值及用途/终止版本门保留；不拿成交额归一化旧模型替代原定义。",
        "source_clock_boundary": "原档案日期不是历史首次HTTP送达证明；经济上市日、PDF署期、档案日期、下载时钟分开保存。",
        "new_returns_loaded": False, "new_models": 0, "new_accounts": 0, "independent_forward_observations": 0,
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False,
        "sources": copied, "targets_sha256": digest(OUT/"targets.parquet"), "code_sha256": digest(Path(__file__))}
    shutil.copy2(Path(__file__), OUT/"code"/Path(__file__).name)
    save(OUT/"protocol.json", protocol, True)
    save(OUT/"freeze.json", {"at": now(), "before_new_documents_and_outcomes": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts]}, True)
    print(f"T13原文范围已固定：{len(selected)}份、{selected.symbol.nunique()}家公司，{len(reusable)}份目标原文可复用。", flush=True)


def request_pdf(item):
    row = dict(item)
    key = row["document_id"]
    record = {"document_id": key, "symbol": row["symbol"], "title": row["clean_title"], "title_kind": row["title_kind"],
              "archive_date": str(pd.Timestamp(row["archive_date"]).date()), "source_url": row["source_url"], "reused": False}
    for attempt in [1, 2]:
        receipt_path = OUT/"receipts"/f"{key}_{attempt}.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
        elif STOP.is_set():
            return {**record, "status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT"}
        else:
            receipt = {"at": now(), "source_url": row["source_url"], "document_id": key, "attempt": attempt}
            try:
                response = requests.get(row["source_url"], headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(10, 25))
                is_pdf = response.content.startswith(b"%PDF")
                raw = OUT/"raw"/f"{key}_{attempt}{'.pdf' if is_pdf else '.bin'}"
                with raw.open("xb") as stream:
                    stream.write(response.content)
                receipt.update(http_status=response.status_code, bytes=len(response.content), raw_path=raw.relative_to(OUT).as_posix(),
                               sha256=digest(raw), status="PDF_DOWNLOADED" if response.status_code == 200 and is_pdf else "NO_USABLE_PDF")
                if response.status_code in [403, 429]:
                    STOP.set()
            except requests.RequestException as exc:
                receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            receipt["completed_at"] = now()
            save(receipt_path, receipt, True)
            time.sleep(.3)
        record.update(status=receipt["status"], raw_path=receipt.get("raw_path"), raw_sha256=receipt.get("sha256"),
                      receipt_path=receipt_path.relative_to(OUT).as_posix(), attempts=attempt)
        if receipt["status"] != "REQUEST_FAILED" and not 500 <= receipt.get("http_status", 0) <= 599:
            break
    return record


def extract(record):
    if record["status"] != "PDF_DOWNLOADED":
        return record
    raw = OUT/record["raw_path"]
    assert digest(raw) == record["raw_sha256"]
    try:
        pdf = pdfium.PdfDocument(raw)
        pages = []
        for page in pdf:
            text = page.get_textpage()
            pages.append(text.get_text_range())
            text.close()
            page.close()
        pdf.close()
        path = OUT/"text"/(record["document_id"]+".json")
        save(path, pages, True)
        record.update(status="PDF_TEXT_SAVED" if len("".join(pages).strip()) >= 50 else "PDF_NO_USABLE_TEXT",
                      text_path=path.relative_to(OUT).as_posix(), text_sha256=digest(path), pages=len(pages),
                      text_characters=len("".join(pages)), first_page_security_code_seen=record["symbol"][:6] in (pages[0] if pages else ""))
    except (pdfium.PdfiumError, ValueError) as exc:
        record.update(status="PDF_TEXT_EXTRACTION_FAILED", error_type=type(exc).__name__)
    return record


def run():
    protocol = read(OUT/"protocol.json")
    assert digest(Path(__file__)) == protocol["code_sha256"]
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(OUT/item["path"]) == item["sha256"], item["path"]
    save(OUT/"run_started.json", {"at": now(), "freeze_sha256": digest(OUT/"freeze.json")}, True)
    targets = pd.read_parquet(OUT/"targets.parquet").to_dict("records")
    reusable = {r["announcementId"]: r for r in read(OUT/"source_evidence/old_four_pdf_cases.json")["facts"]}
    results, pending = [], []
    for row in targets:
        if row["document_id"] in reusable:
            old = reusable[row["document_id"]]
            record = {"document_id": row["document_id"], "symbol": row["symbol"], "title": row["clean_title"],
                "title_kind": row["title_kind"], "archive_date": str(pd.Timestamp(row["archive_date"]).date()),
                "source_url": row["source_url"], "reused": True, "status": "PDF_DOWNLOADED",
                "raw_path": "source_evidence/previous_pdf_cases/"+row["document_id"]+".pdf", "raw_sha256": old["source_pdf_sha256"],
                "prior_receipt": "source_evidence/old_four_pdf_cases.json"}
            record = extract(record)
            save(OUT/"documents"/(row["document_id"]+".json"), record, True)
            results.append(record)
        else:
            pending.append(row)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(request_pdf, row) for row in pending]
        for future in as_completed(futures):
            record = extract(future.result())
            save(OUT/"documents"/(record["document_id"]+".json"), record, True)
            results.append(record)
            if len(results) % 50 == 0:
                print(f"T13已处理{len(results)}/{len(targets)}份，取得文本{sum(r['status']=='PDF_TEXT_SAVED' for r in results)}份。", flush=True)
    results.sort(key=lambda r: (r["archive_date"], r["symbol"], r["document_id"]))
    save(OUT/"documents.json", results, True)
    result = {"at": now(), "study_id": STUDY, "status": "ORIGINALS_COMPLETE_EVENT_FIELDS_PENDING" if all(r["status"] == "PDF_TEXT_SAVED" for r in results) else "PARTIAL_ORIGINALS_EVENT_FIELDS_PENDING",
              "documents": len(results), "complete_texts": sum(r["status"] == "PDF_TEXT_SAVED" for r in results),
              "reused_documents": sum(r["reused"] for r in results), "companies": len({r["symbol"] for r in results}),
              "statuses": pd.Series([r["status"] for r in results]).value_counts().to_dict(),
              "title_counts": pd.Series([r["title_kind"] for r in results]).value_counts().to_dict(),
              "first_page_code_not_confirmed": sum(r["status"] == "PDF_TEXT_SAVED" and not r["first_page_security_code_seen"] for r in results),
              "T13_performance": "NOT_RUN", "T12_performance": "NOT_RUN_SOURCE_GATE", "new_accounts": 0,
              "new_models": 0, "new_return_labels": 0, "independent_forward_observations": 0,
              "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "external_review": "NOT_PERFORMED"}
    save(OUT/"result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
