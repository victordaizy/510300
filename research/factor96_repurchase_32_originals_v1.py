"""按32个已辨认的公司与批准日期补原方案，复用有限查询器，不读取策略收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pypdfium2 as pdfium

from research import factor96_repurchase_missing_originals_v1 as source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_32_originals_v1"
PRIOR = ROOT / "reports/research/510300_factor96_repurchase_remaining_changes_v1"
DIRECTORY = ROOT / "reports/research/510300_corporate_repurchase_public_completion_v1/cninfo_stock_directory.json"
source.OUT = OUT
read, save, digest, now = source.read, source.save, source.digest, source.now


def prepare():
    assert not (OUT / "protocol.json").exists(), "已经固定的查询范围不覆盖"
    for name in ("raw", "receipts", "queries", "text", "documents"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    targets = read(PRIOR / "remaining_original_targets.json")
    directory = {r["code"]: r for r in read(DIRECTORY)["stockList"] if r["category"] == "A股"}
    assert directory["600837"]["zwjc"] == "海通证券"
    previous_paths = [ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1/results/catalogues.json",
                      ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1/catalogues.json"]
    previous = [row for path in previous_paths for row in read(path)]
    windows = []
    for target in targets:
        symbol = target["issuer"]
        predecessor = symbol == "海通证券股份有限公司"
        if predecessor:
            symbol = "600837.SH"
        assert symbol.endswith((".SH", ".SZ")) and symbol[:6] in directory
        day = datetime.fromisoformat(target["reported_approval_date"])
        assert not any(c["window"]["symbol"] == symbol and c["window"]["start"] <= day.date().isoformat() <= c["window"]["end"]
                       for c in previous), "命中旧查询窗口，须复用结果而非重复采集"
        windows.append({
            "key": symbol + "_" + day.strftime("%Y%m%d"), "symbol": symbol,
            "original_approval_date": day.date().isoformat(), "change_document_ids": target["change_document_ids"],
            "org_id": directory[symbol[:6]]["orgId"], "column": "sse" if symbol.endswith(".SH") else "szse",
            "start": (day - timedelta(days=1)).date().isoformat(), "end": (day + timedelta(days=7)).date().isoformat(),
            "issuer_mapping": "PREDECESSOR_ISSUER_DIRECTORY_IDENTITY" if predecessor else "SAME_ISSUER",
            "current_holder_symbol": "601211.SH" if predecessor else symbol,
        })
    assert len(windows) == len({r["key"] for r in windows}) == 32
    protocol = {
        "at": now(), "study_id": "510300_FACTOR96_REPURCHASE_32_ORIGINALS_V1", "windows": windows,
        "previous_turn_classification": "PROGRESS_31_NOTICE_CLAUSES_AND_THREE_ORIGINS_DATA_CREDENTIAL_PENDING",
        "scope": "全部32个明确缺件对象，按原公司和批准日前1日至后7日查巨潮回购公告；不按收益删减对象。",
        "query_rule": "每页30条最多3页，沿用已验证查询器；传输失败最多补1次；403或429后停止新请求。",
        "selection_rule": "沿用标题筛选：回购且方案/报告书/回购股份公告，排除明确限制性股票、法律意见、独董、监事会、提议和持股明细；取得后逐文核对。",
        "duplicate_rule": "重叠日期窗口可返回同一公告，只保存一次原文；保留所有目录关系，不能把报告书和预案重复算成两份方案。",
        "identity_rule": "原海通证券600837与承接方601211分别保存，公司代码及目录标识还须与取得的原文核对。",
        "clock_rule": "原文名义公开日期按日末代理，今天取得时间单列；未证明历史首次发布版本，不回填旧冻结交易研究。",
        "analysis_fields": ["original_approval_date", "original_purpose", "budget_floor", "budget_ceiling", "duration", "approval_state", "plan_revision_identity"],
        "free_float": "CURRENT_SERVICE_CREDENTIAL_EXPIRED_NO_NEW_PROBE", "new_accounts": 0, "goal_achieved": False,
        "delivery_package_required": False, "orders_authorized": False,
    }
    save(OUT / "protocol.json", protocol)
    dependencies = [Path(__file__), Path(source.__file__), DIRECTORY, PRIOR / "remaining_original_targets.json",
                    PRIOR / "review_cards.json", PRIOR / "combined_change_ledger.json", OUT / "protocol.json", *previous_paths]
    save(OUT / "source_freeze.json", {"at": now(), "files": [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in dependencies]})
    print("已固定32个原方案窗口，包含原海通证券的两个窗口；没有读取新策略收益。", flush=True)


def collect():
    assert not (OUT / "source_result.json").exists(), "已经结束的来源查询不重复运行"
    for item in read(OUT / "source_freeze.json")["files"]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    protocol = read(OUT / "protocol.json")
    with ThreadPoolExecutor(max_workers=3) as pool:
        catalogues = list(pool.map(source.query, protocol["windows"]))
    if not (OUT / "catalogues.json").exists():
        save(OUT / "catalogues.json", catalogues)
    selected = {}
    for catalogue in catalogues:
        for row in catalogue["rows"]:
            if not source.selected(row["title"]):
                continue
            key = row["document_id"]
            if key not in selected:
                selected[key] = {**row, "window_keys": []}
            else:
                for field in ("symbol", "source_url", "catalogue_date", "title"):
                    assert row[field] == selected[key][field]
            selected[key]["window_keys"].append(catalogue["window"]["key"])
    candidates = sorted(selected.values(), key=lambda r: (r["catalogue_date"], r["symbol"], r["document_id"]))
    if not (OUT / "selected_pdf_targets.json").exists():
        save(OUT / "selected_pdf_targets.json", candidates)
    acquired = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(source.acquire, "pdf_" + row["document_id"], row["source_url"]): row for row in candidates}
        for future in as_completed(futures):
            acquired[futures[future]["document_id"]] = future.result()
    documents = []
    for row in candidates:
        destination = OUT / "documents" / (row["document_id"] + ".json")
        if destination.exists():
            documents.append(read(destination))
            continue
        receipt, receipt_path = acquired[row["document_id"]]
        record = {**row, "receipt_path": receipt_path, "status": receipt["status"],
                  "raw_path": receipt.get("raw_path"), "raw_sha256": receipt.get("sha256"),
                  "known_at": row["catalogue_date"] + "T23:59:59+08:00", "text_path": None,
                  "historical_first_publication_verified": False, "trading_feature_admitted": False}
        if receipt["status"] == "HTTP_OK" and (OUT / receipt["raw_path"]).read_bytes().startswith(b"%PDF"):
            pdf = pdfium.PdfDocument(OUT / receipt["raw_path"])
            pages = []
            try:
                for page in pdf:
                    text_page = page.get_textpage()
                    try:
                        pages.append(text_page.get_text_range())
                    finally:
                        text_page.close()
                        page.close()
            finally:
                pdf.close()
            target = OUT / "text" / (row["document_id"] + ".json")
            if not target.exists():
                save(target, pages)
            else:
                assert read(target) == pages
            record.update(status="PDF_TEXT_SAVED" if len("".join(pages).strip()) >= 50 else "PDF_NO_USABLE_TEXT",
                          text_path=target.relative_to(OUT).as_posix(), text_sha256=digest(target), pages=len(pages),
                          code_in_first_page=row["symbol"][:6] in pages[0])
        elif receipt["status"] == "HTTP_OK":
            record["status"] = "HTTP_OK_NOT_PDF"
        save(destination, record)
        documents.append(record)
        print(f"原文{row['document_id']}：{record['status']}，{record.get('pages', 0)}页。", flush=True)
    save(OUT / "documents.json", documents)
    receipts = [read(p) for p in sorted((OUT / "receipts").glob("*.json"))]
    result = {
        "at": now(), "query_windows": len(catalogues),
        "complete_query_windows": sum(r["status"] == "COMPLETE_QUERY" for r in catalogues),
        "catalogue_occurrences": sum(len(r["rows"]) for r in catalogues),
        "selected_pdf_documents": len(documents), "complete_pdf_texts": sum(r["status"] == "PDF_TEXT_SAVED" for r in documents),
        "new_http_requests": sum("requested_at" in r for r in receipts),
        "confirmed_originals": "PENDING_CLAUSE_REVIEW", "new_accounts": 0,
        "goal_achieved": False, "delivery_package_created": False,
    }
    save(OUT / "source_result.json", result)
    print(f"32个窗口查询结束，取得{result['complete_pdf_texts']}份原文；尚须核对原方案身份。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="32个具体原方案缺口的有限来源查询")
    parser.add_argument("action", choices=["prepare", "collect"])
    {"prepare": prepare, "collect": collect}[parser.parse_args().action]()
