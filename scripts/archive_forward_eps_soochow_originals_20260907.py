"""归档东吴证券固定原件队列，保存日期证据并复用三份来源样例。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, compact
from research.forward_eps_guosen_history_v1 import identity

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/report/"}


def paths(pilot: bool) -> tuple[Path, Path]:
    label = "510300_forward_eps_soochow_source_pilot_v1" if pilot else "510300_forward_eps_soochow_originals_v1"
    return ROOT / "reports/research" / label, ROOT / "data/raw" / label


def request(url: str) -> bytes:
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(12, 35))
            if response.status_code in (401, 403, 429):
                raise PermissionError(f"公开来源访问限制：{response.status_code}")
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt:
                raise
            time.sleep(1)
    raise RuntimeError("原件请求没有返回")


def one(pilot: bool, row: dict) -> dict:
    out, raw = paths(pilot)
    aid = row["infoCode"]
    receipt = out / "document_records" / f"{aid}.json"
    if receipt.exists():
        return read(receipt)
    try:
        old = paths(True)[0] / "document_records" / f"{aid}.json"
        if not pilot and old.exists():
            previous = read(old)
            source = previous["source"]
            for pathkey, hashkey in [("raw_html_path", "html_sha256"), ("raw_pdf_path", "pdf_sha256")]:
                assert hashlib.sha256((ROOT / source[pathkey]).read_bytes()).hexdigest() == source[hashkey]
            meta = previous["provider_metadata"]
            pages = read(paths(True)[0] / "page_texts" / f"{aid}.json")["pages"]
            reused = old.relative_to(ROOT).as_posix()
        else:
            raw.mkdir(parents=True, exist_ok=True)
            url = "https://data.eastmoney.com/report/zw_stock.jshtml?infocode=" + aid
            hp = raw / f"{aid}.html"
            html = hp.read_bytes() if hp.exists() else request(url)
            hp.write_bytes(html)
            text = html.decode("utf-8-sig")
            marker = re.search(r"var\s+zwinfo\s*=\s*", text)
            if marker is None:
                raise ValueError("原件详情没有研报元数据")
            meta = json.JSONDecoder().raw_decode(text[marker.end():])[0]
            pdfurl = meta["attach_url"]
            if not pdfurl.startswith("https://pdf.dfcfw.com/pdf/"):
                raise ValueError("原件附件不是既定公开来源")
            pp = raw / f"{aid}.pdf"
            payload = pp.read_bytes() if pp.exists() else request(pdfurl)
            if not payload.startswith(b"%PDF-"):
                raise ValueError("附件不是PDF")
            pp.write_bytes(payload)
            doc = pdfium.PdfDocument(payload)
            pages = []
            try:
                for number in range(len(doc)):
                    page = doc[number]
                    tp = page.get_textpage()
                    try:
                        pages.append(tp.get_text_range())
                    finally:
                        tp.close()
                        page.close()
            finally:
                doc.close()
            source = {"report_id": aid, "provider_detail_url": url, "pdf_url": pdfurl,
                      "retrieved_at": now(), "raw_html_path": hp.relative_to(ROOT).as_posix(),
                      "raw_pdf_path": pp.relative_to(ROOT).as_posix(),
                      "html_sha256": hashlib.sha256(html).hexdigest(),
                      "pdf_sha256": hashlib.sha256(payload).hexdigest(), "pdf_pages": len(pages)}
            reused = None
        if meta["info_code"] != aid or str(meta["company_code"]) != "80000031":
            raise ValueError("研报编号或机构不符")
        if row["ts_code"][:6] not in [str(x.get("stock")) for x in meta["security"]]:
            raise ValueError("详情证券不符")
        front = compact("\n".join(pages[:2]))
        matched = row["ts_code"][:6] in front and ("东吴证券" in front or "dwzq.com.cn" in front.lower())
        record = {"report_id": aid, "ts_code": row["ts_code"], "directory_record": row,
                  "source": source, "provider_metadata": meta, "reused_from": reused,
                  "first_two_page_identity_match": matched,
                  "status": "ARCHIVED_FRONT_IDENTITY_MATCH" if matched else "ARCHIVED_FRONT_IDENTITY_PENDING",
                  "new_eps_facts": 0}
        save(out / "page_texts" / f"{aid}.json", {"source": source, "pages": pages}, exclusive=True)
        save(receipt, record, exclusive=True)
        return record
    except PermissionError:
        raise
    except Exception as exc:
        record = {"report_id": aid, "ts_code": row["ts_code"], "status": "ORIGINAL_ARCHIVE_FAILED",
                  "error_type": type(exc).__name__, "error": str(exc), "at": now()}
        save(out / "errors" / f"{aid}.json", record)
        return record


def freeze(pilot: bool) -> None:
    out, raw = paths(pilot)
    out.mkdir(parents=True, exist_ok=False)
    if pilot:
        source = ROOT / "reports/research/510300_forward_eps_multi_institution_probe_v1"
        rows = []
        for year in [2018, 2021, 2024]:
            sample = read(source / f"{year}_directory_sample.json")["reports"]
            selected = max((r for r in sample if r["orgCode"] == "80000031"), key=lambda r: (r["publishDate"], r["infoCode"]))
            rows.append({**selected, "ts_code": "600519.SH"})
        queue = pd.DataFrame(rows)
    else:
        source = ROOT / "reports/research/510300_forward_eps_soochow_directory_v1"
        status = read(source / "result.json")
        assert not status["errors"] and status["pagination_complete_years"] == 12
        queue = pd.read_parquet(source / "historical_member_original_pdf_queue.parquet")
    queue.to_parquet(out / "selected_before_originals.parquet", index=False)
    save(out / "manifest.json", {"registered_at": now(), "institution_code": "80000031", "pilot": pilot,
        "selected_reports": len(queue), "return_selection": False,
        "files": [identity(p) for p in [Path(__file__), out / "selected_before_originals.parquet",
                  ROOT / "docs/510300_FORWARD_EPS_SOOCHOW_HISTORY_V1.md", source / "result.json"]]}, exclusive=True)
    print("东吴原件范围已登记", "样例" if pilot else "历史并集", len(queue), flush=True)


def run(pilot: bool) -> None:
    out, raw = paths(pilot)
    manifest = read(out / "manifest.json")
    for item in manifest["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "原件登记输入变化"
    if (out / "result.json").exists():
        raise FileExistsError("本轮原件归档已经结束")
    rows = pd.read_parquet(out / "selected_before_originals.parquet").to_dict("records")
    done, restricted = [], False
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(one, pilot, row): row for row in rows}
        for future in as_completed(pending):
            row = pending[future]
            try:
                record = future.result()
            except PermissionError as exc:
                restricted = True
                for f in pending:
                    f.cancel()
                record = {"report_id": row["infoCode"], "status": "ACCESS_RESTRICTED", "error": str(exc)}
            except CancelledError:
                record = {"report_id": row["infoCode"], "status": "SKIPPED_AFTER_ACCESS_RESTRICTION"}
            except Exception as exc:
                record = {"report_id": row["infoCode"], "status": "WORKER_FAILED", "error_type": type(exc).__name__, "error": str(exc)}
            done.append(record)
            if len(done) % 25 == 0:
                print("东吴原件归档", len(done), "/", len(rows), "已保存", sum("source" in x for x in done), flush=True)
    good = [x for x in done if "source" in x]
    result = {"study_id": "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_V1", "completed_at": now(),
              "pilot": pilot, "status": "ORIGINALS_ARCHIVED" if len(good) == len(rows) else "ORIGINALS_PARTIAL",
              "selected_reports": len(rows), "archived_reports": len(good),
              "first_two_page_identity_matches": sum(x["first_two_page_identity_match"] for x in good),
              "pdf_pages": sum(x["source"]["pdf_pages"] for x in good),
              "pdf_bytes": sum((ROOT / x["source"]["raw_pdf_path"]).stat().st_size for x in good),
              "reused_reports": sum(bool(x["reused_from"]) for x in good),
              "access_restriction_stop": restricted, "failures": [x for x in done if "source" not in x],
              "new_eps_facts": 0, "new_accounts_generated": 0, "goal_achieved": False}
    save(out / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze(args.pilot) if args.freeze else run(args.pilot)
