"""取得官方公开NFCI实时版本；保留原始响应，按档案版本建立历史可见快照。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SERIES = ("NFCI", "NFCICREDIT", "NFCIRISK")
SOURCE = ROOT / "data/raw/macro/510300_nfci_public_vintages_source_v1"
REPORT = ROOT / "reports/research/510300_nfci_public_vintages_source_v1"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect() -> Path:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f_0800")
    out = SOURCE / stamp
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "study_id": "510300_NFCI_PUBLIC_VINTAGES_SOURCE_V1", "created_at": now(),
        "series": list(SERIES), "observation_start": "2010-01-01",
        "source_vintage_ceiling": "2026-09-25", "mode": "PUBLIC_ARCHIVAL_VINTAGES",
        "source_urls": [f"https://alfred.stlouisfed.org/series/downloaddata?seid={s}" for s in SERIES],
        "availability": "版本日芝加哥当地午夜加一个自然日后才视为可见；执行日09:00使用已可见版本。不是历史HTTP到达时间证明。",
        "use_realtime_end_to_predict": False, "retroactive_current_vintage_substitution": False,
        "source_only": True, "new_model_fits": 0, "new_accounts": 0,
        "authority": "reports/research/510300_new_evidence_resume_20260925/authority_update.json",
    }
    save(out / "protocol.json", protocol)
    receipts: list[dict] = []
    results: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; public historical economic research)"})

    def fetch(name: str, url: str, fields=None) -> bytes:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = {"requested_at": now(), "url": url, "method": "POST" if fields is not None else "GET",
                   "request_fields": fields, "tls_verify": True}
        try:
            response = session.post(url, data=fields, timeout=(12, 55)) if fields is not None else session.get(url, timeout=(12, 40))
            target.write_bytes(response.content)
            receipt.update({"received_at": now(), "http_status": response.status_code, "response_url": response.url,
                            "content_type": response.headers.get("Content-Type"), "http_date": response.headers.get("Date"),
                            "raw_path": target.relative_to(ROOT).as_posix(), "bytes": len(response.content),
                            "sha256": hashlib.sha256(response.content).hexdigest()})
            response.raise_for_status()
            return response.content
        except Exception as error:
            receipt.update({"error_type": type(error).__name__, "error": str(error), "failed_at": now()})
            raise
        finally:
            receipts.append(receipt)
            save(out / "source_receipts.json", receipts)

    try:
        for series in SERIES:
            print(f"正在取得 {series} 的公开历史版本。", flush=True)
            try:
                url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
                raw = fetch(f"{series}/download_form.html", url)
                soup = BeautifulSoup(raw, "html.parser")
                select = soup.find("select", attrs={"name": "form[selected_vintage_dates][]"})
                if select is None:
                    raise ValueError("公开下载表单没有历史版本字段，不能推测历史可用性")
                vintages = sorted({str(o["value"]) for o in select.find_all("option") if o.get("value")})
                vintages = [v for v in vintages if v <= protocol["source_vintage_ceiling"]]
                if not vintages:
                    raise ValueError("截止日期之前没有可用版本")
                end_input = soup.find("input", attrs={"name": "form[obs_end_date]"})
                if end_input is None:
                    raise ValueError("公开下载表单缺少观察结束日")
                end_date = min(str(end_input["value"]), protocol["source_vintage_ceiling"])
                fields = [("form[units]", "lin"), ("form[obs_start_date]", protocol["observation_start"]),
                          ("form[obs_end_date]", end_date), ("form[entered_vintage_dates]", ""),
                          ("form[file_type]", "1"), ("form[file_format]", "csv"), ("form[download_data]", "")]
                fields.extend(("form[selected_vintage_dates][]", d) for d in vintages)
                save(out / series / "selected_vintages.json", vintages)
                body = fetch(f"{series}/realtime_download.zip", url, fields)
                with zipfile.ZipFile(io.BytesIO(body)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("下载档案校验失败")
                    for name in ("README.txt", "obs._by_real-time_period.csv"):
                        (out / series / name).write_bytes(archive.read(name))
                # 独立输出格式核对固定日期，避免只相信一个解析路径。
                fixed_dates = "2015-01-07 2020-03-18 2024-09-25"
                cross = [(k, v) for k, v in fields if k != "form[selected_vintage_dates][]"]
                cross = [(k, "2" if k == "form[file_type]" else fixed_dates if k == "form[entered_vintage_dates]" else v) for k, v in cross]
                cross_body = fetch(f"{series}/crosscheck_download.zip", url, cross)
                with zipfile.ZipFile(io.BytesIO(cross_body)) as archive:
                    for info in archive.infolist():
                        if not info.is_dir():
                            (out / series / ("crosscheck_" + Path(info.filename).name)).write_bytes(archive.read(info))
                results[series] = {"status": "RAW_ARCHIVE_SAVED", "vintage_count": len(vintages),
                                   "first_vintage": vintages[0], "last_vintage": vintages[-1], "observation_end": end_date}
                print(f"{series}：已保存 {len(vintages)} 个版本的原始档案。", flush=True)
            except Exception as error:
                results[series] = {"status": "SOURCE_REQUEST_FAILED", "error_type": type(error).__name__, "error": str(error)}
                print(f"{series} 取得失败，原始响应和失败信息已保留：{type(error).__name__}", flush=True)
        methods = [
            ("alfred_download_help.html", "https://alfred.stlouisfed.org/help/downloaddata"),
            ("nfci_about.html", "https://www.chicagofed.org/research/data/nfci/about"),
            ("nfci_current_method.html", "https://www.chicagofed.org/research/data/nfci/current-data"),
        ]
        for name, url in methods:
            try:
                fetch(name, url)
            except requests.RequestException:
                print(f"方法页 {name} 获取失败，保留失败回执。", flush=True)
    finally:
        session.close()
    status = "RAW_ARCHIVES_SAVED_PENDING_ASOF_VALIDATION" if all(x.get("status") == "RAW_ARCHIVE_SAVED" for x in results.values()) and len(results) == len(SERIES) else "PARTIAL_SOURCE_SNAPSHOT"
    completion = {"completed_at": now(), "status": status, "snapshot": out.relative_to(ROOT).as_posix(),
                  "series": results, "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False,
                  "current_vintage_is_not_historical_signal": True}
    save(out / "completion.json", completion)
    save(REPORT / "latest_source_snapshot.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="取得NFCI公开历史版本，保留每次请求的原始响应。")
    parser.add_argument("--collect", action="store_true", required=True)
    parser.parse_args()
    collect()


if __name__ == "__main__":
    main()
