"""依照 ALFRED 公开下载表单获取新时间戳快照，不覆盖研究冻结输入。"""
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=CHNMAINLANDEPU"


def main() -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f_0800")
    out = ROOT / "data/raw/market/510300_epu_vintages_source_v1" / stamp
    out.mkdir(parents=True, exist_ok=False)
    receipts = []
    session = requests.Session()

    def fetch(name: str, url: str, data=None) -> bytes:
        receipt = {"url": url, "method": "POST" if data is not None else "GET", "params": data,
                   "requested_at": datetime.now().astimezone().isoformat(), "tls_verify": True}
        try:
            response = session.post(url, data=data, timeout=(10, 40)) if data is not None else session.get(url, timeout=(10, 40))
            with (out / name).open("xb") as stream:
                stream.write(response.content)
            receipt.update(http_status=response.status_code, response_url=response.url,
                           raw_path=(out / name).relative_to(ROOT).as_posix(), bytes=len(response.content),
                           sha256=hashlib.sha256(response.content).hexdigest(), http_date=response.headers.get("Date"),
                           received_at=datetime.now().astimezone().isoformat())
            response.raise_for_status()
            return response.content
        except Exception as error:
            receipt.update(status="SOURCE_REQUEST_FAILED", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            receipts.append(receipt)
            (out / "source_receipt.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        raw = fetch("alfred_download_form.html", URL)
        soup = BeautifulSoup(raw, "html.parser")
        select = soup.find("select", attrs={"name": "form[selected_vintage_dates][]"})
        if select is None:
            raise ValueError("公开表单结构变化，停止下载而不猜测参数")
        vintages = [option["value"] for option in select.find_all("option")]
        obs_end = soup.find("input", attrs={"name": "form[obs_end_date]"})["value"]
        fields = [("form[units]", "lin"), ("form[obs_start_date]", "2010-01-01"), ("form[obs_end_date]", obs_end),
                  ("form[entered_vintage_dates]", ""), ("form[file_type]", "1"), ("form[file_format]", "csv"), ("form[download_data]", "")]
        fields += [("form[selected_vintage_dates][]", date) for date in vintages]
        archive_bytes = fetch("alfred_realtime_download.zip", URL, fields)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name in ("README.txt", "obs._by_real-time_period.csv"):
                with (out / name).open("xb") as stream:
                    stream.write(archive.read(name))
        cross = [(name, value) for name, value in fields if name != "form[selected_vintage_dates][]"]
        cross = [(name, "2" if name == "form[file_type]" else "2022-09-09 2023-01-06" if name == "form[entered_vintage_dates]" else value) for name, value in cross]
        archive_bytes = fetch("alfred_crosscheck_matrix.zip", URL, cross)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    with (out / ("crosscheck_" + Path(info.filename).name)).open("xb") as stream:
                        stream.write(archive.read(info))
        for name, url in (("alfred_help.html", "https://alfred.stlouisfed.org/help"),
                          ("alfred_download_help.html", "https://alfred.stlouisfed.org/help/downloaddata"),
                          ("mainland_epu_method.html", "https://www.policyuncertainty.com/china_monthly.html")):
            fetch(name, url)
        summary = {"status": "PUBLIC_SOURCE_SNAPSHOT_SAVED", "completed_at": datetime.now().astimezone().isoformat(),
                   "vintage_count": len(vintages), "new_model_fits": 0, "new_accounts": 0,
                   "existing_frozen_inputs_modified": False, "automatic_daily_collection_started": False}
    except Exception as error:
        summary = {"status": "SOURCE_COLLECTION_FAILED_EVIDENCE_PRESERVED", "completed_at": datetime.now().astimezone().isoformat(),
                   "error_type": type(error).__name__, "error": str(error), "new_model_fits": 0, "new_accounts": 0}
    finally:
        session.close()
    (out / "completion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"输出目录": str(out), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
