"""按三个固定年度获取510300完整年报，逐请求保留来源回执。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_eps_disclosed_holdings_diagnostic_v1.json"
PROTOCOL = ROOT / "docs/510300_EPS_DISCLOSED_HOLDINGS_DIAGNOSTIC_V1_PROTOCOL.md"
OUT = ROOT / "reports/research/510300_eps_disclosed_holdings_diagnostic_v1"
RAW = ROOT / "data/raw/market/510300_eps_disclosed_holdings_v1"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f")
    directory = RAW / stamp
    inputs = [CONFIG, PROTOCOL, Path(__file__), ROOT / config["latest_company_information_source"],
              ROOT / config["institution_evidence_source"]]
    save(OUT / "source_collection_claim.json", {"started_at": now(), "raw_directory": directory.relative_to(ROOT).as_posix(),
         "files": [identity(p) for p in inputs], "new_labels": 0, "new_models": 0, "new_accounts": 0})
    directory.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                            "Referer": "https://www.sse.com.cn/disclosure/fund/announcement/"})
    requests_log = []

    def fetch(name: str, url: str, params: dict | None, pdf: bool = False) -> tuple[bytes, dict]:
        for attempt in range(1, config["maximum_network_attempts_per_object"] + 1):
            record = {"object": name, "attempt": attempt, "url": url, "params": params,
                      "started_at": now(), "tls_verification": True}
            try:
                response = session.get(url, params=params, timeout=config["source_timeout_seconds"])
                record.update({"http_status": response.status_code, "final_url": response.url,
                               "server_date": response.headers.get("Date"), "content_type": response.headers.get("Content-Type")})
                raw_path = directory / f"{name}_attempt{attempt}.{'pdf' if pdf else 'raw'}"
                raw_path.write_bytes(response.content)
                record["raw_file"] = identity(raw_path)
                response.raise_for_status()
                if pdf:
                    if not response.content.startswith(b"%PDF-"):
                        raise ValueError("返回对象不是PDF")
                else:
                    json.loads(response.content)
                record["status"] = "PASS"
                record["completed_at"] = now()
                requests_log.append(record)
                save(directory / f"{name}_attempt{attempt}.receipt.json", record)
                return response.content, record
            except Exception as exc:
                record.update({"status": "EXTERNAL_FREE_SOURCE_FAILED", "completed_at": now(),
                               "error_type": type(exc).__name__, "error": str(exc)})
                requests_log.append(record)
                save(directory / f"{name}_attempt{attempt}.receipt.json", record)
                if attempt == config["maximum_network_attempts_per_object"]:
                    raise
        raise RuntimeError("请求次数状态不一致")

    documents = []
    for year in config["report_years"]:
        try:
            rows, catalogs = [], []
            page = 1
            while True:
                params = {"isPagination": "true", "pageHelp.pageSize": config["query_page_size"],
                          "pageHelp.pageNo": page, "pageHelp.beginPage": page, "pageHelp.cacheSize": 1,
                          "pageHelp.endPage": page, "type": "inParams", "sqlId": "COMMON_PL_JJXX_JJGG_NEW_L",
                          "TITLE": config["query_title"], "SECURITY_CODE": config["fund_code"], "BULLETIN_TYPE": "",
                          "START_DATE": f"{year + 1}-01-01", "END_DATE": f"{year + 1}-04-30",
                          "DATE_DESC": 1, "DATE_ASC": "", "CODE_DESC": "", "CODE_ASC": ""}
                content, receipt = fetch(f"catalog_{year}_page{page}", "https://query.sse.com.cn/commonQuery.do", params)
                data = json.loads(content)
                batch = data["result"]
                total = int(data["pageHelp"]["total"])
                if any(str(x["SECURITY_CODE"]) != config["fund_code"] for x in batch):
                    raise ValueError("公告基金身份不符")
                rows.extend(batch)
                catalogs.append(receipt["raw_file"])
                if len(rows) >= total:
                    if len(rows) != total:
                        raise ValueError("目录计数不一致")
                    break
                page += 1
                if page > config["maximum_query_pages_per_year"]:
                    raise ValueError("目录分页超过预定上限")
            if len({x["URL"] for x in rows}) != len(rows):
                raise ValueError("年度目录含重复路径")
            selected = [x for x in rows if f"{year}年年度报告" in re.sub(r"\s+", "", x["TITLE"])
                        and not any(s in x["TITLE"] for s in config["exclude_titles"])]
            if len(selected) != 1:
                raise ValueError(f"完整原始年度报告不是唯一对象：{len(selected)}")
            item = selected[0]
            url = urljoin("https://www.sse.com.cn", item["URL"])
            if not url.startswith("https://www.sse.com.cn/disclosure/fund/announcement/"):
                raise ValueError("PDF不属于登记的上交所基金公告路径")
            _, receipt = fetch(f"annual_{year}", url, None, pdf=True)
            documents.append({"year": year, "status": "ORIGINAL_PDF_ACQUIRED_NOT_YET_ADMITTED", "metadata": item,
                              "catalogs": catalogs, "pdf": receipt["raw_file"], "retrieved_at": receipt["completed_at"]})
            print(f"{year}年原始报告已保存，待身份、时钟和持仓核对。", flush=True)
        except Exception as exc:
            documents.append({"year": year, "status": "NO_VIEW_SOURCE_COLLECTION_FAILED",
                              "error_type": type(exc).__name__, "error": str(exc)})
            print(f"{year}年来源未完成：{type(exc).__name__}：{exc}", flush=True)
    result = {"study_id": config["study_id"], "completed_at": now(),
              "status": "THREE_FIXED_ORIGINAL_REPORTS_ACQUIRED" if all(x["status"].startswith("ORIGINAL") for x in documents)
                        else "NO_VIEW_INCOMPLETE_FIXED_REPORTS", "documents": documents,
              "network_attempts": len(requests_log), "failed_network_attempts": sum(x["status"] != "PASS" for x in requests_log),
              "new_return_labels": 0, "new_model_fits": 0, "new_accounts": 0}
    save(OUT / "source_collection_result.json", result)
    save(directory / "request_log.json", requests_log)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].startswith("NO_VIEW"):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    main()
