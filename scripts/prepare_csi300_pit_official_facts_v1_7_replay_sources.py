from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_7_gap_replay_sources"
)
DIAGNOSTIC_ROOT = ROOT / "tmp/pit_v1_7_diagnostics"
MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.cninfo.com.cn/",
}

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "announcement_id": "1204679732",
        "ts_code": "600050.SH",
        "period_type": "Q1",
        "report_period": "2018-03-31",
        "fix_family": "NEGATIVE_GUARD_COMBINED_RECEIVABLE_AND_CONTRACT_ASSET",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2018-04-21/1204679732.PDF",
        "expected_size_bytes": 605235,
        "expected_sha256": "3e02fdcb21c26917cf07ef9e884a34a5498879039866fe59f008cd2a0e51783c",
    },
    {
        "announcement_id": "1205243868",
        "ts_code": "600011.SH",
        "period_type": "H1",
        "report_period": "2018-06-30",
        "fix_family": "ADJACENT_NOTE_NET_RECEIVABLE_RECONCILIATION",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2018-08-01/1205243868.PDF",
        "expected_size_bytes": 3062733,
        "expected_sha256": "24efaef2639f70b83df92f93720f9c6f8e74e6c44c06a0905a2a644c3106fe47",
    },
    {
        "announcement_id": "1202613205",
        "ts_code": "601857.SH",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "EXTENDED_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-08-25/1202613205.PDF",
        "expected_size_bytes": 3770664,
        "expected_sha256": "8372830dcf673f086caf8021b2a52f0b6afdfd487a6fa0db22f125b660d87230",
    },
    {
        "announcement_id": "1202642581",
        "ts_code": "000898.SZ",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "EXTENDED_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-08-30/1202642581.PDF",
        "expected_size_bytes": 2244583,
        "expected_sha256": "12997e025cf11dca38aff74ebf6273a7bf7224c3ccda12ab22c5fe01582ad5bc",
    },
    {
        "announcement_id": "1202657301",
        "ts_code": "600871.SH",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "EXTENDED_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-08-31/1202657301.PDF",
        "expected_size_bytes": 2524335,
        "expected_sha256": "a1479934fc9006d19e8039880811fbeb7ba1ceb40812aa4eecf0448c586e98e9",
    },
    {
        "announcement_id": "1202805449",
        "ts_code": "600893.SH",
        "period_type": "Q3",
        "report_period": "2016-09-30",
        "fix_family": "CROSS_PAGE_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-10-29/1202805449.PDF",
        "expected_size_bytes": 978726,
        "expected_sha256": "0710dbb799ce98a138304f5cce2d28a02d29a0c34e01436be39fd05794d7ac27",
    },
    {
        "announcement_id": "1203423195",
        "ts_code": "600406.SH",
        "period_type": "Q1",
        "report_period": "2017-03-31",
        "fix_family": "CROSS_PAGE_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2017-04-29/1203423195.PDF",
        "expected_size_bytes": 482379,
        "expected_sha256": "aabcfc150d84967918c7574f07fd80de307a49b9cd652fcb3727b7023cc4e3a0",
    },
    {
        "announcement_id": "1202268047",
        "ts_code": "600115.SH",
        "period_type": "Q1",
        "report_period": "2016-03-31",
        "fix_family": "QUARTERLY_IMAGE_APPENDIX_DUAL_SCALE_OCR",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-04-29/1202268047.PDF",
        "expected_size_bytes": 2839078,
        "expected_sha256": "7f2f1f73794dda83789aabe5c12d96094fcfc80c0219d81c2fa4c4f06e25b55b",
    },
    {
        "announcement_id": "1202532867",
        "ts_code": "600011.SH",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "DUAL_PERIOD_TEXT_LIABILITY_IDENTITY",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-08-03/1202532867.PDF",
        "expected_size_bytes": 1457394,
        "expected_sha256": "4d5053abbafe8417d7846ace2581f6c361216a5393ce81ccc39c5f299e783ee4",
    },
    {
        "announcement_id": "1202636509",
        "ts_code": "601117.SH",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "EXTENDED_PARENT_NET_PROFIT_IDENTITY",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-08-27/1202636509.PDF",
        "expected_size_bytes": 3505994,
        "expected_sha256": "a8ffe8ccdfe1fe38a07be2d01c833cc4d515e313562f3ed9c8bd90a09f77243a",
    },
    {
        "announcement_id": "1202772943",
        "ts_code": "600549.SH",
        "period_type": "Q3",
        "report_period": "2016-09-30",
        "fix_family": "Q3_EXTENDED_PARENT_NET_PROFIT_IDENTITY",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2016-10-21/1202772943.PDF",
        "expected_size_bytes": 572204,
        "expected_sha256": "45c44e4e03f52b7422310324a73f29ac31ab93ae2ec8838839e9039ae7f9c61e",
    },
    {
        "announcement_id": "1211403108",
        "ts_code": "603993.SH",
        "period_type": "Q3",
        "report_period": "2021-09-30",
        "fix_family": "CROSS_PAGE_PARENT_NET_PROFIT_IDENTITY",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2021-10-28/1211403108.PDF",
        "expected_size_bytes": 784213,
        "expected_sha256": "c821e16dc59403f0d3144d60b15cb0d223dbba359f3ccad4527904999067b3a9",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _obtain(source: dict[str, Any]) -> dict[str, Any]:
    announcement_id = str(source["announcement_id"])
    destination = SOURCE_ROOT / f"{announcement_id}.pdf"
    acquisition_route = "REUSED_EXISTING_FIXED_SOURCE"
    if not destination.exists():
        diagnostic = DIAGNOSTIC_ROOT / f"{announcement_id}.PDF"
        if diagnostic.exists() and _sha256(diagnostic) == source["expected_sha256"]:
            shutil.copyfile(diagnostic, destination)
            acquisition_route = "COPIED_SHA256_VERIFIED_OFFICIAL_DIAGNOSTIC"
        else:
            temporary = destination.with_suffix(".pdf.download")
            with requests.get(
                str(source["official_pdf_url"]),
                headers=HEADERS,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                total = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAXIMUM_PDF_BYTES:
                            raise RuntimeError(f"官方PDF超过100 MiB：{announcement_id}")
                        handle.write(chunk)
            temporary.replace(destination)
            acquisition_route = "DOWNLOADED_FROM_FROZEN_OFFICIAL_URL"
    checks = {
        "pdf_header": destination.read_bytes()[:5] == b"%PDF-",
        "size": destination.stat().st_size == int(source["expected_size_bytes"]),
        "sha256": _sha256(destination) == str(source["expected_sha256"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.7固定官方PDF校验失败：{announcement_id}|{failed}")
    return {
        **source,
        "path": destination.relative_to(ROOT).as_posix(),
        "status": "PASS_FIXED_OFFICIAL_PDF_SHA256_VERIFIED",
        "acquisition_route": acquisition_route,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def main() -> int:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for completed, source in enumerate(SOURCES, start=1):
        row = _obtain(source)
        results.append(row)
        print(
            f"V1.7固定缺口PDF {completed}/{len(SOURCES)}｜"
            f"{row['announcement_id']}｜{row['fix_family']}",
            flush=True,
        )
    payload = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_7_GAP_REPLAY_SOURCES",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "source_host": "static.cninfo.com.cn",
        "target_count": len(results),
        "positive_target_count": 11,
        "negative_guard_target_count": 1,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(MANIFEST_PATH, payload)
    print(f"V1.7固定源清单：{MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    print("本程序未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
