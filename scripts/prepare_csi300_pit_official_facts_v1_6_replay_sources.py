from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_6_gap_replay_sources"
)
MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.cninfo.com.cn/",
}

SOURCES: tuple[dict[str, str], ...] = (
    {
        "announcement_id": "1202600713",
        "ts_code": "600585.SH",
        "period_type": "H1",
        "report_period": "2016-06-30",
        "fix_family": "INVERSE_ORDER_CORE_PARENT_PROFIT_LABEL",
        "official_pdf_url": (
            "https://static.cninfo.com.cn/finalpage/2016-08-23/1202600713.PDF"
        ),
    },
    {
        "announcement_id": "1205362872",
        "ts_code": "000063.SZ",
        "period_type": "H1",
        "report_period": "2018-06-30",
        "fix_family": "NET_RECEIVABLE_DUAL_PERIOD_RECONCILIATION",
        "official_pdf_url": (
            "https://static.cninfo.com.cn/finalpage/2018-08-31/1205362872.PDF"
        ),
    },
    {
        "announcement_id": "1203181067",
        "ts_code": "000703.SZ",
        "period_type": "FY",
        "report_period": "2016-12-31",
        "fix_family": "SUMMARY_ANCHORED_IMAGE_STATEMENT_OCR",
        "official_pdf_url": (
            "https://static.cninfo.com.cn/finalpage/2017-03-21/1203181067.PDF"
        ),
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _download(source: dict[str, str]) -> dict[str, Any]:
    announcement_id = source["announcement_id"]
    destination = SOURCE_ROOT / f"{announcement_id}.pdf"
    if not destination.exists():
        temporary = destination.with_suffix(".pdf.download")
        with requests.get(
            source["official_pdf_url"],
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
    content_header = destination.read_bytes()[:5]
    if content_header != b"%PDF-":
        raise RuntimeError(f"官方文件没有PDF头：{announcement_id}")
    return {
        **source,
        "path": destination.relative_to(ROOT).as_posix(),
        "status": "DOWNLOADED_OFFICIAL_PDF_SHA256_RECORDED",
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def main() -> int:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for completed, source in enumerate(SOURCES, start=1):
        row = _download(source)
        results.append(row)
        print(
            f"V1.6固定缺口PDF {completed}/{len(SOURCES)}｜"
            f"{row['announcement_id']}｜{row['size_bytes']:,}字节",
            flush=True,
        )
    payload = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_6_GAP_REPLAY_SOURCES",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "source_host": "static.cninfo.com.cn",
        "target_count": len(results),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(MANIFEST_PATH, payload)
    print(f"V1.6固定缺口源清单：{MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    print("本程序未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
