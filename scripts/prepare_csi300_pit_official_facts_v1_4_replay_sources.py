from __future__ import annotations

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = (
    ROOT
    / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "checkpoints_v1_3"
)
OUTPUT_ROOT = (
    ROOT
    / "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_4_replay_sources"
)
EXPECTED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_"
    "PDFIUM_PDFPLUMBER_WINDOWS_OCR_V1_7_0"
)
ACCEPTED_HOST = "static.cninfo.com.cn"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_targets() -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("checkpoint_status") != "PARSED_INCOMPLETE":
            continue
        if payload.get("parser_version") != EXPECTED_PARSER_VERSION:
            raise ValueError(f"检查点解析器版本不一致：{path}")
        url = str(payload["official_pdf_url"])
        if urlparse(url).hostname != ACCEPTED_HOST:
            raise ValueError(f"官方PDF主机不符合冻结合同：{url}")
        targets.append(
            {
                "announcement_id": str(payload["announcement_id"]),
                "official_pdf_announcement_id": str(
                    payload["official_pdf_announcement_id"]
                ),
                "url": url,
                "expected_sha256": str(payload["official_pdf_sha256"]),
            }
        )
    if not targets:
        raise RuntimeError("没有找到V1.3不完整检查点")
    return targets


def _download(target: dict[str, str]) -> dict[str, str | int]:
    output_path = OUTPUT_ROOT / (
        f"{target['announcement_id']}__{target['official_pdf_announcement_id']}.pdf"
    )
    if output_path.exists():
        content = output_path.read_bytes()
        if _sha256(content) == target["expected_sha256"] and content.startswith(b"%PDF-"):
            return {
                "announcement_id": target["announcement_id"],
                "status": "REUSED_SHA256_VERIFIED",
                "path": output_path.relative_to(ROOT).as_posix(),
                "size_bytes": len(content),
            }
    response = requests.get(
        target["url"],
        timeout=120,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/128.0 Safari/537.36"
            ),
            "Referer": "https://www.cninfo.com.cn/",
        },
    )
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"下载内容不是PDF：{target['announcement_id']}")
    observed_sha256 = _sha256(content)
    if observed_sha256 != target["expected_sha256"]:
        raise ValueError(
            "官方PDF哈希与V1.3检查点不一致："
            f"{target['announcement_id']}|{observed_sha256}"
        )
    temporary_path = output_path.with_suffix(".pdf.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)
    return {
        "announcement_id": target["announcement_id"],
        "status": "DOWNLOADED_SHA256_VERIFIED",
        "path": output_path.relative_to(ROOT).as_posix(),
        "size_bytes": len(content),
    }


def main() -> int:
    targets = _load_targets()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_download, target): target for target in targets}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['announcement_id']}：{result['status']}|"
                f"{result['size_bytes']:,}字节",
                flush=True,
            )
    results.sort(key=lambda row: str(row["announcement_id"]))
    if len(results) != len(targets):
        raise RuntimeError("V1.4固定重放PDF数量不完整")
    manifest_path = OUTPUT_ROOT / "source_manifest.json"
    manifest = {
        "purpose": "FIXED_V1_3_INCOMPLETE_DOCUMENT_REPLAY_FOR_VERSIONED_V1_4_PARSER",
        "source_checkpoint_root": CHECKPOINT_ROOT.relative_to(ROOT).as_posix(),
        "target_count": len(targets),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"固定重放源：{len(results)}/{len(targets)}；清单：{manifest_path}")
    print("本程序未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
