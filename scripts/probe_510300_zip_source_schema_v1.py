"""用HTTP Range读取公开ZIP目录及CSV表头；不落盘整个年度数据。"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_pressure_zip_schema_probe_v1.json"


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class LimitedRemoteZip(io.RawIOBase):
    """只接受精确范围响应，超预算提前失败；每段保留内容hash而非数据行。"""

    def __init__(self, url: str, size: int, budget: int):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.received = 0
        self.receipts, self.cache = [], {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0:
            raise ValueError("ZIP定位不能为负。")
        self.position = position
        return position

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else min(length, max(0, self.size - self.position))
        if length == 0:
            return b""
        key = (self.position, length)
        if key in self.cache:
            data = self.cache[key]
        else:
            if length > self.budget - self.received:
                raise RuntimeError("ZIP目录或表头超出1MiB总预算，保留未识别。")
            start, end = self.position, self.position + length - 1
            receipt = {"offset": start, "requested_bytes": length, "started_at": now()}
            with requests.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              timeout=15, stream=True) as response:
                receipt["http_status"] = response.status_code
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    self.receipts.append(dict(receipt, status="REJECTED_NONEXACT_RANGE", actual_content_range=response.headers.get("Content-Range")))
                    raise RuntimeError("该公开源未返回精确Range，已终止，不下载整个ZIP。")
                chunks, actual = [], 0
                for chunk in response.iter_content(chunk_size=min(32768, length + 1)):
                    actual += len(chunk)
                    self.received += len(chunk)
                    if actual > length:
                        raise RuntimeError("Range正文长度超出请求。")
                    chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != length:
                raise RuntimeError("Range正文不完整。")
            self.receipts.append(dict(receipt, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                      finished_at=now(), status="EXACT_RANGE_READ"))
            self.cache[key] = data
        self.position += len(data)
        return data


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_directory"]
    if out.exists():
        raise SystemExit("ZIP字段探针结果已存在，拒绝覆盖。")
    out.mkdir(parents=True)
    save_json(out / "registration.json", {"registered_at": now(), "configuration": cfg,
                                           "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                                      for p in [CONFIG, Path(__file__)]]})
    with requests.get(cfg["metadata_url"], timeout=15, headers={"Accept-Encoding": "identity"}, stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(chunk_size=32768):
            if len(body) + len(chunk) > cfg["maximum_http_body_bytes"]:
                raise RuntimeError("ZIP元数据超预算。")
            body.extend(chunk)
    (out / "repo_tree.json").write_bytes(body)
    tree = json.loads(body)
    item = next(row for row in tree if row["path"] == cfg["archive"])
    url = f"https://huggingface.co/datasets/{cfg['repo']}/resolve/{cfg['revision']}/{quote(cfg['archive'])}"
    remote = LimitedRemoteZip(url, int(item["size"]), cfg["maximum_http_body_bytes"] - len(body))
    summary = {"repo": cfg["repo"], "revision": cfg["revision"], "archive": cfg["archive"], "archive_bytes": item["size"],
               "parsed_market_rows": 0, "strategy_returns": "NOT_COMPUTED", "fee_usd": 0}
    try:
        with zipfile.ZipFile(remote) as archive:
            members = archive.infolist()
            catalog = [{"name": m.filename, "compressed_bytes": m.compress_size, "uncompressed_bytes": m.file_size,
                        "compression_method": m.compress_type, "encrypted": bool(m.flag_bits & 1)} for m in members]
            save_json(out / "zip_member_catalog.json", {"members": catalog})
            targets = [m for m in members if "510300" in m.filename and m.filename.lower().endswith(".csv")]
            source_targets = bool(targets)
            if not targets:
                targets = [m for m in members if m.filename.lower().endswith(".csv")][:1]
            headers = []
            for member in targets[:cfg["maximum_header_members"]]:
                if member.flag_bits & 1:
                    headers.append({"member": member.filename, "status": "PASSWORD_REQUIRED_NOT_READ"})
                    continue
                with archive.open(member) as handle:
                    first = handle.readline(cfg["maximum_csv_header_bytes"] + 1)
                if len(first) > cfg["maximum_csv_header_bytes"] or not first.endswith(b"\n"):
                    headers.append({"member": member.filename, "status": "HEADER_TOO_LONG_OR_NOT_LINE_DELIMITED"})
                    continue
                try:
                    text, encoding = first.decode("utf-8-sig"), "utf-8-sig"
                except UnicodeDecodeError:
                    text, encoding = first.decode("gb18030"), "gb18030"
                headers.append({"member": member.filename, "status": "CSV_HEADER_ONLY_READ", "encoding": encoding,
                                "first_line_fields": text.strip(), "header_sha256": hashlib.sha256(first).hexdigest(), "parsed_market_rows": 0})
            summary.update(status="ZIP_CATALOG_AND_SCHEMA_PROFILED_NOT_MARKET_ADMISSION", member_count=len(members),
                           target_named_csv_found=source_targets, schema_headers=headers)
    except (RuntimeError, requests.RequestException, zipfile.BadZipFile, KeyError, UnicodeDecodeError, NotImplementedError) as error:
        summary.update(status="ZIP_SCHEMA_NOT_IDENTIFIED_WITHIN_BUDGET_OR_RANGE_CONTRACT", error=str(error))
    summary.update(completed_at=now(), response_bytes_received=len(body) + remote.received,
                   full_archive_downloaded=False, source_target_coverage_verified=False,
                   raw_byte_use="ZIP目录及用于CSV表头解码的压缩前缀；没有读取价格单元格或生成行情表")
    save_json(out / "range_receipts.json", {"ranges": remote.receipts})
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
