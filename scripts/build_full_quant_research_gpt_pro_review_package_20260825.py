"""构建全项目 GPT Pro 独立审阅压缩包。

本脚本只做快照、脱敏、索引、压缩与完整性验证，不重新运行任何研究，
不修改冻结协议，也不把研究、Paper/Shadow、券商连接、仓位或订单混为一谈。
大型原始数据和重复旧审阅包不直接复制，但会留下相对路径、字节数、
SHA-256 与排除原因，便于外部审阅者判断证据覆盖和复现边界。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
PACKAGE_BASENAME = "QUANT_RESEARCH_ALL_WORK_GPT_PRO_REVIEW_20260825"
INTERNAL_ROOT = "QUANT_RESEARCH_ALL_WORK_GPT_PRO_REVIEW"
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
UPLOAD_MESSAGE_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"
EXCLUSION_HASH_CACHE_PATH = (
    ROOT / "tmp" / f"{PACKAGE_BASENAME}_EXCLUSION_HASH_CACHE.csv"
)
TIME_ZONE = ZoneInfo("Asia/Shanghai")

SMALL_STRUCTURED_LIMIT = 10 * 1024 * 1024
SMALL_BINARY_LIMIT = 2 * 1024 * 1024
MAX_INCLUDED_SOURCE_BYTES = 450 * 1024 * 1024
HASH_WORKERS = 8
FAST_UNREDACTED_USER_REQUEST = True

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".sha256",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

FORMAL_SOURCE_ROOTS = (
    "config",
    "docs",
    "research",
    "scripts",
    "tests",
    "backtest",
    "market_data",
    "src",
)

FULL_EVIDENCE_ROOTS = (
    "reports",
    "data/curated",
    "data/processed",
    "data/reference",
    "data/models",
    "data/backtests",
    "data/forward",
    "data/labels",
)

ROOT_FILES = (
    ".env.example",
    ".gitignore",
    "CONTEXT.md",
    "macd_event_study.py",
    "pytest.ini",
    "requirements.txt",
)

TOP_LEVEL_NOT_ENUMERATED = {
    ".git": "版本库内部对象，不属于交给模型的研究证据",
    ".venv": "本机 Python 虚拟环境，可由 requirements.txt 重建",
    "tmp": "临时下载、渲染与浏览器缓存，不属于正式证据",
    "__pycache__": "Python 解释器缓存，可由源码重建",
    ".pytest_cache": "Pytest 缓存，不属于研究证据",
    ".agents": "本机代理配置，不属于研究证据",
}

STATUS_SOURCE_ROWS = (
    (
        "510300主线-当前前瞻总状态",
        "COLLECTING_FORWARD_WITH_BLOCKERS",
        "reports/audit/PRIORITY_FORWARD_RESEARCH_STATUS_CURRENT.md",
        "以文件内生成时间为准；不得用旧成功状态覆盖新失败回执。",
    ),
    (
        "510300期权盘口前瞻采集",
        "NO_VIEW_PROTOCOL_INTEGRITY_FAILURE",
        "reports/forward/510300_option_orderbook_v1/runner_logs/20260825_runner.log",
        "2026-08-25 在供应商请求前因冻结测试哈希漂移退出；不是零观测。",
    ),
    (
        "510300 T-only V1.1影子账本",
        "NO_VIEW_UNTIL_FORWARD_MATURITY",
        "reports/forward/t_only_forward_v1_1_status.md",
        "仅影子账本；至少252个新交易日且闭合3个周期前不评价。",
    ),
    (
        "510300历史R6",
        "HISTORICALLY_REJECTED",
        "docs/R6_REJECTION_DECISION.md",
        "拒绝后不得用相邻参数、窗口或标签救援。",
    ),
    (
        "A股极端缩波V2/V2.1",
        "NO_REPEATABLE_CONDITIONAL_EXCESS_IDENTIFIED",
        "docs/A_SHARE_HS_EXTREME_COMPRESSION_V2_V2_1_TERMINAL_FREEZE_20260823.md",
        "研究分支冻结关闭，不是可交易Alpha。",
    ),
    (
        "A股ORJ V2执行验收",
        "REJECTED_ORJ_NET_EXCESS_EXECUTION_V2",
        "reports/research/A_SHARE_HS_OFFICIAL_REPORT_ORJ_NET_EXCESS_EXECUTION_V2.md",
        "父研究预测证据未转化为下一开盘、一手、成本后可靠净超额。",
    ),
    (
        "国际化券商研究",
        "NO_VIEW_G3_PROFIT_BRIDGE_INCOMPLETE",
        "reports/research/international_broker_g2_g3_v1_3_status.md",
        "与510300隔离；收益检验、持仓与下单均禁止。",
    ),
    (
        "知乎公开语料研究",
        "BOUNDED_OBSERVABLE_CORPUS_DISCOVERY_ONLY",
        "reports/research/ZHIHU_OLIVER_OBSERVABLE_CORPUS_V2_1_STATUS.md",
        "语料不完整；观点只能登记为候选，不能当策略或Alpha。",
    ),
)


@dataclass(frozen=True)
class ExcludedDirectory:
    relative_path: str
    reason: str
    file_count: int | None
    total_bytes: int | None
    access_error_count: int


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: Any) -> str:
    """分块计算已打开二进制流的 SHA-256。"""

    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    """以 UTF-8 原子写入文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """以稳定、可比较格式原子写入 JSON。"""

    write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def safe_member(value: str) -> str:
    """拒绝绝对路径、空路径与路径穿越。"""

    member = PurePosixPath(value.replace("\\", "/"))
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise ValueError(f"非法包内路径：{value}")
    return member.as_posix()


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def is_cache_or_temporary(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts.intersection({"__pycache__", ".pytest_cache", ".venv", ".git", "tmp"})
        or any(part.startswith(".pytest-tmp-") for part in path.parts)
        or path.suffix.lower() in {".pyc", ".pyo", ".tmp", ".lock"}
        or path.name == ".gitkeep"
    )


def iter_files(root: Path) -> Iterable[Path]:
    """稳定枚举目录中的普通文件。"""

    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


class PackageBuilder:
    """维护唯一源映射，并复制、脱敏、记录来源。"""

    def __init__(self) -> None:
        self.mappings: dict[str, Path] = {}
        self.provenance: list[dict[str, Any]] = []
        self.redaction_counts: Counter[str] = Counter()
        self.privacy_binary_exclusions: set[str] = set()

    @staticmethod
    def _binary_contains_private_marker(source: Path) -> bool:
        """识别无法安全文本脱敏的二进制文件中的本机身份信息。"""

        if source.suffix.lower() in TEXT_SUFFIXES:
            return False
        home = str(Path.home())
        username = Path.home().name
        needles = (
            home.encode("utf-8"),
            home.encode("utf-16le"),
            username.encode("utf-8"),
            username.encode("utf-16le"),
        )
        with source.open("rb") as handle:
            overlap = b""
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sample = overlap + chunk
                if any(needle and needle in sample for needle in needles):
                    return True
                overlap = sample[-512:]
        return False

    def add(self, source: Path, destination: str | None = None) -> None:
        source = source.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(f"缺少打包源文件：{source}")
        if not FAST_UNREDACTED_USER_REQUEST and self._binary_contains_private_marker(source):
            self.privacy_binary_exclusions.add(source.relative_to(ROOT).as_posix())
            return
        if destination is None:
            destination = source.relative_to(ROOT).as_posix()
        destination = safe_member(destination)
        existing = self.mappings.get(destination)
        if existing is not None and existing != source:
            raise ValueError(f"包内目标冲突：{destination}")
        self.mappings[destination] = source

    def add_tree(
        self,
        relative_root: str,
        predicate: Callable[[Path], bool] | None = None,
    ) -> None:
        root = ROOT / relative_root
        for path in iter_files(root):
            if is_cache_or_temporary(path):
                continue
            if predicate is not None and not predicate(path):
                continue
            self.add(path)

    @staticmethod
    def _decode_text(raw: bytes) -> tuple[str, str] | None:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return raw.decode("utf-16"), "utf-16"
            except UnicodeDecodeError:
                pass
        if raw[:4096].count(b"\x00") >= 8:
            try:
                return raw.decode("utf-16le"), "utf-16le"
            except UnicodeDecodeError:
                pass
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        return None

    @staticmethod
    def _redact(value: str) -> tuple[str, list[str]]:
        """删除本机绝对目录与用户名，不改研究状态和数值。"""

        project = str(ROOT)
        home = str(Path.home())
        username = Path.home().name
        replacements = [
            ("PROJECT_ROOT_JSON_ESCAPED", project.replace("\\", "\\\\"), "<PROJECT_ROOT>"),
            ("PROJECT_ROOT_LITERAL", project, "<PROJECT_ROOT>"),
            ("PROJECT_ROOT_FORWARD", project.replace("\\", "/"), "<PROJECT_ROOT>"),
            ("USER_HOME_JSON_ESCAPED", home.replace("\\", "\\\\"), "<USER_HOME>"),
            ("USER_HOME_LITERAL", home, "<USER_HOME>"),
            ("USER_HOME_FORWARD", home.replace("\\", "/"), "<USER_HOME>"),
            ("USERNAME_LITERAL", username, "<REDACTED_USER>"),
        ]
        output = value
        applied: list[str] = []
        for rule_id, old, new in replacements:
            if old and old in output:
                output = output.replace(old, new)
                applied.append(rule_id)

        regex_rules = (
            (
                "GENERIC_WINDOWS_USER_PATH_JSON_ESCAPED",
                re.compile(r"(?i)[A-Z]:\\\\Users\\\\[^\\\"\r\n]+\\\\"),
                "<USER_HOME>\\\\",
            ),
            (
                "GENERIC_WINDOWS_USER_PATH_LITERAL",
                re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"\r\n]+\\"),
                "<USER_HOME>\\",
            ),
            (
                "GENERIC_WINDOWS_USER_PATH_FORWARD",
                re.compile(r"(?i)[A-Z]:/Users/[^/\"\r\n]+/"),
                "<USER_HOME>/",
            ),
        )
        for rule_id, pattern, replacement in regex_rules:
            output, count = pattern.subn(lambda _: replacement, output)
            if count:
                applied.append(rule_id)
        return output, sorted(set(applied))

    def copy_all(self, package_root: Path) -> None:
        total = len(self.mappings)
        for index, (destination, source) in enumerate(sorted(self.mappings.items()), 1):
            target = package_root / Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            if FAST_UNREDACTED_USER_REQUEST:
                shutil.copy2(source, target)
                self.provenance.append(
                    {
                        "source_relative_path": source.relative_to(ROOT).as_posix(),
                        "package_member": destination,
                        "category": destination.split("/", 1)[0],
                        "source_size_bytes": source.stat().st_size,
                        "source_sha256": "",
                        "package_size_bytes": target.stat().st_size,
                        "package_sha256": "",
                        "text_redacted_or_reencoded": False,
                        "source_encoding": "COPIED_UNREDACTED",
                    }
                )
                if index % 500 == 0 or index == total:
                    print(f"已原样复制 {index}/{total} 个纳入文件。", flush=True)
                continue
            source_hash = sha256_file(source)
            redacted = False
            source_encoding = "binary"
            if source.suffix.lower() in TEXT_SUFFIXES:
                raw = source.read_bytes()
                decoded = self._decode_text(raw)
                if decoded is None:
                    shutil.copy2(source, target)
                else:
                    original_text, source_encoding = decoded
                    clean, rules = self._redact(original_text)
                    target.write_text(clean, encoding="utf-8")
                    redacted = bool(rules) or source_encoding not in {"utf-8", "utf-8-sig"}
                    for rule in rules:
                        self.redaction_counts[rule] += 1
            else:
                shutil.copy2(source, target)
            self.provenance.append(
                {
                    "source_relative_path": source.relative_to(ROOT).as_posix(),
                    "package_member": destination,
                    "category": destination.split("/", 1)[0],
                    "source_size_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                    "package_size_bytes": target.stat().st_size,
                    "package_sha256": sha256_file(target),
                    "text_redacted_or_reencoded": redacted,
                    "source_encoding": source_encoding,
                }
            )
            if index % 500 == 0 or index == total:
                print(f"已复制并核验 {index}/{total} 个纳入文件。", flush=True)


def collect_sources(builder: PackageBuilder) -> None:
    """按可上传但覆盖全部研究分支的规则收集源文件。"""

    for relative in ROOT_FILES:
        path = ROOT / relative
        if path.is_file():
            builder.add(path)

    for relative_root in FORMAL_SOURCE_ROOTS:
        builder.add_tree(relative_root)

    for relative_root in FULL_EVIDENCE_ROOTS:
        builder.add_tree(relative_root)

    builder.add_tree(
        "data/features",
        lambda path: path.stat().st_size <= SMALL_BINARY_LIMIT,
    )
    builder.add_tree(
        "data/staging",
        lambda path: (
            path.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".yaml", ".yml"}
            and path.stat().st_size <= SMALL_STRUCTURED_LIMIT
        )
        or (
            path.suffix.lower() == ".parquet"
            and path.stat().st_size <= SMALL_BINARY_LIMIT
        ),
    )
    builder.add_tree(
        "data/audit",
        lambda path: (
            path.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt"}
            and path.stat().st_size <= SMALL_STRUCTURED_LIMIT
        )
        or (
            path.suffix.lower() == ".parquet"
            and path.stat().st_size <= SMALL_BINARY_LIMIT
        ),
    )
    builder.add_tree(
        "paper",
        lambda path: path.stat().st_size <= SMALL_BINARY_LIMIT,
    )
    builder.add_tree(
        "output",
        lambda path: path.stat().st_size <= SMALL_STRUCTURED_LIMIT,
    )
    builder.add_tree(
        "outputs",
        lambda path: path.suffix.lower() == ".xlsx",
    )
    builder.add_tree(".codex-remote-attachments")

    for path in sorted((ROOT / "deliverables").glob("*")):
        if not path.is_file() or path.suffix.lower() == ".zip":
            continue
        if path.name.startswith(PACKAGE_BASENAME):
            continue
        builder.add(path)

    included_source_bytes = sum(path.stat().st_size for path in builder.mappings.values())
    if included_source_bytes > MAX_INCLUDED_SOURCE_BYTES:
        raise RuntimeError(
            "纳入源文件超过上传友好上限："
            f"{included_source_bytes} > {MAX_INCLUDED_SOURCE_BYTES} bytes"
        )


def count_directory_without_hashing(path: Path) -> tuple[int | None, int | None, int]:
    file_count = 0
    total_bytes = 0
    access_errors = 0

    def on_error(_: OSError) -> None:
        nonlocal access_errors
        access_errors += 1

    try:
        for current_root, _, files in os.walk(path, onerror=on_error):
            for name in files:
                candidate = Path(current_root) / name
                try:
                    total_bytes += candidate.stat().st_size
                    file_count += 1
                except OSError:
                    access_errors += 1
    except OSError:
        return None, None, access_errors + 1
    return file_count, total_bytes, access_errors


def exclusion_reason(relative: str, path: Path) -> tuple[str, bool]:
    """返回排除原因以及是否应为排除文件计算 SHA-256。"""

    lower = relative.lower()
    parts = PurePosixPath(relative).parts
    if relative == ".env":
        return "SECRET_ENV_FILE_EXCLUDED_NOT_HASHED", False
    if parts and parts[0] == ".codex":
        return "LOCAL_CODEX_CONFIGURATION_EXCLUDED_NOT_HASHED", False
    if "__pycache__" in parts or path.suffix.lower() in {".pyc", ".pyo", ".lock", ".tmp"}:
        return "GENERATED_CACHE_OR_LOCK_EXCLUDED_NOT_HASHED", False
    if parts and parts[0] == "data" and len(parts) > 1 and parts[1] == "raw":
        return "RAW_SOURCE_DATA_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("data/staging/"):
        if path.suffix.lower() in {".pdf", ".txt", ".html"}:
            return "RAW_ANNOUNCEMENT_DOCUMENT_OR_TEXT_EXCLUDED_WITH_VERIFIED_HASH", True
        return "LARGE_STAGING_PANEL_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("data/audit/"):
        return "LARGE_VISUAL_OR_PANEL_AUDIT_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("data/features/"):
        return "LARGE_REBUILDABLE_FEATURE_PANEL_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("data/derived/"):
        return "LARGE_DERIVED_PANEL_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("paper/"):
        return "LARGE_MUTABLE_FORWARD_PANEL_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("outputs/") and path.suffix.lower() == ".ndjson":
        return "RAW_STREAM_OR_TRANSCRIPT_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("review_packages/"):
        return "DUPLICATE_PRIOR_REVIEW_PACKAGE_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith("deliverables/") and path.suffix.lower() == ".zip":
        return "DUPLICATE_PRIOR_REVIEW_ZIP_EXCLUDED_WITH_VERIFIED_HASH", True
    if lower.startswith(".codex-remote-attachments/"):
        return "PRIVATE_USER_ATTACHMENT_PARAPHRASED_NOT_INCLUDED", True
    if lower.startswith("logs/"):
        return "VOLATILE_OPERATIONAL_LOG_EXCLUDED_WITH_VERIFIED_HASH", True
    if relative in {"long.txt", "share.html", "stream.txt", "stream_decoded.txt", "strings.txt", "="}:
        return "RAW_SCRAPE_DEBUG_OR_EMPTY_ARTIFACT_EXCLUDED_WITH_VERIFIED_HASH", True
    return "NONESSENTIAL_OR_DUPLICATE_ARTIFACT_EXCLUDED_WITH_VERIFIED_HASH", True


def enumerate_workspace_files(
    builder: PackageBuilder,
) -> tuple[list[Path], list[ExcludedDirectory]]:
    """枚举应登记的未纳入文件；环境、缓存和重复解压目录按目录汇总。"""

    included = {path.resolve() for path in builder.mappings.values()}
    excluded_files: list[Path] = []
    excluded_directories: list[ExcludedDirectory] = []

    for top in sorted(ROOT.iterdir()):
        if top.resolve() in included:
            continue
        name = top.name
        if name.startswith(".pytest-tmp-"):
            count, size, errors = count_directory_without_hashing(top)
            excluded_directories.append(
                ExcludedDirectory(
                    relative_path=name,
                    reason="PYTEST_TEMP_DIRECTORY_NOT_ENUMERATED",
                    file_count=count,
                    total_bytes=size,
                    access_error_count=errors,
                )
            )
            continue
        if name in TOP_LEVEL_NOT_ENUMERATED and top.is_dir():
            count, size, errors = count_directory_without_hashing(top)
            excluded_directories.append(
                ExcludedDirectory(
                    relative_path=name,
                    reason=TOP_LEVEL_NOT_ENUMERATED[name],
                    file_count=count,
                    total_bytes=size,
                    access_error_count=errors,
                )
            )
            continue
        if top.is_file():
            if top.resolve() not in included and not top.name.startswith(PACKAGE_BASENAME):
                excluded_files.append(top)
            continue
        if not top.is_dir():
            continue

        if name == "deliverables":
            for child in sorted(top.iterdir()):
                if child.is_file():
                    if child.resolve() not in included and not child.name.startswith(PACKAGE_BASENAME):
                        excluded_files.append(child)
                elif child.is_dir():
                    count, size, errors = count_directory_without_hashing(child)
                    excluded_directories.append(
                        ExcludedDirectory(
                            relative_path=child.relative_to(ROOT).as_posix(),
                            reason="PRIOR_REVIEW_EXTRACTION_DIRECTORY_DUPLICATE_NOT_ENUMERATED",
                            file_count=count,
                            total_bytes=size,
                            access_error_count=errors,
                        )
                    )
            continue

        for path in iter_files(top):
            if path.resolve() in included:
                continue
            if path.name.startswith(PACKAGE_BASENAME):
                continue
            excluded_files.append(path)

    unique_files = sorted(set(excluded_files), key=lambda item: item.relative_to(ROOT).as_posix())
    return unique_files, excluded_directories


def build_exclusion_inventory(
    builder: PackageBuilder,
) -> tuple[list[dict[str, Any]], list[ExcludedDirectory]]:
    """为证据型排除项计算哈希；缓存和秘密文件只记录原因。"""

    files, directories = enumerate_workspace_files(builder)
    work: list[tuple[Path, str, int, int]] = []
    base_rows: dict[str, dict[str, Any]] = {}
    cached_hashes: dict[tuple[str, int, int], str] = {}
    if EXCLUSION_HASH_CACHE_PATH.is_file():
        try:
            with EXCLUSION_HASH_CACHE_PATH.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    key = (
                        row["source_relative_path"],
                        int(row["size_bytes"]),
                        int(row["mtime_ns"]),
                    )
                    digest = row.get("sha256", "")
                    if digest:
                        cached_hashes[key] = digest
        except (OSError, KeyError, TypeError, ValueError):
            cached_hashes = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        reason, should_hash = exclusion_reason(relative, path)
        if relative in builder.privacy_binary_exclusions:
            reason = "BINARY_CONTAINS_LOCAL_IDENTITY_EXCLUDED_WITH_VERIFIED_HASH"
            should_hash = True
        if FAST_UNREDACTED_USER_REQUEST:
            should_hash = False
            reason = reason.replace(
                "WITH_VERIFIED_HASH", "NO_HASH_BY_USER_REQUEST"
            )
        try:
            stat = path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
        except OSError:
            size = None
            mtime_ns = None
            should_hash = False
            reason = "UNREADABLE_FILE_METADATA_ONLY"
        base_rows[relative] = {
            "source_relative_path": relative,
            "size_bytes": size,
            "sha256": "",
            "hash_status": "NOT_COMPUTED_BY_POLICY" if not should_hash else "PENDING",
            "exclusion_reason": reason,
        }
        if should_hash:
            cache_key = (relative, int(size), int(mtime_ns))
            cached = cached_hashes.get(cache_key)
            if cached:
                base_rows[relative]["sha256"] = cached
                base_rows[relative]["hash_status"] = "PASS_CACHE_VALIDATED_BY_SIZE_AND_MTIME"
            else:
                work.append((path, relative, int(size), int(mtime_ns)))

    def compute(item: tuple[Path, str, int, int]) -> tuple[str, str, str, int, int]:
        path, relative, size, mtime_ns = item
        try:
            return relative, sha256_file(path), "PASS", size, mtime_ns
        except OSError:
            return relative, "", "FAILED_UNREADABLE", size, mtime_ns

    if FAST_UNREDACTED_USER_REQUEST:
        print(
            f"按用户要求跳过排除项逐文件哈希；仅登记 {len(base_rows)} 个排除文件。",
            flush=True,
        )
    else:
        print(
            f"开始核验 {len(work)} 个未命中缓存的排除证据；"
            f"已复用 {len(cached_hashes)} 条候选缓存。",
            flush=True,
        )
    cache_rows = [
        {
            "source_relative_path": key[0],
            "size_bytes": key[1],
            "mtime_ns": key[2],
            "sha256": digest,
        }
        for key, digest in cached_hashes.items()
    ]
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        for index, (relative, digest, status, size, mtime_ns) in enumerate(
            executor.map(compute, work), 1
        ):
            base_rows[relative]["sha256"] = digest
            base_rows[relative]["hash_status"] = status
            if status == "PASS":
                cache_rows.append(
                    {
                        "source_relative_path": relative,
                        "size_bytes": size,
                        "mtime_ns": mtime_ns,
                        "sha256": digest,
                    }
                )
            if index % 5000 == 0 or index == len(work):
                print(f"已核验排除证据 {index}/{len(work)} 个文件。", flush=True)
                write_csv(
                    EXCLUSION_HASH_CACHE_PATH,
                    ["source_relative_path", "size_bytes", "mtime_ns", "sha256"],
                    sorted(cache_rows, key=lambda row: row["source_relative_path"]),
                )

    rows = [base_rows[key] for key in sorted(base_rows)]
    failed = [row for row in rows if row["hash_status"] == "FAILED_UNREADABLE"]
    if failed:
        raise RuntimeError(f"有 {len(failed)} 个应哈希的排除证据不可读")
    return rows, directories


def load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def readme_text(
    source_count: int,
    source_bytes: int,
    excluded_count: int,
    excluded_bytes: int,
) -> str:
    return f"""# 全项目量化研究：GPT Pro 独立审阅包

## 这是什么

- 快照日期：2026-08-25（Asia/Shanghai）。
- 包内直接纳入 {source_count:,} 个项目源文件，源文件合计 {source_bytes:,} 字节。
- 另登记 {excluded_count:,} 个未直接复制的文件，合计 {excluded_bytes:,} 字节；按用户要求未对这些排除项逐文件计算哈希。
- 本包覆盖 510300、A股横截面/事件研究、国际化券商研究、公开语料候选研究、前瞻采集与研究治理。
- 本包按用户要求采用未脱敏快速交付；只做一次ZIP CRC与整包SHA-256，不证明任何研究通过，更不授权仓位、订单、券商连接或实盘。

## 当前最高层判断

`NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET`

同时存在的状态不能互相覆盖：

- 历史研究中有大量 `REJECTED_FROZEN`、`NO_VIEW`、`DISCOVERY_ONLY`；
- 510300前瞻总状态为 `COLLECTING_FORWARD_WITH_BLOCKERS`；
- T-only V1.1 仍为 `NO_VIEW_UNTIL_FORWARD_MATURITY`；
- A股 ORJ V2 的成本后执行验收为 `REJECTED_ORJ_NET_EXCESS_EXECUTION_V2`；
- 国际化券商利润桥为 `NO_VIEW`，收益检验仍关闭。

## 必读顺序

1. `00_README_FIRST.md`
2. `01_GPT_PRO_REVIEW_PROMPT.md`
3. `02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md`
4. `03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md`
5. `04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md`
6. `05_EXCLUSIONS_REPRODUCTION_AND_SAFETY.md`
7. `12_STATUS_SOURCE_INDEX.csv`
8. 再按提示词进入 `docs/`、`reports/`、`config/`、`research/`、`scripts/`、`tests/` 与小型数据账本。
9. 最后用 `06_INCLUDED_SOURCE_PROVENANCE.csv`、`07_EXCLUDED_SOURCE_INVENTORY.csv`、`08_INCLUDED_FILE_INDEX.csv` 查看覆盖范围。

## 使用前必须知道

- 不同研究的证据截止日不同，必须读各自协议与回执；不得虚构一个统一市场截止日。
- `docs/DECISIONS.md` 是按时间追加的历史决策，内部存在被后续决定取代的旧边界；不能把所有段落同时视为现行授权。
- 510300主线、A股股票Alpha研究、国际化券商研究和公开语料研究是不同分支；一个分支的证据不得自动授权另一个分支。
- 原始数据未全部装入 ZIP。未恢复 `07_EXCLUDED_SOURCE_INVENTORY.csv` 中的大型数据时，不能把“无法从零重跑”解释成研究逻辑失败，也不能把缺失材料假装成已审阅。
- 包内保留本机用户名、绝对路径和私人聊天截图；这是用户明确要求的未脱敏版本，外发前由用户自行确认接收方。

## 快速包校验范围

构建时只执行一次ZIP CRC检查，并在ZIP外生成整包 `.sha256`。包内文件索引只记录路径和大小，不提供逐文件哈希。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 的独立审阅提示词

你是一名独立的量化研究负责人、反方审计员和研究组合经理。请先完整读取本包的 00—05、12号文件，再按证据路径审阅正文。你的目标不是从旧历史中“找一个最好看的策略”，而是决定下一阶段有限时间、资金和数据预算应该投向哪里，以及哪些方向应该明确停止。

## 一、审阅原则

1. 以冻结协议、运行回执、事件/决策日账本、成本模型和当前状态为准；叙述性总结只能作为导航。
2. 严格区分 `REJECTED_FROZEN`、`NO_VIEW`、`INCOMPLETE`、`DISCOVERY_ONLY`、`PAPER/SHADOW` 与 `LIVE_AUTHORIZED`。缺数据不是零收益，退出码0也不自动是成功。
3. 不得通过改窗口、阈值、市场、年份、成本、标签、基准或分组去救援一个已拒绝版本。真正不同的问题必须另立新版本、事前冻结并给出唯一停止条件。
4. 不得把相关性、Rank IC、事件均值、胜率、利润因子、可复现性或少数正年份直接称为可交易Alpha。
5. 所有建议必须考虑点时可得性、未来泄漏、幸存者偏差、下一可交易时点、涨跌停/停牌/退市、整手、最低佣金、滑点、现金收益、容量和时间稳定性。
6. 不输出买卖指令、目标仓位、个股名单或券商连接方案；只讨论研究方向、证据计划和停止规则。

## 二、必须先解决的矛盾

- `docs/DECISIONS.md` 中有被后续决定取代的旧范围；请建立按日期和版本的有效性判断，不要把旧授权复活。
- 510300现货/现金是交易授权边界；A股横截面/事件研究、国际化券商研究与知乎语料属于隔离的研究分支。
- 历史正点估计与执行验收可能方向相反。请优先解释执行钟、资本加权、成本、不可成交和时间稳定性，而不是偏爱漂亮的父研究指标。
- 当前前瞻采集存在程序失败、协议哈希漂移、陈旧输入和样本未成熟；这些是治理/数据事实，不是策略收益结论。

## 三、请逐方向分类

对包内每个实质性研究方向至少归入以下一类，并给出直接证据路径：

1. `STOP_PERMANENTLY`：问题或机制已经被强反证，继续投入不值得；
2. `STOP_CURRENT_VERSION_RETAIN_EVIDENCE`：该版本冻结拒绝，但证据可作为完全不同新问题的先验；
3. `COMPLETE_MISSING_EVIDENCE`：尚未裁决，先补有限且关键的证据；
4. `CONTINUE_STRICT_FORWARD_ONLY`：不得再看历史调参，只积累冻结后的真正前向样本；
5. `ONE_NEW_FROZEN_TEST`：只允许一个最小的新版本检验，并预先写明淘汰线；
6. `OUT_OF_SCOPE_OR_SEPARATE_RESEARCH`：与510300交易主线隔离，不得映射仓位。

## 四、重点回答

1. 510300历史方向、估值、宏观、技术、日内、主申赎/IOPV/PCF、期权信息桥、鱼中/板块驱动、T-only 前瞻等分支，哪些彻底停止，哪些只继续采集，哪些值得唯一一次新冻结检验？
2. A股低波/缩波、残差趋势、低MAX、低换手、收益季节性、价格延迟、OVP、FXV、ORJ、利润预告、现金要约、现金选择权、可转债配售等方向，分别处于“负结论、NO_VIEW、未完成”中的哪一种？
3. 为什么 ORJ 父研究的正相关和尾档差没有转化为 V2 成本后可靠净超额？这对未来事件研究的设计有什么一般教训？
4. 国际化券商想法是否值得继续补 G2/G3，还是数据披露不足使其边际价值太低？把“跨境需求、汇率、美股长时段交易”拆成哪些可证伪链条，哪一环失败就停止？
5. 知乎公开语料应只作为候选生成器，还是有某一条定义足够完整、值得另立一次严格协议？不得补写作者没有给出的参数后冒充原观点。
6. 当前最稀缺的是什么：新策略想法、可靠点时数据、真正前向样本、执行可行性、工程稳定性，还是研究治理？请给证据排序。

## 五、输出格式

请用中文输出以下六部分：

1. **一页执行结论**：当前总判断、最重要的三件事、最应该停止的三件事；
2. **全方向裁决表**：分支、方向/版本、当前状态、证据路径、分类、继续/停止理由、关键缺口；
3. **优先级组合**：最多三个未来方向，按预期信息价值排序；写出为什么不是历史收益最好看者优先；
4. **30/90/180天路线图**：每阶段最多两个并行研究流，列输入、负责人能力、成本/时间、可交付物、硬停止条件；
5. **明确不做清单**：具体到策略家族、数据救援、调参方式和执行越权；
6. **给用户的决策问题**：只列真正需要用户选择、会改变研究范围或预算的问题；其余自行做证据判断。

最后给出唯一总建议：`PAUSE_AND_FIX_FOUNDATION`、`CONTINUE_FORWARD_ONLY`、`RUN_ONE_NEW_FROZEN_TEST` 或 `STOP_RESEARCH_PROGRAM`，并解释为何。
"""


def user_thoughts_text() -> str:
    return """# 用户思考、硬约束与开放问题

本文件把“用户明确说过的边界”“从既有项目决策可确认的偏好”和“曾讨论但尚未验证的想法”分开。它不是研究结果，也不要求审阅者赞同。

## A. 明确硬约束

1. **交易对象边界**：现行交易主线只允许 510300 ETF 现货与现金；不使用期权、期货、融资融券、卖空、杠杆或成分股。H00300只可作基准，不是模型授权输入。历史上更宽的授权已经被后续决定取代。
2. **研究与执行分离**：研究候选、Paper/Shadow、真实持仓映射、订单生成、券商连接和实盘是不同状态；没有明确新授权就保持全部执行能力关闭。
3. **接受负结论**：拒绝是成果。已经冻结拒绝的缩波/P3、R6、ORJ V2 等分支不能靠有利年份、阈值、窗口、代理、市场列表或成本口径救援。
4. **目标函数简化**：未来新研究不硬求高胜率、高盈亏比、高利润因子、高频率、大容量或高流动性；允许低频、低胜率和小容量，但点时、成本后、可执行净超额必须可靠为正，且不能由单一年份或少数事件驱动。
5. **证据优先**：先冻结数据合同、候选、成本、基准和停止规则，再验证；`NO_VIEW`、`FAILED`、`SKIPPED`、未成熟和删失状态必须原样保留。

## B. 用户持续关注的问题

- 能否识别“鱼中”而不是事后起点：板块驱动、传播、衰竭与指数未来分布必须分层，确认一个周期不等于有预测资格。
- 估值、趋势、风险与执行信息如何分工：慢变量只能提供先验或风险背景，不能自动变成短期买卖信号。
- PCF、IOPV、折溢价、申赎和盘口能否提供真正独立的新信息；在前瞻样本未成熟、采集程序不稳定或冻结哈希漂移时，宁可停止输出，也不补数。
- 是否存在低频、小容量但稳定正期望的A股事件或横截面策略；若执行成本把父研究优势吃掉，应停止而不是继续调参。

## C. 讨论过但仍只是待验证假设的想法

用户曾提供一张私人聊天截图，其中讨论了中国上市券商的国际业务、跨境金融需求、日元/美元汇率和美股延长交易时段等主题，也夹带了主观宏观判断。用户本轮明确要求不脱敏，因此原图已纳入 `.codex-remote-attachments/`；以下仍把它严格改写为待验证的研究问题，而不把私人对话当事实证据：

`跨境需求增长 → 牌照和客户渠道 → 业务量/份额 → 海外经常性利润 → 集团EPS/ROE → 估值是否未充分反映`

任何一环缺失都不能推出“买国际化券商”。美股长时段交易在上线、亚洲时段流量与盈利归属可验证前只能是未来选择权；汇率暴露必须区分客户业务、换算影响和方向性头寸。该分支与510300完全隔离，当前状态仍为研究/补证，不是持仓建议。

## D. 对审阅者的开放问题

1. 如果只能保留两个研究流，应该把资源放在“修复并继续真正前瞻采集”“补齐一个契约型事件证据”“做一次完全不同的新冻结检验”中的哪两个？
2. 哪些看似有经济机制的方向，实际上因样本量、披露质量、成本或可成交性永远不值得继续？
3. 怎样设置一个能阻止无限研究循环的项目级预算和停止机制，而不因为一次 `NO_VIEW` 错杀可补证方向？
4. 哪些数据值得付费，哪些免费数据的缺口已经足以让研究停止？请把建议与具体候选和信息价值绑定。
"""


def project_map_text() -> str:
    current = load_json("reports/audit/priority_forward_research_status_current.json")
    generated_at = current.get("generated_at", "NO_VIEW")
    directions = current.get("directions", {}) if isinstance(current.get("directions"), dict) else {}
    pcf = directions.get("primary_market_pcf_iopv", {}) if isinstance(directions, dict) else {}
    industry = directions.get("industry_expectation_gap", {}) if isinstance(directions, dict) else {}
    low_vol = directions.get("orthogonal_low_vol_replication", {}) if isinstance(directions, dict) else {}
    pcf_gates = pcf.get("gates", {}) if isinstance(pcf, dict) else {}
    quality = pcf_gates.get("quality_audit", {}) if isinstance(pcf_gates, dict) else {}
    return f"""# 项目地图与当前裁决导航

## 1. 510300交易主线

- 现行执行边界：只允许510300 ETF现货与现金，但本包不授权任何执行动作。
- 多个历史方向、估值、宏观、日线、分钟、机器学习与R6家族已拒绝；必须按各自冻结状态审阅，不能重新拼接成“组合策略”。
- 当前前瞻总状态文件生成于 `{generated_at}`，总状态为 `{current.get('overall_status', 'NO_VIEW')}`。
- PCF/IOPV：`{pcf.get('status', 'NO_VIEW')}`；完整质量日 `{pcf.get('full_coverage_days', 'NO_VIEW')}/{quality.get('required', 'NO_VIEW')}`；最近任务 `{pcf.get('task_evidence', {}).get('run_status', 'NO_VIEW') if isinstance(pcf.get('task_evidence'), dict) else 'NO_VIEW'}`。
- 行业预期差：`{industry.get('status', 'NO_VIEW')}`；独立原点 `{industry.get('origin_cluster_count', 'NO_VIEW')}`，成熟原点 `{industry.get('mature_origin_cluster_count', 'NO_VIEW')}`。
- 正交低波复制：`{low_vol.get('status', 'NO_VIEW')}`；治理未通过前不得启动。
- 2026-08-25期权盘口采集在供应商请求前因冻结测试文件哈希漂移退出；O1—O4不得写成0或最新观测。
- T-only V1.1只有3个新交易日、0个闭合周期，保持 `NO_VIEW_UNTIL_FORWARD_MATURITY`。

## 2. A股横截面与事件研究分支

- 极端缩波V2与机制候选V2.1均为0/6，终态为 `NO_REPEATABLE_CONDITIONAL_EXCESS_IDENTIFIED`；P3及相邻条件救援关闭。
- 残差趋势、价格延迟、OVP等多条纯价量/横截面方向已出现冻结反证；低MAX和低换手有局部排序证据但旧协议仍未通过。
- ORJ父研究曾有正Rank IC和正尾档差；独立V2按下一开盘、一手、最低佣金、滑点、不可成交和资本加权后，2021—2025压力成本行业净超额为负，状态 `REJECTED_ORJ_NET_EXCESS_EXECUTION_V2`。
- 利润预告、全面/部分现金要约、现金选择权、可转债一手配售必须区分“经济拒绝”“样本不足”“行情或生命周期证据缺失”；不得统一写成失败或成功。

## 3. 国际化券商独立分支

- 研究命题是跨境需求能否经过牌照、业务量、经常性利润和资本回报传导到集团价值，不是主观宏观判断。
- 当前G2结构只有部分公司年度通过，G3完整利润桥为0/30；`G6_PRICING`、收益检验、510300输入、仓位和订单全部禁止。
- 审阅重点应是补证成本与可得性，而不是提前做股价回测。

## 4. 公开语料候选分支

- 知乎可观测语料不是完整语料，互动量不是正确率，观点缺参数时不能由研究者事后补成最有利策略。
- 只有原文定义足够完整、能登记时点与停止规则的观点，才可能进入另立协议；其余保持知识整理或机制候选。

## 5. 工程与治理层

- 当前存在PowerShell解析失败、过期输入、Windows持久任务未验证、冻结哈希漂移和前瞻样本未成熟。
- 这些故障会污染“是否真正收集到数据”的判断，应先作为研究基础设施问题处理；但修好程序不等于策略有效。
- 外部审阅应把“修复采集”“继续观察”“新策略研究”分别计价，避免工程进展被误报成Alpha进展。
"""


def branch_boundaries_text() -> str:
    return """# 分支边界与冲突裁决规则

## 有效性优先级

同一主题出现矛盾时，按以下顺序判断：

1. 后续明确取代旧范围的用户决定；
2. 冻结协议与对应不可变清单；
3. 同一次运行的回执、状态JSON和结果账本；
4. 独立复算、测试和哈希验证；
5. 叙述性报告与旧聊天摘要。

旧文件不删除是为了审计，不代表它仍有效。必须引用具体版本和状态，禁止把不同版本的最佳部分拼成一个从未冻结过的策略。

## 必须保持隔离

| 分支 | 允许输出 | 禁止跨越 |
|---|---|---|
| 510300历史/前瞻研究 | 数据状态、研究信号、Paper/Shadow账本（仅在其协议允许时） | 自动仓位、订单、券商连接、把A股个股研究混入510300 |
| A股横截面/事件研究 | 历史发现、执行验收、缺口裁决 | 个股订单、把预测证据称为可交易Alpha |
| 国际化券商 | 数据底座、结构/利润桥、未来研究设计 | G3前做G6收益、映射510300或持仓 |
| 公开语料 | 可观察语料、观点登记、候选协议 | 跟单、补写参数、把互动量当收益 |
| 工程运维 | 采集、状态、回执、错误诊断 | 把进程存在或退出码0称为数据/策略成功 |

## 状态语义

- `NO_VIEW`：不能形成合格视图；不是看空、空仓、零收益或失败策略。
- `FAILED` / `PROGRAM_FAILED`：本次程序或数据任务失败；不能沿用旧输出冒充当天结果。
- `SKIPPED`：按协议未运行或供应商未就绪；不能改写为成功。
- `REJECTED_FROZEN`：该版本在冻结门槛下被拒绝；只能另立真正不同的问题。
- `DISCOVERY_ONLY`：允许研究，不允许交易映射。
- `PAPER/SHADOW`：也不等于实盘授权；必须单独验证持仓、订单和券商边界。
"""


def exclusions_text(
    excluded_rows: list[dict[str, Any]],
    excluded_directories: list[ExcludedDirectory],
) -> str:
    reasons = Counter(row["exclusion_reason"] for row in excluded_rows)
    reason_lines = "\n".join(
        f"- `{reason}`：{count:,} 个文件" for reason, count in sorted(reasons.items())
    )
    return f"""# 排除项、复现边界与安全说明

## 为什么不是把约13GB原样塞进ZIP

原项目绝大部分体积来自原始行情、官方公告PDF/逐页文本、大型特征/中间面板、视觉审计页、重复旧审阅包、虚拟环境和缓存。原样外发会造成上传困难、重复证据、隐私风险和“文件多等于结论可靠”的错觉。

本包采用以下口径：

- 代码、协议、测试、报告、状态回执、小型账本和关键派生结果尽量直接纳入；
- 大型或原始证据不复制，只登记相对路径、字节数与原因；
- `.env`只记录存在和排除原因，不计算或外发内容哈希；
- 私人聊天截图按用户要求原样纳入，并在02号文件说明其研究语境；
- `.git`、`.venv`、缓存和临时目录只做目录级汇总，不当研究材料；
- 旧审阅ZIP和旧解压目录不重复嵌套，只保留历史ZIP索引。

## 文件级排除统计

{reason_lines}

另有 {len(excluded_directories):,} 个环境、缓存或重复解压目录只在 `11_EXCLUDED_DIRECTORY_SCOPES.csv` 汇总。

## 复现边界

1. `08_INCLUDED_FILE_INDEX.csv` 只记录包内路径和大小，不是逐文件完整性证明。
2. `06_INCLUDED_SOURCE_PROVENANCE.csv` 记录原样复制关系，不进行文本脱敏或编码转换。
3. `07_EXCLUDED_SOURCE_INVENTORY.csv` 未计算排除项哈希；未恢复大型原始数据并另行校验时不能声称完成了从零全量复现。
4. 本次没有重跑策略、重新估计模型或改冻结协议。旧测试结果必须回到各自回执；本包只做构包验证。
5. 如果外部模型无法读取Parquet，应先读同研究的MD/JSON/CSV报告，不得把工具限制写成数据不存在。
"""


REPRODUCTION_NOTES = """# 独立复核说明

## 包内索引

快速包不提供逐文件验证器。`08_INCLUDED_FILE_INDEX.csv` 只列相对路径和大小；ZIP外的 `.sha256` 用于核对整包文件。

## 研究复现

- 先读取具体研究的配置、冻结清单与运行回执。
- 从 `07_EXCLUDED_SOURCE_INVENTORY.csv` 恢复对应大型源文件，并逐项核对大小和SHA-256。
- 使用项目根目录的 `requirements.txt` 重建环境；不要复用本机 `.venv`。
- 严格按原研究命令运行，保留失败、NO_VIEW、SKIPPED和删失状态。
- 任何实现错误只能发布更正版本和差异说明，不能静默覆盖旧结果。

本包构建过程没有执行全量研究回归测试，也没有做隐私扫描、逐成员哈希或全新解压复核。构包脚本自身只完成语法检查、一次ZIP CRC检查和整包SHA-256。
"""


UPLOAD_MESSAGE = """我上传的是一个按我要求未脱敏、采用快速校验的全项目量化研究审阅包；构建端只做了一次ZIP CRC和整包SHA-256。请先解压并完整阅读 00_README_FIRST.md、01_GPT_PRO_REVIEW_PROMPT.md、02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md、03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md、04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md、05_EXCLUSIONS_REPRODUCTION_AND_SAFETY.md 和 12_STATUS_SOURCE_INDEX.csv，然后按 01_GPT_PRO_REVIEW_PROMPT.md 的六部分格式输出。请不要调参救援已拒绝版本，不要把NO_VIEW当负收益，不要输出仓位或订单；核心任务是决定下一阶段应投入哪些方向、明确停止哪些方向，并给出30/90/180天路线图和硬停止条件。"""


VERIFY_SCRIPT = r'''"""验证解压后的 GPT Pro 审阅包。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "08_FILE_MANIFEST_SHA256.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"文件集合不一致：missing={missing}, extra={extra}")
    failures = []
    for row in rows:
        path = ROOT / Path(row["relative_path"])
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"大小不符：{row['relative_path']}")
        if sha256_file(path) != row["sha256"]:
            failures.append(f"哈希不符：{row['relative_path']}")
    if failures:
        raise RuntimeError("；".join(failures))
    print(f"验证通过：{len(rows)} 个清单文件，0缺失，0额外，SHA-256全部一致。")


if __name__ == "__main__":
    main()
'''


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(package_root: Path, builder: PackageBuilder) -> None:
    write_csv(
        package_root / "06_INCLUDED_SOURCE_PROVENANCE.csv",
        [
            "source_relative_path",
            "package_member",
            "category",
            "source_size_bytes",
            "source_sha256",
            "package_size_bytes",
            "package_sha256",
            "text_redacted_or_reencoded",
            "source_encoding",
        ],
        sorted(builder.provenance, key=lambda row: row["package_member"]),
    )


def write_exclusions(
    package_root: Path,
    rows: list[dict[str, Any]],
    directories: list[ExcludedDirectory],
) -> None:
    write_csv(
        package_root / "07_EXCLUDED_SOURCE_INVENTORY.csv",
        [
            "source_relative_path",
            "size_bytes",
            "sha256",
            "hash_status",
            "exclusion_reason",
        ],
        rows,
    )
    write_csv(
        package_root / "11_EXCLUDED_DIRECTORY_SCOPES.csv",
        [
            "relative_path",
            "reason",
            "file_count",
            "total_bytes",
            "access_error_count",
        ],
        [directory.__dict__ for directory in directories],
    )


def write_status_index(package_root: Path) -> None:
    rows = []
    for branch, status, evidence_path, note in STATUS_SOURCE_ROWS:
        source = ROOT / evidence_path
        rows.append(
            {
                "branch": branch,
                "current_status": status,
                "evidence_path": evidence_path,
                "evidence_present": source.is_file(),
                "source_sha256": "",
                "note": note,
            }
        )
    write_csv(
        package_root / "12_STATUS_SOURCE_INDEX.csv",
        [
            "branch",
            "current_status",
            "evidence_path",
            "evidence_present",
            "source_sha256",
            "note",
        ],
        rows,
    )


def write_prior_review_index(
    package_root: Path,
    excluded_rows: list[dict[str, Any]],
) -> None:
    rows = []
    for row in excluded_rows:
        relative = str(row["source_relative_path"])
        if not relative.lower().endswith(".zip"):
            continue
        if not (
            relative.lower().startswith("deliverables/")
            or relative.lower().startswith("review_packages/")
        ):
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "hash_status": row["hash_status"],
                "superseded_or_duplicate": True,
            }
        )
    write_csv(
        package_root / "13_PRIOR_REVIEW_PACKAGE_INDEX.csv",
        [
            "relative_path",
            "size_bytes",
            "sha256",
            "hash_status",
            "superseded_or_duplicate",
        ],
        rows,
    )


def privacy_scan(package_root: Path) -> dict[str, Any]:
    """扫描用户名、绝对用户路径、密钥标记和高风险凭据字面量。"""

    username = Path.home().name
    home = str(Path.home())
    forbidden_needles = {
        "USER_HOME_UTF8": home.encode("utf-8"),
        "USER_HOME_UTF16LE": home.encode("utf-16le"),
        "USERNAME_UTF8": username.encode("utf-8"),
        "PRIVATE_KEY_MARKER": b"BEGIN " + b"PRIVATE KEY",
        "OPENSSH_PRIVATE_KEY_MARKER": b"BEGIN " + b"OPENSSH " + b"PRIVATE KEY",
    }
    high_risk_patterns = {
        "OPENAI_STYLE_SECRET": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        "GITHUB_STYLE_SECRET": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
        "AWS_ACCESS_KEY": re.compile(rb"AKIA[0-9A-Z]{16}"),
        "BEARER_TOKEN_LITERAL": re.compile(rb"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{24,}"),
    }
    credential_assignment = re.compile(
        r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|tushare[_-]?token)"
        r"\s*[:=]\s*['\"]"
        r"(?!<|your_|test|dummy|example|fake|none|null|replace|env)"
        r"[A-Za-z0-9_./+\-=]{16,}['\"]"
    )
    hits: list[dict[str, str]] = []
    scanned_files = 0
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        scanned_files += 1
        if path.name.lower() == ".env" or relative.lower().endswith("/.env"):
            hits.append({"rule_id": "ENV_FILE_FORBIDDEN", "relative_path": relative})
        raw = path.read_bytes()
        for rule_id, needle in forbidden_needles.items():
            if needle and needle in raw:
                hits.append({"rule_id": rule_id, "relative_path": relative})
        for rule_id, pattern in high_risk_patterns.items():
            if pattern.search(raw):
                hits.append({"rule_id": rule_id, "relative_path": relative})
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = ""
            if credential_assignment.search(text):
                hits.append(
                    {"rule_id": "LITERAL_CREDENTIAL_ASSIGNMENT", "relative_path": relative}
                )
    deduplicated = sorted(
        {json.dumps(hit, ensure_ascii=False, sort_keys=True) for hit in hits}
    )
    final_hits = [json.loads(item) for item in deduplicated]
    return {
        "status": "PASS_ZERO_PRIVACY_OR_CREDENTIAL_HITS" if not final_hits else "FAIL_PRIVACY_SCAN",
        "scanned_file_count": scanned_files,
        "hit_count": len(final_hits),
        "hits": final_hits,
        "rules": sorted(
            [
                *forbidden_needles,
                *high_risk_patterns,
                "ENV_FILE_FORBIDDEN",
                "LITERAL_CREDENTIAL_ASSIGNMENT",
            ]
        ),
    }


def write_file_index(package_root: Path) -> list[dict[str, Any]]:
    """快速包只登记包内文件路径与大小，不计算逐文件哈希。"""

    index_path = package_root / "08_INCLUDED_FILE_INDEX.csv"
    rows: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == index_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    write_csv(
        index_path,
        ["relative_path", "size_bytes"],
        rows,
    )
    return rows


def verify_zip_crc_only(zip_path: Path) -> dict[str, Any]:
    """按用户要求只做一次ZIP CRC和重复成员检查。"""

    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP含重复成员")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC失败：{bad_member}")
    return {
        "status": "PASS_SINGLE_ZIP_CRC_AND_DUPLICATE_MEMBER_CHECK",
        "member_count": len(infos),
        "duplicate_member_count": 0,
    }


def verify_tree(package_root: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = package_root / "08_FILE_MANIFEST_SHA256.csv"
    expected = {str(row["relative_path"]) for row in rows}
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise RuntimeError(
            f"文件集合不一致：missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for row in rows:
        path = package_root / Path(str(row["relative_path"]))
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"文件大小不一致：{row['relative_path']}")
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"文件哈希不一致：{row['relative_path']}")


def create_zip(package_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_root).as_posix()
            archive.write(path, f"{INTERNAL_ROOT}/{relative}")
    os.replace(temporary, destination)


def verify_zip_stream(zip_path: Path, package_root: Path) -> dict[str, Any]:
    expected = {
        f"{INTERNAL_ROOT}/{path.relative_to(package_root).as_posix()}": path
        for path in package_root.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC失败：{bad_member}")
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP含重复成员")
        if set(names) != set(expected):
            raise RuntimeError("ZIP成员集合与构包目录不一致")
        for index, info in enumerate(infos, 1):
            with archive.open(info, "r") as handle:
                archive_hash = sha256_stream(handle)
            if archive_hash != sha256_file(expected[info.filename]):
                raise RuntimeError(f"ZIP字节流哈希不一致：{info.filename}")
            if index % 500 == 0 or index == len(infos):
                print(f"已核验ZIP成员 {index}/{len(infos)} 个。", flush=True)
    return {
        "status": "PASS_ZIP_CRC_MEMBER_SET_AND_STREAM_SHA256",
        "member_count": len(expected),
        "duplicate_member_count": 0,
    }


def verify_fresh_extraction(
    zip_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qra_extract_") as temp_name:
        extraction_root = Path(temp_name)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise RuntimeError(f"ZIP含非法路径：{info.filename}")
            archive.extractall(extraction_root)
        extracted_package = extraction_root / INTERNAL_ROOT
        verify_tree(extracted_package, rows)
        scan = privacy_scan(extracted_package)
        if scan["hit_count"]:
            raise RuntimeError(
                "全新解压后的隐私扫描失败："
                + json.dumps(scan["hits"], ensure_ascii=False, sort_keys=True)
            )
    return {
        "status": "PASS_FRESH_EXTRACTION_MANIFEST_AND_PRIVACY_VERIFIED",
        "manifest_file_count": len(rows),
        "privacy_hit_count": 0,
    }


def summarize_categories(builder: PackageBuilder) -> list[dict[str, Any]]:
    counters: dict[str, dict[str, int]] = defaultdict(lambda: {"file_count": 0, "source_bytes": 0})
    for destination, source in builder.mappings.items():
        category = destination.split("/", 1)[0]
        counters[category]["file_count"] += 1
        counters[category]["source_bytes"] += source.stat().st_size
    return [
        {
            "category": category,
            "file_count": values["file_count"],
            "source_bytes": values["source_bytes"],
        }
        for category, values in sorted(counters.items())
    ]


def main() -> None:
    started_at = datetime.now(TIME_ZONE).isoformat()
    builder = PackageBuilder()
    collect_sources(builder)
    source_bytes = sum(path.stat().st_size for path in builder.mappings.values())
    print(
        f"纳入源文件 {len(builder.mappings)} 个，共 {source_bytes / 1024 / 1024:.2f} MiB。",
        flush=True,
    )
    excluded_rows, excluded_directories = build_exclusion_inventory(builder)
    excluded_bytes = sum(int(row["size_bytes"] or 0) for row in excluded_rows)
    hashed_exclusion_count = sum(
        str(row["hash_status"]).startswith("PASS") for row in excluded_rows
    )
    unhashed_secret_count = sum(
        row["exclusion_reason"] == "SECRET_ENV_FILE_EXCLUDED_NOT_HASHED"
        for row in excluded_rows
    )

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qra_build_") as temp_name:
        package_root = Path(temp_name) / INTERNAL_ROOT
        package_root.mkdir(parents=True, exist_ok=True)
        builder.copy_all(package_root)

        write_text(
            package_root / "00_README_FIRST.md",
            readme_text(
                len(builder.mappings),
                source_bytes,
                len(excluded_rows),
                excluded_bytes,
            ),
        )
        write_text(package_root / "01_GPT_PRO_REVIEW_PROMPT.md", review_prompt_text())
        write_text(
            package_root / "02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md",
            user_thoughts_text(),
        )
        write_text(
            package_root / "03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md",
            project_map_text(),
        )
        write_text(
            package_root / "04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md",
            branch_boundaries_text(),
        )
        write_text(
            package_root / "05_EXCLUSIONS_REPRODUCTION_AND_SAFETY.md",
            exclusions_text(excluded_rows, excluded_directories),
        )
        write_text(package_root / "14_REPRODUCTION_NOTES.md", REPRODUCTION_NOTES)
        write_text(package_root / "15_UPLOAD_MESSAGE.txt", UPLOAD_MESSAGE + "\n")
        write_provenance(package_root, builder)
        write_exclusions(package_root, excluded_rows, excluded_directories)
        write_status_index(package_root)
        write_prior_review_index(package_root, excluded_rows)

        write_json(
            package_root / "09_CHECK_SCOPE.json",
            {
                "status": "FAST_UNREDACTED_BUILD_BY_USER_REQUEST",
                "text_redaction_performed": False,
                "privacy_scan_performed": False,
                "per_file_sha256_performed": False,
                "fresh_extraction_performed": False,
                "zip_crc_performed": True,
                "whole_zip_sha256_performed": True,
            },
        )
        write_json(
            package_root / "10_BUILD_SCOPE_AND_COUNTS.json",
            {
                "schema_version": "1.0.0",
                "package_id": PACKAGE_BASENAME,
                "snapshot_date": "2026-08-25",
                "status": "FAST_UNREDACTED_SOURCE_SELECTION_COMPLETE",
                "included_source_file_count": len(builder.mappings),
                "included_source_bytes": source_bytes,
                "included_categories": summarize_categories(builder),
                "excluded_file_count": len(excluded_rows),
                "excluded_file_bytes": excluded_bytes,
                "excluded_hashed_file_count": hashed_exclusion_count,
                "excluded_directory_scope_count": len(excluded_directories),
                "sensitive_env_file_count_excluded_unhashed": unhashed_secret_count,
                "privacy_scan_performed": False,
                "text_redaction_performed": False,
                "research_rerun_performed": False,
                "frozen_protocol_modified": False,
                "position_mapping_enabled": False,
                "order_generation_enabled": False,
                "broker_connection_enabled": False,
                "live_trading_enabled": False,
            },
        )

        file_index_rows = write_file_index(package_root)
        create_zip(package_root, ZIP_PATH)
        crc_result = verify_zip_crc_only(ZIP_PATH)

    zip_hash = sha256_file(ZIP_PATH)
    write_text(SHA256_PATH, f"{zip_hash}  {ZIP_PATH.name}\n")
    write_text(UPLOAD_MESSAGE_PATH, UPLOAD_MESSAGE + "\n")
    receipt = {
        "schema_version": "1.0.0",
        "package_id": PACKAGE_BASENAME,
        "status": "PASS_UPLOAD_READY_GPT_PRO_REVIEW_ZIP",
        "started_at": started_at,
        "completed_at": datetime.now(TIME_ZONE).isoformat(),
        "snapshot_date": "2026-08-25",
        "zip_path": ZIP_PATH.relative_to(ROOT).as_posix(),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": zip_hash,
        "included_source_file_count": len(builder.mappings),
        "included_source_bytes": source_bytes,
        "indexed_file_count_excluding_index": len(file_index_rows),
        "zip_member_count_including_index": crc_result["member_count"],
        "excluded_file_count": len(excluded_rows),
        "excluded_file_bytes": excluded_bytes,
        "excluded_hashed_file_count": hashed_exclusion_count,
        "excluded_directory_scope_count": len(excluded_directories),
        "privacy_scan_status": "SKIPPED_BY_USER_REQUEST",
        "text_redaction_performed": False,
        "per_file_sha256_performed": False,
        "zip_crc_verification": crc_result,
        "fresh_extraction_verification": "SKIPPED_BY_USER_REQUEST",
        "research_rerun_performed": False,
        "frozen_protocol_modified": False,
        "shadow_authorized": False,
        "position_mapping_enabled": False,
        "orders_generated": False,
        "broker_connected": False,
        "live_authorized": False,
    }
    write_json(RECEIPT_PATH, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
