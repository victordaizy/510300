"""构建 510300 压力传导危险率 V2 的 GPT Pro 全量会话审阅包。

本脚本只收集本次 V2 目标的协议、实现、测试、报告、回执、目标专属数据根，
以及这些文件直接依赖的历史输入。它不会继续冻结或运行反事实 G2，也不会执行
安全、隐私、秘密、恶意文件、脱敏或攻击面审计；仅做交付所需的结构完整性检查。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).absolute().parents[1]
DELIVERABLES = ROOT / "deliverables"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
SNAPSHOT_AT = datetime.now(TIME_ZONE)

PACKAGE_BASENAME = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_SESSION_GPT_PRO_FULL_REVIEW_20260903"
)
INTERNAL_ROOT = PACKAGE_BASENAME
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
UPLOAD_MESSAGE_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"

ORIGINAL_ATTACHMENT = Path(
    r"E:\CodexData\.codex\attachments"
    r"\8712825e-d589-4c9f-90dd-dcac55d9f808\pasted-text-1.txt"
)

TARGET_SCAN_DIRECTORIES = (
    "config",
    "docs",
    "research",
    "scripts",
    "tests",
    "reports/audit",
    "reports/research",
    "reports/data_quality",
    "data/audit",
    "logs",
)
TARGET_TOKENS = (
    "stress_transmission_hazard_v2",
    "STRESS_TRANSMISSION_HAZARD_V2",
)
CORE_FILES = (
    ".gitattributes",
    "pytest.ini",
    "requirements.txt",
)

EXPLICIT_DATA_ROOTS: dict[str, str] = {
    "data/raw/510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1": (
        "四态来源准入探测的原始响应"
    ),
    "data/raw/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1": (
        "四态账本历史采集的原始日行情、公司行动候选与逐证券检查点"
    ),
    "data/curated/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1": (
        "四态分类成分收益与公司行动核对账本"
    ),
    "data/curated/510300_stress_transmission_hazard_v2_mft_feature_execution_v1": (
        "冻结 M/F/T 特征面板及其发布时钟账本"
    ),
    "data/curated/510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1": (
        "G1 使用的 ETF-only B1 历史特征面板"
    ),
}

EXPLICIT_DIRECT_INPUTS: dict[str, str] = {
    "data/raw/market/510300_daily_2015_v2.parquet": "510300 历史未复权行情输入",
    "data/raw/market/H00300_total_return_daily_2015_v2.parquet": (
        "沪深300全收益指数历史输入"
    ),
    "data/raw/constituents/000300_constituent_daily.parquet": (
        "既有成分股日行情种子输入"
    ),
    "data/raw/macro/china_government_bond_yields_daily.parquet": (
        "中国国债收益率历史输入"
    ),
    "data/raw/valuation/000300_pe_official_raw.parquet": (
        "沪深300估值历史输入"
    ),
    "data/reference/510300_dividends.csv": "510300 现金分红参考账本",
    "data/curated/510300_asymmetric_stress_hazard_v1_original_route_g2/"
    "constituent_history_20141101_20200228.parquet": (
        "V2 四态账本使用的早期历史成分行情输入"
    ),
    "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/"
    "pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet": (
        "7天逆回购政策利率首次发布时钟输入"
    ),
    "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/"
    "dr007_daily_20150105_20260814.parquet": "DR007 历史日频输入",
    "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/"
    "000300_daily_pit_membership_20150101_20260814.parquet": (
        "沪深300点时成分与权重输入"
    ),
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "deliverables",
    "review_packages",
}
TEXT_SUFFIXES = {
    ".py",
    ".ps1",
    ".json",
    ".jsonl",
    ".md",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".txt",
    ".toml",
    ".ini",
}
LOCAL_MODULE_PREFIXES = {
    "research",
    "scripts",
    "market_data",
    "src",
    "backtest",
}
PATH_REFERENCE_PATTERN = re.compile(
    r"((?:config|data|docs|reports|research|scripts|tests|market_data|src|backtest)"
    r"[\\/][^\s\"'<>{}\[\]\(\),;:]+)"
)

REQUIRED_NAVIGATION_FILES = (
    "00_README_FIRST.md",
    "01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md",
    "02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt",
    "03_SESSION_STOP_POINT_AND_AUTHORITY.md",
    "04_GATE_TIMELINE_AND_CURRENT_STATUS.md",
    "05_DATA_SCOPE_AND_COMPLETENESS.md",
    "06_DATA_FILE_INDEX.csv",
    "07_INCLUDED_FILE_INDEX.csv",
    "08_REPRODUCTION_AND_VERSION_MAP.md",
    "09_VERIFICATION_BOUNDARY.md",
    "10_MISSING_OR_UNRESOLVED_REFERENCE_INDEX.csv",
    "11_EXPLICIT_DATA_ROOT_INDEX.csv",
    "12_GIT_SNAPSHOT.md",
    "13_PACKAGE_METADATA.json",
)

STRUCTURAL_CHECKS = (
    "SOURCE_FILE_SHA256_INDEX",
    "ZIP_CRC_TESTZIP",
    "DUPLICATE_MEMBER_CHECK",
    "INDEX_MEMBER_AND_SIZE_CHECK",
    "REQUIRED_NAVIGATION_MEMBER_CHECK",
    "EXPLICIT_DATA_ROOT_COVERAGE_CHECK",
    "ORIGINAL_ATTACHMENT_IDENTITY_CHECK",
    "WHOLE_ZIP_SHA256",
)
NOT_PERFORMED = (
    "SECURITY_AUDIT",
    "PRIVACY_SCAN",
    "SECRET_SCAN",
    "MALWARE_SCAN",
    "REDACTION",
    "ATTACK_SURFACE_REVIEW",
    "FRESH_EXTRACTION_REPLAY",
    "FULL_REPOSITORY_REGRESSION",
    "COUNTERFACTUAL_G2_FREEZE",
    "COUNTERFACTUAL_G2_EXECUTION",
)


@dataclass
class SourceRecord:
    path: Path
    reasons: set[str] = field(default_factory=set)


class PackageBuilder:
    def __init__(self) -> None:
        self.sources: dict[str, SourceRecord] = {}
        self.missing_references: dict[str, set[str]] = {}
        self.hash_cache: dict[str, str] = {}
        self.expanded_trees: set[str] = set()

    def add(
        self,
        path: Path | str,
        reason: str,
        *,
        required: bool = False,
    ) -> bool:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            relative = candidate.relative_to(ROOT).as_posix()
        except ValueError as exc:
            if required:
                raise RuntimeError(f"必需文件不在工作区内：{candidate}") from exc
            self.record_missing(str(candidate), reason)
            return False
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in Path(relative).parts):
            if required:
                raise RuntimeError(f"必需文件位于禁止打包目录：{relative}")
            return False
        try:
            is_file = candidate.is_file()
        except OSError as exc:
            if required:
                raise RuntimeError(f"无法读取必需文件：{relative}") from exc
            self.record_missing(relative, reason)
            return False
        if not is_file:
            if required:
                raise RuntimeError(f"缺少必需文件：{relative}")
            self.record_missing(relative, reason)
            return False
        record = self.sources.get(relative)
        if record is None:
            record = SourceRecord(path=candidate)
            self.sources[relative] = record
        record.reasons.add(reason)
        return True

    def add_tree(self, relative_root: str, reason: str, *, required: bool) -> None:
        normalized_root = Path(relative_root).as_posix().rstrip("/")
        if normalized_root in self.expanded_trees:
            return
        self.expanded_trees.add(normalized_root)
        root = ROOT / Path(relative_root)
        if not root.is_dir():
            if required:
                raise RuntimeError(f"缺少必需数据目录：{relative_root}")
            self.record_missing(relative_root, reason)
            return
        found = 0
        for path in iter_files(root):
            self.add(path, reason, required=required)
            found += 1
        if required and found == 0:
            raise RuntimeError(f"必需数据目录为空：{relative_root}")

    def record_missing(self, reference: str, reason: str) -> None:
        normalized = normalize_reference(reference)
        self.missing_references.setdefault(normalized, set()).add(reason)

    @property
    def total_bytes(self) -> int:
        return sum(record.path.stat().st_size for record in self.sources.values())

    def sha256(self, relative: str) -> str:
        cached = self.hash_cache.get(relative)
        if cached is None:
            cached = sha256_file(self.sources[relative].path)
            self.hash_cache[relative] = cached
        return cached


def iter_files(root: Path) -> Iterable[Path]:
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(
            name for name in names if name not in EXCLUDED_DIRECTORY_NAMES
        )
        base = Path(directory)
        for filename in sorted(filenames):
            path = base / filename
            try:
                if path.is_file():
                    yield path
            except OSError:
                continue


def iter_direct_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        try:
            if path.is_file():
                yield path
        except OSError:
            continue


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_reference(raw: str) -> str:
    value = raw.strip().rstrip(".,:;\"')]}><`")
    while "\\\\" in value:
        value = value.replace("\\\\", "\\")
    return value.replace("\\", "/")


def run_git(args: list[str], *, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
    )
    return completed.stdout


def git_z_paths(args: list[str]) -> list[str]:
    raw = run_git(args, binary=True)
    assert isinstance(raw, bytes)
    return [
        item.decode("utf-8", errors="replace").replace("\\", "/")
        for item in raw.split(b"\0")
        if item
    ]


def collect_git_info() -> dict[str, Any]:
    branch = str(run_git(["branch", "--show-current"])).strip()
    head = str(run_git(["rev-parse", "HEAD"])).strip()
    log = str(
        run_git(
            [
                "log",
                "-15",
                "--format=%H%x09%ad%x09%s",
                "--date=iso-strict",
            ]
        )
    ).strip()
    return {
        "branch": branch,
        "head": head,
        "recent_commit_log": log,
        "note": (
            "V2 会话文件主要为未跟踪工作区文件；当前分支名称来自此前 V1 工作，"
            "不能据此推定 V2 已提交。"
        ),
    }


def collect_named_target_files(builder: PackageBuilder) -> None:
    for relative_directory in TARGET_SCAN_DIRECTORIES:
        directory = ROOT / Path(relative_directory)
        for path in iter_direct_files(directory):
            relative = path.relative_to(ROOT).as_posix()
            if any(token.lower() in relative.lower() for token in TARGET_TOKENS):
                builder.add(path, "路径名称明确属于本次 V2 会话")


def collect_explicit_data(builder: PackageBuilder) -> None:
    for relative_root, description in EXPLICIT_DATA_ROOTS.items():
        builder.add_tree(relative_root, description, required=True)
    for relative, description in EXPLICIT_DIRECT_INPUTS.items():
        builder.add(relative, description, required=True)


def add_text_references(
    builder: PackageBuilder,
    path: Path,
    *,
    reason: str,
) -> int:
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            return 0
        text = read_text(path)
    except OSError:
        return 0
    before = len(builder.sources)
    for match in PATH_REFERENCE_PATTERN.finditer(text):
        relative = normalize_reference(match.group(1))
        if (
            not relative
            or not relative.isascii()
            or any(character in relative for character in ("{", "}", "*", "?", "`", "�"))
        ):
            continue
        candidate = ROOT / Path(relative)
        try:
            if candidate.is_file():
                builder.add(candidate, reason)
            elif candidate.is_dir() and any(
                token.lower() in relative.lower() for token in TARGET_TOKENS
            ):
                builder.add_tree(relative, reason, required=False)
            elif not candidate.exists():
                builder.record_missing(relative, reason)
        except OSError:
            builder.record_missing(relative, reason)
    return len(builder.sources) - before


def local_import_candidates(module: str) -> list[Path]:
    if not module:
        return []
    first = module.split(".", 1)[0]
    if first not in LOCAL_MODULE_PREFIXES:
        return []
    path = ROOT.joinpath(*module.split("."))
    return [path.with_suffix(".py"), path / "__init__.py"]


def add_python_imports(builder: PackageBuilder, path: Path) -> int:
    try:
        tree = ast.parse(read_text(path), filename=str(path))
    except (SyntaxError, OSError, UnicodeError):
        return 0
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    before = len(builder.sources)
    for module in sorted(modules):
        for candidate in local_import_candidates(module):
            if candidate.is_file():
                builder.add(
                    candidate,
                    f"本地 Python 导入依赖：{path.relative_to(ROOT).as_posix()}",
                )
    return len(builder.sources) - before


def collect_reference_closure(
    builder: PackageBuilder,
    seed_relatives: set[str],
) -> None:
    for relative in sorted(seed_relatives):
        record = builder.sources.get(relative)
        if record is None:
            continue
        suffix = record.path.suffix.lower()
        if suffix in TEXT_SUFFIXES and not relative.startswith("data/raw/"):
            add_text_references(
                builder,
                record.path,
                reason=f"目标文件直接引用：{relative}",
            )

    scanned_python: set[str] = set()
    while True:
        before = len(builder.sources)
        for relative, record in list(builder.sources.items()):
            suffix = record.path.suffix.lower()
            if suffix == ".py" and relative not in scanned_python:
                scanned_python.add(relative)
                add_python_imports(builder, record.path)
        if len(builder.sources) == before:
            break


def collect_sources(builder: PackageBuilder) -> tuple[dict[str, Any], bytes]:
    if not ORIGINAL_ATTACHMENT.is_file():
        raise RuntimeError(f"缺少原始用户附件：{ORIGINAL_ATTACHMENT}")
    collect_named_target_files(builder)
    collect_explicit_data(builder)
    for relative in CORE_FILES:
        builder.add(relative, "运行环境与测试入口", required=False)
    builder.add(Path(__file__).absolute(), "本压缩包的可复现构建脚本", required=True)
    seed_relatives = set(builder.sources)
    collect_reference_closure(builder, seed_relatives)
    return collect_git_info(), ORIGINAL_ATTACHMENT.read_bytes()


def tracked_state_map(builder: PackageBuilder) -> dict[str, str]:
    tracked = set(git_z_paths(["ls-files", "-z"]))
    modified = set(git_z_paths(["diff", "--name-only", "-z"]))
    staged = set(git_z_paths(["diff", "--cached", "--name-only", "-z"]))
    result: dict[str, str] = {}
    for relative in builder.sources:
        if relative in staged:
            result[relative] = "TRACKED_STAGED"
        elif relative in modified:
            result[relative] = "TRACKED_MODIFIED"
        elif relative in tracked:
            result[relative] = "TRACKED_AT_HEAD"
        else:
            result[relative] = "UNTRACKED_OR_IGNORED"
    return result


def classify_source(relative: str) -> str:
    lower = relative.lower()
    suffix = Path(lower).suffix
    if lower.startswith("data/raw/510300_stress_transmission_hazard_v2"):
        return "TARGET_COLLECTED_RAW_DATA"
    if lower.startswith("data/curated/510300_stress_transmission_hazard_v2"):
        return "TARGET_CURATED_DATA"
    if lower.startswith("data/raw/"):
        return "DIRECT_HISTORICAL_RAW_INPUT"
    if lower.startswith("data/curated/") or lower.startswith("data/reference/"):
        return "DIRECT_HISTORICAL_CURATED_OR_REFERENCE_INPUT"
    if lower.startswith("reports/data_quality/") and suffix in {
        ".parquet",
        ".csv",
        ".json",
    }:
        return "TARGET_DERIVED_DATA_OR_LEDGER"
    if lower.startswith("reports/research/") and suffix in {".parquet", ".csv"}:
        return "TARGET_DERIVED_DATA_OR_LEDGER"
    if lower.startswith("reports/"):
        return "REPORT_STATUS_OR_RECEIPT"
    if lower.startswith("config/") or lower.startswith("docs/"):
        return "PROTOCOL_MANIFEST_OR_DOCUMENT"
    if lower.startswith("tests/"):
        return "TEST"
    if lower.startswith("research/") or lower.startswith("scripts/"):
        return "IMPLEMENTATION"
    return "PROJECT_CONTEXT"


def hash_all_sources(builder: PackageBuilder) -> None:
    pending = [
        (relative, record.path)
        for relative, record in sorted(builder.sources.items())
        if relative not in builder.hash_cache
    ]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(sha256_file, path): relative for relative, path in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            relative = futures[future]
            builder.hash_cache[relative] = future.result()
            if completed_count % 250 == 0 or completed_count == len(pending):
                print(
                    f"逐文件哈希：{completed_count:,}/{len(pending):,}",
                    file=sys.stderr,
                    flush=True,
                )


def source_index_rows(
    builder: PackageBuilder,
    states: dict[str, str],
) -> list[dict[str, Any]]:
    hash_all_sources(builder)
    rows: list[dict[str, Any]] = []
    for relative, record in sorted(builder.sources.items()):
        rows.append(
            {
                "archive_path": f"{INTERNAL_ROOT}/project/{relative}",
                "workspace_logical_path": relative,
                "category": classify_source(relative),
                "bytes": record.path.stat().st_size,
                "sha256": builder.sha256(relative),
                "git_state": states[relative],
                "inclusion_reasons": " | ".join(sorted(record.reasons)),
            }
        )
    return rows


def data_index_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in source_rows
        if str(row["workspace_logical_path"]).startswith("data/")
        or row["category"] == "TARGET_DERIVED_DATA_OR_LEDGER"
    ]


def explicit_data_stats(builder: PackageBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_root, description in EXPLICIT_DATA_ROOTS.items():
        root = ROOT / Path(relative_root)
        disk_files = list(iter_files(root))
        disk_relatives = {
            path.relative_to(ROOT).as_posix() for path in disk_files
        }
        prefix = relative_root.rstrip("/") + "/"
        included_relatives = {
            relative for relative in builder.sources if relative.startswith(prefix)
        }
        missing = sorted(disk_relatives - included_relatives)
        if missing:
            raise RuntimeError(
                f"显式数据目录未完整覆盖：{relative_root}，缺少 {len(missing)} 个文件"
            )
        rows.append(
            {
                "logical_root": relative_root,
                "description": description,
                "file_count": len(disk_relatives),
                "bytes": sum(path.stat().st_size for path in disk_files),
                "coverage": "COMPLETE",
            }
        )
    return rows


def missing_reference_rows(builder: PackageBuilder) -> list[dict[str, str]]:
    return [
        {
            "reference": reference,
            "reasons": " | ".join(sorted(reasons)),
            "interpretation": (
                "引用在当前工作区不存在；可能是计划输出、旧失败路径或文档中的示例。"
            ),
        }
        for reference, reasons in sorted(builder.missing_references.items())
    ]


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def load_json(relative: str) -> dict[str, Any]:
    with (ROOT / Path(relative)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"状态文件不是 JSON 对象：{relative}")
    return payload


def load_authoritative_state() -> dict[str, Any]:
    state = {
        "initial_status_snapshot": load_json(
            "reports/research/510300_stress_transmission_hazard_v2_status.json"
        ),
        "g0_status": load_json(
            "reports/research/510300_stress_transmission_hazard_v2_g0_status_v1_0_1.json"
        ),
        "g1_terminal_status": load_json(
            "reports/research/510300_stress_transmission_hazard_v2_g1_status_v1_0_1.json"
        ),
        "historical_coverage_audit_status": load_json(
            "reports/research/"
            "510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_status_v1.json"
        ),
    }
    g1 = state["g1_terminal_status"]
    audit = state["historical_coverage_audit_status"]
    if g1.get("G1_DATA_AND_EVENTS") != "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY":
        raise RuntimeError("G1 权威状态不再是预期的 NO_VIEW，拒绝构包")
    if audit.get("audit_status") != "PASS_HISTORICAL_COVERAGE_AUDIT_G1_REMAINS_NO_VIEW":
        raise RuntimeError("历史覆盖审计状态与预期不一致，拒绝构包")
    counterfactual_manifest = (
        ROOT
        / "config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1_manifest.json"
    )
    counterfactual_status = (
        ROOT
        / "reports/research/"
        "510300_stress_transmission_hazard_v2_g2_counterfactual_status_v1.json"
    )
    if counterfactual_manifest.exists() or counterfactual_status.exists():
        raise RuntimeError("检测到反事实 G2 已冻结或已输出状态，与会话停止点不一致")
    state["session_stop"] = {
        "user_directive": (
            "中止我们现在做的，把我们这个会话做的一切包括数据，不需要安全审计，"
            "打包压缩包，我要交给gpt pro专家审阅，让他提出下一步战略"
        ),
        "counterfactual_g2_config_exists": True,
        "counterfactual_g2_manifest_exists": False,
        "counterfactual_g2_result_exists": False,
        "counterfactual_g2_frozen": False,
        "counterfactual_g2_executed": False,
        "counterfactual_model_trained": False,
        "prefreeze_python_compile": "PASS_OBSERVED_IN_ACTIVE_SESSION",
        "prefreeze_targeted_test": "8_PASSED_IN_38_64_SECONDS_OBSERVED_IN_ACTIVE_SESSION",
        "prefreeze_test_receipt": "NOT_PERSISTED_AS_IMMUTABLE_RECEIPT",
        "actual_g1_retained": "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY",
    }
    return state


def readme_text(
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    state: dict[str, Any],
) -> str:
    g1 = state["g1_terminal_status"]
    audit = state["historical_coverage_audit_status"]
    metrics = g1["metrics"]
    return f"""# 510300 压力传导危险率 V2：GPT Pro 全量会话审阅包

这是本次 `510300_STRESS_TRANSMISSION_HAZARD_V2` 会话的单一、自包含审阅包。
它保存已冻结的协议与执行链、失败与修正回执、目标内原始/整理/派生数据、
G1 后历史覆盖诊断，以及被用户中止时仍未冻结、未运行的反事实 G2 草案。

## 当前权威结论

- G0：`{state['g0_status']['G0_ENGINEERING_AND_CONTRACT']}`。
- 真实 G1：`{g1['G1_DATA_AND_EVENTS']}`。
- B2/B3 可识别独立事件：{metrics['b2_identifiable_event_count']} / {metrics['b3_identifiable_event_count']}。
- 机制发现最低要求 / 完整模型最低要求：30 / 40。
- G2：`{g1['G2_STRUCTURAL_INCREMENT']}`。
- 历史覆盖诊断：`{audit['audit_status']}`。
- 模型、概率、组合收益与夏普：均未生成或不允许读取。
- 仓位影响：0；Paper/Shadow、券商、订单、实盘均未授权。

早期总状态文件只是阶段快照，不能覆盖后续追加的 G0 通过、G1 `NO_VIEW` 和
历史覆盖审计。请按 `04_GATE_TIMELINE_AND_CURRENT_STATUS.md` 的时间顺序阅读。

## 会话停止点

用户先要求“假设 G1 通过，进行 G2”。为此只创建了隔离的反事实 G2 配置、
说明、实现、冻结器、构建器、重放器和测试；冻结前 Python 编译通过，目标测试
为 `8 passed in 38.64s`。随后用户明确中止，因此：

- 没有生成 G2 manifest；
- 没有训练真实或反事实模型；
- 没有生成预测概率、Log Loss、Brier、PR-AUC、AUC 或 Bootstrap 结果；
- 没有修改真实 G1 `NO_VIEW`；
- 该草案只供专家判断研究价值，不能视为预注册或结果。

## 建议阅读顺序

1. `01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md`
2. `02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt`
3. `03_SESSION_STOP_POINT_AND_AUTHORITY.md`
4. `04_GATE_TIMELINE_AND_CURRENT_STATUS.md`
5. `project/docs/510300_STRESS_TRANSMISSION_HAZARD_V2_PROTOCOL_20260903.md`
6. `project/config/510300_stress_transmission_hazard_v2.yaml`
7. `project/reports/research/510300_stress_transmission_hazard_v2_g1_status_v1_0_1.json`
8. `project/reports/data_quality/510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_COVERAGE_AUDIT_V1.md`
9. `project/reports/research/510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_status_v1.json`
10. `project/config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1.yaml`
11. `project/docs/510300_STRESS_TRANSMISSION_HAZARD_V2_G2_COUNTERFACTUAL_V1_20260903.md`
12. `05_DATA_SCOPE_AND_COMPLETENESS.md`
13. `06_DATA_FILE_INDEX.csv`
14. `07_INCLUDED_FILE_INDEX.csv`
15. `08_REPRODUCTION_AND_VERSION_MAP.md`
16. `09_VERIFICATION_BOUNDARY.md`

## 包规模

- 项目源文件：{len(source_rows):,}
- 项目源文件字节：{sum(int(row['bytes']) for row in source_rows):,}
- 数据及数据型结果文件：{len(data_rows):,}
- 显式目标数据根：{len(explicit_rows)}，均逐文件完整覆盖。
- 原始 GPT Pro 复核与 V2 规范附件另在 ZIP 根目录原样纳入。

## 不可越过的边界

- 本包用于外部专家审阅和下一步战略选择，不是外部审阅已经完成的证据。
- `NO_VIEW` 是有效结论，不是待填补的空白。
- 可执行资产仍仅为 `510300.SH` 与 `CASH_CNY`；成分股、指数和宏观数据仅是研究输入。
- 本次没有安全审计；结构检查边界见 `09_VERIFICATION_BOUNDARY.md`。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 专家的战略审阅任务

你是一名严厉、证据优先的量化研究负责人。请审阅本包中的
`510300_STRESS_TRANSMISSION_HAZARD_V2` 会话全量证据，并提出下一步战略。
不要宣传结果，不要安全审计，不要给买卖指令、目标仓位、券商连接或实盘方案。
每项实证判断必须引用包内相对路径、字段、函数、表格或回执。

## 首先接受这些事实边界

1. 真实 G0 已通过；真实 G1 为 `NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY`。
2. B2/B3 公共样本只有 23 个可识别独立事件，低于 30/40 的冻结门槛。
3. G1 后历史覆盖审计确认：20/60 日当前计数可逐日复现；实现曾丢弃指数纳入前
   已存在的公开价格历史，但补入这部分历史后没有新增可识别事件；审计仍保持 G1 `NO_VIEW`。
4. 审计还记录 15,340 个明确 `source_observed=false` 的成员日、324 个证券，以及
   公司行动未解决、供应商缺口/冲突等四态问题。不得把缺失收益统一填 0。
5. 反事实 G2 是用户一度要求“假设 G1 通过”时产生的未冻结草案。只完成编译与
   8 项单元测试；没有 manifest、模型、概率或指标。不得把它写成 G2 结果。
6. 本会话此前约束为“不用前瞻数据”。如果你认为只有严格前向积累才合理，须把
   它列为“仅当用户重新授权时”的独立路线，不能偷偷并入历史路线。
7. `510300.SH` 与 `CASH_CNY` 是唯一可执行资产；其余仅为信息、基准或归因输入。

## 必须审阅的问题

### A. 协议和工程链是否可信

- 逐门检查 V2 协议、manifest、修正版本、干净进程重放和 append-only 状态语言。
- 判断早期状态快照与后续权威状态是否被正确区分。
- 检查 V1→V1.0.1 的来源选择、输出命名、dtype/文件证据修正是否只修工程缺陷，
  是否改变了经济定义、样本或门槛。
- 检查四态账本 `TRADED_VALID / OFFICIAL_SUSPENSION /
  CORPORATE_ACTION_UNRESOLVED / SUPPLIER_MISSING_OR_CONFLICT` 的证据契约。

### B. G1 的 `NO_VIEW` 是否应被接受

- 复算或抽查 58 个总事件、23 个 B2/B3 可识别事件、119 个可用正样本起点、
  767 个非事件风险日和 886 个公共样本日。
- 评估 30/40 独立事件门槛的统计与经济合理性；禁止在看到当前 23 后下调门槛。
- 审阅历史覆盖审计的上界分析，判断是否还存在一种无需未来数据、无需换标签、
  无需降低覆盖门、且有现实机会把正式事件数提高到门槛的历史数据修复。
- 明确区分“可证明的数据/实现错误”与“为了运行 G2 的结果性救援”。

### C. 未冻结反事实 G2 草案是否有研究价值

- 检查 B0/B1/B2/B3、非负约束 Logistic、L2=1.0、事件权重、整事件切分、年度扩展
  训练、事件/日历年块 Bootstrap 和四时代规则是否无未来泄漏。
- 该草案只比较公共可识别样本，不代表 G1 通过。判断它属于：
  1. `DISCARD_AS_NON_ADMISSIBLE_AND_LOW_INFORMATION`；
  2. `KEEP_AS_EXPLORATORY_DIAGNOSTIC_BUT_DO_NOT_RUN`；
  3. `RUN_ONCE_AS_COUNTERFACTUAL_DIAGNOSTIC_WITH_ZERO_AUTHORITY`；
  4. `REDESIGN_BEFORE_ANY_RUN`。
- 如果选择 3，必须说明它能回答什么、不能回答什么、如何避免事后救援，以及
  为何在真实 G1 失败后仍有足够信息价值。不得把结果升级为正式 G2。

### D. 下一步战略选择

请至少比较以下互斥或分阶段路线：

- `STOP_V2_AND_ARCHIVE`：接受 G1 `NO_VIEW`，停止此分支。
- `HISTORICAL_DATA_REMEDIATION_ONLY`：只修可证明的官方停牌、公司行动或供应商
  历史缺口；不改标签、窗口、覆盖率、事件门或模型。
- `HISTORICAL_COUNTERFACTUAL_DIAGNOSTIC_ONLY`：仅运行隔离 G2 草案，零正式权限。
- `NEW_HISTORICAL_PREREGISTRATION`：若原协议本身不合理，另立版本，但不得复用
  当前历史选择门槛或救援失败结果。
- `STRICT_FORWARD_COLLECTION_ONLY_IF_REAUTHORIZED`：只在用户重新允许未来数据时
  启动，且不能追溯回填为当时可得。
- `SHIFT_RESEARCH_RESOURCES_ELSEWHERE`：把时间投向别的 510300 机制或纯底层数据建设。

对每条路线给出：预期信息增益、数据成本、工程成本、统计功效、主要失败方式、
停止条件、是否违反当前“仅用历史数据”约束，以及推荐资源占比。

## 强制输出格式

### 1. 一句话裁决

从以下选择一个主裁决，并给出不超过 150 字的理由：

- `STOP_AND_ARCHIVE_V2`
- `REPAIR_PROVABLE_HISTORICAL_DATA_GAPS_ONLY`
- `RUN_ONE_COUNTERFACTUAL_DIAGNOSTIC_THEN_ARCHIVE`
- `NEW_PREREGISTERED_V3_REQUIRED`
- `INSUFFICIENT_EVIDENCE_FOR_STRATEGIC_DECISION`

### 2. 证据可靠性表

逐项列 G0、四态来源、四态账本、M/F/T、G1、历史覆盖审计、反事实 G2 草案：
状态、最强证据、最大不确定性、是否足以支撑下一步。

### 3. P0 / P1 / P2 问题

每条给出包内证据路径、影响、最小修复动作。P0 必须足以推翻状态或证明泄漏；
不要把“可以更漂亮”写成 P0。

### 4. 停止/继续/资源矩阵

对上述六条战略路线逐项打分，并明确：现在做、以后条件满足再做、永久停止。

### 5. 30 / 90 / 180 天路线图

分别给出目标、唯一允许的输出、验收门、停止条件和资源预算。若主裁决是停止，
路线图应转向归档、底层数据或其他研究，不得为了填满时间而继续 V2。

### 6. 明确“不再做”清单

至少覆盖：降低事件门、改变 BAD10 阈值/期限、降低 98%/90% 覆盖门、缺失填 0、
信号反转、特征/模型网格、先看组合收益、以成分股作为可执行资产。

### 7. 最终建议

最多 10 项，按信息价值/成本排序。把“已验证事实”“推断”“需要新增证据”分开。
"""


def stop_point_text(state: dict[str, Any]) -> str:
    stop = state["session_stop"]
    return f"""# 会话停止点与权限边界

## 用户最新指令

> {stop['user_directive']}

该指令覆盖了此前“假设 G1 通过，进行 G2”的执行要求。收到中止指令时，
反事实 G2 尚未冻结，也尚未运行。

## 已完成到哪里

- 已创建 G2 反事实草案配置、说明、实现、冻结器、构建器、重放器和单元测试。
- 冻结前 Python 编译：`{stop['prefreeze_python_compile']}`。
- 冻结前目标测试：`{stop['prefreeze_targeted_test']}`。
- 测试输出未另存为不可变回执：`{stop['prefreeze_test_receipt']}`。

## 明确没有发生

- `counterfactual_g2_frozen = false`
- `counterfactual_g2_executed = false`
- `counterfactual_model_trained = false`
- `formal_model_admitted = false`
- `probability_generated = false`
- `prediction_metrics_generated = false`
- `portfolio_evaluation = NOT_ALLOWED`
- `return_evaluation = NOT_ALLOWED`
- `position_impact = 0`

## 权威状态

真实 G1 继续保持 `{stop['actual_g1_retained']}`。草案中的“假设 G1 通过”只是一项
隔离反事实前提，不会改变真实门槛、manifest、状态、仓位或交易权限。
"""


def gate_timeline_text(state: dict[str, Any]) -> str:
    g0 = state["g0_status"]
    g1 = state["g1_terminal_status"]
    audit = state["historical_coverage_audit_status"]
    m = g1["metrics"]
    am = audit["metrics"]
    return f"""# 门槛时间线与当前权威状态

## 1. 协议与 BAD10 账本

- V2 协议已冻结；BAD10 标签、事件合并、事件权重和历史截止日固定。
- BAD10 全体历史样本：{m['total_sample_count']} 日、{m['total_positive_origin_count']} 个正起点、
  {m['total_independent_event_count']} 个独立事件、{m['total_non_event_risk_day_count']} 个非事件风险日。
- 此阶段未读取组合收益或夏普。

## 2. G0

- 最新状态：`{g0['G0_ENGINEERING_AND_CONTRACT']}`。
- M/F/T 构建和修正版干净进程重放完成；原始失败回执保留。
- 下一阶段当时只允许在读实际标签前冻结事件级准入。

## 3. G1

- 最新状态：`{g1['G1_DATA_AND_EVENTS']}`。
- B1 可用样本日：{m['b1_eligible_sample_day_count']}。
- B2/B3 公共样本日：{m['b2_common_sample_day_count']} / {m['b3_common_sample_day_count']}。
- B2/B3 可识别事件：{m['b2_identifiable_event_count']} / {m['b3_identifiable_event_count']}。
- 机制发现要求 ≥30，完整三系数模型要求 ≥40；两者均未通过。
- G2/G3：`{g1['G2_STRUCTURAL_INCREMENT']}` / `{g1['G3_MACRO_INCREMENT']}`。
- 模型训练：false；概率生成：false；组合评估：`NOT_ALLOWED`。

## 4. G1 后历史覆盖审计

- 审计状态：`{audit['audit_status']}`。
- 冻结 20/60 日成员计数逐日复现：
  return20={str(am['current_return20_count_matches_frozen_on_all_dates']).lower()}，
  tail={str(am['current_tail_count_matches_frozen_on_all_dates']).lower()}。
- 发现实现丢弃纳入指数前的公开价格历史：
  `{str(am['implementation_protocol_mismatch_confirmed']).lower()}`。
- 但补入前史后可识别事件上界仍为
  {am['prehistory_price_and_frozen_comovement_diagnostic_event_upper_bound']}，没有新增事件。
- 明确 `source_observed=false` 成员日：{am['explicit_source_observed_false_member_day_count']}，
  涉及证券：{am['explicit_source_observed_false_symbol_count']}。
- 审计权限：`{audit['diagnostic_authority']}`；没有提升 G1。

## 5. 反事实 G2 草案和中止

- 用户一度要求假设 G1 通过，因此创建未冻结的隔离草案。
- 只完成编译与 8 项单元测试；没有 manifest、运行结果或模型。
- 用户随后中止并要求打包。当前正确状态是“草案供审阅”，不是 `G2 PASS/FAIL`。

## 状态读取规则

所有状态按追加时间顺序解释。早期
`project/reports/research/510300_stress_transmission_hazard_v2_status.json`
是协议与 BAD10 阶段快照，不可覆盖后续 G0、G1 和历史覆盖审计文件。
"""


def data_scope_text(
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
) -> str:
    explicit_lines = "\n".join(
        f"- `{row['logical_root']}`：{row['description']}；"
        f"{row['file_count']:,} 文件，{row['bytes']:,} 字节，{row['coverage']}。"
        for row in explicit_rows
    )
    direct_lines = "\n".join(
        f"- `{path}`：{description}。" for path, description in EXPLICIT_DIRECT_INPUTS.items()
    )
    return f"""# 数据范围与完整覆盖

## 范围定义

“本会话包括数据”定义为：本次 V2 目标专属原始、整理、派生数据目录全部文件，
本次协议/代码/manifest 实际引用的历史输入，以及报告中的数据型账本。不会递归
吞入工作区其他策略、旧备份、虚拟环境、缓存、历史交付包或 pytest 临时目录。

## 显式目标数据根

{explicit_lines}

## 强制纳入的直接历史输入

{direct_lines}

## 数据索引

- 数据及数据型结果文件：{len(data_rows):,}。
- 数据总字节：{sum(int(row['bytes']) for row in data_rows):,}。
- 每个文件的逻辑路径、大小、SHA-256、Git 状态和纳入理由见
  `06_DATA_FILE_INDEX.csv`。
- 所有项目源文件见 `07_INCLUDED_FILE_INDEX.csv`。

## 时间与用途边界

- 历史观察截止日：2026-08-14；本包没有为了 G2 读取更晚观测。
- 数据覆盖完整只说明声明范围内的文件已装包，不说明数据语义或研究结论正确。
- 真实 G1 仍是 `NO_VIEW`；缺失文件或不可识别事件不能被解释为零收益、零风险或
  “没有事件”。
"""


def reproduction_text() -> str:
    return """# 复现入口与版本地图

## 推荐顺序

所有命令都应在 ZIP 解压后的 `project/` 目录、Windows PowerShell 中运行。
先建立与 `requirements.txt` 相容的隔离 Python 环境。不要直接运行尚未冻结的
反事实 G2 冻结器或构建器，除非专家裁决和用户另行明确授权。

## 已冻结主线

1. 协议：`config/510300_stress_transmission_hazard_v2.yaml`
2. 四态来源 V1 与 V1.0.1：对应 config、manifest、freeze、probe、tests、receipt。
3. 四态账本 V1 与 V1.0.1：对应 acquisition、build、config、manifest、tests、receipt。
4. M/F/T 执行 V1 与 dtype 规范化 V1.0.1：对应 build、replay、manifest 和 G0 状态。
5. G1 事件准入 V1 与文件证据比较修正 V1.0.1：对应 build、replay 和终局 G1 状态。
6. G1 后历史覆盖审计 V1：对应 freeze、build、replay、报告、状态与回执。

manifest 内记录了各冻结文件和输入的 SHA-256，应以各版本自己的 manifest 为准，
不要将 V1 的失败路径与 V1.0.1 的修正版混成一次成功运行。

## 当前未冻结草案

- `config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1.yaml`
- `docs/510300_STRESS_TRANSMISSION_HAZARD_V2_G2_COUNTERFACTUAL_V1_20260903.md`
- `research/stress_transmission_hazard_v2_g2_counterfactual_v1.py`
- `scripts/freeze_510300_stress_transmission_hazard_v2_g2_counterfactual_v1.py`
- `scripts/build_510300_stress_transmission_hazard_v2_g2_counterfactual_v1.py`
- `scripts/replay_510300_stress_transmission_hazard_v2_g2_counterfactual_v1.py`
- `tests/test_510300_stress_transmission_hazard_v2_g2_counterfactual_v1.py`

它们只通过冻结前编译和目标单元测试，不具备 manifest，也未执行真实历史运算。

## 构包器复现

只读盘点：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\build_510300_stress_transmission_hazard_v2_gpt_pro_full_review_package_20260903.py --inventory-only
```

正式构包要求同名交付物尚不存在：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\build_510300_stress_transmission_hazard_v2_gpt_pro_full_review_package_20260903.py
```
"""


def verification_boundary_text() -> str:
    performed = "\n".join(f"- `{item}=true`" for item in STRUCTURAL_CHECKS)
    skipped = "\n".join(f"- `{item}=false`" for item in NOT_PERFORMED)
    return f"""# 交付检查边界

用户明确要求“不需要安全审计”。因此本次只做交付所需的结构检查，不把它们称为
安全、隐私或研究正确性验证。

## 已执行的结构检查

{performed}

这些检查只能说明：声明文件被索引、ZIP 成员可读、CRC 正常、没有重复成员、
索引大小一致、原附件身份一致、整包有可核对哈希。

## 明确未执行

{skipped}

因此：接收方应把本包视为用户授权的未脱敏研究快照；包完成不代表 GPT Pro 已
审阅，不代表数据/代码经济含义正确，也不授权任何交易动作。
"""


def git_snapshot_text(git_info: dict[str, Any], source_rows: list[dict[str, Any]]) -> str:
    state_counts = Counter(str(row["git_state"]) for row in source_rows)
    count_lines = "\n".join(
        f"- `{key}`：{value:,}" for key, value in sorted(state_counts.items())
    )
    return f"""# Git 快照

- 当前分支：`{git_info['branch']}`
- HEAD：`{git_info['head']}`
- 说明：{git_info['note']}

## 包内文件 Git 状态计数

{count_lines}

未跟踪不等于缺失：本包按磁盘实际内容逐文件索引和计算 SHA-256。但未提交文件
不能仅凭 Git HEAD 复现，因此 ZIP、文件索引和各冻结 manifest 是本次交付快照的
直接身份依据。

## 最近 15 个提交

```text
{git_info['recent_commit_log']}
```
"""


def build_generated_files(
    builder: PackageBuilder,
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, str]],
    git_info: dict[str, Any],
    attachment_bytes: bytes,
    state: dict[str, Any],
) -> dict[str, bytes]:
    metadata = {
        "package_id": PACKAGE_BASENAME,
        "created_at": SNAPSHOT_AT.isoformat(),
        "purpose": "GPT_PRO_EXPERT_STRATEGIC_REVIEW",
        "scope": "FULL_CURRENT_V2_SESSION_WITH_TARGET_DATA_AND_DIRECT_INPUTS",
        "program_id": "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "authoritative_current_state": (
            "G1_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY_G2_NOT_RUN"
        ),
        "counterfactual_g2": {
            "draft_included": True,
            "frozen": False,
            "executed": False,
            "model_trained": False,
            "result_generated": False,
        },
        "source_file_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "explicit_data_root_count": len(explicit_rows),
        "explicit_data_root_coverage": "COMPLETE",
        "missing_or_unresolved_reference_count": len(missing_rows),
        "original_attachment": {
            "source_path": str(ORIGINAL_ATTACHMENT),
            "bytes": len(attachment_bytes),
            "sha256": hashlib.sha256(attachment_bytes).hexdigest(),
            "archive_path": (
                f"{INTERNAL_ROOT}/02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt"
            ),
        },
        "structural_checks": list(STRUCTURAL_CHECKS),
        "not_performed": list(NOT_PERFORMED),
        "security_audit": False,
        "privacy_scan": False,
        "secret_scan": False,
        "malware_scan": False,
        "redaction": False,
        "fresh_extraction_replay": False,
        "position_impact": 0,
        "live_trading_authorized": False,
        "git": git_info,
    }
    generated: dict[str, bytes] = {
        "00_README_FIRST.md": readme_text(
            source_rows, data_rows, explicit_rows, state
        ).encode("utf-8"),
        "01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md": review_prompt_text().encode(
            "utf-8"
        ),
        "02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt": attachment_bytes,
        "03_SESSION_STOP_POINT_AND_AUTHORITY.md": stop_point_text(state).encode(
            "utf-8"
        ),
        "04_GATE_TIMELINE_AND_CURRENT_STATUS.md": gate_timeline_text(state).encode(
            "utf-8"
        ),
        "05_DATA_SCOPE_AND_COMPLETENESS.md": data_scope_text(
            data_rows, explicit_rows
        ).encode("utf-8"),
        "06_DATA_FILE_INDEX.csv": csv_bytes(
            data_rows,
            [
                "archive_path",
                "workspace_logical_path",
                "category",
                "bytes",
                "sha256",
                "git_state",
                "inclusion_reasons",
            ],
        ),
        "07_INCLUDED_FILE_INDEX.csv": csv_bytes(
            source_rows,
            [
                "archive_path",
                "workspace_logical_path",
                "category",
                "bytes",
                "sha256",
                "git_state",
                "inclusion_reasons",
            ],
        ),
        "08_REPRODUCTION_AND_VERSION_MAP.md": reproduction_text().encode("utf-8"),
        "09_VERIFICATION_BOUNDARY.md": verification_boundary_text().encode("utf-8"),
        "10_MISSING_OR_UNRESOLVED_REFERENCE_INDEX.csv": csv_bytes(
            missing_rows,
            ["reference", "reasons", "interpretation"],
        ),
        "11_EXPLICIT_DATA_ROOT_INDEX.csv": csv_bytes(
            explicit_rows,
            ["logical_root", "description", "file_count", "bytes", "coverage"],
        ),
        "12_GIT_SNAPSHOT.md": git_snapshot_text(git_info, source_rows).encode(
            "utf-8"
        ),
        "13_PACKAGE_METADATA.json": (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    missing_navigation = set(REQUIRED_NAVIGATION_FILES) - set(generated)
    if missing_navigation:
        raise RuntimeError(f"缺少导航文件：{sorted(missing_navigation)}")
    return generated


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_zip(
    builder: PackageBuilder,
    generated: dict[str, bytes],
) -> None:
    if ZIP_PATH.exists():
        raise RuntimeError(f"交付包已存在，拒绝覆盖：{ZIP_PATH}")
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{PACKAGE_BASENAME}.", suffix=".tmp", dir=DELIVERABLES
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, payload in sorted(generated.items()):
                archive.writestr(f"{INTERNAL_ROOT}/{name}", payload)
            for relative, record in sorted(builder.sources.items()):
                archive.write(
                    record.path,
                    arcname=f"{INTERNAL_ROOT}/project/{relative}",
                )
        os.replace(temporary, ZIP_PATH)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_zip(
    source_rows: list[dict[str, Any]],
    attachment_bytes: bytes,
) -> dict[str, Any]:
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        names = archive.namelist()
        counts = Counter(names)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise RuntimeError(f"ZIP 存在重复成员：{duplicates[:10]}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC 失败成员：{bad_member}")
        required = {
            f"{INTERNAL_ROOT}/{name}" for name in REQUIRED_NAVIGATION_FILES
        }
        missing_navigation = sorted(required - set(names))
        if missing_navigation:
            raise RuntimeError(f"ZIP 缺少导航成员：{missing_navigation}")
        missing_index_members: list[str] = []
        indexed_size_mismatches: list[str] = []
        info_by_name = {info.filename: info for info in archive.infolist()}
        for row in source_rows:
            archive_path = str(row["archive_path"])
            info = info_by_name.get(archive_path)
            if info is None:
                missing_index_members.append(archive_path)
            elif info.file_size != int(row["bytes"]):
                indexed_size_mismatches.append(archive_path)
        if missing_index_members or indexed_size_mismatches:
            raise RuntimeError(
                "ZIP 索引成员或大小不一致："
                f"missing={len(missing_index_members)}, "
                f"size_mismatch={len(indexed_size_mismatches)}"
            )
        archived_attachment = archive.read(
            f"{INTERNAL_ROOT}/02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt"
        )
        attachment_identity_match = archived_attachment == attachment_bytes
        if not attachment_identity_match:
            raise RuntimeError("ZIP 内原始附件与外部附件不一致")
    return {
        "zip_member_count": len(names),
        "duplicate_member_count": len(duplicates),
        "crc_testzip_bad_member": bad_member,
        "required_navigation_missing_count": len(missing_navigation),
        "source_index_rows": len(source_rows),
        "missing_index_member_count": len(missing_index_members),
        "indexed_size_mismatch_count": len(indexed_size_mismatches),
        "attachment_identity_match": attachment_identity_match,
    }


def upload_message(zip_sha256: str, zip_bytes: int) -> str:
    return f"""请将这个 ZIP 作为一个整体上传给 GPT Pro：

{ZIP_PATH.name}

大小：{zip_bytes:,} 字节
SHA-256：{zip_sha256}

上传后请直接发送以下指令：

请先阅读压缩包根目录的 00_README_FIRST.md，再严格执行
01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md。请依据包内证据审阅，不做安全审计，不给
买卖或实盘指令。重点判断真实 G1 NO_VIEW 是否应接受、未冻结反事实 G2 是否值得
运行，以及下一步应停止、仅修历史数据、另立预注册版本，还是把资源转向其他方向。
请按提示要求输出证据可靠性表、P0/P1/P2、路线矩阵和 30/90/180 天战略。
"""


def inventory_payload(
    builder: PackageBuilder,
    explicit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    category_counts = Counter(classify_source(relative) for relative in builder.sources)
    data_count = sum(
        1
        for relative in builder.sources
        if relative.startswith("data/")
        or classify_source(relative) == "TARGET_DERIVED_DATA_OR_LEDGER"
    )
    return {
        "status": "PASS_INVENTORY_ONLY_NO_PACKAGE_WRITTEN",
        "package_id": PACKAGE_BASENAME,
        "source_file_count": len(builder.sources),
        "source_bytes": builder.total_bytes,
        "data_file_count": data_count,
        "explicit_data_roots": explicit_rows,
        "category_counts": dict(sorted(category_counts.items())),
        "missing_or_unresolved_reference_count": len(builder.missing_references),
        "missing_or_unresolved_references": sorted(builder.missing_references),
        "counterfactual_g2_frozen": False,
        "counterfactual_g2_executed": False,
        "security_audit": False,
    }


def build_package() -> dict[str, Any]:
    output_paths = (ZIP_PATH, SHA256_PATH, RECEIPT_PATH, UPLOAD_MESSAGE_PATH)
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise RuntimeError(f"交付物已存在，拒绝覆盖：{existing_outputs}")
    builder = PackageBuilder()
    git_info, attachment_bytes = collect_sources(builder)
    explicit_rows = explicit_data_stats(builder)
    state = load_authoritative_state()
    states = tracked_state_map(builder)
    source_rows = source_index_rows(builder, states)
    data_rows = data_index_rows(source_rows)
    missing_rows = missing_reference_rows(builder)
    generated = build_generated_files(
        builder,
        source_rows,
        data_rows,
        explicit_rows,
        missing_rows,
        git_info,
        attachment_bytes,
        state,
    )
    write_zip(builder, generated)
    verification = verify_zip(source_rows, attachment_bytes)
    zip_sha256 = sha256_file(ZIP_PATH)
    zip_bytes = ZIP_PATH.stat().st_size
    receipt = {
        "status": (
            "PASS_FULL_SESSION_PACKAGE_CRC_DUPLICATE_INDEX_DATA_COVERAGE_"
            "ATTACHMENT_AND_WHOLE_ZIP_SHA256"
        ),
        "package_id": PACKAGE_BASENAME,
        "created_at": SNAPSHOT_AT.isoformat(),
        "zip_path": ZIP_PATH.relative_to(ROOT).as_posix(),
        "zip_bytes": zip_bytes,
        "zip_sha256": zip_sha256,
        "source_file_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "generated_navigation_file_count": len(generated),
        "explicit_data_root_count": len(explicit_rows),
        "explicit_data_root_coverage": "COMPLETE",
        "missing_or_unresolved_reference_count": len(missing_rows),
        "original_attachment_sha256": hashlib.sha256(attachment_bytes).hexdigest(),
        "authoritative_current_state": (
            "G1_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY_G2_NOT_RUN"
        ),
        "counterfactual_g2_frozen": False,
        "counterfactual_g2_executed": False,
        "model_trained": False,
        "position_impact": 0,
        "verification": verification,
        "structural_checks": list(STRUCTURAL_CHECKS),
        "not_performed": list(NOT_PERFORMED),
        "security_audit": False,
        "privacy_scan": False,
        "secret_scan": False,
        "malware_scan": False,
        "redaction": False,
        "fresh_extraction_replay": False,
    }
    atomic_write_bytes(
        SHA256_PATH,
        f"{zip_sha256}  {ZIP_PATH.name}\n".encode("ascii"),
    )
    atomic_write_bytes(
        RECEIPT_PATH,
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    atomic_write_bytes(
        UPLOAD_MESSAGE_PATH,
        upload_message(zip_sha256, zip_bytes).encode("utf-8"),
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="只读盘点目标范围，不计算逐文件哈希，也不生成任何交付物。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inventory_only:
        builder = PackageBuilder()
        collect_sources(builder)
        explicit_rows = explicit_data_stats(builder)
        load_authoritative_state()
        print(
            json.dumps(
                inventory_payload(builder, explicit_rows),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt = build_package()
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
