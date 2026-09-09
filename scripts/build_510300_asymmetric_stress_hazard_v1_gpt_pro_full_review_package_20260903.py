"""构建 510300 非对称压力风险 V1 的 GPT Pro 全量研究审阅包。

范围包括本目标分支的全部变更、目标专属原始/整理/派生数据、协议实际引用的
输入、原始用户审阅文本、代码、测试、报告和回执。按用户明确要求，不执行
安全、隐私、秘密、恶意文件、脱敏或攻击面审计；仅校验 ZIP 传输完整性。
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
    "510300_ASYMMETRIC_STRESS_HAZARD_V1_GPT_PRO_FULL_REVIEW_20260903"
)
INTERNAL_ROOT = PACKAGE_BASENAME
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
UPLOAD_MESSAGE_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"

ORIGINAL_ATTACHMENT = Path(
    r"E:\CodexData\.codex\attachments"
    r"\3aa2ff69-bbd4-484e-a81b-ff5b9fb9bfc1\pasted-text-1.txt"
)
BASE_REF = "master"

EXPLICIT_DATA_ROOTS: dict[str, str] = {
    "data/raw/510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2":
        "目标内采集的 DR007 原始响应与凭据预检响应",
    "data/raw/510300_asymmetric_stress_hazard_v1_original_route_g2":
        "目标内采集的 G2 成分股历史原始响应",
    "data/raw/remediation/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1":
        "目标内采集的行业和央行源修复原始文件",
    "data/curated/510300_asymmetric_stress_hazard_v1_original_route_g2":
        "目标内整理的 G2 历史成分行情",
    "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1":
        "目标内整理的行业与政策利率数据",
    "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2":
        "目标内整理的精确 DR007 数据",
    "data/processed/510300_structural_signal_power_and_identifiability_audit_v1":
        "本目标启动前置功效识别分析的派生数据",
}

GOAL_SEARCH_ROOTS = (
    "config",
    "docs",
    "reports",
    "research",
    "scripts",
    "tests",
    "logs",
    "data/audit",
)
GOAL_TOKENS = (
    "asymmetric_stress_hazard",
    "stress_hazard_v1",
    "structural_signal_power_and_identifiability",
)
CORE_FILES = (
    ".gitattributes",
    "pytest.ini",
    "requirements.txt",
)
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
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "deliverables",
    "review_packages",
}
PATH_REFERENCE_PATTERN = re.compile(
    r"((?:config|data|docs|reports|research|scripts|tests|market_data|src|backtest)"
    r"[\\/][^\s\"'<>{}\[\]\(\),;:]+)"
)

INTEGRITY_CHECKS = (
    "SOURCE_FILE_SHA256_INDEX",
    "ZIP_CRC_TESTZIP",
    "DUPLICATE_MEMBER_CHECK",
    "WHOLE_ZIP_SHA256",
    "REQUIRED_NAVIGATION_MEMBER_CHECK",
    "EXPLICIT_DATA_ROOT_COVERAGE_CHECK",
)
NOT_PERFORMED = (
    "SECURITY_AUDIT",
    "PRIVACY_SCAN",
    "SECRET_SCAN",
    "MALWARE_SCAN",
    "REDACTION",
    "ATTACK_SURFACE_REVIEW",
    "FULL_REPOSITORY_REGRESSION",
    "FRESH_EXTRACTION_REPLAY",
)


@dataclass
class SourceRecord:
    path: Path
    reasons: set[str] = field(default_factory=set)

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


class PackageBuilder:
    def __init__(self) -> None:
        self.sources: dict[str, SourceRecord] = {}
        self.missing_references: set[str] = set()
        self.hash_cache: dict[str, str] = {}
        self.expanded_trees: set[str] = set()

    def add(self, path: Path, reason: str, *, required: bool = False) -> bool:
        candidate = path if path.is_absolute() else ROOT / path
        try:
            relative = candidate.relative_to(ROOT).as_posix()
        except ValueError as exc:
            if required:
                raise RuntimeError(f"必需文件不在逻辑工作区内：{candidate}") from exc
            self.missing_references.add(str(candidate))
            return False
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in Path(relative).parts):
            if required:
                raise RuntimeError(f"必需文件落入禁止打包目录：{relative}")
            return False
        try:
            is_file = candidate.is_file()
        except OSError as exc:
            if required:
                raise RuntimeError(f"无法读取必需文件：{relative}") from exc
            self.missing_references.add(relative)
            return False
        if not is_file:
            if required:
                raise RuntimeError(f"缺少必需文件：{relative}")
            self.missing_references.add(relative)
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
            self.missing_references.add(relative_root)
            return
        found = 0
        for path in iter_files(root):
            self.add(path, reason, required=required)
            found += 1
        if required and found == 0:
            raise RuntimeError(f"必需数据目录为空：{relative_root}")

    def sha256(self, relative: str) -> str:
        cached = self.hash_cache.get(relative)
        if cached is not None:
            return cached
        digest = sha256_file(self.sources[relative].path)
        self.hash_cache[relative] = digest
        return digest

    @property
    def total_bytes(self) -> int:
        return sum(record.path.stat().st_size for record in self.sources.values())


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str], *, binary: bool = False) -> str | bytes:
    command = ["git", "-c", "core.quotepath=false", *args]
    completed = subprocess.run(
        command,
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


def collect_git_scope(builder: PackageBuilder) -> dict[str, Any]:
    base = str(run_git(["merge-base", "HEAD", BASE_REF])).strip()
    branch = str(run_git(["branch", "--show-current"])).strip()
    head = str(run_git(["rev-parse", "HEAD"])).strip()
    changed = git_z_paths(["diff", "--name-only", "-z", base, "HEAD"])
    for relative in changed:
        builder.add(ROOT / Path(relative), "目标分支相对 master 的变更", required=False)
    commits = str(
        run_git(
            [
                "log",
                "--reverse",
                "--format=%H%x09%ad%x09%s",
                "--date=iso-strict",
                f"{base}..HEAD",
            ]
        )
    ).strip()
    return {
        "branch": branch,
        "head": head,
        "base_ref": BASE_REF,
        "merge_base": base,
        "branch_changed_paths": changed,
        "branch_commit_log": commits,
    }


def collect_goal_named_files(builder: PackageBuilder) -> None:
    for relative_root in GOAL_SEARCH_ROOTS:
        root = ROOT / Path(relative_root)
        if not root.is_dir():
            continue
        for path in iter_files(root):
            relative = path.relative_to(ROOT).as_posix().lower()
            if any(token in relative for token in GOAL_TOKENS):
                builder.add(path, "路径名称明确属于本目标")


def collect_explicit_data(builder: PackageBuilder) -> None:
    for relative, description in EXPLICIT_DATA_ROOTS.items():
        builder.add_tree(relative, description, required=True)


def normalize_reference(raw: str) -> str:
    value = raw.strip().rstrip(".")
    while "\\\\" in value:
        value = value.replace("\\\\", "\\")
    return value.replace("\\", "/")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def add_text_references(
    builder: PackageBuilder,
    text: str,
    *,
    reason: str,
) -> int:
    added_before = len(builder.sources)
    for match in PATH_REFERENCE_PATTERN.finditer(text):
        relative = normalize_reference(match.group(1))
        existing = builder.sources.get(relative)
        if existing is not None:
            existing.reasons.add(reason)
            continue
        if relative.rstrip("/") in builder.expanded_trees:
            continue
        candidate = ROOT / Path(relative)
        try:
            if candidate.is_file():
                builder.add(candidate, reason)
            elif candidate.is_dir() and any(
                token in relative.lower() for token in GOAL_TOKENS
            ):
                builder.add_tree(relative, reason, required=False)
        except OSError:
            builder.missing_references.add(relative)
    return len(builder.sources) - added_before


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
    added_before = len(builder.sources)
    for module in sorted(modules):
        for candidate in local_import_candidates(module):
            if candidate.is_file():
                builder.add(
                    candidate,
                    f"本地 Python 导入依赖：{path.relative_to(ROOT).as_posix()}",
                )
    return len(builder.sources) - added_before


def collect_reference_closure(
    builder: PackageBuilder,
    attachment_text: str,
    seed_relatives: set[str],
) -> None:
    for relative in sorted(seed_relatives):
        record = builder.sources.get(relative)
        if record is None:
            continue
        suffix = record.path.suffix.lower()
        scan_references = (
            suffix in TEXT_SUFFIXES
            and not relative.startswith("data/raw/")
            and record.path.stat().st_size <= 16 * 1024 * 1024
        )
        if not scan_references:
            continue
        try:
            text = read_text(record.path)
        except OSError:
            continue
        add_text_references(
            builder,
            text,
            reason=f"目标文件直接引用：{relative}",
        )

    add_text_references(
        builder,
        attachment_text,
        reason="原始用户审阅文本直接引用的证据",
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
    attachment_bytes = ORIGINAL_ATTACHMENT.read_bytes()
    attachment_text = read_text(ORIGINAL_ATTACHMENT)
    git_info = collect_git_scope(builder)
    collect_goal_named_files(builder)
    collect_explicit_data(builder)
    for relative in CORE_FILES:
        builder.add(ROOT / Path(relative), "运行环境与测试入口", required=False)
    builder.add(Path(__file__).absolute(), "本压缩包可复现构建脚本", required=True)
    seed_relatives = set(builder.sources)
    collect_reference_closure(builder, attachment_text, seed_relatives)
    return git_info, attachment_bytes


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
    if lower.startswith("data/raw/510300_asymmetric_stress_hazard"):
        return "TARGET_COLLECTED_RAW_DATA"
    if lower.startswith("data/raw/remediation/510300_asymmetric_stress_hazard"):
        return "TARGET_COLLECTED_RAW_DATA"
    if lower.startswith("data/curated/510300_asymmetric_stress_hazard"):
        return "TARGET_CURATED_DATA"
    if "structural_signal_power_and_identifiability" in lower and lower.startswith(
        "data/"
    ):
        return "TARGET_DERIVED_PREFLIGHT_DATA"
    if lower.startswith("data/raw/"):
        return "REFERENCED_RAW_INPUT"
    if lower.startswith("data/curated/") or lower.startswith("data/processed/"):
        return "REFERENCED_CURATED_OR_PROCESSED_INPUT"
    if lower.startswith("data/reference/"):
        return "REFERENCED_REFERENCE_INPUT"
    if lower.startswith("reports/research/") and Path(relative).suffix.lower() in {
        ".parquet",
        ".csv",
    }:
        return "TARGET_DERIVED_RESULT_DATA"
    if lower.startswith("reports/"):
        return "REPORT_OR_RECEIPT"
    if lower.startswith("config/") or lower.startswith("docs/"):
        return "PROTOCOL_OR_DOCUMENT"
    if lower.startswith("tests/"):
        return "TEST"
    if lower.startswith("research/") or lower.startswith("scripts/"):
        return "IMPLEMENTATION"
    return "PROJECT_CONTEXT"


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def source_index_rows(
    builder: PackageBuilder,
    states: dict[str, str],
) -> list[dict[str, Any]]:
    pending = [
        (relative, record.path)
        for relative, record in sorted(builder.sources.items())
        if relative not in builder.hash_cache
    ]
    if pending:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(sha256_file, path): relative
                for relative, path in pending
            }
            for completed_count, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                relative = futures[future]
                builder.hash_cache[relative] = future.result()
                if completed_count % 500 == 0 or completed_count == len(pending):
                    print(
                        f"逐文件哈希：{completed_count:,}/{len(pending):,}",
                        file=sys.stderr,
                        flush=True,
                    )
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
        if row["workspace_logical_path"].startswith("data/")
        or row["category"] == "TARGET_DERIVED_RESULT_DATA"
    ]


def explicit_data_stats(builder: PackageBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_root, description in EXPLICIT_DATA_ROOTS.items():
        prefix = relative_root.rstrip("/") + "/"
        members = [
            record
            for relative, record in builder.sources.items()
            if relative.startswith(prefix)
        ]
        disk_files = list(iter_files(ROOT / Path(relative_root)))
        disk_relatives = {
            path.relative_to(ROOT).as_posix()
            for path in disk_files
        }
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
                "file_count": len(members),
                "bytes": sum(record.path.stat().st_size for record in members),
                "coverage": "COMPLETE",
            }
        )
    return rows


def load_final_result() -> dict[str, Any]:
    path = (
        ROOT
        / "reports/research/510300_asymmetric_stress_hazard_v1_g2_mechanism.json"
    )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != (
        "REJECTED_FROZEN_G2_MECHANISM_CHAIN_FAILED_NO_FEATURE_RESCUE"
    ):
        raise RuntimeError("最终 G2 状态与预期终局拒绝不一致")
    return payload


def readme_text(
    *,
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    gate = result["g2_gate"]
    bad10 = gate["internal_bad10_gate"]
    return f"""# 510300 非对称压力风险 V1：GPT Pro 全量审阅包

这是 510300_ASYMMETRIC_STRESS_HAZARD_V1 的单一、自包含研究审阅包。
它包含本目标分支的协议、实现、测试、报告、回执、全部目标内采集数据，以及
协议实际引用的既有输入数据。

## 先看结论，但不要直接接受

- 当前终局状态：{result["status"]}
- 宏观领先门：通过，4 个固定子期中 3 个为正。
- 可评分独立事件 / 非事件块：{bad10["scoreable_event_count"]} / {bad10["scoreable_non_event_count"]}。
- 内部结构 AUC：{bad10["internal_score_roc_auc"]:.12f}。
- 固定价格基线 AUC：{bad10["price_baseline_roc_auc"]:.12f}。
- G3 是否允许：{str(gate["g3_allowed"]).lower()}。

该结论的含义是：按旧文件预注册顺序，V1 在 G2 停止，不能通过改符号、窗口、
特征或模型救援。GPT Pro 的任务是核实这个终止结论是否由正确的数据、代码和
点时口径支持，而不是替它寻找一个事后能通过的新版本。

## 建议阅读顺序

1. 01_GPT_PRO_REVIEW_PROMPT.md
2. 02_ORIGINAL_USER_PASTED_REVIEW.txt
3. 03_USER_DECISIONS_AND_SCOPE.md
4. 04_GATE_TIMELINE_AND_CURRENT_RESULT.md
5. project/docs/510300_ASYMMETRIC_STRESS_HAZARD_V1_CHARTER.md
6. project/config/510300_asymmetric_stress_hazard_v1.yaml
7. project/reports/research/510300_ASYMMETRIC_STRESS_HAZARD_V1_BAD10_CENSUS.md
8. project/reports/research/510300_ASYMMETRIC_STRESS_HAZARD_V1_G2_MECHANISM.md
9. project/reports/research/510300_asymmetric_stress_hazard_v1_g2_mechanism.json
10. 05_DATA_SCOPE_AND_COMPLETENESS.md
11. 06_DATA_FILE_INDEX.csv
12. 07_INCLUDED_FILE_INDEX.csv
13. 08_GIT_HISTORY_AND_REPRODUCIBILITY.md
14. 09_REPRODUCTION_ENTRYPOINTS.md
15. 10_VERIFICATION_BOUNDARY.md

## 包规模

- 项目源文件：{len(source_rows):,}
- 项目源文件字节：{sum(int(row["bytes"]) for row in source_rows):,}
- 数据及数据型结果文件：{len(data_rows):,}
- 显式全量数据目录：{len(explicit_rows)}
- 原始用户附件另作为根目录文件完整纳入。

## 关键边界

- 这是研究审阅包，不是 GPT Pro 已完成审阅的证据。
- 不授权预测上线、仓位、订单、券商连接、Paper/Shadow 或实盘。
- 可执行资产边界仍只有 510300.SH 与 CASH_CNY。
- 不做任何安全审计；具体未执行项目见 10_VERIFICATION_BOUNDARY.md。
- ZIP 完整性校验不等于研究结论正确。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 的审阅任务

你是一名严厉、证据优先的量化研究负责人。请审阅本包中的
510300_ASYMMETRIC_STRESS_HAZARD_V1。不要宣传成果，不要给买卖建议，不要做
安全审计。每项判断都引用包内相对路径、字段、代码函数或数据表。

## 核心问题

1. G0 工程重放、G1 BAD10 独立事件普查、数据源准入与 G2 机制链是否按原始
   章程执行？逐门给出 VALID、INVALID 或 UNCERTAIN。
2. BAD10 的总回报路径、t 日收盘信息、t+1 日开盘起算、10 日窗口、-4%阈值、
   分红、事件合并与非事件块是否实现正确？是否有标签泄漏或独立性夸大？
3. 58 个独立事件和 143 个非事件块中只有 45/123 可评分，缺失是否产生年份、
   行业、停牌、新上市或行情状态选择偏差？
4. 点时成分、历史成分股价格、2020-02-03 数据切换、停牌 NO_VIEW、申万行业
   生效/失效区间、ERP、DR007、政策利率和社融首次发布时间是否真正因果可用？
5. 特别复核冲突：严格时间可识别性预检曾因 record_updated_at 时钟把 G2 判为
   不可评估；随后用户要求按旧文件推进，正式 G2 使用官方 effective_date /
   out_date 历史区间且不把该额外预检作为活动门。请判断：
   - 这是合理恢复原章程，还是削弱 PIT 后得到的不可接受结果？
   - 正式 G2 的行业特征是否存在事后可得性污染？
   - 若存在，只允许修复同一冻结口径的实现/数据错误，不允许结果后调参。
6. 逐项核对 M1-M3、F1-F3、T1-T3、严格历史中秩、中位数、sqrt(F*T)、
   cube_root(M*F*T)、未来 5 日内部变化与价格基线的数学实现。
7. 核对 ROC AUC 单位是否真的是独立事件/非事件块，而不是重叠日；内部 AUC
   0.426378、价格基线 0.573984、差值 -0.147606 是否可复算。
8. 判断宏观门“3/4 子期只要求符号为正”是否过弱，尤其相关性分别约为
   0.098、0.030、-0.009、0.058；但不要据此修改本 V1 门槛救援结果。
9. 检查两次结果生成前的日期精度修复和最终 1.0.3 清单，确认它们只修复
   datetime64 精度，不改变研究语义。
10. 从项目级多重试验、历史复用、有效样本、置信区间、聚类依赖和基线公平性
    判断：当前终局拒绝是否可信，还是结果本身因实现或数据问题不可解释？

## 强制输出格式

### A. 一句话裁决

只能从以下三类选择一类，并解释：

- VALID_TERMINAL_G2_REJECTION
- RESULT_UNRELIABLE_REPAIR_SAME_FROZEN_PROTOCOL
- PROTOCOL_DESIGN_INVALID_NEW_V2_REQUIRED

第二类只允许修复可证明的代码或数据错误，禁止改窗口、符号、阈值、变量或模型。

### B. G0-G2 门槛表

对每道门列：裁决、证据路径、复算结果、最大不确定性、是否影响后续。

### C. P0 / P1 / P2 问题

- P0：足以推翻结果或证明泄漏/错误的缺陷；
- P1：不会立刻推翻，但显著削弱可信度；
- P2：可维护性、表达或次要证据问题。

每条必须给出证据路径、影响和最小修复动作。不要泛泛而谈。

### D. 数据与 PIT 审阅

逐数据源列出：身份、覆盖、发布时间/生效时钟、修订规则、缺失规则、是否足以
支持本次用途。重点审阅 503 只历史成分回填、6,000 个源修复原始文件、DR007
原始响应、行业历史区间和 5 个未映射成分日。

### E. 统计复算

复算或抽查独立样本数、四子期 Spearman、两类 AUC、覆盖损失；指出需要但包内
尚未提供的置信区间或敏感性分析。敏感性分析只能用于评估结论稳健性，不能把
失败版本改成通过。

### F. 停止/继续矩阵

分别对本 V1、同协议代码修复、新 V2 假设、数据底座维护给出：
STOP_PERMANENTLY、REPAIR_ONLY、NEW_PREREGISTRATION_REQUIRED 或 PASSIVE_MAINTENANCE。

### G. 最终行动清单

最多 10 项，按优先级排列。明确哪些事情不应再做。禁止输出目标仓位、交易
阈值、买卖指令、券商连接或实盘计划。
"""


def user_decisions_text() -> str:
    return """# 用户指令与本次打包范围

## 本目标中的关键用户指令

1. 读取原始粘贴文本，并在已经完成的工作基础上做完剩余部分。
2. “没做完的做完”。
3. “不需要审查，就按照旧文件推进”。
4. 将本目标所有内容和搜集到的所有数据打包，交给 GPT Pro 审阅。
5. “不需要任何安全审计”。

## 对这些指令的执行解释

- 正式研究按原始旧文件 G0→G1→G2→G3→G4→G5→G6 的顺序门推进。
- 额外构造过一项严格时间可识别性预检；它作为历史证据保留，但用户明确要求
  回到旧文件路线后，不再把它当作活动门。
- 正式 G2 失败，因此旧文件规定 G3-G6 不允许运行；这属于路线完成，不是遗漏。
- 本包采用“目标内全量、目标外排除”：目标分支全部变更、所有目标专属原始
  数据目录、所有目标内整理/派生数据、协议引用输入、原始用户文本均纳入。
- 不扫描安全、隐私、秘密、恶意文件或攻击面，也不做脱敏。
"""


def gate_timeline_text(result: dict[str, Any]) -> str:
    gate = result["g2_gate"]
    macro = gate["macro_lead_gate"]
    bad10 = gate["internal_bad10_gate"]
    period_lines = "\n".join(
        f"- {row['subperiod_id']}: n={row['scoreable_daily_observations']}, "
        f"Spearman={row['spearman_M_to_internal_change_t_plus_5']:.12f}, "
        f"positive={str(row['strictly_positive']).lower()}"
        for row in macro["subperiods"]
    )
    return f"""# 门槛时间线与当前结果

## 研究时间线

- c5be196：冻结 510300_ASYMMETRIC_STRESS_HAZARD_V1 章程。
- 7c38abb 至 5b845f7：完成 G0 干净重放和一次性 G1 BAD10 普查。
- b0879f1 至 1fb02b5：完成源准入、行业/政策利率修复和精确 DR007 准入。
- ad79a95 / 671d701 / 44b48d5：额外严格时间可识别性预检及其阻断结论。
- 用户随后明确要求不做额外审查，按旧文件原路线推进。
- 4ce8ac3 至 87bb989：采集并整理早期成分股价格历史。
- 015b64a 至 99ec2aa：冻结正式 G2，实现结果前 datetime 精度修复并形成 1.0.3。
- 67e5f60：记录正式 G2 终局拒绝。

## G0

工程重放通过，章程、代码、配置、结果和回执纳入版本链。

## G1

- 独立 BAD10 事件：58。
- 独立非事件 10 日块：143。
- 达到章程要求的 30 / 120，允许进入 G2。

## G2

宏观领先门：

{period_lines}

- 正方向子期：{macro["positive_subperiod_count"]} / {macro["total_subperiods"]}。
- 宏观门通过：{str(macro["passed"]).lower()}。

内部结构对 BAD10：

- 可评分事件：{bad10["scoreable_event_count"]}。
- 可评分非事件块：{bad10["scoreable_non_event_count"]}。
- 覆盖门通过：{str(bad10["coverage_passed"]).lower()}。
- 内部分数 ROC AUC：{bad10["internal_score_roc_auc"]:.15f}。
- 价格基线 ROC AUC：{bad10["price_baseline_roc_auc"]:.15f}。
- AUC 差：{bad10["auc_difference_internal_minus_price"]:.15f}。
- AUC 门通过：{str(bad10["auc_passed"]).lower()}。

## 终局

- 状态：{result["status"]}。
- G3 允许：{str(gate["g3_allowed"]).lower()}。
- 下一允许动作：{gate["next_allowed_action"]}。
- 未训练概率模型，未选择阈值，未读取组合指标，未生成仓位或订单。
"""


def data_scope_text(
    explicit_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
) -> str:
    explicit_lines = "\n".join(
        f"- {row['logical_root']}: {row['file_count']:,} files, "
        f"{row['bytes']:,} bytes, {row['coverage']} — {row['description']}"
        for row in explicit_rows
    )
    categories = Counter(row["category"] for row in data_rows)
    category_lines = "\n".join(
        f"- {category}: {count:,} files"
        for category, count in sorted(categories.items())
    )
    return f"""# 数据范围与完整性

## 明确要求全量覆盖的目标数据目录

{explicit_lines}

构建器逐目录比较磁盘文件集合与包内文件集合；上述目录均要求 COMPLETE，任何
文件未纳入都会使构建失败。

## 数据类别

{category_lines}

除目标专属目录外，包内还包括所有协议或报告能解析到的现存输入文件，例如：
510300/H00300 行情与总回报、分红、点时成分、现有成分行情、官方估值、十年
国债收益率、社融首次发布、行业原始表和其他实际依赖。

逐文件逻辑路径、大小、SHA-256、Git 状态和纳入原因见 06_DATA_FILE_INDEX.csv。
"""


def git_snapshot_text(
    git_info: dict[str, Any],
    states: dict[str, str],
    source_rows: list[dict[str, Any]],
) -> str:
    counts = Counter(states.values())
    state_lines = "\n".join(
        f"- {state}: {count:,}" for state, count in sorted(counts.items())
    )
    commits = git_info["branch_commit_log"] or "(no commits)"
    return f"""# Git 历史与可复现性快照

- 分支：{git_info["branch"]}
- HEAD：{git_info["head"]}
- 基准引用：{git_info["base_ref"]}
- merge-base：{git_info["merge_base"]}
- 分支变更路径数：{len(git_info["branch_changed_paths"]):,}
- 包内项目源文件数：{len(source_rows):,}

## 包内文件 Git 状态

{state_lines}

UNTRACKED_OR_IGNORED 主要包括按用户要求纳入的目标内原始数据；这不等于文件
无效，具体身份应与采集 manifest 和逐文件 SHA-256 对照。TRACKED_MODIFIED
表示打包的是当前实际工作副本，审阅者应核对它是否与冻结 manifest 一致。

## 本目标分支提交

    {commits.replace(chr(10), chr(10) + "    ")}
"""


def reproduction_text() -> str:
    return """# 主要复现入口

在 Windows PowerShell、项目根目录和已安装依赖的 Python 环境中，按协议状态
选择入口。不要重新执行已经消费的一次性动作覆盖旧回执。

## 协议与冻结

- scripts/freeze_510300_asymmetric_stress_hazard_v1.py
- scripts/freeze_510300_asymmetric_stress_hazard_v1_source_contracts_v1.py
- scripts/freeze_510300_asymmetric_stress_hazard_v1_g2_identifiability.py
- scripts/freeze_510300_asymmetric_stress_hazard_v1_g2_mechanism.py

## 运行入口

- scripts/run_510300_asymmetric_stress_hazard_v1_g0_replay.ps1
- scripts/run_510300_asymmetric_stress_hazard_v1_bad10_census.py
- scripts/run_510300_asymmetric_stress_hazard_v1_source_admission_v1.py
- scripts/run_510300_asymmetric_stress_hazard_source_remediation_admission_v1_0_1.py
- scripts/run_510300_asymmetric_stress_hazard_v1_g2_identifiability.py
- scripts/run_510300_asymmetric_stress_hazard_v1_g2_mechanism.py

## 数据采集入口

- scripts/acquire_510300_asymmetric_stress_hazard_source_remediation_v1_0_1.py
- scripts/acquire_510300_asymmetric_stress_hazard_dr007_tushare_v1_0_2.py
- scripts/acquire_510300_asymmetric_stress_hazard_v1_constituent_history.py

## 主要测试

- tests/test_510300_asymmetric_stress_hazard_v1.py
- tests/test_510300_asymmetric_stress_hazard_g0_replay_windows.py
- tests/test_510300_asymmetric_stress_hazard_v1_source_admission_v1.py
- tests/test_510300_asymmetric_stress_hazard_source_remediation_v1_0_1.py
- tests/test_510300_asymmetric_stress_hazard_dr007_tushare_v1_0_2.py
- tests/test_510300_asymmetric_stress_hazard_constituent_history.py
- tests/test_510300_asymmetric_stress_hazard_g2_identifiability_v1.py
- tests/test_510300_asymmetric_stress_hazard_g2_mechanism_v1.py

本包不声称已做新鲜解压后的端到端重放。GPT Pro 可以先做代码和数据复算，再
决定是否需要在隔离环境中运行；不要把重复运行一次性采集/claim 当成审阅步骤。
"""


def verification_boundary_text() -> str:
    performed = "\n".join(f"- {item}" for item in INTEGRITY_CHECKS)
    not_performed = "\n".join(f"- {item}" for item in NOT_PERFORMED)
    return f"""# 校验边界

## 已执行：只用于文件传输完整性

{performed}

## 按用户明确要求未执行

{not_performed}

因此，不应把本包描述为“安全”“已脱敏”“无凭据”“无恶意文件”或“完整端到端
复现通过”。ZIP CRC、重复成员为零和整包 SHA-256 只证明压缩包结构与传输
完整性，不证明研究结论正确，也不证明外部 GPT Pro 已经完成审阅。
"""


def build_generated_files(
    *,
    builder: PackageBuilder,
    git_info: dict[str, Any],
    attachment_bytes: bytes,
) -> tuple[dict[str, bytes], list[dict[str, Any]], list[dict[str, Any]]]:
    states = tracked_state_map(builder)
    source_rows = source_index_rows(builder, states)
    data_rows = data_index_rows(source_rows)
    explicit_rows = explicit_data_stats(builder)
    result = load_final_result()
    branch_rows = [
        {
            "path": relative,
            "exists_in_snapshot": relative in builder.sources,
            "archive_path": (
                f"{INTERNAL_ROOT}/project/{relative}"
                if relative in builder.sources
                else ""
            ),
        }
        for relative in git_info["branch_changed_paths"]
    ]
    generated: dict[str, bytes] = {
        "00_README_FIRST.md": readme_text(
            source_rows=source_rows,
            data_rows=data_rows,
            explicit_rows=explicit_rows,
            result=result,
        ).encode("utf-8"),
        "01_GPT_PRO_REVIEW_PROMPT.md": review_prompt_text().encode("utf-8"),
        "02_ORIGINAL_USER_PASTED_REVIEW.txt": attachment_bytes,
        "03_USER_DECISIONS_AND_SCOPE.md": user_decisions_text().encode("utf-8"),
        "04_GATE_TIMELINE_AND_CURRENT_RESULT.md": gate_timeline_text(result).encode(
            "utf-8"
        ),
        "05_DATA_SCOPE_AND_COMPLETENESS.md": data_scope_text(
            explicit_rows, data_rows
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
        "08_GIT_HISTORY_AND_REPRODUCIBILITY.md": git_snapshot_text(
            git_info, states, source_rows
        ).encode("utf-8"),
        "09_REPRODUCTION_ENTRYPOINTS.md": reproduction_text().encode("utf-8"),
        "10_VERIFICATION_BOUNDARY.md": verification_boundary_text().encode("utf-8"),
        "11_BRANCH_CHANGED_FILE_INDEX.csv": csv_bytes(
            branch_rows,
            ["path", "exists_in_snapshot", "archive_path"],
        ),
        "12_EXPLICIT_DATA_ROOT_INDEX.csv": csv_bytes(
            explicit_rows,
            ["logical_root", "description", "file_count", "bytes", "coverage"],
        ),
    }
    metadata = {
        "package_id": PACKAGE_BASENAME,
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "scope": "ALL_CONTENT_AND_ALL_COLLECTED_DATA_WITHIN_THIS_GOAL",
        "original_attachment": {
            "source_path": str(ORIGINAL_ATTACHMENT),
            "bytes": len(attachment_bytes),
            "sha256": hashlib.sha256(attachment_bytes).hexdigest(),
            "archive_member": f"{INTERNAL_ROOT}/02_ORIGINAL_USER_PASTED_REVIEW.txt",
        },
        "source_file_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "generated_file_count_including_metadata": len(generated) + 1,
        "explicit_data_roots": explicit_rows,
        "missing_text_references_not_packaged": sorted(builder.missing_references),
        "integrity_checks": list(INTEGRITY_CHECKS),
        "not_performed": list(NOT_PERFORMED),
        "git": git_info,
        "final_research_status": result["status"],
        "g3_allowed": bool(result["g2_gate"]["g3_allowed"]),
        "position_impact": int(result["position_impact"]),
    }
    generated["13_PACKAGE_METADATA.json"] = json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    return generated, source_rows, data_rows


def assert_new_outputs() -> None:
    existing = [
        path
        for path in (ZIP_PATH, SHA256_PATH, RECEIPT_PATH, UPLOAD_MESSAGE_PATH)
        if path.exists()
    ]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"交付物已存在，禁止覆盖：{joined}")


def write_zip(
    builder: PackageBuilder,
    generated: dict[str, bytes],
) -> tuple[int, int]:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{PACKAGE_BASENAME}_",
        suffix=".zip.tmp",
        dir=DELIVERABLES,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for relative, record in sorted(builder.sources.items()):
                archive.write(
                    record.path,
                    arcname=f"{INTERNAL_ROOT}/project/{relative}",
                )
            for relative, content in sorted(generated.items()):
                archive.writestr(f"{INTERNAL_ROOT}/{relative}", content)
        temporary_path.replace(ZIP_PATH)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    required = {
        f"{INTERNAL_ROOT}/00_README_FIRST.md",
        f"{INTERNAL_ROOT}/01_GPT_PRO_REVIEW_PROMPT.md",
        f"{INTERNAL_ROOT}/02_ORIGINAL_USER_PASTED_REVIEW.txt",
        f"{INTERNAL_ROOT}/06_DATA_FILE_INDEX.csv",
        f"{INTERNAL_ROOT}/07_INCLUDED_FILE_INDEX.csv",
        f"{INTERNAL_ROOT}/13_PACKAGE_METADATA.json",
        (
            f"{INTERNAL_ROOT}/project/reports/research/"
            "510300_ASYMMETRIC_STRESS_HAZARD_V1_G2_MECHANISM.md"
        ),
        (
            f"{INTERNAL_ROOT}/project/reports/research/"
            "510300_asymmetric_stress_hazard_v1_g2_mechanism.json"
        ),
    }
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = archive.namelist()
        duplicates = len(members) - len(set(members))
        missing_required = sorted(required - set(members))
        bad_member = archive.testzip()
    if duplicates:
        raise RuntimeError(f"ZIP 存在重复成员：{duplicates}")
    if missing_required:
        raise RuntimeError(f"ZIP 缺少关键成员：{missing_required}")
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC 失败成员：{bad_member}")
    expected_count = len(builder.sources) + len(generated)
    if len(members) != expected_count:
        raise RuntimeError(
            f"ZIP 成员数错误：expected={expected_count}, actual={len(members)}"
        )
    return len(members), duplicates


def write_sidecars(
    *,
    builder: PackageBuilder,
    generated: dict[str, bytes],
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    member_count: int,
    duplicate_count: int,
) -> dict[str, Any]:
    package_sha256 = sha256_file(ZIP_PATH)
    SHA256_PATH.write_text(
        f"{package_sha256}  {ZIP_PATH.name}\n",
        encoding="utf-8",
    )
    receipt = {
        "package_id": PACKAGE_BASENAME,
        "zip_path": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": package_sha256,
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "source_file_count": len(source_rows),
        "source_bytes": builder.total_bytes,
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "generated_file_count": len(generated),
        "zip_member_count": member_count,
        "duplicate_member_count": duplicate_count,
        "crc_testzip_bad_member": None,
        "explicit_data_root_count": len(EXPLICIT_DATA_ROOTS),
        "explicit_data_root_coverage": "COMPLETE",
        "status": (
            "PASS_FULL_GOAL_RESEARCH_PACKAGE_CRC_DUPLICATE_MANIFEST_"
            "AND_WHOLE_ZIP_SHA256"
        ),
        "integrity_checks": list(INTEGRITY_CHECKS),
        "not_performed": list(NOT_PERFORMED),
        "security_audit_performed": False,
        "privacy_scan_performed": False,
        "secret_scan_performed": False,
        "malware_scan_performed": False,
        "redaction_performed": False,
        "fresh_extraction_replay_performed": False,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    upload_message = f"""请使用 GPT Pro 审阅附件：{ZIP_PATH.name}

先读压缩包根目录的 00_README_FIRST.md，再严格执行
01_GPT_PRO_REVIEW_PROMPT.md。

这是 510300_ASYMMETRIC_STRESS_HAZARD_V1 的目标内全量包，包含原始需求、全部
目标内采集数据、实际输入、代码、测试、结果和回执。请核实 G0-G2，尤其审查
严格时间预检与后续旧文件路线之间的冲突，以及内部 AUC 0.426378 低于价格基线
0.573984 的终局拒绝是否可信。不要调参救援，不要给交易建议，不要做安全审计。

压缩包 SHA-256：{package_sha256}
完整性：CRC 通过、重复成员 0、显式目标数据目录覆盖 COMPLETE、逐源文件索引
已包含。未做安全、隐私、秘密、恶意文件、脱敏、攻击面或新鲜解压重放审计。
"""
    UPLOAD_MESSAGE_PATH.write_text(upload_message, encoding="utf-8")
    return receipt


def inventory_summary(
    builder: PackageBuilder,
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "package_id": PACKAGE_BASENAME,
        "source_file_count": len(source_rows),
        "source_bytes": builder.total_bytes,
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "categories": dict(Counter(row["category"] for row in source_rows)),
        "explicit_data_roots": explicit_data_stats(builder),
        "missing_reference_count": len(builder.missing_references),
        "missing_references": sorted(builder.missing_references),
        "outputs_written": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="只盘点和计算索引，不写 ZIP 或 sidecar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = PackageBuilder()
    print("阶段 1/4：收集目标文件与引用依赖", file=sys.stderr, flush=True)
    git_info, attachment_bytes = collect_sources(builder)
    print(
        f"已收集 {len(builder.sources):,} 个文件，"
        f"{builder.total_bytes:,} 字节",
        file=sys.stderr,
        flush=True,
    )
    if args.inventory_only:
        source_rows = [
            {
                "workspace_logical_path": relative,
                "category": classify_source(relative),
                "bytes": record.path.stat().st_size,
            }
            for relative, record in sorted(builder.sources.items())
        ]
        data_rows = data_index_rows(source_rows)
        print(
            json.dumps(
                inventory_summary(builder, source_rows, data_rows),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return
    print("阶段 2/4：计算逐文件哈希与生成导航索引", file=sys.stderr, flush=True)
    generated, source_rows, data_rows = build_generated_files(
        builder=builder,
        git_info=git_info,
        attachment_bytes=attachment_bytes,
    )
    assert_new_outputs()
    print("阶段 3/4：写入 ZIP", file=sys.stderr, flush=True)
    member_count, duplicate_count = write_zip(builder, generated)
    print("阶段 4/4：写入整包哈希与构建回执", file=sys.stderr, flush=True)
    receipt = write_sidecars(
        builder=builder,
        generated=generated,
        source_rows=source_rows,
        data_rows=data_rows,
        member_count=member_count,
        duplicate_count=duplicate_count,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
