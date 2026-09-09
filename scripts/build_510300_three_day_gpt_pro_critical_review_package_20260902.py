"""构建 2026-08-31 至 2026-09-02 的 510300 三日 GPT Pro 批判性审阅包。

本脚本只使用 Python 标准库。它收集三日内与 510300/沪深300 直接相关的
协议、代码、测试、报告、状态回执和可携带派生数据，并补入少量重放所需的
历史依赖。按用户要求，不执行安全、隐私、秘密、恶意文件或脱敏审计；仅做
ZIP CRC、重复成员和整包 SHA-256 校验。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
START_AT = datetime(2026, 8, 31, 0, 0, 0, tzinfo=TIME_ZONE)
SNAPSHOT_AT = datetime.now(TIME_ZONE)

PACKAGE_BASENAME = "510300_THREE_DAY_GPT_PRO_CRITICAL_REVIEW_20260831_20260902"
INTERNAL_ROOT = PACKAGE_BASENAME
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
UPLOAD_MESSAGE_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"

MAX_GENERAL_FILE_BYTES = 64 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 8 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 350 * 1024 * 1024

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

RELEVANCE_PATTERNS = (
    "510300",
    "csi300",
    "hs300",
    "h00300",
    "沪深300",
    "priority_forward",
    "primary_market",
    "pit_fundamental_underreaction",
    "post_close_offshore",
    "creation_data_readiness",
    "derivative_pressure",
    "episodic_alpha",
    "total_return_component",
    "cf_dr_rc",
    "structural_equity_risk",
    "structural_prediction",
    "conditional_mechanism",
    "expectations_news",
    "trend_lifecycle",
    "oracle_information",
    "up20",
    "anchored_sparse_mean_reversion",
    "known_state_policy",
    "three_state_trend",
    "gaussian_hmm",
    "multi_scale_nonlinear",
    "absorption_ratio",
)

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "deliverables",
    "review_packages",
    "tmp",
}

SCAN_ROOTS = (
    "config",
    "docs",
    "reports",
    "research",
    "scripts",
    "tests",
    "paper",
    "output",
)

ROOT_CONTEXT_FILES = (
    "CONTEXT.md",
    "RESEARCH_STATUS.md",
    "requirements.txt",
    "pytest.ini",
    "docs/DECISIONS.md",
)

SHARED_CODE_ROOTS = (
    "market_data",
    "src",
    "backtest",
)

STRUCTURAL_DATA_ROOTS = (
    "data/curated/510300_structural_equity_risk_premium_engine_v1",
    "data/processed/510300_structural_equity_risk_premium_engine_v1",
    "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1",
)

CORE_PORTABLE_INPUTS = (
    "data/raw/market/510300_daily_raw.parquet",
    "data/raw/market/510300_daily_raw.metadata.json",
    "data/raw/market/000300_daily_raw.parquet",
    "data/raw/market/000300_daily_raw.metadata.json",
    "data/raw/market/H00300_total_return_daily_raw.parquet",
    "data/raw/market/H00300_total_return_daily_raw.metadata.json",
    "data/processed/510300_daily_total_return.parquet",
    "data/reference/510300_dividends.csv",
    "data/reference/510300_dividends_coverage.json",
    "data/raw/market/510300_1m_tushare_raw.parquet",
    "data/raw/market/510300_1m_tushare_raw.metadata.json",
    "data/raw/market/000300_1m_tushare_raw.parquet",
    "data/raw/market/000300_1m_tushare_raw.metadata.json",
    "data/raw/market/510300_daily_2015_v2.parquet",
    "data/raw/market/H00300_total_return_daily_2015_v2.parquet",
    "data/raw/market/510300_daily_downside_risk_v1.parquet",
    "data/raw/market/000300_price_index_return_dispersion_v1.parquet",
    "data/raw/market/csi300_monthly_return_dispersion_v1.parquet",
    "data/raw/market/csi300_absorption_ratio_v1_total_return_close.parquet",
    "data/reference/510300_downside_risk_price_corrections_v1.csv",
)

PRIORITY_MANIFEST_PATH = ROOT / "config" / "priority_forward_research_operations_v1_8_manifest.json"


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
        self.exclusions: list[dict[str, Any]] = []

    def add(self, path: Path, reason: str) -> bool:
        try:
            if not path.exists() or not path.is_file():
                self.exclusions.append(
                    {
                        "path": self._display_path(path),
                        "reason": "MISSING_OR_NOT_A_FILE",
                        "bytes": None,
                    }
                )
                return False
            relative = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
        except (OSError, ValueError) as exc:
            self.exclusions.append(
                {
                    "path": self._display_path(path),
                    "reason": f"UNREADABLE_OR_OUTSIDE_LOGICAL_ROOT:{type(exc).__name__}",
                    "bytes": None,
                }
            )
            return False

        if size > MAX_GENERAL_FILE_BYTES:
            self.exclusions.append(
                {
                    "path": relative,
                    "reason": "FILE_ABOVE_64_MIB_PORTABLE_LIMIT",
                    "bytes": size,
                }
            )
            return False

        record = self.sources.get(relative)
        if record is None:
            record = SourceRecord(path=path)
            self.sources[relative] = record
        record.reasons.add(reason)
        return True

    def exclude(self, path: Path, reason: str) -> None:
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(path)
        try:
            size: int | None = path.stat().st_size if path.is_file() else None
        except OSError:
            size = None
        self.exclusions.append({"path": relative, "reason": reason, "bytes": size})

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)


def iter_files_safe(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in EXCLUDED_PARTS]
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            try:
                if path.is_file():
                    yield path
            except OSError:
                continue


def local_modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, TIME_ZONE)


def read_text_best_effort(path: Path, limit: int = MAX_TEXT_SCAN_BYTES) -> str:
    raw = path.read_bytes()
    if len(raw) > limit:
        raw = raw[:limit]
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def is_relevant_text_or_path(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix().lower()
    if any(pattern in relative for pattern in RELEVANCE_PATTERNS):
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            return False
        text = read_text_best_effort(path).lower()
    except OSError:
        return False
    return any(pattern in text for pattern in RELEVANCE_PATTERNS)


def collect_recent_relevant_sources(builder: PackageBuilder) -> None:
    for relative_root in SCAN_ROOTS:
        root = ROOT / relative_root
        for path in iter_files_safe(root):
            try:
                if local_modified_at(path) < START_AT:
                    continue
                if path.suffix.lower() == ".pyc" or "__pycache__" in path.parts:
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix.lower() not in {
                    ".parquet",
                }:
                    continue
                if is_relevant_text_or_path(path):
                    builder.add(path, "THREE_DAY_RELEVANT_SOURCE")
            except OSError:
                builder.exclude(path, "RECENT_SOURCE_STAT_FAILED")


def collect_recent_research_data(builder: PackageBuilder) -> None:
    root = ROOT / "data" / "research"
    for path in iter_files_safe(root):
        try:
            if local_modified_at(path) < START_AT:
                continue
            relative = path.relative_to(root)
            if not relative.parts or not relative.parts[0].lower().startswith("510300_"):
                continue
            if path.suffix.lower() not in {".parquet", ".csv", ".json", ".jsonl", ".md"}:
                continue
            builder.add(path, "THREE_DAY_510300_DERIVED_DATA")
        except OSError:
            builder.exclude(path, "RESEARCH_DATA_STAT_FAILED")


def collect_context_and_dependencies(builder: PackageBuilder) -> None:
    for relative in ROOT_CONTEXT_FILES:
        builder.add(ROOT / relative, "PROJECT_CONTEXT")

    for relative_root in SHARED_CODE_ROOTS:
        for path in iter_files_safe(ROOT / relative_root):
            if path.suffix.lower() in TEXT_SUFFIXES and "__pycache__" not in path.parts:
                builder.add(path, "SHARED_REPLAY_CODE")

    for relative_root in STRUCTURAL_DATA_ROOTS:
        for path in iter_files_safe(ROOT / relative_root):
            builder.add(path, "STRUCTURAL_ENGINE_PORTABLE_DATA")

    for relative in CORE_PORTABLE_INPUTS:
        path = ROOT / relative
        if path.exists():
            builder.add(path, "CORE_PORTABLE_INPUT")

    explicit_support = (
        "research/frozen_protocol_support_v1.py",
        "config/priority_forward_research_operations_v1_8_manifest.json",
        "scripts/run_priority_forward_codex_automation_v1_8.py",
    )
    for relative in explicit_support:
        builder.add(ROOT / relative, "EXPLICIT_GOVERNANCE_DEPENDENCY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_priority_manifest_dependencies(
    builder: PackageBuilder,
) -> list[dict[str, Any]]:
    if not PRIORITY_MANIFEST_PATH.exists():
        return []

    manifest = json.loads(PRIORITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    logical_root = ROOT.absolute()
    resolved_root = ROOT.resolve()
    rows: list[dict[str, Any]] = []

    for entry in manifest.get("files", []):
        relative = str(entry.get("path", "")).replace("\\", "/")
        expected_sha = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        logical_path = logical_root / Path(relative)
        exists = logical_path.exists() and logical_path.is_file()
        actual_bytes: int | None = None
        actual_sha: str | None = None
        resolved_path: str | None = None
        contained_after_resolution: bool | None = None
        error: str | None = None

        if exists:
            try:
                resolved = logical_path.resolve()
                resolved_path = str(resolved)
                contained_after_resolution = path_is_relative_to(resolved, resolved_root)
                actual_bytes = logical_path.stat().st_size
                actual_sha = sha256_file(logical_path)
                builder.add(logical_path, "PRIORITY_V1_8_FROZEN_DEPENDENCY")
            except OSError as exc:
                error = f"{type(exc).__name__}:{exc}"
        else:
            error = "MISSING"

        rows.append(
            {
                "path": relative,
                "exists": exists,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "sha256_match": bool(expected_sha and actual_sha and expected_sha == actual_sha),
                "resolved_path": resolved_path,
                "contained_after_resolution": contained_after_resolution,
                "error": error,
            }
        )
    return rows


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_snapshot(builder: PackageBuilder) -> dict[str, Any]:
    tracked_output = run_git(["ls-files"])
    tracked = set(tracked_output.splitlines()) if tracked_output else set()
    selected = set(builder.sources)
    commits_output = run_git(
        [
            "log",
            "--since=2026-08-31 00:00:00 +0800",
            "--date=iso-local",
            "--pretty=format:%h\t%ad\t%s",
        ]
    )
    commits = commits_output.splitlines() if commits_output else []
    return {
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "commits_in_period": commits,
        "selected_file_count": len(selected),
        "selected_tracked_count": len(selected & tracked),
        "selected_untracked_count": len(selected - tracked),
        "selected_untracked_examples": sorted(selected - tracked)[:100],
    }


def report_category(name: str) -> str:
    lower = name.lower()
    if "pit_fundamental" in lower or "official_facts" in lower:
        return "PIT官方事实与覆盖"
    if any(token in lower for token in ("episodic", "derivative", "offshore", "creation")):
        return "情景Alpha与前向"
    if any(token in lower for token in ("structural", "conditional", "cf_dr_rc", "total_return")):
        return "结构机制与测量"
    if any(token in lower for token in ("trend", "hmm", "oracle", "up20", "technical", "absorption")):
        return "技术状态与信息预算"
    if "mean_reversion" in lower:
        return "微观结构与均值回归"
    return "其他510300研究"


def extract_report_index(builder: PackageBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, record in sorted(builder.sources.items()):
        if not relative.startswith("reports/research/") or not relative.lower().endswith(".md"):
            continue
        try:
            text = record.path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        title = ""
        status_line = ""
        for line in text.splitlines():
            stripped = line.strip()
            if not title and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
            if not status_line and re.search(
                r"状态|裁决|最终决定|结论先行|CURRENT_|NO_VIEW|REJECTED|BLOCKED",
                stripped,
                flags=re.IGNORECASE,
            ):
                status_line = stripped
            if title and status_line:
                break
        rows.append(
            {
                "category": report_category(record.path.name),
                "path": relative,
                "modified_at": local_modified_at(record.path).isoformat(),
                "bytes": record.path.stat().st_size,
                "title": title,
                "first_status_or_judgment_line": status_line,
            }
        )
    return rows


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def included_file_index(builder: PackageBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, record in sorted(builder.sources.items()):
        stat = record.path.stat()
        rows.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, TIME_ZONE).isoformat(),
                "selection_reason": ";".join(sorted(record.reasons)),
                "git_tracking": "PENDING_GIT_SNAPSHOT",
            }
        )
    return rows


def render_readme(
    builder: PackageBuilder,
    report_rows: list[dict[str, Any]],
    git_info: dict[str, Any],
) -> str:
    total_bytes = sum(record.path.stat().st_size for record in builder.sources.values())
    category_counts = Counter(row["category"] for row in report_rows)
    categories = "\n".join(
        f"- {category}：{count} 份研究报告" for category, count in sorted(category_counts.items())
    )
    return f"""# 510300 三日量化研究：GPT Pro 批判性审阅包

## 交付结论

本包冻结的自然日范围为 **2026-08-31 00:00:00 至 2026-09-02 当前构建时点（Asia/Shanghai）**。包内纳入 {len(builder.sources):,} 个源文件，共 {total_bytes:,} 字节；其中三日研究报告索引 {len(report_rows)} 份。

这不是“挑最好结果”的宣传包，而是让 GPT Pro 回答：我们是否在三天内铺得太宽、每条线是否挖得不够深、统计与点时证据是否不足、版本和状态是否过度膨胀、哪些工作应该停止。

当前不能从这些文件得出可交易策略已经通过。根权威状态仍把系统定义为 `DISCOVERY_ONLY / ABSTAIN / POSITION_UNSET`，不生成仓位目标、Paper/Shadow 信号、订单、券商连接或实盘授权。部分分支内部使用 `CURRENT_HOLDING_ROUTE=CASH_CNY`，本包要求审阅者专门判断该命名是否会与根状态语义冲突；它不能被解释为用户真实持仓。

## 必读顺序

1. `01_GPT_PRO_REVIEW_PROMPT.md`
2. `02_USER_QUESTION_AND_REVIEW_BOUNDARY.md`
3. `03_THREE_DAY_WORK_MAP.md`
4. `04_KNOWN_WEAKNESSES_TO_ATTACK.md`
5. `05_READING_ROUTE.md`
6. `07_PRIORITY_FORWARD_PATH_CONTAINMENT_SNAPSHOT.md`
7. `09_RESEARCH_REPORT_INDEX.csv`
8. 再按索引进入 `reports/`、`config/`、`research/`、`scripts/`、`tests/` 和 `data/`

## 三日研究板块

{categories or '- 未识别到研究报告。'}

## 证据解释规则

- `NO_VIEW`、`NOT_ALLOWED`、`BLOCKED`、`REJECTED_FROZEN`、`DISCOVERY_ONLY`、`FORWARD_ONLY`、`ABSTAIN` 是不同状态。
- 零候选、零成熟事件或 `None` 收益不等于零收益，更不等于市场没有该机制。
- Oracle/完美未来状态的 `PASS` 只表示条件价值或能力上界，不是现实可预测性通过。
- 会计恒等式、描述性条件地图、风险对象测量通过不等于未来总收益可预测。
- 原数据准入失败与“不完整覆盖条件样本”结果必须分开；条件样本失败不能洗掉原始覆盖阻断，条件样本成功也不能外推到缺失事件。
- 已拒绝分支不得换参数、窗口、方向、样本、代理或成本救援。审阅者可以质疑协议是否合理，但不能用结果后调参把旧分支变成赢家。

## Git 快照提示

- 当前分支：`{git_info.get('branch')}`
- 当前 HEAD：`{git_info.get('head')}`
- 三日内提交数：{len(git_info.get('commits_in_period', []))}
- 包内源文件中 Git 已跟踪：{git_info.get('selected_tracked_count', 0)}；未跟踪：{git_info.get('selected_untracked_count', 0)}

Git 状态不是研究结论，但大量未跟踪产物会削弱可重放性、差异审阅和版本边界，应由 GPT Pro 评价其治理后果。

## 本次校验边界

按用户要求，不做数据安全、隐私、秘密、恶意文件或脱敏审计，也不做全库安全扫描。构包只验证 ZIP CRC、重复成员和整包 SHA-256。`.env`、虚拟环境、缓存、临时 OCR 页面、CNInfo 原始 PDF 全库、旧审阅 ZIP 和大体量原始归档不纳入。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 的任务：严厉审阅我们三天的 510300 研究

你现在扮演独立量化研究负责人、统计审稿人、数据工程负责人和项目组合经理。用户不要鼓励性总结，也不要安全审计；用户要知道我们过去三天到底哪里做得不够、哪里不深入、哪里铺得太宽、哪里方法或实现不好。

请先读 00—07 号导航文件，再沿 09 号研究报告索引进入直接证据。不要只复述报告标题和状态；至少下钻到冻结配置/manifest、研究实现、机器可读结果、测试和派生表。报告与代码冲突时，以可重放证据为准，并明确冲突。

## 审阅纪律

1. 不给买卖指令、仓位建议、目标收益或实盘方案；只做研究裁决和资源配置。
2. 不做隐私、秘密、恶意文件、脱敏或通用安全审计。
3. 不用新参数救援任何已冻结拒绝分支。你可以指出原协议设计得差、门槛武断或问题定义错误，但旧结果必须保留。
4. 缺失数据、`NO_VIEW`、`NOT_ALLOWED`、零候选、程序失败和真实负结果必须严格分开。
5. Oracle、描述统计、会计恒等式、测量有效性与现实预测能力必须严格分开。
6. 包中没有足够证据时写 `CANNOT_VERIFY_FROM_PACKAGE`，并列出缺失的精确文件/字段；不要脑补。
7. 任何实质判断必须引用包内相对路径，最好同时引用配置、代码和结果三层证据。

## 必须深审的代表性链条

### A. 技术、状态与信息预算

- PCA 吸收率为何在上市历史覆盖门前失败，门槛本身是否合理：`reports/research/510300_csi300_absorption_ratio_timing_v1_data_gate_failure.md`
- 多尺度趋势—反转、ADX 三态路由、因果 HMM 是否只是在同一历史上换表达：对应 `reports/research/510300_MULTI_SCALE_NONLINEAR_TREND_REVERSION_V1.md`、`510300_THREE_STATE_TREND_ROUTER_V1_0_1_RESULT.md`、`510300_CAUSAL_GAUSSIAN_HMM_REGIME_ROUTER_V1_RESULT.md`
- 完美状态 Oracle 通过、现实信息预算却停止，是否说明目标定义或研究资源方向错误：`510300_KNOWN_STATE_POLICY_ORACLE_V1_0_1_RESULT.md`、`510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE_RESULT.md`、UP20 V1/V1.1 结果
- 这些分支应视为一个统一多重试验家族，还是多个正交问题？请给出可辩护的 family 边界和 multiplicity 处理。

### B. 结构机制、预测与功效

- 从结构性 ERP 状态面板到条件地图、滚动样本外预测、功效审计、总回报成分账本和 CF/DR/RC 自有对象测量，这条链是否真正逐层深化，还是把同一弱数据拆成更多文件。
- 重点复算 `reports/research/510300_STRUCTURAL_SIGNAL_POWER_AND_IDENTIFIABILITY_AUDIT_V1.md` 中数千至十万年的样本需求。检查公式、效应定义、依赖调整、单位和结论是否合理；若错，请给出正确算法和数量级。
- 判断 `RC` 自有对象测量全 PASS 是否只证明对已实现风险的同义测量，而没有新增预测信息。
- 检查 `TOTAL_RETURN_COMPONENT_LEDGER` 的恒等式闭合是否被过度包装，以及盈利/倍数、成员权重与跟踪残差是否可识别。

### C. 微观结构与情景 Alpha

- 有锚稀疏均值回归在 Phase A 零个成本后完整事件时就停止，是否说明假设定义、阈值或锚点代理在研究前就缺乏经济可行性；不要调参救援。
- 情景 Alpha V1、运行后裁决、聚合 OI V2 一次性关闭是否在科学上充分；重点检查 V2 只有 38 个完整 Pressure 值却要求 prior-252 分位时，协议是否从一开始就不可满足。
- 开盘折价 8 个成熟事件且均值为负，却因预注册 40 个事件保持“样本不足”，这种状态是否有资源价值，何时应停止被动日志。
- 盘后 A50 传导 60 日门、`beta_LCB*r_post>=42bp` 和一级市场 120 日完整门是否有事前功效/经济依据，还是治理数字先行。

### D. PIT 官方事实与不完整覆盖分析

- 9 月 1—2 日出现 V1.6、V1.7、V1.8、V1.9、V1.18、V1.19、V1.21 及多种 direct-PDF/OCR/merge/census 文件。判断这是必要的可追溯演进，还是版本爆炸、脚本碎片化和状态难以重放。
- 原始研究仍 `BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE`；当前完整依赖事件 7,252/8,223=88.192%。在此基础上又做条件样本 Phase 1，是否构成选择偏差或绕过数据门？
- 条件样本最终只有 55 个唯一分数、36 个可形成完整五档的年度×报告类型单元，60D/120D 可评价事件只有 30/23。请判断检验、Bootstrap、分组和跨期稳健性是否有足够独立样本。
- 价格面板 2018—2022 偏训练证券、2024—2026 偏留出证券；请说明这种非矩形缺失如何影响机制结论，是否连“条件样本失败”都不能外推。

### E. 工程与治理

- 检查 `07_PRIORITY_FORWARD_PATH_CONTAINMENT_SNAPSHOT.md`：项目 `data` 为指向 E 盘的 Junction 后，V1.8 manifest 的逻辑相对路径解析到项目根之外。判断 containment 规则与数据根设计谁错了，给出不改写旧回执的版本化修复方案。
- 检查根 `RESEARCH_STATUS.md` 的 `ABSTAIN/POSITION_UNSET` 与 Batch-2 `CURRENT_HOLDING_ROUTE=CASH_CNY` 的语义冲突风险。提出唯一权威字段和状态机。
- 包内源文件大量未被 Git 跟踪，三日内可能没有提交。判断如何影响 manifest、重放、diff、并行研究和失败证据保全。
- 评估“每个分支有冻结文件和测试”是否掩盖了项目级多重试验、注意力稀释、缺少独立复制和没有前向成熟证据。

## 必须回答的问题

1. 一句话总判断：这三天是有效深化，还是高产但过宽、过碎、证据密度低？
2. 最严重的 10 个问题，按 P0/P1/P2 排序。每个问题给证据路径、错误机制、影响范围和具体修复。
3. 哪些结论可信；哪些只在限定样本内可信；哪些被措辞放大；哪些目前不可验证。
4. 哪些工作不够深入：需要补什么最小证据，补完才能回答什么问题。
5. 哪些工作太宽泛：哪些文件/分支其实属于同一个研究家族，应合并管理或停止并行。
6. 统计审阅：有效独立样本、多重检验、重叠 horizon、选择偏差、PIT/修订、基准、成本和最低可检测效应分别哪里有问题。
7. 工程审阅：版本爆炸、单一权威状态、Git/manifest、路径可移植性、重放入口、测试覆盖和状态回执哪里不足。
8. 研究价值审阅：哪些诊断/Oracle/Atlas 对最终决策几乎没有边际价值，属于聪明但低价值的工作。
9. 去留矩阵：`STOP_PERMANENTLY`、`STOP_CURRENT_VERSION_RETAIN_EVIDENCE`、`PASSIVE_DATA_ONLY`、`COMPLETE_ONE_MISSING_EVIDENCE`、`CONTINUE_STRICT_FORWARD_ONLY`、`ONE_NEW_ORTHOGONAL_TEST`、`CANNOT_VERIFY`。
10. 给出 7/30/90 天整改路线，但同时最多允许 2 条主动研究线；列出明确停止条件，禁止再铺新分支。

## 输出格式

请用中文，按下列顺序：

1. **一句话总判断**
2. **最刺耳但最重要的三个事实**
3. **P0/P1/P2 十大问题表**
4. **逐研究家族深度评分表**：问题定义、数据、PIT、统计、经济机制、实现、可重放性、决策价值各 1—5 分
5. **可信/限定可信/夸大/不可验证结论表**
6. **项目级多重试验与有效样本复核**
7. **代码、版本和状态治理复核**
8. **去留矩阵**
9. **7/30/90 天整改路线**
10. **需要补看的最小证据清单**

不要写“整体框架很完善”“继续收集更多数据”“加强风控”之类泛话。每条建议都要能落到一个具体问题、文件、字段、计算或停止动作。
"""


def user_question_text() -> str:
    return """# 用户问题与审阅边界

## 用户真正想问的

请不要替我们包装成果。请直接回答：

- 过去三天做的 510300 研究，哪些地方不够深入？
- 是否同时铺了太多方向，导致每条线都停在协议、诊断或一次性失败层？
- 哪些地方太宽泛、问题定义不清、数据不足、统计功效不足或工程治理过重？
- 哪些结论写得太满，哪些 PASS 实际只是假设上界、描述性或会计闭合？
- 哪些工作做得不好，应该立即停止；哪些值得用有限资源补完？

## 不要做的事

- 不做安全、隐私、秘密、恶意文件或脱敏审计。
- 不给 510300 买卖指令、实际仓位、订单或券商连接建议。
- 不通过改参数、窗口、方向、代理、成本或样本期救援冻结失败。
- 不把 `NO_VIEW`、`NOT_ALLOWED`、零候选或数据阻断解释成零收益或看空。

## 资产边界

可执行资产仍只有 `510300.SH` 与 `CASH_CNY`。000300、H00300、成分股、IF、期权、A50 和宏观数据只能作为观察、归因、基准或信息源，不能被悄悄扩展为交易资产。
"""


def work_map_text() -> str:
    return """# 三日工作地图

## 1. 技术状态、趋势与信息可达性

- PCA 吸收率输入门失败：上市不足 500 日的新成员使 986 个交易日不满足冻结覆盖门，禁止收益评价。
- 多尺度非线性趋势—反转一次性机制检验失败。
- ADX 三态路由与因果 HMM 路由均未达到净 Sharpe 1.2，且状态机制门失败。
- 完美未来状态 Oracle 的条件价值通过，但现实信息预算保守验收停止；UP20 V1 与 V1.1 都冻结拒绝。
- 技术因子准入裁决认为同一历史上没有未使用且合格的回溯候选，只允许新增严格前向证据。

入口报告：

- `reports/research/510300_PHASE_SPECIFIC_TECHNICAL_ADJUDICATION_20260831.md`
- `reports/research/510300_POST_ATLAS_PROJECT_STATE_V1_0_1_RECONCILIATION.md`
- `reports/research/510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE_RESULT.md`

## 2. 结构性 ERP、条件地图、预测与测量

- 第一阶段建立 CF/DR/RC 状态面板，但财务版本、代理权重和 PIT 来源仍有明确限制。
- 条件机制地图没有在全样本与 2021+ 主期限重复得到方向支持。
- 冻结滚动样本外结构预测失败或 NO_VIEW；没有组合评估。
- 功效审计把小幅 MSE 改善的样本需求估到极长年限，需由外部审阅者复算。
- 总回报成分账本完成算术恒等式；CF/DR 自有对象仍多为 PARTIAL/NO_VIEW，RC 自有对象测量 PASS，但不等于回报预测。

入口报告：

- `reports/research/510300_STRUCTURAL_EQUITY_RISK_PREMIUM_ENGINE_V1.md`
- `reports/research/510300_CONDITIONAL_MECHANISM_MAP_V1.md`
- `reports/research/510300_STRUCTURAL_PREDICTION_V1.md`
- `reports/research/510300_STRUCTURAL_SIGNAL_POWER_AND_IDENTIFIABILITY_AUDIT_V1.md`
- `reports/research/510300_TOTAL_RETURN_COMPONENT_LEDGER_V1.md`
- `reports/research/510300_CF_DR_RC_OWN_OBJECT_MEASUREMENT_VALIDITY_V1.md`

## 3. 微观结构与情景 Alpha

- 有锚稀疏均值回归 Phase A 没有一个事件达到成本边界，按规则不运行网格回测。
- 情景 Alpha V1：创建溢价数据合同失败；衍生品压力特征不可构造；开盘折价只有 8 个成熟历史事件且平均为负。
- 聚合 OI V2 只有 38 个完整 Pressure 值，无法形成 prior-252 分位，唯一运行次数消耗并关闭该公开日频代理族。
- 盘后离岸 A50 传导与一级市场数据成熟度账只做前向，当前分别 0/60 与 0/120。

入口报告：

- `reports/research/510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1.md`
- `reports/research/510300_EPISODIC_ALPHA_LIBRARY_V1_POST_RUN_ADJUDICATION.md`
- `reports/research/510300_DERIVATIVE_PRESSURE_RELEASE_V2_AGGREGATE_OI.md`
- `reports/research/510300_EPISODIC_ALPHA_LIBRARY_BATCH_2_RESOURCE_ADJUDICATION.md`

## 4. PIT 官方财务事实管线

- 8 月 31 日至 9 月 2 日连续迭代官方 PDF、direct PDF、文本、OCR、census、merge 与 residual 分类。
- 当前完整依赖事件 7,252/8,223，覆盖率 88.192%，原冻结覆盖门仍未通过。
- 在明确保持原始阻断的前提下，另做不完整覆盖条件样本 Phase 1；60D/120D 冻结机制均失败，第二阶段未授权。
- 条件样本的可评分独立信息非常稀疏，且价格面板呈非矩形选择，应重点审查外推边界。

入口报告：

- `reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_21_DOCUMENT_BY_DOCUMENT.md`
- `reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_RESIDUAL_TERMINAL_CLASSIFICATION_V1.md`
- `reports/research/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_INCOMPLETE_COVERAGE_ANALYSIS_V1.md`

## 5. 优先前向运行链

- V1.8 仍是冻结入口，但项目 `data` 目录迁移为 E 盘 Junction 后，manifest 中的相对数据路径解析到 C 盘项目根之外，触发路径边界失败。
- 这属于程序/配置治理问题，不是新数据采集结果，也不是质量日。
- 当前路径证据见 `07_PRIORITY_FORWARD_PATH_CONTAINMENT_SNAPSHOT.md`；旧运行回执和状态见 `reports/audit/`。
"""


def known_weaknesses_text(report_count: int, git_info: dict[str, Any]) -> str:
    return f"""# 已知疑点：请 GPT Pro 主动攻击，而不是默认接受

以下不是最终判决，而是包的制作者认为必须被外部审阅者证伪或确认的薄弱点。

1. **三日分支密度过高。** 本包索引到 {report_count} 份三日研究报告；大量分支在同一天完成协议、实现、测试和裁决。需要判断这是高效复用，还是没有给机制推理、数据核验和独立复制留下足够时间。
2. **项目级多重试验没有被单个分支的“冻结/不救援”完全解决。** ADX、HMM、多尺度趋势、Atlas、Oracle、UP20、结构状态和情景 Alpha 可能共享同一历史和相邻灵感，应有统一 family 预算。
3. **版本与脚本碎片化。** PIT 官方事实链在两天内跨过多个版本号并产生大量 census/merge/OCR 脚本；需要唯一可重放入口、状态迁移图和最终数据合同，而不是靠文件名推断最新真相。
4. **有效独立样本远小于行数。** 重叠 60D/120D origin、同一发行人多期事件、同一年/行业聚类、只有 23—55 个最终可评价事件，都可能让表面行数失真。
5. **Oracle/Atlas/会计恒等式的决策价值可能被高估。** 它们能解释能力上界或算术闭合，但可能没有缩小现实可预测模型空间。
6. **若门槛从一开始几乎不可满足，失败不一定提供高信息。** 例如 38 个完整 Pressure 值却要求 prior-252 分位、UP20 近乎 100% 精确率、PCA 新成员需完整 500 日历史。需要区分“机制被否证”和“协议没有可行输入域”。
7. **PIT 与代理权重仍是结构性限制。** 市值代理、二级聚合历史值、不可证明版本的历史权重和缺失精确发布时间，可能使复杂模型没有资格谈微小增量。
8. **根状态与分支路由词义可能冲突。** 根状态是 `ABSTAIN/POSITION_UNSET`，Batch-2 写 `CURRENT_HOLDING_ROUTE=CASH_CNY`。若没有统一命名，外部读者可能误解为真实持仓命令。
9. **路径可移植性已被数据迁移打破。** V1.8 的 containment 假设与 Junction 数据根不兼容；manifest 相对路径在语法上位于项目内、解析后却在 E 盘。
10. **版本控制不足。** 本包源文件中已跟踪 {git_info.get('selected_tracked_count', 0)} 个、未跟踪 {git_info.get('selected_untracked_count', 0)} 个；三日内提交数 {len(git_info.get('commits_in_period', []))}。这会让 manifest 哈希、并行修改、失败保全和复现变得脆弱。
11. **测试数量可能掩盖经济验证不足。** 单元测试能证明状态机和输出格式按设计运行，不能证明研究问题、代理、效应门槛或经济机制正确。
12. **权威状态可能滞后于最新工作。** `RESEARCH_STATUS.md` 截至 2026-08-31，而 9 月 2 日又产生 PIT 条件样本结果；需要判断最新结果是否已进入单一权威注册表。

请对每一点回答：成立、部分成立或不成立；给出直接路径和修复优先级。
"""


def reading_route_text() -> str:
    return """# 证据阅读路线

## 第一层：项目状态与用户问题

- `RESEARCH_STATUS.md`
- `CONTEXT.md`
- `01_GPT_PRO_REVIEW_PROMPT.md`
- `03_THREE_DAY_WORK_MAP.md`

## 第二层：先读裁决，再查机器结果

使用 `09_RESEARCH_REPORT_INDEX.csv` 定位报告。每个重要结论至少同时查：

1. `docs/` 或 `config/*.yaml`：问题、数据、时钟、候选、成本和停止门；
2. `config/*manifest.json` 与 `reports/frozen/`：版本与冻结边界；
3. `research/`、`scripts/`：实际实现；
4. `reports/research/*.json` 与 `data/research/`：机器结果；
5. `tests/`：测试证明了什么、没有证明什么。

## 第三层：优先检查四组交叉矛盾

- 根 `ABSTAIN/POSITION_UNSET` 与分支 `CASH_CNY` 路由；
- 原 PIT 覆盖阻断与不完整覆盖条件样本结果；
- Oracle/测量 PASS 与现实预测/组合 NOT_ALLOWED；
- V1.8 相对路径 manifest 与 Junction 解析后的外部数据根。

## 第四层：数据文件

包内包含可携带的三日 `data/research/510300_*` 派生表、结构引擎数据和少量核心日频/分钟输入。原始 CNInfo PDF 全库、OCR 页图和大体量中间检查点未打包。若复核必须依赖被排除材料，请只列最小精确缺口。
"""


def coverage_text(builder: PackageBuilder) -> str:
    included_bytes = sum(record.path.stat().st_size for record in builder.sources.values())
    excluded_bytes = sum(
        int(row["bytes"]) for row in builder.exclusions if isinstance(row.get("bytes"), int)
    )
    return f"""# 覆盖范围与排除项

## 已纳入

- 三日内与 510300/沪深300 直接相关的配置、协议、报告、代码、测试、状态和回执；
- 三日内 `data/research/510300_*` 的可携带派生数据；
- 结构性 ERP 引擎的精选 curated/processed 数据；
- 510300、000300、H00300 的少量核心日频/分钟输入；
- V1.8 优先前向 manifest 及其 41 个冻结依赖，用于复核路径和哈希边界；
- 根项目语义与权威状态。

当前源文件总计 {len(builder.sources):,} 个，{included_bytes:,} 字节。

## 未纳入

- `.env`、API 凭据、虚拟环境、缓存、`__pycache__`；
- `tmp/` 下的 OCR 页图、探针 PDF、临时 receipt 和调试产物；
- CNInfo 原始 PDF/文本全库、大型 staging/checkpoint；
- 旧 GPT Pro 审阅 ZIP、旧交付包和重复归档；
- 单文件超过 64 MiB 的材料；
- 与本次 510300 三日工作无直接关系的其他资产研究。

构包过程中显式登记的缺失/超限排除项 {len(builder.exclusions):,} 个，已知字节合计 {excluded_bytes:,}。完整清单见 `11_EXCLUDED_ITEM_INDEX.csv`。该清单是范围说明，不是全磁盘安全或隐私扫描。
"""


def containment_snapshot_text(rows: list[dict[str, Any]]) -> str:
    failures = [row for row in rows if row.get("contained_after_resolution") is False]
    hash_mismatches = [row for row in rows if row.get("exists") and not row.get("sha256_match")]
    missing = [row for row in rows if not row.get("exists")]
    failure_lines = "\n".join(
        f"- `{row['path']}` → `{row['resolved_path']}`" for row in failures
    ) or "- 无"
    mismatch_lines = "\n".join(
        f"- `{row['path']}`" for row in hash_mismatches[:30]
    ) or "- 无"
    missing_lines = "\n".join(f"- `{row['path']}`" for row in missing[:30]) or "- 无"
    return f"""# V1.8 优先前向路径边界当前快照

- 快照时间：{SNAPSHOT_AT.isoformat()}
- manifest：`config/priority_forward_research_operations_v1_8_manifest.json`
- 冻结依赖数：{len(rows)}
- 解析后越出当前项目根：{len(failures)}
- 当前 SHA-256 与 manifest 不一致：{len(hash_mismatches)}
- 当前缺失：{len(missing)}

## 解析后越界路径

{failure_lines}

当前项目的 `data` 是指向 E 盘研究数据根的 Junction。manifest 中 `data/governance/FREE_SOURCE_REGISTRY_V1_1.csv` 在逻辑路径上位于项目内，但 `resolve()` 后位于 E 盘，因此与“所有冻结文件必须解析在 C 盘项目根内”的旧 containment 假设冲突。该快照只做读取与解析，没有启动采集、没有取得当日 claim、没有写质量日。

## 哈希不一致

{mismatch_lines}

## 缺失依赖

{missing_lines}

完整逐文件数据见 `10_PRIORITY_V1_8_FROZEN_DEPENDENCY_INDEX.csv`。请审阅者判断：应建立显式、版本化、内容寻址的外部数据根合同，还是把冻结治理文件放回 checkout 内；不得静默绕过 containment 或改写旧 manifest。
"""


def git_snapshot_text(info: dict[str, Any]) -> str:
    commits = info.get("commits_in_period", [])
    commit_text = "\n".join(f"- `{line}`" for line in commits) if commits else "- 三日范围内未找到提交。"
    examples = info.get("selected_untracked_examples", [])
    example_text = "\n".join(f"- `{path}`" for path in examples[:40]) or "- 无"
    return f"""# Git 与可重放性快照

- 分支：`{info.get('branch')}`
- HEAD：`{info.get('head')}`
- 包内源文件：{info.get('selected_file_count', 0)}
- Git 已跟踪：{info.get('selected_tracked_count', 0)}
- Git 未跟踪：{info.get('selected_untracked_count', 0)}

## 三日提交

{commit_text}

## 未跟踪示例

{example_text}

这不是用“工作树脏”替代研究证据，而是让审阅者评估：当前版本边界是否足以支持 diff、回滚、重放、并行合并和失败证据永久保存。
"""


def build_generated_files(
    builder: PackageBuilder,
    priority_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    git_info: dict[str, Any],
) -> dict[str, bytes]:
    file_rows = included_file_index(builder)
    tracked_output = run_git(["ls-files"])
    tracked = set(tracked_output.splitlines()) if tracked_output else set()
    for row in file_rows:
        row["git_tracking"] = "TRACKED" if row["path"] in tracked else "UNTRACKED"

    exclusion_rows = sorted(
        builder.exclusions,
        key=lambda row: (str(row.get("reason")), str(row.get("path"))),
    )
    generated: dict[str, bytes] = {
        "00_README_FIRST.md": render_readme(builder, report_rows, git_info).encode("utf-8"),
        "01_GPT_PRO_REVIEW_PROMPT.md": review_prompt_text().encode("utf-8"),
        "02_USER_QUESTION_AND_REVIEW_BOUNDARY.md": user_question_text().encode("utf-8"),
        "03_THREE_DAY_WORK_MAP.md": work_map_text().encode("utf-8"),
        "04_KNOWN_WEAKNESSES_TO_ATTACK.md": known_weaknesses_text(
            len(report_rows), git_info
        ).encode("utf-8"),
        "05_READING_ROUTE.md": reading_route_text().encode("utf-8"),
        "06_COVERAGE_AND_EXCLUSIONS.md": coverage_text(builder).encode("utf-8"),
        "07_PRIORITY_FORWARD_PATH_CONTAINMENT_SNAPSHOT.md": containment_snapshot_text(
            priority_rows
        ).encode("utf-8"),
        "08_GIT_AND_REPRODUCIBILITY_SNAPSHOT.md": git_snapshot_text(git_info).encode(
            "utf-8"
        ),
        "09_RESEARCH_REPORT_INDEX.csv": csv_bytes(
            report_rows,
            [
                "category",
                "path",
                "modified_at",
                "bytes",
                "title",
                "first_status_or_judgment_line",
            ],
        ),
        "10_PRIORITY_V1_8_FROZEN_DEPENDENCY_INDEX.csv": csv_bytes(
            priority_rows,
            [
                "path",
                "exists",
                "expected_bytes",
                "actual_bytes",
                "expected_sha256",
                "actual_sha256",
                "sha256_match",
                "resolved_path",
                "contained_after_resolution",
                "error",
            ],
        ),
        "11_EXCLUDED_ITEM_INDEX.csv": csv_bytes(
            exclusion_rows,
            ["path", "reason", "bytes"],
        ),
        "12_INCLUDED_FILE_INDEX.csv": csv_bytes(
            file_rows,
            ["path", "bytes", "modified_at", "selection_reason", "git_tracking"],
        ),
    }

    metadata = {
        "package_id": PACKAGE_BASENAME,
        "snapshot_timezone": "Asia/Shanghai",
        "period_start": START_AT.isoformat(),
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "scope": "510300_AND_CSI300_DIRECTLY_RELATED_THREE_DAY_WORK",
        "source_file_count": len(builder.sources),
        "source_bytes": sum(record.path.stat().st_size for record in builder.sources.values()),
        "generated_file_count": len(generated) + 1,
        "research_report_count": len(report_rows),
        "validation_tier": "FAST_RESEARCH_REVIEW_NO_SECURITY_AUDIT",
        "performed_checks": [
            "ZIP_CRC",
            "DUPLICATE_MEMBER_CHECK",
            "WHOLE_ZIP_SHA256",
            "PRIORITY_V1_8_FROZEN_41_FILE_HASH_AND_PATH_SNAPSHOT",
        ],
        "not_performed": [
            "SECURITY_AUDIT",
            "PRIVACY_SCAN",
            "SECRET_SCAN",
            "MALWARE_SCAN",
            "REDACTION",
            "FULL_REPOSITORY_REGRESSION",
            "FRESH_EXTRACTION_REPLAY",
        ],
        "git": git_info,
    }
    generated["13_SNAPSHOT_METADATA.json"] = json.dumps(
        metadata, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    return generated


def write_zip(builder: PackageBuilder, generated: dict[str, bytes]) -> tuple[int, int]:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    total_source_bytes = sum(record.path.stat().st_size for record in builder.sources.values())
    if total_source_bytes > MAX_TOTAL_SOURCE_BYTES:
        raise RuntimeError(
            f"源文件总量超过 350 MiB 上限：{total_source_bytes:,} bytes"
        )

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
                    arcname=f"{INTERNAL_ROOT}/{relative}",
                )
            for relative, content in sorted(generated.items()):
                archive.writestr(f"{INTERNAL_ROOT}/{relative}", content)

        temporary_path.replace(ZIP_PATH)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = archive.namelist()
        duplicates = len(members) - len(set(members))
        bad_member = archive.testzip()
    if duplicates != 0:
        raise RuntimeError(f"ZIP 存在重复成员：{duplicates}")
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC 失败成员：{bad_member}")
    return len(members), duplicates


def write_sidecars(
    builder: PackageBuilder,
    generated: dict[str, bytes],
    member_count: int,
    duplicate_count: int,
    priority_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
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
        "period_start": START_AT.isoformat(),
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "source_file_count": len(builder.sources),
        "source_bytes": sum(record.path.stat().st_size for record in builder.sources.values()),
        "generated_file_count": len(generated),
        "zip_member_count": member_count,
        "duplicate_member_count": duplicate_count,
        "research_report_count": len(report_rows),
        "priority_manifest_dependency_count": len(priority_rows),
        "priority_path_containment_failure_count": sum(
            row.get("contained_after_resolution") is False for row in priority_rows
        ),
        "priority_manifest_hash_mismatch_count": sum(
            bool(row.get("exists")) and not bool(row.get("sha256_match"))
            for row in priority_rows
        ),
        "status": "PASS_FAST_RESEARCH_REVIEW_PACKAGE_CRC_DUPLICATE_AND_WHOLE_ZIP_SHA256",
        "security_audit_performed": False,
        "privacy_scan_performed": False,
        "secret_scan_performed": False,
        "malware_scan_performed": False,
        "full_regression_performed": False,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    upload_message = f"""请使用 GPT Pro 审阅附件：{ZIP_PATH.name}

先读压缩包根目录的 00_README_FIRST.md 和 01_GPT_PRO_REVIEW_PROMPT.md。

这次不要做安全审计，也不要给买卖建议。请严厉检查我们 2026-08-31 至 2026-09-02 的 510300 工作：哪里不够深入、哪里铺得太宽、哪里统计或点时证据不足、哪里版本和状态治理不好、哪些工作应该停止。不要泛泛鼓励；每条判断引用包内路径并给出可执行整改或停止动作。

压缩包 SHA-256：{package_sha256}
校验状态：ZIP CRC 通过、重复成员 0、整包 SHA-256 已写入同名 .sha256；未做安全/隐私/秘密/恶意文件审计。
"""
    UPLOAD_MESSAGE_PATH.write_text(upload_message, encoding="utf-8")
    return receipt


def main() -> None:
    builder = PackageBuilder()
    collect_recent_relevant_sources(builder)
    collect_recent_research_data(builder)
    collect_context_and_dependencies(builder)
    priority_rows = collect_priority_manifest_dependencies(builder)
    report_rows = extract_report_index(builder)
    git_info = git_snapshot(builder)
    generated = build_generated_files(builder, priority_rows, report_rows, git_info)
    member_count, duplicate_count = write_zip(builder, generated)
    receipt = write_sidecars(
        builder,
        generated,
        member_count,
        duplicate_count,
        priority_rows,
        report_rows,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
