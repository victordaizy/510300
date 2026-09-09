"""构建沪深A股极端缩波 V2/V2.1 的 ChatGPT Pro 独立审阅包。

本脚本只打包已经存在的冻结协议、实现、结果账本与回执，不重新运行研究，
也不改变任何研究阈值。构包前会复核 V2/V2.1 回执声明的全部直接输入和
全部输出产物；构包后会从 ZIP 字节流以及一次全新解压目录分别复核清单。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
PACKAGE_BASENAME = (
    "A_SHARE_HS_EXTREME_COMPRESSION_V2_V2_1_CHATGPT_PRO_REVIEW_20260823"
)
INTERNAL_ROOT = "A_SHARE_HS_EXTREME_COMPRESSION_V2_V2_1_REVIEW"
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
MAX_AUTO_INCLUDE_DIRECT_INPUT_BYTES = 12_500_000

V2_RECEIPT_PATH = (
    ROOT
    / "data/staging/a_share_hs_extreme_compression_resolution_alpha_discovery_v2"
    / "run_receipt.json"
)
V21_RECEIPT_PATH = (
    ROOT
    / "data/staging/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1"
    / "run_receipt.json"
)
PRIOR_CHAT_ROOT = (
    ROOT / "review_packages/A_SHARE_HS_CHAT_FULL_REVIEW_2026-08-22"
)
PRIOR_CHAT_ZIP = (
    ROOT / "review_packages/A_SHARE_HS_CHAT_FULL_REVIEW_2026-08-22.zip"
)
PARENT_REVIEW_ROOT = (
    ROOT
    / "deliverables"
    / "A_SHARE_HS_EXTREME_COMPRESSION_V1_2_1_CHATGPT_PRO_REVIEW_20260823"
)
PARENT_REVIEW_ZIP = (
    ROOT
    / "deliverables"
    / "A_SHARE_HS_EXTREME_COMPRESSION_V1_2_1_CHATGPT_PRO_REVIEW_20260823.zip"
)

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


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


def atomic_text(path: Path, value: str) -> None:
    """原子写入 UTF-8 文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    """原子写入格式化 JSON。"""

    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def safe_relative_path(value: str) -> str:
    """拒绝绝对路径和路径穿越。"""

    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"非法包内路径：{value}")
    return relative.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要 JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_tree_files(root: Path) -> Iterable[Path]:
    """枚举目录中的正式文件，排除缓存和编译产物。"""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in {"__pycache__", ".pytest_cache", ".venv"} for part in relative_parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        yield path


class PackageBuilder:
    """维护唯一源到包内路径映射，并记录脱敏差异。"""

    def __init__(self) -> None:
        self.mappings: list[tuple[Path, str]] = []
        self.destinations: set[str] = set()
        self.source_to_member: dict[Path, str] = {}
        self.redactions: list[dict[str, Any]] = []

    def add(self, source: Path, destination: str) -> None:
        source = source.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(f"缺少必要源文件：{source}")
        destination = safe_relative_path(destination)
        if destination in self.destinations:
            raise ValueError(f"包内目标重复：{destination}")
        self.mappings.append((source, destination))
        self.destinations.add(destination)
        self.source_to_member.setdefault(source, destination)

    def add_tree(
        self,
        source_root: Path,
        destination_root: str,
        *,
        rename: dict[str, str] | None = None,
        skip: set[str] | None = None,
    ) -> None:
        if not source_root.is_dir():
            raise FileNotFoundError(f"缺少必要源目录：{source_root}")
        rename = rename or {}
        skip = skip or set()
        for source in iter_tree_files(source_root):
            relative = source.relative_to(source_root).as_posix()
            if relative in skip:
                continue
            target_relative = rename.get(relative, relative)
            self.add(source, f"{destination_root}/{target_relative}")

    @staticmethod
    def _redact_text(value: str) -> tuple[str, list[str]]:
        """只替换本机用户名与绝对用户目录，不改变研究字段。"""

        replacements = [
            (
                "PROJECT_ROOT_JSON_ESCAPED",
                str(ROOT).replace("\\", "\\\\"),
                "<PROJECT_ROOT>",
            ),
            ("PROJECT_ROOT_LITERAL", str(ROOT), "<PROJECT_ROOT>"),
            (
                "USER_HOME_JSON_ESCAPED",
                str(Path.home()).replace("\\", "\\\\"),
                "<USER_HOME>",
            ),
            ("USER_HOME_LITERAL", str(Path.home()), "<USER_HOME>"),
            ("USERNAME_LITERAL", "戴周阳", "<REDACTED_USER>"),
        ]
        applied: list[str] = []
        output = value
        for rule_id, old, new in replacements:
            if old and old in output:
                output = output.replace(old, new)
                applied.append(rule_id)
        return output, applied

    def copy_all(self, package_root: Path) -> None:
        for source, destination in sorted(self.mappings, key=lambda item: item[1]):
            target = package_root / Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256_file(source)
            if source.suffix.lower() in TEXT_EXTENSIONS:
                raw = source.read_bytes()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    shutil.copy2(source, target)
                else:
                    redacted, applied = self._redact_text(text)
                    if applied:
                        target.write_bytes(redacted.encode("utf-8"))
                        self.redactions.append(
                            {
                                "source_relative_path": source.relative_to(ROOT).as_posix(),
                                "package_member": destination,
                                "source_sha256": source_hash,
                                "packaged_sha256": sha256_file(target),
                                "replacements": applied,
                            }
                        )
                    else:
                        shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)


def register_current_artifacts(builder: PackageBuilder) -> None:
    """登记 V2/V2.1 协议、实现、报告与核心结果账本。"""

    builder.add(
        ROOT / "config/a_share_hs_extreme_compression_resolution_alpha_discovery_v2.yaml",
        "config/a_share_hs_extreme_compression_resolution_alpha_discovery_v2.yaml",
    )
    builder.add_tree(
        ROOT / "config/a_share_hs_extreme_compression_resolution_alpha_discovery_v2",
        "config/a_share_hs_extreme_compression_resolution_alpha_discovery_v2",
    )
    builder.add(
        ROOT / "config/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1.yaml",
        "config/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1.yaml",
    )

    for relative in [
        "src/a_share_hs_extreme_compression_resolution_alpha_v2.py",
        "src/a_share_hs_extreme_compression_conditional_alpha_v2_1.py",
        "scripts/run_a_share_hs_extreme_compression_resolution_alpha_discovery_v2.py",
        "scripts/run_a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1.py",
        "tests/test_a_share_hs_extreme_compression_resolution_alpha_v2.py",
        "tests/test_a_share_hs_extreme_compression_conditional_alpha_v2_1.py",
        "scripts/build_a_share_hs_extreme_compression_v2_v2_1_chatgpt_pro_review_package.py",
    ]:
        builder.add(ROOT / relative, relative)

    for relative in [
        "reports/research/A_SHARE_HS_EXTREME_COMPRESSION_RESOLUTION_ALPHA_DISCOVERY_V2.md",
        "reports/research/a_share_hs_extreme_compression_resolution_alpha_discovery_v2.json",
        "reports/research/A_SHARE_HS_EXTREME_COMPRESSION_CONDITIONAL_ALPHA_DISCOVERY_V2_1.md",
        "reports/research/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1.json",
        "reports/research/A_SHARE_HS_EXTREME_COMPRESSION_V1_2_1_EXTERNAL_REVIEW_REMEDIATION_SUMMARY.md",
    ]:
        builder.add(ROOT / relative, relative)

    v2_root = (
        ROOT
        / "data/staging/a_share_hs_extreme_compression_resolution_alpha_discovery_v2"
    )
    for name in [
        "event_market_volatility_origin_ledger.parquet",
        "first_vol_expansion_transition_ledger.parquet",
        "market_direction_state_summary.csv",
        "market_state_cumulative_summary.csv",
        "market_state_quintile_summary.csv",
        "market_volatility_state_daily.parquet",
        "policy_market_state_summary.csv",
        "policy_origin_ledger.parquet",
        "policy_outcomes.parquet",
        "primary_hypothesis_results.csv",
        "run_receipt.json",
    ]:
        builder.add(
            v2_root / name,
            (
                "data/staging/"
                "a_share_hs_extreme_compression_resolution_alpha_discovery_v2/"
                f"{name}"
            ),
        )

    v21_root = (
        ROOT
        / "data/staging/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1"
    )
    for name in [
        "annual_stability_summary.csv",
        "conditional_feature_ledger.parquet",
        "expansion_state_outcome_summary.csv",
        "m6_policy_oos_summary.csv",
        "nested_model_fold_metrics.csv",
        "nested_model_oos_predictions.parquet",
        "nested_model_summary.csv",
        "origin_idiosyncratic_feature_ledger.parquet",
        "run_receipt.json",
    ]:
        builder.add(
            v21_root / name,
            (
                "data/staging/"
                "a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1/"
                f"{name}"
            ),
        )


def register_parent_evidence(builder: PackageBuilder) -> None:
    """登记父研究的关键直接输入和可复核抽样。"""

    for relative in [
        "data/raw/all_etf_momentum_v1/H00300_total_return.parquet",
        "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/share_count_history.parquet",
        "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/trading_calendar_observed_open_days.parquet",
        "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1/event_inference_features_v1_1.parquet",
        "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1/factor_input_buckets_v1/bucket_manifest.json",
        "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1_2_1_reproducibility_remediation/breakout_events_v1_2.parquet",
        "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1_2_1_reproducibility_remediation/event_ledger_v1_2.parquet",
        "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1_2_1_reproducibility_remediation/horizon_outcomes_v1_2.parquet",
    ]:
        builder.add(ROOT / relative, relative)

    for directory in [
        "00_REVIEW_CONTEXT",
        "01_PROTOCOL",
        "02_REPORTS",
        "04_SOURCE_CONTRACTS",
        "06_RECEIPTS",
        "07_ENVIRONMENT",
    ]:
        builder.add_tree(
            PARENT_REVIEW_ROOT / directory,
            f"parent_v1_2_1/{directory}",
        )

    for relative in [
        "00_README_FIRST.md",
        "03_EXCLUSIONS_AND_BOUNDARIES.md",
        "04_KEY_RESULTS_AND_STATUS.md",
        "05_INPUT_LINEAGE.json",
    ]:
        builder.add(
            PARENT_REVIEW_ROOT / relative,
            f"parent_v1_2_1/{relative}",
        )

    builder.add(
        PARENT_REVIEW_ROOT / "02_ARTIFACT_MANIFEST.json",
        "parent_v1_2_1/02_ORIGINAL_FULL_PACKAGE_MANIFEST.json",
    )

    for relative in [
        "03_IMPLEMENTATION/01_run_v1_event_study.py",
        "03_IMPLEMENTATION/02_complete_v1_1_inference.py",
        "03_IMPLEMENTATION/03_complete_v1_2_and_v1_2_1.py",
        "03_IMPLEMENTATION/04_run_v1_2_1_full_rebuild_and_tests.py",
        "03_IMPLEMENTATION/05_refit_18_models_from_package.py",
        "03_IMPLEMENTATION/07_v1_regression_tests.py",
        "03_IMPLEMENTATION/08_v1_2_regression_tests.py",
        "03_IMPLEMENTATION/09_v1_2_1_reproducibility_tests.py",
        "05_DATA_TABLES/REPRODUCED/adjusted_contrasts_refit_v1_2_1.csv",
        "05_DATA_TABLES/REVIEW_EXTRACTS/episode_path_all_events_fixed_landmarks.parquet",
        "05_DATA_TABLES/REVIEW_EXTRACTS/episode_path_first_500_event_ids.parquet",
        "05_DATA_TABLES/REVIEW_EXTRACTS/industry_peer_recalculation_first_100_requests.parquet",
        "05_DATA_TABLES/REVISION_AUDITS/mae_mfe_tie_time_changes_v1_2_to_v1_2_1.parquet",
        "05_DATA_TABLES/REVISION_AUDITS/revision_comparison_summary_v1_2_to_v1_2_1.json",
    ]:
        builder.add(
            PARENT_REVIEW_ROOT / relative,
            f"parent_v1_2_1/{relative}",
        )


def register_prior_chat_package(builder: PackageBuilder) -> None:
    """完整纳入此前聊天审阅包的解压内容，并保留原清单副本。"""

    original_manifest = "04_FILE_MANIFEST_SHA256.csv"
    builder.add_tree(
        PRIOR_CHAT_ROOT,
        "prior_chat_review",
        rename={
            original_manifest: "04_ORIGINAL_FILE_MANIFEST_BEFORE_REDACTION.csv"
        },
    )


def direct_input_records(
    v2_receipt: dict[str, Any],
    v21_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study_label, receipt in [("V2", v2_receipt), ("V2.1", v21_receipt)]:
        for record in receipt["input_integrity"]["records"]:
            rows.append(
                {
                    "study": study_label,
                    "path": record["path"],
                    "bytes": int(record["bytes"]),
                    "sha256": record["sha256"],
                }
            )
    return rows


def register_small_direct_inputs(
    builder: PackageBuilder,
    records: list[dict[str, Any]],
) -> None:
    """自动纳入尚未登记且较小的直接输入，避免漏件。"""

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["path"], record)
    for relative, record in sorted(unique.items()):
        source = (ROOT / relative).resolve()
        if source in builder.source_to_member:
            continue
        if int(record["bytes"]) <= MAX_AUTO_INCLUDE_DIRECT_INPUT_BYTES:
            builder.add(source, f"direct_inputs/{safe_relative_path(relative)}")


def verify_receipt_records(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    """按回执逐项复核当前本地文件的存在性、大小和哈希。"""

    failures: list[dict[str, Any]] = []
    checked = 0
    bytes_checked = 0
    seen: set[tuple[str, int, str]] = set()
    for record in rows:
        key = (record["path"], int(record["bytes"]), record["sha256"])
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        path = (ROOT / record["path"]).resolve()
        if not path.is_file():
            failures.append({"path": record["path"], "failure": "MISSING"})
            continue
        actual_size = path.stat().st_size
        bytes_checked += actual_size
        if actual_size != int(record["bytes"]):
            failures.append(
                {
                    "path": record["path"],
                    "failure": "SIZE_MISMATCH",
                    "expected": int(record["bytes"]),
                    "actual": actual_size,
                }
            )
            continue
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        actual_hash = hash_cache[path]
        if actual_hash != record["sha256"]:
            failures.append(
                {
                    "path": record["path"],
                    "failure": "SHA256_MISMATCH",
                    "expected": record["sha256"],
                    "actual": actual_hash,
                }
            )
    result = {
        "label": label,
        "status": "PASS" if not failures else "FAIL",
        "unique_records_checked": checked,
        "bytes_checked": bytes_checked,
        "failures": failures,
    }
    if failures:
        raise AssertionError(f"{label}回执复核失败：{failures[:5]}")
    return result


def artifact_records(
    v2_receipt: dict[str, Any],
    v21_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt in [v2_receipt, v21_receipt]:
        for record in receipt["artifacts"].values():
            rows.append(
                {
                    "path": record["path"],
                    "bytes": int(record["bytes"]),
                    "sha256": record["sha256"],
                }
            )
    return rows


def exclusion_reason(path: str) -> str:
    if "factor_input_buckets_v1/symbol_bucket=" in path:
        return "32个冻结因子输入分桶体积较大；最终特征、结果账本和逐项源哈希已纳入"
    if path.endswith("unified_daily_market.parquet"):
        return "全市场统一逐日面板体积过大；源大小、哈希和完整重建命令已保留"
    if path.endswith("episode_daily_path.parquet"):
        return "全事件逐日路径体积过大；固定landmark和500事件抽样已纳入"
    if path.endswith("point_in_time_industry_daily_prices_v1_2.parquet"):
        return "点时行业日价格中间面板体积较大；100个固定请求复核抽样已纳入"
    if path.endswith("panel_b_breakout_outcomes_v1_2.parquet"):
        return "父研究Panel B大表非本轮核心条件模型输入；父报告、哈希与关键账本已纳入"
    return "未纳入的上游大文件；路径、字节数、SHA-256和用途由回执保留"


def create_source_inventory(
    package_root: Path,
    builder: PackageBuilder,
    records: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        source = (ROOT / record["path"]).resolve()
        member = builder.source_to_member.get(source, "")
        rows.append(
            {
                "study": record["study"],
                "source_path": record["path"],
                "source_bytes": int(record["bytes"]),
                "source_sha256": record["sha256"],
                "source_exists_at_build": source.is_file(),
                "source_hash_reverified_at_build": (
                    source.is_file() and hash_cache.get(source) == record["sha256"]
                ),
                "package_status": "INCLUDED" if member else "EXCLUDED_WITH_VERIFIED_HASH",
                "package_member": member,
                "exclusion_reason": "" if member else exclusion_reason(record["path"]),
            }
        )
    target = package_root / "06_SOURCE_INVENTORY.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def create_prior_chat_manifest(package_root: Path) -> dict[str, Any]:
    """为脱敏后的此前聊天子包重建局部清单。"""

    root = package_root / "prior_chat_review"
    manifest_path = root / "04_FILE_MANIFEST_SHA256.csv"
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(records)
    return {
        "status": "SANITIZED_SUBPACKAGE_MANIFEST_REBUILT",
        "file_count_excluding_manifest": len(records),
        "manifest_path": "prior_chat_review/04_FILE_MANIFEST_SHA256.csv",
    }


def create_parquet_coverage(package_root: Path) -> dict[str, Any]:
    """只读 Parquet 元数据，生成行数和字段类型索引。"""

    records: list[dict[str, Any]] = []
    for top_level in ["data", "parent_v1_2_1"]:
        root = package_root / top_level
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            relative = path.relative_to(package_root).as_posix()
            try:
                parquet = pq.ParquetFile(path)
                schema = parquet.schema_arrow
                records.append(
                    {
                        "package_member": relative,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "row_count": int(parquet.metadata.num_rows),
                        "row_groups": int(parquet.metadata.num_row_groups),
                        "column_count": len(schema),
                        "columns": [
                            {"name": field.name, "type": str(field.type)}
                            for field in schema
                        ],
                        "metadata_status": "PASS",
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "package_member": relative,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "metadata_status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    result = {
        "scope": "INCLUDED_PARQUET_UNDER_DATA_AND_PARENT_V1_2_1",
        "file_count": len(records),
        "all_metadata_readable": all(
            item["metadata_status"] == "PASS" for item in records
        ),
        "files": records,
    }
    atomic_json(package_root / "07_PARQUET_SCHEMA_AND_ROW_COUNTS.json", result)
    if not result["all_metadata_readable"]:
        raise AssertionError("至少一个纳入包的 Parquet 元数据不可读")
    return result


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def create_documents(
    package_root: Path,
    v2: dict[str, Any],
    v21: dict[str, Any],
    inventory: list[dict[str, Any]],
    prior_chat_manifest: dict[str, Any],
    parquet_coverage: dict[str, Any],
    redactions: list[dict[str, Any]],
) -> None:
    primary_pass = sum(
        bool(item["historical_primary_gate_pass"])
        for item in v2["primary_hypotheses"]
    )
    mechanism = v21["conclusion"]["mechanism_candidate_gate_pass_count"]
    excluded = [row for row in inventory if row["package_status"].startswith("EXCLUDED")]
    excluded_unique: dict[str, dict[str, Any]] = {}
    for row in excluded:
        excluded_unique.setdefault(row["source_path"], row)
    excluded_bytes = sum(int(row["source_bytes"]) for row in excluded_unique.values())
    included_unique = {
        row["source_path"]
        for row in inventory
        if row["package_status"] == "INCLUDED"
    }

    readme = f"""# 先读我：沪深A股极端缩波 V2/V2.1 独立审阅包

本 ZIP 是给 ChatGPT Pro 的单一、可上传、已脱敏审阅包。正式 V2/V2.1
数据截止日均为 2026-08-14；其他历史子研究的证据日期以各自报告为准。

## 结论先行

- 父 V2 预冻结主要检验通过：{primary_pass}/6。
- V2.1 固定机制候选门槛通过：{mechanism}/6。
- 最终判断：{v21['conclusion']['status']}。
- V2.1 共生成 {v21['walk_forward']['prediction_rows']:,} 行样本外预测、
  {v21['walk_forward']['fold_metric_rows']} 个折叠指标行；M0-M6 使用同一共同案例。
- 这些结果不构成已验证 alpha、可执行回测、Shadow、实盘或下单授权。

## 阅读顺序

1. 01_CHATGPT_PRO_REVIEW_PROMPT.md
2. 02_CURRENT_HANDOFF.md
3. reports/research/A_SHARE_HS_EXTREME_COMPRESSION_CONDITIONAL_ALPHA_DISCOVERY_V2_1.md
4. reports/research/A_SHARE_HS_EXTREME_COMPRESSION_RESOLUTION_ALPHA_DISCOVERY_V2.md
5. 03_EXCLUSIONS_AND_BOUNDARIES.md
6. 06_SOURCE_INVENTORY.csv
7. 07_PARQUET_SCHEMA_AND_ROW_COUNTS.json
8. 两个 run_receipt.json、冻结配置、源码与测试
9. parent_v1_2_1/ 和 prior_chat_review/，仅在追溯父证据或此前策略问题时阅读

## 包内覆盖

- 此前低风险、缩波、D0/D5/D10、20万元满仓与持股期收益讨论的完整审阅包，
  已作为脱敏后的 prior_chat_review/ 纳入。
- V1.2.1 父研究的协议、报告、实现、回执、关键事件/收益账本和固定复核抽样。
- V2/V2.1 全部协议、代码、测试、报告、回执及核心结果账本。
- {len(included_unique)} 个不同直接输入已收入；{len(excluded_unique)} 个大文件
  未收入，合计 {excluded_bytes:,} 字节，但构包时均重新核对源大小和 SHA-256。
- 纳入的 {parquet_coverage['file_count']} 个当前/父研究 Parquet 均可读取元数据。

## 完整性

05_ARTIFACT_MANIFEST_SHA256.json 覆盖清单自身之外的全部包内文件。
解压后可运行：

    python verify_package.py --package-root .

构包过程还会从 ZIP 字节流逐文件复算，并在全新临时目录解压后再复算一次。
"""

    prompt = f"""# 给 ChatGPT Pro 的独立审阅提示词

请把这个 ZIP 当作一个历史研究证据包，不要先假设策略有效，也不要为了得到
正收益而调参。请先读 00_README_FIRST.md、02_CURRENT_HANDOFF.md、
03_EXCLUSIONS_AND_BOUNDARIES.md，再审阅 V2/V2.1 报告、回执、协议、代码和账本。

## 第一部分：机械复核

1. 运行 python verify_package.py --package-root .，报告文件数、缺失、额外、
   大小或 SHA-256 不一致。
2. 核对两个 run_receipt.json 中的研究 ID、数据截止日、父事件数、直接输入
   总指纹、输入运行后匹配状态和治理字段。
3. 用 07_PARQUET_SCHEMA_AND_ROW_COUNTS.json 与实际 Parquet 元数据交叉核对。
4. 复核 V2 的父结果逐行对账是否确实在 1e-12 内通过，以及 V2.1 是否存在
   未来回填、共同案例不一致、训练/测试重叠或 purge 不足。

## 第二部分：科学结论

请独立判断以下结论是否被证据支持：

- V2 六项预冻结主要检验 0/6 通过，不能声称 P1/P2/P3 有历史确认性超额。
- V2.1 机制候选门槛 0/6 通过，结论应为
  {v21['conclusion']['status']}。
- 负样本外 R²、相对 M5 的负增量、左尾与年度稳定性是否足以支持停止继续
  从同一历史中挑选赢家，而不是继续救援参数。
- 条件触发收益与“所有 episode、未触发按现金 0”的政策价值是否被清楚区分。

## 第三部分：重点概念审查

1. 波动收缩：RV20/RV120PRE、历史分位和 episode 定义是否真正描述收缩，
   还是混入停牌、稀疏成交、价格水平或数据质量伪影。
2. 第一次升波：RV5/RV20 从不高于 1 到高于 1 的交叉是否是合理机制时点；
   它与突破方向、振幅扩张和可交易确认应如何区分。
3. “不是阴跌”与“震荡”：明确审查 NOT(Close < MA120 AND MOM60_5 < 0)
   只是确认阴跌的补集，并不等于震荡。比较 SignedER、OLS 斜率/t/R²、
   回撤路径、市场/行业残差动量和特质波动收缩是否更贴近概念。
4. 动量因子：判断 MOM60_5 是否错配研究目标。若建议替代，只能提出极少数
   预先登记、经济含义不同的候选和新的验证协议；禁止在当前历史上挑最好者。
5. 点时行业同业收益：审查行业映射、排除自身、可得时间、横截面覆盖和
   特质缩波残差是否正确。

## 第四部分：输出格式

请按以下结构回答：

1. 总体决定：可接受 / 需重大修正 / 无法审阅。
2. P0、P1、P2 问题清单；每项引用具体文件、字段、行或回执键。
3. 机械完整性复核结果。
4. 数据泄漏与点时性审计。
5. 统计估计量、多重检验、依赖结构、OOS 和年度稳定性审计。
6. 波动、升波、阴跌/震荡、动量与残差特征的概念审计。
7. 哪些结论可靠，哪些不可靠。
8. 若存在致命错误，给出最小修订协议；若没有，明确建议停止还是只允许一个
   新 ID 的预登记后续研究。
9. 单列仍未解决的成本、涨跌停、T+1、停牌、退市经济终值和组合约束。

不要给当前股票推荐，不要把历史发现转换成交易指令。
"""

    handoff = f"""# 当前研究移交

## 目标

请独立复核“极端波动收缩是否能在点时行业和沪深300基准上产生可重复超额”，
并重点判断我们对波动、升波、不是阴跌、震荡和动量的定义是否错配。

## 已完成状态

- 早期三股票低风险策略、10% TSVolPct、20万元满仓、只算持股期收益、
  D0/D5/D10 重新计算及当前候选解释，保存在 prior_chat_review/。
- V1/V1.2.1 将极端缩波冻结为 episode，并完成父事件和复现性治理；
  关键证据见 parent_v1_2_1/ 和 data/staging/ 下的父账本。
- V2 在冻结的 {v2['parent_event_count']:,} 个 episode 上检验市场状态、
  第一次升波及 P1/P2/P3 的点时行业超额，主要门槛 {primary_pass}/6。
- V2.1 固定 M0-M6 purged walk-forward；共同完整案例
  {v21['conditional_feature_ledger']['common_complete_rows']:,}/
  {v21['conditional_feature_ledger']['rows']:,}
  （{percent(v21['conditional_feature_ledger']['common_complete_rate'])}），
  机制候选门槛 {mechanism}/6。
- 当前冻结结论是 {v21['conclusion']['status']}，不是可交易策略。

## 关键未决问题

- “缩波”是否需要绝对低波、相对自身收缩、市场残差收缩和行业残差收缩同时存在。
- RV5/RV20 首次上穿能否代表从平静进入可预测状态，还是仅是后验事件标签。
- “不持续阴跌”应采用何种路径定义，怎样避免把上涨、V形反弹和高噪声都误称为震荡。
- MOM60_5 是否应由 SignedER、回归斜率/显著性、回撤路径或残差动量替代。
- 当预冻结检验、固定机制门槛和 OOS 增量均失败时，是否应停止同历史迭代。

## 继续工作的硬边界

- 不修改父事件、10%阈值、U0/L0、P1/P2/P3或本轮主要检验来救援结果。
- 不从五等分、模型、期限或年度中挑历史赢家。
- 新定义只能进入新研究 ID、预登记协议和后续未见数据。
- discovery、Shadow、实盘和下单继续隔离。

## Suggested skills

- Python/Parquet 数据核验
- 统计推断与依赖结构审阅
- 时间序列泄漏和 purged walk-forward 审计
- 量化因子概念与可执行性审阅

## 关键路径

- 主结论：reports/research/A_SHARE_HS_EXTREME_COMPRESSION_CONDITIONAL_ALPHA_DISCOVERY_V2_1.md
- 父 V2：reports/research/A_SHARE_HS_EXTREME_COMPRESSION_RESOLUTION_ALPHA_DISCOVERY_V2.md
- 直接输入清单：06_SOURCE_INVENTORY.csv
- 包内哈希：05_ARTIFACT_MANIFEST_SHA256.json
- 排除边界：03_EXCLUSIONS_AND_BOUNDARIES.md
"""

    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path, row in excluded_unique.items():
        if "factor_input_buckets_v1/symbol_bucket=" in path:
            group = "冻结因子输入分桶"
        elif path.endswith("unified_daily_market.parquet"):
            group = "全市场统一逐日面板"
        elif path.endswith("episode_daily_path.parquet"):
            group = "全事件逐日路径"
        elif path.endswith("point_in_time_industry_daily_prices_v1_2.parquet"):
            group = "点时行业日价格中间面板"
        elif path.endswith("panel_b_breakout_outcomes_v1_2.parquet"):
            group = "父研究Panel B大表"
        else:
            group = "其他上游文件"
        groups[group]["files"] += 1
        groups[group]["bytes"] += int(row["source_bytes"])
    group_lines = "\n".join(
        f"- {name}：{values['files']} 个文件，{values['bytes']:,} 字节。"
        for name, values in sorted(groups.items())
    )
    excluded_outputs = [
        (
            "data/staging/a_share_hs_extreme_compression_resolution_alpha_discovery_v2/"
            "stock_volatility_event_path.parquet",
            v2["artifacts"]["stock_volatility_event_path"],
            "V2全部事件0至60日个股波动路径；核心转移账本和摘要已纳入",
        ),
        (
            "data/staging/a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1/"
            "industry_peer_return_panel.parquet",
            v21["artifacts"]["industry_peer_return_panel"],
            "V2.1逐股票排除自身的行业同业日收益中间面板；最终特征账本已纳入",
        ),
    ]
    output_lines = "\n".join(
        (
            f"- {path}：{record['bytes']:,} 字节；SHA-256 {record['sha256']}；"
            f"{reason}。"
        )
        for path, record, reason in excluded_outputs
    )

    exclusions = f"""# 排除项、脱敏与解释边界

## 未收入的直接输入

构包时对两个运行回执声明的全部不同直接输入重新核对了存在性、字节数和
SHA-256。未收入项合计 {len(excluded_unique)} 个、{excluded_bytes:,} 字节：

{group_lines}

每一项的完整路径、字节数、SHA-256、所属研究和排除理由见
06_SOURCE_INVENTORY.csv。缺少这些大文件时，不能声称本 ZIP 可从原始行情
独立全量重跑 V2/V2.1；这是明确的体积边界，不是把数据缺失冒充完整。

## 未收入的两个大型结果/中间面板

{output_lines}

上述两个文件在构包前也按运行回执重新核对了大小和 SHA-256。

## 前序包

- prior_chat_review/ 是此前全聊天审阅包的完整脱敏镜像；原 ZIP 为
  {PRIOR_CHAT_ZIP.name}，源 SHA-256 {sha256_file(PRIOR_CHAT_ZIP)}。
- parent_v1_2_1/ 是 V1.2.1 包的关键证据子集，不是原 151 MB 解压目录的
  完整副本；原 ZIP 为 {PARENT_REVIEW_ZIP.name}，源 SHA-256
  {sha256_file(PARENT_REVIEW_ZIP)}。
- 原始全市场行情库、2,744份上交所源PDF、虚拟环境、缓存、pyc 和
  __pycache__ 没有重复装入。

## 脱敏

只替换本机绝对用户目录和用户名，不改研究数值、字段、阈值或表格。
本轮发生 {len(redactions)} 个文本文件脱敏；源哈希、包内哈希和替换项见
08_REDACTION_LOG.json。prior_chat_review/ 的局部清单已在脱敏后重建，
原清单另存为 04_ORIGINAL_FILE_MANIFEST_BEFORE_REDACTION.csv。

## 尚未解决的研究边界

- 收益为历史收盘到收盘价格研究口径，不含真实成本、滑点和现金收益。
- 未完成涨跌停、停牌成交、下一可交易开盘、T+1和组合持仓冲突验证。
- 退市、换股、合并和现金对价仍缺少完整权威经济终值。
- V2.1 是看过 V2 描述结果后冻结的机制发现层，不是盲测。
- 任何新因子或阈值都必须新建研究 ID 并预登记，不能覆盖本轮失败结果。
"""

    reproduce = """# 复核与复现说明

## 1. 先核验 ZIP 解压内容

在解压后的包根目录执行：

    python verify_package.py --package-root .

成功时状态为 PASS，并报告清单文件数、实际文件数、缺失、额外、大小和
SHA-256 不一致。05_ARTIFACT_MANIFEST_SHA256.json 不包含自身，以避免
自引用；验证器会把它作为允许的唯一额外清单文件。

## 2. 可在本包独立完成的检查

- 读取所有 CSV、JSON、Markdown 和已纳入的 Parquet。
- 复算 V2 六项主要检验表、V2.1 嵌套模型摘要、年度稳定性和政策覆盖摘要。
- 对 nested_model_oos_predictions.parquet 做 OOS 行数、唯一键和共同案例检查。
- 运行两个纯函数测试文件；需要 pytest、numpy、pandas、scikit-learn、
  scipy、duckdb、pyarrow 和 pyyaml。

    python -m pytest tests/test_a_share_hs_extreme_compression_resolution_alpha_v2.py tests/test_a_share_hs_extreme_compression_conditional_alpha_v2_1.py -q

## 3. 完整研究重跑

完整 V2/V2.1 重跑需要先把 06_SOURCE_INVENTORY.csv 中标记为
EXCLUDED_WITH_VERIFIED_HASH 的文件恢复到其 source_path。恢复后，在项目
根目录依次运行：

    python scripts/run_a_share_hs_extreme_compression_resolution_alpha_discovery_v2.py
    python scripts/run_a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1.py

两个脚本会写回其 artifacts 路径，因此应在审阅副本中运行，不要覆盖原冻结
产物。未恢复排除文件时，不要把脚本失败解释成研究结果失效。

## 4. 证据边界

构包成功只证明收集、脱敏、清单和 ZIP 完整性；不证明统计显著、预测有效、
可交易或适合实盘。
"""

    verifier = '''"""核验 ChatGPT Pro 审阅包的逐文件大小和 SHA-256。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="核验审阅包完整性")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest_path = root / "05_ARTIFACT_MANIFEST_SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest["files"]}
    failures = []
    for relative, record in expected.items():
        parts = PurePosixPath(relative).parts
        if not parts or ".." in parts:
            failures.append({"path": relative, "failure": "UNSAFE_PATH"})
            continue
        path = root.joinpath(*parts)
        if not path.is_file():
            failures.append({"path": relative, "failure": "MISSING"})
        elif path.stat().st_size != record["size_bytes"]:
            failures.append({"path": relative, "failure": "SIZE_MISMATCH"})
        elif sha256_file(path) != record["sha256"]:
            failures.append({"path": relative, "failure": "SHA256_MISMATCH"})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    missing_records = sorted(actual - set(expected))
    missing_files = sorted(set(expected) - actual)
    result = {
        "status": "PASS" if not failures and not missing_records and not missing_files else "FAIL",
        "manifest_files_excluding_manifest": len(expected),
        "actual_files_excluding_manifest": len(actual),
        "failures": failures,
        "files_without_manifest_record": missing_records,
        "manifest_records_without_file": missing_files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

    atomic_text(package_root / "00_README_FIRST.md", readme.strip() + "\n")
    atomic_text(
        package_root / "01_CHATGPT_PRO_REVIEW_PROMPT.md",
        prompt.strip() + "\n",
    )
    atomic_text(package_root / "02_CURRENT_HANDOFF.md", handoff.strip() + "\n")
    atomic_text(
        package_root / "03_EXCLUSIONS_AND_BOUNDARIES.md",
        exclusions.strip() + "\n",
    )
    atomic_text(
        package_root / "04_REPRODUCTION_AND_VERIFICATION.md",
        reproduce.strip() + "\n",
    )
    atomic_text(package_root / "verify_package.py", verifier)
    atomic_json(
        package_root / "08_REDACTION_LOG.json",
        {
            "scope": "ABSOLUTE_USER_PATHS_AND_USERNAME_ONLY",
            "redacted_file_count": len(redactions),
            "files": redactions,
        },
    )
    atomic_text(
        package_root / "prior_chat_review/REDACTION_NOTICE.md",
        (
            "# 脱敏镜像说明\n\n"
            "此目录是此前全聊天审阅包的完整文件镜像，但本机绝对用户目录和用户名"
            "已替换。原清单保存在 04_ORIGINAL_FILE_MANIFEST_BEFORE_REDACTION.csv；"
            "04_FILE_MANIFEST_SHA256.csv 是脱敏后重新生成的有效局部清单。\n"
        ),
    )
    atomic_text(
        package_root / "parent_v1_2_1/SUBSET_NOTICE.md",
        (
            "# V1.2.1 父证据子集说明\n\n"
            "此目录只收入父审阅包的协议、报告、代码、回执、源合同和固定复核抽样。"
            "父事件、收益和特征关键账本按原项目相对路径收入 data/；大型上游文件"
            "列入 06_SOURCE_INVENTORY.csv。02_ORIGINAL_FULL_PACKAGE_MANIFEST.json"
            "只描述原完整父包，不是本子目录的现行清单。\n"
        ),
    )
    atomic_json(
        package_root / "09_BUILD_CONTEXT.json",
        {
            "package_basename": PACKAGE_BASENAME,
            "internal_root": INTERNAL_ROOT,
            "built_at": datetime.now(TIME_ZONE).isoformat(),
            "v2_study_id": v2["study_id"],
            "v2_status": v2["status"],
            "v2_1_study_id": v21["study_id"],
            "v2_1_status": v21["status"],
            "v2_1_conclusion": v21["conclusion"]["status"],
            "prior_chat_subpackage_manifest": prior_chat_manifest,
            "research_mutated": False,
            "orders_generated": False,
        },
    )


def build_manifest(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "05_ARTIFACT_MANIFEST_SHA256.json"
    records: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path != manifest_path:
            records.append(
                {
                    "path": path.relative_to(package_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "package_basename": PACKAGE_BASENAME,
        "internal_root": INTERNAL_ROOT,
        "created_at": datetime.now(TIME_ZONE).isoformat(),
        "scope": "ALL_PACKAGE_FILES_EXCEPT_THIS_MANIFEST",
        "file_count": len(records),
        "total_size_bytes_excluding_manifest": sum(
            item["size_bytes"] for item in records
        ),
        "files": records,
    }


def verify_directory(package_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = package_root / "05_ARTIFACT_MANIFEST_SHA256.json"
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    failures: list[dict[str, Any]] = []
    for relative, record in expected.items():
        path = package_root / Path(relative)
        if not path.is_file():
            failures.append({"path": relative, "failure": "MISSING"})
        elif path.stat().st_size != record["size_bytes"]:
            failures.append({"path": relative, "failure": "SIZE_MISMATCH"})
        elif sha256_file(path) != record["sha256"]:
            failures.append({"path": relative, "failure": "SHA256_MISMATCH"})
    extra = sorted(actual - set(expected))
    missing = sorted(set(expected) - actual)
    result = {
        "status": "PASS" if not failures and not extra and not missing else "FAIL",
        "expected_files_excluding_manifest": len(expected),
        "actual_files_excluding_manifest": len(actual),
        "failures": failures,
        "extra_files": extra,
        "missing_files": missing,
    }
    if result["status"] != "PASS":
        raise AssertionError(f"目录清单复核失败：{result}")
    return result


def build_zip(package_root: Path) -> None:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".zip.tmp",
        prefix=f".{PACKAGE_BASENAME}.",
        dir=DELIVERABLES,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(package_root).as_posix()
                    archive.write(path, f"{INTERNAL_ROOT}/{relative}")
        os.replace(temporary, ZIP_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_zip(zip_path: Path) -> dict[str, Any]:
    manifest_member = f"{INTERNAL_ROOT}/05_ARTIFACT_MANIFEST_SHA256.json"
    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise AssertionError(f"ZIP CRC失败：{corrupt}")
        members = [item for item in archive.infolist() if not item.is_dir()]
        for item in members:
            pure = PurePosixPath(item.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise AssertionError(f"ZIP含不安全路径：{item.filename}")
            if not item.filename.startswith(f"{INTERNAL_ROOT}/"):
                raise AssertionError(f"ZIP成员不在唯一根目录：{item.filename}")
        manifest = json.loads(archive.read(manifest_member).decode("utf-8"))
        expected = {
            f"{INTERNAL_ROOT}/{item['path']}": item
            for item in manifest["files"]
        }
        actual = {item.filename for item in members}
        expected_with_manifest = set(expected) | {manifest_member}
        missing = sorted(expected_with_manifest - actual)
        extra = sorted(actual - expected_with_manifest)
        failures: list[dict[str, Any]] = []
        for member, record in expected.items():
            info = archive.getinfo(member)
            if info.file_size != record["size_bytes"]:
                failures.append({"path": member, "failure": "SIZE_MISMATCH"})
                continue
            with archive.open(member, "r") as handle:
                actual_hash = sha256_stream(handle)
            if actual_hash != record["sha256"]:
                failures.append({"path": member, "failure": "SHA256_MISMATCH"})
    result = {
        "status": "PASS" if not missing and not extra and not failures else "FAIL",
        "zip_entries_including_manifest": len(actual),
        "manifest_files_excluding_manifest": len(expected),
        "missing_files": missing,
        "extra_files": extra,
        "failures": failures,
        "crc_test": "PASS",
    }
    if result["status"] != "PASS":
        raise AssertionError(f"ZIP字节流复核失败：{result}")
    return result


def verify_fresh_extraction(zip_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aev21_verify_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for item in archive.infolist():
                pure = PurePosixPath(item.filename)
                if pure.is_absolute() or ".." in pure.parts:
                    raise AssertionError(f"拒绝解压不安全路径：{item.filename}")
            archive.extractall(root)
        package_root = root / INTERNAL_ROOT
        manifest = load_json(
            package_root / "05_ARTIFACT_MANIFEST_SHA256.json"
        )
        result = verify_directory(package_root, manifest)
        result["fresh_temporary_extraction"] = True
        return result


def main() -> int:
    for target in [ZIP_PATH, RECEIPT_PATH, SHA256_PATH]:
        if target.exists():
            raise FileExistsError(f"拒绝覆盖既有交付物：{target}")
    for required in [
        V2_RECEIPT_PATH,
        V21_RECEIPT_PATH,
        PRIOR_CHAT_ROOT,
        PRIOR_CHAT_ZIP,
        PARENT_REVIEW_ROOT,
        PARENT_REVIEW_ZIP,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"缺少必要前序证据：{required}")

    print("读取 V2/V2.1 冻结回执并登记包内文件。", flush=True)
    v2 = load_json(V2_RECEIPT_PATH)
    v21 = load_json(V21_RECEIPT_PATH)
    builder = PackageBuilder()
    register_current_artifacts(builder)
    register_parent_evidence(builder)
    register_prior_chat_package(builder)
    inputs = direct_input_records(v2, v21)
    register_small_direct_inputs(builder, inputs)

    print("重新核对全部直接输入和全部声明产物的大小与 SHA-256。", flush=True)
    hash_cache: dict[Path, str] = {}
    input_verification = verify_receipt_records(
        inputs,
        label="DIRECT_INPUTS",
        hash_cache=hash_cache,
    )
    output_verification = verify_receipt_records(
        artifact_records(v2, v21),
        label="DECLARED_ARTIFACTS",
        hash_cache=hash_cache,
    )

    with tempfile.TemporaryDirectory(prefix="aev21_build_") as temporary:
        package_root = Path(temporary) / INTERNAL_ROOT
        package_root.mkdir(parents=True)
        print(f"复制并脱敏 {len(builder.mappings):,} 个源文件。", flush=True)
        builder.copy_all(package_root)
        inventory = create_source_inventory(
            package_root,
            builder,
            inputs,
            hash_cache,
        )
        atomic_json(
            package_root / "08_REDACTION_LOG.json",
            {
                "scope": "ABSOLUTE_USER_PATHS_AND_USERNAME_ONLY",
                "redacted_file_count": len(builder.redactions),
                "files": builder.redactions,
            },
        )
        atomic_text(
            package_root / "prior_chat_review/REDACTION_NOTICE.md",
            (
                "# 脱敏镜像说明\n\n"
                "此前包中的本机绝对用户目录和用户名已替换；研究数值、字段和阈值"
                "未修改。原清单保留为 04_ORIGINAL_FILE_MANIFEST_BEFORE_REDACTION.csv。"
                "现行局部清单会在脱敏后重新生成。\n"
            ),
        )
        prior_manifest = create_prior_chat_manifest(package_root)
        parquet_coverage = create_parquet_coverage(package_root)
        create_documents(
            package_root,
            v2,
            v21,
            inventory,
            prior_manifest,
            parquet_coverage,
            builder.redactions,
        )
        # create_documents 会更新脱敏日志和子包说明，故再生成一次局部清单。
        prior_manifest = create_prior_chat_manifest(package_root)
        manifest = build_manifest(package_root)
        atomic_json(
            package_root / "05_ARTIFACT_MANIFEST_SHA256.json",
            manifest,
        )
        print(
            f"压缩前复核 {manifest['file_count']:,} 个清单文件。",
            flush=True,
        )
        directory_verification = verify_directory(package_root, manifest)
        build_zip(package_root)

    print("从 ZIP 字节流逐文件复核，并执行一次全新解压复核。", flush=True)
    zip_verification = verify_zip(ZIP_PATH)
    extraction_verification = verify_fresh_extraction(ZIP_PATH)
    zip_hash = sha256_file(ZIP_PATH)
    receipt = {
        "status": "CHATGPT_PRO_REVIEW_ZIP_READY",
        "package_basename": PACKAGE_BASENAME,
        "internal_root": INTERNAL_ROOT,
        "built_at": datetime.now(TIME_ZONE).isoformat(),
        "zip_path": ZIP_PATH.relative_to(ROOT).as_posix(),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": zip_hash,
        "research_mutated": False,
        "input_receipt_verification": input_verification,
        "declared_artifact_verification": output_verification,
        "pre_zip_directory_verification": directory_verification,
        "zip_stream_verification": zip_verification,
        "fresh_extraction_verification": extraction_verification,
        "redacted_file_count": len(builder.redactions),
        "v2_primary_gate_pass_count": sum(
            bool(item["historical_primary_gate_pass"])
            for item in v2["primary_hypotheses"]
        ),
        "v2_1_mechanism_gate_pass_count": v21["conclusion"][
            "mechanism_candidate_gate_pass_count"
        ],
        "v2_1_conclusion": v21["conclusion"]["status"],
    }
    atomic_json(RECEIPT_PATH, receipt)
    atomic_text(SHA256_PATH, f"{zip_hash}  {ZIP_PATH.name}\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "zip": str(ZIP_PATH),
                "zip_bytes": receipt["zip_bytes"],
                "zip_sha256": zip_hash,
                "files_excluding_manifest": manifest["file_count"],
                "zip_entries_including_manifest": zip_verification[
                    "zip_entries_including_manifest"
                ],
                "redacted_file_count": len(builder.redactions),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
