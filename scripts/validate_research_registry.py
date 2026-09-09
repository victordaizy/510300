"""校验事件式研究注册表、证据绑定和研究范围隔离。

本模块只读取研究证据，不运行模型、不生成观点，也不授权仓位或订单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "research_registry.jsonl"

SCOPES = {"ETF_510300", "INTERNATIONAL_BROKER"}
RUN_STATUSES = {"SUCCESS", "FAILED"}
COLLECTION_STATUSES = {"NOT_STARTED", "COLLECTING", "COMPLETE", "BLOCKED"}
RESEARCH_STATUSES = {
    "DISCOVERY_ONLY",
    "PREREGISTERED",
    "FORWARD_COLLECTING",
    "EVALUATION_ELIGIBLE",
    "PASSED_RESEARCH",
    "REJECTED_FROZEN",
}
VIEW_STATUSES = {"VIEW", "NO_VIEW"}
AUTHORIZATIONS = {"NONE", "PAPER_ONLY", "REAL_DISABLED"}
GOVERNANCE_STATUSES = {
    "PASS",
    "BLOCKED_GOVERNANCE_UNVERSIONED",
    "BLOCKED_STATUS_CONFLICT",
    "BLOCKED_SCOPE_CONTAMINATION",
}
ARTIFACT_ROLES = {
    "protocol",
    "implementation",
    "input_manifest",
    "output_report",
    "status_file",
    "configuration",
    "data_contract",
}
BINDING_STATUSES = {"VERSIONED", "UNVERSIONED", "MISSING"}
REQUIRED_ARTIFACT_ROLES = {
    "protocol",
    "implementation",
    "input_manifest",
    "output_report",
    "status_file",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MODEL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_]*$")


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_sha256(path: Path) -> str:
    return sha256_file(path)


def load_registry(path: Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    """逐行读取 JSONL；错误消息保留物理行号，便于第三方复核。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"注册表第{line_number}行不是合法JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"注册表第{line_number}行必须是JSON对象")
            record["_line_number"] = line_number
            records.append(record)
    if not records:
        raise ValueError("研究注册表为空")
    return records


def _parse_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field}必须是带时区的ISO时间")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field}必须包含时区")
    return parsed


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field}必须是YYYY-MM-DD")
    return date.fromisoformat(value)


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("证据路径必须是非空的项目相对路径")
    if "\\" in value:
        raise ValueError(f"注册表路径必须使用正斜杠：{value}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or ":" in value:
        raise ValueError(f"证据路径越出项目根目录：{value}")
    return pure.as_posix()


def _git_lines(root: Path, *args: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line]


def git_state_index(root: Path) -> dict[str, str]:
    """一次性构建 Git 状态索引，避免对每个证据启动一个进程。"""

    tracked = {line.replace("\\", "/") for line in _git_lines(root, "ls-files")}
    modified: set[str] = set()
    untracked: set[str] = set()
    for line in _git_lines(root, "status", "--porcelain=v1", "-uall"):
        if len(line) < 4:
            continue
        state = line[:2]
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        path = raw_path.strip('"').replace("\\", "/")
        if state == "??":
            untracked.add(path)
        else:
            modified.add(path)
    result = {path: "TRACKED_CLEAN" for path in tracked}
    result.update({path: "TRACKED_MODIFIED" for path in modified})
    result.update({path: "UNTRACKED" for path in untracked})
    return result


def discover_unregistered_manifests(
    root: Path, records: Iterable[dict[str, Any]]
) -> list[str]:
    """列出未被任一注册事件引用的配置manifest，不猜测其研究ID。"""

    referenced = {
        artifact.get("path")
        for record in records
        for artifact in record.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("path")
    }
    config_dir = root / "config"
    if not config_dir.exists():
        return []
    candidates = {
        path.relative_to(root).as_posix()
        for path in config_dir.glob("*manifest*.json")
        if path.is_file() and not path.name.startswith("artifact_manifest_")
    }
    return sorted(candidates - referenced)


def latest_records(
    records: Iterable[dict[str, Any]], as_of_date: str | None = None
) -> list[dict[str, Any]]:
    """为每个模型选择指定日期当日或之前最后一个事件。"""

    cutoff = date.fromisoformat(as_of_date) if as_of_date else None
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        event_date = date.fromisoformat(record["as_of_date"])
        if cutoff is not None and event_date > cutoff:
            continue
        model_id = record["model_id"]
        sort_key = (
            event_date,
            datetime.fromisoformat(record["recorded_at"]),
            int(record.get("_line_number", 0)),
        )
        existing = selected.get(model_id)
        if existing is None:
            selected[model_id] = record
            continue
        existing_key = (
            date.fromisoformat(existing["as_of_date"]),
            datetime.fromisoformat(existing["recorded_at"]),
            int(existing.get("_line_number", 0)),
        )
        if sort_key > existing_key:
            selected[model_id] = record
    return [selected[key] for key in sorted(selected)]


def _protocol_hash(record: dict[str, Any]) -> str | None:
    for artifact in record.get("artifacts", []):
        if artifact.get("role") == "protocol":
            return artifact.get("sha256")
    return None


def validate_records(
    records: list[dict[str, Any]],
    *,
    root: Path = ROOT,
    verify_hashes: bool = True,
    audit_coverage: bool = True,
) -> dict[str, Any]:
    """返回可序列化审计结果；不因阻断状态覆盖真实错误。"""

    errors: list[str] = []
    warnings: list[str] = []
    seen_event_ids: set[str] = set()
    model_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    git_states = git_state_index(root)
    latest_event_ids = {
        record["event_id"]
        for record in latest_records(records)
        if isinstance(record.get("event_id"), str)
    }

    required_fields = {
        "schema_version",
        "event_id",
        "event_type",
        "recorded_at",
        "as_of_date",
        "model_id",
        "implementation_id",
        "research_scope",
        "namespace",
        "source_commit",
        "run_status",
        "collection_status",
        "research_status",
        "view_status",
        "authorization",
        "governance_status",
        "status_reason",
        "sample_metrics",
        "artifacts",
        "dependencies",
        "feature_dependencies",
    }

    for ordinal, record in enumerate(records, start=1):
        line = record.get("_line_number", ordinal)
        prefix = f"第{line}行"
        is_latest_event = record.get("event_id") in latest_event_ids
        missing_fields = sorted(required_fields - set(record))
        if missing_fields:
            errors.append(f"{prefix}缺少字段：{missing_fields}")
            continue

        event_id = record["event_id"]
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"{prefix}event_id无效")
        elif event_id in seen_event_ids:
            errors.append(f"{prefix}event_id重复：{event_id}")
        else:
            seen_event_ids.add(event_id)

        if record["schema_version"] != "1.0.0":
            errors.append(f"{prefix}schema_version必须为1.0.0")
        if record["event_type"] != "STATUS_SNAPSHOT":
            errors.append(f"{prefix}event_type必须为STATUS_SNAPSHOT")
        try:
            _parse_datetime(record["recorded_at"], f"{prefix}.recorded_at")
            _parse_date(record["as_of_date"], f"{prefix}.as_of_date")
        except ValueError as exc:
            errors.append(str(exc))

        model_id = record["model_id"]
        if not isinstance(model_id, str) or not MODEL_PATTERN.fullmatch(model_id):
            errors.append(f"{prefix}model_id格式无效：{model_id!r}")
            continue
        model_events[model_id].append(record)

        scope = record["research_scope"]
        if scope not in SCOPES:
            errors.append(f"{prefix}research_scope无效：{scope!r}")
        if record["namespace"] != f"{scope}::{model_id}":
            errors.append(f"{prefix}namespace未与scope/model_id严格绑定")
        if not isinstance(record["implementation_id"], str) or not record[
            "implementation_id"
        ]:
            errors.append(f"{prefix}implementation_id必须为非空字符串")

        enum_checks = (
            ("run_status", RUN_STATUSES),
            ("collection_status", COLLECTION_STATUSES),
            ("research_status", RESEARCH_STATUSES),
            ("view_status", VIEW_STATUSES),
            ("authorization", AUTHORIZATIONS),
            ("governance_status", GOVERNANCE_STATUSES),
        )
        for field, allowed in enum_checks:
            if record[field] not in allowed:
                errors.append(f"{prefix}{field}无效：{record[field]!r}")

        commit = record["source_commit"]
        if commit is not None and (
            not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit)
        ):
            errors.append(f"{prefix}source_commit必须为40位小写Git提交哈希或null")
        if commit is None and record["governance_status"] == "PASS":
            errors.append(f"{prefix}治理PASS时source_commit不得为空")
        if not isinstance(record["status_reason"], str) or not record["status_reason"]:
            errors.append(f"{prefix}status_reason必须为非空字符串")

        metrics = record["sample_metrics"]
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}sample_metrics必须为对象")
        else:
            observed = metrics.get("observed_sample_count")
            mature = metrics.get("mature_sample_count")
            for name, value in (("observed_sample_count", observed), ("mature_sample_count", mature)):
                if value is not None and (not isinstance(value, int) or value < 0):
                    errors.append(f"{prefix}{name}必须为非负整数或null")
            if isinstance(observed, int) and isinstance(mature, int) and mature > observed:
                errors.append(f"{prefix}成熟样本数不得大于观察样本数")

        artifacts = record["artifacts"]
        if not isinstance(artifacts, list):
            errors.append(f"{prefix}artifacts必须为数组")
            artifacts = []
        roles: set[str] = set()
        paths: set[str] = set()
        for artifact_number, artifact in enumerate(artifacts, start=1):
            artifact_prefix = f"{prefix}第{artifact_number}个证据"
            if not isinstance(artifact, dict):
                errors.append(f"{artifact_prefix}必须为对象")
                continue
            role = artifact.get("role")
            if role not in ARTIFACT_ROLES:
                errors.append(f"{artifact_prefix}role无效：{role!r}")
            else:
                roles.add(role)
            binding_status = artifact.get("binding_status")
            if binding_status not in BINDING_STATUSES:
                errors.append(f"{artifact_prefix}binding_status无效：{binding_status!r}")
            path_value = artifact.get("path")
            digest = artifact.get("sha256")
            if path_value is None:
                if digest is not None or binding_status != "MISSING":
                    errors.append(f"{artifact_prefix}缺失路径必须同时使用sha256=null和MISSING")
                continue
            try:
                relative = _safe_relative_path(path_value)
            except ValueError as exc:
                errors.append(f"{artifact_prefix}{exc}")
                continue
            if relative in paths:
                errors.append(f"{prefix}证据路径重复：{relative}")
            paths.add(relative)

            lower_relative = relative.lower()
            if scope == "ETF_510300" and "international_broker" in lower_relative:
                errors.append(f"{artifact_prefix}发生国际券商证据流入510300范围：{relative}")
            if scope == "INTERNATIONAL_BROKER" and (
                lower_relative.startswith("paper/") or "510300" in lower_relative
            ):
                errors.append(f"{artifact_prefix}发生510300证据流入国际券商范围：{relative}")

            absolute = root / PurePosixPath(relative)
            exists = absolute.is_file()
            if is_latest_event and binding_status == "MISSING" and exists:
                errors.append(f"{artifact_prefix}标记MISSING但文件存在：{relative}")
            if is_latest_event and binding_status != "MISSING" and not exists:
                errors.append(f"{artifact_prefix}文件不存在：{relative}")
                continue
            if digest is not None and (
                not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest)
            ):
                errors.append(f"{artifact_prefix}sha256格式无效")
            if binding_status == "VERSIONED" and digest is None:
                errors.append(f"{artifact_prefix}VERSIONED证据必须包含sha256")
            if (
                is_latest_event
                and binding_status == "VERSIONED"
                and git_states.get(relative) != "TRACKED_CLEAN"
            ):
                errors.append(
                    f"{artifact_prefix}声称VERSIONED但Git状态为"
                    f"{git_states.get(relative, 'UNTRACKED_OR_UNKNOWN')}：{relative}"
                )
            if is_latest_event and exists and digest is not None and verify_hashes:
                actual = sha256_file(absolute)
                if actual != digest:
                    errors.append(
                        f"{artifact_prefix}SHA-256不一致：{relative}，"
                        f"expected={digest}，actual={actual}"
                    )

        missing_roles = sorted(REQUIRED_ARTIFACT_ROLES - roles)
        if missing_roles and record["governance_status"] != "BLOCKED_GOVERNANCE_UNVERSIONED":
            errors.append(f"{prefix}缺少关键证据角色但未阻断治理：{missing_roles}")

        dependencies = record["dependencies"]
        if not isinstance(dependencies, list):
            errors.append(f"{prefix}dependencies必须为数组")
        else:
            for dependency in dependencies:
                if not isinstance(dependency, dict):
                    errors.append(f"{prefix}依赖项必须为对象")
                    continue
                dependency_scope = dependency.get("research_scope")
                dependency_id = dependency.get("model_id")
                if dependency_scope != scope:
                    errors.append(
                        f"{prefix}跨研究范围依赖：{model_id}({scope}) -> "
                        f"{dependency_id}({dependency_scope})"
                    )
                if not isinstance(dependency_id, str) or not MODEL_PATTERN.fullmatch(
                    dependency_id
                ):
                    errors.append(f"{prefix}依赖model_id格式无效：{dependency_id!r}")

        feature_dependencies = record["feature_dependencies"]
        if not isinstance(feature_dependencies, list):
            errors.append(f"{prefix}feature_dependencies必须为数组")
        else:
            namespace_prefix = f"{scope}::"
            for feature in feature_dependencies:
                if not isinstance(feature, dict):
                    errors.append(f"{prefix}特征依赖项必须为对象")
                    continue
                feature_scope = feature.get("research_scope")
                feature_namespace = feature.get("namespace")
                if feature_scope != scope or not isinstance(
                    feature_namespace, str
                ) or not feature_namespace.startswith(namespace_prefix):
                    errors.append(
                        f"{prefix}特征跨研究范围：{feature_namespace!r}({feature_scope!r})"
                    )

        if record["authorization"] == "PAPER_ONLY" and scope != "ETF_510300":
            errors.append(f"{prefix}国际券商研究不得获得PAPER_ONLY授权")
        if record.get("live_trading_authorized") is not False:
            errors.append(f"{prefix}live_trading_authorized必须显式为false")

    for model_id, events in model_events.items():
        ordered = sorted(
            events,
            key=lambda item: (
                date.fromisoformat(item["as_of_date"]),
                datetime.fromisoformat(item["recorded_at"]),
                int(item.get("_line_number", 0)),
            ),
        )
        frozen_seen = False
        previous: dict[str, Any] | None = None
        for event in ordered:
            if frozen_seen and event["research_status"] != "REJECTED_FROZEN":
                errors.append(f"{model_id}在REJECTED_FROZEN终态后发生非法回退")
            if event["research_status"] == "REJECTED_FROZEN":
                frozen_seen = True
            if (
                previous is not None
                and previous["collection_status"] == "BLOCKED"
                and event["collection_status"] == "COLLECTING"
                and _protocol_hash(previous) != _protocol_hash(event)
            ):
                errors.append(f"{model_id}从BLOCKED恢复COLLECTING时协议哈希发生变化")
            previous = event

    latest = latest_records(records)
    latest_ids = {record["model_id"] for record in latest}
    for record in latest:
        for dependency in record.get("dependencies", []):
            if dependency.get("model_id") not in latest_ids:
                warnings.append(
                    f"{record['model_id']}依赖的{dependency.get('model_id')}尚未登记；"
                    "不得把依赖存在解释为已验证"
                )

    unregistered = (
        discover_unregistered_manifests(root, records) if audit_coverage else []
    )
    if errors:
        status = "INVALID"
    elif unregistered:
        status = "BLOCKED_GOVERNANCE_UNVERSIONED"
    elif any(record["governance_status"] != "PASS" for record in latest):
        blockers = {record["governance_status"] for record in latest}
        status = sorted(blockers)[0]
    else:
        status = "PASS"

    return {
        "status": status,
        "record_count": len(records),
        "model_count": len(model_events),
        "latest_model_count": len(latest),
        "errors": errors,
        "warnings": warnings,
        "unregistered_manifest_count": len(unregistered),
        "unregistered_manifest_paths": unregistered,
    }


def _clean_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key != "_line_number"}
        for record in records
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验单一权威事件式研究注册表")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strict", action="store_true", help="阻断状态使用退出码2")
    parser.add_argument(
        "--no-hash-check", action="store_true", help="仅检查哈希格式，不读取证据内容"
    )
    parser.add_argument(
        "--no-coverage-audit", action="store_true", help="不扫描未登记manifest"
    )
    args = parser.parse_args(argv)
    registry_path = args.registry.resolve()
    try:
        records = load_registry(registry_path)
        result = validate_records(
            records,
            root=ROOT,
            verify_hashes=not args.no_hash_check,
            audit_coverage=not args.no_coverage_audit,
        )
    except (OSError, ValueError) as exc:
        result = {
            "status": "INVALID",
            "record_count": 0,
            "model_count": 0,
            "latest_model_count": 0,
            "errors": [str(exc)],
            "warnings": [],
            "unregistered_manifest_count": 0,
            "unregistered_manifest_paths": [],
        }
    result["registry"] = registry_path.relative_to(ROOT).as_posix()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "INVALID":
        return 1
    if args.strict and result["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
