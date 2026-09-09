"""从事件式注册表渲染唯一当前研究状态，不读取模型输出自行推断状态。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.validate_research_registry import (
        DEFAULT_REGISTRY,
        ROOT,
        latest_records,
        load_registry,
        registry_sha256,
        validate_records,
    )
except ModuleNotFoundError:  # 兼容直接执行 scripts\\render_research_status.py
    from validate_research_registry import (  # type: ignore[no-redef]
        DEFAULT_REGISTRY,
        ROOT,
        latest_records,
        load_registry,
        registry_sha256,
        validate_records,
    )


DEFAULT_OUTPUT = ROOT / "reports" / "audit" / "research_status_current.json"


def derived_display_status(record: dict[str, Any]) -> str:
    """展示状态仅由正交状态组合，绝不成为新的生命周期状态。"""

    return "__".join(
        (
            record["run_status"],
            record["collection_status"],
            record["research_status"],
            record["view_status"],
            record["authorization"],
        )
    )


def _artifact_summary(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = record.get("artifacts", [])
    return {
        "artifact_count": len(artifacts),
        "versioned_count": sum(
            artifact.get("binding_status") == "VERSIONED" for artifact in artifacts
        ),
        "unversioned_count": sum(
            artifact.get("binding_status") == "UNVERSIONED" for artifact in artifacts
        ),
        "missing_count": sum(
            artifact.get("binding_status") == "MISSING" for artifact in artifacts
        ),
        "paths": [artifact.get("path") for artifact in artifacts],
    }


def render_status(
    records: list[dict[str, Any]],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    as_of_date: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    selected = latest_records(records, as_of_date)
    if not selected:
        raise ValueError("指定日期之前没有注册事件")
    validation = validate_records(
        records,
        root=root,
        verify_hashes=True,
        audit_coverage=True,
    )
    effective_as_of = as_of_date or max(record["as_of_date"] for record in selected)
    rendered_at = max(
        datetime.fromisoformat(record["recorded_at"]) for record in selected
    ).isoformat()

    models: list[dict[str, Any]] = []
    scope_counts = {
        scope: {
            "model_count": 0,
            "view_count": 0,
            "no_view_count": 0,
            "rejected_frozen_count": 0,
            "governance_blocked_count": 0,
        }
        for scope in ("ETF_510300", "INTERNATIONAL_BROKER")
    }
    for record in selected:
        scope = record["research_scope"]
        scope_counts[scope]["model_count"] += 1
        scope_counts[scope]["view_count" if record["view_status"] == "VIEW" else "no_view_count"] += 1
        if record["research_status"] == "REJECTED_FROZEN":
            scope_counts[scope]["rejected_frozen_count"] += 1
        if record["governance_status"] != "PASS":
            scope_counts[scope]["governance_blocked_count"] += 1
        models.append(
            {
                "model_id": record["model_id"],
                "implementation_id": record["implementation_id"],
                "research_scope": scope,
                "namespace": record["namespace"],
                "as_of_date": record["as_of_date"],
                "recorded_at": record["recorded_at"],
                "source_commit": record["source_commit"],
                "run_status": record["run_status"],
                "collection_status": record["collection_status"],
                "research_status": record["research_status"],
                "view_status": record["view_status"],
                "authorization": record["authorization"],
                "governance_status": record["governance_status"],
                "status_reason": record["status_reason"],
                "sample_metrics": record["sample_metrics"],
                "derived_display_status": derived_display_status(record),
                "artifact_summary": _artifact_summary(record),
                "dependencies": record.get("dependencies", []),
                "live_trading_authorized": False,
            }
        )

    return {
        "schema_version": "1.0.0",
        "generated_from": registry_path.relative_to(root).as_posix(),
        "generated_at": rendered_at,
        "as_of_date": effective_as_of,
        "registry_sha256": registry_sha256(registry_path),
        "overall_status": validation["status"],
        "scope_counts": scope_counts,
        "models": models,
        "reconciliation": validation,
        "safety": {
            "status_is_not_trading_instruction": True,
            "success_does_not_mean_research_passed": True,
            "no_view_does_not_mean_short_or_sell": True,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从注册表渲染研究状态")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of", dest="as_of_date", type=str)
    parser.add_argument("--check", action="store_true", help="只校验已渲染文件")
    args = parser.parse_args(argv)
    registry_path = args.registry.resolve()
    output_path = args.output.resolve()
    try:
        if args.as_of_date is not None:
            date.fromisoformat(args.as_of_date)
        records = load_registry(registry_path)
        payload = render_status(
            records,
            registry_path=registry_path,
            as_of_date=args.as_of_date,
            root=ROOT,
        )
        expected = _canonical(payload)
        if args.check:
            if not output_path.exists():
                print(f"状态文件不存在：{output_path}", file=sys.stderr)
                return 1
            actual = output_path.read_text(encoding="utf-8")
            if actual != expected:
                print("状态文件与注册表派生结果不一致：BLOCKED_STATUS_CONFLICT", file=sys.stderr)
                return 1
            print("状态文件与注册表一致。")
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(expected, encoding="utf-8")
        print(f"已生成：{output_path}")
        print(f"总状态：{payload['overall_status']}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"状态渲染失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
