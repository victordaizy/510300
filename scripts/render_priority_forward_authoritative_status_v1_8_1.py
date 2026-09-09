"""以模块入口生成绑定 V1.10 清单的权威状态 V1.8.1。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import render_priority_forward_authoritative_status_v1_8 as base
from scripts.run_priority_forward_codex_automation_v1_10 import (
    verify_manifest as verify_v1_10_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/priority_forward_authoritative_status_v1_8_1.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return base.load_config(path)


def build_report(config: dict, generated_at: datetime) -> dict:
    original_verifier = base.verify_manifest
    base.verify_manifest = verify_v1_10_manifest
    try:
        report = base.build_report(config, generated_at)
    finally:
        base.verify_manifest = original_verifier
    report["schema_version"] = "1.8.1"
    report["report_version"] = config["version"]
    return report


def render_markdown(report: dict) -> str:
    return base.render_markdown(report).replace(
        "# 优先前瞻研究权威状态 V1.8",
        "# 优先前瞻研究权威状态 V1.8.1",
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="生成优先前瞻研究权威状态 V1.8.1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    config = load_config(config_path)
    generated_at = datetime.now(ZoneInfo(str(config["timezone"])))
    report = build_report(config, generated_at)
    base._atomic_text(
        base._path(config["outputs"]["json"]),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    base._atomic_text(
        base._path(config["outputs"]["markdown"]),
        render_markdown(report),
    )
    pcf = report["active_research_streams"][1]
    latest = pcf["latest_task_authority"]
    print(f"权威状态：{report['overall_research_status']}")
    print(f"PCF/IOPV 部署状态：{pcf['deployment_status']}")
    print(
        "PCF/IOPV 最新直接状态："
        f"{latest['collection_status']}，退出码 {latest['task_exit_code']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
