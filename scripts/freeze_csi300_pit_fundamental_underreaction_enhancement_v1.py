from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "csi300_pit_fundamental_underreaction_enhancement_v1.yaml"
MANIFEST_PATH = PROJECT_ROOT / "config" / "csi300_pit_fundamental_underreaction_enhancement_v1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def terminal_status(config: dict[str, Any]) -> dict[str, Any]:
    closure = config["legacy_program_closure"]
    components: list[dict[str, Any]] = []
    for component in closure["components"]:
        evidence_path = PROJECT_ROOT / component["evidence"]
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        observed_status = payload.get("status")
        if observed_status != component["required_status"]:
            raise RuntimeError(
                f"旧项目状态漂移：{component['study_id']}，"
                f"观测={observed_status}，要求={component['required_status']}"
            )
        components.append(
            {
                "study_id": component["study_id"],
                "status": observed_status,
                "evidence": component["evidence"],
                "evidence_sha256": sha256_file(evidence_path),
            }
        )
    return {
        "program_id": closure["timing_program_id"],
        "status": closure["timing_program_status"],
        "dashboard_id": closure["dashboard_id"],
        "dashboard_status": closure["dashboard_status"],
        "components": components,
        "forbidden_reuse": closure["forbidden_reuse"],
        "market_price_read": False,
        "future_return_read": False,
        "position_mapping": False,
        "trading_authorization": False,
    }


def terminal_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 510300 时间序列压力择时研究终局状态",
        "",
        f"- 历史开发总状态：`{payload['status']}`",
        f"- 观察面板状态：`{payload['dashboard_status']}`",
        "- 仓位映射：`false`",
        "- 交易授权：`false`",
        "",
        "## 组成研究",
        "",
    ]
    for component in payload["components"]:
        lines.append(f"- `{component['study_id']}` = `{component['status']}`")
    lines.extend(
        [
            "",
            "压力传导、耗竭、H3 与外部复制证据今后只用于解释市场状态和积累新的独立事件，"
            "不得映射卖出、减仓、回补或现金切换，也不得在原历史上调整阈值、窗口、标签或模型救援。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    terminal = terminal_status(config)
    terminal["frozen_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    artifacts = config["artifacts"]
    terminal_json_path = PROJECT_ROOT / artifacts["terminal_status_json"]
    terminal_markdown_path = PROJECT_ROOT / artifacts["terminal_status_markdown"]
    write_json(terminal_json_path, terminal)
    terminal_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_markdown_path.write_text(terminal_markdown(terminal), encoding="utf-8")

    frozen_paths = [
        CONFIG_PATH,
        PROJECT_ROOT / artifacts["protocol_document"],
        PROJECT_ROOT
        / "research"
        / "csi300_pit_fundamental_underreaction_enhancement_v1_preflight.py",
        PROJECT_ROOT
        / "scripts"
        / "run_csi300_pit_fundamental_underreaction_enhancement_v1_preflight.py",
        Path(__file__).resolve(),
        PROJECT_ROOT
        / "tests"
        / "test_csi300_pit_fundamental_underreaction_enhancement_v1_preflight.py",
        terminal_json_path,
        terminal_markdown_path,
    ]
    files = []
    for path in frozen_paths:
        if not path.exists():
            raise FileNotFoundError(f"冻结文件缺失：{path}")
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_ENHANCEMENT_V1_MANIFEST",
        "status": "FROZEN_BEFORE_OFFICIAL_FACT_ACQUISITION_AND_ANY_FUTURE_RETURN_READ",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "study_id": config["study"]["study_id"],
        "version": config["study"]["version"],
        "files": files,
        "legacy_timing_program_status": terminal["status"],
        "official_fact_acquisition_started": False,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_return_calculated": False,
        "strategy_sharpe_calculated": False,
        "position_mapping": False,
        "trading_authorization": False,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    write_json(MANIFEST_PATH, manifest)
    print(f"冻结清单：{MANIFEST_PATH}")
    print(f"终局状态：{terminal_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

