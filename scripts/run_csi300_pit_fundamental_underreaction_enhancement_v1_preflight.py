from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.csi300_pit_fundamental_underreaction_enhancement_v1_preflight import (  # noqa: E402
    project_path,
    run_preflight,
    sha256_file,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "csi300_pit_fundamental_underreaction_enhancement_v1.yaml"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_report(report: dict, inventory_path: Path) -> str:
    lines = [
        "# 沪深300点时基本面反应不足 V1 数据准入预检",
        "",
        f"- 状态：`{report['status']}`",
        f"- 收益评价：`{report['return_evaluation']}`",
        f"- 通过门：{report['passed_gate_count']}/{report['total_gate_count']}",
        f"- 下一步：`{report['next_allowed_step']}`",
        "",
        "## 目标事件",
        "",
    ]
    for key, value in report["inventory_metrics"].items():
        if not isinstance(value, dict):
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            f"- 无收益目标事件清单：`{inventory_path.relative_to(PROJECT_ROOT).as_posix()}`",
            "",
            "## 数据门",
            "",
            "| 门 | 结果 | 观测 | 要求 |",
            "|---|---:|---|---|",
        ]
    )
    for gate in report["gates"]:
        observed = json.dumps(gate["observed"], ensure_ascii=False, separators=(",", ":"))
        required = json.dumps(gate["required"], ensure_ascii=False, separators=(",", ":"))
        lines.append(
            f"| `{gate['gate_id']}` | {'PASS' if gate['passed'] else 'FAIL'} | "
            f"`{observed}` | `{required}` |"
        )
    lines.extend(
        [
            "",
            "## 治理边界",
            "",
            "本次预检没有读取市场价格、未来收益或未来标签，没有计算信号、组合、夏普、仓位或订单。"
            "只要官方首份原始完整 PDF 的逐项财务事实门未通过，`RETURN_EVALUATION=NOT_ALLOWED`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report, inventory = run_preflight(config)
    report["checked_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["protocol"] = {
        "path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(CONFIG_PATH),
    }

    artifacts = config["artifacts"]
    inventory_path = project_path(artifacts["target_event_inventory"])
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(inventory_path, index=False)
    report["artifacts"] = {
        "target_event_inventory": {
            "path": inventory_path.relative_to(PROJECT_ROOT).as_posix(),
            "rows": int(len(inventory)),
            "sha256": sha256_file(inventory_path),
        }
    }

    json_path = project_path(artifacts["preflight_json"])
    markdown_path = project_path(artifacts["preflight_markdown"])
    write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(report, inventory_path), encoding="utf-8")

    print(f"数据准入状态：{report['status']}")
    print(f"收益评价：{report['return_evaluation']}")
    print(f"通过门：{report['passed_gate_count']}/{report['total_gate_count']}")
    print(f"报告：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

