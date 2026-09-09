"""汇总行业预期差追加式报告的预测原点簇成熟度。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_expectation_gap_origin_maturity import (
    build_origin_cluster_maturity_status,
)


CONFIG_FILE = ROOT / "config" / "industry_expectation_gap_origin_maturity_v1.yaml"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _markdown(report: dict) -> str:
    return "\n".join(
        [
            "# 行业预期差预测原点簇成熟度",
            "",
            f"- 状态：`{report['status']}`",
            f"- 预测原点簇：{report['origin_cluster_count']}",
            f"- 成熟原点簇：{report['mature_origin_cluster_count']}",
            f"- 非重叠60日块：{report['non_overlapping_60d_block_count']}",
            "- 独立时间样本单位：`origin_cluster = prediction_date`",
            "- 同日行业行不计为独立时间样本。",
            "- 原指数观点保持 `NO_VIEW`，禁止事后升级。",
            "- 20簇且4个非重叠60日块才允许校准；40簇且8块才允许模型比较。",
            "- 仓位映射、订单、券商连接和实盘全部关闭。",
            "",
        ]
    )


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    directory = ROOT / config["outputs"]["report_directory"]
    reports = []
    for path in sorted(directory.glob("industry_expectation_gap_forward_evaluation_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluation_version"):
            reports.append(payload)
    maturity = config["maturity"]
    report = build_origin_cluster_maturity_status(
        reports,
        primary_horizon_trading_days=int(config["primary_horizon_trading_days"]),
        calibration_minimum_origin_clusters=int(
            maturity["calibration_minimum_origin_clusters"]
        ),
        calibration_minimum_non_overlapping_blocks=int(
            maturity["calibration_minimum_non_overlapping_60d_blocks"]
        ),
        model_comparison_minimum_origin_clusters=int(
            maturity["model_comparison_minimum_origin_clusters"]
        ),
        model_comparison_minimum_non_overlapping_blocks=int(
            maturity["model_comparison_minimum_non_overlapping_60d_blocks"]
        ),
    )
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source_report_count"] = len(reports)
    json_path = ROOT / config["outputs"]["json"]
    markdown_path = ROOT / config["outputs"]["markdown"]
    _atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _atomic_text(markdown_path, _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

