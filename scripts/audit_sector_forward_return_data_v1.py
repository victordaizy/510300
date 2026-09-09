"""审计并构造点时板块估值—盈利面板，不读取未来收益。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.sector_point_in_time_valuation import (  # noqa: E402
    SectorPanelRules,
    build_sector_point_in_time_panel,
)


CONTRACT_FILE = ROOT / "config" / "sector_forward_return_data_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _render_markdown(report: dict[str, Any]) -> str:
    minimum = report["minimum_snapshot_coverages"]
    return "\n".join(
        [
            "# 510300 点时板块未来收益数据可行性 V1",
            "",
            f"- 状态：`{report['status']}`",
            f"- 失败类别：`{report['failure_category']}`",
            f"- 快照：{report['snapshot_count']}；有效：{report['valid_snapshot_count']}；NO_VIEW：{report['no_view_snapshot_count']}",
            f"- 日期：{report['first_snapshot_date']} 至 {report['last_snapshot_date']}",
            f"- 互不重叠60日时间块：{report['nonoverlapping_60d_time_blocks']}",
            "",
            "## 最低覆盖",
            "",
            *[f"- `{key}`：{value}" for key, value in minimum.items()],
            "",
            "## 治理",
            "",
            "未来收益未读取，预测模型未拟合；数据通过只允许进入目标冻结。仓位、订单和券商连接保持禁用。",
            "",
        ]
    )


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    governance = contract["governance"]
    forbidden = [
        "future_return_reading_allowed",
        "predictive_model_fitting_allowed",
        "feature_selection_allowed",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    ]
    enabled = [key for key in forbidden if governance.get(key) is True]
    if enabled:
        raise ValueError(f"数据审计阶段存在非法启用项：{enabled}")
    inputs = contract["inputs"]
    input_paths = {key: ROOT / value for key, value in inputs.items()}
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"板块数据输入缺失：{missing}")
    rules = SectorPanelRules.from_contract(contract)
    result = build_sector_point_in_time_panel(
        pd.read_parquet(input_paths["weights"]),
        pd.read_parquet(input_paths["financials"]),
        pd.read_parquet(input_paths["constituent_daily"]),
        pd.read_parquet(input_paths["industry_intervals"]),
        rules,
    )
    outputs = contract["outputs"]
    panel_path = ROOT / outputs["sector_panel"]
    audit_path = ROOT / outputs["snapshot_audit"]
    _atomic_parquet(result.sector_panel, panel_path)
    _atomic_parquet(result.snapshot_audit, audit_path)
    coverage_columns = [
        "price_weight_coverage",
        "industry_weight_coverage",
        "earnings_weight_coverage",
        "book_weight_coverage",
        "roe_weight_coverage",
        "eligible_sector_weight_coverage",
    ]
    report = _safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "version": contract["protocol"]["version"],
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            **result.feasibility,
            "sector_row_count": int(len(result.sector_panel)),
            "industry_count": int(result.sector_panel["industry_l1"].nunique()),
            "minimum_snapshot_coverages": {
                column: float(result.snapshot_audit[column].min())
                for column in coverage_columns
            },
            "no_view_failure_counts": result.snapshot_audit.loc[
                result.snapshot_audit["output"].eq("NO_VIEW"), "failure_category"
            ].value_counts().to_dict(),
            "input_hashes": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in input_paths.values()
            },
            "output_hashes": {
                outputs["sector_panel"]: _sha256(panel_path),
                outputs["snapshot_audit"]: _sha256(audit_path),
            },
            "governance": governance,
        }
    )
    json_path = ROOT / outputs["status_json"]
    markdown_path = ROOT / outputs["status_markdown"]
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    _atomic_text(_render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "有效快照": report["valid_snapshot_count"],
                "NO_VIEW快照": report["no_view_snapshot_count"],
                "互不重叠60日块": report["nonoverlapping_60d_time_blocks"],
                "报告": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
