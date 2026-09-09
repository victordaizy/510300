"""按冻结清单构建 510300 点时指数截面与板块贡献。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.index_driver_attribution import (
    AttributionRules,
    IndexDriverAttribution,
    build_index_driver_attribution,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "index_driver_attribution_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest() -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("归因 V1 尚未冻结；先运行冻结脚本")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    changed: list[str] = []
    for section in ("frozen_files", "input_files"):
        for relative, expected_hash in manifest[section].items():
            path = ROOT / relative
            if not path.exists() or sha256(path) != expected_hash:
                changed.append(relative)
    if changed:
        raise RuntimeError(f"归因 V1 冻结文件指纹变化：{changed}")
    return manifest


def _atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _nullable_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _build_report(
    result: IndexDriverAttribution,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    daily = result.daily.sort_values("date").reset_index(drop=True)
    valid_days = daily.loc[daily["valid_for_attribution"]]
    no_view_days = daily.loc[~daily["valid_for_attribution"]]
    latest = daily.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    latest_industries = result.industry.loc[
        result.industry["date"].eq(latest_date)
    ].sort_values("weighted_return_contribution_1d", ascending=False)
    top_drivers = [
        {
            "industry_l1": str(row.industry_l1),
            "industry_weight": float(row.industry_weight),
            "industry_return_1d": _nullable_float(row.industry_return_1d),
            "weighted_return_contribution_1d": _nullable_float(
                row.weighted_return_contribution_1d
            ),
        }
        for row in latest_industries.head(5).itertuples(index=False)
    ]
    failure_counts = {
        str(category): int(count)
        for category, count in no_view_days["failure_category"].value_counts().items()
    }
    status = (
        "PASS_ATTRIBUTION_ONLY"
        if no_view_days.empty
        else "PASS_ATTRIBUTION_ONLY_WITH_NO_VIEW_DAYS"
    )
    if latest["output"] == "NO_VIEW":
        status = "LATEST_NO_VIEW"
    return {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "status": status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_start": str(pd.Timestamp(daily["date"].min()).date()),
        "data_cutoff": str(pd.Timestamp(daily["date"].max()).date()),
        "evidence_scope": "ATTRIBUTION_ONLY_NOT_PREDICTION",
        "row_counts": {
            "cross_section": int(len(result.cross_section)),
            "industry": int(len(result.industry)),
            "daily": int(len(result.daily)),
            "valid_attribution_days": int(len(valid_days)),
            "no_view_days": int(len(no_view_days)),
        },
        "quality_summary": {
            "minimum_price_coverage_weight": _nullable_float(
                daily["price_coverage_weight"].min()
            ),
            "minimum_return_coverage_weight": _nullable_float(
                daily["return_coverage_weight"].min()
            ),
            "minimum_industry_coverage_weight": _nullable_float(
                daily["industry_coverage_weight"].min()
            ),
            "maximum_weight_snapshot_age_calendar_days": int(
                daily["weight_snapshot_age_calendar_days"].max()
            ),
            "maximum_contribution_identity_absolute_error": _nullable_float(
                daily["contribution_identity_absolute_error"].max()
            ),
            "failure_category_counts": failure_counts,
        },
        "latest_attribution": {
            "date": str(latest_date.date()),
            "output": str(latest["output"]),
            "failure_category": str(latest["failure_category"]),
            "weight_snapshot_date": str(
                pd.Timestamp(latest["weight_snapshot_date"]).date()
            ),
            "weight_snapshot_age_calendar_days": int(
                latest["weight_snapshot_age_calendar_days"]
            ),
            "leading_industry_l1": (
                None
                if pd.isna(latest["leading_industry_l1"])
                else str(latest["leading_industry_l1"])
            ),
            "leading_industry_contribution_1d": _nullable_float(
                latest["leading_industry_contribution_1d"]
            ),
            "snapshot_weighted_constituent_return_1d": _nullable_float(
                latest["snapshot_weighted_constituent_return_1d"]
            ),
            "unattributed_contribution_1d": _nullable_float(
                latest["unattributed_contribution_1d"]
            ),
            "top_drivers": top_drivers,
        },
        "taxonomy": contract["inputs"]["industry_intervals"],
        "semantics": contract["semantics"],
        "governance": contract["governance"],
        "manifest_sha256": sha256(MANIFEST_FILE),
        "source_commit": manifest.get("source_commit"),
        "input_hashes": manifest["input_files"],
        "output_hashes": {
            relative: sha256(path)
            for relative, path in output_paths.items()
            if path.exists()
        },
    }


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{value:.4%}"


def _render_markdown(report: dict[str, Any]) -> str:
    latest = report["latest_attribution"]
    lines = [
        "# 510300 点时指数驱动归因 V1 数据质量报告",
        "",
        f"状态：`{report['status']}`  ",
        f"数据截止日：`{report['data_cutoff']}`  ",
        "证据范围：`ATTRIBUTION_ONLY_NOT_PREDICTION`",
        "",
        "## 结论",
        "",
        "本报告只描述同期板块贡献，不读取未来收益，不产生鱼中、鱼尾、仓位或订单结论。",
        "",
        "## 覆盖与恒等式",
        "",
        f"- 点时指数截面：{report['row_counts']['cross_section']} 行。",
        f"- 板块贡献：{report['row_counts']['industry']} 行。",
        f"- 交易日：{report['row_counts']['daily']}；归因有效日：{report['row_counts']['valid_attribution_days']}；NO_VIEW：{report['row_counts']['no_view_days']}。",
        f"- 最低价格权重覆盖：{_percentage(report['quality_summary']['minimum_price_coverage_weight'])}。",
        f"- 最低收益权重覆盖：{_percentage(report['quality_summary']['minimum_return_coverage_weight'])}。",
        f"- 最低行业权重覆盖：{_percentage(report['quality_summary']['minimum_industry_coverage_weight'])}。",
        f"- 最大权重年龄：{report['quality_summary']['maximum_weight_snapshot_age_calendar_days']} 个自然日。",
        f"- 最大贡献恒等式绝对误差：{report['quality_summary']['maximum_contribution_identity_absolute_error']}。",
        "",
        "## 最新归因",
        "",
        f"- 日期：{latest['date']}。",
        f"- 输出：`{latest['output']}`；失败类别：`{latest['failure_category']}`。",
        f"- 权重快照：{latest['weight_snapshot_date']}，年龄 {latest['weight_snapshot_age_calendar_days']} 个自然日。",
        f"- 当日板块驱动：{latest['leading_industry_l1']}；贡献 {_percentage(latest['leading_industry_contribution_1d'])}。",
        f"- 快照加权成分收益：{_percentage(latest['snapshot_weighted_constituent_return_1d'])}。",
        f"- 未归因贡献：{_percentage(latest['unattributed_contribution_1d'])}。",
        "",
        "|排序|中信一级行业|行业权重|行业收益|快照加权贡献|",
        "|---:|---|---:|---:|---:|",
    ]
    for index, row in enumerate(latest["top_drivers"], start=1):
        lines.append(
            f"|{index}|{row['industry_l1']}|{_percentage(row['industry_weight'])}|"
            f"{_percentage(row['industry_return_1d'])}|"
            f"{_percentage(row['weighted_return_contribution_1d'])}|"
        )
    lines.extend(
        [
            "",
            "## 安全状态",
            "",
            "- `FUTURE_RETURN_READING_FORBIDDEN`",
            "- `POSITION_MAPPING_DISABLED`",
            "- `ORDER_GENERATION_DISABLED`",
            "- `BROKER_CONNECTION_DISABLED`",
            "",
            "月度快照权重归因不得冒充中证官方指数逐日贡献。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest = _verify_manifest()
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    rules = AttributionRules.from_contract(contract)
    input_frames = {
        name: pd.read_parquet(ROOT / settings["path"])
        for name, settings in contract["inputs"].items()
    }
    result = build_index_driver_attribution(
        input_frames["weights"],
        input_frames["constituent_daily"],
        input_frames["industry_intervals"],
        rules,
    )
    outputs = contract["outputs"]
    cross_path = ROOT / outputs["cross_section_path"]
    industry_path = ROOT / outputs["industry_path"]
    daily_path = ROOT / outputs["daily_summary_path"]
    json_path = ROOT / outputs["status_json_path"]
    markdown_path = ROOT / outputs["status_markdown_path"]
    _atomic_parquet_write(result.cross_section, cross_path)
    _atomic_parquet_write(result.industry, industry_path)
    _atomic_parquet_write(result.daily, daily_path)
    output_paths = {
        outputs["cross_section_path"]: cross_path,
        outputs["industry_path"]: industry_path,
        outputs["daily_summary_path"]: daily_path,
    }
    report = _build_report(result, contract, manifest, output_paths)
    _atomic_text_write(
        json.dumps(report, ensure_ascii=False, indent=2), json_path
    )
    _atomic_text_write(_render_markdown(report), markdown_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
