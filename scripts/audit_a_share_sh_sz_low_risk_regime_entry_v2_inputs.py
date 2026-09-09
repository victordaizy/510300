"""审计沪深 A 股低风险 V2 的点时输入；禁止读取或计算任何收益结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/a_share_sh_sz_low_risk_regime_entry_v2.yaml"
CORE_INPUT_KEYS = (
    "security_master_history",
    "daily_market",
    "share_count_history",
    "security_status_intervals",
    "industry_l1_intervals",
    "board_execution_rules",
    "trading_calendar",
)
LEGACY_EVIDENCE = {
    "legacy_security_master": ROOT
    / "data/raw/a_share_hash_holdout_v2/stock_master_split.parquet",
    "legacy_daily_basic_signal_dates": ROOT
    / "data/raw/a_share_size_value_development_v1/daily_basic_signal_dates.parquet",
    "legacy_citic_industry_intervals": ROOT
    / "data/raw/reference/a_share_citic_industry_point_in_time.parquet",
    "legacy_sw_industry_intervals_partial": ROOT
    / "data/raw/reference/a_share_sw_industry_point_in_time_partial.parquet",
}


def sha256_file(path: Path) -> str:
    """以分块方式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(relative: str) -> Path:
    """把项目相对路径解析为受控绝对路径。"""

    candidate = (ROOT / relative).resolve()
    candidate.relative_to(ROOT.resolve())
    return candidate


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def inspect_parquet(path: Path, *, value_columns: tuple[str, ...] = ()) -> dict[str, Any]:
    """读取 Parquet 元数据，并只读取明确允许的非收益列。"""

    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    result: dict[str, Any] = {
        "exists": True,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "columns": columns,
    }
    selected = [column for column in value_columns if column in columns]
    if not selected:
        return result
    frame = parquet.read(columns=selected).to_pandas()
    summaries: dict[str, Any] = {}
    for column in selected:
        series = frame[column].dropna()
        item: dict[str, Any] = {
            "non_null_count": int(series.size),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if not series.empty and (
            pd.api.types.is_datetime64_any_dtype(series)
            or column in {"date", "trade_date", "list_date", "delist_date"}
        ):
            converted = pd.to_datetime(series, errors="coerce").dropna()
            if not converted.empty:
                item["min"] = converted.min().isoformat()
                item["max"] = converted.max().isoformat()
        if column in {"exchange", "market", "security_type", "status"}:
            counts = series.astype(str).value_counts(dropna=False).sort_index()
            item["counts"] = {str(key): int(value) for key, value in counts.items()}
        if column in {"ts_code", "con_code"}:
            codes = series.astype(str)
            item["suffix_counts"] = {
                "SH": int(codes.str.endswith(".SH").sum()),
                "SZ": int(codes.str.endswith(".SZ").sum()),
                "BJ": int(codes.str.endswith(".BJ").sum()),
            }
        summaries[column] = item
    result["non_return_column_summaries"] = summaries
    return result


def inspect_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
        "payload": payload,
    }


def audit_required_inputs(contract: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """验证首阶段所需点时输入是否存在且具备冻结字段。"""

    checks: dict[str, Any] = {}
    blockers: list[str] = []
    configured_inputs = contract["inputs"]
    for key in CORE_INPUT_KEYS:
        specification = configured_inputs[key]
        path = project_path(specification["path"])
        check: dict[str, Any] = {
            "path": specification["path"],
            "exists": path.is_file(),
            "required_for_non_return_coverage_audit": True,
        }
        if not path.is_file():
            check["status"] = "MISSING"
            blockers.append(f"缺少点时输入：{specification['path']}")
            checks[key] = check
            continue

        if path.suffix.lower() == ".parquet":
            detail = inspect_parquet(path)
            observed_columns = set(detail["columns"])
            required_columns = set(specification.get("required_columns", []))
            missing_columns = sorted(required_columns - observed_columns)
            check.update(detail)
            check["missing_required_columns"] = missing_columns
            if missing_columns:
                check["status"] = "SCHEMA_INCOMPLETE"
                blockers.append(
                    f"{specification['path']}缺少字段：{', '.join(missing_columns)}"
                )
            else:
                check["status"] = "SCHEMA_PRESENT_NOT_YET_COVERAGE_AUDITED"
        elif path.suffix.lower() in {".yaml", ".yml"}:
            detail = inspect_yaml(path)
            check.update({key_: value for key_, value in detail.items() if key_ != "payload"})
            payload = detail["payload"]
            supported = set(payload.get("boards", [])) if isinstance(payload, dict) else set()
            required = set(specification.get("required_boards", []))
            missing_boards = sorted(required - supported)
            check["observed_boards"] = sorted(supported)
            check["missing_required_boards"] = missing_boards
            if missing_boards:
                check["status"] = "BOARD_RULES_INCOMPLETE"
                blockers.append(
                    f"{specification['path']}缺少板块规则：{', '.join(missing_boards)}"
                )
            else:
                check["status"] = "BOARD_RULES_PRESENT_NOT_YET_COVERAGE_AUDITED"
        else:
            check["status"] = "UNSUPPORTED_FILE_FORMAT"
            blockers.append(f"无法审计输入格式：{specification['path']}")
        checks[key] = check
    return checks, blockers


def audit_legacy_evidence() -> dict[str, Any]:
    """记录现有数据能证明什么；这些文件不自动成为新协议的合格输入。"""

    permitted_columns = {
        "legacy_security_master": (
            "ts_code",
            "exchange",
            "market",
            "list_date",
            "delist_date",
        ),
        "legacy_daily_basic_signal_dates": (
            "ts_code",
            "trade_date",
            "total_share",
            "float_share",
            "total_mv",
            "circ_mv",
        ),
        "legacy_citic_industry_intervals": (
            "con_code",
            "in_date",
            "out_date",
        ),
        "legacy_sw_industry_intervals_partial": (
            "con_code",
            "in_date",
            "out_date",
        ),
    }
    results: dict[str, Any] = {}
    for key, path in LEGACY_EVIDENCE.items():
        if not path.is_file():
            results[key] = {
                "exists": False,
                "path": path.relative_to(ROOT).as_posix(),
            }
            continue
        detail = inspect_parquet(path, value_columns=permitted_columns[key])
        detail["qualified_as_new_protocol_input"] = False
        results[key] = detail
    return results


def _legacy_findings(legacy: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    master = legacy.get("legacy_security_master", {})
    code_summary = (
        master.get("non_return_column_summaries", {})
        .get("ts_code", {})
        .get("suffix_counts", {})
    )
    if code_summary:
        findings.append(
            "旧证券母表代码覆盖："
            f"SH={code_summary.get('SH', 0)}，"
            f"SZ={code_summary.get('SZ', 0)}，"
            f"BJ={code_summary.get('BJ', 0)}；本模型只要求 SH/SZ。"
        )

    daily_basic = legacy.get("legacy_daily_basic_signal_dates", {})
    daily_columns = set(daily_basic.get("columns", []))
    trade_date = (
        daily_basic.get("non_return_column_summaries", {}).get("trade_date", {})
    )
    if daily_basic.get("exists"):
        findings.append(
            "旧 daily_basic 信号日快照范围："
            f"{trade_date.get('min', 'UNKNOWN')} 至 {trade_date.get('max', 'UNKNOWN')}，"
            f"共 {daily_basic.get('rows', 0)} 行。"
        )
        if "available_at" not in daily_columns:
            findings.append(
                "旧 daily_basic 虽含 total_share/float_share，但没有 available_at，"
                "不能直接满足新协议的点时股本证据契约。"
            )

    citic = legacy.get("legacy_citic_industry_intervals", {})
    sw = legacy.get("legacy_sw_industry_intervals_partial", {})
    if citic.get("exists") or sw.get("exists"):
        findings.append(
            "旧点时行业区间行数："
            f"中信={citic.get('rows', 0)}，申万部分集={sw.get('rows', 0)}；"
            "未完成沪深母池逐日覆盖审计。"
        )
    return findings


def build_report(contract: dict[str, Any]) -> dict[str, Any]:
    required_checks, blockers = audit_required_inputs(contract)
    legacy = audit_legacy_evidence()
    status = (
        "BLOCKED_DATA_CONTRACT_INCOMPLETE"
        if blockers
        else "READY_FOR_NON_RETURN_COVERAGE_AUDIT"
    )
    return {
        "schema_version": "A_SHARE_SH_SZ_DATA_READINESS_V1",
        "model_id": contract["protocol"]["model_id"],
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "as_of_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "status": status,
        "view_status": "NO_VIEW",
        "research_status": "DISCOVERY_ONLY",
        "scope": {
            "included_exchanges": contract["universe"]["exchanges"],
            "explicitly_out_of_scope_exchanges": contract["universe"][
                "explicitly_out_of_scope_exchanges"
            ],
            "security_type": contract["universe"]["security_type"],
        },
        "return_values_read": False,
        "return_metrics_computed": False,
        "selection_generated": False,
        "position_mapping_generated": False,
        "orders_generated": False,
        "required_input_checks": required_checks,
        "blocking_reasons": blockers,
        "legacy_evidence_not_qualified_as_new_input": legacy,
        "legacy_findings": _legacy_findings(legacy),
        "next_allowed_step": (
            "补齐点时输入并重新建立新版本数据清单"
            if blockers
            else "运行一次不含收益的母池覆盖审计"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['model_id']} 数据可行性审计",
        "",
        f"- 审计时点：{report['audited_at']}",
        f"- 状态：`{report['status']}`",
        f"- 研究视图：`{report['view_status']}`",
        "- 范围：上交所、深交所普通 A 股；北交所明确不在本模型范围内。",
        "- 收益值读取：否",
        "- 收益指标计算：否",
        "- 选股、仓位或订单生成：否",
        "",
        "## 阻断项",
        "",
    ]
    blockers = report["blocking_reasons"]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- 无；可进入一次性非收益母池覆盖审计。")
    lines.extend(["", "## 现有旧数据只读核对", ""])
    findings = report["legacy_findings"]
    if findings:
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append("- 未找到可用于交叉核对的旧数据。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"下一步仅允许：{report['next_allowed_step']}。",
            "当前不得运行收益回测、报告策略表现、映射 2 万元账户或产生任何交易动作。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--replace-unfrozen",
        action="store_true",
        help="仅在协议清单尚未生成时允许替换旧审计输出",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config_path.relative_to(ROOT.resolve())
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = project_path(contract["paths"]["protocol_manifest"])
    json_path = project_path(contract["paths"]["data_readiness_json"])
    markdown_path = project_path(contract["paths"]["data_readiness_markdown"])
    if manifest_path.exists():
        raise FileExistsError("协议已经冻结，禁止覆盖其数据可行性证据")
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not args.replace_unfrozen:
        names = [path.relative_to(ROOT).as_posix() for path in existing]
        raise FileExistsError(f"未冻结审计输出已存在；显式确认替换后再运行：{names}")

    report = build_report(contract)
    atomic_write_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_value) + "\n",
    )
    atomic_write_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "模型": report["model_id"],
                "状态": report["status"],
                "收益值读取": False,
                "阻断项数量": len(report["blocking_reasons"]),
                "JSON": json_path.relative_to(ROOT).as_posix(),
                "报告": markdown_path.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
