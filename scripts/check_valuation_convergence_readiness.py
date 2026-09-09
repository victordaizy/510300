"""检查510300长期估值收敛研究的数据就绪状态。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "valuation_convergence_data_contract.yaml"
REPORT_FILE = ROOT / "reports" / "data_quality" / "valuation_convergence_readiness.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class DatasetAssessment:
    """单个数据集的结构与日期覆盖检查结果。"""

    dataset_id: str
    path: str
    status: str
    exists: bool
    row_count: int | None
    fields: list[str]
    missing_fields: list[str]
    minimum_date: str | None
    maximum_date: str | None
    errors: list[str]


def _read_fields(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        return list(parquet.ParquetFile(path).schema_arrow.names)
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"不支持的数据格式：{path.suffix}")


def _read_date_column(path: Path, date_field: str) -> pd.Series:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=[date_field])
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=[date_field])
    else:
        raise ValueError(f"不支持的数据格式：{path.suffix}")
    return pd.to_datetime(frame[date_field], errors="coerce")


def _row_count(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        return int(parquet.ParquetFile(path).metadata.num_rows)
    if path.suffix.lower() == ".csv":
        return int(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1)
    raise ValueError(f"不支持的数据格式：{path.suffix}")


def inspect_dataset(
    root: Path,
    dataset_id: str,
    specification: dict[str, Any],
) -> DatasetAssessment:
    """按合同检查文件存在性、字段与日期覆盖。"""

    relative_path = str(specification["path"])
    path = root / Path(relative_path)
    required_fields = [str(field) for field in specification.get("required_fields", [])]
    if not path.exists():
        return DatasetAssessment(
            dataset_id=dataset_id,
            path=relative_path,
            status="MISSING",
            exists=False,
            row_count=None,
            fields=[],
            missing_fields=required_fields,
            minimum_date=None,
            maximum_date=None,
            errors=["文件不存在"],
        )

    try:
        fields = _read_fields(path)
        missing_fields = sorted(set(required_fields) - set(fields))
        errors: list[str] = []
        minimum_date: str | None = None
        maximum_date: str | None = None
        date_field = specification.get("date_field")
        if date_field and date_field in fields:
            dates = _read_date_column(path, str(date_field)).dropna()
            if dates.empty:
                errors.append(f"日期字段{date_field}没有有效值")
            else:
                minimum = dates.min().normalize()
                maximum = dates.max().normalize()
                minimum_date = minimum.date().isoformat()
                maximum_date = maximum.date().isoformat()
                required_start = specification.get("minimum_start_date")
                required_end = specification.get("minimum_end_date")
                start_lag = int(specification.get("maximum_start_lag_calendar_days", 0))
                end_lag = int(specification.get("maximum_end_lag_calendar_days", 0))
                if required_start and minimum > pd.Timestamp(required_start) + pd.Timedelta(days=start_lag):
                    errors.append(
                        f"起始日期晚于合同要求{required_start}及允许滞后{start_lag}天"
                    )
                if required_end and maximum < pd.Timestamp(required_end) - pd.Timedelta(days=end_lag):
                    errors.append(
                        f"结束日期早于合同要求{required_end}及允许滞后{end_lag}天"
                    )
        if missing_fields:
            errors.append("缺少合同字段")
        status = "PASS" if not errors else "FAIL"
        return DatasetAssessment(
            dataset_id=dataset_id,
            path=relative_path,
            status=status,
            exists=True,
            row_count=_row_count(path),
            fields=fields,
            missing_fields=missing_fields,
            minimum_date=minimum_date,
            maximum_date=maximum_date,
            errors=errors,
        )
    except Exception as error:  # 数据审计必须将读取异常写入报告
        return DatasetAssessment(
            dataset_id=dataset_id,
            path=relative_path,
            status="ERROR",
            exists=True,
            row_count=None,
            fields=[],
            missing_fields=required_fields,
            minimum_date=None,
            maximum_date=None,
            errors=[f"读取失败：{type(error).__name__}: {error}"],
        )


def _all_pass(assessments: dict[str, DatasetAssessment], dataset_ids: list[str]) -> bool:
    return all(assessments[dataset_id].status == "PASS" for dataset_id in dataset_ids)


def evaluate_weight_reconstruction(
    root: Path,
    contract: dict[str, Any],
    assessments: dict[str, DatasetAssessment],
) -> dict[str, Any]:
    """判断官方权重或合规重构权重是否可用。"""

    official_ready = assessments["historical_official_weights"].status == "PASS"
    reconstruction = contract["weight_reconstruction"]
    top10_path = root / Path(str(reconstruction["official_top10_path"]))
    top10_fields: list[str] = []
    top10_missing = list(reconstruction["official_top10_required_fields"])
    if top10_path.exists():
        top10_fields = _read_fields(top10_path)
        top10_missing = sorted(set(top10_missing) - set(top10_fields))

    constituent_fields = set(assessments["constituent_daily"].fields)
    alternatives = [set(option) for option in reconstruction["constituent_required_fields_any_of"]]
    matched_alternative = next(
        (sorted(option) for option in alternatives if option.issubset(constituent_fields)),
        None,
    )
    reconstruction_ready = (
        assessments["historical_membership"].status == "PASS"
        and assessments["constituent_daily"].status == "PASS"
        and top10_path.exists()
        and not top10_missing
        and matched_alternative is not None
    )
    return {
        "official_history_ready": official_ready,
        "reconstruction_ready": reconstruction_ready,
        "weight_source_ready": official_ready or reconstruction_ready,
        "top10_history_path": str(top10_path.relative_to(root)),
        "top10_history_exists": top10_path.exists(),
        "top10_history_missing_fields": top10_missing,
        "matched_constituent_field_alternative": matched_alternative,
        "warning": "总市值不能替代调整自由流通市值。" if not reconstruction_ready else None,
    }


def determine_capabilities(
    contract: dict[str, Any],
    assessments: dict[str, DatasetAssessment],
    weight_status: dict[str, Any],
) -> dict[str, bool]:
    """根据数据合同计算每一层研究能力。"""

    capabilities: dict[str, bool] = {}
    for capability_id, specification in contract["capabilities"].items():
        ready = _all_pass(assessments, list(specification["required_datasets"]))
        if specification.get("requires_weight_source"):
            ready = ready and bool(weight_status["weight_source_ready"])
        capabilities[capability_id] = ready
    return capabilities


def determine_overall_status(capabilities: dict[str, bool]) -> str:
    """输出最可操作的总体状态。"""

    if capabilities["component_aggregate_model"] and capabilities["conditional_index_mvp"]:
        return "READY_FOR_FULL_VALUATION_RESEARCH"
    if capabilities["conditional_index_mvp"]:
        return "READY_FOR_CONDITIONAL_INDEX_MVP"
    if capabilities["historical_unconditional_baseline"]:
        return "BASELINE_ONLY_MISSING_POINT_IN_TIME_INPUTS"
    return "BLOCKED_MISSING_CORE_MARKET_DATA"


def build_report(root: Path = ROOT, contract_file: Path = CONTRACT_FILE) -> dict[str, Any]:
    """构建完整就绪报告，供命令行和测试复用。"""

    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    assessments = {
        dataset_id: inspect_dataset(root, dataset_id, specification)
        for dataset_id, specification in contract["datasets"].items()
    }
    weight_status = evaluate_weight_reconstruction(root, contract, assessments)
    capabilities = determine_capabilities(contract, assessments, weight_status)
    missing = [
        dataset_id
        for dataset_id, assessment in assessments.items()
        if assessment.status != "PASS"
    ]
    return {
        "checked_at": datetime.now(TIMEZONE).isoformat(),
        "contract_version": contract["contract_version"],
        "overall_status": determine_overall_status(capabilities),
        "capabilities": capabilities,
        "weight_status": weight_status,
        "missing_or_failed_datasets": missing,
        "datasets": {
            dataset_id: asdict(assessment)
            for dataset_id, assessment in assessments.items()
        },
        "next_action": (
            "补齐国债收益率、点时财务公告数据及合规历史权重输入。"
            if missing
            else "进入模型训练前的点时一致性与数值质量审计。"
        ),
    }


def main() -> int:
    report = build_report()
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["capabilities"]["historical_unconditional_baseline"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
